//! HuggingFace Hub integration for oxillama-py.
//!
//! Downloads GGUF files from a HuggingFace repository using the `hf-hub`
//! crate's synchronous (`blocking`-feature) client, so these bindings stay
//! callable from Python without an async runtime of their own. Download
//! progress renders as an `indicatif` bar via [`PullProgress`] — see its doc
//! comment for why this duplicates `oxillama-cli`'s copy instead of sharing
//! it.

use hf_hub::HFClient;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

/// Resolve an HF access token from the explicit argument, then from common
/// environment variables (`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`).
fn resolve_token(token: Option<&str>) -> Option<String> {
    if let Some(t) = token {
        return Some(t.to_owned());
    }
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGINGFACE_HUB_TOKEN").ok())
}

/// Renders `Engine.from_hub()` / `oxillama_py.hub.load_from_hub()` download
/// progress as an `indicatif` bar.
///
/// This is a near-verbatim duplicate of `oxillama-cli`'s `PullProgress`
/// (`crates/oxillama-cli/src/hub.rs`) — same hf-hub 1.0
/// [`ProgressHandler`](hf_hub::progress::ProgressHandler) contract, same
/// template. There is no crate shared by both `oxillama-cli` and
/// `oxillama-py` that this could live in without misplacing a terminal-UI
/// helper inside a lower layer (`oxillama-runtime`/`oxillama-gguf`), so the
/// two copies are kept independently — if the rendering changes in one,
/// check whether the other should follow.
///
/// Fixes a regression: the hf-hub 0.5 → 1.0 migration (v0.1.4) dropped the
/// old `ApiBuilder::new()` (defaulted `progress = true`) built-in renderer in
/// favour of this callback, and `oxillama-py` did not yet depend on
/// `indicatif` to replace it — see TODO.md's Known Gaps for the dated history.
struct PullProgress {
    bar: indicatif::ProgressBar,
}

impl PullProgress {
    /// Longest filename rendered in the bar's message; longer names are shown
    /// as `..{tail}` — the elision rule hf-hub 0.5 applied.
    const MAX_MESSAGE_CHARS: usize = 30;

    fn new() -> Self {
        // The total is unknown until the `Start` event reports the HEAD size.
        let bar = indicatif::ProgressBar::new(0);
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "{msg} [{elapsed_precise}] [{wide_bar}] {bytes}/{total_bytes} {bytes_per_sec} ({eta})",
            )
            .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar()),
        );
        Self { bar }
    }

    /// Elide `filename` from the left so the message stays one line wide.
    ///
    /// Split on `char` boundaries rather than bytes: repo paths are arbitrary
    /// UTF-8, and byte slicing would panic mid-codepoint.
    fn shorten(filename: &str) -> String {
        let chars = filename.chars().count();
        if chars <= Self::MAX_MESSAGE_CHARS {
            return filename.to_string();
        }
        let tail: String = filename
            .chars()
            .skip(chars - Self::MAX_MESSAGE_CHARS)
            .collect();
        format!("..{tail}")
    }
}

impl hf_hub::progress::ProgressHandler for PullProgress {
    fn on_progress(&self, event: &hf_hub::progress::ProgressEvent) {
        use hf_hub::progress::{DownloadEvent, ProgressEvent};

        // `download_model_from_hub` only ever downloads, so upload events
        // cannot reach here.
        let ProgressEvent::Download(download) = event else {
            return;
        };

        match download {
            DownloadEvent::Start { total_bytes, .. } => self.bar.set_length(*total_bytes),
            // Per-file deltas. Exactly one file downloads per call, so the
            // most recent entry always describes it.
            DownloadEvent::Progress { files } => {
                if let Some(file) = files.last() {
                    self.bar.set_message(Self::shorten(&file.filename));
                    self.bar.set_position(file.bytes_completed);
                }
            }
            // Xet-backed blobs report batch aggregates instead of per-file
            // deltas, and the batch here is that single file.
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                self.bar.set_length(*total_bytes);
                self.bar.set_position(*bytes_completed);
            }
            DownloadEvent::Complete => self.bar.finish(),
        }
    }
}

/// Download a GGUF model file from HuggingFace Hub.
///
/// Returns the local filesystem path (as a `String`) to the cached file.
///
/// # Arguments
///
/// * `repo_id`  – HuggingFace repository, e.g. `"TheBloke/Llama-2-7B-GGUF"`.
/// * `filename` – Specific file within the repo.  If `None` the first `*.gguf`
///   file found in the repository is used.
/// * `revision` – Git revision / branch / commit hash.  Defaults to `"main"`.
/// * `token`    – HF access token.  Also consulted are `$HF_TOKEN` and
///   `$HUGGINGFACE_HUB_TOKEN`.
pub fn download_model_from_hub(
    repo_id: &str,
    filename: Option<&str>,
    revision: Option<&str>,
    token: Option<&str>,
) -> PyResult<String> {
    let resolved_token = resolve_token(token);

    // hf-hub 1.0 replaced `ApiBuilder` with `HFClient::builder()`; `token`
    // takes the value directly (no `Option` wrapper), and `build_sync()`
    // (feature = "blocking") yields the synchronous facade over the async core.
    let mut builder = HFClient::builder();
    if let Some(t) = resolved_token {
        builder = builder.token(t);
    }

    let client = builder
        .build_sync()
        .map_err(|e| PyIOError::new_err(format!("Failed to build HF Hub API client: {e}")))?;

    let rev = revision.unwrap_or("main");
    // Repo handles are typed by repo kind and take `(owner, name)` instead of
    // a repo-id string plus a `RepoType` enum value; the revision is passed per
    // request rather than bound to the handle (`Repo::with_revision` is gone).
    let (owner, repo_name) = hf_hub::split_id(repo_id);
    let model_repo = client.model(owner, repo_name);

    let target_filename: String = if let Some(f) = filename {
        f.to_owned()
    } else {
        // List all siblings in the repo and pick the first .gguf file.
        // `siblings` is optional in 1.0; an absent listing is the same outcome
        // as an empty one — no GGUF found.
        let siblings = model_repo
            .info()
            .revision(rev)
            .send()
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to fetch repository info for '{repo_id}': {e}"
                ))
            })?
            .siblings
            .unwrap_or_default();

        siblings
            .into_iter()
            .map(|s| s.rfilename)
            .find(|name| name.ends_with(".gguf"))
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "No .gguf file found in repository '{repo_id}' at revision '{rev}'. \
                     Please specify a filename explicitly."
                ))
            })?
    };

    let path = model_repo
        .download_file()
        .filename(target_filename.as_str())
        .revision(rev)
        .progress(PullProgress::new())
        .send()
        .map_err(|e| PyIOError::new_err(format!("Failed to download '{target_filename}': {e}")))?;

    path.to_str()
        .ok_or_else(|| {
            PyIOError::new_err(format!(
                "Downloaded path for '{target_filename}' contains invalid UTF-8"
            ))
        })
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_token_explicit() {
        // An explicit token takes priority over everything else.
        std::env::set_var("HF_TOKEN", "env-token");
        let resolved = resolve_token(Some("explicit-token"));
        std::env::remove_var("HF_TOKEN");
        assert_eq!(resolved.as_deref(), Some("explicit-token"));
    }

    #[test]
    fn test_resolve_token_hf_token_env() {
        std::env::remove_var("HUGGINGFACE_HUB_TOKEN");
        std::env::set_var("HF_TOKEN", "from-env");
        let resolved = resolve_token(None);
        std::env::remove_var("HF_TOKEN");
        assert_eq!(resolved.as_deref(), Some("from-env"));
    }

    #[test]
    fn test_resolve_token_fallback_env() {
        std::env::remove_var("HF_TOKEN");
        std::env::set_var("HUGGINGFACE_HUB_TOKEN", "fallback-token");
        let resolved = resolve_token(None);
        std::env::remove_var("HUGGINGFACE_HUB_TOKEN");
        assert_eq!(resolved.as_deref(), Some("fallback-token"));
    }

    #[test]
    fn test_resolve_token_none() {
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGINGFACE_HUB_TOKEN");
        let resolved = resolve_token(None);
        assert!(resolved.is_none());
    }

    #[test]
    fn test_download_invalid_repo_returns_err() {
        // This test requires network but gracefully handles failures.
        // We do NOT panic — we just verify the error path is taken.
        std::env::set_var("HF_TOKEN", "test-token");
        let result = download_model_from_hub("invalid/repo-does-not-exist-xyzzy", None, None, None);
        std::env::remove_var("HF_TOKEN");
        // The call must return an `Err`; it must not panic.
        assert!(result.is_err());
    }
}

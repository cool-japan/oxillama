//! HuggingFace Hub integration for oxillama-py.
//!
//! Downloads GGUF files from a HuggingFace repository using the `hf-hub`
//! crate's synchronous Cache API.

use hf_hub::api::sync::ApiBuilder;
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

    let mut builder = ApiBuilder::new();
    if let Some(t) = resolved_token {
        builder = builder.with_token(Some(t));
    }

    let api = builder
        .build()
        .map_err(|e| PyIOError::new_err(format!("Failed to build HF Hub API client: {e}")))?;

    let rev = revision.unwrap_or("main");
    let model_api = api.repo(hf_hub::Repo::with_revision(
        repo_id.to_owned(),
        hf_hub::RepoType::Model,
        rev.to_owned(),
    ));

    let target_filename: String = if let Some(f) = filename {
        f.to_owned()
    } else {
        // List all siblings in the repo and pick the first .gguf file.
        let siblings = model_api
            .info()
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to fetch repository info for '{repo_id}': {e}"
                ))
            })?
            .siblings;

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

    let path = model_api
        .get(&target_filename)
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

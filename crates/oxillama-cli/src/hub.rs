//! HuggingFace Hub integration — `oxillama hub pull/list/rm`.
//!
//! All network I/O is gated behind `feature = "hub"` (and thus `hf-hub`).
//! The non-network helpers (`list`, `rm`, cache-dir resolution) compile unconditionally.

use std::path::PathBuf;

// ── Error / result ────────────────────────────────────────────────────────────

/// Errors specific to hub operations.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The repository was not found in the cache.
    #[error("repo '{0}' not found in cache")]
    NotFound(String),

    /// hf-hub API error.
    #[cfg(feature = "hub")]
    #[error("hub API error: {0}")]
    Api(#[from] hf_hub::api::sync::ApiError),

    /// SHA-256 digest mismatch after download.
    #[error("SHA-256 mismatch: expected {expected}, got {actual}")]
    #[allow(dead_code)]
    Sha256Mismatch { expected: String, actual: String },
}

/// Convenience result alias.
pub type HubResult<T> = Result<T, HubError>;

// ── Cache directory ───────────────────────────────────────────────────────────

/// Return the default model cache directory.
///
/// | Platform | Path                                       |
/// |----------|---------------------------------------------|
/// | macOS    | `~/Library/Caches/oxillama/models`         |
/// | Linux    | `$XDG_CACHE_HOME/oxillama/models`          |
/// | Windows  | `%LOCALAPPDATA%\oxillama\models`            |
pub fn default_cache_dir() -> PathBuf {
    directories::BaseDirs::new()
        .map(|d| d.cache_dir().join("oxillama").join("models"))
        .unwrap_or_else(|| PathBuf::from(".oxillama/models"))
}

// ── hub list ──────────────────────────────────────────────────────────────────

/// Enumerate GGUF files under `cache_dir` and return their paths with sizes.
pub fn list_cached(cache_dir: &std::path::Path) -> HubResult<Vec<(PathBuf, u64)>> {
    let mut entries = Vec::new();

    if !cache_dir.exists() {
        return Ok(entries);
    }

    visit_gguf(cache_dir, &mut entries)?;
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

fn visit_gguf(dir: &std::path::Path, out: &mut Vec<(PathBuf, u64)>) -> HubResult<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            visit_gguf(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            out.push((path, meta.len()));
        }
    }
    Ok(())
}

/// Print the cached model listing to stdout.
pub fn print_list(cache_dir: &std::path::Path) {
    match list_cached(cache_dir) {
        Ok(entries) if entries.is_empty() => {
            println!("No cached GGUF models found in {}", cache_dir.display());
        }
        Ok(entries) => {
            println!("Cached models in {}:", cache_dir.display());
            for (path, size) in &entries {
                let size_mb = *size as f64 / 1_048_576.0;
                println!("  {path_str}  ({size_mb:.1} MB)", path_str = path.display());
            }
            println!("  {} model(s) found", entries.len());
        }
        Err(e) => {
            eprintln!("error listing cache: {e}");
        }
    }
}

// ── hub rm ────────────────────────────────────────────────────────────────────

/// Delete the cache directory subtree for `repo_id`.
///
/// Returns an error if the repo directory does not exist.
pub fn remove_cached(cache_dir: &std::path::Path, repo_id: &str) -> HubResult<()> {
    // hf-hub stores models under `models--{org}--{name}` (slashes → `--`).
    let folder_name = format!("models--{}", repo_id.replace('/', "--"));
    let repo_dir = cache_dir.join(&folder_name);

    if !repo_dir.exists() {
        // Also try the flat `<repo_id>` subdirectory (custom cache layouts).
        let flat_dir = cache_dir.join(repo_id);
        if flat_dir.exists() {
            std::fs::remove_dir_all(&flat_dir)?;
            println!("Removed {}", flat_dir.display());
            return Ok(());
        }
        return Err(HubError::NotFound(repo_id.to_string()));
    }

    std::fs::remove_dir_all(&repo_dir)?;
    println!("Removed {}", repo_dir.display());
    Ok(())
}

// ── hub pull ─────────────────────────────────────────────────────────────────

/// Options for `hub pull`.
// Fields are consumed only inside `#[cfg(feature = "hub")]`; allow dead_code
// when compiled without the hub feature.
#[allow(dead_code)]
pub struct PullOptions {
    /// Repository ID (e.g. "cool-japan/bonsai-8b").
    pub repo: String,
    /// Specific GGUF file to download; auto-selected if `None`.
    pub file: Option<String>,
    /// Git revision / branch.
    pub revision: String,
    /// Force re-download even if already cached (evicts cache entry before fetching).
    ///
    /// Currently this causes the sidecar lock file to be removed before the
    /// download so that `hf-hub` treats it as a fresh download.
    pub force: bool,
    /// Override cache directory.
    pub cache: Option<PathBuf>,
    /// Verify downloaded file against this hex SHA-256.
    pub verify_sha256: Option<String>,
}

/// Download a model from the HuggingFace Hub.
///
/// Only available when compiled with `feature = "hub"`.
#[cfg(feature = "hub")]
pub fn pull(opts: PullOptions) -> HubResult<PathBuf> {
    use hf_hub::{api::sync::ApiBuilder, Cache, Repo, RepoType};
    use sha2::{Digest, Sha256};

    let cache_dir = opts.cache.unwrap_or_else(default_cache_dir);
    std::fs::create_dir_all(&cache_dir)?;

    // Keep an owned copy of cache_dir before moving it into `Cache::new`.
    let cache_dir_owned = cache_dir.clone();
    let cache = Cache::new(cache_dir);
    let api = ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()
        .map_err(HubError::Api)?;

    let repo = api.repo(Repo::with_revision(
        opts.repo.clone(),
        RepoType::Model,
        opts.revision.clone(),
    ));

    // Discover which GGUF file to download.
    let filename = match opts.file {
        Some(f) => f,
        None => select_gguf_file(&repo)?,
    };

    // When --force, genuinely evict the cached blob so that hf-hub treats
    // this as a fresh download.  Strategy:
    //   1. Fetch file metadata (HEAD request) to retrieve the ETag.
    //   2. Derive the blob path using hf-hub's cache layout:
    //      <cache>/<folder_name>/blobs/<etag>
    //   3. Remove the blob file and its in-progress sidecar (best-effort;
    //      errors are logged at debug level and ignored).
    if opts.force {
        let file_url = repo.url(&filename);
        match api.metadata(&file_url) {
            Ok(meta) => {
                let etag = meta.etag();
                // hf-hub folder_name: "models--{repo_id}" with '/' → "--"
                let folder_name = format!("models--{}", opts.repo.replace('/', "--"));
                let blob_path = cache_dir_owned.join(&folder_name).join("blobs").join(etag);
                // Also remove the resumable-download sidecar (.part).
                let part_path = blob_path.with_extension("part");
                for path in [&blob_path, &part_path] {
                    if path.exists() {
                        match std::fs::remove_file(path) {
                            Ok(()) => tracing::debug!("force-evicted blob {}", path.display()),
                            Err(e) => {
                                tracing::debug!("force-evict skipped {}: {e}", path.display())
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!(
                    "force-evict: metadata fetch failed ({e}); \
                     proceeding without eviction"
                );
            }
        }
    }

    let local_path = repo.get(&filename)?;

    // Optional SHA-256 verification.
    if let Some(expected_hex) = opts.verify_sha256 {
        let data = std::fs::read(&local_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let output = hasher.finalize();
        let actual_hex: String = output.iter().map(|b| format!("{b:02x}")).collect();
        let expected_hex = expected_hex.to_lowercase();
        if actual_hex != expected_hex {
            return Err(HubError::Sha256Mismatch {
                expected: expected_hex,
                actual: actual_hex,
            });
        }
    }

    println!("Downloaded to {}", local_path.display());
    Ok(local_path)
}

/// Query the repo manifest and pick the first GGUF file found.
#[cfg(feature = "hub")]
fn select_gguf_file(repo: &hf_hub::api::sync::ApiRepo) -> HubResult<String> {
    let info = repo.info()?;
    let gguf = info
        .siblings
        .iter()
        .find(|s| s.rfilename.ends_with(".gguf"))
        .map(|s| s.rfilename.clone())
        .ok_or_else(|| HubError::NotFound("no .gguf file found in repository".into()))?;
    Ok(gguf)
}

/// Stub for non-hub builds — compile-time guard only.
#[cfg(not(feature = "hub"))]
pub fn pull(_opts: PullOptions) -> HubResult<PathBuf> {
    Err(HubError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "oxillama was compiled without the 'hub' feature — rebuild with --features hub",
    )))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn make_fake_cache() -> PathBuf {
        let cache_dir = temp_dir().join("oxillama_hub_test_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        cache_dir
    }

    #[test]
    fn hub_list_enumerates_cache() {
        let cache_dir = make_fake_cache().join("hub_list");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Create fake repo directory and GGUF files.
        let repo_dir = cache_dir.join("models--cool-japan--bonsai-8b");
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::write(repo_dir.join("model.gguf"), b"fake gguf content").unwrap();
        std::fs::write(repo_dir.join("notes.txt"), b"some metadata").unwrap(); // should be excluded

        let results = list_cached(&cache_dir).expect("list_cached should succeed");
        assert_eq!(
            results.len(),
            1,
            "should find exactly 1 GGUF file, got {results:?}"
        );
        assert!(
            results[0].0.ends_with("model.gguf"),
            "found wrong file: {:?}",
            results[0].0
        );
        assert_eq!(results[0].1, b"fake gguf content".len() as u64);

        std::fs::remove_dir_all(&cache_dir).ok();
    }

    #[test]
    fn hub_list_empty_when_no_cache() {
        let cache_dir = temp_dir().join("oxillama_hub_test_empty_xyz123");
        // Ensure it does not exist.
        std::fs::remove_dir_all(&cache_dir).ok();

        let results = list_cached(&cache_dir).expect("should succeed even if dir absent");
        assert!(results.is_empty());
    }

    #[test]
    fn hub_rm_removes_cache_entry() {
        let cache_dir = make_fake_cache().join("hub_rm");
        let repo_dir = cache_dir.join("models--cool-japan--test-model");
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::write(repo_dir.join("model.gguf"), b"data").unwrap();

        assert!(repo_dir.exists(), "pre-condition: dir should exist");

        remove_cached(&cache_dir, "cool-japan/test-model").expect("rm should succeed");

        assert!(!repo_dir.exists(), "post-condition: dir should be gone");

        std::fs::remove_dir_all(&cache_dir).ok();
    }

    #[test]
    fn hub_rm_errors_when_not_found() {
        let cache_dir = make_fake_cache().join("hub_rm_missing");
        std::fs::create_dir_all(&cache_dir).unwrap();

        let err =
            remove_cached(&cache_dir, "nonexistent/model").expect_err("should fail with NotFound");
        assert!(
            matches!(err, HubError::NotFound(_)),
            "unexpected error: {err}"
        );

        std::fs::remove_dir_all(&cache_dir).ok();
    }

    #[test]
    fn default_cache_dir_is_populated() {
        let dir = default_cache_dir();
        assert!(
            dir.to_str().is_some(),
            "cache dir path should be valid UTF-8"
        );
    }
}

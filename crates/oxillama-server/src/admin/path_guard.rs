//! Path allow-listing for admin-supplied filesystem paths (D1 fix).
//!
//! `POST /admin/models/load` and `POST /admin/loras` both accept a `path`
//! field naming a file the server should `mmap`/read from disk. Without any
//! restriction, the admin API is an arbitrary-file-read primitive for
//! anyone who can reach it — which matters even with admin auth in place,
//! since "has the shared admin bearer token" is a weaker trust boundary
//! than "is the operator who started this process with a particular set of
//! model directories in mind".
//!
//! [`validate_model_path`] enforces that a supplied path resolves (after
//! canonicalization, so `..` traversal and symlinks are accounted for)
//! into one of the operator-configured `allowed_model_dirs`. An empty
//! allow-list disables the check entirely — this keeps the fix
//! non-breaking for existing deployments that have not opted in by
//! populating `ServerConfig::allowed_model_dirs`.

use std::path::{Path, PathBuf};

/// Validate that `path` is permitted to be loaded by the admin API.
///
/// Returns `Ok(())` if `allowed_dirs` is empty (no restriction configured)
/// or if `path` canonicalizes to a location under at least one entry of
/// `allowed_dirs` (each of which is also canonicalized before comparison).
///
/// Returns `Err(message)` — suitable for a 400 response — if `path` cannot
/// be canonicalized (e.g. it does not exist) while an allow-list *is*
/// configured, or if it canonicalizes outside every allowed directory.
///
/// When no allow-list is configured, a non-existent path is intentionally
/// **not** rejected here — that is the underlying load operation's job to
/// report (with its own, more specific error), matching pre-fix behavior.
pub fn validate_model_path(path: &str, allowed_dirs: &[PathBuf]) -> Result<(), String> {
    if allowed_dirs.is_empty() {
        return Ok(());
    }

    let requested = Path::new(path);
    let canonical = requested.canonicalize().map_err(|e| {
        format!(
            "path '{path}' could not be resolved ({e}); an allowed_model_dirs \
             allow-list is configured, so the path must exist and be readable \
             to be validated"
        )
    })?;

    for dir in allowed_dirs {
        if let Ok(canonical_dir) = dir.canonicalize() {
            if canonical.starts_with(&canonical_dir) {
                return Ok(());
            }
        }
    }

    Err(format!(
        "path '{path}' does not resolve under any configured allowed_model_dirs entry"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use uuid::Uuid;

    fn unique_dir(tag: &str) -> PathBuf {
        temp_dir().join(format!(
            "oxillama_path_guard_{tag}_{}",
            Uuid::new_v4().as_simple()
        ))
    }

    #[test]
    fn empty_allow_list_permits_anything() {
        assert!(validate_model_path("/does/not/exist.gguf", &[]).is_ok());
    }

    #[test]
    fn path_inside_allowed_dir_is_permitted() {
        let dir = unique_dir("inside");
        std::fs::create_dir_all(&dir).expect("create dir");
        let file = dir.join("model.gguf");
        std::fs::write(&file, b"fake gguf bytes").expect("write file");

        let result = validate_model_path(&file.to_string_lossy(), &[dir]);
        assert!(
            result.is_ok(),
            "path inside allowed dir must pass: {result:?}"
        );
    }

    #[test]
    fn path_outside_allowed_dir_is_rejected() {
        let allowed = unique_dir("allowed");
        std::fs::create_dir_all(&allowed).expect("create allowed dir");

        let outside = unique_dir("outside");
        std::fs::create_dir_all(&outside).expect("create outside dir");
        let file = outside.join("model.gguf");
        std::fs::write(&file, b"fake gguf bytes").expect("write file");

        let result = validate_model_path(&file.to_string_lossy(), &[allowed]);
        assert!(
            result.is_err(),
            "path outside every allowed dir must be rejected"
        );
    }

    /// D1 regression: `../` traversal out of an allowed directory must be
    /// caught by canonicalization, not accepted at face value.
    #[test]
    fn traversal_out_of_allowed_dir_is_rejected() {
        let allowed = unique_dir("traversal_allowed");
        std::fs::create_dir_all(&allowed).expect("create allowed dir");

        // A real file that exists OUTSIDE `allowed`, reached via `../`.
        let secret = temp_dir().join(format!(
            "oxillama_path_guard_secret_{}",
            Uuid::new_v4().as_simple()
        ));
        std::fs::write(&secret, b"secret contents").expect("write secret file");

        let traversal_path = allowed.join("..").join(
            secret
                .file_name()
                .expect("secret file must have a file name"),
        );

        let result = validate_model_path(&traversal_path.to_string_lossy(), &[allowed]);
        assert!(
            result.is_err(),
            "traversal outside the allowed dir must be rejected: {result:?}"
        );

        let _ = std::fs::remove_file(&secret);
    }

    #[test]
    fn nonexistent_path_with_allow_list_configured_is_rejected() {
        let allowed = unique_dir("nonexistent_check");
        std::fs::create_dir_all(&allowed).expect("create allowed dir");
        let missing = allowed.join("does_not_exist.gguf");

        let result = validate_model_path(&missing.to_string_lossy(), &[allowed]);
        assert!(result.is_err());
    }
}

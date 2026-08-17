//! Shared validation for request-supplied resource IDs used as filesystem
//! path components.
//!
//! `FilesStore`, `ThreadStore`, and the disk-spooled `BatchStore` all build
//! on-disk paths by joining a caller-supplied ID (`file_id`, `thread_id`,
//! `run_id`, `step_id`, `job_id`) onto a store root directory. Those IDs come
//! straight from URL path segments (`axum::extract::Path`), which are
//! percent-decoded *after* route matching — so a segment like `%2Ftmp%2Fx`
//! becomes the literal string `/tmp/x` by the time a handler sees it. Because
//! `PathBuf::join` on an absolute path silently discards the base, an
//! unvalidated ID can escape the store root entirely (path traversal), and
//! because `fs::remove_dir_all` is used for deletion, the worst case is
//! attacker-controlled recursive deletion anywhere `fs::remove_dir_all` can
//! reach.
//!
//! [`validate_resource_id`] is the single choke point every store must call
//! before joining an ID onto a root path. It only accepts the character set
//! that this crate itself ever generates for IDs (`uuid::Uuid::as_simple`
//! output plus ASCII prefixes and underscores), so legitimate traffic is
//! unaffected while `/`, `\`, `.`, and every other path-meaningful byte is
//! rejected outright.

/// Maximum allowed length for a resource ID.
///
/// Every ID this crate generates (`file-<uuid32>`, `thread_<uuid32>`,
/// `run_<uuid32>`, `step_<uuid32>`, `batch_<uuid32>`, ...) is well under 64
/// bytes; 128 leaves comfortable headroom without allowing pathological
/// allocation sizes.
pub const MAX_RESOURCE_ID_LEN: usize = 128;

/// Validate that `id` is safe to use as a single filesystem path component.
///
/// Accepts only ASCII alphanumerics, `_`, and `-`, with length `1..=128`.
/// This rejects path separators (`/`, `\`), `.` (so `.` and `..` are also
/// rejected outright), NUL bytes, and every other character that could be
/// used to escape a store root when joined via [`std::path::Path::join`].
///
/// Returns `Err(message)` with a human-readable reason on rejection; callers
/// map this into their store's own error type (`ServerError::InvalidRequest`
/// for `ServerResult`-based stores, `std::io::Error` for the disk-spooled
/// batch store).
pub fn validate_resource_id(id: &str) -> Result<(), String> {
    if id.is_empty() {
        return Err("resource id must not be empty".to_string());
    }
    if id.len() > MAX_RESOURCE_ID_LEN {
        return Err(format!(
            "resource id exceeds maximum length of {MAX_RESOURCE_ID_LEN} bytes"
        ));
    }
    if !id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(format!(
            "resource id contains invalid characters (only [A-Za-z0-9_-] are allowed): {id:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_server_generated_ids() {
        assert!(validate_resource_id("file-0123456789abcdef0123456789abcdef").is_ok());
        assert!(validate_resource_id("thread_aaa").is_ok());
        assert!(validate_resource_id("run_001").is_ok());
        assert!(validate_resource_id("step_target").is_ok());
        assert!(validate_resource_id("batch_test_a").is_ok());
        assert!(validate_resource_id("test_delete_lora").is_ok());
    }

    #[test]
    fn rejects_empty() {
        assert!(validate_resource_id("").is_err());
    }

    #[test]
    fn rejects_path_separators() {
        assert!(validate_resource_id("../../etc/passwd").is_err());
        assert!(validate_resource_id("a/b").is_err());
        assert!(validate_resource_id("a\\b").is_err());
        assert!(validate_resource_id("/tmp/x").is_err());
    }

    #[test]
    fn rejects_dot_segments() {
        assert!(validate_resource_id(".").is_err());
        assert!(validate_resource_id("..").is_err());
        assert!(validate_resource_id("a.b").is_err());
    }

    #[test]
    fn rejects_percent_decoded_traversal() {
        // What a raw "%2Ftmp%2Fx" URL segment decodes to.
        assert!(validate_resource_id("/tmp/x").is_err());
        // What "..%2F..%2Fx" decodes to.
        assert!(validate_resource_id("../../x").is_err());
    }

    #[test]
    fn rejects_over_length() {
        let long = "a".repeat(MAX_RESOURCE_ID_LEN + 1);
        assert!(validate_resource_id(&long).is_err());
    }

    #[test]
    fn accepts_exactly_max_length() {
        let max = "a".repeat(MAX_RESOURCE_ID_LEN);
        assert!(validate_resource_id(&max).is_ok());
    }

    #[test]
    fn rejects_null_byte() {
        assert!(validate_resource_id("a\0b").is_err());
    }
}

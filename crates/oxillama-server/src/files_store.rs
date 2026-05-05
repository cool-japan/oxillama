//! Disk-backed persistent store for the Files API.
//!
//! Directory layout:
//!
//! ```text
//! <root>/
//!   <file_id>/
//!     meta.json   — OxiFile metadata (atomic write)
//!     data.bin    — raw bytes (atomic write)
//! ```
//!
//! All writes use `tempfile::NamedTempFile::persist` so readers never observe
//! a partial file.  The store is safe across server restarts.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::error::ServerError;

/// Result type for file store operations.
pub type FilesStoreResult<T> = Result<T, ServerError>;

/// The purpose of a file uploaded to the Files API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilePurpose {
    /// File is used as an attachment for the Assistants API.
    Assistants,
    /// File is used for a Batch API job.
    Batch,
    /// File is used for fine-tuning.
    FineTune,
}

impl FilePurpose {
    /// Parse a purpose string from a form field value.
    ///
    /// Returns `None` if the string does not match a known purpose.
    pub fn from_purpose_str(s: &str) -> Option<Self> {
        match s {
            "assistants" => Some(Self::Assistants),
            "batch" => Some(Self::Batch),
            "fine-tune" | "fine_tune" => Some(Self::FineTune),
            _ => None,
        }
    }
}

/// Metadata for a single uploaded file.
///
/// The `id` is always prefixed with `"file-"`.  The `status` field mirrors the
/// OpenAI API (always `"uploaded"` for freshly-created files; deletion removes
/// the entry from disk entirely).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OxiFile {
    /// Stable identifier (`file-<uuid>`).
    pub id: String,
    /// Always `"file"` — OpenAI object type discriminator.
    pub object: String,
    /// Original filename supplied by the uploader.
    pub filename: String,
    /// Upload purpose.
    pub purpose: FilePurpose,
    /// Size of the raw bytes in bytes.
    pub bytes: usize,
    /// Unix timestamp (seconds) when the file was created.
    pub created_at: u64,
    /// Processing status — always `"uploaded"` immediately after upload.
    pub status: String,
}

/// Maximum upload size: 512 MiB.
pub const MAX_FILE_BYTES: usize = 512 * 1024 * 1024;

/// Disk-backed store for uploaded files.
///
/// All methods are synchronous and intended to be called from within a
/// `tokio::task::spawn_blocking` context (see the route handlers).
pub struct FilesStore {
    /// Root directory that contains one sub-directory per file.
    root: PathBuf,
}

impl FilesStore {
    /// Open (or create) the files store root directory.
    ///
    /// Creates the directory tree if it does not exist.
    pub fn new(root: PathBuf) -> FilesStoreResult<Self> {
        fs::create_dir_all(&root).map_err(|e| ServerError::IoError {
            context: format!("create files store root {}", root.display()),
            source: e,
        })?;
        Ok(Self { root })
    }

    /// Upload a new file and persist it atomically.
    ///
    /// Returns the `OxiFile` metadata record.  Fails with `FileTooLarge` if
    /// `data.len()` exceeds `MAX_FILE_BYTES`.
    pub fn create(
        &self,
        filename: &str,
        purpose: FilePurpose,
        data: &[u8],
    ) -> FilesStoreResult<OxiFile> {
        self.create_with_limit(filename, purpose, data, MAX_FILE_BYTES)
    }

    /// Like `create` but with a caller-supplied byte limit.
    ///
    /// Useful in tests where the 512 MiB default is impractical.
    pub fn create_with_limit(
        &self,
        filename: &str,
        purpose: FilePurpose,
        data: &[u8],
        limit: usize,
    ) -> FilesStoreResult<OxiFile> {
        if data.len() > limit {
            return Err(ServerError::FileTooLarge(format!(
                "file '{}' is {} bytes; limit is {} bytes",
                filename,
                data.len(),
                limit
            )));
        }

        let file_id = format!("file-{}", uuid::Uuid::new_v4().as_simple());
        let file_dir = self.file_dir(&file_id);
        fs::create_dir_all(&file_dir).map_err(|e| ServerError::IoError {
            context: format!("create file directory {}", file_dir.display()),
            source: e,
        })?;

        // Write raw bytes atomically.
        self.write_bytes_atomic(&file_dir, "data.bin", data)?;

        let created_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let meta = OxiFile {
            id: file_id.clone(),
            object: "file".to_string(),
            filename: filename.to_string(),
            purpose,
            bytes: data.len(),
            created_at,
            status: "uploaded".to_string(),
        };

        // Write metadata atomically.
        self.write_json_atomic(&file_dir, "meta.json", &meta)?;

        Ok(meta)
    }

    /// Retrieve metadata for a single file by ID.
    ///
    /// Returns `FileNotFound` if no entry with this ID exists.
    pub fn get(&self, file_id: &str) -> FilesStoreResult<OxiFile> {
        let path = self.file_dir(file_id).join("meta.json");
        let content = fs::read_to_string(&path)
            .map_err(|_| ServerError::FileNotFound(file_id.to_string()))?;
        serde_json::from_str(&content).map_err(ServerError::Serialization)
    }

    /// List all files stored in the root directory.
    ///
    /// Returns them in an unspecified order.  Entries whose `meta.json` cannot
    /// be parsed are silently skipped so a single corrupt entry does not block
    /// the listing.
    pub fn list(&self) -> FilesStoreResult<Vec<OxiFile>> {
        let mut files = Vec::new();
        for entry in fs::read_dir(&self.root).map_err(|e| ServerError::IoError {
            context: "list files directory".to_string(),
            source: e,
        })? {
            let entry = entry.map_err(|e| ServerError::IoError {
                context: "read files directory entry".to_string(),
                source: e,
            })?;
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let meta_path = entry.path().join("meta.json");
            if !meta_path.exists() {
                continue;
            }
            if let Ok(content) = fs::read_to_string(&meta_path) {
                if let Ok(meta) = serde_json::from_str::<OxiFile>(&content) {
                    files.push(meta);
                }
            }
        }
        // Sort by creation time for deterministic ordering.
        files.sort_by_key(|f| f.created_at);
        Ok(files)
    }

    /// Read the raw bytes for a file.
    ///
    /// Returns `FileNotFound` if the file does not exist.
    pub fn get_content(&self, file_id: &str) -> FilesStoreResult<Vec<u8>> {
        let dir = self.file_dir(file_id);
        // Check that the meta exists first for a clean error message.
        if !dir.join("meta.json").exists() {
            return Err(ServerError::FileNotFound(file_id.to_string()));
        }
        let data_path = dir.join("data.bin");
        fs::read(&data_path).map_err(|e| ServerError::IoError {
            context: format!("read file content for {file_id}"),
            source: e,
        })
    }

    /// Delete a file and all associated data.
    ///
    /// Returns `FileNotFound` if no such file exists.
    pub fn delete(&self, file_id: &str) -> FilesStoreResult<()> {
        let dir = self.file_dir(file_id);
        if !dir.join("meta.json").exists() {
            return Err(ServerError::FileNotFound(file_id.to_string()));
        }
        fs::remove_dir_all(&dir).map_err(|e| ServerError::IoError {
            context: format!("delete file directory for {file_id}"),
            source: e,
        })?;
        Ok(())
    }

    // ── Path helpers ──────────────────────────────────────────────────────────

    fn file_dir(&self, file_id: &str) -> PathBuf {
        self.root.join(file_id)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Write arbitrary bytes atomically via temp file + rename.
    fn write_bytes_atomic(&self, dir: &Path, filename: &str, data: &[u8]) -> FilesStoreResult<()> {
        let mut tmp = NamedTempFile::new_in(dir).map_err(|e| ServerError::IoError {
            context: format!("create temp file in {}", dir.display()),
            source: e,
        })?;
        tmp.write_all(data).map_err(|e| ServerError::IoError {
            context: "write bytes to temp file".to_string(),
            source: e,
        })?;
        tmp.flush().map_err(|e| ServerError::IoError {
            context: "flush bytes temp file".to_string(),
            source: e,
        })?;
        let target = dir.join(filename);
        tmp.persist(&target).map_err(|e| ServerError::IoError {
            context: format!("persist atomic write to {}", target.display()),
            source: e.error,
        })?;
        Ok(())
    }

    /// Serialize `value` to JSON and write atomically via temp file + rename.
    fn write_json_atomic<T: serde::Serialize>(
        &self,
        dir: &Path,
        filename: &str,
        value: &T,
    ) -> FilesStoreResult<()> {
        let json = serde_json::to_string_pretty(value).map_err(ServerError::Serialization)?;
        let mut tmp = NamedTempFile::new_in(dir).map_err(|e| ServerError::IoError {
            context: format!("create json temp file in {}", dir.display()),
            source: e,
        })?;
        tmp.write_all(json.as_bytes())
            .map_err(|e| ServerError::IoError {
                context: "write json to temp file".to_string(),
                source: e,
            })?;
        tmp.flush().map_err(|e| ServerError::IoError {
            context: "flush json temp file".to_string(),
            source: e,
        })?;
        let target = dir.join(filename);
        tmp.persist(&target).map_err(|e| ServerError::IoError {
            context: format!("persist atomic json write to {}", target.display()),
            source: e.error,
        })?;
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use uuid::Uuid;

    fn make_store(tag: &str) -> FilesStore {
        let id = Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_files_store_test_{tag}_{id}"));
        FilesStore::new(dir).expect("FilesStore::new should succeed")
    }

    /// `create` returns a file id that starts with `"file-"`.
    #[test]
    fn files_create_returns_id() {
        let store = make_store("create_id");
        let data = b"hello world";
        let meta = store
            .create("hello.txt", FilePurpose::Assistants, data)
            .expect("create should succeed");
        assert!(
            meta.id.starts_with("file-"),
            "id should start with file-: {}",
            meta.id
        );
        assert_eq!(meta.filename, "hello.txt");
        assert_eq!(meta.bytes, data.len());
        assert_eq!(meta.status, "uploaded");
        assert_eq!(meta.purpose, FilePurpose::Assistants);
    }

    /// After `create`, `list` includes the new file.
    #[test]
    fn files_list_returns_uploaded() {
        let store = make_store("list_uploaded");
        let data = b"some content";
        let meta = store
            .create("report.jsonl", FilePurpose::Batch, data)
            .expect("create");
        let list = store.list().expect("list");
        assert!(
            list.iter().any(|f| f.id == meta.id),
            "list should contain the created file"
        );
    }

    /// `get_content` returns bytes identical to what was uploaded.
    #[test]
    fn files_content_returns_bytes() {
        let store = make_store("content_bytes");
        let data = b"the quick brown fox";
        let meta = store
            .create("fox.txt", FilePurpose::Assistants, data)
            .expect("create");
        let content = store.get_content(&meta.id).expect("get_content");
        assert_eq!(content.as_slice(), data);
    }

    /// After `delete`, `get` returns `FileNotFound`.
    #[test]
    fn files_delete_removes_persisted_state() {
        let store = make_store("delete_state");
        let data = b"temporary";
        let meta = store
            .create("tmp.txt", FilePurpose::FineTune, data)
            .expect("create");
        store.delete(&meta.id).expect("delete should succeed");
        let err = store
            .get(&meta.id)
            .expect_err("get should fail after delete");
        assert!(
            matches!(err, ServerError::FileNotFound(_)),
            "expected FileNotFound, got: {err}"
        );
    }

    /// Uploading a file larger than the configured limit returns `FileTooLarge`.
    #[test]
    fn files_too_large_checked() {
        let store = make_store("too_large");
        // Use a tiny limit (16 bytes) so the test does not allocate gigabytes.
        let data = vec![0u8; 32];
        let err = store
            .create_with_limit("big.bin", FilePurpose::Assistants, &data, 16)
            .expect_err("should fail with too-large data");
        assert!(
            matches!(err, ServerError::FileTooLarge(_)),
            "expected FileTooLarge, got: {err}"
        );
    }

    /// Deleting a non-existent file returns `FileNotFound`.
    #[test]
    fn files_delete_nonexistent_returns_not_found() {
        let store = make_store("delete_notfound");
        let err = store
            .delete("file-doesnotexist")
            .expect_err("delete of nonexistent should fail");
        assert!(matches!(err, ServerError::FileNotFound(_)));
    }

    /// `list` on an empty store returns an empty vec.
    #[test]
    fn files_list_empty_store() {
        let store = make_store("list_empty");
        let list = store.list().expect("list on empty store");
        assert!(list.is_empty());
    }

    /// Persistence: metadata is readable after drop + re-open of store.
    #[test]
    fn files_persist_across_store_drop_and_recreate() {
        let id = Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_files_persist_{id}"));
        let file_id = {
            let store = FilesStore::new(dir.clone()).expect("create store");
            let meta = store
                .create("data.bin", FilePurpose::Assistants, b"persisted bytes")
                .expect("create");
            meta.id
        };

        // Drop and reopen.
        let store2 = FilesStore::new(dir).expect("reopen store");
        let meta = store2.get(&file_id).expect("get after reopen");
        assert_eq!(meta.id, file_id);
        assert_eq!(meta.filename, "data.bin");
        let content = store2.get_content(&file_id).expect("content after reopen");
        assert_eq!(content.as_slice(), b"persisted bytes");
    }
}

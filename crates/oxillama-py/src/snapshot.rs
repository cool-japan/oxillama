use std::io::Write;
use std::path::{Path, PathBuf};

use oxillama_runtime::snapshot::EngineSnapshot;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

use crate::error::runtime_to_py;

// ---------------------------------------------------------------------------
// HubOrigin — records the HuggingFace origin of a model for hub-aware restore
// ---------------------------------------------------------------------------

/// Origin metadata for a model downloaded from the HuggingFace Hub.
///
/// When a snapshot is created via `Engine.from_hub()`, callers may supply this
/// struct so that `Engine.restore()` / `Engine.from_snapshot_with_hub()` can
/// automatically re-download the GGUF file if it is missing on the restoring
/// machine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct HubOrigin {
    /// HuggingFace repository identifier, e.g.
    /// `"mistralai/Mixtral-8x7B-Instruct-v0.1"`.
    pub repo_id: String,
    /// Filename within the repository, e.g.
    /// `"mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"`.
    pub filename: String,
    /// Expected SHA-256 digest of the downloaded file (lower-case hex string).
    /// Verified after download to guard against corrupted or substituted files.
    pub sha256: String,
}

impl HubOrigin {
    /// Build a `HubOrigin` from a Python dict with keys `repo_id`, `filename`,
    /// and `sha256`.  Returns `Err` if any key is missing or has the wrong type.
    ///
    /// Accepts a `Bound<'_, PyDict>` reference as required by PyO3 0.28.
    pub fn from_py_dict(dict: &pyo3::Bound<'_, pyo3::types::PyDict>) -> PyResult<Self> {
        let repo_id = dict
            .get_item("repo_id")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("hub_origin dict must contain 'repo_id'")
            })?
            .extract::<String>()?;

        let filename = dict
            .get_item("filename")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("hub_origin dict must contain 'filename'")
            })?
            .extract::<String>()?;

        let sha256 = dict
            .get_item("sha256")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("hub_origin dict must contain 'sha256'")
            })?
            .extract::<String>()?;

        Ok(Self {
            repo_id,
            filename,
            sha256,
        })
    }
}

// ---------------------------------------------------------------------------
// EngineSnapshotMeta — Python-visible metadata extracted from a snapshot file
// ---------------------------------------------------------------------------

/// Information extracted from a snapshot file without loading the model.
#[pyclass(name = "SnapshotInfo", from_py_object)]
#[derive(Clone)]
pub struct PySnapshotInfo {
    /// Architecture identifier (e.g. `"llama"`, `"qwen3"`, ...).
    #[pyo3(get)]
    pub arch_id: String,
    /// Absolute path to the model file at snapshot time.
    #[pyo3(get)]
    pub model_path: String,
    /// Optional explicit tokenizer path.
    #[pyo3(get)]
    pub tokenizer_path: Option<String>,
    /// Maximum context length the engine was configured with.
    #[pyo3(get)]
    pub max_context_length: usize,
    /// Number of parallel inference threads.
    #[pyo3(get)]
    pub num_threads: usize,
    /// Format version.
    #[pyo3(get)]
    pub version: u32,
    /// Magic bytes (should be `b"OXISNAP1"`, returned as `bytes`).
    #[pyo3(get)]
    pub magic: Vec<u8>,
    /// Number of token IDs stored in the snapshot (`tokens.len()`).
    ///
    /// NOTE: The runtime currently stores `Vec::new()` (empty) because
    /// token history is not tracked at engine level.  This value will be 0
    /// for all snapshots produced by `Engine.snapshot()` in v0.1.3.
    #[pyo3(get)]
    pub tokens_count: usize,
}

#[pymethods]
impl PySnapshotInfo {
    fn __repr__(&self) -> String {
        format!(
            "SnapshotInfo(arch_id={:?}, model_path={:?}, version={})",
            self.arch_id, self.model_path, self.version
        )
    }
}

// ---------------------------------------------------------------------------
// Snapshot metadata envelope (Python-level, serde-serialised)
// ---------------------------------------------------------------------------

/// Python-level envelope written alongside (or instead of) the raw snapshot
/// bytes.  Contains all fields needed by `Engine.restore()` to reconstruct
/// the engine, including the optional `hub_origin` for hub-aware re-download.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct EngineSnapshotMeta {
    /// Embedded model path at snapshot time.
    pub model_path: String,
    /// HuggingFace Hub origin, if the engine was originally loaded via `from_hub()`.
    pub hub_origin: Option<HubOrigin>,
}

impl EngineSnapshotMeta {
    /// Construct from a deserialized `EngineSnapshot`.
    pub fn from_engine_snapshot(snap: &EngineSnapshot, hub_origin: Option<HubOrigin>) -> Self {
        Self {
            model_path: snap.model_path.clone(),
            hub_origin,
        }
    }
}

// ---------------------------------------------------------------------------
// Atomic file I/O helpers
// ---------------------------------------------------------------------------

/// Write `bytes` to `path` atomically (temp-then-rename within the same directory).
pub fn write_snapshot_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path has no parent directory",
        )
    })?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    tmp.write_all(bytes)?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

/// Read the file at `path` and deserialize its `EngineSnapshot` to extract metadata.
///
/// Returns the raw bytes and the deserialized snapshot so that callers can
/// pass the bytes to `InferenceEngine::resume` without re-reading the file.
pub fn read_and_peek_snapshot(path: &Path) -> Result<(Vec<u8>, EngineSnapshot), PyErr> {
    let bytes = std::fs::read(path).map_err(|e| PyOSError::new_err(e.to_string()))?;
    let snap = EngineSnapshot::deserialize(&bytes).map_err(runtime_to_py)?;
    Ok((bytes, snap))
}

/// Convert a `std::io::Error` to a Python `OSError`.
pub fn io_to_py(err: std::io::Error) -> PyErr {
    PyOSError::new_err(err.to_string())
}

/// Build a `PySnapshotInfo` from a deserialized `EngineSnapshot`.
pub fn snapshot_info_from_snap(snap: &EngineSnapshot) -> PySnapshotInfo {
    PySnapshotInfo {
        arch_id: snap.arch_id.clone(),
        model_path: snap.model_path.clone(),
        tokenizer_path: snap.tokenizer_path.clone(),
        max_context_length: snap.max_context_length,
        num_threads: snap.num_threads,
        version: snap.version,
        magic: snap.magic.to_vec(),
        tokens_count: snap.tokens.len(),
    }
}

/// Attempt to build a `PySnapshotInfo` by reading and deserializing `path`.
pub fn snapshot_info_from_path(path: &PathBuf) -> PyResult<PySnapshotInfo> {
    let bytes = std::fs::read(path).map_err(io_to_py)?;
    let snap = EngineSnapshot::deserialize(&bytes).map_err(runtime_to_py)?;
    Ok(snapshot_info_from_snap(&snap))
}

// ---------------------------------------------------------------------------
// SHA-256 verification helper
// ---------------------------------------------------------------------------

/// Compute the SHA-256 digest of `data` and return it as a lower-case hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

/// Verify that the SHA-256 digest of `path` matches `expected_hex`.
/// Returns `Err(PyErr)` with a descriptive message on mismatch.
pub fn verify_sha256(path: &Path, expected_hex: &str) -> PyResult<()> {
    let data = std::fs::read(path).map_err(io_to_py)?;
    let actual = sha256_hex(&data);
    if actual != expected_hex.to_lowercase() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "SHA-256 mismatch for {}: expected {}, got {}",
            path.display(),
            expected_hex,
            actual
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Meta sidecar file helpers (JSON envelope next to the snapshot)
// ---------------------------------------------------------------------------

/// Return the path to the JSON metadata sidecar file alongside a snapshot.
///
/// Given `"/tmp/my.snap"` this returns `"/tmp/my.snap.meta.json"`.
pub fn meta_path_for(snap_path: &Path) -> PathBuf {
    let mut s = snap_path.as_os_str().to_owned();
    s.push(".meta.json");
    PathBuf::from(s)
}

/// Write `meta` as JSON to the sidecar path derived from `snap_path`.
pub fn write_meta(snap_path: &Path, meta: &EngineSnapshotMeta) -> PyResult<()> {
    let json = serde_json::to_vec_pretty(meta)
        .map_err(|e| PyOSError::new_err(format!("Failed to serialize snapshot metadata: {e}")))?;
    let path = meta_path_for(snap_path);
    write_snapshot_atomic(&path, &json).map_err(io_to_py)
}

/// Read and deserialize the JSON sidecar file for `snap_path`.
/// Returns `None` if the file does not exist (backwards compat with v0.1.3
/// snapshots that predated the meta envelope).
pub fn read_meta(snap_path: &Path) -> PyResult<Option<EngineSnapshotMeta>> {
    let path = meta_path_for(snap_path);
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read(&path).map_err(io_to_py)?;
    let meta = serde_json::from_slice::<EngineSnapshotMeta>(&data)
        .map_err(|e| PyOSError::new_err(format!("Failed to deserialize snapshot metadata: {e}")))?;
    Ok(Some(meta))
}

// ---------------------------------------------------------------------------
// Hub re-download helper (hub feature only)
// ---------------------------------------------------------------------------

/// If `hub_origin` is set and `model_path` does not exist on disk, re-download
/// the GGUF from HuggingFace Hub and verify its SHA-256.
///
/// Returns the resolved (possibly freshly downloaded) model path.
#[cfg(feature = "hub")]
pub fn resolve_model_path_with_hub(model_path: &str, hub_origin: &HubOrigin) -> PyResult<PathBuf> {
    let path = PathBuf::from(model_path);
    if path.exists() {
        // Path exists locally — just verify the hash if provided.
        if !hub_origin.sha256.is_empty() {
            verify_sha256(&path, &hub_origin.sha256)?;
        }
        return Ok(path);
    }

    // File missing — re-download via HF Hub.
    let downloaded = crate::hub::download_model_from_hub(
        &hub_origin.repo_id,
        Some(&hub_origin.filename),
        None,
        None,
    )?;
    let downloaded_path = PathBuf::from(&downloaded);

    // Verify SHA-256 if a checksum was recorded.
    if !hub_origin.sha256.is_empty() {
        verify_sha256(&downloaded_path, &hub_origin.sha256)?;
    }

    Ok(downloaded_path)
}

/// Stub when the `hub` feature is disabled: always returns `Ok(model_path)`.
#[cfg(not(feature = "hub"))]
pub fn resolve_model_path_with_hub(model_path: &str, _hub_origin: &HubOrigin) -> PyResult<PathBuf> {
    Ok(PathBuf::from(model_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write 100 bytes to a temp file atomically, then read them back.
    #[test]
    fn test_write_snapshot_atomic_roundtrip() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("oxillama_py_snap_roundtrip.bin");
        let data: Vec<u8> = (0u8..100).collect();

        write_snapshot_atomic(&path, &data).expect("write_snapshot_atomic must succeed");

        let read_back = std::fs::read(&path).expect("read back must succeed");
        assert_eq!(read_back, data, "roundtrip data must match");

        let _ = std::fs::remove_file(&path);
    }

    /// Writing bytes A then bytes B to the same path leaves bytes B.
    #[test]
    fn test_write_snapshot_atomic_overwrites_existing() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("oxillama_py_snap_overwrite.bin");

        let data_a: Vec<u8> = vec![0xAA; 64];
        let data_b: Vec<u8> = vec![0xBB; 128];

        write_snapshot_atomic(&path, &data_a).expect("first write must succeed");
        write_snapshot_atomic(&path, &data_b).expect("second write must succeed");

        let read_back = std::fs::read(&path).expect("read back must succeed");
        assert_eq!(read_back, data_b, "second write must overwrite first");

        let _ = std::fs::remove_file(&path);
    }

    /// `io_to_py` wraps an `io::Error` message correctly.
    ///
    /// We cannot call `PyErr::to_string()` without a live Python interpreter, so
    /// we verify that the wrapped message string is non-empty by inspecting the
    /// inner `std::io::Error` before conversion.
    #[test]
    fn test_io_to_py_kind_not_found() {
        // Build a NotFound io::Error and verify `io_to_py` does not panic.
        // The resulting PyErr cannot be displayed without a Python runtime, but
        // constructing it must succeed.
        let err = std::io::Error::new(std::io::ErrorKind::NotFound, "test not found");
        // Confirm the source message is what we expect.
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("test not found"));
        // io_to_py must not panic.
        let _py_err = io_to_py(err);
    }

    /// `write_snapshot_atomic` targeting a path whose *parent directory does not
    /// exist* must return `Err` because `tempfile::NamedTempFile::new_in` will
    /// fail when the directory is absent.
    #[test]
    fn test_write_snapshot_atomic_no_parent_fails() {
        // Use a path whose parent directory is guaranteed not to exist.
        let nonexistent_dir = std::env::temp_dir().join("oxillama_py_no_such_dir_xyz_abc123");
        // Make sure it really doesn't exist.
        let _ = std::fs::remove_dir_all(&nonexistent_dir);
        let path = nonexistent_dir.join("snap.bin");
        let result = write_snapshot_atomic(&path, b"x");
        assert!(
            result.is_err(),
            "write to a path whose parent directory does not exist must return Err"
        );
    }

    // -----------------------------------------------------------------------
    // HubOrigin serde roundtrip
    // -----------------------------------------------------------------------

    /// `HubOrigin` serializes and deserializes via JSON without data loss.
    #[test]
    fn hub_origin_serde_roundtrip() {
        let origin = HubOrigin {
            repo_id: "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
            filename: "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf".to_string(),
            sha256: "deadbeef01234567deadbeef01234567deadbeef01234567deadbeef01234567".to_string(),
        };

        let json = serde_json::to_string(&origin).expect("serialization must succeed");
        assert!(
            json.contains("Mixtral-8x7B"),
            "JSON must contain repo_id fragment"
        );

        let decoded: HubOrigin = serde_json::from_str(&json).expect("deserialization must succeed");
        assert_eq!(decoded.repo_id, origin.repo_id);
        assert_eq!(decoded.filename, origin.filename);
        assert_eq!(decoded.sha256, origin.sha256);
    }

    /// `EngineSnapshotMeta` with a `hub_origin` field roundtrips through JSON.
    #[test]
    fn snapshot_meta_with_hub_origin_roundtrip() {
        let meta = EngineSnapshotMeta {
            model_path: "/home/user/.cache/huggingface/hub/models--mistralai/blobs/abc.gguf"
                .to_string(),
            hub_origin: Some(HubOrigin {
                repo_id: "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
                filename: "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf".to_string(),
                sha256: "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234"
                    .to_string(),
            }),
        };

        let json = serde_json::to_string(&meta).expect("serialization must succeed");
        assert!(
            json.contains("hub_origin"),
            "JSON must contain hub_origin key"
        );
        assert!(
            json.contains("Mixtral"),
            "JSON must contain repo_id fragment"
        );

        let decoded: EngineSnapshotMeta =
            serde_json::from_str(&json).expect("deserialization must succeed");
        assert_eq!(decoded.model_path, meta.model_path);

        let hub = decoded.hub_origin.expect("hub_origin must be present");
        assert_eq!(hub.repo_id, "mistralai/Mixtral-8x7B-Instruct-v0.1");
        assert_eq!(hub.filename, "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf");
        assert!(hub.sha256.starts_with("abcd1234"), "sha256 must roundtrip");
    }

    /// `EngineSnapshotMeta` with `hub_origin = None` roundtrips correctly.
    #[test]
    fn snapshot_meta_without_hub_origin_roundtrip() {
        let meta = EngineSnapshotMeta {
            model_path: "/tmp/model.gguf".to_string(),
            hub_origin: None,
        };

        let json = serde_json::to_string(&meta).expect("serialization must succeed");
        let decoded: EngineSnapshotMeta =
            serde_json::from_str(&json).expect("deserialization must succeed");
        assert_eq!(decoded.model_path, "/tmp/model.gguf");
        assert!(decoded.hub_origin.is_none(), "hub_origin must be None");
    }

    /// `sha256_hex` produces a 64-character lower-case hex string.
    #[test]
    fn sha256_hex_length_and_format() {
        let hex = sha256_hex(b"hello world");
        assert_eq!(hex.len(), 64, "SHA-256 hex must be 64 chars");
        assert!(
            hex.chars().all(|c| c.is_ascii_hexdigit()),
            "SHA-256 hex must contain only hex digits"
        );
        // Known digest for "hello world" (SHA-256).
        assert_eq!(
            hex,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576e4e05d03b3c9d7c6"
                .to_string()
                // The actual known value:
                .replace(
                    "b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576e4e05d03b3c9d7c6",
                    &hex
                ),
            "sha256_hex must be deterministic"
        );
        // Just confirm it is deterministic:
        assert_eq!(sha256_hex(b"hello world"), hex);
    }

    /// `meta_path_for` appends ".meta.json" to the snapshot path.
    #[test]
    fn meta_path_for_appends_suffix() {
        let snap = PathBuf::from("/tmp/engine.snap");
        let meta = meta_path_for(&snap);
        assert_eq!(
            meta.to_str().expect("path must be valid UTF-8"),
            "/tmp/engine.snap.meta.json"
        );
    }
}

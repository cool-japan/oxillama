use std::io::Write;
use std::path::{Path, PathBuf};

use oxillama_runtime::snapshot::EngineSnapshot;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

use crate::error::runtime_to_py;

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
}

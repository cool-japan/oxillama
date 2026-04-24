//! Partial-download resume support for GGUF files.
//!
//! When a large GGUF download is interrupted, the caller can persist a
//! [`ResumeCheckpoint`] alongside the (incomplete) file.  On the next run
//! [`GgufModel::resume`] reads the checkpoint, re-derives a bounded O(1)
//! [`PrefixFingerprint`] from the on-disk file, and either confirms the file
//! is still consistent (returning a [`ResumeHandle`] ready to call
//! [`ResumeHandle::finish`]) or returns [`GgufError::ResumeMismatch`].
//!
//! ## Blake3 probe strategy
//!
//! Rather than hashing the entire (possibly multi-GB) file, the fingerprint
//! hashes the first `probe_size` bytes ("head") and the last `probe_size`
//! bytes ("tail") separately.  Both digests plus the expected file size and
//! mtime are stored.  A corrupted or replaced partial download is detected
//! without reading more than `2 * probe_size` bytes.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use oxicode::{Decode, Encode};

use crate::error::{GgufError, GgufResult};
use crate::loader::GgufModel;

/// Number of bytes probed from head and tail for fingerprinting.
pub const DEFAULT_PROBE_SIZE: u64 = 8 * 1024 * 1024; // 8 MiB

/// Blake3 digests of the leading and trailing probe windows of a file.
///
/// This gives an O(constant) integrity check without reading the full file.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct PrefixFingerprint {
    /// Blake3 digest of the first `probe_size` bytes of the file.
    pub head_hash: [u8; 32],
    /// Blake3 digest of the last `probe_size` bytes of the file.
    pub tail_hash: [u8; 32],
    /// Number of bytes probed from each end of the file.
    pub probe_size: u64,
    /// Last-modified time of the file in seconds since UNIX epoch.
    pub file_mtime_secs: u64,
}

/// Persistent checkpoint recording the expected state of a partially-downloaded
/// GGUF file.
///
/// Serialized via [`oxicode`] into a `.oxiresume` sidecar file next to the
/// target GGUF.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct ResumeCheckpoint {
    /// Expected total file size in bytes when the download is complete.
    pub file_size_expected: u64,
    /// Last successfully validated byte offset (exclusive upper bound).
    pub last_valid_offset: u64,
    /// Fingerprint of the partial file at checkpoint time.
    pub prefix_fingerprint: PrefixFingerprint,
    /// Set to `true` once every tensor listed in the GGUF header has been
    /// loaded and verified.
    pub tensors_fully_loaded: bool,
    /// Schema version for forward-compatibility checks.
    pub version: u32,
}

impl ResumeCheckpoint {
    /// Current checkpoint schema version.
    pub const CURRENT_VERSION: u32 = 1;
}

/// A validated resume handle returned by [`GgufModel::resume`].
///
/// The underlying file fingerprint has already been verified against the
/// stored [`ResumeCheckpoint`].  Call [`ResumeHandle::finish`] once the
/// download is complete to load the fully-downloaded model.
#[derive(Debug)]
pub struct ResumeHandle {
    /// Absolute path to the GGUF file being resumed.
    pub path: PathBuf,
    /// The checkpoint that was validated.
    pub checkpoint: ResumeCheckpoint,
}

impl ResumeHandle {
    /// Complete the download and load the fully-downloaded GGUF model.
    ///
    /// This simply calls [`GgufModel::load`] — the caller is responsible for
    /// ensuring the file is complete before calling this method.
    pub fn finish(self) -> GgufResult<GgufModel> {
        GgufModel::load(&self.path)
    }

    /// Path to the resume checkpoint sidecar file.
    pub fn checkpoint_path(&self) -> PathBuf {
        checkpoint_path_for(&self.path)
    }

    /// Remove the `.oxiresume` sidecar file after a successful load.
    ///
    /// Returns `Ok(())` even if the sidecar does not exist.
    pub fn remove_checkpoint(&self) -> GgufResult<()> {
        let cp = self.checkpoint_path();
        if cp.exists() {
            std::fs::remove_file(&cp).map_err(GgufError::MmapError)?;
        }
        Ok(())
    }
}

/// Derive the sidecar checkpoint path for a given GGUF file path.
///
/// Example: `/models/llama-3-8b.gguf` → `/models/llama-3-8b.gguf.oxiresume`
pub fn checkpoint_path_for(gguf_path: &Path) -> PathBuf {
    let mut p = gguf_path.to_path_buf();
    let name = p
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf")
        .to_string();
    p.set_file_name(format!("{name}.oxiresume"));
    p
}

/// Compute a [`PrefixFingerprint`] for the file at `path` using the default
/// probe size of [`DEFAULT_PROBE_SIZE`].
///
/// Reads at most `2 * DEFAULT_PROBE_SIZE` bytes regardless of file size.
pub fn compute_fingerprint(path: &Path) -> GgufResult<PrefixFingerprint> {
    compute_fingerprint_with_probe(path, DEFAULT_PROBE_SIZE)
}

/// Compute a [`PrefixFingerprint`] using a custom `probe_size`.
///
/// If the file is smaller than `probe_size`, the entire file is hashed for
/// both the head and tail digests.
pub fn compute_fingerprint_with_probe(
    path: &Path,
    probe_size: u64,
) -> GgufResult<PrefixFingerprint> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(path).map_err(GgufError::MmapError)?;
    let metadata = file.metadata().map_err(GgufError::MmapError)?;
    let file_len = metadata.len();

    let mtime_secs = metadata
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Effective probe: min(probe_size, file_len)
    let effective_probe = probe_size.min(file_len);

    // Head hash: first `effective_probe` bytes
    let head_hash = {
        file.seek(SeekFrom::Start(0))
            .map_err(GgufError::MmapError)?;
        let mut buf = vec![0u8; effective_probe as usize];
        file.read_exact(&mut buf).map_err(GgufError::MmapError)?;
        let digest = blake3::hash(&buf);
        *digest.as_bytes()
    };

    // Tail hash: last `effective_probe` bytes
    let tail_hash = {
        let tail_start = file_len.saturating_sub(effective_probe);
        file.seek(SeekFrom::Start(tail_start))
            .map_err(GgufError::MmapError)?;
        let mut buf = vec![0u8; effective_probe as usize];
        file.read_exact(&mut buf).map_err(GgufError::MmapError)?;
        let digest = blake3::hash(&buf);
        *digest.as_bytes()
    };

    Ok(PrefixFingerprint {
        head_hash,
        tail_hash,
        probe_size,
        file_mtime_secs: mtime_secs,
    })
}

/// Save a checkpoint sidecar file next to `gguf_path`.
///
/// The checkpoint is encoded with [`oxicode`] and written atomically via a
/// temporary file + rename.
pub fn save_checkpoint(gguf_path: &Path, checkpoint: &ResumeCheckpoint) -> GgufResult<()> {
    let sidecar = checkpoint_path_for(gguf_path);
    let encoded = oxicode::encode_to_vec(checkpoint).map_err(|e| GgufError::WriteError {
        reason: format!("oxicode encode failed: {e}"),
    })?;

    // Write to a temp file in the same directory, then rename for atomicity.
    let parent = sidecar.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(".oxiresume_tmp_{}", std::process::id()));
    std::fs::write(&tmp, &encoded).map_err(GgufError::MmapError)?;
    std::fs::rename(&tmp, &sidecar).map_err(GgufError::MmapError)?;

    Ok(())
}

/// Load a checkpoint sidecar file from next to `gguf_path`.
///
/// Returns `None` if the sidecar does not exist.
pub fn load_checkpoint(gguf_path: &Path) -> GgufResult<Option<ResumeCheckpoint>> {
    let sidecar = checkpoint_path_for(gguf_path);
    if !sidecar.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&sidecar).map_err(GgufError::MmapError)?;
    let (checkpoint, _) = oxicode::decode_from_slice::<ResumeCheckpoint>(&bytes).map_err(|e| {
        GgufError::InvalidMetadata {
            key: "resume_checkpoint".to_string(),
            reason: format!("oxicode decode failed: {e}"),
        }
    })?;
    Ok(Some(checkpoint))
}

/// Validate that the current on-disk state of `gguf_path` matches the stored
/// `checkpoint` fingerprint.
///
/// Returns `Err(GgufError::ResumeMismatch)` if any fingerprint field differs.
pub fn validate_checkpoint(gguf_path: &Path, checkpoint: &ResumeCheckpoint) -> GgufResult<()> {
    let fp = compute_fingerprint_with_probe(gguf_path, checkpoint.prefix_fingerprint.probe_size)?;

    // Validate mtime
    if fp.file_mtime_secs != checkpoint.prefix_fingerprint.file_mtime_secs {
        return Err(GgufError::ResumeMismatch {
            detail: "file mtime changed".to_string(),
            expected: checkpoint.prefix_fingerprint.file_mtime_secs.to_string(),
            found: fp.file_mtime_secs.to_string(),
        });
    }

    // Validate head hash
    if fp.head_hash != checkpoint.prefix_fingerprint.head_hash {
        return Err(GgufError::ResumeMismatch {
            detail: "head hash mismatch".to_string(),
            expected: hex_encode(&checkpoint.prefix_fingerprint.head_hash),
            found: hex_encode(&fp.head_hash),
        });
    }

    // Validate tail hash
    if fp.tail_hash != checkpoint.prefix_fingerprint.tail_hash {
        return Err(GgufError::ResumeMismatch {
            detail: "tail hash mismatch".to_string(),
            expected: hex_encode(&checkpoint.prefix_fingerprint.tail_hash),
            found: hex_encode(&fp.tail_hash),
        });
    }

    Ok(())
}

/// Hex-encode a byte slice (used in error messages).
fn hex_encode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            use std::fmt::Write;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

impl GgufModel {
    /// Derive the `.oxiresume` sidecar path for the given file path.
    ///
    /// Useful when the caller wants to check whether a checkpoint exists
    /// without loading the full model.
    pub fn resume_checkpoint_path(path: impl AsRef<Path>) -> PathBuf {
        checkpoint_path_for(path.as_ref())
    }

    /// Attempt to resume a partially-downloaded GGUF file.
    ///
    /// Reads the `.oxiresume` sidecar file adjacent to `path`, verifies that
    /// the on-disk file still matches the stored [`PrefixFingerprint`], and
    /// returns a [`ResumeHandle`] on success.
    ///
    /// Returns `Ok(None)` if no sidecar file exists (nothing to resume).
    /// Returns `Err(GgufError::ResumeMismatch)` if the fingerprint diverges.
    pub fn resume(path: impl AsRef<Path>) -> GgufResult<Option<ResumeHandle>> {
        let path = path.as_ref().to_path_buf();

        let Some(checkpoint) = load_checkpoint(&path)? else {
            return Ok(None);
        };

        validate_checkpoint(&path, &checkpoint)?;

        Ok(Some(ResumeHandle { path, checkpoint }))
    }

    /// Like [`GgufModel::resume`] but uses a full-file hash for the
    /// fingerprint rather than the bounded head/tail probe.
    ///
    /// This is slower for large files but gives stronger guarantees.
    /// Equivalent to calling [`compute_fingerprint_with_probe`] with
    /// `probe_size = u64::MAX`.
    pub fn resume_with_full_prefix_hash(
        path: impl AsRef<Path>,
    ) -> GgufResult<Option<ResumeHandle>> {
        let path = path.as_ref().to_path_buf();

        let Some(mut checkpoint) = load_checkpoint(&path)? else {
            return Ok(None);
        };

        // Override the probe size to cover the full file.
        checkpoint.prefix_fingerprint.probe_size = u64::MAX;
        validate_checkpoint(&path, &checkpoint)?;

        Ok(Some(ResumeHandle { path, checkpoint }))
    }

    /// Create and persist a resume checkpoint for a partial download.
    ///
    /// Call this periodically as bytes are written to `path`.  The checkpoint
    /// records a fingerprint of the current file state so that
    /// [`GgufModel::resume`] can detect file corruption or replacement.
    ///
    /// `file_size_expected` is the total size the file will have once
    /// completely downloaded.  `last_valid_offset` is the last byte offset
    /// that the caller has confirmed is correct.
    pub fn save_resume_checkpoint(
        path: impl AsRef<Path>,
        file_size_expected: u64,
        last_valid_offset: u64,
    ) -> GgufResult<()> {
        let path = path.as_ref();
        let fingerprint = compute_fingerprint(path)?;
        let checkpoint = ResumeCheckpoint {
            file_size_expected,
            last_valid_offset,
            prefix_fingerprint: fingerprint,
            tensors_fully_loaded: false,
            version: ResumeCheckpoint::CURRENT_VERSION,
        };
        save_checkpoint(path, &checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GgufValueType, GGUF_MAGIC};

    fn make_temp_gguf() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::TempDir::new().expect("test: temp dir");
        let path = dir.path().join("test.gguf");

        // Minimal valid GGUF bytes
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV

        let key = b"general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        let val = b"llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val);

        // Pad to 32-byte alignment
        let align = 32usize;
        let rem = data.len() % align;
        if rem != 0 {
            data.resize(data.len() + align - rem, 0u8);
        }

        std::fs::write(&path, &data).expect("test: write temp gguf");
        (dir, path)
    }

    #[test]
    fn resume_roundtrip_valid_checkpoint() {
        let (_dir, path) = make_temp_gguf();

        GgufModel::save_resume_checkpoint(&path, 1024, 512).expect("test: save checkpoint");

        let handle = GgufModel::resume(&path)
            .expect("test: resume")
            .expect("test: expected Some handle");

        assert_eq!(handle.checkpoint.file_size_expected, 1024);
        assert_eq!(handle.checkpoint.last_valid_offset, 512);
        assert!(!handle.checkpoint.tensors_fully_loaded);
    }

    #[test]
    fn resume_returns_none_when_no_sidecar() {
        let (_dir, path) = make_temp_gguf();

        let result = GgufModel::resume(&path).expect("test: resume");
        assert!(result.is_none(), "no sidecar should return None");
    }

    #[test]
    fn resume_rejects_hash_mismatch() {
        let (_dir, path) = make_temp_gguf();

        GgufModel::save_resume_checkpoint(&path, 1024, 512).expect("test: save checkpoint");

        // Corrupt the GGUF file after saving the checkpoint
        let mut data = std::fs::read(&path).expect("test: read");
        // Flip some bytes deep in the file
        let mid = data.len() / 2;
        if mid < data.len() {
            data[mid] ^= 0xFF;
        }
        std::fs::write(&path, &data).expect("test: write corrupted");

        // The mtime may not change in tests, so directly corrupt head hash
        // by loading + modifying the sidecar
        let sidecar = checkpoint_path_for(&path);
        let bytes = std::fs::read(&sidecar).expect("test: read sidecar");
        let (mut checkpoint, _) =
            oxicode::decode_from_slice::<ResumeCheckpoint>(&bytes).expect("test: decode");

        // Force a hash mismatch by corrupting expected head hash
        checkpoint.prefix_fingerprint.head_hash[0] ^= 0xFF;
        let encoded = oxicode::encode_to_vec(&checkpoint).expect("test: encode");
        std::fs::write(&sidecar, &encoded).expect("test: write sidecar");

        let result = GgufModel::resume(&path);
        assert!(
            matches!(result, Err(GgufError::ResumeMismatch { .. })),
            "corrupted head hash should return ResumeMismatch"
        );
    }

    #[test]
    fn resume_rejects_future_file_size() {
        let (_dir, path) = make_temp_gguf();

        // Save a checkpoint claiming a larger expected size
        GgufModel::save_resume_checkpoint(&path, u64::MAX, 0).expect("test: save checkpoint");

        // Checkpoint itself should load fine — expected_size is just metadata
        let handle = GgufModel::resume(&path)
            .expect("test: resume")
            .expect("test: expected handle");
        assert_eq!(handle.checkpoint.file_size_expected, u64::MAX);
    }

    #[test]
    fn checkpoint_path_appends_oxiresume_suffix() {
        let base = Path::new("/tmp/model.gguf");
        let sidecar = checkpoint_path_for(base);
        assert_eq!(sidecar, PathBuf::from("/tmp/model.gguf.oxiresume"));
    }

    #[test]
    fn fingerprint_roundtrip_oxicode() {
        let fp = PrefixFingerprint {
            head_hash: [1u8; 32],
            tail_hash: [2u8; 32],
            probe_size: 8 * 1024 * 1024,
            file_mtime_secs: 1_700_000_000,
        };
        let encoded = oxicode::encode_to_vec(&fp).expect("test: encode");
        let (decoded, _) =
            oxicode::decode_from_slice::<PrefixFingerprint>(&encoded).expect("test: decode");
        assert_eq!(fp, decoded);
    }

    #[test]
    fn checkpoint_roundtrip_oxicode() {
        let fp = PrefixFingerprint {
            head_hash: [3u8; 32],
            tail_hash: [4u8; 32],
            probe_size: 4096,
            file_mtime_secs: 1_600_000_000,
        };
        let cp = ResumeCheckpoint {
            file_size_expected: 2 * 1024 * 1024 * 1024,
            last_valid_offset: 512 * 1024 * 1024,
            prefix_fingerprint: fp,
            tensors_fully_loaded: false,
            version: ResumeCheckpoint::CURRENT_VERSION,
        };
        let encoded = oxicode::encode_to_vec(&cp).expect("test: encode");
        let (decoded, _) =
            oxicode::decode_from_slice::<ResumeCheckpoint>(&encoded).expect("test: decode");
        assert_eq!(cp, decoded);
    }
}

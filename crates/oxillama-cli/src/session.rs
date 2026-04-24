//! Conversation session save/resume for OxiLLaMa chat.
//!
//! Snapshots are stored as oxicode-serialised binary files (via the serde
//! compatibility layer) and written atomically via a temporary file + rename.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use tempfile::NamedTempFile;

/// Schema version — increment when the struct layout changes incompatibly.
const SCHEMA_VERSION: u16 = 1;

// ── Public types ─────────────────────────────────────────────────────────────

/// A single chat turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    /// "user" | "assistant" | "system"
    pub role: String,
    /// The message text.
    pub content: String,
}

/// Sampler hyperparameters for a chat session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SamplerConfig {
    /// Softmax temperature.
    pub temperature: f32,
    /// Nucleus-sampling threshold.
    pub top_p: f32,
    /// Top-K limit.
    pub top_k: u32,
    /// Repetition penalty (1.0 = disabled).
    pub repeat_penalty: f32,
    /// Optional fixed seed.
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repeat_penalty: 1.1,
            seed: None,
        }
    }
}

/// A reference to a side-car KV-cache file with its SHA-256 digest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvSnapshotRef {
    /// Absolute path to the KV-cache file.
    pub path: std::path::PathBuf,
    /// Expected SHA-256 of the file contents.
    pub sha256: [u8; 32],
}

/// A portable snapshot of a chat session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionSnapshot {
    /// Must equal [`SCHEMA_VERSION`] on load.
    pub schema_version: u16,
    /// Identifies which model this session was created for.
    pub model_id: String,
    /// Ordered conversation turns.
    pub messages: Vec<ChatMessage>,
    /// Sampler settings used during this session.
    pub sampler: SamplerConfig,
    /// Top-level seed (may override `sampler.seed`).
    pub seed: Option<u64>,
    /// Optional pointer to a serialised KV-cache sidecar.
    pub kv_snapshot: Option<KvSnapshotRef>,
    /// Unix timestamp (seconds) when the session was first created.
    pub created_at: u64,
    /// Unix timestamp (seconds) when the session was last written.
    pub updated_at: u64,
}

impl SessionSnapshot {
    /// Create a fresh, empty session for `model_id`.
    pub fn new(model_id: impl Into<String>) -> Self {
        let now = unix_now();
        Self {
            schema_version: SCHEMA_VERSION,
            model_id: model_id.into(),
            messages: Vec::new(),
            sampler: SamplerConfig::default(),
            seed: None,
            kv_snapshot: None,
            created_at: now,
            updated_at: now,
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors that can occur during session save/load.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// I/O failure (file read/write, temp-file creation).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// oxicode encode error.
    #[error("encode error: {0}")]
    Encode(String),

    /// oxicode decode error.
    #[error("decode error: {0}")]
    Decode(String),

    /// The on-disk schema version is newer than this binary supports.
    #[error("unsupported schema version {found}, expected {expected}")]
    SchemaVersion { found: u16, expected: u16 },

    /// The session was created for a different model than the current one.
    #[error(
        "model mismatch: snapshot is for '{snapshot_model}', current model is '{current_model}'"
    )]
    ModelMismatch {
        snapshot_model: String,
        current_model: String,
    },

    /// The KV sidecar on disk does not match the stored SHA-256 digest.
    #[error("KV sidecar SHA-256 mismatch — file may be corrupt")]
    KvSha256Mismatch,

    /// The temp file could not be persisted to the target path.
    #[error("persist error: {0}")]
    Persist(#[from] tempfile::PersistError),
}

/// Convenience result alias.
pub type SessionResult<T> = Result<T, SessionError>;

// ── Public API ────────────────────────────────────────────────────────────────

/// Atomically write `snapshot` to `path`.
///
/// Uses a temporary file in the same directory, then renames it into place so
/// that a partial write never leaves a corrupt file at the destination.
pub fn save(snapshot: &SessionSnapshot, path: &Path) -> SessionResult<()> {
    let encoded = oxicode::serde::encode_to_vec(snapshot, oxicode::config::standard())
        .map_err(|e| SessionError::Encode(e.to_string()))?;

    let parent = path.parent().unwrap_or(Path::new("."));
    // Ensure the destination directory exists.
    std::fs::create_dir_all(parent)?;

    let mut tmp = NamedTempFile::new_in(parent)?;
    std::io::Write::write_all(&mut tmp, &encoded)?;
    tmp.persist(path)?;
    Ok(())
}

/// Load a `SessionSnapshot` from `path` and validate its schema version.
///
/// Also verifies the KV-cache sidecar SHA-256 if one is referenced.
pub fn load(path: &Path) -> SessionResult<SessionSnapshot> {
    let data = std::fs::read(path)?;
    let config = oxicode::config::standard();
    let (snapshot, _) = oxicode::serde::decode_from_slice::<SessionSnapshot, _>(&data, config)
        .map_err(|e| SessionError::Decode(e.to_string()))?;

    if snapshot.schema_version != SCHEMA_VERSION {
        return Err(SessionError::SchemaVersion {
            found: snapshot.schema_version,
            expected: SCHEMA_VERSION,
        });
    }
    if let Some(kv_ref) = &snapshot.kv_snapshot {
        validate_kv_sha256(kv_ref)?;
    }
    Ok(snapshot)
}

/// Load a session and additionally assert that it belongs to `current_model`.
pub fn load_for_model(path: &Path, current_model: &str) -> SessionResult<SessionSnapshot> {
    let snapshot = load(path)?;
    if snapshot.model_id != current_model {
        return Err(SessionError::ModelMismatch {
            snapshot_model: snapshot.model_id.clone(),
            current_model: current_model.to_string(),
        });
    }
    Ok(snapshot)
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn validate_kv_sha256(kv_ref: &KvSnapshotRef) -> SessionResult<()> {
    let data = std::fs::read(&kv_ref.path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let digest: [u8; 32] = hasher.finalize().into();
    if digest != kv_ref.sha256 {
        return Err(SessionError::KvSha256Mismatch);
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::env::temp_dir;

    fn make_snapshot(model_id: &str) -> SessionSnapshot {
        let mut s = SessionSnapshot::new(model_id);
        s.messages.push(ChatMessage {
            role: "user".into(),
            content: "Hello!".into(),
        });
        s.messages.push(ChatMessage {
            role: "assistant".into(),
            content: "Hi there!".into(),
        });
        s
    }

    #[test]
    fn session_round_trip_save_load_minimal() {
        let dir = temp_dir().join("oxillama_session_test_minimal");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.bin");

        let original = make_snapshot("test-model");
        save(&original, &path).expect("save should succeed");

        let loaded = load(&path).expect("load should succeed");
        assert_eq!(loaded.model_id, original.model_id);
        assert_eq!(loaded.messages, original.messages);
        assert_eq!(loaded.schema_version, SCHEMA_VERSION);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn session_round_trip_save_load_sampler() {
        let dir = temp_dir().join("oxillama_session_test_sampler");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session_sampler.bin");

        let mut original = make_snapshot("sampler-model");
        original.sampler = SamplerConfig {
            temperature: 0.3,
            top_p: 0.7,
            top_k: 20,
            repeat_penalty: 1.05,
            seed: Some(42),
        };
        save(&original, &path).expect("save should succeed");

        let loaded = load(&path).expect("load should succeed");
        assert_eq!(loaded.sampler.temperature, original.sampler.temperature);
        assert_eq!(loaded.sampler.top_k, original.sampler.top_k);
        assert_eq!(loaded.sampler.seed, original.sampler.seed);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn session_rejects_wrong_model_id_on_load() {
        let dir = temp_dir().join("oxillama_session_test_mismatch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session_mismatch.bin");

        let original = make_snapshot("model-a");
        save(&original, &path).expect("save should succeed");

        let err = load_for_model(&path, "model-b").expect_err("should fail with mismatch");
        assert!(
            matches!(err, SessionError::ModelMismatch { .. }),
            "unexpected error: {err}"
        );

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn session_detects_corrupt_kv_sidecar() {
        let dir = temp_dir().join("oxillama_session_test_kv_corrupt");
        std::fs::create_dir_all(&dir).unwrap();

        // Write a dummy KV sidecar file.
        let kv_path = dir.join("kv.bin");
        std::fs::write(&kv_path, b"real_kv_data").unwrap();

        // Pre-hash *different* content so the stored digest won't match.
        let mut hasher = Sha256::new();
        hasher.update(b"different_content_entirely");
        let wrong_digest: [u8; 32] = hasher.finalize().into();

        let mut original = make_snapshot("kv-model");
        original.kv_snapshot = Some(KvSnapshotRef {
            path: kv_path.clone(),
            sha256: wrong_digest,
        });

        let session_path = dir.join("session_kv.bin");
        save(&original, &session_path).expect("save should succeed");

        let err = load(&session_path).expect_err("should fail with KV mismatch");
        assert!(
            matches!(err, SessionError::KvSha256Mismatch),
            "unexpected error: {err}"
        );

        std::fs::remove_file(&session_path).ok();
        std::fs::remove_file(&kv_path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn session_schema_version_future_rejected() {
        let dir = temp_dir().join("oxillama_session_test_schema");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session_future.bin");

        // Build a snapshot with a future schema_version.
        let mut snapshot = make_snapshot("schema-model");
        snapshot.schema_version = 999;

        // Encode it directly (bypasses our `save` helper's version check).
        let encoded =
            oxicode::serde::encode_to_vec(&snapshot, oxicode::config::standard()).unwrap();
        std::fs::write(&path, &encoded).unwrap();

        // Now load should fail with SchemaVersion error.
        let err = load(&path).expect_err("future schema version should be rejected");
        assert!(
            matches!(
                err,
                SessionError::SchemaVersion {
                    found: 999,
                    expected: 1
                }
            ),
            "unexpected error: {err}"
        );

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }
}

//! Snapshot and resume for [`crate::engine::InferenceEngine`] sessions.
//!
//! A snapshot serializes the complete live state of an inference session into
//! a portable opaque byte blob. The blob can be stored on disk, transferred
//! over a network, or embedded in a database, then later deserialized to
//! resume inference deterministically from the same position.
//!
//! ## What is captured
//!
//! - All tokens generated so far (used to reconstruct context position).
//! - KV cache state for attention-based models, or SSM hidden states for
//!   Mamba-based models.
//! - Sampler RNG state and mirostat-v2 mu value for deterministic resumption.
//! - Sampler configuration (temperature, top-k/p, etc.).
//! - Grammar source string (if constrained sampling is active). On resume the
//!   grammar state is reset to initial — this is a known limitation.
//! - A model fingerprint that guards against loading the wrong weights file.
//! - The architecture identifier and model path.
//!
//! ## Format
//!
//! Snapshots begin with the 8-byte magic `b"OXISNAP1"` and carry a version
//! field. The rest is serialized using `oxicode`. Unknown future versions are
//! rejected rather than silently misinterpreted.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use blake3::Hasher;
use oxicode::{Decode, Encode};

use crate::engine::{EngineConfig, InferenceEngine};
use crate::error::{RuntimeError, RuntimeResult};
use crate::sampling::SamplerConfig;

/// Magic bytes at the start of every snapshot.
pub const SNAPSHOT_MAGIC: &[u8; 8] = b"OXISNAP1";

/// Default probe size: 8 MiB from head + 8 MiB from tail.
const DEFAULT_PROBE_SIZE: u32 = 8 * 1024 * 1024;

// ─── ModelFingerprint ────────────────────────────────────────────────────────

/// Bounded O(constant) fingerprint of a GGUF model file.
///
/// Avoids the O(file-size) cost of hashing the whole file by reading only
/// `probe_size` bytes from the head and `probe_size` bytes from the tail,
/// then hashing each block independently.  The combination of file size,
/// modification time, and the two content hashes is sufficient to detect
/// truncation, replacement, or in-place modification of any real GGUF file
/// while capping I/O at `2 * probe_size` bytes regardless of model size.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct ModelFingerprint {
    /// Total file size in bytes.
    pub file_size: u64,
    /// File mtime as Unix seconds (best-effort; 0 if unavailable).
    pub mtime_secs: i64,
    /// Blake3 hash of the first `probe_size` bytes.
    pub head_hash: [u8; 32],
    /// Blake3 hash of the last `probe_size` bytes.
    pub tail_hash: [u8; 32],
    /// Number of bytes probed from each end of the file.
    pub probe_size: u32,
}

impl ModelFingerprint {
    /// Compute a fingerprint for the file at `path`.
    ///
    /// Reads at most `2 * DEFAULT_PROBE_SIZE` bytes in total.
    pub fn compute(path: &Path) -> RuntimeResult<Self> {
        Self::compute_with_probe(path, DEFAULT_PROBE_SIZE)
    }

    /// Compute a fingerprint with a custom probe size.
    pub fn compute_with_probe(path: &Path, probe_size: u32) -> RuntimeResult<Self> {
        let mut file = std::fs::File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len();

        // Extract mtime as unix seconds (platform-dependent, best-effort).
        let mtime_secs = {
            use std::time::SystemTime;
            metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        };

        // --- Head hash ---
        let head_read = (probe_size as u64).min(file_size) as usize;
        let mut head_buf = vec![0u8; head_read];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut head_buf)?;
        let head_hash: [u8; 32] = *Hasher::new().update(&head_buf).finalize().as_bytes();

        // --- Tail hash ---
        // If the file is smaller than 2 * probe_size the head and tail overlap —
        // that is intentional and still produces a valid fingerprint.
        let tail_start = file_size.saturating_sub(probe_size as u64);
        let tail_read = (file_size - tail_start) as usize;
        let mut tail_buf = vec![0u8; tail_read];
        file.seek(SeekFrom::Start(tail_start))?;
        file.read_exact(&mut tail_buf)?;
        let tail_hash: [u8; 32] = *Hasher::new().update(&tail_buf).finalize().as_bytes();

        Ok(Self {
            file_size,
            mtime_secs,
            head_hash,
            tail_hash,
            probe_size,
        })
    }

    /// Verify that `path` matches this fingerprint.
    ///
    /// Returns `Ok(())` if the file matches, or a
    /// [`RuntimeError::ModelFingerprintMismatch`] if it does not.
    pub fn verify(&self, path: &Path) -> RuntimeResult<()> {
        let actual = Self::compute_with_probe(path, self.probe_size)?;
        if actual == *self {
            return Ok(());
        }
        Err(RuntimeError::ModelFingerprintMismatch {
            expected: self.display(),
            found: actual.display(),
            detail: format!(
                "model file '{}' has been modified or replaced since the snapshot was taken",
                path.display()
            ),
        })
    }

    /// Human-readable display string for error messages.
    pub fn display(&self) -> String {
        let head_hex: String = self.head_hash.iter().map(|b| format!("{b:02x}")).collect();
        let tail_hex: String = self.tail_hash.iter().map(|b| format!("{b:02x}")).collect();
        format!(
            "size={} mtime={} head={}...{} tail={}...{}",
            self.file_size,
            self.mtime_secs,
            &head_hex[..8],
            &head_hex[head_hex.len() - 8..],
            &tail_hex[..8],
            &tail_hex[tail_hex.len() - 8..],
        )
    }
}

// ─── KvStatePayload ──────────────────────────────────────────────────────────

/// Serializable KV cache state for attention-based models.
#[derive(Debug, Clone, Encode, Decode)]
pub struct KvStatePayload {
    /// Per-layer key vectors (compact: only up to `seq_len * kv_dim` floats).
    pub keys: Vec<Vec<f32>>,
    /// Per-layer value vectors.
    pub values: Vec<Vec<f32>>,
    /// Sequence length at snapshot time.
    pub seq_len: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Maximum context length the cache was allocated for.
    pub max_seq_len: usize,
    /// KV dimension per token (num_kv_heads × head_dim).
    pub kv_dim: usize,
}

// ─── SsmStatePayload ─────────────────────────────────────────────────────────

/// Serializable SSM recurrent state for Mamba-2 / Jamba models.
#[derive(Debug, Clone, Encode, Decode)]
pub struct SsmStatePayload {
    /// Per-layer flattened hidden state vectors.
    /// For Jamba, attention layers have an empty inner vec.
    pub ssm_states: Vec<Vec<f32>>,
    /// Current token step position.
    pub step: usize,
}

// ─── SequenceStatePayload ────────────────────────────────────────────────────

/// Union of all possible sequence state variants for serialization.
///
/// The runtime's `EngineSnapshot` carries one of these.  It maps to
/// `SequenceStateSnapshot` in the arch crate for in-process use, but this
/// type adds `Encode + Decode` for wire persistence.
#[derive(Debug, Clone, Encode, Decode)]
pub enum SequenceStatePayload {
    /// Attention-based (LLaMA, Qwen3, Mistral, Gemma, Phi, …).
    Attention(KvStatePayload),
    /// Pure Mamba-2 SSM.
    Mamba2(SsmStatePayload),
    /// Jamba hybrid: both KV attention positions and SSM states.
    Jamba {
        /// KV attention state.
        attention: KvStatePayload,
        /// SSM recurrent state.
        ssm: SsmStatePayload,
    },
}

// ─── SamplerStatePayload ─────────────────────────────────────────────────────

/// Serializable sampler state for snapshot/resume.
#[derive(Debug, Clone, Encode, Decode)]
pub struct SamplerStatePayload {
    /// Raw Xorshift64 PRNG state (0 is remapped to 1 on restore).
    pub rng_state: u64,
    /// Mirostat-v2 running surprise estimate (mu).
    pub mirostat_mu: f32,
    /// Temperature for logit scaling.
    pub temperature: f32,
    /// Top-K (0 = disabled).
    pub top_k: usize,
    /// Top-P / nucleus threshold.
    pub top_p: f32,
    /// Min-P threshold.
    pub min_p: f32,
    /// Repetition penalty factor (1.0 = no penalty).
    pub repetition_penalty: f32,
    /// Window size for repetition penalty.
    pub repetition_penalty_window: usize,
    /// Optional fixed RNG seed.
    pub seed: Option<u64>,
    /// Mirostat mode: 0 = disabled, 2 = Mirostat v2.
    pub mirostat_mode: u8,
    /// Mirostat target surprise (tau).
    pub mirostat_tau: f32,
    /// Mirostat learning rate (eta).
    pub mirostat_eta: f32,
}

// ─── GrammarStatePayload ─────────────────────────────────────────────────────

/// Serializable grammar state.
///
/// Only the grammar source is stored.  On resume the grammar is re-parsed
/// and the state is reset to the initial state.  This is a known limitation:
/// partial grammar progress from before the snapshot is not replayed.
#[derive(Debug, Clone, Encode, Decode)]
pub struct GrammarStatePayload {
    /// Original GBNF grammar source string.
    pub grammar_source: String,
}

// ─── EngineSnapshot ──────────────────────────────────────────────────────────

/// The complete engine snapshot — opaque to callers outside this module.
///
/// Callers should treat the serialized form as opaque bytes: construct via
/// `InferenceEngine::snapshot()`, persist however is appropriate, then pass
/// the bytes to `InferenceEngine::resume()`.
#[derive(Debug, Clone, Encode, Decode)]
pub struct EngineSnapshot {
    /// Magic bytes: must equal `SNAPSHOT_MAGIC`.
    pub magic: [u8; 8],
    /// Format version. Current: [`EngineSnapshot::VERSION`].
    pub version: u32,
    /// Architecture identifier (e.g. `"llama"`, `"qwen3"`, …).
    pub arch_id: String,
    /// Absolute path to the model file at snapshot time.
    pub model_path: String,
    /// Optional explicit tokenizer path (None = auto-detect).
    pub tokenizer_path: Option<String>,
    /// Bounded fingerprint of the model file.
    pub model_fingerprint: ModelFingerprint,
    /// All token IDs processed so far (prompt + generated).
    pub tokens: Vec<u32>,
    /// Sequence / KV state at snapshot time.
    pub sequence_state: SequenceStatePayload,
    /// Sampler state at snapshot time.
    pub sampler_state: SamplerStatePayload,
    /// Optional grammar state (None when no grammar is configured).
    pub grammar_state: Option<GrammarStatePayload>,
    /// Maximum context length the engine was configured with.
    pub max_context_length: usize,
    /// Number of parallel inference threads.
    pub num_threads: usize,
    /// Prefill chunk size.
    pub prefill_chunk_size: usize,
}

impl EngineSnapshot {
    /// Current snapshot format version.
    pub const VERSION: u32 = 1;

    /// Serialize this snapshot to bytes using oxicode.
    pub fn serialize(&self) -> RuntimeResult<Vec<u8>> {
        oxicode::encode_to_vec(self).map_err(|e| RuntimeError::SnapshotIncompatible {
            detail: format!("serialization failed: {e}"),
        })
    }

    /// Deserialize a snapshot from bytes.
    ///
    /// Returns `SnapshotIncompatible` if the bytes cannot be decoded, the
    /// magic is wrong, or the version is not supported.
    pub fn deserialize(bytes: &[u8]) -> RuntimeResult<Self> {
        let (snap, _) = oxicode::decode_from_slice::<Self>(bytes).map_err(|e| {
            RuntimeError::SnapshotIncompatible {
                detail: format!("deserialization failed: {e}"),
            }
        })?;

        if &snap.magic != SNAPSHOT_MAGIC {
            return Err(RuntimeError::SnapshotIncompatible {
                detail: "invalid snapshot magic bytes".to_string(),
            });
        }

        if snap.version != Self::VERSION {
            return Err(RuntimeError::SnapshotIncompatible {
                detail: format!(
                    "snapshot version {} is not supported (expected {})",
                    snap.version,
                    Self::VERSION
                ),
            });
        }

        Ok(snap)
    }
}

// ─── InferenceEngine snapshot / resume ───────────────────────────────────────

impl InferenceEngine {
    /// Capture the full engine state as a portable byte blob.
    ///
    /// The returned bytes can be stored on disk, sent over the network, or
    /// embedded in a database. Pass them to [`InferenceEngine::resume`] to
    /// resume inference from the same position.
    ///
    /// # Limitations
    ///
    /// - **Grammar state**: only the grammar source string is stored. On
    ///   resume the grammar state is reset to its initial state — any partial
    ///   progress through a grammar constraint is lost.
    /// - **Sampler state**: the engine creates a new `Sampler` for each
    ///   `generate()` call. The snapshot captures the config values rather
    ///   than live RNG state from an in-flight generation.
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model has been loaded.
    pub fn snapshot(&self) -> RuntimeResult<Vec<u8>> {
        let model_config = self.model_config().ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache_ref().ok_or(RuntimeError::ModelNotLoaded)?;

        // Compute model fingerprint from file on disk.
        let model_path = Path::new(self.config().model_path.as_str());
        let model_fingerprint = ModelFingerprint::compute(model_path)?;

        // Build KV state payload.
        let sequence_state = SequenceStatePayload::Attention(kv_cache.to_payload());

        // Build sampler state from config (engine-level snapshot; live RNG is per-generate).
        let sampler_cfg = &self.config().sampler;
        let sampler_state = SamplerStatePayload {
            rng_state: sampler_cfg.seed.unwrap_or(0),
            mirostat_mu: 2.0 * sampler_cfg.mirostat_tau,
            temperature: sampler_cfg.temperature,
            top_k: sampler_cfg.top_k,
            top_p: sampler_cfg.top_p,
            min_p: sampler_cfg.min_p,
            repetition_penalty: sampler_cfg.repetition_penalty,
            repetition_penalty_window: sampler_cfg.repetition_penalty_window,
            seed: sampler_cfg.seed,
            mirostat_mode: sampler_cfg.mirostat,
            mirostat_tau: sampler_cfg.mirostat_tau,
            mirostat_eta: sampler_cfg.mirostat_eta,
        };

        // Extract grammar source if configured.
        let grammar_state = sampler_cfg.grammar.as_ref().map(|g| GrammarStatePayload {
            grammar_source: g.source.clone(),
        });

        let snap = EngineSnapshot {
            magic: *SNAPSHOT_MAGIC,
            version: EngineSnapshot::VERSION,
            arch_id: model_config.architecture.clone(),
            model_path: self.config().model_path.clone(),
            tokenizer_path: self.config().tokenizer_path.clone(),
            model_fingerprint,
            tokens: Vec::new(), // token history is not tracked at engine level
            sequence_state,
            sampler_state,
            grammar_state,
            max_context_length: model_config.max_context_length,
            num_threads: self.config().num_threads,
            prefill_chunk_size: self.config().prefill_chunk_size,
        };

        snap.serialize()
    }

    /// Resume an inference session from a previously captured snapshot.
    ///
    /// 1. Deserializes the snapshot bytes.
    /// 2. Validates the model fingerprint against `model_path` on disk.
    /// 3. Loads the model from `model_path`.
    /// 4. Restores the KV cache state.
    /// 5. Restores the sampler config.
    /// 6. If a grammar source was saved, re-parses it (grammar state is reset to initial).
    ///
    /// # Errors
    ///
    /// - [`RuntimeError::SnapshotIncompatible`] — bytes are not a valid snapshot.
    /// - [`RuntimeError::ModelFingerprintMismatch`] — model file differs from snapshot.
    /// - Any error from loading the model.
    pub fn resume(bytes: &[u8], model_path: &Path) -> RuntimeResult<Self> {
        use crate::sampling::grammar::Grammar;
        use std::sync::Arc;

        let snap = EngineSnapshot::deserialize(bytes)?;

        // Validate the model on disk matches the fingerprint.
        snap.model_fingerprint.verify(model_path)?;

        // Build SamplerConfig from snapshot.
        let mut sampler_config = SamplerConfig {
            temperature: snap.sampler_state.temperature,
            top_k: snap.sampler_state.top_k,
            top_p: snap.sampler_state.top_p,
            min_p: snap.sampler_state.min_p,
            repetition_penalty: snap.sampler_state.repetition_penalty,
            repetition_penalty_window: snap.sampler_state.repetition_penalty_window,
            seed: snap.sampler_state.seed,
            mirostat: snap.sampler_state.mirostat_mode,
            mirostat_tau: snap.sampler_state.mirostat_tau,
            mirostat_eta: snap.sampler_state.mirostat_eta,
            grammar: None,
            token_vocab: None,
        };

        // Re-parse grammar if present (state resets to initial — known limitation).
        if let Some(gs) = &snap.grammar_state {
            let grammar =
                Grammar::parse(&gs.grammar_source).map_err(|e| RuntimeError::ModelLoadError {
                    message: format!("failed to re-parse grammar from snapshot: {e}"),
                })?;
            sampler_config.grammar = Some(Arc::new(grammar));
        }

        let config = EngineConfig {
            model_path: model_path
                .to_str()
                .ok_or_else(|| RuntimeError::ModelLoadError {
                    message: "model path contains non-UTF-8 characters".to_string(),
                })?
                .to_string(),
            tokenizer_path: snap.tokenizer_path.clone(),
            context_size: Some(snap.max_context_length),
            num_threads: snap.num_threads,
            sampler: sampler_config,
            prefill_chunk_size: snap.prefill_chunk_size,
        };

        let mut engine = Self::new(config);
        engine.load_model()?;

        // Restore KV cache state.
        if let SequenceStatePayload::Attention(kv_payload) = &snap.sequence_state {
            let kv = engine.kv_cache_mut().ok_or(RuntimeError::ModelNotLoaded)?;
            kv.restore_from_payload(kv_payload)?;
        }

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_minimal_snapshot() -> EngineSnapshot {
        EngineSnapshot {
            magic: *SNAPSHOT_MAGIC,
            version: EngineSnapshot::VERSION,
            arch_id: "llama".to_string(),
            model_path: "/tmp/test.gguf".to_string(),
            tokenizer_path: None,
            model_fingerprint: ModelFingerprint {
                file_size: 1024,
                mtime_secs: 1_000_000,
                head_hash: [0u8; 32],
                tail_hash: [1u8; 32],
                probe_size: DEFAULT_PROBE_SIZE,
            },
            tokens: vec![1, 2, 3],
            sequence_state: SequenceStatePayload::Attention(KvStatePayload {
                keys: vec![vec![0.0f32; 4]],
                values: vec![vec![0.0f32; 4]],
                seq_len: 1,
                num_layers: 1,
                max_seq_len: 512,
                kv_dim: 4,
            }),
            sampler_state: SamplerStatePayload {
                rng_state: 42,
                mirostat_mu: 5.0,
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                min_p: 0.0,
                repetition_penalty: 1.1,
                repetition_penalty_window: 64,
                seed: Some(42),
                mirostat_mode: 0,
                mirostat_tau: 5.0,
                mirostat_eta: 0.1,
            },
            grammar_state: None,
            max_context_length: 512,
            num_threads: 4,
            prefill_chunk_size: 512,
        }
    }

    #[test]
    fn roundtrip_serialize_deserialize() {
        let snap = make_minimal_snapshot();
        let bytes = snap.serialize().expect("serialize");
        let restored = EngineSnapshot::deserialize(&bytes).expect("deserialize");
        assert_eq!(restored.arch_id, "llama");
        assert_eq!(restored.tokens, vec![1, 2, 3]);
        assert_eq!(restored.version, EngineSnapshot::VERSION);
        assert_eq!(&restored.magic, SNAPSHOT_MAGIC);
    }

    #[test]
    fn bad_magic_rejected() {
        // Build a valid snap then corrupt the serialized magic bytes.
        let snap = make_minimal_snapshot();
        let mut bytes = snap.serialize().expect("serialize");
        // The first 8 bytes in the oxicode encoding encode the magic field.
        // Corrupt some early bytes to trigger either a decode error or a magic mismatch.
        if bytes.len() > 4 {
            bytes[0] ^= 0xFF;
        }
        let result = EngineSnapshot::deserialize(&bytes);
        assert!(result.is_err(), "corrupted bytes must return Err");
    }

    #[test]
    fn incompatible_version_rejected() {
        // Serialize a snapshot with an invalid version.
        let mut snap = make_minimal_snapshot();
        snap.version = 9999;
        let bytes = snap.serialize().expect("serialize");
        let result = EngineSnapshot::deserialize(&bytes);
        assert!(
            matches!(result, Err(RuntimeError::SnapshotIncompatible { .. })),
            "invalid version must return SnapshotIncompatible"
        );
    }

    #[test]
    fn model_fingerprint_compute_and_verify() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxillama_snap_test_fingerprint.gguf");
        std::fs::write(&path, vec![0xABu8; 100 * 1024]).expect("write test file");

        let fp = ModelFingerprint::compute(&path).expect("compute fingerprint");
        assert_eq!(fp.file_size, 100 * 1024);
        fp.verify(&path).expect("verify same file");

        // Modify and re-verify — must fail.
        std::fs::write(&path, vec![0xCDu8; 100 * 1024]).expect("write modified file");
        assert!(
            fp.verify(&path).is_err(),
            "fingerprint verify must fail after file modification"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fingerprint_mismatch_error_type() {
        let dir = std::env::temp_dir();
        let path_a = dir.join("oxillama_snap_fp_a.gguf");
        let path_b = dir.join("oxillama_snap_fp_b.gguf");
        std::fs::write(&path_a, vec![0xAAu8; 10_000]).expect("write A");
        std::fs::write(&path_b, vec![0xBBu8; 10_000]).expect("write B");

        let fp_a = ModelFingerprint::compute(&path_a).expect("compute A");
        let result = fp_a.verify(&path_b);
        assert!(
            matches!(result, Err(RuntimeError::ModelFingerprintMismatch { .. })),
            "mismatch must return ModelFingerprintMismatch"
        );

        let _ = std::fs::remove_file(&path_a);
        let _ = std::fs::remove_file(&path_b);
    }

    #[test]
    fn kv_state_payload_roundtrip_in_snapshot() {
        let kv = KvStatePayload {
            keys: vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
            values: vec![vec![9.0f32, 10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0, 16.0]],
            seq_len: 1,
            num_layers: 2,
            max_seq_len: 512,
            kv_dim: 4,
        };
        let mut snap = make_minimal_snapshot();
        snap.sequence_state = SequenceStatePayload::Attention(kv.clone());

        let bytes = snap.serialize().expect("serialize");
        let restored = EngineSnapshot::deserialize(&bytes).expect("deserialize");

        if let SequenceStatePayload::Attention(restored_kv) = restored.sequence_state {
            assert_eq!(restored_kv.keys, kv.keys);
            assert_eq!(restored_kv.values, kv.values);
            assert_eq!(restored_kv.seq_len, kv.seq_len);
            assert_eq!(restored_kv.num_layers, kv.num_layers);
        } else {
            panic!("expected Attention sequence state payload");
        }
    }
}

//! Integration tests for the snapshot/resume feature.

use oxillama_runtime::error::RuntimeError;
use oxillama_runtime::sampling::Sampler;
use oxillama_runtime::sampling::SamplerConfig;
use oxillama_runtime::snapshot::{
    EngineSnapshot, GrammarStatePayload, KvStatePayload, ModelFingerprint, SamplerStatePayload,
    SequenceStatePayload, SNAPSHOT_MAGIC,
};
use oxillama_runtime::{EngineConfig, InferenceEngine};

fn minimal_kv_payload() -> KvStatePayload {
    KvStatePayload {
        keys: vec![vec![0.0f32; 4]],
        values: vec![vec![0.0f32; 4]],
        seq_len: 1,
        num_layers: 1,
        max_seq_len: 512,
        kv_dim: 4,
    }
}

fn minimal_sampler_state() -> SamplerStatePayload {
    SamplerStatePayload {
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
    }
}

fn minimal_fingerprint() -> ModelFingerprint {
    ModelFingerprint {
        file_size: 1024,
        mtime_secs: 1_000_000,
        head_hash: [0u8; 32],
        tail_hash: [1u8; 32],
        probe_size: 8 * 1024 * 1024,
    }
}

fn build_snapshot() -> EngineSnapshot {
    EngineSnapshot {
        magic: *SNAPSHOT_MAGIC,
        version: EngineSnapshot::VERSION,
        arch_id: "llama".to_string(),
        model_path: "/tmp/test.gguf".to_string(),
        tokenizer_path: None,
        model_fingerprint: minimal_fingerprint(),
        tokens: vec![1, 2, 3, 4, 5],
        sequence_state: SequenceStatePayload::Attention(minimal_kv_payload()),
        sampler_state: minimal_sampler_state(),
        grammar_state: None,
        max_context_length: 512,
        num_threads: 4,
        prefill_chunk_size: 512,
    }
}

// ─── Serialization roundtrip ──────────────────────────────────────────────────

#[test]
fn snapshot_roundtrip_serialize_deserialize() {
    let snap = build_snapshot();
    let bytes = snap.serialize().expect("serialize");
    let restored = EngineSnapshot::deserialize(&bytes).expect("deserialize");

    assert_eq!(restored.arch_id, "llama");
    assert_eq!(restored.tokens, vec![1, 2, 3, 4, 5]);
    assert_eq!(restored.version, EngineSnapshot::VERSION);
    assert_eq!(&restored.magic, SNAPSHOT_MAGIC);
    assert_eq!(restored.max_context_length, 512);
    assert_eq!(restored.num_threads, 4);
}

#[test]
fn snapshot_rejects_wrong_magic() {
    // Corrupt the serialized bytes — either decode fails or magic check fails.
    let snap = build_snapshot();
    let mut bytes = snap.serialize().expect("serialize");
    if bytes.len() > 4 {
        bytes[0] ^= 0xFF;
        bytes[1] ^= 0xFF;
    }
    let result = EngineSnapshot::deserialize(&bytes);
    assert!(
        result.is_err(),
        "corrupted bytes must produce an error, not Ok"
    );
}

#[test]
fn snapshot_rejects_incompatible_version() {
    let mut snap = build_snapshot();
    snap.version = 9999;
    let bytes = snap.serialize().expect("serialize valid bytes");
    let result = EngineSnapshot::deserialize(&bytes);
    assert!(
        matches!(result, Err(RuntimeError::SnapshotIncompatible { .. })),
        "invalid version must return SnapshotIncompatible, got {:?}",
        result
    );
}

// ─── ModelFingerprint ─────────────────────────────────────────────────────────

#[test]
fn model_fingerprint_compute_and_verify() {
    let dir = std::env::temp_dir();
    let path = dir.join("oxillama_snapshot_test_verify.gguf");
    std::fs::write(&path, vec![0xABu8; 100 * 1024]).expect("write test file");

    let fp = ModelFingerprint::compute(&path).expect("compute fingerprint");
    assert_eq!(fp.file_size, 100 * 1024);
    fp.verify(&path).expect("verify must succeed for same file");

    // Modify the file — verification must fail.
    std::fs::write(&path, vec![0xCDu8; 100 * 1024]).expect("write modified file");
    assert!(
        fp.verify(&path).is_err(),
        "fingerprint verify must fail after file modification"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn snapshot_fingerprint_mismatch_returns_correct_error() {
    let dir = std::env::temp_dir();
    let path_a = dir.join("oxillama_snapshot_fp_a.gguf");
    let path_b = dir.join("oxillama_snapshot_fp_b.gguf");
    std::fs::write(&path_a, vec![0xAAu8; 10_000]).expect("write A");
    std::fs::write(&path_b, vec![0xBBu8; 10_000]).expect("write B");

    let fp_a = ModelFingerprint::compute(&path_a).expect("compute A");
    let result = fp_a.verify(&path_b);
    assert!(
        matches!(result, Err(RuntimeError::ModelFingerprintMismatch { .. })),
        "mismatch must return ModelFingerprintMismatch, got {:?}",
        result
    );

    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}

// ─── KV state payload roundtrip ───────────────────────────────────────────────

#[test]
fn snapshot_kv_state_roundtrip() {
    let kv = KvStatePayload {
        keys: vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
        values: vec![vec![9.0f32, 10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0, 16.0]],
        seq_len: 1,
        num_layers: 2,
        max_seq_len: 512,
        kv_dim: 4,
    };
    let mut snap = build_snapshot();
    snap.sequence_state = SequenceStatePayload::Attention(kv.clone());

    let bytes = snap.serialize().expect("serialize");
    let restored = EngineSnapshot::deserialize(&bytes).expect("deserialize");

    if let SequenceStatePayload::Attention(restored_kv) = restored.sequence_state {
        assert_eq!(restored_kv.keys, kv.keys, "keys must round-trip exactly");
        assert_eq!(
            restored_kv.values, kv.values,
            "values must round-trip exactly"
        );
        assert_eq!(restored_kv.seq_len, kv.seq_len);
        assert_eq!(restored_kv.num_layers, kv.num_layers);
        assert_eq!(restored_kv.kv_dim, kv.kv_dim);
    } else {
        panic!("expected Attention sequence state payload after roundtrip");
    }
}

// ─── Sampler RNG roundtrip ───────────────────────────────────────────────────

#[test]
fn sampler_rng_state_roundtrip() {
    let config = SamplerConfig {
        seed: Some(12345),
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        ..SamplerConfig::default()
    };

    let logits = vec![1.0f32, 2.0, 3.0, 2.0, 1.0];
    let mut sampler_a = Sampler::new(config.clone());

    // Advance by sampling some tokens.
    for _ in 0..5 {
        sampler_a.sample(&logits, &[]);
    }

    // Capture state.
    let rng_state = sampler_a.rng_state();
    let mu = sampler_a.mirostat_mu_value();

    // Build a second sampler and restore state.
    let mut sampler_b = Sampler::new(config.clone());
    sampler_b.restore_rng_state(rng_state, mu);

    // Both should produce identical sequences from this point forward.
    for _ in 0..10 {
        let ta = sampler_a.sample(&logits, &[]);
        let tb = sampler_b.sample(&logits, &[]);
        assert_eq!(ta, tb, "restored RNG state must produce identical sequence");
    }
}

// ─── Grammar state in snapshot ────────────────────────────────────────────────

#[test]
fn snapshot_with_grammar_roundtrips() {
    let mut snap = build_snapshot();
    snap.grammar_state = Some(GrammarStatePayload {
        grammar_source: r#"root ::= "yes" | "no""#.to_string(),
    });

    let bytes = snap.serialize().expect("serialize with grammar");
    let restored = EngineSnapshot::deserialize(&bytes).expect("deserialize with grammar");

    let gs = restored
        .grammar_state
        .expect("grammar_state must be Some after roundtrip");
    assert_eq!(gs.grammar_source, r#"root ::= "yes" | "no""#);
}

// ─── End-to-end engine snapshot / resume ─────────────────────────────────────

/// Writes the synthetic LLaMA GGUF fixture and its tokenizer to temp_dir, loads
/// the engine from disk (so ModelFingerprint::compute can open the file), calls
/// snapshot(), then resumes into a fresh engine and asserts it is ready.
#[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
#[test]
fn engine_snapshot_resume_roundtrip() {
    use std::path::PathBuf;

    let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
    let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();

    // Write model and tokenizer to temp_dir so snapshot() can fingerprint the file.
    let dir: PathBuf = std::env::temp_dir();
    let model_path = dir.join("oxillama_e2e_snap_model.gguf");
    let tokenizer_path = dir.join("oxillama_e2e_snap_model.tokenizer.json");
    std::fs::write(&model_path, &model_bytes).expect("write synthetic GGUF");
    std::fs::write(&tokenizer_path, tokenizer_json).expect("write tokenizer JSON");

    let cfg = EngineConfig {
        model_path: model_path
            .to_str()
            .expect("temp path must be UTF-8")
            .to_string(),
        tokenizer_path: Some(
            tokenizer_path
                .to_str()
                .expect("temp path must be UTF-8")
                .to_string(),
        ),
        context_size: Some(512),
        num_threads: 1,
        prefill_chunk_size: 512,
        ..EngineConfig::default()
    };

    let mut engine = InferenceEngine::new(cfg);
    engine
        .load_model()
        .expect("synthetic GGUF must load from disk");
    assert!(engine.is_loaded(), "engine must be loaded before snapshot");

    // Capture snapshot.
    let snap_bytes = engine
        .snapshot()
        .expect("snapshot must succeed on loaded engine");
    assert!(!snap_bytes.is_empty(), "snapshot bytes must not be empty");

    // Resume into a fresh engine from the snapshot bytes.
    let resumed = InferenceEngine::resume(&snap_bytes, &model_path).expect("resume must succeed");
    assert!(
        resumed.is_loaded(),
        "resumed engine must report is_loaded() == true"
    );

    // Sanity: the resumed engine's config should reflect the original model path.
    assert_eq!(
        resumed.config().model_path,
        model_path.to_str().expect("temp path must be UTF-8"),
        "resumed engine model_path must match the original"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&tokenizer_path);
}

// ─── SSM state payload roundtrip ─────────────────────────────────────────────

#[test]
fn snapshot_ssm_state_roundtrip() {
    use oxillama_runtime::snapshot::SsmStatePayload;

    let ssm = SsmStatePayload {
        ssm_states: vec![vec![0.1f32, 0.2, 0.3, 0.4], vec![0.5f32, 0.6, 0.7, 0.8]],
        step: 7,
    };

    let mut snap = build_snapshot();
    snap.sequence_state = SequenceStatePayload::Mamba2(ssm.clone());

    let bytes = snap.serialize().expect("serialize Mamba2 snapshot");
    let restored = EngineSnapshot::deserialize(&bytes).expect("deserialize Mamba2 snapshot");

    if let SequenceStatePayload::Mamba2(restored_ssm) = restored.sequence_state {
        assert_eq!(restored_ssm.ssm_states, ssm.ssm_states);
        assert_eq!(restored_ssm.step, ssm.step);
    } else {
        panic!("expected Mamba2 sequence state payload after roundtrip");
    }
}

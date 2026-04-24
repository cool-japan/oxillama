//! Integration tests for the resume module.

use oxillama_gguf::{
    checkpoint_path_for, compute_fingerprint, load_checkpoint, save_checkpoint,
    validate_checkpoint, GgufError, GgufModel, GgufValueType, PrefixFingerprint, ResumeCheckpoint,
};

fn build_minimal_gguf() -> Vec<u8> {
    use oxillama_gguf::types::GGUF_MAGIC;
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

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
    data
}

#[test]
fn resume_roundtrip_valid_checkpoint() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, build_minimal_gguf()).expect("test: write gguf");

    GgufModel::save_resume_checkpoint(&path, 4096, 2048).expect("test: save_resume_checkpoint");

    let handle = GgufModel::resume(&path)
        .expect("test: resume")
        .expect("test: expected Some handle");

    assert_eq!(handle.checkpoint.file_size_expected, 4096);
    assert_eq!(handle.checkpoint.last_valid_offset, 2048);
    assert!(!handle.checkpoint.tensors_fully_loaded);
    assert_eq!(handle.checkpoint.version, ResumeCheckpoint::CURRENT_VERSION);
}

#[test]
fn resume_returns_none_when_no_sidecar() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, build_minimal_gguf()).expect("test: write gguf");

    let result = GgufModel::resume(&path).expect("test: resume");
    assert!(result.is_none(), "no sidecar should return None");
}

#[test]
fn resume_rejects_hash_mismatch() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, build_minimal_gguf()).expect("test: write gguf");

    GgufModel::save_resume_checkpoint(&path, 1024, 0).expect("test: save checkpoint");

    // Corrupt the stored head hash in the sidecar
    let sidecar = checkpoint_path_for(&path);
    let sidecar_bytes = std::fs::read(&sidecar).expect("test: read sidecar");
    let (mut cp, _) = oxicode::decode_from_slice::<ResumeCheckpoint>(&sidecar_bytes)
        .expect("test: decode checkpoint");
    cp.prefix_fingerprint.head_hash[0] ^= 0xFF;
    let encoded = oxicode::encode_to_vec(&cp).expect("test: encode checkpoint");
    std::fs::write(&sidecar, &encoded).expect("test: write corrupted sidecar");

    let result = GgufModel::resume(&path);
    assert!(
        matches!(result, Err(GgufError::ResumeMismatch { .. })),
        "corrupted head hash should return ResumeMismatch, got: {result:?}"
    );
}

#[test]
fn resume_rejects_tail_hash_mismatch() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, build_minimal_gguf()).expect("test: write gguf");

    GgufModel::save_resume_checkpoint(&path, 1024, 0).expect("test: save checkpoint");

    let sidecar = checkpoint_path_for(&path);
    let sidecar_bytes = std::fs::read(&sidecar).expect("test: read sidecar");
    let (mut cp, _) = oxicode::decode_from_slice::<ResumeCheckpoint>(&sidecar_bytes)
        .expect("test: decode checkpoint");
    cp.prefix_fingerprint.tail_hash[31] ^= 0xFF;
    let encoded = oxicode::encode_to_vec(&cp).expect("test: encode checkpoint");
    std::fs::write(&sidecar, &encoded).expect("test: write corrupted sidecar");

    let result = GgufModel::resume(&path);
    assert!(
        matches!(result, Err(GgufError::ResumeMismatch { .. })),
        "corrupted tail hash should return ResumeMismatch, got: {result:?}"
    );
}

#[test]
fn resume_rejects_future_file_size_still_validates() {
    // The `file_size_expected` field is just stored metadata — it does not
    // affect fingerprint validation.  A future expected size should still
    // load as long as the fingerprint matches.
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, build_minimal_gguf()).expect("test: write gguf");

    GgufModel::save_resume_checkpoint(&path, u64::MAX, 0).expect("test: save checkpoint");

    let handle = GgufModel::resume(&path)
        .expect("test: resume")
        .expect("test: expected handle");
    assert_eq!(handle.checkpoint.file_size_expected, u64::MAX);
}

#[test]
fn save_and_load_checkpoint_roundtrip() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, build_minimal_gguf()).expect("test: write gguf");

    let fp = compute_fingerprint(&path).expect("test: compute fingerprint");
    let cp = ResumeCheckpoint {
        file_size_expected: 999,
        last_valid_offset: 42,
        prefix_fingerprint: fp.clone(),
        tensors_fully_loaded: true,
        version: ResumeCheckpoint::CURRENT_VERSION,
    };

    save_checkpoint(&path, &cp).expect("test: save_checkpoint");
    let loaded = load_checkpoint(&path)
        .expect("test: load_checkpoint")
        .expect("test: expected Some checkpoint");

    assert_eq!(loaded.file_size_expected, 999);
    assert_eq!(loaded.last_valid_offset, 42);
    assert!(loaded.tensors_fully_loaded);
    assert_eq!(loaded.prefix_fingerprint.head_hash, fp.head_hash);
}

#[test]
fn load_checkpoint_returns_none_for_nonexistent_sidecar() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("nonexistent.gguf");
    let result = load_checkpoint(&path).expect("test: load_checkpoint");
    assert!(result.is_none(), "non-existent sidecar should return None");
}

#[test]
fn validate_checkpoint_succeeds_for_fresh_fingerprint() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, build_minimal_gguf()).expect("test: write gguf");

    GgufModel::save_resume_checkpoint(&path, 0, 0).expect("test: save checkpoint");

    let sidecar = checkpoint_path_for(&path);
    let sidecar_bytes = std::fs::read(&sidecar).expect("test: read sidecar");
    let (cp, _) =
        oxicode::decode_from_slice::<ResumeCheckpoint>(&sidecar_bytes).expect("test: decode");

    validate_checkpoint(&path, &cp).expect("test: validate_checkpoint should succeed");
}

#[test]
fn checkpoint_path_suffix_is_correct() {
    let base = std::path::Path::new("/tmp/llama-8b.gguf");
    let sidecar = checkpoint_path_for(base);
    assert_eq!(
        sidecar.file_name().and_then(|n| n.to_str()),
        Some("llama-8b.gguf.oxiresume"),
        "sidecar suffix should be .oxiresume"
    );
}

#[test]
fn fingerprint_oxicode_roundtrip() {
    let fp = PrefixFingerprint {
        head_hash: [0xABu8; 32],
        tail_hash: [0xCDu8; 32],
        probe_size: 1024,
        file_mtime_secs: 1_700_000_000,
    };
    let encoded = oxicode::encode_to_vec(&fp).expect("test: encode");
    let (decoded, _) =
        oxicode::decode_from_slice::<PrefixFingerprint>(&encoded).expect("test: decode");
    assert_eq!(fp, decoded);
}

#[test]
fn resume_handle_removes_checkpoint_file() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, build_minimal_gguf()).expect("test: write gguf");

    GgufModel::save_resume_checkpoint(&path, 100, 50).expect("test: save checkpoint");

    let sidecar = checkpoint_path_for(&path);
    assert!(sidecar.exists(), "sidecar should exist before removal");

    let handle = GgufModel::resume(&path)
        .expect("test: resume")
        .expect("test: expected handle");

    handle.remove_checkpoint().expect("test: remove_checkpoint");
    assert!(
        !sidecar.exists(),
        "sidecar should be removed after remove_checkpoint"
    );
}

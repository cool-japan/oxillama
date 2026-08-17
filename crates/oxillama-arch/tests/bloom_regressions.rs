//! Regression tests for confirmed BLOOM correctness defects (BL1-BL3).
//!
//! Each synthetic GGUF is built directly with `oxillama_gguf::GgufWriter`
//! (the always-available production writer, not the `test-utils` fixture in
//! `oxillama-gguf/src/test_utils/arch.rs`) — the shared `build_minimal_bloom_gguf()`
//! builder there is outside this crate's ownership boundary for the Bloom
//! fixture and predates BL1 (it does not carry `token_embd_norm`), so this
//! file does not depend on it.
//!
//! BL1 was verified to fail before the fix and pass after by `git stash`-ing
//! `bloom/model.rs` and re-running
//! `cargo nextest run -p oxillama-arch --test bloom_regressions`; see the fix
//! report for the captured before/after output.

#![cfg(feature = "bloom")]

use oxillama_arch::bloom::{load_bloom_from_gguf, BloomArchitecture};
use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::{ArchError, ArchResult};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess, ModelArchitecture};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

const H: usize = 32; // hidden_size
const HEADS: usize = 4;
const HEAD_DIM: usize = H / HEADS;
const FFN: usize = 64;
const VOCAB: usize = 8;

struct TestKv {
    kv_dim: usize,
    max_seq: usize,
    position: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl TestKv {
    fn new(kv_dim: usize, max_seq: usize) -> Self {
        Self {
            kv_dim,
            max_seq,
            position: 0,
            keys: vec![0.0f32; max_seq * kv_dim],
            values: vec![0.0f32; max_seq * kv_dim],
        }
    }
}

impl KvCacheAccess for TestKv {
    fn seq_len(&self) -> usize {
        self.position
    }
    fn store_kv(&mut self, _layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let offset = self.position * self.kv_dim;
        let ck = key.len().min(self.kv_dim);
        let cv = value.len().min(self.kv_dim);
        self.keys[offset..offset + ck].copy_from_slice(&key[..ck]);
        self.values[offset..offset + cv].copy_from_slice(&value[..cv]);
        Ok(())
    }
    fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
        let end = (self.position + 1) * self.kv_dim;
        Ok(&self.keys[..end])
    }
    fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
        let end = (self.position + 1) * self.kv_dim;
        Ok(&self.values[..end])
    }
    fn advance(&mut self) {
        self.position = (self.position + 1).min(self.max_seq.saturating_sub(1));
    }
}

fn zeros(n: usize) -> Vec<u8> {
    vec![0u8; n * 4]
}

/// Build a minimal 1-layer BLOOM GGUF. `with_token_embd_norm` controls
/// whether `token_embd_norm.weight`/`.bias` are written — omitting them
/// reproduces the tensor set BL1 was filed against. `qkv_bias_len` overrides
/// the element count of `blk.0.attn_qkv.bias` (`None` = correct length
/// `HEADS * 3 * HEAD_DIM`), so BL2 can exercise an otherwise-complete,
/// otherwise-loadable checkpoint with only that one tensor malformed.
fn build_bloom_gguf(with_token_embd_norm: bool, qkv_bias_len: Option<usize>) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("bloom".to_string()),
    );
    w.add_metadata("bloom.embedding_length", MetadataValue::Uint32(H as u32));
    w.add_metadata(
        "bloom.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("bloom.block_count", MetadataValue::Uint32(1));
    w.add_metadata(
        "bloom.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "bloom.attention.head_count_kv",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata("bloom.context_length", MetadataValue::Uint32(128));
    w.add_metadata("bloom.vocab_size", MetadataValue::Uint32(VOCAB as u32));

    w.add_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &zeros(VOCAB * H),
    );
    if with_token_embd_norm {
        w.add_tensor(
            "token_embd_norm.weight",
            &[H as u64],
            GgufTensorType::F32,
            &zeros(H),
        );
        w.add_tensor(
            "token_embd_norm.bias",
            &[H as u64],
            GgufTensorType::F32,
            &zeros(H),
        );
    }
    w.add_tensor(
        "blk.0.attn_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    w.add_tensor(
        "blk.0.attn_norm.bias",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    let qkv_out = HEADS * 3 * HEAD_DIM;
    let bias_len = qkv_bias_len.unwrap_or(qkv_out);
    w.add_tensor(
        "blk.0.attn_qkv.weight",
        &[H as u64, qkv_out as u64],
        GgufTensorType::F32,
        &zeros(H * qkv_out),
    );
    w.add_tensor(
        "blk.0.attn_qkv.bias",
        &[bias_len as u64],
        GgufTensorType::F32,
        &zeros(bias_len),
    );
    w.add_tensor(
        "blk.0.attn_output.weight",
        &[H as u64, H as u64],
        GgufTensorType::F32,
        &zeros(H * H),
    );
    w.add_tensor(
        "blk.0.attn_output.bias",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    w.add_tensor(
        "blk.0.ffn_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    w.add_tensor(
        "blk.0.ffn_norm.bias",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    w.add_tensor(
        "blk.0.ffn_up.weight",
        &[H as u64, FFN as u64],
        GgufTensorType::F32,
        &zeros(H * FFN),
    );
    w.add_tensor(
        "blk.0.ffn_up.bias",
        &[FFN as u64],
        GgufTensorType::F32,
        &zeros(FFN),
    );
    w.add_tensor(
        "blk.0.ffn_down.weight",
        &[FFN as u64, H as u64],
        GgufTensorType::F32,
        &zeros(FFN * H),
    );
    w.add_tensor(
        "blk.0.ffn_down.bias",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    w.add_tensor(
        "output_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    w.add_tensor(
        "output_norm.bias",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic bloom GGUF must serialize");
    bytes
}

fn parse_and_config(bytes: Vec<u8>) -> (GgufModel, ModelConfig) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
    let config =
        ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
    (gguf, config)
}

// ─── BL1: token_embd_norm is required and actually applied ────────────────────

/// A checkpoint carrying `token_embd_norm.weight`/`.bias` (every real BLOOM
/// checkpoint) must load successfully.
#[test]
fn bl1_checkpoint_with_token_embd_norm_loads() {
    let (gguf, config) = parse_and_config(build_bloom_gguf(true, None));
    let result = load_bloom_from_gguf(&gguf, &config);
    assert!(
        result.is_ok(),
        "a checkpoint with token_embd_norm must load, got: {:?}",
        result.err()
    );
}

/// A checkpoint MISSING `token_embd_norm.weight` must fail loudly. Before the
/// fix, `load_bloom_from_gguf` never looked up this tensor at all (`rg
/// token_embd_norm` across the whole workspace returned nothing), so this
/// same fixture loaded "successfully" — silently skipping the normalization
/// every layer depends on.
#[test]
fn bl1_checkpoint_missing_token_embd_norm_is_a_loud_error() {
    let (gguf, config) = parse_and_config(build_bloom_gguf(false, None));
    let result = load_bloom_from_gguf(&gguf, &config);
    match result {
        Err(ArchError::MissingTensor { name }) => {
            assert!(
                name.contains("token_embd_norm"),
                "expected MissingTensor naming token_embd_norm, got: {name}"
            );
        }
        Err(other) => panic!("expected MissingTensor, got a different error: {other:?}"),
        Ok(_) => panic!("a checkpoint without token_embd_norm must not load successfully"),
    }
}

/// End-to-end forward pass after loading must produce finite vocab-sized
/// logits.
#[test]
fn bl1_forward_after_load_is_finite() {
    let (gguf, config) = parse_and_config(build_bloom_gguf(true, None));
    let mut model = load_bloom_from_gguf(&gguf, &config).expect("load bloom");
    let mut kv = TestKv::new(HEADS * HEAD_DIM, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
    }
}

/// `ModelArchitecture::build_from_gguf` must delegate to
/// `load_bloom_from_gguf` (the registry-driven load path the runtime engine
/// is being migrated onto), not the default `NotSupported`.
#[test]
fn bl_build_from_gguf_delegates_to_the_real_loader() {
    let (gguf, config) = parse_and_config(build_bloom_gguf(true, None));
    let arch = BloomArchitecture::new();
    let mut model = arch
        .build_from_gguf(&gguf, &config)
        .expect("build_from_gguf must succeed");
    let mut kv = TestKv::new(HEADS * HEAD_DIM, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
}

// ─── Shared: context bounds / OOV token guards, tied-embedding fallback ───────

#[test]
fn bl_missing_output_weight_falls_back_to_tied_token_embd() {
    // build_bloom_gguf never writes `output.weight`, so every test above
    // already exercises the tied fallback; this test names it explicitly.
    let (gguf, config) = parse_and_config(build_bloom_gguf(true, None));
    let mut model = load_bloom_from_gguf(&gguf, &config).expect("load bloom");
    let mut kv = TestKv::new(HEADS * HEAD_DIM, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
}

#[test]
fn bl_prefill_past_max_context_length_is_an_error() {
    let (gguf, config) = parse_and_config(build_bloom_gguf(true, None));
    let mut model = load_bloom_from_gguf(&gguf, &config).expect("load bloom");
    let mut kv = TestKv::new(HEADS * HEAD_DIM, 256);
    let tokens = vec![0u32; config.max_context_length + 1];
    let result = model.forward(&tokens, &mut kv);
    assert!(
        result.is_err(),
        "a prompt longer than max_context_length must error, not panic"
    );
}

#[test]
fn bl_out_of_vocabulary_token_is_an_error() {
    let (gguf, config) = parse_and_config(build_bloom_gguf(true, None));
    let mut model = load_bloom_from_gguf(&gguf, &config).expect("load bloom");
    let mut kv = TestKv::new(HEADS * HEAD_DIM, 8);
    let result = model.forward(&[VOCAB as u32 + 5], &mut kv);
    assert!(
        result.is_err(),
        "an out-of-vocabulary token id must error, not index out of bounds"
    );
}

// ─── BL2: mismatched bias tensor length is a loud error, not zero-fill ────────

/// A `attn_qkv.bias` tensor whose declared length disagrees with
/// `(num_heads * 3) * head_dim` must be rejected. Before the fix,
/// `bloom/model.rs` zero-padded a short bias or truncated a long one into the
/// expected width instead of erroring — wrong output with no error, for a
/// case that should never legitimately arise on a well-formed checkpoint.
///
/// Uses the OTHERWISE-COMPLETE `build_bloom_gguf` fixture (every other
/// tensor correct) with only `attn_qkv.bias` shortened by one element, so
/// this is a true before/after discriminator: pre-fix, the old zero-pad
/// fallback let this exact checkpoint load "successfully" (silently wrong
/// bias values); a variant of this fixture that ALSO omitted a later
/// required tensor would have failed with `MissingTensor` even before the
/// fix and proven nothing about the bias-length check specifically.
#[test]
fn bl2_mismatched_qkv_bias_length_is_rejected() {
    let qkv_out = HEADS * 3 * HEAD_DIM;
    let (gguf, config) = parse_and_config(build_bloom_gguf(true, Some(qkv_out - 1)));
    let result = load_bloom_from_gguf(&gguf, &config);
    match result {
        Err(ArchError::InvalidShape { name, .. }) => {
            assert!(
                name.contains("attn_qkv.bias"),
                "expected InvalidShape naming attn_qkv.bias, got: {name}"
            );
        }
        Err(other) => panic!("expected InvalidShape, got a different error: {other:?}"),
        Ok(_) => panic!("a mismatched-length bias tensor must not load successfully"),
    }
}

// ─── mem::take(buf_logits) safety: BLOOM has no runtime-engine integration
// test today (unlike command_r's `test_generate_command_r_arch`), so this is
// the only regression guard against the exact bug class caught in
// command_r/model.rs during this fix: `forward()` hands `buf_logits` to the
// caller by `std::mem::take`, which leaves it empty; without a resize guard
// at the top of the NEXT `forward()` call, the second call panics/errors
// instead of producing `vocab_size` logits. ────────────────────────────────

#[test]
fn bl_two_consecutive_forward_calls_both_succeed() {
    let (gguf, config) = parse_and_config(build_bloom_gguf(true, None));
    let mut model = load_bloom_from_gguf(&gguf, &config).expect("load bloom");
    let mut kv = TestKv::new(HEADS * HEAD_DIM, 8);
    let first = model.forward(&[0u32], &mut kv).expect("first forward");
    assert_eq!(first.len(), VOCAB);
    let second = model.forward(&[1u32], &mut kv).expect("second forward");
    assert_eq!(
        second.len(),
        VOCAB,
        "second forward() must still produce vocab_size logits"
    );
    for (i, &v) in second.iter().enumerate() {
        assert!(
            v.is_finite(),
            "logit[{i}] must be finite on 2nd call, got {v}"
        );
    }
}

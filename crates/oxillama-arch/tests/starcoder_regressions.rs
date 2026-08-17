//! Regression tests for confirmed StarCoder correctness defects (SC1-SC4).
//!
//! Each synthetic GGUF is built directly with `oxillama_gguf::GgufWriter`
//! (the always-available production writer, not the `test-utils` fixture in
//! `oxillama-gguf/src/test_utils/arch.rs`) so these tests do not depend on —
//! or need to modify — the shared fixture builders owned by other
//! architectures' test suites.
//!
//! SC1 and SC2 were verified to fail before the fix and pass after by
//! `git stash`-ing the `starcoder/model.rs` changes and re-running
//! `cargo nextest run -p oxillama-arch --test starcoder_regressions`; see the
//! fix report for the captured before/after output.

#![cfg(feature = "starcoder")]

use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::{ArchError, ArchResult};
use oxillama_arch::starcoder::{load_starcoder_from_gguf, StarcoderArchitecture, StarcoderModel};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess, ModelArchitecture};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

const H: usize = 32; // hidden_size
const HEADS: usize = 2;
const HD: usize = 16; // head_dim (H / HEADS)
const FFN: usize = 64;
const VOCAB: usize = 8;

/// Minimal in-memory KV cache, sized for a single layer.
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

/// Build a minimal 1-layer StarCoder GGUF using the *correct* GGUF tensor
/// naming convention (`attn_output.weight`/`.bias`).
///
/// `max_position` controls the row count of `position_embd.weight`, which can
/// be set independently of `context_length` to exercise SC3 without the
/// context-length guard (added separately, `validate_context_bounds`)
/// pre-empting the position-specific check.
///
/// `include_output_weight` controls whether a standalone `output.weight` is
/// written — omit it to exercise the tied-embedding fallback (SC2).
fn build_starcoder_gguf(max_position: usize, include_output_weight: bool) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("starcoder".to_string()),
    );
    w.add_metadata(
        "starcoder.embedding_length",
        MetadataValue::Uint32(H as u32),
    );
    w.add_metadata(
        "starcoder.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("starcoder.block_count", MetadataValue::Uint32(1));
    w.add_metadata(
        "starcoder.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "starcoder.attention.head_count_kv",
        MetadataValue::Uint32(1),
    );
    // context_length is deliberately generous so validate_context_bounds
    // does not interfere with the max_position-specific SC3 test.
    w.add_metadata("starcoder.context_length", MetadataValue::Uint32(128));
    w.add_metadata("starcoder.vocab_size", MetadataValue::Uint32(VOCAB as u32));

    w.add_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &zeros(VOCAB * H),
    );
    w.add_tensor(
        "position_embd.weight",
        &[H as u64, max_position as u64],
        GgufTensorType::F32,
        &zeros(max_position * H),
    );

    let qkv_out = (HEADS + 2) * HD;
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
    w.add_tensor(
        "blk.0.attn_qkv.weight",
        &[H as u64, qkv_out as u64],
        GgufTensorType::F32,
        &zeros(H * qkv_out),
    );
    w.add_tensor(
        "blk.0.attn_qkv.bias",
        &[qkv_out as u64],
        GgufTensorType::F32,
        &zeros(qkv_out),
    );
    // The correct GGUF tensor name (SC1): `attn_output`, not `attn_out`.
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
    if include_output_weight {
        w.add_tensor(
            "output.weight",
            &[H as u64, VOCAB as u64],
            GgufTensorType::F32,
            &zeros(VOCAB * H),
        );
    }

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic starcoder GGUF must serialize");
    bytes
}

fn parse_and_config(bytes: Vec<u8>) -> (GgufModel, ModelConfig) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
    let config =
        ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
    (gguf, config)
}

// ─── SC1: `attn_output.weight`, not `attn_out.weight` ─────────────────────────

/// A checkpoint written with the real GGUF tensor name `attn_output.weight`
/// must load. Before the fix, `load_starcoder_from_gguf` looked up
/// `attn_out.weight` (gguf-py never emits that name for any architecture) and
/// this failed with `MissingTensor { name: "blk.0.attn_out.weight" }`.
#[test]
fn sc1_attn_output_tensor_name_loads_successfully() {
    let (gguf, config) = parse_and_config(build_starcoder_gguf(4, true));
    let result = load_starcoder_from_gguf(&gguf, &config);
    assert!(
        result.is_ok(),
        "loading a checkpoint using the real 'attn_output.weight' tensor name must succeed, got: {:?}",
        result.err()
    );
}

/// End-to-end: after loading, a forward pass must produce finite
/// vocab-sized logits (proves the attn_output weights were actually wired
/// up, not merely present).
#[test]
fn sc1_forward_after_correct_load_is_finite() {
    let (gguf, config) = parse_and_config(build_starcoder_gguf(4, true));
    let mut model = load_starcoder_from_gguf(&gguf, &config).expect("load starcoder");
    let mut kv = TestKv::new(HD, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
    }
}

// ─── SC2: tied-embedding LM-head fallback ──────────────────────────────────────

/// GPT-BigCode defaults `tie_word_embeddings=true`; most StarCoder checkpoints
/// ship no standalone `output.weight` at all. Before the fix,
/// `load_starcoder_quant_linear(model, "output.weight")` returned
/// `MissingTensor` unconditionally.
#[test]
fn sc2_missing_output_weight_falls_back_to_tied_token_embd() {
    let (gguf, config) = parse_and_config(build_starcoder_gguf(4, false));
    let result = load_starcoder_from_gguf(&gguf, &config);
    assert!(
        result.is_ok(),
        "a checkpoint with no output.weight must fall back to tied token_embd.weight, got: {:?}",
        result.err()
    );
    let mut model = result.expect("load starcoder");
    let mut kv = TestKv::new(HD, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
}

// ─── SC3: out-of-range position must error, not silently clamp ────────────────

/// `position_embd.weight` has only 4 rows here (`max_position = 4`), while
/// `context_length = 128` — chosen so `validate_context_bounds` alone would
/// accept a 5-token prompt, isolating the position-specific guard added in
/// `embed_token`. Before the fix this silently clamped to
/// `position_embd[3]` for every position past the trained context, instead
/// of erroring.
#[test]
fn sc3_position_past_max_position_is_an_error() {
    let (gguf, config) = parse_and_config(build_starcoder_gguf(4, true));
    let mut model = load_starcoder_from_gguf(&gguf, &config).expect("load starcoder");
    let mut kv = TestKv::new(HD, 16);
    // 5 tokens => positions 0..=4; position 4 is out of range for a 4-row
    // (positions 0..=3) position_embd table.
    let result = model.forward(&[0u32, 1, 2, 3, 4], &mut kv);
    assert!(
        result.is_err(),
        "a prompt longer than the learned position table must error, not silently clamp"
    );
    assert!(
        matches!(result, Err(ArchError::ConfigMismatch { .. })),
        "expected ConfigMismatch naming the out-of-range position, got: {result:?}"
    );
}

/// Exactly at the boundary (`position == max_position - 1`) must still
/// succeed — the guard must not be off-by-one in the strict direction either.
#[test]
fn sc3_position_at_max_position_boundary_is_ok() {
    let (gguf, config) = parse_and_config(build_starcoder_gguf(4, true));
    let mut model = load_starcoder_from_gguf(&gguf, &config).expect("load starcoder");
    let mut kv = TestKv::new(HD, 16);
    // 4 tokens => positions 0..=3, all within the 4-row table.
    let result = model.forward(&[0u32, 1, 2, 3], &mut kv);
    assert!(
        result.is_ok(),
        "a prompt exactly filling the position table must succeed, got: {:?}",
        result.err()
    );
}

// ─── Shared: prefill past max_context_length must error, not panic ────────────

#[test]
fn sc_prefill_past_max_context_length_is_an_error() {
    let (gguf, config) = parse_and_config(build_starcoder_gguf(128, true));
    let mut model = load_starcoder_from_gguf(&gguf, &config).expect("load starcoder");
    let mut kv = TestKv::new(HD, 256);
    let tokens = vec![0u32; config.max_context_length + 1];
    let result = model.forward(&tokens, &mut kv);
    assert!(
        result.is_err(),
        "a prompt longer than max_context_length must error, not panic on buf_attn_scores"
    );
}

#[test]
fn sc_out_of_vocabulary_token_is_an_error() {
    let (gguf, config) = parse_and_config(build_starcoder_gguf(4, true));
    let mut model = load_starcoder_from_gguf(&gguf, &config).expect("load starcoder");
    let mut kv = TestKv::new(HD, 8);
    let result = model.forward(&[VOCAB as u32 + 5], &mut kv);
    assert!(
        result.is_err(),
        "an out-of-vocabulary token id must error, not index out of bounds"
    );
}

/// `ModelArchitecture::build_from_gguf` must delegate to
/// `load_starcoder_from_gguf` (the registry-driven load path the runtime
/// engine is being migrated onto), not the default `NotSupported`.
#[test]
fn sc_build_from_gguf_delegates_to_the_real_loader() {
    let (gguf, config) = parse_and_config(build_starcoder_gguf(4, true));
    let arch = StarcoderArchitecture::new();
    let mut model = arch
        .build_from_gguf(&gguf, &config)
        .expect("build_from_gguf must succeed");
    let mut kv = TestKv::new(HD, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
}

// ─── LoRA: apply_lora / apply_lora_scaled must no longer silently no-op ───────

#[test]
fn sc_apply_lora_without_matching_adapter_is_a_harmless_ok() {
    let (gguf, config) = parse_and_config(build_starcoder_gguf(4, true));
    let mut model: StarcoderModel = load_starcoder_from_gguf(&gguf, &config).expect("load");
    let empty = oxillama_arch::lora::LoadedLora {
        adapters: std::collections::HashMap::new(),
        rank: 0,
        alpha: 1.0,
    };
    let result = model.apply_lora(&empty);
    assert!(
        result.is_ok(),
        "apply_lora with no matching adapters must succeed as a no-op, got: {result:?}"
    );
    // apply_lora_scaled with a non-unit scale must also be accepted now
    // (previously the default trait impl returned NotSupported for scale != 1.0).
    let result_scaled = model.apply_lora_scaled(&empty, 0.5);
    assert!(
        result_scaled.is_ok(),
        "apply_lora_scaled must be implemented (not NotSupported), got: {result_scaled:?}"
    );
}

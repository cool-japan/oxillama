//! Regression tests for confirmed Phi correctness defects (G5, G8, G9, G10).
//!
//! G6 (Phi SWA) is not exercised here: the reference (`llama-model.cpp`,
//! `case LLM_ARCH_PHI3:`) forcibly disables sliding-window attention for
//! this architecture, so "wiring it up" would be a behavioral regression
//! relative to llama.cpp, not a fix — see the comment at
//! `phi::model::PhiModel::attention`. G7 (bias loading) is not applicable to
//! plain Phi-3 either: `LLM_ARCH_PHI3`'s tensor map has zero bias tensors in
//! the reference; the PhiMoE-specific bias support is exercised in
//! `tests/phi_moe_regressions.rs`.

#![cfg(feature = "phi")]

use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::ArchResult;
use oxillama_arch::phi::{load_phi_from_gguf, PhiModel};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

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
        self.keys[offset..offset + key.len()].copy_from_slice(key);
        self.values[offset..offset + value.len()].copy_from_slice(value);
        Ok(())
    }
    fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.keys[..(self.position + 1) * self.kv_dim])
    }
    fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.values[..(self.position + 1) * self.kv_dim])
    }
    fn advance(&mut self) {
        self.position = (self.position + 1).min(self.max_seq.saturating_sub(1));
    }
}

const H: usize = 16;
const HEADS: usize = 2;
const KV_HEADS: usize = 2;
const HD: usize = 8;
const FFN: usize = 32;
const VOCAB: usize = 16;

fn zeros(n: usize) -> Vec<u8> {
    vec![0u8; n * 4]
}

/// Build a minimal 1-layer Phi GGUF. `dimension_count`, when `Some`, sets
/// `phi3.rope.dimension_count`; `with_output` controls whether a standalone
/// `output.weight` is written (vs. relying on the tied-embedding fallback).
fn build_phi_gguf(dimension_count: Option<u32>, with_output: bool) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("phi3".to_string()),
    );
    w.add_metadata("phi3.embedding_length", MetadataValue::Uint32(H as u32));
    w.add_metadata(
        "phi3.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("phi3.block_count", MetadataValue::Uint32(1));
    w.add_metadata(
        "phi3.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "phi3.attention.head_count_kv",
        MetadataValue::Uint32(KV_HEADS as u32),
    );
    w.add_metadata(
        "phi3.attention.key_length",
        MetadataValue::Uint32(HD as u32),
    );
    w.add_metadata("phi3.context_length", MetadataValue::Uint32(64));
    w.add_metadata("phi3.vocab_size", MetadataValue::Uint32(VOCAB as u32));
    w.add_metadata("phi3.rope.freq_base", MetadataValue::Float32(10000.0));
    if let Some(dc) = dimension_count {
        w.add_metadata("phi3.rope.dimension_count", MetadataValue::Uint32(dc));
    }

    w.add_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &zeros(VOCAB * H),
    );
    w.add_tensor(
        "output_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    if with_output {
        w.add_tensor(
            "output.weight",
            &[H as u64, VOCAB as u64],
            GgufTensorType::F32,
            &zeros(VOCAB * H),
        );
    }

    let q_dim = HEADS * HD;
    let kv_dim = KV_HEADS * HD;
    let qkv_dim = q_dim + 2 * kv_dim;
    let p = "blk.0";
    w.add_tensor(
        &format!("{p}.attn_norm.weight"),
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    w.add_tensor(
        &format!("{p}.attn_qkv.weight"),
        &[H as u64, qkv_dim as u64],
        GgufTensorType::F32,
        &zeros(H * qkv_dim),
    );
    w.add_tensor(
        &format!("{p}.attn_output.weight"),
        &[q_dim as u64, H as u64],
        GgufTensorType::F32,
        &zeros(q_dim * H),
    );
    w.add_tensor(
        &format!("{p}.ffn_norm.weight"),
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    w.add_tensor(
        &format!("{p}.ffn_gate.weight"),
        &[H as u64, FFN as u64],
        GgufTensorType::F32,
        &zeros(H * FFN),
    );
    w.add_tensor(
        &format!("{p}.ffn_up.weight"),
        &[H as u64, FFN as u64],
        GgufTensorType::F32,
        &zeros(H * FFN),
    );
    w.add_tensor(
        &format!("{p}.ffn_down.weight"),
        &[FFN as u64, H as u64],
        GgufTensorType::F32,
        &zeros(FFN * H),
    );

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic phi GGUF must serialize");
    bytes
}

fn parse_and_config(bytes: Vec<u8>) -> (GgufModel, ModelConfig) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
    let config =
        ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
    (gguf, config)
}

fn load(bytes: Vec<u8>) -> PhiModel {
    let (gguf, config) = parse_and_config(bytes);
    load_phi_from_gguf(&gguf, &config).expect("phi fixture must load")
}

// ─── G5: rope_dims from {arch}.rope.dimension_count, defaulting to head_dim ──

#[test]
fn g5_rope_dims_defaults_to_full_head_dim_when_dimension_count_absent() {
    // The bug: the old code read `{arch}.rope.partial_rotary_factor`, a key
    // no GGUF converter ever writes, so it always missed and fell back to a
    // hardcoded 0.5 — half of every head, unconditionally. Absent
    // `dimension_count`, the fix must default to the FULL head_dim instead.
    let model = load(build_phi_gguf(None, true));
    assert_eq!(
        model.rope_dims, HD,
        "absent {{arch}}.rope.dimension_count must default to the full head_dim (full rotary), \
         not head_dim/2"
    );
}

#[test]
fn g5_rope_dims_read_from_dimension_count_when_present() {
    // A genuine partial-rotary checkpoint (dimension_count < head_dim) must
    // be honored, not silently replaced by a hardcoded factor.
    let model = load(build_phi_gguf(Some(4), true));
    assert_eq!(
        model.rope_dims, 4,
        "explicit {{arch}}.rope.dimension_count must be read verbatim"
    );
}

#[test]
fn g5_rope_dims_zero_in_metadata_still_falls_back_to_head_dim() {
    // A zero dimension_count is nonsensical (RoPE would rotate nothing);
    // the loader must not blindly accept it.
    let model = load(build_phi_gguf(Some(0), true));
    assert_eq!(
        model.rope_dims, HD,
        "a zero dimension_count must fall back to head_dim"
    );
}

// ─── G8: out-of-vocabulary token ids error instead of panicking ──────────────

#[test]
fn g8_out_of_vocab_token_errors_instead_of_panicking() {
    let mut model = load(build_phi_gguf(None, true));
    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(kv_dim, 64);

    let result = model.forward(&[VOCAB as u32], &mut kv);
    assert!(
        result.is_err(),
        "token id == vocab_size must error, not panic on the buf_hidden slice"
    );
    let result2 = model.forward(&[u32::MAX], &mut kv);
    assert!(
        result2.is_err(),
        "a wildly out-of-range token id must not overflow into a panic"
    );
}

// ─── G9: prefill past max_context_length errors instead of panicking ─────────

#[test]
fn g9_prefill_past_max_context_length_errors_instead_of_panicking() {
    let mut model = load(build_phi_gguf(None, true));
    assert_eq!(model.max_context_length(), 64);
    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(kv_dim, 128);
    let too_many_tokens = vec![0u32; 65];
    let result = model.forward(&too_many_tokens, &mut kv);
    assert!(
        result.is_err(),
        "a prefill longer than max_context_length must error, not run RopeTable::apply/\
         buf_attn_scores past their max_context_length-sized allocation"
    );
}

#[test]
fn g9_prefill_exactly_at_max_context_length_is_ok() {
    let mut model = load(build_phi_gguf(None, true));
    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(kv_dim, 128);
    let exactly_at_limit = vec![0u32; 64];
    let result = model.forward(&exactly_at_limit, &mut kv);
    assert!(
        result.is_ok(),
        "filling exactly to max_context_length must still succeed: {:?}",
        result.err()
    );
}

// ─── G10: tied-embedding LM-head fallback ─────────────────────────────────────

#[test]
fn g10_missing_output_weight_falls_back_to_tied_token_embd() {
    let (gguf, config) = parse_and_config(build_phi_gguf(None, false));
    assert!(
        !gguf.file.tensors.contains("output.weight"),
        "fixture must model a tied checkpoint (no standalone output.weight)"
    );
    let model = load_phi_from_gguf(&gguf, &config)
        .expect("a tied checkpoint (no output.weight) must still load");
    assert_eq!(
        model.output.weight.shape,
        vec![VOCAB, H],
        "tied LM head must keep token_embd's dimensions"
    );

    // And it must actually be usable for a forward pass.
    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(kv_dim, 64);
    let mut model = model;
    let logits = model
        .forward(&[0u32], &mut kv)
        .expect("forward through the tied LM head must succeed");
    assert_eq!(logits.len(), VOCAB);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
    }
}

#[test]
fn g10_explicit_output_weight_still_takes_precedence() {
    let model = load(build_phi_gguf(None, true));
    // With an explicit `output.weight` present, the loader must use it
    // rather than the tied fallback (both are all-zero here, so this just
    // confirms the loader didn't error/skip the explicit tensor).
    assert_eq!(model.output.weight.shape, vec![VOCAB, H]);
}

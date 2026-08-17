//! Regression tests for confirmed Phi-MoE correctness defects (G5, G7, G8, G9, G10).
//!
//! G6 is not applicable here either — `LLM_ARCH_PHIMOE` shares `LLM_ARCH_PHI3`'s
//! forced-dense hparams path in llama.cpp (see the note in
//! `phi_moe::model::PhiMoeModel::attention`). G13 (renormalized top-2 routing,
//! and removal of the dead `partial_rotary_factor` field) is exercised by the
//! existing inline tests in `phi_moe::model`/`phi_moe::config`.

#![cfg(feature = "phimoe")]

use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::ArchResult;
use oxillama_arch::phi_moe::{load_phi_moe_from_gguf, PhiMoeModel};
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
const FFN: usize = 8; // per-expert intermediate size
const VOCAB: usize = 16;
const NUM_EXPERTS: usize = 2;

fn zeros(n: usize) -> Vec<u8> {
    vec![0u8; n * 4]
}

/// Build a minimal 1-layer Phi-MoE GGUF.
///
/// - `dimension_count`: sets `phimoe.rope.dimension_count` when `Some` (G5).
/// - `with_output`: whether a standalone `output.weight` is written (G10).
/// - `with_bias`: whether `attn_norm.bias`/`attn_output.bias`/`ffn_norm.bias`/
///   `output_norm.bias`/`output.bias` are written (G7), each set to `1.0` so
///   a forward pass can observe whether they were actually added.
fn build_phi_moe_gguf(dimension_count: Option<u32>, with_output: bool, with_bias: bool) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("phimoe".to_string()),
    );
    w.add_metadata("phimoe.embedding_length", MetadataValue::Uint32(H as u32));
    w.add_metadata(
        "phimoe.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("phimoe.block_count", MetadataValue::Uint32(1));
    w.add_metadata(
        "phimoe.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "phimoe.attention.head_count_kv",
        MetadataValue::Uint32(KV_HEADS as u32),
    );
    w.add_metadata(
        "phimoe.attention.key_length",
        MetadataValue::Uint32(HD as u32),
    );
    w.add_metadata("phimoe.context_length", MetadataValue::Uint32(64));
    w.add_metadata("phimoe.vocab_size", MetadataValue::Uint32(VOCAB as u32));
    w.add_metadata("phimoe.rope.freq_base", MetadataValue::Float32(10000.0));
    w.add_metadata(
        "phimoe.expert_count",
        MetadataValue::Uint32(NUM_EXPERTS as u32),
    );
    w.add_metadata("phimoe.expert_used_count", MetadataValue::Uint32(1));
    if let Some(dc) = dimension_count {
        w.add_metadata("phimoe.rope.dimension_count", MetadataValue::Uint32(dc));
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
    if with_bias {
        w.add_tensor(
            "output_norm.bias",
            &[H as u64],
            GgufTensorType::F32,
            &f32_bytes(&[1.0f32; H]),
        );
    }
    if with_output {
        w.add_tensor(
            "output.weight",
            &[H as u64, VOCAB as u64],
            GgufTensorType::F32,
            &zeros(VOCAB * H),
        );
        if with_bias {
            w.add_tensor(
                "output.bias",
                &[VOCAB as u64],
                GgufTensorType::F32,
                &f32_bytes(&[1.0f32; VOCAB]),
            );
        }
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
    if with_bias {
        w.add_tensor(
            &format!("{p}.attn_norm.bias"),
            &[H as u64],
            GgufTensorType::F32,
            &f32_bytes(&[1.0f32; H]),
        );
    }
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
    if with_bias {
        w.add_tensor(
            &format!("{p}.attn_output.bias"),
            &[H as u64],
            GgufTensorType::F32,
            &f32_bytes(&[1.0f32; H]),
        );
    }
    w.add_tensor(
        &format!("{p}.ffn_norm.weight"),
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    if with_bias {
        w.add_tensor(
            &format!("{p}.ffn_norm.bias"),
            &[H as u64],
            GgufTensorType::F32,
            &f32_bytes(&[1.0f32; H]),
        );
    }
    w.add_tensor(
        &format!("{p}.ffn_gate_inp.weight"),
        &[H as u64, NUM_EXPERTS as u64],
        GgufTensorType::F32,
        &zeros(H * NUM_EXPERTS),
    );
    w.add_tensor(
        &format!("{p}.ffn_gate_exps.weight"),
        &[H as u64, FFN as u64, NUM_EXPERTS as u64],
        GgufTensorType::F32,
        &zeros(H * FFN * NUM_EXPERTS),
    );
    w.add_tensor(
        &format!("{p}.ffn_up_exps.weight"),
        &[H as u64, FFN as u64, NUM_EXPERTS as u64],
        GgufTensorType::F32,
        &zeros(H * FFN * NUM_EXPERTS),
    );
    w.add_tensor(
        &format!("{p}.ffn_down_exps.weight"),
        &[FFN as u64, H as u64, NUM_EXPERTS as u64],
        GgufTensorType::F32,
        &zeros(FFN * H * NUM_EXPERTS),
    );

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic phi_moe GGUF must serialize");
    bytes
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn parse_and_config(bytes: Vec<u8>) -> (GgufModel, ModelConfig) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
    let config =
        ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
    (gguf, config)
}

fn load(bytes: Vec<u8>) -> PhiMoeModel {
    let (gguf, config) = parse_and_config(bytes);
    load_phi_moe_from_gguf(&gguf, &config).expect("phi_moe fixture must load")
}

// ─── G5: rope_dims from {arch}.rope.dimension_count, defaulting to head_dim ──

#[test]
fn g5_rope_dims_defaults_to_full_head_dim_when_dimension_count_absent() {
    let model = load(build_phi_moe_gguf(None, true, false));
    assert_eq!(
        model.rope_dims, HD,
        "absent {{arch}}.rope.dimension_count must default to the full head_dim"
    );
}

#[test]
fn g5_rope_dims_read_from_dimension_count_when_present() {
    let model = load(build_phi_moe_gguf(Some(4), true, false));
    assert_eq!(model.rope_dims, 4);
}

// ─── G7: optional bias tensors are loaded and actually applied ───────────────

#[test]
fn g7_bias_tensors_load_when_present() {
    let model = load(build_phi_moe_gguf(None, true, true));
    let layer = &model.layers[0];
    assert!(
        layer.attn_norm_bias.is_some(),
        "attn_norm.bias must load when present"
    );
    assert!(
        layer.attn_output_bias.is_some(),
        "attn_output.bias must load when present"
    );
    assert!(
        layer.ffn_norm_bias.is_some(),
        "ffn_norm.bias must load when present"
    );
    assert!(
        model.output_norm_bias.is_some(),
        "output_norm.bias must load when present"
    );
    assert!(
        model.output_bias.is_some(),
        "output.bias must load when present"
    );
}

#[test]
fn g7_bias_tensors_are_none_when_absent() {
    let model = load(build_phi_moe_gguf(None, true, false));
    let layer = &model.layers[0];
    assert!(layer.attn_norm_bias.is_none());
    assert!(layer.attn_output_bias.is_none());
    assert!(layer.ffn_norm_bias.is_none());
    assert!(model.output_norm_bias.is_none());
    assert!(model.output_bias.is_none());
}

/// Build a 1-layer fixture with `attn_output.weight`/`ffn_*`/`router` all
/// zero (so only bias terms can inject a nonzero value) but `output_norm`
/// and `output` given signal-preserving weights (`1.0` and identity,
/// respectively — `VOCAB == H` here, so identity is square), so a chain of
/// THREE biases (`attn_output.bias` → residual → `output_norm.bias` →
/// `output.bias`) is independently observable in the final logits rather
/// than being zeroed out by a downstream zero-weight matmul/RMSNorm.
fn build_bias_chain_gguf(with_bias: bool) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("phimoe".to_string()),
    );
    w.add_metadata("phimoe.embedding_length", MetadataValue::Uint32(H as u32));
    w.add_metadata(
        "phimoe.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("phimoe.block_count", MetadataValue::Uint32(1));
    w.add_metadata(
        "phimoe.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "phimoe.attention.head_count_kv",
        MetadataValue::Uint32(KV_HEADS as u32),
    );
    w.add_metadata(
        "phimoe.attention.key_length",
        MetadataValue::Uint32(HD as u32),
    );
    w.add_metadata("phimoe.context_length", MetadataValue::Uint32(64));
    w.add_metadata("phimoe.vocab_size", MetadataValue::Uint32(VOCAB as u32));
    w.add_metadata("phimoe.rope.freq_base", MetadataValue::Float32(10000.0));
    w.add_metadata(
        "phimoe.expert_count",
        MetadataValue::Uint32(NUM_EXPERTS as u32),
    );
    w.add_metadata("phimoe.expert_used_count", MetadataValue::Uint32(1));

    w.add_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &zeros(VOCAB * H),
    );
    // Signal-preserving: RMSNorm with weight=1 passes a constant vector
    // through at ~unit scale instead of RMSNorm-with-weight=0's hard zero.
    w.add_tensor(
        "output_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &f32_bytes(&[1.0f32; H]),
    );
    // Identity: logits[v] = hidden_final[v] for v < H (VOCAB == H here).
    let mut identity = vec![0.0f32; VOCAB * H];
    for i in 0..H.min(VOCAB) {
        identity[i * H + i] = 1.0;
    }
    w.add_tensor(
        "output.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &f32_bytes(&identity),
    );
    if with_bias {
        w.add_tensor(
            "output_norm.bias",
            &[H as u64],
            GgufTensorType::F32,
            &f32_bytes(&[1.0f32; H]),
        );
        w.add_tensor(
            "output.bias",
            &[VOCAB as u64],
            GgufTensorType::F32,
            &f32_bytes(&[1.0f32; VOCAB]),
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
    if with_bias {
        w.add_tensor(
            &format!("{p}.attn_output.bias"),
            &[H as u64],
            GgufTensorType::F32,
            &f32_bytes(&[1.0f32; H]),
        );
    }
    w.add_tensor(
        &format!("{p}.ffn_norm.weight"),
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    w.add_tensor(
        &format!("{p}.ffn_gate_inp.weight"),
        &[H as u64, NUM_EXPERTS as u64],
        GgufTensorType::F32,
        &zeros(H * NUM_EXPERTS),
    );
    w.add_tensor(
        &format!("{p}.ffn_gate_exps.weight"),
        &[H as u64, FFN as u64, NUM_EXPERTS as u64],
        GgufTensorType::F32,
        &zeros(H * FFN * NUM_EXPERTS),
    );
    w.add_tensor(
        &format!("{p}.ffn_up_exps.weight"),
        &[H as u64, FFN as u64, NUM_EXPERTS as u64],
        GgufTensorType::F32,
        &zeros(H * FFN * NUM_EXPERTS),
    );
    w.add_tensor(
        &format!("{p}.ffn_down_exps.weight"),
        &[FFN as u64, H as u64, NUM_EXPERTS as u64],
        GgufTensorType::F32,
        &zeros(FFN * H * NUM_EXPERTS),
    );

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic phi_moe bias-chain GGUF must serialize");
    bytes
}

/// The bias tensors must actually be ADDED, not just loaded and ignored.
///
/// With every projection weight zero except an identity `output.weight` and
/// a unit `output_norm.weight`, only `attn_output.bias` can inject a nonzero
/// residual, which `output_norm.bias` and `output.bias` then each add their
/// own contribution on top of — three independent bias applications chained
/// together, landing at ≈3.0 (1.0 from each) instead of exactly 0.0.
#[test]
fn g7_bias_chain_is_actually_applied_to_logits() {
    let mut with_bias = load(build_bias_chain_gguf(true));
    let mut without_bias = load(build_bias_chain_gguf(false));

    let kv_dim = KV_HEADS * HD;
    let mut kv_a = TestKv::new(kv_dim, 64);
    let mut kv_b = TestKv::new(kv_dim, 64);

    let logits_with_bias = with_bias
        .forward(&[0u32], &mut kv_a)
        .expect("forward with bias must succeed");
    let logits_without_bias = without_bias
        .forward(&[0u32], &mut kv_b)
        .expect("forward without bias must succeed");

    // No bias anywhere: hidden never leaves zero, so every logit is exactly 0.0.
    for (i, &v) in logits_without_bias.iter().enumerate() {
        assert_eq!(
            v, 0.0,
            "logit[{i}] must be exactly 0 with no bias tensors present"
        );
    }
    // attn_output.bias (≈1.0 after the residual) + output_norm.bias (≈1.0
    // after RMSNorm-with-weight=1 passes ≈1.0 through) + output.bias (1.0)
    // ≈ 3.0. The RMSNorm division introduces a tiny `eps`-scale error, so
    // this allows a small tolerance rather than requiring bit-exactness.
    for (i, &v) in logits_with_bias.iter().enumerate() {
        assert!(
            (v - 3.0).abs() < 1e-3,
            "logit[{i}] must reflect all three chained biases (~3.0), got {v}"
        );
    }
}

// ─── G8: out-of-vocabulary token ids error instead of panicking ──────────────

#[test]
fn g8_out_of_vocab_token_errors_instead_of_panicking() {
    let mut model = load(build_phi_moe_gguf(None, true, false));
    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(kv_dim, 64);

    let result = model.forward(&[VOCAB as u32], &mut kv);
    assert!(
        result.is_err(),
        "token id == vocab_size must error, not panic"
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
    let mut model = load(build_phi_moe_gguf(None, true, false));
    assert_eq!(model.max_context_length(), 64);
    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(kv_dim, 128);
    let too_many_tokens = vec![0u32; 65];
    let result = model.forward(&too_many_tokens, &mut kv);
    assert!(
        result.is_err(),
        "a prefill longer than max_context_length must error, not panic"
    );
}

#[test]
fn g9_prefill_exactly_at_max_context_length_is_ok() {
    let mut model = load(build_phi_moe_gguf(None, true, false));
    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(kv_dim, 128);
    let exactly_at_limit = vec![0u32; 64];
    let result = model.forward(&exactly_at_limit, &mut kv);
    assert!(
        result.is_ok(),
        "exactly at max_context_length must succeed: {:?}",
        result.err()
    );
}

// ─── G10: tied-embedding LM-head fallback ─────────────────────────────────────

#[test]
fn g10_missing_output_weight_falls_back_to_tied_token_embd() {
    let (gguf, config) = parse_and_config(build_phi_moe_gguf(None, false, false));
    assert!(!gguf.file.tensors.contains("output.weight"));
    let mut model = load_phi_moe_from_gguf(&gguf, &config)
        .expect("a tied checkpoint (no output.weight) must still load");
    assert_eq!(
        model.output_weight.len(),
        VOCAB * H,
        "tied LM head must keep token_embd's dimensions"
    );

    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(kv_dim, 64);
    let logits = model
        .forward(&[0u32], &mut kv)
        .expect("forward through the tied LM head must succeed");
    assert_eq!(logits.len(), VOCAB);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
    }
}

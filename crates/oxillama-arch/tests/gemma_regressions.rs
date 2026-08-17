//! Regression tests for confirmed Gemma correctness defects (G1-G11).
//!
//! Each test is named after the defect it guards. Where practical the test
//! was run against the pre-fix `gemma/model.rs` (via `git stash`) to confirm
//! it fails there and passes after the fix — see the fix report for the
//! captured before/after output.

#![cfg(feature = "gemma")]

use oxillama_arch::common::linear::QuantLinear;
use oxillama_arch::common::rms_norm::RmsNorm;
use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::ArchResult;
use oxillama_arch::gemma::{load_gemma_from_gguf, GemmaLayer, GemmaModel};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};
use oxillama_quant::{KernelDispatcher, QuantTensor};
use std::sync::Arc;

/// Minimal in-memory KV cache sized for a single layer/model under test.
struct TestKv {
    kv_dim: usize,
    max_seq: usize,
    position: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl TestKv {
    fn new(n_layers: usize, kv_dim: usize, max_seq: usize) -> Self {
        Self {
            kv_dim,
            max_seq,
            position: 0,
            keys: vec![vec![0.0f32; max_seq * kv_dim]; n_layers],
            values: vec![vec![0.0f32; max_seq * kv_dim]; n_layers],
        }
    }
}

impl KvCacheAccess for TestKv {
    fn seq_len(&self) -> usize {
        self.position
    }
    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let offset = self.position * self.kv_dim;
        let ck = key.len().min(self.kv_dim);
        let cv = value.len().min(self.kv_dim);
        self.keys[layer][offset..offset + ck].copy_from_slice(&key[..ck]);
        self.values[layer][offset..offset + cv].copy_from_slice(&value[..cv]);
        Ok(())
    }
    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        let end = (self.position + 1) * self.kv_dim;
        Ok(&self.keys[layer][..end])
    }
    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        let end = (self.position + 1) * self.kv_dim;
        Ok(&self.values[layer][..end])
    }
    fn advance(&mut self) {
        self.position = (self.position + 1).min(self.max_seq.saturating_sub(1));
    }
}

/// Build a zero-initialized F32 `QuantLinear` of math shape `[out, in]`.
fn f32_linear(out_features: usize, in_features: usize) -> QuantLinear {
    let data = vec![0u8; out_features * in_features * 4];
    let tensor = QuantTensor::new(data, vec![out_features, in_features], GgufTensorType::F32);
    QuantLinear::new(tensor, None)
}

fn f32_kernel() -> Arc<dyn oxillama_quant::QuantKernel> {
    KernelDispatcher::new()
        .get_kernel(GgufTensorType::F32)
        .expect("F32 kernel must be registered")
        .into()
}

// ─── G1: buf_attn_out sized `num_heads * head_dim`, not `hidden_size` ─────────

/// Gemma-2-9B's real attention geometry: hidden_size=3584, 16 query heads,
/// 8 KV heads, head_dim=256. `16 * 256 = 4096 != 3584`, so a
/// `buf_attn_out` sized as `hidden_size` is 512 elements too short — writing
/// head 14's slice (`[3584..3840]`) into a 3584-element `Vec` panics before
/// the fix (verified via `git stash` against the pre-fix file: "index out of
/// bounds ... range end index 3840 out of range for slice of length 3584" at
/// `gemma/model.rs`, inside `attention()`).
///
/// Layer/vocab/intermediate sizes are kept tiny (this is about the
/// attention-buffer shape, not about loading a real 9B checkpoint), per
/// `GemmaModel::new`'s doc comment on `buf_attn_out`.
#[test]
fn g1_gemma2_9b_shaped_attention_does_not_panic() {
    const HIDDEN: usize = 3584;
    const NUM_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 256;
    const INTERMEDIATE: usize = 32;
    const VOCAB: usize = 8;

    let config = ModelConfig {
        architecture: "gemma2".to_string(),
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        num_layers: 1,
        num_attention_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        vocab_size: VOCAB,
        max_context_length: 16,
        ..ModelConfig::default()
    };

    let kernel = f32_kernel();
    let q_dim = NUM_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;

    let layer = GemmaLayer {
        attn_norm: RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
        attn_post_norm: None,
        attn_q: f32_linear(q_dim, HIDDEN),
        attn_k: f32_linear(kv_dim, HIDDEN),
        attn_v: f32_linear(kv_dim, HIDDEN),
        attn_output: f32_linear(HIDDEN, q_dim),
        ffn_norm: RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
        ffn_post_norm: None,
        ffn_gate: f32_linear(INTERMEDIATE, HIDDEN),
        ffn_up: f32_linear(INTERMEDIATE, HIDDEN),
        ffn_down: f32_linear(HIDDEN, INTERMEDIATE),
        use_sliding_window: false,
        attn_q_norm: None,
        attn_k_norm: None,
        attn_q_kernel: kernel.clone(),
        attn_k_kernel: kernel.clone(),
        attn_v_kernel: kernel.clone(),
        attn_output_kernel: kernel.clone(),
        ffn_gate_kernel: kernel.clone(),
        ffn_up_kernel: kernel.clone(),
        ffn_down_kernel: kernel.clone(),
    };

    let output_norm = RmsNorm::new(vec![1.0; HIDDEN], 1e-5);

    let mut model = GemmaModel::new(
        config,
        vec![0.0f32; VOCAB * HIDDEN],
        vec![layer],
        output_norm,
        None, // weight-tied LM head
        0.0,
        0.0,
        None,
    )
    .expect("GemmaModel::new must succeed for a well-formed layer set");

    let mut kv = TestKv::new(1, kv_dim, 16);
    let result = model.forward(&[0u32], &mut kv);
    assert!(
        result.is_ok(),
        "Gemma-2-9B-shaped forward must not panic/error on head 14's slice: {:?}",
        result.err()
    );
    let logits = result.expect("checked above");
    assert_eq!(logits.len(), VOCAB);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
    }
}

// ─── Shared minimal-GGUF builder for G2/G3/G8/G9 (loader-level tests) ─────────

const H: usize = 16;
const HEADS: usize = 2;
const KV_HEADS: usize = 2;
const HD: usize = 8; // head_dim: HEADS * HD == H, keeps this fixture boring on G1's axis
const FFN: usize = 32;
const VOCAB: usize = 16;

fn zeros(n: usize) -> Vec<u8> {
    vec![0u8; n * 4]
}

/// Build a minimal N-layer Gemma GGUF. `gemma3_tensors` adds the
/// `attn_q_norm`/`attn_k_norm`/`attn_post_norm`/`ffn_post_norm` tensors every
/// Gemma-3 checkpoint ships; `extra_metadata` layers on architecture-specific
/// keys (softcap, sliding window, rope bases).
fn build_gemma_gguf(
    architecture: &str,
    num_layers: usize,
    gemma3_tensors: bool,
    extra_metadata: &[(&str, MetadataValue)],
) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String(architecture.to_string()),
    );
    w.add_metadata(
        &format!("{architecture}.embedding_length"),
        MetadataValue::Uint32(H as u32),
    );
    w.add_metadata(
        &format!("{architecture}.feed_forward_length"),
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata(
        &format!("{architecture}.block_count"),
        MetadataValue::Uint32(num_layers as u32),
    );
    w.add_metadata(
        &format!("{architecture}.attention.head_count"),
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        &format!("{architecture}.attention.head_count_kv"),
        MetadataValue::Uint32(KV_HEADS as u32),
    );
    w.add_metadata(
        &format!("{architecture}.attention.key_length"),
        MetadataValue::Uint32(HD as u32),
    );
    w.add_metadata(
        &format!("{architecture}.context_length"),
        MetadataValue::Uint32(64),
    );
    w.add_metadata(
        &format!("{architecture}.vocab_size"),
        MetadataValue::Uint32(VOCAB as u32),
    );
    w.add_metadata(
        &format!("{architecture}.rope.freq_base"),
        MetadataValue::Float32(10000.0),
    );
    for (k, v) in extra_metadata {
        w.add_metadata(k, v.clone());
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

    let q_dim = HEADS * HD;
    let kv_dim = KV_HEADS * HD;
    for i in 0..num_layers {
        let p = format!("blk.{i}");
        w.add_tensor(
            &format!("{p}.attn_norm.weight"),
            &[H as u64],
            GgufTensorType::F32,
            &zeros(H),
        );
        w.add_tensor(
            &format!("{p}.attn_q.weight"),
            &[H as u64, q_dim as u64],
            GgufTensorType::F32,
            &zeros(H * q_dim),
        );
        w.add_tensor(
            &format!("{p}.attn_k.weight"),
            &[H as u64, kv_dim as u64],
            GgufTensorType::F32,
            &zeros(H * kv_dim),
        );
        w.add_tensor(
            &format!("{p}.attn_v.weight"),
            &[H as u64, kv_dim as u64],
            GgufTensorType::F32,
            &zeros(H * kv_dim),
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
        if gemma3_tensors {
            w.add_tensor(
                &format!("{p}.attn_post_norm.weight"),
                &[H as u64],
                GgufTensorType::F32,
                &zeros(H),
            );
            w.add_tensor(
                &format!("{p}.ffn_post_norm.weight"),
                &[H as u64],
                GgufTensorType::F32,
                &zeros(H),
            );
            w.add_tensor(
                &format!("{p}.attn_q_norm.weight"),
                &[HD as u64],
                GgufTensorType::F32,
                &zeros(HD),
            );
            w.add_tensor(
                &format!("{p}.attn_k_norm.weight"),
                &[HD as u64],
                GgufTensorType::F32,
                &zeros(HD),
            );
        }
    }

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic gemma GGUF must serialize");
    bytes
}

fn parse_and_config(bytes: Vec<u8>) -> (GgufModel, ModelConfig) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
    let config =
        ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
    (gguf, config)
}

// ─── G2: correct soft-cap metadata keys, hard error for gemma2 ───────────────

#[test]
fn g2_gemma2_load_fails_without_softcap_metadata() {
    let bytes = build_gemma_gguf("gemma2", 1, false, &[]);
    let (gguf, config) = parse_and_config(bytes);
    let result = load_gemma_from_gguf(&gguf, &config);
    assert!(
        result.is_err(),
        "gemma2 with no attn/final softcap metadata must fail to load, not silently disable capping"
    );
}

#[test]
fn g2_gemma2_load_succeeds_with_correct_keys() {
    let bytes = build_gemma_gguf(
        "gemma2",
        1,
        false,
        &[
            (
                "gemma2.attn_logit_softcapping",
                MetadataValue::Float32(50.0),
            ),
            (
                "gemma2.final_logit_softcapping",
                MetadataValue::Float32(30.0),
            ),
        ],
    );
    let (gguf, config) = parse_and_config(bytes);
    let model = load_gemma_from_gguf(&gguf, &config).expect("must load with correct keys");
    assert!((model.attn_logit_softcap - 50.0).abs() < 1e-6);
    assert!((model.final_logit_softcap - 30.0).abs() < 1e-6);
}

#[test]
fn g2_gemma_v1_ignores_legacy_softcap_keys_present_under_wrong_architecture() {
    // Same fixture shape `build_minimal_gemma_gguf()` (oxillama-gguf
    // test_utils) uses: a v1 "gemma" checkpoint with the WRONG
    // (`attention.logit_softcap`) key names set. Gemma 1 must still load
    // with soft-capping disabled, not error and not apply the values.
    let bytes = build_gemma_gguf(
        "gemma",
        1,
        false,
        &[
            (
                "gemma.attention.logit_softcap",
                MetadataValue::Float32(50.0),
            ),
            ("gemma.final_logit_softcap", MetadataValue::Float32(30.0)),
        ],
    );
    let (gguf, config) = parse_and_config(bytes);
    let model = load_gemma_from_gguf(&gguf, &config).expect("gemma v1 must always load");
    assert_eq!(model.attn_logit_softcap, 0.0);
    assert_eq!(model.final_logit_softcap, 0.0);
}

// ─── G3: Gemma-3 gets its own SWA pattern, QK-norm, and dual RoPE bases ───────

#[test]
fn g3_gemma3_swa_pattern_and_qk_norm_load_correctly() {
    let bytes = build_gemma_gguf("gemma3", 7, true, &[]);
    let (gguf, config) = parse_and_config(bytes);
    let model = load_gemma_from_gguf(&gguf, &config).expect("gemma3 fixture must load");

    // set_swa_pattern(6): local for il % 6 < 5, global at il % 6 == 5.
    let expected_local = [true, true, true, true, true, false, true];
    for (i, &expect_local) in expected_local.iter().enumerate() {
        assert_eq!(
            model.layers[i].use_sliding_window, expect_local,
            "gemma3 layer {i}: expected use_sliding_window={expect_local}"
        );
    }

    for (i, layer) in model.layers.iter().enumerate() {
        assert!(
            layer.attn_q_norm.is_some(),
            "gemma3 layer {i} must load attn_q_norm when the tensor is present"
        );
        assert!(
            layer.attn_k_norm.is_some(),
            "gemma3 layer {i} must load attn_k_norm when the tensor is present"
        );
    }
}

#[test]
fn g3_gemma2_keeps_the_i_mod_2_pattern_not_gemma3s() {
    let bytes = build_gemma_gguf(
        "gemma2",
        4,
        false,
        &[
            (
                "gemma2.attn_logit_softcapping",
                MetadataValue::Float32(50.0),
            ),
            (
                "gemma2.final_logit_softcapping",
                MetadataValue::Float32(30.0),
            ),
        ],
    );
    let (gguf, config) = parse_and_config(bytes);
    let model = load_gemma_from_gguf(&gguf, &config).expect("gemma2 fixture must load");
    let expected_local = [true, false, true, false];
    for (i, &expect_local) in expected_local.iter().enumerate() {
        assert_eq!(
            model.layers[i].use_sliding_window, expect_local,
            "gemma2 layer {i}: i % 2 == 0 pattern must be preserved"
        );
    }
    // Gemma-2 has no QK-norm tensors in this fixture.
    assert!(model.layers[0].attn_q_norm.is_none());
}

#[test]
fn g3_gemma3_uses_distinct_local_rope_base_when_present() {
    let bytes = build_gemma_gguf(
        "gemma3",
        2,
        true,
        &[("gemma3.rope.freq_base_swa", MetadataValue::Float32(10000.0))],
    );
    let (gguf, config) = parse_and_config(bytes);
    // Global base defaults to 10000.0 too (no override in this fixture) —
    // rebuild with a DIFFERENT global base so the two tables are genuinely
    // distinct.
    let bytes2 = build_gemma_gguf(
        "gemma3",
        2,
        true,
        &[
            ("gemma3.rope.freq_base", MetadataValue::Float32(1_000_000.0)),
            ("gemma3.rope.freq_base_swa", MetadataValue::Float32(10000.0)),
        ],
    );
    let (gguf2, config2) = parse_and_config(bytes2);
    let _ = (gguf, config); // first fixture only demonstrates the key name is read

    let model = load_gemma_from_gguf(&gguf2, &config2).expect("gemma3 fixture must load");
    assert!(
        model.rope_local.is_some(),
        "distinct freq_base_swa must produce a separate local RoPE table"
    );
}

#[test]
fn g3_gemma3_shares_one_rope_table_when_freq_base_swa_absent() {
    let bytes = build_gemma_gguf("gemma3", 2, true, &[]);
    let (gguf, config) = parse_and_config(bytes);
    let model = load_gemma_from_gguf(&gguf, &config).expect("gemma3 fixture must load");
    assert!(
        model.rope_local.is_none(),
        "absent freq_base_swa must not allocate a second RoPE table"
    );
}

// ─── G8: out-of-vocabulary token ids error instead of panicking ──────────────

#[test]
fn g8_out_of_vocab_token_errors_instead_of_panicking() {
    let bytes = build_gemma_gguf("gemma", 1, false, &[]);
    let (gguf, config) = parse_and_config(bytes);
    let mut model = load_gemma_from_gguf(&gguf, &config).expect("fixture must load");
    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(1, kv_dim, 64);

    let result = model.forward(&[VOCAB as u32], &mut kv);
    assert!(
        result.is_err(),
        "token id == vocab_size must be reported as an error, not panic on the slice index"
    );

    let result2 = model.forward(&[u32::MAX], &mut kv);
    assert!(
        result2.is_err(),
        "a wildly out-of-range token id must not overflow into a panic either"
    );
}

// ─── G9: prefill past max_context_length errors instead of panicking ─────────

#[test]
fn g9_prefill_past_max_context_length_errors_instead_of_panicking() {
    let bytes = build_gemma_gguf("gemma", 1, false, &[]);
    let (gguf, config) = parse_and_config(bytes);
    let mut model = load_gemma_from_gguf(&gguf, &config).expect("fixture must load");
    assert_eq!(model.max_context_length(), 64);

    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(1, kv_dim, 128);
    let too_many_tokens = vec![0u32; 65]; // one past max_context_length=64
    let result = model.forward(&too_many_tokens, &mut kv);
    assert!(
        result.is_err(),
        "a prefill longer than max_context_length must error, not run RopeTable::apply/\
         buf_attn_scores past their max_context_length-sized allocation"
    );
}

#[test]
fn g9_prefill_exactly_at_max_context_length_is_ok() {
    let bytes = build_gemma_gguf("gemma", 1, false, &[]);
    let (gguf, config) = parse_and_config(bytes);
    let mut model = load_gemma_from_gguf(&gguf, &config).expect("fixture must load");
    let kv_dim = KV_HEADS * HD;
    let mut kv = TestKv::new(1, kv_dim, 128);
    let exactly_at_limit = vec![0u32; 64];
    let result = model.forward(&exactly_at_limit, &mut kv);
    assert!(
        result.is_ok(),
        "filling exactly to max_context_length must still succeed: {:?}",
        result.err()
    );
}

// ─── G11: a corrupted/undecodable optional post-norm must not go silent ──────

#[test]
fn g11_gemma1_without_post_norm_tensors_loads_with_none() {
    // Sanity check for the "genuinely absent" branch `load_optional_rms_norm`
    // must still return `Ok(None)` for: gemma-1 fixtures never ship
    // `attn_post_norm.weight`/`ffn_post_norm.weight`.
    let bytes = build_gemma_gguf("gemma", 1, false, &[]);
    let (gguf, config) = parse_and_config(bytes);
    let model = load_gemma_from_gguf(&gguf, &config).expect("fixture must load");
    assert!(model.layers[0].attn_post_norm.is_none());
    assert!(model.layers[0].ffn_post_norm.is_none());
}

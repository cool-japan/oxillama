//! MiniCPM (base) GGUF round-trip and regression tests.
//!
//! Two kinds of test live here:
//!
//! * **New capability** — `minicpm_round_trip_from_gguf_fixture` and its
//!   neighbours exercise `load_minicpm_from_gguf`, which did not exist before.
//!   MiniCPM was listed as "implemented" while having no GGUF loader at all;
//!   only `MiniCpmForward::new(config, …)` could build a model, so no
//!   checkpoint on disk was ever loadable.  These tests could not have been
//!   written against the old code.
//!
//! * **Regression** — `m1_*`, `m2_*`, `m3_*` pin the three confirmed
//!   correctness defects.  Each test's doc comment states why the old code was
//!   wrong.
//!
//! Fixtures come from two places.  `oxillama_gguf::test_utils::
//! build_minimal_minicpm_gguf()` is the shared fixture; its tensors are
//! zero-filled, which is fine for shape/parse assertions but useless for a
//! numeric comparison (0 · anything is 0).  The numeric tests therefore build
//! their own GGUFs with `oxillama_gguf::GgufWriter` and deterministic
//! non-trivial weights, differing *only* in the three scale keys.

#![cfg(feature = "minicpm")]

use oxillama_arch::common::rope::{RopeStyle, RopeTable};
use oxillama_arch::config::{rope_style_for_arch, ModelConfig};
use oxillama_arch::error::{ArchError, ArchResult};
use oxillama_arch::minicpm::{
    default_logit_scale, default_residual_scale, load_minicpm_from_gguf, MiniCpmConfig,
    DEFAULT_EMBEDDING_SCALE,
};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

// Shape of both the shared fixture and the locally-built ones.
const H: usize = 32; // hidden_size
const HEADS: usize = 4;
const KV_HEADS: usize = 2; // GQA: n_embd_gqa = 2 * 8 = 16
const HD: usize = H / HEADS; // head_dim = 8
const FFN: usize = 64;
const VOCAB: usize = 32;
const CTX: usize = 128;
const LAYERS: usize = 1;

const KV_DIM: usize = KV_HEADS * HD;

// ── Minimal KV cache ─────────────────────────────────────────────────────────

/// In-memory KV cache laid out exactly like the runtime's: one contiguous
/// `kv_dim`-wide row per token.
struct TestKv {
    kv_dim: usize,
    max_seq: usize,
    position: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl TestKv {
    fn new(layers: usize, kv_dim: usize, max_seq: usize) -> Self {
        Self {
            kv_dim,
            max_seq,
            position: 0,
            keys: vec![vec![0.0f32; max_seq * kv_dim]; layers],
            values: vec![vec![0.0f32; max_seq * kv_dim]; layers],
        }
    }
}

impl KvCacheAccess for TestKv {
    fn seq_len(&self) -> usize {
        self.position
    }
    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let offset = self.position * self.kv_dim;
        let end = offset + self.kv_dim;
        if layer >= self.keys.len() || end > self.keys[layer].len() {
            return Err(ArchError::ForwardPassError {
                layer,
                message: "test kv cache overflow".to_string(),
            });
        }
        self.keys[layer][offset..end].copy_from_slice(&key[..self.kv_dim]);
        self.values[layer][offset..end].copy_from_slice(&value[..self.kv_dim]);
        Ok(())
    }
    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        let end = ((self.position + 1) * self.kv_dim).min(self.keys[layer].len());
        Ok(&self.keys[layer][..end])
    }
    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        let end = ((self.position + 1) * self.kv_dim).min(self.values[layer].len());
        Ok(&self.values[layer][..end])
    }
    fn advance(&mut self) {
        self.position = (self.position + 1).min(self.max_seq);
    }
    fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

fn fresh_kv() -> TestKv {
    TestKv::new(LAYERS, KV_DIM, CTX + 2)
}

// ── Local GGUF construction ──────────────────────────────────────────────────

/// Deterministic, non-trivial F32 payload.
///
/// The shared fixture is zero-filled, which makes every logit zero regardless
/// of the scale factors; the numeric comparison in `m2_scales_change_logits`
/// needs weights that actually propagate signal.
fn weights(n: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = (((i * 3 + seed) % 7) as f32) * 0.1 - 0.3;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Which of the three MiniCPM scale keys to write, and with what value.
#[derive(Clone, Copy, Default)]
struct Scales {
    embedding: Option<f32>,
    residual: Option<f32>,
    logit: Option<f32>,
}

impl Scales {
    fn all(embedding: f32, residual: f32, logit: f32) -> Self {
        Self {
            embedding: Some(embedding),
            residual: Some(residual),
            logit: Some(logit),
        }
    }

    /// No scale keys at all — the "old GGUF" case that must fall back to
    /// llama.cpp's backward-compatibility defaults.
    fn omitted() -> Self {
        Self::default()
    }
}

/// Build a 1-layer MiniCPM GGUF with LLaMA's tensor layout.
///
/// `include_output_weight = false` omits `output.weight` to exercise the
/// tied-embedding fallback that llama.cpp spells `TENSOR_NOT_REQUIRED` +
/// `TENSOR_DUPLICATED` for `LLM_ARCH_MINICPM`.
fn build_minicpm(scales: Scales, include_output_weight: bool) -> Vec<u8> {
    build_minicpm_with_q_width(scales, include_output_weight, HEADS * HD)
}

/// As [`build_minicpm`], but with an explicit `attn_q` output width so a
/// deliberately mis-shaped checkpoint can be produced.
fn build_minicpm_with_q_width(
    scales: Scales,
    include_output_weight: bool,
    q_out_features: usize,
) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("minicpm".to_string()),
    );
    w.add_metadata("minicpm.embedding_length", MetadataValue::Uint32(H as u32));
    w.add_metadata(
        "minicpm.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("minicpm.block_count", MetadataValue::Uint32(LAYERS as u32));
    w.add_metadata(
        "minicpm.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "minicpm.attention.head_count_kv",
        MetadataValue::Uint32(KV_HEADS as u32),
    );
    w.add_metadata("minicpm.context_length", MetadataValue::Uint32(CTX as u32));
    w.add_metadata("minicpm.vocab_size", MetadataValue::Uint32(VOCAB as u32));
    w.add_metadata(
        "minicpm.attention.layer_norm_rms_epsilon",
        MetadataValue::Float32(1e-5),
    );
    w.add_metadata("minicpm.rope.freq_base", MetadataValue::Float32(10000.0));
    if let Some(v) = scales.embedding {
        w.add_metadata("minicpm.embedding_scale", MetadataValue::Float32(v));
    }
    if let Some(v) = scales.residual {
        w.add_metadata("minicpm.residual_scale", MetadataValue::Float32(v));
    }
    if let Some(v) = scales.logit {
        w.add_metadata("minicpm.logit_scale", MetadataValue::Float32(v));
    }

    // GGUF `ne` is fastest-changing-first, so a weight mapping
    // in_features -> out_features is written [in_features, out_features].
    w.add_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &weights(VOCAB * H, 1),
    );
    w.add_tensor(
        "output_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &weights(H, 2),
    );
    w.add_tensor(
        "blk.0.attn_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &weights(H, 3),
    );
    w.add_tensor(
        "blk.0.attn_q.weight",
        &[H as u64, q_out_features as u64],
        GgufTensorType::F32,
        &weights(H * q_out_features, 4),
    );
    w.add_tensor(
        "blk.0.attn_k.weight",
        &[H as u64, KV_DIM as u64],
        GgufTensorType::F32,
        &weights(H * KV_DIM, 5),
    );
    w.add_tensor(
        "blk.0.attn_v.weight",
        &[H as u64, KV_DIM as u64],
        GgufTensorType::F32,
        &weights(H * KV_DIM, 6),
    );
    w.add_tensor(
        "blk.0.attn_output.weight",
        &[(HEADS * HD) as u64, H as u64],
        GgufTensorType::F32,
        &weights(HEADS * HD * H, 7),
    );
    w.add_tensor(
        "blk.0.ffn_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &weights(H, 8),
    );
    w.add_tensor(
        "blk.0.ffn_gate.weight",
        &[H as u64, FFN as u64],
        GgufTensorType::F32,
        &weights(H * FFN, 9),
    );
    w.add_tensor(
        "blk.0.ffn_up.weight",
        &[H as u64, FFN as u64],
        GgufTensorType::F32,
        &weights(H * FFN, 10),
    );
    w.add_tensor(
        "blk.0.ffn_down.weight",
        &[FFN as u64, H as u64],
        GgufTensorType::F32,
        &weights(FFN * H, 11),
    );
    if include_output_weight {
        w.add_tensor(
            "output.weight",
            &[H as u64, VOCAB as u64],
            GgufTensorType::F32,
            &weights(VOCAB * H, 12),
        );
    }

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic minicpm GGUF must serialize");
    bytes
}

fn parse(bytes: Vec<u8>) -> (GgufModel, ModelConfig) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata must parse");
    (gguf, config)
}

// ── 1. Round-trip (new capability) ───────────────────────────────────────────

/// **New capability.** Load the shared `build_minimal_minicpm_gguf()` fixture
/// end to end and run a forward pass.
///
/// Before this change `oxillama-arch` had no `load_minicpm_from_gguf` at all,
/// so no MiniCPM GGUF could be turned into a runnable model.
#[test]
fn minicpm_round_trip_from_gguf_fixture() {
    let bytes = oxillama_gguf::test_utils::build_minimal_minicpm_gguf();
    let (gguf, config) = parse(bytes);
    assert_eq!(config.architecture, "minicpm");

    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("fixture must load");
    assert_eq!(model.vocab_size(), VOCAB);
    assert_eq!(model.hidden_size(), H);
    assert_eq!(model.max_context_length(), CTX);

    let mut kv = fresh_kv();
    let logits = model.forward(&[1u32, 2, 3], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must all be finite"
    );
}

/// **New capability.** The loader honours the tied-embedding fallback: a GGUF
/// without `output.weight` reuses `token_embd.weight` as the LM head, exactly
/// as llama.cpp does for `LLM_ARCH_MINICPM`.
#[test]
fn minicpm_loads_without_output_weight_via_tied_embedding() {
    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), false));
    let mut model =
        load_minicpm_from_gguf(&gguf, &config).expect("tied-embedding checkpoint must load");
    let mut kv = fresh_kv();
    let logits = model.forward(&[1u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// **New capability.** `embed()` returns the post-output-norm hidden state
/// (hidden_size, not vocab_size) and `embed_all()` returns one per token.
#[test]
fn minicpm_embed_paths_return_hidden_states() {
    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));
    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("load");

    let mut kv = fresh_kv();
    let embedded = model.embed(&[1u32, 2], &mut kv).expect("embed");
    assert_eq!(embedded.len(), H);

    let mut kv_all = fresh_kv();
    let all = model
        .embed_all(&[1u32, 2, 3], &mut kv_all)
        .expect("embed_all");
    assert_eq!(all.len(), 3 * H);
}

// ── 2. M1 regression: per-token prefill ──────────────────────────────────────

/// **M1 regression.** A 3-token prompt in a single `forward()` must advance
/// the KV cache by exactly 3.
///
/// The old `run_layers` read `let token = tokens[tokens.len() - 1];` under a
/// comment reading "For simplicity, process the last token (single-token step
/// mode)".  Every prompt token but the last was discarded, one KV row was
/// written, and the model was conditioned on a single token no matter how long
/// the prompt was — so this assertion returned 1 against the old code.
#[test]
fn m1_multi_token_prefill_advances_cache_once_per_token() {
    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));
    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("load");

    let mut kv = fresh_kv();
    model.forward(&[1u32, 2, 3], &mut kv).expect("forward");
    assert_eq!(
        kv.seq_len(),
        3,
        "three prompt tokens must write three KV rows, not one"
    );
}

/// **M1 regression.** `embed()` takes the same per-token path.
#[test]
fn m1_embed_also_advances_cache_once_per_token() {
    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));
    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("load");

    let mut kv = fresh_kv();
    model.embed(&[1u32, 2, 3, 4], &mut kv).expect("embed");
    assert_eq!(kv.seq_len(), 4);
}

/// **M1 regression.** Prefilling `[a, b]` then decoding `[c]` must land in the
/// same state as prefilling `[a, b, c]` in one call — which is only true if
/// every token is actually processed at its own position.
#[test]
fn m1_incremental_and_batched_prefill_agree() {
    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));
    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("load");

    let mut kv_batch = fresh_kv();
    let batched = model
        .forward(&[5u32, 6, 7], &mut kv_batch)
        .expect("batched");

    let mut kv_step = fresh_kv();
    model.forward(&[5u32, 6], &mut kv_step).expect("prefill");
    let stepped = model.forward(&[7u32], &mut kv_step).expect("decode");

    assert_eq!(kv_batch.seq_len(), kv_step.seq_len());
    for (i, (a, b)) in batched.iter().zip(stepped.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "logit {i} diverges: batched {a} vs stepped {b}"
        );
    }
}

// ── 3. M2 regression: the three scale factors ────────────────────────────────

/// **M2 regression.** The three MiniCPM scale factors are read from GGUF
/// metadata with their real values.
///
/// This test cannot even *compile* against the pre-fix `MiniCpmConfig`:
/// `residual_scale` and `logit_scale` did not exist as fields, and
/// `embedding_scale` was hard-coded to `1.0` in `from_model_config` (derived
/// from a `dim_model_base` that defaulted to `hidden_size`).  Its doc comment
/// described `embedding_scale` as `hidden_size / dim_model_base`, which is
/// **`logit_scale`'s** formula in `convert_hf_to_gguf.py::MiniCPMModel`, not
/// `embedding_scale`'s (that one is the raw HF `scale_emb`).
#[test]
fn m2_scale_factors_are_read_from_metadata() {
    let bytes = oxillama_gguf::test_utils::build_minimal_minicpm_gguf();
    let (gguf, config) = parse(bytes);
    let cfg = MiniCpmConfig::from_metadata(&config, &gguf.file.metadata).expect("minicpm config");

    assert!(
        (cfg.embedding_scale - 2.0).abs() < 1e-6,
        "minicpm.embedding_scale = 2.0, got {}",
        cfg.embedding_scale
    );
    assert!(
        (cfg.residual_scale - 0.5).abs() < 1e-6,
        "minicpm.residual_scale = 0.5, got {}",
        cfg.residual_scale
    );
    assert!(
        (cfg.logit_scale - 4.0).abs() < 1e-6,
        "minicpm.logit_scale = 4.0, got {}",
        cfg.logit_scale
    );
}

/// **M2 regression.** The scales are wired into the compute path, not merely
/// parsed: two checkpoints with byte-identical weights but different scale
/// keys must produce different logits.
///
/// The pre-fix forward pass applied **none** of the three, so both runs would
/// have been identical here.
#[test]
fn m2_scales_change_logits() {
    let (gguf_scaled, cfg_scaled) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));
    let (gguf_plain, cfg_plain) = parse(build_minicpm(Scales::all(1.0, 1.0, 1.0), true));

    let mut scaled = load_minicpm_from_gguf(&gguf_scaled, &cfg_scaled).expect("load scaled");
    let mut plain = load_minicpm_from_gguf(&gguf_plain, &cfg_plain).expect("load plain");

    let mut kv_a = fresh_kv();
    let mut kv_b = fresh_kv();
    let a = scaled.forward(&[1u32, 2, 3], &mut kv_a).expect("scaled");
    let b = plain.forward(&[1u32, 2, 3], &mut kv_b).expect("plain");

    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().any(|v| v.abs() > 1e-6),
        "the fixture weights must produce non-zero logits, otherwise this \
         comparison proves nothing"
    );
    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
        "embedding_scale / residual_scale / logit_scale must alter the logits"
    );
}

/// **M2 regression.** Isolating `logit_scale`: dividing the logits by 4
/// instead of 1, with everything else equal, must scale every logit by 0.25.
///
/// `granite.cpp` ends with `ggml_scale(ctx0, cur, 1.0f /
/// hparams.f_logit_scale)`, i.e. the logits are *divided* by `logit_scale`.
#[test]
fn m2_logit_scale_divides_the_logits() {
    let (gguf_one, cfg_one) = parse(build_minicpm(Scales::all(2.0, 0.5, 1.0), true));
    let (gguf_four, cfg_four) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));

    let mut one = load_minicpm_from_gguf(&gguf_one, &cfg_one).expect("load");
    let mut four = load_minicpm_from_gguf(&gguf_four, &cfg_four).expect("load");

    let mut kv_a = fresh_kv();
    let mut kv_b = fresh_kv();
    let unscaled = one.forward(&[1u32, 2], &mut kv_a).expect("forward");
    let quartered = four.forward(&[1u32, 2], &mut kv_b).expect("forward");

    assert!(unscaled.iter().any(|v| v.abs() > 1e-6));
    for (i, (u, q)) in unscaled.iter().zip(quartered.iter()).enumerate() {
        assert!(
            (u * 0.25 - q).abs() < 1e-4,
            "logit {i}: expected {} * 0.25 == {}, got {}",
            u,
            u * 0.25,
            q
        );
    }
}

/// **M2 regression.** A GGUF that predates the three keys falls back to
/// llama.cpp's backward-compatibility defaults from
/// `src/llama-model.cpp`'s `case LLM_ARCH_MINICPM:` — **not** to
/// `ModelConfig::logit_scale`'s generic default of `1.0`.
#[test]
fn m2_missing_scale_keys_use_llama_cpp_defaults() {
    let (gguf, config) = parse(build_minicpm(Scales::omitted(), true));
    let cfg = MiniCpmConfig::from_metadata(&config, &gguf.file.metadata).expect("minicpm config");

    assert!(
        (cfg.embedding_scale - DEFAULT_EMBEDDING_SCALE).abs() < 1e-6,
        "embedding_scale must default to 12.0, got {}",
        cfg.embedding_scale
    );
    assert!(
        (cfg.residual_scale - default_residual_scale(LAYERS)).abs() < 1e-6,
        "residual_scale must default to 1.4 / sqrt(n_layers) = {}, got {}",
        default_residual_scale(LAYERS),
        cfg.residual_scale
    );
    assert!(
        (cfg.logit_scale - default_logit_scale(H)).abs() < 1e-6,
        "logit_scale must default to 256 / hidden_size = {}, got {}",
        default_logit_scale(H),
        cfg.logit_scale
    );
    assert!(
        (cfg.logit_scale - 1.0).abs() > 1e-6,
        "the generic ModelConfig::logit_scale default of 1.0 is wrong for MiniCPM"
    );

    // The defaults must still produce a runnable model.
    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("load");
    let mut kv = fresh_kv();
    let logits = model.forward(&[1u32], &mut kv).expect("forward");
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ── 4. M3: RoPE style ────────────────────────────────────────────────────────

/// **M3 (already resolved upstream).** MiniCPM is `LLAMA_ROPE_TYPE_NORM`, and
/// the loader consumes the shared selector rather than hard-coding NeoX.
#[test]
fn m3_minicpm_uses_norm_rope_style() {
    assert_eq!(rope_style_for_arch("minicpm"), RopeStyle::Norm);

    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));
    assert_eq!(config.rope_style(), RopeStyle::Norm);
    // The loader must accept the file that declares that architecture.
    load_minicpm_from_gguf(&gguf, &config).expect("load");
}

// ── 5. Shared Part-3 guards ──────────────────────────────────────────────────

/// An out-of-vocabulary token id is reported, never used to index the
/// embedding table.
#[test]
fn out_of_vocabulary_token_returns_error() {
    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));
    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("load");
    let mut kv = fresh_kv();

    let err = model.forward(&[VOCAB as u32], &mut kv);
    assert!(
        matches!(err, Err(ArchError::ConfigMismatch { .. })),
        "an OOV token id must return ConfigMismatch, got {err:?}"
    );
    assert_eq!(
        kv.seq_len(),
        0,
        "a rejected prompt must not touch the cache"
    );
}

/// A prompt longer than the model's context is reported before any buffer is
/// indexed by position.
#[test]
fn context_overflow_returns_error() {
    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));
    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("load");
    let mut kv = fresh_kv();

    let tokens = vec![1u32; CTX + 1];
    let err = model.forward(&tokens, &mut kv);
    assert!(
        matches!(err, Err(ArchError::ConfigMismatch { .. })),
        "an over-long prompt must return ConfigMismatch, got {err:?}"
    );
    assert_eq!(kv.seq_len(), 0);
}

/// A missing required tensor is named, not panicked on.
#[test]
fn missing_required_tensor_is_reported() {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("minicpm".to_string()),
    );
    w.add_metadata("minicpm.embedding_length", MetadataValue::Uint32(H as u32));
    w.add_metadata(
        "minicpm.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("minicpm.block_count", MetadataValue::Uint32(1));
    w.add_metadata(
        "minicpm.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "minicpm.attention.head_count_kv",
        MetadataValue::Uint32(KV_HEADS as u32),
    );
    w.add_metadata("minicpm.context_length", MetadataValue::Uint32(CTX as u32));
    w.add_metadata("minicpm.vocab_size", MetadataValue::Uint32(VOCAB as u32));
    // Only the embedding table — every block tensor is missing.
    w.add_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &weights(VOCAB * H, 1),
    );
    let mut bytes = Vec::new();
    w.write_to(&mut bytes).expect("serialize");

    let (gguf, config) = parse(bytes);
    // `MiniCpmForward` is not `Debug` (it owns mmap-backed weights), so the
    // success arm is collapsed before matching.
    let err = load_minicpm_from_gguf(&gguf, &config).map(|_| ());
    match err {
        Err(ArchError::MissingTensor { name }) => {
            assert!(
                name.contains("blk.0"),
                "the error must name the missing tensor, got {name}"
            );
        }
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

/// A projection whose GGUF shape contradicts the configuration is rejected at
/// load time rather than corrupting a GEMV.
#[test]
fn mismatched_projection_shape_is_reported() {
    // Otherwise complete checkpoint whose query projection is half the width
    // `head_count * head_dim` demands.
    let bytes = build_minicpm_with_q_width(Scales::all(2.0, 0.5, 4.0), true, HEADS * HD / 2);
    let (gguf, config) = parse(bytes);
    match load_minicpm_from_gguf(&gguf, &config).map(|_| ()) {
        Err(ArchError::InvalidShape { name, .. }) => {
            assert!(name.contains("attn_q"), "unexpected tensor named: {name}");
        }
        other => panic!("expected InvalidShape for blk.0.attn_q.weight, got {other:?}"),
    }
}

/// A `logit_scale` of zero would make the graph divide by zero.
#[test]
fn zero_logit_scale_is_rejected_at_load() {
    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 0.0), true));
    assert!(matches!(
        load_minicpm_from_gguf(&gguf, &config),
        Err(ArchError::InvalidConfig { .. })
    ));
}

// ── 6. Independent reference forward ─────────────────────────────────────────
//
// Everything above compares the model against itself ("these two runs differ").
// The tests below compare it against an independently written plain-`f32`
// forward pass built from the same weight generator, so they pin *values*, not
// just differences: the GQA KV stride, the orientation of every non-square
// projection, and the exact placement of all three scale factors.

/// Same values [`weights`] serialises, as `f32`.
fn f32_weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 3 + seed) % 7) as f32) * 0.1 - 0.3)
        .collect()
}

/// `output[i] = (x[i] / rms(x)) * weight[i]`, matching `common::rms_norm`.
fn ref_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

/// Row-major `[out_features, in_features]` matrix-vector product.
fn ref_gemv(w: &[f32], x: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    (0..out_features)
        .map(|o| {
            w[o * in_features..(o + 1) * in_features]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn ref_softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// A from-scratch MiniCPM forward pass over the fixture built by
/// [`build_minicpm`], written directly against llama.cpp's `granite.cpp`
/// graph rather than by reading `minicpm/forward.rs`.
///
/// `gguf_linear_shape` reverses a tensor's `ne` and leaves the payload alone,
/// so a GGUF written with dims `[in, out]` *is* a row-major `[out, in]`
/// matrix — which is how the weights are indexed here.
///
/// RoPE comes from the shared [`RopeTable`] because the rotation itself is not
/// what these tests pin; everything else is recomputed.
fn reference_logits(
    tokens: &[u32],
    embedding_scale: f32,
    residual_scale: f32,
    logit_scale: f32,
) -> Vec<f32> {
    const EPS: f32 = 1e-5;
    const Q_DIM: usize = HEADS * HD;

    let tok_embd = f32_weights(VOCAB * H, 1);
    let output_norm_w = f32_weights(H, 2);
    let attn_norm_w = f32_weights(H, 3);
    let wq = f32_weights(H * Q_DIM, 4);
    let wk = f32_weights(H * KV_DIM, 5);
    let wv = f32_weights(H * KV_DIM, 6);
    let wo = f32_weights(Q_DIM * H, 7);
    let ffn_norm_w = f32_weights(H, 8);
    let w_gate = f32_weights(H * FFN, 9);
    let w_up = f32_weights(H * FFN, 10);
    let w_down = f32_weights(FFN * H, 11);
    let w_out = f32_weights(VOCAB * H, 12);

    let rope = RopeTable::new_standard_with_style(HD, CTX, 10000.0, RopeStyle::Norm);
    let heads_per_kv = HEADS / KV_HEADS;
    let attn_scale = 1.0 / (HD as f32).sqrt();

    // One contiguous `KV_DIM`-wide row per cached token.
    let mut cached_keys: Vec<f32> = Vec::new();
    let mut cached_values: Vec<f32> = Vec::new();
    let mut normed = vec![0.0f32; H];

    for (position, &token) in tokens.iter().enumerate() {
        // Embedding + embedding_scale (once, before layer 0).
        let base = token as usize * H;
        let mut x: Vec<f32> = tok_embd[base..base + H]
            .iter()
            .map(|v| v * embedding_scale)
            .collect();

        // ── Attention ───────────────────────────────────────────────────
        let h = ref_rms_norm(&x, &attn_norm_w, EPS);
        let mut q = ref_gemv(&wq, &h, Q_DIM, H);
        let mut k = ref_gemv(&wk, &h, KV_DIM, H);
        let v = ref_gemv(&wv, &h, KV_DIM, H);
        for head in 0..HEADS {
            rope.apply(&mut q[head * HD..(head + 1) * HD], position);
        }
        for head in 0..KV_HEADS {
            rope.apply(&mut k[head * HD..(head + 1) * HD], position);
        }
        cached_keys.extend_from_slice(&k);
        cached_values.extend_from_slice(&v);

        let seq_len = position + 1;
        let mut attn_heads = vec![0.0f32; Q_DIM];
        for head in 0..HEADS {
            let kv_head = head / heads_per_kv;
            let q_head = &q[head * HD..(head + 1) * HD];

            let mut scores = vec![0.0f32; seq_len];
            for (pos, score) in scores.iter_mut().enumerate() {
                // The stride is the full KV row, not `head_dim`.
                let off = pos * KV_DIM + kv_head * HD;
                *score = q_head
                    .iter()
                    .zip(&cached_keys[off..off + HD])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * attn_scale;
            }
            ref_softmax(&mut scores);

            for (pos, &w) in scores.iter().enumerate() {
                let off = pos * KV_DIM + kv_head * HD;
                for d in 0..HD {
                    attn_heads[head * HD + d] += w * cached_values[off + d];
                }
            }
        }

        // residual_scale on the attention output, then the residual add.
        let attn_out = ref_gemv(&wo, &attn_heads, H, Q_DIM);
        for (xi, a) in x.iter_mut().zip(attn_out.iter()) {
            *xi += a * residual_scale;
        }

        // ── FFN ─────────────────────────────────────────────────────────
        let hf = ref_rms_norm(&x, &ffn_norm_w, EPS);
        let mut gate = ref_gemv(&w_gate, &hf, FFN, H);
        let up = ref_gemv(&w_up, &hf, FFN, H);
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            let silu = *g / (1.0 + (-*g).exp());
            *g = silu * u;
        }

        // residual_scale on the FFN output, then the second residual add.
        let ffn_out = ref_gemv(&w_down, &gate, H, FFN);
        for (xi, f) in x.iter_mut().zip(ffn_out.iter()) {
            *xi += f * residual_scale;
        }

        normed = ref_rms_norm(&x, &output_norm_w, EPS);
    }

    let mut logits = ref_gemv(&w_out, &normed, VOCAB, H);
    let inv = 1.0 / logit_scale;
    for l in logits.iter_mut() {
        *l *= inv;
    }
    logits
}

/// **M4 regression + value check.** The loaded model must agree numerically
/// with the independent reference forward pass.
///
/// This is the test that discriminates the GQA KV stride (M4).  The old `sdpa`
/// used `kv_stride = head_dim` instead of `n_kv_heads * head_dim`, so for
/// `kv_heads = 2` the query for KV head 1 at position `p` read the slice where
/// KV head 0 of position `p + 1` lives.  That is in-bounds and finite, so no
/// "results differ" style assertion can see it — only a value comparison can.
/// The fixture is deliberately grouped (`4` query heads over `2` KV heads);
/// the defect is invisible at `n_kv_heads == 1`.
///
/// The same comparison pins the three scale *placements* (scaling the branch
/// rather than the stream, at both residual adds) and the orientation of every
/// non-square projection — `attn_k`/`attn_v` are `16 × 32` and `ffn_down` is
/// `32 × 64`, so a transposed read cannot agree by symmetry.
#[test]
fn m4_matches_independent_reference_forward() {
    let (gguf, config) = parse(build_minicpm(Scales::all(2.0, 0.5, 4.0), true));
    assert_eq!(config.num_kv_heads, KV_HEADS, "fixture must be GQA");
    assert_ne!(
        config.num_attention_heads, config.num_kv_heads,
        "fixture must actually exercise grouped-query attention"
    );

    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("load");
    let mut kv = fresh_kv();
    let tokens = [3u32, 9, 17];
    let got = model.forward(&tokens, &mut kv).expect("forward");
    let want = reference_logits(&tokens, 2.0, 0.5, 4.0);

    assert_eq!(got.len(), want.len());
    assert!(
        want.iter().any(|v| v.abs() > 1e-3),
        "the reference must produce non-trivial logits, otherwise agreement \
         proves nothing"
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "logit {i}: model {g} vs reference {w}"
        );
    }
}

/// The reference agreement must also hold for a single-token step and for the
/// no-op scale configuration.
#[test]
fn m4_reference_agreement_holds_for_single_token_and_unit_scales() {
    let (gguf, config) = parse(build_minicpm(Scales::all(1.0, 1.0, 1.0), true));
    let mut model = load_minicpm_from_gguf(&gguf, &config).expect("load");

    let mut kv = fresh_kv();
    let got = model.forward(&[5u32], &mut kv).expect("forward");
    let want = reference_logits(&[5u32], 1.0, 1.0, 1.0);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "logit {i}: model {g} vs reference {w}"
        );
    }
}

/// The reference itself must be sensitive to the KV stride, otherwise the
/// agreement test above would pass against a mis-strided implementation.
///
/// Recomputing the reference with the *wrong* stride (`head_dim` instead of
/// `n_kv_heads * head_dim`) must change the answer — which is exactly the
/// difference between the old `sdpa` and the fixed one.
#[test]
fn m4_wrong_kv_stride_would_change_the_answer() {
    let tokens = [3u32, 9, 17];
    let correct = reference_logits(&tokens, 2.0, 0.5, 4.0);
    let mis_strided = reference_logits_with_kv_stride(&tokens, 2.0, 0.5, 4.0, HD);
    assert!(
        correct
            .iter()
            .zip(mis_strided.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "if the two strides agreed, the reference could not discriminate M4"
    );
}

/// [`reference_logits`] with an injectable KV-cache stride, so a test can show
/// the wrong stride really does produce a different answer.
fn reference_logits_with_kv_stride(
    tokens: &[u32],
    embedding_scale: f32,
    residual_scale: f32,
    logit_scale: f32,
    kv_stride: usize,
) -> Vec<f32> {
    const EPS: f32 = 1e-5;
    const Q_DIM: usize = HEADS * HD;

    let tok_embd = f32_weights(VOCAB * H, 1);
    let output_norm_w = f32_weights(H, 2);
    let attn_norm_w = f32_weights(H, 3);
    let wq = f32_weights(H * Q_DIM, 4);
    let wk = f32_weights(H * KV_DIM, 5);
    let wv = f32_weights(H * KV_DIM, 6);
    let wo = f32_weights(Q_DIM * H, 7);
    let ffn_norm_w = f32_weights(H, 8);
    let w_gate = f32_weights(H * FFN, 9);
    let w_up = f32_weights(H * FFN, 10);
    let w_down = f32_weights(FFN * H, 11);
    let w_out = f32_weights(VOCAB * H, 12);

    let rope = RopeTable::new_standard_with_style(HD, CTX, 10000.0, RopeStyle::Norm);
    let heads_per_kv = HEADS / KV_HEADS;
    let attn_scale = 1.0 / (HD as f32).sqrt();

    let mut cached_keys: Vec<f32> = Vec::new();
    let mut cached_values: Vec<f32> = Vec::new();
    let mut normed = vec![0.0f32; H];

    for (position, &token) in tokens.iter().enumerate() {
        let base = token as usize * H;
        let mut x: Vec<f32> = tok_embd[base..base + H]
            .iter()
            .map(|v| v * embedding_scale)
            .collect();

        let h = ref_rms_norm(&x, &attn_norm_w, EPS);
        let mut q = ref_gemv(&wq, &h, Q_DIM, H);
        let mut k = ref_gemv(&wk, &h, KV_DIM, H);
        let v = ref_gemv(&wv, &h, KV_DIM, H);
        for head in 0..HEADS {
            rope.apply(&mut q[head * HD..(head + 1) * HD], position);
        }
        for head in 0..KV_HEADS {
            rope.apply(&mut k[head * HD..(head + 1) * HD], position);
        }
        cached_keys.extend_from_slice(&k);
        cached_values.extend_from_slice(&v);

        let seq_len = position + 1;
        let mut attn_heads = vec![0.0f32; Q_DIM];
        for head in 0..HEADS {
            let kv_head = head / heads_per_kv;
            let q_head = &q[head * HD..(head + 1) * HD];

            let mut scores = vec![0.0f32; seq_len];
            for (pos, score) in scores.iter_mut().enumerate() {
                let off = (pos * kv_stride + kv_head * HD).min(cached_keys.len() - HD);
                *score = q_head
                    .iter()
                    .zip(&cached_keys[off..off + HD])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * attn_scale;
            }
            ref_softmax(&mut scores);

            for (pos, &w) in scores.iter().enumerate() {
                let off = (pos * kv_stride + kv_head * HD).min(cached_values.len() - HD);
                for d in 0..HD {
                    attn_heads[head * HD + d] += w * cached_values[off + d];
                }
            }
        }

        let attn_out = ref_gemv(&wo, &attn_heads, H, Q_DIM);
        for (xi, a) in x.iter_mut().zip(attn_out.iter()) {
            *xi += a * residual_scale;
        }

        let hf = ref_rms_norm(&x, &ffn_norm_w, EPS);
        let mut gate = ref_gemv(&w_gate, &hf, FFN, H);
        let up = ref_gemv(&w_up, &hf, FFN, H);
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            let silu = *g / (1.0 + (-*g).exp());
            *g = silu * u;
        }

        let ffn_out = ref_gemv(&w_down, &gate, H, FFN);
        for (xi, f) in x.iter_mut().zip(ffn_out.iter()) {
            *xi += f * residual_scale;
        }

        normed = ref_rms_norm(&x, &output_norm_w, EPS);
    }

    let mut logits = ref_gemv(&w_out, &normed, VOCAB, H);
    let inv = 1.0 / logit_scale;
    for l in logits.iter_mut() {
        *l *= inv;
    }
    logits
}

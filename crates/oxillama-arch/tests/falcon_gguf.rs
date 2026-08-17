//! Falcon GGUF loading and forward-pass integration tests.
//!
//! # What is being tested
//!
//! `load_falcon_from_gguf` is a **new capability**: before this change the
//! Falcon architecture had no GGUF loader at all.  `FalconArchitecture::build`
//! unconditionally returned
//! `ArchError::MissingTensor { name: "token_embd.weight (...)" }`, and nothing
//! anywhere else in the workspace could turn a Falcon checkpoint into a
//! `FalconForward`.  The round-trip test below therefore has no "before"
//! failing state to demonstrate — there was no code path to fail.
//!
//! The other tests in this file *are* regressions and each names the defect it
//! pins, with the pre-fix behaviour spelled out.
//!
//! # Reference
//!
//! Every tensor name and topology claim here is taken from
//! `~/work/refs/llama.cpp`:
//! * `src/llama-model.cpp`, `case LLM_ARCH_FALCON:` in `load_tensors()`
//! * `src/llama-arch.cpp`, `LLM_TENSOR_NAMES` (`ATTN_NORM_2` →
//!   `"blk.%d.attn_norm_2"`)
//! * `src/models/falcon.cpp`, `llm_build_falcon`

use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::{ArchError, ArchResult};
use oxillama_arch::falcon::{load_falcon_from_gguf, FalconConfig, FalconForward};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

// ── Test scaffolding ─────────────────────────────────────────────────────────

/// Minimal in-memory KV cache.
///
/// Layout matches what the Falcon attention loop expects: every stored
/// position appends all K/V heads back to back, so a position's stride is
/// `n_kv_heads * head_dim`.
struct TestKv {
    seq_len: usize,
    keys: Vec<Vec<f32>>,
    vals: Vec<Vec<f32>>,
}

impl TestKv {
    fn new(n_layers: usize) -> Self {
        Self {
            seq_len: 0,
            keys: vec![Vec::new(); n_layers],
            vals: vec![Vec::new(); n_layers],
        }
    }
}

impl KvCacheAccess for TestKv {
    fn seq_len(&self) -> usize {
        self.seq_len
    }
    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        self.keys[layer].extend_from_slice(key);
        self.vals[layer].extend_from_slice(value);
        Ok(())
    }
    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.keys[layer])
    }
    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.vals[layer])
    }
    fn advance(&mut self) {
        self.seq_len += 1;
    }
}

/// Parse GGUF bytes and load a Falcon model out of them.
fn load(bytes: Vec<u8>) -> (GgufModel, FalconForward) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic Falcon GGUF must parse");
    let config =
        ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
    let model = load_falcon_from_gguf(&gguf, &config).expect("load_falcon_from_gguf must succeed");
    (gguf, model)
}

// ── Hand-built Falcon-40B-shaped fixture (with `attn_norm_2`) ────────────────

const HIDDEN: usize = 32;
const HEADS: usize = 4;
const HEAD_DIM: usize = 8;
const KV_HEADS: usize = 2;
const FFN: usize = 64;
const VOCAB: usize = 32;
const CONTEXT: usize = 128;
/// Fused QKV width: `n_embd + 2 * n_embd_gqa` = 32 + 2 * (2 * 8) = 64.
const QKV_OUT: usize = HIDDEN + 2 * (KV_HEADS * HEAD_DIM);

/// Deterministic small non-zero weights, so the forward pass exercises real
/// arithmetic instead of multiplying zeros.
fn pattern_bytes(n: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = (((i + seed) % 7) as f32 - 3.0) * 0.05;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Build a one-layer Falcon GGUF.
///
/// * `with_attn_norm_2` — emit `blk.0.attn_norm_2.weight/.bias`, the pair
///   llama.cpp declares `TENSOR_NOT_REQUIRED` and only Falcon-40B ships.
/// * `with_output` — emit a standalone `output.weight`.  When `false` the
///   checkpoint is *tied* and the LM head must fall back to
///   `token_embd.weight` (`TENSOR_DUPLICATED` in llama.cpp).
/// * `omit` — tensor names to leave out entirely (for the "required tensor is
///   missing" cases).
fn build_falcon_gguf_omitting(with_attn_norm_2: bool, with_output: bool, omit: &[&str]) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    for (key, value) in [
        (
            "general.architecture",
            MetadataValue::String("falcon".to_string()),
        ),
        (
            "falcon.embedding_length",
            MetadataValue::Uint32(HIDDEN as u32),
        ),
        (
            "falcon.feed_forward_length",
            MetadataValue::Uint32(FFN as u32),
        ),
        ("falcon.block_count", MetadataValue::Uint32(1)),
        (
            "falcon.attention.head_count",
            MetadataValue::Uint32(HEADS as u32),
        ),
        (
            "falcon.attention.head_count_kv",
            MetadataValue::Uint32(KV_HEADS as u32),
        ),
        (
            "falcon.context_length",
            MetadataValue::Uint32(CONTEXT as u32),
        ),
        ("falcon.vocab_size", MetadataValue::Uint32(VOCAB as u32)),
        (
            "falcon.attention.layer_norm_epsilon",
            MetadataValue::Float32(1e-5),
        ),
        ("falcon.rope.freq_base", MetadataValue::Float32(10000.0)),
    ] {
        writer.add_metadata(key, value);
    }

    // GGUF writes `ne` fastest-changing-first, so every weight is declared as
    // [in_features, out_features] — the reverse of the math shape.
    let matrices: &[(&str, usize, usize, usize)] = &[
        ("token_embd.weight", HIDDEN, VOCAB, 1),
        ("blk.0.attn_qkv.weight", HIDDEN, QKV_OUT, 2),
        ("blk.0.attn_output.weight", HIDDEN, HIDDEN, 3),
        ("blk.0.ffn_up.weight", HIDDEN, FFN, 4),
        ("blk.0.ffn_down.weight", FFN, HIDDEN, 5),
    ];
    for (name, in_features, out_features, seed) in matrices {
        if omit.contains(name) {
            continue;
        }
        writer.add_tensor(
            name,
            &[*in_features as u64, *out_features as u64],
            GgufTensorType::F32,
            &pattern_bytes(in_features * out_features, *seed),
        );
    }

    if with_output {
        writer.add_tensor(
            "output.weight",
            &[HIDDEN as u64, VOCAB as u64],
            GgufTensorType::F32,
            &pattern_bytes(HIDDEN * VOCAB, 6),
        );
    }

    let mut vectors: Vec<(&str, usize)> = vec![
        ("output_norm.weight", 7),
        ("output_norm.bias", 8),
        ("blk.0.attn_norm.weight", 9),
        ("blk.0.attn_norm.bias", 10),
    ];
    if with_attn_norm_2 {
        vectors.push(("blk.0.attn_norm_2.weight", 11));
        vectors.push(("blk.0.attn_norm_2.bias", 12));
    }
    for (name, seed) in vectors {
        if omit.contains(&name) {
            continue;
        }
        writer.add_tensor(
            name,
            &[HIDDEN as u64],
            GgufTensorType::F32,
            &pattern_bytes(HIDDEN, seed),
        );
    }

    let mut bytes = Vec::new();
    writer.write_to(&mut bytes).expect("GGUF must serialize");
    bytes
}

/// [`build_falcon_gguf_omitting`] with a complete tensor set.
fn build_falcon_gguf(with_attn_norm_2: bool, with_output: bool) -> Vec<u8> {
    build_falcon_gguf_omitting(with_attn_norm_2, with_output, &[])
}

// ── 1. Round trip on the shared fixture ─────────────────────────────────────

/// Load `build_minimal_falcon_gguf()` and run one forward pass.
///
/// **New capability, not a regression.**  Before this change there was no
/// Falcon GGUF loader at all — `FalconArchitecture::build()` returned
/// `MissingTensor` for every input and no other entry point existed — so there
/// is no pre-fix state in which this test "failed differently"; it could not
/// have been written.
#[test]
fn falcon_gguf_round_trip_produces_finite_logits() {
    let (gguf, mut model) = load(oxillama_gguf::test_utils::build_minimal_falcon_gguf());

    // The shared fixture is Falcon-2 shaped: one norm per layer, GQA, no
    // linear biases.
    assert_eq!(model.layers.len(), 1);
    assert!(
        model.layers[0].attn_norm_2.is_none(),
        "the shared fixture has no attn_norm_2 (Falcon-2 / Falcon-7B shape)"
    );
    assert!(
        !gguf.file.tensors.contains("blk.0.attn_qkv.bias"),
        "Falcon has no linear biases"
    );
    assert_eq!(model.cfg.n_heads, 4);
    assert_eq!(model.cfg.n_kv_heads, 2, "fixture is GQA");
    assert_eq!(model.cfg.head_dim, 8);
    assert_eq!(model.cfg.hidden_size, 32);
    assert_eq!(model.cfg.intermediate_size, 64);

    let mut cache = TestKv::new(model.cfg.n_layers);
    let logits = model
        .forward(&[0u32, 1u32, 2u32], &mut cache)
        .expect("forward");

    assert_eq!(
        logits.len(),
        model.vocab_size(),
        "logits must be vocab_size wide"
    );
    assert_eq!(logits.len(), VOCAB);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must contain no NaN/Inf: {logits:?}"
    );
    assert_eq!(cache.seq_len(), 3, "one KV slot consumed per token");
}

/// The hand-built fixture carries non-zero weights, so this checks the
/// arithmetic actually runs (the shared fixture is zero-filled).
#[test]
fn falcon_gguf_round_trip_with_nonzero_weights_is_finite() {
    let (_gguf, mut model) = load(build_falcon_gguf(false, true));
    let mut cache = TestKv::new(model.cfg.n_layers);

    let logits = model.forward(&[3u32, 4u32], &mut cache).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
    assert!(
        logits.iter().any(|v| v.abs() > 0.0),
        "non-zero weights must produce non-zero logits"
    );
}

/// `output.weight` is `TENSOR_NOT_REQUIRED`; a tied checkpoint must load.
#[test]
fn falcon_tied_lm_head_falls_back_to_token_embd() {
    let (gguf, mut model) = load(build_falcon_gguf(false, false));
    assert!(
        !gguf.file.tensors.contains("output.weight"),
        "fixture must model a tied checkpoint"
    );
    assert_eq!(
        model.output.weight.shape,
        vec![VOCAB, HIDDEN],
        "tied LM head keeps token_embd's dimensions"
    );

    let mut cache = TestKv::new(model.cfg.n_layers);
    let logits = model.forward(&[1u32], &mut cache).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ── 2. F1: unconditional RoPE + parallel attention ──────────────────────────

/// F1 regression — the topology flags must not be derived from the activation
/// string.
///
/// Pre-fix code:
///
/// ```ignore
/// let rope = config.activation != "gelu";
/// let alibi = !rope;
/// let parallel_attn = alibi;
/// ```
///
/// For the fixture's metadata the crate's default activation for `falcon` is
/// `"silu"`, so that formula produced `parallel_attn = false` — every real
/// Falcon checkpoint ran the *sequential* graph.  Forcing
/// `activation = "gelu"` flipped it the other way and produced
/// `rope = false, alibi = true` — ALiBi, which `llm_build_falcon` does not
/// implement at all.  Both assertions below are false under that formula.
#[test]
fn falcon_config_is_unconditionally_rope_and_parallel() {
    let gguf = GgufModel::from_bytes(oxillama_gguf::test_utils::build_minimal_falcon_gguf())
        .expect("fixture must parse");
    let base = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata");

    // (a) default activation ("silu" for falcon): pre-fix parallel_attn = false.
    let cfg = FalconConfig::from_model_config(&base).expect("config");
    assert!(
        cfg.parallel_attn,
        "llm_build_falcon sums ffn_out + attn_out + residual unconditionally"
    );
    assert!(cfg.rope);
    assert!(!cfg.alibi);
    assert!(cfg.n_kv_heads < cfg.n_heads, "fixture is GQA");

    // (b) activation forced to "gelu": pre-fix rope = false, alibi = true.
    let mut gelu = base.clone();
    gelu.activation = "gelu".to_string();
    let cfg = FalconConfig::from_model_config(&gelu).expect("config");
    assert!(
        cfg.rope,
        "ggml_rope_ext is applied to Q and K with no config branch"
    );
    assert!(
        !cfg.alibi,
        "there is no ALiBi code path in llm_build_falcon"
    );
    assert!(cfg.parallel_attn);
}

// ── 3. F2: Falcon-40B's second pre-attention LayerNorm ──────────────────────

/// F2 regression — `blk.{i}.attn_norm_2.weight/.bias` must be loaded.
///
/// Pre-fix, `FalconLayer` had no `attn_norm_2` field, `tensor_names.rs`
/// declared no such pattern, and (there being no loader) nothing read the
/// tensor.  A Falcon-40B checkpoint's second norm was therefore dropped on the
/// floor and the attention branch was fed `attn_norm`'s output — a different
/// model.
#[test]
fn falcon_loads_attn_norm_2_when_present() {
    let (gguf, model) = load(build_falcon_gguf(true, true));

    assert!(
        gguf.file.tensors.contains("blk.0.attn_norm_2.weight"),
        "fixture must ship the Falcon-40B second norm"
    );
    let layer = &model.layers[0];
    let norm_2 = layer
        .attn_norm_2
        .as_ref()
        .expect("attn_norm_2 must be loaded when the tensor is present");
    assert_eq!(norm_2.weight.len(), HIDDEN);
    assert!(
        norm_2.bias.is_some(),
        "attn_norm_2.bias must be loaded alongside its weight"
    );

    // The two norms must be distinct objects, not the same weights read twice.
    assert_ne!(
        norm_2.weight, layer.attn_norm.weight,
        "attn_norm_2 must come from its own tensor"
    );
}

/// The dual-norm layer runs end to end.
#[test]
fn falcon_dual_norm_forward_is_finite() {
    let (_gguf, mut model) = load(build_falcon_gguf(true, true));
    let mut cache = TestKv::new(model.cfg.n_layers);
    let logits = model.forward(&[0u32, 5u32], &mut cache).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// Falcon-7B / Falcon-2 checkpoints have no `attn_norm_2`, and loading one
/// must not require it (`TENSOR_NOT_REQUIRED`).
#[test]
fn falcon_without_attn_norm_2_still_loads() {
    let (gguf, model) = load(build_falcon_gguf(false, true));
    assert!(!gguf.file.tensors.contains("blk.0.attn_norm_2.weight"));
    assert!(model.layers[0].attn_norm_2.is_none());
}

// ── 4. F3: no silent norm substitution ──────────────────────────────────────

/// F3 — the silent `ffn_norm` → `attn_norm` fallback is gone, and for the
/// GGUF-driven path the question is moot.
///
/// The old code ran a *sequential* branch whenever `parallel_attn` was false
/// (which, per F1, was every real Falcon checkpoint) and, finding no
/// `ffn_norm`, silently reused `attn_norm` as the pre-FFN norm.  With F1 fixed
/// the GGUF-derived config is always parallel, so the parallel graph — where
/// `attn_norm` legitimately feeds the FFN, per `build_ffn(attn_norm, ...)` —
/// is the only one a checkpoint can select.  There is nothing left to guess.
///
/// llama.cpp's `LLM_ARCH_FALCON` loader creates no `ffn_norm` tensor at all,
/// so a loaded layer must report `None` and the forward pass must still
/// succeed.
#[test]
fn falcon_gguf_path_never_needs_a_fallback_norm() {
    let (gguf, mut model) = load(oxillama_gguf::test_utils::build_minimal_falcon_gguf());

    assert!(
        !gguf.file.tensors.contains("blk.0.ffn_norm.weight"),
        "Falcon checkpoints ship no ffn_norm"
    );
    assert!(
        model.layers.iter().all(|l| l.ffn_norm.is_none()),
        "no ffn_norm may be invented at load time"
    );
    assert!(
        model.cfg.parallel_attn,
        "the GGUF-derived config can only be parallel, so no pre-FFN norm is needed"
    );

    let mut cache = TestKv::new(model.cfg.n_layers);
    assert!(
        model.forward(&[0u32], &mut cache).is_ok(),
        "the parallel graph runs without any ffn_norm"
    );
}

// ── 5. Out-of-vocabulary token ids ──────────────────────────────────────────

/// An out-of-range token id must be an error, never a panic — this path is
/// reachable from the HTTP server with attacker-supplied ids.
#[test]
fn falcon_out_of_vocab_token_returns_error() {
    let (_gguf, mut model) = load(oxillama_gguf::test_utils::build_minimal_falcon_gguf());
    let mut cache = TestKv::new(model.cfg.n_layers);

    let vocab = model.vocab_size() as u32;
    for token in [vocab, vocab + 1, u32::MAX] {
        match model.forward(&[token], &mut cache) {
            Err(ArchError::ConfigMismatch { param, .. }) => {
                assert_eq!(param, "token_id", "must name the offending parameter");
            }
            other => panic!("token {token} must be rejected, got {other:?}"),
        }
    }
}

/// `embed()` shares the same guard.
#[test]
fn falcon_embed_rejects_out_of_vocab_token() {
    let (_gguf, mut model) = load(oxillama_gguf::test_utils::build_minimal_falcon_gguf());
    let mut cache = TestKv::new(model.cfg.n_layers);
    assert!(model.embed(&[VOCAB as u32], &mut cache).is_err());

    // …and still works for a valid id.
    let embedding = model.embed(&[0u32], &mut cache).expect("embed");
    assert_eq!(embedding.len(), model.hidden_size());
    assert!(embedding.iter().all(|v| v.is_finite()));
}

// ── 6. Context-length overflow ──────────────────────────────────────────────

/// Prefill longer than `max_context_length` must be rejected before any RoPE
/// table lookup or score-buffer write.
#[test]
fn falcon_context_overflow_returns_error() {
    let (_gguf, mut model) = load(oxillama_gguf::test_utils::build_minimal_falcon_gguf());
    let max_ctx = model.max_context_length();
    assert_eq!(max_ctx, CONTEXT);

    let mut cache = TestKv::new(model.cfg.n_layers);
    let overlong = vec![0u32; max_ctx + 1];
    match model.forward(&overlong, &mut cache) {
        Err(ArchError::ConfigMismatch { param, .. }) => {
            assert_eq!(param, "context_length");
        }
        other => panic!("an over-long prompt must be rejected, got {other:?}"),
    }

    // The same check applies when the overflow comes from an already-populated
    // cache rather than from a single long prompt.
    let mut cache = TestKv::new(model.cfg.n_layers);
    model.forward(&[0u32; 8], &mut cache).expect("prefill");
    let rest = vec![0u32; max_ctx];
    assert!(
        model.forward(&rest, &mut cache).is_err(),
        "start_pos + n_tokens must be checked, not just n_tokens"
    );
}

/// Exactly `max_context_length` tokens is legal.
#[test]
fn falcon_full_context_prefill_is_accepted() {
    let (_gguf, mut model) = load(oxillama_gguf::test_utils::build_minimal_falcon_gguf());
    let max_ctx = model.max_context_length();
    let mut cache = TestKv::new(model.cfg.n_layers);
    let full = vec![0u32; max_ctx];
    let logits = model
        .forward(&full, &mut cache)
        .expect("full-context prefill");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ── LayerNorm bias is required, not "optional if you feel like it" ──────────

/// `attn_norm.bias` and `output_norm.bias` are `create_tensor(..., 0)` in
/// llama.cpp — required.
///
/// A checkpoint without one must be rejected, not loaded with the shift term
/// quietly dropped: an all-zero beta is a different model, and that silent
/// substitution is exactly the failure class F3 was about.
#[test]
fn falcon_missing_required_layer_norm_bias_is_reported() {
    for missing in ["output_norm.bias", "blk.0.attn_norm.bias"] {
        let bytes = build_falcon_gguf_omitting(false, true, &[missing]);
        let gguf = GgufModel::from_bytes(bytes).expect("must parse");
        let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata");
        match load_falcon_from_gguf(&gguf, &config).err() {
            Some(ArchError::MissingTensor { name }) => {
                assert_eq!(name, missing, "error must name the missing bias");
            }
            other => panic!("{missing} must be required, got {other:?}"),
        }
    }
}

/// `attn_norm_2`'s bias really is optional (`TENSOR_NOT_REQUIRED` for both the
/// weight and the bias), so a weight-only second norm still loads.
#[test]
fn falcon_attn_norm_2_bias_is_optional() {
    let bytes = build_falcon_gguf_omitting(true, true, &["blk.0.attn_norm_2.bias"]);
    let gguf = GgufModel::from_bytes(bytes).expect("must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata");
    let model = load_falcon_from_gguf(&gguf, &config).expect("weight-only attn_norm_2 must load");
    let norm_2 = model.layers[0]
        .attn_norm_2
        .as_ref()
        .expect("attn_norm_2 must still be loaded");
    assert!(norm_2.bias.is_none());
}

// ── KV-cache layout contract ────────────────────────────────────────────────

/// Pin the KV-cache stride the attention loop indexes with.
///
/// `oxillama-runtime`'s `KvCache` stores one `kv_dim = num_kv_heads *
/// head_dim` block per position and `get_keys` returns `stored_len * kv_dim`
/// floats, so head `kv_head` at position `pos` lives at
/// `pos * (n_kv_heads * head_dim) + kv_head * head_dim`.
///
/// The pre-existing `sdpa` used `kv_stride = head_dim`, which only coincides
/// with that contract when `n_kv_heads == 1` (Falcon-7B's MQA).  For every GQA
/// Falcon — 40B and Falcon-2 alike — it read the wrong keys and values.  This
/// fixture is GQA (`n_kv_heads = 2`), so the stride is observable here.
#[test]
fn falcon_kv_cache_uses_kv_dim_stride() {
    let (_gguf, mut model) = load(oxillama_gguf::test_utils::build_minimal_falcon_gguf());
    let n_kv = model.cfg.n_kv_heads;
    let head_dim = model.cfg.head_dim;
    assert!(n_kv > 1, "fixture must be GQA for this to mean anything");

    let mut cache = TestKv::new(model.cfg.n_layers);
    model
        .forward(&[0u32, 1u32, 2u32], &mut cache)
        .expect("forward");

    assert_eq!(
        cache.keys[0].len(),
        3 * n_kv * head_dim,
        "one kv_dim block per position, matching oxillama-runtime's KvCache"
    );
    assert_eq!(cache.vals[0].len(), 3 * n_kv * head_dim);
}

// ── Registry entry point ────────────────────────────────────────────────────

/// `ModelArchitecture::build_from_gguf` must route to the real loader.
///
/// `build()` only receives a `TensorStore` (descriptors, no payload) and can
/// therefore only report `MissingTensor`; `build_from_gguf()` gets the mapping
/// and must produce a running model.
#[test]
fn falcon_architecture_builds_from_gguf() {
    use oxillama_arch::falcon::FalconArchitecture;
    use oxillama_arch::traits::ModelArchitecture;

    let gguf = GgufModel::from_bytes(oxillama_gguf::test_utils::build_minimal_falcon_gguf())
        .expect("fixture must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata");
    let arch = FalconArchitecture::new();

    // `build()` cannot load — it has no payload.
    assert!(matches!(
        arch.build(&config, &gguf.file.tensors).err(),
        Some(ArchError::MissingTensor { .. })
    ));

    // `build_from_gguf()` can.
    let mut model = arch
        .build_from_gguf(&gguf, &config)
        .expect("build_from_gguf must load a Falcon model");
    assert_eq!(model.vocab_size(), VOCAB);
    assert_eq!(model.hidden_size(), HIDDEN);

    let mut cache = TestKv::new(1);
    let logits = model.forward(&[0u32], &mut cache).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ── Loader error reporting ──────────────────────────────────────────────────

/// A required tensor that is absent must be named, not panicked on.
#[test]
fn falcon_missing_required_tensor_is_reported() {
    // Drop `blk.0.attn_qkv.weight` by rebuilding the fixture without it.
    let mut writer = GgufWriter::new();
    for (key, value) in [
        (
            "general.architecture",
            MetadataValue::String("falcon".to_string()),
        ),
        (
            "falcon.embedding_length",
            MetadataValue::Uint32(HIDDEN as u32),
        ),
        (
            "falcon.feed_forward_length",
            MetadataValue::Uint32(FFN as u32),
        ),
        ("falcon.block_count", MetadataValue::Uint32(1)),
        (
            "falcon.attention.head_count",
            MetadataValue::Uint32(HEADS as u32),
        ),
        (
            "falcon.attention.head_count_kv",
            MetadataValue::Uint32(KV_HEADS as u32),
        ),
        (
            "falcon.context_length",
            MetadataValue::Uint32(CONTEXT as u32),
        ),
        ("falcon.vocab_size", MetadataValue::Uint32(VOCAB as u32)),
    ] {
        writer.add_metadata(key, value);
    }
    writer.add_tensor(
        "token_embd.weight",
        &[HIDDEN as u64, VOCAB as u64],
        GgufTensorType::F32,
        &pattern_bytes(HIDDEN * VOCAB, 1),
    );
    for name in [
        "output_norm.weight",
        "output_norm.bias",
        "blk.0.attn_norm.weight",
        "blk.0.attn_norm.bias",
    ] {
        writer.add_tensor(
            name,
            &[HIDDEN as u64],
            GgufTensorType::F32,
            &pattern_bytes(HIDDEN, 2),
        );
    }
    let mut bytes = Vec::new();
    writer.write_to(&mut bytes).expect("GGUF must serialize");

    let gguf = GgufModel::from_bytes(bytes).expect("must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata");
    match load_falcon_from_gguf(&gguf, &config).err() {
        Some(ArchError::MissingTensor { name }) => {
            assert!(
                name.contains("attn_qkv"),
                "error must name the missing tensor, got {name:?}"
            );
        }
        other => panic!("expected MissingTensor for the absent fused QKV, got {other:?}"),
    }
}

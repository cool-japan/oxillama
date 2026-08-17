//! GGUF round-trip and regression tests for the GPT-NeoX architecture.
//!
//! # New capability
//!
//! `load_gpt_neox_from_gguf` did not exist before this suite: `gpt_neox` had
//! only `GptNeoxModel::new(config, …)` over pre-materialised weights, and
//! `GptNeoxArchitecture::build()` returned a `MissingTensor` sentinel.  No
//! GPT-NeoX checkpoint on disk could reach the forward pass, so
//! [`gptneox_gguf_round_trip`] is a **new-capability** test rather than a
//! regression: there was no prior behaviour to regress from.
//!
//! # Regressions covered
//!
//! | Id | Defect |
//! |----|--------|
//! | N1 | Declared tensor names (`ln1`, `ln2`, separate `attn_q`/`attn_k`/`attn_v`) matched **no** real GGUF |
//! | N2 | `GptNeoxLayer` had no bias fields, so every required GPT-NeoX bias was dropped |
//! | N3 | Only the parallel-residual branch existed; `use_parallel_residual = false` was unsupported |
//! | N4 | The RoPE table was built over `head_dim`, giving the wrong frequency ladder for partial rotary |
//! | N5 | Attention-output row stride clamped with `.min(hidden_size)` instead of rejecting a bad geometry |
//! | —  | OOV token ids and over-long prompts panicked instead of returning `Err` |
//!
//! Every fixture below is built with the production
//! [`oxillama_gguf::GgufWriter`] (the shared `test-utils` builder has no `Bool`
//! metadata variant, which N3 needs), except the round-trip test, which uses
//! the shared `build_minimal_gpt_neox_gguf()` fixture on purpose.

#![cfg(feature = "gptneox")]

use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::{ArchError, ArchResult};
use oxillama_arch::gpt_neox::tensor_names::expand_layer_pattern;
use oxillama_arch::gpt_neox::{load_gpt_neox_from_gguf, GptNeoxArchitecture, GptNeoxModel};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess, ModelArchitecture};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

// ─── Hand-built fixture geometry ─────────────────────────────────────────────

const H: usize = 32; // hidden_size / n_embd
const HEADS: usize = 2; // head_dim is derived as H / HEADS = 16
const FFN: usize = 64;
const VOCAB: usize = 8;
const CTX: usize = 64;
const N_ROT: usize = 4; // rotary_pct 0.25 × head_dim 16
/// `n_embd + 2 * n_embd_gqa` — MHA, so `n_embd_gqa == n_embd`.
const QKV: usize = H + 2 * H;

// ─── Minimal KV cache ────────────────────────────────────────────────────────

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

    fn for_config(config: &ModelConfig) -> Self {
        Self::new(
            config.num_layers.max(1),
            config.num_kv_heads * config.head_dim,
            config.max_context_length,
        )
    }
}

impl KvCacheAccess for TestKv {
    fn seq_len(&self) -> usize {
        self.position
    }

    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let offset = self.position * self.kv_dim;
        let dim = self.kv_dim;
        let dst_k = self
            .keys
            .get_mut(layer)
            .and_then(|k| k.get_mut(offset..offset + dim))
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: format!("layer {layer} out of range"),
            })?;
        dst_k.copy_from_slice(&key[..dim.min(key.len())]);
        let dst_v = self
            .values
            .get_mut(layer)
            .and_then(|v| v.get_mut(offset..offset + dim))
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: format!("layer {layer} out of range"),
            })?;
        dst_v.copy_from_slice(&value[..dim.min(value.len())]);
        Ok(())
    }

    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        let end = (self.position + 1) * self.kv_dim;
        self.keys
            .get(layer)
            .and_then(|k| k.get(..end))
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: format!("layer {layer} out of range"),
            })
    }

    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        let end = (self.position + 1) * self.kv_dim;
        self.values
            .get(layer)
            .and_then(|v| v.get(..end))
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: format!("layer {layer} out of range"),
            })
    }

    fn advance(&mut self) {
        self.position = (self.position + 1).min(self.max_seq.saturating_sub(1));
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

// ─── Fixture construction ────────────────────────────────────────────────────

/// Deterministic pseudo-random f32 values in `base ± amp`.
///
/// Every weight varies along the output dimension.  A matrix of identical rows
/// yields an output that is constant across the hidden dimension, which the
/// final LayerNorm's mean-subtraction then deletes — that would make
/// "changing X changes the output" assertions pass vacuously.
fn pattern(n: usize, seed: u32, base: f32, amp: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = ((state >> 9) as f32 / 8_388_608.0) - 1.0;
            base + amp * unit
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Knobs for the hand-built GPT-NeoX fixture.
#[derive(Clone, Copy, Default)]
struct FixtureOpts {
    /// `Some(v)` writes `gptneox.use_parallel_residual = v`; `None` omits the
    /// key entirely, exercising the documented default of `true`.
    parallel_residual: Option<bool>,
    /// Tensor name to leave out of the file (defect N2's required-bias check).
    omit: Option<&'static str>,
    /// `Some(v)` writes `gptneox.attention.key_length = v` (defect N5).
    key_length: Option<usize>,
    /// Write `blk.0.attn_qkv.weight` with only `n_embd` output rows instead of
    /// the required `n_embd + 2 * n_embd_gqa`.
    shrink_qkv: bool,
    /// Omit `gptneox.rope.dimension_count`, exercising the `n_rot = head_dim`
    /// fallback llama.cpp uses.
    omit_rope_dim: bool,
}

/// Build a complete single-layer GPT-NeoX GGUF with varying, non-zero weights.
///
/// Shapes follow `llama-model.cpp`'s `case LLM_ARCH_GPTNEOX:` exactly, written
/// in GGUF's in-features-first `ne` order.
fn build_fixture(opts: FixtureOpts) -> Vec<u8> {
    let mut writer = GgufWriter::new();

    for (key, value) in [
        (
            "general.architecture",
            MetadataValue::String("gptneox".to_string()),
        ),
        (
            "general.name",
            MetadataValue::String("gptneox-fixture".to_string()),
        ),
        ("gptneox.embedding_length", MetadataValue::Uint32(H as u32)),
        (
            "gptneox.feed_forward_length",
            MetadataValue::Uint32(FFN as u32),
        ),
        ("gptneox.block_count", MetadataValue::Uint32(1)),
        (
            "gptneox.attention.head_count",
            MetadataValue::Uint32(HEADS as u32),
        ),
        (
            "gptneox.attention.head_count_kv",
            MetadataValue::Uint32(HEADS as u32),
        ),
        ("gptneox.context_length", MetadataValue::Uint32(CTX as u32)),
        ("gptneox.vocab_size", MetadataValue::Uint32(VOCAB as u32)),
        (
            "gptneox.attention.layer_norm_epsilon",
            MetadataValue::Float32(1e-5),
        ),
        ("gptneox.rope.freq_base", MetadataValue::Float32(10000.0)),
    ] {
        writer.add_metadata(key, value);
    }

    if !opts.omit_rope_dim {
        writer.add_metadata(
            "gptneox.rope.dimension_count",
            MetadataValue::Uint32(N_ROT as u32),
        );
    }
    if let Some(parallel) = opts.parallel_residual {
        writer.add_metadata(
            "gptneox.use_parallel_residual",
            MetadataValue::Bool(parallel),
        );
    }
    if let Some(key_length) = opts.key_length {
        writer.add_metadata(
            "gptneox.attention.key_length",
            MetadataValue::Uint32(key_length as u32),
        );
    }

    // (name, ne dims, seed, base, amplitude)
    let tensors: &[(&str, &[u64], u32, f32, f32)] = &[
        ("token_embd.weight", &[H as u64, VOCAB as u64], 1, 0.0, 0.5),
        ("output_norm.weight", &[H as u64], 2, 1.0, 0.1),
        ("output_norm.bias", &[H as u64], 3, 0.0, 0.05),
        ("output.weight", &[H as u64, VOCAB as u64], 4, 0.0, 0.5),
        ("blk.0.attn_norm.weight", &[H as u64], 5, 1.0, 0.1),
        ("blk.0.attn_norm.bias", &[H as u64], 6, 0.0, 0.05),
        (
            "blk.0.attn_qkv.weight",
            &[H as u64, QKV as u64],
            7,
            0.0,
            0.2,
        ),
        ("blk.0.attn_qkv.bias", &[QKV as u64], 8, 0.0, 0.3),
        (
            "blk.0.attn_output.weight",
            &[H as u64, H as u64],
            9,
            0.0,
            0.2,
        ),
        ("blk.0.attn_output.bias", &[H as u64], 10, 0.0, 0.3),
        ("blk.0.ffn_norm.weight", &[H as u64], 11, 1.0, 0.1),
        ("blk.0.ffn_norm.bias", &[H as u64], 12, 0.0, 0.05),
        ("blk.0.ffn_up.weight", &[H as u64, FFN as u64], 13, 0.0, 0.2),
        ("blk.0.ffn_up.bias", &[FFN as u64], 14, 0.0, 0.3),
        (
            "blk.0.ffn_down.weight",
            &[FFN as u64, H as u64],
            15,
            0.0,
            0.2,
        ),
        ("blk.0.ffn_down.bias", &[H as u64], 16, 0.0, 0.3),
    ];

    for (name, dims, seed, base, amp) in tensors {
        if opts.omit == Some(name) {
            continue;
        }
        let shrunk: Vec<u64>;
        let dims: &[u64] = if opts.shrink_qkv && *name == "blk.0.attn_qkv.weight" {
            shrunk = vec![H as u64, H as u64];
            &shrunk
        } else {
            dims
        };
        let n: usize = dims.iter().map(|&d| d as usize).product();
        writer.add_tensor(
            name,
            dims,
            GgufTensorType::F32,
            &f32_bytes(&pattern(n, *seed, *base, *amp)),
        );
    }

    let mut buf = Vec::new();
    writer.write_to(&mut buf).expect("fixture must serialize");
    buf
}

fn load(bytes: Vec<u8>) -> ArchResult<(ModelConfig, GptNeoxModel)> {
    let gguf = GgufModel::from_bytes(bytes)?;
    let config = ModelConfig::from_metadata(&gguf.file.metadata)?;
    let model = load_gpt_neox_from_gguf(&gguf, &config)?;
    Ok((config, model))
}

// ─── 1. Round trip (new capability) ──────────────────────────────────────────

/// A GPT-NeoX GGUF loads and produces finite, correctly-sized logits.
///
/// This is a NEW-CAPABILITY test: no GGUF loader existed for this
/// architecture before, so there is no earlier behaviour to compare against.
/// It runs against the shared `build_minimal_gpt_neox_gguf()` fixture, whose
/// tensor payloads are all-zero — enough to prove the tensor names, shapes and
/// wiring line up end to end.
#[test]
fn gptneox_gguf_round_trip() {
    let gguf = GgufModel::from_bytes(oxillama_gguf::test_utils::build_minimal_gpt_neox_gguf())
        .expect("fixture GGUF must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("config");

    assert_eq!(config.architecture, "gptneox");
    assert_eq!(config.hidden_size, 64);
    assert_eq!(config.head_dim, 16);
    assert_eq!(config.num_attention_heads, 4);
    assert_eq!(config.vocab_size, 32);

    let mut model = load_gpt_neox_from_gguf(&gguf, &config).expect("load_gpt_neox_from_gguf");
    assert_eq!(model.layers.len(), 1);
    assert!(
        model.use_parallel_residual,
        "the fixture omits use_parallel_residual, so the documented default (true) must apply"
    );

    let mut kv = TestKv::for_config(&config);
    let logits = model.forward(&[3u32], &mut kv).expect("forward");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must be finite: {logits:?}"
    );
}

/// The same file must also route through the registry's `build_from_gguf`.
#[test]
fn gptneox_builds_through_the_architecture_plugin() {
    let gguf = GgufModel::from_bytes(oxillama_gguf::test_utils::build_minimal_gpt_neox_gguf())
        .expect("fixture GGUF must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("config");

    let mut model = GptNeoxArchitecture::new()
        .build_from_gguf(&gguf, &config)
        .expect("build_from_gguf must load a real checkpoint");

    let mut kv = TestKv::for_config(&config);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), config.vocab_size);
}

// ─── 2. N1: declared tensor names are the ones a real GGUF carries ───────────

/// Defect N1.
///
/// The pre-fix `tensor_names()` declared `blk.{i}.ln1.weight`,
/// `blk.{i}.ln2.weight` and three separate `attn_q`/`attn_k`/`attn_v`
/// projections.  None of those strings exists in llama.cpp, in `gguf-py`'s
/// `TENSOR_NAMES`, or in any converted checkpoint — so the declared list could
/// never match a real file.  This test pins the list to the shared fixture,
/// which is byte-for-byte a valid GGUF.
#[test]
fn gptneox_declared_tensor_names_exist_in_a_real_gguf() {
    let gguf = GgufModel::from_bytes(oxillama_gguf::test_utils::build_minimal_gpt_neox_gguf())
        .expect("fixture GGUF must parse");

    let patterns = GptNeoxArchitecture::new().tensor_names();
    assert!(!patterns.is_empty());

    for p in &patterns {
        let name = expand_layer_pattern(&p.pattern, 0);
        assert!(
            gguf.file.tensors.contains(&name),
            "declared pattern '{}' expands to '{name}', which the fixture GGUF does not contain",
            p.pattern
        );
    }

    let names: Vec<&str> = patterns.iter().map(|p| p.pattern.as_str()).collect();
    assert!(
        names.contains(&"blk.{i}.attn_qkv.weight"),
        "the fused QKV tensor must be declared: {names:?}"
    );
    assert!(names.contains(&"blk.{i}.attn_qkv.bias"));
    assert!(names.contains(&"blk.{i}.attn_norm.weight"));
    assert!(names.contains(&"blk.{i}.ffn_norm.weight"));

    for name in &names {
        for phantom in ["ln1", "ln2", "attn_q.", "attn_k.", "attn_v."] {
            assert!(
                !name.contains(phantom),
                "'{name}' contains '{phantom}', which appears in no real GPT-NeoX GGUF"
            );
        }
    }
}

/// The loader must read the fused QKV by name; splitting it into three
/// projections would make these lookups fail.
#[test]
fn gptneox_loader_reads_fused_qkv_by_name() {
    let gguf = GgufModel::from_bytes(oxillama_gguf::test_utils::build_minimal_gpt_neox_gguf())
        .expect("fixture GGUF must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("config");
    let model = load_gpt_neox_from_gguf(&gguf, &config).expect("load");

    let layer = &model.layers[0];
    // n_embd + 2 * n_embd_gqa = 64 + 128 = 192 output rows over 64 inputs.
    assert_eq!(layer.attn_qkv.out_features, 192);
    assert_eq!(layer.attn_qkv.in_features, 64);
    assert_eq!(layer.attn_qkv_bias.len(), 192);

    for absent in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.ln1.weight",
        "blk.0.ln2.weight",
    ] {
        assert!(
            !gguf.file.tensors.contains(absent),
            "'{absent}' must not exist in a GPT-NeoX GGUF"
        );
    }
}

// ─── 3. N2: every bias is required and actually applied ──────────────────────

/// Defect N2 (absence half).
///
/// `bqkv` is created with `flags = 0` in `llama-model.cpp`, so a checkpoint
/// without it is malformed.  The pre-fix layer had no bias fields at all and
/// would have loaded such a file happily.
#[test]
fn gptneox_missing_qkv_bias_is_reported() {
    let err = load(build_fixture(FixtureOpts {
        omit: Some("blk.0.attn_qkv.bias"),
        ..FixtureOpts::default()
    }))
    .err();

    match err {
        Some(ArchError::MissingTensor { ref name }) => {
            assert_eq!(name, "blk.0.attn_qkv.bias");
        }
        other => panic!("expected MissingTensor for blk.0.attn_qkv.bias, got {other:?}"),
    }
}

/// The other five required per-layer/global biases are equally mandatory.
#[test]
fn gptneox_every_required_bias_is_mandatory() {
    for missing in [
        "blk.0.attn_norm.bias",
        "blk.0.attn_output.bias",
        "blk.0.ffn_norm.bias",
        "blk.0.ffn_up.bias",
        "blk.0.ffn_down.bias",
        "output_norm.bias",
    ] {
        let err = load(build_fixture(FixtureOpts {
            omit: Some(missing),
            ..FixtureOpts::default()
        }))
        .err();
        match err {
            Some(ArchError::MissingTensor { ref name }) => assert_eq!(name, missing),
            other => panic!("omitting '{missing}' must be MissingTensor, got {other:?}"),
        }
    }
}

/// Defect N2 (application half): the loaded bias values must reach the
/// arithmetic, not merely be stored.
#[test]
fn gptneox_biases_are_applied_in_the_forward_pass() {
    let (config, mut with_bias) = load(build_fixture(FixtureOpts::default())).expect("load");
    let (_, mut zeroed) = load(build_fixture(FixtureOpts::default())).expect("load");
    for layer in zeroed.layers.iter_mut() {
        layer.attn_qkv_bias.fill(0.0);
        layer.attn_out_bias.fill(0.0);
        layer.ffn_up_bias.fill(0.0);
        layer.ffn_down_bias.fill(0.0);
    }

    let mut kv1 = TestKv::for_config(&config);
    let mut kv2 = TestKv::for_config(&config);
    let a = with_bias.forward(&[2u32], &mut kv1).expect("with bias");
    let b = zeroed.forward(&[2u32], &mut kv2).expect("zeroed bias");

    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
        "GPT-NeoX projection biases must change the logits: {a:?} vs {b:?}"
    );
}

// ─── 4. N3: both residual topologies work and differ ─────────────────────────

/// Defect N3.
///
/// `llm_build_gptneox` branches its entire per-layer computation on
/// `hparams.use_par_res`; only the parallel formula was implemented, hard-coded.
/// A `use_parallel_residual = false` checkpoint (which llama.cpp loads happily)
/// was silently evaluated with the wrong graph.
#[test]
fn gptneox_sequential_residual_loads_and_differs_from_parallel() {
    let (config, mut parallel) = load(build_fixture(FixtureOpts {
        parallel_residual: Some(true),
        ..FixtureOpts::default()
    }))
    .expect("parallel fixture");
    let (_, mut sequential) = load(build_fixture(FixtureOpts {
        parallel_residual: Some(false),
        ..FixtureOpts::default()
    }))
    .expect("sequential fixture");

    assert!(parallel.use_parallel_residual);
    assert!(!sequential.use_parallel_residual);

    let mut kv1 = TestKv::for_config(&config);
    let mut kv2 = TestKv::for_config(&config);
    let a = parallel
        .forward(&[1u32], &mut kv1)
        .expect("parallel forward");
    let b = sequential
        .forward(&[1u32], &mut kv2)
        .expect("sequential forward");

    assert_eq!(a.len(), VOCAB);
    assert_eq!(b.len(), VOCAB);
    assert!(a.iter().all(|v| v.is_finite()));
    assert!(b.iter().all(|v| v.is_finite()));
    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
        "the two residual topologies must not produce identical logits: {a:?} vs {b:?}"
    );
}

/// An omitted `use_parallel_residual` key defaults to `true`, matching
/// `convert_hf_to_gguf.py`'s `hparams.get("use_parallel_residual", True)`.
#[test]
fn gptneox_absent_parallel_residual_key_defaults_to_true() {
    let (_, absent) = load(build_fixture(FixtureOpts::default())).expect("load");
    assert!(absent.use_parallel_residual);
}

// ─── 5. N4: the RoPE ladder comes from n_rot, not head_dim ───────────────────

/// Defect N4.
///
/// The table used to be built from `head_dim` and then truncated to the
/// leading `n_rot / 2` entries.  ggml computes
/// `theta_scale = powf(freq_base, -2.0f / n_dims)` with `n_dims = n_rot`, so
/// truncating a `head_dim`-wide ladder produces the wrong frequency for every
/// rotated pair whenever `n_rot != head_dim` — i.e. always, for Pythia.
#[test]
fn gptneox_rope_table_is_built_over_rope_dimension_count() {
    let gguf = GgufModel::from_bytes(oxillama_gguf::test_utils::build_minimal_gpt_neox_gguf())
        .expect("fixture GGUF must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("config");
    let model = load_gpt_neox_from_gguf(&gguf, &config).expect("load");

    // Fixture: gptneox.rope.dimension_count = 4, head_dim = 16.
    assert_eq!(model.rotary_dims, 4, "n_rot must come from the metadata");
    assert_eq!(
        model.rope.half_dim, 2,
        "the table must hold n_rot/2 = 2 rotation pairs"
    );
    assert_ne!(
        model.rope.half_dim,
        config.head_dim / 2,
        "a head_dim/2 = 8 pair table is exactly defect N4"
    );

    // theta_1 = base^(-2*1/n_rot) = 10000^(-0.5); a head_dim-derived table
    // would have produced 10000^(-2/16) here instead.
    let want = 1.0f32 / 10000.0f32.powf(2.0 / 4.0);
    let got = model.rope.cos[model.rope.half_dim + 1].acos();
    assert!(
        (got - want).abs() < 1e-5,
        "frequency for pair 1 must be {want}, got {got}"
    );
    let head_dim_derived = 1.0f32 / 10000.0f32.powf(2.0 / 16.0);
    assert!(
        (got - head_dim_derived).abs() > 1e-3,
        "the ladder must NOT match the head_dim-derived one"
    );
}

/// An absent `rope.dimension_count` falls back to **`head_dim`** (full
/// rotary), not to `0.25 × head_dim`.
///
/// `llama-model.cpp:618-620` seeds `hparams.n_rot = hparams.n_embd_head_k;`
/// and only then reads the *optional*
/// `get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, /*required=*/false)`.
/// Hard-coding a 0.25 fallback would silently under-rotate every full-rotary
/// checkpoint that omits the key.
#[test]
fn gptneox_absent_rope_dimension_count_falls_back_to_head_dim() {
    let (config, mut model) = load(build_fixture(FixtureOpts {
        omit_rope_dim: true,
        ..FixtureOpts::default()
    }))
    .expect("load");

    assert_eq!(config.head_dim, H / HEADS, "fixture head_dim");
    assert_eq!(
        model.rotary_dims, config.head_dim,
        "the fallback must be head_dim (llama-model.cpp:618), not 0.25 * head_dim"
    );
    assert_ne!(
        model.rotary_dims,
        config.head_dim / 4,
        "a 0.25 * head_dim fallback would under-rotate a full-rotary checkpoint"
    );
    assert_eq!(model.rope.half_dim, config.head_dim / 2);

    let mut kv = TestKv::for_config(&config);
    let logits = model.forward(&[1u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
}

// ─── 6. N5: an inconsistent attention geometry is an error, not a clamp ──────

/// Defect N5.
///
/// The attention-output projection's row stride used to be
/// `(num_heads * head_dim).min(hidden_size)`.  When the two disagree the
/// clamp silently reads every weight row from the wrong offset; llama.cpp
/// hard-codes `wo` as `{n_embd, n_embd}`, so the only correct response is to
/// reject the checkpoint.
#[test]
fn gptneox_rejects_inconsistent_head_geometry() {
    // key_length = 24 → head_dim 24, so num_heads * head_dim = 48 != n_embd 32.
    let err = load(build_fixture(FixtureOpts {
        key_length: Some(24),
        ..FixtureOpts::default()
    }))
    .err();

    match err {
        Some(ArchError::ConfigMismatch { ref param, .. }) => {
            assert!(
                param.contains("head_count") || param.contains("key_length"),
                "unexpected param '{param}'"
            );
        }
        other => panic!("expected ConfigMismatch for a bad head geometry, got {other:?}"),
    }
}

/// A fused-QKV tensor that is not `n_embd + 2 * n_embd_gqa` rows tall is
/// reported by name rather than read past.
///
/// A loader that split QKV into three separate projections would accept this
/// `[n_embd, n_embd]` tensor as a plain `attn_q` and silently produce garbage.
#[test]
fn gptneox_wrong_qkv_shape_is_reported() {
    let err = load(build_fixture(FixtureOpts {
        shrink_qkv: true,
        ..FixtureOpts::default()
    }))
    .err();

    match err {
        Some(ArchError::InvalidShape {
            ref name,
            ref expected,
            ref got,
        }) => {
            assert_eq!(name, "blk.0.attn_qkv.weight");
            assert_eq!(expected, &vec![QKV, H]);
            assert_eq!(got, &vec![H, H]);
        }
        other => panic!("expected InvalidShape naming the fused QKV, got {other:?}"),
    }
}

// ─── 7. Shared guards: OOV ids and over-long prompts ─────────────────────────

/// An out-of-vocabulary id must be an `Err`, never an out-of-range slice.
#[test]
fn gptneox_out_of_vocabulary_token_errors() {
    let (config, mut model) = load(build_fixture(FixtureOpts::default())).expect("load");
    let mut kv = TestKv::for_config(&config);

    let err = model.forward(&[VOCAB as u32], &mut kv).err();
    assert!(
        matches!(err, Some(ArchError::ConfigMismatch { .. })),
        "expected ConfigMismatch for an OOV id, got {err:?}"
    );

    let err = model.forward(&[u32::MAX], &mut kv).err();
    assert!(
        matches!(err, Some(ArchError::ConfigMismatch { .. })),
        "expected ConfigMismatch for u32::MAX, got {err:?}"
    );
}

/// A prompt longer than the precomputed context must be an `Err` before any
/// RoPE-table or score-buffer index happens.
#[test]
fn gptneox_overlong_prompt_errors() {
    let (config, mut model) = load(build_fixture(FixtureOpts::default())).expect("load");
    let mut kv = TestKv::for_config(&config);

    let tokens = vec![0u32; config.max_context_length + 1];
    let err = model.forward(&tokens, &mut kv).err();
    assert!(
        matches!(err, Some(ArchError::ConfigMismatch { .. })),
        "expected ConfigMismatch for an over-long prompt, got {err:?}"
    );

    // `embed()` shares the same guard.
    let err = model.embed(&tokens, &mut kv).err();
    assert!(
        matches!(err, Some(ArchError::ConfigMismatch { .. })),
        "embed() must be guarded too, got {err:?}"
    );
}

/// Multi-token prefill inside the context bound still works and stays finite.
#[test]
fn gptneox_multi_token_prefill_is_finite() {
    let (config, mut model) = load(build_fixture(FixtureOpts::default())).expect("load");
    let mut kv = TestKv::for_config(&config);

    let logits = model
        .forward(&[0u32, 1, 2, 3, 4], &mut kv)
        .expect("prefill");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");

    let hidden = model.embed(&[5u32], &mut kv).expect("embed");
    assert_eq!(hidden.len(), H);
    assert!(hidden.iter().all(|v| v.is_finite()));
}

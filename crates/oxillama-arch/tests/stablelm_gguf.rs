//! StableLM GGUF loading + forward-pass regressions.
//!
//! Before this file existed StableLM had **no GGUF loader at all**: the
//! architecture was reachable only through `StablelmModel::new(config, …)`
//! with hand-built `Vec<f32>` weights, so no test in the workspace ever loaded
//! a StableLM checkpoint.  `stablelm_round_trip_*` below is therefore a
//! *new-capability* test, not a regression: it is the first end-to-end
//! GGUF → `forward()` path for this architecture.
//!
//! Everything else in this file is a regression for one of the six defects
//! fixed alongside the loader (S1–S6), each verified against
//! `~/work/refs/llama.cpp/src/models/stablelm.cpp` (`llm_build_stablelm`) and
//! `~/work/refs/llama.cpp/src/llama-model.cpp` (`case LLM_ARCH_STABLELM:`).

use std::collections::BTreeMap;

use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::{ArchError, ArchResult};
use oxillama_arch::registry::ArchitectureRegistry;
use oxillama_arch::stablelm::{load_stablelm_from_gguf, StablelmArchitecture, StablelmModel};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess, ModelArchitecture};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

// ─── Fixture geometry ────────────────────────────────────────────────────────
//
// Deliberately identical to `oxillama_gguf::test_utils::build_minimal_stablelm_gguf()`
// so the two are directly comparable: hidden 64, 4 query heads over 16-wide
// heads, 2 KV heads (GQA), FFN 128, vocab 32, one layer, and — critically for
// S3 — `rope.dimension_count = 4`, i.e. a rotary width that is neither
// `head_dim` (16) nor `head_dim / 2` (8).

const HIDDEN: usize = 64;
const VOCAB: usize = 32;
const FFN: usize = 128;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const N_ROT: usize = 4;
const CTX: usize = 128;
const EPS: f32 = 1e-5;

// ─── Deterministic tensor payloads ───────────────────────────────────────────

/// Deterministic pseudo-random weights in `[-0.1, 0.1]`.
///
/// The shared `build_minimal_stablelm_gguf()` fixture is **zero-filled**, which
/// makes every logit zero and every "output A differs from output B" assertion
/// vacuous.  Every fixture built here therefore carries non-trivial,
/// per-index-distinct values.
fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((((s >> 33) as u32 % 2001) as f32) / 1000.0 - 1.0) * 0.1
        })
        .collect()
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Reference LayerNorm, used to compute expected values independently of the
/// implementation under test.
fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight)
        .zip(bias)
        .map(|((v, w), b)| (v - mean) * inv_std * w + b)
        .collect()
}

// ─── Fixture builder ─────────────────────────────────────────────────────────

/// A synthetic single-layer StableLM GGUF plus the f32 values that went into it.
struct Fixture {
    bytes: Vec<u8>,
    tensors: BTreeMap<String, Vec<f32>>,
}

/// Knobs covering every checkpoint shape llama.cpp's `LLM_ARCH_STABLELM`
/// branch admits.
#[derive(Clone)]
struct FixtureSpec {
    /// Emit `blk.0.ffn_norm.weight` / `.bias`.  Absent ⇒ parallel residual.
    ffn_norm: bool,
    /// Zero out `attn_norm.weight` and `.bias`, forcing `inpSA == 0`.
    zero_attn_norm: bool,
    /// Explicit `(q, k, v)` bias payloads (`None` ⇒ the tensors are omitted).
    qkv_bias: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    /// Explicit `blk.0.attn_q_norm.weight` payload, `[n_head * head_dim]`.
    q_norm: Option<Vec<f32>>,
    /// Override `stablelm.attention.key_length` (and size the attention
    /// tensors to match), so a `n_head * head_dim != n_embd` model can be built.
    key_length: Option<usize>,
    /// Override `stablelm.attention.head_count_kv`.
    kv_heads: usize,
}

impl Default for FixtureSpec {
    fn default() -> Self {
        Self {
            ffn_norm: true,
            zero_attn_norm: false,
            qkv_bias: None,
            q_norm: None,
            key_length: None,
            kv_heads: KV_HEADS,
        }
    }
}

impl FixtureSpec {
    fn build(&self) -> Fixture {
        let head_dim = self.key_length.unwrap_or(HEAD_DIM);
        let q_dim = HEADS * head_dim;
        let kv_dim = self.kv_heads * head_dim;

        let mut writer = GgufWriter::new();
        for (key, value) in [
            (
                "general.architecture",
                MetadataValue::String("stablelm".to_string()),
            ),
            (
                "general.name",
                MetadataValue::String("stablelm-fixture".to_string()),
            ),
            (
                "stablelm.embedding_length",
                MetadataValue::Uint32(HIDDEN as u32),
            ),
            (
                "stablelm.feed_forward_length",
                MetadataValue::Uint32(FFN as u32),
            ),
            ("stablelm.block_count", MetadataValue::Uint32(1)),
            (
                "stablelm.attention.head_count",
                MetadataValue::Uint32(HEADS as u32),
            ),
            (
                "stablelm.attention.head_count_kv",
                MetadataValue::Uint32(self.kv_heads as u32),
            ),
            ("stablelm.context_length", MetadataValue::Uint32(CTX as u32)),
            ("stablelm.vocab_size", MetadataValue::Uint32(VOCAB as u32)),
            (
                "stablelm.attention.layer_norm_epsilon",
                MetadataValue::Float32(EPS),
            ),
            (
                "stablelm.rope.dimension_count",
                MetadataValue::Uint32(N_ROT as u32),
            ),
            ("stablelm.rope.freq_base", MetadataValue::Float32(10000.0)),
        ] {
            writer.add_metadata(key, value);
        }
        if let Some(kl) = self.key_length {
            writer.add_metadata(
                "stablelm.attention.key_length",
                MetadataValue::Uint32(kl as u32),
            );
            writer.add_metadata(
                "stablelm.attention.value_length",
                MetadataValue::Uint32(kl as u32),
            );
        }

        let mut tensors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let emit = |writer: &mut GgufWriter,
                    tensors: &mut BTreeMap<String, Vec<f32>>,
                    name: &str,
                    ne: &[u64],
                    values: Vec<f32>| {
            writer.add_tensor(name, ne, GgufTensorType::F32, &to_bytes(&values));
            tensors.insert(name.to_string(), values);
        };

        // GGUF writes `ne` fastest-changing-first, so a weight mapping
        // `in → out` is declared `[in, out]`.
        emit(
            &mut writer,
            &mut tensors,
            "token_embd.weight",
            &[HIDDEN as u64, VOCAB as u64],
            pseudo(VOCAB * HIDDEN, 1),
        );
        emit(
            &mut writer,
            &mut tensors,
            "output_norm.weight",
            &[HIDDEN as u64],
            pseudo(HIDDEN, 2).iter().map(|v| 1.0 + v).collect(),
        );
        emit(
            &mut writer,
            &mut tensors,
            "output_norm.bias",
            &[HIDDEN as u64],
            pseudo(HIDDEN, 3),
        );
        emit(
            &mut writer,
            &mut tensors,
            "output.weight",
            &[HIDDEN as u64, VOCAB as u64],
            pseudo(VOCAB * HIDDEN, 4),
        );

        let (attn_norm_w, attn_norm_b) = if self.zero_attn_norm {
            (vec![0.0f32; HIDDEN], vec![0.0f32; HIDDEN])
        } else {
            (
                pseudo(HIDDEN, 5).iter().map(|v| 1.0 + v).collect(),
                pseudo(HIDDEN, 6),
            )
        };
        emit(
            &mut writer,
            &mut tensors,
            "blk.0.attn_norm.weight",
            &[HIDDEN as u64],
            attn_norm_w,
        );
        emit(
            &mut writer,
            &mut tensors,
            "blk.0.attn_norm.bias",
            &[HIDDEN as u64],
            attn_norm_b,
        );

        emit(
            &mut writer,
            &mut tensors,
            "blk.0.attn_q.weight",
            &[HIDDEN as u64, q_dim as u64],
            pseudo(q_dim * HIDDEN, 7),
        );
        emit(
            &mut writer,
            &mut tensors,
            "blk.0.attn_k.weight",
            &[HIDDEN as u64, kv_dim as u64],
            pseudo(kv_dim * HIDDEN, 8),
        );
        emit(
            &mut writer,
            &mut tensors,
            "blk.0.attn_v.weight",
            &[HIDDEN as u64, kv_dim as u64],
            pseudo(kv_dim * HIDDEN, 9),
        );
        emit(
            &mut writer,
            &mut tensors,
            "blk.0.attn_output.weight",
            &[q_dim as u64, HIDDEN as u64],
            pseudo(HIDDEN * q_dim, 10),
        );

        if let Some((q_bias, k_bias, v_bias)) = self.qkv_bias.as_ref() {
            emit(
                &mut writer,
                &mut tensors,
                "blk.0.attn_q.bias",
                &[q_dim as u64],
                q_bias.clone(),
            );
            emit(
                &mut writer,
                &mut tensors,
                "blk.0.attn_k.bias",
                &[kv_dim as u64],
                k_bias.clone(),
            );
            emit(
                &mut writer,
                &mut tensors,
                "blk.0.attn_v.bias",
                &[kv_dim as u64],
                v_bias.clone(),
            );
        }

        if let Some(q_norm) = self.q_norm.as_ref() {
            // `{n_embd_head_k, n_head}` — per-head, NOT a shared [head_dim].
            emit(
                &mut writer,
                &mut tensors,
                "blk.0.attn_q_norm.weight",
                &[head_dim as u64, HEADS as u64],
                q_norm.clone(),
            );
            emit(
                &mut writer,
                &mut tensors,
                "blk.0.attn_k_norm.weight",
                &[head_dim as u64, self.kv_heads as u64],
                vec![1.0f32; kv_dim],
            );
        }

        if self.ffn_norm {
            emit(
                &mut writer,
                &mut tensors,
                "blk.0.ffn_norm.weight",
                &[HIDDEN as u64],
                pseudo(HIDDEN, 11).iter().map(|v| 1.0 + v).collect(),
            );
            emit(
                &mut writer,
                &mut tensors,
                "blk.0.ffn_norm.bias",
                &[HIDDEN as u64],
                pseudo(HIDDEN, 12),
            );
        }

        emit(
            &mut writer,
            &mut tensors,
            "blk.0.ffn_gate.weight",
            &[HIDDEN as u64, FFN as u64],
            pseudo(FFN * HIDDEN, 13),
        );
        emit(
            &mut writer,
            &mut tensors,
            "blk.0.ffn_up.weight",
            &[HIDDEN as u64, FFN as u64],
            pseudo(FFN * HIDDEN, 14),
        );
        emit(
            &mut writer,
            &mut tensors,
            "blk.0.ffn_down.weight",
            &[FFN as u64, HIDDEN as u64],
            pseudo(HIDDEN * FFN, 15),
        );

        let mut bytes = Vec::new();
        writer
            .write_to(&mut bytes)
            .expect("synthetic StableLM GGUF must serialize");
        Fixture { bytes, tensors }
    }
}

// ─── Loading helpers ─────────────────────────────────────────────────────────

fn load(bytes: Vec<u8>) -> StablelmModel {
    try_load(bytes).expect("load_stablelm_from_gguf must succeed")
}

fn try_load(bytes: Vec<u8>) -> ArchResult<StablelmModel> {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic StableLM GGUF must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata must parse");
    load_stablelm_from_gguf(&gguf, &config)
}

/// Load with a hand-built [`ModelConfig`], bypassing `from_metadata`.
///
/// Needed for geometries `ModelConfig::from_metadata` already rejects upstream,
/// so the loader's own guard can still be exercised in isolation.
fn try_load_with(bytes: Vec<u8>, config: &ModelConfig) -> ArchResult<StablelmModel> {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic StableLM GGUF must parse");
    load_stablelm_from_gguf(&gguf, config)
}

fn manual_config(num_kv_heads: usize, head_dim: usize) -> ModelConfig {
    ModelConfig {
        architecture: "stablelm".to_string(),
        hidden_size: HIDDEN,
        intermediate_size: FFN,
        num_layers: 1,
        num_attention_heads: HEADS,
        num_kv_heads,
        head_dim,
        vocab_size: VOCAB,
        max_context_length: CTX,
        rms_norm_eps: EPS,
        rope_freq_base: 10000.0,
        ..ModelConfig::default()
    }
}

/// Flat single-layer KV cache.
struct TestKv {
    keys: Vec<f32>,
    values: Vec<f32>,
    kv_dim: usize,
    seq_len: usize,
}

impl TestKv {
    fn new(kv_dim: usize) -> Self {
        Self {
            keys: vec![0.0; CTX * kv_dim],
            values: vec![0.0; CTX * kv_dim],
            kv_dim,
            seq_len: 0,
        }
    }
}

impl KvCacheAccess for TestKv {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn store_kv(&mut self, _layer: usize, k: &[f32], v: &[f32]) -> ArchResult<()> {
        let off = self.seq_len * self.kv_dim;
        self.keys[off..off + k.len()].copy_from_slice(k);
        self.values[off..off + v.len()].copy_from_slice(v);
        Ok(())
    }

    fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.keys[..(self.seq_len + 1) * self.kv_dim])
    }

    fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.values[..(self.seq_len + 1) * self.kv_dim])
    }

    fn advance(&mut self) {
        self.seq_len += 1;
    }
}

fn run(model: &mut StablelmModel, tokens: &[u32]) -> Vec<f32> {
    let kv_dim = model.config.num_kv_heads * model.config.head_dim;
    let mut kv = TestKv::new(kv_dim);
    model
        .forward(tokens, &mut kv)
        .expect("forward must succeed")
}

fn embed(model: &mut StablelmModel, tokens: &[u32]) -> Vec<f32> {
    let kv_dim = model.config.num_kv_heads * model.config.head_dim;
    let mut kv = TestKv::new(kv_dim);
    model.embed(tokens, &mut kv).expect("embed must succeed")
}

// ─── 1. Round trip (new capability) ──────────────────────────────────────────

/// **New capability, not a regression.** StableLM had no GGUF loader at all
/// before this change, so the shared fixture had never been loaded.
#[test]
fn stablelm_round_trip_shared_fixture() {
    let bytes = oxillama_gguf::test_utils::build_minimal_stablelm_gguf();
    let mut model = load(bytes);

    assert_eq!(model.config.hidden_size, HIDDEN);
    assert_eq!(model.config.num_attention_heads, HEADS);
    assert_eq!(model.config.num_kv_heads, KV_HEADS);
    assert_eq!(model.config.head_dim, HEAD_DIM);
    assert_eq!(model.layers.len(), 1);
    assert!(
        model.layers[0].ffn_norm.is_some(),
        "the shared fixture ships blk.0.ffn_norm.weight ⇒ sequential residual"
    );
    assert!(model.layers[0].attn_q_norm.is_none());
    assert!(model.layers[0].attn_q.bias.is_none());

    let logits = run(&mut model, &[0]);
    assert_eq!(logits.len(), VOCAB, "logits must be vocab_size wide");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "every logit must be finite"
    );
}

/// The same round trip over a fixture with non-trivial weights, so the logits
/// are actually informative rather than the zero-filled fixture's all-zeros.
#[test]
fn stablelm_round_trip_nontrivial_weights() {
    let mut model = load(FixtureSpec::default().build().bytes);
    let logits = run(&mut model, &[3]);

    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
    assert!(
        logits.iter().any(|v| v.abs() > 1e-9),
        "non-zero weights must produce non-zero logits"
    );
}

// ─── S1: residual topology ───────────────────────────────────────────────────

/// **S1 regression.** The two residual topologies must actually be distinct
/// code paths.
///
/// The old implementation ran *neither* llama.cpp branch: it applied
/// `attn_norm` and `ffn_norm` to the **same** pre-residual hidden state and
/// added both outputs at once (`*h += a + f`), and it declared `ffn_norm`
/// required, so the parallel checkpoint (StableLM 2 12B) could not even be
/// described.
#[test]
fn s1_sequential_and_parallel_branches_differ() {
    let sequential = FixtureSpec {
        ffn_norm: true,
        ..FixtureSpec::default()
    };
    let parallel = FixtureSpec {
        ffn_norm: false,
        ..FixtureSpec::default()
    };

    let mut seq_model = load(sequential.build().bytes);
    let mut par_model = load(parallel.build().bytes);

    assert!(!seq_model.layers[0].is_parallel_residual());
    assert!(
        par_model.layers[0].is_parallel_residual(),
        "a checkpoint without ffn_norm.weight must select the parallel residual"
    );

    let seq = run(&mut seq_model, &[5]);
    let par = run(&mut par_model, &[5]);

    assert!(seq.iter().all(|v| v.is_finite()));
    assert!(par.iter().all(|v| v.is_finite()));
    let max_delta = seq
        .iter()
        .zip(&par)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-5,
        "the parallel branch must be a different computation; max |Δ| = {max_delta}"
    );
}

/// **S1 regression, exact.** In the parallel topology the FFN reads `inpSA` —
/// the *output of `attn_norm`* — never a second norm of the residual.
///
/// With `attn_norm` zeroed, `inpSA == 0`, so Q/K/V are zero, the attention
/// output is zero, and (because the FFN has no biases) the FFN output is zero
/// too.  The residual therefore leaves the block **exactly** as the token's
/// embedding row, and `embed()` returns `output_norm(embedding)`.
///
/// Had the FFN instead been fed a norm of the raw residual — what the old code
/// did — its input would have been non-zero and the equality would fail.  The
/// sequential fixture below is precisely that case, and is asserted to differ.
#[test]
fn s1_parallel_ffn_consumes_attn_norm_output() {
    let spec = FixtureSpec {
        ffn_norm: false,
        zero_attn_norm: true,
        ..FixtureSpec::default()
    };
    let fixture = spec.build();
    let token = 7usize;

    let embd = &fixture.tensors["token_embd.weight"][token * HIDDEN..(token + 1) * HIDDEN];
    let expected = layer_norm(
        embd,
        &fixture.tensors["output_norm.weight"],
        &fixture.tensors["output_norm.bias"],
        EPS,
    );

    let mut model = load(fixture.bytes);
    let got = embed(&mut model, &[token as u32]);

    assert_eq!(got.len(), HIDDEN);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "dim {i}: parallel block with a zeroed attn_norm must pass the residual \
             through untouched — got {g}, expected {e}"
        );
    }

    // Same weights, but with ffn_norm present: the FFN now reads
    // LN(ffn_inp) != 0, so the output must NOT match the pass-through.
    let seq = FixtureSpec {
        ffn_norm: true,
        zero_attn_norm: true,
        ..FixtureSpec::default()
    };
    let mut seq_model = load(seq.build().bytes);
    let seq_out = embed(&mut seq_model, &[token as u32]);
    let max_delta = seq_out
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-5,
        "the sequential branch must feed the FFN a *norm of the residual*, \
         which cannot be the identity; max |Δ| = {max_delta}"
    );
}

// ─── S2: optional biases and per-head QK-norm ────────────────────────────────

/// **S2 regression.** `blk.0.attn_q/k/v.bias` were not loaded at all — the old
/// `StablelmLayer` had no field for them and the (nonexistent) loader could not
/// have read them.  Stable LM 2 1.6B ships all three.
///
/// Note the two-token prompts: with a single position the causal softmax is
/// trivially `1.0`, so Q and K cannot influence the output at all and a
/// Q-bias-only assertion would pass vacuously against a no-op implementation.
#[test]
fn s2_qkv_biases_are_loaded_and_applied() {
    let q_len = HEADS * HEAD_DIM;
    let kv_len = KV_HEADS * HEAD_DIM;
    let nonzero =
        |n: usize, seed: u64| -> Vec<f32> { pseudo(n, seed).iter().map(|v| v + 0.5).collect() };

    let no_bias = FixtureSpec::default();
    let zero_bias = FixtureSpec {
        qkv_bias: Some((vec![0.0; q_len], vec![0.0; kv_len], vec![0.0; kv_len])),
        ..FixtureSpec::default()
    };
    // Only `bq` is non-zero, so the difference can come from nothing but the
    // query bias flowing through the attention scores.
    let q_only = FixtureSpec {
        qkv_bias: Some((nonzero(q_len, 99), vec![0.0; kv_len], vec![0.0; kv_len])),
        ..FixtureSpec::default()
    };
    let all_three = FixtureSpec {
        qkv_bias: Some((
            nonzero(q_len, 99),
            nonzero(kv_len, 100),
            nonzero(kv_len, 101),
        )),
        ..FixtureSpec::default()
    };

    let mut m_none = load(no_bias.build().bytes);
    let mut m_zero = load(zero_bias.build().bytes);
    let mut m_q = load(q_only.build().bytes);
    let mut m_all = load(all_three.build().bytes);

    assert!(m_none.layers[0].attn_q.bias.is_none());
    assert!(
        m_zero.layers[0].attn_q.bias.is_some(),
        "a present attn_q.bias tensor must be loaded"
    );
    assert_eq!(
        m_all.layers[0].attn_q.bias.as_ref().map(Vec::len),
        Some(q_len)
    );
    assert_eq!(
        m_all.layers[0].attn_k.bias.as_ref().map(Vec::len),
        Some(kv_len),
        "bk is `{{n_embd_gqa}}`, narrower than bq under GQA"
    );
    assert_eq!(
        m_all.layers[0].attn_v.bias.as_ref().map(Vec::len),
        Some(kv_len)
    );

    let prompt = [9u32, 21];
    let none = run(&mut m_none, &prompt);
    let zero = run(&mut m_zero, &prompt);
    let q = run(&mut m_q, &prompt);
    let all = run(&mut m_all, &prompt);

    for (a, b) in none.iter().zip(&zero) {
        assert!(
            (a - b).abs() < 1e-5,
            "an all-zero bias must be a no-op ({a} vs {b})"
        );
    }

    let max_delta = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let dq = max_delta(&none, &q);
    assert!(
        dq > 1e-5,
        "a non-zero bq must reach the attention scores; max |Δ| = {dq}"
    );
    let dall = max_delta(&q, &all);
    assert!(
        dall > 1e-5,
        "non-zero bk/bv must change the output too; max |Δ| = {dall}"
    );
}

/// **S2 regression, the decisive one.** StableLM's `attn_q_norm` is
/// `{n_embd_head_k, n_head}` — a *distinct* weight vector per head — unlike
/// Qwen3's single shared `[head_dim]` vector.
///
/// Both fixtures below share **head 0's slice** (all ones) and differ only in
/// heads 1..3.  An implementation that reads head 0's slice and reuses it for
/// every head — the Qwen3 shape assumption — produces byte-identical output
/// for the two, so this assertion is what separates the two readings.
#[test]
fn s2_qk_norm_uses_a_distinct_weight_slice_per_head() {
    let mut same_head0_ones = vec![1.0f32; HEADS * HEAD_DIM];
    same_head0_ones[HEAD_DIM..].fill(0.0); // heads 1..3 zeroed

    let same_head0_all_ones = vec![1.0f32; HEADS * HEAD_DIM]; // heads 1..3 kept

    // Sanity: the two agree on head 0 …
    assert_eq!(
        same_head0_ones[..HEAD_DIM],
        same_head0_all_ones[..HEAD_DIM],
        "the fixtures must be indistinguishable if only head 0 is consulted"
    );
    // … and disagree on every other head, which is the only signal available.
    assert_ne!(
        same_head0_ones[HEAD_DIM..],
        same_head0_all_ones[HEAD_DIM..],
        "heads 1.. must differ, otherwise the test proves nothing"
    );

    let a = FixtureSpec {
        q_norm: Some(same_head0_ones),
        ..FixtureSpec::default()
    };
    let b = FixtureSpec {
        q_norm: Some(same_head0_all_ones),
        ..FixtureSpec::default()
    };

    let mut model_a = load(a.build().bytes);
    let mut model_b = load(b.build().bytes);

    let q_norm = model_a.layers[0]
        .attn_q_norm
        .as_ref()
        .expect("attn_q_norm must be loaded when the tensor is present");
    assert_eq!(
        q_norm.weight.len(),
        HEADS * HEAD_DIM,
        "the QK-norm weight must be per-head ({HEADS} × {HEAD_DIM}), not a shared [head_dim]"
    );
    assert_eq!(q_norm.num_heads, HEADS);
    assert_eq!(
        model_a.layers[0]
            .attn_k_norm
            .as_ref()
            .expect("attn_k_norm loaded")
            .num_heads,
        KV_HEADS,
        "the K norm covers n_head_kv heads, not n_head"
    );

    // Two tokens: with a single position the causal softmax is trivially 1.0
    // and Q/K — hence the QK norm — cannot influence the output at all.
    let out_a = run(&mut model_a, &[11, 4]);
    let out_b = run(&mut model_b, &[11, 4]);
    let delta = out_a
        .iter()
        .zip(&out_b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        delta > 1e-5,
        "heads 1..{HEADS} must use their OWN weight slices; reusing head 0's for all \
         heads would make these two fixtures identical (max |Δ| = {delta})"
    );
}

// ─── S3 / S4: partial RoPE ───────────────────────────────────────────────────

/// **S3 regression.** The RoPE table must be built over `n_rot`, not
/// `head_dim`.
///
/// Two independent checks:
///
/// * the table's rotary width is `rope.dimension_count` (4) — not `head_dim`
///   (16) and not `head_dim / 2` (8);
/// * the *frequency denominator* is `n_rot`.  With `θ_i = base^(-2i/n_rot)`,
///   `θ_1 = 10000^(-1/2) = 0.01`.  The old table used `head_dim`, giving
///   `θ_1 = 10000^(-1/8) ≈ 0.3162` — `cos(0.01) ≈ 0.99995` versus
///   `cos(0.3162) ≈ 0.9504`, far outside any tolerance.
#[test]
fn s3_rope_table_is_built_over_n_rot() {
    let model = load(FixtureSpec::default().build().bytes);

    assert_eq!(model.rotary_dims(), N_ROT, "rotary width must be n_rot");
    assert_eq!(
        2 * model.rope.half_dim,
        N_ROT,
        "the table's rotary dimension must be {N_ROT}, not head_dim ({HEAD_DIM}) \
         nor head_dim/2 ({})",
        HEAD_DIM / 2
    );
    assert_ne!(2 * model.rope.half_dim, HEAD_DIM);

    // Row for position 1: [cos(θ_0), cos(θ_1)] with θ_0 = 1, θ_1 = 0.01.
    let half = model.rope.half_dim;
    let theta0 = model.rope.cos[half];
    let theta1 = model.rope.cos[half + 1];
    assert!(
        (theta0 - 1.0f32.cos()).abs() < 1e-5,
        "θ_0 must be 1.0 (base^0); got cos = {theta0}"
    );
    assert!(
        (theta1 - 0.01f32.cos()).abs() < 1e-5,
        "θ_1 must be base^(-2/n_rot) = 0.01; got cos = {theta1} \
         (a head_dim-wide table would give cos(0.3162) ≈ 0.9504)"
    );
}

/// **S4 regression.** The public `apply_partial_rope` must agree with the
/// decode path, and must pair `(i, i + n_rot/2)`.
///
/// The old code was wrong three ways and the old test could not see any of
/// them, because it was written with `head_dim = 4`:
///
/// * `apply_partial_rope` delegated to `RopeTable::apply` over a slice of
///   length `rotary_dims`, while the table's `half_dim` was `head_dim / 2`.
///   For head_dim 64 / rotary 16 that indexed `x[16]` in a 16-element slice.
/// * it paired `(i, i + head_dim/2)` while the decode path paired
///   `(i, i + rotary_dims/2)` — the two disagreed outright.
/// * the old assertion only checked that dims `>= rotary_dims` were
///   *unchanged*, which the broken code satisfied vacuously (it rotated
///   nothing at all).
///
/// This test uses the fixture's realistic 16/4 split and asserts the pairing
/// directly, in both directions.
#[test]
fn s4_partial_rope_pairs_at_n_rot_over_two() {
    let model = load(FixtureSpec::default().build().bytes);
    let half = N_ROT / 2; // 2

    // A single 1.0 at index `n_rot/2` must bleed into index 0: the NeoX
    // rotation is x[0] = x[0]·cos − x[half]·sin.
    let mut head = vec![0.0f32; HEAD_DIM];
    head[half] = 1.0;
    model.apply_partial_rope(&mut head, 1);
    assert!(
        head[0].abs() > 1e-3,
        "index {half} (= n_rot/2) must be the rotation partner of index 0; got {}",
        head[0]
    );
    assert!(
        (head[half] - 1.0f32.cos()).abs() < 1e-4,
        "index {half} must be rotated by θ_0 = 1.0; got {}",
        head[half]
    );

    // A single 1.0 at index `head_dim/2` must NOT touch index 0 — that was the
    // old pairing, and index 8 is outside the rotary window entirely.
    let mut head = vec![0.0f32; HEAD_DIM];
    head[HEAD_DIM / 2] = 1.0;
    model.apply_partial_rope(&mut head, 1);
    assert!(
        head[0].abs() < 1e-9,
        "index {} (= head_dim/2) must NOT be index 0's partner; got {}",
        HEAD_DIM / 2,
        head[0]
    );
    assert!(
        (head[HEAD_DIM / 2] - 1.0).abs() < 1e-9,
        "dimensions outside the rotary window must pass through unchanged"
    );

    // Everything from n_rot onwards is untouched.
    let mut head: Vec<f32> = (0..HEAD_DIM).map(|i| (i + 1) as f32).collect();
    let original = head.clone();
    model.apply_partial_rope(&mut head, 3);
    for i in N_ROT..HEAD_DIM {
        assert!(
            (head[i] - original[i]).abs() < 1e-9,
            "dim {i} is outside [0, {N_ROT}) and must be unchanged"
        );
    }
    assert!(
        (0..N_ROT).any(|i| (head[i] - original[i]).abs() > 1e-6),
        "the rotary prefix must actually rotate"
    );
}

// ─── S5: head geometry ───────────────────────────────────────────────────────

/// **S5 regression.** `num_heads * head_dim != hidden_size` must be an error.
///
/// The old code used `(num_heads * head_dim).min(hidden_size)` as the *row
/// stride* of the row-major `[out, in]` output projection, so every row after
/// the first read from the wrong offset — silently.
#[test]
fn s5_head_geometry_mismatch_is_an_error() {
    // key_length = 24 ⇒ 4 heads × 24 = 96 ≠ hidden_size 64.  Every projection
    // is sized consistently with 24, so the shape checks pass and the geometry
    // check is what fires.
    let spec = FixtureSpec {
        key_length: Some(24),
        ..FixtureSpec::default()
    };
    let err = try_load(spec.build().bytes);
    match err {
        Err(ArchError::ConfigMismatch { param, .. }) => {
            assert!(
                param.contains("head_dim"),
                "the error must name the head geometry, got '{param}'"
            );
        }
        Err(other) => panic!("expected ConfigMismatch, got {other:?}"),
        Ok(_) => panic!("num_heads * head_dim != hidden_size must not load"),
    }
}

/// A `num_kv_heads` that does not divide `num_attention_heads` used to fall
/// back to `heads_per_kv = 1` via `checked_div().unwrap_or(1)`, mapping every
/// query head onto KV head `h` and reading past the cache.
///
/// `ModelConfig::from_metadata` already rejects this geometry upstream, so the
/// config here is hand-built to reach the loader's own guard.
#[test]
fn s5_non_dividing_kv_head_count_is_an_error() {
    let spec = FixtureSpec {
        kv_heads: 3, // 4 % 3 != 0; the K/V tensors are sized 3 × 16 to match
        ..FixtureSpec::default()
    };
    let config = manual_config(3, HEAD_DIM);
    match try_load_with(spec.build().bytes, &config) {
        Err(ArchError::ConfigMismatch { param, .. }) => {
            assert!(
                param.contains("kv_heads"),
                "the error must name num_kv_heads, got '{param}'"
            );
        }
        Err(other) => panic!("expected ConfigMismatch, got {other:?}"),
        Ok(_) => panic!("a non-dividing num_kv_heads must be rejected"),
    }
}

// ─── S6: KV cache borrowing ──────────────────────────────────────────────────

/// **S6 regression (behavioural).** The per-layer, per-token
/// `kv_cache.get_keys(..)?.to_vec()` / `get_values(..)?.to_vec()` full-cache
/// deep copies are gone — `attention()` now borrows both slices, mirroring
/// `qwen3::Qwen3Model::attention`.
///
/// Removing a copy is not directly observable, so this covers the property the
/// copy was protecting: a multi-token call that reads several cached positions
/// per layer still produces correct, finite output, and the last token's logits
/// match what an incremental decode of the same prefix produces.
#[test]
fn s6_multi_token_forward_matches_incremental_decode() {
    let mut batched = load(FixtureSpec::default().build().bytes);
    let mut incremental = load(FixtureSpec::default().build().bytes);
    let kv_dim = KV_HEADS * HEAD_DIM;

    let tokens = [4u32, 17, 2, 29];

    let mut kv_all = TestKv::new(kv_dim);
    let all = batched
        .forward(&tokens, &mut kv_all)
        .expect("multi-token forward must succeed");
    assert_eq!(kv_all.seq_len(), tokens.len(), "every token must advance");
    assert!(all.iter().all(|v| v.is_finite()));

    let mut kv_step = TestKv::new(kv_dim);
    let mut last = Vec::new();
    for t in tokens {
        last = incremental
            .forward(&[t], &mut kv_step)
            .expect("single-token forward must succeed");
    }

    for (i, (a, b)) in all.iter().zip(&last).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "logit {i}: batched {a} vs incremental {b}"
        );
    }
}

// ─── Part 3: input validation ────────────────────────────────────────────────

/// An out-of-vocabulary token id used to index `token_embd` unchecked and
/// panic.  It must be an error.
#[test]
fn oov_token_id_is_an_error_not_a_panic() {
    let mut model = load(FixtureSpec::default().build().bytes);
    let mut kv = TestKv::new(KV_HEADS * HEAD_DIM);

    match model.forward(&[VOCAB as u32], &mut kv) {
        Err(ArchError::ConfigMismatch { param, .. }) => {
            assert!(param.contains("token"), "error must name the token id");
        }
        Err(other) => panic!("expected ConfigMismatch, got {other:?}"),
        Ok(_) => panic!("an out-of-vocabulary token id must not be accepted"),
    }
}

/// Prefill past `max_context_length` used to index `buf_attn_scores`
/// (sized `max_context_length`) with the raw position and abort the process —
/// reachable from the HTTP server with an attacker-controlled prompt length.
#[test]
fn context_overflow_is_an_error_not_a_panic() {
    let mut model = load(FixtureSpec::default().build().bytes);

    // A single call longer than the context.
    let mut kv = TestKv::new(KV_HEADS * HEAD_DIM);
    let long: Vec<u32> = (0..(CTX + 1)).map(|i| (i % VOCAB) as u32).collect();
    assert!(
        matches!(
            model.forward(&long, &mut kv),
            Err(ArchError::ConfigMismatch { .. })
        ),
        "a prompt longer than max_context_length must be rejected"
    );

    // A continuation that would step past the end.
    let mut kv = TestKv::new(KV_HEADS * HEAD_DIM);
    kv.seq_len = CTX;
    assert!(
        matches!(
            model.forward(&[1], &mut kv),
            Err(ArchError::ConfigMismatch { .. })
        ),
        "decoding at position max_context_length must be rejected"
    );

    // And embed() is guarded identically.
    let mut kv = TestKv::new(KV_HEADS * HEAD_DIM);
    kv.seq_len = CTX;
    assert!(model.embed(&[1], &mut kv).is_err());
}

/// A truncated / mis-shaped embedding table must be caught at load time.
#[test]
fn short_token_embedding_is_rejected_at_load() {
    let mut spec_bytes = FixtureSpec::default().build().bytes;
    // Re-declare a larger vocabulary than the table holds by rebuilding with a
    // vocab_size the tensors do not satisfy.
    let gguf = GgufModel::from_bytes(std::mem::take(&mut spec_bytes)).expect("parses");
    let mut config = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata");
    config.vocab_size = VOCAB * 4;
    match load_stablelm_from_gguf(&gguf, &config) {
        Err(ArchError::InvalidShape { name, .. }) => {
            assert_eq!(name, "token_embd.weight");
        }
        Err(other) => panic!("expected InvalidShape, got {other:?}"),
        Ok(_) => panic!("an embedding table too short for vocab_size must be rejected"),
    }
}

// ─── Registry / tensor-name plumbing ─────────────────────────────────────────

#[test]
fn stablelm_registry_lookup() {
    let registry = ArchitectureRegistry::with_builtins();
    let arch = registry
        .get("stablelm")
        .expect("stablelm must be registered");
    assert_eq!(arch.arch_id(), "stablelm");
}

/// `tensor_names()` must mirror llama.cpp's `create_tensor` flags: `ffn_norm`,
/// the q/k/v biases and the QK norms are `TENSOR_NOT_REQUIRED`.
#[test]
fn stablelm_tensor_names_match_reference_requirements() {
    let arch = StablelmArchitecture::new();
    let names = arch.tensor_names();
    assert!(!names.is_empty());

    let by_pattern: BTreeMap<&str, bool> = names
        .iter()
        .map(|p| (p.pattern.as_str(), p.required))
        .collect();

    for req in [
        "token_embd.weight",
        "output_norm.weight",
        "output_norm.bias",
        "output.weight",
        "blk.{i}.attn_norm.weight",
        "blk.{i}.attn_norm.bias",
        "blk.{i}.attn_q.weight",
        "blk.{i}.attn_k.weight",
        "blk.{i}.attn_v.weight",
        "blk.{i}.attn_output.weight",
        "blk.{i}.ffn_gate.weight",
        "blk.{i}.ffn_up.weight",
        "blk.{i}.ffn_down.weight",
    ] {
        assert_eq!(
            by_pattern.get(req),
            Some(&true),
            "'{req}' is flags = 0 in llama-model.cpp ⇒ required"
        );
    }

    for opt in [
        "blk.{i}.ffn_norm.weight",
        "blk.{i}.ffn_norm.bias",
        "blk.{i}.attn_q.bias",
        "blk.{i}.attn_k.bias",
        "blk.{i}.attn_v.bias",
        "blk.{i}.attn_q_norm.weight",
        "blk.{i}.attn_k_norm.weight",
    ] {
        assert_eq!(
            by_pattern.get(opt),
            Some(&false),
            "'{opt}' is TENSOR_NOT_REQUIRED in llama-model.cpp ⇒ optional"
        );
    }

    for p in &names {
        assert!(
            !p.description.is_empty(),
            "{} needs a description",
            p.pattern
        );
    }
}

/// The registry's payload-aware entry point routes to the new loader.
#[test]
fn stablelm_build_from_gguf_routes_to_the_loader() {
    let bytes = oxillama_gguf::test_utils::build_minimal_stablelm_gguf();
    let gguf = GgufModel::from_bytes(bytes).expect("parses");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata");

    let arch = StablelmArchitecture::new();
    let mut model = arch
        .build_from_gguf(&gguf, &config)
        .expect("build_from_gguf must load a StableLM checkpoint");

    assert_eq!(model.vocab_size(), VOCAB);
    assert_eq!(model.hidden_size(), HIDDEN);
    assert_eq!(model.max_context_length(), CTX);

    let mut kv = TestKv::new(KV_HEADS * HEAD_DIM);
    let logits = model.forward(&[1], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
}

//! Granite-3.x is a real, loadable architecture now — and its four multipliers
//! are actually applied.
//!
//! # The "before" state
//!
//! There was nothing to regress against.  `granite/mod.rs` was a 25-line plugin
//! whose `build()` returned
//!
//! ```text
//! MissingTensor { name: "token_embd.weight (use GraniteModel::from_gguf for full loading)" }
//! ```
//!
//! naming a `GraniteModel` type that **did not exist anywhere in the tree**
//! (`rg GraniteModel crates/` matched exactly that one string).  There was no
//! loader, no `ForwardPass`, no `build_from_gguf` override — so
//! `general.architecture = "granite"` could not be loaded at all — and not one
//! line of the crate read `granite.embedding_scale`,
//! `granite.residual_scale`, `granite.attention.scale` or
//! `granite.logit_scale`.  Every test in this file is new capability, not a
//! repaired one; none of them could have been compiled against the old code.
//!
//! # What the multiplier tests prove
//!
//! Parsing the four keys and dropping them is invisible: the model still runs
//! and still returns finite logits.  So each multiplier is checked by building
//! two checkpoints that differ **only** in that one metadata key and comparing
//! the logits:
//!
//! * `logit_scale` has a closed form — llama.cpp's
//!   `ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale)` is a **division**, so
//!   `logit_scale = 2.0` must *halve* every logit.  That is checked
//!   elementwise, and it is the one test that pins the *direction*.
//! * `embedding_scale`, `residual_scale` and `attention.scale` survive no
//!   closed form (RMSNorm is scale-invariant, softmax is not linear), so those
//!   assert "the output changed" — plus, for `residual_scale`, two zeroed-branch
//!   fixtures that isolate each of the two residual adds separately.

use oxillama_arch::granite::{load_granite_from_gguf, GraniteModel, GraniteScales};
use oxillama_arch::{
    ArchError, ArchResult, ArchitectureRegistry, ForwardPass, KvCacheAccess, ModelConfig,
};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

const HIDDEN: usize = 32;
const INTER: usize = 64;
const VOCAB: usize = 48;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const ATTN_DIM: usize = HEADS * HEAD_DIM;
const KV_DIM: usize = KV_HEADS * HEAD_DIM;
const LAYERS: usize = 2;
const CTX: usize = 32;

/// Enough tokens that the attention softmax spans more than one position and
/// RoPE sees a non-zero position — both are identities at `seq_len == 1`.
const PROMPT: &[u32] = &[3, 11, 5, 29];

// ─── Deterministic weights ───────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[-1, 1)`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 8_388_608.0 - 1.0
    }
}

/// `n` little-endian f32 weights in `[-0.1, 0.1)`.
fn weight_bytes(n: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        out.extend_from_slice(&(0.1 * rng.next_f32()).to_le_bytes());
    }
    out
}

/// `n` little-endian f32 norm weights near 1.0.
fn norm_bytes(n: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        out.extend_from_slice(&(1.0 + 0.05 * rng.next_f32()).to_le_bytes());
    }
    out
}

// ─── Fixture ─────────────────────────────────────────────────────────────────

/// Everything a caller may vary about the synthetic Granite checkpoint.
#[derive(Default, Clone)]
struct Fixture {
    /// Extra `granite.*` metadata (the multipliers under test).
    scales: Vec<(&'static str, MetadataValue)>,
    /// Tensor-name **suffix** whose payload is forced to all zeros in every
    /// layer, isolating one residual branch.  Matching by suffix (not by exact
    /// name) matters: with `LAYERS > 1`, zeroing only `blk.0.*` leaves the same
    /// branch live in every other layer and the "isolation" is a fiction.
    zeroed: Option<&'static str>,
    /// Declared `granite.vocab_size`; defaults to the real row count.
    declared_vocab: Option<u32>,
    /// Omit `output.weight` so the LM head must fall back to `token_embd`.
    tie_lm_head: bool,
}

impl Fixture {
    fn with_scale(mut self, key: &'static str, value: f32) -> Self {
        self.scales.push((key, MetadataValue::Float32(value)));
        self
    }

    fn with_bool(mut self, key: &'static str, value: bool) -> Self {
        self.scales.push((key, MetadataValue::Bool(value)));
        self
    }

    /// Zero every tensor whose name ends with `suffix`, in all layers.
    fn zeroing(mut self, suffix: &'static str) -> Self {
        self.zeroed = Some(suffix);
        self
    }

    /// Serialize a one-or-more-layer Granite GGUF.
    ///
    /// Every weight is F32 so the arithmetic in the multiplier tests is exact
    /// rather than quantization-limited; the loader path exercised is the same
    /// one a Q4_K checkpoint takes (`crate::common::loader`).
    fn build(&self) -> Vec<u8> {
        let mut rng = Rng::new(0x0067_7261_6E69_7465);
        let mut writer = GgufWriter::new();

        let vocab = self.declared_vocab.unwrap_or(VOCAB as u32);
        for (key, value) in [
            (
                "general.architecture".to_string(),
                MetadataValue::String("granite".to_string()),
            ),
            (
                "granite.embedding_length".to_string(),
                MetadataValue::Uint32(HIDDEN as u32),
            ),
            (
                "granite.feed_forward_length".to_string(),
                MetadataValue::Uint32(INTER as u32),
            ),
            (
                "granite.block_count".to_string(),
                MetadataValue::Uint32(LAYERS as u32),
            ),
            (
                "granite.attention.head_count".to_string(),
                MetadataValue::Uint32(HEADS as u32),
            ),
            (
                "granite.attention.head_count_kv".to_string(),
                MetadataValue::Uint32(KV_HEADS as u32),
            ),
            (
                "granite.attention.key_length".to_string(),
                MetadataValue::Uint32(HEAD_DIM as u32),
            ),
            (
                "granite.attention.value_length".to_string(),
                MetadataValue::Uint32(HEAD_DIM as u32),
            ),
            (
                "granite.attention.layer_norm_rms_epsilon".to_string(),
                MetadataValue::Float32(1e-5),
            ),
            (
                "granite.context_length".to_string(),
                MetadataValue::Uint32(CTX as u32),
            ),
            (
                "granite.vocab_size".to_string(),
                MetadataValue::Uint32(vocab),
            ),
            (
                "granite.rope.freq_base".to_string(),
                MetadataValue::Float32(10000.0),
            ),
        ] {
            writer.add_metadata(&key, value);
        }
        for (key, value) in &self.scales {
            writer.add_metadata(key, value.clone());
        }

        // GGUF writes `ne` fastest-changing-first: [in_features, out_features].
        let add = |w: &mut GgufWriter, name: &str, out_f: usize, in_f: usize, rng: &mut Rng| {
            let data = if self.zeroed.is_some_and(|z| name.ends_with(z)) {
                vec![0u8; out_f * in_f * 4]
            } else {
                weight_bytes(out_f * in_f, rng)
            };
            w.add_tensor(
                name,
                &[in_f as u64, out_f as u64],
                GgufTensorType::F32,
                &data,
            );
        };

        add(&mut writer, "token_embd.weight", VOCAB, HIDDEN, &mut rng);

        for i in 0..LAYERS {
            let p = format!("blk.{i}");
            add(
                &mut writer,
                &format!("{p}.attn_q.weight"),
                ATTN_DIM,
                HIDDEN,
                &mut rng,
            );
            add(
                &mut writer,
                &format!("{p}.attn_k.weight"),
                KV_DIM,
                HIDDEN,
                &mut rng,
            );
            add(
                &mut writer,
                &format!("{p}.attn_v.weight"),
                KV_DIM,
                HIDDEN,
                &mut rng,
            );
            add(
                &mut writer,
                &format!("{p}.attn_output.weight"),
                HIDDEN,
                ATTN_DIM,
                &mut rng,
            );
            add(
                &mut writer,
                &format!("{p}.ffn_gate.weight"),
                INTER,
                HIDDEN,
                &mut rng,
            );
            add(
                &mut writer,
                &format!("{p}.ffn_up.weight"),
                INTER,
                HIDDEN,
                &mut rng,
            );
            add(
                &mut writer,
                &format!("{p}.ffn_down.weight"),
                HIDDEN,
                INTER,
                &mut rng,
            );

            for name in [
                format!("{p}.attn_norm.weight"),
                format!("{p}.ffn_norm.weight"),
            ] {
                let bytes = norm_bytes(HIDDEN, &mut rng);
                writer.add_tensor(&name, &[HIDDEN as u64], GgufTensorType::F32, &bytes);
            }
        }

        let bytes = norm_bytes(HIDDEN, &mut rng);
        writer.add_tensor(
            "output_norm.weight",
            &[HIDDEN as u64],
            GgufTensorType::F32,
            &bytes,
        );
        if !self.tie_lm_head {
            add(&mut writer, "output.weight", VOCAB, HIDDEN, &mut rng);
        }

        let mut out = Vec::new();
        writer
            .write_to(&mut out)
            .expect("test: synthetic Granite GGUF must serialize");
        out
    }

    fn load(&self) -> ArchResult<GraniteModel> {
        let gguf = GgufModel::from_bytes(self.build()).expect("test: fixture must parse");
        let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("test: fixture config");
        load_granite_from_gguf(&gguf, &config)
    }

    /// Load and run [`PROMPT`] through the model, returning the logits.
    fn logits(&self) -> Vec<f32> {
        let mut model = self.load().expect("test: fixture must load");
        let mut kv = TestKv::new();
        model
            .forward(PROMPT, &mut kv)
            .expect("test: forward must succeed")
    }
}

// ─── KV cache double ─────────────────────────────────────────────────────────

/// Flat per-layer KV cache that records how often `advance()` was called.
struct TestKv {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    seq_len: usize,
    stored_len: usize,
    advances: usize,
}

impl TestKv {
    fn new() -> Self {
        Self {
            keys: vec![vec![0.0; (CTX + 2) * KV_DIM]; LAYERS],
            values: vec![vec![0.0; (CTX + 2) * KV_DIM]; LAYERS],
            seq_len: 0,
            stored_len: 0,
            advances: 0,
        }
    }
}

impl KvCacheAccess for TestKv {
    fn seq_len(&self) -> usize {
        self.seq_len
    }
    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let off = self.seq_len * KV_DIM;
        self.keys[layer][off..off + KV_DIM].copy_from_slice(&key[..KV_DIM]);
        self.values[layer][off..off + KV_DIM].copy_from_slice(&value[..KV_DIM]);
        if self.stored_len <= self.seq_len {
            self.stored_len = self.seq_len + 1;
        }
        Ok(())
    }
    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.keys[layer][..self.stored_len * KV_DIM])
    }
    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.values[layer][..self.stored_len * KV_DIM])
    }
    fn advance(&mut self) {
        self.advances += 1;
        self.seq_len += 1;
        if self.stored_len < self.seq_len {
            self.stored_len = self.seq_len;
        }
    }
    fn kv_dim(&self) -> usize {
        KV_DIM
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ─── Round trip ──────────────────────────────────────────────────────────────

/// A `general.architecture = "granite"` GGUF loads and runs.
///
/// New capability: before this change there was no loader at all and
/// `GraniteArchitecture::build()` was the only entry point, returning
/// `MissingTensor`.
#[test]
fn granite_gguf_round_trips_and_forwards() {
    let model = Fixture::default()
        .load()
        .expect("granite fixture must load");
    assert_eq!(model.layers.len(), LAYERS);
    assert_eq!(model.config.hidden_size, HIDDEN);
    assert_eq!(model.config.head_dim, HEAD_DIM);
    assert_eq!(model.config.vocab_size, VOCAB);
    assert_eq!(model.token_embd.len(), VOCAB * HIDDEN);
    // NORM pairing — `LLM_ARCH_GRANITE` is `LLAMA_ROPE_TYPE_NORM`.
    assert_eq!(
        model.rope.style(),
        oxillama_arch::common::rope::RopeStyle::Norm
    );
    assert_eq!(model.rope.half_dim, HEAD_DIM / 2);

    let logits = Fixture::default().logits();
    assert_eq!(logits.len(), VOCAB);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "every logit must be finite"
    );
    assert!(
        logits.iter().any(|v| *v != 0.0),
        "a non-degenerate checkpoint must not produce all-zero logits"
    );
}

/// A tied checkpoint (no `output.weight`) reuses `token_embd.weight`, as
/// llama.cpp does via `TENSOR_NOT_REQUIRED` + `TENSOR_DUPLICATED`.
#[test]
fn tied_lm_head_falls_back_to_token_embd() {
    let fixture = Fixture {
        tie_lm_head: true,
        ..Fixture::default()
    };
    let logits = fixture.logits();
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// `advance()` exactly once per token, after every layer has written its K/V.
///
/// The documented contract (`oxillama-runtime/src/kv_cache/mod.rs`); advancing
/// per layer would make each layer see a different sequence length.
#[test]
fn kv_cache_advances_exactly_once_per_token() {
    let mut model = Fixture::default().load().expect("test: load");
    let mut kv = TestKv::new();
    model.forward(PROMPT, &mut kv).expect("test: forward");
    assert_eq!(
        kv.advances,
        PROMPT.len(),
        "one advance per token, not per layer ({LAYERS} layers × {} tokens)",
        PROMPT.len()
    );
    assert_eq!(kv.seq_len(), PROMPT.len());
}

/// `embed()` stops before the LM head and returns a `hidden_size` vector.
#[test]
fn embed_returns_the_hidden_state() {
    let mut model = Fixture::default().load().expect("test: load");
    let mut kv = TestKv::new();
    let embedding = model.embed(PROMPT, &mut kv).expect("test: embed");
    assert_eq!(embedding.len(), HIDDEN);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

// ─── The four multipliers are applied ────────────────────────────────────────

/// `granite.logit_scale` **divides** — and does so exactly.
///
/// `src/models/granite.cpp` ends with
/// `cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);`, the converter
/// writes HF's `logits_scaling` uninverted, and HF Granite does
/// `logits = logits / config.logits_scaling`.  So `logit_scale = 2.0` must
/// **halve** the logits — the opposite of doubling them.
#[test]
fn logit_scale_divides_the_logits() {
    let baseline = Fixture::default().logits();
    let scaled = Fixture::default()
        .with_scale("granite.logit_scale", 2.0)
        .logits();

    assert_eq!(baseline.len(), scaled.len());
    for (i, (b, s)) in baseline.iter().zip(scaled.iter()).enumerate() {
        let want = b * 0.5;
        assert!(
            (s - want).abs() <= 1e-6 * want.abs().max(1.0),
            "logit[{i}]: logit_scale=2.0 must halve {b} to {want}, got {s}"
        );
    }
    assert!(
        max_abs_diff(&baseline, &scaled) > 0.0,
        "the fixture must produce non-zero logits for this test to mean anything"
    );
}

/// A `logit_scale` below 1 magnifies, confirming the direction is not an
/// artefact of the value 2.0.
#[test]
fn logit_scale_below_one_magnifies() {
    let baseline = Fixture::default().logits();
    let scaled = Fixture::default()
        .with_scale("granite.logit_scale", 0.5)
        .logits();
    for (i, (b, s)) in baseline.iter().zip(scaled.iter()).enumerate() {
        let want = b * 2.0;
        assert!(
            (s - want).abs() <= 1e-6 * want.abs().max(1.0),
            "logit[{i}]: logit_scale=0.5 must double {b} to {want}, got {s}"
        );
    }
}

/// `granite.embedding_scale` reaches the output.
///
/// No closed form survives: RMSNorm is scale-invariant, so scaling the
/// embedding leaves the first `attn_norm` output (and hence Q/K/V) unchanged —
/// but the residual stream still carries the scaled embedding, so the logits
/// must move.
#[test]
fn embedding_scale_changes_the_output() {
    let baseline = Fixture::default().logits();
    let scaled = Fixture::default()
        .with_scale("granite.embedding_scale", 12.0)
        .logits();
    assert!(
        max_abs_diff(&baseline, &scaled) > 1e-5,
        "granite.embedding_scale was parsed and dropped: logits unchanged"
    );
}

/// `granite.attention.scale` reaches the output.
///
/// It replaces `1/sqrt(head_dim)` inside the softmax, so it only has an effect
/// once a query attends to more than one key — hence the multi-token
/// [`PROMPT`].
#[test]
fn attention_scale_changes_the_output() {
    let baseline = Fixture::default().logits();
    let scaled = Fixture::default()
        .with_scale("granite.attention.scale", 0.0078125)
        .logits();
    assert!(
        max_abs_diff(&baseline, &scaled) > 1e-5,
        "granite.attention.scale was parsed and dropped: logits unchanged"
    );
}

/// The key really is `granite.attention.scale`, not `granite.attention_scale`.
///
/// `llama-arch.cpp`: `{ LLM_KV_ATTENTION_SCALE, "%s.attention.scale" }`;
/// `constants.py`: `SCALE = "{arch}.attention.scale"`.
#[test]
fn misspelled_attention_scale_key_is_inert() {
    let baseline = Fixture::default().logits();
    let misspelled = Fixture::default()
        .with_scale("granite.attention_scale", 0.0078125)
        .logits();
    assert_eq!(
        max_abs_diff(&baseline, &misspelled),
        0.0,
        "`granite.attention_scale` is not a GGUF key and must have no effect"
    );
}

/// `granite.residual_scale` is applied at the **post-attention** residual add.
///
/// `ffn_down.weight` is zeroed **in every layer**, so the FFN branch
/// contributes nothing anywhere; any change must come from
/// `ffn_inp = residual_scale · attn + inpSA`.
#[test]
fn residual_scale_applies_to_the_attention_residual_add() {
    let base = Fixture::default().zeroing("ffn_down.weight");
    let unscaled = base.clone().logits();
    let scaled = base.with_scale("granite.residual_scale", 0.22).logits();
    assert!(
        max_abs_diff(&unscaled, &scaled) > 1e-5,
        "granite.residual_scale is not applied at the post-attention residual add"
    );
}

/// `granite.residual_scale` is applied at the **post-FFN** residual add too.
///
/// `attn_output.weight` is zeroed **in every layer**, so the attention branch
/// contributes nothing anywhere; any change must come from the second
/// `ggml_scale` in `build_layer_ffn`.
#[test]
fn residual_scale_applies_to_the_ffn_residual_add() {
    let base = Fixture::default().zeroing("attn_output.weight");
    let unscaled = base.clone().logits();
    let scaled = base.with_scale("granite.residual_scale", 0.22).logits();
    assert!(
        max_abs_diff(&unscaled, &scaled) > 1e-5,
        "granite.residual_scale is not applied at the post-FFN residual add"
    );
}

/// All four together, at IBM's published Granite-3.0-2B values.
#[test]
fn the_full_granite_scale_set_is_honoured() {
    let baseline = Fixture::default().logits();
    let granite = Fixture::default()
        .with_scale("granite.embedding_scale", 12.0)
        .with_scale("granite.residual_scale", 0.22)
        .with_scale("granite.attention.scale", 0.0078125)
        .with_scale("granite.logit_scale", 8.0)
        .logits();
    assert!(granite.iter().all(|v| v.is_finite()));
    assert!(
        max_abs_diff(&baseline, &granite) > 1e-5,
        "a fully-specified Granite checkpoint must not match an unscaled one"
    );
}

/// A literal `0.0` means "no scaling", not "multiply the branch by zero"
/// (llama.cpp guards each on `!= 0.0f`).
#[test]
fn zero_scale_sentinels_are_inert() {
    let baseline = Fixture::default().logits();
    let zeroed = Fixture::default()
        .with_scale("granite.embedding_scale", 0.0)
        .with_scale("granite.residual_scale", 0.0)
        .with_scale("granite.attention.scale", 0.0)
        .logits();
    assert_eq!(
        max_abs_diff(&baseline, &zeroed),
        0.0,
        "0.0 sentinels must leave the model bit-identical to an unscaled one"
    );
}

/// `granite.rope.scaling.finetuned = false` switches RoPE off entirely
/// (`const bool use_rope = hparams.rope_finetuned;`), and it defaults to true.
#[test]
fn rope_finetuned_switches_rope_off() {
    let with_rope = Fixture::default().logits();
    let without = Fixture::default()
        .with_bool("granite.rope.scaling.finetuned", false)
        .logits();
    assert!(
        max_abs_diff(&with_rope, &without) > 1e-5,
        "rope_finetuned = false must skip the RoPE rotation"
    );

    let explicit_true = Fixture::default()
        .with_bool("granite.rope.scaling.finetuned", true)
        .logits();
    assert_eq!(
        max_abs_diff(&with_rope, &explicit_true),
        0.0,
        "rope_finetuned defaults to true for Granite"
    );
}

// ─── Loader hardening ────────────────────────────────────────────────────────

/// An over-declared `vocab_size` is reported at load, not indexed past at
/// decode.
///
/// `ModelConfig::from_metadata` falls back to the tokenizer token array and
/// then to a hard-coded 32000 when `{arch}.vocab_size` is absent, so the
/// declared vocabulary genuinely can exceed the embedding table's real row
/// count.  Checking a token id against `config.vocab_size` alone would still
/// walk off the end.
#[test]
fn over_declared_vocab_size_is_rejected_at_load() {
    let fixture = Fixture {
        declared_vocab: Some(VOCAB as u32 * 2),
        ..Fixture::default()
    };
    match fixture.load() {
        Err(ArchError::InvalidShape { name, .. }) => {
            assert_eq!(name, "token_embd.weight");
        }
        Err(other) => panic!("expected InvalidShape naming token_embd.weight, got {other}"),
        Ok(_) => panic!("a table with half the declared rows must not load"),
    }
}

/// A token id past the end of the vocabulary is an error, never a panic.
#[test]
fn out_of_range_token_is_reported() {
    let mut model = Fixture::default().load().expect("test: load");
    let mut kv = TestKv::new();
    let err = model
        .forward(&[VOCAB as u32], &mut kv)
        .expect_err("an out-of-vocabulary id must be rejected");
    assert!(
        matches!(err, ArchError::ConfigMismatch { .. }),
        "expected ConfigMismatch, got {err}"
    );
}

/// A prompt longer than the context window is rejected before any RoPE lookup.
#[test]
fn over_long_prompt_is_rejected() {
    let mut model = Fixture::default().load().expect("test: load");
    let mut kv = TestKv::new();
    let tokens: Vec<u32> = (0..(CTX as u32 + 1)).map(|i| i % VOCAB as u32).collect();
    let err = model
        .forward(&tokens, &mut kv)
        .expect_err("a prompt longer than the context must be rejected");
    assert!(
        matches!(err, ArchError::ConfigMismatch { .. }),
        "expected ConfigMismatch, got {err}"
    );
}

/// A truncated payload is reported by the shared bounds-checked loader rather
/// than slicing `&data[off..off + block_bytes]` past the end.
///
/// The cut removes the last tensor's final row from the *data* section, which
/// the GGUF header does not describe — so the file still parses and the
/// truncation is only discoverable at weight-load time.  That is exactly the
/// case the per-architecture `dequant_to_f32` copies got wrong.  The parse is
/// asserted rather than tolerated: if `GgufModel::from_bytes` ever grows a
/// data-section length check, this test must be re-cut instead of silently
/// becoming vacuous.
#[test]
fn truncated_gguf_is_reported_not_panicked() {
    let bytes = Fixture::default().build();
    let cut = bytes.len() - HIDDEN * 4;
    let gguf = GgufModel::from_bytes(bytes[..cut].to_vec()).expect(
        "the header is intact, so the truncated fixture must still parse and \
         reach the loader — re-cut the fixture if the parser has tightened",
    );
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("test: config");
    assert!(
        load_granite_from_gguf(&gguf, &config).is_err(),
        "a truncated checkpoint must be reported by the loader, not decoded"
    );
}

// ─── Registry routing ────────────────────────────────────────────────────────

/// `general.architecture = "granite"` is reachable through the registry.
///
/// `build_from_gguf` had no Granite override at all, so the registry's default
/// reported `"architecture 'granite' has not implemented build_from_gguf()"`
/// and the engine could not load a Granite checkpoint by architecture id.
#[test]
fn registry_routes_granite_to_a_runnable_model() {
    let registry = ArchitectureRegistry::with_builtins();
    let arch = registry.get("granite").expect("granite must be registered");

    let gguf = GgufModel::from_bytes(Fixture::default().build()).expect("test: parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("test: config");

    let mut model = arch
        .build_from_gguf(&gguf, &config)
        .expect("registry must build a Granite model from GGUF");
    assert_eq!(model.vocab_size(), VOCAB);
    assert_eq!(model.hidden_size(), HIDDEN);
    assert_eq!(model.max_context_length(), CTX);

    let mut kv = TestKv::new();
    let logits = model
        .forward(PROMPT, &mut kv)
        .expect("registry-built model must run");
    assert_eq!(logits.len(), VOCAB);
}

/// The multipliers read by the loader are exactly the ones in the file.
#[test]
fn scales_come_from_the_gguf_metadata() {
    let bytes = Fixture::default()
        .with_scale("granite.embedding_scale", 12.0)
        .with_scale("granite.residual_scale", 0.22)
        .with_scale("granite.attention.scale", 0.0078125)
        .with_scale("granite.logit_scale", 8.0)
        .build();
    let gguf = GgufModel::from_bytes(bytes).expect("test: parse");
    let scales = GraniteScales::from_metadata(&gguf.file.metadata, "granite");
    assert!((scales.embedding - 12.0).abs() < 1e-6);
    assert!((scales.residual - 0.22).abs() < 1e-6);
    assert!((scales.attention - 0.0078125).abs() < 1e-9);
    assert!((scales.logit - 8.0).abs() < 1e-6);
    assert!(scales.rope_finetuned);
    assert!(!scales.is_identity());

    let model = Fixture::default()
        .with_scale("granite.logit_scale", 8.0)
        .load()
        .expect("test: load");
    assert!((model.scales.logit - 8.0).abs() < 1e-6);
}

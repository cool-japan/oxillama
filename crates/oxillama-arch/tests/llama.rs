//! LLaMA correctness regressions, over a synthetic `Q4_K` checkpoint.
//!
//! Every test here pins a bug that made the llama path either produce silently
//! wrong text or abort the process:
//!
//! | Test | What used to happen |
//! |------|---------------------|
//! | [`rope_uses_the_norm_convention`] | NeoX half-split rotation on NORM-permuted weights → incoherent output |
//! | [`attention_width_wider_than_hidden_does_not_panic`] | `buf_attn_out` sized `hidden_size`, indexed by `num_heads * head_dim` |
//! | [`out_of_vocabulary_token_is_an_error`] | unchecked `token_embd[offset..offset + hidden]` slice |
//! | [`tied_embedding_checkpoint_loads`] | `output.weight` demanded unconditionally → Llama-3.2-1B/3B unloadable |
//! | [`over_long_prompt_is_an_error`] | walked off the end of the RoPE table |
//!
//! The projections are **Q4_K**, not F32, on purpose: the fused-Q8 GEMV path and
//! the batched prefill are both gated on the weight kernel advertising a fused
//! activation route, and only the K-quants do.  An F32 fixture would quietly
//! bypass everything this file is meant to cover.

use oxillama_arch::llama::{load_llama_from_gguf, LlamaModel};
use oxillama_arch::{ArchResult, ForwardPass, KvCacheAccess, ModelConfig};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

// Geometry chosen so nothing lines up by accident:
//   * ATTN_DIM (512) != HIDDEN (256) — a buffer sized from the wrong one trips.
//   * HEADS != KV_HEADS, so grouped-query head mapping is exercised.
//   * VOCAB != HIDDEN and VOCAB != ATTN_DIM.
//   * Two layers, so per-layer KV staging cannot be confused with one buffer.
//   * Every `in_features` is a multiple of the 256-weight Q4_K block.
const HIDDEN: usize = 256;
const FFN: usize = 512;
const VOCAB: usize = 768;
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const ATTN_DIM: usize = HEADS * HEAD_DIM; // 512
const KV_DIM: usize = KV_HEADS * HEAD_DIM; // 128
const LAYERS: usize = 2;
const CTX: usize = 128;

/// Bytes per Q4_K block (2 d + 2 dmin + 12 packed scales + 128 nibbles).
const Q4_K_BYTES: usize = 144;
/// Weights per Q4_K block.
const Q4_K_BLOCK: usize = 256;

/// Deterministic xorshift64* — reproducible fixtures without a PRNG dep.
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
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }
    /// Uniform in `[-1, 1)`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 8_388_608.0 - 1.0
    }
}

/// `n_rows × n_cols` of Q4_K.  Every byte pattern is a legal block, so random
/// scale/nibble payloads are valid — and unusually harsh — test weights.
fn q4_k_matrix(n_rows: usize, n_cols: usize, rng: &mut Rng) -> Vec<u8> {
    let blocks = n_rows * n_cols.div_ceil(Q4_K_BLOCK);
    let mut data = Vec::with_capacity(blocks * Q4_K_BYTES);
    for _ in 0..blocks {
        // Keep d/dmin small so accumulated dot products stay well inside fp32.
        let d = half::f16::from_f32(rng.next_f32() * 0.05);
        let dmin = half::f16::from_f32(rng.next_f32() * 0.02);
        data.extend_from_slice(&d.to_bits().to_le_bytes());
        data.extend_from_slice(&dmin.to_bits().to_le_bytes());
        for _ in 0..12 + 128 {
            data.push(rng.next_u8());
        }
    }
    data
}

/// F32 vector payload with values near 1.0 (a plausible RMSNorm scale).
fn f32_vec_bytes(n: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        out.extend_from_slice(&(1.0 + 0.1 * rng.next_f32()).to_le_bytes());
    }
    out
}

/// What to vary about the fixture.
#[derive(Clone, Copy)]
struct Fixture {
    /// Emit a standalone `output.weight`; `false` models a tied checkpoint.
    with_output: bool,
    /// Declared context length.
    ctx: usize,
    /// Declared vocabulary size.  Over-declaring it — which `config.rs` does
    /// whenever `{arch}.vocab_size` is missing — is exactly how an in-range
    /// token id can address a row the embedding table does not have.
    declared_vocab: usize,
}

impl Default for Fixture {
    fn default() -> Self {
        Self {
            with_output: true,
            ctx: CTX,
            declared_vocab: VOCAB,
        }
    }
}

/// Serialize a two-layer LLaMA checkpoint whose projections are all Q4_K.
fn build_fixture(opts: Fixture) -> Vec<u8> {
    let mut rng = Rng::new(0x11A_11A5);
    let mut writer = GgufWriter::new();

    for (key, value) in [
        (
            "general.architecture",
            MetadataValue::String("llama".to_string()),
        ),
        (
            "llama.embedding_length",
            MetadataValue::Uint32(HIDDEN as u32),
        ),
        (
            "llama.feed_forward_length",
            MetadataValue::Uint32(FFN as u32),
        ),
        ("llama.block_count", MetadataValue::Uint32(LAYERS as u32)),
        (
            "llama.attention.head_count",
            MetadataValue::Uint32(HEADS as u32),
        ),
        (
            "llama.attention.head_count_kv",
            MetadataValue::Uint32(KV_HEADS as u32),
        ),
        (
            "llama.attention.key_length",
            MetadataValue::Uint32(HEAD_DIM as u32),
        ),
        (
            "llama.attention.value_length",
            MetadataValue::Uint32(HEAD_DIM as u32),
        ),
        (
            "llama.context_length",
            MetadataValue::Uint32(opts.ctx as u32),
        ),
        (
            "llama.vocab_size",
            MetadataValue::Uint32(opts.declared_vocab as u32),
        ),
        ("llama.rope.freq_base", MetadataValue::Float32(10000.0)),
    ] {
        writer.add_metadata(key, value);
    }

    // GGUF writes `ne` fastest-changing-first, so every weight is declared as
    // [in_features, out_features] — the reverse of the math shape.
    let add_q4k =
        |writer: &mut GgufWriter, name: &str, out_f: usize, in_f: usize, rng: &mut Rng| {
            let data = q4_k_matrix(out_f, in_f, rng);
            writer.add_tensor(
                name,
                &[in_f as u64, out_f as u64],
                GgufTensorType::Q4K,
                &data,
            );
        };

    add_q4k(&mut writer, "token_embd.weight", VOCAB, HIDDEN, &mut rng);

    for i in 0..LAYERS {
        let p = format!("blk.{i}");
        for (name, out_f, in_f) in [
            (format!("{p}.attn_q.weight"), ATTN_DIM, HIDDEN),
            (format!("{p}.attn_k.weight"), KV_DIM, HIDDEN),
            (format!("{p}.attn_v.weight"), KV_DIM, HIDDEN),
            (format!("{p}.attn_output.weight"), HIDDEN, ATTN_DIM),
            (format!("{p}.ffn_gate.weight"), FFN, HIDDEN),
            (format!("{p}.ffn_up.weight"), FFN, HIDDEN),
            (format!("{p}.ffn_down.weight"), HIDDEN, FFN),
        ] {
            add_q4k(&mut writer, &name, out_f, in_f, &mut rng);
        }
        for name in [
            format!("{p}.attn_norm.weight"),
            format!("{p}.ffn_norm.weight"),
        ] {
            let bytes = f32_vec_bytes(HIDDEN, &mut rng);
            writer.add_tensor(&name, &[HIDDEN as u64], GgufTensorType::F32, &bytes);
        }
    }

    let bytes = f32_vec_bytes(HIDDEN, &mut rng);
    writer.add_tensor(
        "output_norm.weight",
        &[HIDDEN as u64],
        GgufTensorType::F32,
        &bytes,
    );

    if opts.with_output {
        add_q4k(&mut writer, "output.weight", VOCAB, HIDDEN, &mut rng);
    }

    let mut out = Vec::new();
    writer
        .write_to(&mut out)
        .expect("synthetic LLaMA GGUF must serialize");
    out
}

fn load_with(opts: Fixture) -> LlamaModel {
    let gguf = GgufModel::from_bytes(build_fixture(opts)).expect("fixture must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata");
    load_llama_from_gguf(&gguf, &config).expect("fixture must load")
}

fn load_model() -> LlamaModel {
    load_with(Fixture::default())
}

/// Flat per-layer KV cache with the same visibility rules as the runtime's:
/// a key written at the current position is readable before `advance()`.
///
/// `seq_len` is settable so a test can place a token at an arbitrary absolute
/// position without replaying the prefix.
struct TestKv {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    seq_len: usize,
    stored_len: usize,
}

impl TestKv {
    fn new() -> Self {
        Self {
            keys: vec![vec![0.0; (CTX + 2) * KV_DIM]; LAYERS],
            values: vec![vec![0.0; (CTX + 2) * KV_DIM]; LAYERS],
            seq_len: 0,
            stored_len: 0,
        }
    }

    /// Pretend `n` tokens are already cached (their K/V stay zero).
    fn primed(n: usize) -> Self {
        let mut kv = Self::new();
        kv.seq_len = n;
        kv.stored_len = n;
        kv
    }

    /// Layer 0's key at absolute position `pos`, for head `kv_head`.
    fn key_head(&self, pos: usize, kv_head: usize) -> Vec<f32> {
        let off = pos * KV_DIM + kv_head * HEAD_DIM;
        self.keys[0][off..off + HEAD_DIM].to_vec()
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
        self.seq_len += 1;
        if self.stored_len < self.seq_len {
            self.stored_len = self.seq_len;
        }
    }
    fn kv_dim(&self) -> usize {
        KV_DIM
    }
}

// ── TASK F: RoPE convention ──────────────────────────────────────────────────

/// LLaMA rotates **consecutive** pairs `(x[2j], x[2j+1])`, not NeoX's
/// half-split `(x[j], x[j + head_dim/2])`.
///
/// llama.cpp: `llama_model_rope_type` maps `LLM_ARCH_LLAMA` to
/// `LLAMA_ROPE_TYPE_NORM` (`src/llama-model.cpp`), and
/// `convert_hf_to_gguf.py`'s `LlamaModel` sets `undo_permute = True`, which
/// reorders every `q_proj`/`k_proj` row precisely so that the NORM pairing is
/// the right one.  Applying the NeoX rotation to those permuted rows pairs each
/// dimension with the wrong partner and rotates it by the wrong angle — finite
/// logits, incoherent text, no error anywhere.
///
/// The probe: the key written into the cache at position 0 is unrotated (RoPE at
/// position 0 is the identity), so it *is* the pre-RoPE vector.  Feeding the
/// same token to a fresh model whose cache claims one prior token gives the same
/// pre-RoPE vector rotated for position 1.  Comparing that against both
/// conventions identifies which one ran.
#[test]
fn rope_uses_the_norm_convention() {
    let token = 5u32;

    let mut m0 = load_model();
    let mut kv0 = TestKv::new();
    m0.forward(&[token], &mut kv0).expect("position 0");
    let unrotated = kv0.key_head(0, 0);

    let mut m1 = load_model();
    let mut kv1 = TestKv::primed(1);
    m1.forward(&[token], &mut kv1).expect("position 1");
    let rotated = kv1.key_head(1, 0);

    assert!(
        unrotated.iter().any(|v| v.abs() > 1e-6),
        "fixture must produce a non-trivial key, got {unrotated:?}"
    );

    // NORM: pairs (2j, 2j+1).
    let mut norm = unrotated.clone();
    let half = m1.rope.half_dim;
    for j in 0..half {
        let (c, s) = (m1.rope.cos[half + j], m1.rope.sin[half + j]);
        let (x0, x1) = (norm[2 * j], norm[2 * j + 1]);
        norm[2 * j] = x0 * c - x1 * s;
        norm[2 * j + 1] = x0 * s + x1 * c;
    }
    // NeoX: pairs (j, j + half) — what `RopeTable::apply` does.
    let mut neox = unrotated.clone();
    m1.rope.apply(&mut neox, 1);

    let close = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-4 * x.abs().max(1.0))
    };

    assert!(
        !close(&norm, &neox),
        "the fixture must distinguish the two conventions"
    );
    assert!(
        close(&rotated, &norm),
        "llama must rotate consecutive pairs (LLAMA_ROPE_TYPE_NORM).\n\
         cached  = {:?}\n expected(NORM) = {:?}\n neox   = {:?}",
        &rotated[..8],
        &norm[..8],
        &neox[..8]
    );
}

// ── TASK F: buffer sizing ────────────────────────────────────────────────────

/// `num_heads * head_dim` is not always `hidden_size`.
///
/// The fixture attends over 8 × 64 = 512 with a 256-wide residual stream, which
/// is the same decoupling Qwen3-4B has.  `buf_attn_out` used to be sized
/// `hidden_size` and indexed `h * head_dim .. (h+1) * head_dim`, so head 4
/// sliced past the end and the process aborted.
#[test]
fn attention_width_wider_than_hidden_does_not_panic() {
    assert_ne!(ATTN_DIM, HIDDEN, "the fixture must decouple the two widths");

    let mut model = load_model();
    let mut kv = TestKv::new();
    let logits = model
        .forward(&[1u32], &mut kv)
        .expect("a model whose attention is wider than its residual stream must run");

    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ── TASK F: out-of-vocabulary token ──────────────────────────────────────────

/// A token id past the embedding table is an error, not an abort.
///
/// `vocab_size` is routinely over-declared: `ModelConfig::from_metadata` falls
/// back to the tokenizer token-array length and then to a hard-coded 32000, so a
/// server can legitimately be handed an id the table has no row for.
#[test]
fn out_of_vocabulary_token_is_an_error() {
    let mut model = load_model();
    let mut kv = TestKv::new();

    let err = model
        .forward(&[VOCAB as u32], &mut kv)
        .expect_err("token id == vocab_size must be rejected");
    let text = err.to_string();
    assert!(
        text.contains("token id"),
        "error should name the offending token id, got: {text}"
    );

    // And a wildly out-of-range id, which the old slice would have read from
    // gigabytes past the table.
    assert!(model.forward(&[u32::MAX], &mut kv).is_err());
}

/// The same guard covers an over-declared `vocab_size`, where the id is inside
/// `config.vocab_size` but outside the actual table.
#[test]
fn token_inside_an_over_declared_vocab_is_an_error() {
    let mut model = load_with(Fixture {
        declared_vocab: VOCAB * 2,
        ..Fixture::default()
    });
    let mut kv = TestKv::new();
    assert_eq!(model.vocab_size(), VOCAB * 2);
    assert!(
        model.forward(&[(VOCAB + 1) as u32], &mut kv).is_err(),
        "an id inside the declared vocab but outside the table must error"
    );
}

// ── TASK F: tied embeddings ──────────────────────────────────────────────────

/// Llama-3.2-1B/3B ship no `output.weight` and reuse `token_embd.weight`.
///
/// llama.cpp loads `output` with `TENSOR_NOT_REQUIRED` and falls back to
/// `tok_embd`.  Demanding it made every tied checkpoint — and every LLaVA build
/// on top of one — refuse to load.
#[test]
fn tied_embedding_checkpoint_loads() {
    let opts = Fixture {
        with_output: false,
        ..Fixture::default()
    };
    let gguf = GgufModel::from_bytes(build_fixture(opts)).expect("fixture must parse");
    assert!(
        !gguf.file.tensors.contains("output.weight"),
        "fixture must model a tied checkpoint"
    );

    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata");
    let mut model =
        load_llama_from_gguf(&gguf, &config).expect("a tied checkpoint must load, not error");

    assert_eq!(
        model.output.weight.shape,
        vec![VOCAB, HIDDEN],
        "the tied LM head must keep token_embd's dimensions"
    );
    assert_eq!(
        model.output.weight.tensor_type,
        GgufTensorType::Q4K,
        "the tied LM head must stay quantized"
    );

    let mut kv = TestKv::new();
    let logits = model.forward(&[3u32], &mut kv).expect("tied head must run");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// With neither tensor present the error still names `output.weight`.
#[test]
fn missing_lm_head_and_embedding_reports_output_weight() {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataValue::String("llama".to_string()),
    );
    writer.add_metadata("llama.block_count", MetadataValue::Uint32(0));
    let mut bytes = Vec::new();
    writer.write_to(&mut bytes).expect("must serialize");
    let gguf = GgufModel::from_bytes(bytes).expect("must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("metadata");

    let Err(err) = load_llama_from_gguf(&gguf, &config) else {
        panic!("a headless GGUF must fail to load");
    };
    assert!(
        err.to_string().contains("token_embd.weight"),
        "the first missing tensor should be reported, got: {err}"
    );
}

// ── TASK F: context guard ────────────────────────────────────────────────────

/// An over-long prompt is rejected before a weight is touched.
///
/// Past `max_context_length` the RoPE table has no entry for the position: the
/// old code indexed straight past it and aborted.
#[test]
fn over_long_prompt_is_an_error() {
    let mut model = load_model();
    let mut kv = TestKv::new();
    let tokens: Vec<u32> = (0..CTX + 1).map(|i| (i % VOCAB) as u32).collect();

    let err = model
        .forward(&tokens, &mut kv)
        .expect_err("a prompt longer than the context window must be rejected");
    assert!(
        err.to_string().contains("context overflow"),
        "error should explain the overflow, got: {err}"
    );
}

/// A prompt that exactly fills the window is still accepted, and a prompt that
/// overflows an already-warm cache is rejected.
#[test]
fn exactly_full_context_is_accepted_and_the_next_token_is_not() {
    let mut model = load_model();
    let mut kv = TestKv::new();
    let tokens: Vec<u32> = (0..CTX).map(|i| (i % VOCAB) as u32).collect();

    model
        .forward(&tokens, &mut kv)
        .expect("a prompt of exactly max_context_length must be accepted");
    assert_eq!(kv.seq_len(), CTX);
    assert!(
        model.forward(&[1u32], &mut kv).is_err(),
        "one more token past a full window must be rejected"
    );
}

// ── TASK C: quantized embedding table ────────────────────────────────────────

/// The embedding table is kept in its GGUF form, not expanded to f32.
///
/// Llama-3-8B's `[128256, 4096]` table is 2.10 GB as `Vec<f32>`; the fixture's
/// is small, so the observable property is the *representation*: a quantized
/// table exposes the checkpoint's row count and hidden size without ever
/// materialising a float per weight.
#[test]
fn embedding_table_stays_quantized() {
    let model = load_model();
    assert!(
        matches!(
            model.token_embd,
            oxillama_arch::llama::TokenEmbedding::Quantized { .. }
        ),
        "a block-aligned Q4_K embedding table must not be bulk-dequantized"
    );
    assert_eq!(model.token_embd.vocab_size(), VOCAB);
    assert_eq!(model.token_embd.hidden_size(), HIDDEN);
}

// ── TASK E: logits are moved, not cloned ─────────────────────────────────────

/// Two consecutive decodes must both return full-length logits.
///
/// `forward` hands `buf_logits` to the caller by `mem::take`, which leaves the
/// field empty; the next call has to restore its length before the LM head
/// writes into it.  Getting that wrong would surface as a zero-length (or
/// stale) second result.
#[test]
fn consecutive_decodes_each_return_fresh_logits() {
    let mut model = load_model();
    let mut kv = TestKv::new();

    let first = model.forward(&[1u32], &mut kv).expect("first decode");
    let second = model.forward(&[2u32], &mut kv).expect("second decode");
    let third = model.forward(&[3u32], &mut kv).expect("third decode");

    for (i, l) in [&first, &second, &third].iter().enumerate() {
        assert_eq!(l.len(), VOCAB, "decode {i} returned {} logits", l.len());
        assert!(l.iter().all(|v| v.is_finite()), "decode {i} not finite");
    }
    assert!(
        first
            .iter()
            .zip(&second)
            .any(|(a, b)| (a - b).abs() > f32::EPSILON),
        "different tokens at different positions must give different logits"
    );
}

/// `embed` returns the hidden state, `forward` the logits — different widths.
#[test]
fn embed_returns_hidden_state_not_logits() {
    let mut model = load_model();
    let mut kv = TestKv::new();
    let hidden = model.embed(&[1u32], &mut kv).expect("embed");
    assert_eq!(hidden.len(), HIDDEN);
    assert_ne!(HIDDEN, VOCAB);
    assert!(hidden.iter().all(|v| v.is_finite()));
}

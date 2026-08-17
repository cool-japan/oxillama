//! Attention must work against a KV cache that cannot lend a `&[f32]`.
//!
//! Until 0.1.4 every architecture read its cache through
//! [`KvCacheAccess::get_keys`]/[`get_values`], which hand out a **borrow** of a
//! contiguous `f32` run.  Two perfectly legitimate cache layouts cannot produce
//! one:
//!
//! * `KvCache` with `KvCacheDtype::F16` — the elements are `f16`, so there is
//!   no `&[f32]` in memory to lend.  Selecting it made the forward pass fail on
//!   the very first layer of the very first token.
//! * `PagedKvCache` with a multi-page sequence — the data *is* `f32`, but not
//!   contiguous.
//!
//! Both answer `get_keys()` with a typed error and serve `for_each_key()`
//! instead, so this file pins the two properties that make that arrangement
//! usable:
//!
//! | Test | Property |
//! |------|----------|
//! | [`gathered_f32_cache_is_bit_identical`] | the gather path is **exact** — a non-borrowable *f32* cache produces bit-identical logits, so the migration changed no arithmetic |
//! | [`gathered_f32_batched_prefill_is_bit_identical`] | likewise for the separate batched-prefill kernel |
//! | [`f16_decode_equals_f32_pre_rounded_exactly`] | decode over f16 storage is **bit-identical** to decode over a borrowable f32 cache holding the same values pre-rounded through f16 |
//! | [`f16_chunked_prefill_equals_f32_pre_rounded_exactly`] | the same, for the batched-prefill kernel, driven in two chunks so it really reads stored K/V |
//! | [`f16_storage_actually_perturbs_the_logits`] | control: the f16 buffers are genuinely in use |
//! | [`single_chunk_prefill_does_not_read_stored_kv`] | control: a one-chunk prefill attends only to its own staged f32 K/V |
//! | [`f16_decode_stays_within_a_few_percent_of_f32`] | crash barrier on the size of the f16 perturbation |
//! | [`f16_storage_is_half_the_bytes`] | the point of the exercise: the cache really is half the size |
//!
//! ## Why the exact assertion is the `PreRounded` one
//!
//! Comparing f16 KV against f32 KV on this fixture measures the *fixture*, not
//! the implementation: the weights are uniformly random Q4_K, so attention
//! scores have a huge dynamic range, the softmax sits close to a hard argmax,
//! and a `2^-11` perturbation of one key can flip which position wins.  Measured
//! drift is ~10.6 on a logit scale of ~597 (1.8%) — a number that says nothing
//! about whether the f16 *path* is correct.
//!
//! So the load-bearing tests compare f16 storage against an f32 cache fed
//! **f16-rounded values**, which is bit-exact and carries no conditioning
//! assumption whatsoever.  The f32-vs-f16 comparison is kept only as a crash
//! barrier, with its bound stated for what it is.

use half::f16;
use oxillama_arch::llama::{load_llama_from_gguf, LlamaModel};
use oxillama_arch::{ArchError, ArchResult, ForwardPass, KvCacheAccess, ModelConfig};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

// Same geometry as `tests/llama.rs`: nothing lines up by accident, GQA is
// exercised (HEADS != KV_HEADS), and every `in_features` is a multiple of the
// 256-weight Q4_K block.
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

/// `n_rows × n_cols` of Q4_K.  Every byte pattern is a legal block.
fn q4_k_matrix(n_rows: usize, n_cols: usize, rng: &mut Rng) -> Vec<u8> {
    let blocks = n_rows * n_cols.div_ceil(Q4_K_BLOCK);
    let mut data = Vec::with_capacity(blocks * Q4_K_BYTES);
    for _ in 0..blocks {
        let d = f16::from_f32(rng.next_f32() * 0.05);
        let dmin = f16::from_f32(rng.next_f32() * 0.02);
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

/// Serialize a two-layer LLaMA checkpoint whose projections are all Q4_K.
fn build_fixture() -> Vec<u8> {
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
        ("llama.context_length", MetadataValue::Uint32(CTX as u32)),
        ("llama.vocab_size", MetadataValue::Uint32(VOCAB as u32)),
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
    add_q4k(&mut writer, "output.weight", VOCAB, HIDDEN, &mut rng);

    let mut out = Vec::new();
    writer
        .write_to(&mut out)
        .expect("synthetic LLaMA GGUF must serialize");
    out
}

fn load_model() -> LlamaModel {
    let gguf = GgufModel::from_bytes(build_fixture()).expect("fixture must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata");
    load_llama_from_gguf(&gguf, &config).expect("fixture must load")
}

// ── Cache doubles ────────────────────────────────────────────────────────────

/// How a cache double stores its elements.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Storage {
    /// `f32` elements, borrowable — the historical fast path.
    F32Borrow,
    /// `f32` elements the cache refuses to lend, standing in for a paged layout.
    /// The gather path must reproduce the borrow path *exactly* here.
    F32Gather,
    /// `f16` elements — `KvCacheDtype::F16`'s shape: no borrow possible, and
    /// every element round-trips through half precision.
    F16,
    /// `f32` elements that were *pre-rounded* through `f16` on the way in, then
    /// stored and lent as ordinary borrowable `f32`.
    ///
    /// This is the control for [`Storage::F16`]: it holds numerically identical
    /// data but takes the untouched borrow path, so any difference between the
    /// two isolates the f16 read path itself from the f16 rounding.
    F32PreRounded,
}

/// A `KvCacheAccess` whose element type and borrowability are configurable,
/// with the runtime cache's visibility rule: a key written at the current
/// position is readable before `advance()`.
struct DualKv {
    storage: Storage,
    keys_f32: Vec<Vec<f32>>,
    values_f32: Vec<Vec<f32>>,
    keys_f16: Vec<Vec<f16>>,
    values_f16: Vec<Vec<f16>>,
    seq_len: usize,
    stored_len: usize,
}

impl DualKv {
    fn new(storage: Storage) -> Self {
        let elems = (CTX + 2) * KV_DIM;
        Self {
            storage,
            keys_f32: vec![vec![0.0; elems]; LAYERS],
            values_f32: vec![vec![0.0; elems]; LAYERS],
            keys_f16: vec![vec![f16::ZERO; elems]; LAYERS],
            values_f16: vec![vec![f16::ZERO; elems]; LAYERS],
            seq_len: 0,
            stored_len: 0,
        }
    }

    /// Bytes held by the K and V buffers actually in use for this storage mode.
    fn live_bytes(&self) -> usize {
        match self.storage {
            Storage::F32Borrow | Storage::F32Gather | Storage::F32PreRounded => {
                (self.keys_f32.len() + self.values_f32.len())
                    * self.keys_f32[0].len()
                    * core::mem::size_of::<f32>()
            }
            Storage::F16 => {
                (self.keys_f16.len() + self.values_f16.len())
                    * self.keys_f16[0].len()
                    * core::mem::size_of::<f16>()
            }
        }
    }

    fn rows(&self, layer: usize, keys: bool, f: &mut dyn FnMut(usize, &[f32])) {
        match self.storage {
            Storage::F32Borrow | Storage::F32Gather | Storage::F32PreRounded => {
                let buf = if keys {
                    &self.keys_f32[layer]
                } else {
                    &self.values_f32[layer]
                };
                for (pos, row) in buf[..self.stored_len * KV_DIM]
                    .chunks_exact(KV_DIM)
                    .enumerate()
                {
                    f(pos, row);
                }
            }
            Storage::F16 => {
                let buf = if keys {
                    &self.keys_f16[layer]
                } else {
                    &self.values_f16[layer]
                };
                let mut row = vec![0.0f32; KV_DIM];
                for pos in 0..self.stored_len {
                    let off = pos * KV_DIM;
                    for (dst, src) in row.iter_mut().zip(buf[off..off + KV_DIM].iter()) {
                        *dst = src.to_f32();
                    }
                    f(pos, &row);
                }
            }
        }
    }
}

impl KvCacheAccess for DualKv {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let off = self.seq_len * KV_DIM;
        match self.storage {
            Storage::F32Borrow | Storage::F32Gather => {
                self.keys_f32[layer][off..off + KV_DIM].copy_from_slice(&key[..KV_DIM]);
                self.values_f32[layer][off..off + KV_DIM].copy_from_slice(&value[..KV_DIM]);
            }
            Storage::F32PreRounded => {
                for (dst, &s) in self.keys_f32[layer][off..off + KV_DIM]
                    .iter_mut()
                    .zip(key[..KV_DIM].iter())
                {
                    *dst = f16::from_f32(s).to_f32();
                }
                for (dst, &s) in self.values_f32[layer][off..off + KV_DIM]
                    .iter_mut()
                    .zip(value[..KV_DIM].iter())
                {
                    *dst = f16::from_f32(s).to_f32();
                }
            }
            Storage::F16 => {
                for (dst, &s) in self.keys_f16[layer][off..off + KV_DIM]
                    .iter_mut()
                    .zip(key[..KV_DIM].iter())
                {
                    *dst = f16::from_f32(s);
                }
                for (dst, &s) in self.values_f16[layer][off..off + KV_DIM]
                    .iter_mut()
                    .zip(value[..KV_DIM].iter())
                {
                    *dst = f16::from_f32(s);
                }
            }
        }
        if self.stored_len <= self.seq_len {
            self.stored_len = self.seq_len + 1;
        }
        Ok(())
    }

    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        match self.storage {
            Storage::F32Borrow | Storage::F32PreRounded => {
                Ok(&self.keys_f32[layer][..self.stored_len * KV_DIM])
            }
            _ => Err(ArchError::ForwardPassError {
                layer,
                message: "this cache cannot lend a contiguous &[f32] of keys".to_string(),
            }),
        }
    }

    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        match self.storage {
            Storage::F32Borrow | Storage::F32PreRounded => {
                Ok(&self.values_f32[layer][..self.stored_len * KV_DIM])
            }
            _ => Err(ArchError::ForwardPassError {
                layer,
                message: "this cache cannot lend a contiguous &[f32] of values".to_string(),
            }),
        }
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

    fn for_each_key(&self, layer: usize, f: &mut dyn FnMut(usize, &[f32])) -> ArchResult<()> {
        self.rows(layer, true, f);
        Ok(())
    }

    fn for_each_value(&self, layer: usize, f: &mut dyn FnMut(usize, &[f32])) -> ArchResult<()> {
        self.rows(layer, false, f);
        Ok(())
    }
}

// ── Drivers ──────────────────────────────────────────────────────────────────

/// Feed `tokens` one at a time (the decode path) and return the final logits.
fn decode_one_by_one(model: &mut LlamaModel, storage: Storage, tokens: &[u32]) -> Vec<f32> {
    let mut kv = DualKv::new(storage);
    let mut logits = Vec::new();
    for &t in tokens {
        logits = model.forward(&[t], &mut kv).expect("decode forward");
    }
    logits
}

/// Feed `tokens` in a single call (the batched-prefill path) and return the
/// final logits.
fn prefill_in_one_batch(model: &mut LlamaModel, storage: Storage, tokens: &[u32]) -> Vec<f32> {
    let mut kv = DualKv::new(storage);
    model.forward(tokens, &mut kv).expect("prefill forward")
}

/// Feed `tokens` as two consecutive multi-token calls.
///
/// This is the configuration that makes the batched-prefill kernel actually
/// *read* the KV cache: within one chunk it attends to its own staged, still-f32
/// K/V, so a single-chunk prefill never touches stored elements at all (measured:
/// f16 and f32 single-chunk prefills agree bit-for-bit).  The second chunk has to
/// go through `fetch_keys`/`fetch_values` for the first chunk's positions.
fn prefill_in_two_chunks(model: &mut LlamaModel, storage: Storage, tokens: &[u32]) -> Vec<f32> {
    let mut kv = DualKv::new(storage);
    let split = tokens.len() / 2;
    model
        .forward(&tokens[..split], &mut kv)
        .expect("first prefill chunk");
    model
        .forward(&tokens[split..], &mut kv)
        .expect("second prefill chunk")
}

/// Largest absolute difference between two logit vectors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit vectors must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Largest absolute logit magnitude — the scale drift is measured against.
fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |acc, x| acc.max(x.abs()))
}

/// Relative drift bound for `f16` KV on **this fixture** — see the file header.
///
/// Deliberately loose, and deliberately not the load-bearing assertion: the
/// exact statement about the f16 path is
/// [`f16_decode_equals_f32_pre_rounded_exactly`].
const F16_RELATIVE_BOUND: f32 = 0.05;

const TOKENS: [u32; 6] = [1, 7, 42, 300, 5, 99];

// ── Tests ────────────────────────────────────────────────────────────────────

/// The gather path must be arithmetically invisible.
///
/// `Storage::F32Gather` holds the same `f32` bits as `Storage::F32Borrow` and
/// only differs in *how* attention reads them.  Anything other than bit-equality
/// here means the migration to `fetch_keys`/`fetch_values` changed the maths —
/// which would silently move every architecture's output, not just f16's.
#[test]
fn gathered_f32_cache_is_bit_identical() {
    let mut model = load_model();
    let borrowed = decode_one_by_one(&mut model, Storage::F32Borrow, &TOKENS);
    let gathered = decode_one_by_one(&mut model, Storage::F32Gather, &TOKENS);
    assert_eq!(
        borrowed, gathered,
        "a non-borrowable f32 cache must produce bit-identical logits"
    );
}

/// The same, for the separate batched-prefill attention kernel.
#[test]
fn gathered_f32_batched_prefill_is_bit_identical() {
    let mut model = load_model();
    let borrowed = prefill_in_one_batch(&mut model, Storage::F32Borrow, &TOKENS);
    let gathered = prefill_in_one_batch(&mut model, Storage::F32Gather, &TOKENS);
    assert_eq!(
        borrowed, gathered,
        "batched prefill over a gathered f32 cache must be bit-identical"
    );
}

/// **The load-bearing assertion.** Decoding over `f16` storage must produce
/// *bit-identical* logits to decoding over an ordinary borrowable `f32` cache
/// that was fed the same values pre-rounded through `f16`.
///
/// Both runs see numerically identical keys and values; they differ only in how
/// attention gets at them (`for_each_key` conversion vs a plain borrow).  Zero
/// tolerance, and no assumption about the fixture's conditioning:
///
/// * a dropped, duplicated or mis-ordered row → mismatch;
/// * conversion applied to the wrong element → mismatch;
/// * accumulation silently narrowed to `f16` → mismatch;
/// * a `f16::from_f32` rounding mode that disagrees with the storage layer's →
///   mismatch.
///
/// Before this wave the f16 run did not produce *drifted* logits — it produced
/// an error, because `get_keys()` cannot borrow `&[f32]` from a `Vec<f16>`.
#[test]
fn f16_decode_equals_f32_pre_rounded_exactly() {
    let mut model = load_model();
    let via_f16 = decode_one_by_one(&mut model, Storage::F16, &TOKENS);
    let via_pre_rounded = decode_one_by_one(&mut model, Storage::F32PreRounded, &TOKENS);
    assert_eq!(via_f16.len(), VOCAB, "logits must cover the vocabulary");
    assert_eq!(
        via_f16, via_pre_rounded,
        "f16 storage must add nothing beyond f16 rounding of the stored elements"
    );
}

/// The same bit-exact property for the batched-prefill kernel, driven in two
/// chunks so the second chunk genuinely reads stored (f16) K/V.
#[test]
fn f16_chunked_prefill_equals_f32_pre_rounded_exactly() {
    let mut model = load_model();
    let via_f16 = prefill_in_two_chunks(&mut model, Storage::F16, &TOKENS);
    let via_pre_rounded = prefill_in_two_chunks(&mut model, Storage::F32PreRounded, &TOKENS);
    assert_eq!(
        via_f16, via_pre_rounded,
        "batched prefill over f16 storage must add nothing beyond f16 rounding"
    );
}

/// The f16 read path must actually be *exercised* — a control for the two tests
/// above, which would also pass if `Storage::F16` silently kept f32 data.
#[test]
fn f16_storage_actually_perturbs_the_logits() {
    let mut model = load_model();
    let exact = decode_one_by_one(&mut model, Storage::F32Borrow, &TOKENS);
    let rounded = decode_one_by_one(&mut model, Storage::F16, &TOKENS);
    assert!(
        max_abs_diff(&exact, &rounded) > 0.0,
        "f16 storage produced bit-identical logits — the f16 buffers were never used"
    );
}

/// f16 KV must stay within a few percent of the f32 run in *relative* terms.
///
/// This is a crash barrier, not a precision claim.  The fixture's weights are
/// uniformly random Q4_K, which gives attention scores a huge dynamic range; the
/// softmax is then close to a hard argmax and a `2^-11` perturbation of a key can
/// flip which position wins, so the observed drift (~1.8% of the logit scale)
/// reflects the fixture's conditioning far more than f16's precision.  A trained
/// checkpoint is much better behaved — which is why llama.cpp ships f16 KV as its
/// *default* — and the end-to-end evidence for that lives in the real-model runs,
/// not here.  What this test catches is an f16 path that is wrong by an order of
/// magnitude rather than by rounding.
#[test]
fn f16_decode_stays_within_a_few_percent_of_f32() {
    let mut model = load_model();
    let exact = decode_one_by_one(&mut model, Storage::F32Borrow, &TOKENS);
    let rounded = decode_one_by_one(&mut model, Storage::F16, &TOKENS);
    let scale = max_abs(&exact);
    assert!(scale > 0.0, "the fixture must produce non-zero logits");
    let relative = max_abs_diff(&exact, &rounded) / scale;
    assert!(
        relative <= F16_RELATIVE_BOUND,
        "f16 KV drifted {relative} relative to a logit scale of {scale}, \
         bound {F16_RELATIVE_BOUND}"
    );
}

/// A single-chunk prefill never reads stored K/V, so it must be *insensitive* to
/// the storage dtype.
///
/// Pinning this stops the chunked test above from being quietly weakened into a
/// single-chunk one, which would assert nothing at all.
#[test]
fn single_chunk_prefill_does_not_read_stored_kv() {
    let mut model = load_model();
    let exact = prefill_in_one_batch(&mut model, Storage::F32Borrow, &TOKENS);
    let rounded = prefill_in_one_batch(&mut model, Storage::F16, &TOKENS);
    assert_eq!(
        exact, rounded,
        "a one-chunk prefill attends only to its own staged f32 K/V"
    );
}

/// The reason for all of the above: half the bytes.
#[test]
fn f16_storage_is_half_the_bytes() {
    let f32_kv = DualKv::new(Storage::F32Borrow);
    let f16_kv = DualKv::new(Storage::F16);
    assert_eq!(
        f16_kv.live_bytes() * 2,
        f32_kv.live_bytes(),
        "f16 KV storage must be exactly half of f32"
    );
}

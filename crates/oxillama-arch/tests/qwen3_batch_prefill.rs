//! Batched prefill must be indistinguishable from the per-token loop.
//!
//! `Qwen3Model::forward` sends any multi-token call through the batched path in
//! `qwen3::batch`, which runs `PREFILL_TILE` tokens per pass over the weights.
//! That is only a safe default because the two paths agree **bit for bit**: the
//! CLI samples at temperature, so a one-ULP logit difference is a different
//! word on screen.
//!
//! The kernel half of that claim is pinned in `oxillama-quant`
//! (`gemv_parity::q4_k_batched_matmul_is_bit_identical_to_sequential` and its
//! Q6_K twin).  This file pins the *model* half — attention masking, the KV
//! staging/commit dance, RoPE positions, the residual stream and the tiling
//! seam — by running a whole synthetic Qwen3 checkpoint both ways.
//!
//! The fixture's projections are **Q4_K**, not F32, on purpose: the batched
//! route is gated on the weight kernel advertising a fused Q8_0 activation
//! path, and only the K-quants do.  An F32 fixture would make every assertion
//! below a tautology.

use oxillama_arch::qwen3::{load_qwen3_from_gguf, Qwen3Model};
use oxillama_arch::{ArchResult, ForwardPass, KvCacheAccess, ModelConfig};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

// Geometry chosen so nothing lines up by accident:
//   * HIDDEN is exactly one Q4_K block, FFN is two — both the single-block and
//     the multi-block row loops run.
//   * HEADS != KV_HEADS, so grouped-query head mapping is exercised.
//   * ATTN_DIM != HIDDEN, so a buffer sized from the wrong one would trip.
//   * Two layers, so per-layer KV staging cannot be confused with a single
//     shared buffer.
const HIDDEN: usize = 256;
const FFN: usize = 512;
const VOCAB: usize = 512;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const ATTN_DIM: usize = HEADS * HEAD_DIM;
const KV_DIM: usize = KV_HEADS * HEAD_DIM;
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

/// `n_rows x n_cols` of Q4_K.  Every byte pattern is a legal block, so random
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

/// Serialize a two-layer Qwen3 checkpoint whose projections are all Q4_K.
fn build_fixture() -> Vec<u8> {
    let mut rng = Rng::new(0x0BA7_C4ED);
    let mut writer = GgufWriter::new();

    for (key, value) in [
        (
            "general.architecture",
            MetadataValue::String("qwen3".to_string()),
        ),
        (
            "qwen3.embedding_length",
            MetadataValue::Uint32(HIDDEN as u32),
        ),
        (
            "qwen3.feed_forward_length",
            MetadataValue::Uint32(FFN as u32),
        ),
        ("qwen3.block_count", MetadataValue::Uint32(LAYERS as u32)),
        (
            "qwen3.attention.head_count",
            MetadataValue::Uint32(HEADS as u32),
        ),
        (
            "qwen3.attention.head_count_kv",
            MetadataValue::Uint32(KV_HEADS as u32),
        ),
        (
            "qwen3.attention.key_length",
            MetadataValue::Uint32(HEAD_DIM as u32),
        ),
        (
            "qwen3.attention.value_length",
            MetadataValue::Uint32(HEAD_DIM as u32),
        ),
        ("qwen3.context_length", MetadataValue::Uint32(CTX as u32)),
        ("qwen3.vocab_size", MetadataValue::Uint32(VOCAB as u32)),
        ("qwen3.rope.freq_base", MetadataValue::Float32(10000.0)),
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

    // Tied checkpoint: token_embd doubles as the LM head.
    add_q4k(&mut writer, "token_embd.weight", VOCAB, HIDDEN, &mut rng);

    for i in 0..LAYERS {
        let p = format!("blk.{i}");
        add_q4k(
            &mut writer,
            &format!("{p}.attn_q.weight"),
            ATTN_DIM,
            HIDDEN,
            &mut rng,
        );
        add_q4k(
            &mut writer,
            &format!("{p}.attn_k.weight"),
            KV_DIM,
            HIDDEN,
            &mut rng,
        );
        add_q4k(
            &mut writer,
            &format!("{p}.attn_v.weight"),
            KV_DIM,
            HIDDEN,
            &mut rng,
        );
        add_q4k(
            &mut writer,
            &format!("{p}.attn_output.weight"),
            HIDDEN,
            ATTN_DIM,
            &mut rng,
        );
        add_q4k(
            &mut writer,
            &format!("{p}.ffn_gate.weight"),
            FFN,
            HIDDEN,
            &mut rng,
        );
        add_q4k(
            &mut writer,
            &format!("{p}.ffn_up.weight"),
            FFN,
            HIDDEN,
            &mut rng,
        );
        add_q4k(
            &mut writer,
            &format!("{p}.ffn_down.weight"),
            HIDDEN,
            FFN,
            &mut rng,
        );

        for name in [
            format!("{p}.attn_norm.weight"),
            format!("{p}.ffn_norm.weight"),
        ] {
            let bytes = f32_vec_bytes(HIDDEN, &mut rng);
            writer.add_tensor(&name, &[HIDDEN as u64], GgufTensorType::F32, &bytes);
        }
        for name in [
            format!("{p}.attn_q_norm.weight"),
            format!("{p}.attn_k_norm.weight"),
        ] {
            let bytes = f32_vec_bytes(HEAD_DIM, &mut rng);
            writer.add_tensor(&name, &[HEAD_DIM as u64], GgufTensorType::F32, &bytes);
        }
    }

    let bytes = f32_vec_bytes(HIDDEN, &mut rng);
    writer.add_tensor(
        "output_norm.weight",
        &[HIDDEN as u64],
        GgufTensorType::F32,
        &bytes,
    );

    let mut out = Vec::new();
    writer
        .write_to(&mut out)
        .expect("synthetic Qwen3 GGUF must serialize");
    out
}

fn load_model() -> Qwen3Model {
    let gguf = GgufModel::from_bytes(build_fixture()).expect("fixture must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata");
    load_qwen3_from_gguf(&gguf, &config).expect("fixture must load")
}

/// Flat per-layer KV cache with the same visibility rules as the runtime's:
/// a key written at the current position is readable before `advance()`.
struct TestKv {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    seq_len: usize,
    stored_len: usize,
}

impl TestKv {
    fn new() -> Self {
        Self {
            keys: vec![vec![0.0; CTX * KV_DIM]; LAYERS],
            values: vec![vec![0.0; CTX * KV_DIM]; LAYERS],
            seq_len: 0,
            stored_len: 0,
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
        self.seq_len += 1;
        if self.stored_len < self.seq_len {
            self.stored_len = self.seq_len;
        }
    }
    fn kv_dim(&self) -> usize {
        KV_DIM
    }
}

/// A prompt long enough to span several [`PREFILL_TILE`]-sized tiles and end on
/// a ragged one (`40 = 2 * 16 + 8`), so the tiling seam is covered.
fn prompt_tokens(n: usize) -> Vec<u32> {
    let mut rng = Rng::new(0xC0FFEE);
    (0..n)
        .map(|_| (rng.next_u64() % VOCAB as u64) as u32)
        .collect()
}

fn assert_bit_identical(batched: &[f32], sequential: &[f32], what: &str) {
    assert_eq!(batched.len(), sequential.len(), "{what}: length differs");
    let mismatches = batched
        .iter()
        .zip(sequential)
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .take(4)
        .map(|(i, (a, b))| format!("[{i}] batched={a:e} sequential={b:e}"))
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "{what}: batched prefill diverged from the per-token loop at {} of {} slots; first: {}",
        batched
            .iter()
            .zip(sequential)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        batched.len(),
        mismatches.join(", ")
    );
}

/// The fixture must actually reach the batched path, or every other assertion
/// in this file compares the per-token loop with itself.
#[test]
fn q4_k_fixture_reaches_the_batched_path() {
    let model = load_model();
    let kernel = model
        .dispatcher
        .get_kernel(model.layers[0].attn_q.weight.tensor_type)
        .expect("Q4_K kernel must exist");
    if kernel.name() == "Q4_K_NEON" {
        assert!(
            model.layers[0].attn_q.q8_fused_blocks(&*kernel).is_some(),
            "the NEON Q4_K kernel must advertise the fused path, otherwise the \
             batched prefill tests below are vacuous"
        );
    }
}

/// One `forward` over the whole prompt == one `forward` per token.
#[test]
fn batched_prefill_logits_match_sequential() {
    let tokens = prompt_tokens(40);

    let mut batched_model = load_model();
    let mut batched_kv = TestKv::new();
    let batched = batched_model
        .forward(&tokens, &mut batched_kv)
        .expect("batched prefill");

    let mut seq_model = load_model();
    let mut seq_kv = TestKv::new();
    let mut sequential = Vec::new();
    for &token in &tokens {
        sequential = seq_model
            .forward(&[token], &mut seq_kv)
            .expect("per-token prefill");
    }

    assert_bit_identical(&batched, &sequential, "logits after a 40-token prompt");
}

/// The KV cache the batched path leaves behind must be the one the per-token
/// path would have written — otherwise the *next* token decodes differently
/// even though the prefill logits matched.
#[test]
fn batched_prefill_leaves_an_identical_kv_cache() {
    let tokens = prompt_tokens(40);

    let mut batched_model = load_model();
    let mut batched_kv = TestKv::new();
    batched_model
        .forward(&tokens, &mut batched_kv)
        .expect("batched prefill");

    let mut seq_model = load_model();
    let mut seq_kv = TestKv::new();
    for &token in &tokens {
        seq_model
            .forward(&[token], &mut seq_kv)
            .expect("per-token prefill");
    }

    assert_eq!(batched_kv.seq_len(), seq_kv.seq_len(), "sequence length");
    for layer in 0..LAYERS {
        assert_bit_identical(
            batched_kv.get_keys(layer).expect("keys"),
            seq_kv.get_keys(layer).expect("keys"),
            &format!("layer {layer} keys"),
        );
        assert_bit_identical(
            batched_kv.get_values(layer).expect("values"),
            seq_kv.get_values(layer).expect("values"),
            &format!("layer {layer} values"),
        );
    }

    // And the very next decode step must agree too.
    let next = [7u32];
    let batched_next = batched_model
        .forward(&next, &mut batched_kv)
        .expect("decode after batched prefill");
    let seq_next = seq_model
        .forward(&next, &mut seq_kv)
        .expect("decode after per-token prefill");
    assert_bit_identical(&batched_next, &seq_next, "first decode step");
}

/// Prompts that are shorter than, equal to and longer than one tile, plus a
/// prompt that continues a non-empty cache (the engine's chunked prefill).
#[test]
fn batched_prefill_matches_at_every_tile_boundary() {
    for &n in &[2usize, 3, 15, 16, 17, 32, 33] {
        let tokens = prompt_tokens(n);

        let mut batched_model = load_model();
        let mut batched_kv = TestKv::new();
        let batched = batched_model
            .forward(&tokens, &mut batched_kv)
            .expect("batched prefill");

        let mut seq_model = load_model();
        let mut seq_kv = TestKv::new();
        let mut sequential = Vec::new();
        for &token in &tokens {
            sequential = seq_model
                .forward(&[token], &mut seq_kv)
                .expect("per-token prefill");
        }

        assert_bit_identical(&batched, &sequential, &format!("{n}-token prompt"));
    }
}

/// A second multi-token `forward` on a warm cache — what the runtime does when
/// `prefill_chunk_size` splits a long prompt — must also match.
#[test]
fn chunked_batched_prefill_matches_sequential() {
    let tokens = prompt_tokens(40);

    let mut chunked_model = load_model();
    let mut chunked_kv = TestKv::new();
    let mut chunked = Vec::new();
    for chunk in tokens.chunks(13) {
        chunked = chunked_model
            .forward(chunk, &mut chunked_kv)
            .expect("chunked batched prefill");
    }

    let mut seq_model = load_model();
    let mut seq_kv = TestKv::new();
    let mut sequential = Vec::new();
    for &token in &tokens {
        sequential = seq_model
            .forward(&[token], &mut seq_kv)
            .expect("per-token prefill");
    }

    assert_bit_identical(&chunked, &sequential, "logits after a chunked prompt");
}

/// `embed` shares `run_layers` with `forward`, so it must batch identically.
#[test]
fn batched_embed_matches_sequential() {
    let tokens = prompt_tokens(20);

    let mut batched_model = load_model();
    let mut batched_kv = TestKv::new();
    let batched = batched_model
        .embed(&tokens, &mut batched_kv)
        .expect("batched embed");

    let mut seq_model = load_model();
    let mut seq_kv = TestKv::new();
    let mut sequential = Vec::new();
    for &token in &tokens {
        sequential = seq_model
            .embed(&[token], &mut seq_kv)
            .expect("per-token embed");
    }

    assert_bit_identical(&batched, &sequential, "embedding after a 20-token prompt");
}

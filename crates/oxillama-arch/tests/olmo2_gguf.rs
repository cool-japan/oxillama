//! OLMo2 GGUF loader round-trip plus regression tests for O1–O5.
//!
//! Everything here is checked against `~/work/refs/llama.cpp`:
//!
//! * `src/models/olmo2.cpp` (`llm_build_olmo2`) — graph order, whole-vector
//!   QK-norm before the head reshape, post-norm *before* each residual add,
//!   `1/sqrt(head_dim)` attention scale, SwiGLU FFN with no pre-FFN norm.
//! * `src/llama-model.cpp`, `case LLM_ARCH_OLMO2:` — tensor set and widths,
//!   notably `attn_q_norm {n_embd}` vs `attn_k_norm {n_head_kv * n_embd_head}`.
//! * `src/llama-arch.cpp` lines 355 / 359 — the post-norm tensor *names*.
//!
//! Two fixtures are used:
//!
//! * `oxillama_gguf::test_utils::build_minimal_olmo2_gguf()` — the shared
//!   fixture.  Its tensor payloads are all zeros, so it can prove that loading
//!   and shapes work but cannot discriminate between two normalization
//!   schemes.
//! * [`build_olmo2_gguf`] below — built directly with the production
//!   `GgufWriter` and filled with deterministic non-zero weights, so it *can*.

#![cfg(feature = "olmo2")]

use oxillama_arch::common::rope::{RopeStyle, RopeTable};
use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::ArchResult;
use oxillama_arch::olmo2::{load_olmo2_from_gguf, olmo2_tensor_name_patterns};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

// ─── Dimensions of the hand-built fixture ────────────────────────────────────

const H: usize = 16; // n_embd
const HEADS: usize = 4;
const KV_HEADS: usize = 2; // GQA
const HD: usize = H / HEADS; // head_dim = 4
const QW: usize = HEADS * HD; // 16 = n_embd
const KVW: usize = KV_HEADS * HD; // 8  = n_embd_gqa (deliberately != QW)
const FFN: usize = 24;
const VOCAB: usize = 6;
const LAYERS: usize = 2;
const CTX: usize = 12;
const EPS: f32 = 1e-5;
const ROPE_BASE: f32 = 10_000.0;

// ─── Test KV cache ───────────────────────────────────────────────────────────

/// Flat `[seq_len, kv_dim]` KV cache — the layout every production
/// `KvCacheAccess` in this workspace uses (`kv_cache/mod.rs`:
/// `offset = seq_len * kv_dim`).
///
/// `kv_dim` is `n_kv_heads * head_dim`, **not** `head_dim`: a cache sized the
/// MQA way would hide O5.
struct TestKv {
    kv_dim: usize,
    seq_len: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl TestKv {
    fn new(kv_dim: usize, num_layers: usize) -> Self {
        Self {
            kv_dim,
            seq_len: 0,
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
        }
    }
}

impl KvCacheAccess for TestKv {
    fn seq_len(&self) -> usize {
        self.seq_len
    }
    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let end = (self.seq_len + 1) * self.kv_dim;
        self.keys[layer].resize(end, 0.0);
        self.values[layer].resize(end, 0.0);
        let off = self.seq_len * self.kv_dim;
        self.keys[layer][off..end].copy_from_slice(&key[..self.kv_dim]);
        self.values[layer][off..end].copy_from_slice(&value[..self.kv_dim]);
        Ok(())
    }
    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.keys[layer])
    }
    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.values[layer])
    }
    fn advance(&mut self) {
        self.seq_len += 1;
    }
    fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

// ─── Deterministic weights ───────────────────────────────────────────────────

/// Deterministic pseudo-random values in roughly `[-0.35, 0.35]`.
///
/// A fixed seed per tensor keeps every run — and every fixture variant —
/// bit-identical, so the differential assertions below are stable.
fn weights(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((state >> 33) as u32 as f64) / (u32::MAX as f64);
            ((u - 0.5) * 0.7) as f32
        })
        .collect()
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Every tensor of the hand-built fixture, kept in f32 so the reference
/// implementation can use exactly the same numbers the GGUF carries.
#[derive(Clone)]
struct RefLayer {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn: Vec<f32>,
    post_ffn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

#[derive(Clone)]
struct RefModel {
    token_embd: Vec<f32>,
    layers: Vec<RefLayer>,
    output_norm: Vec<f32>,
    output: Vec<f32>,
}

fn build_ref_model() -> RefModel {
    let layers = (0..LAYERS)
        .map(|i| {
            let s = (i as u64 + 1) * 1000;
            RefLayer {
                q: weights(s + 1, QW * H),
                k: weights(s + 2, KVW * H),
                v: weights(s + 3, KVW * H),
                o: weights(s + 4, H * QW),
                // Norm scales are centred on 1.0 with a wide spread (roughly
                // [0.3, 1.7]): near-uniform scales would make a per-head norm
                // numerically indistinguishable from a whole-vector one.
                q_norm: weights(s + 5, QW).iter().map(|v| 1.0 + 2.0 * v).collect(),
                k_norm: weights(s + 6, KVW).iter().map(|v| 1.0 + 2.0 * v).collect(),
                post_attn: weights(s + 7, H).iter().map(|v| 1.0 + 2.0 * v).collect(),
                post_ffn: weights(s + 8, H).iter().map(|v| 1.0 + 2.0 * v).collect(),
                gate: weights(s + 9, FFN * H),
                up: weights(s + 10, FFN * H),
                down: weights(s + 11, H * FFN),
            }
        })
        .collect();

    RefModel {
        token_embd: weights(7, VOCAB * H),
        layers,
        output_norm: weights(8, H).iter().map(|v| 1.0 + 2.0 * v).collect(),
        output: weights(9, VOCAB * H),
    }
}

/// Serialize a [`RefModel`] as a GGUF v3 file using the production writer.
///
/// GGUF `ne` is fastest-changing-first, so a weight mapping
/// `in_features → out_features` is written `[in_features, out_features]` —
/// exactly what llama.cpp's `create_tensor(..., {n_embd, n_embd_gqa}, 0)` calls
/// mean.
fn build_olmo2_gguf(m: &RefModel) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("olmo2".to_string()),
    );
    w.add_metadata("olmo2.embedding_length", MetadataValue::Uint32(H as u32));
    w.add_metadata(
        "olmo2.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("olmo2.block_count", MetadataValue::Uint32(LAYERS as u32));
    w.add_metadata(
        "olmo2.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "olmo2.attention.head_count_kv",
        MetadataValue::Uint32(KV_HEADS as u32),
    );
    w.add_metadata("olmo2.context_length", MetadataValue::Uint32(CTX as u32));
    w.add_metadata("olmo2.vocab_size", MetadataValue::Uint32(VOCAB as u32));
    w.add_metadata(
        "olmo2.attention.layer_norm_rms_epsilon",
        MetadataValue::Float32(EPS),
    );
    w.add_metadata("olmo2.rope.freq_base", MetadataValue::Float32(ROPE_BASE));

    w.add_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &to_bytes(&m.token_embd),
    );

    for (i, l) in m.layers.iter().enumerate() {
        w.add_tensor(
            &format!("blk.{i}.attn_q.weight"),
            &[H as u64, QW as u64],
            GgufTensorType::F32,
            &to_bytes(&l.q),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_k.weight"),
            &[H as u64, KVW as u64],
            GgufTensorType::F32,
            &to_bytes(&l.k),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_v.weight"),
            &[H as u64, KVW as u64],
            GgufTensorType::F32,
            &to_bytes(&l.v),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_output.weight"),
            &[QW as u64, H as u64],
            GgufTensorType::F32,
            &to_bytes(&l.o),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_q_norm.weight"),
            &[QW as u64],
            GgufTensorType::F32,
            &to_bytes(&l.q_norm),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_k_norm.weight"),
            &[KVW as u64],
            GgufTensorType::F32,
            &to_bytes(&l.k_norm),
        );
        w.add_tensor(
            &format!("blk.{i}.post_attention_norm.weight"),
            &[H as u64],
            GgufTensorType::F32,
            &to_bytes(&l.post_attn),
        );
        w.add_tensor(
            &format!("blk.{i}.ffn_gate.weight"),
            &[H as u64, FFN as u64],
            GgufTensorType::F32,
            &to_bytes(&l.gate),
        );
        w.add_tensor(
            &format!("blk.{i}.ffn_up.weight"),
            &[H as u64, FFN as u64],
            GgufTensorType::F32,
            &to_bytes(&l.up),
        );
        w.add_tensor(
            &format!("blk.{i}.ffn_down.weight"),
            &[FFN as u64, H as u64],
            GgufTensorType::F32,
            &to_bytes(&l.down),
        );
        w.add_tensor(
            &format!("blk.{i}.post_ffw_norm.weight"),
            &[H as u64],
            GgufTensorType::F32,
            &to_bytes(&l.post_ffn),
        );
    }

    w.add_tensor(
        "output_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &to_bytes(&m.output_norm),
    );
    w.add_tensor(
        "output.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &to_bytes(&m.output),
    );

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic OLMo2 GGUF must serialize");
    bytes
}

// ─── Independent reference implementation of llm_build_olmo2 ─────────────────

fn gemv(w: &[f32], x: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    (0..out_features)
        .map(|r| {
            w[r * in_features..(r + 1) * in_features]
                .iter()
                .zip(x.iter())
                .map(|(&a, &b)| a * b)
                .sum()
        })
        .collect()
}

/// RMSNorm over the **whole** vector, exactly as `build_norm(..., LLM_NORM_RMS)`.
fn rms(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "RMSNorm weight must match the vector it normalizes"
    );
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v * inv * w)
        .collect()
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Run `llm_build_olmo2` by hand for `tokens`, returning the last token's
/// logits.
///
/// Deliberately written against the reference graph rather than against
/// `Olmo2Forward`: it keeps K/V as `[pos][kv_dim]` rows (so a wrong GQA stride
/// cannot be reproduced), normalizes Q/K as single whole-vector calls (so a
/// per-head norm cannot be reproduced), and processes every token (so a
/// last-token-only prefill cannot be reproduced).
fn reference_logits(m: &RefModel, tokens: &[u32]) -> Vec<f32> {
    let rope = RopeTable::new_standard_with_style(HD, CTX, ROPE_BASE, RopeStyle::Neox);
    let gqa = HEADS / KV_HEADS;

    let mut k_cache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); LAYERS];
    let mut v_cache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); LAYERS];
    let mut last_hidden = vec![0.0f32; H];

    for (pos, &tok) in tokens.iter().enumerate() {
        let mut x = m.token_embd[tok as usize * H..(tok as usize + 1) * H].to_vec();

        for (li, l) in m.layers.iter().enumerate() {
            let inp_sa = x.clone();

            let mut q = gemv(&l.q, &x, QW, H);
            let mut k = gemv(&l.k, &x, KVW, H);
            let v = gemv(&l.v, &x, KVW, H);

            // ONE norm over the full vector, BEFORE the head reshape.
            q = rms(&q, &l.q_norm, EPS);
            k = rms(&k, &l.k_norm, EPS);

            for h in 0..HEADS {
                rope.apply(&mut q[h * HD..(h + 1) * HD], pos);
            }
            for h in 0..KV_HEADS {
                rope.apply(&mut k[h * HD..(h + 1) * HD], pos);
            }

            k_cache[li].push(k);
            v_cache[li].push(v);

            let mut attn = vec![0.0f32; QW];
            let scale = 1.0 / (HD as f32).sqrt();
            for h in 0..HEADS {
                let kh = h / gqa;
                let qh = &q[h * HD..(h + 1) * HD];
                let mut scores: Vec<f32> = (0..=pos)
                    .map(|p| {
                        let kv = &k_cache[li][p][kh * HD..(kh + 1) * HD];
                        qh.iter().zip(kv.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                softmax(&mut scores);
                for (p, &sw) in scores.iter().enumerate() {
                    let vv = &v_cache[li][p][kh * HD..(kh + 1) * HD];
                    for d in 0..HD {
                        attn[h * HD + d] += sw * vv[d];
                    }
                }
            }

            // post-norm BEFORE the residual add
            let attn_out = rms(&gemv(&l.o, &attn, H, QW), &l.post_attn, EPS);
            let ffn_inp: Vec<f32> = inp_sa
                .iter()
                .zip(attn_out.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            // No pre-FFN norm: the FFN reads ffn_inp directly.
            let mut g = gemv(&l.gate, &ffn_inp, FFN, H);
            let u = gemv(&l.up, &ffn_inp, FFN, H);
            for (gi, &ui) in g.iter_mut().zip(u.iter()) {
                *gi = *gi / (1.0 + (-*gi).exp()) * ui;
            }
            let ffn_out = rms(&gemv(&l.down, &g, H, FFN), &l.post_ffn, EPS);

            x = ffn_inp
                .iter()
                .zip(ffn_out.iter())
                .map(|(&a, &b)| a + b)
                .collect();
        }
        last_hidden = x;
    }

    let normed = rms(&last_hidden, &m.output_norm, EPS);
    gemv(&m.output, &normed, VOCAB, H)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn parse(bytes: Vec<u8>) -> (GgufModel, ModelConfig) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
    let config =
        ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
    (gguf, config)
}

fn logits_for(m: &RefModel, tokens: &[u32]) -> Vec<f32> {
    let (gguf, config) = parse(build_olmo2_gguf(m));
    let mut model = load_olmo2_from_gguf(&gguf, &config).expect("hand-built fixture must load");
    let mut cache = TestKv::new(KVW, LAYERS);
    model.forward(tokens, &mut cache).expect("forward")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Round-trip — NEW CAPABILITY
// ═════════════════════════════════════════════════════════════════════════════

/// **New capability.**  Before this change OLMo2 had no GGUF loader at all —
/// only `Olmo2Forward::new(config, ...)` with pre-dequantized components — so
/// `Olmo2Architecture::build()` returned `MissingTensor` and no OLMo2
/// checkpoint could be run.  This is the first test that loads one.
#[test]
fn o0_round_trip_shared_fixture_loads_and_runs() {
    let (gguf, config) = parse(oxillama_gguf::test_utils::build_minimal_olmo2_gguf());
    assert_eq!(config.architecture, "olmo2");

    let mut model = load_olmo2_from_gguf(&gguf, &config).expect("shared OLMo2 fixture must load");

    // hidden = 32, heads = 4 → head_dim = 8, kv_heads = 2 → kv_dim = 16.
    let mut cache = TestKv::new(16, 1);
    let logits = model
        .forward(&[0u32, 1, 2], &mut cache)
        .expect("forward must succeed");

    assert_eq!(logits.len(), 32, "logits must be vocab_size long");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "every logit must be finite"
    );
    assert_eq!(model.vocab_size(), 32);
    assert_eq!(model.hidden_size(), 32);
    assert_eq!(model.max_context_length(), 128);
}

/// The hand-built (non-zero) fixture loads and matches an independent
/// transcription of `llm_build_olmo2`.
///
/// This single assertion covers O1 (all tokens processed, one KV advance
/// each), O2 (whole-vector QK-norm at the correct Q/K widths) and O5 (GQA KV
/// stride `pos * kv_dim + kv_head * head_dim`) simultaneously: the reference
/// cannot express any of the three defects.
#[test]
fn o1_o2_o5_matches_reference_graph() {
    let m = build_ref_model();
    let tokens = [3u32, 1, 4, 0, 5];

    let got = logits_for(&m, &tokens);
    let want = reference_logits(&m, &tokens);

    assert_eq!(got.len(), VOCAB);
    let diff = max_abs_diff(&got, &want);
    // Measured separation on this fixture (each defect was reintroduced and the
    // diff recorded): correct = 2.4e-7 (f32 summation noise), per-head QK-norm
    // = 6.0e-5, `head_dim`-strided GQA KV cache = 1.9e-3.  1e-5 sits between the
    // noise floor and the smallest real defect with ~40x headroom either way.
    assert!(
        diff < 1e-5,
        "loader/forward must match the llama.cpp graph; max|Δ| = {diff}\n got: {got:?}\nwant: {want:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. O1 — every prompt token is processed
// ═════════════════════════════════════════════════════════════════════════════

/// O1: `run_layers` used to do
/// `let token = tokens[tokens.len() - 1]; let position = kv_cache.seq_len();`
/// with a single `kv_cache.advance()` outside any loop, so a 3-token prompt
/// advanced the cache by 1 and the first two tokens never entered the KV cache.
#[test]
fn o1_multi_token_prefill_advances_cache_once_per_token() {
    let m = build_ref_model();
    let (gguf, config) = parse(build_olmo2_gguf(&m));
    let mut model = load_olmo2_from_gguf(&gguf, &config).expect("load");
    let mut cache = TestKv::new(KVW, LAYERS);

    model.forward(&[1u32, 2, 3], &mut cache).expect("forward");
    assert_eq!(
        cache.seq_len(),
        3,
        "a 3-token prompt must advance the KV cache by exactly 3"
    );
    // And the cache must actually hold three rows per layer.
    for layer in 0..LAYERS {
        assert_eq!(
            cache.get_keys(layer).expect("keys").len(),
            3 * KVW,
            "layer {layer} must hold 3 key rows of {KVW} elements"
        );
    }
}

/// `embed()` routes through the same per-token path.
#[test]
fn o1_embed_also_processes_every_token() {
    let m = build_ref_model();
    let (gguf, config) = parse(build_olmo2_gguf(&m));
    let mut model = load_olmo2_from_gguf(&gguf, &config).expect("load");
    let mut cache = TestKv::new(KVW, LAYERS);

    let hidden = model.embed(&[1u32, 2, 3, 4], &mut cache).expect("embed");
    assert_eq!(hidden.len(), H, "embed() returns hidden_size, not vocab");
    assert_eq!(cache.seq_len(), 4);
}

/// Prefilling 2 tokens then decoding 1 must equal prefilling all 3 at once.
///
/// This is the property the single-token shortcut destroyed: with the old code
/// the "prefill" leg cached nothing but the last token, so the two paths
/// disagreed.
#[test]
fn o1_incremental_decode_matches_single_shot_prefill() {
    let m = build_ref_model();
    let tokens = [2u32, 5, 1];

    let one_shot = logits_for(&m, &tokens);

    let (gguf, config) = parse(build_olmo2_gguf(&m));
    let mut model = load_olmo2_from_gguf(&gguf, &config).expect("load");
    let mut cache = TestKv::new(KVW, LAYERS);
    model.forward(&tokens[..2], &mut cache).expect("prefill");
    let incremental = model.forward(&tokens[2..], &mut cache).expect("decode");

    let diff = max_abs_diff(&one_shot, &incremental);
    assert!(
        diff < 1e-5,
        "prefill+decode must equal a single prefill; max|Δ| = {diff}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. O2 — QK-norm is ONE whole-vector RMSNorm at the correct widths
// ═════════════════════════════════════════════════════════════════════════════

/// The loader reads `attn_k_norm` at its own declared width.
///
/// The shared fixture ships `blk.0.attn_k_norm.weight` with 16 elements
/// (`n_head_kv(2) * head_dim(8)`) while `attn_q_norm` has 32 (`n_embd`).  A
/// loader that assumed both were `n_embd` would either truncate or read past
/// the tensor.
#[test]
fn o2_k_norm_is_loaded_at_its_true_gqa_width() {
    let (gguf, config) = parse(oxillama_gguf::test_utils::build_minimal_olmo2_gguf());
    let model = load_olmo2_from_gguf(&gguf, &config).expect("load");

    assert_eq!(
        model.layers[0].attn_q_norm.weight.len(),
        32,
        "attn_q_norm is n_embd wide"
    );
    assert_eq!(
        model.layers[0].attn_k_norm.weight.len(),
        16,
        "attn_k_norm is n_head_kv * head_dim wide, NOT n_embd"
    );
}

/// O2: Q-norm weights beyond the first head must affect the output.
///
/// Two fixtures differ **only** in `attn_q_norm[HD..QW]` — the entries that a
/// per-head implementation would never read, because `RmsNorm::forward`
/// indexes `self.weight[i]` relative to the slice it is handed and every head
/// is handed `weight[0..head_dim]`.  Under the old per-head code both variants
/// produce identical logits; under whole-vector normalization they must not.
#[test]
fn o2_q_norm_tail_changes_the_output() {
    let base = build_ref_model();
    let mut variant = base.clone();
    for (i, w) in variant.layers[0].q_norm.iter_mut().enumerate() {
        if i >= HD {
            *w += 1.0;
        }
    }
    // The first head's slice is byte-identical in both.
    assert_eq!(
        base.layers[0].q_norm[..HD],
        variant.layers[0].q_norm[..HD],
        "the two fixtures must agree on head 0's weight slice"
    );

    let tokens = [1u32, 4, 2];
    let a = logits_for(&base, &tokens);
    let b = logits_for(&variant, &tokens);
    let diff = max_abs_diff(&a, &b);
    // The buggy per-head path never reads `weight[head_dim..]`, so its two
    // variants are bit-identical and this diff is EXACTLY 0.0 (measured).  Any
    // threshold above zero therefore separates the two implementations; 1e-6
    // is far below the ~2e-4 the correct implementation produces here.
    assert!(
        diff > 1e-6,
        "attn_q_norm entries past head 0 must be used; max|Δ| = {diff} \
         (a per-head norm reuses weight[0..head_dim] for every head)"
    );
}

/// O2: the same for K-norm, over the narrower `n_embd_gqa` buffer.
#[test]
fn o2_k_norm_tail_changes_the_output() {
    let base = build_ref_model();
    let mut variant = base.clone();
    for (i, w) in variant.layers[0].k_norm.iter_mut().enumerate() {
        if i >= HD {
            *w += 1.0;
        }
    }
    assert_eq!(base.layers[0].k_norm[..HD], variant.layers[0].k_norm[..HD]);

    // At least two positions are needed: with a single cached key the softmax
    // is 1.0 regardless of the scores, so K would not influence the output.
    let tokens = [1u32, 4, 2];
    let a = logits_for(&base, &tokens);
    let b = logits_for(&variant, &tokens);
    let diff = max_abs_diff(&a, &b);
    // Exactly 0.0 under the per-head implementation — see the Q-norm test.
    assert!(
        diff > 1e-6,
        "attn_k_norm entries past kv head 0 must be used; max|Δ| = {diff}"
    );
}

/// A checkpoint whose `attn_k_norm` is `n_embd` wide (the shape the buggy
/// implementation implicitly assumed) is rejected with a named error instead
/// of panicking inside `RmsNorm::forward`.
#[test]
fn o2_mis_sized_k_norm_is_a_named_error() {
    let m = build_ref_model();

    // Control: the correctly-sized fixture loads.
    let (gguf, config) = parse(build_olmo2_gguf_with_k_norm_width(&m, KVW));
    assert!(load_olmo2_from_gguf(&gguf, &config).is_ok());

    // An `n_embd`-wide attn_k_norm — the shape the per-head implementation
    // implicitly assumed — must be rejected by name.
    let (gguf, config) = parse(build_olmo2_gguf_with_k_norm_width(&m, QW));
    let err = load_olmo2_from_gguf(&gguf, &config);
    assert!(
        matches!(
            err,
            Err(oxillama_arch::error::ArchError::InvalidShape { .. })
        ),
        "an n_embd-wide attn_k_norm must be InvalidShape, got {:?}",
        err.err()
    );
}

/// Variant of [`build_olmo2_gguf`] that writes `blk.0.attn_k_norm` at an
/// arbitrary declared width.
fn build_olmo2_gguf_with_k_norm_width(m: &RefModel, width: usize) -> Vec<u8> {
    let fixed = m.clone();
    // `build_olmo2_gguf` writes `k_norm` with a hard-coded `KVW` dimension, so
    // build the file here with the requested width instead.
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("olmo2".to_string()),
    );
    w.add_metadata("olmo2.embedding_length", MetadataValue::Uint32(H as u32));
    w.add_metadata(
        "olmo2.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("olmo2.block_count", MetadataValue::Uint32(LAYERS as u32));
    w.add_metadata(
        "olmo2.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "olmo2.attention.head_count_kv",
        MetadataValue::Uint32(KV_HEADS as u32),
    );
    w.add_metadata("olmo2.context_length", MetadataValue::Uint32(CTX as u32));
    w.add_metadata("olmo2.vocab_size", MetadataValue::Uint32(VOCAB as u32));
    w.add_metadata(
        "olmo2.attention.layer_norm_rms_epsilon",
        MetadataValue::Float32(EPS),
    );
    w.add_metadata("olmo2.rope.freq_base", MetadataValue::Float32(ROPE_BASE));
    w.add_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &to_bytes(&fixed.token_embd),
    );
    for (i, l) in fixed.layers.iter().enumerate() {
        let k_norm_width = if i == 0 { width } else { KVW };
        w.add_tensor(
            &format!("blk.{i}.attn_q.weight"),
            &[H as u64, QW as u64],
            GgufTensorType::F32,
            &to_bytes(&l.q),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_k.weight"),
            &[H as u64, KVW as u64],
            GgufTensorType::F32,
            &to_bytes(&l.k),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_v.weight"),
            &[H as u64, KVW as u64],
            GgufTensorType::F32,
            &to_bytes(&l.v),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_output.weight"),
            &[QW as u64, H as u64],
            GgufTensorType::F32,
            &to_bytes(&l.o),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_q_norm.weight"),
            &[QW as u64],
            GgufTensorType::F32,
            &to_bytes(&l.q_norm),
        );
        w.add_tensor(
            &format!("blk.{i}.attn_k_norm.weight"),
            &[k_norm_width as u64],
            GgufTensorType::F32,
            &to_bytes(&vec![1.0f32; k_norm_width]),
        );
        w.add_tensor(
            &format!("blk.{i}.post_attention_norm.weight"),
            &[H as u64],
            GgufTensorType::F32,
            &to_bytes(&l.post_attn),
        );
        w.add_tensor(
            &format!("blk.{i}.ffn_gate.weight"),
            &[H as u64, FFN as u64],
            GgufTensorType::F32,
            &to_bytes(&l.gate),
        );
        w.add_tensor(
            &format!("blk.{i}.ffn_up.weight"),
            &[H as u64, FFN as u64],
            GgufTensorType::F32,
            &to_bytes(&l.up),
        );
        w.add_tensor(
            &format!("blk.{i}.ffn_down.weight"),
            &[FFN as u64, H as u64],
            GgufTensorType::F32,
            &to_bytes(&l.down),
        );
        w.add_tensor(
            &format!("blk.{i}.post_ffw_norm.weight"),
            &[H as u64],
            GgufTensorType::F32,
            &to_bytes(&l.post_ffn),
        );
    }
    w.add_tensor(
        "output_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &to_bytes(&fixed.output_norm),
    );
    w.add_tensor(
        "output.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &to_bytes(&fixed.output),
    );
    let mut bytes = Vec::new();
    w.write_to(&mut bytes).expect("must serialize");
    bytes
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. O4 — post-norm tensor names
// ═════════════════════════════════════════════════════════════════════════════

/// O4: the declared post-norm names are llama.cpp's, not the invented ones.
///
/// This test FAILS against the pre-fix `tensor_names.rs`, which declared
/// `blk.{i}.attn_post_norm.weight` / `blk.{i}.ffn_post_norm.weight`.  The
/// shared `LLM_TENSOR_NAMES` table in `src/llama-arch.cpp` resolves
/// `LLM_TENSOR_ATTN_POST_NORM` to `"blk.%d.post_attention_norm"` (line 355) and
/// `LLM_TENSOR_FFN_POST_NORM` to `"blk.%d.post_ffw_norm"` (line 359);
/// `gguf-py/gguf/constants.py` lines 964 / 969 agree, and
/// `tensor_mapping.py` marks both entries `# gemma2 olmo2`.
#[test]
fn o4_post_norm_patterns_use_the_real_gguf_names() {
    let patterns: Vec<String> = olmo2_tensor_name_patterns()
        .into_iter()
        .map(|p| p.pattern)
        .collect();

    assert!(
        patterns.contains(&"blk.{i}.post_attention_norm.weight".to_string()),
        "missing blk.{{i}}.post_attention_norm.weight; got {patterns:?}"
    );
    assert!(
        patterns.contains(&"blk.{i}.post_ffw_norm.weight".to_string()),
        "missing blk.{{i}}.post_ffw_norm.weight; got {patterns:?}"
    );
    assert!(
        !patterns.contains(&"blk.{i}.attn_post_norm.weight".to_string()),
        "the pre-fix name 'attn_post_norm' must be gone"
    );
    assert!(
        !patterns.contains(&"blk.{i}.ffn_post_norm.weight".to_string()),
        "the pre-fix name 'ffn_post_norm' must be gone"
    );
}

/// O4, end to end: a checkpoint carrying the *old* names fails to load.
///
/// If the loader still looked for `attn_post_norm` / `ffn_post_norm`, this
/// checkpoint (which uses the real names) would be the one that failed.
#[test]
fn o4_loader_reads_the_real_post_norm_tensors() {
    let m = build_ref_model();
    let (gguf, config) = parse(build_olmo2_gguf(&m));
    assert!(gguf
        .file
        .tensors
        .contains("blk.0.post_attention_norm.weight"));
    assert!(gguf.file.tensors.contains("blk.0.post_ffw_norm.weight"));
    assert!(!gguf.file.tensors.contains("blk.0.attn_post_norm.weight"));
    assert!(load_olmo2_from_gguf(&gguf, &config).is_ok());
}

/// Removing `blk.0.post_ffw_norm.weight` makes the load fail with
/// `MissingTensor` — the tensor is required (`flags = 0` in llama.cpp), so it
/// must not be silently defaulted to identity.
#[test]
fn o4_missing_post_ffw_norm_is_missing_tensor() {
    let m = build_ref_model();
    let bytes = build_olmo2_gguf(&m);
    let (gguf, config) = parse(bytes);
    // Load once to prove the fixture is otherwise good.
    assert!(load_olmo2_from_gguf(&gguf, &config).is_ok());

    // Now build one without the post-FFN norm on layer 0.
    let mut w = GgufWriter::new();
    for (key, value) in gguf.file.metadata.iter() {
        w.add_metadata(key, value.clone());
    }
    for name in gguf.file.tensors.names() {
        if name == "blk.0.post_ffw_norm.weight" {
            continue;
        }
        let info = gguf.file.tensors.get(name).expect("tensor info");
        let data = gguf.tensor_data(name).expect("tensor data");
        w.add_tensor(name, &info.dimensions, info.tensor_type, data);
    }
    let mut bytes = Vec::new();
    w.write_to(&mut bytes).expect("serialize");

    let (gguf, config) = parse(bytes);
    let err = load_olmo2_from_gguf(&gguf, &config);
    assert!(
        matches!(
            err,
            Err(oxillama_arch::error::ArchError::MissingTensor { .. })
        ),
        "a missing post_ffw_norm must be MissingTensor, got {:?}",
        err.err()
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 4b. LM head: declared shape and the documented tied-embedding fallback
// ═════════════════════════════════════════════════════════════════════════════

/// Rebuild `bytes` while dropping `drop` and overriding the declared dimensions
/// of `retype` (name → dims), so a single tensor can be corrupted in isolation.
fn rewrite_gguf(bytes: Vec<u8>, drop: &[&str], retype: &[(&str, Vec<u64>)]) -> Vec<u8> {
    let gguf = GgufModel::from_bytes(bytes).expect("parse");
    let mut w = GgufWriter::new();
    for (key, value) in gguf.file.metadata.iter() {
        w.add_metadata(key, value.clone());
    }
    for name in gguf.file.tensors.names() {
        if drop.contains(&name.as_str()) {
            continue;
        }
        let info = gguf.file.tensors.get(name).expect("tensor info");
        let data = gguf.tensor_data(name).expect("tensor data");
        let dims = retype
            .iter()
            .find(|(n, _)| *n == name.as_str())
            .map(|(_, d)| d.clone())
            .unwrap_or_else(|| info.dimensions.clone());
        w.add_tensor(name, &dims, info.tensor_type, data);
    }
    let mut out = Vec::new();
    w.write_to(&mut out).expect("serialize");
    out
}

/// An `output.weight` with more rows than `vocab_size` is rejected.
///
/// `QuantLinear::forward` sizes the GEMV from the tensor's own shape, so an
/// over-wide LM head would write past the `vocab_size`-long logits buffer.
#[test]
fn lm_head_row_count_is_validated() {
    let m = build_ref_model();
    // Declare `[H, VOCAB]` as `[VOCAB, H]`: same element count, transposed
    // meaning — out_features becomes H (16) instead of VOCAB (6).
    let bad = rewrite_gguf(
        build_olmo2_gguf(&m),
        &[],
        &[("output.weight", vec![VOCAB as u64, H as u64])],
    );
    let (gguf, config) = parse(bad);
    let err = load_olmo2_from_gguf(&gguf, &config);
    assert!(
        matches!(
            err,
            Err(oxillama_arch::error::ArchError::InvalidShape { .. })
        ),
        "a mis-shaped output.weight must be InvalidShape, got {:?}",
        err.err()
    );
}

/// The documented tied-embedding fallback: a checkpoint with no
/// `output.weight` loads by reusing `token_embd.weight`.
///
/// llama.cpp creates OLMo2's `output.weight` with `flags = 0` (required), so no
/// conforming checkpoint needs this — but `load_lm_head` makes it a safe
/// superset, and `token_embd.weight` is `{n_embd, n_vocab}`, i.e. exactly the
/// shape the LM-head check expects.
#[test]
fn lm_head_falls_back_to_tied_token_embeddings() {
    let m = build_ref_model();
    let tied = rewrite_gguf(build_olmo2_gguf(&m), &["output.weight"], &[]);
    let (gguf, config) = parse(tied);
    assert!(!gguf.file.tensors.contains("output.weight"));

    let mut model =
        load_olmo2_from_gguf(&gguf, &config).expect("a tied checkpoint must load via token_embd");
    let mut cache = TestKv::new(KVW, LAYERS);
    let logits = model.forward(&[1u32, 2, 3], &mut cache).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Part-3 shared items: OOV ids and context overflow
// ═════════════════════════════════════════════════════════════════════════════

/// An out-of-vocabulary token id is an error, not a panic.
#[test]
fn p3_oov_token_id_errors() {
    let (gguf, config) = parse(oxillama_gguf::test_utils::build_minimal_olmo2_gguf());
    let mut model = load_olmo2_from_gguf(&gguf, &config).expect("load");
    let mut cache = TestKv::new(16, 1);

    let err = model.forward(&[32u32], &mut cache); // vocab_size == 32
    assert!(err.is_err(), "token id 32 is out of a 32-token vocabulary");
    let err = model.forward(&[0u32, 1, 9_999], &mut cache);
    assert!(err.is_err(), "an OOV id anywhere in the prompt must error");
}

/// A prompt longer than the model's context is an error, not an out-of-bounds
/// write into `buf_attn_scores`.
///
/// This path is reachable straight from the HTTP server's prompt field.
#[test]
fn p3_context_overflow_errors() {
    let m = build_ref_model();
    let (gguf, config) = parse(build_olmo2_gguf(&m));
    assert_eq!(config.max_context_length, CTX);
    let mut model = load_olmo2_from_gguf(&gguf, &config).expect("load");

    let mut cache = TestKv::new(KVW, LAYERS);
    let too_many: Vec<u32> = (0..CTX + 1).map(|i| (i % VOCAB) as u32).collect();
    assert!(
        model.forward(&too_many, &mut cache).is_err(),
        "a prompt of {} tokens must not fit a {CTX}-token context",
        CTX + 1
    );

    // Exactly at the limit is fine, and one more token afterwards is not.
    let mut cache = TestKv::new(KVW, LAYERS);
    let exact: Vec<u32> = (0..CTX).map(|i| (i % VOCAB) as u32).collect();
    assert!(model.forward(&exact, &mut cache).is_ok());
    assert!(
        model.forward(&[0u32], &mut cache).is_err(),
        "decoding past the context bound must error"
    );
}

/// An empty prompt is an error rather than an index panic.
#[test]
fn p3_empty_prompt_errors() {
    let m = build_ref_model();
    let (gguf, config) = parse(build_olmo2_gguf(&m));
    let mut model = load_olmo2_from_gguf(&gguf, &config).expect("load");
    let mut cache = TestKv::new(KVW, LAYERS);
    assert!(model.forward(&[], &mut cache).is_err());
}

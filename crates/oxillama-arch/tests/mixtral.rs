//! Mixtral is a real transformer block now, not a placeholder.
//!
//! `mixtral::MixtralModel` used to contain, verbatim:
//!
//! ```text
//! // For this reference implementation, Q = K = V = norm(hidden).
//! self.buf_q[i] = norm_slice[i % norm_slice.len()];
//! ```
//!
//! `MixtralLayer` held only two RMSNorms and an FFN — no attention weights at
//! all — `MixtralModel` had no `RopeTable`, and there was no
//! `load_mixtral_from_gguf`.  The type was `pub`, implemented `ForwardPass`,
//! returned finite logits and never raised an error, so it *ran* and meant
//! nothing.  None of the tests below could even be written against it: the
//! fields they read did not exist and the loader they call did not exist.
//!
//! The experts are stacked `Q4_K` tensors, as a real checkpoint's are, so the
//! zero-copy per-expert slicing is exercised rather than bypassed.

use oxillama_arch::llama::load_llama_from_gguf;
use oxillama_arch::mixtral::{load_mixtral_from_gguf, MixtralModel};
use oxillama_arch::{ArchResult, ForwardPass, KvCacheAccess, ModelConfig};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

const HIDDEN: usize = 256;
const INTER: usize = 256;
const VOCAB: usize = 512;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const ATTN_DIM: usize = HEADS * HEAD_DIM;
const KV_DIM: usize = KV_HEADS * HEAD_DIM;
const LAYERS: usize = 1;
const CTX: usize = 32;
const EXPERTS: usize = 4;
const EXPERTS_USED: usize = 2;

const Q4_K_BYTES: usize = 144;
const Q4_K_BLOCK: usize = 256;

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
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 8_388_608.0 - 1.0
    }
}

fn q4_k_matrix(n_rows: usize, n_cols: usize, rng: &mut Rng) -> Vec<u8> {
    let blocks = n_rows * n_cols.div_ceil(Q4_K_BLOCK);
    let mut data = Vec::with_capacity(blocks * Q4_K_BYTES);
    for _ in 0..blocks {
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

fn f32_vec_bytes(n: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        out.extend_from_slice(&(1.0 + 0.1 * rng.next_f32()).to_le_bytes());
    }
    out
}

/// Build a one-layer MoE checkpoint under `arch` as the declared architecture.
///
/// `v_seed` perturbs only `attn_v.weight`, which is how
/// [`attention_actually_uses_the_value_projection`] isolates that tensor.
fn build_fixture(arch: &str, v_seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(0x1CE_C0FFEE);
    let mut writer = GgufWriter::new();

    for (key, value) in [
        (
            "general.architecture".to_string(),
            MetadataValue::String(arch.to_string()),
        ),
        (
            format!("{arch}.embedding_length"),
            MetadataValue::Uint32(HIDDEN as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            MetadataValue::Uint32(INTER as u32),
        ),
        (
            format!("{arch}.block_count"),
            MetadataValue::Uint32(LAYERS as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            MetadataValue::Uint32(HEADS as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            MetadataValue::Uint32(KV_HEADS as u32),
        ),
        (
            format!("{arch}.attention.key_length"),
            MetadataValue::Uint32(HEAD_DIM as u32),
        ),
        (
            format!("{arch}.attention.value_length"),
            MetadataValue::Uint32(HEAD_DIM as u32),
        ),
        (
            format!("{arch}.context_length"),
            MetadataValue::Uint32(CTX as u32),
        ),
        (
            format!("{arch}.vocab_size"),
            MetadataValue::Uint32(VOCAB as u32),
        ),
        (
            format!("{arch}.expert_count"),
            MetadataValue::Uint32(EXPERTS as u32),
        ),
        (
            format!("{arch}.expert_used_count"),
            MetadataValue::Uint32(EXPERTS_USED as u32),
        ),
        (
            format!("{arch}.rope.freq_base"),
            MetadataValue::Float32(10000.0),
        ),
    ] {
        writer.add_metadata(&key, value);
    }

    // GGUF writes `ne` fastest-changing-first: [in_features, out_features].
    let add_q4k = |w: &mut GgufWriter, name: &str, out_f: usize, in_f: usize, rng: &mut Rng| {
        let data = q4_k_matrix(out_f, in_f, rng);
        w.add_tensor(
            name,
            &[in_f as u64, out_f as u64],
            GgufTensorType::Q4K,
            &data,
        );
    };

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
        // `attn_v` gets its own generator so a caller can vary it alone.
        let mut v_rng = Rng::new(v_seed);
        add_q4k(
            &mut writer,
            &format!("{p}.attn_v.weight"),
            KV_DIM,
            HIDDEN,
            &mut v_rng,
        );
        add_q4k(
            &mut writer,
            &format!("{p}.attn_output.weight"),
            HIDDEN,
            ATTN_DIM,
            &mut rng,
        );

        // MoE: router plus three stacked `[n_expert, out, in]` expert tensors.
        let router = f32_vec_bytes(EXPERTS * HIDDEN, &mut rng);
        writer.add_tensor(
            &format!("{p}.ffn_gate_inp.weight"),
            &[HIDDEN as u64, EXPERTS as u64],
            GgufTensorType::F32,
            &router,
        );
        for (name, out_f, in_f) in [
            (format!("{p}.ffn_gate_exps.weight"), INTER, HIDDEN),
            (format!("{p}.ffn_up_exps.weight"), INTER, HIDDEN),
            (format!("{p}.ffn_down_exps.weight"), HIDDEN, INTER),
        ] {
            let data = q4_k_matrix(out_f * EXPERTS, in_f, &mut rng);
            writer.add_tensor(
                &name,
                &[in_f as u64, out_f as u64, EXPERTS as u64],
                GgufTensorType::Q4K,
                &data,
            );
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
        .expect("synthetic Mixtral GGUF must serialize");
    out
}

fn load_mixtral(v_seed: u64) -> MixtralModel {
    let gguf = GgufModel::from_bytes(build_fixture("mixtral", v_seed)).expect("fixture must parse");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata");
    load_mixtral_from_gguf(&gguf, &config).expect("fixture must load")
}

/// Flat per-layer KV cache; records how many times `advance()` was called.
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

    fn primed(n: usize) -> Self {
        let mut kv = Self::new();
        kv.seq_len = n;
        kv.stored_len = n;
        kv
    }

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

// ── The block has real weights ───────────────────────────────────────────────

/// `MixtralLayer` carries the four attention projections a transformer needs.
///
/// The placeholder's `MixtralLayer` was `{ attn_norm, ffn_norm, moe_ffn }` —
/// this test could not be compiled against it.
#[test]
fn layer_has_the_four_attention_projections() {
    let model = load_mixtral(1);
    let layer = &model.layers[0];

    assert_eq!(layer.attn_q.out_features, ATTN_DIM);
    assert_eq!(layer.attn_q.in_features, HIDDEN);
    assert_eq!(layer.attn_k.out_features, KV_DIM);
    assert_eq!(layer.attn_v.out_features, KV_DIM);
    assert_eq!(layer.attn_output.in_features, ATTN_DIM);
    assert_eq!(layer.attn_output.out_features, HIDDEN);
    assert_eq!(model.rope.half_dim, HEAD_DIM / 2);
}

/// The value projection actually participates.
///
/// Two checkpoints identical except for `attn_v.weight` must produce different
/// logits.  The placeholder set `V = norm(hidden)` regardless of any weight, so
/// its output was invariant under this change.
#[test]
fn attention_actually_uses_the_value_projection() {
    let mut a = load_mixtral(0xAAAA);
    let mut b = load_mixtral(0xBBBB);

    assert_ne!(
        a.layers[0].attn_v.weight.data.as_slice(),
        b.layers[0].attn_v.weight.data.as_slice(),
        "the two fixtures must differ in attn_v"
    );
    assert_eq!(
        a.layers[0].attn_q.weight.data.as_slice(),
        b.layers[0].attn_q.weight.data.as_slice(),
        "and must agree everywhere else"
    );

    let mut kv_a = TestKv::new();
    let mut kv_b = TestKv::new();
    let la = a.forward(&[3u32], &mut kv_a).expect("forward a");
    let lb = b.forward(&[3u32], &mut kv_b).expect("forward b");

    assert!(
        la.iter().zip(&lb).any(|(x, y)| (x - y).abs() > 1e-6),
        "changing attn_v must change the logits; it did not, so the value \
         projection is being ignored"
    );
}

/// The experts stay in their GGUF form; nothing is dequantized at load.
///
/// Mixtral-8x7B's 8 × 3 expert matrices are ~5.6 GB per layer as f32 — the old
/// `Expert { gate: Vec<f32>, .. }` layout could never have loaded one.
#[test]
fn experts_stay_quantized_and_share_one_payload() {
    let model = load_mixtral(1);
    let moe = &model.layers[0].moe_ffn;

    assert_eq!(moe.num_experts(), EXPERTS);
    assert_eq!(moe.top_k, EXPERTS_USED);
    for (i, expert) in moe.experts.iter().enumerate() {
        assert_eq!(
            expert.gate.weight.tensor_type,
            GgufTensorType::Q4K,
            "expert {i} gate must stay Q4_K"
        );
        assert_eq!(expert.gate.out_features, INTER);
        assert_eq!(expert.gate.in_features, HIDDEN);
        assert_eq!(expert.down.out_features, HIDDEN);
        assert_eq!(expert.down.in_features, INTER);
    }
    // Distinct experts must be distinct slices, not the same one repeated.
    assert_ne!(
        moe.experts[0].gate.weight.data.as_slice(),
        moe.experts[1].gate.weight.data.as_slice(),
        "each expert must get its own slice of the stacked tensor"
    );
}

// ── Forward pass ─────────────────────────────────────────────────────────────

#[test]
fn forward_produces_finite_vocab_logits() {
    let mut model = load_mixtral(1);
    let mut kv = TestKv::new();
    let logits = model.forward(&[1u32, 2, 3], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit {i} must be finite, got {v}");
    }
}

/// The KV cache advances exactly once per token, after every layer has written.
#[test]
fn kv_cache_advances_once_per_token() {
    let mut model = load_mixtral(1);
    let mut kv = TestKv::new();
    model.forward(&[1u32, 2, 3, 4], &mut kv).expect("forward");
    assert_eq!(kv.advances, 4, "one advance per token, not per layer");
    assert_eq!(kv.seq_len(), 4);
}

/// Mixtral rotates **consecutive** pairs, like LLaMA.
///
/// `convert_hf_to_gguf.py` maps `MixtralForCausalLM` onto its `LlamaModel`
/// converter (`MODEL_ARCH.LLAMA`, `undo_permute = True`) and
/// `llama_model_rope_type` gives `LLM_ARCH_LLAMA` → `LLAMA_ROPE_TYPE_NORM`, so
/// the NeoX half-split is wrong here for exactly the same reason it is wrong for
/// llama.  The probe reads the key written at position 0 (RoPE is the identity
/// there) and compares the position-1 key against both conventions.
#[test]
fn rope_uses_the_norm_convention() {
    let token = 5u32;

    let mut m0 = load_mixtral(1);
    let mut kv0 = TestKv::new();
    m0.forward(&[token], &mut kv0).expect("position 0");
    let unrotated = kv0.key_head(0, 0);

    let mut m1 = load_mixtral(1);
    let mut kv1 = TestKv::primed(1);
    m1.forward(&[token], &mut kv1).expect("position 1");
    let rotated = kv1.key_head(1, 0);

    assert!(
        unrotated.iter().any(|v| v.abs() > 1e-6),
        "fixture must produce a non-trivial key"
    );

    let half = m1.rope.half_dim;
    let mut norm = unrotated.clone();
    for j in 0..half {
        let (c, s) = (m1.rope.cos[half + j], m1.rope.sin[half + j]);
        let (x0, x1) = (norm[2 * j], norm[2 * j + 1]);
        norm[2 * j] = x0 * c - x1 * s;
        norm[2 * j + 1] = x0 * s + x1 * c;
    }
    let mut neox = unrotated.clone();
    m1.rope.apply(&mut neox, 1);

    let close = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-4 * x.abs().max(1.0))
    };
    assert!(!close(&norm, &neox), "the fixture must distinguish the two");
    assert!(
        close(&rotated, &norm),
        "Mixtral must rotate consecutive pairs (LLAMA_ROPE_TYPE_NORM)"
    );
}

/// An out-of-vocabulary token id errors instead of aborting the process.
#[test]
fn out_of_vocabulary_token_is_an_error() {
    let mut model = load_mixtral(1);
    let mut kv = TestKv::new();
    assert!(model.forward(&[VOCAB as u32], &mut kv).is_err());
    assert!(model.forward(&[u32::MAX], &mut kv).is_err());
}

/// An over-long prompt is rejected before a weight is touched.
#[test]
fn over_long_prompt_is_an_error() {
    let mut model = load_mixtral(1);
    let mut kv = TestKv::new();
    let tokens: Vec<u32> = (0..CTX + 1).map(|i| (i % VOCAB) as u32).collect();
    let err = model
        .forward(&tokens, &mut kv)
        .expect_err("a prompt longer than the context window must be rejected");
    assert!(err.to_string().contains("context overflow"), "got: {err}");
}

// ── Agreement with the llama path ────────────────────────────────────────────

/// The same weights loaded as `"llama"` and as `"mixtral"` must compute the
/// same thing.
///
/// This is the property that matters in practice: llama.cpp writes
/// `general.architecture = "llama"` for every Mixtral checkpoint, so a real
/// Mixtral GGUF goes through [`load_llama_from_gguf`] and its
/// `FfnVariant::Moe`.  `crate::mixtral` only ever sees GGUFs from converters
/// that name the architecture explicitly — and those two routes must not drift.
#[test]
fn mixtral_and_llama_loaders_agree_on_the_same_weights() {
    let mixtral_gguf =
        GgufModel::from_bytes(build_fixture("mixtral", 7)).expect("mixtral fixture parses");
    let mixtral_config =
        ModelConfig::from_metadata(&mixtral_gguf.file.metadata).expect("mixtral metadata");
    let mut mixtral =
        load_mixtral_from_gguf(&mixtral_gguf, &mixtral_config).expect("mixtral loads");

    let llama_gguf =
        GgufModel::from_bytes(build_fixture("llama", 7)).expect("llama fixture parses");
    let llama_config =
        ModelConfig::from_metadata(&llama_gguf.file.metadata).expect("llama metadata");
    assert_eq!(
        llama_config.num_experts, EXPERTS,
        "the llama-arch twin must still be recognised as MoE"
    );
    let mut llama = load_llama_from_gguf(&llama_gguf, &llama_config).expect("llama loads");

    let mut kv_m = TestKv::new();
    let mut kv_l = TestKv::new();
    let from_mixtral = mixtral
        .forward(&[9u32], &mut kv_m)
        .expect("mixtral forward");
    let from_llama = llama.forward(&[9u32], &mut kv_l).expect("llama forward");

    let mismatches = from_mixtral
        .iter()
        .zip(&from_llama)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        mismatches,
        0,
        "the two loaders disagree on {mismatches} of {} logits",
        from_llama.len()
    );
}

// ── Registry ─────────────────────────────────────────────────────────────────

#[test]
fn registry_lookup() {
    use oxillama_arch::ArchitectureRegistry;

    let registry = ArchitectureRegistry::with_builtins();
    let arch = registry.get("mixtral").expect("registry.get('mixtral')");
    assert_eq!(arch.arch_id(), "mixtral");

    let names: Vec<String> = arch.tensor_names().into_iter().map(|p| p.pattern).collect();
    for required in [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.{i}.ffn_gate_inp.weight",
        "blk.{i}.ffn_gate_exps.weight",
        "blk.{i}.ffn_up_exps.weight",
        "blk.{i}.ffn_down_exps.weight",
    ] {
        assert!(
            names.iter().any(|n| n == required),
            "tensor_names should list '{required}'"
        );
    }
}

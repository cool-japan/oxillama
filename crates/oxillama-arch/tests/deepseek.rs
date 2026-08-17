//! Integration tests for DeepSeek-V2 / V2-Lite.
//!
//! Driven through the GGUF fixtures, which now carry the **real**
//! `LLM_ARCH_DEEPSEEK2` tensor names.  The old fixture invented
//! `blk.N.ffn_exp.{e}.ffn_gate.weight` / `blk.N.ffn_shared_exp.{e}.*`, names
//! that appear nowhere in llama.cpp or `gguf-py`, so a loader written against it
//! could not read a single real checkpoint.

#![cfg(feature = "deepseek")]

use oxillama_arch::deepseek::{load_deepseek_from_gguf, DeepSeekModel, FfnKind, QProjection};
use oxillama_arch::error::ArchResult;
use oxillama_arch::registry::ArchitectureRegistry;
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};

/// DeepSeek keeps its own `MlaLatentCache`, so the external KV cache is unused.
struct NoKv;
impl KvCacheAccess for NoKv {
    fn seq_len(&self) -> usize {
        0
    }
    fn store_kv(&mut self, _: usize, _: &[f32], _: &[f32]) -> ArchResult<()> {
        Ok(())
    }
    fn get_keys(&self, _: usize) -> ArchResult<&[f32]> {
        Ok(&[])
    }
    fn get_values(&self, _: usize) -> ArchResult<&[f32]> {
        Ok(&[])
    }
    fn advance(&mut self) {}
}

fn load_v2() -> DeepSeekModel {
    let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("V2 fixture parses");
    load_deepseek_from_gguf(&gguf).expect("load_deepseek_from_gguf")
}

fn load_lite() -> DeepSeekModel {
    let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_lite_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("Lite fixture parses");
    load_deepseek_from_gguf(&gguf).expect("load_deepseek_from_gguf (Lite)")
}

#[test]
fn deepseek_registered_in_registry() {
    let reg = ArchitectureRegistry::with_builtins();
    assert!(reg.contains("deepseek2"));
    assert_eq!(reg.get("deepseek2").expect("get").arch_id(), "deepseek2");
}

/// The registry can now build DeepSeek; the default `build_from_gguf` returned
/// `NotSupported`.
#[test]
fn deepseek_reachable_through_registry() {
    let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("fixture parses");
    let config = oxillama_arch::config::ModelConfig::from_metadata(&gguf.file.metadata)
        .expect("config parses");
    let model = ArchitectureRegistry::with_builtins()
        .get("deepseek2")
        .expect("get deepseek2")
        .build_from_gguf(&gguf, &config)
        .expect("registry must be able to build DeepSeek from a GGUF");
    assert_eq!(model.vocab_size(), 32);
}

#[test]
fn v2_forward_shape_and_finiteness() {
    let mut model = load_v2();
    let mut kv = NoKv;
    let logits = model.forward(&[1u32, 2], &mut kv).expect("forward");
    assert_eq!(logits.len(), 32);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn v2_embed_returns_hidden_size() {
    let mut model = load_v2();
    let mut kv = NoKv;
    assert_eq!(model.embed(&[1u32], &mut kv).expect("embed").len(), 32);
}

/// `leading_dense_block_count = 1`: layer 0 dense, layer 1 routed MoE.
#[test]
fn v2_layer_zero_is_dense_and_layer_one_is_moe() {
    let model = load_v2();
    assert!(matches!(model.layers[0].ffn, FfnKind::Dense(_)));
    match &model.layers[1].ffn {
        FfnKind::Moe(m) => {
            assert_eq!(m.num_experts(), 2);
            assert!(m.has_shared_expert(), "one shared expert of width n_ff_exp");
        }
        FfnKind::Dense(_) => panic!("layer 1 must be a MoE layer"),
    }
}

/// DeepSeek-V2 ships `norm_topk_prob = false`; without an explicit
/// `expert_weights_norm` key the weights must **not** be re-normalised.
#[test]
fn v2_does_not_renormalise_top_k_weights_by_default() {
    let model = load_v2();
    match &model.layers[1].ffn {
        FfnKind::Moe(m) => assert!(
            !m.config().norm_weights,
            "DeepSeek-V2 must not re-normalise the top-k weights"
        ),
        FfnKind::Dense(_) => panic!("layer 1 must be a MoE layer"),
    }
}

/// The V2 fixture uses the real tensor names.
#[test]
fn v2_fixture_uses_the_llama_cpp_tensor_names() {
    let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("fixture parses");
    for name in [
        "blk.0.attn_q_a.weight",
        "blk.0.attn_q_a_norm.weight",
        "blk.0.attn_q_b.weight",
        "blk.0.attn_kv_a_mqa.weight",
        "blk.1.ffn_gate_exps.weight",
        "blk.1.ffn_down_shexp.weight",
    ] {
        assert!(gguf.file.tensors.contains(name), "missing {name}");
    }
    for gone in [
        "blk.0.attn_q_a_proj.weight",
        "blk.0.attn_kv_a_proj.weight",
        "blk.1.ffn_exp.0.ffn_gate.weight",
        "blk.1.ffn_shared_exp.0.ffn_gate.weight",
    ] {
        assert!(
            !gguf.file.tensors.contains(gone),
            "{gone} is not a llama.cpp tensor name"
        );
    }
}

/// D2: `qk_nope_head_dim = key_length − rope.dimension_count`.
///
/// The fixture declares `key_length = 8` and `rope.dimension_count = 4`, so the
/// nope slice is 4 wide.  Reading `qk_nope_head_dim` straight off `key_length`
/// (what `config.rs` does) would give 8 and split `attn_q_b`'s 16 outputs into
/// two heads of 12 — past the end of the row.
#[test]
fn head_dims_come_from_key_length_minus_rope_dimension_count() {
    let model = load_v2();
    let mla = &model.layers[0].mla_config;
    assert_eq!(mla.qk_nope_head_dim, 4, "8 - 4");
    assert_eq!(mla.qk_rope_head_dim, 4);
    assert_eq!(mla.qk_head_dim(), 8);
}

/// D3: DeepSeek-V2-Lite has `q_lora_rank = null` and one `blk.N.attn_q.weight`.
///
/// This could not be loaded at all before: `load_mla_weights` unconditionally
/// required `attn_q_a*` / `attn_q_b*` and `mla_forward` unconditionally ran
/// `w_q_a → q_a_norm → w_q_b`.  There is no "failing before" to show — the
/// capability did not exist.
#[test]
fn lite_variant_loads_and_runs() {
    let mut model = load_lite();
    assert!(
        model.layers[0].mla_weights.q.is_lite(),
        "a checkpoint with blk.0.attn_q.weight must take the dense Q path"
    );
    assert!(matches!(
        model.layers[0].mla_weights.q,
        QProjection::Dense { .. }
    ));
    let mut kv = NoKv;
    let logits = model.forward(&[1u32, 2], &mut kv).expect("Lite forward");
    assert_eq!(logits.len(), 32);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// `reset_sequence()` clears the latent cache, so two identical requests give
/// identical logits without rebuilding the model.
///
/// The old test worked around the missing reset by constructing a second model
/// from the same RNG seed.
#[test]
fn reset_sequence_makes_a_second_request_reproducible() {
    let mut model = load_v2();
    let mut kv = NoKv;
    let first = model.forward(&[0u32, 1], &mut kv).expect("first");
    model.reset_sequence();
    let second = model.forward(&[0u32, 1], &mut kv).expect("second");
    assert_eq!(first.len(), second.len());
    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit {i} must be bit-identical after reset_sequence(): {a} vs {b}"
        );
    }
}

/// Without a reset the latent cache keeps growing, and a prompt longer than the
/// declared context must be reported rather than overrunning the cache.
#[test]
fn overlong_prompt_is_reported() {
    let mut model = load_v2();
    let mut kv = NoKv;
    let max = model.max_context_length();
    let tokens: Vec<u32> = (0..=max as u32).map(|t| t % 4).collect();
    assert!(model.forward(&tokens, &mut kv).is_err());
}

#[test]
fn out_of_vocab_token_is_reported() {
    let mut model = load_v2();
    let mut kv = NoKv;
    assert!(model.forward(&[9999u32], &mut kv).is_err());
}

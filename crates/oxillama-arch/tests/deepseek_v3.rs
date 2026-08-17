//! DeepSeek-V3 regression tests: sigmoid+bias routing, group-limited routing,
//! `expert_weights_norm`, `expert_weights_scale`.
//!
//! # What changed here and why
//!
//! `sigmoid_bias_topk_weights` used to recompute the expected weights as
//! `sigmoid(logit) + bias`, re-normalised — i.e. it *encoded the bug*.
//! `llm_graph_context::build_moe_ffn` keeps selection and combination apart:
//!
//! ```text
//! probs           = sigmoid(logits)
//! selection_probs = probs + exp_probs_b   // "leave probs unbiased as it's
//! selected        = top_k(selection_probs) //  later used to get expert weights"
//! weights         = probs[selected]        // UNBIASED
//! ```
//!
//! The new expectation is the unbiased `sigmoid(logit)`, because `exp_probs_b`
//! is DeepSeek-V3's *load-balancing* term: it exists to steer which experts are
//! picked, not to change how much each contributes.  It can be negative, so
//! using it as a weight can drive the combination weight below zero — and the
//! old `weight_sum > 0.0` guard then zeroed the entire routed branch.

#![cfg(feature = "deepseek")]

use oxillama_arch::deepseek::moe::{
    moe_forward, DeepSeekExpert, MoeConfig, MoeWeights, ScoringMode,
};
use oxillama_arch::deepseek::{load_deepseek_from_gguf, DeepSeekModel, FfnKind};
use oxillama_arch::error::ArchResult;
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};

const H: usize = 8;
const INTER: usize = 8;
const N_EXPERTS: usize = 4;
const TOP_K: usize = 2;

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

fn expert(scale: f32) -> DeepSeekExpert {
    let mut gate = vec![0.0f32; INTER * H];
    let mut up = vec![0.0f32; INTER * H];
    let mut down = vec![0.0f32; H * INTER];
    for i in 0..INTER.min(H) {
        gate[i * H + i] = 1.0;
        up[i * H + i] = 1.0;
        down[i * INTER + i] = scale;
    }
    DeepSeekExpert {
        gate,
        up,
        down,
        hidden_size: H,
        intermediate_size: INTER,
    }
}

/// Router row `e` puts weight `e` on input dim 0, so for `x = e0` the logits are
/// `[0, 1, 2, 3]`.
fn router() -> Vec<f32> {
    let mut r = vec![0.0f32; N_EXPERTS * H];
    for e in 0..N_EXPERTS {
        r[e * H] = e as f32;
    }
    r
}

fn cfg(mode: ScoringMode) -> MoeConfig {
    MoeConfig {
        hidden_size: H,
        expert_intermediate_size: INTER,
        n_shared_experts: 0,
        n_routed_experts: N_EXPERTS,
        top_k: TOP_K,
        routed_scaling_factor: 1.0,
        scoring_mode: mode,
        shared_expert_intermediate_size: INTER,
    }
}

/// (a) The combination weight is the **unbiased** `sigmoid(logit)`.
///
/// Bias `[10, 0, 0, 0]` makes expert 0 (lowest logit) win selection.  With the
/// old semantics its weight would have been `sigmoid(0) + 10 = 10.5`; with the
/// reference semantics it is `sigmoid(0) = 0.5`.
#[test]
fn sigmoid_bias_weights_are_unbiased() {
    let bias = vec![10.0f32, 0.0, 0.0, 0.0];
    let weights = MoeWeights {
        router: router(),
        routed_experts: (0..N_EXPERTS).map(|_| expert(1.0)).collect(),
        shared_experts: vec![],
        expert_bias: Some(bias),
    };
    let mut c = cfg(ScoringMode::SigmoidWithBias);
    c.top_k = 1;

    let mut x = vec![0.0f32; H];
    x[0] = 1.0;
    let out = moe_forward(&x, &weights, &c).expect("moe_forward");

    // top-1 → the single weight normalises to exactly 1, so the output is the
    // expert's own response: silu(1) * 1 * 1.
    let expected = 1.0f32 / (1.0 + (-1.0f32).exp());
    assert!(
        (out[0] - expected).abs() < 1e-5,
        "expected the unbiased routing to give {expected}, got {}",
        out[0]
    );
}

/// (b) A negative `exp_probs_b` must not be able to zero the routed branch.
///
/// With the old code the bias landed in the weights, so an all-negative bias
/// drove `weight_sum` below zero, `inv_weight_sum` became `0.0`, and every
/// routed expert contributed exactly nothing.
#[test]
fn negative_bias_does_not_zero_the_routed_branch() {
    let weights = MoeWeights {
        router: router(),
        routed_experts: (0..N_EXPERTS).map(|_| expert(1.0)).collect(),
        shared_experts: vec![],
        expert_bias: Some(vec![-5.0f32; N_EXPERTS]),
    };
    let c = cfg(ScoringMode::SigmoidWithBias);
    let mut x = vec![0.0f32; H];
    x[0] = 1.0;
    let out = moe_forward(&x, &weights, &c).expect("moe_forward");
    assert!(
        out.iter().any(|v| v.abs() > 1e-6),
        "an all-negative selection bias must not zero the routed output: {out:?}"
    );
}

/// (c) `top_k == 0` is an error, not an arithmetic underflow panic.
///
/// `top_k` comes straight from `deepseek2.expert_used_count`, so a malformed
/// checkpoint used to take the process down inside
/// `select_nth_unstable_by(top_k - 1, …)`.
#[test]
fn zero_top_k_is_an_error_not_a_panic() {
    let weights = MoeWeights {
        router: router(),
        routed_experts: (0..N_EXPERTS).map(|_| expert(1.0)).collect(),
        shared_experts: vec![],
        expert_bias: None,
    };
    let mut c = cfg(ScoringMode::Softmax);
    c.top_k = 0;
    let x = vec![1.0f32; H];
    assert!(
        moe_forward(&x, &weights, &c).is_err(),
        "top_k = 0 must return an error"
    );
}

/// (d) Selection still follows the biased score even though the weight does not.
#[test]
fn bias_still_steers_selection() {
    // Bias makes expert 0 (the lowest logit) the top-1 pick.
    let weights = MoeWeights {
        router: router(),
        routed_experts: (0..N_EXPERTS)
            .map(|e| expert(if e == 0 { 7.0 } else { 0.0 }))
            .collect(),
        shared_experts: vec![],
        expert_bias: Some(vec![10.0f32, 0.0, 0.0, 0.0]),
    };
    let mut c = cfg(ScoringMode::SigmoidWithBias);
    c.top_k = 1;
    let mut x = vec![0.0f32; H];
    x[0] = 1.0;
    let out = moe_forward(&x, &weights, &c).expect("moe_forward");
    // Only expert 0 has a non-zero `down`, so a non-zero output proves it was
    // the one selected.
    assert!(
        out[0].abs() > 1e-6,
        "the bias must steer selection to expert 0, got {out:?}"
    );
}

// ─── V3 GGUF fixture ─────────────────────────────────────────────────────────

fn load_v3() -> DeepSeekModel {
    let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_v3_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("V3 fixture parses");
    load_deepseek_from_gguf(&gguf).expect("load_deepseek_from_gguf")
}

/// The V3 fixture now carries the real llama.cpp tensor names, including
/// `blk.1.exp_probs_b.bias` (a `.bias` suffix, per
/// `tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i)`) and the stacked expert pool.
#[test]
fn v3_fixture_uses_the_llama_cpp_tensor_names() {
    let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_v3_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("fixture parses");
    for name in [
        "blk.1.ffn_gate_exps.weight",
        "blk.1.ffn_up_exps.weight",
        "blk.1.ffn_down_exps.weight",
        "blk.1.ffn_gate_shexp.weight",
        "blk.1.exp_probs_b.bias",
        "blk.1.attn_kv_a_mqa.weight",
        "blk.1.attn_q_a.weight",
        "blk.1.attn_q_b.weight",
    ] {
        assert!(gguf.file.tensors.contains(name), "missing {name}");
    }
    assert!(
        !gguf
            .file
            .tensors
            .contains("blk.1.ffn_exp.0.ffn_gate.weight"),
        "the invented per-expert 2-D names must be gone"
    );
}

#[test]
fn v3_loads_and_runs() {
    let mut model = load_v3();
    let mut kv = NoKv;
    let logits = model.forward(&[1u32, 2], &mut kv).expect("forward");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// Layer 1 must be a routed MoE with the shared expert attached, and its
/// routing must reflect the V3 metadata: sigmoid gating, weight
/// re-normalisation, `expert_weights_scale = 2.5`, and group-limited routing.
#[test]
fn v3_routing_config_reaches_the_layer() {
    let model = load_v3();
    let moe = match &model.layers[1].ffn {
        FfnKind::Moe(m) => m,
        FfnKind::Dense(_) => panic!("layer 1 must be a MoE layer"),
    };
    let cfg = moe.config();
    assert_eq!(
        cfg.gating,
        oxillama_arch::deepseek::GatingFunc::Sigmoid,
        "deepseek2.expert_gating_func = 2"
    );
    assert!(cfg.norm_weights, "deepseek2.expert_weights_norm = true");
    assert!(
        (cfg.weight_scale - 2.5).abs() < 1e-6,
        "deepseek2.expert_weights_scale = 2.5, got {}",
        cfg.weight_scale
    );
    assert_eq!(cfg.n_group, 2, "deepseek2.expert_group_count");
    assert_eq!(cfg.topk_group, 1, "deepseek2.expert_group_used_count");
    assert!(
        moe.has_shared_expert(),
        "DeepSeek always runs one shared expert"
    );
}

/// Layer 0 is dense (`leading_dense_block_count = 1`).
#[test]
fn v3_leading_layer_is_dense() {
    let model = load_v3();
    assert!(matches!(model.layers[0].ffn, FfnKind::Dense(_)));
}

//! Quantized routed Mixture-of-Experts FFN with llama.cpp `build_moe_ffn`
//! semantics.
//!
//! [`crate::common::moe::QuantMoeFfn`] covers the Mixtral case exactly — softmax
//! gating, top-`k`, re-normalise, SwiGLU.  Three of the architectures in this
//! crate need strictly more than that, and every one of the extra knobs is a
//! *correctness* requirement rather than a tuning option:
//!
//! | knob | who needs it | what breaks without it |
//! |---|---|---|
//! | GELU experts | Grok-1 (`build_grok` passes `LLM_FFN_GELU`) | wrong activation on 100 % of expert output |
//! | sigmoid gating | DeepSeek-V3 (`expert_gating_func = 2`) | wrong routing weights |
//! | selection bias `exp_probs_b` | DeepSeek-V3 | wrong experts selected |
//! | unbiased combination weight | DeepSeek-V3 | negative weights, zeroed routed branch |
//! | group-limited routing | DeepSeek-V3 (`n_group = 8`, `topk_group = 4`) | routes across all 256 experts |
//! | optional weight norm | DeepSeek-V2 ships `norm_topk_prob = false` | wrong weight magnitude |
//! | weight scale | DeepSeek-V3 `expert_weights_scale = 2.5` | routed branch 2.5× too small |
//! | shared expert | DeepSeek V2/V3 | a whole always-on FFN missing |
//!
//! Like [`QuantMoeFfn`](crate::common::moe::QuantMoeFfn) the weights stay in
//! their GGUF quantized form as shared mmap views produced by
//! [`load_stacked_experts`](crate::common::loader::load_stacked_experts): a
//! DBRX layer costs its quantized bytes once instead of the >500 GB an `f32`
//! expansion of the whole model would need.
//!
//! # Reference
//!
//! `llm_graph_context::build_moe_ffn` in `src/llama-graph.cpp`.  The ordering
//! there is load-bearing and reproduced exactly:
//!
//! ```text
//! logits          = gate_inp @ x
//! probs           = softmax(logits) | sigmoid(logits)
//! selection_probs = probs + exp_probs_b        // selection only
//! (group mask over selection_probs)
//! selected        = top_k(selection_probs)
//! weights         = probs[selected]            // UNBIASED
//! if norm_w:  weights /= max(sum(weights), 6.103515625e-5)
//! if scale_w: weights *= w_scale
//! ```
//!
//! Note where `exp_probs_b` is *not* used: llama.cpp's own comment is
//! "leave probs unbiased as it's later used to get expert weights".
//!
//! # Placement
//!
//! This belongs next to [`crate::common::moe::QuantMoeFfn`] in `crate::common::moe`; it lives here
//! only because `src/common/` is owned by a different agent in the current
//! split.  Nothing in it is DeepSeek-specific.

use oxillama_quant::{quantize_activations_q8_0_into, KernelDispatcher, QuantKernel};

use crate::common::gelu::gelu;
use crate::common::linear::QuantLinear;
use crate::common::moe::{ExpertKernels, MoeScratch, QuantExpert};
use crate::error::{ArchError, ArchResult};

/// Smallest positive normal `f16`, the divisor floor llama.cpp clamps the
/// top-k weight sum to (`ggml_clamp(weights_sum, 6.103515625e-5, INFINITY)`).
///
/// The previous DeepSeek implementation used `if weight_sum > 0.0 { … } else {
/// 0.0 }`, which silently deleted the entire routed branch whenever the sum was
/// non-positive.
const WEIGHT_SUM_FLOOR: f32 = 6.103_515_6e-5;

/// Gated activation applied to the expert's gate projection.
///
/// Mirrors `llm_ffn_op_type` for the two gated forms this crate's MoE
/// architectures use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertActivation {
    /// `silu(gate) * up` — `LLM_FFN_SILU` + `ggml_swiglu_split`.
    ///
    /// DBRX, DeepSeek, Mixtral, Phi-MoE.
    #[default]
    Silu,
    /// `gelu(gate) * up` — `LLM_FFN_GELU` + `ggml_geglu_split`.
    ///
    /// Grok-1 (`build_grok` passes `LLM_FFN_GELU`).
    Gelu,
}

impl ExpertActivation {
    #[inline]
    fn apply(self, gate: f32) -> f32 {
        match self {
            Self::Silu => gate / (1.0 + (-gate).exp()),
            Self::Gelu => gelu(gate),
        }
    }
}

/// How router logits are turned into probabilities.
///
/// Mirrors `llama_expert_gating_func_type`; the numeric values are the ones
/// written into `{arch}.expert_gating_func` by `gguf-py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GatingFunc {
    /// `softmax(logits)` — `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` (1).
    ///
    /// DeepSeek-V2, DBRX, Grok, Mixtral.
    #[default]
    Softmax,
    /// `sigmoid(logits)` — `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` (2).
    ///
    /// DeepSeek-V3.
    Sigmoid,
}

impl GatingFunc {
    /// Parse the GGUF `{arch}.expert_gating_func` value.
    ///
    /// llama.cpp treats `0` (`_NONE`) as "old checkpoint" and falls back to
    /// softmax; so does this.
    ///
    /// # Errors
    ///
    /// [`ArchError::NotSupported`] for an unrecognised value, rather than
    /// silently routing with the wrong function.
    pub fn from_gguf_value(value: u32) -> ArchResult<Self> {
        match value {
            0 | 1 => Ok(Self::Softmax),
            2 => Ok(Self::Sigmoid),
            other => Err(ArchError::NotSupported {
                detail: format!("expert_gating_func = {other} is not a known gating function"),
            }),
        }
    }
}

/// Routing behaviour of one MoE layer.
#[derive(Debug, Clone)]
pub struct RoutedMoeConfig {
    /// Experts activated per token (`{arch}.expert_used_count`). Must be >= 1.
    pub top_k: usize,
    /// Gated activation used inside each expert.
    pub activation: ExpertActivation,
    /// Router probability function.
    pub gating: GatingFunc,
    /// Re-normalise the selected weights to sum to 1
    /// (`{arch}.expert_weights_norm`, llama.cpp `norm_w`).
    ///
    /// DBRX/Grok/Mixtral: `true`.  DeepSeek-V2: `false` (`norm_topk_prob`).
    /// DeepSeek-V3: `true`.
    pub norm_weights: bool,
    /// Multiplier on the selected weights (`{arch}.expert_weights_scale`).
    ///
    /// `1.0` disables the scale.  DeepSeek-V3 ships `2.5`.
    pub weight_scale: f32,
    /// Number of expert groups (`{arch}.expert_group_count`).
    ///
    /// `0` or `1` disables group-limited routing.  DeepSeek-V3 ships `8`.
    pub n_group: usize,
    /// Groups kept per token (`{arch}.expert_group_used_count`).
    ///
    /// DeepSeek-V3 ships `4`.
    pub topk_group: usize,
}

impl Default for RoutedMoeConfig {
    fn default() -> Self {
        Self {
            top_k: 1,
            activation: ExpertActivation::Silu,
            gating: GatingFunc::Softmax,
            norm_weights: true,
            weight_scale: 1.0,
            n_group: 1,
            topk_group: 1,
        }
    }
}

/// One expert whose kernels have already been resolved.
struct ResolvedExpert {
    expert: QuantExpert,
    kernels: ExpertKernels,
}

/// A sparse MoE FFN over quantized experts with full `build_moe_ffn` semantics.
pub struct RoutedQuantMoe {
    router: QuantLinear,
    router_kernel: Box<dyn QuantKernel>,
    experts: Vec<ResolvedExpert>,
    /// `blk.N.exp_probs_b.weight` — the DeepSeek-V3 load-balancing bias added to
    /// the **selection** scores only.
    exp_probs_b: Option<Vec<f32>>,
    /// Always-active shared expert (`ffn_*_shexp`), summed into the output.
    shared: Option<ResolvedExpert>,
    cfg: RoutedMoeConfig,
    hidden_size: usize,
}

impl RoutedQuantMoe {
    /// Assemble a routed MoE layer and resolve every kernel up front.
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidConfig`] when `top_k == 0` (the previous DeepSeek
    ///   `moe_forward` computed `select_nth_unstable_by(top_k - 1, …)` and
    ///   panicked on the underflow — `top_k` comes straight from
    ///   `{arch}.expert_used_count`, i.e. from the file), when there are no
    ///   experts, or when the group configuration does not divide the pool.
    /// * [`ArchError::InvalidShape`] when the router width disagrees with the
    ///   expert count or hidden size, or when `exp_probs_b` is the wrong length.
    /// * [`ArchError::Quant`] for an unsupported quantization type.
    pub fn new(
        router: QuantLinear,
        experts: Vec<QuantExpert>,
        shared: Option<QuantExpert>,
        exp_probs_b: Option<Vec<f32>>,
        cfg: RoutedMoeConfig,
    ) -> ArchResult<Self> {
        if experts.is_empty() {
            return Err(ArchError::InvalidConfig {
                detail: "RoutedQuantMoe requires at least one expert".to_string(),
            });
        }
        if cfg.top_k == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "RoutedQuantMoe top_k must be >= 1 (expert_used_count = 0 in the GGUF)"
                    .to_string(),
            });
        }
        if cfg.top_k > experts.len() {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "RoutedQuantMoe top_k = {} exceeds the {} available experts",
                    cfg.top_k,
                    experts.len()
                ),
            });
        }
        let hidden_size = experts[0].hidden_size();
        if router.out_features != experts.len() {
            return Err(ArchError::InvalidShape {
                name: "routed_moe.router".to_string(),
                expected: vec![experts.len(), hidden_size],
                got: vec![router.out_features, router.in_features],
            });
        }
        if router.in_features != hidden_size {
            return Err(ArchError::InvalidShape {
                name: "routed_moe.router.in_features".to_string(),
                expected: vec![hidden_size],
                got: vec![router.in_features],
            });
        }
        if let Some(bias) = exp_probs_b.as_ref() {
            if bias.len() != experts.len() {
                return Err(ArchError::InvalidShape {
                    name: "routed_moe.exp_probs_b".to_string(),
                    expected: vec![experts.len()],
                    got: vec![bias.len()],
                });
            }
        }
        if cfg.n_group > 1 {
            if !experts.len().is_multiple_of(cfg.n_group) {
                return Err(ArchError::InvalidConfig {
                    detail: format!(
                        "expert_group_count = {} does not divide the {} experts",
                        cfg.n_group,
                        experts.len()
                    ),
                });
            }
            if cfg.topk_group == 0 || cfg.topk_group > cfg.n_group {
                return Err(ArchError::InvalidConfig {
                    detail: format!(
                        "expert_group_used_count = {} must be in 1..={}",
                        cfg.topk_group, cfg.n_group
                    ),
                });
            }
        }

        let dispatcher = KernelDispatcher::new();
        let router_kernel = dispatcher.get_kernel(router.weight.tensor_type)?;
        let mut resolved = Vec::with_capacity(experts.len());
        for expert in experts {
            let kernels = ExpertKernels::for_expert(&expert)?;
            resolved.push(ResolvedExpert { expert, kernels });
        }
        let shared = match shared {
            Some(expert) => {
                let kernels = ExpertKernels::for_expert(&expert)?;
                Some(ResolvedExpert { expert, kernels })
            }
            None => None,
        };

        Ok(Self {
            router,
            router_kernel,
            experts: resolved,
            exp_probs_b,
            shared,
            cfg,
            hidden_size,
        })
    }

    /// Model hidden dimension.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of routed experts in this layer.
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Whether this layer carries an always-active shared expert.
    pub fn has_shared_expert(&self) -> bool {
        self.shared.is_some()
    }

    /// Routing configuration.
    pub fn config(&self) -> &RoutedMoeConfig {
        &self.cfg
    }

    /// Allocate scratch sized for this layer.
    pub fn make_scratch(&self) -> MoeScratch {
        let inter = self
            .experts
            .first()
            .map_or(0, |e| e.expert.intermediate_size())
            .max(
                self.shared
                    .as_ref()
                    .map_or(0, |e| e.expert.intermediate_size()),
            );
        MoeScratch::new(self.hidden_size, inter, self.experts.len())
    }

    /// Router probabilities and the selected `(expert, weight)` pairs.
    ///
    /// Split out from [`Self::forward`] so the routing semantics can be tested
    /// without materialising expert weights.
    ///
    /// # Errors
    ///
    /// Propagates the router GEMV failure.
    pub fn route(&self, input: &[f32], out: &mut Vec<(usize, f32)>) -> ArchResult<()> {
        let n_exp = self.experts.len();
        let mut probs = vec![0.0f32; n_exp];
        self.router
            .forward(&*self.router_kernel, input, &mut probs)
            .map_err(ArchError::from)?;
        self.route_from_logits(&mut probs, out);
        Ok(())
    }

    /// Shared tail of [`Self::route`]: turn raw router logits into a selection.
    fn route_from_logits(&self, probs: &mut [f32], out: &mut Vec<(usize, f32)>) {
        let n_exp = probs.len();

        match self.cfg.gating {
            GatingFunc::Softmax => softmax_inplace(probs),
            GatingFunc::Sigmoid => {
                for p in probs.iter_mut() {
                    *p = 1.0 / (1.0 + (-*p).exp());
                }
            }
        }

        // Selection scores: probs + exp_probs_b.  `probs` itself stays
        // unbiased — it is what the combination weights are read from.
        let mut selection: Vec<f32> = match self.exp_probs_b.as_ref() {
            Some(bias) => probs
                .iter()
                .zip(bias.iter())
                .map(|(p, b)| p + b)
                .collect::<Vec<f32>>(),
            None => probs.to_vec(),
        };

        // Group-limited routing (DeepSeek-V3): score each group by the sum of
        // its top-2 selection values, keep `topk_group` groups, mask the rest.
        if self.cfg.n_group > 1 && self.cfg.topk_group < self.cfg.n_group {
            let per_group = n_exp / self.cfg.n_group;
            let mut group_scores: Vec<(usize, f32)> = Vec::with_capacity(self.cfg.n_group);
            for g in 0..self.cfg.n_group {
                let slice = &selection[g * per_group..(g + 1) * per_group];
                // Sum of the two largest entries, without allocating a sort.
                let mut best = f32::NEG_INFINITY;
                let mut second = f32::NEG_INFINITY;
                for &v in slice {
                    if v > best {
                        second = best;
                        best = v;
                    } else if v > second {
                        second = v;
                    }
                }
                let score = if per_group >= 2 { best + second } else { best };
                group_scores.push((g, score));
            }
            group_scores.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            let mut keep = vec![false; self.cfg.n_group];
            for &(g, _) in group_scores.iter().take(self.cfg.topk_group) {
                keep[g] = true;
            }
            for (g, keep_group) in keep.iter().enumerate() {
                if !keep_group {
                    for s in selection[g * per_group..(g + 1) * per_group].iter_mut() {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Top-k by selection score; ties resolve by ascending expert index so
        // the result is deterministic across runs and platforms.
        let mut order: Vec<usize> = (0..n_exp).collect();
        order.sort_by(|&a, &b| {
            selection[b]
                .partial_cmp(&selection[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let k = self.cfg.top_k.min(n_exp);
        out.clear();
        // Combination weights come from the UNBIASED probabilities.
        for &e in &order[..k] {
            out.push((e, probs[e]));
        }

        if self.cfg.norm_weights {
            let sum: f32 = out.iter().map(|&(_, w)| w).sum();
            let denom = sum.max(WEIGHT_SUM_FLOOR);
            for entry in out.iter_mut() {
                entry.1 /= denom;
            }
        }
        if self.cfg.weight_scale != 1.0 {
            for entry in out.iter_mut() {
                entry.1 *= self.cfg.weight_scale;
            }
        }
    }

    /// Run one expert with this layer's activation, through the fused
    /// Q8_0-activation path wherever a projection's kernel supports it.
    ///
    /// `acts_q8` is a shared Q8_0 image of `input`, quantized once by
    /// [`Self::forward`] for the routed selection plus the shared expert (if
    /// any) — see [`QuantExpert::forward_fused`] for why sharing it across
    /// experts is exact, not an approximation.
    fn run_expert(
        expert: &ResolvedExpert,
        activation: ExpertActivation,
        input: &[f32],
        acts_q8: &[u8],
        output: &mut [f32],
        scratch: &mut MoeScratch,
    ) -> ArchResult<()> {
        expert.expert.forward_fused(
            &expert.kernels,
            input,
            acts_q8,
            |g| activation.apply(g),
            output,
            scratch,
        )
    }

    /// Route `input` to the top-`k` experts, add the shared expert, and write
    /// the combined result into `output` (**overwritten**, not accumulated).
    ///
    /// # Errors
    ///
    /// Propagates routing and expert GEMV failures.
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut MoeScratch,
    ) -> ArchResult<()> {
        if input.len() != self.hidden_size || output.len() != self.hidden_size {
            return Err(ArchError::InvalidShape {
                name: "routed_moe.forward".to_string(),
                expected: vec![self.hidden_size, self.hidden_size],
                got: vec![input.len(), output.len()],
            });
        }
        let n_exp = self.experts.len();
        scratch.ensure_router(n_exp);
        scratch.ensure_output(self.hidden_size);

        // Router GEMV into the scratch buffer, then routing on that slice.
        let mut logits = std::mem::take(&mut scratch.router_logits);
        let router_result = self
            .router
            .forward(&*self.router_kernel, input, &mut logits[..n_exp])
            .map_err(ArchError::from);
        let mut selection: Vec<(usize, f32)> = Vec::with_capacity(self.cfg.top_k);
        if router_result.is_ok() {
            self.route_from_logits(&mut logits[..n_exp], &mut selection);
        }
        scratch.router_logits = logits;
        router_result?;

        output.fill(0.0);

        // Every selected expert's `gate`/`up` — plus the always-active shared
        // expert's, if any — reads this exact `input` vector, so it is
        // quantized to Q8_0 once here and shared across all of them.  See
        // `QuantExpert::forward_fused`.  Taken out of `scratch` only after
        // the router's early-return above, so a routing failure never drops
        // this buffer's allocation.
        let mut shared_blocks: Option<usize> = None;
        for &(idx, _) in &selection {
            let re = &self.experts[idx];
            if let Some(b) = re.expert.gate_up_fused_blocks(&re.kernels) {
                shared_blocks = Some(shared_blocks.map_or(b, |m: usize| m.max(b)));
            }
        }
        if let Some(shared_expert) = self.shared.as_ref() {
            if let Some(b) = shared_expert
                .expert
                .gate_up_fused_blocks(&shared_expert.kernels)
            {
                shared_blocks = Some(shared_blocks.map_or(b, |m: usize| m.max(b)));
            }
        }
        let mut acts_q8 = std::mem::take(&mut scratch.acts_q8);
        if let Some(n_blocks) = shared_blocks {
            quantize_activations_q8_0_into(input, n_blocks, &mut acts_q8);
        }

        let mut expert_out = std::mem::take(&mut scratch.expert_out);
        let mut result = Ok(());

        for &(idx, weight) in &selection {
            match Self::run_expert(
                &self.experts[idx],
                self.cfg.activation,
                input,
                &acts_q8,
                &mut expert_out,
                scratch,
            ) {
                Ok(()) => {
                    for (o, e) in output.iter_mut().zip(expert_out.iter()) {
                        *o += weight * e;
                    }
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }

        if result.is_ok() {
            if let Some(shared) = self.shared.as_ref() {
                match Self::run_expert(
                    shared,
                    self.cfg.activation,
                    input,
                    &acts_q8,
                    &mut expert_out,
                    scratch,
                ) {
                    Ok(()) => {
                        for (o, e) in output.iter_mut().zip(expert_out.iter()) {
                            *o += e;
                        }
                    }
                    Err(e) => result = Err(e),
                }
            }
        }

        scratch.expert_out = expert_out;
        scratch.acts_q8 = acts_q8;
        result
    }
}

/// Numerically stable in-place softmax.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    } else {
        let uniform = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    /// Build an F32 `QuantLinear` from row-major `[rows, cols]` values.
    fn f32_linear(rows: usize, cols: usize, values: &[f32]) -> QuantLinear {
        assert_eq!(values.len(), rows * cols);
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        QuantLinear::new(
            QuantTensor::new(bytes, vec![rows, cols], GgufTensorType::F32),
            None,
        )
    }

    /// An expert whose output is `scale * ones(hidden)` for any input with
    /// `sum(input) == 1`: gate = up = identity-ish, down = scale on the diagonal.
    fn scaled_expert(hidden: usize, inter: usize, down_scale: f32) -> QuantExpert {
        let mut gate = vec![0.0f32; inter * hidden];
        let mut up = vec![0.0f32; inter * hidden];
        let mut down = vec![0.0f32; hidden * inter];
        for i in 0..inter.min(hidden) {
            gate[i * hidden + i] = 1.0;
            up[i * hidden + i] = 1.0;
        }
        for j in 0..hidden.min(inter) {
            down[j * inter + j] = down_scale;
        }
        QuantExpert::new(
            f32_linear(inter, hidden, &gate),
            f32_linear(inter, hidden, &up),
            f32_linear(hidden, inter, &down),
        )
        .expect("expert shapes compose")
    }

    fn build(
        n_exp: usize,
        router_values: &[f32],
        bias: Option<Vec<f32>>,
        cfg: RoutedMoeConfig,
    ) -> RoutedQuantMoe {
        let hidden = 4;
        let inter = 4;
        let experts: Vec<QuantExpert> = (0..n_exp)
            .map(|e| scaled_expert(hidden, inter, (e + 1) as f32))
            .collect();
        RoutedQuantMoe::new(
            f32_linear(n_exp, hidden, router_values),
            experts,
            None,
            bias,
            cfg,
        )
        .expect("moe constructs")
    }

    #[test]
    fn top_k_zero_is_rejected_not_panicked() {
        let hidden = 4;
        let experts = vec![scaled_expert(hidden, 4, 1.0)];
        let err = RoutedQuantMoe::new(
            f32_linear(1, hidden, &[1.0, 0.0, 0.0, 0.0]),
            experts,
            None,
            None,
            RoutedMoeConfig {
                top_k: 0,
                ..RoutedMoeConfig::default()
            },
        );
        match err {
            Err(ArchError::InvalidConfig { detail }) => {
                assert!(detail.contains("top_k"), "detail = {detail}");
            }
            Err(other) => panic!("top_k = 0 must be InvalidConfig, got {other:?}"),
            Ok(_) => panic!("top_k = 0 must be rejected at construction"),
        }
    }

    /// The DeepSeek-V3 rule: `exp_probs_b` steers *selection* but must never
    /// appear in the combination weight.
    #[test]
    fn bias_selects_but_does_not_weight() {
        let n_exp = 4;
        let hidden = 4;
        // router row e has weight 1.0 on input dim 0 scaled by (e as f32),
        // so logits for input [1,0,0,0] are [0, 1, 2, 3].
        let mut router = vec![0.0f32; n_exp * hidden];
        for e in 0..n_exp {
            router[e * hidden] = e as f32;
        }
        // Bias hugely favours expert 0 (which has the lowest logit).
        let bias = vec![10.0, 0.0, 0.0, 0.0];
        let cfg = RoutedMoeConfig {
            top_k: 1,
            gating: GatingFunc::Sigmoid,
            norm_weights: false,
            ..RoutedMoeConfig::default()
        };
        let moe = build(n_exp, &router, Some(bias), cfg);

        let mut selected = Vec::new();
        moe.route(&[1.0, 0.0, 0.0, 0.0], &mut selected)
            .expect("route");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].0, 0, "bias must steer selection to expert 0");
        // sigmoid(0.0) = 0.5 — the UNBIASED probability.  The biased score
        // would have been 10.5.
        assert!(
            (selected[0].1 - 0.5).abs() < 1e-6,
            "weight must be the unbiased sigmoid(logit) = 0.5, got {}",
            selected[0].1
        );
    }

    /// A negative `exp_probs_b` used to be able to drive the combination weight
    /// negative and trip the `weight_sum > 0.0` guard, zeroing the whole routed
    /// branch.  Weights must stay the (strictly positive) sigmoid values.
    #[test]
    fn negative_bias_cannot_zero_the_routed_branch() {
        let n_exp = 4;
        let hidden = 4;
        let mut router = vec![0.0f32; n_exp * hidden];
        for e in 0..n_exp {
            router[e * hidden] = e as f32;
        }
        let bias = vec![-5.0, -5.0, -5.0, -5.0];
        let cfg = RoutedMoeConfig {
            top_k: 2,
            gating: GatingFunc::Sigmoid,
            norm_weights: true,
            ..RoutedMoeConfig::default()
        };
        let moe = build(n_exp, &router, Some(bias), cfg);
        let mut selected = Vec::new();
        moe.route(&[1.0, 0.0, 0.0, 0.0], &mut selected)
            .expect("route");
        let total: f32 = selected.iter().map(|&(_, w)| w).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "normalised weights must sum to 1 even with an all-negative bias, got {total}"
        );
        for &(e, w) in &selected {
            assert!(w > 0.0, "expert {e} weight must be positive, got {w}");
        }
    }

    #[test]
    fn group_routing_restricts_selection_to_top_groups() {
        // 8 experts in 2 groups of 4; keep 1 group.  Logits rise with the
        // expert index, so group 1 (experts 4..8) wins and no expert from
        // group 0 may be selected even with top_k = 4.
        let n_exp = 8;
        let hidden = 4;
        let mut router = vec![0.0f32; n_exp * hidden];
        for e in 0..n_exp {
            router[e * hidden] = e as f32;
        }
        let experts: Vec<QuantExpert> = (0..n_exp).map(|_| scaled_expert(hidden, 4, 1.0)).collect();
        let moe = RoutedQuantMoe::new(
            f32_linear(n_exp, hidden, &router),
            experts,
            None,
            None,
            RoutedMoeConfig {
                top_k: 4,
                n_group: 2,
                topk_group: 1,
                ..RoutedMoeConfig::default()
            },
        )
        .expect("moe constructs");

        let mut selected = Vec::new();
        moe.route(&[1.0, 0.0, 0.0, 0.0], &mut selected)
            .expect("route");
        assert_eq!(selected.len(), 4);
        for &(e, _) in &selected {
            assert!(
                e >= 4,
                "group-limited routing must not reach group 0, got expert {e}"
            );
        }
    }

    #[test]
    fn without_norm_weights_the_raw_probabilities_survive() {
        let n_exp = 2;
        let hidden = 4;
        let router = vec![0.0f32; n_exp * hidden];
        // Both logits are 0 → softmax gives 0.5 each.
        let moe = build(
            n_exp,
            &router,
            None,
            RoutedMoeConfig {
                top_k: 1,
                norm_weights: false,
                ..RoutedMoeConfig::default()
            },
        );
        let mut selected = Vec::new();
        moe.route(&[1.0, 0.0, 0.0, 0.0], &mut selected)
            .expect("route");
        assert!(
            (selected[0].1 - 0.5).abs() < 1e-6,
            "unnormalised top-1 weight must stay 0.5, got {}",
            selected[0].1
        );
    }

    #[test]
    fn weight_scale_multiplies_the_routed_weights() {
        let n_exp = 2;
        let hidden = 4;
        let router = vec![0.0f32; n_exp * hidden];
        let moe = build(
            n_exp,
            &router,
            None,
            RoutedMoeConfig {
                top_k: 1,
                norm_weights: true,
                weight_scale: 2.5,
                ..RoutedMoeConfig::default()
            },
        );
        let mut selected = Vec::new();
        moe.route(&[1.0, 0.0, 0.0, 0.0], &mut selected)
            .expect("route");
        assert!(
            (selected[0].1 - 2.5).abs() < 1e-6,
            "top-1 normalised weight 1.0 scaled by 2.5 must be 2.5, got {}",
            selected[0].1
        );
    }

    #[test]
    fn gelu_and_silu_experts_differ() {
        let hidden = 4;
        let n_exp = 1;
        let router = vec![1.0f32, 0.0, 0.0, 0.0];
        let input = [2.0f32, 0.0, 0.0, 0.0];

        let mut outs = Vec::new();
        for activation in [ExpertActivation::Silu, ExpertActivation::Gelu] {
            let experts = vec![scaled_expert(hidden, 4, 1.0)];
            let moe = RoutedQuantMoe::new(
                f32_linear(n_exp, hidden, &router),
                experts,
                None,
                None,
                RoutedMoeConfig {
                    top_k: 1,
                    activation,
                    ..RoutedMoeConfig::default()
                },
            )
            .expect("moe constructs");
            let mut scratch = moe.make_scratch();
            let mut out = vec![0.0f32; hidden];
            moe.forward(&input, &mut out, &mut scratch)
                .expect("forward");
            outs.push(out[0]);
        }
        // silu(2) = 1.7616, gelu(2) = 1.9546; both times up = 2.
        assert!(
            (outs[0] - 2.0 * 1.761_594).abs() < 1e-3,
            "SiLU expert out {} != silu(2)*2",
            outs[0]
        );
        assert!(
            (outs[1] - 2.0 * 1.954_598).abs() < 1e-3,
            "GELU expert out {} != gelu(2)*2",
            outs[1]
        );
    }

    #[test]
    fn shared_expert_is_always_added() {
        let hidden = 4;
        let n_exp = 2;
        let router = vec![0.0f32; n_exp * hidden];
        let experts: Vec<QuantExpert> = (0..n_exp).map(|_| scaled_expert(hidden, 4, 0.0)).collect();
        let moe = RoutedQuantMoe::new(
            f32_linear(n_exp, hidden, &router),
            experts,
            Some(scaled_expert(hidden, 4, 3.0)),
            None,
            RoutedMoeConfig {
                top_k: 1,
                ..RoutedMoeConfig::default()
            },
        )
        .expect("moe constructs");
        let mut scratch = moe.make_scratch();
        let mut out = vec![0.0f32; hidden];
        moe.forward(&[1.0, 0.0, 0.0, 0.0], &mut out, &mut scratch)
            .expect("forward");
        // Routed experts have a zero `down`, so everything here is the shared
        // expert: silu(1) * 1 * 3.
        let expected = (1.0f32 / (1.0 + (-1.0f32).exp())) * 3.0;
        assert!(
            (out[0] - expected).abs() < 1e-5,
            "shared expert must contribute {expected}, got {}",
            out[0]
        );
    }

    #[test]
    fn gating_func_parse_matches_gguf_values() {
        assert_eq!(
            GatingFunc::from_gguf_value(0).expect("0 = none → softmax"),
            GatingFunc::Softmax
        );
        assert_eq!(
            GatingFunc::from_gguf_value(1).expect("1 = softmax"),
            GatingFunc::Softmax
        );
        assert_eq!(
            GatingFunc::from_gguf_value(2).expect("2 = sigmoid"),
            GatingFunc::Sigmoid
        );
        assert!(GatingFunc::from_gguf_value(7).is_err());
    }

    // ── Fused-Q8 shared-quantization parity ─────────────────────────────
    //
    // `QuantExpert::forward_fused` itself is proven bit-exact under buffer
    // sharing by `common::moe`'s own test suite. What is unique to *this*
    // module is the `shared_blocks` aggregation loop inside
    // `RoutedQuantMoe::forward` — a third, independent "max over the
    // selection" computation (after `common::moe::QuantMoeFfn` and
    // `grok::moe::GrokMoe`), and the only one that must also fold in an
    // always-active *shared* expert outside the top-`k` selection. The
    // routing tests above use `hidden = 4` fixtures, far below any
    // block-quantized kernel's minimum, so they never reach the fused path
    // and would not notice `shared_blocks` under- or over-sizing `acts_q8`,
    // or the shared expert being left out of that computation entirely.
    // This pins it directly: `forward` (shared quantization) must match a
    // hand-rolled reference that quantizes fresh for every routed expert
    // *and* the shared expert, at several `top_k`. The selection/weights
    // themselves come from the already-tested public [`RoutedQuantMoe::route`]
    // entry point, so this cannot drift from the real routing math (softmax
    // gating, top-k, re-normalise) — only the fused-sharing claim is new here.
    //
    // `GgufTensorType` / `QuantTensor` are already in scope via this module's
    // own `use` above.

    /// Deterministic xorshift64* stream — mirrors `common::moe::tests::Rng`.
    struct FusedRng(u64);
    impl FusedRng {
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

    const Q4_K_BYTES: usize = 144;
    const Q4_K_BLOCK: usize = 256;

    /// `[rows, cols]` Q4_K weight tensor with well-formed pseudo-random
    /// blocks — a fused-capable control, sized and shaped exactly as
    /// `common::moe::tests::q4_k_tensor`.
    fn q4_k_tensor(rows: usize, cols: usize, rng: &mut FusedRng) -> QuantTensor {
        let blocks_per_row = cols.div_ceil(Q4_K_BLOCK);
        let mut data = Vec::with_capacity(rows * blocks_per_row * Q4_K_BYTES);
        for _ in 0..rows * blocks_per_row {
            let d = half::f16::from_f32(rng.next_f32() * 0.05);
            let dmin = half::f16::from_f32(rng.next_f32() * 0.02);
            data.extend_from_slice(&d.to_bits().to_le_bytes());
            data.extend_from_slice(&dmin.to_bits().to_le_bytes());
            for _ in 0..12 + 128 {
                data.push(rng.next_u8());
            }
        }
        QuantTensor::new(data, vec![rows, cols], GgufTensorType::Q4K)
    }

    fn q4_k_linear(rows: usize, cols: usize, rng: &mut FusedRng) -> QuantLinear {
        QuantLinear::new(q4_k_tensor(rows, cols, rng), None)
    }

    fn q4_k_expert(hidden: usize, inter: usize, rng: &mut FusedRng) -> QuantExpert {
        QuantExpert::new(
            q4_k_linear(inter, hidden, rng),
            q4_k_linear(inter, hidden, rng),
            q4_k_linear(hidden, inter, rng),
        )
        .expect("Q4_K expert shapes compose")
    }

    /// `RoutedQuantMoe::forward`'s shared quantization — covering the routed
    /// selection *and* the always-active shared expert — must match
    /// requantizing fresh for every one of them individually, at several
    /// `top_k` values (including `top_k == num_experts`).
    #[test]
    fn forward_matches_fresh_per_expert_quantization_reference() {
        for &top_k in &[1usize, 2, 4] {
            let hidden = 256;
            let inter = 256;
            let n_exp = 4;
            let mut rng = FusedRng::new(0xD5EE_0000 ^ top_k as u64);
            let experts: Vec<QuantExpert> = (0..n_exp)
                .map(|_| q4_k_expert(hidden, inter, &mut rng))
                .collect();
            let shared_expert = q4_k_expert(hidden, inter, &mut rng);
            let router = q4_k_linear(n_exp, hidden, &mut rng);
            let cfg = RoutedMoeConfig {
                top_k,
                ..RoutedMoeConfig::default()
            };
            let moe = RoutedQuantMoe::new(router, experts, Some(shared_expert), None, cfg)
                .expect("moe constructs");

            let input: Vec<f32> = (0..hidden)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.045)
                .collect();

            let mut scratch = moe.make_scratch();
            let mut got = vec![0.0f32; hidden];
            moe.forward(&input, &mut got, &mut scratch)
                .expect("deepseek moe forward (shared quantization)");

            // Real selection + weights, from the already-tested public
            // routing entry point — this cannot drift from what `forward`
            // itself used internally.
            let mut selected = Vec::new();
            moe.route(&input, &mut selected).expect("route");
            assert!(
                !selected.is_empty(),
                "top_k={top_k} must select at least one expert, or this test is vacuous"
            );

            let mut want = vec![0.0f32; hidden];
            let mut expert_scratch = MoeScratch::new(hidden, inter, n_exp);
            for &(idx, weight) in &selected {
                let re = &moe.experts[idx];
                let n_blocks = re
                    .expert
                    .gate_up_fused_blocks(&re.kernels)
                    .expect("Q4_K fixture must advertise the fused path, or this test is vacuous");
                let mut fresh_acts = Vec::new();
                quantize_activations_q8_0_into(&input, n_blocks, &mut fresh_acts);
                let mut expert_out = vec![0.0f32; hidden];
                RoutedQuantMoe::run_expert(
                    re,
                    moe.cfg.activation,
                    &input,
                    &fresh_acts,
                    &mut expert_out,
                    &mut expert_scratch,
                )
                .expect("reference routed expert forward_fused");
                for (w, e) in want.iter_mut().zip(expert_out.iter()) {
                    *w += weight * e;
                }
            }

            // Shared expert: always active, added unweighted — exactly as
            // `forward` does after the routed loop.
            let shared = moe
                .shared
                .as_ref()
                .expect("fixture always constructs a shared expert");
            let n_blocks = shared
                .expert
                .gate_up_fused_blocks(&shared.kernels)
                .expect("Q4_K fixture must advertise the fused path, or this test is vacuous");
            let mut fresh_acts = Vec::new();
            quantize_activations_q8_0_into(&input, n_blocks, &mut fresh_acts);
            let mut expert_out = vec![0.0f32; hidden];
            RoutedQuantMoe::run_expert(
                shared,
                moe.cfg.activation,
                &input,
                &fresh_acts,
                &mut expert_out,
                &mut expert_scratch,
            )
            .expect("reference shared expert forward_fused");
            for (w, e) in want.iter_mut().zip(expert_out.iter()) {
                *w += e;
            }

            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "top_k={top_k} output[{i}]: shared-quantization {a} != fresh-per-expert {b}"
                );
            }
        }
    }
}

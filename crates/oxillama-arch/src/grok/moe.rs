//! Grok-1's sparse MoE FFN: identical routing to Mixtral, **GELU** experts.
//!
//! `build_grok` (`src/models/grok.cpp`) calls
//!
//! ```text
//! build_moe_ffn(cur, ffn_gate_inp, ffn_up_exps, ffn_gate_exps, ffn_down_exps,
//!               nullptr, n_expert, n_expert_used,
//!               LLM_FFN_GELU, /* norm_w = */ true,
//!               /* scale_w = */ false, 0.0,
//!               LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
//! ```
//!
//! `LLM_FFN_GELU` with a gate resolves to `ggml_geglu_split(gate, up)`, i.e.
//! `gelu(gate) * up`.  This crate's [`QuantMoeFfn`](crate::common::moe::QuantMoeFfn)
//! hard-codes SiLU, and Grok used to run through `DeepSeekExpert::forward`
//! whose SiLU is likewise hard-coded — so every expert in every layer applied
//! the wrong non-linearity.
//!
//! Routing (softmax → top-k → re-normalise) is byte-for-byte the same as
//! `QuantMoeFfn`'s; only `GrokMoe::activation` differs.  Weights stay in their
//! GGUF quantized form as shared mmap views.

use oxillama_quant::{quantize_activations_q8_0_into, KernelDispatcher, QuantKernel};

use crate::common::gelu::gelu;
use crate::common::linear::QuantLinear;
use crate::common::moe::{ExpertKernels, MoeScratch, QuantExpert};
use crate::error::{ArchError, ArchResult};

/// Smallest positive normal `f16` — llama.cpp's
/// `ggml_clamp(weights_sum, 6.103515625e-5, INFINITY)` divisor floor.
const WEIGHT_SUM_FLOOR: f32 = 6.103_515_6e-5;

/// A sparse MoE FFN with GELU-gated quantized experts.
pub struct GrokMoe {
    router: QuantLinear,
    router_kernel: Box<dyn QuantKernel>,
    experts: Vec<QuantExpert>,
    expert_kernels: Vec<ExpertKernels>,
    top_k: usize,
    hidden_size: usize,
}

impl GrokMoe {
    /// Assemble the layer and resolve every kernel up front.
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidConfig`] when there are no experts or
    ///   `top_k == 0` (`grok.expert_used_count` comes from the file).
    /// * [`ArchError::InvalidShape`] when the router does not match the pool.
    /// * [`ArchError::Quant`] for an unsupported quantization type.
    pub fn new(router: QuantLinear, experts: Vec<QuantExpert>, top_k: usize) -> ArchResult<Self> {
        if experts.is_empty() {
            // llama.cpp: `throw std::runtime_error("Grok model cannot have zero experts")`.
            return Err(ArchError::InvalidConfig {
                detail: "Grok model cannot have zero experts".to_string(),
            });
        }
        if top_k == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "Grok MoE top_k must be >= 1 (expert_used_count = 0 in the GGUF)"
                    .to_string(),
            });
        }
        let hidden_size = experts[0].hidden_size();
        if router.out_features != experts.len() || router.in_features != hidden_size {
            return Err(ArchError::InvalidShape {
                name: "grok_moe.router".to_string(),
                expected: vec![experts.len(), hidden_size],
                got: vec![router.out_features, router.in_features],
            });
        }
        let dispatcher = KernelDispatcher::new();
        let router_kernel = dispatcher.get_kernel(router.weight.tensor_type)?;
        let expert_kernels = experts
            .iter()
            .map(ExpertKernels::for_expert)
            .collect::<ArchResult<Vec<_>>>()?;
        Ok(Self {
            router,
            router_kernel,
            experts,
            expert_kernels,
            top_k,
            hidden_size,
        })
    }

    /// Model hidden dimension.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of experts in the pool.
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Allocate scratch sized for this layer.
    pub fn make_scratch(&self) -> MoeScratch {
        let inter = self
            .experts
            .first()
            .map_or(0, QuantExpert::intermediate_size);
        MoeScratch::new(self.hidden_size, inter, self.experts.len())
    }

    /// `gelu(gate(x)) * up(x)` → `down(...)`, through the fused Q8_0-activation
    /// path wherever a projection's kernel supports it.
    ///
    /// `acts_q8` is a shared Q8_0 image of `input`, quantized once by
    /// [`Self::forward`] for however many experts top-`k` selects — see
    /// [`QuantExpert::forward_fused`] for why sharing it across experts is
    /// exact, not an approximation.
    fn run_expert(
        expert: &QuantExpert,
        kernels: &ExpertKernels,
        input: &[f32],
        acts_q8: &[u8],
        output: &mut [f32],
        scratch: &mut MoeScratch,
    ) -> ArchResult<()> {
        expert.forward_fused(kernels, input, acts_q8, gelu, output, scratch)
    }

    /// Route `input` to the top-`k` experts and write the combination into
    /// `output` (**overwritten**, not accumulated).
    ///
    /// # Errors
    ///
    /// Propagates the router and expert GEMV failures.
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut MoeScratch,
    ) -> ArchResult<()> {
        let n_exp = self.experts.len();
        scratch.ensure_router(n_exp);
        scratch.ensure_output(self.hidden_size);

        self.router
            .forward(
                &*self.router_kernel,
                input,
                &mut scratch.router_logits[..n_exp],
            )
            .map_err(ArchError::from)?;

        // Numerically stable softmax over the router logits.
        {
            let probs = &mut scratch.router_logits[..n_exp];
            let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for p in probs.iter_mut() {
                *p = (*p - max).exp();
                sum += *p;
            }
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            } else {
                let uniform = 1.0 / n_exp as f32;
                for p in probs.iter_mut() {
                    *p = uniform;
                }
            }
        }

        let order = &mut scratch.order;
        order.clear();
        order.extend(0..n_exp);
        let probs = &scratch.router_logits[..n_exp];
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let k = self.top_k.min(n_exp);
        let sum: f32 = scratch.order[..k]
            .iter()
            .map(|&i| scratch.router_logits[i])
            .sum();
        let denom = sum.max(WEIGHT_SUM_FLOOR);
        let selection: Vec<(usize, f32)> = scratch.order[..k]
            .iter()
            .map(|&i| (i, scratch.router_logits[i] / denom))
            .collect();

        output.fill(0.0);

        // Every selected expert's `gate`/`up` reads this exact `input`
        // vector, so it is quantized to Q8_0 once and shared — see
        // `QuantExpert::forward_fused`.
        let mut shared_blocks: Option<usize> = None;
        for &(idx, _) in &selection {
            if let Some(b) = self.experts[idx].gate_up_fused_blocks(&self.expert_kernels[idx]) {
                shared_blocks = Some(shared_blocks.map_or(b, |m: usize| m.max(b)));
            }
        }
        let mut acts_q8 = std::mem::take(&mut scratch.acts_q8);
        if let Some(n_blocks) = shared_blocks {
            quantize_activations_q8_0_into(input, n_blocks, &mut acts_q8);
        }

        let mut expert_out = std::mem::take(&mut scratch.expert_out);
        let mut result = Ok(());
        for (idx, weight) in selection {
            match Self::run_expert(
                &self.experts[idx],
                &self.expert_kernels[idx],
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
        scratch.expert_out = expert_out;
        scratch.acts_q8 = acts_q8;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grok::testkit::TinyWeights;

    /// One expert with `gate = up = I`, `down = I`: the layer output for a
    /// one-hot input is exactly the gated activation.
    fn identity_expert(hidden: usize, w: &mut TinyWeights) -> QuantExpert {
        let mut eye = vec![0.0f32; hidden * hidden];
        for i in 0..hidden {
            eye[i * hidden + i] = 1.0;
        }
        QuantExpert::new(
            w.linear_from(hidden, hidden, &eye),
            w.linear_from(hidden, hidden, &eye),
            w.linear_from(hidden, hidden, &eye),
        )
        .expect("expert shapes compose")
    }

    /// `build_grok` passes `LLM_FFN_GELU`; the experts used to run SiLU.
    #[test]
    fn experts_use_gelu_not_silu() {
        let hidden = 4;
        let mut w = TinyWeights::new(3);
        let router = w.linear_from(1, hidden, &[1.0, 0.0, 0.0, 0.0]);
        let moe =
            GrokMoe::new(router, vec![identity_expert(hidden, &mut w)], 1).expect("MoE constructs");
        let mut scratch = moe.make_scratch();
        let mut out = vec![0.0f32; hidden];
        let x = [2.0f32, 0.0, 0.0, 0.0];
        moe.forward(&x, &mut out, &mut scratch).expect("forward");

        let want_gelu = gelu(2.0) * 2.0; // gelu(gate) * up
        let want_silu = (2.0f32 / (1.0 + (-2.0f32).exp())) * 2.0;
        assert!(
            (out[0] - want_gelu).abs() < 1e-4,
            "expected geglu {want_gelu}, got {}",
            out[0]
        );
        assert!(
            (want_gelu - want_silu).abs() > 1e-2,
            "the test only discriminates if GELU and SiLU differ here"
        );
    }

    #[test]
    fn zero_top_k_is_rejected() {
        let hidden = 4;
        let mut w = TinyWeights::new(3);
        let router = w.linear_from(1, hidden, &[1.0, 0.0, 0.0, 0.0]);
        assert!(GrokMoe::new(router, vec![identity_expert(hidden, &mut w)], 0).is_err());
    }

    #[test]
    fn zero_experts_is_rejected() {
        let mut w = TinyWeights::new(3);
        let router = w.linear(1, 4);
        assert!(GrokMoe::new(router, Vec::new(), 1).is_err());
    }

    // ── Fused-Q8 shared-quantization parity ─────────────────────────────
    //
    // `QuantExpert::forward_fused` itself is proven bit-exact under buffer
    // sharing by `common::moe`'s own test suite. What is unique to *this*
    // module is the `shared_blocks` aggregation loop inside `GrokMoe::forward`
    // (a second, independent "max over the selection" computation) — the
    // GELU/routing tests above use `hidden = 4` `TinyWeights` fixtures, far
    // below any block-quantized kernel's minimum, so they never reach the
    // fused path and would not notice that loop under- or over-sizing
    // `acts_q8`. This pins it directly: `forward` (shared quantization) must
    // match a hand-rolled reference that quantizes fresh per selected expert,
    // at several `top_k` — mirroring
    // `common::moe::tests::quant_moe_ffn_forward_matches_fresh_per_expert_quantization_reference`.

    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

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

    /// `GrokMoe::forward`'s shared quantization must match requantizing fresh
    /// for each selected expert, at several `top_k` values (including
    /// `top_k == num_experts`).
    #[test]
    fn forward_matches_fresh_per_expert_quantization_reference() {
        for &top_k in &[1usize, 2, 4] {
            let hidden = 256;
            let inter = 256;
            let n_exp = 4;
            let mut rng = FusedRng::new(0x6A0C_0000 ^ top_k as u64);
            let experts: Vec<QuantExpert> = (0..n_exp)
                .map(|_| q4_k_expert(hidden, inter, &mut rng))
                .collect();
            let router = q4_k_linear(n_exp, hidden, &mut rng);
            let moe = GrokMoe::new(router, experts, top_k).expect("moe constructs");

            let input: Vec<f32> = (0..hidden)
                .map(|i| ((i % 19) as f32 - 9.0) * 0.04)
                .collect();

            let mut scratch = moe.make_scratch();
            let mut got = vec![0.0f32; hidden];
            moe.forward(&input, &mut got, &mut scratch)
                .expect("grok moe forward (shared quantization)");

            // Read back the real selection: `forward` leaves `order`/
            // `router_logits` populated by the routing step, so this cannot
            // drift from the real routing logic.
            let k = top_k.min(n_exp);
            let sum: f32 = scratch.order[..k]
                .iter()
                .map(|&i| scratch.router_logits[i])
                .sum();
            let denom = sum.max(WEIGHT_SUM_FLOOR);

            let mut want = vec![0.0f32; hidden];
            let mut expert_scratch = MoeScratch::new(hidden, inter, n_exp);
            for &idx in &scratch.order[..k] {
                let weight = scratch.router_logits[idx] / denom;
                let expert = &moe.experts[idx];
                let kernels = &moe.expert_kernels[idx];
                let n_blocks = expert
                    .gate_up_fused_blocks(kernels)
                    .expect("Q4_K fixture must advertise the fused path, or this test is vacuous");
                let mut fresh_acts = Vec::new();
                quantize_activations_q8_0_into(&input, n_blocks, &mut fresh_acts);
                let mut expert_out = vec![0.0f32; hidden];
                expert
                    .forward_fused(
                        kernels,
                        &input,
                        &fresh_acts,
                        gelu,
                        &mut expert_out,
                        &mut expert_scratch,
                    )
                    .expect("reference expert forward_fused");
                for (w, e) in want.iter_mut().zip(expert_out.iter()) {
                    *w += weight * e;
                }
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

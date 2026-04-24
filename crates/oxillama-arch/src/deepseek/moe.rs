//! DeepSeek-V2 Mixture-of-Experts (MoE) FFN layer.
//!
//! Implements the DeepSeek-V2 sparse MoE where:
//! - `n_shared_experts` experts always run (no routing, summed into output).
//! - `n_routed_experts` experts are gated; only the `top_k` highest-scoring ones
//!   contribute to the output per token.
//!
//! Routing score computation supports two modes:
//! - [`ScoringMode::Softmax`]: standard softmax over router logits.
//! - [`ScoringMode::SigmoidWithBias`]: element-wise sigmoid applied to logits,
//!   then bias added, used by DeepSeek-V3.
//!
//! All expert weights are stored as `f32` (dequantized at load time).
//! The expert SwiGLU FFN follows the same gate/up/down layout as the
//! standard LLaMA FFN in `crate::common::moe`.

use crate::error::{ArchError, ArchResult};

// ─── Routing mode ─────────────────────────────────────────────────────────────

/// Scoring mode for the expert router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringMode {
    /// Softmax over router logits then top-k selection and renormalisation.
    Softmax,
    /// Element-wise sigmoid on logits, then add per-expert bias (DeepSeek-V3).
    SigmoidWithBias,
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for one DeepSeek MoE FFN.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Intermediate size of each individual expert's SwiGLU FFN.
    pub expert_intermediate_size: usize,
    /// Number of shared experts (always active per token).
    pub n_shared_experts: usize,
    /// Total number of routed experts in the pool.
    pub n_routed_experts: usize,
    /// Number of routed experts activated per token (top-k).
    pub top_k: usize,
    /// Scaling factor applied to the routing scores before accumulation.
    pub routed_scaling_factor: f32,
    /// Scoring mode (softmax vs sigmoid+bias).
    pub scoring_mode: ScoringMode,
    /// Intermediate size of each shared expert's SwiGLU FFN.
    pub shared_expert_intermediate_size: usize,
}

// ─── Per-expert SwiGLU weights ────────────────────────────────────────────────

/// Weights for a single SwiGLU expert (gate + up + down projections).
///
/// Layout follows row-major GGUF convention:
/// - `gate`:  `[intermediate_size × hidden_size]`
/// - `up`:    `[intermediate_size × hidden_size]`
/// - `down`:  `[hidden_size × intermediate_size]`
///
/// All weights are stored as `f32` (pre-dequantised at model load time).
pub struct DeepSeekExpert {
    /// Gate projection: `[intermediate_size, hidden_size]` flattened row-major.
    pub gate: Vec<f32>,
    /// Up projection: `[intermediate_size, hidden_size]` flattened row-major.
    pub up: Vec<f32>,
    /// Down projection: `[hidden_size, intermediate_size]` flattened row-major.
    pub down: Vec<f32>,
    /// Model hidden dimension (= number of input/output features).
    pub hidden_size: usize,
    /// Expert FFN intermediate dimension.
    pub intermediate_size: usize,
}

impl DeepSeekExpert {
    /// Validate that all weight buffers have the expected sizes.
    fn validate(&self) -> ArchResult<()> {
        let exp_gate_up = self.intermediate_size * self.hidden_size;
        let exp_down = self.hidden_size * self.intermediate_size;
        if self.gate.len() != exp_gate_up {
            return Err(ArchError::InvalidShape {
                name: "expert.gate".to_string(),
                expected: vec![self.intermediate_size, self.hidden_size],
                got: vec![self.gate.len()],
            });
        }
        if self.up.len() != exp_gate_up {
            return Err(ArchError::InvalidShape {
                name: "expert.up".to_string(),
                expected: vec![self.intermediate_size, self.hidden_size],
                got: vec![self.up.len()],
            });
        }
        if self.down.len() != exp_down {
            return Err(ArchError::InvalidShape {
                name: "expert.down".to_string(),
                expected: vec![self.hidden_size, self.intermediate_size],
                got: vec![self.down.len()],
            });
        }
        Ok(())
    }

    /// Forward pass: SwiGLU FFN.
    ///
    /// Computes:
    /// ```text
    /// gate_vec = silu(W_gate @ input)
    /// up_vec   = W_up   @ input
    /// output   = W_down @ (gate_vec ⊙ up_vec)
    /// ```
    ///
    /// # Arguments
    /// * `input`  - Slice of length `hidden_size`.
    /// * `output` - Mutable slice of length `hidden_size` (overwritten).
    ///
    /// # Errors
    /// Returns [`ArchError::InvalidShape`] if weight buffers are inconsistent.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ArchResult<()> {
        self.validate()?;

        let n = self.intermediate_size;
        let h = self.hidden_size;

        let mut gate_vec = vec![0.0f32; n];
        let mut up_vec = vec![0.0f32; n];

        // Gate GEMV: gate_vec[i] = dot(gate[i, :], input)
        for (i, g) in gate_vec.iter_mut().enumerate() {
            let row = &self.gate[i * h..(i + 1) * h];
            *g = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
        }

        // SiLU activation: silu(x) = x * sigmoid(x)
        for g in gate_vec.iter_mut() {
            let sigmoid = 1.0 / (1.0 + (-*g).exp());
            *g *= sigmoid;
        }

        // Up GEMV: up_vec[i] = dot(up[i, :], input)
        for (i, u) in up_vec.iter_mut().enumerate() {
            let row = &self.up[i * h..(i + 1) * h];
            *u = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
        }

        // Element-wise product: gate_vec *= up_vec
        for (g, u) in gate_vec.iter_mut().zip(up_vec.iter()) {
            *g *= u;
        }

        // Down GEMV: output[j] = dot(down[j, :], gate_vec)
        for (j, o) in output.iter_mut().enumerate() {
            let row = &self.down[j * n..(j + 1) * n];
            *o = row.iter().zip(gate_vec.iter()).map(|(w, x)| w * x).sum();
        }

        Ok(())
    }
}

// ─── MoE FFN weights ──────────────────────────────────────────────────────────

/// Weights for one DeepSeek MoE FFN layer.
///
/// Holds the router projection, the pool of routed experts, the shared experts,
/// and an optional per-expert bias (used in `SigmoidWithBias` mode).
pub struct MoeWeights {
    /// Router projection: `[n_routed_experts, hidden_size]` row-major f32.
    pub router: Vec<f32>,
    /// Pool of routed experts (length = `n_routed_experts`).
    pub routed_experts: Vec<DeepSeekExpert>,
    /// Shared experts (always active, length = `n_shared_experts`).
    pub shared_experts: Vec<DeepSeekExpert>,
    /// Optional per-expert bias used in `SigmoidWithBias` scoring mode.
    /// Length must equal `n_routed_experts` when present.
    pub expert_bias: Option<Vec<f32>>,
}

// ─── Public forward function ──────────────────────────────────────────────────

/// Run the DeepSeek MoE FFN forward pass for a single token.
///
/// # Arguments
/// * `x`       - Input hidden state, slice of length `cfg.hidden_size`.
/// * `weights` - MoE layer weights.
/// * `cfg`     - MoE configuration.
///
/// # Returns
/// Output hidden state (same length as `x`), accumulated from shared and
/// selected routed experts.
///
/// # Errors
/// Returns `ArchError` on shape mismatches or configuration errors.
pub fn moe_forward(x: &[f32], weights: &MoeWeights, cfg: &MoeConfig) -> ArchResult<Vec<f32>> {
    if x.len() != cfg.hidden_size {
        return Err(ArchError::InvalidShape {
            name: "moe_forward input".to_string(),
            expected: vec![cfg.hidden_size],
            got: vec![x.len()],
        });
    }
    if weights.routed_experts.len() != cfg.n_routed_experts {
        return Err(ArchError::InvalidConfig {
            detail: format!(
                "moe_forward: expected {} routed experts, got {}",
                cfg.n_routed_experts,
                weights.routed_experts.len()
            ),
        });
    }
    if weights.shared_experts.len() != cfg.n_shared_experts {
        return Err(ArchError::InvalidConfig {
            detail: format!(
                "moe_forward: expected {} shared experts, got {}",
                cfg.n_shared_experts,
                weights.shared_experts.len()
            ),
        });
    }
    if cfg.top_k > cfg.n_routed_experts {
        return Err(ArchError::InvalidConfig {
            detail: format!(
                "moe_forward: top_k={} exceeds n_routed_experts={}",
                cfg.top_k, cfg.n_routed_experts
            ),
        });
    }

    let mut output = vec![0.0f32; cfg.hidden_size];
    let mut expert_out = vec![0.0f32; cfg.hidden_size];

    // ── Shared experts (always active) ────────────────────────────────────────
    for shared_expert in &weights.shared_experts {
        let mut shared_out = vec![0.0f32; cfg.hidden_size];
        shared_expert.forward(x, &mut shared_out)?;
        for (o, s) in output.iter_mut().zip(shared_out.iter()) {
            *o += s;
        }
    }

    // ── Routing scores ────────────────────────────────────────────────────────
    // Router GEMV: scores[e] = dot(router[e, :], x)
    let n_experts = cfg.n_routed_experts;
    let h = cfg.hidden_size;
    let mut scores = vec![0.0f32; n_experts];

    for (e, score) in scores.iter_mut().enumerate() {
        let row = &weights.router[e * h..(e + 1) * h];
        *score = row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
    }

    // Apply scoring function
    match cfg.scoring_mode {
        ScoringMode::Softmax => {
            softmax_inplace(&mut scores);
        }
        ScoringMode::SigmoidWithBias => {
            // Sigmoid element-wise
            for s in scores.iter_mut() {
                *s = 1.0 / (1.0 + (-*s).exp());
            }
            // Add per-expert bias if available
            if let Some(ref bias) = weights.expert_bias {
                if bias.len() != n_experts {
                    return Err(ArchError::InvalidShape {
                        name: "expert_bias".to_string(),
                        expected: vec![n_experts],
                        got: vec![bias.len()],
                    });
                }
                for (s, &b) in scores.iter_mut().zip(bias.iter()) {
                    *s += b;
                }
            }
        }
    }

    // ── Top-k selection ───────────────────────────────────────────────────────
    // Select the top_k experts by score (partial sort via linear scan).
    let mut selected: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    // Partial sort: move top_k highest scores to the front.
    selected.select_nth_unstable_by(cfg.top_k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_selected = selected[..cfg.top_k].to_vec();

    // Normalise routing weights for the selected experts.
    let weight_sum: f32 = top_selected.iter().map(|&(_, w)| w).sum();
    let inv_weight_sum = if weight_sum > 0.0 {
        1.0 / weight_sum
    } else {
        0.0
    };

    // ── Routed expert accumulation ────────────────────────────────────────────
    for (expert_idx, raw_weight) in top_selected {
        let normalised_weight = raw_weight * inv_weight_sum * cfg.routed_scaling_factor;
        let expert = &weights.routed_experts[expert_idx];
        expert_out.fill(0.0);
        expert.forward(x, &mut expert_out)?;
        for (o, e) in output.iter_mut().zip(expert_out.iter()) {
            *o += normalised_weight * e;
        }
    }

    Ok(output)
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
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal LCG pseudo-random number generator (deterministic, no deps).
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Mask mantissa bits only, force exponent=127 → value in [1.0, 2.0)
            let mantissa = (self.state >> 33) as u32 & 0x007f_ffff;
            let bits = mantissa | 0x3f80_0000u32;
            (f32::from_bits(bits) - 1.5) * 0.1 // in [-0.05, 0.05)
        }

        fn fill(&mut self, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.next_f32();
            }
        }
    }

    fn make_expert(lcg: &mut Lcg, hidden: usize, intermediate: usize) -> DeepSeekExpert {
        let mut gate = vec![0.0f32; intermediate * hidden];
        let mut up = vec![0.0f32; intermediate * hidden];
        let mut down = vec![0.0f32; hidden * intermediate];
        lcg.fill(&mut gate);
        lcg.fill(&mut up);
        lcg.fill(&mut down);
        DeepSeekExpert {
            gate,
            up,
            down,
            hidden_size: hidden,
            intermediate_size: intermediate,
        }
    }

    fn make_weights(lcg: &mut Lcg, cfg: &MoeConfig, mode: ScoringMode) -> MoeWeights {
        let h = cfg.hidden_size;
        let n = cfg.n_routed_experts;

        let mut router = vec![0.0f32; n * h];
        lcg.fill(&mut router);

        let routed_experts = (0..n)
            .map(|_| make_expert(lcg, h, cfg.expert_intermediate_size))
            .collect();

        let shared_experts = (0..cfg.n_shared_experts)
            .map(|_| make_expert(lcg, h, cfg.shared_expert_intermediate_size))
            .collect();

        let expert_bias = if mode == ScoringMode::SigmoidWithBias {
            let mut bias = vec![0.0f32; n];
            lcg.fill(&mut bias);
            Some(bias)
        } else {
            None
        };

        MoeWeights {
            router,
            routed_experts,
            shared_experts,
            expert_bias,
        }
    }

    /// (a) topk_selects_largest: top-1 selection must pick the expert with
    /// the highest router score.
    #[test]
    fn topk_selects_largest() {
        let n_routed = 4;
        let h = 4;

        // Craft a router where expert 2 gets a very high score for input [1,0,0,0].
        // router[e, :] × input = router[e, 0].
        // Set row 2 high, others low.
        let mut router = vec![0.0f32; n_routed * h];
        router[0] = -1.0; // expert 0 first weight: score ≈ -1 for input [1,0,0,0]
        router[h] = 0.0; // expert 1 first weight: score = 0
        router[2 * h] = 5.0; // expert 2: score = 5 (highest)
        router[3 * h] = 1.0; // expert 3: score = 1

        // Build trivial experts (identity-like, just need to not error).
        let make_trivial = |id: f32| -> DeepSeekExpert {
            let inter = 4;
            let mut gate = vec![0.0f32; inter * h];
            let mut up = vec![0.0f32; inter * h];
            // down is identity-ish: diagonal = 1.0
            let mut down = vec![0.0f32; h * inter];
            // small weight on [0,0] so output is proportional to id
            gate[0] = id;
            up[0] = 1.0;
            down[0] = 1.0;
            DeepSeekExpert {
                gate,
                up,
                down,
                hidden_size: h,
                intermediate_size: inter,
            }
        };

        let routed_experts = (0..n_routed).map(|e| make_trivial(e as f32)).collect();

        let weights = MoeWeights {
            router,
            routed_experts,
            shared_experts: vec![],
            expert_bias: None,
        };

        let cfg = MoeConfig {
            hidden_size: h,
            expert_intermediate_size: 4,
            n_shared_experts: 0,
            n_routed_experts: n_routed,
            top_k: 1,
            routed_scaling_factor: 1.0,
            scoring_mode: ScoringMode::Softmax,
            shared_expert_intermediate_size: 4,
        };

        let x = vec![1.0f32, 0.0, 0.0, 0.0];
        let out = moe_forward(&x, &weights, &cfg).expect("moe_forward must succeed");

        // Expert 2 is selected. Its gate[0] = 2.0, up[0] = 1.0, down[0] = 1.0.
        // gate_vec[0] = silu(2.0 * 1.0) ≈ silu(2.0) = 2 * sigmoid(2) ≈ 1.761
        // up_vec[0] = 1.0 * 1.0 = 1.0
        // prod = 1.761 * 1.0 = 1.761
        // out[0] = 1.761 * down[0] = 1.761 * 1.0 (down[0,0] = 1.0)
        // The key assertion is that expert 2 was selected and contributed.
        // We verify by checking the output is non-trivially distinct from the
        // result we'd get if expert 0 (gate=0) or expert 1 (gate=1) were chosen.
        let silu_2 = 2.0f32 / (1.0 + (-2.0f32).exp());
        let expert2_out0 = silu_2 * 1.0; // gate * up for dim 0
        assert!(
            (out[0] - expert2_out0).abs() < 1e-4,
            "top-1 should select expert 2; out[0]={} expected≈{expert2_out0}",
            out[0]
        );
    }

    /// (b) softmax_sums_to_one: router softmax weights over all experts sum to 1.
    #[test]
    fn softmax_sums_to_one() {
        let mut scores = vec![1.0f32, 2.0, 3.0, -1.0, 0.5];
        softmax_inplace(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax must sum to 1.0, got {sum}"
        );
        for &s in &scores {
            assert!(s >= 0.0, "softmax values must be non-negative, got {s}");
        }
    }

    /// (c) shared_expert_always_active: shared experts run regardless of routing.
    ///
    /// Uses zero routed experts (top_k=0 is invalid — we use n_shared=1, n_routed=1,
    /// top_k=1 but disable their contribution by zeroing the expert output,
    /// and assert the shared expert's contribution is present).
    ///
    /// Concretely: shared expert is a scalar-multiply-by-2 (gate=1, up=1, down=2).
    /// Routed expert is all-zeros (contributes nothing meaningful).
    /// Output[0] must contain the shared expert contribution.
    #[test]
    fn shared_expert_always_active() {
        let h = 4;
        let inter = 4;

        // Shared expert: gate[0,0]=1, up[0,0]=1, down[0,0]=2 → out[0] ≈ silu(1)*2
        let mut shared_gate = vec![0.0f32; inter * h];
        let mut shared_up = vec![0.0f32; inter * h];
        let mut shared_down = vec![0.0f32; h * inter];
        shared_gate[0] = 1.0;
        shared_up[0] = 1.0;
        shared_down[0] = 2.0;
        let shared_expert = DeepSeekExpert {
            gate: shared_gate,
            up: shared_up,
            down: shared_down,
            hidden_size: h,
            intermediate_size: inter,
        };

        // Routed expert: all zeros → zero output
        let zero_expert = DeepSeekExpert {
            gate: vec![0.0f32; inter * h],
            up: vec![0.0f32; inter * h],
            down: vec![0.0f32; h * inter],
            hidden_size: h,
            intermediate_size: inter,
        };

        // Router: single expert with any score (it will be selected)
        let router = vec![1.0f32; h]; // score = sum(x) for 1 routed expert

        let weights = MoeWeights {
            router,
            routed_experts: vec![zero_expert],
            shared_experts: vec![shared_expert],
            expert_bias: None,
        };

        let cfg = MoeConfig {
            hidden_size: h,
            expert_intermediate_size: inter,
            n_shared_experts: 1,
            n_routed_experts: 1,
            top_k: 1,
            routed_scaling_factor: 1.0,
            scoring_mode: ScoringMode::Softmax,
            shared_expert_intermediate_size: inter,
        };

        let x = vec![1.0f32, 0.0, 0.0, 0.0];
        let out = moe_forward(&x, &weights, &cfg).expect("moe_forward must succeed");

        // Expected: shared expert contributes silu(1*1) * 1 * 2 = silu(1)*2 to out[0]
        let silu_1 = 1.0f32 / (1.0 + (-1.0f32).exp()); // silu(1) = sigmoid(1)
        let expected = silu_1 * 2.0;
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "shared expert must contribute; out[0]={} expected≈{expected}",
            out[0]
        );
    }

    /// Additional test for SigmoidWithBias scoring mode.
    #[test]
    fn sigmoid_with_bias_scoring() {
        let n_routed = 3;
        let h = 4;
        let mut lcg = Lcg::new(42);
        let cfg = MoeConfig {
            hidden_size: h,
            expert_intermediate_size: 8,
            n_shared_experts: 0,
            n_routed_experts: n_routed,
            top_k: 1,
            routed_scaling_factor: 1.0,
            scoring_mode: ScoringMode::SigmoidWithBias,
            shared_expert_intermediate_size: 8,
        };
        let weights = make_weights(&mut lcg, &cfg, ScoringMode::SigmoidWithBias);
        let mut x = vec![0.0f32; h];
        lcg.fill(&mut x);

        let out = moe_forward(&x, &weights, &cfg);
        assert!(out.is_ok(), "sigmoid_with_bias moe_forward must not error");
        let out = out.expect("sigmoid_with_bias out");
        assert_eq!(out.len(), h, "output must have hidden_size elements");
    }
}

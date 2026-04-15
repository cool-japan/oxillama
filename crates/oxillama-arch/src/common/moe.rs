//! Sparse Mixture-of-Experts FFN layer.
//!
//! Implements the Mixtral-style sparse MoE where a fixed set of expert
//! SwiGLU FFNs are gated by a learned router, and only the top-K experts
//! are activated per token.
//!
//! All expert weights are stored as `f32` (dequantized at load time). A
//! follow-up PR will add quantized expert paths via `QuantLinear`.

use crate::error::{ArchError, ArchResult};

/// A single SwiGLU expert (same computation as standard LLaMA FFN).
///
/// Weight layout follows row-major GGUF convention:
/// - `gate`: `[intermediate_size, hidden_size]` — gate projection
/// - `up`:   `[intermediate_size, hidden_size]` — up projection
/// - `down`: `[hidden_size, intermediate_size]` — down projection
pub struct Expert {
    /// Gate projection: `[intermediate_size, hidden_size]` (row-major).
    pub gate: Vec<f32>,
    /// Up projection: `[intermediate_size, hidden_size]` (row-major).
    pub up: Vec<f32>,
    /// Down projection: `[hidden_size, intermediate_size]` (row-major).
    pub down: Vec<f32>,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
}

impl Expert {
    /// Validate that all weight buffers have the expected sizes.
    fn validate(&self) -> ArchResult<()> {
        let expected_gate_up = self.intermediate_size * self.hidden_size;
        let expected_down = self.hidden_size * self.intermediate_size;
        if self.gate.len() != expected_gate_up {
            return Err(ArchError::InvalidShape {
                name: "expert.gate".to_string(),
                expected: vec![self.intermediate_size, self.hidden_size],
                got: vec![self.gate.len()],
            });
        }
        if self.up.len() != expected_gate_up {
            return Err(ArchError::InvalidShape {
                name: "expert.up".to_string(),
                expected: vec![self.intermediate_size, self.hidden_size],
                got: vec![self.up.len()],
            });
        }
        if self.down.len() != expected_down {
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
    /// output   = W_down @ (gate_vec * up_vec)
    /// ```
    ///
    /// # Arguments
    /// * `input`  – slice of length `hidden_size`
    /// * `output` – mutable slice of length `hidden_size` (overwritten)
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
            *g = row
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
        }

        // SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        for g in gate_vec.iter_mut() {
            let sigmoid = 1.0 / (1.0 + (-*g).exp());
            *g *= sigmoid;
        }

        // Up GEMV: up_vec[i] = dot(up[i, :], input)
        for (i, u) in up_vec.iter_mut().enumerate() {
            let row = &self.up[i * h..(i + 1) * h];
            *u = row
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
        }

        // Element-wise multiply: gate_vec * up_vec (store into up_vec)
        for (u, &g) in up_vec.iter_mut().zip(gate_vec.iter()) {
            *u *= g;
        }

        // Down GEMV: output[i] = dot(down[i, :], up_vec)
        for (i, o) in output.iter_mut().enumerate() {
            let row = &self.down[i * n..(i + 1) * n];
            *o = row
                .iter()
                .zip(up_vec.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
        }

        Ok(())
    }
}

/// Sparse Mixture-of-Experts FFN layer.
///
/// Each forward call:
/// 1. Computes router logits via `router @ input`
/// 2. Applies softmax over all experts
/// 3. Selects the top-K experts by softmax weight
/// 4. Re-normalises the top-K weights to sum to 1
/// 5. Runs each selected expert and accumulates `weight * expert_out`
pub struct MoeFfn {
    /// Expert router weight matrix: `[num_experts, hidden_size]` (row-major).
    pub router: Vec<f32>,
    /// All expert FFNs.
    pub experts: Vec<Expert>,
    /// Number of experts to activate per token (top-K).
    pub top_k: usize,
    /// Total number of experts.
    pub num_experts: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
}

impl MoeFfn {
    /// Validate structural invariants.
    fn validate(&self) -> ArchResult<()> {
        let expected_router = self.num_experts * self.hidden_size;
        if self.router.len() != expected_router {
            return Err(ArchError::InvalidShape {
                name: "moe.router".to_string(),
                expected: vec![self.num_experts, self.hidden_size],
                got: vec![self.router.len()],
            });
        }
        if self.experts.len() != self.num_experts {
            return Err(ArchError::InvalidShape {
                name: "moe.experts".to_string(),
                expected: vec![self.num_experts],
                got: vec![self.experts.len()],
            });
        }
        if self.top_k == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "MoeFfn top_k must be >= 1".to_string(),
            });
        }
        Ok(())
    }

    /// Forward pass: route input to top-K experts, combine outputs.
    ///
    /// # Arguments
    /// * `input`  – slice of length `hidden_size`
    /// * `output` – mutable slice of length `hidden_size` (overwritten, not accumulated)
    ///
    /// # Errors
    /// Returns [`ArchError`] on shape/config mismatches or if any expert fails.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ArchResult<()> {
        self.validate()?;

        let n_exp = self.num_experts;

        // 1. Router logits: router_logits[i] = dot(router[i, :], input)
        let mut router_logits = vec![0.0f32; n_exp];
        for (i, logit) in router_logits.iter_mut().enumerate() {
            let row = &self.router[i * self.hidden_size..(i + 1) * self.hidden_size];
            *logit = row
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
        }

        // 2. Numerically stable softmax over router logits
        let max_logit = router_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_vals: Vec<f32> = router_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .collect();
        let exp_sum: f32 = exp_vals.iter().sum();
        // Normalise (guard against degenerate zero sum)
        if exp_sum > 0.0 {
            for v in exp_vals.iter_mut() {
                *v /= exp_sum;
            }
        } else {
            let uniform = 1.0 / n_exp as f32;
            for v in exp_vals.iter_mut() {
                *v = uniform;
            }
        }

        // 3. Select top-K experts by descending softmax weight
        let mut indices: Vec<usize> = (0..n_exp).collect();
        // Partial sort: find top_k largest weights.
        // Using sort_unstable_by for full sort (simpler and correct for small n_exp).
        indices.sort_unstable_by(|&a, &b| {
            exp_vals[b]
                .partial_cmp(&exp_vals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let effective_k = self.top_k.min(n_exp);
        let top_k_indices = &indices[..effective_k];

        // 4. Renormalise weights for the selected experts to sum to 1
        let selected_sum: f32 = top_k_indices.iter().map(|&i| exp_vals[i]).sum();

        // 5. Compute and accumulate weighted expert outputs
        for v in output.iter_mut() {
            *v = 0.0;
        }
        let mut expert_out = vec![0.0f32; self.hidden_size];

        for &expert_idx in top_k_indices {
            let weight = if selected_sum > 1e-9 {
                exp_vals[expert_idx] / selected_sum
            } else {
                1.0 / effective_k as f32
            };

            self.experts[expert_idx].forward(input, &mut expert_out)?;

            for (o, e) in output.iter_mut().zip(expert_out.iter()) {
                *o += weight * e;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an all-ones expert of the given size.
    fn make_expert(h: usize, n: usize) -> Expert {
        Expert {
            gate: vec![1.0; n * h],
            up: vec![1.0; n * h],
            down: vec![1.0; h * n],
            hidden_size: h,
            intermediate_size: n,
        }
    }

    #[test]
    fn test_expert_forward_shape() {
        let e = make_expert(4, 8);
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        e.forward(&input, &mut output)
            .expect("expert forward should succeed");
        // With all-ones weights, output should be non-zero
        assert!(
            output.iter().any(|v| v.abs() > 1e-6),
            "output should have non-zero elements, got {output:?}"
        );
    }

    #[test]
    fn test_expert_forward_silu_zero_input() {
        // With all-zero input, gate and up projections are zero, so output is zero.
        let e = make_expert(4, 8);
        let input = vec![0.0f32; 4];
        let mut output = vec![1.0f32; 4]; // pre-fill with non-zero
        e.forward(&input, &mut output)
            .expect("expert forward should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "output[{i}] = {v} should be ~0 for zero input"
            );
        }
    }

    #[test]
    fn test_expert_invalid_gate_size_errors() {
        let e = Expert {
            gate: vec![1.0; 3], // wrong size (should be 4*8 = 32)
            up: vec![1.0; 32],
            down: vec![1.0; 32],
            hidden_size: 4,
            intermediate_size: 8,
        };
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        assert!(
            e.forward(&input, &mut output).is_err(),
            "mismatched gate size should return error"
        );
    }

    #[test]
    fn test_moe_ffn_top1_routes_to_single_expert() {
        let h = 4;
        let n = 8;
        let num_experts = 4;
        // Router: only expert 0 row[0] = 1.0, all others zero.
        // For input [1,0,0,0], router logit for expert 0 is 1.0, rest are 0.0.
        // Softmax will strongly prefer expert 0.
        let mut router = vec![0.0f32; num_experts * h];
        router[0] = 1.0; // expert 0, dimension 0

        let experts: Vec<Expert> = (0..num_experts).map(|_| make_expert(h, n)).collect();
        let moe = MoeFfn {
            router,
            experts,
            top_k: 1,
            num_experts,
            hidden_size: h,
        };

        let input = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut output = vec![0.0f32; h];
        moe.forward(&input, &mut output)
            .expect("MoE forward should succeed");
        assert!(
            output.iter().any(|v| v.abs() > 1e-6),
            "top-1 MoE output should be non-zero, got {output:?}"
        );
    }

    #[test]
    fn test_moe_ffn_top2_combines_experts() {
        let h = 4;
        let n = 4;
        let num_experts = 4;
        // Uniform router weights → all experts get equal logits → equal softmax.
        // top-2 selects any 2; renormalised weights are 0.5 each.
        let router = vec![1.0f32; num_experts * h];
        let experts: Vec<Expert> = (0..num_experts).map(|_| make_expert(h, n)).collect();
        let moe = MoeFfn {
            router,
            experts,
            top_k: 2,
            num_experts,
            hidden_size: h,
        };

        let input = vec![1.0f32; h];
        let mut output = vec![0.0f32; h];
        moe.forward(&input, &mut output)
            .expect("top-2 MoE forward should succeed");
        assert!(
            output.iter().any(|v| v.abs() > 1e-6),
            "top-2 MoE output should be non-zero, got {output:?}"
        );
    }

    #[test]
    fn test_moe_ffn_top_k_exceeds_num_experts_clamps() {
        // top_k = 10 but only 4 experts: should clamp to 4 without panic.
        let h = 4;
        let n = 4;
        let num_experts = 4;
        let router = vec![1.0f32; num_experts * h];
        let experts: Vec<Expert> = (0..num_experts).map(|_| make_expert(h, n)).collect();
        let moe = MoeFfn {
            router,
            experts,
            top_k: 10,
            num_experts,
            hidden_size: h,
        };

        let input = vec![1.0f32; h];
        let mut output = vec![0.0f32; h];
        moe.forward(&input, &mut output)
            .expect("top_k > num_experts should clamp, not panic");
    }

    #[test]
    fn test_moe_ffn_zero_top_k_errors() {
        let h = 4;
        let n = 4;
        let num_experts = 2;
        let router = vec![1.0f32; num_experts * h];
        let experts: Vec<Expert> = (0..num_experts).map(|_| make_expert(h, n)).collect();
        let moe = MoeFfn {
            router,
            experts,
            top_k: 0, // invalid
            num_experts,
            hidden_size: h,
        };

        let input = vec![1.0f32; h];
        let mut output = vec![0.0f32; h];
        assert!(
            moe.forward(&input, &mut output).is_err(),
            "top_k = 0 should return an error"
        );
    }

    #[test]
    fn test_moe_ffn_invalid_router_size_errors() {
        let h = 4;
        let n = 4;
        let num_experts = 2;
        // Wrong router size
        let router = vec![1.0f32; 3]; // should be 2*4 = 8
        let experts: Vec<Expert> = (0..num_experts).map(|_| make_expert(h, n)).collect();
        let moe = MoeFfn {
            router,
            experts,
            top_k: 1,
            num_experts,
            hidden_size: h,
        };

        let input = vec![1.0f32; h];
        let mut output = vec![0.0f32; h];
        assert!(
            moe.forward(&input, &mut output).is_err(),
            "mismatched router size should return error"
        );
    }

    #[test]
    fn test_moe_ffn_output_is_deterministic() {
        // Running forward twice on same input produces identical output.
        let h = 4;
        let n = 4;
        let num_experts = 4;
        let router = vec![0.5f32; num_experts * h];
        let experts: Vec<Expert> = (0..num_experts).map(|_| make_expert(h, n)).collect();
        let moe = MoeFfn {
            router,
            experts,
            top_k: 2,
            num_experts,
            hidden_size: h,
        };

        let input = vec![0.3f32, 0.7, 0.1, 0.9];
        let mut output1 = vec![0.0f32; h];
        let mut output2 = vec![0.0f32; h];
        moe.forward(&input, &mut output1).expect("first forward");
        moe.forward(&input, &mut output2).expect("second forward");
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "forward should be deterministic: {a} != {b}"
            );
        }
    }
}

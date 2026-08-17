//! Sparse Mixture-of-Experts FFN layer.
//!
//! Implements the Mixtral-style sparse MoE where a fixed set of expert
//! SwiGLU FFNs are gated by a learned router, and only the top-K experts
//! are activated per token.
//!
//! Two expert representations live here:
//!
//! * [`QuantExpert`] / [`QuantMoeFfn`] — the **production** path.  Weights stay
//!   in their GGUF quantized form as shared mmap views
//!   ([`load_stacked_experts`](crate::common::loader::load_stacked_experts)) and
//!   the compute is dispatched through the quantization kernels, which are
//!   SIMD-accelerated and parallelised over rows by
//!   `oxillama_quant::parallel::for_each_row`.
//! * [`Expert`] / [`MoeFfn`] — the legacy `f32` path, kept so existing loaders
//!   keep compiling.  It dequantizes every expert at load time, which needs
//!   ~180 GB for Mixtral-8x7B, >500 GB for DBRX and ~5 GB for a single Phi-MoE
//!   layer.  Architecture owners must migrate to [`QuantMoeFfn`].

use oxillama_quant::{quantize_activations_q8_0_into, KernelDispatcher, QuantKernel};

use crate::common::linear::QuantLinear;
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
        // Allocates exactly the two `intermediate_size` vectors the original
        // implementation did — `expert_out` / router buffers stay empty.
        let mut scratch = MoeScratch::new(0, self.intermediate_size, 0);
        self.forward_with_scratch(input, output, &mut scratch)
    }

    /// Forward pass reusing caller-owned scratch buffers.
    ///
    /// `forward` allocates two `intermediate_size` vectors on every call, and
    /// MoE calls it `top_k` times per token per layer.  Hoist a [`MoeScratch`]
    /// into the model and call this instead.
    ///
    /// # Errors
    /// Returns [`ArchError::InvalidShape`] if weight buffers are inconsistent.
    pub fn forward_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut MoeScratch,
    ) -> ArchResult<()> {
        self.validate()?;

        let n = self.intermediate_size;
        let h = self.hidden_size;
        scratch.ensure_activations(n);
        let gate_vec = &mut scratch.gate[..n];
        let up_vec = &mut scratch.up[..n];

        // Gate GEMV: gate_vec[i] = silu(dot(gate[i, :], input)).
        // Row-parallel via the same helper every quantization kernel uses;
        // rows are never split so the result stays bit-identical.
        let gate_w = &self.gate;
        oxillama_quant::parallel::for_each_row(gate_vec, n, h, |i, g| {
            let row = &gate_w[i * h..(i + 1) * h];
            let acc = row
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
            // SiLU: x * sigmoid(x)
            *g = acc / (1.0 + (-acc).exp());
        });

        // Up GEMV fused with the element-wise gate multiply.
        let up_w = &self.up;
        let gate_ro: &[f32] = gate_vec;
        oxillama_quant::parallel::for_each_row(up_vec, n, h, |i, u| {
            let row = &up_w[i * h..(i + 1) * h];
            *u = row
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>()
                * gate_ro[i];
        });

        // Down GEMV: output[i] = dot(down[i, :], up_vec)
        let down_w = &self.down;
        let act: &[f32] = up_vec;
        oxillama_quant::parallel::for_each_row(output, h, n, |i, o| {
            let row = &down_w[i * n..(i + 1) * n];
            *o = row.iter().zip(act.iter()).map(|(w, x)| w * x).sum::<f32>();
        });

        Ok(())
    }
}

/// Reusable scratch buffers for one MoE forward pass.
///
/// [`Expert::forward`] and [`MoeFfn::forward`] allocate an
/// `intermediate_size` vector twice per selected expert per token; `MoeFfn`
/// allocates two more.  Hoist one of these into the model and use the
/// `*_with_scratch` entry points instead.
#[derive(Debug, Default, Clone)]
pub struct MoeScratch {
    /// Gate activations, `intermediate_size` long.
    pub gate: Vec<f32>,
    /// Up activations (also holds `gate * up`), `intermediate_size` long.
    pub up: Vec<f32>,
    /// One expert's output, `hidden_size` long.
    pub expert_out: Vec<f32>,
    /// Router logits / softmax weights, `num_experts` long.
    pub router_logits: Vec<f32>,
    /// Expert indices sorted by descending weight, `num_experts` long.
    pub order: Vec<usize>,
    /// Q8_0 image of `input`, quantized **once** per [`QuantMoeFfn::forward`]
    /// call and shared by every selected expert's `gate`/`up` — see
    /// [`QuantExpert::forward_fused`].  Empty whenever no selected expert's
    /// `gate`/`up` kernel opts into the fused path.
    pub acts_q8: Vec<u8>,
    /// Q8_0 image of one expert's SwiGLU intermediate (`gate * silu`), used by
    /// [`QuantExpert::forward_fused`] for the `down` projection.  Unlike
    /// [`Self::acts_q8`] this cannot be shared across experts — each expert
    /// computes its own intermediate from the same `input` — so it is
    /// requantized fresh inside every `forward_fused` call whose `down`
    /// kernel supports fusion.
    pub down_q8: Vec<u8>,
}

impl MoeScratch {
    /// Allocate scratch for the given layer geometry.
    pub fn new(hidden_size: usize, intermediate_size: usize, num_experts: usize) -> Self {
        Self {
            gate: vec![0.0; intermediate_size],
            up: vec![0.0; intermediate_size],
            expert_out: vec![0.0; hidden_size],
            router_logits: vec![0.0; num_experts],
            order: (0..num_experts).collect(),
            acts_q8: Vec::new(),
            down_q8: Vec::new(),
        }
    }

    /// Grow the activation buffers if the layer is wider than expected.
    ///
    /// Deliberately does **not** touch [`Self::expert_out`]: `QuantMoeFfn`
    /// hands that buffer out by value while an expert runs, and re-growing it
    /// from inside the expert would allocate on every call — defeating the
    /// scratch entirely.
    pub fn ensure_activations(&mut self, intermediate_size: usize) {
        if self.gate.len() < intermediate_size {
            self.gate.resize(intermediate_size, 0.0);
        }
        if self.up.len() < intermediate_size {
            self.up.resize(intermediate_size, 0.0);
        }
    }

    /// Grow the per-expert output buffer.
    pub fn ensure_output(&mut self, hidden_size: usize) {
        if self.expert_out.len() < hidden_size {
            self.expert_out.resize(hidden_size, 0.0);
        }
    }

    /// Grow both the activation and output buffers.
    pub fn ensure(&mut self, hidden_size: usize, intermediate_size: usize) {
        self.ensure_activations(intermediate_size);
        self.ensure_output(hidden_size);
    }

    /// Grow the router buffers if the layer has more experts than expected.
    pub fn ensure_router(&mut self, num_experts: usize) {
        if self.router_logits.len() < num_experts {
            self.router_logits.resize(num_experts, 0.0);
        }
        if self.order.len() < num_experts {
            self.order.clear();
            self.order.extend(0..num_experts);
        }
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
        let inter = self.experts.first().map_or(0, |e| e.intermediate_size);
        let mut scratch = MoeScratch::new(self.hidden_size, inter, self.num_experts);
        self.forward_with_scratch(input, output, &mut scratch)
    }

    /// Forward pass reusing caller-owned scratch buffers.
    ///
    /// `forward` allocates the router logits, the softmax weights, the expert
    /// output and two `intermediate_size` vectors per selected expert — on
    /// every token of every MoE layer.  Hold one [`MoeScratch`] in the model and
    /// call this instead.
    ///
    /// # Errors
    /// Returns [`ArchError`] on shape/config mismatches or if any expert fails.
    pub fn forward_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut MoeScratch,
    ) -> ArchResult<()> {
        self.validate()?;

        let n_exp = self.num_experts;
        scratch.ensure_router(n_exp);
        scratch.ensure_output(self.hidden_size);
        scratch.ensure_activations(self.experts.first().map_or(0, |e| e.intermediate_size));

        // 1. Router logits: router_logits[i] = dot(router[i, :], input)
        let mut router_logits = std::mem::take(&mut scratch.router_logits);
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
        let indices = &mut scratch.order;
        indices.clear();
        indices.extend(0..n_exp);
        // Full sort: `n_exp` is small (8–128) and this keeps ties deterministic.
        indices.sort_unstable_by(|&a, &b| {
            exp_vals[b]
                .partial_cmp(&exp_vals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let effective_k = self.top_k.min(n_exp);

        // 4. Renormalise weights for the selected experts to sum to 1
        let selected_sum: f32 = indices[..effective_k].iter().map(|&i| exp_vals[i]).sum();

        // 5. Compute and accumulate weighted expert outputs
        output.fill(0.0);
        let mut expert_out = std::mem::take(&mut scratch.expert_out);
        let selected: Vec<(usize, f32)> = indices[..effective_k]
            .iter()
            .map(|&e| {
                let weight = if selected_sum > 1e-9 {
                    exp_vals[e] / selected_sum
                } else {
                    1.0 / effective_k as f32
                };
                (e, weight)
            })
            .collect();

        let mut result = Ok(());
        for (expert_idx, weight) in selected {
            match self.experts[expert_idx].forward_with_scratch(input, &mut expert_out, scratch) {
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
        scratch.router_logits = router_logits;
        result
    }
}

/// A SwiGLU expert whose weights stay in their GGUF quantized format.
///
/// This is the production counterpart to [`Expert`].  The three projections are
/// [`QuantLinear`]s over **shared mmap views** produced by
/// [`load_stacked_experts`](crate::common::loader::load_stacked_experts), so a
/// Mixtral-8x7B layer costs the same resident memory as the file itself rather
/// than the ~180 GB an `f32` expansion would need.
///
/// Compute is dispatched through the quantization kernels, which are both
/// SIMD-specialised and row-parallel via
/// `oxillama_quant::parallel::for_each_row`.
pub struct QuantExpert {
    /// Gate projection: `[intermediate_size, hidden_size]`.
    pub gate: QuantLinear,
    /// Up projection: `[intermediate_size, hidden_size]`.
    pub up: QuantLinear,
    /// Down projection: `[hidden_size, intermediate_size]`.
    pub down: QuantLinear,
}

impl QuantExpert {
    /// Assemble an expert from its three projections.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] if the projections do not compose
    /// (`gate` and `up` must agree, and `down` must invert them).
    pub fn new(gate: QuantLinear, up: QuantLinear, down: QuantLinear) -> ArchResult<Self> {
        if gate.out_features != up.out_features || gate.in_features != up.in_features {
            return Err(ArchError::InvalidShape {
                name: "quant_expert.up".to_string(),
                expected: vec![gate.out_features, gate.in_features],
                got: vec![up.out_features, up.in_features],
            });
        }
        if down.in_features != gate.out_features || down.out_features != gate.in_features {
            return Err(ArchError::InvalidShape {
                name: "quant_expert.down".to_string(),
                expected: vec![gate.in_features, gate.out_features],
                got: vec![down.out_features, down.in_features],
            });
        }
        Ok(Self { gate, up, down })
    }

    /// Model hidden dimension.
    pub fn hidden_size(&self) -> usize {
        self.gate.in_features
    }

    /// FFN intermediate dimension.
    pub fn intermediate_size(&self) -> usize {
        self.gate.out_features
    }

    /// `down(silu(gate(x)) * up(x))` using caller-owned scratch.
    ///
    /// # Errors
    ///
    /// Propagates kernel dispatch and GEMV failures.
    pub fn forward(
        &self,
        kernels: &ExpertKernels,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut MoeScratch,
    ) -> ArchResult<()> {
        let n = self.intermediate_size();
        scratch.ensure_activations(n);

        self.gate
            .forward(&*kernels.gate, input, &mut scratch.gate[..n])
            .map_err(ArchError::from)?;
        self.up
            .forward(&*kernels.up, input, &mut scratch.up[..n])
            .map_err(ArchError::from)?;

        for (u, g) in scratch.up[..n].iter_mut().zip(scratch.gate[..n].iter()) {
            let silu = *g / (1.0 + (-*g).exp());
            *u *= silu;
        }

        self.down
            .forward(&*kernels.down, &scratch.up[..n], output)
            .map_err(ArchError::from)?;
        Ok(())
    }

    /// Q8_0 activation-block count a shared quantization of `input` needs to
    /// cover both `gate` and `up`, or `None` when neither projection's kernel
    /// opts into the fused path.
    ///
    /// Mirrors the `.max()` the dense attention/FFN paths take over several
    /// `Option<usize>`s that all read the same activation vector (see
    /// `llama::model::attention`'s `q_fused`/`k_fused`/`v_fused`): a
    /// mixed-precision layer whose `gate` and `up` use different quantization
    /// types still gets a buffer long enough for the wider of the two.
    pub fn gate_up_fused_blocks(&self, kernels: &ExpertKernels) -> Option<usize> {
        self.gate
            .q8_fused_blocks(&*kernels.gate)
            .into_iter()
            .chain(self.up.q8_fused_blocks(&*kernels.up))
            .max()
    }

    /// `down(activation(gate(x)) * up(x))`, routing whichever of `gate`/`up`/
    /// `down` support it through the fused Q8_0-activation kernel path.
    ///
    /// `acts_q8` must be a Q8_0 image of `input` with at least
    /// [`Self::gate_up_fused_blocks`]`(kernels)` blocks whenever that returns
    /// `Some` — callers quantize `input` **once** and pass the same buffer for
    /// every expert selected for this token, which is the entire benefit: the
    /// alternative (quantizing per expert) would redo the identical `f32 →
    /// i8` conversion `top_k` times over identical values.  Because Q8_0
    /// quantization is a pure, stateless function of the input floats (no
    /// shared scale, no cross-call state — see
    /// [`oxillama_quant::quantize_activations_q8_0_into`]), sharing the
    /// buffer changes nothing numerically versus requantizing per expert.
    ///
    /// `down`'s input is each expert's own SwiGLU intermediate, so it cannot
    /// share `acts_q8`; it is quantized fresh into `scratch.down_q8`
    /// whenever `down`'s kernel supports fusion.
    ///
    /// Each of `gate`/`up`/`down` is gated independently — exactly like the
    /// dense FFN in `llama::model::feed_forward` — so a kernel that does not
    /// support the fused path (its `q8_fused_blocks` returns `None`) simply
    /// falls back to the ordinary f32-activation GEMV for that one
    /// projection and `acts_q8`/`down_q8` are not read for it.
    ///
    /// # Errors
    ///
    /// Propagates kernel dispatch and GEMV failures.
    pub fn forward_fused<F: Fn(f32) -> f32>(
        &self,
        kernels: &ExpertKernels,
        input: &[f32],
        acts_q8: &[u8],
        activation: F,
        output: &mut [f32],
        scratch: &mut MoeScratch,
    ) -> ArchResult<()> {
        let n = self.intermediate_size();
        scratch.ensure_activations(n);

        match self.gate.q8_fused_blocks(&*kernels.gate) {
            Some(_) => self
                .gate
                .forward_q8_fused(&*kernels.gate, input, acts_q8, &mut scratch.gate[..n])
                .map_err(ArchError::from)?,
            None => self
                .gate
                .forward(&*kernels.gate, input, &mut scratch.gate[..n])
                .map_err(ArchError::from)?,
        }
        match self.up.q8_fused_blocks(&*kernels.up) {
            Some(_) => self
                .up
                .forward_q8_fused(&*kernels.up, input, acts_q8, &mut scratch.up[..n])
                .map_err(ArchError::from)?,
            None => self
                .up
                .forward(&*kernels.up, input, &mut scratch.up[..n])
                .map_err(ArchError::from)?,
        }

        for (u, g) in scratch.up[..n].iter_mut().zip(scratch.gate[..n].iter()) {
            *u *= activation(*g);
        }

        match self.down.q8_fused_blocks(&*kernels.down) {
            Some(n_blocks) => {
                let MoeScratch { up, down_q8, .. } = scratch;
                quantize_activations_q8_0_into(&up[..n], n_blocks, down_q8);
                self.down
                    .forward_q8_fused(&*kernels.down, &up[..n], down_q8, output)
                    .map_err(ArchError::from)?;
            }
            None => self
                .down
                .forward(&*kernels.down, &scratch.up[..n], output)
                .map_err(ArchError::from)?,
        }
        Ok(())
    }
}

/// Kernels for one expert's three projections, resolved once at load time.
///
/// Kernel lookup is a type dispatch, so resolving it inside the per-token,
/// per-expert loop is pure overhead.
pub struct ExpertKernels {
    /// Kernel for the gate projection's quantization type.
    pub gate: Box<dyn QuantKernel>,
    /// Kernel for the up projection's quantization type.
    pub up: Box<dyn QuantKernel>,
    /// Kernel for the down projection's quantization type.
    pub down: Box<dyn QuantKernel>,
}

impl ExpertKernels {
    /// Resolve the three kernels for `expert`.
    ///
    /// # Errors
    ///
    /// [`ArchError::Quant`] if any projection uses an unsupported quantization
    /// type.
    pub fn for_expert(expert: &QuantExpert) -> ArchResult<Self> {
        let dispatcher = KernelDispatcher::new();
        Ok(Self {
            gate: dispatcher.get_kernel(expert.gate.weight.tensor_type)?,
            up: dispatcher.get_kernel(expert.up.weight.tensor_type)?,
            down: dispatcher.get_kernel(expert.down.weight.tensor_type)?,
        })
    }
}

/// Sparse Mixture-of-Experts FFN over quantized experts.
///
/// Same routing semantics as [`MoeFfn`]: softmax over the router logits,
/// top-`k` selection, re-normalisation of the selected weights, weighted sum of
/// the expert outputs.
pub struct QuantMoeFfn {
    /// Router projection: `[num_experts, hidden_size]`.
    pub router: QuantLinear,
    /// All expert FFNs.
    pub experts: Vec<QuantExpert>,
    /// Number of experts to activate per token (top-K).
    pub top_k: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    router_kernel: Box<dyn QuantKernel>,
    expert_kernels: Vec<ExpertKernels>,
}

impl QuantMoeFfn {
    /// Assemble a quantized MoE layer and resolve every kernel up front.
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidConfig`] if `top_k == 0` or there are no experts.
    /// * [`ArchError::InvalidShape`] if the router width disagrees with the
    ///   expert count or hidden size.
    /// * [`ArchError::Quant`] for an unsupported quantization type.
    pub fn new(router: QuantLinear, experts: Vec<QuantExpert>, top_k: usize) -> ArchResult<Self> {
        if experts.is_empty() {
            return Err(ArchError::InvalidConfig {
                detail: "QuantMoeFfn requires at least one expert".to_string(),
            });
        }
        if top_k == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "QuantMoeFfn top_k must be >= 1".to_string(),
            });
        }
        let hidden_size = experts[0].hidden_size();
        if router.out_features != experts.len() {
            return Err(ArchError::InvalidShape {
                name: "quant_moe.router".to_string(),
                expected: vec![experts.len(), hidden_size],
                got: vec![router.out_features, router.in_features],
            });
        }
        if router.in_features != hidden_size {
            return Err(ArchError::InvalidShape {
                name: "quant_moe.router.in_features".to_string(),
                expected: vec![hidden_size],
                got: vec![router.in_features],
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
            experts,
            top_k,
            hidden_size,
            router_kernel,
            expert_kernels,
        })
    }

    /// Number of experts in this layer.
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

    /// Route `input` to the top-`k` experts and combine their outputs.
    ///
    /// `output` is **overwritten**, not accumulated.
    ///
    /// # Errors
    ///
    /// Propagates routing and expert failures.
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut MoeScratch,
    ) -> ArchResult<()> {
        let n_exp = self.experts.len();
        scratch.ensure_router(n_exp);
        scratch.ensure_output(self.hidden_size);
        scratch.ensure_activations(
            self.experts
                .first()
                .map_or(0, QuantExpert::intermediate_size),
        );

        self.router
            .forward(
                &*self.router_kernel,
                input,
                &mut scratch.router_logits[..n_exp],
            )
            .map_err(ArchError::from)?;

        softmax_inplace(&mut scratch.router_logits[..n_exp]);

        let weights = &scratch.router_logits[..n_exp];
        let order = &mut scratch.order;
        order.clear();
        order.extend(0..n_exp);
        order.sort_unstable_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let effective_k = self.top_k.min(n_exp);
        let selected_sum: f32 = order[..effective_k].iter().map(|&i| weights[i]).sum();

        output.fill(0.0);

        // `order` and `router_logits` are read-only from here; copy the small
        // selection out so the expert scratch can be borrowed mutably.
        let mut selection: [(usize, f32); 32] = [(0, 0.0); 32];
        let mut selection_len = 0usize;
        let mut spill: Vec<(usize, f32)> = Vec::new();
        for &idx in &scratch.order[..effective_k] {
            let w = if selected_sum > 1e-9 {
                scratch.router_logits[idx] / selected_sum
            } else {
                1.0 / effective_k as f32
            };
            if selection_len < selection.len() {
                selection[selection_len] = (idx, w);
                selection_len += 1;
            } else {
                spill.push((idx, w));
            }
        }

        // Every selected expert's `gate`/`up` reads this exact `input`
        // vector — expert selection just decides *which* weight rows dot it
        // against, not what the activation side is — so it is quantized to
        // Q8_0 **once** here and shared across all of them.  Quantizing per
        // expert would redo the identical `f32 → i8` conversion
        // `selection_len + spill.len()` times over identical values; that is
        // the whole win the fused path is for.  Mirrors the dense FFN's
        // gate/up share in `llama::model::feed_forward`, generalised from a
        // fixed 2 consumers to however many experts top-`k` selects.
        //
        // `down`'s input differs per expert (each expert's own SwiGLU
        // intermediate), so it is quantized separately inside
        // `QuantExpert::forward_fused`.
        let mut shared_blocks: Option<usize> = None;
        for &(idx, _) in selection[..selection_len].iter().chain(spill.iter()) {
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
        for &(idx, weight) in selection[..selection_len].iter().chain(spill.iter()) {
            let expert = &self.experts[idx];
            let kernels = &self.expert_kernels[idx];
            result = expert.forward_fused(
                kernels,
                input,
                &acts_q8,
                |g| g / (1.0 + (-g).exp()),
                &mut expert_out,
                scratch,
            );
            match &result {
                // Accumulate only on success: a failed expert leaves garbage.
                Ok(()) => {
                    for (o, e) in output.iter_mut().zip(expert_out.iter()) {
                        *o += weight * e;
                    }
                }
                Err(_) => break,
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

    // ── Fused Q8_0-activation path (QuantExpert::forward_fused) ────────────
    //
    // Two *different* claims are at stake here and they need different
    // tolerances:
    //
    // 1. Fused-Q8-activation GEMV vs f32-activation GEMV: only agree to
    //    within Q8_0's quantization error — see `oxillama-quant`'s
    //    `gemv_parity.rs` (`FUSED_TOL = 2.0e-2`, explicitly NOT bit-exact).
    //    None of the tests below make this comparison.
    // 2. Sharing one quantized buffer across several experts vs quantizing a
    //    fresh copy per expert: Q8_0 quantization is a pure, stateless
    //    function of the input floats (no shared scale, no cross-call state
    //    — see `quantize_activations_q8_0_into`'s own doc comment), so this
    //    MUST be bit-exact.  Every test below checks *this* claim, mirroring
    //    `llama_batch_prefill.rs`'s `assert_bit_identical` /
    //    `oxillama-quant`'s `activation_buffer_reuse_is_stateless`.

    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    /// Deterministic xorshift64* stream — reproducible fixtures, no PRNG dep.
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

    const Q4_K_BYTES: usize = 144;
    const Q4_K_BLOCK: usize = 256;

    /// `[rows, cols]` Q4_K weight tensor with well-formed pseudo-random
    /// blocks — a fused-capable control.  `d`/`dmin` are kept small (matching
    /// `oxillama-quant/tests/gemv_parity.rs` and
    /// `oxillama-arch/tests/llama_batch_prefill.rs`) so accumulated dot
    /// products stay finite; the packed scale/qs bytes are fully random since
    /// every bit pattern is a legal Q4_K block.
    fn q4_k_tensor(rows: usize, cols: usize, rng: &mut Rng) -> QuantTensor {
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

    fn q4_k_linear(rows: usize, cols: usize, rng: &mut Rng) -> QuantLinear {
        QuantLinear::new(q4_k_tensor(rows, cols, rng), None)
    }

    fn q4_k_expert(hidden: usize, inter: usize, rng: &mut Rng) -> QuantExpert {
        QuantExpert::new(
            q4_k_linear(inter, hidden, rng),
            q4_k_linear(inter, hidden, rng),
            q4_k_linear(hidden, inter, rng),
        )
        .expect("Q4_K expert shapes compose")
    }

    /// `[rows, cols]` tensor of `tensor_type`, filled with pseudo-random
    /// bytes.  Used only for the "kernel does not support fusion" controls,
    /// where `forward` and `forward_fused` must take the *identical* code
    /// path — so the numeric meaning of the weights is irrelevant, only that
    /// both calls see the same bytes.  F32 gets its own branch because
    /// `GgufTensorType::F32.block_size()` is `0` (not a block format at all);
    /// every real quantized type accepts arbitrary bytes as a legal block
    /// (see `gemv_parity.rs`'s `make_rows`).
    fn random_tensor(
        tensor_type: GgufTensorType,
        rows: usize,
        cols: usize,
        rng: &mut Rng,
    ) -> QuantTensor {
        if tensor_type == GgufTensorType::F32 {
            let mut data = Vec::with_capacity(rows * cols * 4);
            for _ in 0..rows * cols {
                data.extend_from_slice(&(rng.next_f32() * 0.3).to_le_bytes());
            }
            return QuantTensor::new(data, vec![rows, cols], tensor_type);
        }
        let block_size = tensor_type.block_size();
        let block_bytes = tensor_type.block_bytes();
        let blocks_per_row = cols.div_ceil(block_size.max(1));
        let mut data = Vec::with_capacity(rows * blocks_per_row * block_bytes);
        for _ in 0..rows * blocks_per_row * block_bytes {
            data.push(rng.next_u8());
        }
        QuantTensor::new(data, vec![rows, cols], tensor_type)
    }

    fn random_linear(
        tensor_type: GgufTensorType,
        rows: usize,
        cols: usize,
        rng: &mut Rng,
    ) -> QuantLinear {
        QuantLinear::new(random_tensor(tensor_type, rows, cols, rng), None)
    }

    /// A quantized type this crate's dispatcher serves but that has not opted
    /// into the fused path — the "genuinely block-quantized, still falls
    /// back" control `GgufTensorType::F32` cannot exercise (F32 has no block
    /// structure at all).  Discovered at runtime instead of hard-coded, so
    /// the test keeps meaning if a future kernel adds a fused override.
    fn unfused_block_quantized_type() -> GgufTensorType {
        let dispatcher = KernelDispatcher::new();
        dispatcher
            .supported_types()
            .into_iter()
            .find(|&ty| {
                if ty == GgufTensorType::F32 || ty == GgufTensorType::F16 {
                    return false;
                }
                match dispatcher.get_kernel(ty) {
                    Ok(kernel) => {
                        kernel.block_size() > 0 && kernel.q8_fused_acts_blocks(256).is_none()
                    }
                    Err(_) => false,
                }
            })
            .expect("at least one supported quantized type must not support fusion")
    }

    /// Sharing one Q8_0-quantized `input` across several experts' `gate`/`up`
    /// must be bit-for-bit identical to quantizing a fresh copy per expert —
    /// the entire justification for sharing the buffer instead of
    /// requantizing inside the per-expert loop.
    #[test]
    fn forward_fused_shared_acts_match_fresh_per_expert_quantization() {
        let hidden = 256;
        let inter = 256;
        let mut rng = Rng::new(0xE5FE_D000);
        let experts: Vec<QuantExpert> = (0..3)
            .map(|_| q4_k_expert(hidden, inter, &mut rng))
            .collect();
        let kernels: Vec<ExpertKernels> = experts
            .iter()
            .map(|e| ExpertKernels::for_expert(e).expect("kernels resolve"))
            .collect();

        let input: Vec<f32> = (0..hidden)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();

        let max_blocks = experts
            .iter()
            .zip(&kernels)
            .filter_map(|(e, k)| e.gate_up_fused_blocks(k))
            .max()
            .expect("Q4_K fixture must advertise the fused path, or this test is vacuous");

        let mut shared_acts = Vec::new();
        quantize_activations_q8_0_into(&input, max_blocks, &mut shared_acts);

        for (idx, (expert, kernel)) in experts.iter().zip(&kernels).enumerate() {
            let n_blocks = expert
                .gate_up_fused_blocks(kernel)
                .expect("every expert in this fixture shares the Q4_K tensor type");
            let mut fresh_acts = Vec::new();
            quantize_activations_q8_0_into(&input, n_blocks, &mut fresh_acts);

            let mut scratch_shared = MoeScratch::new(hidden, inter, 1);
            let mut scratch_fresh = MoeScratch::new(hidden, inter, 1);
            let mut out_shared = vec![0.0f32; hidden];
            let mut out_fresh = vec![0.0f32; hidden];
            let silu = |g: f32| g / (1.0 + (-g).exp());

            expert
                .forward_fused(
                    kernel,
                    &input,
                    &shared_acts,
                    silu,
                    &mut out_shared,
                    &mut scratch_shared,
                )
                .expect("shared-buffer forward_fused");
            expert
                .forward_fused(
                    kernel,
                    &input,
                    &fresh_acts,
                    silu,
                    &mut out_fresh,
                    &mut scratch_fresh,
                )
                .expect("fresh-buffer forward_fused");

            for (i, (a, b)) in out_shared.iter().zip(out_fresh.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "expert {idx} output[{i}]: shared-buffer {a} != fresh-buffer {b}"
                );
            }
        }
    }

    /// `QuantMoeFfn::forward`'s shared quantization must be exact relative to
    /// quantizing fresh for each selected expert, at several `top_k` values
    /// (including `top_k == num_experts`, which selects every expert every
    /// call).  The reference is rebuilt from the *actual* selection
    /// `forward` leaves behind in `scratch` (`order`/`router_logits` are not
    /// overwritten after selection) instead of re-deriving softmax/top-k
    /// here, so this test cannot drift from the real routing logic.
    #[test]
    fn quant_moe_ffn_forward_matches_fresh_per_expert_quantization_reference() {
        for &top_k in &[1usize, 2, 4] {
            let hidden = 256;
            let inter = 256;
            let n_exp = 4;
            let mut rng = Rng::new(0xB007_0000 ^ top_k as u64);
            let experts: Vec<QuantExpert> = (0..n_exp)
                .map(|_| q4_k_expert(hidden, inter, &mut rng))
                .collect();
            let router = q4_k_linear(n_exp, hidden, &mut rng);
            let moe = QuantMoeFfn::new(router, experts, top_k).expect("moe constructs");

            let input: Vec<f32> = (0..hidden)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
                .collect();

            let mut scratch = moe.make_scratch();
            let mut got = vec![0.0f32; hidden];
            moe.forward(&input, &mut got, &mut scratch)
                .expect("moe forward (shared quantization)");

            // Read the real selection back out of `scratch`: `order[..k]`
            // holds the selected expert indices and `router_logits` still
            // holds the softmax weights they were chosen from.
            let effective_k = top_k.min(n_exp);
            let order: Vec<usize> = scratch.order[..effective_k].to_vec();
            let selected_sum: f32 = order.iter().map(|&i| scratch.router_logits[i]).sum();

            // Rebuild the expected output requantizing `input` fresh for
            // *each* selected expert individually.
            let mut want = vec![0.0f32; hidden];
            let mut expert_scratch = MoeScratch::new(hidden, inter, n_exp);
            let silu = |g: f32| g / (1.0 + (-g).exp());
            for &idx in &order {
                let weight = if selected_sum > 1e-9 {
                    scratch.router_logits[idx] / selected_sum
                } else {
                    1.0 / effective_k as f32
                };
                let expert = &moe.experts[idx];
                let kernels = ExpertKernels::for_expert(expert).expect("kernels resolve");
                let n_blocks = expert
                    .gate_up_fused_blocks(&kernels)
                    .expect("Q4_K fixture must advertise the fused path");
                let mut fresh_acts = Vec::new();
                quantize_activations_q8_0_into(&input, n_blocks, &mut fresh_acts);
                let mut expert_out = vec![0.0f32; hidden];
                expert
                    .forward_fused(
                        &kernels,
                        &input,
                        &fresh_acts,
                        silu,
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

    /// A kernel that does not support the fused path takes the *same* code
    /// path inside `forward_fused` as inside `forward` — so the two methods
    /// must agree bit for bit, not merely both succeed.  Covers F32 (no
    /// block structure at all) and a genuinely block-quantized type that
    /// still has no fused override, which is the family the trait's default
    /// `matvec_q8_fused` refuses outright (block_size() != 32).
    #[test]
    fn forward_fused_matches_forward_when_kernel_lacks_fusion() {
        let unfused_quantized = unfused_block_quantized_type();
        for (tag, tensor_type) in [
            ("f32", GgufTensorType::F32),
            ("unfused_quantized", unfused_quantized),
        ] {
            let hidden = 256;
            let inter = 256;
            let mut rng = Rng::new(0xFA11_BACC ^ (tag.len() as u64));
            let expert = QuantExpert::new(
                random_linear(tensor_type, inter, hidden, &mut rng),
                random_linear(tensor_type, inter, hidden, &mut rng),
                random_linear(tensor_type, hidden, inter, &mut rng),
            )
            .expect("expert shapes compose");
            let kernels = ExpertKernels::for_expert(&expert).expect("kernels resolve");
            assert!(
                expert.gate_up_fused_blocks(&kernels).is_none(),
                "{tag}: fixture must not advertise the fused path, or this test is vacuous"
            );

            let input: Vec<f32> = (0..hidden)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.07)
                .collect();
            let silu = |g: f32| g / (1.0 + (-g).exp());

            let mut scratch_a = MoeScratch::new(hidden, inter, 1);
            let mut scratch_b = MoeScratch::new(hidden, inter, 1);
            let mut via_forward = vec![0.0f32; hidden];
            let mut via_fused = vec![0.0f32; hidden];

            expert
                .forward(&kernels, &input, &mut via_forward, &mut scratch_a)
                .unwrap_or_else(|e| panic!("{tag}: forward failed: {e}"));
            // No fused-capable kernel means `acts_q8` is never read; an empty
            // slice proves it.
            expert
                .forward_fused(&kernels, &input, &[], silu, &mut via_fused, &mut scratch_b)
                .unwrap_or_else(|e| panic!("{tag}: forward_fused failed: {e}"));

            for (i, (a, b)) in via_forward.iter().zip(via_fused.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{tag} output[{i}]: forward {a} != forward_fused {b} (fallback must be exact)"
                );
            }
        }
    }

    /// Per-projection gating: `gate`/`down` on a fused-capable Q4_K kernel,
    /// `up` on a non-fused F32 kernel.  `QuantExpert::new` only checks
    /// shapes, so this is a legal (if unusual) expert.  `forward_fused` must
    /// dispatch `gate`/`down` through the fused kernel and `up` through the
    /// plain GEMV independently — checked by hand-assembling the same three
    /// calls outside `forward_fused` and requiring a bit-exact match.
    #[test]
    fn forward_fused_gates_each_projection_independently() {
        let hidden = 256;
        let inter = 256;
        let mut rng = Rng::new(0x1A33_ED00);
        let gate = q4_k_linear(inter, hidden, &mut rng);
        let up = random_linear(GgufTensorType::F32, inter, hidden, &mut rng);
        let down = q4_k_linear(hidden, inter, &mut rng);
        let expert = QuantExpert::new(gate, up, down).expect("mixed expert constructs");
        let kernels = ExpertKernels::for_expert(&expert).expect("kernels resolve");

        let gate_blocks = expert.gate.q8_fused_blocks(&*kernels.gate);
        let up_blocks = expert.up.q8_fused_blocks(&*kernels.up);
        let down_blocks = expert.down.q8_fused_blocks(&*kernels.down);
        assert!(gate_blocks.is_some(), "gate (Q4_K) must be fused-capable");
        assert!(up_blocks.is_none(), "up (F32) must not be fused-capable");
        assert!(down_blocks.is_some(), "down (Q4_K) must be fused-capable");

        let input: Vec<f32> = (0..hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let mut acts_q8 = Vec::new();
        quantize_activations_q8_0_into(&input, gate_blocks.expect("checked above"), &mut acts_q8);

        // Hand-assemble the reference outside `forward_fused`: gate through
        // the fused kernel, up through the plain f32 GEMV, down through the
        // fused kernel over a fresh quantization of the SwiGLU intermediate.
        let mut want_gate = vec![0.0f32; inter];
        expert
            .gate
            .forward_q8_fused(&*kernels.gate, &input, &acts_q8, &mut want_gate)
            .expect("gate fused reference");
        let mut want_up = vec![0.0f32; inter];
        expert
            .up
            .forward(&*kernels.up, &input, &mut want_up)
            .expect("up unfused reference");
        let mut want_mid = vec![0.0f32; inter];
        for i in 0..inter {
            let silu = want_gate[i] / (1.0 + (-want_gate[i]).exp());
            want_mid[i] = want_up[i] * silu;
        }
        let mut down_acts = Vec::new();
        quantize_activations_q8_0_into(
            &want_mid,
            down_blocks.expect("checked above"),
            &mut down_acts,
        );
        let mut want_out = vec![0.0f32; hidden];
        expert
            .down
            .forward_q8_fused(&*kernels.down, &want_mid, &down_acts, &mut want_out)
            .expect("down fused reference");

        let mut scratch = MoeScratch::new(hidden, inter, 1);
        let mut got_out = vec![0.0f32; hidden];
        expert
            .forward_fused(
                &kernels,
                &input,
                &acts_q8,
                |g| g / (1.0 + (-g).exp()),
                &mut got_out,
                &mut scratch,
            )
            .expect("forward_fused (mixed)");

        for (i, (a, b)) in got_out.iter().zip(want_out.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "mixed-fusion output[{i}]: forward_fused {a} != hand-assembled reference {b}"
            );
        }
    }
}

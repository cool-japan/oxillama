//! Linear (fully connected) layer supporting quantized weights.
//!
//! Wraps a [`QuantTensor`] and dispatches to the appropriate quantization
//! kernel for forward passes.
//!
//! Optionally holds a [`LoraAdapter`] that adds a low-rank correction after
//! the main GEMV, enabling LoRA fine-tuned inference without modifying the
//! base quantized weights.

use std::sync::Arc;

use oxillama_quant::{LoraAdapter, QuantKernel, QuantTensor};

/// Convert a GGUF tensor's dimension list into `[out_features, in_features]`.
///
/// GGUF stores `ne` fastest-changing-first, so a weight mapping
/// `in_features → out_features` is written as `[in_features, out_features]` —
/// llama.cpp builds it with `ggml_new_tensor_2d(ctx, ty, n_in, n_out)`.
/// [`QuantLinear`] and every kernel behind it expect the mathematical
/// row-major order, which is the reverse.  Only the shape flips: `ne[1]` rows
/// of `ne[0]` contiguous values already *are* a row-major
/// `[out_features, in_features]` matrix, so the payload is left untouched.
///
/// Vectors (bias, norm weights) are unaffected — reversing a 1-element list is
/// a no-op.
pub fn gguf_linear_shape(dimensions: &[u64]) -> Vec<usize> {
    dimensions.iter().rev().map(|&d| d as usize).collect()
}

/// A linear layer with quantized weights and optional LoRA correction.
///
/// Stores the weight matrix in its quantized GGUF format and uses
/// the corresponding [`QuantKernel`] for efficient computation.
/// When a [`LoraAdapter`] is attached, the output of each forward call
/// is corrected by `B @ (A @ input) * scale` before being returned.
pub struct QuantLinear {
    /// Quantized weight tensor (shape: [out_features, in_features]).
    pub weight: QuantTensor,
    /// Optional bias vector (FP32, length: out_features).
    pub bias: Option<Vec<f32>>,
    /// Output feature count (number of rows in weight matrix).
    pub out_features: usize,
    /// Input feature count (number of columns in weight matrix).
    pub in_features: usize,
    /// Optional LoRA correction applied after the main GEMV.
    ///
    /// Shared via `Arc` so that the same adapter can be referenced from
    /// multiple layers without cloning the weight data.
    ///
    /// This is the *single-adapter* slot kept for backwards compatibility;
    /// [`QuantLinear::push_lora`] accumulates into a separate stack so that
    /// several adapters can be composed with independent scale multipliers.
    pub lora: Option<Arc<LoraAdapter>>,
    /// Accumulated LoRA stack: `(adapter, extra_scale)` applied in order after
    /// [`Self::lora`].
    ///
    /// `set_lora` **replaces** the single slot, which silently discarded every
    /// earlier adapter when a stack was applied; `push_lora` appends here
    /// instead, so `apply_lora_stack` composes correctly.
    lora_stack: Vec<(Arc<LoraAdapter>, f32)>,
}

/// Apply `B @ (A @ input) * (adapter.scale * extra_scale)` to `output`.
///
/// `LoraAdapter::apply` folds only the adapter's intrinsic `alpha / rank`
/// scale; stacking needs the per-entry multiplier as well, so the correction is
/// recomputed here.  `rank` is small (typically 8–128), so the intermediate is
/// cheap.
fn apply_lora_scaled(adapter: &LoraAdapter, input: &[f32], output: &mut [f32], extra_scale: f32) {
    let rank = adapter.rank;
    let in_f = adapter.in_features;
    if rank == 0 || in_f == 0 || input.len() < in_f {
        return;
    }
    if adapter.a.len() < rank * in_f || adapter.b.len() < adapter.out_features * rank {
        return;
    }

    let mut tmp = vec![0.0f32; rank];
    for (i, slot) in tmp.iter_mut().enumerate() {
        let row = &adapter.a[i * in_f..(i + 1) * in_f];
        *slot = row
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>();
    }

    let scale = adapter.scale * extra_scale;
    let n_out = adapter.out_features.min(output.len());
    for (o, out_slot) in output.iter_mut().enumerate().take(n_out) {
        let row = &adapter.b[o * rank..(o + 1) * rank];
        *out_slot += scale * row.iter().zip(tmp.iter()).map(|(w, x)| w * x).sum::<f32>();
    }
}

impl QuantLinear {
    /// Create a new quantized linear layer with no LoRA adapter.
    pub fn new(weight: QuantTensor, bias: Option<Vec<f32>>) -> Self {
        let out_features = if weight.shape.is_empty() {
            0
        } else {
            weight.shape[0]
        };
        let in_features = if weight.shape.len() < 2 {
            0
        } else {
            weight.shape[1]
        };

        Self {
            weight,
            bias,
            out_features,
            in_features,
            lora: None,
            lora_stack: Vec::new(),
        }
    }

    /// Attach a LoRA adapter to this layer.
    ///
    /// Replaces any previously attached adapter in the single-adapter slot.
    /// Use [`Self::push_lora`] to *compose* adapters instead of replacing.
    pub fn set_lora(&mut self, lora: Arc<LoraAdapter>) {
        self.lora = Some(lora);
    }

    /// Append a LoRA adapter with an extra scale multiplier.
    ///
    /// Unlike [`Self::set_lora`] this **accumulates**: applying a
    /// [`LoraStack`](crate::lora::LoraStack) of three adapters leaves all three
    /// corrections in effect, each with its own multiplier, which is what
    /// stacking is supposed to mean.
    pub fn push_lora(&mut self, lora: Arc<LoraAdapter>, scale: f32) {
        self.lora_stack.push((lora, scale));
    }

    /// Remove every attached LoRA adapter (both the single slot and the stack).
    pub fn clear_lora(&mut self) {
        self.lora = None;
        self.lora_stack.clear();
    }

    /// Number of adapters currently composed on top of this layer.
    pub fn lora_count(&self) -> usize {
        self.lora_stack.len() + usize::from(self.lora.is_some())
    }

    /// Forward pass: compute `output = weight @ input + bias [+ lora_delta]`.
    ///
    /// Uses the provided kernel for quantized matmul.  If a LoRA adapter is
    /// attached, the correction `B @ (A @ input) * scale` is added to the
    /// output after the bias.
    pub fn forward(
        &self,
        kernel: &dyn QuantKernel,
        input: &[f32],
        output: &mut [f32],
    ) -> oxillama_quant::QuantResult<()> {
        kernel.gemv(&self.weight, input, output)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for (o, &b) in output.iter_mut().zip(bias.iter()) {
                *o += b;
            }
        }

        // Apply LoRA correction if attached
        if let Some(ref lora) = self.lora {
            lora.apply(input, output)?;
        }

        for (adapter, scale) in &self.lora_stack {
            apply_lora_scaled(adapter, input, output, *scale);
        }

        Ok(())
    }

    /// Q8_0 activation blocks needed to run this layer through
    /// [`Self::forward_q8_fused`], or `None` when `kernel` has no fused
    /// override worth dispatching to.
    ///
    /// Callers size one scratch buffer from the maximum over the layers that
    /// share an activation vector, quantize once, and then hand the same
    /// buffer to every one of them.
    pub fn q8_fused_blocks(&self, kernel: &dyn QuantKernel) -> Option<usize> {
        kernel.q8_fused_acts_blocks(self.in_features)
    }

    /// Forward pass over an activation vector that has **already** been
    /// quantized to Q8_0 by [`oxillama_quant::quantize_activations_q8_0_into`].
    ///
    /// Semantically identical to [`Self::forward`] except that the activation
    /// side of every dot product is the Q8_0 reconstruction of `input` rather
    /// than `input` itself.  That is the whole point: the `f32 → i8`
    /// conversion happens once per matmul *input* instead of once per weight
    /// *row*, and the row loop then multiplies i8 by i8 in integer registers.
    ///
    /// `input` is still required because the bias and the LoRA correction are
    /// f32 paths that must see the unquantized activation.
    ///
    /// [`QuantKernel::matvec_q8_fused`] accumulates, so `output` is zeroed
    /// first — unlike `gemv`, which overwrites.
    ///
    /// # Adoption
    ///
    /// This entry point is architecture-agnostic, but only `qwen3` currently
    /// calls it: that is the architecture whose decode speed was measured
    /// (6.34 → 12.98 tok/s on Qwen3-4B `Q4_K_M` / Apple M3), and a fused path
    /// switched on for an architecture nobody benchmarked would be an
    /// unmeasured numerics change.  Wiring another architecture is mechanical
    /// — quantize each activation vector once with
    /// [`oxillama_quant::quantize_activations_q8_0_into`], size the block
    /// count from [`Self::q8_fused_blocks`] over the layers that share that
    /// vector, and call this instead of [`Self::forward`] wherever the count
    /// is `Some`.
    pub fn forward_q8_fused(
        &self,
        kernel: &dyn QuantKernel,
        input: &[f32],
        acts_q8: &[u8],
        output: &mut [f32],
    ) -> oxillama_quant::QuantResult<()> {
        let n_rows = self.out_features.min(output.len());
        output[..n_rows].fill(0.0);
        kernel.matvec_q8_fused(
            &self.weight.data,
            acts_q8,
            output,
            self.out_features,
            self.in_features,
        )?;

        if let Some(ref bias) = self.bias {
            for (o, &b) in output.iter_mut().zip(bias.iter()) {
                *o += b;
            }
        }

        if let Some(ref lora) = self.lora {
            lora.apply(input, output)?;
        }

        for (adapter, scale) in &self.lora_stack {
            apply_lora_scaled(adapter, input, output, *scale);
        }

        Ok(())
    }

    /// Batched sibling of [`Self::forward_q8_fused`]: `m` tokens through one
    /// weight matrix in a single sweep.
    ///
    /// `acts_q8` holds the `m` activation vectors back to back, as produced by
    /// [`oxillama_quant::quantize_activations_q8_0_batch_into`] with the block
    /// count from [`Self::q8_fused_blocks`].
    ///
    /// `output` is **feature-major** `[out_features][m]` — output row `r` for
    /// token `t` lands at `output[r * m + t]`.  That is the kernel's native
    /// layout (it keeps a weight row's `m` results contiguous on one thread);
    /// callers that want token-major hidden states transpose afterwards.
    ///
    /// The result for every `(row, token)` is bit-identical to calling
    /// [`Self::forward_q8_fused`] for that token alone: the kernels hoist the
    /// weight decode out of the token loop but never reassociate the `K`
    /// accumulation, and the bias is added once per element either way.
    ///
    /// # LoRA
    ///
    /// A LoRA correction is per token and is **not** applied here.  Callers
    /// must keep LoRA-patched layers on the per-token path; the Qwen3 prefill
    /// checks this before choosing the batched route.
    pub fn forward_q8_fused_batch(
        &self,
        kernel: &dyn QuantKernel,
        acts_q8: &[u8],
        output: &mut [f32],
        m: usize,
    ) -> oxillama_quant::QuantResult<()> {
        let n_rows = self.out_features.min(output.len() / m.max(1));
        output[..n_rows * m].fill(0.0);
        kernel.matmul_q8_fused(
            &self.weight.data,
            acts_q8,
            output,
            self.out_features,
            self.in_features,
            m,
        )?;

        if let Some(ref bias) = self.bias {
            for (row, &b) in bias.iter().enumerate().take(n_rows) {
                for o in output[row * m..(row + 1) * m].iter_mut() {
                    *o += b;
                }
            }
        }

        Ok(())
    }

    /// Batched forward pass: compute `output = weight @ input_matrix + bias`.
    ///
    /// Note: LoRA correction is not applied in batch mode because each row of
    /// the input would require an independent correction.  For batched inference
    /// with LoRA, call [`forward`](Self::forward) per token.
    pub fn forward_batch(
        &self,
        kernel: &dyn QuantKernel,
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
    ) -> oxillama_quant::QuantResult<()> {
        kernel.gemm(
            &self.weight,
            input,
            output,
            batch_size,
            self.out_features,
            self.in_features,
        )?;

        // Add bias if present (broadcast across batch)
        if let Some(ref bias) = self.bias {
            for row in 0..batch_size {
                let row_offset = row * self.out_features;
                for (j, &b) in bias.iter().enumerate() {
                    output[row_offset + j] += b;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::{LoraAdapter, QuantTensor};

    use super::*;

    /// Build a small F32 QuantTensor from row-major weight values.
    fn f32_tensor(weights: &[f32], rows: usize, cols: usize) -> QuantTensor {
        let mut data = Vec::with_capacity(weights.len() * 4);
        for &w in weights {
            data.extend_from_slice(&w.to_le_bytes());
        }
        QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32)
    }

    /// Build an F32 kernel dispatcher just for the test.
    fn f32_kernel() -> Box<dyn QuantKernel> {
        use oxillama_quant::KernelDispatcher;
        KernelDispatcher::new()
            .get_kernel(GgufTensorType::F32)
            .expect("F32 kernel must be available")
    }

    /// A QuantLinear without LoRA should produce the plain GEMV result.
    #[test]
    fn test_forward_no_lora() {
        // Weight matrix: [[2, 0], [0, 3]]  (2×2, F32)
        let tensor = f32_tensor(&[2.0f32, 0.0, 0.0, 3.0], 2, 2);
        let linear = QuantLinear::new(tensor, None);
        let kernel = f32_kernel();

        let input = vec![4.0f32, 5.0];
        let mut output = vec![0.0f32; 2];
        linear
            .forward(&*kernel, &input, &mut output)
            .expect("forward ok");

        // W @ input = [2*4 + 0*5, 0*4 + 3*5] = [8, 15]
        assert!((output[0] - 8.0).abs() < 1e-5, "output[0]={}", output[0]);
        assert!((output[1] - 15.0).abs() < 1e-5, "output[1]={}", output[1]);
    }

    /// After `set_lora()`, the LoRA correction is added to the GEMV result.
    ///
    /// W = I₂ (2×2 identity), so W @ input = input = [1, 2].
    /// A = I₂, B = I₂, scale = 1.0  →  delta = input = [1, 2].
    /// Expected output = [1+1, 2+2] = [2, 4].
    #[test]
    fn test_forward_with_lora_identity() {
        let tensor = f32_tensor(&[1.0f32, 0.0, 0.0, 1.0], 2, 2); // I₂
        let mut linear = QuantLinear::new(tensor, None);

        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // I₂, rank=2, in=2
        let b = vec![1.0f32, 0.0, 0.0, 1.0]; // I₂, out=2, rank=2
        let adapter = LoraAdapter::new(a, b, 2, 1.0, 2, 2).expect("valid adapter");
        linear.set_lora(Arc::new(adapter));

        let kernel = f32_kernel();
        let input = vec![1.0f32, 2.0];
        let mut output = vec![0.0f32; 2];
        linear
            .forward(&*kernel, &input, &mut output)
            .expect("forward ok");

        assert!((output[0] - 2.0).abs() < 1e-5, "output[0]={}", output[0]);
        assert!((output[1] - 4.0).abs() < 1e-5, "output[1]={}", output[1]);
    }

    /// After `clear_lora()`, the LoRA correction is no longer applied.
    #[test]
    fn test_clear_lora() {
        let tensor = f32_tensor(&[1.0f32, 0.0, 0.0, 1.0], 2, 2);
        let mut linear = QuantLinear::new(tensor, None);

        let adapter = LoraAdapter::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            2,
            1.0,
            2,
            2,
        )
        .expect("valid adapter");
        linear.set_lora(Arc::new(adapter));
        linear.clear_lora();

        let kernel = f32_kernel();
        let input = vec![3.0f32, 7.0];
        let mut output = vec![0.0f32; 2];
        linear
            .forward(&*kernel, &input, &mut output)
            .expect("forward ok");

        // No LoRA: W @ input = I₂ @ [3,7] = [3,7]
        assert!((output[0] - 3.0).abs() < 1e-5, "output[0]={}", output[0]);
        assert!((output[1] - 7.0).abs() < 1e-5, "output[1]={}", output[1]);
    }

    /// LoRA is applied after the bias, not before.
    #[test]
    fn test_forward_lora_applied_after_bias() {
        // W = I₂, bias = [10, 10]
        let tensor = f32_tensor(&[1.0f32, 0.0, 0.0, 1.0], 2, 2);
        let mut linear = QuantLinear::new(tensor, Some(vec![10.0f32, 10.0]));

        // LoRA: A=I₂, B=I₂, scale=1.0 → delta = input
        let adapter = LoraAdapter::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            2,
            1.0,
            2,
            2,
        )
        .expect("valid adapter");
        linear.set_lora(Arc::new(adapter));

        let kernel = f32_kernel();
        let input = vec![2.0f32, 3.0];
        let mut output = vec![0.0f32; 2];
        linear
            .forward(&*kernel, &input, &mut output)
            .expect("forward ok");

        // W @ input + bias + delta = [2+10+2, 3+10+3] = [14, 16]
        assert!((output[0] - 14.0).abs() < 1e-5, "output[0]={}", output[0]);
        assert!((output[1] - 16.0).abs() < 1e-5, "output[1]={}", output[1]);
    }

    /// GGUF's in-features-first order becomes `[out_features, in_features]`.
    ///
    /// Qwen3-4B's `attn_q.weight` is stored as `[2560, 4096]`; reading it
    /// verbatim makes the GEMV demand a 4096-long activation vector from a
    /// 2560-wide residual stream.
    #[test]
    fn test_gguf_linear_shape_reverses_2d_dims() {
        assert_eq!(gguf_linear_shape(&[2560, 4096]), vec![4096, 2560]);
        assert_eq!(gguf_linear_shape(&[9728, 2560]), vec![2560, 9728]);
    }

    /// 1-D tensors (bias, norm scales) pass through unchanged.
    #[test]
    fn test_gguf_linear_shape_leaves_vectors_alone() {
        assert_eq!(gguf_linear_shape(&[128]), vec![128]);
        assert_eq!(gguf_linear_shape(&[]), Vec::<usize>::new());
    }

    /// `lora` field is None for a freshly constructed QuantLinear.
    #[test]
    fn test_new_lora_is_none() {
        let tensor = f32_tensor(&[1.0f32], 1, 1);
        let linear = QuantLinear::new(tensor, None);
        assert!(linear.lora.is_none());
    }
}

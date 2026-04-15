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
    pub lora: Option<Arc<LoraAdapter>>,
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
        }
    }

    /// Attach a LoRA adapter to this layer.
    ///
    /// Replaces any previously attached adapter.  Pass `None` to remove the
    /// correction entirely.
    pub fn set_lora(&mut self, lora: Arc<LoraAdapter>) {
        self.lora = Some(lora);
    }

    /// Remove any attached LoRA adapter.
    pub fn clear_lora(&mut self) {
        self.lora = None;
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

    /// `lora` field is None for a freshly constructed QuantLinear.
    #[test]
    fn test_new_lora_is_none() {
        let tensor = f32_tensor(&[1.0f32], 1, 1);
        let linear = QuantLinear::new(tensor, None);
        assert!(linear.lora.is_none());
    }
}

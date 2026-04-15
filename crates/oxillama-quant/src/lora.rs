//! LoRA (Low-Rank Adaptation) correction for quantized linear layers.
//!
//! At inference time, a LoRA adapter adds a low-rank correction to the weight
//! matrix output:
//!
//! ```text
//! output = W @ input + B @ (A @ input) * scale
//! ```
//!
//! where `A` is `[rank × in_features]` and `B` is `[out_features × rank]`.
//! Because `rank` is typically 4–64, the two small matrix-vector products are
//! negligible compared to the main quantized GEMV.

use crate::error::{QuantError, QuantResult};

/// Low-rank adaptation matrices for a single linear layer.
///
/// Computes the correction: `delta = B @ (A @ input) * scale`
/// where A is `[rank × in_features]` (row-major) and B is
/// `[out_features × rank]` (row-major).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// A matrix (row-major, shape `[rank, in_features]`).
    pub a: Vec<f32>,
    /// B matrix (row-major, shape `[out_features, rank]`).
    pub b: Vec<f32>,
    /// Rank `r`.
    pub rank: usize,
    /// Scale factor: `lora_alpha / rank`.
    pub scale: f32,
    /// Input feature count.
    pub in_features: usize,
    /// Output feature count.
    pub out_features: usize,
}

impl LoraAdapter {
    /// Create a new LoRA adapter, validating matrix dimensions.
    ///
    /// # Errors
    /// Returns [`QuantError::DimensionMismatch`] if the matrix sizes are
    /// inconsistent with `rank`, `in_features`, or `out_features`.
    pub fn new(
        a: Vec<f32>,
        b: Vec<f32>,
        rank: usize,
        scale: f32,
        in_features: usize,
        out_features: usize,
    ) -> QuantResult<Self> {
        let expected_a = rank * in_features;
        if a.len() != expected_a {
            return Err(QuantError::DimensionMismatch {
                expected: expected_a,
                got: a.len(),
            });
        }
        let expected_b = out_features * rank;
        if b.len() != expected_b {
            return Err(QuantError::DimensionMismatch {
                expected: expected_b,
                got: b.len(),
            });
        }
        Ok(Self {
            a,
            b,
            rank,
            scale,
            in_features,
            out_features,
        })
    }

    /// Apply the LoRA correction in-place: `output += B @ (A @ input) * scale`.
    ///
    /// # Arguments
    /// * `input`  – FP32 input vector, length ≥ `in_features`.
    /// * `output` – FP32 output vector, length ≥ `out_features`; modified in place.
    ///
    /// # Errors
    /// Returns [`QuantError::DimensionMismatch`] if either slice is too short.
    pub fn apply(&self, input: &[f32], output: &mut [f32]) -> QuantResult<()> {
        if input.len() < self.in_features {
            return Err(QuantError::DimensionMismatch {
                expected: self.in_features,
                got: input.len(),
            });
        }
        if output.len() < self.out_features {
            return Err(QuantError::DimensionMismatch {
                expected: self.out_features,
                got: output.len(),
            });
        }

        // Step 1: r_vec = A @ input   (rank × 1)
        let mut r_vec = vec![0.0f32; self.rank];
        for (i, r) in r_vec.iter_mut().enumerate().take(self.rank) {
            let row_start = i * self.in_features;
            let row = &self.a[row_start..row_start + self.in_features];
            *r = row
                .iter()
                .zip(input[..self.in_features].iter())
                .map(|(&a, &x)| a * x)
                .sum();
        }

        // Step 2: output += B @ r_vec * scale   (out_features × 1)
        let s = self.scale;
        for (i, out) in output.iter_mut().enumerate().take(self.out_features) {
            let row_start = i * self.rank;
            let row = &self.b[row_start..row_start + self.rank];
            let delta: f32 = row
                .iter()
                .zip(r_vec.iter())
                .map(|(&b, &r)| b * r)
                .sum::<f32>()
                * s;
            *out += delta;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic rank-2 correctness test with hand-computed expected values.
    ///
    /// A = [[1, 0], [0, 1]]  (2×2 identity),  rank = 2, in = 2
    /// B = [[2, 0], [0, 2]]  (2×2 diagonal),  out = 2
    /// scale = 1.0
    /// input = [3, 4]
    ///
    /// r_vec = A @ input = [3, 4]
    /// delta = B @ r_vec * 1.0 = [6, 8]
    /// output (initially [0,0]) → [6, 8]
    #[test]
    fn test_apply_rank2_correctness() {
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let b = vec![2.0f32, 0.0, 0.0, 2.0]; // 2×2 diagonal
        let adapter = LoraAdapter::new(a, b, 2, 1.0, 2, 2).expect("valid adapter");

        let input = vec![3.0f32, 4.0];
        let mut output = vec![0.0f32, 0.0];
        adapter.apply(&input, &mut output).expect("apply ok");

        assert!(
            (output[0] - 6.0).abs() < 1e-6,
            "output[0] should be 6, got {}",
            output[0]
        );
        assert!(
            (output[1] - 8.0).abs() < 1e-6,
            "output[1] should be 8, got {}",
            output[1]
        );
    }

    /// Scale factor is applied correctly.
    #[test]
    fn test_apply_scale() {
        // A and B are 1×1 identity matrices (rank=1, in=1, out=1)
        let a = vec![1.0f32];
        let b = vec![1.0f32];
        let scale = 0.5;
        let adapter = LoraAdapter::new(a, b, 1, scale, 1, 1).expect("valid adapter");

        let input = vec![4.0f32];
        let mut output = vec![10.0f32]; // non-zero base to test accumulation
        adapter.apply(&input, &mut output).expect("apply ok");

        // delta = 1 * (1 * 4) * 0.5 = 2.0; output = 10 + 2 = 12
        assert!(
            (output[0] - 12.0).abs() < 1e-6,
            "output[0] should be 12, got {}",
            output[0]
        );
    }

    /// LoRA correction adds to existing output (accumulation behaviour).
    #[test]
    fn test_apply_accumulates() {
        let a = vec![1.0f32];
        let b = vec![1.0f32];
        let adapter = LoraAdapter::new(a, b, 1, 1.0, 1, 1).expect("valid adapter");

        let input = vec![1.0f32];
        let mut output = vec![5.0f32];
        adapter.apply(&input, &mut output).expect("apply ok");
        assert!(
            (output[0] - 6.0).abs() < 1e-6,
            "output should accumulate: 5 + 1 = 6, got {}",
            output[0]
        );
    }

    /// Dimension mismatch on construction: wrong `a` length.
    #[test]
    fn test_new_dimension_mismatch_a() {
        let a = vec![1.0f32]; // should be rank*in = 2
        let b = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 = 4 floats
        let result = LoraAdapter::new(a, b, 2, 1.0, 1, 2);
        assert!(
            matches!(result, Err(QuantError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            result
        );
    }

    /// Dimension mismatch on construction: wrong `b` length.
    #[test]
    fn test_new_dimension_mismatch_b() {
        let a = vec![1.0f32, 0.0]; // rank=1, in=2 → ok
        let b = vec![1.0f32]; // should be out*rank = 2*1 = 2
        let result = LoraAdapter::new(a, b, 1, 1.0, 2, 2);
        assert!(
            matches!(result, Err(QuantError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            result
        );
    }

    /// apply() returns error when input slice is too short.
    #[test]
    fn test_apply_input_too_short() {
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2
        let b = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2
        let adapter = LoraAdapter::new(a, b, 2, 1.0, 2, 2).expect("valid adapter");

        let input = vec![1.0f32]; // too short: needs 2
        let mut output = vec![0.0f32, 0.0];
        let result = adapter.apply(&input, &mut output);
        assert!(
            matches!(result, Err(QuantError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            result
        );
    }

    /// apply() returns error when output slice is too short.
    #[test]
    fn test_apply_output_too_short() {
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2
        let b = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2
        let adapter = LoraAdapter::new(a, b, 2, 1.0, 2, 2).expect("valid adapter");

        let input = vec![1.0f32, 2.0];
        let mut output = vec![0.0f32]; // too short: needs 2
        let result = adapter.apply(&input, &mut output);
        assert!(
            matches!(result, Err(QuantError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            result
        );
    }

    /// Zero-rank adapter (rank=0) is handled without panicking.
    /// Construction should succeed since 0*in = 0 and out*0 = 0.
    #[test]
    fn test_zero_rank_adapter() {
        let adapter = LoraAdapter::new(vec![], vec![], 0, 1.0, 4, 4).expect("rank=0 valid");
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32, 0.0, 0.0, 0.0];
        adapter.apply(&input, &mut output).expect("apply ok");
        // delta is zero for all outputs since rank=0
        for &v in &output {
            assert!(
                v.abs() < 1e-9,
                "zero-rank adapter should not modify output, got {v}"
            );
        }
    }
}

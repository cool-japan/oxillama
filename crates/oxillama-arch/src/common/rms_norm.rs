//! RMSNorm (Root Mean Square Layer Normalization).
//!
//! Used by LLaMA, Qwen3, Mistral, and most modern transformer architectures
//! as a faster alternative to LayerNorm.

/// RMSNorm layer with learned scale parameters.
///
/// Computes: `output[i] = (x[i] / rms(x)) * weight[i]`
/// where `rms(x) = sqrt(mean(x^2) + eps)`
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Learned scale weights (one per hidden dimension).
    pub weight: Vec<f32>,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl RmsNorm {
    /// Create a new RMSNorm layer.
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Apply RMSNorm to an input vector in-place.
    ///
    /// # Arguments
    /// * `x` - Input/output vector of length `hidden_size`.
    pub fn forward(&self, x: &mut [f32]) {
        let n = x.len();
        if n == 0 {
            return;
        }

        // Compute RMS: sqrt(mean(x^2) + eps)
        let mut sum_sq: f32 = 0.0;
        for &val in x.iter() {
            sum_sq += val * val;
        }
        let rms = (sum_sq / n as f32 + self.eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Normalize and scale
        for (i, val) in x.iter_mut().enumerate() {
            *val = *val * inv_rms * self.weight[i];
        }
    }

    /// Apply RMSNorm, writing the result to a separate output buffer.
    pub fn forward_to(&self, input: &[f32], output: &mut [f32]) {
        let n = input.len();
        if n == 0 {
            return;
        }

        let mut sum_sq: f32 = 0.0;
        for &val in input.iter() {
            sum_sq += val * val;
        }
        let rms = (sum_sq / n as f32 + self.eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..n {
            output[i] = input[i] * inv_rms * self.weight[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_identity_weights() {
        let weight = vec![1.0; 4];
        let norm = RmsNorm::new(weight, 1e-5);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        norm.forward(&mut x);

        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let expected_rms = (7.5_f32 + 1e-5).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| v / expected_rms)
            .collect();

        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
        }
    }

    #[test]
    fn test_rms_norm_forward_to_matches_forward() {
        // forward_to(input, out) should produce the same result as forward(&mut x.clone())
        let weight = vec![1.0f32, 2.0, 0.5, 1.5];
        let norm = RmsNorm::new(weight, 1e-5);
        let input = vec![3.0f32, -1.0, 2.0, 0.5];

        // forward_to path
        let mut out_to = vec![0.0f32; 4];
        norm.forward_to(&input, &mut out_to);

        // forward (in-place) path
        let mut out_inplace = input.clone();
        norm.forward(&mut out_inplace);

        for (i, (a, b)) in out_to.iter().zip(out_inplace.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "forward_to[{i}]={a} != forward[{i}]={b}"
            );
        }
    }

    #[test]
    fn test_rms_norm_non_unit_weights_scale_output() {
        // Doubling the weight should double the output.
        let input = vec![1.0f32, 2.0, 3.0];
        let norm1 = RmsNorm::new(vec![1.0f32; 3], 1e-5);
        let norm2 = RmsNorm::new(vec![2.0f32; 3], 1e-5);

        let mut out1 = input.clone();
        let mut out2 = input.clone();
        norm1.forward(&mut out1);
        norm2.forward(&mut out2);

        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (b - 2.0 * a).abs() < 1e-6,
                "out2[{i}] should be 2×out1[{i}]: {b} vs 2×{a}"
            );
        }
    }

    #[test]
    fn test_rms_norm_empty_does_not_panic() {
        let norm = RmsNorm::new(vec![], 1e-5);
        let mut x: Vec<f32> = vec![];
        norm.forward(&mut x); // must not panic
    }

    #[test]
    fn test_rms_norm_forward_to_empty_does_not_panic() {
        let norm = RmsNorm::new(vec![], 1e-5);
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        norm.forward_to(&input, &mut output); // must not panic
    }

    #[test]
    fn test_rms_norm_output_rms_is_approximately_one() {
        // After RMSNorm with identity weights, the RMS of the output should be ~1.
        let n = 8;
        let weight = vec![1.0f32; n];
        let norm = RmsNorm::new(weight, 1e-5);
        let input: Vec<f32> = (1..=n as i32).map(|i| i as f32).collect();
        let mut out = input.clone();
        norm.forward(&mut out);

        let sum_sq: f32 = out.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-4,
            "output RMS should be ~1.0 with identity weights, got {rms}"
        );
    }
}

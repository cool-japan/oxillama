//! LayerNorm (Layer Normalization) with optional bias.
//!
//! Used by GPT-style architectures such as StarCoder/GPT-BigCode.
//! Unlike RMSNorm (which only normalises by the root mean square),
//! LayerNorm subtracts the mean before dividing by the standard deviation.
//!
//! Formula: `y[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]`

/// LayerNorm layer with learned scale (`weight`) and optional shift (`bias`).
///
/// When `bias` is `None` the shift term is omitted (i.e., bias is treated as
/// all-zeros), matching architectures that use bias-free layer normalisation.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Learned scale weights (gamma), one per hidden dimension.
    pub weight: Vec<f32>,
    /// Optional learned shift (beta), one per hidden dimension.
    pub bias: Option<Vec<f32>>,
    /// Small constant added to variance for numerical stability.
    pub eps: f32,
}

impl LayerNorm {
    /// Create a new LayerNorm with the given scale and optional bias.
    pub fn new(weight: Vec<f32>, bias: Option<Vec<f32>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Apply LayerNorm to `input`, writing the normalised output to `output`.
    ///
    /// Both slices must have the same length.
    ///
    /// `output[i] = (input[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]`
    pub fn forward_to(&self, input: &[f32], output: &mut [f32]) {
        let n = input.len();
        if n == 0 {
            return;
        }

        // Compute mean
        let mean: f32 = input.iter().sum::<f32>() / n as f32;

        // Compute variance: E[(x - mean)^2]
        let var: f32 = input
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;

        let inv_std = 1.0 / (var + self.eps).sqrt();

        match &self.bias {
            Some(bias) => {
                for ((o, &inp), (&w, &b)) in output
                    .iter_mut()
                    .zip(input.iter())
                    .zip(self.weight.iter().zip(bias.iter()))
                {
                    *o = (inp - mean) * inv_std * w + b;
                }
            }
            None => {
                for ((o, &inp), &w) in output.iter_mut().zip(input.iter()).zip(self.weight.iter()) {
                    *o = (inp - mean) * inv_std * w;
                }
            }
        }
    }

    /// Apply LayerNorm in-place.
    ///
    /// `x[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]`
    pub fn forward(&self, x: &mut [f32]) {
        let n = x.len();
        if n == 0 {
            return;
        }

        let mean: f32 = x.iter().sum::<f32>() / n as f32;
        let var: f32 = x
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;

        let inv_std = 1.0 / (var + self.eps).sqrt();

        match &self.bias {
            Some(bias) => {
                for ((v, &w), &b) in x.iter_mut().zip(self.weight.iter()).zip(bias.iter()) {
                    *v = (*v - mean) * inv_std * w + b;
                }
            }
            None => {
                for (v, &w) in x.iter_mut().zip(self.weight.iter()) {
                    *v = (*v - mean) * inv_std * w;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_identity_weights_no_bias() {
        // With weight=1 and no bias, output should be z-score of input.
        let weight = vec![1.0f32; 4];
        let norm = LayerNorm::new(weight, None, 1e-5);
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        norm.forward_to(&input, &mut output);

        // mean = 2.5, var = 1.25
        let mean = 2.5_f32;
        let var = 1.25_f32;
        let inv_std = 1.0 / (var + 1e-5_f32).sqrt();

        for (i, &o) in output.iter().enumerate() {
            let expected = (input[i] - mean) * inv_std;
            assert!(
                (o - expected).abs() < 1e-4,
                "output[{i}] = {o}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let weight = vec![2.0f32; 2];
        let bias = vec![0.5f32; 2];
        let norm = LayerNorm::new(weight, Some(bias), 1e-5);
        let input = vec![1.0f32, 3.0]; // mean=2, var=1
        let mut output = vec![0.0f32; 2];
        norm.forward_to(&input, &mut output);

        // mean=2, var=1, inv_std ≈ 1/(1+1e-5)^0.5 ≈ 1.0
        // output[0] = (1-2)*1.0 * 2 + 0.5 = -2 + 0.5 = -1.5
        // output[1] = (3-2)*1.0 * 2 + 0.5 =  2 + 0.5 =  2.5
        assert!((output[0] - (-1.5)).abs() < 1e-4, "output[0]={}", output[0]);
        assert!((output[1] - 2.5).abs() < 1e-4, "output[1]={}", output[1]);
    }

    #[test]
    fn test_layer_norm_forward_inplace() {
        let weight = vec![1.0f32; 3];
        let norm = LayerNorm::new(weight, None, 1e-5);
        let mut x = vec![1.0f32, 2.0, 3.0];
        norm.forward(&mut x);

        // After normalisation the mean should be ~0 and std ~1.
        let mean: f32 = x.iter().sum::<f32>() / 3.0;
        assert!(
            mean.abs() < 1e-4,
            "mean after LayerNorm should be ~0, got {mean}"
        );
    }

    #[test]
    fn test_layer_norm_empty() {
        let norm = LayerNorm::new(vec![], None, 1e-5);
        let mut x: Vec<f32> = vec![];
        norm.forward(&mut x); // must not panic
    }

    #[test]
    fn test_layer_norm_uniform_input() {
        // Uniform input => zero variance. LayerNorm should produce all zeros
        // (with identity weights, no bias) because (x - mean) = 0 for all elements.
        let weight = vec![1.0f32; 4];
        let norm = LayerNorm::new(weight, None, 1e-5);
        let mut x = vec![3.0f32; 4];
        norm.forward(&mut x);
        for &v in x.iter() {
            assert!(
                v.abs() < 1e-4,
                "uniform input should give ~0 output, got {v}"
            );
        }
    }
}

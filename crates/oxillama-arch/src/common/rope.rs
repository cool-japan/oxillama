//! Rotary Position Embedding (RoPE).
//!
//! Applies rotary position embeddings to query and key tensors,
//! enabling the model to encode relative positional information.
//!
//! Supports three scaling modes:
//! - [`RopeScalingType::Standard`]: no scaling (default)
//! - [`RopeScalingType::Linear`]: divide all frequencies by `scaling_factor`
//! - [`RopeScalingType::Yarn`]: NTK-by-parts scaling (YaRN paper)

/// RoPE frequency scaling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RopeScalingType {
    /// Standard RoPE — no scaling (default).
    #[default]
    Standard,
    /// Linear frequency scaling: divide all frequencies by `rope_scaling_factor`.
    /// Extends context proportionally.
    Linear,
    /// YaRN (NTK-by-parts): scales low-frequency dims less aggressively,
    /// preserving short-range attention quality while extending long-range context.
    Yarn,
}

/// Precomputed RoPE frequency table.
///
/// Contains cos/sin values for each position and head dimension pair.
/// These are computed once during model loading and reused for every token.
#[derive(Debug, Clone)]
pub struct RopeTable {
    /// Cosine values: [max_seq_len, head_dim / 2].
    pub cos: Vec<f32>,
    /// Sine values: [max_seq_len, head_dim / 2].
    pub sin: Vec<f32>,
    /// Half the head dimension (number of rotation pairs).
    pub half_dim: usize,
    /// Maximum precomputed sequence length.
    pub max_seq_len: usize,
}

/// Compute the adjusted frequency for dimension index `i` under the given scaling.
fn compute_freq(
    i: usize,
    half_dim: usize,
    base: f32,
    scaling_type: RopeScalingType,
    scaling_factor: f32,
) -> f32 {
    let base_freq = 1.0 / base.powf((2 * i) as f32 / (2 * half_dim) as f32);
    match scaling_type {
        RopeScalingType::Standard => base_freq,
        RopeScalingType::Linear => base_freq / scaling_factor,
        RopeScalingType::Yarn => {
            // YaRN NTK-by-parts (https://arxiv.org/abs/2309.00071):
            // Smoothly interpolates between fully-scaled and unscaled frequencies
            // based on the wavelength relative to the original context length.
            let beta_fast = 32.0_f32;
            let beta_slow = 1.0_f32;
            let original_context = 4096.0_f32;
            let wavelength = (2.0 * std::f32::consts::PI) / base_freq;
            let low = original_context / beta_fast;
            let high = original_context / beta_slow;
            let ramp = ((wavelength - low) / (high - low).max(1e-6)).clamp(0.0, 1.0);
            // ramp=0 → fully scaled; ramp=1 → unscaled
            (base_freq / scaling_factor) * (1.0 - ramp) + base_freq * ramp
        }
    }
}

impl RopeTable {
    /// Precompute the RoPE frequency table with a given scaling strategy.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head.
    /// * `max_seq_len` - Maximum sequence length to precompute.
    /// * `base` - Base frequency (typically 10000.0).
    /// * `scaling_type` - Which scaling strategy to apply.
    /// * `scaling_factor` - Scaling multiplier (`1.0` = no scaling).
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        scaling_type: RopeScalingType,
        scaling_factor: f32,
    ) -> Self {
        let half_dim = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = compute_freq(i, half_dim, base, scaling_type, scaling_factor);
                let theta = pos as f32 * freq;
                cos.push(theta.cos());
                sin.push(theta.sin());
            }
        }

        Self {
            cos,
            sin,
            half_dim,
            max_seq_len,
        }
    }

    /// Create a standard (unscaled) RoPE table. Equivalent to the original `new()`.
    pub fn new_standard(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        Self::new(head_dim, max_seq_len, base, RopeScalingType::Standard, 1.0)
    }

    /// Apply RoPE to a single head vector in-place.
    ///
    /// # Arguments
    /// * `x` - Head vector of length `head_dim`.
    /// * `position` - Sequence position index.
    pub fn apply(&self, x: &mut [f32], position: usize) {
        let half = self.half_dim;
        let offset = position * half;

        for i in 0..half {
            let x0 = x[i];
            let x1 = x[i + half];
            let cos_val = self.cos[offset + i];
            let sin_val = self.sin[offset + i];
            x[i] = x0 * cos_val - x1 * sin_val;
            x[i + half] = x0 * sin_val + x1 * cos_val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_position_zero() {
        let table = RopeTable::new(4, 16, 10000.0, RopeScalingType::Standard, 1.0);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let original = x.clone();
        table.apply(&mut x, 0);

        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
        }
    }

    #[test]
    fn standard_and_linear_scale_1_match() {
        let t1 = RopeTable::new(64, 8, 10000.0, RopeScalingType::Standard, 1.0);
        let t2 = RopeTable::new(64, 8, 10000.0, RopeScalingType::Linear, 1.0);
        for (a, b) in t1.cos.iter().zip(t2.cos.iter()) {
            assert!((a - b).abs() < 1e-6, "cos mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn linear_scaling_extends_freq_range() {
        let standard = RopeTable::new(64, 8, 10000.0, RopeScalingType::Standard, 1.0);
        let linear = RopeTable::new(64, 8, 10000.0, RopeScalingType::Linear, 4.0);
        let differs = standard
            .cos
            .iter()
            .zip(linear.cos.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "Linear-4x should differ from standard");
    }

    #[test]
    fn yarn_produces_valid_values() {
        let yarn = RopeTable::new(64, 16, 10000.0, RopeScalingType::Yarn, 4.0);
        for v in yarn.cos.iter().chain(yarn.sin.iter()) {
            assert!(v.abs() <= 1.0 + 1e-6, "out of range: {v}");
        }
    }

    #[test]
    fn apply_in_place_unchanged_shape() {
        let table = RopeTable::new_standard(8, 4, 10000.0);
        let mut x = vec![1.0f32; 8];
        table.apply(&mut x, 0);
        assert_eq!(x.len(), 8);
    }
}

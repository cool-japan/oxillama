//! Rotary Position Embedding (RoPE).
//!
//! Applies rotary position embeddings to query and key tensors,
//! enabling the model to encode relative positional information.

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

impl RopeTable {
    /// Precompute the RoPE frequency table.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head.
    /// * `max_seq_len` - Maximum sequence length to precompute.
    /// * `base` - Base frequency (typically 10000.0).
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        let half_dim = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half_dim + i] = angle.cos();
                sin[pos * half_dim + i] = angle.sin();
            }
        }

        Self {
            cos,
            sin,
            half_dim,
            max_seq_len,
        }
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
        let table = RopeTable::new(4, 16, 10000.0);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let original = x.clone();
        table.apply(&mut x, 0);

        // At position 0, cos(0) = 1.0 and sin(0) = 0.0
        // So the output should be approximately the same as input.
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
        }
    }
}

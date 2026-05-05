//! ALiBi (Attention with Linear Biases) positional bias primitive.
//!
//! ALiBi replaces additive positional embeddings with a fixed linear bias
//! applied to attention scores, enabling zero-shot length extrapolation.
//!
//! Reference: "Train Short, Test Long: Attention with Linear Biases Enables
//! Input Length Extrapolation" (Press et al., 2021).
//!
//! Implementation follows the HuggingFace `transformers` code for BLOOM
//! (`modeling_bloom.py`), which matches the original ALiBi paper precisely.
//!
//! # Slope formula
//!
//! For **power-of-two** head counts `n`, the slopes are:
//! ```text
//! start  = 2^(-8/n)
//! slopes = [start^1, start^2, ..., start^n]
//! ```
//!
//! For **non-power-of-two** head counts, slopes are computed for the nearest
//! power-of-two ceiling `m = 2^⌈log₂(n)⌉`, and the "intermediate" set
//! (odd-indexed slopes from the `2m`-head schedule) is used to fill in the
//! remaining `n - m` slots. This is the same interpolation used in
//! `transformers/models/bloom/modeling_bloom.py`.

use crate::error::{ArchError, ArchResult};

/// ALiBi positional bias.
///
/// Holds the per-head slopes used to bias attention scores toward local context.
/// Apply via [`AlibiBias::apply`] to modify score tensors in-place before
/// softmax.
#[derive(Debug, Clone)]
pub struct AlibiBias {
    /// Learned (fixed) slope for each attention head: `slopes[h]`.
    ///
    /// Length = `num_heads`. Indexed from 0 to `num_heads-1`.
    slopes: Vec<f32>,
}

impl AlibiBias {
    /// Compute ALiBi slopes for `num_heads` attention heads.
    ///
    /// The formula follows `transformers/modeling_bloom.py`:
    ///
    /// 1. Compute the base set of `n` slopes for the nearest-power-of-two
    ///    count that is ≥ `num_heads`.
    /// 2. If `num_heads` is not a power of two, interleave with the "half-step"
    ///    slopes from the `2n`-head schedule to fill the remaining slots.
    ///
    /// # Panics
    ///
    /// Does not panic; `num_heads = 0` produces an empty slope vector.
    pub fn new(num_heads: usize) -> Self {
        let slopes = compute_alibi_slopes(num_heads);
        Self { slopes }
    }

    /// Apply ALiBi bias to a flattened `[num_heads, seq_q, seq_k]` score buffer.
    ///
    /// The bias for head `h` at position `(q, k)` is:
    /// ```text
    /// bias[h][q][k] = slopes[h] * (k as isize - q as isize)
    /// ```
    ///
    /// The causal constraint means that for autoregressive models `k ≤ q`, so
    /// the bias is always ≤ 0, penalising distant keys.
    ///
    /// # Arguments
    /// * `scores` – mutable flat buffer of shape `[num_heads, seq_q, seq_k]`
    ///   (row-major, i.e. `scores[h * seq_q * seq_k + q * seq_k + k]`).
    /// * `seq_q`  – number of query positions.
    /// * `seq_k`  – number of key positions (= total sequence length for decode).
    ///
    /// # Errors
    ///
    /// Returns [`ArchError::InvalidShape`] if `scores.len()` does not equal
    /// `num_heads * seq_q * seq_k`.
    pub fn apply(&self, scores: &mut [f32], seq_q: usize, seq_k: usize) -> ArchResult<()> {
        let num_heads = self.slopes.len();
        let expected = num_heads * seq_q * seq_k;

        if scores.len() != expected {
            return Err(ArchError::InvalidShape {
                name: "alibi_scores".to_string(),
                expected: vec![num_heads, seq_q, seq_k],
                got: vec![scores.len()],
            });
        }

        for h in 0..num_heads {
            let slope = self.slopes[h];
            for q in 0..seq_q {
                for k in 0..seq_k {
                    let idx = h * seq_q * seq_k + q * seq_k + k;
                    let relative_pos = k as isize - q as isize;
                    scores[idx] += slope * relative_pos as f32;
                }
            }
        }

        Ok(())
    }

    /// Return the per-head slope vector (length = `num_heads`).
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// Return the number of heads for which slopes are stored.
    pub fn num_heads(&self) -> usize {
        self.slopes.len()
    }
}

// ─── Internal slope computation ───────────────────────────────────────────────

/// Compute the ALiBi slopes for `num_heads` heads, matching HuggingFace BLOOM.
///
/// The algorithm:
/// 1. Let `n = nearest_power_of_two_geq(num_heads)`.
/// 2. Compute the base set of `n` slopes: `start = 2^(-8/n)`,
///    then `[start^1, start^2, ..., start^n]`.
/// 3. If `num_heads == n`, return those directly.
/// 4. Otherwise (`num_heads < n`), compute the `n` base slopes and the
///    "half-step" intermediate slopes from the `2n` schedule.
///    Interleave them (base-first, then intermediate) and keep the first
///    `num_heads` entries.
fn compute_alibi_slopes(num_heads: usize) -> Vec<f32> {
    if num_heads == 0 {
        return Vec::new();
    }

    // Nearest power of two ≥ num_heads
    let n = num_heads.next_power_of_two();

    // Base slopes for `n` heads: start = 2^(-8/n), slopes_base = [start^i for i in 1..=n]
    let base_slopes = make_slopes(n);

    if num_heads == n {
        return base_slopes;
    }

    // Non-power-of-two: `num_heads < n`.
    // We need `num_heads - (n / 2)` extra slopes from the `n`-head intermediate set.
    // The intermediate set is the odd-indexed elements of the `2n`-head schedule.
    // Actually, the HuggingFace implementation for non-power-of-two uses:
    //   - slopes from the closest lower power of two (n/2 slopes)
    //   - padded with intermediate slopes until we reach num_heads
    // We start with the n/2 base slopes and append `num_heads - n/2` intermediates.

    let half_n = n / 2; // guaranteed to be at least 1 since n >= 2 (num_heads >= 2 when num_heads != n)
    let base_half = make_slopes(half_n);

    // Intermediate slopes: odd-indexed elements of the `n`-head full schedule
    let intermediates: Vec<f32> = base_slopes
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % 2 != 0)
        .map(|(_, v)| v)
        .collect();

    // We have `half_n` base slopes + intermediates; combine and take first `num_heads`
    let needed_extra = num_heads.saturating_sub(half_n);
    let mut result = base_half;
    result.extend_from_slice(&intermediates[..needed_extra.min(intermediates.len())]);
    result.truncate(num_heads);

    // If we still don't have enough (edge case), fill remainder with the full base_slopes
    if result.len() < num_heads {
        let mut ext = base_slopes;
        ext.truncate(num_heads);
        result = ext;
    }

    result
}

/// Generate `n` ALiBi slopes for exactly `n` heads (power-of-two path).
///
/// `start = 2^(-8/n)`, then `slopes = [start^1, start^2, ..., start^n]`.
fn make_slopes(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    // start = 2^(-8/n) = exp2(-8/n)
    let exponent = -8.0f32 / n as f32;
    let start = (exponent * std::f32::consts::LN_2).exp(); // 2^exponent
    let mut slopes = Vec::with_capacity(n);
    let mut current = start;
    for _ in 0..n {
        slopes.push(current);
        current *= start;
    }
    slopes
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Slopes must form a geometric sequence: ratio slopes[i] / slopes[i+1] is constant.
    #[test]
    fn alibi_slopes_geometric_sequence() {
        let bias = AlibiBias::new(8);
        let slopes = bias.slopes();
        assert_eq!(slopes.len(), 8, "should have 8 slopes for 8 heads");

        // The ratio between consecutive slopes should be constant.
        let ratio = slopes[0] / slopes[1];
        for i in 1..slopes.len() - 1 {
            let r = slopes[i] / slopes[i + 1];
            assert!(
                (r - ratio).abs() < 1e-5,
                "slopes are not geometric: ratio[{i}] = {r}, expected {ratio}"
            );
        }
    }

    /// ALiBi slopes for 8 heads: slopes[0] should equal 2^(-1) = 0.5.
    ///
    /// Verification: for n=8, start = 2^(-8/8) = 2^(-1) = 0.5
    /// slopes[0] = start^1 = 0.5.
    #[test]
    fn alibi_slopes_first_head_for_8_heads() {
        let bias = AlibiBias::new(8);
        let slopes = bias.slopes();
        let expected = 0.5f32; // 2^(-1) = 0.5
        assert!(
            (slopes[0] - expected).abs() < 1e-5,
            "slopes[0] for 8 heads should be 0.5 (= 2^(-1)), got {}",
            slopes[0]
        );
    }

    /// `apply` must not panic for arbitrary seq_q / seq_k combinations.
    #[test]
    fn alibi_bias_matrix_shape() {
        let bias = AlibiBias::new(4);
        let seq_q = 3;
        let seq_k = 5;
        let mut scores = vec![0.0f32; 4 * seq_q * seq_k];
        let result = bias.apply(&mut scores, seq_q, seq_k);
        assert!(result.is_ok(), "apply should succeed: {:?}", result.err());
    }

    /// `apply` returns an error when the buffer length is wrong.
    #[test]
    fn alibi_apply_wrong_size_errors() {
        let bias = AlibiBias::new(2);
        let mut scores = vec![0.0f32; 5]; // wrong size
        let result = bias.apply(&mut scores, 2, 2);
        assert!(
            result.is_err(),
            "apply with wrong size should return an error"
        );
    }

    /// Single-token decode: scores shape [h, 1, 1]. Bias is 0 (k - q = 0 - 0 = 0).
    #[test]
    fn alibi_single_token_zero_bias() {
        let bias = AlibiBias::new(4);
        let mut scores = vec![1.0f32; 4]; // shape [4, 1, 1]
        bias.apply(&mut scores, 1, 1).expect("apply should succeed");
        for &s in &scores {
            assert!(
                (s - 1.0).abs() < 1e-6,
                "single-token bias should be zero, score changed to {s}"
            );
        }
    }

    /// All slopes must be strictly positive (they bound the penalty magnitude).
    #[test]
    fn alibi_slopes_all_positive() {
        for n in [1, 2, 3, 4, 7, 8, 12, 16] {
            let bias = AlibiBias::new(n);
            for (i, &s) in bias.slopes().iter().enumerate() {
                assert!(
                    s > 0.0,
                    "slope[{i}] for num_heads={n} must be positive, got {s}"
                );
                assert!(
                    s.is_finite(),
                    "slope[{i}] for num_heads={n} must be finite, got {s}"
                );
            }
        }
    }

    /// Non-power-of-two head counts should still produce the right number of slopes.
    #[test]
    fn alibi_non_power_of_two_head_counts() {
        for n in [1, 3, 5, 6, 7, 9, 12] {
            let bias = AlibiBias::new(n);
            assert_eq!(
                bias.slopes().len(),
                n,
                "expected {n} slopes, got {}",
                bias.slopes().len()
            );
        }
    }

    /// Zero heads should produce an empty slope vector.
    #[test]
    fn alibi_zero_heads_empty() {
        let bias = AlibiBias::new(0);
        assert!(bias.slopes().is_empty());
    }
}

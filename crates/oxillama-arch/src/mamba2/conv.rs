//! 1-D depthwise causal convolution for Mamba-2 models.
//!
//! Implements the depthwise conv1d used in the Mamba-2 input mixing stage.
//! Each channel is filtered independently by a kernel of width `d_conv`.
//! The convolution is **causal**: position `t` can only see positions ≤ `t`.
//! Left-zero-padding ensures the output has the same length as the input.
//!
//! After the convolution, a SiLU (= `x * sigmoid(x)`) activation is applied
//! element-wise.

// ─── Public function ──────────────────────────────────────────────────────────

/// Causal 1-D depthwise convolution with SiLU activation.
///
/// Each output element at position `t` and channel `i` is:
///
/// ```text
/// raw[t, i] = sum_{k=0}^{d_conv-1} weight[i, k] * x[(t - (d_conv-1) + k), i]
///             (positions < 0 are treated as zero)
/// out[t, i]  = silu(raw[t, i] + bias[i])
/// ```
///
/// # Arguments
/// * `x`       – Input `[seq_len × d_inner]` row-major.
/// * `weight`  – Kernel `[d_inner × d_conv]` row-major.
/// * `bias`    – Bias `[d_inner]`.
/// * `seq_len` – Number of input tokens.
/// * `d_inner` – Number of channels (depthwise: one kernel per channel).
/// * `d_conv`  – Convolution kernel width (causal receptive field).
///
/// # Returns
/// Output `[seq_len × d_inner]` row-major after SiLU activation.
///
/// # Panics
/// Panics if `x.len() != seq_len * d_inner`, `weight.len() != d_inner * d_conv`,
/// or `bias.len() != d_inner`. These invariants are callers' responsibility.
pub fn conv1d_depthwise(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    seq_len: usize,
    d_inner: usize,
    d_conv: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), seq_len * d_inner);
    debug_assert_eq!(weight.len(), d_inner * d_conv);
    debug_assert_eq!(bias.len(), d_inner);

    let mut out = vec![0.0f32; seq_len * d_inner];

    for t in 0..seq_len {
        for i in 0..d_inner {
            let mut acc = bias[i];
            // Convolve with the d_conv-wide causal kernel.
            // k=0 corresponds to the oldest position in the receptive field.
            for k in 0..d_conv {
                // Source position: t - (d_conv - 1) + k
                // When this is negative, the input is implicitly zero (left-pad).
                let src_signed = t as i64 - (d_conv as i64 - 1) + k as i64;
                if src_signed >= 0 {
                    let src = src_signed as usize;
                    let w = weight[i * d_conv + k];
                    acc += w * x[src * d_inner + i];
                }
            }
            // Apply SiLU: x * sigmoid(x)
            out[t * d_inner + i] = acc * sigmoid(acc);
        }
    }

    out
}

/// Numerically stable sigmoid: `1 / (1 + exp(-x))`.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// conv1d_depthwise matches a scalar Python-style reference implementation.
    ///
    /// Reference (Python):
    /// ```python
    /// def conv1d_depthwise_ref(x, weight, bias, seq_len, d_inner, d_conv):
    ///     out = np.zeros((seq_len, d_inner))
    ///     for t in range(seq_len):
    ///         for i in range(d_inner):
    ///             acc = bias[i]
    ///             for k in range(d_conv):
    ///                 src = t - (d_conv - 1) + k
    ///                 if src >= 0:
    ///                     acc += weight[i, k] * x[src, i]
    ///             out[t, i] = acc * sigmoid(acc)
    ///     return out
    /// ```
    #[test]
    fn conv1d_depthwise_matches_reference() {
        let seq_len = 4;
        let d_inner = 2;
        let d_conv = 3;

        // x[t, i] = (t * d_inner + i) as f32 * 0.1 (small values)
        // x layout: [x[0,0], x[0,1], x[1,0], x[1,1], x[2,0], x[2,1], x[3,0], x[3,1]]
        let x: Vec<f32> = (0..seq_len * d_inner).map(|idx| idx as f32 * 0.1).collect();

        // weight[i, k] = 1.0 / (d_conv as f32) so each kernel sums to 1.
        let weight: Vec<f32> = vec![1.0 / d_conv as f32; d_inner * d_conv];
        // bias = zero
        let bias: Vec<f32> = vec![0.0f32; d_inner];

        // Scalar reference.
        let mut reference = vec![0.0f32; seq_len * d_inner];
        for t in 0..seq_len {
            for i in 0..d_inner {
                let mut acc = bias[i];
                for k in 0..d_conv {
                    let src_signed = t as i64 - (d_conv as i64 - 1) + k as i64;
                    if src_signed >= 0 {
                        let src = src_signed as usize;
                        acc += weight[i * d_conv + k] * x[src * d_inner + i];
                    }
                }
                let silu = acc / (1.0 + (-acc).exp());
                reference[t * d_inner + i] = silu;
            }
        }

        let result = conv1d_depthwise(&x, &weight, &bias, seq_len, d_inner, d_conv);

        assert_eq!(result.len(), reference.len());
        for (idx, (got, expected)) in result.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "result[{idx}] = {got} != reference[{idx}] = {expected}"
            );
        }
    }

    /// Left-padding (zero) for positions before the sequence start is correct.
    #[test]
    fn conv1d_causal_padding_is_zero() {
        let seq_len = 1;
        let d_inner = 1;
        let d_conv = 3;

        // Single token: receptive field covers positions -2, -1, 0 of the input.
        // Positions -2 and -1 are zero (left-pad). Position 0 is x[0, 0].
        let x = vec![1.0f32];
        // Each kernel element = 1.0 so raw = 0 + 0 + 1*x[0,0] = 1.0.
        let weight = vec![1.0f32; d_inner * d_conv];
        let bias = vec![0.0f32; d_inner];

        let result = conv1d_depthwise(&x, &weight, &bias, seq_len, d_inner, d_conv);

        // raw = 1.0; silu(1.0) = 1.0 * sigmoid(1.0)
        let expected = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!(
            (result[0] - expected).abs() < 1e-6,
            "result={}, expected={expected}",
            result[0]
        );
    }

    /// Bias is correctly added before SiLU.
    #[test]
    fn conv1d_bias_applied_before_silu() {
        let seq_len = 1;
        let d_inner = 1;
        let d_conv = 1;

        // With zero input and kernel, raw = bias.
        let x = vec![0.0f32];
        let weight = vec![0.0f32; d_inner * d_conv];
        let bias = vec![2.0f32];

        let result = conv1d_depthwise(&x, &weight, &bias, seq_len, d_inner, d_conv);

        // raw = 2.0; silu(2.0) = 2 * sigmoid(2) ≈ 1.761
        let silu_2 = 2.0f32 / (1.0 + (-2.0f32).exp());
        assert!(
            (result[0] - silu_2).abs() < 1e-5,
            "bias must be applied before SiLU: got {}, expected {silu_2}",
            result[0]
        );
    }

    /// Zero input and zero bias → zero output (SiLU(0) = 0).
    #[test]
    fn conv1d_zero_input_zero_output() {
        let seq_len = 5;
        let d_inner = 3;
        let d_conv = 4;
        let x = vec![0.0f32; seq_len * d_inner];
        let weight = vec![0.5f32; d_inner * d_conv];
        let bias = vec![0.0f32; d_inner];
        let result = conv1d_depthwise(&x, &weight, &bias, seq_len, d_inner, d_conv);
        // SiLU(0) = 0, so all outputs should be zero.
        for (i, &v) in result.iter().enumerate() {
            assert!(
                v.abs() < 1e-7,
                "result[{i}] = {v} should be ~0 for zero input"
            );
        }
    }
}

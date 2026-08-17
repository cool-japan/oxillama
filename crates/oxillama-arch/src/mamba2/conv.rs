//! 1-D depthwise causal convolution for Mamba-2 models.
//!
//! Implements the depthwise conv1d used in the Mamba-2 input mixing stage.
//! Each channel is filtered independently by a kernel of width `d_conv`.
//! The convolution is **causal**: position `t` can only see positions ≤ `t`.
//!
//! After the convolution, a SiLU (= `x * sigmoid(x)`) activation is applied
//! element-wise, matching `xBC = ggml_silu(ctx0, ggml_add(ctx0,
//! ggml_ssm_conv(...), ssm_conv1d_b))` in `src/models/mamba-base.cpp`.
//!
//! # Two entry points
//!
//! | Function | History before position 0 |
//! |---|---|
//! | [`conv1d_depthwise`] | zero-padded (a whole-sequence, fresh-state convolution) |
//! | [`conv1d_depthwise_stateful`] | taken from a [`ConvRing`], then updated |
//!
//! A recurrent decoder **must** use the stateful form.  Calling the stateless
//! form with `seq_len == 1` leaves only tap `d_conv - 1` non-zero: taps
//! `0 ..= d_conv-2` all read left-padding, so the convolution degenerates into
//! `silu(bias + w[d_conv-1] * x)` and the block forgets everything before the
//! current token.
//!
//! # Tap ordering
//!
//! `k = 0` is the **oldest** position in the receptive field, matching
//! `ggml_compute_forward_ssm_conv_f32`:
//!
//! ```text
//! for (int i0 = 0; i0 < nc; ++i0) { sumf += s[i0 + i1*ncs] * c[i0 + i1*nc]; }
//! ```
//!
//! where `s` starts at offset `i2` (the token index) into a row whose first
//! `d_conv - 1` entries are the carried-over conv state.

use crate::error::{ArchError, ArchResult};
use crate::mamba2::state::ConvRing;

// ─── Public functions ─────────────────────────────────────────────────────────

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

/// Causal 1-D depthwise convolution with SiLU activation, carrying state.
///
/// Identical to [`conv1d_depthwise`] except that positions before the start of
/// `x` are read from `ring` instead of being treated as zero, and `ring` is
/// advanced by `seq_len` positions afterwards.  This is the Rust equivalent of
/// llama.cpp's
///
/// ```text
/// conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);
/// ...copy the last d_conv-1 columns back into conv_states_all...
/// xBC = ggml_silu(ggml_add(ggml_ssm_conv(ctx0, conv_x, ssm_conv1d), ssm_conv1d_b));
/// ```
///
/// # Arguments
/// * `x`        – Input `[seq_len × channels]` row-major.
/// * `weight`   – Kernel `[channels × d_conv]` row-major (`k = 0` = oldest tap).
/// * `bias`     – Bias `[channels]`.
/// * `ring`     – Shift register holding the previous `d_conv - 1` positions.
/// * `seq_len`  – Number of input tokens in this chunk.
/// * `channels` – Depthwise channel count (`d_inner + 2*n_group*d_state` for
///   Mamba-2, since `x`, `B` and `C` share the convolution).
/// * `d_conv`   – Kernel width.
///
/// # Returns
/// Output `[seq_len × channels]` row-major after SiLU.
///
/// # Errors
///
/// [`ArchError::InvalidShape`] when any slice length disagrees with the
/// declared dimensions, or when `ring` was built for a different geometry.
pub fn conv1d_depthwise_stateful(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    ring: &mut ConvRing,
    seq_len: usize,
    channels: usize,
    d_conv: usize,
) -> ArchResult<Vec<f32>> {
    let shape_err = |what: &str, expected: Vec<usize>, got: usize| ArchError::InvalidShape {
        name: format!("mamba2.conv1d.{what}"),
        expected,
        got: vec![got],
    };

    if d_conv == 0 {
        return Err(ArchError::InvalidConfig {
            detail: "mamba2.conv1d: d_conv must be >= 1".to_string(),
        });
    }
    if x.len() != seq_len * channels {
        return Err(shape_err("input", vec![seq_len, channels], x.len()));
    }
    if weight.len() != channels * d_conv {
        return Err(shape_err("weight", vec![channels, d_conv], weight.len()));
    }
    if bias.len() != channels {
        return Err(shape_err("bias", vec![channels], bias.len()));
    }
    if ring.conv_dim() != channels || ring.width() != d_conv - 1 {
        return Err(shape_err(
            "state",
            vec![d_conv - 1, channels],
            ring.width() * ring.conv_dim(),
        ));
    }

    let width = ring.width();
    let mut out = vec![0.0f32; seq_len * channels];

    for t in 0..seq_len {
        for ch in 0..channels {
            let mut acc = bias[ch];
            for k in 0..d_conv {
                // Source position relative to the start of this chunk.
                let time = t as i64 - (d_conv as i64 - 1) + k as i64;
                let sample = if time >= 0 {
                    x[time as usize * channels + ch]
                } else {
                    // time is in [-width, -1]; slot `width + time` is that
                    // position in the ring (slot 0 = oldest).
                    ring.get((width as i64 + time) as usize, ch)
                };
                acc += weight[ch * d_conv + k] * sample;
            }
            out[t * channels + ch] = acc * sigmoid(acc);
        }
    }

    ring.push_window(x, seq_len)?;

    Ok(out)
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

    // ─── Stateful variant ─────────────────────────────────────────────────

    /// Streaming one token at a time must equal convolving the whole sequence.
    ///
    /// This is the property the stateless call in the Mamba-2 block violated:
    /// with `seq_len == 1` and no carried state, taps `0 ..= d_conv-2` always
    /// read zero padding.
    #[test]
    fn stateful_streaming_matches_whole_sequence() {
        let seq_len = 6;
        let channels = 3;
        let d_conv = 4;

        let x: Vec<f32> = (0..seq_len * channels)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
            .collect();
        let weight: Vec<f32> = (0..channels * d_conv)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.3)
            .collect();
        let bias: Vec<f32> = (0..channels).map(|i| i as f32 * 0.1).collect();

        // Whole-sequence reference: a fresh ring is all-zero, so this must
        // agree with the stateless (zero-padded) implementation too.
        let mut ring_batch = ConvRing::new(channels, d_conv);
        let batch = conv1d_depthwise_stateful(
            &x,
            &weight,
            &bias,
            &mut ring_batch,
            seq_len,
            channels,
            d_conv,
        )
        .expect("batch conv");

        let stateless = conv1d_depthwise(&x, &weight, &bias, seq_len, channels, d_conv);
        for (i, (a, b)) in batch.iter().zip(stateless.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "fresh state must reproduce zero-padding at index {i}: {a} vs {b}"
            );
        }

        // Token-at-a-time streaming through a carried ring.
        let mut ring_stream = ConvRing::new(channels, d_conv);
        let mut streamed = Vec::with_capacity(seq_len * channels);
        for t in 0..seq_len {
            let step = conv1d_depthwise_stateful(
                &x[t * channels..(t + 1) * channels],
                &weight,
                &bias,
                &mut ring_stream,
                1,
                channels,
                d_conv,
            )
            .expect("streamed conv");
            streamed.extend_from_slice(&step);
        }

        assert_eq!(streamed.len(), batch.len());
        for (i, (a, b)) in streamed.iter().zip(batch.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "streamed[{i}] = {a} != batch[{i}] = {b}: the conv state is not \
                 being carried across single-token steps"
            );
        }
    }

    /// Every tap must be reachable once history exists.
    ///
    /// Kernel `[1, 0, 0]` selects the position `d_conv-1` steps back.  After
    /// two tokens the third step must return `silu(x[0])`, which is only
    /// possible if the ring retains position 0.
    #[test]
    fn stateful_oldest_tap_reads_history() {
        let channels = 1;
        let d_conv = 3;
        let weight = vec![1.0f32, 0.0, 0.0];
        let bias = vec![0.0f32];

        let mut ring = ConvRing::new(channels, d_conv);
        let inputs = [2.0f32, -1.0, 0.5, 3.0];
        let mut outs = Vec::new();
        for v in inputs {
            let o = conv1d_depthwise_stateful(&[v], &weight, &bias, &mut ring, 1, channels, d_conv)
                .expect("step");
            outs.push(o[0]);
        }

        // Step 2 sees position 0 through tap k=0.
        let expected = inputs[0] / (1.0 + (-inputs[0]).exp());
        assert!(
            (outs[2] - expected).abs() < 1e-6,
            "tap 0 at t=2 must read x[0]={}: got {}, expected {expected}",
            inputs[0],
            outs[2]
        );
        // Step 3 sees position 1.
        let expected3 = inputs[1] / (1.0 + (-inputs[1]).exp());
        assert!(
            (outs[3] - expected3).abs() < 1e-6,
            "tap 0 at t=3 must read x[1]={}: got {}",
            inputs[1],
            outs[3]
        );
    }

    /// Clearing the ring restores fresh-sequence behaviour exactly.
    #[test]
    fn stateful_clear_restores_fresh_behaviour() {
        let channels = 2;
        let d_conv = 3;
        let weight = vec![0.5f32, -0.25, 0.75, 1.0, 0.5, -0.5];
        let bias = vec![0.1f32, -0.1];
        let x = vec![1.0f32, 2.0];

        let mut ring = ConvRing::new(channels, d_conv);
        let first = conv1d_depthwise_stateful(&x, &weight, &bias, &mut ring, 1, channels, d_conv)
            .expect("first");
        let polluted =
            conv1d_depthwise_stateful(&x, &weight, &bias, &mut ring, 1, channels, d_conv)
                .expect("second");
        assert!(
            first
                .iter()
                .zip(polluted.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "history must change the result, otherwise the test proves nothing"
        );

        ring.clear();
        let after_clear =
            conv1d_depthwise_stateful(&x, &weight, &bias, &mut ring, 1, channels, d_conv)
                .expect("after clear");
        for (i, (a, b)) in first.iter().zip(after_clear.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "index {i}: {a} vs {b}");
        }
    }

    /// Mismatched slice lengths are typed errors, not panics.
    #[test]
    fn stateful_rejects_bad_shapes() {
        let mut ring = ConvRing::new(2, 3);
        let w = vec![0.0f32; 6];
        let b = vec![0.0f32; 2];
        assert!(
            conv1d_depthwise_stateful(&[0.0; 3], &w, &b, &mut ring, 1, 2, 3).is_err(),
            "short input must error"
        );
        assert!(
            conv1d_depthwise_stateful(&[0.0; 2], &[0.0; 5], &b, &mut ring, 1, 2, 3).is_err(),
            "short weight must error"
        );
        assert!(
            conv1d_depthwise_stateful(&[0.0; 2], &w, &[0.0; 1], &mut ring, 1, 2, 3).is_err(),
            "short bias must error"
        );
        let mut wrong = ConvRing::new(4, 3);
        assert!(
            conv1d_depthwise_stateful(&[0.0; 2], &w, &b, &mut wrong, 1, 2, 3).is_err(),
            "ring built for another geometry must error"
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

//! Selective-scan (SSM) primitive for Mamba-2 models.
//!
//! Implements the sequential O(n) selective scan used in Mamba-2 blocks.
//!
//! ## Key mathematical detail
//!
//! The `A` matrix is stored in GGUF as `log(A)` (log of the absolute value of
//! the negative diagonal). The discrete-time update rule is:
//!
//! ```text
//! A_discrete[t, s, i] = exp(-Δ[t, i] * exp(log_A[s, i]))
//!                      ≠ exp(-Δ[t, i] * log_A[s, i])   ← WRONG
//! ```
//!
//! Then:
//! ```text
//! B_discrete[t, s, i] = Δ[t, i] * B[t, s]
//! h[s, i]             = A_discrete * h[s, i] + B_discrete * u[t, i]
//! y[t, i]            += C[t, s] * h[s, i]
//! y[t, i]            += D[i] * u[t, i]  (skip connection)
//! ```

use crate::common::sequence_state::SsmLayerState;

// ─── Public function ──────────────────────────────────────────────────────────

/// Sequential selective scan for one Mamba-2 SSM layer.
///
/// # Arguments
/// * `u`       – Input `[seq_len × d_inner]` row-major.
/// * `delta`   – Time steps `[seq_len × d_inner]` row-major (already softplus'd).
/// * `log_a`   – Log-parameterised A: `[d_state × d_inner]` row-major.
///   **Must be exp'd before discrete-time conversion.**
/// * `b`       – Input-dependent B: `[seq_len × d_state]` row-major.
/// * `c`       – Output matrix C: `[seq_len × d_state]` row-major.
/// * `d`       – Skip-connection bias: `[d_inner]`.
/// * `seq_len` – Number of input tokens.
/// * `d_inner` – Inner dimension (channels).
/// * `d_state` – SSM state dimension.
/// * `state`   – Mutable per-layer recurrent state (updated in-place).
///
/// # Returns
/// Output `[seq_len × d_inner]` row-major.
///
/// # Panics (debug only)
/// Asserts that slice lengths match the declared dimensions.
#[allow(clippy::too_many_arguments)]
pub fn selective_scan_sequential(
    u: &[f32],
    delta: &[f32],
    log_a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
    state: &mut SsmLayerState,
) -> Vec<f32> {
    debug_assert_eq!(u.len(), seq_len * d_inner);
    debug_assert_eq!(delta.len(), seq_len * d_inner);
    debug_assert_eq!(log_a.len(), d_state * d_inner);
    debug_assert_eq!(b.len(), seq_len * d_state);
    debug_assert_eq!(c.len(), seq_len * d_state);
    debug_assert_eq!(d.len(), d_inner);
    debug_assert_eq!(state.h.len(), d_state * d_inner);

    let mut y = vec![0.0f32; seq_len * d_inner];

    for t in 0..seq_len {
        let u_t = &u[t * d_inner..(t + 1) * d_inner];
        let delta_t = &delta[t * d_inner..(t + 1) * d_inner];
        let b_t = &b[t * d_state..(t + 1) * d_state];
        let c_t = &c[t * d_state..(t + 1) * d_state];
        let y_t = &mut y[t * d_inner..(t + 1) * d_inner];

        for i in 0..d_inner {
            let dt = delta_t[i];

            for s in 0..d_state {
                // A_discrete = exp(-dt * exp(log_A[s, i]))
                let a_disc = (-dt * log_a[s * d_inner + i].exp()).exp();
                // B_discrete = dt * B[t, s]
                let b_disc = dt * b_t[s];

                // Recurrent update: h[s, i] = A_disc * h[s, i] + B_disc * u[t, i]
                let h_idx = s * d_inner + i;
                state.h[h_idx] = a_disc * state.h[h_idx] + b_disc * u_t[i];

                // Accumulate: y[t, i] += C[t, s] * h[s, i]
                y_t[i] += c_t[s] * state.h[h_idx];
            }

            // Skip connection: y[t, i] += D[i] * u[t, i]
            y_t[i] += d[i] * u_t[i];
        }
    }

    y
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::sequence_state::SsmLayerState;

    const TOL: f32 = 1e-5;

    /// Compute a scalar reference for the selective scan.
    ///
    /// This is a direct port of the mathematical definition, intentionally
    /// verbose for clarity.
    #[allow(clippy::too_many_arguments)]
    fn reference_scan(
        u: &[f32],
        delta: &[f32],
        log_a: &[f32],
        b: &[f32],
        c: &[f32],
        d: &[f32],
        seq_len: usize,
        d_inner: usize,
        d_state: usize,
        h_init: &[f32],
    ) -> Vec<f32> {
        let mut h = h_init.to_vec(); // [d_state × d_inner]
        let mut y = vec![0.0f32; seq_len * d_inner];

        for t in 0..seq_len {
            let u_t = &u[t * d_inner..(t + 1) * d_inner];
            let delta_t = &delta[t * d_inner..(t + 1) * d_inner];
            let b_t = &b[t * d_state..(t + 1) * d_state];
            let c_t = &c[t * d_state..(t + 1) * d_state];

            for i in 0..d_inner {
                let dt = delta_t[i];
                for s in 0..d_state {
                    let a_disc = (-dt * log_a[s * d_inner + i].exp()).exp();
                    let b_disc = dt * b_t[s];
                    let h_idx = s * d_inner + i;
                    h[h_idx] = a_disc * h[h_idx] + b_disc * u_t[i];
                    y[t * d_inner + i] += c_t[s] * h[h_idx];
                }
                y[t * d_inner + i] += d[i] * u_t[i];
            }
        }

        y
    }

    /// ssm_scan_matches_reference:
    /// 32-token sequence, d_inner=8, d_state=4.
    /// Compare against the scalar reference with tolerance 1e-5.
    #[test]
    fn ssm_scan_matches_reference() {
        let seq_len = 32;
        let d_inner = 8;
        let d_state = 4;

        // Deterministic inputs using a tiny LCG.
        struct Lcg(u64);
        impl Lcg {
            fn next_f32(&mut self) -> f32 {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let mantissa = (self.0 >> 33) as u32 & 0x007f_ffff;
                let bits = mantissa | 0x3f80_0000u32;
                (f32::from_bits(bits) - 1.5) * 0.1
            }
            fn fill(&mut self, buf: &mut [f32]) {
                for v in buf.iter_mut() {
                    *v = self.next_f32();
                }
            }
        }

        let mut lcg = Lcg(1234);

        let mut u = vec![0.0f32; seq_len * d_inner];
        let mut delta_raw = vec![0.0f32; seq_len * d_inner];
        // log_A: small negative values (since A < 1 for stability).
        let mut log_a = vec![0.0f32; d_state * d_inner];
        let mut b_mat = vec![0.0f32; seq_len * d_state];
        let mut c_mat = vec![0.0f32; seq_len * d_state];
        let mut d_vec = vec![0.0f32; d_inner];

        lcg.fill(&mut u);
        lcg.fill(&mut delta_raw);
        // log_A should be small to keep A_disc close to 1; use small negative vals.
        for v in log_a.iter_mut() {
            *v = lcg.next_f32().abs() * 0.5;
        }
        lcg.fill(&mut b_mat);
        lcg.fill(&mut c_mat);
        lcg.fill(&mut d_vec);

        // Apply softplus to delta: log(1 + exp(x)).
        let delta: Vec<f32> = delta_raw
            .iter()
            .map(|&x| if x > 20.0 { x } else { (1.0 + x.exp()).ln() })
            .collect();

        let h_init = vec![0.0f32; d_state * d_inner];
        let mut state = SsmLayerState::new(d_state, d_inner);

        let result = selective_scan_sequential(
            &u, &delta, &log_a, &b_mat, &c_mat, &d_vec, seq_len, d_inner, d_state, &mut state,
        );

        let reference = reference_scan(
            &u, &delta, &log_a, &b_mat, &c_mat, &d_vec, seq_len, d_inner, d_state, &h_init,
        );

        assert_eq!(result.len(), reference.len());
        for (idx, (got, exp)) in result.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - exp).abs() < TOL,
                "result[{idx}] = {got} != reference[{idx}] = {exp} (diff={})",
                (got - exp).abs()
            );
        }
    }

    /// State is correctly carried across tokens.
    ///
    /// Run 4 tokens with the same input. Then reset state and run again.
    /// The two outputs should be bit-for-bit identical.
    #[test]
    fn ssm_determinism_after_state_reset() {
        let seq_len = 4;
        let d_inner = 4;
        let d_state = 2;

        let u = vec![0.1f32; seq_len * d_inner];
        let delta = vec![0.5f32; seq_len * d_inner];
        let log_a = vec![0.2f32; d_state * d_inner];
        let b_mat = vec![0.3f32; seq_len * d_state];
        let c_mat = vec![0.4f32; seq_len * d_state];
        let d_vec = vec![0.5f32; d_inner];

        let mut state1 = SsmLayerState::new(d_state, d_inner);
        let out1 = selective_scan_sequential(
            &u,
            &delta,
            &log_a,
            &b_mat,
            &c_mat,
            &d_vec,
            seq_len,
            d_inner,
            d_state,
            &mut state1,
        );

        // Reset to zero initial state.
        let mut state2 = SsmLayerState::new(d_state, d_inner);
        let out2 = selective_scan_sequential(
            &u,
            &delta,
            &log_a,
            &b_mat,
            &c_mat,
            &d_vec,
            seq_len,
            d_inner,
            d_state,
            &mut state2,
        );

        for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "output[{i}] must be bit-identical after reset"
            );
        }
    }

    /// Verify the log(A) interpretation: the test fails if we use `log_a` directly
    /// instead of `exp(log_a)`.
    #[test]
    fn ssm_loga_not_used_directly() {
        let seq_len = 2;
        let d_inner = 1;
        let d_state = 1;

        let u = vec![1.0f32; seq_len * d_inner];
        let delta = vec![1.0f32; seq_len * d_inner];
        // log_A = 1.0 → A_real = exp(1.0) ≈ 2.718
        // A_disc = exp(-delta * A_real) = exp(-2.718) ≈ 0.066
        let log_a = vec![1.0f32; d_state * d_inner];
        let b_mat = vec![1.0f32; seq_len * d_state];
        let c_mat = vec![1.0f32; seq_len * d_state];
        let d_vec = vec![0.0f32; d_inner];

        let mut state = SsmLayerState::new(d_state, d_inner);
        let out = selective_scan_sequential(
            &u, &delta, &log_a, &b_mat, &c_mat, &d_vec, seq_len, d_inner, d_state, &mut state,
        );

        // Reference: a_disc = exp(-1.0 * exp(1.0)), h_0 = 0*a_disc + 1*1*1.0 = 1.0
        // y[0] = c[0]*h[0] + d*u = 1.0 * 1.0 + 0 = 1.0
        let a_disc = (-(1.0f32.exp())).exp();
        // y[1]: h = a_disc * 1.0 + 1.0, y = h + 0
        let h1 = a_disc * 1.0 + 1.0;
        let expected_y0 = 1.0f32;
        let expected_y1 = h1;

        assert!(
            (out[0] - expected_y0).abs() < 1e-5,
            "out[0]={} expected {expected_y0}",
            out[0]
        );
        assert!(
            (out[1] - expected_y1).abs() < 1e-5,
            "out[1]={} expected {expected_y1}",
            out[1]
        );
    }
}

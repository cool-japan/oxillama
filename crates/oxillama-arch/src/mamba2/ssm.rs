//! Selective-scan (SSM) primitives.
//!
//! This module holds **two** scans, because Mamba-1 and Mamba-2 parameterise
//! `A` and `Δ` completely differently:
//!
//! | Function | `A` | `Δ` | Used by |
//! |---|---|---|---|
//! | [`selective_scan_sequential`] | `[d_state × d_inner]` | per channel | Mamba-1 / Jamba |
//! | [`selective_scan_mamba2`] | one scalar per **head** | one scalar per **head** | Mamba-2 |
//!
//! ## Mamba-1 convention (`selective_scan_sequential`)
//!
//! The caller supplies `log_a` and the scan forms
//!
//! ```text
//! A_discrete[t, s, i] = exp(-Δ[t, i] * exp(log_A[s, i]))
//!                      ≠ exp(-Δ[t, i] * log_A[s, i])   ← WRONG
//! B_discrete[t, s, i] = Δ[t, i] * B[t, s]
//! h[s, i]             = A_discrete * h[s, i] + B_discrete * u[t, i]
//! y[t, i]            += C[t, s] * h[s, i]
//! y[t, i]            += D[i] * u[t, i]  (skip connection)
//! ```
//!
//! ## Mamba-2 convention (`selective_scan_mamba2`)
//!
//! **`blk.N.ssm_a` in a GGUF file is `A` itself, already negative — not
//! `log(A)`.**  `convert_hf_to_gguf.py::Mamba2Model.modify_tensors` writes
//! `data_torch = -torch.exp(data_torch)` for every `.A_log` tensor, and
//! `ggml_compute_forward_ssm_scan_f32` then consumes it verbatim:
//!
//! ```text
//! const float dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
//! const float dA           = expf(dt_soft_plus * A[h]);
//! ```
//!
//! So [`selective_scan_mamba2`] applies **no** `exp` and **no** negation to
//! `a`.  Passing an `A_log` tensor to it would be a silent correctness bug.

use crate::common::sequence_state::SsmLayerState;
use crate::error::{ArchError, ArchResult};

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

// ─── Mamba-2 selective scan ───────────────────────────────────────────────────

/// Geometry of one Mamba-2 selective scan.
///
/// Bundled into a struct because the scan otherwise needs eleven arguments and
/// the indexing is easy to get wrong: `x` is head-major
/// (`x[t][h][i1]`), while `B`/`C` are group-major (`B[t][g][i0]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba2ScanDims {
    /// Number of tokens in this chunk.
    pub seq_len: usize,
    /// Number of SSM heads (`ssm.time_step_rank`).
    pub n_head: usize,
    /// Channels per head (`d_inner / n_head`).
    pub head_dim: usize,
    /// SSM state dimension.
    pub d_state: usize,
    /// Number of B/C groups (`ssm.group_count`).
    pub n_group: usize,
}

impl Mamba2ScanDims {
    /// Inner width, `n_head * head_dim`.
    pub fn d_inner(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Width of one token's `B` (or `C`) slice, `n_group * d_state`.
    pub fn bc_width(&self) -> usize {
        self.n_group * self.d_state
    }
}

/// Softplus, matching `ggml_compute_softplus_f32` in `ggml/src/ggml-impl.h`:
/// `(input > 20.0f) ? input : logf(1 + expf(input))`.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

/// Sequential Mamba-2 selective scan.
///
/// A direct transcription of the `src3->ne[0] == 1` branch of
/// `ggml_compute_forward_ssm_scan_f32` (`ggml/src/ggml-cpu/ops.cpp`), which is
/// the branch llama.cpp takes for Mamba-2 because `ssm_a` has shape
/// `{1, n_head}`:
///
/// ```text
/// dt_soft_plus = softplus(dt[h]);
/// dA           = exp(dt_soft_plus * A[h]);
/// g            = h / (nh / ng);            // repeat_interleave
/// x_dt         = x[i1 + h*nr] * dt_soft_plus;
/// state        = s0[i0 + ii*nc] * dA + B[i0 + g*nc] * x_dt;
/// y[ii]       += state * C[i0 + g*nc];
/// ```
///
/// The `D` skip (`y = ggml_add(y, ggml_mul(x, ssm_d))` in
/// `build_mamba2_layer`) is folded in here; like `ssm_a`, `ssm_d` is
/// **per head**, so channel `i1` of head `h` adds `d_skip[h] * x[..]`.
///
/// # Layouts
///
/// * `x`       – `[seq_len × d_inner]`, head-major: `x[t*d_inner + h*head_dim + i1]`.
/// * `dt`      – `[seq_len × n_head]`, **raw** (bias and softplus applied here).
/// * `dt_bias` – `[n_head]` (`blk.N.ssm_dt.bias`).
/// * `a`       – `[n_head]` (`blk.N.ssm_a`), used verbatim; see the module docs.
/// * `b`, `c`  – `[seq_len × n_group*d_state]`, group-major.
/// * `d_skip`  – `[n_head]` (`blk.N.ssm_d`).
/// * `state.h` – `[d_state × head_dim × n_head]` indexed
///   `h[(h*head_dim + i1) * d_state + i0]`, matching ggml's
///   `reshape_4d(states, d_state, head_dim, n_head, ...)`.
///
/// # Returns
/// `y` of `[seq_len × d_inner]`, head-major.
///
/// # Errors
///
/// [`ArchError::InvalidShape`] when any slice disagrees with `dims`, or
/// [`ArchError::InvalidConfig`] when `n_head` is not a multiple of `n_group`
/// (an invariant `ggml_ssm_scan` asserts).
#[allow(clippy::too_many_arguments)]
pub fn selective_scan_mamba2(
    x: &[f32],
    dt: &[f32],
    dt_bias: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d_skip: &[f32],
    dims: Mamba2ScanDims,
    state: &mut SsmLayerState,
) -> ArchResult<Vec<f32>> {
    let Mamba2ScanDims {
        seq_len,
        n_head,
        head_dim,
        d_state,
        n_group,
    } = dims;

    if n_head == 0 || n_group == 0 || d_state == 0 || head_dim == 0 {
        return Err(ArchError::InvalidConfig {
            detail: format!(
                "mamba2.scan: n_head={n_head}, head_dim={head_dim}, d_state={d_state}, \
                 n_group={n_group} must all be non-zero"
            ),
        });
    }
    if n_head % n_group != 0 {
        return Err(ArchError::InvalidConfig {
            detail: format!(
                "mamba2.scan: n_head ({n_head}) must be divisible by n_group ({n_group})"
            ),
        });
    }

    let d_inner = dims.d_inner();
    let bc_width = dims.bc_width();

    let shape_err = |what: &str, expected: Vec<usize>, got: usize| ArchError::InvalidShape {
        name: format!("mamba2.scan.{what}"),
        expected,
        got: vec![got],
    };

    if x.len() != seq_len * d_inner {
        return Err(shape_err("x", vec![seq_len, d_inner], x.len()));
    }
    if dt.len() != seq_len * n_head {
        return Err(shape_err("dt", vec![seq_len, n_head], dt.len()));
    }
    if dt_bias.len() != n_head {
        return Err(shape_err("dt_bias", vec![n_head], dt_bias.len()));
    }
    if a.len() != n_head {
        return Err(shape_err("a", vec![n_head], a.len()));
    }
    if d_skip.len() != n_head {
        return Err(shape_err("d", vec![n_head], d_skip.len()));
    }
    if b.len() != seq_len * bc_width {
        return Err(shape_err("b", vec![seq_len, bc_width], b.len()));
    }
    if c.len() != seq_len * bc_width {
        return Err(shape_err("c", vec![seq_len, bc_width], c.len()));
    }
    if state.h.len() != d_state * d_inner {
        return Err(shape_err("state", vec![d_state * d_inner], state.h.len()));
    }

    let heads_per_group = n_head / n_group;
    let mut y = vec![0.0f32; seq_len * d_inner];

    for t in 0..seq_len {
        let b_t = &b[t * bc_width..(t + 1) * bc_width];
        let c_t = &c[t * bc_width..(t + 1) * bc_width];
        let x_t = &x[t * d_inner..(t + 1) * d_inner];
        let y_t = &mut y[t * d_inner..(t + 1) * d_inner];

        for h in 0..n_head {
            let dt_soft_plus = softplus(dt[t * n_head + h] + dt_bias[h]);
            // A is consumed verbatim: it is already `-exp(A_log)` in GGUF.
            let d_a = (dt_soft_plus * a[h]).exp();
            let g = h / heads_per_group;
            let g_off = g * d_state;

            for i1 in 0..head_dim {
                let ii = i1 + h * head_dim;
                let x_ii = x_t[ii];
                let x_dt = x_ii * dt_soft_plus;
                let h_off = ii * d_state;

                let mut sumf = 0.0f32;
                for i0 in 0..d_state {
                    let prev = state.h[h_off + i0];
                    let s = prev * d_a + b_t[g_off + i0] * x_dt;
                    sumf += s * c_t[g_off + i0];
                    state.h[h_off + i0] = s;
                }

                // Per-head D skip connection.
                y_t[ii] = sumf + d_skip[h] * x_ii;
            }
        }
    }

    Ok(y)
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

    // ─── Mamba-2 scan ─────────────────────────────────────────────────────

    fn scan_dims(seq_len: usize) -> Mamba2ScanDims {
        Mamba2ScanDims {
            seq_len,
            n_head: 4,
            head_dim: 2,
            d_state: 3,
            n_group: 2,
        }
    }

    /// `selective_scan_mamba2` reproduces the scalar loop in
    /// `ggml_compute_forward_ssm_scan_f32` (the `src3->ne[0] == 1` branch).
    #[test]
    fn mamba2_scan_matches_ggml_reference() {
        let dims = scan_dims(5);
        let d_inner = dims.d_inner();
        let bc = dims.bc_width();

        let f = |i: usize, k: usize| ((i * 7 + k) % 11) as f32 * 0.11 - 0.5;
        let x: Vec<f32> = (0..dims.seq_len * d_inner).map(|i| f(i, 1)).collect();
        let dt: Vec<f32> = (0..dims.seq_len * dims.n_head).map(|i| f(i, 2)).collect();
        let dt_bias: Vec<f32> = (0..dims.n_head).map(|i| f(i, 3)).collect();
        let a: Vec<f32> = (0..dims.n_head).map(|i| -(0.3 + i as f32 * 0.2)).collect();
        let b: Vec<f32> = (0..dims.seq_len * bc).map(|i| f(i, 4)).collect();
        let c: Vec<f32> = (0..dims.seq_len * bc).map(|i| f(i, 5)).collect();
        let d_skip: Vec<f32> = (0..dims.n_head).map(|i| f(i, 6)).collect();

        // Scalar transcription of ops.cpp, written independently below.
        let mut h_ref = vec![0.0f32; dims.d_state * d_inner];
        let mut y_ref = vec![0.0f32; dims.seq_len * d_inner];
        let heads_per_group = dims.n_head / dims.n_group;
        for t in 0..dims.seq_len {
            for head in 0..dims.n_head {
                let raw = dt[t * dims.n_head + head] + dt_bias[head];
                let sp = if raw > 20.0 {
                    raw
                } else {
                    (1.0f32 + raw.exp()).ln()
                };
                let d_a = (sp * a[head]).exp();
                let g = head / heads_per_group;
                for i1 in 0..dims.head_dim {
                    let ii = i1 + head * dims.head_dim;
                    let x_dt = x[t * d_inner + ii] * sp;
                    let mut acc = 0.0f32;
                    for i0 in 0..dims.d_state {
                        let hi = ii * dims.d_state + i0;
                        let bi = t * bc + g * dims.d_state + i0;
                        h_ref[hi] = h_ref[hi] * d_a + b[bi] * x_dt;
                        acc += h_ref[hi] * c[bi];
                    }
                    y_ref[t * d_inner + ii] = acc + d_skip[head] * x[t * d_inner + ii];
                }
            }
        }

        let mut state = SsmLayerState::new(dims.d_state, d_inner);
        let y = selective_scan_mamba2(&x, &dt, &dt_bias, &a, &b, &c, &d_skip, dims, &mut state)
            .expect("scan");

        assert_eq!(y.len(), y_ref.len());
        for (i, (got, want)) in y.iter().zip(y_ref.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "y[{i}] = {got} != reference {want}"
            );
        }
        for (i, (got, want)) in state.h.iter().zip(h_ref.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "state.h[{i}] = {got} != reference {want}"
            );
        }
    }

    /// `A` is used verbatim: no `exp`, no negation.
    ///
    /// `blk.N.ssm_a` already holds `-exp(A_log)` (see the module docs), so the
    /// decay for one step must be exactly `exp(softplus(dt) * a)`.
    #[test]
    fn mamba2_scan_uses_a_verbatim() {
        let dims = Mamba2ScanDims {
            seq_len: 2,
            n_head: 1,
            head_dim: 1,
            d_state: 1,
            n_group: 1,
        };
        let a = vec![-2.0f32];
        let dt = vec![0.0f32, 0.0];
        let dt_bias = vec![0.0f32];
        let x = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 1.0];
        let c = vec![1.0f32, 1.0];
        let d_skip = vec![0.0f32];

        let mut state = SsmLayerState::new(1, 1);
        let y = selective_scan_mamba2(&x, &dt, &dt_bias, &a, &b, &c, &d_skip, dims, &mut state)
            .expect("scan");

        // softplus(0) = ln 2.
        let sp = 2.0f32.ln();
        // t=0: h = 0*dA + 1*(1*sp) = sp ; y = sp.
        assert!((y[0] - sp).abs() < 1e-6, "y[0] = {} != {sp}", y[0]);
        // t=1: x = 0, so h = sp * exp(sp * a) and y = h.
        let expected = sp * (sp * a[0]).exp();
        assert!(
            (y[1] - expected).abs() < 1e-6,
            "y[1] = {} != {expected}; A must be used verbatim (dA = exp(softplus(dt)*A))",
            y[1]
        );
        // Had the scan re-applied exp/negation it would use exp(-sp*exp(-2.0)).
        let wrong = sp * (-sp * a[0].exp()).exp();
        assert!(
            (y[1] - wrong).abs() > 1e-6,
            "y[1] matches the log-parameterised formula; A was not used verbatim"
        );
    }

    /// `B`/`C` are shared across the heads of a group (`repeat_interleave`).
    #[test]
    fn mamba2_scan_maps_heads_to_groups() {
        // 4 heads, 2 groups -> heads 0,1 use group 0 and heads 2,3 use group 1.
        let dims = Mamba2ScanDims {
            seq_len: 1,
            n_head: 4,
            head_dim: 1,
            d_state: 1,
            n_group: 2,
        };
        let x = vec![1.0f32; 4];
        let dt = vec![0.0f32; 4];
        let dt_bias = vec![0.0f32; 4];
        let a = vec![-1.0f32; 4];
        // Group 0 -> B = 1, group 1 -> B = 10.
        let b = vec![1.0f32, 10.0];
        let c = vec![1.0f32, 1.0];
        let d_skip = vec![0.0f32; 4];

        let mut state = SsmLayerState::new(1, 4);
        let y = selective_scan_mamba2(&x, &dt, &dt_bias, &a, &b, &c, &d_skip, dims, &mut state)
            .expect("scan");

        assert!((y[0] - y[1]).abs() < 1e-6, "heads 0 and 1 share group 0");
        assert!((y[2] - y[3]).abs() < 1e-6, "heads 2 and 3 share group 1");
        assert!(
            (y[2] - 10.0 * y[0]).abs() < 1e-5,
            "group 1 has B = 10x group 0: {} vs {}",
            y[2],
            y[0]
        );
    }

    /// `D` is per head, not per channel.
    #[test]
    fn mamba2_scan_d_skip_is_per_head() {
        let dims = Mamba2ScanDims {
            seq_len: 1,
            n_head: 2,
            head_dim: 2,
            d_state: 1,
            n_group: 1,
        };
        let x = vec![1.0f32; 4];
        let dt = vec![-30.0f32; 2]; // softplus(-30) ~ 0 -> the scan term vanishes
        let dt_bias = vec![0.0f32; 2];
        let a = vec![-1.0f32; 2];
        let b = vec![0.0f32];
        let c = vec![0.0f32];
        let d_skip = vec![0.25f32, 4.0];

        let mut state = SsmLayerState::new(1, 4);
        let y = selective_scan_mamba2(&x, &dt, &dt_bias, &a, &b, &c, &d_skip, dims, &mut state)
            .expect("scan");

        assert!((y[0] - 0.25).abs() < 1e-5, "head 0 channel 0: {}", y[0]);
        assert!((y[1] - 0.25).abs() < 1e-5, "head 0 channel 1: {}", y[1]);
        assert!((y[2] - 4.0).abs() < 1e-5, "head 1 channel 0: {}", y[2]);
        assert!((y[3] - 4.0).abs() < 1e-5, "head 1 channel 1: {}", y[3]);
    }

    /// Streaming one token at a time equals scanning the whole chunk.
    #[test]
    fn mamba2_scan_state_carries_across_calls() {
        let dims = scan_dims(4);
        let d_inner = dims.d_inner();
        let bc = dims.bc_width();

        let f = |i: usize, k: usize| ((i * 5 + k) % 9) as f32 * 0.2 - 0.7;
        let x: Vec<f32> = (0..dims.seq_len * d_inner).map(|i| f(i, 1)).collect();
        let dt: Vec<f32> = (0..dims.seq_len * dims.n_head).map(|i| f(i, 2)).collect();
        let dt_bias: Vec<f32> = (0..dims.n_head).map(|i| f(i, 3)).collect();
        let a: Vec<f32> = (0..dims.n_head).map(|i| -(0.5 + i as f32 * 0.1)).collect();
        let b: Vec<f32> = (0..dims.seq_len * bc).map(|i| f(i, 4)).collect();
        let c: Vec<f32> = (0..dims.seq_len * bc).map(|i| f(i, 5)).collect();
        let d_skip: Vec<f32> = (0..dims.n_head).map(|i| f(i, 6)).collect();

        let mut batch_state = SsmLayerState::new(dims.d_state, d_inner);
        let batch = selective_scan_mamba2(
            &x,
            &dt,
            &dt_bias,
            &a,
            &b,
            &c,
            &d_skip,
            dims,
            &mut batch_state,
        )
        .expect("batch scan");

        let mut step_state = SsmLayerState::new(dims.d_state, d_inner);
        let mut streamed = Vec::new();
        let one = Mamba2ScanDims { seq_len: 1, ..dims };
        for t in 0..dims.seq_len {
            let out = selective_scan_mamba2(
                &x[t * d_inner..(t + 1) * d_inner],
                &dt[t * dims.n_head..(t + 1) * dims.n_head],
                &dt_bias,
                &a,
                &b[t * bc..(t + 1) * bc],
                &c[t * bc..(t + 1) * bc],
                &d_skip,
                one,
                &mut step_state,
            )
            .expect("step scan");
            streamed.extend_from_slice(&out);
        }

        for (i, (a_v, b_v)) in batch.iter().zip(streamed.iter()).enumerate() {
            assert!(
                (a_v - b_v).abs() < 1e-5,
                "y[{i}]: batch {a_v} != streamed {b_v}"
            );
        }
    }

    /// Shape mismatches are typed errors rather than panics.
    #[test]
    fn mamba2_scan_rejects_bad_shapes() {
        let dims = scan_dims(1);
        let d_inner = dims.d_inner();
        let bc = dims.bc_width();
        let ok_x = vec![0.0f32; d_inner];
        let ok_dt = vec![0.0f32; dims.n_head];
        let ok_head = vec![0.0f32; dims.n_head];
        let ok_bc = vec![0.0f32; bc];
        let mut state = SsmLayerState::new(dims.d_state, d_inner);

        assert!(
            selective_scan_mamba2(
                &ok_x[..d_inner - 1],
                &ok_dt,
                &ok_head,
                &ok_head,
                &ok_bc,
                &ok_bc,
                &ok_head,
                dims,
                &mut state
            )
            .is_err(),
            "short x must error"
        );
        assert!(
            selective_scan_mamba2(
                &ok_x,
                &ok_dt,
                &ok_head,
                &ok_head[..dims.n_head - 1],
                &ok_bc,
                &ok_bc,
                &ok_head,
                dims,
                &mut state
            )
            .is_err(),
            "short a must error"
        );
        assert!(
            selective_scan_mamba2(
                &ok_x,
                &ok_dt,
                &ok_head,
                &ok_head,
                &ok_bc[..bc - 1],
                &ok_bc,
                &ok_head,
                dims,
                &mut state
            )
            .is_err(),
            "short B must error"
        );

        let bad = Mamba2ScanDims { n_group: 3, ..dims };
        assert!(
            selective_scan_mamba2(
                &ok_x,
                &ok_dt,
                &ok_head,
                &ok_head,
                &vec![0.0f32; 3 * dims.d_state],
                &vec![0.0f32; 3 * dims.d_state],
                &ok_head,
                bad,
                &mut state
            )
            .is_err(),
            "n_head not divisible by n_group must error"
        );
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

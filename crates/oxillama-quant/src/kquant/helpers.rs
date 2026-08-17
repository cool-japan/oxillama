//! Shared scalar helpers for the K-quant encoders.
//!
//! Every function here is a literal port of the correspondingly-named
//! `static` helper in llama.cpp's `ggml/src/ggml-quants.c` (commit
//! `ba7e817ee`).  "Literal" is load-bearing: these routines are what decide
//! the exact bytes a K-quant super-block ends up with, and several of them
//! depend on C semantics that a "cleaner" rewrite silently changes —
//! round-half-to-even in [`nearest_int`], the truncate-then-clamp order in
//! the callers, the `l * l` association in the accumulators.  The comments
//! call out each of those.

/// `GROUP_MAX_EPS` from `ggml-quants.c`: the threshold below which a group
/// of weights is treated as all-zero.
pub(crate) const GROUP_MAX_EPS: f32 = 1e-15;

/// Round `fval` to the nearest integer, ties to even.
///
/// Port of ggml's `nearest_int`:
///
/// ```c
/// static inline int nearest_int(float fval) {
///     assert(fabsf(fval) <= 4194303.f);
///     float val = fval + 12582912.f;
///     int i; memcpy(&i, &val, sizeof(int));
///     return (i & 0x007fffff) - 0x00400000;
/// }
/// ```
///
/// Adding `12582912.0 == 1.5 * 2^23` forces the IEEE-754 round-to-nearest-even
/// mode to round the value into the integer grid and leaves the result in the
/// mantissa. This is **not** the same as `f32::round`, which rounds halves
/// *away from zero*: `nearest_int(2.5) == 2` but `2.5f32.round() == 3.0`.
/// Substituting `round()` changes the quantized bytes on every input that
/// lands exactly on a half — precisely the plateau/ramp inputs a real weight
/// matrix is full of.
///
/// The C `assert` on the input range is deliberately not reproduced: release
/// builds of ggml compile it out, so inputs beyond `±4194303` are garbage in
/// both implementations rather than a panic in one of them.
#[inline]
pub(crate) fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0f32;
    ((val.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Unpack the 6-bit scale/min pair for sub-block `j` from a Q4_K/Q5_K
/// `scales[12]` array.
///
/// Port of ggml's `get_scale_min_k4`. The encoder needs it because its second
/// pass re-reads the scales it just packed, so any asymmetry between the pack
/// and the unpack would change the quantized values.
#[inline]
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Search for the scale that minimises the weighted squared error of a
/// symmetric (zero-centred) quantization of `x` onto `[-nmax, nmax-1]`.
///
/// Port of ggml's `make_qx_quants`. Only the `rmse_type == 1`, `qw == None`
/// configuration is reachable from `quantize_row_q6_K_ref`, but the whole
/// function is ported so the weight-selection ladder stays visible and any
/// future caller gets the same behaviour.
///
/// On return, `l[i]` holds `q_i + nmax` (a non-negative code) and the return
/// value is the scale `d` such that `x_i ≈ d * (l[i] - nmax)`.
pub(crate) fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [i8],
    rmse_type: i32,
    qw: Option<&[f32]>,
) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x.iter().take(n) {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        // all zero
        for slot in l.iter_mut().take(n) {
            *slot = 0;
        }
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let q = nearest_int(iscale * x[i]);
            l[i] = (nmax + q.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let mut rmse_type = rmse_type;
    let mut return_early = false;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }

    // `w` selection ladder, verbatim from the C ternary chain.
    let weight = |i: usize| -> f32 {
        match qw {
            Some(w) => w[i],
            None => match rmse_type {
                1 => x[i] * x[i],
                2 => 1.0,
                3 => x[i].abs(),
                _ => x[i].abs().sqrt(),
            },
        }
    };

    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (q + nmax) as i8;
        let w = weight(i);
        sumlx += w * x[i] * q as f32;
        suml2 += w * q as f32 * q as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9i32..=9 {
        if is == 0 {
            continue;
        }
        iscale = -((nmax as f32) + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for (i, &xi) in x.iter().take(n).enumerate() {
            let q = nearest_int(iscale * xi).clamp(-nmax, nmax - 1);
            let w = weight(i);
            sumlx += w * xi * q as f32;
            suml2 += w * q as f32 * q as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let q = nearest_int(iscale * x[i]);
                l[i] = (nmax + q.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Coordinate-descent refinement of a symmetric 3-bit quantization.
///
/// Port of ggml's `make_q3_quants`; only the `do_rmse == true` path is
/// reachable from `quantize_row_q3_K_ref`.
pub(crate) fn make_q3_quants(n: usize, nmax: i32, x: &[f32], l: &mut [i8], do_rmse: bool) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x.iter().take(n) {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        for slot in l.iter_mut().take(n) {
            *slot = 0;
        }
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for i in 0..n {
            let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            l[i] = q as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * q as f32;
            suml2 += w * q as f32 * q as f32;
        }
        for _itry in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let mut slx = sumlx - w * x[i] * l[i] as f32;
                if slx > 0.0 {
                    let mut sl2 = suml2 - w * l[i] as f32 * l[i] as f32;
                    let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                    if new_l as i8 != l[i] {
                        slx += w * x[i] * new_l as f32;
                        sl2 += w * new_l as f32 * new_l as f32;
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for slot in l.iter_mut().take(n) {
            *slot += nmax as i8;
        }
        return if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    }
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (q + nmax) as i8;
    }
    1.0 / iscale
}

/// Search for the (scale, min) pair that minimises the weighted error of an
/// affine quantization `x ≈ scale * q + min` with `q ∈ [0, nmax]`.
///
/// Port of ggml's `make_qkx2_quants`. This is the workhorse behind Q4_K, Q5_K
/// and Q2_K: it starts from the naive min/max scale, then sweeps `nstep + 1`
/// candidate scales and, for each, solves the 2×2 weighted least-squares
/// normal equations for the best (scale, min) given the codes that scale
/// produces.
///
/// `the_min` receives `-min` — the value the block format stores, which is
/// *subtracted* on the decode side.
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for slot in l.iter_mut().take(n) {
            *slot = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min));
        l[i] = q.clamp(0, nmax) as u8;
        let diff = scale * l[i] as f32 + min - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        let w = weights[i];
        best_error += w * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let w = weights[i];
            let qf = q as f32;
            sum_l += w * qf;
            sum_l2 += w * qf * qf;
            sum_xl += w * qf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                let w = weights[i];
                cur_error += w * diff;
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `nearest_int` rounds halves to even, unlike `f32::round`.
    ///
    /// Golden values produced by running llama.cpp's `nearest_int` verbatim
    /// (see `tests/golden_kquant_encoders.rs` for the harness description).
    #[test]
    fn nearest_int_rounds_half_to_even() {
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(3.5), 4);
        assert_eq!(nearest_int(-0.5), 0);
        assert_eq!(nearest_int(-1.5), -2);
        assert_eq!(nearest_int(-2.5), -2);
        // f32::round() disagrees on every one of the ties above.
        assert_ne!(nearest_int(2.5), 2.5f32.round() as i32);
    }

    #[test]
    fn nearest_int_matches_plain_rounding_away_from_ties() {
        assert_eq!(nearest_int(0.49), 0);
        assert_eq!(nearest_int(0.51), 1);
        assert_eq!(nearest_int(-0.51), -1);
        assert_eq!(nearest_int(123.4), 123);
        assert_eq!(nearest_int(-123.6), -124);
        assert_eq!(nearest_int(0.0), 0);
    }

    #[test]
    fn get_scale_min_k4_round_trips_low_half() {
        let mut q = [0u8; 12];
        q[0] = 37;
        q[4] = 21;
        let (d, m) = get_scale_min_k4(0, &q);
        assert_eq!((d, m), (37, 21));
    }

    #[test]
    fn get_scale_min_k4_round_trips_high_half() {
        // Pack ls=45, lm=52 into slot j=5 exactly as the Q4_K encoder does.
        let (ls, lm) = (45u8, 52u8);
        let j = 5usize;
        let mut q = [0u8; 12];
        q[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        q[j - 4] |= (ls >> 4) << 6;
        q[j] |= (lm >> 4) << 6;
        assert_eq!(get_scale_min_k4(j, &q), (ls, lm));
    }
}

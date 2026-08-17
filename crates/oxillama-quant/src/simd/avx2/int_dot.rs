//! Integer dot-product primitives shared by the fused Q8_0-activation AVX2
//! kernels (Q4_K, Q6_K, Q2_K, Q3_K, ...).
//!
//! The fused decode path quantizes the activation vector to Q8_0 once per
//! matmul input, which turns every weight row into a pure **integer** dot
//! product: `Σ q_w · q_a`, with `q_w` unsigned (K-quant nibbles/quants are
//! stored unsigned in the GGUF byte layout) and `q_a` signed `i8` (Q8_0
//! activations). This module is the one place that knows how to evaluate
//! 32-wide slices of such a dot product on x86_64 with AVX2.
//!
//! # Why VPMADDUBSW + VPMADDWD
//!
//! AVX2 has no single "unsigned x signed, 32 lanes of i8 -> i32" instruction.
//! The two-step idiom used here — `VPMADDUBSW` ([`core::arch::x86_64::_mm256_maddubs_epi16`],
//! unsigned x signed byte pairs -> saturating `i16`) followed by `VPMADDWD`
//! ([`core::arch::x86_64::_mm256_madd_epi16`], `i16` pairs -> `i32`) — is the
//! standard way to get there. Unlike NEON's SDOT
//! (see `crate::simd::neon::int_dot`), it needs no optional-extension
//! runtime dispatch: both instructions are part of the baseline AVX2
//! instruction set, so once `is_x86_feature_detected!("avx2")` is true
//! (checked once by [`crate::dispatch::KernelDispatcher`] /
//! [`crate::simd::cached_capabilities`] before any AVX2 kernel is
//! constructed), both intrinsics are always safe to call — there is no
//! widening-fallback path to select between the way NEON needs one for CPUs
//! without `dotprod`.
//!
//! # Saturation
//!
//! `VPMADDUBSW`'s per-pair intermediate sum is a **saturating** `i16`, so
//! this module is not exact for arbitrary `u8`/`i8` inputs — see
//! [`dot_u8_i8`]'s doc for the exact bound every caller in this crate relies
//! on to stay clear of that saturation.
//!
//! # Layout
//!
//! [`load_q8_act`] reads one 34-byte Q8_0 activation block (2-byte
//! little-endian FP16 scale + 32 `i8` values) directly into a `__m256i`,
//! masking lanes at or past a ragged-K tail's `valid` count to exact zero so
//! a partial final block contributes nothing extra — matching the scalar
//! fallback's explicit bounds check
//! (`let w_end = (w_start + bs).min(n_cols);` in `traits.rs`) byte for byte.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

/// Largest batch a single pass of a `*_row_batch_avx2` prefill kernel
/// handles.
///
/// Must stay equal to `crate::simd::neon::int_dot::MAX_FUSED_BATCH` (defined
/// separately there because that module is aarch64-gated and unreachable
/// from x86_64 — there is no single shared location both platforms could
/// pull this constant from without one platform depending on the other's
/// cfg-gated module).
pub(crate) const MAX_FUSED_BATCH: usize = 32;

/// Horizontal sum of the eight `i32` lanes of `v`.
///
/// # Safety
/// Caller must have the `avx2` CPU feature.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn hsum_i32(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(hi, lo);
    let shuf = _mm_shuffle_epi32(sum128, 0b10_11_00_01);
    let sums = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sums, 0b00_00_10_10);
    let sums2 = _mm_add_epi32(sums, shuf2);
    _mm_cvtsi128_si32(sums2)
}

/// `Σ_{i<32} w[i] * a[i]`, with `w`'s 32 bytes read as unsigned `u8`
/// (`0..=255`) and `a`'s 32 bytes read as signed `i8`.
///
/// Exact for every caller in this crate: Q4_K nibbles are `0..=15` and Q6_K
/// quants are `0..=63`, both far under the `255` this function's `u8`
/// interpretation actually allows; Q8_0 activations satisfy `|a| <= 128` (an
/// `i8`'s range is `-128..=127`, so `128` is the tightest uniform bound).
/// The computation runs `VPMADDUBSW` (unsigned x signed -> saturating `i16`
/// pairs) then `VPMADDWD` (`i16` pairs -> `i32`, exact — `VPMADDWD` does not
/// saturate) then a horizontal reduce. No saturation occurs for any real
/// caller because the per-*pair* product magnitude — the quantity
/// `VPMADDUBSW` actually clamps — stays under the `i16` bound of `32767`:
/// `w <= 63, |a| <= 128 => |pair sum| <= 2 * 63 * 128 = 16128 < 32767`,
/// using Q6_K's `w <= 63` as the tighter of the two real callers' bounds
/// (Q4_K's `w <= 15` leaves even more headroom).
///
/// # Safety
/// Caller must have the `avx2` CPU feature.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn dot_u8_i8(w: __m256i, a: __m256i) -> i32 {
    let ones16 = _mm256_set1_epi16(1);
    hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(w, a), ones16))
}

/// `Σ_{i<32} a[i]`, `a`'s 32 bytes read as signed `i8`.
///
/// Used for the K-quant "minimum" correction terms, which need `Σ q_a` over
/// the same 32-lane window as [`dot_u8_i8`]'s dot product. Implemented the
/// same way as [`dot_u8_i8`], against an all-ones unsigned operand instead
/// of a real weight block, which is exact across the *entire* `i8` range:
/// the saturating `i16` intermediate holds at most `2 * 127 = 254` in
/// magnitude, nowhere near the `32767` saturation bound, so — unlike
/// [`dot_u8_i8`] — this one needs no caller-specific bound to stay exact.
///
/// # Safety
/// Caller must have the `avx2` CPU feature.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn sum_i8(a: __m256i) -> i32 {
    let ones16 = _mm256_set1_epi16(1);
    hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_set1_epi8(1), a),
        ones16,
    ))
}

/// Load a Q8_0 activation block's 32 `i8` values as a `__m256i` (lanes at or
/// past `valid` zeroed) plus its `f32` scale.
///
/// Masking is what makes a ragged-K tail exact: a column past `n_cols` must
/// contribute an exact zero product, matching the f32 GEMV's bounds check.
///
/// # Safety
/// `acts_q8.len() >= (q8_blk + 1) * 34`, caller has avx2.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn load_q8_act(acts_q8: &[u8], q8_blk: usize, valid: usize) -> (f32, __m256i) {
    let base = q8_blk * 34;
    let ab = &acts_q8[base..base + 34];
    let d_a = half::f16::from_bits(u16::from_le_bytes([ab[0], ab[1]])).to_f32();
    if valid >= 32 {
        (
            d_a,
            _mm256_loadu_si256(ab.as_ptr().add(2) as *const __m256i),
        )
    } else {
        let mut buf = [0i8; 32];
        for (i, slot) in buf.iter_mut().enumerate().take(valid) {
            *slot = ab[2 + i] as i8;
        }
        (d_a, _mm256_loadu_si256(buf.as_ptr() as *const __m256i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the value this module's doc promises to keep in sync with
    /// `crate::simd::neon::int_dot::MAX_FUSED_BATCH`. That constant lives in
    /// an aarch64-gated module and does not exist in this (x86_64)
    /// compilation, so it cannot be cross-checked directly here — this test
    /// is the next best thing: an accidental change to this side shows up
    /// as a diff a reviewer must explain, same as changing the NEON side
    /// would. (Also gives `MAX_FUSED_BATCH` a real use: nothing in
    /// production calls it yet — see the module's `dead_code` note above —
    /// so without this test it would be flagged unused even in the test
    /// build.)
    #[test]
    fn max_fused_batch_is_32() {
        assert_eq!(MAX_FUSED_BATCH, 32);
    }

    /// Plain, non-saturating scalar oracle for [`dot_u8_i8`]. Exact for
    /// every case fed to it below, because each case is built (see
    /// [`build_dot_case`]) to keep every `VPMADDUBSW` pair sum within the
    /// `i16` saturation bound — so an un-saturated scalar sum is the correct
    /// oracle, not an approximation.
    fn scalar_dot_u8_i8(w: &[u8; 32], a: &[i8; 32]) -> i32 {
        w.iter()
            .zip(a.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum()
    }

    fn scalar_sum_i8(a: &[i8; 32]) -> i32 {
        a.iter().map(|&y| y as i32).sum()
    }

    /// SAFETY: avx2 confirmed by every caller before this runs.
    unsafe fn load_u8x32(v: &[u8; 32]) -> __m256i {
        _mm256_loadu_si256(v.as_ptr() as *const __m256i)
    }

    /// SAFETY: avx2 confirmed by every caller before this runs.
    unsafe fn load_i8x32(v: &[i8; 32]) -> __m256i {
        _mm256_loadu_si256(v.as_ptr() as *const __m256i)
    }

    /// Build a uniform-fill `(w, a)` pair for [`dot_u8_i8`]'s extremes test,
    /// keeping every `VPMADDUBSW` pair sum inside `i16` range.
    ///
    /// `w == 255` is the one value in the requested set
    /// (`{0, 15, 63, 255}`) that can saturate: a fully-uniform
    /// `w=255, a=-128` pair sums to `2 * 255 * -128 = -65280`, outside
    /// `i16`'s `-32768..=32767`. No real caller ever hits this (Q4_K/Q6_K
    /// weights cap at 63 — see [`dot_u8_i8`]'s doc) but the test still wants
    /// to exercise `w`'s true `u8` max (confirming 255 is read as unsigned
    /// 255, not a sign-extended -1). So for `w == 255` only, every odd lane
    /// of `a` is zeroed: each pair then sums to at most `255 * 128 = 32640`,
    /// safely under the saturation bound, while lanes 0, 2, 4, ... still
    /// carry the requested extreme `a` value.
    fn build_dot_case(w_val: u8, a_val: i8) -> ([u8; 32], [i8; 32]) {
        let w = [w_val; 32];
        let mut a = [a_val; 32];
        if w_val == 255 {
            for slot in a.iter_mut().skip(1).step_by(2) {
                *slot = 0;
            }
        }
        (w, a)
    }

    #[test]
    fn hsum_i32_sums_all_eight_lanes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let lanes: [i32; 8] = [1, -2, 3, -4, 5, -6, 7, 1000];
        let want: i32 = lanes.iter().sum();
        // SAFETY: avx2 confirmed above.
        let got = unsafe { hsum_i32(_mm256_loadu_si256(lanes.as_ptr() as *const __m256i)) };
        assert_eq!(got, want);
    }

    #[test]
    fn dot_u8_i8_matches_scalar_at_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // w: 0, 15 (Q4_K's max nibble), 63 (Q6_K's max quant), 255 (u8's
        // true max). a: 0, 127, -128 (i8's full range).
        for &w_val in &[0u8, 15, 63, 255] {
            for &a_val in &[0i8, 127, -128] {
                let (w, a) = build_dot_case(w_val, a_val);
                let want = scalar_dot_u8_i8(&w, &a);
                // SAFETY: avx2 confirmed above.
                let got = unsafe { dot_u8_i8(load_u8x32(&w), load_i8x32(&a)) };
                assert_eq!(got, want, "w_val={w_val} a_val={a_val} w={w:?} a={a:?}");
            }
        }
    }

    #[test]
    fn sum_i8_matches_scalar_at_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for &a_val in &[0i8, 127, -128] {
            let a = [a_val; 32];
            let want = scalar_sum_i8(&a);
            // SAFETY: avx2 confirmed above.
            let got = unsafe { sum_i8(load_i8x32(&a)) };
            assert_eq!(got, want, "a_val={a_val}");
        }
    }

    #[test]
    fn load_q8_act_masks_lanes_at_or_past_valid() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let scale = half::f16::from_f32(0.25);
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&scale.to_bits().to_le_bytes());
        let mut quants = [0i8; 32];
        for (i, slot) in quants.iter_mut().enumerate() {
            // -16..=15: distinct per lane, spans negative/zero/positive.
            *slot = (i as i32 - 16) as i8;
        }
        for &q in &quants {
            block.push(q as u8);
        }
        assert_eq!(block.len(), 34);

        for &valid in &[0usize, 17, 32] {
            // SAFETY: avx2 confirmed above; block.len() == 34 == (0 + 1) * 34.
            let (d_a, vec) = unsafe { load_q8_act(&block, 0, valid) };
            assert!(
                (d_a - 0.25).abs() < 1e-6,
                "scale mismatch for valid={valid}: got {d_a}"
            );

            let mut out = [0i8; 32];
            // SAFETY: avx2 confirmed above; `out` is exactly one __m256i.
            unsafe {
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, vec);
            }
            for i in 0..32 {
                let expect = if i < valid { quants[i] } else { 0 };
                assert_eq!(out[i], expect, "lane {i} mismatch for valid={valid}");
            }
        }
    }
}

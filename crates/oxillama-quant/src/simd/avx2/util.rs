//! AVX2 utility intrinsics shared across Q4_0, Q8_0, and future kernels.
//!
//! All functions in this module are `unsafe` and require the `avx2` and `fma`
//! CPU features.  The feature gate at the top restricts compilation to the
//! correct platform; callers must additionally verify CPU support at runtime.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

/// Horizontal sum of a 256-bit packed-float register.
///
/// Reduces eight FP32 lanes to a single `f32` scalar using the
/// hadd/extract pattern that avoids the slower `_mm256_permutevar8x32_ps`
/// path.
///
/// # Safety
/// Requires the `avx` CPU feature (subset of `avx2`).
#[target_feature(enable = "avx")]
pub unsafe fn hsum_f32_avx(v: __m256) -> f32 {
    // Sum adjacent pairs: [a+b, c+d, e+f, g+h, a+b, c+d, e+f, g+h]
    let hadd0 = _mm256_hadd_ps(v, v);
    // Sum adjacent pairs again: [a+b+c+d, e+f+g+h, ...]
    let hadd1 = _mm256_hadd_ps(hadd0, hadd0);

    // Extract the two 128-bit lanes and add them
    let lo = _mm256_castps256_ps128(hadd1);
    let hi = _mm256_extractf128_ps(hadd1, 1);
    let sum128 = _mm_add_ps(lo, hi);

    // The final sum sits in the lowest lane
    _mm_cvtss_f32(sum128)
}

/// Read two bytes from `bytes` as a little-endian IEEE 754 FP16 value and
/// return the FP32 equivalent.
///
/// Uses the `half` crate for the conversion, which handles denormals,
/// infinities, and NaNs correctly.
///
/// # Safety
/// `bytes` must be at least 2 bytes long.  The caller is responsible for
/// ensuring the slice bounds are valid before calling this function.
#[inline(always)]
pub unsafe fn f16_to_f32(bytes: &[u8]) -> f32 {
    // SAFETY: caller guarantees bytes.len() >= 2
    half::f16::from_le_bytes([*bytes.get_unchecked(0), *bytes.get_unchecked(1)]).to_f32()
}

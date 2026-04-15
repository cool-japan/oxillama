//! AVX-512 utility intrinsics shared across Q4_0, Q8_0, Q4_K, and Q1_0_G128 kernels.
//!
//! All functions in this module are `unsafe` and require the `avx512f` CPU feature.
//! The feature gate at the top restricts compilation to the correct platform.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

/// Horizontal sum of a 512-bit packed-float register.
///
/// Reduces sixteen FP32 lanes to a single `f32` scalar using the
/// `_mm512_reduce_add_ps` intrinsic.
///
/// # Safety
/// Requires the `avx512f` CPU feature.
#[target_feature(enable = "avx512f")]
pub unsafe fn hsum_f32_avx512(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
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

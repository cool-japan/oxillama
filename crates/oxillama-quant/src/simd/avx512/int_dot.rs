//! Integer lane primitives shared by every AVX-512 fused Q8_0-activation kernel.
//!
//! The fused decode path (`QuantKernel::matvec_q8_fused`) never leaves the
//! integer domain inside a block: it computes the exact `i32` dot product of a
//! block's 32 weight quants against 32 Q8_0 activation quants, then applies a
//! single `f32` scale.  That is also what ggml does
//! (`sumf += (d_x*d_y) * sumi` in `ggml/src/ggml-cpu/quants.c`), which is why
//! the golden vectors in `tests/avx512_fused_goldens.rs` can be taken straight
//! from llama.cpp's C.
//!
//! # Two tiers, one arithmetic result
//!
//! | Tier | Feature gate | 32-lane dot costs |
//! |------|--------------|-------------------|
//! | [`Bw`]   | `avx512f` + `avx512bw` | 2 × `VPMOVSXBW` + 1 × `VPMADDWD` + reduce |
//! | [`Base`] | `avx512f` only         | 4 × `VPMOVSXBD` + 2 × `VPMULLD` + 1 × `VPADDD` + reduce |
//!
//! **AVX-512BW is a separate `CPUID` bit from AVX-512F.**  Every `epi8`/`epi16`
//! instruction — `_mm512_cvtepi8_epi16`, `_mm512_madd_epi16` — is BW-only;
//! AVX-512F defines nothing narrower than 32-bit lanes.  Knights Landing and
//! Knights Mill are AVX-512F without BW, so the tier is chosen from
//! [`crate::dispatch::SimdCapabilities::avx512bw`] at runtime rather than
//! assumed.
//!
//! Both tiers are **exact integer arithmetic over the same products**, so they
//! return bit-identical `i32` values; only the instruction count differs.  No
//! intermediate can overflow: the widest case is 32 lanes of `31 × -128`,
//! i.e. `|Σ| ≤ 126_976`, and `VPMADDWD`'s pairwise sums stay under
//! `2 × 255 × 128 = 65_280`.
//!
//! # AVX-512-VNNI is deliberately *not* used
//!
//! `_mm512_dpbusd_epi32` would halve the instruction count again, but it is a
//! third independent feature bit (Cascade Lake and later) and its `u8 × i8`
//! operand typing needs the abs/sign fixup that llama.cpp writes as
//! `mul_sum_i8_pairs_float` — a fixup with no 512-bit `VPSIGNB` to build it
//! from.  Adding a third untestable code path on a machine that cannot execute
//! any of them is a bad trade; the BW path already reaches one instruction per
//! 32 multiply-accumulates.  See this crate's README for the verification
//! status.
//!
//! # 256-bit intrinsics inside `avx512f` functions
//!
//! `_mm256_*`/`_mm_*` intrinsics appear inside `#[target_feature(enable =
//! "avx512f")]` bodies here and throughout `simd/avx512/`.  That is sound
//! because LLVM's x86 feature graph makes `avx512f` imply `avx2`, `fma` and
//! `f16c` (`X86.td`: `FeatureAVX512 … [FeatureAVX2, FeatureFMA, FeatureF16C]`),
//! so those instructions are enabled for codegen and inline normally.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

/// Whether the AVX-512BW tier may be used on this CPU.
///
/// Reads the cached `CPUID` result — see
/// [`crate::dispatch::SimdCapabilities::avx512bw`].  Callers hoist this out of
/// their row loop and pick the wrapper once per `matvec_q8_fused` call.
#[inline]
pub(crate) fn have_avx512bw() -> bool {
    crate::simd::cached_capabilities().avx512bw
}

/// Integer lane arithmetic over 32 8-bit values held in a 256-bit register.
///
/// Implemented twice — [`Bw`] and [`Base`] — with identical results.  Every
/// method is `#[inline(always)]` and carries **no** `#[target_feature]`
/// attribute of its own: the attribute lives on the outer row kernel, and
/// these bodies inherit it by being inlined into it.  That is what keeps the
/// BW instructions confined to callers that verified the BW bit.
///
/// # Safety
///
/// Every method requires the caller's CPU to support the features its tier
/// documents (`avx512f`, plus `avx512bw` for [`Bw`]).
pub(crate) trait Int8Lanes {
    /// `Σ_{i<32} x[i] * y[i]` with both operands read as **signed** `i8`.
    ///
    /// # Safety
    /// Caller must have the tier's CPU features.
    unsafe fn dot32_i8(x: __m256i, y: __m256i) -> i32;

    /// `Σ_{i<32} x[i] * y[i]` with `x` read as **unsigned** `u8` and `y` as
    /// signed `i8`.  Used by the formats whose quants are unsigned before the
    /// scale is applied (Q5_1's `w = d·q + m`).
    ///
    /// # Safety
    /// Caller must have the tier's CPU features.
    unsafe fn dot32_u8_i8(x: __m256i, y: __m256i) -> i32;

    /// `Σ_{i<32} y[i]` with `y` read as signed `i8`.
    ///
    /// # Safety
    /// Caller must have the tier's CPU features.
    unsafe fn sum32_i8(y: __m256i) -> i32;
}

/// AVX-512BW tier: 16-bit lanes, `VPMADDWD` does 32 multiply-accumulates per
/// instruction.
pub(crate) struct Bw;

/// AVX-512F-only tier: 32-bit lanes, `VPMULLD` does 16 multiplies per
/// instruction.  Bit-identical to [`Bw`]; exists so a Knights Landing part
/// (AVX-512F without AVX-512BW) still gets a working 512-bit kernel instead of
/// an illegal instruction.
pub(crate) struct Base;

impl Int8Lanes for Bw {
    #[inline(always)]
    unsafe fn dot32_i8(x: __m256i, y: __m256i) -> i32 {
        // Scalar model (tests/avx512_fused_goldens.rs::model::dot32_i8):
        //   xw[i] = x[i] as i16 (sign-extended)          <- _mm512_cvtepi8_epi16
        //   yw[i] = y[i] as i16 (sign-extended)          <- _mm512_cvtepi8_epi16
        //   p[k]  = xw[2k]*yw[2k] + xw[2k+1]*yw[2k+1]    <- _mm512_madd_epi16
        //   result = Σ_{k<16} p[k]                       <- _mm512_reduce_add_epi32
        let xw = _mm512_cvtepi8_epi16(x);
        let yw = _mm512_cvtepi8_epi16(y);
        _mm512_reduce_add_epi32(_mm512_madd_epi16(xw, yw))
    }

    #[inline(always)]
    unsafe fn dot32_u8_i8(x: __m256i, y: __m256i) -> i32 {
        // Identical to `dot32_i8` except `x` is zero-extended.  Zero-extended
        // u8 lands in 0..=255, which is non-negative as i16, so the signed
        // `VPMADDWD` still computes the intended product.
        let xw = _mm512_cvtepu8_epi16(x);
        let yw = _mm512_cvtepi8_epi16(y);
        _mm512_reduce_add_epi32(_mm512_madd_epi16(xw, yw))
    }

    #[inline(always)]
    unsafe fn sum32_i8(y: __m256i) -> i32 {
        // Σ y[i] == dot(y, ones): VPMADDWD against a vector of 1s adds
        // adjacent pairs into i32 lanes, then the tree reduction finishes it.
        let yw = _mm512_cvtepi8_epi16(y);
        _mm512_reduce_add_epi32(_mm512_madd_epi16(yw, _mm512_set1_epi16(1)))
    }
}

impl Int8Lanes for Base {
    #[inline(always)]
    unsafe fn dot32_i8(x: __m256i, y: __m256i) -> i32 {
        // Scalar model (tests/avx512_fused_goldens.rs::model::dot32_i8):
        //   lo half = lanes 0..16, hi half = lanes 16..32
        //   each widened i8 -> i32, multiplied 16-wide, summed.
        let xl = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(x));
        let yl = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(y));
        let xh = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(x, 1));
        let yh = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(y, 1));
        let p = _mm512_add_epi32(_mm512_mullo_epi32(xl, yl), _mm512_mullo_epi32(xh, yh));
        _mm512_reduce_add_epi32(p)
    }

    #[inline(always)]
    unsafe fn dot32_u8_i8(x: __m256i, y: __m256i) -> i32 {
        let xl = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(x));
        let yl = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(y));
        let xh = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(x, 1));
        let yh = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(y, 1));
        let p = _mm512_add_epi32(_mm512_mullo_epi32(xl, yl), _mm512_mullo_epi32(xh, yh));
        _mm512_reduce_add_epi32(p)
    }

    #[inline(always)]
    unsafe fn sum32_i8(y: __m256i) -> i32 {
        let yl = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(y));
        let yh = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(y, 1));
        _mm512_reduce_add_epi32(_mm512_add_epi32(yl, yh))
    }
}

/// Expand the low 16 bits of a Q5_0/Q5_1 `qh` word into 16 bytes that are
/// `0x10` where the bit is set and `0x00` where it is clear.
///
/// This is the fifth bit of each 5-bit quant, pre-shifted into position so it
/// can be `OR`-ed straight onto a 4-bit nibble.
///
/// Pure AVX-512F: the bit pattern *is* an opmask, `_mm512_maskz_set1_epi32`
/// turns it into sixteen 32-bit lanes of `0x10`/`0`, and `VPMOVDB`
/// (`_mm512_cvtepi32_epi8`) truncates those lanes to sixteen bytes in order.
/// The AVX2 kernel builds the same 16 bytes with a scalar
/// `core::array::from_fn` loop plus a load; this replaces that with three
/// instructions and no memory round-trip.
///
/// Scalar model (`tests/avx512_fused_goldens.rs::model::expand_qh_nibble_bits`):
/// `out[i] = if (bits >> i) & 1 == 1 { 0x10 } else { 0x00 }`.
///
/// # Safety
/// Caller must have the `avx512f` CPU feature.
#[inline(always)]
pub(crate) unsafe fn expand_qh_bits_16(bits: u16) -> __m128i {
    _mm512_cvtepi32_epi8(_mm512_maskz_set1_epi32(bits, 0x10))
}

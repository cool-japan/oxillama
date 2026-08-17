//! AVX-512 accelerated quantization kernels (x86_64 only, `simd-avx512` feature).
//!
//! All kernels in this module require the `avx512f` CPU feature and are
//! guarded by `#[target_feature(enable = "avx512f")]` on their inner
//! functions.  The [`crate::dispatch::KernelDispatcher`] checks for AVX-512
//! support at runtime before constructing any of these kernels.
//!
//! ## Kernels
//!
//! | Struct | Format | Block size | Block bytes | Throughput vs AVX2 |
//! |--------|--------|-----------|-------------|-------------------|
//! | [`Q4_0Avx512`]      | Q4_0       | 32  | 18  | ~2× |
//! | [`Q4_1Avx512`]      | Q4_1       | Q4_1       | 32  | 20  | ~2× |
//! | [`Q8_0Avx512`]      | Q8_0       | 32  | 34  | ~2× |
//! | [`Q8_1Avx512`]      | Q8_1       | 32  | 36  | ~2× |
//! | [`Q2_KAvx512`]      | Q2_K       | 256 | 84  | ~2× |
//! | [`Q3_KAvx512`]      | Q3_K       | 256 | 110 | ~2× |
//! | [`Q4_KAvx512`]      | Q4_K       | 256 | 144 | ~2× |
//! | [`Q5_KAvx512`]      | Q5_K       | 256 | 176 | ~2× |
//! | [`Q5_1Avx512`]      | Q5_1       | 32  | 24  | ~2× |
//! | [`Q6_KAvx512`]      | Q6_K       | 256 | 210 | ~2× |
//! | [`Q1_0G128Avx512`]  | Q1_0_G128  | 128 | 18  | ~2× |
//! | [`Tq1_0Avx512`]     | TQ1_0      | 256 | 54  | ~2× |
//! | [`Tq2_0Avx512`]     | TQ2_0      | 256 | 66  | ~2× |
//! | [`Q5_0Avx512`]      | Q5_0       | 32  | 22  | ~2× |
//! | [`Q8_KAvx512`]      | Q8_K       | 256 | 292 | ~2× |
//! | [`Iq2XxsAvx512`]    | IQ2_XXS    | 256 | 66  | ~2× |
//! | [`Iq2XsAvx512`]     | IQ2_XS     | 256 | 74  | ~2× |
//! | [`Iq3SAvx512`]      | IQ3_S      | 256 | 110 | ~2× |
//! | [`Iq4XsAvx512`]     | IQ4_XS     | 256 | 136 | ~2× |
//!
//! The "throughput vs AVX2" column records the *design intent* of the 512-bit
//! rewrite (half as many passes over the same data).  It is **not** a measured
//! number, and no measurement on AVX-512 hardware has been made — see
//! "Verification status" below.
//!
//! ## Fused Q8_0-activation GEMV (`matvec_q8_fused`)
//!
//! `dispatch.rs` selects this tier *before* AVX2, so anything AVX2 overrides and
//! AVX-512 does not is silently **lost** when the `simd-avx512` feature is
//! enabled.  That is exactly what happened to the fused decode path.  The tier
//! now covers every format AVX2 covers:
//!
//! | Format | AVX-512 `matvec_q8_fused` | Gate (`q8_fused_acts_blocks`) |
//! |--------|---------------------------|-------------------------------|
//! | Q8_0, Q8_1, Q5_0, Q5_1 | native 512-bit ([`fused`]) | `ceil(K/32)`, as AVX2 |
//! | Q4_0                   | native 512-bit ([`fused`]) | none — mirrors AVX2, which also leaves Q4_0 ungated |
//! | Q2_K, Q3_K, Q4_K, Q6_K | delegated to AVX2, deliberately | `ceil(K/256)*8`, as AVX2 |
//! | Q5_K                   | delegated to AVX2, deliberately | none — mirrors AVX2 |
//!
//! The K-quant delegation is a decision, not an omission: their AVX2 row
//! kernels interleave per-sub-block `f32` scale/min combinations with the
//! integer dots, and re-deriving that in 512-bit lanes cannot be validated on
//! the hardware available here.  Every AVX-512F CPU also has AVX2+FMA, so
//! delegating costs nothing relative to the AVX2 tier and removes the
//! regression.  See each K-quant kernel's `matvec_q8_fused` for the per-format
//! note.
//!
//! ## Verification status
//!
//! **No code in this module has been executed on AVX-512 hardware.**  What has
//! been verified: cross-compilation and Clippy for `x86_64-unknown-linux-gnu`
//! with `simd-avx512` (alone and combined with `simd-avx2`), and golden tests
//! that hold a scalar model of each fused kernel's lane arithmetic against
//! constants produced by executing llama.cpp's C reference
//! (`tests/avx512_fused_goldens.rs`).  The `#[test]`s inside these modules that
//! call the intrinsics directly are gated on `is_x86_feature_detected!` and
//! skip silently on the aarch64 development host; they are what a future
//! AVX-512 CI run should exercise first.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

pub mod fused;
pub mod int_dot;
pub mod iq2_xs;
pub mod iq2_xxs;
pub mod iq3_s;
pub mod iq4_xs;
pub mod q1_0_g128;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod q8_1;
pub mod q8_k;
pub mod tq1_0;
pub mod tq2_0;
mod util;

pub use iq2_xs::Iq2XsAvx512;
pub use iq2_xxs::Iq2XxsAvx512;
pub use iq3_s::Iq3SAvx512;
pub use iq4_xs::Iq4XsAvx512;
pub use q1_0_g128::Q1_0G128Avx512;
pub use q2_k::Q2_KAvx512;
pub use q3_k::Q3_KAvx512;
pub use q4_0::Q4_0Avx512;
pub use q4_1::Q4_1Avx512;
pub use q4_k::Q4_KAvx512;
pub use q5_0::Q5_0Avx512;
pub use q5_1::Q5_1Avx512;
pub use q5_k::Q5_KAvx512;
pub use q6_k::Q6_KAvx512;
pub use q8_0::Q8_0Avx512;
pub use q8_1::Q8_1Avx512;
pub use q8_k::Q8_KAvx512;
pub use tq1_0::Tq1_0Avx512;
pub use tq2_0::Tq2_0Avx512;

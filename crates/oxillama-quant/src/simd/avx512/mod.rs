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

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

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

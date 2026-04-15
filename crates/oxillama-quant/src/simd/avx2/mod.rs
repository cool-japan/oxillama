//! AVX2+FMA quantization kernels for x86_64.
//!
//! This module is only compiled when both the `simd-avx2` feature flag
//! is enabled and the target architecture is `x86_64`.
//!
//! Runtime CPU feature detection is the responsibility of the caller
//! (e.g., [`crate::dispatch::KernelDispatcher`] or
//! [`crate::simd::cached_capabilities`]).  The kernels here assume that
//! the calling code has already verified that `avx2` and `fma` are
//! available via [`std::arch::is_x86_feature_detected!`].

#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod iq2_xxs;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod q1_0_g128;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod q2_k;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod q3_k;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod q4_0;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod q4_k;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod q5_k;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod q6_k;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod q8_0;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod q8_k;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
mod util;

#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use iq2_xxs::Iq2XxsAvx2;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use q1_0_g128::Q1_0G128Avx2;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use q2_k::Q2_KAvx2;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use q3_k::Q3_KAvx2;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use q4_0::Q4_0Avx2;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use q4_k::Q4_KAvx2;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use q5_k::Q5_KAvx2;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use q6_k::Q6_KAvx2;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use q8_0::Q8_0Avx2;
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub use q8_k::Q8_KAvx2;

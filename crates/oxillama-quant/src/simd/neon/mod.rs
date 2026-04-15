//! AArch64 NEON SIMD kernels for GGUF quantization formats.
//!
//! All items in this module are gated on `#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]`.
//! On other targets this module compiles to nothing.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q1_0_g128;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q4_0;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q4_k;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q5_k;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q6_k;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q8_0;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q8_k;

#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q1_0_g128::Q1_0G128Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q4_0::Q4_0Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q4_k::Q4_KNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q5_k::Q5_KNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q6_k::Q6_KNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q8_0::Q8_0Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q8_k::Q8_KNeon;

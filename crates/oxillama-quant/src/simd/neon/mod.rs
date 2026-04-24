//! AArch64 NEON SIMD kernels for GGUF quantization formats.
//!
//! All items in this module are gated on `#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]`.
//! On other targets this module compiles to nothing.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod iq1_m;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod iq1_s;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod iq2_s;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod iq2_xs;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod iq2_xxs;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod iq3_s;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod iq3_xxs;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod iq4_nl;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod iq4_xs;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q1_0_g128;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q2_k;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q3_k;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q4_0;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q4_1;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q4_k;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q5_0;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q5_1;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q5_k;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q6_k;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q8_0;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q8_1;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod q8_k;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod tq1_0;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod tq2_0;

#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use iq1_m::Iq1MNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use iq1_s::Iq1SNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use iq2_s::Iq2SNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use iq2_xs::Iq2XsNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use iq2_xxs::Iq2XxsNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use iq3_s::Iq3SNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use iq3_xxs::Iq3XxsNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use iq4_nl::Iq4NlNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use iq4_xs::Iq4XsNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q1_0_g128::Q1_0G128Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q2_k::Q2_KNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q3_k::Q3_KNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q4_0::Q4_0Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q4_1::Q4_1Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q4_k::Q4_KNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q5_0::Q5_0Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q5_1::Q5_1Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q5_k::Q5_KNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q6_k::Q6_KNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q8_0::Q8_0Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q8_1::Q8_1Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use q8_k::Q8_KNeon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use tq1_0::Tq1_0Neon;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub use tq2_0::Tq2_0Neon;

//! # oxillama-quant
//!
//! Quantization kernel library for OxiLLaMa.
//!
//! Provides dequantization and fused matmul operations for all GGUF
//! quantization formats. Each format has three implementation tiers:
//!
//! 1. **Reference (naive)** — Pure scalar Rust for correctness.
//! 2. **Portable SIMD** — Cross-platform vectorization.
//! 3. **Platform SIMD** — AVX2, AVX-512, NEON intrinsics.
//!
//! ## Supported Formats (planned)
//!
//! | Category | Types |
//! |----------|-------|
//! | Legacy | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 |
//! | K-Quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K |
//! | I-Quants | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL |
//! | 1-Bit | Q1_0_G128 (from OxiBonsai) |
//! | Float | F16, BF16, F32 |

pub mod dispatch;
pub mod error;
pub mod lora;
pub mod parallel;
pub mod quantize;
pub mod reference;
pub mod simd;
pub mod traits;
pub mod types;

pub use dispatch::{global_dispatcher, CachedDispatcher, KernelDispatcher};
pub use error::{QuantError, QuantResult};
pub use lora::LoraAdapter;
pub use quantize::{
    dequantize_to_f32, quantize_f16_to_q4_0, quantize_f16_to_q8_0, quantize_f32_to_q4_0,
    quantize_f32_to_q8_0,
};
pub use traits::QuantKernel;
pub use types::{BlockInfo, QuantTensor};

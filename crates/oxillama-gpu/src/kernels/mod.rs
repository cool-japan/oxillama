//! Kernel registry — GPU-accelerated GEMV operations.
//!
//! The [`GpuKernel`] trait defines the interface for all GPU-backed tensor
//! operations.  Two concrete implementations are provided:
//!
//! - [`Q4_0GpuKernel`] — dequantises Q4_0 on the CPU then dispatches a pure
//!   f32 GEMV compute shader on the GPU.
//! - [`Q8_0GpuKernel`] — same pattern for Q8_0.
//!
//! When the `gpu` feature is disabled both types compile as zero-size structs
//! whose `gemv` method always returns `GpuError::NoAdapter`.

pub mod batched_gemv;
pub mod f16_accumulator;
pub mod q1_0_g128;
pub mod q4_0;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

pub use batched_gemv::{batched_gemv_f32, BatchedGemvConfig, BatchedGpuKernel};
#[cfg(any(feature = "gpu", test))]
pub use f16_accumulator::{dequant_q4_0_to_f16, dequant_q8_0_to_f16};
#[cfg(feature = "gpu")]
pub use f16_accumulator::{f16_gemv, upload_f16};
pub use f16_accumulator::{supports_f16, F16AccumulatorConfig};
pub use q1_0_g128::Q1_0_G128GpuKernel;
pub use q4_0::Q4_0GpuKernel;
pub use q4_k::Q4_KGpuKernel;
pub use q5_k::Q5_KGpuKernel;
pub use q6_k::Q6_KGpuKernel;
pub use q8_0::Q8_0GpuKernel;

use crate::context::GpuContext;
use crate::error::GpuResult;

/// Trait for GPU-accelerated GEMV operations.
///
/// Implementations are expected to:
/// 1. Dequantise `weight_bytes` into an f32 buffer (CPU side).
/// 2. Upload weights and `input` to the GPU.
/// 3. Dispatch the f32 GEMV compute shader.
/// 4. Read back the results into `output`.
///
/// When `feature = "gpu"` is absent, `gemv` must return
/// `Err(GpuError::NoAdapter)` so callers can fall back to CPU kernels.
pub trait GpuKernel: Send + Sync {
    /// Compute `output[i] = Σ_j weight[i*cols+j] * input[j]`.
    ///
    /// - `weight_bytes` — raw quantised weight bytes.
    /// - `input`        — input vector, length `cols`.
    /// - `output`       — output vector, length `rows`.
    fn gemv(
        &self,
        ctx: &GpuContext,
        weight_bytes: &[u8],
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> GpuResult<()>;
}

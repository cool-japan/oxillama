//! Core traits for quantization kernels.
//!
//! Every quantization type (Q4_0, Q4_K, Q8_0, Q1_0_G128, etc.) implements
//! the [`QuantKernel`] trait, providing dequantization and fused matrix
//! multiply operations.

use crate::error::QuantResult;
use crate::types::QuantTensor;

/// Trait for quantization-specific compute kernels.
///
/// Each GGUF quantization format provides its own implementation with
/// three tiers:
/// 1. **Reference (naive):** Pure scalar Rust for correctness verification.
/// 2. **Portable SIMD:** `std::simd` for cross-platform vectorization.
/// 3. **Platform-specific:** AVX2, AVX-512, NEON intrinsics behind safe wrappers.
pub trait QuantKernel: Send + Sync {
    /// Dequantize a single block to FP32 values.
    ///
    /// # Arguments
    /// * `block` - Raw bytes of one quantized block.
    /// * `output` - Destination buffer for dequantized FP32 values.
    ///   Must have length >= [`block_size()`](Self::block_size).
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()>;

    /// Fused quantized-matrix x FP32-vector product (GEMV).
    ///
    /// Computes `output = quant_matrix @ input` where `quant_matrix` is stored
    /// in the quantized format and `input`/`output` are FP32 vectors.
    ///
    /// # Arguments
    /// * `quant_matrix` - The quantized weight matrix.
    /// * `input` - FP32 input vector of length K.
    /// * `output` - FP32 output vector of length N (rows of the matrix).
    fn gemv(
        &self,
        quant_matrix: &QuantTensor,
        input: &[f32],
        output: &mut [f32],
    ) -> QuantResult<()>;

    /// Fused quantized-matrix x FP32-matrix product (GEMM).
    ///
    /// Computes `output = quant_matrix @ input_matrix` for batched operations
    /// (e.g., prompt prefill).
    ///
    /// # Arguments
    /// * `quant_matrix` - The quantized weight matrix (N x K).
    /// * `input` - Row-major FP32 input matrix [M x K].
    /// * `output` - Row-major FP32 output matrix [M x N].
    /// * `m` - Number of rows in the input/output matrices.
    /// * `n` - Number of rows in the weight matrix (output columns).
    /// * `k` - Shared inner dimension.
    fn gemm(
        &self,
        quant_matrix: &QuantTensor,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> QuantResult<()>;

    /// Number of weights per quantized block.
    fn block_size(&self) -> usize;

    /// Number of bytes per quantized block.
    fn block_bytes(&self) -> usize;

    /// Display name of this quantization type (e.g., "Q4_0", "Q1_0_G128").
    fn name(&self) -> &'static str;
}

//! Core traits for quantization kernels.
//!
//! Every quantization type (Q4_0, Q4_K, Q8_0, Q1_0_G128, etc.) implements
//! the [`QuantKernel`] trait, providing dequantization and fused matrix
//! multiply operations.

use crate::error::{QuantError, QuantResult};
use crate::types::QuantTensor;

/// Q8_0 block constants — used by the `matvec_q8_fused` default implementation.
const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

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

    /// Fused dequant + Q8_0 activation GEMV.
    ///
    /// Computes `out[row] += Σ_block (dequant_q4_weight_block · dequant_q8_0_act_block)`
    /// in a single pass through quantized registers — no f32 scratch buffer needed
    /// by SIMD overrides.
    ///
    /// The default implementation is a scalar fallback that dequantizes each block
    /// to a 32-element f32 scratch and dot-products with the f32-converted Q8_0
    /// activations.  Platform-specific kernels (AVX2, NEON) override this method
    /// to keep everything in SIMD registers.
    ///
    /// # Arguments
    /// * `weights` — raw bytes of `n_rows × blocks_per_row × block_bytes` for
    ///   the weight matrix in this kernel's native format.
    /// * `acts_q8` — raw bytes of `blocks_per_row × 34` for the Q8_0 activation
    ///   vector (34 bytes = 2-byte FP16 scale + 32 × i8 values).
    /// * `out` — output accumulator, length must be at least `n_rows`.
    ///   Values are **added** to the existing content (caller must zero
    ///   if a fresh GEMV is desired).
    /// * `n_rows` — number of output elements (weight-matrix rows).
    /// * `n_cols` — inner dimension K; must be a multiple of `block_size()`.
    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        // --- Default scalar fallback ---
        // Validate dimensions.
        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }

        let bs = self.block_size();
        let bb = self.block_bytes();

        if bs == 0 {
            return Err(QuantError::KernelError {
                message: "block_size() returned 0 — cannot fuse GEMV".to_string(),
            });
        }

        let blocks_per_row = n_cols.div_ceil(bs);
        let row_bytes = blocks_per_row * bb;
        let acts_needed = blocks_per_row * Q8_0_BLOCK_BYTES;

        if weights.len() < n_rows * row_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < acts_needed {
            return Err(QuantError::BufferTooSmall {
                needed: acts_needed,
                available: acts_q8.len(),
            });
        }

        // Scratch buffer for one dequantized weight block.
        let mut w_scratch = vec![0.0f32; bs];
        // Scratch buffer for one dequantized Q8_0 activation block.
        let mut a_scratch = [0.0f32; Q8_0_BLOCK_SIZE];

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                // Dequantize weight block into w_scratch.
                let w_block_start = row_start + blk * bb;
                let w_block = &weights[w_block_start..w_block_start + bb];
                self.dequant_block(w_block, &mut w_scratch)?;

                // Dequantize Q8_0 activation block into a_scratch.
                let a_block_start = blk * Q8_0_BLOCK_BYTES;
                let a_block = &acts_q8[a_block_start..a_block_start + Q8_0_BLOCK_BYTES];
                let d_a =
                    half::f16::from_bits(u16::from_le_bytes([a_block[0], a_block[1]])).to_f32();
                let q8_bytes = &a_block[2..];

                let w_start = blk * bs;
                let w_end = (w_start + bs).min(n_cols);
                let valid = w_end - w_start;

                for i in 0..valid {
                    let q = q8_bytes[i] as i8;
                    a_scratch[i] = q as f32 * d_a;
                }

                // Dot product.
                for i in 0..valid {
                    sum += w_scratch[i] * a_scratch[i];
                }
            }

            *out_val += sum;
        }

        Ok(())
    }

    /// Number of weights per quantized block.
    fn block_size(&self) -> usize;

    /// Number of bytes per quantized block.
    fn block_bytes(&self) -> usize;

    /// Display name of this quantization type (e.g., "Q4_0", "Q1_0_G128").
    fn name(&self) -> &'static str;
}

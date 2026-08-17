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

        // This scalar fallback assumes exactly one 32-element Q8_0 activation
        // block per weight block — true only for 32-weight formats (Q4_0,
        // Q5_0, Q5_1, Q8_0, Q8_1). K-quants and Q8_K (block_size 256),
        // Q1_0_G128 (block_size 128), and every IQ/ternary type (block_size
        // 256) all need `block_size() / 32` activation blocks per weight
        // block, indexed `blk * n_sub + sub` — a layout only a
        // kernel-specific override can know. Refuse cleanly here instead of
        // indexing `a_scratch` (fixed at `Q8_0_BLOCK_SIZE` = 32 elements) past
        // its end: that used to panic with "index out of bounds: the len is
        // 32 but the index is 32" for every kernel dispatched through this
        // default with block_size() != 32 (Q8_K, Q1_0_G128, IQ1_S, IQ1_M,
        // IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, TQ1_0, TQ2_0).
        if bs != Q8_0_BLOCK_SIZE {
            return Err(QuantError::KernelError {
                message: format!(
                    "matvec_q8_fused has no default implementation for block_size() == {bs} \
                     (the scalar fallback only supports {Q8_0_BLOCK_SIZE}-weight formats); \
                     kernel '{}' must override matvec_q8_fused itself, or must not be driven \
                     through the fused path (q8_fused_acts_blocks should return None)",
                    self.name()
                ),
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

    /// Batched sibling of [`Self::matvec_q8_fused`]: one weight matrix against
    /// `m` Q8_0 activation vectors.
    ///
    /// This is the prefill kernel.  Prompt processing is bandwidth-bound
    /// because every token re-streams the entire weight matrix through the
    /// core; running `m` tokens per pass reads the weights **once** and pays
    /// only the (integer, SDOT-rate) arithmetic `m` times.
    ///
    /// # Layout
    ///
    /// * `acts_q8` — `m` activation vectors laid out back to back, each
    ///   [`Self::q8_fused_acts_blocks`]`(n_cols)` blocks of 34 bytes.  Vector
    ///   `t` starts at `t * acts_blocks * 34`.
    /// * `out` — **feature-major** `[n_rows][m]`: weight row `row` owns the
    ///   contiguous slice `out[row * m .. (row + 1) * m]`, and `out[row*m + t]`
    ///   is the dot product of weight row `row` with activation vector `t`.
    ///   Feature-major is what lets a weight row stay on one thread and keeps
    ///   its `m` results contiguous; the caller transposes if it wants
    ///   token-major hidden states.
    /// * Values are **added** to `out`, matching [`Self::matvec_q8_fused`].
    ///
    /// # Numerics
    ///
    /// Overrides must produce, for every `(row, t)`, the bit-identical value
    /// that [`Self::matvec_q8_fused`] would produce for that row against
    /// vector `t` alone.  Hoisting the weight decode out of the `t` loop is
    /// allowed (it is the whole point); reassociating the `K` accumulation is
    /// not.
    ///
    /// The default implementation delegates to [`Self::matvec_q8_fused`] once
    /// per vector and scatters, which is correct for every kernel but wins
    /// nothing — the weights are still read `m` times.
    fn matmul_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
        m: usize,
    ) -> QuantResult<()> {
        if m == 0 {
            return Ok(());
        }
        if out.len() < n_rows * m {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows * m,
                got: out.len(),
            });
        }

        // Key the fallback on the dispatch gate, not directly on
        // `block_size()`. `q8_fused_acts_blocks` returning `Some(..)` is the
        // kernel's own promise about its activation layout (see that
        // method's doc below): when it is open, trust it and delegate to
        // `matvec_q8_fused` — virtually dispatched, so it always reaches the
        // kernel's own override, which is exactly what makes this safe even
        // for block_size() != 32 kernels (Q4_K/Q6_K/Q2_K/Q3_K's AVX2/NEON
        // kernels all open this gate without overriding `matmul_q8_fused`
        // itself). Only when the gate is closed (`None`) do we fall back to
        // requiring `block_size() == 32`, refusing otherwise — that matches
        // `matvec_q8_fused`'s default exactly, and refusing here (before
        // computing a stride) avoids a `BufferTooSmall` below that would
        // describe the wrong problem. See `matvec_q8_fused`'s default for
        // the full explanation.
        let acts_blocks = match self.q8_fused_acts_blocks(n_cols) {
            Some(n) => n,
            None => {
                if self.block_size() != Q8_0_BLOCK_SIZE {
                    return Err(QuantError::KernelError {
                        message: format!(
                            "matmul_q8_fused has no default implementation for block_size() == {} \
                             (the scalar fallback only supports {Q8_0_BLOCK_SIZE}-weight formats); \
                             kernel '{}' must override matmul_q8_fused/matvec_q8_fused itself, or \
                             must not be driven through the fused path (q8_fused_acts_blocks should \
                             return None)",
                            self.block_size(),
                            self.name()
                        ),
                    });
                }
                n_cols.div_ceil(Q8_0_BLOCK_SIZE)
            }
        };
        let acts_stride = acts_blocks * Q8_0_BLOCK_BYTES;

        let mut scratch = vec![0.0f32; n_rows];
        for t in 0..m {
            scratch.fill(0.0);
            let start = t * acts_stride;
            let vec_acts =
                acts_q8
                    .get(start..start + acts_stride)
                    .ok_or(QuantError::BufferTooSmall {
                        needed: (t + 1) * acts_stride,
                        available: acts_q8.len(),
                    })?;
            self.matvec_q8_fused(weights, vec_acts, &mut scratch, n_rows, n_cols)?;
            for (row, &v) in scratch.iter().enumerate() {
                out[row * m + t] += v;
            }
        }
        Ok(())
    }

    /// Number of Q8_0 activation blocks [`Self::matvec_q8_fused`] consumes for
    /// an inner dimension of `n_cols`, or `None` when this kernel must not be
    /// driven through the fused path.
    ///
    /// This is the **dispatch gate** for the fused decode path, and it is
    /// deliberately opt-in per kernel for two independent reasons:
    ///
    /// * *Correctness.*  The trait's default `matvec_q8_fused` assumes one
    ///   Q8_0 activation block per weight block, which is only true for
    ///   32-weight formats.  A K-quant (256 weights per block) needs eight,
    ///   and its SIMD override indexes them as `blk * 8 + sub`.  Returning the
    ///   count from the kernel keeps that layout knowledge in one place.
    /// * *Speed.*  A fused kernel is only worth dispatching to when it beats
    ///   this kernel's own [`Self::gemv`]; a scalar "fused" body that walks the
    ///   same weights with none of the SIMD is strictly slower.  Kernels that
    ///   have not been measured to win keep the default `None` and stay on the
    ///   f32-activation GEMV.
    ///
    /// Returning `Some(n)` is a promise that `matvec_q8_fused` reads at most
    /// `n * 34` bytes of `acts_q8`.
    fn q8_fused_acts_blocks(&self, n_cols: usize) -> Option<usize> {
        let _ = n_cols;
        None
    }

    /// Number of weights per quantized block.
    fn block_size(&self) -> usize;

    /// Number of bytes per quantized block.
    fn block_bytes(&self) -> usize;

    /// Display name of this quantization type (e.g., "Q4_0", "Q1_0_G128").
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::KernelDispatcher;

    /// Regression test for the panic an auditor found by executing the
    /// default `matvec_q8_fused` against the real dispatcher on aarch64:
    /// `index out of bounds: the len is 32 but the index is 32` for every
    /// dispatched kernel whose `block_size()` is not 32 (all K-quants, Q8_K,
    /// Q1_0_G128, every IQ type, and the ternary TQ1_0/TQ2_0). This iterates
    /// every type [`KernelDispatcher::supported_types`] returns, invokes the
    /// dispatched kernel's `matvec_q8_fused` with a well-formed but minimal
    /// buffer, and asserts the call returns cleanly — `Ok` for kernels that
    /// have their own override (or a genuine 32-weight format falling
    /// through to the fixed default) and `Err` (never a panic) for kernels
    /// with block_size() != 32 relying on the unmodified default.
    ///
    /// This test FAILED (panicked) before the `block_size() != 32` guard was
    /// added to the default `matvec_q8_fused`, and PASSES after it.
    #[test]
    fn default_matvec_q8_fused_never_panics_for_any_dispatched_kernel() {
        let dispatcher = KernelDispatcher::new();
        for tensor_type in dispatcher.supported_types() {
            let kernel = match dispatcher.get_kernel(tensor_type) {
                Ok(k) => k,
                Err(_) => continue, // Type not constructible in this build; not this test's concern.
            };

            let bs = kernel.block_size();
            let bb = kernel.block_bytes();
            if bs == 0 || bb == 0 {
                // F32/F16/BF16-style "1 weight per block" kernels do not
                // participate in the Q8_0-activation fused path at all.
                continue;
            }

            let n_rows = 1usize;
            let n_cols = bs; // exactly one weight block
            let weights = vec![0u8; bb];
            // One Q8_0 activation block (34 bytes) per `block_size()/32`
            // sub-blocks — generous enough that a correct override finds
            // enough bytes; a refusing default never reads it anyway.
            let acts_blocks = bs.div_ceil(32).max(1);
            let acts = vec![0u8; acts_blocks * 34];
            let mut out = vec![0.0f32; n_rows];

            // The assertion is simply "this does not panic": both `Ok(())`
            // (kernel handles it, correctly or via the 32-block default) and
            // `Err(_)` (kernel/default correctly refuses) are acceptable
            // outcomes; a panic is not.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                kernel.matvec_q8_fused(&weights, &acts, &mut out, n_rows, n_cols)
            }));
            assert!(
                result.is_ok(),
                "matvec_q8_fused panicked for {tensor_type:?} (kernel '{}', block_size={bs})",
                kernel.name()
            );

            // Kernels whose block_size() != 32 and that have not overridden
            // matvec_q8_fused must refuse with an Err, never silently
            // succeed with a wrong answer or panic.
            if bs != Q8_0_BLOCK_SIZE {
                let inner = result.expect("checked above");
                // A kernel MAY override matvec_q8_fused (e.g. NEON/AVX2
                // fused kernels) and return Ok — that is fine. What matters
                // is that nothing panicked, which is already asserted above.
                let _ = inner;
            }
        }
    }

    /// Focused check of the default (non-overridden) behavior: a bare
    /// `block_size() != 32` kernel driven through the *trait default* must
    /// return `Err`, not panic and not silently miscompute. `Iq2XxsRef` has
    /// block_size 256 and does not override `matvec_q8_fused`.
    #[test]
    fn default_matvec_q8_fused_refuses_non_32_block_size() {
        use crate::reference::Iq2XxsRef;
        let kernel = Iq2XxsRef;
        assert_eq!(kernel.block_size(), 256);

        let weights = vec![0u8; kernel.block_bytes()];
        let acts = vec![0u8; 8 * 34]; // 256 / 32 = 8 Q8_0 blocks
        let mut out = vec![0.0f32; 1];

        let result = kernel.matvec_q8_fused(&weights, &acts, &mut out, 1, 256);
        assert!(
            result.is_err(),
            "default matvec_q8_fused must refuse block_size() != 32, got {result:?}"
        );
    }

    /// Same refusal, but through `matmul_q8_fused`'s default, and asserting
    /// it fails *before* misreporting a buffer-size problem — the error
    /// message must name block_size/override, not a byte count.
    #[test]
    fn default_matmul_q8_fused_refuses_non_32_block_size() {
        use crate::reference::Iq2XxsRef;
        let kernel = Iq2XxsRef;

        let weights = vec![0u8; kernel.block_bytes()];
        let acts = vec![0u8; 8 * 34];
        let mut out = vec![0.0f32; 1];

        let result = kernel.matmul_q8_fused(&weights, &acts, &mut out, 1, 256, 1);
        match result {
            Err(QuantError::KernelError { message }) => {
                assert!(
                    message.contains("block_size"),
                    "error should explain the block_size mismatch, got: {message}"
                );
            }
            other => panic!("expected a KernelError naming the block_size mismatch, got {other:?}"),
        }
    }
}

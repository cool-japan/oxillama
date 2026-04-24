//! AVX2+FMA accelerated Q4_0 quantization kernel.
//!
//! Q4_0 block layout (18 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..18]  — 16 packed bytes encoding 32 × 4-bit unsigned nibbles
//!
//! Each weight reconstructs as `(nibble − 8) × d`.
//! Nibble order: for byte `b[i]`, `lo = b[i] & 0x0F` → weight `2i`,
//!                                `hi = b[i] >> 4`   → weight `2i+1`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q4_0: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q4_0 block: 2 (FP16 scale) + 16 (nibble data).
pub const BLOCK_BYTES: usize = 18;

/// AVX2+FMA accelerated Q4_0 kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
pub struct Q4_0Avx2;

impl QuantKernel for Q4_0Avx2 {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: BLOCK_SIZE,
                available: output.len(),
            });
        }

        // SAFETY: we verified block.len() >= 18 and output.len() >= 32 above.
        // The CPU features avx2+fma are guaranteed by KernelDispatcher.
        unsafe { dequant_block_avx2(block, output) }
        Ok(())
    }

    fn gemv(
        &self,
        quant_matrix: &QuantTensor,
        input: &[f32],
        output: &mut [f32],
    ) -> QuantResult<()> {
        let n_rows = quant_matrix.shape[0];
        let n_cols = if quant_matrix.shape.len() > 1 {
            quant_matrix.shape[1]
        } else {
            quant_matrix.n_elements() / n_rows
        };

        if input.len() < n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_cols,
                got: input.len(),
            });
        }
        if output.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: output.len(),
            });
        }

        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let row_bytes = blocks_per_row * BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            // SAFETY: row and block bounds are checked above (n_rows, n_cols).
            // CPU avx2+fma support is guaranteed by KernelDispatcher.
            *out = unsafe {
                gemv_row_avx2(
                    &quant_matrix.data[row_start..row_start + row_bytes],
                    input,
                    blocks_per_row,
                    n_cols,
                )
            };
        }

        Ok(())
    }

    fn gemm(
        &self,
        quant_matrix: &QuantTensor,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> QuantResult<()> {
        for row in 0..m {
            let input_row = &input[row * k..(row + 1) * k];
            let output_row = &mut output[row * n..(row + 1) * n];
            self.gemv(quant_matrix, input_row, output_row)?;
        }
        Ok(())
    }

    /// Fused Q4_0 weight × Q8_0 activation GEMV using AVX2+FMA intrinsics.
    ///
    /// Computes `out[row] += Σ_block (q4_0_weight · q8_0_act)` with ACCUMULATE semantics.
    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }

        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let row_bytes = blocks_per_row * BLOCK_BYTES;
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

        for row in 0..n_rows {
            let row_start = row * row_bytes;
            // SAFETY: bounds checked above; CPU avx2+fma guaranteed by KernelDispatcher.
            let row_sum = unsafe {
                fused_q4_0_q8_0_row_avx2(
                    &weights[row_start..row_start + row_bytes],
                    acts_q8,
                    blocks_per_row,
                    n_cols,
                )
            };
            out[row] += row_sum; // ACCUMULATE
        }

        Ok(())
    }

    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q4_0"
    }
}

/// Q8_0 block bytes for fused GEMV.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Compute fused Q4_0 weight × Q8_0 activation dot product for one row using AVX2+FMA.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * Q8_0_BLOCK_BYTES`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_q4_0_q8_0_row_avx2(
    row_data: &[u8],
    acts_q8: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;
    let eight_i32 = _mm256_set1_epi32(8);

    for blk in 0..blocks_per_row {
        // Weight block.
        let w_off = blk * BLOCK_BYTES;
        // SAFETY: row_data.len() == blocks_per_row * BLOCK_BYTES; blk < blocks_per_row.
        let w_block = &row_data[w_off..w_off + BLOCK_BYTES];
        let d_w = f16_to_f32(w_block);

        // Q8_0 activation block (1:1 with weight block).
        let a_off = blk * Q8_0_BLOCK_BYTES;
        // SAFETY: acts_q8.len() >= blocks_per_row * Q8_0_BLOCK_BYTES.
        let a_block = &acts_q8[a_off..a_off + Q8_0_BLOCK_BYTES];
        let d_a = f16_to_f32(a_block);
        let scale = d_w * d_a;

        let input_offset = blk * BLOCK_SIZE;
        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: full block.
            // SAFETY: w_block has 18 bytes; nibbles at [2..18] = 16 bytes.
            // a_block has 34 bytes; q8 at [2..34] = 32 i8s.
            let raw = _mm_loadu_si128(w_block.as_ptr().add(2) as *const __m128i);
            let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
            let lo_bytes = _mm_and_si128(raw, mask_lo);
            let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo);

            // Interleave lo/hi to match sequential Q8_0 layout.
            let weights_0_15 = _mm_unpacklo_epi8(lo_bytes, hi_bytes);
            let weights_16_31 = _mm_unpackhi_epi8(lo_bytes, hi_bytes);

            // Load Q8_0 i8 activations.
            // SAFETY: a_block[2..34] = 32 valid i8 bytes.
            let qa_ptr = a_block.as_ptr().add(2) as *const __m128i;
            let qa_0 = _mm_loadu_si128(qa_ptr);
            let qa_1 = _mm_loadu_si128(qa_ptr.add(1));

            // Compute dot(w - 8, qa) = dot(w, qa) - 8 * sum(qa).
            // Use _mm256_madd_epi16 for 16-bit multiply-add after widening.
            // Convert u8 nibbles to i16 and subtract 8, then multiply by i8 activations.

            // Group A (weights 0-7): first 8 bytes of weights_0_15 × first 8 i8s of qa_0.
            let w_a = _mm256_sub_epi32(_mm256_cvtepu8_epi32(weights_0_15), eight_i32);
            let q_a = _mm256_cvtepi8_epi32(qa_0);
            let mut acc = _mm256_mullo_epi32(w_a, q_a);

            // Group B (weights 8-15): next 8 bytes of weights_0_15 × next 8 i8s of qa_0.
            let w0_hi = _mm_srli_si128(weights_0_15, 8);
            let w_b = _mm256_sub_epi32(_mm256_cvtepu8_epi32(w0_hi), eight_i32);
            let qa0_hi = _mm_srli_si128(qa_0, 8);
            let q_b = _mm256_cvtepi8_epi32(qa0_hi);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w_b, q_b));

            // Group C (weights 16-23): first 8 bytes of weights_16_31 × first 8 i8s of qa_1.
            let w_c = _mm256_sub_epi32(_mm256_cvtepu8_epi32(weights_16_31), eight_i32);
            let q_c = _mm256_cvtepi8_epi32(qa_1);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w_c, q_c));

            // Group D (weights 24-31): next 8 bytes of weights_16_31 × next 8 i8s of qa_1.
            let w1_hi = _mm_srli_si128(weights_16_31, 8);
            let w_d = _mm256_sub_epi32(_mm256_cvtepu8_epi32(w1_hi), eight_i32);
            let qa1_hi = _mm_srli_si128(qa_1, 8);
            let q_d = _mm256_cvtepi8_epi32(qa1_hi);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w_d, q_d));

            // Horizontal sum of i32 accumulator.
            let dot_i32 = hsum_i32_avx(acc);
            row_sum += scale * dot_i32 as f32;
        } else if remaining > 0 {
            // Scalar tail path.
            let q8_bytes = &a_block[2..];
            let valid = remaining;

            for i in 0..(valid / 2) {
                let byte = w_block[2 + i];
                let q_lo = (byte & 0x0F) as i32 - 8;
                let q_hi = ((byte >> 4) & 0x0F) as i32 - 8;
                let a_lo = q8_bytes[i * 2] as i8 as i32;
                let a_hi = q8_bytes[i * 2 + 1] as i8 as i32;
                row_sum += scale * (q_lo * a_lo + q_hi * a_hi) as f32;
            }
            if valid % 2 == 1 {
                let i = valid / 2;
                let byte = w_block[2 + i];
                let q_lo = (byte & 0x0F) as i32 - 8;
                let a_lo = q8_bytes[i * 2] as i8 as i32;
                row_sum += scale * (q_lo * a_lo) as f32;
            }
        }
    }

    row_sum
}

/// Horizontal sum of an `__m256i` i32 register.
///
/// # Safety
/// Caller must have `avx2` CPU feature.
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32_avx(v: __m256i) -> i32 {
    // Fold high 128 into low 128.
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(hi, lo);
    // Horizontal add within 128 bits.
    let shuf = _mm_shuffle_epi32(sum128, 0b10_11_00_01);
    let sums = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sums, 0b00_00_10_10);
    let sums2 = _mm_add_epi32(sums, shuf2);
    _mm_cvtsi128_si32(sums2)
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 18-byte Q4_0 block to 32 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 18`
/// - `output.len() >= 32`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale.
    // SAFETY: block.len() >= 18 ≥ 2 — guaranteed by caller.
    let d = f16_to_f32(block);

    let vd = _mm256_set1_ps(d);

    // Load the 16 nibble bytes.
    // SAFETY: block.ptr + 2 is valid because block.len() >= 18.
    let raw = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);

    // Split each byte into its low and high nibble.
    // Note: there is no _mm_srli_epi8 in x86.  We use _mm_srli_epi16
    // (16-bit right shift) and then mask each byte to 0x0F to strip
    // the cross-byte contamination introduced by the 16-bit shift.
    let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
    let lo_bytes = _mm_and_si128(raw, mask_lo); // low nibbles in each byte
    let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo); // high nibbles

    // Interleave: first16 = [lo0,hi0,lo1,hi1,...,lo7,hi7]  (weights 0-15)
    //             last16  = [lo8,hi8,...,lo15,hi15]         (weights 16-31)
    let first16 = _mm_unpacklo_epi8(lo_bytes, hi_bytes);
    let last16 = _mm_unpackhi_epi8(lo_bytes, hi_bytes);

    // Convert i8→i32→f32 in four groups of 8, subtract 8, scale by d.
    // Groups: first16[0..8], first16[8..16], last16[0..8], last16[8..16]

    let eight_i32 = _mm256_set1_epi32(8);

    // Group A: first 8 weights from first16
    let a_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(first16), eight_i32);
    let a_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(a_i32), vd);

    // Group B: next 8 weights from first16 (shifted by 8 bytes)
    let first16_hi = _mm_srli_si128(first16, 8);
    let b_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(first16_hi), eight_i32);
    let b_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i32), vd);

    // Group C: first 8 weights from last16
    let c_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(last16), eight_i32);
    let c_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(c_i32), vd);

    // Group D: next 8 weights from last16
    let last16_hi = _mm_srli_si128(last16, 8);
    let d_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(last16_hi), eight_i32);
    let d_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(d_i32), vd);

    // Store all 32 values.
    // SAFETY: output.len() >= 32 — guaranteed by caller.
    let ptr = output.as_mut_ptr();
    _mm256_storeu_ps(ptr, a_f32);
    _mm256_storeu_ps(ptr.add(8), b_f32);
    _mm256_storeu_ps(ptr.add(16), c_f32);
    _mm256_storeu_ps(ptr.add(24), d_f32);
}

/// Compute the dot product of one row of a Q4_0 matrix with an FP32 vector.
///
/// Returns the scalar result for this row.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_avx2(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    let eight_i32 = _mm256_set1_epi32(8);

    for blk in 0..blocks_per_row {
        let block_offset = blk * BLOCK_BYTES;
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;

        // Read FP16 scale.
        // SAFETY: block.len() == BLOCK_BYTES == 18 ≥ 2.
        let d = f16_to_f32(block);

        // Load 16 nibble bytes.
        // SAFETY: block.ptr + 2 valid because BLOCK_BYTES == 18.
        let raw = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);

        let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
        let lo_bytes = _mm_and_si128(raw, mask_lo);
        let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo);

        let first16 = _mm_unpacklo_epi8(lo_bytes, hi_bytes);
        let last16 = _mm_unpackhi_epi8(lo_bytes, hi_bytes);

        // Check whether this block is fully within bounds.
        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 32 weights are valid — use 4 AVX2 FMA lanes.
            // SAFETY: input_offset + 32 <= n_cols <= input.len().
            let inp_ptr = input.as_ptr().add(input_offset);

            // Group A (weights 0-7)
            let wa_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(first16), eight_i32);
            let wa_f32 = _mm256_cvtepi32_ps(wa_i32);
            let ia = _mm256_loadu_ps(inp_ptr);
            let mut acc = _mm256_mul_ps(wa_f32, ia);

            // Group B (weights 8-15)
            let first16_hi = _mm_srli_si128(first16, 8);
            let wb_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(first16_hi), eight_i32);
            let wb_f32 = _mm256_cvtepi32_ps(wb_i32);
            let ib = _mm256_loadu_ps(inp_ptr.add(8));
            acc = _mm256_fmadd_ps(wb_f32, ib, acc);

            // Group C (weights 16-23)
            let wc_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(last16), eight_i32);
            let wc_f32 = _mm256_cvtepi32_ps(wc_i32);
            let ic = _mm256_loadu_ps(inp_ptr.add(16));
            acc = _mm256_fmadd_ps(wc_f32, ic, acc);

            // Group D (weights 24-31)
            let last16_hi = _mm_srli_si128(last16, 8);
            let wd_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(last16_hi), eight_i32);
            let wd_f32 = _mm256_cvtepi32_ps(wd_i32);
            let id = _mm256_loadu_ps(inp_ptr.add(24));
            acc = _mm256_fmadd_ps(wd_f32, id, acc);

            row_sum += hsum_f32_avx(acc) * d;
        } else if remaining > 0 {
            // Tail path: partial block — fall back to scalar to avoid OOB reads.
            // Reconstruct nibbles from raw bytes in the block.
            let mut partial_sum = 0.0f32;
            for i in 0..BLOCK_SIZE / 2 {
                let byte = *block.get_unchecked(2 + i);
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                let idx = input_offset + i * 2;
                if idx + 1 < n_cols {
                    partial_sum += lo as f32 * input[idx];
                    partial_sum += hi as f32 * input[idx + 1];
                } else if idx < n_cols {
                    partial_sum += lo as f32 * input[idx];
                }
            }
            row_sum += partial_sum * d;
        }
        // remaining == 0: block is fully out of bounds, skip
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx2"))]
mod tests {
    use super::*;
    use crate::reference::q4_0::Q4_0Ref;

    fn make_q4_0_block(scale: f32, nibbles: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(nibbles);
        block
    }

    /// Build a single-row QuantTensor from a raw block.
    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q4_0)
    }

    #[test]
    fn test_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return; // skip on machines without AVX2
        }
        let nibbles: [u8; 16] = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB,
            0xED, 0x0F,
        ];
        let block = make_q4_0_block(0.25, &nibbles);

        let mut out_avx2 = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];

        let avx2 = Q4_0Avx2;
        let refk = Q4_0Ref;

        avx2.dequant_block(&block, &mut out_avx2).unwrap();
        refk.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_gemv_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let nibbles: [u8; 16] = [
            0x89, 0x7A, 0x6B, 0x5C, 0x4D, 0x3E, 0x2F, 0x10, 0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5,
            0x96, 0x87,
        ];
        let scale = 0.5f32;
        let block = make_q4_0_block(scale, &nibbles);
        let tensor_avx2 = make_tensor(block.clone(), 32);
        let tensor_ref = make_tensor(block, 32);

        // Use distinct input values to detect permutation bugs.
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_0Avx2.gemv(&tensor_avx2, &input, &mut out_avx2).unwrap();
        Q4_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-4,
            "gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_partial_block() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 20 columns — one partial block of 20 out of 32
        let nibbles = [0x88u8; 16]; // all zeros after -8
        let block = make_q4_0_block(1.0, &nibbles);
        let tensor_avx2 = make_tensor(block.clone(), 20);
        let tensor_ref = make_tensor(block, 20);

        let input = vec![1.0f32; 20];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_0Avx2.gemv(&tensor_avx2, &input, &mut out_avx2).unwrap();
        Q4_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-4,
            "partial gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemm_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 2 rows, 32 cols, batch of 2 inputs (M=2)
        let nibbles: [u8; 16] = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB,
            0xED, 0x0F,
        ];
        let block = make_q4_0_block(0.25, &nibbles);
        let two_row_data = [block.as_slice(), block.as_slice()].concat();
        let tensor_avx2 = QuantTensor::new(
            two_row_data.clone(),
            vec![2, 32],
            oxillama_gguf::GgufTensorType::Q4_0,
        );
        let tensor_ref = QuantTensor::new(
            two_row_data,
            vec![2, 32],
            oxillama_gguf::GgufTensorType::Q4_0,
        );

        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();
        let mut out_avx2 = vec![0.0f32; 4];
        let mut out_ref = vec![0.0f32; 4];

        Q4_0Avx2
            .gemm(&tensor_avx2, &input, &mut out_avx2, 2, 2, 32)
            .unwrap();
        Q4_0Ref
            .gemm(&tensor_ref, &input, &mut out_ref, 2, 2, 32)
            .unwrap();

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "gemm mismatch at [{i}]: avx2={a}, ref={r}"
            );
        }
    }

    // ── matvec_q8_fused ───────────────────────────────────────────────────

    fn make_q8_0_block(scale: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    #[test]
    fn avx2_fused_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let nibbles: [u8; 16] = [
            0x5A, 0xF0, 0x13, 0x7E, 0xC2, 0x48, 0x9D, 0x6B, 0xA3, 0x2F, 0x71, 0xE4, 0x0C, 0x58,
            0xB6, 0xD9,
        ];
        let w_block = make_q4_0_block(0.25, &nibbles);
        let q8_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, -1, 2, -3, 4, -5, 6, -7,
            8, -9, 10, -11, 12, -13, 14, -15, 16,
        ];
        let a_block = make_q8_0_block(0.5, &q8_vals);

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_0Avx2
            .matvec_q8_fused(&w_block, &a_block, &mut out_avx2, 1, 32)
            .expect("avx2 fused");
        crate::reference::q4_0::matvec_q8_fused_reference(&w_block, &a_block, &mut out_ref, 1, 32)
            .expect("ref fused");

        let err = (out_avx2[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "avx2_fused_matches_reference: avx2={} ref={} err={}",
            out_avx2[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn avx2_fused_multi_row() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let n_rows = 4usize;
        let n_cols = 64usize;
        let blocks_per_row = 2usize;
        let nibbles: [u8; 16] = [
            0x13, 0x57, 0x9B, 0xDF, 0x24, 0x68, 0xAC, 0xE0, 0x5F, 0x3A, 0x72, 0x8D, 0xC6, 0x4E,
            0x91, 0xB7,
        ];
        let scales = [0.1f32, 0.25f32, 0.5f32, 1.0f32];
        let d_a = 0.5f32;
        let q8_vals: [i8; 32] = [
            2, 4, -6, 8, -10, 12, -14, 16, 1, -3, 5, -7, 9, -11, 13, -15, 0, 1, -2, 3, -4, 5, -6,
            7, -8, 9, -10, 11, -12, 13, -14, 15,
        ];

        let mut weights: Vec<u8> = Vec::new();
        for &s in &scales {
            for _ in 0..blocks_per_row {
                weights.extend_from_slice(&make_q4_0_block(s, &nibbles));
            }
        }

        let mut acts: Vec<u8> = Vec::new();
        for _ in 0..blocks_per_row {
            acts.extend_from_slice(&make_q8_0_block(d_a, &q8_vals));
        }

        let mut out_avx2 = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q4_0Avx2
            .matvec_q8_fused(&weights, &acts, &mut out_avx2, n_rows, n_cols)
            .expect("avx2 fused multi-row");
        crate::reference::q4_0::matvec_q8_fused_reference(
            &weights,
            &acts,
            &mut out_ref,
            n_rows,
            n_cols,
        )
        .expect("ref fused multi-row");

        for i in 0..n_rows {
            let err = (out_avx2[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "row {i}: avx2={} ref={} err={}",
                out_avx2[i],
                out_ref[i],
                err
            );
        }
    }

    #[test]
    fn avx2_fused_accumulate_semantics() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let w_block = make_q4_0_block(1.0, &[0x88u8; 16]); // zero weights
        let a_block = make_q8_0_block(1.0, &[0i8; 32]);

        let mut out = vec![42.0f32; 1];
        Q4_0Avx2
            .matvec_q8_fused(&w_block, &a_block, &mut out, 1, 32)
            .expect("avx2 fused accumulate");
        assert!(
            (out[0] - 42.0).abs() < 1e-5,
            "accumulation broken: got {}",
            out[0]
        );
    }
}

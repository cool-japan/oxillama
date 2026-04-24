//! AVX-512 accelerated Q4_1 quantization kernel.
//!
//! Q4_1 block layout (20 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..4]   — FP16 minimum `m` (little-endian)
//! - bytes[4..20]  — 16 packed bytes encoding 32 × 4-bit unsigned nibbles
//!
//! Each weight reconstructs as `nibble × d + m`.
//! Nibble order: for byte `b[i]`, `lo = b[i] & 0x0F` → weight `2i`,
//!                                `hi = b[i] >> 4`   → weight `2i+1`.
//! Nibbles are unsigned (0..=15).
//!
//! ## AVX-512 strategy
//!
//! Process all 32 weights in **two** AVX-512 (16-wide) passes instead of
//! the AVX2 kernel's four 8-wide passes.  The formula `nibble × d + m` is
//! implemented as `fmadd(nibble_f32, vd, vm)`.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q4_1: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q4_1 block: 2 (FP16 d) + 2 (FP16 m) + 16 (nibble data).
pub const BLOCK_BYTES: usize = 20;

/// AVX-512 accelerated Q4_1 kernel.
///
/// Requires the `avx512f` CPU feature.  The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q4_1Avx512;

impl QuantKernel for Q4_1Avx512 {
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

        // SAFETY: block.len() >= 20 and output.len() >= 32 verified above.
        // CPU avx512f support guaranteed by KernelDispatcher.
        unsafe { dequant_block_avx512(block, output) }
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
            // SAFETY: row and block bounds are checked above.
            // CPU avx512f support is guaranteed by KernelDispatcher.
            *out = unsafe {
                gemv_row_avx512(
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

    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q4_1"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 20-byte Q4_1 block to 32 FP32 values using AVX-512.
///
/// # Safety
/// - `block.len() >= 20`
/// - `output.len() >= 32`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale d and minimum m.
    // SAFETY: block.len() >= 20 >= 4 — guaranteed by caller.
    let d = f16_to_f32(block);
    let m = f16_to_f32(&block[2..]);

    let vd = _mm512_set1_ps(d);
    let vm = _mm512_set1_ps(m);

    // Load the 16 nibble bytes (qs starts at byte offset 4).
    // SAFETY: block.ptr + 4 valid because block.len() >= 20.
    let raw = _mm_loadu_si128(block.as_ptr().add(4) as *const __m128i);

    // Split each byte into its low and high nibble.
    let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
    let lo_bytes = _mm_and_si128(raw, mask_lo); // low nibbles per byte
    let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo); // high nibbles per byte

    // Interleave: first16 = [lo0,hi0,lo1,hi1,...,lo7,hi7]  (weights 0-15)
    //             last16  = [lo8,hi8,...,lo15,hi15]         (weights 16-31)
    let first16 = _mm_unpacklo_epi8(lo_bytes, hi_bytes);
    let last16 = _mm_unpackhi_epi8(lo_bytes, hi_bytes);

    // Convert u8→u32→f32, then FMA: d * nibble + m — 2 AVX-512 passes.

    // Group A: weights 0-15 from first16
    // SAFETY: _mm512_cvtepu8_epi32 reads 16 bytes from the 128-bit source.
    let a_u32 = _mm512_cvtepu8_epi32(first16);
    let a_f32 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a_u32), vd, vm);

    // Group B: weights 16-31 from last16
    let b_u32 = _mm512_cvtepu8_epi32(last16);
    let b_f32 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(b_u32), vd, vm);

    // Store all 32 values.
    // SAFETY: output.len() >= 32 — guaranteed by caller.
    let ptr = output.as_mut_ptr();
    _mm512_storeu_ps(ptr, a_f32);
    _mm512_storeu_ps(ptr.add(16), b_f32);
}

/// Compute the dot product of one row of a Q4_1 matrix with an FP32 vector.
///
/// Returns the scalar result for this row.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn gemv_row_avx512(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut acc_wd = _mm512_setzero_ps(); // accumulates (nibble * d) * input
    let mut acc_m = _mm512_setzero_ps(); // accumulates m * input

    for blk in 0..blocks_per_row {
        let block_offset = blk * BLOCK_BYTES;
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;

        // Read FP16 scale d and minimum m.
        // SAFETY: block.len() == BLOCK_BYTES == 20 >= 4.
        let d = f16_to_f32(block);
        let m = f16_to_f32(&block[2..]);

        let vd = _mm512_set1_ps(d);
        let vm = _mm512_set1_ps(m);

        // Load 16 nibble bytes (qs at offset 4).
        // SAFETY: block.ptr + 4 valid because BLOCK_BYTES == 20.
        let raw = _mm_loadu_si128(block.as_ptr().add(4) as *const __m128i);

        let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
        let lo_bytes = _mm_and_si128(raw, mask_lo);
        let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo);

        let first16 = _mm_unpacklo_epi8(lo_bytes, hi_bytes);
        let last16 = _mm_unpackhi_epi8(lo_bytes, hi_bytes);

        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 32 weights valid — 2 AVX-512 FMA passes.
            // SAFETY: input_offset + 32 <= n_cols <= input.len().
            let inp_ptr = input.as_ptr().add(input_offset);

            // Group A (weights 0-15): nibble_f32 * d * input + acc_wd
            let wa_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(first16));
            let ia = _mm512_loadu_ps(inp_ptr);
            acc_wd = _mm512_fmadd_ps(_mm512_mul_ps(wa_f32, vd), ia, acc_wd);
            acc_m = _mm512_fmadd_ps(vm, ia, acc_m);

            // Group B (weights 16-31)
            let wb_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(last16));
            let ib = _mm512_loadu_ps(inp_ptr.add(16));
            acc_wd = _mm512_fmadd_ps(_mm512_mul_ps(wb_f32, vd), ib, acc_wd);
            acc_m = _mm512_fmadd_ps(vm, ib, acc_m);
        } else if remaining > 0 {
            // Tail path: partial block — scalar to avoid OOB reads.
            let mut partial = [0.0f32; BLOCK_SIZE];
            let pp = partial.as_mut_ptr();
            _mm512_storeu_ps(
                pp,
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(first16)), vd, vm),
            );
            _mm512_storeu_ps(
                pp.add(16),
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(last16)), vd, vm),
            );

            let mut scalar_sum = 0.0f32;
            for j in 0..remaining {
                scalar_sum += partial[j] * input[input_offset + j];
            }
            // Accumulate scalar result into acc_wd (acc_m already folded into partial).
            acc_wd = _mm512_add_ps(acc_wd, _mm512_set1_ps(scalar_sum));
        }
        // remaining == 0: block fully out of bounds, skip.
    }

    // Total = Σ (nibble * d * input) + Σ (m * input)
    hsum_f32_avx512(acc_wd) + hsum_f32_avx512(acc_m)
}

// ---------------------------------------------------------------------------
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
mod tests {
    use super::*;
    use crate::reference::Q4_1Ref;

    fn make_block(d: f32, m: f32, nibbles: &[u8; 32]) -> Vec<u8> {
        let d_f16 = half::f16::from_f32(d);
        let m_f16 = half::f16::from_f32(m);
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        block.extend_from_slice(&m_f16.to_bits().to_le_bytes());
        for i in 0..16 {
            let lo = nibbles[2 * i] & 0x0F;
            let hi = nibbles[2 * i + 1] & 0x0F;
            block.push(lo | (hi << 4));
        }
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q4_1)
    }

    fn avx512_available() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
    }

    #[test]
    fn test_dequant_matches_reference() {
        if !avx512_available() {
            return;
        }
        let mut nibbles = [0u8; 32];
        for (i, n) in nibbles.iter_mut().enumerate() {
            *n = (i % 16) as u8;
        }
        let block = make_block(0.25, 1.0, &nibbles);

        let mut out_avx512 = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];

        Q4_1Avx512.dequant_block(&block, &mut out_avx512).unwrap();
        Q4_1Ref.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_all_zeros() {
        if !avx512_available() {
            return;
        }
        let block = make_block(1.0, 0.0, &[0u8; 32]);
        let mut out = vec![0.0f32; 32];
        Q4_1Avx512.dequant_block(&block, &mut out).unwrap();
        for &v in &out {
            assert!((v).abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_with_bias() {
        if !avx512_available() {
            return;
        }
        let block = make_block(0.5, 2.0, &[0u8; 32]);
        let mut out = vec![0.0f32; 32];
        Q4_1Avx512.dequant_block(&block, &mut out).unwrap();
        // nibble=0, so weight = 0 * 0.5 + 2.0 = 2.0
        for &v in &out {
            assert!((v - 2.0).abs() < 1e-4, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn test_gemv_matches_reference() {
        if !avx512_available() {
            return;
        }
        let mut nibbles = [0u8; 32];
        for (i, n) in nibbles.iter_mut().enumerate() {
            *n = (i % 16) as u8;
        }
        let block = make_block(0.5, 0.25, &nibbles);
        let tensor_avx512 = make_tensor(block.clone(), 32);
        let tensor_ref = make_tensor(block, 32);

        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_1Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q4_1Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 1e-3,
            "gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_partial_block() {
        if !avx512_available() {
            return;
        }
        // 20 columns — one partial block of 20 out of 32
        let block = make_block(1.0, 0.5, &[8u8; 32]);
        let tensor_avx512 = make_tensor(block.clone(), 20);
        let tensor_ref = make_tensor(block, 20);

        let input = vec![1.0f32; 20];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_1Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q4_1Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 1e-3,
            "partial gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemm_matches_reference() {
        if !avx512_available() {
            return;
        }
        let mut nibbles = [0u8; 32];
        for (i, n) in nibbles.iter_mut().enumerate() {
            *n = ((i * 3 + 1) % 16) as u8;
        }
        let block = make_block(0.25, 1.0, &nibbles);
        let two_row_data = [block.as_slice(), block.as_slice()].concat();
        let tensor_avx512 = QuantTensor::new(
            two_row_data.clone(),
            vec![2, 32],
            oxillama_gguf::GgufTensorType::Q4_1,
        );
        let tensor_ref = QuantTensor::new(
            two_row_data,
            vec![2, 32],
            oxillama_gguf::GgufTensorType::Q4_1,
        );

        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();
        let mut out_avx512 = vec![0.0f32; 4];
        let mut out_ref = vec![0.0f32; 4];

        Q4_1Avx512
            .gemm(&tensor_avx512, &input, &mut out_avx512, 2, 2, 32)
            .unwrap();
        Q4_1Ref
            .gemm(&tensor_ref, &input, &mut out_ref, 2, 2, 32)
            .unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "gemm mismatch at [{i}]: avx512={a}, ref={r}"
            );
        }
    }
}

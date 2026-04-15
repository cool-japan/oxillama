//! AVX2+FMA accelerated Q8_K quantization kernel.
//!
//! Q8_K block layout (292 bytes per 256 weights):
//! - bytes\[0..4\]    — f32 super-block scale `d` (little-endian)
//! - bytes\[4..260\]  — 256 × int8 signed quantized values (qs)
//! - bytes\[260..292\] — 16 × int16 block sums (bsums, unused here)
//!
//! Each weight reconstructs as `qs[i] × d`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::hsum_f32_avx;
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q8_K: 256 weights per block.
const BLOCK_SIZE: usize = 256;
/// Bytes per Q8_K block: 4 (f32 scale) + 256 (i8 data) + 32 (bsums).
const BLOCK_BYTES: usize = 292;
/// Offset of the quantized values within a block.
const QS_OFFSET: usize = 4;

/// AVX2+FMA accelerated Q8_K kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q8_KAvx2;

impl QuantKernel for Q8_KAvx2 {
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

        // SAFETY: block.len() >= 292 and output.len() >= 256 verified above.
        // CPU avx2+fma support guaranteed by KernelDispatcher.
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
            // SAFETY: row/block bounds verified above.
            // CPU avx2+fma support guaranteed by KernelDispatcher.
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

    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q8_K"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 292-byte Q8_K block to 256 FP32 values using AVX2.
///
/// Processes 256 i8 values in 32 groups of 8, using `_mm256_cvtepi8_epi32`
/// to widen i8→i32 in a single step, then convert to f32 and multiply by
/// the super-block scale.
///
/// # Safety
/// - `block.len() >= 292`
/// - `output.len() >= 256`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // Read f32 scale from bytes[0..4].
    // SAFETY: block.len() >= 292 >= 4.
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    let vd = _mm256_set1_ps(d);

    let qs_ptr = block.as_ptr().add(QS_OFFSET);
    let out_ptr = output.as_mut_ptr();

    // Process 256 values in 8 chunks of 32 i8s each.
    // Each chunk: load 256-bit (32 i8s), split into 4 groups of 8,
    // widen i8→i32→f32, scale, and store.
    for chunk in 0..8_usize {
        let base = chunk * 32;

        // SAFETY: qs_ptr + base..+32 is within block[4..260].
        let raw256 = _mm256_loadu_si256(qs_ptr.add(base).cast::<__m256i>());

        // Split into two 128-bit halves.
        let lo128 = _mm256_castsi256_si128(raw256);
        let hi128 = _mm256_extracti128_si256(raw256, 1);

        // Group A: i8[0..8] → i32 → f32 × d
        let a_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo128)), vd);
        // Group B: i8[8..16] → i32 → f32 × d
        let b_f32 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo128, 8))),
            vd,
        );
        // Group C: i8[16..24] → i32 → f32 × d
        let c_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi128)), vd);
        // Group D: i8[24..32] → i32 → f32 × d
        let d_f32 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi128, 8))),
            vd,
        );

        // Store 32 f32 values.
        // SAFETY: out_ptr + base..+32 is within output[0..256].
        _mm256_storeu_ps(out_ptr.add(base), a_f32);
        _mm256_storeu_ps(out_ptr.add(base + 8), b_f32);
        _mm256_storeu_ps(out_ptr.add(base + 16), c_f32);
        _mm256_storeu_ps(out_ptr.add(base + 24), d_f32);
    }
}

/// Compute dot product of one Q8_K row with an FP32 vector using AVX2+FMA.
///
/// For each block: load i8 quants, widen to f32, FMA with input, accumulate,
/// then multiply by the super-block scale.
///
/// # Safety
/// - `row_data.len() >= blocks_per_row * BLOCK_BYTES`
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

    for blk in 0..blocks_per_row {
        let block_offset = blk * BLOCK_BYTES;
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;

        // Read f32 scale.
        // SAFETY: block.len() == BLOCK_BYTES == 292 >= 4.
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);

        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: full block — process 256 values via AVX2.
            // 256 i8s processed in 32 groups of 8 (using _mm256_cvtepi8_epi32).
            let qs_ptr = block.as_ptr().add(QS_OFFSET);
            let inp_ptr = input.as_ptr().add(input_offset);

            let mut acc = _mm256_setzero_ps();

            // 8 chunks of 32 i8s, each chunk produces 4 FMA ops of 8 lanes.
            for chunk in 0..8_usize {
                let base = chunk * 32;

                // SAFETY: qs_ptr + base..+32 within block[4..260].
                let raw256 = _mm256_loadu_si256(qs_ptr.add(base).cast::<__m256i>());
                let lo128 = _mm256_castsi256_si128(raw256);
                let hi128 = _mm256_extracti128_si256(raw256, 1);

                // Group A: i8[0..8] × input[0..8]
                let wa = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo128));
                // SAFETY: inp_ptr + base..+8 within input.
                let ia = _mm256_loadu_ps(inp_ptr.add(base));
                acc = _mm256_fmadd_ps(wa, ia, acc);

                // Group B: i8[8..16] × input[8..16]
                let wb = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo128, 8)));
                let ib = _mm256_loadu_ps(inp_ptr.add(base + 8));
                acc = _mm256_fmadd_ps(wb, ib, acc);

                // Group C: i8[16..24] × input[16..24]
                let wc = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi128));
                let ic = _mm256_loadu_ps(inp_ptr.add(base + 16));
                acc = _mm256_fmadd_ps(wc, ic, acc);

                // Group D: i8[24..32] × input[24..32]
                let wd = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi128, 8)));
                let id = _mm256_loadu_ps(inp_ptr.add(base + 24));
                acc = _mm256_fmadd_ps(wd, id, acc);
            }

            row_sum += hsum_f32_avx(acc) * d;
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid OOB reads.
            let mut partial_sum = 0.0f32;
            for i in 0..remaining {
                // SAFETY: block[QS_OFFSET + i] valid because remaining <= BLOCK_SIZE == 256
                // and BLOCK_BYTES == 292 = 4 + 256 + 32.
                let q = *block.get_unchecked(QS_OFFSET + i) as i8;
                partial_sum += q as f32 * input[input_offset + i];
            }
            row_sum += partial_sum * d;
        }
        // remaining == 0: out of bounds, skip
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx2"))]
mod tests {
    use super::*;
    use crate::reference::q8_k::Q8KRef;

    fn make_q8_k_block(d: f32, qs: &[i8; 256]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&d.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        // bsums: 16 × int16 (32 bytes), zero for testing
        block.extend_from_slice(&[0u8; 32]);
        block
    }

    fn make_tensor(block_data: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(
            block_data,
            vec![1, n_cols],
            oxillama_gguf::GgufTensorType::Q8K,
        )
    }

    #[test]
    fn test_dequant_block_zeros() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let qs = [0i8; 256];
        let block = make_q8_k_block(1.5, &qs);
        let mut output = vec![0.0f32; 256];
        Q8_KAvx2
            .dequant_block(&block, &mut output)
            .expect("dequant failed");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-6, "expected 0.0 at index {i}, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_positive() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = (i % 127) as i8 + 1; // 1..127 repeating
        }
        let d = 0.25f32;
        let block = make_q8_k_block(d, &qs);
        let mut output = vec![0.0f32; 256];
        Q8_KAvx2
            .dequant_block(&block, &mut output)
            .expect("dequant failed");
        for (i, &v) in output.iter().enumerate() {
            let expected = d * qs[i] as f32;
            assert!(
                (v - expected).abs() < 1e-5,
                "mismatch at index {i}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = (i as i8).wrapping_sub(64);
        }
        let block = make_q8_k_block(0.5, &qs);

        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q8_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant failed");
        Q8KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant failed");

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
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i32) - 100) as i8;
        }
        let block = make_q8_k_block(0.25, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q8_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv failed");
        Q8KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv failed");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-2,
            "gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_multi_row() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 2 rows × 256 columns
        let mut qs1 = [0i8; 256];
        let mut qs2 = [0i8; 256];
        for (i, (q1, q2)) in qs1.iter_mut().zip(qs2.iter_mut()).enumerate() {
            *q1 = (i % 50) as i8;
            *q2 = -((i % 30) as i8);
        }
        let mut data = make_q8_k_block(0.5, &qs1);
        data.extend_from_slice(&make_q8_k_block(0.3, &qs2));

        let tensor_avx2 = QuantTensor::new(
            data.clone(),
            vec![2, 256],
            oxillama_gguf::GgufTensorType::Q8K,
        );
        let tensor_ref = QuantTensor::new(data, vec![2, 256], oxillama_gguf::GgufTensorType::Q8K);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.005).collect();
        let mut out_avx2 = vec![0.0f32; 2];
        let mut out_ref = vec![0.0f32; 2];

        Q8_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv failed");
        Q8KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv failed");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-2,
                "gemv row {i} mismatch: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_negative_values() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = (((i % 128) + 1) as i8).wrapping_neg(); // -1..-128 repeating
        }
        let d = 0.1f32;
        let block = make_q8_k_block(d, &qs);
        let mut output = vec![0.0f32; 256];
        Q8_KAvx2
            .dequant_block(&block, &mut output)
            .expect("dequant failed");
        for (i, &v) in output.iter().enumerate() {
            let expected = d * qs[i] as f32;
            assert!(
                (v - expected).abs() < 1e-5,
                "mismatch at index {i}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_buffer_too_small_errors() {
        let small_block = vec![0u8; 100];
        let mut output = vec![0.0f32; 256];
        assert!(Q8_KAvx2.dequant_block(&small_block, &mut output).is_err());

        let block = vec![0u8; BLOCK_BYTES];
        let mut small_output = vec![0.0f32; 100];
        assert!(Q8_KAvx2.dequant_block(&block, &mut small_output).is_err());
    }

    #[test]
    fn test_block_size_and_name() {
        assert_eq!(Q8_KAvx2.block_size(), 256);
        assert_eq!(Q8_KAvx2.block_bytes(), 292);
        assert_eq!(Q8_KAvx2.name(), "Q8_K");
    }
}

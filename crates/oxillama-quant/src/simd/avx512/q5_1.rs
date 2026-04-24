//! AVX-512 accelerated Q5_1 quantization kernel.
//!
//! Q5_1 block layout (24 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..4]   — FP16 minimum `m` (little-endian)
//! - bytes[4..8]   — `qh`: 32 high bits packed as u32 LE (bit i = high bit of weight i)
//! - bytes[8..24]  — `qs`: 32 × lower 4 bits packed (2 per byte)
//!
//! Weight formula: `output[i] = d * q5[i] + m`  where `q5[i] ∈ [0..31]` is unsigned.
//!
//! q5[i] = (qs[i].lo4 | ((qh >> i) & 1) << 4)  for i in 0..16
//! q5[i+16] = (qs[i].hi4 | ((qh >> (i+16)) & 1) << 4)  for i in 0..16
//!
//! Q5_1 is UNSIGNED — do NOT apply Q5_0's -16 bias.
//!
//! ## AVX-512 strategy
//!
//! The high-bit expansion uses `core::array::from_fn` to unpack qh bits into
//! per-lane `[u8; 16]` arrays, then loads those into `__m128i` registers.
//! Each of the two 16-element groups is widened to 16 × i32 with
//! `_mm512_cvtepu8_epi32`, converted to f32, and scaled: `d * q5 + m`.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q5_1: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q5_1 block: 2 (d) + 2 (m) + 4 (qh) + 16 (qs).
pub const BLOCK_BYTES: usize = 24;

/// AVX-512 accelerated Q5_1 kernel.
///
/// Requires the `avx512f` CPU feature.  The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q5_1Avx512;

impl QuantKernel for Q5_1Avx512 {
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

        // SAFETY: block.len() >= 24 and output.len() >= 32 verified above.
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
        "Q5_1"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 24-byte Q5_1 block to 32 FP32 values using AVX-512.
///
/// Q5_1 is unsigned ([0..31]): no -16 centering.  Affine formula: `d * q5 + m`.
///
/// # Safety
/// - `block.len() >= 24`
/// - `output.len() >= 32`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    // SAFETY: block.len() >= 24 >= 4 — guaranteed by caller.
    let d = f16_to_f32(block);
    let m = f16_to_f32(&block[2..]);
    let vd = _mm512_set1_ps(d);
    let vm = _mm512_set1_ps(m);

    // Read qh as a u32 little-endian.
    // SAFETY: block[4..8] valid (block.len() >= 24).
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);

    // Load 16 qs bytes.
    // SAFETY: block.ptr + 8 valid (block.len() >= 24).
    let qs = _mm_loadu_si128(block.as_ptr().add(8) as *const __m128i);

    let mask4 = _mm_set1_epi8(0x0F_u8 as i8);
    let lo_nib = _mm_and_si128(qs, mask4); // low nibbles (weights 0..15 lo part)
    let hi_nib = _mm_and_si128(_mm_srli_epi16(qs, 4), mask4); // high nibbles (weights 16..31 hi part)

    // Expand qh into per-element 5th bit (shifted to bit position 4).
    // lo group: bits 0..16 → weights 0..15
    // hi group: bits 16..32 → weights 16..31
    let qh_lo_bits: [u8; 16] = core::array::from_fn(|i| (((qh >> i) & 1) << 4) as u8);
    let qh_hi_bits: [u8; 16] = core::array::from_fn(|i| (((qh >> (i + 16)) & 1) << 4) as u8);

    // SAFETY: qh_lo_bits and qh_hi_bits are stack-allocated 16-byte arrays.
    let vqh_lo = _mm_loadu_si128(qh_lo_bits.as_ptr() as *const __m128i);
    let vqh_hi = _mm_loadu_si128(qh_hi_bits.as_ptr() as *const __m128i);

    // Combine: q5[i] = lo_nib[i] | qh_lo[i]  (bit 4 is the 5th bit)
    let q5_lo = _mm_or_si128(lo_nib, vqh_lo); // 5-bit unsigned, weights 0-15
    let q5_hi = _mm_or_si128(hi_nib, vqh_hi); // 5-bit unsigned, weights 16-31

    // Widen to 16 × i32 per group (AVX-512), convert to f32, then FMA: d * q5 + m.

    // Group A: weights 0-15
    // SAFETY: _mm512_cvtepu8_epi32 reads 16 bytes from q5_lo.
    let a_u32 = _mm512_cvtepu8_epi32(q5_lo);
    let a_f32 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a_u32), vd, vm);

    // Group B: weights 16-31
    let b_u32 = _mm512_cvtepu8_epi32(q5_hi);
    let b_f32 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(b_u32), vd, vm);

    // Store all 32 values.
    // SAFETY: output.len() >= 32 — guaranteed by caller.
    let ptr = output.as_mut_ptr();
    _mm512_storeu_ps(ptr, a_f32);
    _mm512_storeu_ps(ptr.add(16), b_f32);
}

/// Compute the dot product of one row of a Q5_1 matrix with an FP32 vector.
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
    let mut acc_wd = _mm512_setzero_ps(); // accumulates (q5 * d) * input
    let mut acc_m = _mm512_setzero_ps(); // accumulates m * input

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &row_data[bo..bo + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;

        // SAFETY: block.len() == BLOCK_BYTES == 24 >= 4.
        let d = f16_to_f32(block);
        let m = f16_to_f32(&block[2..]);
        let vd = _mm512_set1_ps(d);
        let vm = _mm512_set1_ps(m);

        // SAFETY: block[4..8] valid (BLOCK_BYTES == 24).
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);

        // SAFETY: block.ptr + 8 valid (BLOCK_BYTES == 24).
        let qs = _mm_loadu_si128(block.as_ptr().add(8) as *const __m128i);

        let mask4 = _mm_set1_epi8(0x0F_u8 as i8);
        let lo_nib = _mm_and_si128(qs, mask4);
        let hi_nib = _mm_and_si128(_mm_srli_epi16(qs, 4), mask4);

        let qh_lo_bits: [u8; 16] = core::array::from_fn(|i| (((qh >> i) & 1) << 4) as u8);
        let qh_hi_bits: [u8; 16] = core::array::from_fn(|i| (((qh >> (i + 16)) & 1) << 4) as u8);

        // SAFETY: qh_lo_bits and qh_hi_bits are valid 16-byte stack arrays.
        let vqh_lo = _mm_loadu_si128(qh_lo_bits.as_ptr() as *const __m128i);
        let vqh_hi = _mm_loadu_si128(qh_hi_bits.as_ptr() as *const __m128i);

        let q5_lo = _mm_or_si128(lo_nib, vqh_lo);
        let q5_hi = _mm_or_si128(hi_nib, vqh_hi);

        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 32 weights valid — 2 AVX-512 passes.
            // SAFETY: input_offset + 32 <= n_cols <= input.len().
            let inp_ptr = input.as_ptr().add(input_offset);

            // Group A (weights 0-15)
            let wa_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q5_lo));
            let ia = _mm512_loadu_ps(inp_ptr);
            acc_wd = _mm512_fmadd_ps(_mm512_mul_ps(wa_f32, vd), ia, acc_wd);
            acc_m = _mm512_fmadd_ps(vm, ia, acc_m);

            // Group B (weights 16-31)
            let wb_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q5_hi));
            let ib = _mm512_loadu_ps(inp_ptr.add(16));
            acc_wd = _mm512_fmadd_ps(_mm512_mul_ps(wb_f32, vd), ib, acc_wd);
            acc_m = _mm512_fmadd_ps(vm, ib, acc_m);
        } else if remaining > 0 {
            // Scalar tail: materialize q5 values then dot.
            let mut partial = [0.0f32; BLOCK_SIZE];
            let pp = partial.as_mut_ptr();
            _mm512_storeu_ps(
                pp,
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q5_lo)), vd, vm),
            );
            _mm512_storeu_ps(
                pp.add(16),
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q5_hi)), vd, vm),
            );

            let mut scalar_sum = 0.0f32;
            for j in 0..remaining {
                scalar_sum += partial[j] * input[input_offset + j];
            }
            acc_wd = _mm512_add_ps(acc_wd, _mm512_set1_ps(scalar_sum));
        }
        // remaining == 0: fully out of bounds, skip.
    }

    // Total = Σ (q5 * d * input) + Σ (m * input)
    hsum_f32_avx512(acc_wd) + hsum_f32_avx512(acc_m)
}

// ---------------------------------------------------------------------------
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
mod tests {
    use super::*;
    use crate::reference::q5_1::Q5_1Ref;

    fn make_block(d: f32, m: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q5_1)
    }

    fn avx512_available() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
    }

    #[test]
    fn test_dequant_zeros() {
        if !avx512_available() {
            return;
        }
        let block = make_block(0.0, 0.0, 0, &[0; 16]);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx512_out = vec![0.0f32; BLOCK_SIZE];
        Q5_1Ref.dequant_block(&block, &mut ref_out).unwrap();
        Q5_1Avx512.dequant_block(&block, &mut avx512_out).unwrap();
        for (i, (r, a)) in ref_out.iter().zip(avx512_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r} avx512={a}");
        }
    }

    #[test]
    fn test_dequant_max() {
        if !avx512_available() {
            return;
        }
        // qh = 0xFFFFFFFF, qs = 0xFF → q5=31; d=1.0, m=0.0 → weight=31.0
        let block = make_block(1.0, 0.0, 0xFFFF_FFFF, &[0xFF; 16]);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx512_out = vec![0.0f32; BLOCK_SIZE];
        Q5_1Ref.dequant_block(&block, &mut ref_out).unwrap();
        Q5_1Avx512.dequant_block(&block, &mut avx512_out).unwrap();
        for (i, (r, a)) in ref_out.iter().zip(avx512_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r} avx512={a}");
        }
    }

    #[test]
    fn test_dequant_alternating() {
        if !avx512_available() {
            return;
        }
        let qh: u32 = 0x5A5A_5A5A;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 9 + 3) & 0xFF) as u8;
        }
        let block = make_block(0.5, 0.25, qh, &qs);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx512_out = vec![0.0f32; BLOCK_SIZE];
        Q5_1Ref.dequant_block(&block, &mut ref_out).unwrap();
        Q5_1Avx512.dequant_block(&block, &mut avx512_out).unwrap();
        for (i, (r, a)) in ref_out.iter().zip(avx512_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r:.6} avx512={a:.6}");
        }
    }

    #[test]
    fn test_gemv_matches_reference() {
        if !avx512_available() {
            return;
        }
        let qh: u32 = 0x5A5A_5A5A;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 9 + 3) & 0xFF) as u8;
        }
        let block = make_block(0.5, 0.25, qh, &qs);
        let tensor_avx512 = make_tensor(block.clone(), 32);
        let tensor_ref = make_tensor(block, 32);

        let input: Vec<f32> = (0..BLOCK_SIZE).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let mut ref_out = vec![0.0f32; 1];
        let mut avx512_out = vec![0.0f32; 1];

        Q5_1Ref.gemv(&tensor_ref, &input, &mut ref_out).unwrap();
        Q5_1Avx512
            .gemv(&tensor_avx512, &input, &mut avx512_out)
            .unwrap();

        assert!(
            (ref_out[0] - avx512_out[0]).abs() < 1e-3,
            "gemv: ref={} avx512={}",
            ref_out[0],
            avx512_out[0]
        );
    }

    #[test]
    fn test_gemv_partial_block() {
        if !avx512_available() {
            return;
        }
        let block = make_block(0.5, 0.1, 0xAAAA_AAAA, &[0x55; 16]);
        let tensor_avx512 = make_tensor(block.clone(), 20);
        let tensor_ref = make_tensor(block, 20);

        let input = vec![1.0f32; 20];
        let mut ref_out = vec![0.0f32; 1];
        let mut avx512_out = vec![0.0f32; 1];

        Q5_1Ref.gemv(&tensor_ref, &input, &mut ref_out).unwrap();
        Q5_1Avx512
            .gemv(&tensor_avx512, &input, &mut avx512_out)
            .unwrap();

        assert!(
            (ref_out[0] - avx512_out[0]).abs() < 1e-3,
            "partial gemv: ref={} avx512={}",
            ref_out[0],
            avx512_out[0]
        );
    }

    #[test]
    fn test_empty_block_errors() {
        let mut out = vec![0.0f32; BLOCK_SIZE];
        assert!(Q5_1Avx512.dequant_block(&[], &mut out).is_err());
    }
}

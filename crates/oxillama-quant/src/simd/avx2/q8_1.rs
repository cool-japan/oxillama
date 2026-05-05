//! AVX2+FMA accelerated Q8_1 quantization kernel.
//!
//! Q8_1 block layout (36 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..4]   — FP16 sum `s` = d * Σqs (stored but unused in GEMV)
//! - bytes[4..36]  — 32 × int8 signed quantised values
//!
//! GEMV mirrors `reference::q8_1::Q8_1Ref::gemv` exactly:
//!   plain f32 activations × `d * qs[i]`, `s` is not used.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q8_1: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q8_1 block.
pub const BLOCK_BYTES: usize = 36;

/// AVX2+FMA accelerated Q8_1 kernel.
#[allow(non_camel_case_types)]
pub struct Q8_1Avx2;

impl QuantKernel for Q8_1Avx2 {
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
        "Q8_1"
    }

    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        use crate::error::QuantError;

        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let row_bytes = blocks_per_row * BLOCK_BYTES;
        let q8_block_bytes: usize = 34;

        if weights.len() < n_rows * row_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < blocks_per_row * q8_block_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: blocks_per_row * q8_block_bytes,
                available: acts_q8.len(),
            });
        }

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let partial = unsafe {
                fused_q8_1_q8_0_row_avx2(
                    &weights[row_start..row_start + row_bytes],
                    acts_q8,
                    blocks_per_row,
                    n_cols,
                )
            };
            *out_val += partial;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 36-byte Q8_1 block to 32 FP32 values using AVX2.
///
/// Mirrors reference: `d * qs[i]`. The `s` field at bytes[2..4] is unused.
///
/// # Safety
/// - `block.len() >= 36`
/// - `output.len() >= 32`
/// - CPU must support `avx2`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let vd = _mm256_set1_ps(d);

    // Load 32 int8 values from bytes[4..36]
    let q_raw = _mm256_loadu_si256(block.as_ptr().add(4) as *const __m256i);

    // Widen i8 → i16 → i32 → f32 in four groups of 8
    // Lower 128-bit lane (bytes 0..15 of q_raw = quant values 0..15)
    let q_lo128 = _mm256_castsi256_si128(q_raw);
    let q_lo_i16 = _mm256_cvtepi8_epi16(q_lo128);
    let q0_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_lo_i16));
    let q1_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_lo_i16, 1));

    // Upper 128-bit lane (quant values 16..31)
    let q_hi128 = _mm256_extracti128_si256(q_raw, 1);
    let q_hi_i16 = _mm256_cvtepi8_epi16(q_hi128);
    let q2_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_hi_i16));
    let q3_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_hi_i16, 1));

    let w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q0_i32), vd);
    let w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q1_i32), vd);
    let w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q2_i32), vd);
    let w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q3_i32), vd);

    let ptr = output.as_mut_ptr();
    _mm256_storeu_ps(ptr, w0);
    _mm256_storeu_ps(ptr.add(8), w1);
    _mm256_storeu_ps(ptr.add(16), w2);
    _mm256_storeu_ps(ptr.add(24), w3);
}

/// Compute the dot product of one row of a Q8_1 matrix with an FP32 vector.
///
/// Mirrors reference exactly: `d * Σ(qs[i] * inp[i])`. The `s` field is unused.
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
    let mut acc = _mm256_setzero_ps();

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &row_data[bo..bo + BLOCK_BYTES];
        let col_start = blk * BLOCK_SIZE;
        let col_end = (col_start + BLOCK_SIZE).min(n_cols);
        let inp = &input[col_start..col_end];

        let d = f16_to_f32(block);
        let vd = _mm256_set1_ps(d);

        // Load 32 int8 values from bytes[4..36]
        let q_raw = _mm256_loadu_si256(block.as_ptr().add(4) as *const __m256i);

        let q_lo128 = _mm256_castsi256_si128(q_raw);
        let q_lo_i16 = _mm256_cvtepi8_epi16(q_lo128);
        let q0_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_lo_i16));
        let q1_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_lo_i16, 1));

        let q_hi128 = _mm256_extracti128_si256(q_raw, 1);
        let q_hi_i16 = _mm256_cvtepi8_epi16(q_hi128);
        let q2_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_hi_i16));
        let q3_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_hi_i16, 1));

        let qf0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q0_i32), vd);
        let qf1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q1_i32), vd);
        let qf2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q2_i32), vd);
        let qf3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q3_i32), vd);

        let avail = inp.len();

        let n_a = avail.min(8);
        let mut inp_a = [0.0f32; 8];
        inp_a[..n_a].copy_from_slice(&inp[..n_a]);
        let vi0 = _mm256_loadu_ps(inp_a.as_ptr());
        acc = _mm256_fmadd_ps(qf0, vi0, acc);

        let n_b = avail.saturating_sub(8).min(8);
        let mut inp_b = [0.0f32; 8];
        if n_b > 0 {
            inp_b[..n_b].copy_from_slice(&inp[8..8 + n_b]);
        }
        let vi1 = _mm256_loadu_ps(inp_b.as_ptr());
        acc = _mm256_fmadd_ps(qf1, vi1, acc);

        let n_c = avail.saturating_sub(16).min(8);
        let mut inp_c = [0.0f32; 8];
        if n_c > 0 {
            inp_c[..n_c].copy_from_slice(&inp[16..16 + n_c]);
        }
        let vi2 = _mm256_loadu_ps(inp_c.as_ptr());
        acc = _mm256_fmadd_ps(qf2, vi2, acc);

        let n_d = avail.saturating_sub(24).min(8);
        let mut inp_d = [0.0f32; 8];
        if n_d > 0 {
            inp_d[..n_d].copy_from_slice(&inp[24..24 + n_d]);
        }
        let vi3 = _mm256_loadu_ps(inp_d.as_ptr());
        acc = _mm256_fmadd_ps(qf3, vi3, acc);
    }

    hsum_f32_avx(acc)
}

// ---------------------------------------------------------------------------
// Fused matvec: Q8_1 weights × Q8_0 activations
// ---------------------------------------------------------------------------

/// Fused dequant+dot for Q8_1 weights × Q8_0 activations.
///
/// Q8_1 layout: d (f16) at [0..2], s (f16, unused) at [2..4], qs (i8×32) at [4..36].
/// Formula: `d_w * Σ(qs_weight_i * d_a * q8_0_act_i)`
///
/// # Safety
/// - `weights_row.len() >= blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 34`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_q8_1_q8_0_row_avx2(
    weights_row: &[u8],
    acts_q8: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    const Q8_BLOCK_BYTES: usize = 34;
    let mut acc = _mm256_setzero_ps();

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &weights_row[bo..bo + BLOCK_BYTES];
        let col_start = blk * BLOCK_SIZE;

        // Q8_1 weight: scale at [0..2], qs (int8) at [4..36], s at [2..4] unused
        let d_w = f16_to_f32(block);
        let vd_w = _mm256_set1_ps(d_w);

        // Load 32 int8 weight quants from byte offset 4
        let w_raw = _mm256_loadu_si256(block.as_ptr().add(4) as *const __m256i);
        let w_lo128 = _mm256_castsi256_si128(w_raw);
        let w_hi128 = _mm256_extracti128_si256(w_raw, 1);
        let w_lo_i16 = _mm256_cvtepi8_epi16(w_lo128);
        let w_hi_i16 = _mm256_cvtepi8_epi16(w_hi128);
        let wq0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w_lo_i16)));
        let wq1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(w_lo_i16, 1)));
        let wq2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w_hi_i16)));
        let wq3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(w_hi_i16, 1)));
        // Scale by d_w
        let wf0 = _mm256_mul_ps(wq0, vd_w);
        let wf1 = _mm256_mul_ps(wq1, vd_w);
        let wf2 = _mm256_mul_ps(wq2, vd_w);
        let wf3 = _mm256_mul_ps(wq3, vd_w);

        // Decode Q8_0 activation block (34 bytes: 2 f16 scale + 32 i8)
        let ab = blk * Q8_BLOCK_BYTES;
        let a_block = &acts_q8[ab..ab + Q8_BLOCK_BYTES];
        let d_a = f16_to_f32(a_block);
        let vd_a = _mm256_set1_ps(d_a);

        let q8_raw = _mm256_loadu_si256(a_block.as_ptr().add(2) as *const __m256i);
        let q8_lo128 = _mm256_castsi256_si128(q8_raw);
        let q8_hi128 = _mm256_extracti128_si256(q8_raw, 1);
        let q8_lo_i16 = _mm256_cvtepi8_epi16(q8_lo128);
        let q8_hi_i16 = _mm256_cvtepi8_epi16(q8_hi128);
        let qa0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q8_lo_i16)));
        let qa1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(
            q8_lo_i16, 1,
        )));
        let qa2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q8_hi_i16)));
        let qa3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(
            q8_hi_i16, 1,
        )));
        let act0 = _mm256_mul_ps(qa0, vd_a);
        let act1 = _mm256_mul_ps(qa1, vd_a);
        let act2 = _mm256_mul_ps(qa2, vd_a);
        let act3 = _mm256_mul_ps(qa3, vd_a);

        let col_end = (col_start + BLOCK_SIZE).min(n_cols);
        let avail = col_end - col_start;

        // Group A: weight[0..8] × act[0..8]
        let n_a = avail.min(8);
        let mut act_a = [0.0f32; 8];
        _mm256_storeu_ps(act_a.as_mut_ptr(), act0);
        let mut buf_a = [0.0f32; 8];
        buf_a[..n_a].copy_from_slice(&act_a[..n_a]);
        acc = _mm256_fmadd_ps(wf0, _mm256_loadu_ps(buf_a.as_ptr()), acc);

        // Group B: weight[8..16] × act[8..16]
        let n_b = avail.saturating_sub(8).min(8);
        let mut act_b = [0.0f32; 8];
        _mm256_storeu_ps(act_b.as_mut_ptr(), act1);
        let mut buf_b = [0.0f32; 8];
        if n_b > 0 {
            buf_b[..n_b].copy_from_slice(&act_b[..n_b]);
        }
        acc = _mm256_fmadd_ps(wf1, _mm256_loadu_ps(buf_b.as_ptr()), acc);

        // Group C: weight[16..24] × act[16..24]
        let n_c = avail.saturating_sub(16).min(8);
        let mut act_c = [0.0f32; 8];
        _mm256_storeu_ps(act_c.as_mut_ptr(), act2);
        let mut buf_c = [0.0f32; 8];
        if n_c > 0 {
            buf_c[..n_c].copy_from_slice(&act_c[..n_c]);
        }
        acc = _mm256_fmadd_ps(wf2, _mm256_loadu_ps(buf_c.as_ptr()), acc);

        // Group D: weight[24..32] × act[24..32]
        let n_d = avail.saturating_sub(24).min(8);
        let mut act_d = [0.0f32; 8];
        _mm256_storeu_ps(act_d.as_mut_ptr(), act3);
        let mut buf_d = [0.0f32; 8];
        if n_d > 0 {
            buf_d[..n_d].copy_from_slice(&act_d[..n_d]);
        }
        acc = _mm256_fmadd_ps(wf3, _mm256_loadu_ps(buf_d.as_ptr()), acc);
    }

    hsum_f32_avx(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::reference::q8_1::Q8_1Ref;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    fn make_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        let s: f32 = d * qs.iter().map(|&q| q as f32).sum::<f32>();
        block.extend_from_slice(&half::f16::from_f32(s).to_bits().to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    fn avx2_available() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    #[test]
    fn dequant_matches_reference_zeros() {
        if !avx2_available() {
            return;
        }
        let block = make_block(0.0, &[0; 32]);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];
        Q8_1Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q8_1Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");
        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r} avx2={a}");
        }
    }

    #[test]
    fn dequant_matches_reference_mixed() {
        if !avx2_available() {
            return;
        }
        let mut qs = [0i8; 32];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i16 * 7 - 64).clamp(-128, 127)) as i8;
        }
        let block = make_block(0.5, &qs);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];
        Q8_1Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q8_1Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");
        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r:.6} avx2={a:.6}");
        }
    }

    #[test]
    fn empty_block_errors() {
        let mut out = vec![0.0f32; BLOCK_SIZE];
        let err = Q8_1Avx2.dequant_block(&[], &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn gemv_matches_reference() {
        if !avx2_available() {
            return;
        }
        let mut qs = [0i8; 32];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i16 * 7 - 64).clamp(-128, 127)) as i8;
        }
        let block = make_block(0.5, &qs);

        let tensor = QuantTensor {
            data: block.clone().into(),
            shape: vec![1, BLOCK_SIZE],
            tensor_type: oxillama_gguf::GgufTensorType::Q8_1,
        };

        let input: Vec<f32> = (0..BLOCK_SIZE).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let mut ref_out = vec![0.0f32; 1];
        let mut avx2_out = vec![0.0f32; 1];

        Q8_1Ref
            .gemv(&tensor, &input, &mut ref_out)
            .expect("ref gemv");
        Q8_1Avx2
            .gemv(&tensor, &input, &mut avx2_out)
            .expect("avx2 gemv");

        assert!(
            (ref_out[0] - avx2_out[0]).abs() < 1e-3,
            "gemv: ref={} avx2={}",
            ref_out[0],
            avx2_out[0]
        );
    }

    #[test]
    fn fused_q8_1_avx2_matches_reference() {
        if !avx2_available() {
            return;
        }

        let mut qs_w = [0i8; 32];
        for (i, q) in qs_w.iter_mut().enumerate() {
            *q = ((i as i16 * 7 - 64).clamp(-128, 127)) as i8;
        }
        let weight_block = make_block(0.5, &qs_w);

        // Build a Q8_0 activation block (34 bytes): 2-byte f16 scale + 32 i8 values
        let d_a = 0.25f32;
        let mut acts_raw = [0i8; 32];
        for (i, v) in acts_raw.iter_mut().enumerate() {
            *v = ((i as i16 * 5 - 40).clamp(-128, 127)) as i8;
        }
        let mut acts_block = Vec::with_capacity(34);
        acts_block.extend_from_slice(&half::f16::from_f32(d_a).to_bits().to_le_bytes());
        for &v in &acts_raw {
            acts_block.push(v as u8);
        }

        // Reference: dequant weight then dot with scaled activations
        let mut w_dequant = vec![0.0f32; BLOCK_SIZE];
        Q8_1Avx2
            .dequant_block(&weight_block, &mut w_dequant)
            .expect("ref dequant w");
        let acts_f32: Vec<f32> = acts_raw.iter().map(|&v| v as f32 * d_a).collect();
        let expected: f32 = w_dequant
            .iter()
            .zip(acts_f32.iter())
            .map(|(w, a)| w * a)
            .sum();

        // AVX2 fused (additive — start with 0)
        let mut out_avx2 = vec![0.0f32; 1];
        Q8_1Avx2
            .matvec_q8_fused(&weight_block, &acts_block, &mut out_avx2, 1, BLOCK_SIZE)
            .expect("avx2 fused");

        assert!(
            (out_avx2[0] - expected).abs() < 0.1,
            "fused_q8_1_avx2: got={} expected={}",
            out_avx2[0],
            expected
        );

        // Also compare against reference Q8_1Ref::matvec_q8_fused
        let mut out_ref = vec![0.0f32; 1];
        Q8_1Ref
            .matvec_q8_fused(&weight_block, &acts_block, &mut out_ref, 1, BLOCK_SIZE)
            .expect("ref fused");
        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.1,
            "fused: avx2={} ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }
}

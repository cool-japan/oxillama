//! oxiblas-backed GEMM kernels for F32, F16, and BF16 weight tensors.
//!
//! These kernels implement the [`crate::traits::QuantKernel`] trait for unquantized float
//! weight formats by delegating batched matrix multiplication to oxiblas.
//!
//! # Design
//!
//! oxiblas uses column-major storage internally.  The weight matrix (stored
//! row-major in the raw GGUF byte buffer) and the row-major input matrix are
//! each transposed into temporary column-major `Mat<f32>` views before calling
//! `oxiblas::gemm`.  The result is read back from the column-major output
//! matrix into the caller's row-major `output` slice.
//!
//! GEMV (single-vector case) is handled with a scalar dot-product loop to
//! avoid the overhead of two transposes for M = 1.
//!
//! Tolerances achieved vs. reference:
//! - F32 → 1e-6 (no precision loss)
//! - F16 → 1e-3  (limited by f16 precision)
//! - BF16 → 1e-2 (limited by bf16 precision)

use oxiblas::{gemm, Mat};

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Convert a row-major f32 slice (N×K) into a column-major `Mat<f32>` (N rows, K cols).
///
/// Column-major means element `(i, j)` lives at index `i + j * row_stride`
/// (where row_stride ≥ N for alignment).  The returned `Mat` is freshly
/// allocated; its `row_stride` is determined by oxiblas alignment rules.
fn row_major_to_colmaj_mat(data: &[f32], n_rows: usize, n_cols: usize) -> Mat<f32> {
    let mut mat = Mat::<f32>::zeros(n_rows, n_cols);
    let rs = mat.row_stride();
    let raw = unsafe { std::slice::from_raw_parts_mut(mat.as_mut_ptr(), rs * n_cols) };
    for r in 0..n_rows {
        for c in 0..n_cols {
            raw[r + c * rs] = data[r * n_cols + c];
        }
    }
    mat
}

// ---------------------------------------------------------------------------
// F32 kernel
// ---------------------------------------------------------------------------

/// oxiblas-backed F32 kernel.
///
/// Single-element blocks (1 weight = 4 bytes, raw f32 LE).
pub struct F32OxiblasKernel;

impl F32OxiblasKernel {
    /// Decode the entire weight matrix as a row-major f32 Vec (N × K).
    fn decode_weights(
        &self,
        quant_matrix: &QuantTensor,
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<Vec<f32>> {
        let needed = n_rows * n_cols * 4;
        if quant_matrix.data.len() < needed {
            return Err(QuantError::FloatGemmFailed(format!(
                "F32 weight buffer too small: need {needed} bytes, have {}",
                quant_matrix.data.len()
            )));
        }
        let mut w = Vec::with_capacity(n_rows * n_cols);
        for chunk in quant_matrix.data[..needed].chunks_exact(4) {
            w.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(w)
    }
}

impl QuantKernel for F32OxiblasKernel {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < 4 {
            return Err(QuantError::BufferTooSmall {
                needed: 4,
                available: block.len(),
            });
        }
        if output.is_empty() {
            return Err(QuantError::BufferTooSmall {
                needed: 1,
                available: 0,
            });
        }
        output[0] = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
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
        let w = self.decode_weights(quant_matrix, n_rows, n_cols)?;
        for (r, out_val) in output.iter_mut().enumerate().take(n_rows) {
            let mut sum = 0.0f32;
            for c in 0..n_cols {
                sum += w[r * n_cols + c] * input[c];
            }
            *out_val = sum;
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
        // Weight matrix: N rows × K cols  (n_rows=n, n_cols=k)
        let w = self.decode_weights(quant_matrix, n, k)?;
        if input.len() < m * k {
            return Err(QuantError::FloatGemmFailed(format!(
                "F32 gemm: input too small ({} < {})",
                input.len(),
                m * k
            )));
        }
        if output.len() < m * n {
            return Err(QuantError::FloatGemmFailed(format!(
                "F32 gemm: output too small ({} < {})",
                output.len(),
                m * n
            )));
        }

        // Build column-major oxiblas matrices.
        // W:   n×k weight matrix  →  A(n, k) in oxiblas colmaj
        // X^T: k×m input (input is m×k row-major, we need k×m colmaj = input itself colmaj)
        // Y:   n×m output colmaj

        let a = row_major_to_colmaj_mat(&w, n, k); // A: n×k
                                                   // input is m rows × k cols (row-major).  We need B: k×m (k rows, m cols) column-major.
                                                   // Column-major B(k, m): element (r, c) → r + c*rs.  From row-major input (m, k): input[c*k + r].
        let mut b = Mat::<f32>::zeros(k, m);
        {
            let rs_b = b.row_stride();
            let raw_b = unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr(), rs_b * m) };
            for r in 0..k {
                for c in 0..m {
                    raw_b[r + c * rs_b] = input[c * k + r];
                }
            }
        }
        let mut c = Mat::<f32>::zeros(n, m);

        gemm(1.0_f32, a.as_ref(), b.as_ref(), 0.0_f32, c.as_mut());

        // Read result: C is n×m colmaj → output is m×n row-major (caller expects m output rows, n cols)
        // Wait: the QuantKernel::gemm signature says output is [M x N] row-major where the
        // weight matrix has N rows (output cols).  So output[row_of_input][weight_row].
        // That means output[i][j] = Σ_k input[i][k] * W[j][k], which is (X · W^T)[i][j].
        // But we computed W · X^T = (n×k) × (k×m) → (n×m) colmaj.
        // C result: row=weight_row (j), col=input_row (i). So C(j, i) maps to output[i][j].
        {
            let rs_c = c.row_stride();
            let raw_c = unsafe { std::slice::from_raw_parts(c.as_ptr(), rs_c * m) };
            for i in 0..m {
                // input row
                for j in 0..n {
                    // weight row (output col)
                    output[i * n + j] = raw_c[j + i * rs_c];
                }
            }
        }
        Ok(())
    }

    fn block_size(&self) -> usize {
        1
    }
    fn block_bytes(&self) -> usize {
        4
    }
    fn name(&self) -> &'static str {
        "F32-oxiblas"
    }
}

// ---------------------------------------------------------------------------
// F16 kernel
// ---------------------------------------------------------------------------

/// oxiblas-backed F16 kernel.
///
/// Single-element blocks (1 weight = 2 bytes, raw f16 LE → converted to f32).
pub struct F16OxiblasKernel;

impl F16OxiblasKernel {
    fn decode_weights(
        &self,
        quant_matrix: &QuantTensor,
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<Vec<f32>> {
        let needed = n_rows * n_cols * 2;
        if quant_matrix.data.len() < needed {
            return Err(QuantError::FloatGemmFailed(format!(
                "F16 weight buffer too small: need {needed} bytes, have {}",
                quant_matrix.data.len()
            )));
        }
        let mut w = Vec::with_capacity(n_rows * n_cols);
        for chunk in quant_matrix.data[..needed].chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            w.push(half::f16::from_bits(bits).to_f32());
        }
        Ok(w)
    }
}

impl QuantKernel for F16OxiblasKernel {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < 2 {
            return Err(QuantError::BufferTooSmall {
                needed: 2,
                available: block.len(),
            });
        }
        if output.is_empty() {
            return Err(QuantError::BufferTooSmall {
                needed: 1,
                available: 0,
            });
        }
        let bits = u16::from_le_bytes([block[0], block[1]]);
        output[0] = half::f16::from_bits(bits).to_f32();
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
        let w = self.decode_weights(quant_matrix, n_rows, n_cols)?;
        for (r, out_val) in output.iter_mut().enumerate().take(n_rows) {
            let mut sum = 0.0f32;
            for c in 0..n_cols {
                sum += w[r * n_cols + c] * input[c];
            }
            *out_val = sum;
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
        let w = self.decode_weights(quant_matrix, n, k)?;
        if input.len() < m * k {
            return Err(QuantError::FloatGemmFailed(format!(
                "F16 gemm: input too small ({} < {})",
                input.len(),
                m * k
            )));
        }
        if output.len() < m * n {
            return Err(QuantError::FloatGemmFailed(format!(
                "F16 gemm: output too small ({} < {})",
                output.len(),
                m * n
            )));
        }

        let a = row_major_to_colmaj_mat(&w, n, k);
        let mut b = Mat::<f32>::zeros(k, m);
        {
            let rs_b = b.row_stride();
            let raw_b = unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr(), rs_b * m) };
            for r in 0..k {
                for c in 0..m {
                    raw_b[r + c * rs_b] = input[c * k + r];
                }
            }
        }
        let mut c = Mat::<f32>::zeros(n, m);
        gemm(1.0_f32, a.as_ref(), b.as_ref(), 0.0_f32, c.as_mut());
        {
            let rs_c = c.row_stride();
            let raw_c = unsafe { std::slice::from_raw_parts(c.as_ptr(), rs_c * m) };
            for i in 0..m {
                for j in 0..n {
                    output[i * n + j] = raw_c[j + i * rs_c];
                }
            }
        }
        Ok(())
    }

    fn block_size(&self) -> usize {
        1
    }
    fn block_bytes(&self) -> usize {
        2
    }
    fn name(&self) -> &'static str {
        "F16-oxiblas"
    }
}

// ---------------------------------------------------------------------------
// BF16 kernel
// ---------------------------------------------------------------------------

/// Convert a BF16 bit pattern to f32.
///
/// BF16 is the upper 16 bits of an IEEE 754 f32, so conversion is zero-cost:
/// shift the bits into the high half of a u32 and reinterpret.
#[inline]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// oxiblas-backed BF16 kernel.
///
/// Single-element blocks (1 weight = 2 bytes, raw bf16 LE → converted to f32).
pub struct Bf16OxiblasKernel;

impl Bf16OxiblasKernel {
    fn decode_weights(
        &self,
        quant_matrix: &QuantTensor,
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<Vec<f32>> {
        let needed = n_rows * n_cols * 2;
        if quant_matrix.data.len() < needed {
            return Err(QuantError::FloatGemmFailed(format!(
                "BF16 weight buffer too small: need {needed} bytes, have {}",
                quant_matrix.data.len()
            )));
        }
        let mut w = Vec::with_capacity(n_rows * n_cols);
        for chunk in quant_matrix.data[..needed].chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            w.push(bf16_bits_to_f32(bits));
        }
        Ok(w)
    }
}

impl QuantKernel for Bf16OxiblasKernel {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < 2 {
            return Err(QuantError::BufferTooSmall {
                needed: 2,
                available: block.len(),
            });
        }
        if output.is_empty() {
            return Err(QuantError::BufferTooSmall {
                needed: 1,
                available: 0,
            });
        }
        let bits = u16::from_le_bytes([block[0], block[1]]);
        output[0] = bf16_bits_to_f32(bits);
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
        let w = self.decode_weights(quant_matrix, n_rows, n_cols)?;
        for (r, out_val) in output.iter_mut().enumerate().take(n_rows) {
            let mut sum = 0.0f32;
            for c in 0..n_cols {
                sum += w[r * n_cols + c] * input[c];
            }
            *out_val = sum;
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
        let w = self.decode_weights(quant_matrix, n, k)?;
        if input.len() < m * k {
            return Err(QuantError::FloatGemmFailed(format!(
                "BF16 gemm: input too small ({} < {})",
                input.len(),
                m * k
            )));
        }
        if output.len() < m * n {
            return Err(QuantError::FloatGemmFailed(format!(
                "BF16 gemm: output too small ({} < {})",
                output.len(),
                m * n
            )));
        }

        let a = row_major_to_colmaj_mat(&w, n, k);
        let mut b = Mat::<f32>::zeros(k, m);
        {
            let rs_b = b.row_stride();
            let raw_b = unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr(), rs_b * m) };
            for r in 0..k {
                for c in 0..m {
                    raw_b[r + c * rs_b] = input[c * k + r];
                }
            }
        }
        let mut c = Mat::<f32>::zeros(n, m);
        gemm(1.0_f32, a.as_ref(), b.as_ref(), 0.0_f32, c.as_mut());
        {
            let rs_c = c.row_stride();
            let raw_c = unsafe { std::slice::from_raw_parts(c.as_ptr(), rs_c * m) };
            for i in 0..m {
                for j in 0..n {
                    output[i * n + j] = raw_c[j + i * rs_c];
                }
            }
        }
        Ok(())
    }

    fn block_size(&self) -> usize {
        1
    }
    fn block_bytes(&self) -> usize {
        2
    }
    fn name(&self) -> &'static str {
        "BF16-oxiblas"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{Bf16Ref, F16Ref, F32Ref};
    use crate::types::QuantTensor;
    use oxillama_gguf::GgufTensorType;

    // ------ helpers ------

    fn make_f32_tensor(data: &[f32], shape: Vec<usize>) -> QuantTensor {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        QuantTensor::new(bytes, shape, GgufTensorType::F32)
    }

    fn make_f16_tensor(data: &[f32], shape: Vec<usize>) -> QuantTensor {
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &v in data {
            bytes.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
        }
        QuantTensor::new(bytes, shape, GgufTensorType::F16)
    }

    fn make_bf16_tensor(data: &[f32], shape: Vec<usize>) -> QuantTensor {
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &v in data {
            let bf16_bits = (v.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bf16_bits.to_le_bytes());
        }
        QuantTensor::new(bytes, shape, GgufTensorType::Bf16)
    }

    // ------ F32 ------

    #[test]
    fn test_f32_gemv_matches_reference() {
        let vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3 matrix
        let tensor = make_f32_tensor(&vals, vec![2, 3]);
        let input = vec![1.0f32, 1.0, 1.0];

        let mut out_ref = vec![0.0f32; 2];
        F32Ref
            .gemv(&tensor, &input, &mut out_ref)
            .expect("ref gemv");

        let mut out_oxi = vec![0.0f32; 2];
        F32OxiblasKernel
            .gemv(&tensor, &input, &mut out_oxi)
            .expect("oxi gemv");

        for (r, o) in out_ref.iter().zip(out_oxi.iter()) {
            assert!((r - o).abs() < 1e-6, "F32 gemv mismatch: ref={r}, oxi={o}");
        }
    }

    #[test]
    fn test_f32_gemm_matches_reference() {
        // 3×4 weight matrix, 2 input rows (m=2, n=3, k=4)
        let w_vals: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let tensor = make_f32_tensor(&w_vals, vec![3, 4]);

        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect(); // 2×4
        let m = 2;
        let n = 3;
        let k = 4;

        let mut out_ref = vec![0.0f32; m * n];
        F32Ref
            .gemm(&tensor, &input, &mut out_ref, m, n, k)
            .expect("ref gemm");

        let mut out_oxi = vec![0.0f32; m * n];
        F32OxiblasKernel
            .gemm(&tensor, &input, &mut out_oxi, m, n, k)
            .expect("oxi gemm");

        for (i, (r, o)) in out_ref.iter().zip(out_oxi.iter()).enumerate() {
            assert!(
                (r - o).abs() < 1e-5,
                "F32 gemm mismatch at [{i}]: ref={r}, oxi={o}"
            );
        }
    }

    #[test]
    fn test_f32_gemm_identity() {
        // 2×2 identity weight, 2 input rows
        let w_vals = [1.0f32, 0.0, 0.0, 1.0];
        let tensor = make_f32_tensor(&w_vals, vec![2, 2]);
        let input = vec![3.0f32, 5.0, 7.0, 11.0]; // 2×2
        let mut output = vec![0.0f32; 4];
        F32OxiblasKernel
            .gemm(&tensor, &input, &mut output, 2, 2, 2)
            .expect("f32 identity gemm");
        assert!((output[0] - 3.0).abs() < 1e-5, "output[0]={}", output[0]);
        assert!((output[1] - 5.0).abs() < 1e-5, "output[1]={}", output[1]);
        assert!((output[2] - 7.0).abs() < 1e-5, "output[2]={}", output[2]);
        assert!((output[3] - 11.0).abs() < 1e-5, "output[3]={}", output[3]);
    }

    #[test]
    fn test_f32_kernel_metadata() {
        assert_eq!(F32OxiblasKernel.block_size(), 1);
        assert_eq!(F32OxiblasKernel.block_bytes(), 4);
        assert_eq!(F32OxiblasKernel.name(), "F32-oxiblas");
    }

    #[test]
    fn test_f32_dequant_block() {
        let val = 42.5f32; // exact in IEEE 754 f32
        let mut block = val.to_le_bytes().to_vec();
        block.extend_from_slice(&[0u8; 4]); // extra bytes
        let mut out = [0.0f32; 1];
        F32OxiblasKernel
            .dequant_block(&block, &mut out)
            .expect("dequant");
        assert!((out[0] - val).abs() < 1e-6);
    }

    // ------ F16 ------

    #[test]
    fn test_f16_gemv_matches_reference() {
        let vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = make_f16_tensor(&vals, vec![2, 3]);
        let input = vec![1.0f32, 1.0, 1.0];

        let mut out_ref = vec![0.0f32; 2];
        F16Ref
            .gemv(&tensor, &input, &mut out_ref)
            .expect("ref gemv");

        let mut out_oxi = vec![0.0f32; 2];
        F16OxiblasKernel
            .gemv(&tensor, &input, &mut out_oxi)
            .expect("oxi gemv");

        for (r, o) in out_ref.iter().zip(out_oxi.iter()) {
            assert!((r - o).abs() < 1e-3, "F16 gemv mismatch: ref={r}, oxi={o}");
        }
    }

    #[test]
    fn test_f16_gemm_matches_reference() {
        let w_vals: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let tensor = make_f16_tensor(&w_vals, vec![3, 4]);
        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let (m, n, k) = (2, 3, 4);

        let mut out_ref = vec![0.0f32; m * n];
        F16Ref
            .gemm(&tensor, &input, &mut out_ref, m, n, k)
            .expect("ref gemm");

        let mut out_oxi = vec![0.0f32; m * n];
        F16OxiblasKernel
            .gemm(&tensor, &input, &mut out_oxi, m, n, k)
            .expect("oxi gemm");

        for (i, (r, o)) in out_ref.iter().zip(out_oxi.iter()).enumerate() {
            assert!(
                (r - o).abs() < 1e-3,
                "F16 gemm mismatch at [{i}]: ref={r}, oxi={o}"
            );
        }
    }

    #[test]
    fn test_f16_kernel_metadata() {
        assert_eq!(F16OxiblasKernel.block_size(), 1);
        assert_eq!(F16OxiblasKernel.block_bytes(), 2);
        assert_eq!(F16OxiblasKernel.name(), "F16-oxiblas");
    }

    #[test]
    fn test_f16_dequant_block() {
        let val = 3.125f32; // exact in f16
        let bits = half::f16::from_f32(val).to_bits();
        let block = bits.to_le_bytes().to_vec();
        let mut out = [0.0f32; 1];
        F16OxiblasKernel
            .dequant_block(&block, &mut out)
            .expect("dequant");
        assert!(
            (out[0] - val).abs() < 1e-3,
            "expected ~{val}, got {}",
            out[0]
        );
    }

    // ------ BF16 ------

    #[test]
    fn test_bf16_gemv_matches_reference() {
        let vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = make_bf16_tensor(&vals, vec![2, 3]);
        let input = vec![1.0f32, 1.0, 1.0];

        let mut out_ref = vec![0.0f32; 2];
        Bf16Ref
            .gemv(&tensor, &input, &mut out_ref)
            .expect("ref gemv");

        let mut out_oxi = vec![0.0f32; 2];
        Bf16OxiblasKernel
            .gemv(&tensor, &input, &mut out_oxi)
            .expect("oxi gemv");

        for (r, o) in out_ref.iter().zip(out_oxi.iter()) {
            assert!((r - o).abs() < 1e-2, "BF16 gemv mismatch: ref={r}, oxi={o}");
        }
    }

    #[test]
    fn test_bf16_gemm_matches_reference() {
        let w_vals: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let tensor = make_bf16_tensor(&w_vals, vec![3, 4]);
        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let (m, n, k) = (2, 3, 4);

        let mut out_ref = vec![0.0f32; m * n];
        Bf16Ref
            .gemm(&tensor, &input, &mut out_ref, m, n, k)
            .expect("ref gemm");

        let mut out_oxi = vec![0.0f32; m * n];
        Bf16OxiblasKernel
            .gemm(&tensor, &input, &mut out_oxi, m, n, k)
            .expect("oxi gemm");

        for (i, (r, o)) in out_ref.iter().zip(out_oxi.iter()).enumerate() {
            assert!(
                (r - o).abs() < 1e-2,
                "BF16 gemm mismatch at [{i}]: ref={r}, oxi={o}"
            );
        }
    }

    #[test]
    fn test_bf16_kernel_metadata() {
        assert_eq!(Bf16OxiblasKernel.block_size(), 1);
        assert_eq!(Bf16OxiblasKernel.block_bytes(), 2);
        assert_eq!(Bf16OxiblasKernel.name(), "BF16-oxiblas");
    }

    #[test]
    fn test_bf16_dequant_block() {
        // 1.0 in BF16 = 0x3F80
        let block = 0x3F80u16.to_le_bytes().to_vec();
        let mut out = [0.0f32; 1];
        Bf16OxiblasKernel
            .dequant_block(&block, &mut out)
            .expect("dequant");
        assert!((out[0] - 1.0).abs() < 1e-2, "expected ~1.0, got {}", out[0]);
    }

    #[test]
    fn test_dispatch_routes_f32_to_oxiblas() {
        use crate::dispatch::KernelDispatcher;
        use oxillama_gguf::GgufTensorType;

        let dispatcher = KernelDispatcher::new();
        let kernel = dispatcher
            .get_kernel(GgufTensorType::F32)
            .expect("dispatch F32");
        assert_eq!(kernel.name(), "F32-oxiblas");
    }

    #[test]
    fn test_dispatch_routes_f16_to_oxiblas() {
        use crate::dispatch::KernelDispatcher;
        use oxillama_gguf::GgufTensorType;

        let dispatcher = KernelDispatcher::new();
        let kernel = dispatcher
            .get_kernel(GgufTensorType::F16)
            .expect("dispatch F16");
        assert_eq!(kernel.name(), "F16-oxiblas");
    }

    #[test]
    fn test_dispatch_routes_bf16_to_oxiblas() {
        use crate::dispatch::KernelDispatcher;
        use oxillama_gguf::GgufTensorType;

        let dispatcher = KernelDispatcher::new();
        let kernel = dispatcher
            .get_kernel(GgufTensorType::Bf16)
            .expect("dispatch BF16");
        assert_eq!(kernel.name(), "BF16-oxiblas");
    }
}

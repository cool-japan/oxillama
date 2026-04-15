//! Parallel (multi-threaded) wrappers for quantized matrix operations.
//!
//! When the `parallel` feature is enabled (default), uses rayon to parallelize
//! across output rows in GEMV/GEMM, which is the primary source of latency in
//! LLM inference. Each row's dot product is independent, making this
//! embarrassingly parallel.
//!
//! When `parallel` is disabled (e.g. for `wasm32-unknown-unknown`), falls back
//! to identical single-threaded logic so the same API compiles everywhere.
//!
//! The underlying single-row computation still uses the same scalar
//! (or SIMD) kernels — this module only adds thread-level parallelism.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::{QuantError, QuantResult};
use crate::types::QuantTensor;

/// Parallel GEMV: compute `output = quant_matrix @ input` using rayon.
///
/// Each output row is computed independently in parallel. This is the
/// primary optimization for autoregressive decode (single-token inference).
///
/// # Arguments
/// * `quant_matrix` - Quantized weight matrix [n_rows x n_cols].
/// * `input` - FP32 input vector of length n_cols.
/// * `output` - FP32 output vector of length n_rows (written in parallel).
/// * `block_size` - Number of weights per quantized block.
/// * `block_bytes` - Number of bytes per quantized block.
/// * `row_dot` - Function that computes the dot product for one row.
///   Signature: `fn(row_data: &[u8], input: &[f32], n_cols: usize) -> f32`
pub fn parallel_gemv<F>(
    quant_matrix: &QuantTensor,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
    block_bytes: usize,
    row_dot: F,
) -> QuantResult<()>
where
    F: Fn(&[u8], &[f32], usize) -> f32 + Send + Sync,
{
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

    let blocks_per_row = n_cols.div_ceil(block_size);
    let row_bytes = blocks_per_row * block_bytes;
    let data = &quant_matrix.data;

    #[cfg(feature = "parallel")]
    {
        output[..n_rows]
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, out)| {
                let row_start = row * row_bytes;
                let row_data = &data[row_start..row_start + row_bytes];
                *out = row_dot(row_data, input, n_cols);
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        output[..n_rows]
            .iter_mut()
            .enumerate()
            .for_each(|(row, out)| {
                let row_start = row * row_bytes;
                let row_data = &data[row_start..row_start + row_bytes];
                *out = row_dot(row_data, input, n_cols);
            });
    }

    Ok(())
}

/// Dimensions for a parallel GEMM operation.
pub struct GemmDims {
    /// Number of input rows (batch size).
    pub m: usize,
    /// Number of output columns (weight rows).
    pub n: usize,
    /// Shared inner dimension.
    pub k: usize,
    /// Number of weights per quantized block.
    pub block_size: usize,
    /// Number of bytes per quantized block.
    pub block_bytes: usize,
}

/// Parallel GEMM: compute `output = input_matrix @ quant_matrix^T` using rayon.
///
/// Parallelizes across rows of the input matrix (batch dimension).
/// Each input row's GEMV is independent.
pub fn parallel_gemm<F>(
    quant_matrix: &QuantTensor,
    input: &[f32],
    output: &mut [f32],
    dims: &GemmDims,
    row_dot: F,
) -> QuantResult<()>
where
    F: Fn(&[u8], &[f32], usize) -> f32 + Send + Sync,
{
    let blocks_per_row = dims.k.div_ceil(dims.block_size);
    let weight_row_bytes = blocks_per_row * dims.block_bytes;
    let data = &quant_matrix.data;

    // Parallelize across input batch rows
    #[cfg(feature = "parallel")]
    {
        output
            .par_chunks_mut(dims.n)
            .enumerate()
            .take(dims.m)
            .for_each(|(batch_row, out_row)| {
                let inp_row = &input[batch_row * dims.k..(batch_row + 1) * dims.k];
                for (weight_row, out) in out_row.iter_mut().enumerate().take(dims.n) {
                    let row_start = weight_row * weight_row_bytes;
                    let row_data = &data[row_start..row_start + weight_row_bytes];
                    *out = row_dot(row_data, inp_row, dims.k);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        output
            .chunks_mut(dims.n)
            .enumerate()
            .take(dims.m)
            .for_each(|(batch_row, out_row)| {
                let inp_row = &input[batch_row * dims.k..(batch_row + 1) * dims.k];
                for (weight_row, out) in out_row.iter_mut().enumerate().take(dims.n) {
                    let row_start = weight_row * weight_row_bytes;
                    let row_data = &data[row_start..row_start + weight_row_bytes];
                    *out = row_dot(row_data, inp_row, dims.k);
                }
            });
    }

    Ok(())
}

/// Minimum number of rows before engaging parallel execution.
/// For small matrices, thread overhead exceeds the benefit.
pub const PARALLEL_ROW_THRESHOLD: usize = 64;

/// Check whether parallel execution is worthwhile for the given dimensions.
///
/// Returns `true` if the matrix is large enough to benefit from parallelism.
pub fn should_parallelize(n_rows: usize, n_cols: usize) -> bool {
    // Heuristic: parallelize when total work exceeds threshold.
    // Each row does O(n_cols) work, so total is O(n_rows * n_cols).
    n_rows >= PARALLEL_ROW_THRESHOLD && n_cols >= 256
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::Q8_0Ref;
    use crate::traits::QuantKernel;

    fn make_q8_0_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        let d_bits = half::f16::from_f32(d).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    #[test]
    fn test_parallel_gemv_matches_sequential() {
        // Build a 4x32 matrix (4 rows, each 1 block of Q8_0)
        let n_rows = 4;
        let n_cols = 32;
        let mut data = Vec::new();
        for row in 0..n_rows {
            let mut qs = [0i8; 32];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((row as i16 * 7 + i as i16 * 3 - 48).clamp(-128, 127)) as i8;
            }
            data.extend_from_slice(&make_q8_0_block(0.5, &qs));
        }

        let tensor = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32 * 0.1) - 1.6).collect();

        // Sequential reference
        let kernel = Q8_0Ref;
        let mut seq_output = vec![0.0f32; n_rows];
        kernel.gemv(&tensor, &input, &mut seq_output).unwrap();

        // Parallel
        let mut par_output = vec![0.0f32; n_rows];
        parallel_gemv(
            &tensor,
            &input,
            &mut par_output,
            32,
            34,
            |row_data, inp, _n_cols| {
                // Q8_0 row dot: d * sum(qs[i] * inp[i])
                let d =
                    half::f16::from_bits(u16::from_le_bytes([row_data[0], row_data[1]])).to_f32();
                let qs = &row_data[2..34];
                let mut sum = 0.0f32;
                for (i, &q) in qs.iter().enumerate() {
                    sum += (q as i8) as f32 * inp[i];
                }
                d * sum
            },
        )
        .unwrap();

        for (i, (&s, &p)) in seq_output.iter().zip(par_output.iter()).enumerate() {
            assert!(
                (s - p).abs() < 1e-4,
                "row {i}: sequential={s}, parallel={p}"
            );
        }
    }

    #[test]
    fn test_should_parallelize() {
        assert!(!should_parallelize(1, 256));
        assert!(!should_parallelize(32, 256));
        assert!(should_parallelize(64, 256));
        assert!(should_parallelize(4096, 4096));
        assert!(!should_parallelize(128, 32));
    }

    #[test]
    fn test_parallel_gemv_input_too_small_errors() {
        let tensor = QuantTensor::new(
            make_q8_0_block(1.0, &[0i8; 32]),
            vec![1, 32],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let input = vec![0.0f32; 4]; // need 32
        let mut output = vec![0.0f32; 1];
        let result = parallel_gemv(&tensor, &input, &mut output, 32, 34, |_, _, _| 0.0);
        assert!(result.is_err(), "too-small input should error");
    }

    #[test]
    fn test_parallel_gemv_output_too_small_errors() {
        let tensor = QuantTensor::new(
            make_q8_0_block(1.0, &[0i8; 32]),
            vec![2, 32],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let input = vec![0.0f32; 32];
        let mut output = vec![0.0f32; 1]; // need 2
        let result = parallel_gemv(&tensor, &input, &mut output, 32, 34, |_, _, _| 0.0);
        assert!(result.is_err(), "too-small output should error");
    }

    #[test]
    fn test_parallel_gemm_basic() {
        // 2 weight rows of 32 cols Q8_0; 1 batch input row
        let n_rows = 2usize;
        let n_cols = 32usize;
        let mut data = Vec::new();
        for row in 0..n_rows {
            let mut qs = [0i8; 32];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((row as i16 + i as i16) % 10) as i8;
            }
            data.extend_from_slice(&make_q8_0_block(0.25, &qs));
        }
        let tensor = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let m = 1usize;
        let k = n_cols;
        let input = vec![1.0f32; k]; // batch of 1 input row
        let mut output = vec![0.0f32; m * n_rows]; // [m x n_rows]

        let dims = GemmDims {
            m,
            n: n_rows,
            k,
            block_size: 32,
            block_bytes: 34,
        };

        let result = parallel_gemm(&tensor, &input, &mut output, &dims, |row_data, inp, _nc| {
            let d = half::f16::from_bits(u16::from_le_bytes([row_data[0], row_data[1]])).to_f32();
            let qs = &row_data[2..34];
            let mut sum = 0.0f32;
            for (i, &q) in qs.iter().enumerate() {
                if i < inp.len() {
                    sum += (q as i8) as f32 * inp[i];
                }
            }
            d * sum
        });
        assert!(result.is_ok(), "parallel_gemm should succeed: {result:?}");
    }
}

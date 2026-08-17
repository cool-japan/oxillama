//! K-quant **encoders**: FP32 → Q2_K / Q3_K / Q4_K / Q5_K / Q6_K block bytes.
//!
//! The rest of this crate decodes K-quants; this module is the inverse, and it
//! is what makes `oxillama quantize --target Q4_K_M` (and friends) possible at
//! all. Each encoder is a literal port of the corresponding
//! `quantize_row_*_K_ref` in llama.cpp's `ggml/src/ggml-quants.c` (commit
//! `ba7e817ee`) and is held to **byte-for-byte** equality with the C original
//! by `tests/golden_kquant_encoders.rs`, whose expected values were produced by
//! compiling and running those C functions verbatim.
//!
//! ## Why "literal port" is the specification, not a style choice
//!
//! A K-quant super-block is not a simple `round(x / d)`. Q4_K and Q5_K run a
//! 21- (resp. 16-) step search over candidate scales, solving weighted normal
//! equations at each step and keeping the best; Q6_K runs a 19-step search
//! plus an RMSE refinement. Every one of those decisions is taken on `f32`
//! comparisons whose outcome flips with the last bit of the accumulator. Any
//! of the following "harmless" changes produces different — still plausible,
//! still round-trippable, but *not* llama.cpp — bytes:
//!
//! * using `f32::round()` instead of ggml's round-half-to-**even**
//!   `helpers::nearest_int`;
//! * clamping before narrowing to `u8` instead of after (Q4_K/Q5_K scales);
//! * accumulating in `f64`, or letting an FMA fuse `w * x * l`;
//! * recomputing the codes of a sub-block whose reconstructed scale is zero
//!   instead of leaving the first-pass codes in place.
//!
//! ## Row-length requirement
//!
//! K-quant rows must be a whole number of 256-weight super-blocks; ggml
//! asserts this and provides no ragged-tail path. The encoders here return
//! [`QuantError::RowNotBlockAligned`] rather than silently padding, because a
//! zero-padded tail would shift every subsequent row of the tensor. Callers
//! that need to quantize a tensor whose row length is not divisible by 256 do
//! what llama.cpp does and pick a different type for that tensor.

pub(crate) mod helpers;
mod q2_k;
mod q3_k;
mod q4_k;
mod q5_k;
mod q6_k;

use crate::error::{QuantError, QuantResult};

pub use q2_k::Q2_K_BLOCK_BYTES;
pub use q3_k::Q3_K_BLOCK_BYTES;
pub use q4_k::Q4_K_BLOCK_BYTES;
pub use q5_k::Q5_K_BLOCK_BYTES;
pub use q6_k::Q6_K_BLOCK_BYTES;

/// Weights per K-quant super-block (`QK_K` in ggml).
pub const QK_K: usize = 256;

/// Number of 32-weight sub-blocks in a super-block (Q4_K, Q5_K).
pub(crate) const SUB_BLOCKS_32: usize = QK_K / 32;

/// Number of 16-weight sub-blocks in a super-block (Q2_K, Q3_K, Q6_K).
pub(crate) const SUB_BLOCKS_16: usize = QK_K / 16;

/// Reject a row that is not a whole number of super-blocks.
fn check_row(data: &[f32], quant_type: &'static str) -> QuantResult<usize> {
    if !data.len().is_multiple_of(QK_K) {
        return Err(QuantError::RowNotBlockAligned {
            quant_type,
            block_size: QK_K,
            row_len: data.len(),
        });
    }
    Ok(data.len() / QK_K)
}

macro_rules! encoder_fn {
    ($name:ident, $encode:path, $bytes:expr, $label:literal, $doc:literal) => {
        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns [`QuantError::RowNotBlockAligned`] when `data.len()` is not
        /// a multiple of 256.
        pub fn $name(data: &[f32]) -> QuantResult<Vec<u8>> {
            let n_blocks = check_row(data, $label)?;
            let mut out = Vec::with_capacity(n_blocks * $bytes);
            for b in 0..n_blocks {
                $encode(&data[b * QK_K..(b + 1) * QK_K], &mut out);
            }
            debug_assert_eq!(out.len(), n_blocks * $bytes);
            Ok(out)
        }
    };
}

encoder_fn!(
    quantize_f32_to_q2_k,
    q2_k::encode_q2_k_block,
    Q2_K_BLOCK_BYTES,
    "Q2_K",
    "Quantize FP32 values to Q2_K (84 bytes per 256 weights)."
);
encoder_fn!(
    quantize_f32_to_q3_k,
    q3_k::encode_q3_k_block,
    Q3_K_BLOCK_BYTES,
    "Q3_K",
    "Quantize FP32 values to Q3_K (110 bytes per 256 weights)."
);
encoder_fn!(
    quantize_f32_to_q4_k,
    q4_k::encode_q4_k_block,
    Q4_K_BLOCK_BYTES,
    "Q4_K",
    "Quantize FP32 values to Q4_K (144 bytes per 256 weights)."
);
encoder_fn!(
    quantize_f32_to_q5_k,
    q5_k::encode_q5_k_block,
    Q5_K_BLOCK_BYTES,
    "Q5_K",
    "Quantize FP32 values to Q5_K (176 bytes per 256 weights)."
);
encoder_fn!(
    quantize_f32_to_q6_k,
    q6_k::encode_q6_k_block,
    Q6_K_BLOCK_BYTES,
    "Q6_K",
    "Quantize FP32 values to Q6_K (210 bytes per 256 weights)."
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every encoder must refuse a row that is not a whole number of 256-weight
    /// super-blocks — ggml asserts the same precondition and has no ragged path.
    #[test]
    fn partial_superblock_is_rejected() {
        for len in [1usize, 31, 32, 33, 255, 257, 511] {
            let data = vec![0.25f32; len];
            for (name, res) in [
                ("Q2_K", quantize_f32_to_q2_k(&data)),
                ("Q3_K", quantize_f32_to_q3_k(&data)),
                ("Q4_K", quantize_f32_to_q4_k(&data)),
                ("Q5_K", quantize_f32_to_q5_k(&data)),
                ("Q6_K", quantize_f32_to_q6_k(&data)),
            ] {
                match res {
                    Err(QuantError::RowNotBlockAligned {
                        quant_type,
                        block_size,
                        row_len,
                    }) => {
                        assert_eq!(quant_type, name);
                        assert_eq!(block_size, 256);
                        assert_eq!(row_len, len);
                    }
                    other => panic!("{name} len={len}: expected RowNotBlockAligned, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let empty: Vec<f32> = Vec::new();
        assert!(quantize_f32_to_q4_k(&empty).expect("q4_k").is_empty());
        assert!(quantize_f32_to_q5_k(&empty).expect("q5_k").is_empty());
        assert!(quantize_f32_to_q6_k(&empty).expect("q6_k").is_empty());
        assert!(quantize_f32_to_q2_k(&empty).expect("q2_k").is_empty());
        assert!(quantize_f32_to_q3_k(&empty).expect("q3_k").is_empty());
    }

    #[test]
    fn output_sizes_match_block_bytes() {
        let data: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) / 256.0).collect();
        assert_eq!(quantize_f32_to_q2_k(&data).expect("q2_k").len(), 2 * 84);
        assert_eq!(quantize_f32_to_q3_k(&data).expect("q3_k").len(), 2 * 110);
        assert_eq!(quantize_f32_to_q4_k(&data).expect("q4_k").len(), 2 * 144);
        assert_eq!(quantize_f32_to_q5_k(&data).expect("q5_k").len(), 2 * 176);
        assert_eq!(quantize_f32_to_q6_k(&data).expect("q6_k").len(), 2 * 210);
    }

    /// An all-zero super-block hits ggml's `memset(&y[i], 0, sizeof(block))`
    /// short-circuit in Q6_K.
    #[test]
    fn q6_k_all_zero_block_is_all_zero_bytes() {
        let data = vec![0.0f32; 256];
        let out = quantize_f32_to_q6_k(&data).expect("q6_k");
        assert_eq!(out.len(), 210);
        assert!(out.iter().all(|&b| b == 0), "expected 210 zero bytes");
    }
}

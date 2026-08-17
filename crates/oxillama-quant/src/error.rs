//! Error types for quantization operations.

use thiserror::Error;

/// Result type alias for quantization operations.
pub type QuantResult<T> = Result<T, QuantError>;

/// Errors that can occur during quantization kernel operations.
#[derive(Error, Debug)]
pub enum QuantError {
    /// The requested quantization type is not yet implemented.
    #[error("unsupported quantization type: {quant_type}")]
    UnsupportedType {
        /// Display name of the unsupported type.
        quant_type: String,
    },

    /// Block size does not match expected value for the quantization type.
    #[error("block size mismatch: expected {expected}, got {got}")]
    BlockSizeMismatch {
        /// Expected block size.
        expected: usize,
        /// Actual block size.
        got: usize,
    },

    /// Matrix/vector dimension mismatch in a kernel operation.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
    },

    /// A low-level kernel computation error.
    #[error("kernel error: {message}")]
    KernelError {
        /// Description of the kernel error.
        message: String,
    },

    /// Block count does not match the expected value for the tensor dimensions.
    #[error("block count mismatch: expected {expected} blocks, got {got}")]
    BlockCountMismatch {
        /// Expected number of blocks.
        expected: usize,
        /// Actual number of blocks.
        got: usize,
    },

    /// Data buffer is too small for the operation.
    #[error("buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall {
        /// Required buffer size in bytes.
        needed: usize,
        /// Available buffer size in bytes.
        available: usize,
    },

    /// Internal error (e.g., lock poisoning).
    #[error("internal error: {message}")]
    Internal {
        /// Description of the internal error.
        message: String,
    },

    /// Float GEMM (F16/BF16/F32) computation failure.
    #[error("float GEMM failed: {0}")]
    FloatGemmFailed(String),

    /// A row handed to an encoder is not a whole number of blocks.
    ///
    /// ggml's `quantize_row_*_ref` functions all begin with
    /// `assert(k % QK == 0)`, so a K-quant row must be a multiple of 256 and a
    /// legacy row a multiple of 32. This is *not* something a caller can pad
    /// its way out of: padding the flattened tensor only pads the last row,
    /// and a quantized tensor's rows must each start on a block boundary for
    /// the decoder to find them. Model-level callers handle the situation the
    /// way llama.cpp does — by falling back to a type whose block size does
    /// divide the row (see `oxillama-cli`'s `fallback_for_incompatible_row`).
    #[error("row length {row_len} is not a multiple of the {quant_type} block size {block_size}")]
    RowNotBlockAligned {
        /// Display name of the target quantization type.
        quant_type: &'static str,
        /// Required block size in weights.
        block_size: usize,
        /// The offending row length in weights.
        row_len: usize,
    },
}

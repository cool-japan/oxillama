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
}

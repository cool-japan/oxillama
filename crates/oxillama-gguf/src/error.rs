//! Error types for the GGUF parser.

use thiserror::Error;

/// Result type alias for GGUF operations.
pub type GgufResult<T> = Result<T, GgufError>;

/// Errors that can occur during GGUF parsing and tensor loading.
#[derive(Error, Debug)]
pub enum GgufError {
    /// Invalid GGUF magic number in file header.
    #[error("invalid GGUF magic number: expected 0x46475547, got 0x{magic:08X}")]
    InvalidMagic {
        /// The magic number found in the file.
        magic: u32,
    },

    /// Unsupported GGUF format version.
    #[error("unsupported GGUF version: {version} (supported: 1, 2, 3)")]
    UnsupportedVersion {
        /// The version number found in the file.
        version: u32,
    },

    /// Invalid or missing metadata entry.
    #[error("invalid metadata for key '{key}': {reason}")]
    InvalidMetadata {
        /// The metadata key that caused the error.
        key: String,
        /// Description of what went wrong.
        reason: String,
    },

    /// A required tensor was not found in the model file.
    #[error("tensor not found: '{name}'")]
    TensorNotFound {
        /// The name of the missing tensor.
        name: String,
    },

    /// Unsupported quantization type encountered.
    #[error("unsupported quantization type: id={type_id}")]
    UnsupportedQuantType {
        /// The numeric type ID from the GGUF file.
        type_id: u32,
    },

    /// Memory mapping failed.
    #[error("memory mapping failed: {0}")]
    MmapError(#[from] std::io::Error),

    /// Unexpected end of file during parsing.
    #[error("unexpected end of file at offset {offset}")]
    UnexpectedEof {
        /// The byte offset where the EOF occurred.
        offset: u64,
    },

    /// Data alignment error.
    #[error("alignment error: expected {expected}-byte alignment at offset {offset}")]
    AlignmentError {
        /// Expected alignment in bytes.
        expected: usize,
        /// The actual byte offset.
        offset: u64,
    },

    /// Invalid string encoding in GGUF data.
    #[error("invalid UTF-8 string at offset {offset}: {source}")]
    InvalidString {
        /// Byte offset of the invalid string.
        offset: u64,
        /// The underlying UTF-8 error.
        source: std::string::FromUtf8Error,
    },

    /// Error during GGUF write operations.
    #[error("write error: {reason}")]
    WriteError {
        /// Description of what went wrong.
        reason: String,
    },

    /// Integrity validation error.
    #[error("integrity error for tensor '{tensor_name}': {reason}")]
    IntegrityError {
        /// Name of the tensor that failed validation.
        tensor_name: String,
        /// Description of what went wrong.
        reason: String,
    },
}

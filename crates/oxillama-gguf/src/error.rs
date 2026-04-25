//! Error types for the GGUF parser.

#[cfg(not(feature = "std"))]
use alloc::string::{FromUtf8Error, String};
#[cfg(feature = "std")]
use std::string::FromUtf8Error;

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

    /// I/O or memory mapping error (std environments only).
    #[cfg(feature = "std")]
    #[error("I/O error: {0}")]
    MmapError(#[from] std::io::Error),

    /// I/O error representation for no_std environments.
    #[cfg(not(feature = "std"))]
    #[error("I/O error: {0}")]
    MmapError(alloc::borrow::Cow<'static, str>),

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
        source: FromUtf8Error,
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

    /// Tensor hash mismatch during validation.
    #[error("tensor hash mismatch for '{name}': expected {expected}, got {actual}")]
    HashMismatch {
        /// Name of the tensor.
        name: String,
        /// Expected hash (hex-encoded).
        expected: String,
        /// Actual hash (hex-encoded).
        actual: String,
    },

    /// Resume checkpoint mismatch (file changed since checkpoint was saved).
    #[error("resume mismatch: {detail} (expected={expected}, found={found})")]
    ResumeMismatch {
        /// Human-readable description of the field that mismatched.
        detail: String,
        /// Expected value as a string.
        expected: String,
        /// Actual found value as a string.
        found: String,
    },

    /// Shard header or metadata inconsistency across shards.
    #[error("shard mismatch: {detail}")]
    ShardMismatch {
        /// Description of the inconsistency.
        detail: String,
    },

    /// A tensor name appears in more than one shard.
    #[error("duplicate tensor '{name}' across shards")]
    ShardDuplicateTensor {
        /// Name of the duplicated tensor.
        name: String,
    },

    /// Attempt to quantize a tensor that is already in a quantized format.
    #[error("cannot requantize '{name}': already {existing}")]
    CannotRequantize {
        /// Name of the tensor.
        name: String,
        /// Existing quantization format description.
        existing: String,
    },
}

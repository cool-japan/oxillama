//! Error types for model architecture operations.

use thiserror::Error;

/// Result type alias for architecture operations.
pub type ArchResult<T> = Result<T, ArchError>;

/// Errors that can occur during model architecture operations.
#[derive(Error, Debug)]
pub enum ArchError {
    /// Unknown or unsupported model architecture.
    #[error("unknown architecture: '{arch_id}'")]
    UnknownArchitecture {
        /// The architecture identifier from GGUF metadata.
        arch_id: String,
    },

    /// Model configuration parameter mismatch.
    #[error("config mismatch for '{param}': expected {expected}, got {got}")]
    ConfigMismatch {
        /// Name of the configuration parameter.
        param: String,
        /// Expected value.
        expected: String,
        /// Actual value found.
        got: String,
    },

    /// Tensor shape does not match expected dimensions.
    #[error("tensor shape mismatch for '{tensor}': expected {expected:?}, got {got:?}")]
    TensorShapeMismatch {
        /// Tensor name.
        tensor: String,
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Error during forward pass computation.
    #[error("forward pass error at layer {layer}: {message}")]
    ForwardPassError {
        /// Layer index where the error occurred.
        layer: usize,
        /// Description of the error.
        message: String,
    },

    /// Missing required tensor in the model.
    #[error("missing required tensor: '{name}'")]
    MissingTensor {
        /// Name of the missing tensor.
        name: String,
    },

    /// Error propagated from GGUF parsing.
    #[error("GGUF error: {0}")]
    Gguf(#[from] oxillama_gguf::GgufError),

    /// Error propagated from quantization kernel.
    #[error("quantization error: {0}")]
    Quant(#[from] oxillama_quant::QuantError),

    /// Operation not supported by this architecture.
    #[error("operation not supported: {detail}")]
    NotSupported {
        /// Description of what is not supported.
        detail: String,
    },

    /// Tensor or weight buffer has an unexpected shape.
    ///
    /// Used by MoE and other layers that validate buffer sizes at runtime.
    #[error("invalid shape for '{name}': expected {expected:?}, got {got:?}")]
    InvalidShape {
        /// Descriptive name of the buffer or weight.
        name: String,
        /// Expected shape (as a dimension list).
        expected: Vec<usize>,
        /// Actual dimensions (as a dimension list, may be a flat length).
        got: Vec<usize>,
    },

    /// An invalid configuration parameter was detected.
    #[error("invalid configuration: {detail}")]
    InvalidConfig {
        /// Human-readable explanation.
        detail: String,
    },

    /// Unsupported operation for this architecture (grammar/batching contexts).
    #[error("unsupported operation: {message}")]
    UnsupportedOperation {
        /// Description of the unsupported operation.
        message: String,
    },

    /// A LoRA adapter is incompatible with the loaded model.
    ///
    /// Returned by `with_lora_stack()` when rank, dimension, or architecture
    /// constraints are violated.
    #[error("LoRA adapter incompatible: {detail}")]
    LoraIncompatible {
        /// Human-readable explanation.
        detail: String,
    },
}

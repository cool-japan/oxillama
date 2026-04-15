//! Error types for the GPU compute backend.

use thiserror::Error;

/// Errors that can occur in the GPU compute backend.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No GPU adapter was found (headless CI, no GPU hardware, etc.).
    #[error("No GPU adapter available")]
    NoAdapter,

    /// The GPU device request failed.
    #[error("GPU device request failed: {0}")]
    DeviceRequest(String),

    /// A buffer's actual size did not match the expected size.
    #[error("Buffer size mismatch: expected {expected}, got {got}")]
    BufferSize { expected: usize, got: usize },

    /// The requested quantization type has no GPU kernel implementation.
    #[error("Unsupported quant type for GPU: {name}")]
    UnsupportedType { name: String },

    /// Shader compilation or pipeline creation failed.
    #[error("Shader compilation failed: {detail}")]
    ShaderCompilation { detail: String },

    /// A GPU readback or mapping operation failed.
    #[error("GPU buffer mapping failed: {detail}")]
    BufferMap { detail: String },
}

/// Convenience `Result` alias for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

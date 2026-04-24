//! Error types for the inference runtime.

use thiserror::Error;

/// Result type alias for runtime operations.
pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// Errors that can occur during inference.
#[derive(Error, Debug)]
pub enum RuntimeError {
    /// No model has been loaded yet.
    #[error("no model loaded")]
    ModelNotLoaded,

    /// Tokenizer is not available because neither `tokenizer-wasm` nor `tokenizer-onig`
    /// feature is enabled.
    ///
    /// Enable the `tokenizer-wasm` feature (default, pure Rust) to use
    /// the HuggingFace tokenizers library.
    #[error("tokenizer not available: rebuild with the `tokenizer-wasm` feature enabled")]
    TokenizerNotAvailable,

    /// Tokenizer initialization or encoding/decoding failed.
    #[error("tokenizer error: {message}")]
    TokenizerError {
        /// Description of the tokenizer error.
        message: String,
    },

    /// Sampling operation failed.
    #[error("sampling error: {message}")]
    SamplingError {
        /// Description of the sampling error.
        message: String,
    },

    /// KV cache has reached its maximum capacity.
    #[error("KV cache full: maximum context length {max_ctx} reached")]
    KvCacheFull {
        /// Maximum context length supported.
        max_ctx: usize,
    },

    /// Model file could not be loaded.
    #[error("model loading error: {message}")]
    ModelLoadError {
        /// Description of the loading error.
        message: String,
    },

    /// Generation was interrupted or cancelled.
    #[error("generation cancelled")]
    Cancelled,

    /// Error propagated from architecture layer.
    #[error("architecture error: {0}")]
    Arch(#[from] oxillama_arch::ArchError),

    /// Error propagated from GGUF parser.
    #[error("GGUF error: {0}")]
    Gguf(#[from] oxillama_gguf::GgufError),

    /// Error propagated from quantization kernel.
    #[error("quantization error: {0}")]
    Quant(#[from] oxillama_quant::QuantError),

    /// I/O error during model loading.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Grammar error from GBNF constrained sampling.
    #[error("grammar error: {0}")]
    Grammar(#[from] crate::sampling::grammar::GrammarError),

    /// Attention computation error.
    #[error("attention error: {message}")]
    AttentionError {
        /// Description of the attention error.
        message: String,
    },

    /// Snapshot format version is incompatible with this runtime.
    #[error("snapshot incompatible: {detail}")]
    SnapshotIncompatible {
        /// Details about the incompatibility.
        detail: String,
    },

    /// Model fingerprint in snapshot does not match the file on disk.
    #[error("model fingerprint mismatch: expected={expected}, found={found}, detail={detail}")]
    ModelFingerprintMismatch {
        /// The fingerprint expected (from snapshot).
        expected: String,
        /// The fingerprint found (computed from disk).
        found: String,
        /// Additional detail about the mismatch.
        detail: String,
    },

    /// Offload pager read past end of backing store.
    #[error("offload: unexpected EOF at offset {offset}, needed {needed} bytes, {available} available")]
    OffloadEof {
        /// Byte offset at which the read was attempted.
        offset: u64,
        /// Number of bytes requested.
        needed: usize,
        /// Number of bytes available from `offset` to end.
        available: usize,
    },

    /// A tensor name was not found in the weight offset map.
    #[error("tensor not found in weight map: {0}")]
    TensorNotFound(String),

    /// An internal RwLock or Mutex was poisoned.
    #[error("lock poisoned")]
    LockPoisoned,
}

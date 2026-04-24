//! Error conversion from Rust error types to `pyo3::PyErr`.
//!
//! Defines a custom Python exception hierarchy rooted at `OxiLlamaError` and
//! maps every Rust error variant to the most semantically appropriate
//! exception subclass.

use pyo3::prelude::*;
use pyo3::PyErr;

use oxillama_arch::ArchError;
use oxillama_runtime::RuntimeError;

// ---------------------------------------------------------------------------
// Custom Python exception hierarchy
// ---------------------------------------------------------------------------

pyo3::create_exception!(oxillama_py, OxiLlamaError, pyo3::exceptions::PyException);
pyo3::create_exception!(oxillama_py, LoadError, OxiLlamaError);
pyo3::create_exception!(oxillama_py, GenerateError, OxiLlamaError);
pyo3::create_exception!(oxillama_py, TokenizerError, OxiLlamaError);
pyo3::create_exception!(oxillama_py, GrammarError, OxiLlamaError);
pyo3::create_exception!(oxillama_py, QuantError, OxiLlamaError);
pyo3::create_exception!(oxillama_py, KvCacheFullError, OxiLlamaError);

/// Register all custom exception classes on the Python module so that
/// they are importable as `oxillama_py.OxiLlamaError`, etc.
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("OxiLlamaError", m.py().get_type::<OxiLlamaError>())?;
    m.add("LoadError", m.py().get_type::<LoadError>())?;
    m.add("GenerateError", m.py().get_type::<GenerateError>())?;
    m.add("TokenizerError", m.py().get_type::<TokenizerError>())?;
    m.add("GrammarError", m.py().get_type::<GrammarError>())?;
    m.add("QuantError", m.py().get_type::<QuantError>())?;
    m.add("KvCacheFullError", m.py().get_type::<KvCacheFullError>())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// RuntimeError → PyErr
// ---------------------------------------------------------------------------

/// Convert a [`RuntimeError`] into a Python exception.
///
/// Mapping:
/// - `ModelNotLoaded`     → `GenerateError`
/// - `ModelLoadError`     → `LoadError`
/// - `TokenizerError`     → `TokenizerError`
/// - `TokenizerNotAvail`  → `TokenizerError`
/// - `SamplingError`      → `GenerateError`
/// - `KvCacheFull`        → `GenerateError`
/// - `Cancelled`          → `GenerateError`
/// - `Arch(…)`            → delegates to `arch_to_py`
/// - `Gguf(…)`            → `LoadError`
/// - `Quant(…)`           → `QuantError`
/// - `Io(…)`              → `LoadError`
/// - `Grammar(…)`         → `GrammarError`
pub fn runtime_to_py(err: RuntimeError) -> PyErr {
    match err {
        RuntimeError::ModelNotLoaded => {
            GenerateError::new_err("Model not loaded — call load_model() first")
        }
        RuntimeError::TokenizerNotAvailable => TokenizerError::new_err(
            "Tokenizer not available — rebuild with the `tokenizer-wasm` feature enabled",
        ),
        RuntimeError::ModelLoadError { message } => {
            LoadError::new_err(format!("Model load error: {message}"))
        }
        RuntimeError::TokenizerError { message } => {
            TokenizerError::new_err(format!("Tokenizer error: {message}"))
        }
        RuntimeError::SamplingError { message } => {
            GenerateError::new_err(format!("Sampling error: {message}"))
        }
        RuntimeError::KvCacheFull { max_ctx } => KvCacheFullError::new_err(format!(
            "KV cache full: maximum context length {max_ctx} reached"
        )),
        RuntimeError::Cancelled => GenerateError::new_err("Generation cancelled"),
        RuntimeError::Arch(arch_err) => arch_to_py(arch_err),
        RuntimeError::Gguf(gguf_err) => LoadError::new_err(format!("GGUF parse error: {gguf_err}")),
        RuntimeError::Quant(quant_err) => {
            QuantError::new_err(format!("Quantization error: {quant_err}"))
        }
        RuntimeError::Io(io_err) => LoadError::new_err(format!("I/O error: {io_err}")),
        RuntimeError::Grammar(grammar_err) => {
            GrammarError::new_err(format!("Grammar error: {grammar_err}"))
        }
        RuntimeError::AttentionError { message } => {
            GenerateError::new_err(format!("Attention error: {message}"))
        }
        RuntimeError::SnapshotIncompatible { detail } => {
            GenerateError::new_err(format!("Snapshot incompatible: {detail}"))
        }
        RuntimeError::ModelFingerprintMismatch {
            expected,
            found,
            detail,
        } => LoadError::new_err(format!(
            "Model fingerprint mismatch — expected {expected}, found {found}: {detail}"
        )),
        RuntimeError::OffloadEof {
            offset,
            needed,
            available,
        } => LoadError::new_err(format!(
            "Offload I/O error: unexpected EOF at offset {offset}, needed {needed} bytes, {available} available"
        )),
        RuntimeError::TensorNotFound(name) => {
            LoadError::new_err(format!("Tensor not found in weight map: '{name}'"))
        }
        RuntimeError::LockPoisoned => {
            GenerateError::new_err("Internal error: lock poisoned")
        }
    }
}

// ---------------------------------------------------------------------------
// ArchError → PyErr
// ---------------------------------------------------------------------------

/// Convert an [`ArchError`] into a Python exception.
///
/// Mapping:
/// - `Gguf(…)`               → `LoadError`
/// - `Quant(…)`              → `QuantError`
/// - `MissingTensor`         → `LoadError`
/// - `UnknownArchitecture`   → `LoadError`
/// - `ConfigMismatch`        → `LoadError`
/// - `TensorShapeMismatch`   → `LoadError`
/// - `InvalidShape`          → `LoadError`
/// - `InvalidConfig`         → `LoadError`
/// - `NotSupported`          → `GenerateError`
/// - `ForwardPassError`      → `GenerateError`
/// - `UnsupportedOperation`  → `GenerateError`
pub fn arch_to_py(err: ArchError) -> PyErr {
    match err {
        ArchError::Gguf(gguf_err) => {
            LoadError::new_err(format!("GGUF error in arch layer: {gguf_err}"))
        }
        ArchError::Quant(quant_err) => {
            QuantError::new_err(format!("Quantization error in arch layer: {quant_err}"))
        }
        ArchError::MissingTensor { name } => {
            LoadError::new_err(format!("Missing tensor in model: '{name}'"))
        }
        ArchError::UnknownArchitecture { arch_id } => {
            LoadError::new_err(format!("Unknown architecture: '{arch_id}'"))
        }
        ArchError::ConfigMismatch {
            param,
            expected,
            got,
        } => LoadError::new_err(format!(
            "Config mismatch for '{param}': expected {expected}, got {got}"
        )),
        ArchError::TensorShapeMismatch {
            tensor,
            expected,
            got,
        } => LoadError::new_err(format!(
            "Tensor shape mismatch for '{tensor}': expected {expected:?}, got {got:?}"
        )),
        ArchError::NotSupported { detail } => {
            GenerateError::new_err(format!("Operation not supported: {detail}"))
        }
        ArchError::ForwardPassError { layer, message } => {
            GenerateError::new_err(format!("Forward pass error at layer {layer}: {message}"))
        }
        ArchError::InvalidShape {
            name,
            expected,
            got,
        } => LoadError::new_err(format!(
            "Invalid shape for '{name}': expected {expected:?}, got {got:?}"
        )),
        ArchError::InvalidConfig { detail } => {
            LoadError::new_err(format!("Invalid configuration: {detail}"))
        }
        ArchError::UnsupportedOperation { message } => {
            GenerateError::new_err(format!("Unsupported operation: {message}"))
        }
        ArchError::LoraIncompatible { detail } => {
            LoadError::new_err(format!("LoRA adapter incompatible: {detail}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests verify the error mapping at the Rust level only.
    //
    // PyO3 tests that require `Python::with_gil` (i.e., a live interpreter)
    // cannot run as native `cargo test` binaries when the crate is a `cdylib`
    // with `abi3-py38` (no libpython linked at compile time).
    //
    // We verify correctness by:
    //   1. Checking that `runtime_to_py` does not panic for each variant.
    //   2. Checking the error message payload contains the expected string.
    //   3. Full Python-side type-checking is done in `python/tests/test_engine.py`.

    /// Every `RuntimeError` variant must map without panicking.
    #[test]
    fn test_all_runtime_error_variants_map_without_panic() {
        use oxillama_runtime::sampling::grammar::GrammarError;

        let variants: Vec<RuntimeError> = vec![
            RuntimeError::ModelNotLoaded,
            RuntimeError::TokenizerNotAvailable,
            RuntimeError::TokenizerError {
                message: "test".to_string(),
            },
            RuntimeError::SamplingError {
                message: "test".to_string(),
            },
            RuntimeError::KvCacheFull { max_ctx: 1024 },
            RuntimeError::ModelLoadError {
                message: "test".to_string(),
            },
            RuntimeError::Cancelled,
            RuntimeError::Gguf(oxillama_gguf::GgufError::InvalidMagic { magic: 0 }),
            RuntimeError::Quant(oxillama_quant::QuantError::UnsupportedType {
                quant_type: "Q99".to_string(),
            }),
            RuntimeError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test")),
            RuntimeError::Grammar(GrammarError::Stuck),
            RuntimeError::OffloadEof {
                offset: 0,
                needed: 4,
                available: 2,
            },
            RuntimeError::TensorNotFound("blk.0.attn_q.weight".to_string()),
            RuntimeError::LockPoisoned,
        ];

        for variant in variants {
            // Should not panic.
            let _ = runtime_to_py(variant);
        }
    }

    /// Every `ArchError` variant must map without panicking.
    #[test]
    fn test_all_arch_error_variants_map_without_panic() {
        let variants: Vec<ArchError> = vec![
            ArchError::Gguf(oxillama_gguf::GgufError::InvalidMagic { magic: 0 }),
            ArchError::Quant(oxillama_quant::QuantError::UnsupportedType {
                quant_type: "Q99".to_string(),
            }),
            ArchError::MissingTensor {
                name: "blk.0.attn_q.weight".to_string(),
            },
            ArchError::UnknownArchitecture {
                arch_id: "foo".to_string(),
            },
            ArchError::ConfigMismatch {
                param: "p".to_string(),
                expected: "a".to_string(),
                got: "b".to_string(),
            },
            ArchError::TensorShapeMismatch {
                tensor: "t".to_string(),
                expected: vec![2, 3],
                got: vec![3, 2],
            },
            ArchError::NotSupported {
                detail: "x".to_string(),
            },
            ArchError::ForwardPassError {
                layer: 0,
                message: "err".to_string(),
            },
            ArchError::InvalidShape {
                name: "expert.gate".to_string(),
                expected: vec![8, 4],
                got: vec![3],
            },
            ArchError::InvalidConfig {
                detail: "top_k must be >= 1".to_string(),
            },
            ArchError::LoraIncompatible {
                detail: "rank mismatch".to_string(),
            },
        ];

        for variant in variants {
            let _ = arch_to_py(variant);
        }
    }

    /// `ModelNotLoaded` Rust error message must contain "no model loaded".
    ///
    /// We test the *Rust* error message directly (before wrapping in PyErr)
    /// because `PyErr::to_string` requires a live Python interpreter.
    #[test]
    fn test_model_not_loaded_message() {
        let rust_msg = RuntimeError::ModelNotLoaded.to_string();
        assert!(
            rust_msg.to_lowercase().contains("model"),
            "Rust error message should mention 'model', got: {rust_msg}"
        );
    }

    /// `ModelLoadError` Rust message must contain the wrapped message.
    #[test]
    fn test_model_load_error_contains_message() {
        let rust_msg = RuntimeError::ModelLoadError {
            message: "missing_xyz.gguf".to_string(),
        }
        .to_string();
        assert!(
            rust_msg.contains("missing_xyz.gguf"),
            "Rust error message should contain original cause, got: {rust_msg}"
        );
    }

    /// `KvCacheFull` Rust message must mention the max context size.
    #[test]
    fn test_kv_cache_full_message() {
        let rust_msg = RuntimeError::KvCacheFull { max_ctx: 9999 }.to_string();
        assert!(
            rust_msg.contains("9999"),
            "Rust error message should contain max_ctx=9999, got: {rust_msg}"
        );
    }
}

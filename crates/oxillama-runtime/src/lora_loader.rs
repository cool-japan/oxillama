//! Runtime API for applying a loaded LoRA adapter to an [`InferenceEngine`].
//!
//! ## Usage
//!
//! After calling [`InferenceEngine::load_model`], use [`apply_lora`] to patch
//! the model's linear layers with the LoRA correction matrices.
//!
//! ```no_run
//! use oxillama_runtime::{EngineConfig, InferenceEngine};
//! use oxillama_runtime::lora_loader::apply_lora;
//!
//! let config = EngineConfig { model_path: "model.gguf".into(), ..Default::default() };
//! let mut engine = InferenceEngine::new(config);
//! engine.load_model().unwrap();
//! apply_lora(&mut engine, "adapter.gguf").unwrap();
//! ```
//!
//! ## How it works
//!
//! 1. The adapter GGUF is parsed and dequantized into a [`LoadedLora`].
//! 2. `engine.apply_lora_adapters(&lora)` is called, which delegates to the
//!    architecture-specific `ForwardPass::apply_lora` implementation.
//! 3. Each `QuantLinear` layer whose tensor name matches an entry in the
//!    adapter map receives an `Arc<LoraAdapter>`, which is applied after
//!    the main GEMV on every subsequent forward call.

use oxillama_arch::lora::LoadedLora;

use crate::engine::InferenceEngine;
use crate::error::{RuntimeError, RuntimeResult};

/// Load a LoRA adapter GGUF file and apply it to a loaded [`InferenceEngine`].
///
/// The engine must have a model loaded (via [`InferenceEngine::load_model`]).
/// This function patches the model's `QuantLinear` layers with LoRA correction
/// matrices in-place.  Subsequent forward passes will include the correction.
///
/// # Errors
///
/// Returns [`RuntimeError::ModelLoadError`] if the adapter GGUF cannot be
/// parsed.  Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded.
pub fn apply_lora(engine: &mut InferenceEngine, lora_path: &str) -> RuntimeResult<()> {
    let lora = LoadedLora::load(lora_path).map_err(|e| RuntimeError::ModelLoadError {
        message: format!("LoRA load failed: {e}"),
    })?;

    let rank = lora.rank;
    let alpha = lora.alpha;
    let n_adapters = lora.num_adapters();

    engine.apply_lora_adapters(&lora)?;

    tracing::info!(
        path = %lora_path,
        rank = rank,
        alpha = alpha,
        adapters = n_adapters,
        "LoRA adapter applied"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineConfig, InferenceEngine};

    /// `apply_lora` on a model-less engine should return `ModelLoadError`
    /// (because the GGUF file doesn't exist), not `ModelNotLoaded`.
    #[test]
    fn test_apply_lora_missing_file() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let path = std::env::temp_dir().join("nonexistent_adapter_xyz.gguf");
        let path_str = path.to_string_lossy();
        let result = apply_lora(&mut engine, &path_str);
        assert!(
            result.is_err(),
            "apply_lora with missing file should return Err"
        );
        assert!(
            matches!(result, Err(RuntimeError::ModelLoadError { .. })),
            "expected ModelLoadError for missing adapter file, got {:?}",
            result
        );
    }

    /// `apply_lora` with a file that exists but contains garbage bytes must return
    /// `ModelLoadError` from the GGUF parse failure, not panic or return Ok.
    #[test]
    fn test_apply_lora_garbage_bytes_errors() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxillama_lora_bad_magic_test.gguf");
        std::fs::write(
            &tmp,
            b"GARBAGE BYTES - DEFINITELY NOT A GGUF FILE 9876543210",
        )
        .expect("write temp file");
        let path = tmp
            .to_str()
            .expect("temp path must be valid UTF-8")
            .to_string();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = apply_lora(&mut engine, &path);
        let _ = std::fs::remove_file(&tmp);
        assert!(
            result.is_err(),
            "apply_lora with garbage GGUF content should return Err"
        );
        assert!(
            matches!(result, Err(RuntimeError::ModelLoadError { .. })),
            "expected ModelLoadError for invalid GGUF content, got {:?}",
            result
        );
    }

    /// `apply_lora` with an empty file must return `ModelLoadError` (GGUF parse
    /// error for truncated/missing header), not panic.
    #[test]
    fn test_apply_lora_empty_file_errors() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxillama_lora_empty_test.gguf");
        std::fs::write(&tmp, b"").expect("write empty temp file");
        let path = tmp
            .to_str()
            .expect("temp path must be valid UTF-8")
            .to_string();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = apply_lora(&mut engine, &path);
        let _ = std::fs::remove_file(&tmp);
        assert!(
            result.is_err(),
            "apply_lora with empty file should return Err"
        );
        assert!(
            matches!(result, Err(RuntimeError::ModelLoadError { .. })),
            "expected ModelLoadError for empty file, got {:?}",
            result
        );
    }
}

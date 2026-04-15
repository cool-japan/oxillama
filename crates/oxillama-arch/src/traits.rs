//! Core traits defining the model architecture plugin system.
//!
//! Every model family (LLaMA, Qwen3, Mistral, etc.) implements
//! [`ModelArchitecture`] to register itself, and [`ForwardPass`]
//! for the actual inference computation.

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::lora::LoadedLora;
use oxillama_gguf::TensorStore;

/// Pattern for matching expected tensor names in a model file.
#[derive(Debug, Clone)]
pub struct TensorNamePattern {
    /// Regex or glob pattern for tensor names.
    pub pattern: String,
    /// Human-readable description of what this tensor represents.
    pub description: String,
    /// Whether this tensor is required for the architecture.
    pub required: bool,
}

/// Trait for a model architecture plugin.
///
/// Implementations register themselves with the [`ArchitectureRegistry`](crate::registry::ArchitectureRegistry)
/// and provide the ability to build a runnable model from GGUF data.
pub trait ModelArchitecture: Send + Sync {
    /// Architecture identifier string (matches GGUF `general.architecture` metadata).
    ///
    /// Examples: `"llama"`, `"qwen3"`, `"mistral"`, `"gemma"`, `"phi"`.
    fn arch_id(&self) -> &str;

    /// Build a runnable model from configuration and loaded tensors.
    ///
    /// This is called once during model loading. The returned [`ForwardPass`]
    /// implementation owns the model weights and is used for inference.
    fn build(
        &self,
        config: &ModelConfig,
        tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>>;

    /// Expected tensor name patterns for this architecture.
    ///
    /// Used for validation and diagnostics when loading a model file.
    fn tensor_names(&self) -> Vec<TensorNamePattern>;
}

/// Trait for running forward passes through a loaded model.
///
/// Implementations own the model weights and maintain any mutable state
/// needed during inference (e.g., internal buffers).
pub trait ForwardPass: Send + Sync {
    /// Run one forward pass, returning logits for the next token prediction.
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs for this step.
    /// * `kv_cache` - Mutable reference to the key-value cache.
    ///
    /// # Returns
    /// A vector of logits with length equal to the vocabulary size.
    fn forward(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess)
        -> ArchResult<Vec<f32>>;

    /// Run one forward pass, returning the post-output-norm hidden state
    /// (not projected through the LM head).
    ///
    /// This is the embedding extraction path: it runs all transformer layers
    /// and applies the final RMSNorm, but stops before the LM-head projection
    /// that maps hidden_size → vocab_size. The returned vector has length
    /// `hidden_size`, not `vocab_size`.
    ///
    /// The default implementation returns [`ArchError::NotSupported`].
    /// Each architecture overrides this with a concrete implementation.
    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let _ = (tokens, kv_cache);
        Err(ArchError::NotSupported {
            detail: "embed() not implemented for this architecture".to_string(),
        })
    }

    /// Returns the model's vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Returns the model's maximum context length.
    fn max_context_length(&self) -> usize;

    /// Returns the model's hidden size (embedding dimension).
    fn hidden_size(&self) -> usize;

    /// Apply LoRA adapter corrections to this model's linear layers.
    ///
    /// Walks the model's `QuantLinear` fields and calls
    /// [`QuantLinear::set_lora`](crate::common::linear::QuantLinear::set_lora)
    /// for every layer whose name appears in `lora.adapters`.
    ///
    /// The default implementation is a no-op: models that do not yet support
    /// LoRA patching will silently ignore the adapter.  Override this method
    /// in each architecture implementation that supports LoRA.
    fn apply_lora(&mut self, lora: &LoadedLora) -> ArchResult<()> {
        let _ = lora;
        Ok(())
    }
}

/// Minimal KV cache interface used by forward pass implementations.
///
/// This trait is defined in `oxillama-arch` to avoid a circular dependency
/// with `oxillama-runtime` where the full KV cache lives.
pub trait KvCacheAccess: Send + Sync {
    /// Get the current sequence length (number of cached tokens).
    fn seq_len(&self) -> usize;

    /// Store key and value tensors for a layer at the current position.
    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()>;

    /// Retrieve all cached keys for a layer up to the current sequence length.
    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]>;

    /// Retrieve all cached values for a layer up to the current sequence length.
    fn get_values(&self, layer: usize) -> ArchResult<&[f32]>;

    /// Advance the cache position by one token.
    ///
    /// Called after all layers have stored their K/V for the current token.
    fn advance(&mut self);
}

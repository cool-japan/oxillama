//! Core traits defining the model architecture plugin system.
//!
//! Every model family (LLaMA, Qwen3, Mistral, etc.) implements
//! [`ModelArchitecture`] to register itself, and [`ForwardPass`]
//! for the actual inference computation.

use crate::common::sequence_state::{AttentionSequenceState, SequenceState};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::lora::{LoadedLora, LoraStack};
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

    /// Returns the sliding-window attention configuration for this model.
    ///
    /// Returns `Some((window_size, is_interleaved))` when the architecture
    /// uses SWA on at least some layers, or `None` for pure global attention.
    fn swa_config(&self) -> Option<(u32, bool)> {
        None
    }
}

/// A single request's slot within the shared KV pool.
///
/// Each in-flight request receives one `KvSlot` that identifies which
/// position in the KV pool belongs to it.  The slot is released back to the
/// pool when the request finishes (EOS or max-token limit reached).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSlot {
    /// Unique identifier of the request that owns this slot.
    pub request_id: u64,
    /// Index into the shared KV cache pool (e.g. the row within a paged KV
    /// cache or the sequence slot index in a flat pool).
    pub kv_cache_idx: usize,
    /// Current sequence position (number of tokens committed so far).
    pub position: usize,
}

impl KvSlot {
    /// Construct a new `KvSlot`.
    pub fn new(request_id: u64, kv_cache_idx: usize, position: usize) -> Self {
        Self {
            request_id,
            kv_cache_idx,
            position,
        }
    }
}

/// A view over the KV caches of multiple concurrent requests for batched
/// decode attention.
///
/// During the decode phase each request has already accumulated keys and
/// values from the prefill + prior decode steps.  `BatchedKvView` provides
/// the batched-attention kernel with access to per-request KV slices without
/// requiring the caller to lay out memory in any particular way.
///
/// Implementors typically wrap a pool of KV cache buffers indexed by [`KvSlot`].
pub trait BatchedKvView: Sync {
    /// Number of concurrent request slots in this batch.
    fn slot_count(&self) -> usize;

    /// Return the flattened key and value slices for slot `slot`.
    ///
    /// Both slices have length `position(slot) * kv_dim`, laid out as
    /// `[seq_len, kv_dim]` in row-major order.
    ///
    /// # Panics
    ///
    /// Implementations are permitted to panic if `slot >= slot_count()`.
    fn kv_for_slot(&self, slot: usize) -> (&[f32], &[f32]);

    /// Number of KV tokens already committed for slot `slot`
    /// (= the sequence position the next token will be written to).
    fn position(&self, slot: usize) -> usize;
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

    /// KV dimension per token (num_kv_heads * head_dim).
    ///
    /// Returns `0` by default, which signals that per-token iteration is not
    /// available via the default [`for_each_key`](Self::for_each_key) /
    /// [`for_each_value`](Self::for_each_value) helpers.  Implementations that
    /// know their KV dimension should override this.
    fn kv_dim(&self) -> usize {
        0
    }

    /// Iterate over every cached key token for `layer`, calling `f(pos, key_data)`.
    ///
    /// The default implementation chunks `get_keys()` using [`kv_dim()`](Self::kv_dim).
    /// Paged implementations override this to avoid assembling a contiguous slice.
    ///
    /// # Errors
    ///
    /// Returns [`ArchError::NotSupported`] if `kv_dim()` returns `0`.
    /// Propagates any error from [`get_keys()`](Self::get_keys).
    fn for_each_key(&self, layer: usize, f: &mut dyn FnMut(usize, &[f32])) -> ArchResult<()> {
        let dim = self.kv_dim();
        if dim == 0 {
            return Err(ArchError::NotSupported {
                detail: "kv_dim() not implemented; cannot iterate per-token keys".to_string(),
            });
        }
        let keys = self.get_keys(layer)?;
        for (pos, slice) in keys.chunks_exact(dim).enumerate() {
            f(pos, slice);
        }
        Ok(())
    }

    /// Iterate over every cached value token for `layer`, calling `f(pos, value_data)`.
    ///
    /// The default implementation chunks `get_values()` using [`kv_dim()`](Self::kv_dim).
    /// Paged implementations override this to avoid assembling a contiguous slice.
    ///
    /// # Errors
    ///
    /// Returns [`ArchError::NotSupported`] if `kv_dim()` returns `0`.
    /// Propagates any error from [`get_values()`](Self::get_values).
    fn for_each_value(&self, layer: usize, f: &mut dyn FnMut(usize, &[f32])) -> ArchResult<()> {
        let dim = self.kv_dim();
        if dim == 0 {
            return Err(ArchError::NotSupported {
                detail: "kv_dim() not implemented; cannot iterate per-token values".to_string(),
            });
        }
        let values = self.get_values(layer)?;
        for (pos, slice) in values.chunks_exact(dim).enumerate() {
            f(pos, slice);
        }
        Ok(())
    }
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

    /// Apply an ordered stack of LoRA adapters.
    ///
    /// Default implementation: applies each adapter in the stack in order via
    /// [`apply_lora`](Self::apply_lora), ignoring per-entry scale multipliers.
    /// Override for architectures that support scaled stacking.
    fn apply_lora_stack(&mut self, stack: &LoraStack) -> ArchResult<()> {
        for (lora, _scale) in stack.entries() {
            self.apply_lora(lora)?;
        }
        Ok(())
    }

    /// Returns the sliding-window attention configuration for this loaded model.
    ///
    /// Returns `Some((window_size, is_interleaved))` when the model uses SWA
    /// on at least some layers, or `None` for pure global attention.
    fn swa_config(&self) -> Option<(u32, bool)> {
        None
    }

    /// Set a persistent LoRA adapter stack that applies to all subsequent
    /// `forward()` calls.
    ///
    /// Default implementation is a no-op — architectures that do not support
    /// LoRA stacking silently ignore the call (compatible with the existing
    /// `apply_lora_stack` interface).
    ///
    /// # Errors
    ///
    /// Returns [`ArchError::LoraIncompatible`] if the adapter's rank or
    /// dimensions are incompatible with this model.
    fn with_lora_stack(&mut self, _stack: LoraStack) -> ArchResult<()> {
        Ok(())
    }

    /// Allocate a fresh per-sequence state object for this model.
    ///
    /// The runtime calls this once per pool slot at model load time.
    /// Default implementation returns an [`AttentionSequenceState`] suitable
    /// for all KV-cache-based architectures.
    ///
    /// SSM and hybrid architectures **must** override this to return the
    /// correct state type (e.g. [`Mamba2SequenceState`] or `JambaSequenceState`).
    ///
    /// [`Mamba2SequenceState`]: crate::common::sequence_state::Mamba2SequenceState
    fn allocate_sequence_state(&self, max_context_length: usize) -> Box<dyn SequenceState> {
        Box::new(AttentionSequenceState::new(max_context_length))
    }

    /// Run a batched decode-phase forward pass across multiple concurrent requests.
    ///
    /// Each slot in `kv_view` corresponds to one batch element.  `q_batch` is
    /// laid out as `[batch_size, num_heads, head_dim]` in row-major order.
    ///
    /// The default implementation returns [`ArchError::NotSupported`].
    /// Architectures that support continuous batching override this.
    ///
    /// # Arguments
    ///
    /// * `q_batch`    - Query tensor, shape `[batch_size, num_heads, head_dim]`.
    /// * `kv_view`    - Per-slot KV cache view.
    /// * `num_heads`  - Number of query attention heads.
    /// * `head_dim`   - Per-head dimension.
    /// * `scale`      - Softmax scale factor (typically `1 / sqrt(head_dim)`).
    ///
    /// # Returns
    ///
    /// Output tensor with layout `[batch_size, num_heads, head_dim]`.
    fn forward_batched(
        &mut self,
        q_batch: &[f32],
        kv_view: &dyn BatchedKvView,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> ArchResult<Vec<f32>> {
        let _ = (q_batch, kv_view, num_heads, head_dim, scale);
        Err(ArchError::NotSupported {
            detail: "forward_batched() not implemented for this architecture".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ArchError;

    /// A minimal stub implementing ForwardPass to test the default
    /// `forward_batched` returns NotSupported.
    struct StubModel;

    impl ForwardPass for StubModel {
        fn forward(
            &mut self,
            _tokens: &[u32],
            _kv_cache: &mut dyn KvCacheAccess,
        ) -> ArchResult<Vec<f32>> {
            Ok(vec![])
        }

        fn vocab_size(&self) -> usize {
            1
        }

        fn max_context_length(&self) -> usize {
            1
        }

        fn hidden_size(&self) -> usize {
            1
        }
    }

    /// A minimal BatchedKvView for testing.
    struct EmptyKvView;
    impl BatchedKvView for EmptyKvView {
        fn slot_count(&self) -> usize {
            0
        }

        fn kv_for_slot(&self, _slot: usize) -> (&[f32], &[f32]) {
            (&[], &[])
        }

        fn position(&self, _slot: usize) -> usize {
            0
        }
    }

    #[test]
    fn forward_batched_default_returns_not_supported() {
        let mut model = StubModel;
        let view = EmptyKvView;
        let result = model.forward_batched(&[], &view, 2, 4, 0.5);
        match result {
            Err(ArchError::NotSupported { detail }) => {
                assert!(
                    detail.contains("forward_batched"),
                    "error detail should mention forward_batched, got: {detail}"
                );
            }
            other => panic!("expected NotSupported, got: {other:?}"),
        }
    }

    #[test]
    fn forward_batched_empty_batch_via_default_is_not_supported() {
        // The default implementation always returns NotSupported regardless of
        // batch size — it cannot know the correct answer without weights.
        let mut model = StubModel;
        let view = EmptyKvView;
        let result = model.forward_batched(&[], &view, 1, 8, 1.0);
        assert!(result.is_err(), "default must return Err");
    }

    #[test]
    fn kv_slot_construction() {
        let slot = KvSlot::new(42, 7, 100);
        assert_eq!(slot.request_id, 42);
        assert_eq!(slot.kv_cache_idx, 7);
        assert_eq!(slot.position, 100);
    }

    #[test]
    fn kv_cache_access_default_kv_dim_is_zero() {
        /// Minimal KvCacheAccess impl that does not override kv_dim().
        struct MinimalCache;
        impl KvCacheAccess for MinimalCache {
            fn seq_len(&self) -> usize {
                0
            }
            fn store_kv(&mut self, _layer: usize, _key: &[f32], _value: &[f32]) -> ArchResult<()> {
                Ok(())
            }
            fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
                Ok(&[])
            }
            fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
                Ok(&[])
            }
            fn advance(&mut self) {}
        }

        let cache = MinimalCache;
        assert_eq!(cache.kv_dim(), 0, "default kv_dim must be 0");

        // for_each_key must return NotSupported when kv_dim == 0
        let mut called = false;
        let result = cache.for_each_key(0, &mut |_, _| {
            called = true;
        });
        assert!(result.is_err(), "must return Err when kv_dim() == 0");
        assert!(!called, "callback must not be invoked");
    }
}

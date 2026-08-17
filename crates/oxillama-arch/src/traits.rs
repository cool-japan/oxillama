//! Core traits defining the model architecture plugin system.
//!
//! Every model family (LLaMA, Qwen3, Mistral, etc.) implements
//! [`ModelArchitecture`] to register itself, and [`ForwardPass`]
//! for the actual inference computation.

use crate::common::sequence_state::{AttentionSequenceState, SequenceState};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::lora::{LoadedLora, LoraStack};
use oxillama_gguf::{GgufModel, TensorStore};
use std::sync::Arc;

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

/// One stored [`QuantKernel`](oxillama_quant::QuantKernel) binding inside a
/// loaded model, as handed to [`ForwardPass::remap_quant_kernels`].
///
/// A site identifies *where* in the model a kernel is bound and *what weight*
/// it is dispatched for; a backend decides from `weight.tensor_type` and
/// `weight.shape` whether it can serve that binding itself.
///
/// `role` is drawn from a fixed, stable set — `"attn_q"`, `"attn_k"`,
/// `"attn_v"`, `"attn_output"`, `"ffn_gate"`, `"ffn_up"`, `"ffn_down"`,
/// `"output"` — and is part of the contract: an architecture must not rename
/// its projections here, even when its own GGUF tensor names differ.
pub struct QuantKernelSite<'a> {
    /// Transformer block index, or `None` for the LM head.
    pub layer: Option<usize>,
    /// Stable short name of the projection this kernel serves.
    pub role: &'static str,
    /// The quantized weight tensor this kernel is dispatched for.
    pub weight: &'a oxillama_quant::QuantTensor,
}

/// The visitor [`ForwardPass::remap_quant_kernels`] drives.
///
/// Receives one [`QuantKernelSite`] together with the kernel currently stored
/// there, and returns the kernel to store instead — returning the argument
/// unchanged is the identity remap.
pub type QuantKernelRemap<'a> = dyn FnMut(
        QuantKernelSite<'_>,
        Arc<dyn oxillama_quant::QuantKernel>,
    ) -> Arc<dyn oxillama_quant::QuantKernel>
    + 'a;

/// Hand one stored kernel binding to `f` and store the returned kernel back.
///
/// `weight` and `slot` must be disjoint places (distinct fields of the same
/// layer or model), otherwise the caller cannot form both borrows.
#[cfg(any(feature = "llama", feature = "qwen3"))]
pub(crate) fn remap_kernel_slot(
    f: &mut QuantKernelRemap<'_>,
    layer: Option<usize>,
    role: &'static str,
    weight: &oxillama_quant::QuantTensor,
    slot: &mut Arc<dyn oxillama_quant::QuantKernel>,
) {
    let current = Arc::clone(slot);
    *slot = f(
        QuantKernelSite {
            layer,
            role,
            weight,
        },
        current,
    );
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

    /// Build a runnable model from a fully-loaded GGUF file.
    ///
    /// [`Self::build`] only receives the tensor *metadata* table, which is why
    /// ~22 of the 27 implementations return
    /// `MissingTensor { name: "… (use X::from_gguf for full loading)" }` and
    /// the registry can build nothing.  This entry point receives the payload
    /// too, so every architecture can route its existing `from_gguf` loader
    /// through the registry instead of forcing the engine to keep a hard-coded
    /// `match` over architecture names.
    ///
    /// # Migration
    ///
    /// Each architecture overrides this with a one-line delegation:
    ///
    /// ```ignore
    /// fn build_from_gguf(
    ///     &self,
    ///     model: &GgufModel,
    ///     config: &ModelConfig,
    /// ) -> ArchResult<Box<dyn ForwardPass>> {
    ///     Ok(Box::new(LlamaModel::from_gguf(model, config)?))
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// The default returns [`ArchError::NotSupported`]; architectures that have
    /// not migrated yet are therefore reported explicitly rather than
    /// pretending a tensor is missing.
    fn build_from_gguf(
        &self,
        model: &GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        let _ = (model, config);
        Err(ArchError::NotSupported {
            detail: format!(
                "architecture '{}' has not implemented build_from_gguf()",
                self.arch_id()
            ),
        })
    }

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

    /// Run one forward pass, returning per-token hidden states for **all** tokens
    /// as a flat `[seq_len × hidden_size]` vector in row-major order.
    ///
    /// This is the multi-token embedding extraction path used when a pooling
    /// mode other than `Last` is requested. The returned vector has length
    /// `seq_len * hidden_size`.
    ///
    /// The default implementation returns [`ArchError::NotSupported`].
    /// Architectures that require multi-token pooling should override this.
    fn embed_all(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let _ = (tokens, kv_cache);
        Err(ArchError::NotSupported {
            detail: "embed_all() not implemented for this architecture; use embed() instead"
                .to_string(),
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
    /// The default implementation returns [`ArchError::NotSupported`].
    ///
    /// It used to return `Ok(())`, which meant that the five engine-reachable
    /// architectures without an override (qwen3, mistral, gemma, phi,
    /// starcoder) accepted an adapter, ignored it, and reported success.  A
    /// loud failure is strictly better than a silent no-op.
    ///
    /// # Errors
    ///
    /// [`ArchError::NotSupported`] unless the architecture overrides this.
    fn apply_lora(&mut self, lora: &LoadedLora) -> ArchResult<()> {
        let _ = lora;
        Err(ArchError::NotSupported {
            detail: "apply_lora() not implemented for this architecture".to_string(),
        })
    }

    /// Apply one LoRA adapter with an extra scale multiplier.
    ///
    /// This is the primitive [`apply_lora_stack`](Self::apply_lora_stack)
    /// builds on.  The default delegates to [`apply_lora`](Self::apply_lora)
    /// when `scale == 1.0` and otherwise reports that the architecture cannot
    /// honour the multiplier — previously the multiplier was simply dropped.
    ///
    /// Architectures implement this by calling
    /// [`QuantLinear::push_lora`](crate::common::linear::QuantLinear::push_lora)
    /// (which **accumulates**) instead of
    /// [`QuantLinear::set_lora`](crate::common::linear::QuantLinear::set_lora)
    /// (which **replaces**).
    ///
    /// # Errors
    ///
    /// [`ArchError::NotSupported`] when the architecture cannot apply the
    /// adapter or cannot honour a non-unit scale.
    fn apply_lora_scaled(&mut self, lora: &LoadedLora, scale: f32) -> ArchResult<()> {
        if (scale - 1.0).abs() <= f32::EPSILON {
            self.apply_lora(lora)
        } else {
            Err(ArchError::NotSupported {
                detail: format!(
                    "apply_lora_scaled(scale = {scale}) not implemented for this architecture; \
                     the per-entry LoRA scale would be silently discarded"
                ),
            })
        }
    }

    /// Apply an ordered stack of LoRA adapters.
    ///
    /// Each entry is applied through
    /// [`apply_lora_scaled`](Self::apply_lora_scaled) so its scale multiplier
    /// is honoured; the previous implementation discarded every `_scale`.
    ///
    /// # Errors
    ///
    /// Propagates the first per-entry failure.
    fn apply_lora_stack(&mut self, stack: &LoraStack) -> ArchResult<()> {
        for (lora, scale) in stack.entries() {
            self.apply_lora_scaled(lora, *scale)?;
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

    /// Offer every stored [`QuantKernel`](oxillama_quant::QuantKernel) binding
    /// to `_f`, replacing each with whatever `_f` returns.
    ///
    /// This is how an out-of-tree backend (a GPU offloader, an instrumenting
    /// proxy) swaps the kernels a loaded model dispatches through without this
    /// crate depending on it: `_f` receives the [`QuantKernelSite`] describing
    /// the binding plus the kernel currently stored there, and returns the
    /// kernel to store instead — returning the argument unchanged leaves the
    /// model exactly as it was.
    ///
    /// # Contract
    ///
    /// An implementation MUST visit every kernel binding the decode and
    /// prefill paths actually read — the attention projections, the dense-FFN
    /// projections and the LM head — passing the kernel stored at that site
    /// together with the weight tensor it serves, and MUST store the returned
    /// `Arc` back into that site.  The default visits nothing, which is the
    /// correct answer for an architecture that does not support remapping.
    ///
    /// # Out of contract
    ///
    /// * MoE expert and router kernels.  They live in companion structs and no
    ///   architecture exposes them here yet.
    /// * Any path that re-dispatches a kernel per call from the model's
    ///   `KernelDispatcher` instead of reading a stored `Arc` — the tiled
    ///   multi-token prefill of `llama`/`qwen3` and the token-embedding row
    ///   lookup both do this, so a remapped kernel does **not** apply to them.
    ///   A caller that needs the remap to hold for prompt processing must
    ///   drive prefill one token at a time.
    fn remap_quant_kernels(&mut self, _f: &mut QuantKernelRemap<'_>) {}

    /// Set a persistent LoRA adapter stack that applies to all subsequent
    /// `forward()` calls.
    ///
    /// The default delegates to [`apply_lora_stack`](Self::apply_lora_stack)
    /// so that architectures which support LoRA at all also support the
    /// persistent form.  It used to return `Ok(())` unconditionally, making the
    /// call a no-op for every architecture except Jamba.
    ///
    /// # Errors
    ///
    /// Returns [`ArchError::LoraIncompatible`] if the adapter's rank or
    /// dimensions are incompatible with this model, or
    /// [`ArchError::NotSupported`] when the architecture has no LoRA support.
    fn with_lora_stack(&mut self, stack: LoraStack) -> ArchResult<()> {
        self.apply_lora_stack(&stack)
    }

    /// Reset all per-sequence state so the model can start a fresh request.
    ///
    /// The runtime **must** call this whenever a KV-cache slot is handed to a
    /// new request (and the CLI must call it between prompts).  Without it:
    ///
    /// * DeepSeek's `MlaLatentCache` keeps growing until `append` errors, and
    ///   before that leaks the previous request's tokens into `attend_len`;
    /// * Mamba-2 / Jamba carry the previous sequence's `h` into the new one;
    /// * DBRX / Grok never return `current_pos` to 0.
    ///
    /// The default is a no-op, which is correct for stateless architectures
    /// whose only state is the externally-owned [`KvCacheAccess`].  Every
    /// architecture that owns *internal* mutable state must override it — see
    /// the per-architecture table in this crate's audit notes.
    fn reset_sequence(&mut self) {}

    /// Remove all LoRA adapters from every `QuantLinear` in this model.
    ///
    /// The inverse of [`Self::apply_lora_stack`]: sets `lora = None` on every linear
    /// layer that was patched.  The default is a no-op; architectures that
    /// override [`Self::apply_lora`] must also override this.
    fn unapply_all_loras(&mut self) {}

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

/// Kernel decorator shared by the per-architecture `remap_quant_kernels` tests.
#[cfg(all(test, any(feature = "llama", feature = "qwen3")))]
pub(crate) mod remap_test_support {
    use oxillama_quant::{QuantKernel, QuantResult, QuantTensor};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A [`QuantKernel`] that forwards every method to `inner` and counts the
    /// matmul entry points it was actually driven through.
    ///
    /// The count is what keeps the parity test honest: identical logits prove
    /// nothing unless the decorated kernels were on the path that produced
    /// them.
    pub(crate) struct CountingKernel {
        inner: Arc<dyn QuantKernel>,
        calls: Arc<AtomicUsize>,
    }

    impl CountingKernel {
        /// Wrap `inner`, sharing `calls` across every wrapper of one model.
        pub(crate) fn wrap(
            inner: Arc<dyn QuantKernel>,
            calls: Arc<AtomicUsize>,
        ) -> Arc<dyn QuantKernel> {
            Arc::new(Self { inner, calls })
        }
    }

    impl QuantKernel for CountingKernel {
        fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
            self.inner.dequant_block(block, output)
        }

        fn gemv(
            &self,
            quant_matrix: &QuantTensor,
            input: &[f32],
            output: &mut [f32],
        ) -> QuantResult<()> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.gemv(quant_matrix, input, output)
        }

        fn gemm(
            &self,
            quant_matrix: &QuantTensor,
            input: &[f32],
            output: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> QuantResult<()> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.gemm(quant_matrix, input, output, m, n, k)
        }

        fn matvec_q8_fused(
            &self,
            weights: &[u8],
            acts_q8: &[u8],
            out: &mut [f32],
            n_rows: usize,
            n_cols: usize,
        ) -> QuantResult<()> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner
                .matvec_q8_fused(weights, acts_q8, out, n_rows, n_cols)
        }

        fn matmul_q8_fused(
            &self,
            weights: &[u8],
            acts_q8: &[u8],
            out: &mut [f32],
            n_rows: usize,
            n_cols: usize,
            m: usize,
        ) -> QuantResult<()> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner
                .matmul_q8_fused(weights, acts_q8, out, n_rows, n_cols, m)
        }

        fn q8_fused_acts_blocks(&self, n_cols: usize) -> Option<usize> {
            self.inner.q8_fused_acts_blocks(n_cols)
        }

        fn block_size(&self) -> usize {
            self.inner.block_size()
        }

        fn block_bytes(&self) -> usize {
            self.inner.block_bytes()
        }

        fn name(&self) -> &'static str {
            self.inner.name()
        }
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

    /// An architecture that does not override the visitor exposes no sites,
    /// so a backend can tell "nothing to offload" from "offloaded nothing".
    #[test]
    fn remap_quant_kernels_default_visits_nothing() {
        let mut model = StubModel;
        let mut visited = 0usize;
        model.remap_quant_kernels(&mut |_site, kernel| {
            visited += 1;
            kernel
        });
        assert_eq!(visited, 0, "the default implementation must visit no sites");
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

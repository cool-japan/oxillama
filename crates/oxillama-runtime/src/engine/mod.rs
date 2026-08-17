// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
/// Sequence-length threshold above which the engine routes attention through
/// the memory-efficient tiled flash-attention kernel rather than the naïve
/// full-score-matrix path.
///
/// At and above this threshold the O(N²) memory cost of materialising the
/// full attention matrix becomes a bottleneck; the tiled kernel keeps memory
/// at O(BQ × BK) per tile instead.
///
/// The actual dispatch lives inside `oxillama-arch`'s `ForwardPass::forward`
/// implementations.  This constant is exported so that arch crates and
/// callers can apply the same policy without hard-coding the threshold.
pub const FLASH_ATTN_THRESHOLD: usize = 512;
use crate::chat_template::ChatTemplate;
use crate::embedding::{pool_hidden_states, PoolingMode};
use crate::error::{RuntimeError, RuntimeResult};
use crate::gpu_backend::{self, GpuPolicy, GpuStatus};
use crate::kv_cache::{KvCache, KvCacheDtype, KvCacheSnapshot};
use crate::metrics::{EngineMetrics, MetricsSnapshot};
use crate::offload::{LayerPager, OffloadPolicy};
use crate::sampling::SamplerConfig;
use crate::tokenizer_bridge::TokenizerBridge;
use oxillama_arch::config::ModelConfig;
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_arch::ArchitectureRegistry;
use oxillama_gguf::{GgufModel, MetadataStore};
use std::sync::Arc;

mod generation;

use generation::{prefill_chunked, run_decode_loop, DecodeContext};
pub use generation::{FinishReason, GenerationConfig, GenerationOutcome};

/// Default context length used when the caller does not specify one.
///
/// GGUF metadata reports the *training* context, which for Llama-3.1-8B is
/// 131072.  Allocating a KV cache that large up front costs
/// `32 layers × 131072 × 1024 × 4 bytes × 2 ≈ 34 GB` and kills the process
/// before the first token — so an unspecified context is clamped to this bound.
/// Set [`EngineConfig::context_size`] explicitly to ask for more.
pub const DEFAULT_MAX_CONTEXT: usize = 4096;
/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Path to the tokenizer JSON file (if not embedded in GGUF).
    pub tokenizer_path: Option<String>,
    /// Context size override.
    ///
    /// `None` means "pick a safe default": `min(n_ctx_train, `
    /// [`DEFAULT_MAX_CONTEXT`]`)`.  It does **not** mean "use the training
    /// context", because that is 131072 tokens for Llama-3.1 and the KV cache
    /// for it does not fit in memory.
    pub context_size: Option<usize>,
    /// Number of threads for parallel computation.
    ///
    /// Forwarded to [`oxillama_quant::parallel::set_num_threads`] when the
    /// model is loaded, sizing the **dedicated** GEMV thread pool that splits
    /// weight-matrix rows across cores.  rayon's *global* pool is left
    /// untouched, so a library consumer's own rayon work is unaffected.
    ///
    /// `0` means "auto": use [`std::thread::available_parallelism`].  The
    /// `OXILLAMA_NUM_THREADS` environment variable overrides this value.  The
    /// pool is process-wide and built once, so the first engine to load a
    /// model fixes the width for the lifetime of the process.
    pub num_threads: usize,
    /// Sampling configuration.
    pub sampler: SamplerConfig,
    /// Prefill chunk size: how many prompt tokens to process per forward call.
    ///
    /// Set to 0 or `usize::MAX` to process the entire prompt in one batch.
    /// Smaller values reduce peak memory usage for long prompts at the cost
    /// of slightly higher overhead from multiple forward calls.
    /// Default: 512.
    pub prefill_chunk_size: usize,
    /// CPU/disk offload policy.
    ///
    /// Controls which model weights are kept resident in RAM and which are
    /// evicted to disk and reloaded on demand.
    ///
    /// Default: [`OffloadPolicy::None`] — all weights remain in RAM, matching
    /// classic llama.cpp behaviour.
    pub offload_policy: OffloadPolicy,
    /// Element type used for KV cache **storage**.
    ///
    /// [`KvCacheDtype::F16`] halves the cache — for Llama-3-8B at a 4096-token
    /// context that is 1 GiB down to 512 MiB — at the cost of `f16` rounding on
    /// every key and value written.  Attention itself still accumulates in
    /// `f32`; only the stored elements narrow, which is exactly how llama.cpp
    /// arranges its own `--cache-type-k`/`--cache-type-v`.
    ///
    /// Default: [`KvCacheDtype::F32`].  llama.cpp defaults to `f16`; OxiLLaMa
    /// keeps the lossless dtype as the default so that upgrading never changes
    /// anyone's output silently.
    pub kv_dtype: KvCacheDtype,
    /// GPU offload policy, applied once per model load.
    ///
    /// [`GpuPolicy::On`] is a hard requirement, not a hint: if no device can be
    /// bound — including because the crate was built without the `gpu` feature
    /// — loading fails with [`RuntimeError::GpuUnavailable`] rather than
    /// quietly running on the CPU.  Inspect
    /// [`InferenceEngine::gpu_status`] afterwards to see what the device
    /// actually took.
    ///
    /// Default: [`GpuPolicy::Off`].
    pub gpu: GpuPolicy,
}
impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: None,
            context_size: None,
            // 0 = auto: size the GEMV pool from `available_parallelism()`.
            // Until 0.1.4 this field was 4 but nothing ever read it; hard-coding
            // 4 now would cap decode at half the cores on an 8-way machine.
            num_threads: 0,
            sampler: SamplerConfig::default(),
            prefill_chunk_size: 512,
            offload_policy: OffloadPolicy::None,
            kv_dtype: KvCacheDtype::F32,
            gpu: GpuPolicy::Off,
        }
    }
}
impl EngineConfig {
    /// Set the CPU/disk offload policy, consuming self and returning the
    /// updated config (builder pattern).
    pub fn with_offload(mut self, policy: OffloadPolicy) -> Self {
        self.offload_policy = policy;
        self
    }

    /// Set the KV cache storage dtype, consuming self and returning the updated
    /// config (builder pattern).
    pub fn with_kv_dtype(mut self, dtype: KvCacheDtype) -> Self {
        self.kv_dtype = dtype;
        self
    }

    /// Set the GPU offload policy, consuming self and returning the updated
    /// config (builder pattern).
    pub fn with_gpu(mut self, gpu: GpuPolicy) -> Self {
        self.gpu = gpu;
        self
    }
}
/// The main inference engine.
///
/// Manages model loading, forward pass execution, and token generation.
/// The full pipeline: load GGUF → parse metadata → build architecture → generate.
pub struct InferenceEngine {
    config: EngineConfig,
    /// Loaded GGUF model (None until load_model is called).
    gguf_model: Option<GgufModel>,
    /// Parsed model configuration from GGUF metadata.
    model_config: Option<ModelConfig>,
    /// Forward pass implementation (architecture-specific).
    forward_pass: Option<Box<dyn ForwardPass>>,
    /// Key-value cache.
    kv_cache: Option<KvCache>,
    /// Tokenizer bridge.
    tokenizer: Option<TokenizerBridge>,
    /// EOS token ID for stopping generation.
    eos_token_id: Option<u32>,
    /// Live metrics counters.
    metrics: Arc<EngineMetrics>,
    /// Stack of active LoRA adapters (in insertion order).
    lora_stack: oxillama_arch::LoraStack,
    /// Optional CPU/disk layer pager (None when offload_policy is None).
    ///
    /// When present, linear-layer forward passes can call
    /// `layer_pager.acquire(&tensor_id)` to get (or load from disk) the
    /// raw quantized bytes for a given weight tensor.  This is the
    /// graceful-fallback path: if `layer_pager` is `None`, existing in-RAM
    /// weight references are used unchanged.
    ///
    /// Full integration with the arch-layer forward kernels (wiring through
    /// `acquire` at each GEMM site) requires changes in `oxillama-arch` and
    /// is deferred to a follow-up subtask (R1-arch integration).
    layer_pager: Option<Arc<LayerPager>>,
    /// What the GPU took at the last successful load, or `None` when
    /// [`EngineConfig::gpu`] is [`GpuPolicy::Off`].
    gpu_status: Option<GpuStatus>,
}
impl InferenceEngine {
    /// Create a new inference engine with the given configuration.
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config,
            gguf_model: None,
            model_config: None,
            forward_pass: None,
            kv_cache: None,
            tokenizer: None,
            eos_token_id: None,
            metrics: EngineMetrics::new(),
            lora_stack: oxillama_arch::LoraStack::new(),
            layer_pager: None,
            gpu_status: None,
        }
    }
    /// What the GPU is holding for the currently loaded model.
    ///
    /// `None` means no GPU was requested (or no model is loaded).  A `Some`
    /// whose [`GpuStatus::resident_tensors`] is 0 means a device was bound but
    /// nothing met the eligibility rules — see [`crate::gpu_backend`].
    pub fn gpu_status(&self) -> Option<&GpuStatus> {
        self.gpu_status.as_ref()
    }
    /// Forward [`EngineConfig::num_threads`] to the quantization crate's
    /// dedicated GEMV thread pool.
    ///
    /// Called at the start of every model-loading entry point, i.e. before the
    /// first GEMV can possibly run.  The pool is process-wide and built once,
    /// so only the first engine in a process gets to size it; later engines
    /// (and the `0` = auto value) are silently accepted, which is logged at
    /// debug level rather than treated as an error.
    fn apply_thread_config(&self) {
        let requested = self.config.num_threads;
        let accepted = oxillama_quant::parallel::set_num_threads(requested);
        tracing::debug!(
            requested,
            accepted,
            effective = oxillama_quant::parallel::num_threads(),
            "GEMV thread pool configured"
        );
    }
    /// Return a reference to the active layer pager, if offloading is enabled.
    ///
    /// This is the inspection / integration hook that arch-layer code (or
    /// higher-level callers) can use to acquire tensors on demand.  When
    /// the pager is `None`, the engine is running in the default fully-in-RAM
    /// mode.
    pub fn layer_pager(&self) -> Option<&Arc<LayerPager>> {
        self.layer_pager.as_ref()
    }
    /// Attach a pre-built [`LayerPager`] to this engine.
    ///
    /// This is the integration point for callers that construct their own
    /// pager (e.g. from a custom [`PagerSource`][crate::offload::PagerSource])
    /// and want to inject it rather than relying on the engine to build one
    /// automatically from the GGUF file.
    pub fn set_layer_pager(&mut self, pager: Arc<LayerPager>) {
        self.layer_pager = Some(pager);
    }
    /// Load the model from an in-memory GGUF byte buffer.
    ///
    /// This is the preferred entry point for environments that cannot access
    /// the filesystem, such as `wasm32-unknown-unknown`.  The tokenizer must be
    /// provided separately as a JSON string because GGUF metadata rarely
    /// contains the full HuggingFace `tokenizer.json`.
    ///
    /// The loading pipeline is identical to `load_model` except:
    /// - The GGUF data comes from the supplied `model_bytes` slice (copied into
    ///   owned storage inside [`GgufModel::from_bytes`]).
    /// - The tokenizer is loaded from `tokenizer_json` rather than a file path.
    ///
    /// Any `context_size` override from [`EngineConfig`] is still applied.
    #[tracing::instrument(
        skip_all,
        fields(n_bytes = model_bytes.len(), arch = tracing::field::Empty)
    )]
    pub fn load_model_from_bytes(
        &mut self,
        model_bytes: &[u8],
        tokenizer_json: &str,
    ) -> RuntimeResult<()> {
        self.apply_thread_config();
        let gguf = GgufModel::from_bytes(model_bytes.to_vec())?;
        tracing::Span::current().record("arch", gguf.architecture().unwrap_or("unknown"));
        tracing::info!(
            arch = gguf.architecture().unwrap_or("unknown"),
            tensors = gguf.file.header.tensor_count,
            "GGUF file parsed from bytes"
        );
        let mut model_config = ModelConfig::from_metadata(&gguf.file.metadata)?;
        model_config.max_context_length =
            resolve_context_size(self.config.context_size, model_config.max_context_length);
        tracing::info!(
            arch = % model_config.architecture, layers = model_config.num_layers, hidden
            = model_config.hidden_size, heads = model_config.num_attention_heads,
            kv_heads = model_config.num_kv_heads, vocab = model_config.vocab_size, ctx =
            model_config.max_context_length, "model config loaded from bytes"
        );
        let mut forward_pass = build_forward_pass(&gguf, &model_config)?;
        let gpu_status = gpu_backend::apply_gpu_policy(
            forward_pass.as_mut(),
            &self.config.gpu,
            model_config.num_layers,
        )?;
        let kv_dim = model_config.num_kv_heads * model_config.head_dim;
        let kv_cache = KvCache::with_dtype(
            model_config.num_layers,
            model_config.max_context_length,
            kv_dim,
            self.config.kv_dtype,
        );
        tracing::info!(
            layers = model_config.num_layers,
            max_ctx = model_config.max_context_length,
            kv_dim = kv_dim,
            dtype = self.config.kv_dtype.as_str(),
            "KV cache initialized (from-bytes path)"
        );
        let mut tokenizer = TokenizerBridge::from_bytes(tokenizer_json.as_bytes())?;
        // GGUF metadata stays authoritative for BOS/EOS even when the caller
        // supplied a tokenizer.json.
        tokenizer.apply_gguf_specials(&gguf.file.metadata);
        let eos_token_id = tokenizer.eos_token_id();
        tracing::info!(
            vocab_size = tokenizer.vocab_size(), eos = ? eos_token_id, n_eog = tokenizer
            .eog_token_ids().len(), "tokenizer loaded from JSON string"
        );
        self.model_config = Some(model_config);
        self.forward_pass = Some(forward_pass);
        self.kv_cache = Some(kv_cache);
        self.tokenizer = Some(tokenizer);
        self.eos_token_id = eos_token_id;
        self.gguf_model = Some(gguf);
        self.gpu_status = gpu_status;
        Ok(())
    }
    /// Load the model from the configured path.
    ///
    /// This performs the full loading pipeline:
    /// 1. Parse GGUF file (header, metadata, tensor info)
    /// 2. Extract model configuration from metadata
    /// 3. Build the architecture-specific forward pass
    /// 4. Initialize KV cache
    /// 5. Load tokenizer
    #[tracing::instrument(
        skip_all,
        fields(model = %self.config.model_path, arch = tracing::field::Empty)
    )]
    pub fn load_model(&mut self) -> RuntimeResult<()> {
        let path = Path::new(&self.config.model_path);
        if !path.exists() {
            return Err(RuntimeError::ModelLoadError {
                message: format!("model file not found: {}", self.config.model_path),
            });
        }
        self.apply_thread_config();
        tracing::info!(path = % self.config.model_path, "loading GGUF model");
        let gguf = GgufModel::load(&self.config.model_path)?;
        tracing::Span::current().record("arch", gguf.architecture().unwrap_or("unknown"));
        tracing::info!(
            arch = gguf.architecture().unwrap_or("unknown"),
            tensors = gguf.file.header.tensor_count,
            "GGUF file parsed"
        );
        let mut model_config = ModelConfig::from_metadata(&gguf.file.metadata)?;
        model_config.max_context_length =
            resolve_context_size(self.config.context_size, model_config.max_context_length);
        tracing::info!(
            arch = % model_config.architecture, layers = model_config.num_layers, hidden
            = model_config.hidden_size, heads = model_config.num_attention_heads,
            kv_heads = model_config.num_kv_heads, vocab = model_config.vocab_size, ctx =
            model_config.max_context_length, "model config loaded"
        );
        let mut forward_pass = build_forward_pass(&gguf, &model_config)?;
        let gpu_status = gpu_backend::apply_gpu_policy(
            forward_pass.as_mut(),
            &self.config.gpu,
            model_config.num_layers,
        )?;
        let kv_dim = model_config.num_kv_heads * model_config.head_dim;
        let kv_cache = KvCache::with_dtype(
            model_config.num_layers,
            model_config.max_context_length,
            kv_dim,
            self.config.kv_dtype,
        );
        tracing::info!(
            layers = model_config.num_layers,
            max_ctx = model_config.max_context_length,
            kv_dim = kv_dim,
            dtype = self.config.kv_dtype.as_str(),
            "KV cache initialized"
        );
        let tokenizer = load_tokenizer(&self.config, &gguf.file.metadata)?;
        let eos_token_id = tokenizer.eos_token_id();
        tracing::info!(
            vocab_size = tokenizer.vocab_size(), eos = ? eos_token_id, bos = ? tokenizer
            .bos_token_id(), n_eog = tokenizer.eog_token_ids().len(), gguf_backed = tokenizer
            .is_gguf_backed(), "tokenizer loaded"
        );
        if eos_token_id.is_none() && tokenizer.eog_token_ids().is_empty() {
            tracing::warn!(
                "no end-of-generation token could be resolved — generation will always \
                 run to max_tokens"
            );
        }
        self.model_config = Some(model_config);
        self.forward_pass = Some(forward_pass);
        self.kv_cache = Some(kv_cache);
        self.tokenizer = Some(tokenizer);
        self.eos_token_id = eos_token_id;
        self.gguf_model = Some(gguf);
        self.gpu_status = gpu_status;
        Ok(())
    }
    /// Generate tokens from a prompt.
    ///
    /// Runs the full generation pipeline:
    /// 1. Tokenize the prompt
    /// 2. Prefill: process all prompt tokens through the model
    /// 3. Decode: autoregressive generation until EOS or max_tokens
    ///
    /// The callback is invoked with each decoded chunk of text as it's generated.
    /// Chunks are complete UTF-8 sequences, never partial ones.
    ///
    /// # Precondition
    ///
    /// The KV cache is **not** reset by this call.  Call [`Self::reset`] first
    /// unless you intend the prompt to continue the previous sequence.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        callback: impl FnMut(&str),
    ) -> RuntimeResult<String> {
        let config = GenerationConfig {
            max_tokens,
            sampler: self.config.sampler.clone(),
            ..GenerationConfig::default()
        };
        Ok(self.generate_detailed(prompt, &config, callback)?.text)
    }
    /// Generate tokens using an explicit sampler config instead of the engine default.
    ///
    /// This is the preferred entry point for per-request sampler customization
    /// (e.g., grammar-constrained sampling from the API server).
    ///
    /// # Precondition
    ///
    /// The KV cache is **not** reset by this call.  Call [`Self::reset`] first
    /// unless you intend the prompt to continue the previous sequence.
    pub fn generate_with_config(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler_config: SamplerConfig,
        callback: impl FnMut(&str),
    ) -> RuntimeResult<String> {
        let config = GenerationConfig {
            max_tokens,
            sampler: sampler_config,
            ..GenerationConfig::default()
        };
        Ok(self.generate_detailed(prompt, &config, callback)?.text)
    }
    /// Generate from `prompt`, reporting why generation stopped.
    ///
    /// This is the entry point an OpenAI-compatible server wants: the returned
    /// [`GenerationOutcome`] distinguishes natural completion from truncation
    /// (`finish_reason`), names the stop sequence that matched, and reports
    /// prompt/completion token counts for the `usage` block.  Stop sequences
    /// ([`GenerationConfig::stop`]) are only reachable through this method.
    ///
    /// Prefill is batched according to [`EngineConfig::prefill_chunk_size`].
    ///
    /// # Precondition
    ///
    /// The KV cache is **not** reset by this call.  Call [`Self::reset`] first
    /// unless you intend the prompt to continue the previous sequence.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded, or any
    /// tokenizer / forward-pass error.
    pub fn generate_detailed(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        callback: impl FnMut(&str),
    ) -> RuntimeResult<GenerationOutcome> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let prompt_tokens =
            tokenizer.encode_with(prompt, config.add_special, config.parse_special)?;
        self.generate_detailed_with_tokens(prompt_tokens, config, callback)
    }
    /// Generate from an already-tokenized prompt, reporting why generation stopped.
    ///
    /// Identical to [`Self::generate_detailed`] except that the caller supplies
    /// `prompt_tokens` directly instead of a prompt string — [`GenerationConfig::add_special`]
    /// and [`GenerationConfig::parse_special`] are therefore not consulted, since there is no
    /// text left to tokenize. This is the entry point for callers that need control over BOS/EOS
    /// insertion finer than that single boolean can express (e.g. forcing a BOS token onto a
    /// tokenizer whose own policy would not add one, which requires inspecting and possibly
    /// editing the id sequence before the forward pass ever runs).
    ///
    /// # Precondition
    ///
    /// The KV cache is **not** reset by this call.  Call [`Self::reset`] first
    /// unless you intend the prompt to continue the previous sequence.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded, or any
    /// forward-pass error.
    pub fn generate_detailed_with_tokens(
        &mut self,
        prompt_tokens: Vec<u32>,
        config: &GenerationConfig,
        mut callback: impl FnMut(&str),
    ) -> RuntimeResult<GenerationOutcome> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        if prompt_tokens.is_empty() {
            return Ok(GenerationOutcome {
                text: String::new(),
                finish_reason: FinishReason::Eos,
                generated_tokens: Vec::new(),
                prompt_tokens: 0,
                stop_sequence: None,
            });
        }
        tracing::debug!(n_tokens = prompt_tokens.len(), "prompt tokenized");
        let logits = prefill_chunked(
            forward_pass.as_mut(),
            kv_cache,
            &self.metrics,
            &prompt_tokens,
            self.config.prefill_chunk_size,
        )?;
        let mut recent_tokens = prompt_tokens;
        run_decode_loop(
            DecodeContext {
                forward_pass: forward_pass.as_mut(),
                kv_cache,
                tokenizer,
                metrics: &self.metrics,
            },
            config,
            logits,
            &mut recent_tokens,
            &mut callback,
        )
    }
    /// Build the vocabulary byte table, used for grammar-constrained sampling.
    ///
    /// Returns `None` if no tokenizer is loaded.
    pub fn vocab_bytes(&self) -> Option<Vec<(u32, Vec<u8>)>> {
        self.tokenizer.as_ref().map(|t| t.vocab_bytes())
    }
    /// Apply a loaded LoRA adapter to the model's linear layers.
    ///
    /// Delegates to the architecture-specific [`ForwardPass::apply_lora`]
    /// implementation, which walks the model's layers and attaches
    /// [`LoraAdapter`](oxillama_quant::LoraAdapter) instances to each
    /// matching `QuantLinear` field.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model has been loaded.
    pub fn apply_lora_adapters(
        &mut self,
        lora: &oxillama_arch::lora::LoadedLora,
    ) -> RuntimeResult<()> {
        let fp = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        fp.apply_lora(lora).map_err(RuntimeError::Arch)?;
        Ok(())
    }
    /// Push a LoRA adapter onto the stack with a per-entry scale multiplier.
    ///
    /// The adapter is applied additively during inference:
    /// `output += scale · (alpha/rank) · B @ A @ input`
    pub fn push_lora(&mut self, lora: std::sync::Arc<oxillama_arch::lora::LoadedLora>, scale: f32) {
        self.lora_stack.push(lora, scale);
    }
    /// Remove the last adapter pushed onto the stack.
    ///
    /// Returns `None` if the stack is empty.
    pub fn pop_lora(&mut self) -> Option<(std::sync::Arc<oxillama_arch::lora::LoadedLora>, f32)> {
        self.lora_stack.pop()
    }
    /// Remove all LoRA adapters from the stack.
    pub fn clear_loras(&mut self) {
        self.lora_stack.clear();
    }
    /// Inspect the current LoRA stack.
    pub fn lora_stack(&self) -> &oxillama_arch::LoraStack {
        &self.lora_stack
    }
    /// Apply the stacked LoRA adapters to the loaded model's linear layers.
    ///
    /// This is a hot-swap operation: it can be called at any time without
    /// reloading the model.  If the stack is empty this is a no-op.
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model has been loaded.
    pub fn apply_lora_stack(&mut self) -> RuntimeResult<()> {
        if self.lora_stack.is_empty() {
            return Ok(());
        }
        let fp = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        fp.apply_lora_stack(&self.lora_stack)
            .map_err(RuntimeError::Arch)?;
        Ok(())
    }
    /// Remove all LoRA adapters from the loaded model's linear layers.
    ///
    /// Clears the `lora_stack` and calls `unapply_all_loras` on the forward
    /// pass so every `QuantLinear.lora` field is set back to `None`.
    ///
    /// This is the necessary counterpart to `apply_lora_stack` for per-request
    /// LoRA hot-swap: push adapters, apply, generate, then unapply.
    ///
    /// Does nothing when no model is loaded.
    pub fn unapply_all_loras(&mut self) {
        self.lora_stack.clear();
        if let Some(fp) = self.forward_pass.as_mut() {
            fp.unapply_all_loras();
        }
    }
    /// Restore the KV cache from a cached prefix snapshot and run prefill for
    /// the suffix tokens that follow the cached prefix.
    ///
    /// This is the prefix-KV-cache fast path: instead of re-prefilling the
    /// entire prompt from scratch, the engine restores the KV state for the
    /// longest matching cached prefix and only runs the forward pass for the
    /// remaining suffix tokens.
    ///
    /// Restriction: the cached prefix must start at position 0 (i.e. it was
    /// stored from the beginning of a sequence).  This matches how
    /// `PrefixKvCache::store` snapshots KV state.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded.
    pub fn prime_with_prefix(
        &mut self,
        cached: &crate::kv_cache::prefix::CachedKvState,
        restore_to: usize,
        suffix_tokens: &[u32],
    ) -> RuntimeResult<Vec<f32>> {
        if suffix_tokens.is_empty() {
            return Err(RuntimeError::ModelLoadError {
                message: "prime_with_prefix: suffix_tokens must contain at least one token"
                    .to_string(),
            });
        }
        // A primed prefix begins a NEW sequence: whatever per-architecture
        // state the previous request left behind (DeepSeek's MLA latent cache,
        // Mamba-2 / Jamba's SSM hidden state, DBRX / Grok's position counter)
        // must go before the restored KV is used, or the two sequences blend.
        if let Some(fp) = self.forward_pass.as_mut() {
            fp.reset_sequence();
        }
        {
            let kv = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
            kv.restore_from_snapshot(cached.keys(), cached.values(), restore_to)?;
        }
        // A restored prefix is precisely a KV-cache hit; the engine's own
        // counters used to record neither hits nor misses anywhere, leaving
        // `MetricsSnapshot::kv_cache_hit_rate` pinned at 0.0 even when the
        // prefix cache was doing all the work.
        self.metrics.record_kv_hit();
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        let logits = forward_pass
            .forward(suffix_tokens, kv)
            .map_err(RuntimeError::Arch)?;
        Ok(logits)
    }
    /// Run the autoregressive decode loop starting from pre-computed logits.
    ///
    /// Unlike [`Self::generate_with_config`], this does **not** run prefill — the
    /// caller must have already primed the KV cache (via [`Self::prime_with_prefix`]
    /// or a full prefill) and obtained the initial logits from that step.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded.
    pub fn generate_with_logits(
        &mut self,
        prompt_tokens: &[u32],
        initial_logits: Vec<f32>,
        max_tokens: usize,
        sampler_config: SamplerConfig,
        callback: impl FnMut(&str),
    ) -> RuntimeResult<String> {
        let config = GenerationConfig {
            max_tokens,
            sampler: sampler_config,
            ..GenerationConfig::default()
        };
        Ok(self
            .generate_with_logits_detailed(prompt_tokens, initial_logits, &config, callback)?
            .text)
    }
    /// [`Self::generate_with_logits`] reporting a [`FinishReason`].
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded.
    pub fn generate_with_logits_detailed(
        &mut self,
        prompt_tokens: &[u32],
        initial_logits: Vec<f32>,
        config: &GenerationConfig,
        mut callback: impl FnMut(&str),
    ) -> RuntimeResult<GenerationOutcome> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        let mut recent_tokens: Vec<u32> = prompt_tokens.to_vec();
        run_decode_loop(
            DecodeContext {
                forward_pass: forward_pass.as_mut(),
                kv_cache,
                tokenizer,
                metrics: &self.metrics,
            },
            config,
            initial_logits,
            &mut recent_tokens,
            &mut callback,
        )
    }
    /// Returns whether a model is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.forward_pass.is_some()
    }
    /// Returns the engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
    /// Returns the model configuration, if loaded.
    pub fn model_config(&self) -> Option<&ModelConfig> {
        self.model_config.as_ref()
    }
    /// Returns a shared reference to the KV cache, if a model is loaded.
    pub(crate) fn kv_cache_ref(&self) -> Option<&KvCache> {
        self.kv_cache.as_ref()
    }
    /// Returns a mutable reference to the KV cache, if a model is loaded.
    pub(crate) fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        self.kv_cache.as_mut()
    }
    /// Store the current KV cache state into a `PrefixKvCache` under `tokens`.
    ///
    /// This is the public integration point for server-side prefix caching:
    /// after a successful generation pass the worker calls this to persist the
    /// KV state so future requests sharing the same prefix can skip prefill.
    ///
    /// Returns `true` when an entry was stored.  If no model is loaded (KV
    /// cache absent) the call is a silent no-op returning `false`.
    ///
    /// The snapshot [`PrefixKvCache::store`] writes is truncated to
    /// `tokens.len()` regardless of what this call passes as its `seq_len`
    /// argument. That argument is `kv.seq_len()` — after a decode loop, the
    /// prompt **plus every generated token** — which `store` only uses as a
    /// floor check (refusing a KV state shorter than the trie key), never as
    /// the stored length; passing the inflated value here is what a prior
    /// version of `store` stored verbatim, retaining memory for a completion
    /// the trie key never mentions. See [`PrefixKvCache::store`] for the
    /// invariant it now enforces regardless of this caller.
    ///
    /// [`PrefixKvCache::store`]: crate::kv_cache::prefix::PrefixKvCache::store
    pub fn store_kv_in_prefix_cache(
        &mut self,
        tokens: &[u32],
        prefix_cache: &mut crate::kv_cache::prefix::PrefixKvCache,
    ) -> bool {
        let Some(kv) = self.kv_cache.as_mut() else {
            return false;
        };
        let seq_len = kv.seq_len();
        let kv_dim = kv.kv_dim();
        let num_layers = kv.num_layers();
        prefix_cache.store(tokens, kv, seq_len, kv_dim, num_layers)
    }
    /// Record that a prefix-cache lookup found nothing usable.
    ///
    /// The hit side is recorded by [`Self::prime_with_prefix`]; a miss is only
    /// visible to the caller that performed the lookup, so it reports it here.
    /// Without both, `MetricsSnapshot::kv_cache_hit_rate` is permanently 0.0.
    pub fn record_kv_cache_miss(&self) {
        self.metrics.record_kv_miss();
    }
    /// Reset the KV cache and all per-sequence architecture state.
    ///
    /// This is the "start a new conversation" boundary.  Clearing the KV cache
    /// alone is not enough: architectures that own *internal* per-sequence
    /// state — DeepSeek's `MlaLatentCache`, Mamba-2's and Jamba's SSM hidden
    /// state `h`, DBRX's and Grok's `current_pos` — otherwise carry it across
    /// the reset and leak the previous request into the next one.
    /// `ForwardPass::reset_sequence` defaults to a no-op, so stateless
    /// architectures pay nothing.
    pub fn reset(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            cache.clear();
        }
        if let Some(ref mut fp) = self.forward_pass {
            fp.reset_sequence();
        }
    }
    /// Tokenize text and return token IDs.
    ///
    /// Requires that a model (and thus a tokenizer) has been loaded.
    pub fn tokenize(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        tokenizer.encode(text)
    }
    /// Tokenize text with explicit control over special-token handling.
    ///
    /// [`Self::tokenize`] is `tokenize_with(text, true, true)`. Callers that
    /// pre-render a chat template emitting its own literal BOS marker
    /// (`<|begin_of_text|>`, `<s>` — see
    /// [`ChatTemplate::emits_literal_bos`]) must pass `add_special = false`
    /// so the tokenizer does not add a second one. Getting this wrong in a
    /// prefix-cache lookup is worse than getting it wrong once: the cached
    /// token sequence would then disagree with the sequence the decode path
    /// actually prefills, and the restored KV state would belong to a
    /// different prompt.
    ///
    /// Requires that a model (and thus a tokenizer) has been loaded.
    pub fn tokenize_with(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> RuntimeResult<Vec<u32>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        tokenizer.encode_with(text, add_special, parse_special)
    }
    /// Prefill the KV cache with the given token sequence without returning logits.
    ///
    /// Processes tokens in batches of [`EngineConfig::prefill_chunk_size`],
    /// updating the KV cache at each position.  The last token's logits are
    /// discarded; callers typically follow up with `forward_one` to begin
    /// autoregressive generation.
    #[tracing::instrument(skip_all, fields(n_tokens = tokens.len()))]
    pub fn prefill(&mut self, tokens: &[u32]) -> RuntimeResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        prefill_chunked(
            forward_pass.as_mut(),
            kv_cache,
            &self.metrics,
            tokens,
            self.config.prefill_chunk_size,
        )?;
        Ok(())
    }
    /// Run a batched prefill forward pass for the given chunk of tokens.
    ///
    /// This is the per-chunk entry point for the chunked-prefill scheduler
    /// fairness path (A3).  It differs from `prefill` in two ways:
    ///
    /// 1. It accepts a multi-token slice and dispatches a *single* batched
    ///    forward call, matching the `generate` path's chunked prefill logic.
    /// 2. It returns the logits of the last token in the chunk so that the
    ///    caller can immediately begin decode sampling if `pos_end` equals the
    ///    full prompt length.
    ///
    /// `pos_start` is the KV-cache position at which this chunk begins.  It
    /// must equal the current `kv_cache.seq_len()` on entry; the parameter is
    /// provided explicitly so that callers (e.g. the scheduler) can assert the
    /// invariant in debug builds.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded, or
    /// any arch-level error from the forward pass.
    #[tracing::instrument(skip_all, fields(n_tokens = tokens.len(), pos_start))]
    pub fn forward_prefill(&mut self, tokens: &[u32], pos_start: usize) -> RuntimeResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(RuntimeError::ModelLoadError {
                message: "forward_prefill called with empty token slice".to_string(),
            });
        }
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        debug_assert_eq!(
            kv_cache.seq_len(),
            pos_start,
            "forward_prefill: pos_start ({pos_start}) must equal kv_cache.seq_len() ({})",
            kv_cache.seq_len(),
        );
        let logits = forward_pass.forward(tokens, kv_cache)?;
        Ok(logits)
    }
    /// Run a single autoregressive decode step for `token` and return logits.
    ///
    /// This is the per-step entry point for the chunked-prefill scheduler
    /// fairness path (A3).  It is semantically equivalent to `forward_one`
    /// but named differently to make the prefill/decode distinction explicit
    /// in call sites inside the engine and scheduler integration layer.
    ///
    /// `pos` is the current sequence position (= `kv_cache.seq_len()`).  It
    /// is accepted as a parameter so that callers can assert the invariant.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded.
    #[tracing::instrument(skip_all, fields(token, pos))]
    pub fn forward_decode(&mut self, token: u32, pos: usize) -> RuntimeResult<Vec<f32>> {
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        debug_assert_eq!(
            kv_cache.seq_len(),
            pos,
            "forward_decode: pos ({pos}) must equal kv_cache.seq_len() ({})",
            kv_cache.seq_len(),
        );
        let logits = forward_pass.forward(&[token], kv_cache)?;
        Ok(logits)
    }
    /// Run a single forward pass for `token` and return raw logits.
    ///
    /// The KV cache is updated (one position advanced).
    pub fn forward_one(&mut self, token: u32) -> RuntimeResult<Vec<f32>> {
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        let logits = forward_pass.forward(&[token], kv_cache)?;
        Ok(logits)
    }
    /// Returns `true` if `token` ends generation for this model.
    ///
    /// This tests membership in the full end-of-generation set rather than
    /// equality with a single id: Llama-3-Instruct ends a turn with
    /// `<|eot_id|>` while its `eos_token_id` is `<|end_of_text|>`, and Qwen
    /// ends with `<|im_end|>`.
    pub fn is_eos(&self, token: u32) -> bool {
        match self.tokenizer.as_ref() {
            Some(tokenizer) => tokenizer.is_eog(token),
            None => self.eos_token_id == Some(token),
        }
    }
    /// The primary EOS token id, if the model declares one.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
    /// Every token that ends generation for this model.
    pub fn eog_token_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .tokenizer
            .as_ref()
            .map(|t| t.eog_token_ids().iter().copied().collect())
            .unwrap_or_default();
        ids.sort_unstable();
        ids
    }
    /// The tokenizer bridge, if a model is loaded.
    pub fn tokenizer(&self) -> Option<&TokenizerBridge> {
        self.tokenizer.as_ref()
    }
    /// Resolve the loaded model's chat template family from its own GGUF
    /// metadata (`tokenizer.chat_template`, falling back to a vocabulary
    /// fingerprint — see [`ChatTemplate::resolve`]).
    ///
    /// Returns `None` when no model is loaded. Callers that want a usable
    /// value regardless (chat template selection is a UX nicety, not a
    /// precondition) can `unwrap_or_default()` into
    /// [`ChatTemplate::ChatMl`].
    ///
    /// This reads the GGUF metadata the engine already holds, so it costs no
    /// additional file I/O — unlike [`ChatTemplate::resolve_from_path`],
    /// which re-opens the file.
    pub fn chat_template(&self) -> Option<ChatTemplate> {
        self.gguf_model
            .as_ref()
            .map(|g| ChatTemplate::resolve(&g.file.metadata))
    }
    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, token: u32) -> RuntimeResult<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        tokenizer.decode(&[token])
    }
    /// Returns a shared reference to the engine's live metrics counters.
    pub fn metrics(&self) -> Arc<EngineMetrics> {
        Arc::clone(&self.metrics)
    }
    /// Returns a point-in-time [`MetricsSnapshot`] of the engine's counters.
    pub fn metrics_snapshot(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }
    /// Capture a [`KvCacheSnapshot`] from the current KV cache state.
    ///
    /// Returns `None` if no model (and thus no KV cache) is loaded.
    pub fn kv_snapshot(&self) -> Option<KvCacheSnapshot> {
        self.kv_cache.as_ref().map(|c| c.snapshot())
    }
    /// Restore the KV cache state from a previously captured [`KvCacheSnapshot`].
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded.
    pub fn kv_restore(&mut self, snapshot: &KvCacheSnapshot) -> RuntimeResult<()> {
        let kv = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        kv.restore_from_snapshot(&snapshot.keys, &snapshot.values, snapshot.seq_len)
    }
    /// Truncate the KV cache to `n` tokens.
    ///
    /// After this call the engine behaves as if only `n` tokens have been
    /// processed.  This is the low-level primitive used by speculative
    /// decoding on divergence rollback.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model is loaded.
    pub fn truncate(&mut self, n: usize) -> RuntimeResult<()> {
        let kv = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        kv.truncate(n);
        Ok(())
    }
    /// Return the current KV cache sequence length.
    ///
    /// Returns 0 if no model is loaded.
    pub fn kv_cache_seq_len(&self) -> usize {
        self.kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0)
    }
    /// Returns the model's hidden state dimension, if a model is loaded.
    pub fn hidden_size(&self) -> Option<usize> {
        self.model_config.as_ref().map(|c| c.hidden_size)
    }
    /// Compute a semantic embedding vector for the given text using `PoolingMode::Last`.
    ///
    /// This is a convenience wrapper around [`Self::embed_with`]. Runs tokenization →
    /// full transformer layers → final RMSNorm, then L2-normalises the resulting
    /// `hidden_size`-dimensional vector. The KV cache is reset before the pass
    /// so that embeddings for different inputs are independent of each other.
    ///
    /// Returns `RuntimeError::ModelNotLoaded` if no model has been loaded.
    pub fn embed(&mut self, text: &str) -> RuntimeResult<Vec<f32>> {
        self.embed_with(text, PoolingMode::Last)
    }
    /// Compute a semantic embedding vector for the given text using the specified
    /// pooling strategy.
    ///
    /// Runs tokenization → full transformer layers → final RMSNorm → pooling,
    /// then L2-normalises the resulting `hidden_size`-dimensional vector.
    /// The KV cache is reset before the pass so that embeddings for different
    /// inputs are independent of each other.
    ///
    /// # Pooling modes
    ///
    /// * [`PoolingMode::Last`] — last token hidden state (causal / decoder models).
    /// * [`PoolingMode::Mean`] — mean across all token positions.
    /// * [`PoolingMode::Max`]  — elementwise max across all token positions.
    /// * [`PoolingMode::Cls`]  — first token hidden state (BERT / encoder models).
    ///
    /// Returns `RuntimeError::ModelNotLoaded` if no model has been loaded.
    pub fn embed_with(&mut self, text: &str, mode: PoolingMode) -> RuntimeResult<Vec<f32>> {
        self.reset();
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        let tokens = {
            let tok = self
                .tokenizer
                .as_ref()
                .ok_or(RuntimeError::ModelNotLoaded)?;
            tok.encode(text)?
        };
        if tokens.is_empty() {
            let dim = self
                .model_config
                .as_ref()
                .map(|c| c.hidden_size)
                .unwrap_or(0);
            return Ok(vec![0.0f32; dim]);
        }
        let seq_len = tokens.len();
        let hidden_size = forward_pass.hidden_size();
        let all_hidden = forward_pass.embed_all(&tokens, kv_cache);
        let hidden = match all_hidden {
            Ok(states) if states.len() == seq_len * hidden_size && seq_len > 0 => {
                pool_hidden_states(&states, seq_len, hidden_size, mode)?
            }
            _ => forward_pass.embed(&tokens, kv_cache)?,
        };
        let norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            Ok(hidden.into_iter().map(|x| x / norm).collect())
        } else {
            Ok(hidden)
        }
    }
    /// Extract embedding vectors for multiple input texts using `PoolingMode::Last`.
    ///
    /// This is a convenience wrapper around [`Self::embed_batch_with`].
    /// Each text is processed independently with a fresh KV cache.
    pub fn embed_batch(&mut self, texts: &[String]) -> RuntimeResult<Vec<Vec<f32>>> {
        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embed_batch_with(&str_refs, PoolingMode::Last)
    }
    /// Extract embedding vectors for multiple input texts using the specified
    /// pooling strategy.
    ///
    /// Each text is processed independently with a fresh KV cache.
    /// The output order matches the input order.
    ///
    /// Returns `RuntimeError::ModelNotLoaded` if no model has been loaded.
    pub fn embed_batch_with(
        &mut self,
        texts: &[&str],
        mode: PoolingMode,
    ) -> RuntimeResult<Vec<Vec<f32>>> {
        {
            let _fp = self
                .forward_pass
                .as_ref()
                .ok_or(RuntimeError::ModelNotLoaded)?;
            let _tok = self
                .tokenizer
                .as_ref()
                .ok_or(RuntimeError::ModelNotLoaded)?;
        }
        let mut embeddings = Vec::with_capacity(texts.len());
        for &text in texts {
            embeddings.push(self.embed_with(text, mode)?);
        }
        Ok(embeddings)
    }
}
/// Build the forward pass from a loaded GGUF model.
///
/// Routes through [`ArchitectureRegistry`], which is the single place
/// architectures declare themselves.  Until 0.1.4 this was a hard-coded
/// `match` over seven names while the registry already knew 26 — every other
/// architecture in `oxillama-arch` was unreachable from the engine no matter
/// how complete its implementation was, and reported as
/// `unsupported architecture: '<name>'` as if it did not exist.
///
/// Three failure modes are now distinguished:
///
/// * **not registered** — [`ArchError::UnknownArchitecture`], typically a GGUF
///   naming an architecture this build was compiled without.  The message
///   lists what *is* available so a missing Cargo feature is obvious.
/// * **registered but not loadable** — [`ArchError::NotSupported`] from the
///   default [`ModelArchitecture::build_from_gguf`], naming the architecture
///   that has not migrated its loader.
/// * **loadable but the file is wrong** — the architecture's own error.
///
/// Feature gating is unchanged: `with_builtins()` registers each architecture
/// under its own `#[cfg(feature = ...)]`, so a disabled architecture is simply
/// absent from the registry and produces the first error above rather than a
/// link failure.
fn build_forward_pass(
    gguf: &GgufModel,
    config: &ModelConfig,
) -> RuntimeResult<Box<dyn ForwardPass>> {
    let registry = ArchitectureRegistry::with_builtins();
    let arch = registry.get(&config.architecture).map_err(|e| match e {
        oxillama_arch::ArchError::UnknownArchitecture { arch_id } => {
            let mut available = registry.list();
            available.sort_unstable();
            RuntimeError::ModelLoadError {
                message: format!(
                    "unsupported architecture: '{arch_id}' — this build knows: {}",
                    available.join(", ")
                ),
            }
        }
        other => RuntimeError::Arch(other),
    })?;
    tracing::debug!(
        arch = %config.architecture,
        resolved = arch.arch_id(),
        "architecture resolved through the registry"
    );
    match arch.build_from_gguf(gguf, config) {
        // Registered, but its loader has not migrated to the plugin trait yet.
        // Fall back to the direct loader so architectures that worked before
        // the registry landed keep working.  Each arm here disappears as its
        // owner overrides `build_from_gguf`.
        Err(oxillama_arch::ArchError::NotSupported { .. }) => {
            tracing::debug!(
                arch = %config.architecture,
                "architecture has no build_from_gguf override; using the direct loader"
            );
            build_forward_pass_direct(gguf, config)
        }
        other => Ok(other?),
    }
}

/// Direct per-architecture loaders, for architectures that have not yet
/// overridden [`ModelArchitecture::build_from_gguf`].
///
/// This is a transitional path, not the intended one — see
/// [`build_forward_pass`].  It exists because switching the engine to the
/// registry before the per-architecture overrides existed took every
/// previously-working architecture offline.
fn build_forward_pass_direct(
    gguf: &GgufModel,
    config: &ModelConfig,
) -> RuntimeResult<Box<dyn ForwardPass>> {
    match config.architecture.as_str() {
        #[cfg(feature = "llama")]
        "llama" => Ok(Box::new(oxillama_arch::llama::load_llama_from_gguf(
            gguf, config,
        )?)),
        #[cfg(feature = "qwen3")]
        "qwen3" => Ok(Box::new(oxillama_arch::qwen3::load_qwen3_from_gguf(
            gguf, config,
        )?)),
        #[cfg(feature = "mistral")]
        "mistral" => Ok(Box::new(oxillama_arch::mistral::load_mistral_from_gguf(
            gguf, config,
        )?)),
        #[cfg(feature = "gemma")]
        "gemma" | "gemma2" | "gemma3" => Ok(Box::new(oxillama_arch::gemma::load_gemma_from_gguf(
            gguf, config,
        )?)),
        #[cfg(feature = "phi")]
        "phi3" | "phi" => Ok(Box::new(oxillama_arch::phi::load_phi_from_gguf(
            gguf, config,
        )?)),
        #[cfg(feature = "command-r")]
        "command-r" => Ok(Box::new(
            oxillama_arch::command_r::load_command_r_from_gguf(gguf, config)?,
        )),
        #[cfg(feature = "starcoder")]
        "starcoder" => Ok(Box::new(
            oxillama_arch::starcoder::load_starcoder_from_gguf(gguf, config)?,
        )),
        arch => Err(RuntimeError::ModelLoadError {
            message: format!(
                "architecture '{arch}' is registered but has neither a \
                 build_from_gguf() override nor a direct loader"
            ),
        }),
    }
}
/// Resolve the effective context length.
///
/// An explicit [`EngineConfig::context_size`] always wins.  Otherwise the
/// training context from GGUF metadata is clamped to [`DEFAULT_MAX_CONTEXT`],
/// because models routinely advertise a context whose KV cache does not fit in
/// memory (Llama-3.1-8B reports 131072, which is ~34 GB of cache).
fn resolve_context_size(requested: Option<usize>, n_ctx_train: usize) -> usize {
    match requested {
        Some(0) | None => {
            let chosen = n_ctx_train.clamp(1, DEFAULT_MAX_CONTEXT);
            if chosen < n_ctx_train {
                tracing::info!(
                    n_ctx_train,
                    n_ctx = chosen,
                    "context length clamped to the default bound; \
                     set EngineConfig::context_size to raise it"
                );
            } else {
                tracing::info!(n_ctx = chosen, "context length");
            }
            chosen
        }
        Some(ctx) => {
            if ctx > n_ctx_train {
                tracing::warn!(
                    n_ctx = ctx,
                    n_ctx_train,
                    "requested context exceeds the model's training context"
                );
            } else {
                tracing::info!(n_ctx = ctx, "context length (explicit)");
            }
            ctx
        }
    }
}

/// Load the tokenizer, trying each source in priority order.
///
/// 1. An explicit `--tokenizer` path.
/// 2. The vocabulary embedded in the GGUF file itself (`tokenizer.ggml.*`).
/// 3. A `tokenizer.json` sidecar next to the model.
///
/// The GGUF-embedded vocabulary deliberately outranks the sidecar: it is the
/// vocabulary the weights were quantized against, it always agrees with the
/// model's special-token ids, and it decodes byte-exactly.  A sidecar that
/// happens to sit in the model directory is no longer allowed to shadow it.
///
/// Whichever source wins, `tokenizer.ggml.*_token_id` metadata is overlaid on
/// top so BOS/EOS are always the model's own.
fn load_tokenizer(
    config: &EngineConfig,
    metadata: &MetadataStore,
) -> RuntimeResult<TokenizerBridge> {
    if let Some(ref path) = config.tokenizer_path {
        tracing::info!(path, "loading tokenizer from the configured path");
        let mut bridge = TokenizerBridge::from_file(path)?;
        bridge.apply_gguf_specials(metadata);
        return Ok(bridge);
    }

    if TokenizerBridge::metadata_has_vocab(metadata) {
        match TokenizerBridge::from_gguf_metadata(metadata) {
            Ok(bridge) => return Ok(bridge),
            Err(e) => tracing::warn!(
                error = %e,
                "GGUF-embedded vocabulary could not be used; falling back to a sidecar"
            ),
        }
    }

    let model_dir = Path::new(&config.model_path)
        .parent()
        .unwrap_or(Path::new("."));
    let sidecar = model_dir.join("tokenizer.json");
    if sidecar.exists() {
        tracing::info!(path = %sidecar.display(), "loading tokenizer.json sidecar");
        // `from_path` avoids the old `to_str().unwrap_or("tokenizer.json")`,
        // which silently retried a CWD-relative path for non-UTF-8 directories.
        let mut bridge = TokenizerBridge::from_path(&sidecar)?;
        bridge.apply_gguf_specials(metadata);
        return Ok(bridge);
    }

    Err(RuntimeError::TokenizerError {
        message: format!(
            "no tokenizer found: the GGUF has no '{}' metadata and no tokenizer.json \
             exists next to the model — pass an explicit tokenizer path",
            crate::gguf_vocab::KEY_TOKENS
        ),
    })
}

#[cfg(test)]
mod tests;

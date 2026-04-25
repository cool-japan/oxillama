//! Main inference engine — orchestrates model loading and text generation.

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

use oxillama_arch::config::ModelConfig;
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::GgufModel;

use crate::error::{RuntimeError, RuntimeResult};
use crate::kv_cache::{KvCache, KvCacheSnapshot};
use crate::metrics::{EngineMetrics, MetricsSnapshot};
use crate::offload::{LayerPager, OffloadPolicy};
use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer_bridge::TokenizerBridge;
use std::sync::Arc;
use std::time::Instant;

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Path to the tokenizer JSON file (if not embedded in GGUF).
    pub tokenizer_path: Option<String>,
    /// Context size override (None = use model default).
    pub context_size: Option<usize>,
    /// Number of threads for parallel computation.
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: None,
            context_size: None,
            num_threads: 4,
            sampler: SamplerConfig::default(),
            prefill_chunk_size: 512,
            offload_policy: OffloadPolicy::None,
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
        }
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
    pub fn load_model_from_bytes(
        &mut self,
        model_bytes: &[u8],
        tokenizer_json: &str,
    ) -> RuntimeResult<()> {
        // ── Step 1: Parse GGUF from owned bytes ──────────────────────────────
        let gguf = GgufModel::from_bytes(model_bytes.to_vec())?;
        tracing::info!(
            arch = gguf.architecture().unwrap_or("unknown"),
            tensors = gguf.file.header.tensor_count,
            "GGUF file parsed from bytes"
        );

        // ── Step 2: Extract model configuration ──────────────────────────────
        let mut model_config = ModelConfig::from_metadata(&gguf.file.metadata)?;
        if let Some(ctx) = self.config.context_size {
            model_config.max_context_length = ctx;
        }

        tracing::info!(
            arch = %model_config.architecture,
            layers = model_config.num_layers,
            hidden = model_config.hidden_size,
            heads = model_config.num_attention_heads,
            kv_heads = model_config.num_kv_heads,
            vocab = model_config.vocab_size,
            ctx = model_config.max_context_length,
            "model config loaded from bytes"
        );

        // ── Step 3: Build forward pass ────────────────────────────────────────
        let forward_pass = build_forward_pass(&gguf, &model_config)?;

        // ── Step 4: KV cache ──────────────────────────────────────────────────
        let kv_dim = model_config.num_kv_heads * model_config.head_dim;
        let kv_cache = KvCache::new(
            model_config.num_layers,
            model_config.max_context_length,
            kv_dim,
        );
        tracing::info!(
            layers = model_config.num_layers,
            max_ctx = model_config.max_context_length,
            kv_dim = kv_dim,
            "KV cache initialized (from-bytes path)"
        );

        // ── Step 5: Tokenizer from JSON string ────────────────────────────────
        let tokenizer = TokenizerBridge::from_bytes(tokenizer_json.as_bytes())?;
        let eos_token_id = tokenizer.eos_token_id();
        tracing::info!(
            vocab_size = tokenizer.vocab_size(),
            eos = ?eos_token_id,
            "tokenizer loaded from JSON string"
        );

        self.model_config = Some(model_config);
        self.forward_pass = Some(forward_pass);
        self.kv_cache = Some(kv_cache);
        self.tokenizer = Some(tokenizer);
        self.eos_token_id = eos_token_id;
        self.gguf_model = Some(gguf);

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
    pub fn load_model(&mut self) -> RuntimeResult<()> {
        let path = Path::new(&self.config.model_path);
        if !path.exists() {
            return Err(RuntimeError::ModelLoadError {
                message: format!("model file not found: {}", self.config.model_path),
            });
        }

        tracing::info!(path = %self.config.model_path, "loading GGUF model");

        // Step 1: Load and parse GGUF
        let gguf = GgufModel::load(&self.config.model_path)?;
        tracing::info!(
            arch = gguf.architecture().unwrap_or("unknown"),
            tensors = gguf.file.header.tensor_count,
            "GGUF file parsed"
        );

        // Step 2: Extract model config from metadata
        let mut model_config = ModelConfig::from_metadata(&gguf.file.metadata)?;

        // Apply context size override
        if let Some(ctx) = self.config.context_size {
            model_config.max_context_length = ctx;
        }

        tracing::info!(
            arch = %model_config.architecture,
            layers = model_config.num_layers,
            hidden = model_config.hidden_size,
            heads = model_config.num_attention_heads,
            kv_heads = model_config.num_kv_heads,
            vocab = model_config.vocab_size,
            ctx = model_config.max_context_length,
            "model config loaded"
        );

        // Step 3: Build architecture-specific forward pass
        let forward_pass = build_forward_pass(&gguf, &model_config)?;

        // Step 4: Initialize KV cache
        let kv_dim = model_config.num_kv_heads * model_config.head_dim;
        let kv_cache = KvCache::new(
            model_config.num_layers,
            model_config.max_context_length,
            kv_dim,
        );
        tracing::info!(
            layers = model_config.num_layers,
            max_ctx = model_config.max_context_length,
            kv_dim = kv_dim,
            "KV cache initialized"
        );

        // Step 5: Load tokenizer
        let tokenizer = load_tokenizer(&self.config, &gguf)?;
        let eos_token_id = tokenizer.eos_token_id();
        tracing::info!(
            vocab_size = tokenizer.vocab_size(),
            eos = ?eos_token_id,
            "tokenizer loaded"
        );

        self.model_config = Some(model_config);
        self.forward_pass = Some(forward_pass);
        self.kv_cache = Some(kv_cache);
        self.tokenizer = Some(tokenizer);
        self.eos_token_id = eos_token_id;
        self.gguf_model = Some(gguf);

        Ok(())
    }

    /// Generate tokens from a prompt.
    ///
    /// Runs the full generation pipeline:
    /// 1. Tokenize the prompt
    /// 2. Prefill: process all prompt tokens through the model
    /// 3. Decode: autoregressive generation until EOS or max_tokens
    ///
    /// The callback is invoked with each decoded token's text as it's generated.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut callback: impl FnMut(&str),
    ) -> RuntimeResult<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;

        // Step 1: Tokenize prompt
        let prompt_tokens = tokenizer.encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        tracing::debug!(n_tokens = prompt_tokens.len(), "prompt tokenized");

        // Track recent tokens for repetition penalty
        let mut recent_tokens = prompt_tokens.clone();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut output_text = String::new();

        // Step 2: Batch prefill — process all prompt tokens through the model.
        //
        // Instead of processing tokens one-at-a-time (N separate forward calls
        // each computing and discarding full logits), we batch them into chunks.
        // The architecture's forward() handles multi-token input: it iterates
        // internally and only computes logits for the last hidden state.
        //
        // For very long prompts, we chunk into `prefill_chunk_size` pieces to
        // bound peak memory usage in the attention computation.
        let chunk_size = if self.config.prefill_chunk_size == 0 {
            prompt_tokens.len()
        } else {
            self.config.prefill_chunk_size
        };

        let mut logits = if prompt_tokens.len() <= chunk_size {
            // Short prompt: single batch forward
            tracing::debug!(
                chunk = 1,
                tokens = prompt_tokens.len(),
                "prefill: single batch"
            );
            let prefill_start = Instant::now();
            let result = forward_pass.forward(&prompt_tokens, kv_cache)?;
            self.metrics
                .record_prefill(prompt_tokens.len() as u64, prefill_start.elapsed());
            result
        } else {
            // Long prompt: chunked prefill
            let n_chunks = prompt_tokens.len().div_ceil(chunk_size);
            tracing::debug!(
                n_chunks = n_chunks,
                chunk_size = chunk_size,
                total = prompt_tokens.len(),
                "prefill: chunked"
            );

            let prefill_start = Instant::now();
            let mut last_logits = Vec::new();
            for (i, chunk) in prompt_tokens.chunks(chunk_size).enumerate() {
                tracing::trace!(
                    chunk_idx = i,
                    chunk_len = chunk.len(),
                    kv_pos = kv_cache.seq_len(),
                    "prefill chunk"
                );
                last_logits = forward_pass.forward(chunk, kv_cache)?;
            }
            self.metrics
                .record_prefill(prompt_tokens.len() as u64, prefill_start.elapsed());
            last_logits
        };

        // Create a stateful sampler so grammar state is maintained across tokens.
        let mut sampler = Sampler::new(self.config.sampler.clone());

        // Step 3: Autoregressive decode loop
        self.metrics.record_request_start();
        for _step in 0..max_tokens {
            // Sample next token (grammar masking happens inside the sampler)
            let next_token = sampler.sample(&logits, &recent_tokens);

            // Check for EOS
            if Some(next_token) == self.eos_token_id {
                tracing::debug!("EOS token generated, stopping");
                break;
            }

            // Check context length
            if kv_cache.seq_len() >= forward_pass.max_context_length() {
                tracing::warn!("context length reached, stopping generation");
                break;
            }

            // Decode token to text
            let token_text = tokenizer.decode(&[next_token])?;
            callback(&token_text);
            output_text.push_str(&token_text);

            // Track for repetition penalty
            recent_tokens.push(next_token);
            generated_tokens.push(next_token);

            // Forward pass for next token
            let decode_start = Instant::now();
            logits = forward_pass.forward(&[next_token], kv_cache)?;
            self.metrics.record_decode_token(decode_start.elapsed());
        }
        self.metrics.record_request_complete();

        tracing::info!(
            prompt_tokens = prompt_tokens.len(),
            generated_tokens = generated_tokens.len(),
            "generation complete"
        );

        Ok(output_text)
    }

    /// Generate tokens using an explicit sampler config instead of the engine default.
    ///
    /// This is the preferred entry point for per-request sampler customization
    /// (e.g., grammar-constrained sampling from the API server).
    pub fn generate_with_config(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler_config: SamplerConfig,
        mut callback: impl FnMut(&str),
    ) -> RuntimeResult<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;

        let prompt_tokens = tokenizer.encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        let mut recent_tokens = prompt_tokens.clone();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut output_text = String::new();

        for &token in &prompt_tokens[..prompt_tokens.len() - 1] {
            forward_pass.forward(&[token], kv_cache)?;
        }

        let last = *prompt_tokens.last().ok_or(RuntimeError::ModelNotLoaded)?;
        let mut logits = forward_pass.forward(&[last], kv_cache)?;

        let mut sampler = Sampler::new(sampler_config);
        self.metrics.record_request_start();
        for _step in 0..max_tokens {
            let next_token = sampler.sample(&logits, &recent_tokens);

            if Some(next_token) == self.eos_token_id {
                tracing::debug!("EOS token generated, stopping");
                break;
            }

            if kv_cache.seq_len() >= forward_pass.max_context_length() {
                tracing::warn!("context length reached, stopping generation");
                break;
            }

            let token_text = tokenizer.decode(&[next_token])?;
            callback(&token_text);
            output_text.push_str(&token_text);

            recent_tokens.push(next_token);
            generated_tokens.push(next_token);

            let decode_start = Instant::now();
            logits = forward_pass.forward(&[next_token], kv_cache)?;
            self.metrics.record_decode_token(decode_start.elapsed());
        }
        self.metrics.record_request_complete();

        tracing::info!(
            prompt_tokens = prompt_tokens.len(),
            generated_tokens = generated_tokens.len(),
            "generation (with custom config) complete"
        );

        Ok(output_text)
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

    // -------------------------------------------------------------------------
    // Multi-LoRA hot-swap
    // -------------------------------------------------------------------------

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

    /// Reset the KV cache (for starting a new conversation).
    pub fn reset(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            cache.clear();
        }
    }

    // -------------------------------------------------------------------------
    // Speculative-decoding primitives
    // -------------------------------------------------------------------------

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

    /// Prefill the KV cache with the given token sequence without returning logits.
    ///
    /// Processes all tokens in order, updating the KV cache at each position.
    /// The last token's logits are discarded; callers typically follow up with
    /// `forward_one` to begin autoregressive generation.
    pub fn prefill(&mut self, tokens: &[u32]) -> RuntimeResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;
        for &token in tokens {
            forward_pass.forward(&[token], kv_cache)?;
        }
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

    /// Returns `true` if `token` is the EOS token for this model.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos_token_id == Some(token)
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
        kv.restore_from_snapshot(&snapshot.keys, &snapshot.values, snapshot.seq_len);
        Ok(())
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

    /// Compute a semantic embedding vector for the given text.
    ///
    /// Runs tokenization → full transformer layers → final RMSNorm, then
    /// L2-normalises the resulting `hidden_size`-dimensional vector.
    /// The KV cache is reset before the pass so that embeddings for
    /// different inputs are independent of each other.
    ///
    /// Returns `RuntimeError::ModelNotLoaded` if no model has been loaded.
    pub fn embed(&mut self, text: &str) -> RuntimeResult<Vec<f32>> {
        // Step 1: Reset the KV cache so this embedding is independent.
        // Must happen before we take any partial borrows below.
        self.reset();

        // Step 2: Validate that all components are loaded.
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;

        // Step 3: Tokenize. We need the tokenizer reference independently,
        // so read it before we borrow forward_pass/kv_cache mutably.
        let tokens = {
            let tok = self
                .tokenizer
                .as_ref()
                .ok_or(RuntimeError::ModelNotLoaded)?;
            tok.encode(text)?
        };

        if tokens.is_empty() {
            // Return a zero vector of the appropriate dimension if available,
            // otherwise an empty vector. An empty input has no well-defined embedding.
            let dim = self
                .model_config
                .as_ref()
                .map(|c| c.hidden_size)
                .unwrap_or(0);
            return Ok(vec![0.0f32; dim]);
        }

        // Step 4: Run the embed forward pass (all layers + output_norm, no LM head).
        let hidden = forward_pass.embed(&tokens, kv_cache)?;

        // Step 5: L2-normalise the hidden vector for cosine-similarity compatibility.
        let norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            Ok(hidden.into_iter().map(|x| x / norm).collect())
        } else {
            Ok(hidden)
        }
    }

    /// Extract embedding vectors for multiple input texts (batch variant).
    ///
    /// Runs the model forward pass and returns the final hidden state
    /// (post-normalization, pre-LM-head) as the embedding vector.
    /// Each text is processed independently with a fresh KV cache.
    pub fn embed_batch(&mut self, texts: &[String]) -> RuntimeResult<Vec<Vec<f32>>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let forward_pass = self
            .forward_pass
            .as_mut()
            .ok_or(RuntimeError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut().ok_or(RuntimeError::ModelNotLoaded)?;

        let hidden_size = forward_pass.hidden_size();
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            // Reset cache for each text (embeddings are independent)
            kv_cache.clear();

            let tokens = tokenizer.encode(text)?;
            if tokens.is_empty() {
                embeddings.push(vec![0.0f32; hidden_size]);
                continue;
            }

            // Process all tokens and get the final hidden state
            let hidden_state = forward_pass.embed(&tokens, kv_cache)?;

            // L2 normalize the embedding vector
            let norm: f32 = hidden_state.iter().map(|x| x * x).sum::<f32>().sqrt();
            let embedding = if norm > 1e-12 {
                hidden_state.iter().map(|x| x / norm).collect()
            } else {
                hidden_state
            };

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

/// Build the forward pass from a loaded GGUF model.
fn build_forward_pass(
    gguf: &GgufModel,
    config: &ModelConfig,
) -> RuntimeResult<Box<dyn ForwardPass>> {
    match config.architecture.as_str() {
        #[cfg(feature = "llama")]
        "llama" => {
            let model = oxillama_arch::llama::load_llama_from_gguf(gguf, config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "qwen3")]
        "qwen3" => {
            let model = oxillama_arch::qwen3::load_qwen3_from_gguf(gguf, config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "mistral")]
        "mistral" => {
            let model = oxillama_arch::mistral::load_mistral_from_gguf(gguf, config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "gemma")]
        "gemma" | "gemma2" | "gemma3" => {
            let model = oxillama_arch::gemma::load_gemma_from_gguf(gguf, config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "phi")]
        "phi3" | "phi" => {
            let model = oxillama_arch::phi::load_phi_from_gguf(gguf, config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "command-r")]
        "command-r" => {
            let model = oxillama_arch::command_r::load_command_r_from_gguf(gguf, config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "starcoder")]
        "starcoder" => {
            let model = oxillama_arch::starcoder::load_starcoder_from_gguf(gguf, config)?;
            Ok(Box::new(model))
        }
        arch => Err(RuntimeError::ModelLoadError {
            message: format!("unsupported architecture: '{arch}'"),
        }),
    }
}

/// Load the tokenizer, trying multiple sources.
fn load_tokenizer(config: &EngineConfig, gguf: &GgufModel) -> RuntimeResult<TokenizerBridge> {
    // Try explicit tokenizer path first
    if let Some(ref path) = config.tokenizer_path {
        return TokenizerBridge::from_file(path);
    }

    // Try to extract tokenizer from GGUF metadata
    if let Some(tokenizer_json) = gguf
        .file
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|_| {
            // If there's a full tokenizer JSON in metadata, use it
            gguf.file
                .metadata
                .get("tokenizer.huggingface.json")
                .and_then(|v| v.as_str())
        })
    {
        return TokenizerBridge::from_bytes(tokenizer_json.as_bytes());
    }

    // Try to find tokenizer.json next to the model file
    let model_dir = Path::new(&config.model_path)
        .parent()
        .unwrap_or(Path::new("."));
    let tokenizer_path = model_dir.join("tokenizer.json");
    if tokenizer_path.exists() {
        return TokenizerBridge::from_file(tokenizer_path.to_str().unwrap_or("tokenizer.json"));
    }

    Err(RuntimeError::TokenizerError {
        message: "no tokenizer found: provide --tokenizer path or place tokenizer.json next to the model file".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── A3: forward_prefill / forward_decode tests ───────────────────────────

    /// `forward_prefill` must return `ModelNotLoaded` when no model is loaded.
    #[test]
    fn test_forward_prefill_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.forward_prefill(&[1, 2, 3], 0);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded from forward_prefill, got {result:?}"
        );
    }

    /// `forward_prefill` with empty token slice must return an error
    /// (even if a model were loaded — callers must supply at least one token).
    #[test]
    fn test_forward_prefill_empty_slice_errors() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        // No model loaded; empty slice should error with ModelNotLoaded because
        // the empty-slice guard fires first (returns ModelLoadError), but any
        // error is acceptable — the point is that it never returns Ok.
        let result = engine.forward_prefill(&[], 0);
        assert!(
            result.is_err(),
            "forward_prefill with empty slice must return Err, got Ok"
        );
    }

    /// `forward_decode` must return `ModelNotLoaded` when no model is loaded.
    #[test]
    fn test_forward_decode_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.forward_decode(42, 0);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded from forward_decode, got {result:?}"
        );
    }

    // ── A3: forward_prefill / forward_decode with loaded model ────────────────

    /// `forward_prefill` with a loaded model must return a logits vector whose
    /// length equals the model's vocab size (32 in the synthetic fixture).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_forward_prefill_returns_logits_after_load() {
        let mut engine = make_loaded_engine();
        // Fresh engine: KV cache is empty so pos_start = 0.
        let result = engine.forward_prefill(&[3, 4, 5], 0);
        assert!(
            result.is_ok(),
            "forward_prefill must return Ok when model is loaded, got {result:?}"
        );
        let logits = result.expect("forward_prefill Ok");
        assert_eq!(
            logits.len(),
            32,
            "logits length must equal vocab_size=32, got {}",
            logits.len()
        );
    }

    /// `forward_decode` with a loaded model must return a logits vector of
    /// the correct vocab-size length.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_forward_decode_returns_logits_after_load() {
        let mut engine = make_loaded_engine();
        // Prefill one token to prime the KV cache (pos becomes 1).
        engine
            .forward_prefill(&[3], 0)
            .expect("prefill must succeed");
        // Now KV cache seq_len == 1.
        let result = engine.forward_decode(4, 1);
        assert!(
            result.is_ok(),
            "forward_decode must return Ok when model is loaded, got {result:?}"
        );
        let logits = result.expect("forward_decode Ok");
        assert_eq!(
            logits.len(),
            32,
            "logits length must equal vocab_size=32, got {}",
            logits.len()
        );
    }

    /// Verify that chunked-prefill produces the same final logits as
    /// single-shot prefill (the core KV-state invariant from A3).
    ///
    /// Both paths must agree on the logit vector produced after processing
    /// the same prompt tokens, within floating-point tolerance.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn chunked_prefill_kv_matches_singleshot() {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let prompt_tokens = vec![3u32, 4, 5, 6];

        // ── Single-shot path ──────────────────────────────────────────────────
        let mut engine_single = InferenceEngine::new(EngineConfig::default());
        engine_single
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("single-shot load");
        // Fresh engine: pos_start = 0.
        let logits_single = engine_single
            .forward_prefill(&prompt_tokens, 0)
            .expect("single-shot prefill");

        // ── Chunked path (chunk = 2) ──────────────────────────────────────────
        let mut engine_chunked = InferenceEngine::new(EngineConfig::default());
        engine_chunked
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("chunked load");

        let mut logits_chunked = Vec::new();
        let chunk_size = 2usize;
        let mut pos = 0usize;
        for slice in prompt_tokens.chunks(chunk_size) {
            logits_chunked = engine_chunked
                .forward_prefill(slice, pos)
                .expect("chunked prefill");
            pos += slice.len();
        }

        // ── Compare logits ────────────────────────────────────────────────────
        assert_eq!(
            logits_single.len(),
            logits_chunked.len(),
            "logit vector lengths must match"
        );
        let tol = 1e-4f32;
        let max_diff = logits_single
            .iter()
            .zip(logits_chunked.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < tol,
            "chunked and single-shot prefill logits differ by {max_diff} > tolerance {tol}"
        );
    }

    // ── End A3 ────────────────────────────────────────────────────────────────

    /// embed() must return an error when no model has been loaded,
    /// rather than panicking or producing a garbage vector.
    #[test]
    fn test_embed_returns_err_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.embed("hello world");
        assert!(
            result.is_err(),
            "embed() should return Err when no model is loaded"
        );
    }

    /// hidden_size() returns None when no model is loaded.
    #[test]
    fn test_hidden_size_none_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(
            engine.hidden_size().is_none(),
            "hidden_size() should be None before load_model()"
        );
    }

    /// is_loaded() must be false for a freshly created engine.
    #[test]
    fn test_is_loaded_false_initially() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_model_config_none_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(engine.model_config().is_none());
    }

    #[test]
    fn test_config_roundtrip() {
        let cfg = EngineConfig {
            model_path: "test.gguf".to_string(),
            num_threads: 8,
            ..EngineConfig::default()
        };
        let engine = InferenceEngine::new(cfg);
        assert_eq!(engine.config().model_path, "test.gguf");
        assert_eq!(engine.config().num_threads, 8);
    }

    #[test]
    fn test_generate_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.generate("hello", 10, |_| {});
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }

    #[test]
    fn test_generate_with_config_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.generate_with_config("hello", 5, SamplerConfig::greedy(), |_| {});
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }

    #[test]
    fn test_tokenize_errors_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.tokenize("hello world");
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }

    #[test]
    fn test_prefill_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.prefill(&[1, 2, 3]);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }

    #[test]
    fn test_prefill_empty_slice_ok_when_no_model() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        // Empty slice is a no-op and returns Ok regardless of model state.
        let result = engine.prefill(&[]);
        assert!(result.is_ok(), "empty prefill should be Ok, got {result:?}");
    }

    #[test]
    fn test_forward_one_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.forward_one(42);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }

    #[test]
    fn test_decode_token_errors_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.decode_token(1);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }

    #[test]
    fn test_is_eos_false_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(!engine.is_eos(0));
        assert!(!engine.is_eos(u32::MAX));
    }

    #[test]
    fn test_vocab_bytes_none_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(engine.vocab_bytes().is_none());
    }

    #[test]
    fn test_reset_does_not_panic_when_no_kv_cache() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine.reset(); // should be a no-op, not a panic
    }

    #[test]
    fn test_apply_lora_adapters_errors_when_not_loaded() {
        use oxillama_arch::lora::LoadedLora;
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let lora = LoadedLora {
            rank: 8,
            alpha: 1.0,
            adapters: std::collections::HashMap::new(),
        };
        let result = engine.apply_lora_adapters(&lora);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }

    #[test]
    fn test_load_model_missing_file_errors() {
        let cfg = EngineConfig {
            model_path: "/nonexistent/path/model_abc_xyz.gguf".to_string(),
            ..EngineConfig::default()
        };
        let mut engine = InferenceEngine::new(cfg);
        let result = engine.load_model();
        assert!(
            matches!(result, Err(RuntimeError::ModelLoadError { .. })),
            "expected ModelLoadError for missing file, got {result:?}"
        );
    }

    #[test]
    fn test_load_model_from_bytes_bad_magic_errors() {
        let cfg = EngineConfig::default();
        let mut engine = InferenceEngine::new(cfg);
        // Bytes that look nothing like a GGUF file (wrong magic)
        let bad_bytes = b"THIS IS NOT A GGUF FILE AT ALL";
        let result = engine.load_model_from_bytes(bad_bytes, "{}");
        assert!(
            result.is_err(),
            "load_model_from_bytes with garbage bytes should error, got Ok(())"
        );
    }

    #[test]
    fn test_load_model_from_bytes_empty_errors() {
        let cfg = EngineConfig::default();
        let mut engine = InferenceEngine::new(cfg);
        let result = engine.load_model_from_bytes(&[], "{}");
        assert!(
            result.is_err(),
            "load_model_from_bytes with empty bytes should error"
        );
    }

    #[test]
    fn test_engine_config_default_fields() {
        let cfg = EngineConfig::default();
        assert!(
            cfg.model_path.is_empty(),
            "default model_path should be empty"
        );
        assert!(
            cfg.tokenizer_path.is_none(),
            "default tokenizer_path should be None"
        );
        assert!(
            cfg.context_size.is_none(),
            "default context_size should be None"
        );
        assert_eq!(cfg.num_threads, 4, "default num_threads should be 4");
    }

    #[test]
    fn test_engine_config_context_override() {
        let cfg = EngineConfig {
            context_size: Some(2048),
            ..EngineConfig::default()
        };
        assert_eq!(cfg.context_size, Some(2048));
    }

    #[test]
    fn test_generate_with_config_errors_when_not_loaded_variant() {
        // Additional variant with explicit SamplerConfig fields
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let sc = SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            ..SamplerConfig::default()
        };
        let result = engine.generate_with_config("test prompt", 5, sc, |_| {});
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }

    /// load_model() with a file that *exists* but contains garbage (not valid GGUF)
    /// must return an error without panicking, exercising the GgufModel::load parse path.
    #[test]
    fn test_load_model_existing_invalid_file_errors() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxillama_engine_bad_magic_test.gguf");
        // Write garbage bytes that will fail GGUF magic-byte check.
        std::fs::write(&tmp, b"NOT A GGUF FILE AT ALL - GARBAGE BYTES 0123456789")
            .expect("write temp file");
        let cfg = EngineConfig {
            model_path: tmp
                .to_str()
                .expect("temp path must be valid UTF-8")
                .to_string(),
            ..EngineConfig::default()
        };
        let mut engine = InferenceEngine::new(cfg);
        let result = engine.load_model();
        // Clean up before asserting so the file is always removed.
        let _ = std::fs::remove_file(&tmp);
        assert!(
            result.is_err(),
            "load_model with invalid GGUF content should return Err"
        );
    }

    /// is_loaded() must remain false after a failed load_model() call.
    #[test]
    fn test_is_loaded_remains_false_after_failed_load() {
        let cfg = EngineConfig {
            model_path: "/nonexistent/guaranteed_missing_model.gguf".to_string(),
            ..EngineConfig::default()
        };
        let mut engine = InferenceEngine::new(cfg);
        // This must fail (file doesn't exist).
        let _ = engine.load_model();
        assert!(
            !engine.is_loaded(),
            "is_loaded() must be false after a failed load_model()"
        );
    }

    /// EngineConfig implements Clone; verify the clone is independent.
    #[test]
    fn test_engine_config_clone_is_independent() {
        let original = EngineConfig {
            model_path: "original.gguf".to_string(),
            num_threads: 16,
            context_size: Some(4096),
            ..EngineConfig::default()
        };
        let mut cloned = original.clone();
        cloned.model_path = "cloned.gguf".to_string();
        cloned.num_threads = 1;
        // Original must be unaffected.
        assert_eq!(original.model_path, "original.gguf");
        assert_eq!(original.num_threads, 16);
        assert_eq!(original.context_size, Some(4096));
    }

    // -------------------------------------------------------------------------
    // Tests backed by the synthetic GGUF fixture (requires tokenizer feature)
    // -------------------------------------------------------------------------

    /// Return a loaded engine from the synthetic GGUF + tokenizer.
    /// These tests are only meaningful when a tokenizer backend is active.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    fn make_loaded_engine() -> InferenceEngine {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("synthetic GGUF must load successfully");
        engine
    }

    /// load_model_from_bytes with the synthetic fixture must succeed and
    /// set is_loaded() to true.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_load_model_from_bytes_succeeds() {
        let engine = make_loaded_engine();
        assert!(
            engine.is_loaded(),
            "is_loaded() must be true after a successful load_model_from_bytes()"
        );
    }

    /// model_config() must be Some with the expected hidden size after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_hidden_size_after_load() {
        let engine = make_loaded_engine();
        let hs = engine.hidden_size();
        assert_eq!(
            hs,
            Some(32),
            "hidden_size() must be Some(32) after loading the synthetic model, got {hs:?}"
        );
    }

    /// tokenize() must return Ok with at least one token after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_tokenize_after_load() {
        let engine = make_loaded_engine();
        let result = engine.tokenize("abc");
        assert!(
            result.is_ok(),
            "tokenize() must return Ok after model is loaded, got {result:?}"
        );
        let tokens = result.expect("tokenize succeeded");
        assert!(
            !tokens.is_empty(),
            "tokenize('abc') must produce at least one token"
        );
    }

    /// is_eos() must return true for token id 2 (</s> = EOS in the synthetic tokenizer).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_is_eos_after_load() {
        let engine = make_loaded_engine();
        assert!(
            engine.is_eos(2),
            "is_eos(2) must be true — </s> is the EOS token in the synthetic tokenizer"
        );
        assert!(
            !engine.is_eos(3),
            "is_eos(3) must be false — token 3 ('a') is not EOS"
        );
    }

    /// decode_token(3) should decode successfully (token 3 = 'a' in the synthetic vocab).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_decode_token_after_load() {
        let engine = make_loaded_engine();
        let result = engine.decode_token(3);
        assert!(
            result.is_ok(),
            "decode_token(3) must return Ok, got {result:?}"
        );
    }

    /// generate() must return Ok after loading; the returned string may be empty
    /// if the EOS token is sampled immediately, but it must not panic or error.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_after_load() {
        let mut engine = make_loaded_engine();
        let result = engine.generate("a", 3, |_| {});
        assert!(
            result.is_ok(),
            "generate() must return Ok after model is loaded, got {result:?}"
        );
    }

    /// generate() with max_tokens=5 must produce at most 5 tokens of output.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_respects_max_tokens() {
        let mut engine = make_loaded_engine();
        let max = 5usize;
        // Count callback invocations as a proxy for generated tokens.
        let mut count = 0usize;
        let result = engine.generate("a", max, |_tok| {
            count += 1;
        });
        assert!(result.is_ok(), "generate() must return Ok, got {result:?}");
        assert!(
            count <= max,
            "callback was invoked {count} times but max_tokens={max}"
        );
    }

    /// generate_streaming — count callback invocations to verify the streaming path fires.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_streaming_calls_callback() {
        let mut engine = make_loaded_engine();
        let mut invocations = 0usize;
        let max_tokens = 4;
        let result = engine.generate("a", max_tokens, |_piece| {
            invocations += 1;
        });
        assert!(
            result.is_ok(),
            "generate() streaming path must return Ok, got {result:?}"
        );
        // invocations may be 0 if EOS is sampled on the first step; just assert <= max.
        assert!(
            invocations <= max_tokens,
            "streaming callback fired {invocations} > max_tokens={max_tokens}"
        );
    }

    /// embed() must return Ok with a non-empty vector after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_embed_after_load() {
        let mut engine = make_loaded_engine();
        let result = engine.embed("a");
        assert!(
            result.is_ok(),
            "embed() must return Ok after model is loaded, got {result:?}"
        );
        let vec = result.expect("embed succeeded");
        assert!(!vec.is_empty(), "embed() must return a non-empty vector");
    }

    /// embed() must return a vector of length == hidden_size (32 for the synthetic model).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_embed_returns_hidden_size_vector() {
        let mut engine = make_loaded_engine();
        let vec = engine
            .embed("a")
            .expect("embed() must succeed after loading");
        assert_eq!(
            vec.len(),
            32,
            "embed() vector length must equal hidden_size=32, got {}",
            vec.len()
        );
    }

    /// Reload: loading the model a second time must succeed and leave is_loaded() true.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_reload_model_succeeds() {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());

        // First load.
        engine
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("first load must succeed");
        assert!(engine.is_loaded(), "is_loaded() after first load");

        // Second load (reload).
        engine
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("second (re)load must succeed");
        assert!(
            engine.is_loaded(),
            "is_loaded() after reload must still be true"
        );
    }

    /// vocab_bytes() must return Some with non-empty entries after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_vocab_bytes_some_after_load() {
        let engine = make_loaded_engine();
        let vb = engine.vocab_bytes();
        assert!(
            vb.is_some(),
            "vocab_bytes() must be Some after model is loaded"
        );
        let entries = vb.expect("vocab_bytes is Some");
        assert!(
            !entries.is_empty(),
            "vocab_bytes() must contain at least one entry"
        );
    }

    /// model_config() must return Some with correct metadata after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_model_config_some_after_load() {
        let engine = make_loaded_engine();
        let cfg = engine.model_config();
        assert!(cfg.is_some(), "model_config() must be Some after loading");
        let mc = cfg.expect("model_config is Some");
        assert_eq!(mc.architecture, "llama", "architecture must be 'llama'");
        assert_eq!(
            mc.num_layers, 1,
            "num_layers must be 1 for the synthetic model"
        );
        assert_eq!(mc.vocab_size, 32, "vocab_size must be 32");
    }

    /// reset() must not panic when a model is loaded, and the engine remains usable.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_reset_when_loaded_does_not_panic() {
        let mut engine = make_loaded_engine();
        engine.reset(); // must not panic
                        // After reset, basic queries still work.
        assert!(
            engine.is_loaded(),
            "is_loaded() must still be true after reset()"
        );
        assert_eq!(engine.hidden_size(), Some(32));
    }

    // -------------------------------------------------------------------------
    // Architecture forward-pass integration tests
    // -------------------------------------------------------------------------
    // Each test loads the synthetic GGUF for a specific architecture, verifies
    // is_loaded(), runs generate() for 2 tokens, and verifies the call
    // succeeds.  Two architectures additionally run embed() to cover the
    // embedding endpoint code path.

    /// Qwen3 forward pass: load, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_qwen3_arch() {
        use oxillama_gguf::test_utils::{build_minimal_qwen3_gguf, minimal_tokenizer_json};

        let bytes = build_minimal_qwen3_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load qwen3");
        assert!(engine.is_loaded(), "qwen3: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate qwen3");
    }

    /// Qwen3 embed: load and call embed().
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_embed_qwen3_arch() {
        use oxillama_gguf::test_utils::{build_minimal_qwen3_gguf, minimal_tokenizer_json};

        let bytes = build_minimal_qwen3_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load qwen3 for embed");
        let vec = engine.embed("abc").expect("test: embed qwen3");
        assert_eq!(
            vec.len(),
            32,
            "qwen3 embed must return hidden_size=32 vector"
        );
    }

    /// Mistral forward pass: load, generate with sliding window, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_mistral_arch() {
        use oxillama_gguf::test_utils::{build_minimal_mistral_gguf, minimal_tokenizer_json};

        let bytes = build_minimal_mistral_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load mistral");
        assert!(engine.is_loaded(), "mistral: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate mistral");
    }

    /// Gemma forward pass: load with soft-capping metadata, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_gemma_arch() {
        use oxillama_gguf::test_utils::{build_minimal_gemma_gguf, minimal_tokenizer_json};

        let bytes = build_minimal_gemma_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load gemma");
        assert!(engine.is_loaded(), "gemma: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate gemma");
    }

    /// Gemma embed: load and call embed() to cover the Gemma embedding path.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_embed_gemma_arch() {
        use oxillama_gguf::test_utils::{build_minimal_gemma_gguf, minimal_tokenizer_json};

        let bytes = build_minimal_gemma_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load gemma for embed");
        let vec = engine.embed("abc").expect("test: embed gemma");
        assert_eq!(
            vec.len(),
            32,
            "gemma embed must return hidden_size=32 vector"
        );
    }

    /// Phi-3 forward pass: merged QKV + partial RoPE, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_phi3_arch() {
        use oxillama_gguf::test_utils::{build_minimal_phi3_gguf, minimal_tokenizer_json};

        let bytes = build_minimal_phi3_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load phi3");
        assert!(engine.is_loaded(), "phi3: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate phi3");
    }

    /// Command-R forward pass: logit scaling, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_command_r_arch() {
        use oxillama_gguf::test_utils::{build_minimal_command_r_gguf, minimal_tokenizer_json};

        let bytes = build_minimal_command_r_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load command-r");
        assert!(engine.is_loaded(), "command-r: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate command-r");
    }

    /// StarCoder forward pass: MQA + absolute position embeddings, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_starcoder_arch() {
        use oxillama_gguf::test_utils::{build_minimal_starcoder_gguf, minimal_tokenizer_json};

        let bytes = build_minimal_starcoder_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load starcoder");
        assert!(engine.is_loaded(), "starcoder: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate starcoder");
    }

    // -------------------------------------------------------------------------
    // Multi-LoRA hot-swap tests
    // -------------------------------------------------------------------------

    #[test]
    fn lora_stack_push_pop() {
        use oxillama_arch::lora::LoadedLora;
        use oxillama_quant::LoraAdapter;
        use std::collections::HashMap;
        use std::sync::Arc;

        fn make_lora() -> Arc<LoadedLora> {
            let adapter = LoraAdapter::new(vec![0.0f32; 4 * 8], vec![0.0f32; 8 * 4], 4, 1.0, 8, 8)
                .expect("valid lora adapter");
            let mut adapters = HashMap::new();
            adapters.insert("test.weight".to_string(), Arc::new(adapter));
            Arc::new(LoadedLora {
                adapters,
                rank: 4,
                alpha: 1.0,
            })
        }

        let mut engine = InferenceEngine::new(EngineConfig::default());

        // Initially empty
        assert!(engine.lora_stack().is_empty());
        assert_eq!(engine.lora_stack().len(), 0);

        // Push two adapters
        engine.push_lora(make_lora(), 1.0);
        engine.push_lora(make_lora(), 0.5);
        assert_eq!(engine.lora_stack().len(), 2);
        assert!(!engine.lora_stack().is_empty());

        // Pop one
        let popped = engine.pop_lora();
        assert!(popped.is_some());
        let (_, scale) = popped.expect("pop must return Some");
        assert!((scale - 0.5).abs() < 1e-6);
        assert_eq!(engine.lora_stack().len(), 1);

        // Clear
        engine.clear_loras();
        assert!(engine.lora_stack().is_empty());

        // Pop from empty returns None
        assert!(engine.pop_lora().is_none());
    }

    #[test]
    fn lora_apply_stack_errors_when_not_loaded() {
        use oxillama_arch::lora::LoadedLora;
        use oxillama_quant::LoraAdapter;
        use std::collections::HashMap;
        use std::sync::Arc;

        let adapter = LoraAdapter::new(vec![0.0f32; 4 * 8], vec![0.0f32; 8 * 4], 4, 1.0, 8, 8)
            .expect("valid lora adapter");
        let mut adapters = HashMap::new();
        adapters.insert("test.weight".to_string(), Arc::new(adapter));
        let lora = Arc::new(LoadedLora {
            adapters,
            rank: 4,
            alpha: 1.0,
        });

        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine.push_lora(lora, 1.0);
        let result = engine.apply_lora_stack();
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {:?}",
            result
        );
    }
}

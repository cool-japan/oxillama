//! Phi transformer forward pass implementation.
//!
//! Phi-3/4 architecture with merged QKV projections and partial RoPE.
//!
//! Architecture: embedding → N×(RMSNorm → GQA → residual → RMSNorm → SwiGLU FFN → residual) → RMSNorm → LM head
//!
//! The main difference from LLaMA is that Q, K, V are packed into a single
//! `attn_qkv.weight` tensor, and RoPE may be applied to only a fraction of
//! each head's dimensions, controlled by `{arch}.rope.dimension_count`
//! (`rope_dims`, defaulting to the full `head_dim` — see `load_phi_from_gguf`).

use crate::common::linear::{gguf_linear_shape, QuantLinear};
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantKernel, QuantTensor};
use std::sync::Arc;

/// A single Phi transformer layer.
pub struct PhiLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Merged Q/K/V projection [q_dim + kv_dim + kv_dim, hidden_size].
    pub attn_qkv: QuantLinear,
    /// Output projection [hidden_size, num_heads * head_dim].
    pub attn_output: QuantLinear,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN gate projection [intermediate_size, hidden_size].
    pub ffn_gate: QuantLinear,
    /// FFN up projection [intermediate_size, hidden_size].
    pub ffn_up: QuantLinear,
    /// FFN down projection [hidden_size, intermediate_size].
    pub ffn_down: QuantLinear,

    // Resolved kernels — one per projection, looked up from the tensor's
    // quantization type exactly once, in `load_phi_from_gguf`, instead of
    // once per layer per token on the decode hot path. `pub` so a layer can
    // be constructed outside the loader (e.g. from a test) too.
    pub attn_qkv_kernel: Arc<dyn QuantKernel>,
    pub attn_output_kernel: Arc<dyn QuantKernel>,
    pub ffn_gate_kernel: Arc<dyn QuantKernel>,
    pub ffn_up_kernel: Arc<dyn QuantKernel>,
    pub ffn_down_kernel: Arc<dyn QuantKernel>,
}

/// Complete Phi model.
pub struct PhiModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Token embedding weights [vocab_size, hidden_size] stored as f32.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<PhiLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head (unembedding) projection. Falls back to a view over
    /// `token_embd.weight` when the checkpoint ties input/output embeddings
    /// and ships no standalone `output.weight` — see `load_lm_head`.
    pub output: QuantLinear,
    /// Resolved kernel for `output` — see `PhiLayer`'s kernel fields.
    output_kernel: Arc<dyn QuantKernel>,
    /// RoPE precomputed frequency table.
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    /// Number of head dimensions RoPE is applied to (partial rotary).
    ///
    /// Read from `{arch}.rope.dimension_count`; defaults to the full
    /// `head_dim` (standard Phi-3 is full rotary — see `load_phi_from_gguf`).
    pub rope_dims: usize,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_qkv: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    /// Output of `attn_output`'s projection, added into `buf_hidden`.
    /// Preallocated to `hidden_size` here instead of a fresh
    /// `vec![0.0; hidden_size]` inside `attention()` on every call.
    buf_proj_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl PhiModel {
    /// Create a new PhiModel from preloaded weights.
    ///
    /// `rope_dims`: number of leading dimensions per head that RoPE rotates
    /// (e.g. `head_dim` for full rotary, matching plain Phi-3).
    ///
    /// Fails if `output`'s tensor type — or any layer projection's — has no
    /// registered [`oxillama_quant::QuantKernel`]: every kernel is resolved
    /// once here rather than on every forward pass.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<PhiLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
        rope_dims: usize,
    ) -> ArchResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;

        let rope = RopeTable::new(
            rope_dims,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
        );
        let dispatcher = KernelDispatcher::new();
        let output_kernel = resolve_kernel(&dispatcher, &output)?;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;

        Ok(Self {
            config,
            token_embd,
            layers,
            output_norm,
            output,
            output_kernel,
            rope,
            dispatcher,
            rope_dims,
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_qkv: vec![0.0; qkv_dim],
            buf_q: vec![0.0; q_dim],
            buf_k: vec![0.0; kv_dim],
            buf_v: vec![0.0; kv_dim],
            buf_attn_out: vec![0.0; hidden_size],
            buf_proj_out: vec![0.0; hidden_size],
            buf_gate: vec![0.0; intermediate_size],
            buf_up: vec![0.0; intermediate_size],
            buf_ffn_out: vec![0.0; hidden_size],
            buf_logits: vec![0.0; vocab_size],
            buf_attn_scores: vec![0.0; max_ctx],
        })
    }

    /// Load the residual stream with `token`'s embedding row.
    ///
    /// Bound-checked: an out-of-vocabulary `token` (possible whenever
    /// `vocab_size` is over- or under-estimated from GGUF metadata) is
    /// reported as an error instead of panicking on the slice index. This
    /// path runs on every decoded token, including ones sourced from an
    /// HTTP request body, so it must never panic.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let hidden_size = self.config.hidden_size;
        let offset = (token as usize)
            .checked_mul(hidden_size)
            .ok_or_else(|| out_of_vocab_error(token, self.config.vocab_size))?;
        let end = offset
            .checked_add(hidden_size)
            .ok_or_else(|| out_of_vocab_error(token, self.config.vocab_size))?;
        let src = self
            .token_embd
            .get(offset..end)
            .ok_or_else(|| out_of_vocab_error(token, self.config.vocab_size))?;
        self.buf_hidden.copy_from_slice(src);
        Ok(())
    }

    /// Split merged QKV output into separate Q, K, V buffers.
    fn split_qkv(&mut self) {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        self.buf_q.copy_from_slice(&self.buf_qkv[..q_dim]);
        self.buf_k
            .copy_from_slice(&self.buf_qkv[q_dim..q_dim + kv_dim]);
        self.buf_v
            .copy_from_slice(&self.buf_qkv[q_dim + kv_dim..q_dim + 2 * kv_dim]);
    }

    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let heads_per_kv = num_heads / num_kv_heads;

        // Merged QKV projection. Kernel resolved once at load time — see the
        // comment on `PhiLayer`'s kernel fields.
        let qkv_kernel: &dyn QuantKernel = &*layer.attn_qkv_kernel;
        layer
            .attn_qkv
            .forward(qkv_kernel, &self.buf_norm, &mut self.buf_qkv)?;

        // Split into Q, K, V
        self.split_qkv();

        // Apply partial RoPE to Q and K (only first `rope_dims` dims per head)
        for h in 0..num_heads {
            let q_head = &mut self.buf_q[h * head_dim..h * head_dim + self.rope_dims];
            self.rope.apply(q_head, position);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..h * head_dim + self.rope_dims];
            self.rope.apply(k_head, position);
        }

        // Store K, V in cache
        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let seq_len = position + 1;
        let scale = 1.0 / (head_dim as f32).sqrt();

        self.buf_attn_out.fill(0.0);

        // NOTE on sliding-window attention: Phi-3-mini/-small checkpoints
        // carry a `{arch}.attention.sliding_window` value, but current
        // llama.cpp (`llama-model.cpp`, `case LLM_ARCH_PHI3:`) forcibly
        // disables SWA for this architecture — `swa_type = LLAMA_SWA_TYPE_NONE;
        // n_swa = 0; set_swa_pattern(1);` — with the comment "Phi SWA is
        // currently disabled - results might be suboptimal for some models"
        // and a TODO citing ggml-org/llama.cpp#13676: the conversion scripts
        // do not correctly populate `n_swa`/`n_swa_pattern`, so a window
        // computed from that metadata could be actively wrong rather than
        // merely absent. `llm_build_phi3<false>` (dense, no windowing) is
        // therefore always instantiated for `LLM_ARCH_PHI3`/`LLM_ARCH_PHIMOE`
        // regardless of the sliding_window key. This loop intentionally
        // mirrors that: full causal attention over every cached position,
        // not windowed. Do not wire `effective_attention_span`/
        // `config.sliding_window` in here without re-checking whether
        // upstream has fixed the converter.
        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in 0..seq_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_offset..k_offset + head_dim];

                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * k_vec[d];
                }
                self.buf_attn_scores[pos] = score * scale;
            }

            softmax_inplace(&mut self.buf_attn_scores[..seq_len]);

            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for pos in 0..seq_len {
                let v_offset = pos * kv_dim + kv_head * head_dim;
                let v_vec = &cached_values[v_offset..v_offset + head_dim];
                let w = self.buf_attn_scores[pos];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        // Project attention output. `buf_proj_out` is preallocated engine
        // state (see its field doc) instead of a per-call `vec![0.0; ...]`.
        let o_kernel: &dyn QuantKernel = &*self.layers[layer_idx].attn_output_kernel;
        let layer = &self.layers[layer_idx];
        layer
            .attn_output
            .forward(o_kernel, &self.buf_attn_out, &mut self.buf_proj_out)?;

        for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_proj_out.iter()) {
            *h += p;
        }

        Ok(())
    }

    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];

        let gate_kernel: &dyn QuantKernel = &*layer.ffn_gate_kernel;
        let up_kernel: &dyn QuantKernel = &*layer.ffn_up_kernel;
        let down_kernel: &dyn QuantKernel = &*layer.ffn_down_kernel;

        layer
            .ffn_gate
            .forward(gate_kernel, &self.buf_norm, &mut self.buf_gate)?;
        layer
            .ffn_up
            .forward(up_kernel, &self.buf_norm, &mut self.buf_up)?;

        swiglu_inplace(&mut self.buf_gate, &self.buf_up);

        layer
            .ffn_down
            .forward(down_kernel, &self.buf_gate, &mut self.buf_ffn_out)?;

        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }

        Ok(())
    }
}

impl ForwardPass for PhiModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();
        check_context_length(self.config.max_context_length, start_pos, tokens.len())?;

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            self.embed_token(token)?;

            for layer_idx in 0..self.layers.len() {
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                self.attention(layer_idx, position, kv_cache)?;

                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                self.feed_forward(layer_idx)?;
            }

            kv_cache.advance();
        }

        self.output_norm.forward(&mut self.buf_hidden);

        if self.buf_logits.len() != self.config.vocab_size {
            self.buf_logits.resize(self.config.vocab_size, 0.0);
        }
        let output_kernel: &dyn QuantKernel = &*self.output_kernel;
        self.output
            .forward(output_kernel, &self.buf_hidden, &mut self.buf_logits)?;

        // Hand the freshly computed logits to the caller by ownership
        // transfer instead of a `.clone()` allocation-and-memcpy on every
        // decoded token; the resize above repairs `buf_logits` on next call.
        Ok(std::mem::take(&mut self.buf_logits))
    }

    /// Extract the post-output-norm hidden state for embedding.
    ///
    /// Identical to `forward()` up to and including `output_norm.forward()`.
    /// Stops SHORT of the LM-head projection (output.weight) that maps
    /// hidden_size → vocab_size. Partial RoPE is preserved as in the normal
    /// forward pass. Returns a `hidden_size`-dimensional vector suitable for
    /// L2-normalised semantic embeddings.
    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();
        check_context_length(self.config.max_context_length, start_pos, tokens.len())?;

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            self.embed_token(token)?;

            for layer_idx in 0..self.layers.len() {
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                self.attention(layer_idx, position, kv_cache)?;

                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                self.feed_forward(layer_idx)?;
            }

            kv_cache.advance();
        }

        // Final norm on the last token's hidden state.
        // Does NOT project through the LM head — returns hidden state directly.
        self.output_norm.forward(&mut self.buf_hidden);

        Ok(self.buf_hidden.clone())
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn max_context_length(&self) -> usize {
        self.config.max_context_length
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}

/// In-place softmax over a slice.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Build the out-of-vocabulary error for a bad token id.
fn out_of_vocab_error(token: u32, vocab_size: usize) -> ArchError {
    ArchError::ConfigMismatch {
        param: "token id".to_string(),
        expected: format!("< {vocab_size}"),
        got: token.to_string(),
    }
}

/// Validate that a forward/embed call will not run any position past
/// `max_context_length`.
///
/// `RopeTable::apply` and `buf_attn_scores` are both sized to
/// `max_context_length` and indexed by raw position with no further
/// checking; an over-long prompt at prefill time (the decode loop is
/// separately guarded upstream, but prefill is reachable directly from the
/// HTTP server with attacker-controlled input) would otherwise panic on the
/// first out-of-range index instead of returning an error.
fn check_context_length(
    max_context_length: usize,
    start_pos: usize,
    n_tokens: usize,
) -> ArchResult<()> {
    let end = start_pos
        .checked_add(n_tokens)
        .filter(|&end| end <= max_context_length);
    if end.is_some() {
        return Ok(());
    }
    Err(ArchError::InvalidConfig {
        detail: format!(
            "context length exceeded: start_pos={start_pos} + {n_tokens} tokens > max_context_length={max_context_length}"
        ),
    })
}

/// Load a Phi model from a `GgufModel`.
pub fn load_phi_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<PhiModel> {
    let dispatcher = KernelDispatcher::new();

    // Number of rotated dimensions per head, from the standard GGUF key
    // `{arch}.rope.dimension_count`. No GGUF converter writes
    // `{arch}.rope.partial_rotary_factor` (that key does not exist in
    // gguf-py's `Keys.Rope`); `convert_hf_to_gguf.py`'s `Phi3MiniModel`
    // writes `add_rope_dimension_count(int(rot_pct * n_embd) // n_head)`,
    // which is `head_dim` (full rotary) whenever the HF config has no
    // `partial_rotary_factor` — true for plain Phi-3. Reading the old wrong
    // key always missed and silently rotated only half of every head.
    // `llama-model.cpp` defaults `hparams.n_rot` to `n_embd_head_k`
    // (`head_dim`) before applying this same key, which this mirrors.
    let rope_dims = model
        .file
        .metadata
        .get_u32(&format!("{}.rope.dimension_count", config.architecture))
        .map(|v| v as usize)
        .ok()
        .filter(|&v| v > 0)
        .unwrap_or(config.head_dim);

    // Load token embeddings
    let embd_data = model.tensor_data("token_embd.weight")?;
    let embd_info = model.file.tensors.get("token_embd.weight")?;
    let token_embd = dequant_to_f32(embd_info, embd_data, &dispatcher)?;

    // The table must carry at least `vocab_size * hidden_size` rows (`>=`,
    // not `==`: some converters keep padding rows beyond the tokenizer's
    // declared vocabulary). Falling short turns an in-range `token` into an
    // out-of-bounds `buf_hidden` copy.
    let min_embd_len = config.vocab_size.saturating_mul(config.hidden_size);
    if token_embd.len() < min_embd_len {
        return Err(ArchError::InvalidShape {
            name: "token_embd.weight".to_string(),
            expected: vec![config.vocab_size, config.hidden_size],
            got: vec![token_embd.len()],
        });
    }

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;

        let attn_qkv = load_quant_linear(model, &format!("{prefix}.attn_qkv.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        // Resolve every projection's kernel once, here, instead of on every
        // token of every generation this model ever serves.
        let attn_qkv_kernel = resolve_kernel(&dispatcher, &attn_qkv)?;
        let attn_output_kernel = resolve_kernel(&dispatcher, &attn_output)?;
        let ffn_gate_kernel = resolve_kernel(&dispatcher, &ffn_gate)?;
        let ffn_up_kernel = resolve_kernel(&dispatcher, &ffn_up)?;
        let ffn_down_kernel = resolve_kernel(&dispatcher, &ffn_down)?;

        layers.push(PhiLayer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_qkv,
            attn_qkv_kernel,
            attn_output,
            attn_output_kernel,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            ffn_gate,
            ffn_gate_kernel,
            ffn_up,
            ffn_up_kernel,
            ffn_down,
            ffn_down_kernel,
        });
    }

    // Load final norm and output projection
    let output_norm_weight = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_weight, config.rms_norm_eps);
    let output = load_lm_head(model)?;

    PhiModel::new(
        config.clone(),
        token_embd,
        layers,
        output_norm,
        output,
        rope_dims,
    )
}

/// Resolve `linear`'s kernel once, wrapped for cheap sharing.
fn resolve_kernel(
    dispatcher: &KernelDispatcher,
    linear: &QuantLinear,
) -> ArchResult<Arc<dyn QuantKernel>> {
    Ok(dispatcher.get_kernel(linear.weight.tensor_type)?.into())
}

/// Load the LM head, falling back to the tied input embedding.
///
/// Some Phi checkpoints tie the input and output embeddings and ship no
/// standalone `output.weight`. When it is absent, `token_embd.weight` is
/// reused as the LM head (mirroring `qwen3::load_lm_head` and llama.cpp's
/// own `if (output == NULL) { output = tok_embd; }` fallback for
/// `LLM_ARCH_PHI3`). Both tensors carry the same dimensions, and the reuse
/// goes through [`load_quant_linear`], so the head stays quantized.
fn load_lm_head(model: &oxillama_gguf::GgufModel) -> ArchResult<QuantLinear> {
    if !model.file.tensors.contains("output.weight")
        && model.file.tensors.contains("token_embd.weight")
    {
        return load_quant_linear(model, "token_embd.weight");
    }
    load_quant_linear(model, "output.weight")
}

/// Load a quantized linear layer from GGUF.
fn load_quant_linear(model: &oxillama_gguf::GgufModel, name: &str) -> ArchResult<QuantLinear> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let shape = gguf_linear_shape(&info.dimensions);
    let tensor_type = info.tensor_type;
    // Shared mmap-backed view, not a `to_vec()` copy: the GEMV kernels only
    // ever read `&[u8]` out of the payload, so a private copy would double
    // the checkpoint's resident cost for nothing.
    let data = model.tensor_bytes(name)?;
    let tensor = QuantTensor::from_shared(data, shape, tensor_type);
    Ok(QuantLinear::new(tensor, None))
}

/// Load an RMSNorm weight vector from GGUF.
fn load_rms_norm_weight(model: &oxillama_gguf::GgufModel, name: &str) -> ArchResult<Vec<f32>> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model.tensor_data(name)?;
    let dispatcher = KernelDispatcher::new();
    dequant_to_f32(info, data, &dispatcher)
}

/// Dequantize tensor data to f32.
fn dequant_to_f32(
    info: &oxillama_gguf::TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    let n_elements = info.n_elements() as usize;
    let tensor_type = info.tensor_type;

    if tensor_type == oxillama_gguf::GgufTensorType::F32 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(4).enumerate().take(n_elements) {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    if tensor_type == oxillama_gguf::GgufTensorType::F16 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(2).enumerate().take(n_elements) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            out[i] = half::f16::from_bits(bits).to_f32();
        }
        return Ok(out);
    }

    let kernel = dispatcher.get_kernel(tensor_type)?;
    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();
    let n_blocks = n_elements.div_ceil(block_size);

    let mut out = vec![0.0f32; n_elements];
    for blk in 0..n_blocks {
        let data_offset = blk * block_bytes;
        let out_offset = blk * block_size;
        let block_data = &data[data_offset..data_offset + block_bytes];
        let out_slice = &mut out[out_offset..out_offset.saturating_add(block_size).min(n_elements)];
        kernel.dequant_block(block_data, out_slice)?;
    }

    Ok(out)
}

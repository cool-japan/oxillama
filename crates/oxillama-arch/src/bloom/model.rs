//! BLOOM transformer forward pass implementation.
//!
//! BigScience BLOOM uses the following architecture:
//! - Pre-LayerNorm (not RMSNorm) with bias, applied before both attention and FFN.
//! - ALiBi positional bias instead of RoPE — no rotary embeddings.
//! - Fused QKV projection (`attn_qkv.weight/bias`) instead of separate Q/K/V.
//! - Multi-Head Attention (MHA) — not GQA, `num_kv_heads = num_attention_heads`.
//! - Standard GELU FFN (not SwiGLU): `output = W_down @ gelu(W_up @ norm(x)) + bias`.
//! - All linear layers have bias vectors.
//!
//! ## Tensor naming convention (GGUF)
//!
//! | Tensor name                          | Description                          |
//! |--------------------------------------|--------------------------------------|
//! | `token_embd.weight`                  | Token embedding matrix               |
//! | `output_norm.weight`                 | Final LayerNorm scale                |
//! | `output_norm.bias`                   | Final LayerNorm bias                 |
//! | `blk.{l}.attn_norm.weight`           | Pre-attention LayerNorm scale        |
//! | `blk.{l}.attn_norm.bias`             | Pre-attention LayerNorm bias         |
//! | `blk.{l}.attn_qkv.weight`            | Fused QKV weight                     |
//! | `blk.{l}.attn_qkv.bias`              | Fused QKV bias                       |
//! | `blk.{l}.attn_output.weight`         | Attention output projection weight   |
//! | `blk.{l}.attn_output.bias`           | Attention output projection bias     |
//! | `blk.{l}.ffn_norm.weight`            | Pre-FFN LayerNorm scale              |
//! | `blk.{l}.ffn_norm.bias`              | Pre-FFN LayerNorm bias               |
//! | `blk.{l}.ffn_up.weight`              | FFN W1 weight (projects to 4×hidden) |
//! | `blk.{l}.ffn_up.bias`                | FFN W1 bias                          |
//! | `blk.{l}.ffn_down.weight`            | FFN W2 weight                        |
//! | `blk.{l}.ffn_down.bias`              | FFN W2 bias                          |

use crate::common::alibi::AlibiBias;
use crate::common::gelu::gelu_inplace;
use crate::common::layer_norm::LayerNorm;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, GgufTensorType, TensorStore};

use super::config::BloomConfig;

// ─── Layer definition ─────────────────────────────────────────────────────────

/// A single BLOOM transformer layer.
pub struct BloomLayer {
    /// Pre-attention LayerNorm with bias.
    pub attn_norm: LayerNorm,
    /// Fused QKV projection weight `[(num_heads * 3) * head_dim, hidden_size]` (f32).
    pub attn_qkv_weight: Vec<f32>,
    /// Fused QKV bias `[(num_heads * 3) * head_dim]` (f32).
    pub attn_qkv_bias: Vec<f32>,
    /// Attention output projection weight `[hidden_size, num_heads * head_dim]` (f32).
    pub attn_output_weight: Vec<f32>,
    /// Attention output projection bias `[hidden_size]` (f32).
    pub attn_output_bias: Vec<f32>,
    /// Pre-FFN LayerNorm with bias.
    pub ffn_norm: LayerNorm,
    /// FFN up/gate projection weight `[ffn_size, hidden_size]` (f32).
    pub ffn_up_weight: Vec<f32>,
    /// FFN up bias `[ffn_size]` (f32).
    pub ffn_up_bias: Vec<f32>,
    /// FFN down projection weight `[hidden_size, ffn_size]` (f32).
    pub ffn_down_weight: Vec<f32>,
    /// FFN down bias `[hidden_size]` (f32).
    pub ffn_down_bias: Vec<f32>,
}

// ─── Model ────────────────────────────────────────────────────────────────────

/// Complete BLOOM model with ALiBi attention.
pub struct BloomModel {
    /// Base model configuration.
    pub config: ModelConfig,
    /// BLOOM-specific configuration.
    pub bloom_config: BloomConfig,
    /// Token embeddings `[vocab_size, hidden_size]` (f32).
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<BloomLayer>,
    /// Final LayerNorm (with bias).
    pub output_norm: LayerNorm,
    /// LM head weight `[vocab_size, hidden_size]` (f32).
    /// In BLOOM this is tied to token_embd (same matrix), but we store separately.
    pub output_weight: Vec<f32>,
    /// ALiBi positional bias handler.
    alibi: AlibiBias,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_qkv: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_attn_proj: Vec<f32>,
    buf_ffn_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl BloomModel {
    /// Construct a BLOOM model from pre-loaded weights.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<BloomLayer>,
        output_norm: LayerNorm,
        output_weight: Vec<f32>,
    ) -> Self {
        let bloom_config = BloomConfig::from(&config);
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let qkv_dim = num_heads * 3 * head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let kv_dim = num_heads * head_dim; // MHA: kv_heads = num_heads

        let alibi = AlibiBias::new(num_heads);

        Self {
            config,
            bloom_config,
            token_embd,
            layers,
            output_norm,
            output_weight,
            alibi,
            buf_hidden: vec![0.0f32; hidden_size],
            buf_norm: vec![0.0f32; hidden_size],
            buf_qkv: vec![0.0f32; qkv_dim],
            buf_q: vec![0.0f32; num_heads * head_dim],
            buf_k: vec![0.0f32; kv_dim],
            buf_v: vec![0.0f32; kv_dim],
            buf_attn_out: vec![0.0f32; num_heads * head_dim],
            buf_attn_proj: vec![0.0f32; hidden_size],
            buf_ffn_up: vec![0.0f32; intermediate_size],
            buf_ffn_out: vec![0.0f32; hidden_size],
            buf_logits: vec![0.0f32; vocab_size],
            buf_attn_scores: vec![0.0f32; max_ctx * num_heads], // flat [num_heads, seq_k]
        }
    }

    /// Copy the embedding for `token` into `buf_hidden`.
    fn embed_token(&mut self, token: u32) {
        let h = self.config.hidden_size;
        let offset = token as usize * h;
        self.buf_hidden
            .copy_from_slice(&self.token_embd[offset..offset + h]);
    }

    /// Dense matrix-vector product: `out[i] = dot(W[i, :], x)`.
    ///
    /// `W` is stored row-major `[out_dim, in_dim]`.
    fn gemv(w: &[f32], x: &[f32], out: &mut [f32], in_dim: usize) {
        for (i, o) in out.iter_mut().enumerate() {
            let row = &w[i * in_dim..(i + 1) * in_dim];
            *o = row.iter().zip(x.iter()).map(|(w, x)| w * x).sum();
        }
    }

    /// Split the flattened QKV projection into separate Q, K, V buffers.
    ///
    /// Layout: `[Q_0, Q_1, ..., K_0, K_1, ..., V_0, V_1, ...]`
    /// where each block has length `num_heads * head_dim`.
    fn split_qkv(&mut self) {
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let q_len = num_heads * head_dim;
        let k_len = num_heads * head_dim;
        let v_len = num_heads * head_dim;

        self.buf_q.copy_from_slice(&self.buf_qkv[..q_len]);
        self.buf_k
            .copy_from_slice(&self.buf_qkv[q_len..q_len + k_len]);
        self.buf_v
            .copy_from_slice(&self.buf_qkv[q_len + k_len..q_len + k_len + v_len]);
    }

    /// Run scaled dot-product attention with ALiBi bias (no RoPE).
    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = num_heads * head_dim; // MHA
        let qkv_in_dim = self.config.hidden_size;
        let qkv_out_dim = num_heads * 3 * head_dim;
        let seq_len = position + 1;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let layer = &self.layers[layer_idx];

        // QKV projection: buf_qkv = W_qkv @ norm(x) + b_qkv
        Self::gemv(
            &layer.attn_qkv_weight,
            &self.buf_norm,
            &mut self.buf_qkv,
            qkv_in_dim,
        );
        // Add QKV bias
        for i in 0..qkv_out_dim.min(self.buf_qkv.len()) {
            self.buf_qkv[i] += layer.attn_qkv_bias[i];
        }

        // Split into Q, K, V
        self.split_qkv();

        // Store K, V in cache
        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = kv_cache.get_keys(layer_idx)?.to_vec();
        let cached_values = kv_cache.get_values(layer_idx)?.to_vec();

        // Compute raw attention scores: scores[h, 0, k] for decode (seq_q = 1)
        // shape: [num_heads, 1, seq_len] -> flattened as [num_heads * seq_len]
        // We use a flat score buffer here sized [num_heads * seq_len].
        let total_score_elems = num_heads * seq_len;
        if self.buf_attn_scores.len() < total_score_elems {
            self.buf_attn_scores.resize(total_score_elems, 0.0f32);
        }

        for h in 0..num_heads {
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];
            for k_pos in 0..seq_len {
                let k_off = k_pos * kv_dim + h * head_dim;
                let k_vec = &cached_keys[k_off..k_off + head_dim];
                let score: f32 = q_head.iter().zip(k_vec).map(|(q, k)| q * k).sum::<f32>() * scale;
                self.buf_attn_scores[h * seq_len + k_pos] = score;
            }
        }

        // Apply ALiBi bias: shape [num_heads, seq_q=1, seq_k=seq_len]
        // Operate on a slice of size num_heads * 1 * seq_len = num_heads * seq_len
        {
            let score_slice = &mut self.buf_attn_scores[..total_score_elems];
            // For decode: seq_q = 1, q = 0
            // ALiBi bias[h][0][k] = slopes[h] * (k - q) where q = position
            // We need to reshape conceptually to [num_heads, 1, seq_len].
            // Use a temporary contiguous buffer.
            let mut tmp = vec![0.0f32; num_heads * seq_len];
            tmp[..total_score_elems].copy_from_slice(&score_slice[..total_score_elems]);
            // Apply ALiBi: bias[h][0][k] = slopes[h] * (k - position)
            for h in 0..num_heads {
                let slope = self.alibi.slopes()[h];
                for k_pos in 0..seq_len {
                    let relative = k_pos as isize - position as isize;
                    tmp[h * seq_len + k_pos] += slope * relative as f32;
                }
            }
            score_slice[..total_score_elems].copy_from_slice(&tmp[..total_score_elems]);
        }

        // Softmax per head (over seq_len key positions)
        for h in 0..num_heads {
            let s = &mut self.buf_attn_scores[h * seq_len..(h + 1) * seq_len];
            softmax_inplace(s);
        }

        // Accumulate attention output
        self.buf_attn_out.fill(0.0);
        for h in 0..num_heads {
            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for k_pos in 0..seq_len {
                let w = self.buf_attn_scores[h * seq_len + k_pos];
                let v_off = k_pos * kv_dim + h * head_dim;
                let v_vec = &cached_values[v_off..v_off + head_dim];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        // Output projection: proj = W_o @ attn_out + b_o
        let attn_in_dim = num_heads * head_dim;
        let attn_out_copy = self.buf_attn_out.clone();
        let hidden_size = self.config.hidden_size;
        let o_weight = &self.layers[layer_idx].attn_output_weight;
        let o_bias = &self.layers[layer_idx].attn_output_bias;
        Self::gemv(
            o_weight,
            &attn_out_copy,
            &mut self.buf_attn_proj,
            attn_in_dim,
        );
        for (proj, &b) in self
            .buf_attn_proj
            .iter_mut()
            .zip(o_bias.iter())
            .take(hidden_size)
        {
            *proj += b;
        }

        Ok(())
    }

    /// Compute GELU FFN and add to residual.
    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];
        let hidden_size = self.config.hidden_size;
        let ffn_size = self.config.intermediate_size;

        // W1 projection (with bias): ffn_up = gelu(W_up @ norm(x) + b_up)
        Self::gemv(
            &layer.ffn_up_weight,
            &self.buf_norm,
            &mut self.buf_ffn_up,
            hidden_size,
        );
        for i in 0..ffn_size {
            self.buf_ffn_up[i] += layer.ffn_up_bias[i];
        }
        // GELU activation (standard GELU, not SwiGLU)
        gelu_inplace(&mut self.buf_ffn_up);

        // W2 projection (with bias): ffn_out = W_down @ ffn_up + b_down
        let up_copy = self.buf_ffn_up.clone();
        Self::gemv(
            &layer.ffn_down_weight,
            &up_copy,
            &mut self.buf_ffn_out,
            ffn_size,
        );
        for i in 0..hidden_size {
            self.buf_ffn_out[i] += layer.ffn_down_bias[i];
        }

        // Add FFN output to residual
        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }

        Ok(())
    }

    /// Run one transformer layer.
    fn layer_forward(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        // Pre-attention LayerNorm
        {
            let hidden = self.buf_hidden.clone();
            self.layers[layer_idx]
                .attn_norm
                .forward_to(&hidden, &mut self.buf_norm);
        }

        // Attention with ALiBi
        self.attention(layer_idx, position, kv_cache)?;

        // Residual: hidden += attn_proj
        {
            let attn_proj = self.buf_attn_proj.clone();
            for (h, &a) in self.buf_hidden.iter_mut().zip(attn_proj.iter()) {
                *h += a;
            }
        }

        // Pre-FFN LayerNorm
        {
            let hidden = self.buf_hidden.clone();
            self.layers[layer_idx]
                .ffn_norm
                .forward_to(&hidden, &mut self.buf_norm);
        }

        // FFN (GELU) + residual add is inside feed_forward
        self.feed_forward(layer_idx)?;

        Ok(())
    }
}

impl ForwardPass for BloomModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;
            self.embed_token(token);

            for layer_idx in 0..self.layers.len() {
                self.layer_forward(layer_idx, position, kv_cache)?;
            }
            kv_cache.advance();
        }

        // Final LayerNorm
        let hidden_final = self.buf_hidden.clone();
        let mut normed = vec![0.0f32; self.config.hidden_size];
        self.output_norm.forward_to(&hidden_final, &mut normed);

        // LM head: vocab_size × hidden_size
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        self.buf_logits.fill(0.0);
        for v in 0..vocab_size {
            let row = &self.output_weight[v * hidden_size..(v + 1) * hidden_size];
            self.buf_logits[v] = row.iter().zip(normed.iter()).map(|(w, &x)| w * x).sum();
        }

        Ok(self.buf_logits.clone())
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;
            self.embed_token(token);

            for layer_idx in 0..self.layers.len() {
                self.layer_forward(layer_idx, position, kv_cache)?;
            }
            kv_cache.advance();
        }

        let hidden_final = self.buf_hidden.clone();
        let mut normed = vec![0.0f32; self.config.hidden_size];
        self.output_norm.forward_to(&hidden_final, &mut normed);

        Ok(normed)
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

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// Numerically stable in-place softmax.
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
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

// ─── Architecture plugin ──────────────────────────────────────────────────────

/// BLOOM architecture plugin.
pub struct BloomArchitecture;

impl BloomArchitecture {
    /// Create a new `BloomArchitecture` instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BloomArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for BloomArchitecture {
    fn arch_id(&self) -> &str {
        "bloom"
    }

    fn build(
        &self,
        config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        if config.num_attention_heads == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "num_attention_heads".to_string(),
                expected: ">0".to_string(),
                got: "0".to_string(),
            });
        }
        if config.hidden_size == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "hidden_size".to_string(),
                expected: ">0".to_string(),
                got: "0".to_string(),
            });
        }
        // The full loading path is in `load_bloom_from_gguf`.
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use load_bloom_from_gguf for full loading)".to_string(),
        })
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        let mut patterns = vec![
            TensorNamePattern {
                pattern: "token_embd.weight".to_string(),
                description: "Token embedding matrix".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.weight".to_string(),
                description: "Final LayerNorm scale".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.bias".to_string(),
                description: "Final LayerNorm bias".to_string(),
                required: true,
            },
        ];

        let per_layer_tensors: &[(&str, &str, bool)] = &[
            (
                "blk.{l}.attn_norm.weight",
                "Pre-attention LayerNorm scale",
                true,
            ),
            (
                "blk.{l}.attn_norm.bias",
                "Pre-attention LayerNorm bias",
                true,
            ),
            (
                "blk.{l}.attn_qkv.weight",
                "Fused QKV projection weight",
                true,
            ),
            ("blk.{l}.attn_qkv.bias", "Fused QKV projection bias", true),
            (
                "blk.{l}.attn_output.weight",
                "Attention output projection weight",
                true,
            ),
            (
                "blk.{l}.attn_output.bias",
                "Attention output projection bias",
                true,
            ),
            ("blk.{l}.ffn_norm.weight", "Pre-FFN LayerNorm scale", true),
            ("blk.{l}.ffn_norm.bias", "Pre-FFN LayerNorm bias", true),
            ("blk.{l}.ffn_up.weight", "FFN W1 projection weight", true),
            ("blk.{l}.ffn_up.bias", "FFN W1 projection bias", true),
            ("blk.{l}.ffn_down.weight", "FFN W2 projection weight", true),
            ("blk.{l}.ffn_down.bias", "FFN W2 projection bias", true),
        ];

        for (pat, desc, req) in per_layer_tensors {
            patterns.push(TensorNamePattern {
                pattern: pat.to_string(),
                description: desc.to_string(),
                required: *req,
            });
        }

        patterns
    }
}

// ─── GGUF loader ──────────────────────────────────────────────────────────────

/// Load a BLOOM model from a parsed GGUF file.
pub fn load_bloom_from_gguf(model: &GgufModel, config: &ModelConfig) -> ArchResult<BloomModel> {
    let dispatcher = oxillama_quant::KernelDispatcher::new();
    let num_heads = config.num_attention_heads;
    let head_dim = config.head_dim;
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let qkv_dim = num_heads * 3 * head_dim;

    // Token embeddings
    let token_embd = load_f32_tensor(model, "token_embd.weight", &dispatcher)?;

    // Transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        // Pre-attention LayerNorm
        let attn_ln_w = load_f32_tensor(model, &format!("{prefix}.attn_norm.weight"), &dispatcher)?;
        let attn_ln_b = load_f32_tensor(model, &format!("{prefix}.attn_norm.bias"), &dispatcher)?;
        let attn_norm = LayerNorm::new(attn_ln_w, Some(attn_ln_b), config.rms_norm_eps);

        // Fused QKV
        let attn_qkv_weight =
            load_f32_tensor(model, &format!("{prefix}.attn_qkv.weight"), &dispatcher)?;
        let attn_qkv_bias_raw =
            load_f32_tensor(model, &format!("{prefix}.attn_qkv.bias"), &dispatcher)?;
        let mut attn_qkv_bias = vec![0.0f32; qkv_dim];
        attn_qkv_bias[..attn_qkv_bias_raw.len().min(qkv_dim)]
            .copy_from_slice(&attn_qkv_bias_raw[..attn_qkv_bias_raw.len().min(qkv_dim)]);

        // Attention output projection
        let attn_output_weight =
            load_f32_tensor(model, &format!("{prefix}.attn_output.weight"), &dispatcher)?;
        let attn_output_bias_raw =
            load_f32_tensor(model, &format!("{prefix}.attn_output.bias"), &dispatcher)?;
        let mut attn_output_bias = vec![0.0f32; hidden_size];
        attn_output_bias[..attn_output_bias_raw.len().min(hidden_size)]
            .copy_from_slice(&attn_output_bias_raw[..attn_output_bias_raw.len().min(hidden_size)]);

        // Pre-FFN LayerNorm
        let ffn_ln_w = load_f32_tensor(model, &format!("{prefix}.ffn_norm.weight"), &dispatcher)?;
        let ffn_ln_b = load_f32_tensor(model, &format!("{prefix}.ffn_norm.bias"), &dispatcher)?;
        let ffn_norm = LayerNorm::new(ffn_ln_w, Some(ffn_ln_b), config.rms_norm_eps);

        // FFN up (W1)
        let ffn_up_weight =
            load_f32_tensor(model, &format!("{prefix}.ffn_up.weight"), &dispatcher)?;
        let ffn_up_bias_raw =
            load_f32_tensor(model, &format!("{prefix}.ffn_up.bias"), &dispatcher)?;
        let mut ffn_up_bias = vec![0.0f32; intermediate_size];
        ffn_up_bias[..ffn_up_bias_raw.len().min(intermediate_size)]
            .copy_from_slice(&ffn_up_bias_raw[..ffn_up_bias_raw.len().min(intermediate_size)]);

        // FFN down (W2)
        let ffn_down_weight =
            load_f32_tensor(model, &format!("{prefix}.ffn_down.weight"), &dispatcher)?;
        let ffn_down_bias_raw =
            load_f32_tensor(model, &format!("{prefix}.ffn_down.bias"), &dispatcher)?;
        let mut ffn_down_bias = vec![0.0f32; hidden_size];
        ffn_down_bias[..ffn_down_bias_raw.len().min(hidden_size)]
            .copy_from_slice(&ffn_down_bias_raw[..ffn_down_bias_raw.len().min(hidden_size)]);

        layers.push(BloomLayer {
            attn_norm,
            attn_qkv_weight,
            attn_qkv_bias,
            attn_output_weight,
            attn_output_bias,
            ffn_norm,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
        });
    }

    // Final LayerNorm
    let output_norm_w = load_f32_tensor(model, "output_norm.weight", &dispatcher)?;
    let output_norm_b = load_f32_tensor(model, "output_norm.bias", &dispatcher)?;
    let output_norm = LayerNorm::new(output_norm_w, Some(output_norm_b), config.rms_norm_eps);

    // LM head — BLOOM ties weights with token_embd. If output.weight is present, use it.
    // Otherwise fall back to token_embd (weight tying).
    let output_weight = if model.file.tensors.contains("output.weight") {
        load_f32_tensor(model, "output.weight", &dispatcher)?
    } else {
        // Weight tying
        token_embd.clone()
    };

    Ok(BloomModel::new(
        config.clone(),
        token_embd,
        layers,
        output_norm,
        output_weight,
    ))
}

/// Load a tensor as `f32`, dequantizing if necessary.
fn load_f32_tensor(
    model: &GgufModel,
    name: &str,
    dispatcher: &oxillama_quant::KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model.tensor_data(name)?;
    let n_elements = info.n_elements() as usize;
    let tensor_type = info.tensor_type;

    if tensor_type == GgufTensorType::F32 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(4).enumerate().take(n_elements) {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    if tensor_type == GgufTensorType::F16 {
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::error::ArchResult;
    use crate::registry::ArchitectureRegistry;
    use crate::traits::KvCacheAccess;

    // ── KV cache stub ─────────────────────────────────────────────────────────

    struct SimpleKvCache {
        n_layers: usize,
        kv_dim: usize,
        max_seq: usize,
        position: usize,
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
    }

    impl SimpleKvCache {
        fn new(n_layers: usize, kv_dim: usize, max_seq: usize) -> Self {
            Self {
                n_layers,
                kv_dim,
                max_seq,
                position: 0,
                keys: vec![vec![0.0f32; max_seq * kv_dim]; n_layers],
                values: vec![vec![0.0f32; max_seq * kv_dim]; n_layers],
            }
        }
    }

    impl KvCacheAccess for SimpleKvCache {
        fn seq_len(&self) -> usize {
            self.position
        }
        fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
            if layer >= self.n_layers {
                return Err(ArchError::InvalidConfig {
                    detail: format!("layer {layer} out of range"),
                });
            }
            let offset = self.position * self.kv_dim;
            let ck = key.len().min(self.kv_dim);
            let cv = value.len().min(self.kv_dim);
            self.keys[layer][offset..offset + ck].copy_from_slice(&key[..ck]);
            self.values[layer][offset..offset + cv].copy_from_slice(&value[..cv]);
            Ok(())
        }
        fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
            if layer >= self.n_layers {
                return Err(ArchError::InvalidConfig {
                    detail: format!("layer {layer} out of range"),
                });
            }
            let end = (self.position + 1) * self.kv_dim;
            Ok(&self.keys[layer][..end])
        }
        fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
            if layer >= self.n_layers {
                return Err(ArchError::InvalidConfig {
                    detail: format!("layer {layer} out of range"),
                });
            }
            let end = (self.position + 1) * self.kv_dim;
            Ok(&self.values[layer][..end])
        }
        fn advance(&mut self) {
            self.position = (self.position + 1).min(self.max_seq - 1);
        }
    }

    // ── Helper: build a tiny BLOOM model ─────────────────────────────────────

    fn minimal_config() -> ModelConfig {
        let hidden = 64usize;
        let heads = 8usize;
        let head_dim = hidden / heads; // 8
        ModelConfig {
            architecture: "bloom".to_string(),
            hidden_size: hidden,
            intermediate_size: hidden * 4, // 256
            num_layers: 1,
            num_attention_heads: heads,
            num_kv_heads: heads, // MHA
            head_dim,
            vocab_size: 32,
            max_context_length: 64,
            rms_norm_eps: 1e-5,
            ..ModelConfig::default()
        }
    }

    fn build_tiny_bloom() -> BloomModel {
        let config = minimal_config();
        let h = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let qkv_dim = num_heads * 3 * head_dim;
        let ffn_size = config.intermediate_size;
        let v = config.vocab_size;

        let ln = LayerNorm::new(vec![1.0f32; h], Some(vec![0.0f32; h]), 1e-5f32);
        let layer = BloomLayer {
            attn_norm: ln.clone(),
            attn_qkv_weight: vec![0.01f32; qkv_dim * h],
            attn_qkv_bias: vec![0.0f32; qkv_dim],
            attn_output_weight: vec![0.01f32; h * (num_heads * head_dim)],
            attn_output_bias: vec![0.0f32; h],
            ffn_norm: ln.clone(),
            ffn_up_weight: vec![0.01f32; ffn_size * h],
            ffn_up_bias: vec![0.0f32; ffn_size],
            ffn_down_weight: vec![0.01f32; h * ffn_size],
            ffn_down_bias: vec![0.0f32; h],
        };

        let final_ln = LayerNorm::new(vec![1.0f32; h], Some(vec![0.0f32; h]), 1e-5f32);
        let token_embd = vec![0.01f32; v * h];
        let output_weight = vec![0.01f32; v * h];

        BloomModel::new(config, token_embd, vec![layer], final_ln, output_weight)
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn bloom_registry_lookup() {
        let reg = ArchitectureRegistry::with_builtins();
        let arch = reg.get("bloom");
        assert!(
            arch.is_ok(),
            "registry.get('bloom') must succeed; got: {:?}",
            arch.err()
        );
        assert_eq!(arch.expect("bloom arch").arch_id(), "bloom");
    }

    #[test]
    fn bloom_tensor_names_complete() {
        let arch = BloomArchitecture::new();
        let names = arch.tensor_names();
        assert!(
            !names.is_empty(),
            "tensor_names() must return at least one pattern"
        );
        let has_embd = names.iter().any(|n| n.pattern == "token_embd.weight");
        assert!(has_embd, "tensor_names must include 'token_embd.weight'");
        let has_qkv = names.iter().any(|n| n.pattern.contains("attn_qkv"));
        assert!(has_qkv, "tensor_names must include attn_qkv pattern");
    }

    #[test]
    fn bloom_alibi_slopes_match_reference() {
        // For 8 heads, start = 2^(-8/8) = 2^(-1) = 0.5.
        // slopes[0] = start^1 = 0.5.
        let bias = AlibiBias::new(8);
        let slope_0 = bias.slopes()[0];
        let expected = 0.5f32;
        assert!(
            (slope_0 - expected).abs() < 1e-5,
            "slopes[0] for 8 heads should be 0.5 (= 2^(-1)), got {slope_0}"
        );
    }

    #[test]
    fn bloom_forward_shape_and_finite() {
        let mut model = build_tiny_bloom();
        let vocab = model.vocab_size();
        let kv_dim = model.config.num_attention_heads * model.config.head_dim;
        let mut kv = SimpleKvCache::new(1, kv_dim, model.max_context_length());

        let result = model.forward(&[0u32], &mut kv);
        assert!(
            result.is_ok(),
            "bloom forward must succeed: {:?}",
            result.err()
        );
        let logits = result.expect("logits");
        assert_eq!(logits.len(), vocab, "logits must have vocab_size elements");
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
        }
    }

    #[test]
    fn bloom_no_rope_present() {
        // BLOOM uses ALiBi, so tensor names must NOT contain rope_freqs.
        let arch = BloomArchitecture::new();
        let names = arch.tensor_names();
        let has_rope = names.iter().any(|n| n.pattern.contains("rope_freqs"));
        assert!(!has_rope, "BLOOM tensor names must not include rope_freqs");
    }

    /// Load BLOOM from a minimal synthetic GGUF — checks end-to-end path.
    #[test]
    fn bloom_load_from_synthetic_gguf() {
        use oxillama_gguf::test_utils::build_minimal_bloom_gguf;
        let bytes = build_minimal_bloom_gguf();
        let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("parse GGUF");
        let config =
            crate::config::ModelConfig::from_metadata(&gguf.file.metadata).expect("parse config");
        let mut model = load_bloom_from_gguf(&gguf, &config).expect("load bloom");
        let kv_dim = config.num_attention_heads * config.head_dim;
        let mut kv = SimpleKvCache::new(config.num_layers, kv_dim, config.max_context_length);
        let logits = model.forward(&[0u32], &mut kv).expect("forward");
        assert_eq!(logits.len(), config.vocab_size);
        for &v in &logits {
            assert!(v.is_finite(), "logit must be finite: {v}");
        }
    }
}

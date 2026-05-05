//! Phi-3.5-MoE transformer forward pass implementation.
//!
//! Phi-3.5-MoE is a Mixture-of-Experts variant of the Phi-3 architecture.
//! It uses the Phi-3 attention style with merged QKV projections and partial
//! RoPE, combined with a sparse MoE FFN where the top-2 experts (of 16) are
//! activated per token.
//!
//! ## Architecture
//!
//! ```text
//! embed → for each layer:
//!   RMSNorm(hidden) → QKV → partial-RoPE → GQA → output-proj → residual
//!   RMSNorm(hidden) → router → top-2 expert selection →
//!                     SwiGLU expert FFN × top-2 → weighted sum → residual
//! → RMSNorm → LM head
//! ```
//!
//! ## Tensor naming (GGUF)
//!
//! | Tensor name                           | Description                         |
//! |---------------------------------------|-------------------------------------|
//! | `token_embd.weight`                   | Token embedding matrix              |
//! | `output_norm.weight`                  | Final RMSNorm scale                 |
//! | `output.weight`                       | LM head projection                  |
//! | `blk.{l}.attn_norm.weight`            | Pre-attention RMSNorm scale         |
//! | `blk.{l}.attn_qkv.weight`             | Fused QKV (same as Phi-3)           |
//! | `blk.{l}.attn_output.weight`          | Attention output projection         |
//! | `blk.{l}.ffn_norm.weight`             | Pre-FFN RMSNorm scale               |
//! | `blk.{l}.ffn_gate_inp.weight`         | Router: `[num_experts, hidden_size]`|
//! | `blk.{l}.ffn_gate_exps.weight`        | Stacked gate projections            |
//! | `blk.{l}.ffn_up_exps.weight`          | Stacked up projections              |
//! | `blk.{l}.ffn_down_exps.weight`        | Stacked down projections            |

use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, GgufTensorType, TensorStore};

use super::config::PhiMoeConfig;

// ─── Layer definition ─────────────────────────────────────────────────────────

/// A single Phi-MoE transformer layer.
pub struct PhiMoeLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Fused QKV weight `[(num_q + 2*num_kv) * head_dim, hidden_size]` (f32).
    pub attn_qkv_weight: Vec<f32>,
    /// Attention output projection weight `[hidden_size, num_q * head_dim]` (f32).
    pub attn_output_weight: Vec<f32>,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// Router weight matrix `[num_experts, hidden_size]` (f32).
    pub router_weight: Vec<f32>,
    /// Gate projections for all experts: `num_experts` rows of
    /// `[intermediate_size, hidden_size]` stored contiguously.
    pub ffn_gate_exps: Vec<f32>,
    /// Up projections for all experts: same layout as `ffn_gate_exps`.
    pub ffn_up_exps: Vec<f32>,
    /// Down projections for all experts: `num_experts` rows of
    /// `[hidden_size, intermediate_size]` stored contiguously.
    pub ffn_down_exps: Vec<f32>,
}

// ─── Model ────────────────────────────────────────────────────────────────────

/// Complete Phi-3.5-MoE model.
pub struct PhiMoeModel {
    /// Base model configuration.
    pub config: ModelConfig,
    /// Phi-MoE specific configuration.
    pub phi_moe_config: PhiMoeConfig,
    /// Token embeddings `[vocab_size, hidden_size]` (f32).
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<PhiMoeLayer>,
    /// Final RMSNorm.
    pub output_norm: RmsNorm,
    /// LM head weight `[vocab_size, hidden_size]` (f32).
    pub output_weight: Vec<f32>,
    /// Precomputed partial RoPE frequency table.
    rope: RopeTable,
    /// Number of head dimensions to apply RoPE to.
    rope_dims: usize,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_qkv: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_attn_proj: Vec<f32>,
    buf_router_logits: Vec<f32>,
    buf_expert_gate: Vec<f32>,
    buf_expert_up: Vec<f32>,
    buf_expert_out: Vec<f32>,
    buf_moe_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl PhiMoeModel {
    /// Construct a Phi-MoE model from pre-loaded weights.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<PhiMoeLayer>,
        output_norm: RmsNorm,
        output_weight: Vec<f32>,
        partial_rotary_factor: f32,
    ) -> Self {
        let phi_moe_config = PhiMoeConfig::from(&config);
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let num_experts = phi_moe_config.num_experts;

        let rope_dims = ((head_dim as f32 * partial_rotary_factor) as usize).max(2);
        let rope_dims = (rope_dims & !1).min(head_dim); // must be even

        let rope = RopeTable::new(
            rope_dims,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
        );

        Self {
            config,
            phi_moe_config,
            token_embd,
            layers,
            output_norm,
            output_weight,
            rope,
            rope_dims,
            buf_hidden: vec![0.0f32; hidden_size],
            buf_norm: vec![0.0f32; hidden_size],
            buf_qkv: vec![0.0f32; qkv_dim],
            buf_q: vec![0.0f32; q_dim],
            buf_k: vec![0.0f32; kv_dim],
            buf_v: vec![0.0f32; kv_dim],
            buf_attn_out: vec![0.0f32; q_dim],
            buf_attn_proj: vec![0.0f32; hidden_size],
            buf_router_logits: vec![0.0f32; num_experts],
            buf_expert_gate: vec![0.0f32; intermediate_size],
            buf_expert_up: vec![0.0f32; intermediate_size],
            buf_expert_out: vec![0.0f32; hidden_size],
            buf_moe_out: vec![0.0f32; hidden_size],
            buf_logits: vec![0.0f32; vocab_size],
            buf_attn_scores: vec![0.0f32; max_ctx],
        }
    }

    fn embed_token(&mut self, token: u32) {
        let h = self.config.hidden_size;
        let offset = token as usize * h;
        self.buf_hidden
            .copy_from_slice(&self.token_embd[offset..offset + h]);
    }

    /// Dense matrix-vector product: `out[i] = dot(W[i, :], x)`.
    fn gemv(w: &[f32], x: &[f32], out: &mut [f32], in_dim: usize) {
        for (i, o) in out.iter_mut().enumerate() {
            let row = &w[i * in_dim..(i + 1) * in_dim];
            *o = row.iter().zip(x.iter()).map(|(w, x)| w * x).sum();
        }
    }

    fn split_qkv(&mut self) {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let q_len = num_heads * head_dim;
        let kv_len = num_kv_heads * head_dim;

        self.buf_q.copy_from_slice(&self.buf_qkv[..q_len]);
        self.buf_k
            .copy_from_slice(&self.buf_qkv[q_len..q_len + kv_len]);
        self.buf_v
            .copy_from_slice(&self.buf_qkv[q_len + kv_len..q_len + 2 * kv_len]);
    }

    /// Run attention with partial RoPE and GQA.
    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let heads_per_kv = num_heads.max(1) / num_kv_heads.max(1);
        let seq_len = position + 1;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let hidden_size = self.config.hidden_size;
        let q_out_dim = num_heads * head_dim;
        let qkv_dim = q_out_dim + 2 * kv_dim;

        let layer = &self.layers[layer_idx];

        // Fused QKV projection
        Self::gemv(
            &layer.attn_qkv_weight,
            &self.buf_norm,
            &mut self.buf_qkv,
            hidden_size,
        );

        // Split
        self.split_qkv();

        // Partial RoPE applied to first `rope_dims` dimensions per head
        let rope_dims = self.rope_dims;
        let _ = qkv_dim; // used above

        // Snapshot cos/sin to avoid split-borrow
        let rotate_half = rope_dims / 2;
        let rope_offset = position * self.rope.half_dim;
        let rope_cos: Vec<f32> =
            if rotate_half > 0 && rope_offset + rotate_half <= self.rope.cos.len() {
                self.rope.cos[rope_offset..rope_offset + rotate_half].to_vec()
            } else {
                vec![]
            };
        let rope_sin: Vec<f32> =
            if rotate_half > 0 && rope_offset + rotate_half <= self.rope.sin.len() {
                self.rope.sin[rope_offset..rope_offset + rotate_half].to_vec()
            } else {
                vec![]
            };

        for h in 0..num_heads {
            let q_head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
            apply_partial_rope(q_head, &rope_cos, &rope_sin, rope_dims);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            apply_partial_rope(k_head, &rope_cos, &rope_sin, rope_dims);
        }

        // Store K, V
        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = kv_cache.get_keys(layer_idx)?.to_vec();
        let cached_values = kv_cache.get_values(layer_idx)?.to_vec();

        self.buf_attn_out.fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in 0..seq_len {
                let k_off = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_off..k_off + head_dim];
                let score: f32 = q_head.iter().zip(k_vec).map(|(q, k)| q * k).sum::<f32>() * scale;
                self.buf_attn_scores[pos] = score;
            }

            softmax_inplace(&mut self.buf_attn_scores[..seq_len]);

            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for pos in 0..seq_len {
                let v_off = pos * kv_dim + kv_head * head_dim;
                let v_vec = &cached_values[v_off..v_off + head_dim];
                let w = self.buf_attn_scores[pos];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        // Output projection
        let attn_out_copy = self.buf_attn_out.clone();
        let attn_in_dim = num_heads * head_dim;
        Self::gemv(
            &self.layers[layer_idx].attn_output_weight,
            &attn_out_copy,
            &mut self.buf_attn_proj,
            attn_in_dim,
        );

        Ok(())
    }

    /// Run the sparse MoE FFN (top-K expert selection with SwiGLU per expert).
    fn moe_ffn(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_experts = self.phi_moe_config.num_experts;
        let top_k = self.phi_moe_config.num_experts_per_tok;

        // 1. Router logits: [num_experts] = W_router @ hidden_norm
        Self::gemv(
            &layer.router_weight,
            &self.buf_norm,
            &mut self.buf_router_logits,
            hidden_size,
        );

        // 2. Softmax over router logits
        {
            let logits = &mut self.buf_router_logits[..num_experts];
            softmax_inplace(logits);
        }

        // 3. Select top-K experts
        let mut indices: Vec<usize> = (0..num_experts).collect();
        let router_logits_copy: Vec<f32> = self.buf_router_logits[..num_experts].to_vec();
        indices.sort_unstable_by(|&a, &b| {
            router_logits_copy[b]
                .partial_cmp(&router_logits_copy[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let effective_k = top_k.min(num_experts);
        let top_k_indices = &indices[..effective_k];

        // 4. Re-normalise weights for the selected experts
        let selected_sum: f32 = top_k_indices.iter().map(|&i| router_logits_copy[i]).sum();

        // 5. Accumulate weighted expert outputs
        self.buf_moe_out.fill(0.0);
        let norm_input = self.buf_norm.clone();

        for &expert_idx in top_k_indices {
            let weight = if selected_sum > 1e-9 {
                router_logits_copy[expert_idx] / selected_sum
            } else {
                1.0 / effective_k as f32
            };

            // Each expert is stored at offset `expert_idx * intermediate_size * hidden_size`
            let gate_offset = expert_idx * intermediate_size * hidden_size;
            let up_offset = expert_idx * intermediate_size * hidden_size;
            let down_offset = expert_idx * hidden_size * intermediate_size;

            let gate_w =
                &layer.ffn_gate_exps[gate_offset..gate_offset + intermediate_size * hidden_size];
            let up_w = &layer.ffn_up_exps[up_offset..up_offset + intermediate_size * hidden_size];
            let down_w =
                &layer.ffn_down_exps[down_offset..down_offset + hidden_size * intermediate_size];

            // SwiGLU: gate_vec = silu(W_gate @ x), up_vec = W_up @ x, out = W_down @ (gate_vec * up_vec)
            Self::gemv(gate_w, &norm_input, &mut self.buf_expert_gate, hidden_size);
            Self::gemv(up_w, &norm_input, &mut self.buf_expert_up, hidden_size);

            // SiLU activation on gate
            for g in self.buf_expert_gate.iter_mut() {
                *g *= 1.0 / (1.0 + (-*g).exp()); // silu(x) = x * sigmoid(x)
            }

            // Element-wise gate * up
            for (g, &u) in self
                .buf_expert_gate
                .iter_mut()
                .zip(self.buf_expert_up.iter())
            {
                *g *= u;
            }

            let gate_copy = self.buf_expert_gate.clone();
            Self::gemv(
                down_w,
                &gate_copy,
                &mut self.buf_expert_out,
                intermediate_size,
            );

            // Accumulate with weight
            for (o, &e) in self.buf_moe_out.iter_mut().zip(self.buf_expert_out.iter()) {
                *o += weight * e;
            }
        }

        // Add MoE output to residual
        for (h, &m) in self.buf_hidden.iter_mut().zip(self.buf_moe_out.iter()) {
            *h += m;
        }

        Ok(())
    }

    fn layer_forward(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        // Pre-attention RMSNorm
        {
            let hidden = self.buf_hidden.clone();
            self.layers[layer_idx]
                .attn_norm
                .forward_to(&hidden, &mut self.buf_norm);
        }

        self.attention(layer_idx, position, kv_cache)?;

        // Residual add
        {
            let proj = self.buf_attn_proj.clone();
            for (h, &a) in self.buf_hidden.iter_mut().zip(proj.iter()) {
                *h += a;
            }
        }

        // Pre-FFN RMSNorm
        {
            let hidden = self.buf_hidden.clone();
            self.layers[layer_idx]
                .ffn_norm
                .forward_to(&hidden, &mut self.buf_norm);
        }

        // MoE FFN (residual add is inside moe_ffn)
        self.moe_ffn(layer_idx)?;

        Ok(())
    }
}

/// Apply partial RoPE using pre-extracted cos/sin slices.
fn apply_partial_rope(head: &mut [f32], rope_cos: &[f32], rope_sin: &[f32], rotary_dims: usize) {
    if rotary_dims == 0 || rope_cos.is_empty() || rope_sin.is_empty() {
        return;
    }
    let rotate_half = rotary_dims / 2;
    for i in 0..rotate_half
        .min(rope_cos.len())
        .min(head.len().saturating_sub(rotate_half))
    {
        let x0 = head[i];
        let x1 = head[i + rotate_half];
        head[i] = x0 * rope_cos[i] - x1 * rope_sin[i];
        head[i + rotate_half] = x0 * rope_sin[i] + x1 * rope_cos[i];
    }
}

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

impl ForwardPass for PhiMoeModel {
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

        // Final RMSNorm
        self.output_norm.forward(&mut self.buf_hidden);

        // LM head
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        let hidden_final = self.buf_hidden.clone();
        self.buf_logits.fill(0.0);
        for v in 0..vocab_size {
            let row = &self.output_weight[v * hidden_size..(v + 1) * hidden_size];
            self.buf_logits[v] = row
                .iter()
                .zip(hidden_final.iter())
                .map(|(w, &x)| w * x)
                .sum();
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

// ─── Architecture plugin ──────────────────────────────────────────────────────

/// Phi-3.5-MoE architecture plugin.
pub struct PhiMoeArchitecture;

impl PhiMoeArchitecture {
    /// Create a new `PhiMoeArchitecture` instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for PhiMoeArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for PhiMoeArchitecture {
    fn arch_id(&self) -> &str {
        "phimoe"
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
        if config.num_experts == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "num_experts".to_string(),
                expected: ">0".to_string(),
                got: "0".to_string(),
            });
        }
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use load_phi_moe_from_gguf for full loading)".to_string(),
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
                description: "Final RMSNorm scale".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head projection".to_string(),
                required: true,
            },
        ];

        let per_layer_tensors: &[(&str, &str, bool)] = &[
            (
                "blk.{l}.attn_norm.weight",
                "Pre-attention RMSNorm scale",
                true,
            ),
            (
                "blk.{l}.attn_qkv.weight",
                "Fused QKV projection weight",
                true,
            ),
            (
                "blk.{l}.attn_output.weight",
                "Attention output projection",
                true,
            ),
            ("blk.{l}.ffn_norm.weight", "Pre-FFN RMSNorm scale", true),
            ("blk.{l}.ffn_gate_inp.weight", "MoE router weight", true),
            (
                "blk.{l}.ffn_gate_exps.weight",
                "Stacked gate projections",
                true,
            ),
            ("blk.{l}.ffn_up_exps.weight", "Stacked up projections", true),
            (
                "blk.{l}.ffn_down_exps.weight",
                "Stacked down projections",
                true,
            ),
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

/// Load a Phi-MoE model from a parsed GGUF file.
pub fn load_phi_moe_from_gguf(model: &GgufModel, config: &ModelConfig) -> ArchResult<PhiMoeModel> {
    let dispatcher = oxillama_quant::KernelDispatcher::new();
    let hidden_size = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_dim = q_dim + 2 * kv_dim;
    let intermediate_size = config.intermediate_size;
    let num_experts = config.num_experts.max(1);

    // Partial rotary factor from GGUF metadata
    let partial_rotary_factor = model
        .file
        .metadata
        .get_f32(&format!(
            "{}.rope.partial_rotary_factor",
            config.architecture
        ))
        .unwrap_or(0.5);

    // Token embeddings
    let token_embd = load_f32_tensor(model, "token_embd.weight", &dispatcher)?;

    // Transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        // Pre-attention RMSNorm
        let attn_norm_w =
            load_f32_tensor(model, &format!("{prefix}.attn_norm.weight"), &dispatcher)?;
        let attn_norm = RmsNorm::new(attn_norm_w, config.rms_norm_eps);

        // Fused QKV
        let attn_qkv_weight =
            load_f32_tensor(model, &format!("{prefix}.attn_qkv.weight"), &dispatcher)?;

        // Attention output projection
        let attn_output_weight =
            load_f32_tensor(model, &format!("{prefix}.attn_output.weight"), &dispatcher)?;

        // Pre-FFN RMSNorm
        let ffn_norm_w = load_f32_tensor(model, &format!("{prefix}.ffn_norm.weight"), &dispatcher)?;
        let ffn_norm = RmsNorm::new(ffn_norm_w, config.rms_norm_eps);

        // Router: [num_experts, hidden_size]
        let router_weight =
            load_f32_tensor(model, &format!("{prefix}.ffn_gate_inp.weight"), &dispatcher)?;

        // Stacked expert weight tensors.
        // Shape in GGUF: [num_experts, ffn_hidden, hidden_size] for gate/up,
        //                [num_experts, hidden_size, ffn_hidden] for down.
        let ffn_gate_exps = load_f32_tensor(
            model,
            &format!("{prefix}.ffn_gate_exps.weight"),
            &dispatcher,
        )?;
        let ffn_up_exps =
            load_f32_tensor(model, &format!("{prefix}.ffn_up_exps.weight"), &dispatcher)?;
        let ffn_down_exps = load_f32_tensor(
            model,
            &format!("{prefix}.ffn_down_exps.weight"),
            &dispatcher,
        )?;

        // Validate expert weight sizes
        let expected_gate_up = num_experts * intermediate_size * hidden_size;
        let expected_down = num_experts * hidden_size * intermediate_size;
        if ffn_gate_exps.len() != expected_gate_up {
            return Err(ArchError::InvalidShape {
                name: format!("{prefix}.ffn_gate_exps.weight"),
                expected: vec![num_experts, intermediate_size, hidden_size],
                got: vec![ffn_gate_exps.len()],
            });
        }
        if ffn_up_exps.len() != expected_gate_up {
            return Err(ArchError::InvalidShape {
                name: format!("{prefix}.ffn_up_exps.weight"),
                expected: vec![num_experts, intermediate_size, hidden_size],
                got: vec![ffn_up_exps.len()],
            });
        }
        if ffn_down_exps.len() != expected_down {
            return Err(ArchError::InvalidShape {
                name: format!("{prefix}.ffn_down_exps.weight"),
                expected: vec![num_experts, hidden_size, intermediate_size],
                got: vec![ffn_down_exps.len()],
            });
        }

        let _ = (qkv_dim, kv_dim); // used in split_qkv via config

        layers.push(PhiMoeLayer {
            attn_norm,
            attn_qkv_weight,
            attn_output_weight,
            ffn_norm,
            router_weight,
            ffn_gate_exps,
            ffn_up_exps,
            ffn_down_exps,
        });
    }

    // Final RMSNorm and output projection
    let output_norm_w = load_f32_tensor(model, "output_norm.weight", &dispatcher)?;
    let output_norm = RmsNorm::new(output_norm_w, config.rms_norm_eps);
    let output_weight = load_f32_tensor(model, "output.weight", &dispatcher)?;

    Ok(PhiMoeModel::new(
        config.clone(),
        token_embd,
        layers,
        output_norm,
        output_weight,
        partial_rotary_factor,
    ))
}

/// Load a tensor as f32, dequantizing if necessary.
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

    // ── Helper: build a tiny PhiMoE model ────────────────────────────────────

    fn minimal_phi_moe_config() -> ModelConfig {
        let hidden = 64usize;
        let num_heads = 8usize;
        let num_kv_heads = 4usize;
        let head_dim = hidden / num_heads; // 8
        ModelConfig {
            architecture: "phimoe".to_string(),
            hidden_size: hidden,
            intermediate_size: 64, // per-expert intermediate
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            vocab_size: 32,
            max_context_length: 64,
            rms_norm_eps: 1e-5,
            num_experts: 4,
            num_experts_used: 2,
            ..ModelConfig::default()
        }
    }

    fn build_tiny_phi_moe() -> PhiMoeModel {
        let config = minimal_phi_moe_config();
        let h = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let ffn = config.intermediate_size;
        let n_exp = config.num_experts;
        let v = config.vocab_size;

        let layer = PhiMoeLayer {
            attn_norm: RmsNorm::new(vec![1.0f32; h], 1e-5),
            attn_qkv_weight: vec![0.01f32; qkv_dim * h],
            attn_output_weight: vec![0.01f32; h * q_dim],
            ffn_norm: RmsNorm::new(vec![1.0f32; h], 1e-5),
            router_weight: vec![0.01f32; n_exp * h],
            ffn_gate_exps: vec![0.01f32; n_exp * ffn * h],
            ffn_up_exps: vec![0.01f32; n_exp * ffn * h],
            ffn_down_exps: vec![0.01f32; n_exp * h * ffn],
        };

        let output_norm = RmsNorm::new(vec![1.0f32; h], 1e-5);
        let token_embd = vec![0.01f32; v * h];
        let output_weight = vec![0.01f32; v * h];

        PhiMoeModel::new(
            config,
            token_embd,
            vec![layer],
            output_norm,
            output_weight,
            0.5,
        )
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn phi_moe_registry_lookup() {
        let reg = ArchitectureRegistry::with_builtins();
        let arch = reg.get("phimoe");
        assert!(
            arch.is_ok(),
            "registry.get('phimoe') must succeed; got: {:?}",
            arch.err()
        );
        assert_eq!(arch.expect("phimoe arch").arch_id(), "phimoe");
    }

    #[test]
    fn phi_moe_tensor_names_complete() {
        let arch = PhiMoeArchitecture::new();
        let names = arch.tensor_names();
        assert!(
            !names.is_empty(),
            "tensor_names() must return at least one pattern"
        );
        let has_embd = names.iter().any(|n| n.pattern == "token_embd.weight");
        assert!(has_embd, "must contain 'token_embd.weight'");
        let has_router = names.iter().any(|n| n.pattern.contains("ffn_gate_inp"));
        assert!(has_router, "must contain router (ffn_gate_inp) pattern");
        let has_experts = names.iter().any(|n| n.pattern.contains("ffn_gate_exps"));
        assert!(has_experts, "must contain expert (ffn_gate_exps) pattern");
    }

    /// Top-2 expert weights after softmax: their sum before re-normalisation is ≤ 1.0.
    #[test]
    fn phi_moe_top2_routing_softmax_normalised() {
        let config = minimal_phi_moe_config();
        let phi_cfg = PhiMoeConfig::from(&config);
        assert_eq!(phi_cfg.num_experts_per_tok, 2, "should use top-2");

        // After router softmax, all expert weights sum to 1.0.
        // The top-2 sum must be ≤ 1.0.
        let num_experts = phi_cfg.num_experts;
        let mut logits: Vec<f32> = (0..num_experts).map(|i| (i as f32) * 0.1).collect();
        softmax_inplace(&mut logits);

        let mut sorted = logits.clone();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top2_sum: f32 = sorted[..2].iter().sum();

        assert!(
            top2_sum <= 1.0 + 1e-5,
            "top-2 softmax weights sum must be ≤ 1, got {top2_sum}"
        );
    }

    #[test]
    fn phi_moe_forward_shape_and_finite() {
        let mut model = build_tiny_phi_moe();
        let vocab = model.vocab_size();
        let kv_dim = model.config.num_kv_heads * model.config.head_dim;
        let mut kv = SimpleKvCache::new(1, kv_dim, model.max_context_length());

        let result = model.forward(&[0u32], &mut kv);
        assert!(
            result.is_ok(),
            "phimoe forward must succeed: {:?}",
            result.err()
        );
        let logits = result.expect("logits");
        assert_eq!(logits.len(), vocab, "logits must have vocab_size elements");
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
        }
    }

    /// `partial_rotary_factor` stored in config should be 0.5 for Phi-3.5-MoE.
    #[test]
    fn phi_moe_partial_rope_first_25pct() {
        let config = minimal_phi_moe_config();
        let phi_cfg = PhiMoeConfig::from(&config);
        assert!(
            (phi_cfg.partial_rotary_factor - 0.5).abs() < 1e-6,
            "Phi-3.5-MoE partial_rotary_factor should be 0.5, got {}",
            phi_cfg.partial_rotary_factor
        );
    }

    /// Load PhiMoE from a minimal synthetic GGUF — checks end-to-end loading.
    #[test]
    fn phi_moe_load_from_synthetic_gguf() {
        use oxillama_gguf::test_utils::build_minimal_phi_moe_gguf;
        let bytes = build_minimal_phi_moe_gguf();
        let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("parse GGUF");
        let config =
            crate::config::ModelConfig::from_metadata(&gguf.file.metadata).expect("parse config");
        let mut model = load_phi_moe_from_gguf(&gguf, &config).expect("load phi_moe");
        let kv_dim = config.num_kv_heads * config.head_dim;
        let mut kv = SimpleKvCache::new(config.num_layers, kv_dim, config.max_context_length);
        let logits = model.forward(&[0u32], &mut kv).expect("forward");
        assert_eq!(logits.len(), config.vocab_size);
        for &v in &logits {
            assert!(v.is_finite(), "logit must be finite: {v}");
        }
    }
}

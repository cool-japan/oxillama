//! GPT-NeoX transformer forward pass implementation.
//!
//! GPT-NeoX (EleutherAI) shares the parallel residual structure with StableLM
//! but predates it. Both branches of the residual receive the same pre-norm
//! input, and their outputs are combined into a single residual update:
//!
//! ```text
//! y = x + Attention(LN_1(x)) + FFN(LN_2(x))
//! ```
//!
//! ## Key differences from LLaMA
//!
//! - **Parallel residual** (same as StableLM): attn and FFN share the same
//!   pre-norm input and are added together to the residual simultaneously.
//! - **Two separate LayerNorms** — `ln1` (pre-attention) and `ln2` (pre-FFN),
//!   each with a learned bias.
//! - **Partial RoPE** — same mechanism as StableLM: only the first
//!   `partial_rotary_factor × head_dim` dimensions of Q/K are rotated.
//! - **GELU activation** instead of SiLU for the FFN gate.
//!
//! ## Tensor names (GGUF)
//!
//! - `token_embd.weight` — Token embedding matrix
//! - `blk.{i}.ln1.weight` / `.bias` — Pre-attention LayerNorm
//! - `blk.{i}.ln2.weight` / `.bias` — Pre-FFN LayerNorm
//! - `blk.{i}.attn_q.weight` — Q projection
//! - `blk.{i}.attn_k.weight` — K projection
//! - `blk.{i}.attn_v.weight` — V projection
//! - `blk.{i}.attn_output.weight` — Attention output projection
//! - `blk.{i}.ffn_up.weight` — FFN up/gate projection
//! - `blk.{i}.ffn_down.weight` — FFN down projection
//! - `output_norm.weight` / `.bias` — Final LayerNorm
//! - `output.weight` — LM head

use crate::common::layer_norm::LayerNorm;
use crate::common::rope::RopeTable;
use crate::config::ModelConfig;
use crate::error::ArchResult;
use crate::traits::{ForwardPass, KvCacheAccess};

/// Apply partial RoPE to one head vector using separate rope table reference.
///
/// Free function variant so callers can simultaneously hold mutable borrows
/// on the head buffer fields alongside an immutable borrow on `rope`.
fn apply_partial_rope_free(
    rope: &RopeTable,
    head: &mut [f32],
    position: usize,
    rotary_dims: usize,
) {
    let head_dim = head.len();
    if rotary_dims == 0 || head_dim == 0 {
        return;
    }
    let rotate_half = rotary_dims / 2;
    let offset = position * rope.half_dim;

    for i in 0..rotate_half {
        let x0 = head[i];
        let x1 = head[i + rotate_half];
        let cos_val = rope.cos[offset + i];
        let sin_val = rope.sin[offset + i];
        head[i] = x0 * cos_val - x1 * sin_val;
        head[i + rotate_half] = x0 * sin_val + x1 * cos_val;
    }
    // head[rotary_dims..head_dim] is unchanged.
}

/// Apply partial RoPE using pre-extracted cos/sin vectors.
///
/// This avoids the split-borrow conflict that arises when `rope` is a field
/// of the same struct as `head`.  The caller must pre-extract the relevant
/// cos/sin slices before mutably borrowing `head`.
fn apply_rope_from_precomputed(
    head: &mut [f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    rotary_dims: usize,
) {
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

/// Partial rotary factor for GPT-NeoX (matches StableLM convention).
///
/// This is stored per-model and defaults to 0.25 (25% of head_dim rotated).
pub const DEFAULT_PARTIAL_ROTARY_FACTOR: f32 = 0.25;

/// A single GPT-NeoX transformer layer.
///
/// Uses two separate LayerNorms with bias (`ln1` for attention, `ln2` for FFN).
/// The output of both branches is added to the same residual simultaneously.
pub struct GptNeoxLayer {
    /// Pre-attention LayerNorm (with bias).
    pub ln1: LayerNorm,
    /// Pre-FFN LayerNorm (with bias).
    pub ln2: LayerNorm,
    /// Q projection `[num_heads * head_dim, hidden_size]` (f32).
    pub attn_q: Vec<f32>,
    /// K projection `[num_kv_heads * head_dim, hidden_size]` (f32).
    pub attn_k: Vec<f32>,
    /// V projection `[num_kv_heads * head_dim, hidden_size]` (f32).
    pub attn_v: Vec<f32>,
    /// Attention output projection `[hidden_size, num_heads * head_dim]` (f32).
    pub attn_output: Vec<f32>,
    /// FFN up/gate combined projection `[intermediate_size, hidden_size]` (f32).
    ///
    /// In GPT-NeoX the FFN is a dense 2-layer MLP: `GELU(W_up @ x)` followed
    /// by `W_down @` that result. There is no separate gate projection.
    pub ffn_up: Vec<f32>,
    /// FFN down projection `[hidden_size, intermediate_size]` (f32).
    pub ffn_down: Vec<f32>,
}

/// Complete GPT-NeoX model.
pub struct GptNeoxModel {
    /// Base model configuration.
    pub config: ModelConfig,
    /// Fraction of head_dim to apply RoPE to (default 0.25).
    pub partial_rotary_factor: f32,
    /// Token embeddings `[vocab_size, hidden_size]` (f32).
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<GptNeoxLayer>,
    /// Final LayerNorm (with bias).
    pub output_norm: LayerNorm,
    /// LM head weights `[vocab_size, hidden_size]` (f32).
    pub output_weights: Vec<f32>,
    /// Precomputed RoPE frequency table.
    pub rope: RopeTable,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_ln1: Vec<f32>,
    buf_ln2: Vec<f32>,
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

impl GptNeoxModel {
    /// Create a new `GptNeoxModel` from pre-loaded weights.
    pub fn new(
        config: ModelConfig,
        partial_rotary_factor: f32,
        token_embd: Vec<f32>,
        layers: Vec<GptNeoxLayer>,
        output_norm: LayerNorm,
        output_weights: Vec<f32>,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;

        let rope = RopeTable::new(
            head_dim,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
        );

        Self {
            config,
            partial_rotary_factor,
            token_embd,
            layers,
            output_norm,
            output_weights,
            rope,
            buf_hidden: vec![0.0f32; hidden_size],
            buf_ln1: vec![0.0f32; hidden_size],
            buf_ln2: vec![0.0f32; hidden_size],
            buf_q: vec![0.0f32; num_heads * head_dim],
            buf_k: vec![0.0f32; num_kv_heads * head_dim],
            buf_v: vec![0.0f32; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0f32; num_heads * head_dim],
            buf_attn_proj: vec![0.0f32; hidden_size],
            buf_ffn_up: vec![0.0f32; intermediate_size],
            buf_ffn_out: vec![0.0f32; hidden_size],
            buf_logits: vec![0.0f32; vocab_size],
            buf_attn_scores: vec![0.0f32; max_ctx],
        }
    }

    /// Compute the number of rotated dimensions given the current `partial_rotary_factor`.
    ///
    /// The result is always even (required for RoPE pair rotation) and
    /// clamped to `[0, head_dim]`.
    pub fn rotary_dims(&self, head_dim: usize) -> usize {
        let raw = (self.partial_rotary_factor * head_dim as f32) as usize;
        (raw & !1).min(head_dim)
    }

    fn embed_token(&mut self, token: u32) {
        let h = self.config.hidden_size;
        let offset = token as usize * h;
        self.buf_hidden
            .copy_from_slice(&self.token_embd[offset..offset + h]);
    }

    /// Apply partial RoPE to a single head vector in-place.
    ///
    /// Only dims `[0, rotary_dims)` are rotated; `[rotary_dims, head_dim)` are
    /// left unchanged. Verified by `gptneox_partial_rope_correctness`.
    ///
    /// Delegates to `apply_partial_rope_free` to make the function safe
    /// to call with a locally-owned `head` slice.
    pub fn apply_partial_rope(&self, head: &mut [f32], position: usize, rotary_dims: usize) {
        apply_partial_rope_free(&self.rope, head, position, rotary_dims);
    }

    /// Dense GEMV: `out[i] = dot(W[i, :], x)`.
    fn gemv(w: &[f32], x: &[f32], out: &mut [f32], in_dim: usize) {
        for (i, o) in out.iter_mut().enumerate() {
            let row = &w[i * in_dim..(i + 1) * in_dim];
            *o = row.iter().zip(x.iter()).map(|(w, x)| w * x).sum();
        }
    }

    /// GELU activation (approximate; matches GPT-NeoX original).
    ///
    /// Uses the tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
    fn gelu(x: f32) -> f32 {
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
    }

    /// Run causal scaled dot-product attention.
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
        let heads_per_kv = num_heads.checked_div(num_kv_heads).unwrap_or(1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = position + 1;
        let rotary_dims = self.rotary_dims(head_dim);
        let hidden_size = self.config.hidden_size;

        let ln1_out = self.buf_ln1.clone();

        // Q, K, V projections
        Self::gemv(
            &self.layers[layer_idx].attn_q,
            &ln1_out,
            &mut self.buf_q,
            hidden_size,
        );
        Self::gemv(
            &self.layers[layer_idx].attn_k,
            &ln1_out,
            &mut self.buf_k[..kv_dim],
            hidden_size,
        );
        Self::gemv(
            &self.layers[layer_idx].attn_v,
            &ln1_out,
            &mut self.buf_v[..kv_dim],
            hidden_size,
        );

        // Partial RoPE — snapshot cos/sin to avoid split-borrow conflict.
        let rotate_half = rotary_dims / 2;
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
            apply_rope_from_precomputed(q_head, &rope_cos, &rope_sin, rotary_dims);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            apply_rope_from_precomputed(k_head, &rope_cos, &rope_sin, rotary_dims);
        }

        // Cache K, V
        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = kv_cache.get_keys(layer_idx)?.to_vec();
        let cached_values = kv_cache.get_values(layer_idx)?.to_vec();

        self.buf_attn_out.fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in 0..seq_len {
                let k_off = pos * kv_dim + kv_head * head_dim;
                let k_slice = &cached_keys[k_off..k_off + head_dim];
                let score: f32 =
                    q_head.iter().zip(k_slice).map(|(q, k)| q * k).sum::<f32>() * scale;
                self.buf_attn_scores[pos] = score;
            }

            softmax_inplace(&mut self.buf_attn_scores[..seq_len]);

            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for pos in 0..seq_len {
                let v_off = pos * kv_dim + kv_head * head_dim;
                let v_slice = &cached_values[v_off..v_off + head_dim];
                let w = self.buf_attn_scores[pos];
                for d in 0..head_dim {
                    out_head[d] += w * v_slice[d];
                }
            }
        }

        // Output projection
        let attn_out_copy = self.buf_attn_out.clone();
        let proj_in_dim = num_heads * head_dim;
        Self::gemv(
            &self.layers[layer_idx].attn_output,
            &attn_out_copy,
            &mut self.buf_attn_proj,
            proj_in_dim.min(hidden_size),
        );

        Ok(())
    }

    /// Compute GELU FFN: `out = W_down @ GELU(W_up @ ln2_out)`.
    fn feed_forward(&mut self, layer_idx: usize) {
        let ffn_in = self.buf_ln2.clone();
        let h = self.config.hidden_size;
        let n = self.config.intermediate_size;

        // up projection + GELU activation
        Self::gemv(
            &self.layers[layer_idx].ffn_up,
            &ffn_in,
            &mut self.buf_ffn_up,
            h,
        );
        for v in self.buf_ffn_up.iter_mut() {
            *v = Self::gelu(*v);
        }

        // down projection
        let up_copy = self.buf_ffn_up.clone();
        Self::gemv(
            &self.layers[layer_idx].ffn_down,
            &up_copy,
            &mut self.buf_ffn_out,
            n,
        );
    }

    /// Run one layer with the parallel residual pattern.
    ///
    /// `y = x + Attn(LN1(x)) + FFN(LN2(x))`
    fn layer_forward(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let hidden = self.buf_hidden.clone();

        // Apply both norms to the same pre-residual hidden state.
        self.layers[layer_idx]
            .ln1
            .forward_to(&hidden, &mut self.buf_ln1);
        self.layers[layer_idx]
            .ln2
            .forward_to(&hidden, &mut self.buf_ln2);

        // Attention path → buf_attn_proj
        self.attention(layer_idx, position, kv_cache)?;

        // FFN path → buf_ffn_out
        self.feed_forward(layer_idx);

        // Parallel residual update: y = x + attn + ffn
        let attn_proj = self.buf_attn_proj.clone();
        let ffn_out = self.buf_ffn_out.clone();
        for ((h, &a), &f) in self
            .buf_hidden
            .iter_mut()
            .zip(attn_proj.iter())
            .zip(ffn_out.iter())
        {
            *h += a + f;
        }

        Ok(())
    }
}

impl ForwardPass for GptNeoxModel {
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

        // LM head projection
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        self.buf_logits.fill(0.0);
        for v in 0..vocab_size {
            let row = &self.output_weights[v * hidden_size..(v + 1) * hidden_size];
            let sum: f32 = row.iter().zip(normed.iter()).map(|(w, &x)| w * x).sum();
            self.buf_logits[v] = sum;
        }

        Ok(self.buf_logits.clone())
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

// ─── Test helpers ─────────────────────────────────────────────────────────────

/// Build a minimal `GptNeoxLayer` with near-zero weights for testing.
#[cfg(test)]
pub fn make_test_layer(hidden_size: usize, intermediate_size: usize) -> GptNeoxLayer {
    let ln1 = LayerNorm::new(
        vec![1.0f32; hidden_size],
        Some(vec![0.0f32; hidden_size]),
        1e-5,
    );
    let ln2 = LayerNorm::new(
        vec![1.0f32; hidden_size],
        Some(vec![0.0f32; hidden_size]),
        1e-5,
    );

    let qkv_size = hidden_size * hidden_size;
    let ffn_up_size = intermediate_size * hidden_size;
    let ffn_down_size = hidden_size * intermediate_size;

    GptNeoxLayer {
        ln1,
        ln2,
        attn_q: vec![0.01f32; qkv_size],
        attn_k: vec![0.01f32; qkv_size],
        attn_v: vec![0.01f32; qkv_size],
        attn_output: vec![0.01f32; qkv_size],
        ffn_up: vec![0.01f32; ffn_up_size],
        ffn_down: vec![0.01f32; ffn_down_size],
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::error::{ArchError, ArchResult};
    use crate::gpt_neox::GptNeoxArchitecture;
    use crate::registry::ArchitectureRegistry;
    use crate::traits::ModelArchitecture;

    fn minimal_config(hidden_size: usize) -> ModelConfig {
        let num_heads = 2usize;
        let head_dim = hidden_size / num_heads;
        ModelConfig {
            architecture: "gptneox".to_string(),
            hidden_size,
            intermediate_size: 16,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads: num_heads,
            head_dim,
            vocab_size: 4,
            max_context_length: 8,
            ..ModelConfig::default()
        }
    }

    fn make_model(mc: &ModelConfig, partial_rotary_factor: f32) -> GptNeoxModel {
        let h = mc.hidden_size;
        let v = mc.vocab_size;
        let token_embd = vec![0.01f32; v * h];
        let layer = make_test_layer(h, mc.intermediate_size);
        let output_norm = LayerNorm::new(vec![1.0f32; h], Some(vec![0.0f32; h]), 1e-5);
        let output_weights = vec![0.01f32; v * h];

        GptNeoxModel::new(
            mc.clone(),
            partial_rotary_factor,
            token_embd,
            vec![layer],
            output_norm,
            output_weights,
        )
    }

    /// Minimal KV cache for forward-pass tests.
    struct SimpleKvCache {
        kv_dim: usize,
        max_seq: usize,
        n_layers: usize,
        position: usize,
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
    }

    impl SimpleKvCache {
        fn new(n_layers: usize, kv_dim: usize, max_seq: usize) -> Self {
            Self {
                kv_dim,
                max_seq,
                n_layers,
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

    // ── Registry lookup ───────────────────────────────────────────────────────

    #[test]
    fn gptneox_registry_lookup() {
        let registry = ArchitectureRegistry::with_builtins();
        let arch = registry.get("gptneox");
        assert!(
            arch.is_ok(),
            "registry.get('gptneox') must succeed; got: {:?}",
            arch.err()
        );
        assert_eq!(arch.expect("gptneox arch").arch_id(), "gptneox");
    }

    // ── Tensor names ──────────────────────────────────────────────────────────

    #[test]
    fn gptneox_tensor_names_complete() {
        let arch = GptNeoxArchitecture::new();
        let names = arch.tensor_names();
        assert!(
            !names.is_empty(),
            "tensor_names() must return at least one pattern"
        );
        for tp in &names {
            assert!(!tp.pattern.is_empty(), "pattern must not be empty");
            assert!(!tp.description.is_empty(), "description must not be empty");
        }
        let required_patterns = [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.{i}.ln1.weight",
            "blk.{i}.ln1.bias",
            "blk.{i}.ln2.weight",
            "blk.{i}.ln2.bias",
        ];
        let pattern_strs: Vec<&str> = names.iter().map(|n| n.pattern.as_str()).collect();
        for req in required_patterns {
            assert!(
                pattern_strs.contains(&req),
                "tensor_names should contain '{req}'"
            );
        }
    }

    // ── Parallel residual shape ───────────────────────────────────────────────

    /// Output of GPT-NeoX forward pass must equal vocab_size.
    #[test]
    fn gptneox_parallel_residual_shape() {
        let mc = minimal_config(8);
        let vocab_size = mc.vocab_size;
        let num_kv_heads = mc.num_kv_heads;
        let head_dim = mc.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut model = make_model(&mc, 0.25);
        let mut kv_cache = SimpleKvCache::new(1, kv_dim, mc.max_context_length);

        let result = model.forward(&[0u32], &mut kv_cache);
        assert!(
            result.is_ok(),
            "GPT-NeoX forward must succeed: {:?}",
            result.err()
        );
        let logits = result.expect("logits");
        assert_eq!(
            logits.len(),
            vocab_size,
            "logits must have vocab_size={vocab_size} elements, got {}",
            logits.len()
        );
    }

    // ── Partial RoPE correctness ──────────────────────────────────────────────

    /// Verify that partial RoPE rotates only the first `rotary_dims` dimensions
    /// and leaves the rest unchanged.
    #[test]
    fn gptneox_partial_rope_correctness() {
        let mc = minimal_config(8);
        let model = make_model(&mc, 0.25);

        // Use head_dim = 64 for a meaningful test (25% → 16 rotary dims).
        let big_head_dim = 64usize;
        let rotary_dims = model.rotary_dims(big_head_dim); // 25% of 64 = 16

        let mut head = vec![0.0f32; big_head_dim];
        for (i, v) in head.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        let original = head.clone();

        // Apply partial RoPE at position 1 (non-trivial values).
        model.apply_partial_rope(&mut head, 1, rotary_dims);

        // The rotated range [0, rotary_dims) should have changed.
        let rotated_changed = (0..rotary_dims).any(|i| (head[i] - original[i]).abs() > 1e-6);
        assert!(
            rotated_changed,
            "rotated dims should change at position != 0, got head[..{rotary_dims}]={:?}",
            &head[..rotary_dims]
        );

        // The unrotated range [rotary_dims, big_head_dim) must be identical.
        for i in rotary_dims..big_head_dim {
            assert!(
                (head[i] - original[i]).abs() < 1e-9,
                "dim {i} (outside rotary range) must be unchanged; got {} vs {}",
                head[i],
                original[i]
            );
        }

        assert_eq!(rotary_dims, 16, "25% of 64 must be 16 rotary dims");
    }

    // ── Parallel residual is deterministic ────────────────────────────────────

    #[test]
    fn gptneox_parallel_residual_deterministic() {
        let mc = minimal_config(8);
        let num_kv_heads = mc.num_kv_heads;
        let head_dim = mc.head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut m1 = make_model(&mc, 0.25);
        let mut m2 = make_model(&mc, 0.25);
        let mut kv1 = SimpleKvCache::new(1, kv_dim, mc.max_context_length);
        let mut kv2 = SimpleKvCache::new(1, kv_dim, mc.max_context_length);

        let out1 = m1.forward(&[2u32], &mut kv1).expect("forward1");
        let out2 = m2.forward(&[2u32], &mut kv2).expect("forward2");

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "gptneox forward must be deterministic: {a} != {b}"
            );
        }
    }
}

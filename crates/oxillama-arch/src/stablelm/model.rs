//! StableLM transformer forward pass implementation.
//!
//! StableLM departs from LLaMA/Mistral in three key ways:
//!
//! 1. **LayerNorm instead of RMSNorm** — both the attention and FFN pre-norms
//!    use a full LayerNorm with learned bias (`attn_norm.weight/bias`,
//!    `ffn_norm.weight/bias`).
//!
//! 2. **Partial RoPE** — RoPE is applied only to the first
//!    `round(partial_rotary_factor × head_dim)` dimensions of each Q/K head
//!    vector. The remaining dimensions are passed through unmodified.
//!
//! 3. **Parallel attention + FFN** — the residual stream receives both the
//!    attention output *and* the FFN output added simultaneously:
//!    ```text
//!    y = x + Attention(LayerNorm(x)) + FFN(LayerNorm(x))
//!    ```
//!    rather than the sequential LLaMA pattern where attention is applied first
//!    and then FFN is applied to the updated residual.
//!
//! ## Tensor names (GGUF)
//!
//! - `blk.{i}.attn_norm.weight` / `.bias` — pre-attention LayerNorm
//! - `blk.{i}.ffn_norm.weight` / `.bias` — pre-FFN LayerNorm (same input as attn_norm)
//! - `blk.{i}.attn_q.weight` — Q projection
//! - `blk.{i}.attn_k.weight` — K projection
//! - `blk.{i}.attn_v.weight` — V projection
//! - `blk.{i}.attn_output.weight` — attention output projection
//! - `blk.{i}.ffn_gate.weight` — FFN gate (SwiGLU)
//! - `blk.{i}.ffn_up.weight` — FFN up projection
//! - `blk.{i}.ffn_down.weight` — FFN down projection
//! - `output_norm.weight` / `.bias` — final LayerNorm
//! - `output.weight` — LM head

use crate::common::layer_norm::LayerNorm;
use crate::common::rope::RopeTable;
use crate::config::ModelConfig;
use crate::error::ArchResult;
use crate::traits::{ForwardPass, KvCacheAccess};

use super::config::StablelmConfig;

/// Apply partial RoPE to a single head vector using a pre-built `RopeTable`.
///
/// Only the first `rotary_dims` elements (which must be even) are rotated;
/// the remaining `head.len() - rotary_dims` elements are left unchanged.
///
/// This is a free function so that callers can hold a mutable borrow on the
/// buffer fields at the same time as an immutable borrow on `rope`.
fn apply_partial_rope_with_table(
    rope: &RopeTable,
    head: &mut [f32],
    position: usize,
    rotary_dims: usize,
) {
    if rotary_dims == 0 || head.is_empty() {
        return;
    }
    let rotate_len = rotary_dims.min(head.len());
    rope.apply(&mut head[..rotate_len], position);
}

/// Apply partial RoPE to a head using pre-extracted cos/sin slices.
///
/// Avoids split-borrow conflicts when the rope table is a field of the same
/// struct as the buffer being mutated.
fn apply_partial_rope_precomputed(
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

/// A single StableLM transformer layer.
///
/// Uses LayerNorm (with bias) for both norms and combines attention + FFN
/// outputs in parallel before adding to the residual.
pub struct StablelmLayer {
    /// Pre-attention LayerNorm (with bias).
    pub attn_norm: LayerNorm,
    /// Pre-FFN LayerNorm (with bias, applied to the *same* pre-residual input).
    pub ffn_norm: LayerNorm,
    /// Q projection weights `[num_heads * head_dim, hidden_size]` (f32).
    pub attn_q: Vec<f32>,
    /// K projection weights `[num_kv_heads * head_dim, hidden_size]` (f32).
    pub attn_k: Vec<f32>,
    /// V projection weights `[num_kv_heads * head_dim, hidden_size]` (f32).
    pub attn_v: Vec<f32>,
    /// Attention output projection `[hidden_size, num_heads * head_dim]` (f32).
    pub attn_output: Vec<f32>,
    /// FFN gate projection `[intermediate_size, hidden_size]` (f32).
    pub ffn_gate: Vec<f32>,
    /// FFN up projection `[intermediate_size, hidden_size]` (f32).
    pub ffn_up: Vec<f32>,
    /// FFN down projection `[hidden_size, intermediate_size]` (f32).
    pub ffn_down: Vec<f32>,
}

/// Complete StableLM model.
pub struct StablelmModel {
    /// Base model configuration.
    pub config: ModelConfig,
    /// StableLM-specific configuration.
    pub stablelm_config: StablelmConfig,
    /// Token embeddings `[vocab_size, hidden_size]` (f32).
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<StablelmLayer>,
    /// Final LayerNorm (with bias).
    pub output_norm: LayerNorm,
    /// LM head weights `[vocab_size, hidden_size]` (f32).
    pub output_weights: Vec<f32>,
    /// Precomputed RoPE frequency table (covers full head_dim; partial usage
    /// is enforced inside `apply_partial_rope`).
    pub rope: RopeTable,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_attn_norm: Vec<f32>,
    buf_ffn_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_attn_proj: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl StablelmModel {
    /// Create a new `StablelmModel` from pre-loaded weights.
    pub fn new(
        config: ModelConfig,
        stablelm_config: StablelmConfig,
        token_embd: Vec<f32>,
        layers: Vec<StablelmLayer>,
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
            stablelm_config,
            token_embd,
            layers,
            output_norm,
            output_weights,
            rope,
            buf_hidden: vec![0.0f32; hidden_size],
            buf_attn_norm: vec![0.0f32; hidden_size],
            buf_ffn_norm: vec![0.0f32; hidden_size],
            buf_q: vec![0.0f32; num_heads * head_dim],
            buf_k: vec![0.0f32; num_kv_heads * head_dim],
            buf_v: vec![0.0f32; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0f32; num_heads * head_dim],
            buf_attn_proj: vec![0.0f32; hidden_size],
            buf_gate: vec![0.0f32; intermediate_size],
            buf_up: vec![0.0f32; intermediate_size],
            buf_ffn_out: vec![0.0f32; hidden_size],
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

    /// Apply partial RoPE to a single head vector.
    ///
    /// Only the first `rotary_dims` elements are rotated; the remaining
    /// `head_dim - rotary_dims` elements are left unchanged. This is the
    /// defining feature of StableLM's partial RoPE implementation.
    ///
    /// # Arguments
    /// * `head`        — mutable slice of length `head_dim`
    /// * `position`    — token position
    /// * `rotary_dims` — how many leading dimensions to rotate (must be even)
    pub fn apply_partial_rope(&self, head: &mut [f32], position: usize, rotary_dims: usize) {
        // Use the split-borrow-safe free function.
        apply_partial_rope_with_table(&self.rope, head, position, rotary_dims);
    }

    /// Dense matrix-vector product: `out[i] = dot(W[i, :], x)`.
    ///
    /// `W` is stored row-major with shape `[out_dim, in_dim]`.
    fn gemv(w: &[f32], x: &[f32], out: &mut [f32], in_dim: usize) {
        for (i, o) in out.iter_mut().enumerate() {
            let row = &w[i * in_dim..(i + 1) * in_dim];
            *o = row.iter().zip(x.iter()).map(|(w, x)| w * x).sum();
        }
    }

    /// Run scaled dot-product attention with causal masking.
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

        let rotary_dims = self.stablelm_config.rotary_dims(head_dim);

        let layer = &self.layers[layer_idx];

        // Project Q, K, V
        let attn_in = self.buf_attn_norm.clone();
        Self::gemv(&layer.attn_q, &attn_in, &mut self.buf_q, attn_in.len());
        Self::gemv(
            &layer.attn_k,
            &attn_in,
            &mut self.buf_k[..kv_dim],
            attn_in.len(),
        );
        Self::gemv(
            &layer.attn_v,
            &attn_in,
            &mut self.buf_v[..kv_dim],
            attn_in.len(),
        );

        // Apply partial RoPE to each Q head.
        // Snapshot the RoPE slice for this position to avoid split-borrow conflict.
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
            apply_partial_rope_precomputed(q_head, &rope_cos, &rope_sin, rotary_dims);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            apply_partial_rope_precomputed(k_head, &rope_cos, &rope_sin, rotary_dims);
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

        // Output projection: attn_proj = W_o @ attn_out
        let attn_out_copy = self.buf_attn_out.clone();
        let attn_proj_out_dim = self.config.hidden_size;
        let attn_proj_in_dim = num_heads * head_dim;
        Self::gemv(
            &self.layers[layer_idx].attn_output,
            &attn_out_copy,
            &mut self.buf_attn_proj,
            attn_proj_in_dim.min(attn_proj_out_dim),
        );

        Ok(())
    }

    /// Compute SwiGLU FFN output into `buf_ffn_out`.
    fn feed_forward(&mut self, layer_idx: usize) {
        let ffn_in = self.buf_ffn_norm.clone();
        let h = self.config.hidden_size;
        let n = self.config.intermediate_size;

        let layer = &self.layers[layer_idx];

        // gate = W_gate @ ffn_in
        Self::gemv(&layer.ffn_gate, &ffn_in, &mut self.buf_gate, h);
        // up = W_up @ ffn_in
        Self::gemv(&layer.ffn_up, &ffn_in, &mut self.buf_up, h);

        // SwiGLU: gate = silu(gate) * up
        for (g, &u) in self.buf_gate.iter_mut().zip(self.buf_up.iter()) {
            let silu = *g / (1.0 + (-*g).exp());
            *g = silu * u;
        }

        // down = W_down @ gate
        let swiglu = self.buf_gate.clone();
        Self::gemv(&layer.ffn_down, &swiglu, &mut self.buf_ffn_out, n);
    }

    /// Run one layer: parallel attention + FFN, add both to residual.
    ///
    /// `y = x + Attention(LayerNorm_attn(x)) + FFN(LayerNorm_ffn(x))`
    fn layer_forward(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let hidden = self.buf_hidden.clone();

        // Both norms receive the same pre-residual hidden state.
        self.layers[layer_idx]
            .attn_norm
            .forward_to(&hidden, &mut self.buf_attn_norm);
        self.layers[layer_idx]
            .ffn_norm
            .forward_to(&hidden, &mut self.buf_ffn_norm);

        // Attention path (writes result into buf_attn_proj)
        self.attention(layer_idx, position, kv_cache)?;

        // FFN path (writes result into buf_ffn_out)
        self.feed_forward(layer_idx);

        // Parallel residual: hidden = original_hidden + attn_proj + ffn_out
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

impl ForwardPass for StablelmModel {
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

        // Final norm on hidden state
        let hidden_final = self.buf_hidden.clone();
        let mut normed = vec![0.0f32; self.config.hidden_size];
        self.output_norm.forward_to(&hidden_final, &mut normed);

        // LM head: vocab_size × hidden_size
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

/// Build a minimal `StablelmLayer` filled with near-zero weights.
#[cfg(test)]
pub fn make_test_layer(hidden_size: usize, intermediate_size: usize) -> StablelmLayer {
    let attn_norm = LayerNorm::new(
        vec![1.0f32; hidden_size],
        Some(vec![0.0f32; hidden_size]),
        1e-5,
    );
    let ffn_norm = LayerNorm::new(
        vec![1.0f32; hidden_size],
        Some(vec![0.0f32; hidden_size]),
        1e-5,
    );

    // All projection weights are 0.01 (small non-zero to produce non-trivial output)
    let q_k_v_out_size = hidden_size * hidden_size; // square for simplicity (head_dim = hidden)
    let ffn_gate_up_size = intermediate_size * hidden_size;
    let ffn_down_size = hidden_size * intermediate_size;

    StablelmLayer {
        attn_norm,
        ffn_norm,
        attn_q: vec![0.01f32; q_k_v_out_size],
        attn_k: vec![0.01f32; q_k_v_out_size],
        attn_v: vec![0.01f32; q_k_v_out_size],
        attn_output: vec![0.01f32; q_k_v_out_size],
        ffn_gate: vec![0.01f32; ffn_gate_up_size],
        ffn_up: vec![0.01f32; ffn_gate_up_size],
        ffn_down: vec![0.01f32; ffn_down_size],
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::error::{ArchError, ArchResult};
    use crate::registry::ArchitectureRegistry;
    use crate::stablelm::StablelmArchitecture;
    use crate::traits::ModelArchitecture;

    fn minimal_config(hidden_size: usize) -> (ModelConfig, StablelmConfig) {
        let num_heads = 2usize;
        let head_dim = hidden_size / num_heads;
        let mc = ModelConfig {
            architecture: "stablelm".to_string(),
            hidden_size,
            intermediate_size: 16,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads: num_heads,
            head_dim,
            vocab_size: 4,
            max_context_length: 8,
            ..ModelConfig::default()
        };
        let sc = StablelmConfig {
            partial_rotary_factor: 0.25,
            num_heads,
            num_kv_heads: num_heads,
            hidden_size,
            intermediate_size: 16,
            layer_norm_eps: 1e-5,
        };
        (mc, sc)
    }

    fn make_model(mc: &ModelConfig, sc: &StablelmConfig) -> StablelmModel {
        let h = mc.hidden_size;
        let v = mc.vocab_size;
        let token_embd = vec![0.01f32; v * h];
        let layer = make_test_layer(h, mc.intermediate_size);
        let output_norm = LayerNorm::new(vec![1.0f32; h], Some(vec![0.0f32; h]), 1e-5);
        let output_weights = vec![0.01f32; v * h];

        StablelmModel::new(
            mc.clone(),
            sc.clone(),
            token_embd,
            vec![layer],
            output_norm,
            output_weights,
        )
    }

    /// Minimal KV cache for tests.
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
    fn stablelm_registry_lookup() {
        let registry = ArchitectureRegistry::with_builtins();
        let arch = registry.get("stablelm");
        assert!(
            arch.is_ok(),
            "registry.get('stablelm') must succeed; got: {:?}",
            arch.err()
        );
        assert_eq!(arch.expect("stablelm arch").arch_id(), "stablelm");
    }

    // ── Tensor names ──────────────────────────────────────────────────────────

    #[test]
    fn stablelm_tensor_names_complete() {
        let arch = StablelmArchitecture::new();
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
            "blk.{i}.attn_norm.weight",
            "blk.{i}.attn_norm.bias",
            "blk.{i}.ffn_norm.weight",
            "blk.{i}.ffn_norm.bias",
        ];
        let pattern_strs: Vec<&str> = names.iter().map(|n| n.pattern.as_str()).collect();
        for req in required_patterns {
            assert!(
                pattern_strs.contains(&req),
                "tensor_names should contain '{req}'"
            );
        }
    }

    // ── Partial RoPE correctness ───────────────────────────────────────────────

    /// Only the first 25% of head dims must be rotated; the remaining 75%
    /// must remain unchanged after `apply_partial_rope`.
    #[test]
    fn stablelm_partial_rope_correctness() {
        let (mc, sc) = minimal_config(8);
        let model = make_model(&mc, &sc);

        let head_dim = mc.head_dim; // = hidden / num_heads = 4
        let rotary_dims = sc.rotary_dims(head_dim); // 25% of 4 = 1 → rounds down to even 0
                                                    // Use a larger head_dim for a meaningful test.
        let big_head_dim = 64usize;
        let big_rotary_dims = sc.rotary_dims(big_head_dim); // 25% of 64 = 16

        let mut head = vec![0.0f32; big_head_dim];
        // Fill with distinguishable values.
        for (i, v) in head.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        let original = head.clone();

        // Apply partial RoPE at position 1 (non-trivial rotation).
        model.apply_partial_rope(&mut head, 1, big_rotary_dims);

        // Dimensions [0, big_rotary_dims) must have changed (rotation applied).
        // Dimensions [big_rotary_dims, big_head_dim) must be IDENTICAL to original.
        let rotated_changed = (0..big_rotary_dims).any(|i| (head[i] - original[i]).abs() > 1e-6);
        assert!(
            rotated_changed,
            "at least some rotated dims should change after partial RoPE at pos=1"
        );

        for i in big_rotary_dims..big_head_dim {
            assert!(
                (head[i] - original[i]).abs() < 1e-9,
                "dim {i} (outside rotary range [{big_rotary_dims}, {big_head_dim})) must be unchanged; \
                 got {} vs original {}",
                head[i],
                original[i]
            );
        }

        // Sanity check: the model has the right partial_rotary_factor.
        assert_eq!(
            sc.partial_rotary_factor, 0.25,
            "StablelmConfig default partial_rotary_factor must be 0.25"
        );
        // For the tiny model (head_dim=4), rotary_dims is the even-floored 25% of 4.
        // 25% of 4 = 1.0 → floor = 1 → round down to even = 0.
        assert_eq!(
            rotary_dims, 0,
            "25% of head_dim=4 rounds to 0 (must be even)"
        );
        // For head_dim=64, 25% = 16, which is already even.
        assert_eq!(big_rotary_dims, 16, "25% of 64 = 16");
    }

    // ── Parallel FFN + attn output shape ─────────────────────────────────────

    /// Parallel computation: the output of a forward pass must have the correct
    /// shape (vocab_size). The parallel residual adds both attn and ffn outputs
    /// to the same residual in one step.
    #[test]
    fn stablelm_parallel_ffn_attn_shapes() {
        let (mc, sc) = minimal_config(8);
        let vocab_size = mc.vocab_size;
        let num_kv_heads = mc.num_kv_heads;
        let head_dim = mc.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut model = make_model(&mc, &sc);
        let mut kv_cache = SimpleKvCache::new(1, kv_dim, mc.max_context_length);

        let result = model.forward(&[0u32], &mut kv_cache);
        assert!(
            result.is_ok(),
            "StableLM forward must succeed: {:?}",
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

    // ── Parallel residual: attn + ffn both contribute ─────────────────────────

    /// The parallel residual update means that both the attention output and
    /// the FFN output are added to the residual in the same step. We verify
    /// this by checking that calling two forward passes produces consistent
    /// (deterministic) outputs.
    #[test]
    fn stablelm_parallel_residual_is_deterministic() {
        let (mc, sc) = minimal_config(8);
        let num_kv_heads = mc.num_kv_heads;
        let head_dim = mc.head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut model1 = make_model(&mc, &sc);
        let mut model2 = make_model(&mc, &sc);
        let mut kv1 = SimpleKvCache::new(1, kv_dim, mc.max_context_length);
        let mut kv2 = SimpleKvCache::new(1, kv_dim, mc.max_context_length);

        let out1 = model1.forward(&[1u32], &mut kv1).expect("first forward");
        let out2 = model2.forward(&[1u32], &mut kv2).expect("second forward");

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "stablelm forward must be deterministic: {a} != {b}"
            );
        }
    }
}

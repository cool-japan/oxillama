//! Falcon transformer forward pass.
//!
//! Implements both Falcon-1 (parallel attention+FFN, ALiBi) and
//! Falcon-2 (sequential attention then FFN, GQA, RoPE) in a single struct
//! driven by the [`FalconConfig`] flags.
//!
//! ## Parallel-attention mode (Falcon-1)
//!
//! ```text
//!   x ──[LayerNorm]──┬──[QKV]──[Attn]──[O proj]──┐
//!                    │                              add ──[+residual]── x'
//!                    └──[FFN up]──[act]──[FFN down]─┘
//! ```
//!
//! ## Sequential mode (Falcon-2 / standard transformer)
//!
//! ```text
//!   x ──[attn_norm]──[QKV]──[Attn]──[O proj]──[+residual]──[ffn_norm]──[FFN]──[+residual]── x'
//! ```

use crate::common::gelu::gelu_inplace;
use crate::common::layer_norm::LayerNorm;
use crate::common::rope::RopeTable;
use crate::error::{ArchError, ArchResult};
use crate::falcon::config::FalconConfig;
use crate::llama::softmax_inplace;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantKernel, QuantTensor};

// ── Layer helpers ────────────────────────────────────────────────────────────

/// A single Falcon transformer layer.
///
/// Fields are `pub` to allow construction from a loader (e.g.
/// `load_falcon_from_gguf`).
pub struct FalconLayer {
    /// Pre-attention LayerNorm (used for both branches in parallel mode).
    pub attn_norm: LayerNorm,
    /// Optional separate pre-FFN LayerNorm (sequential mode / Falcon-2).
    pub ffn_norm: Option<LayerNorm>,
    /// Fused QKV weight `[(n_heads + 2*n_kv_heads) * head_dim, hidden_size]`.
    pub attn_qkv_weight: QuantTensor,
    /// Optional fused QKV bias.
    pub attn_qkv_bias: Option<Vec<f32>>,
    /// Attention output projection `[hidden_size, n_heads*head_dim]`.
    pub attn_out_weight: QuantTensor,
    /// Optional attention output bias.
    pub attn_out_bias: Option<Vec<f32>>,
    /// FFN up projection `[intermediate_size, hidden_size]`.
    pub ffn_up_weight: QuantTensor,
    /// Optional FFN up bias.
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection `[hidden_size, intermediate_size]`.
    pub ffn_down_weight: QuantTensor,
    /// Optional FFN down bias.
    pub ffn_down_bias: Option<Vec<f32>>,
}

// ── Full model ───────────────────────────────────────────────────────────────

/// Loaded Falcon model capable of running forward passes.
pub struct FalconForward {
    /// Falcon-specific hyperparameters.
    pub cfg: FalconConfig,
    /// Dequantised token embedding table `[vocab_size * hidden_size]`.
    pub token_embd: Vec<f32>,
    /// All transformer layers.
    pub layers: Vec<FalconLayer>,
    /// Final LayerNorm.
    pub output_norm: LayerNorm,
    /// LM head weight `[vocab_size * hidden_size]` (dequantised f32).
    pub output_weight: Vec<f32>,
    /// Quantisation kernel dispatcher.
    pub dispatcher: KernelDispatcher,
    /// Precomputed RoPE table (used when `cfg.rope == true`).
    rope_table: Option<RopeTable>,

    // ── Scratch buffers (reused across tokens) ───────────────────────────
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_qkv: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_ffn_mid: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
    buf_parallel: Vec<f32>,
}

impl FalconForward {
    /// Construct a [`FalconForward`] from its components.
    pub fn new(
        cfg: FalconConfig,
        token_embd: Vec<f32>,
        layers: Vec<FalconLayer>,
        output_norm: LayerNorm,
        output_weight: Vec<f32>,
        max_context_length: usize,
    ) -> Self {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let qkv_total = (n_heads + 2 * n_kv) * head_dim;
        let intermediate = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let dispatcher = KernelDispatcher::new();

        let rope_table = if cfg.rope {
            Some(RopeTable::new_standard(
                head_dim,
                max_context_length,
                cfg.rope_freq_base,
            ))
        } else {
            None
        };

        Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output_weight,
            dispatcher,
            rope_table,
            buf_hidden: vec![0.0; hidden],
            buf_norm: vec![0.0; hidden],
            buf_qkv: vec![0.0; qkv_total],
            buf_q: vec![0.0; n_heads * head_dim],
            buf_k: vec![0.0; n_kv * head_dim],
            buf_v: vec![0.0; n_kv * head_dim],
            buf_attn_out: vec![0.0; hidden],
            buf_ffn_mid: vec![0.0; intermediate],
            buf_ffn_out: vec![0.0; hidden],
            buf_logits: vec![0.0; vocab],
            buf_attn_scores: vec![0.0; max_context_length],
            buf_parallel: vec![0.0; hidden],
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Obtain the kernel for the given tensor type.
    fn kernel_for_tensor(&self, weight: &QuantTensor) -> ArchResult<Box<dyn QuantKernel>> {
        self.dispatcher
            .get_kernel(weight.tensor_type)
            .map_err(ArchError::from)
    }

    /// Add bias (if present) to `buf`.
    fn add_bias_opt(buf: &mut [f32], bias: &Option<Vec<f32>>) {
        if let Some(b) = bias {
            for (o, &bv) in buf.iter_mut().zip(b.iter()) {
                *o += bv;
            }
        }
    }

    /// Embed a single token (no positional encoding — that is applied later).
    fn embed_token(&self, token: u32) -> ArchResult<Vec<f32>> {
        let hs = self.cfg.hidden_size;
        let off = token as usize * hs;
        let end = off + hs;
        if end > self.token_embd.len() {
            return Err(ArchError::ConfigMismatch {
                param: "token id".to_string(),
                expected: format!("< {}", self.cfg.vocab_size),
                got: token.to_string(),
            });
        }
        Ok(self.token_embd[off..end].to_vec())
    }

    /// Run fused QKV projection and split into Q/K/V buffers.
    ///
    /// Q  → `buf_q`  `[n_heads * head_dim]`
    /// K  → `buf_k`  `[n_kv_heads * head_dim]`
    /// V  → `buf_v`  `[n_kv_heads * head_dim]`
    fn project_qkv(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];
        let kernel = self.kernel_for_tensor(&layer.attn_qkv_weight)?;

        // We need to call kernel GEMV but we cannot hold a borrow on `self` at
        // the same time as accessing `buf_*`.  Use raw pointer access.
        //
        // Safety: `kernel` exists for the duration of this call; `buf_norm` is
        // not aliased by `buf_qkv`.
        let norm_slice: &[f32] = &self.buf_norm;
        let qkv_slice: &mut [f32] = &mut self.buf_qkv;

        kernel
            .gemv(&layer.attn_qkv_weight, norm_slice, qkv_slice)
            .map_err(ArchError::from)?;

        Self::add_bias_opt(qkv_slice, &layer.attn_qkv_bias);

        // Split: Q first, then K, then V
        let n_heads = self.cfg.n_heads;
        let n_kv = self.cfg.n_kv_heads;
        let head_dim = self.cfg.head_dim;
        let q_len = n_heads * head_dim;
        let k_len = n_kv * head_dim;

        let qkv = &self.buf_qkv;
        self.buf_q.copy_from_slice(&qkv[..q_len]);
        self.buf_k.copy_from_slice(&qkv[q_len..q_len + k_len]);
        self.buf_v
            .copy_from_slice(&qkv[q_len + k_len..q_len + 2 * k_len]);

        Ok(())
    }

    /// Apply ALiBi bias to attention scores for one head.
    ///
    /// Adds `slope * (position - key_position)` for each key position to the
    /// already-scaled dot-product scores.
    fn apply_alibi(scores: &mut [f32], seq_len: usize, current_pos: usize, slope: f32) {
        for (k_pos, score) in scores[..seq_len].iter_mut().enumerate() {
            let distance = current_pos as f32 - k_pos as f32;
            *score -= slope * distance;
        }
    }

    /// Scaled dot-product attention for one head with optional ALiBi.
    ///
    /// `q`            — query for this head `[head_dim]`
    /// `cached_keys`  — all cached key vectors `[seq_len * kv_head_dim]`
    /// `cached_vals`  — all cached value vectors `[seq_len * kv_head_dim]`
    /// `kv_head`      — which K/V head index to use (for GQA mapping)
    /// `output`       — output slice for this head `[head_dim]`
    /// `seq_len`      — number of valid positions currently in cache
    /// `current_pos`  — current token position (for ALiBi)
    /// `alibi_slope`  — pre-computed ALiBi slope (0.0 → ALiBi disabled)
    #[allow(clippy::too_many_arguments)]
    fn sdpa(
        q: &[f32],
        cached_keys: &[f32],
        cached_vals: &[f32],
        kv_head: usize,
        output: &mut [f32],
        scores: &mut [f32],
        head_dim: usize,
        seq_len: usize,
        current_pos: usize,
        alibi_slope: f32,
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = head_dim; // each cached kv entry is head_dim floats
        let kv_off = kv_head * head_dim; // offset into fused K or V slice

        // Compute dot products Q·K_t for all cached positions
        for (k_pos, score) in scores[..seq_len].iter_mut().enumerate() {
            let k_base = k_pos * kv_stride + kv_off;
            let k_vec = &cached_keys[k_base..k_base + head_dim];
            let dot: f32 = q.iter().zip(k_vec.iter()).map(|(&a, &b)| a * b).sum();
            *score = dot * scale;
        }

        // ALiBi positional bias
        if alibi_slope != 0.0 {
            Self::apply_alibi(&mut scores[..seq_len], seq_len, current_pos, alibi_slope);
        }

        // Causal mask: future positions get -inf (seq_len == current_pos+1, so
        // all cached positions are past/present → no masking needed here).

        softmax_inplace(&mut scores[..seq_len]);

        // Weighted sum of values
        output.iter_mut().for_each(|x| *x = 0.0);
        for (k_pos, &w) in scores[..seq_len].iter().enumerate() {
            let v_base = k_pos * kv_stride + kv_off;
            let v_vec = &cached_vals[v_base..v_base + head_dim];
            for (o, &v) in output.iter_mut().zip(v_vec.iter()) {
                *o += w * v;
            }
        }
    }

    /// Multi-head attention for a single token at `position`.
    ///
    /// Assumes `buf_q`, `buf_k`, `buf_v` are already filled and `buf_norm` is
    /// the layer-normalised input.  Writes the output-projected result into
    /// `buf_attn_out`.
    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let n_heads = self.cfg.n_heads;
        let n_kv = self.cfg.n_kv_heads;
        let head_dim = self.cfg.head_dim;

        // Apply RoPE to Q and K when configured
        if self.cfg.rope {
            if let Some(ref table) = self.rope_table {
                // Apply RoPE in-place to each Q head
                for h in 0..n_heads {
                    let off = h * head_dim;
                    table.apply(&mut self.buf_q[off..off + head_dim], position);
                }
                // Apply RoPE to each K head
                for h in 0..n_kv {
                    let off = h * head_dim;
                    table.apply(&mut self.buf_k[off..off + head_dim], position);
                }
            }
        }

        // Store K/V into cache  (flattened: all kv_heads concatenated)
        kv_cache.store_kv(layer_idx, &self.buf_k, &self.buf_v)?;

        let cached_keys = kv_cache.get_keys(layer_idx)?;
        let cached_vals = kv_cache.get_values(layer_idx)?;
        let seq_len = position + 1;

        // GQA group size: how many Q heads share one K/V head
        let groups = n_heads / n_kv;

        // Run SDPA for each Q head, accumulate into buf_attn_out
        self.buf_attn_out.iter_mut().for_each(|x| *x = 0.0);

        // We need mutable access to buf_attn_scores and alibi config inside
        // the loop while also reading buf_q.  To avoid borrow conflicts we
        // split out only the values we need before the loop.
        let alibi = self.cfg.alibi;
        let n_heads_for_slope = n_heads;

        // Stack-allocate a small head output buffer (head_dim typically ≤ 256)
        let mut head_out = vec![0.0_f32; head_dim];

        for h in 0..n_heads {
            let kv_head = h / groups;
            let q_off = h * head_dim;

            let alibi_slope = if alibi {
                FalconConfig::alibi_slope(h, n_heads_for_slope)
            } else {
                0.0
            };

            let q_slice = &self.buf_q[q_off..q_off + head_dim];

            Self::sdpa(
                q_slice,
                cached_keys,
                cached_vals,
                kv_head,
                &mut head_out,
                &mut self.buf_attn_scores,
                head_dim,
                seq_len,
                position,
                alibi_slope,
            );

            // Write head output into correct slice of attn_out
            let out_off = h * head_dim;
            self.buf_attn_out[out_off..out_off + head_dim].copy_from_slice(&head_out);
        }

        // Output projection: attn_out = W_o @ (all head outputs) + bias
        let layer = &self.layers[layer_idx];
        let out_kernel = self.kernel_for_tensor(&layer.attn_out_weight)?;
        let attn_in = self.buf_attn_out.clone(); // temp copy to avoid alias
                                                 // Reuse buf_parallel as a scratch for the projected output
        out_kernel
            .gemv(&layer.attn_out_weight, &attn_in, &mut self.buf_parallel)
            .map_err(ArchError::from)?;
        Self::add_bias_opt(&mut self.buf_parallel, &layer.attn_out_bias);

        // Copy projected attention result back into buf_attn_out
        self.buf_attn_out.copy_from_slice(&self.buf_parallel);

        Ok(())
    }

    /// Feed-forward network for a single layer.
    ///
    /// Uses GELU activation (Falcon-1) or SiLU (Falcon-2 uses silu in practice
    /// but libtransformers Falcon still uses GELU).  For simplicity we apply
    /// GELU consistently; the caller may check `FalconConfig` if a separate
    /// activation is needed.
    ///
    /// Input is taken from `buf_norm`.  Output is written into `buf_ffn_out`.
    fn ffn(&mut self, layer_idx: usize, from_norm: &[f32]) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];

        // Up projection
        let up_kernel = self.kernel_for_tensor(&layer.ffn_up_weight)?;
        let mid = &mut self.buf_ffn_mid;
        up_kernel
            .gemv(&layer.ffn_up_weight, from_norm, mid)
            .map_err(ArchError::from)?;
        Self::add_bias_opt(mid, &layer.ffn_up_bias);

        // GELU activation
        gelu_inplace(&mut self.buf_ffn_mid);

        // Down projection
        let ffn_in = self.buf_ffn_mid.clone();
        let down_kernel = self.kernel_for_tensor(&layer.ffn_down_weight)?;
        let out = &mut self.buf_ffn_out;
        down_kernel
            .gemv(&layer.ffn_down_weight, &ffn_in, out)
            .map_err(ArchError::from)?;
        Self::add_bias_opt(out, &layer.ffn_down_bias);

        Ok(())
    }

    /// Run a single Falcon decoder layer.
    ///
    /// Handles both parallel (Falcon-1) and sequential (Falcon-2) modes.
    fn layer_forward(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let hidden_size = self.cfg.hidden_size;

        // 1. Pre-attention layer norm → buf_norm
        let layer_attn_norm = self.layers[layer_idx].attn_norm.clone();
        layer_attn_norm.forward_to(&self.buf_hidden, &mut self.buf_norm);

        // 2. Fused QKV projection
        self.project_qkv(layer_idx)?;

        // 3. Attention (writes to buf_attn_out)
        self.attention(layer_idx, position, kv_cache)?;

        if self.cfg.parallel_attn {
            // ── Parallel mode ────────────────────────────────────────────────
            // FFN uses the SAME layer-norm input as the attention branch.
            // We clone buf_norm because ffn() will read from its argument.
            let ffn_input = self.buf_norm.clone();
            self.ffn(layer_idx, &ffn_input)?;

            // Sum attn + ffn outputs, then add residual
            for i in 0..hidden_size {
                self.buf_hidden[i] += self.buf_attn_out[i] + self.buf_ffn_out[i];
            }
        } else {
            // ── Sequential mode ──────────────────────────────────────────────
            // Add attention residual
            for i in 0..hidden_size {
                self.buf_hidden[i] += self.buf_attn_out[i];
            }

            // Pre-FFN layer norm (Falcon-2 has a separate ffn_norm)
            let ffn_norm_input = self.buf_hidden.clone();
            let ffn_norm = self.layers[layer_idx]
                .ffn_norm
                .clone()
                .unwrap_or_else(|| layer_attn_norm.clone());
            let mut norm_for_ffn = vec![0.0_f32; hidden_size];
            ffn_norm.forward_to(&ffn_norm_input, &mut norm_for_ffn);

            // FFN
            self.ffn(layer_idx, &norm_for_ffn)?;

            // Add FFN residual
            for i in 0..hidden_size {
                self.buf_hidden[i] += self.buf_ffn_out[i];
            }
        }

        Ok(())
    }

    /// Run the full Falcon forward pass for a sequence of tokens.
    ///
    /// Returns logits for the last token only.
    fn forward_tokens(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let seq_start = kv_cache.seq_len();

        for (step, &token) in tokens.iter().enumerate() {
            let position = seq_start + step;

            // Embed token
            let emb = self.embed_token(token)?;
            self.buf_hidden.copy_from_slice(&emb);

            // Run all transformer layers
            let n_layers = self.cfg.n_layers;
            for layer_idx in 0..n_layers {
                self.layer_forward(layer_idx, position, kv_cache)?;
            }

            kv_cache.advance();
        }

        // Final layer norm
        let hidden_clone = self.buf_hidden.clone();
        let mut normed = vec![0.0_f32; self.cfg.hidden_size];
        self.output_norm.forward_to(&hidden_clone, &mut normed);

        // LM head: logits = output_weight @ normed
        // output_weight is [vocab_size, hidden_size] stored row-major f32
        let vocab = self.cfg.vocab_size;
        let hs = self.cfg.hidden_size;
        for v in 0..vocab {
            let row = &self.output_weight[v * hs..(v + 1) * hs];
            self.buf_logits[v] = row.iter().zip(normed.iter()).map(|(&w, &x)| w * x).sum();
        }

        Ok(self.buf_logits.clone())
    }
}

// ── ForwardPass impl ─────────────────────────────────────────────────────────

impl ForwardPass for FalconForward {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.forward_tokens(tokens, kv_cache)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn max_context_length(&self) -> usize {
        self.buf_attn_scores.len()
    }

    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::layer_norm::LayerNorm;
    use crate::error::ArchResult;
    use crate::falcon::config::FalconConfig;
    use crate::traits::KvCacheAccess;

    // ── Minimal KV cache for tests ────────────────────────────────────────────
    struct DummyKvCache {
        seq_len: usize,
        keys: Vec<Vec<f32>>,
        vals: Vec<Vec<f32>>,
    }

    impl DummyKvCache {
        fn new(n_layers: usize) -> Self {
            Self {
                seq_len: 0,
                keys: vec![Vec::new(); n_layers],
                vals: vec![Vec::new(); n_layers],
            }
        }
    }

    impl KvCacheAccess for DummyKvCache {
        fn seq_len(&self) -> usize {
            self.seq_len
        }
        fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
            self.keys[layer].extend_from_slice(key);
            self.vals[layer].extend_from_slice(value);
            Ok(())
        }
        fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.keys[layer])
        }
        fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.vals[layer])
        }
        fn advance(&mut self) {
            self.seq_len += 1;
        }
    }

    fn make_dummy_layer(cfg: &FalconConfig) -> FalconLayer {
        use oxillama_gguf::GgufTensorType;
        use oxillama_quant::QuantTensor;

        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let qkv_total = (n_heads + 2 * n_kv) * head_dim;

        let make_tensor = |rows: usize, cols: usize| QuantTensor {
            tensor_type: GgufTensorType::F32,
            shape: vec![rows, cols],
            data: vec![0u8; rows * cols * 4],
        };

        FalconLayer {
            attn_norm: LayerNorm::new(vec![1.0; hidden], Some(vec![0.0; hidden]), 1e-5),
            ffn_norm: None,
            attn_qkv_weight: make_tensor(qkv_total, hidden),
            attn_qkv_bias: None,
            attn_out_weight: make_tensor(hidden, n_heads * head_dim),
            attn_out_bias: None,
            ffn_up_weight: make_tensor(intermediate, hidden),
            ffn_up_bias: None,
            ffn_down_weight: make_tensor(hidden, intermediate),
            ffn_down_bias: None,
        }
    }

    fn make_falcon1_forward() -> FalconForward {
        let cfg = FalconConfig {
            n_heads: 4,
            n_kv_heads: 1,
            n_layers: 2,
            hidden_size: 32,
            vocab_size: 64,
            intermediate_size: 64,
            norm_eps: 1e-5,
            parallel_attn: true,
            alibi: true,
            rope: false,
            rope_freq_base: 10_000.0,
            head_dim: 8,
        };

        let layers: Vec<FalconLayer> = (0..cfg.n_layers).map(|_| make_dummy_layer(&cfg)).collect();

        let token_embd = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        let output_weight = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        let output_norm = LayerNorm::new(
            vec![1.0; cfg.hidden_size],
            Some(vec![0.0; cfg.hidden_size]),
            1e-5,
        );

        FalconForward::new(cfg, token_embd, layers, output_norm, output_weight, 128)
    }

    fn make_falcon2_forward() -> FalconForward {
        let cfg = FalconConfig {
            n_heads: 4,
            n_kv_heads: 2,
            n_layers: 2,
            hidden_size: 32,
            vocab_size: 64,
            intermediate_size: 64,
            norm_eps: 1e-5,
            parallel_attn: false,
            alibi: false,
            rope: true,
            rope_freq_base: 10_000.0,
            head_dim: 8,
        };

        let mut layers: Vec<FalconLayer> =
            (0..cfg.n_layers).map(|_| make_dummy_layer(&cfg)).collect();

        // Falcon-2 has a separate ffn_norm
        for layer in layers.iter_mut() {
            layer.ffn_norm = Some(LayerNorm::new(
                vec![1.0; cfg.hidden_size],
                Some(vec![0.0; cfg.hidden_size]),
                1e-5,
            ));
        }

        let token_embd = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        let output_weight = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        let output_norm = LayerNorm::new(
            vec![1.0; cfg.hidden_size],
            Some(vec![0.0; cfg.hidden_size]),
            1e-5,
        );

        FalconForward::new(cfg, token_embd, layers, output_norm, output_weight, 128)
    }

    #[test]
    fn test_falcon1_forward_returns_vocab_size_logits() {
        let mut model = make_falcon1_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let logits = model.forward(&[0u32], &mut cache).unwrap();
        assert_eq!(logits.len(), 64, "logits must match vocab_size");
    }

    #[test]
    fn test_falcon2_forward_returns_vocab_size_logits() {
        let mut model = make_falcon2_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let logits = model.forward(&[0u32], &mut cache).unwrap();
        assert_eq!(logits.len(), 64, "logits must match vocab_size");
    }

    #[test]
    fn test_forward_multi_token() {
        let mut model = make_falcon1_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let logits = model.forward(&[0u32, 1u32, 2u32], &mut cache).unwrap();
        assert_eq!(logits.len(), 64);
    }

    #[test]
    fn test_kv_cache_advances_correctly() {
        let mut model = make_falcon1_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        assert_eq!(cache.seq_len(), 0);
        model.forward(&[0u32], &mut cache).unwrap();
        assert_eq!(cache.seq_len(), 1);
        model.forward(&[1u32], &mut cache).unwrap();
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn test_vocab_size_and_hidden_size_accessors() {
        let model = make_falcon1_forward();
        assert_eq!(model.vocab_size(), 64);
        assert_eq!(model.hidden_size(), 32);
        assert_eq!(model.max_context_length(), 128);
    }

    #[test]
    fn test_invalid_token_returns_error() {
        let mut model = make_falcon1_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        // Token 64 is out-of-vocab (vocab_size = 64)
        let result = model.forward(&[64u32], &mut cache);
        assert!(result.is_err(), "Out-of-vocab token should return an error");
    }
}

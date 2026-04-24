//! OLMo2 transformer forward pass.
//!
//! OLMo2 uses post-norm style with per-head QK-norm:
//!
//! ## Forward per layer
//!
//! ```text
//!   residual_attn = x
//!   q = x @ Wq
//!   k = x @ Wk
//!   v = x @ Wv
//!   q = qk_norm(q, q_norm)   ← per-head QK-norm
//!   k = qk_norm(k, k_norm)
//!   q, k = rope(q, pos), rope(k, pos)
//!   attn_out = sdpa(q, k, v, kv_cache)
//!   attn_out = attn_out @ Wo
//!   attn_out = rms_norm(attn_out, attn_post_norm)  ← post-norm
//!   x = residual_attn + attn_out
//!
//!   residual_ffn = x
//!   ffn_out = swiglu(x, Wgate, Wup, Wdown)
//!   ffn_out = rms_norm(ffn_out, ffn_post_norm)     ← post-norm
//!   x = residual_ffn + ffn_out
//! ```

use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::error::{ArchError, ArchResult};
use crate::llama::softmax_inplace;
use crate::olmo2::config::Olmo2Config;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantKernel, QuantTensor};

// ── Layer storage ─────────────────────────────────────────────────────────────

/// A single OLMo2 transformer layer.
pub struct Olmo2Layer {
    /// Per-head query RMSNorm.
    pub q_norm: RmsNorm,
    /// Per-head key RMSNorm.
    pub k_norm: RmsNorm,
    /// Post-attention RMSNorm.
    pub attn_post_norm: RmsNorm,
    /// Post-FFN RMSNorm.
    pub ffn_post_norm: RmsNorm,
    /// Query projection `[n_heads * head_dim, hidden_size]`.
    pub attn_q: QuantTensor,
    /// Key projection `[n_kv_heads * head_dim, hidden_size]`.
    pub attn_k: QuantTensor,
    /// Value projection `[n_kv_heads * head_dim, hidden_size]`.
    pub attn_v: QuantTensor,
    /// Attention output projection `[hidden_size, n_heads * head_dim]`.
    pub attn_out: QuantTensor,
    /// FFN gate projection (SwiGLU) `[intermediate_size, hidden_size]`.
    pub ffn_gate: QuantTensor,
    /// FFN up projection `[intermediate_size, hidden_size]`.
    pub ffn_up: QuantTensor,
    /// FFN down projection `[hidden_size, intermediate_size]`.
    pub ffn_down: QuantTensor,
}

// ── Full model ────────────────────────────────────────────────────────────────

/// Loaded OLMo2 model capable of running forward passes.
pub struct Olmo2Forward {
    /// OLMo2-specific hyperparameters.
    pub cfg: Olmo2Config,
    /// Dequantised token embedding table `[vocab_size * hidden_size]`.
    pub token_embd: Vec<f32>,
    /// All transformer layers.
    pub layers: Vec<Olmo2Layer>,
    /// Final output RMSNorm.
    pub output_norm: RmsNorm,
    /// LM head weight `[vocab_size * hidden_size]` (dequantised f32).
    pub output_weight: Vec<f32>,
    /// Quantisation kernel dispatcher.
    pub dispatcher: KernelDispatcher,
    /// Precomputed RoPE table.
    rope_table: RopeTable,

    // ── Scratch buffers ────────────────────────────────────────────────────
    buf_hidden: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl Olmo2Forward {
    /// Construct an [`Olmo2Forward`] from pre-loaded components.
    pub fn new(
        cfg: Olmo2Config,
        token_embd: Vec<f32>,
        layers: Vec<Olmo2Layer>,
        output_norm: RmsNorm,
        output_weight: Vec<f32>,
        max_context_length: usize,
    ) -> Self {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let intermediate = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let dispatcher = KernelDispatcher::new();
        let rope_table = RopeTable::new_standard(head_dim, max_context_length, cfg.rope_freq_base);

        Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output_weight,
            dispatcher,
            rope_table,
            buf_hidden: vec![0.0; hidden],
            buf_q: vec![0.0; n_heads * head_dim],
            buf_k: vec![0.0; n_kv * head_dim],
            buf_v: vec![0.0; n_kv * head_dim],
            buf_attn_out: vec![0.0; hidden],
            buf_gate: vec![0.0; intermediate],
            buf_up: vec![0.0; intermediate],
            buf_ffn_out: vec![0.0; hidden],
            buf_logits: vec![0.0; vocab],
            buf_attn_scores: vec![0.0; max_context_length],
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn kernel_for(&self, weight: &QuantTensor) -> ArchResult<Box<dyn QuantKernel>> {
        self.dispatcher
            .get_kernel(weight.tensor_type)
            .map_err(ArchError::from)
    }

    fn embed_token(&self, token: u32) -> ArchResult<Vec<f32>> {
        let hs = self.cfg.hidden_size;
        let off = token as usize * hs;
        let end = off + hs;
        if end > self.token_embd.len() {
            return Err(ArchError::ConfigMismatch {
                param: "token_id".to_string(),
                expected: format!("< {}", self.cfg.vocab_size),
                got: token.to_string(),
            });
        }
        Ok(self.token_embd[off..end].to_vec())
    }

    /// Apply per-head RMSNorm to a packed multi-head tensor.
    ///
    /// `buf` has shape `[n_heads, head_dim]` laid out contiguously.
    /// `norm` has `weight` of length `head_dim` (shared across heads).
    fn apply_qk_norm(buf: &mut [f32], norm: &RmsNorm, head_dim: usize) {
        let n_heads = buf.len() / head_dim.max(1);
        for h in 0..n_heads {
            let off = h * head_dim;
            norm.forward(&mut buf[off..off + head_dim]);
        }
    }

    /// Scaled dot-product attention for one head.
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
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = head_dim;
        let kv_off = kv_head * head_dim;

        for (k_pos, score) in scores[..seq_len].iter_mut().enumerate() {
            let k_base = k_pos * kv_stride + kv_off;
            let k_vec = &cached_keys[k_base..k_base + head_dim];
            *score = q
                .iter()
                .zip(k_vec.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                * scale;
        }

        softmax_inplace(&mut scores[..seq_len]);

        output.iter_mut().for_each(|x| *x = 0.0);
        for (k_pos, &w) in scores[..seq_len].iter().enumerate() {
            let v_base = k_pos * kv_stride + kv_off;
            let v_vec = &cached_vals[v_base..v_base + head_dim];
            for (o, &v) in output.iter_mut().zip(v_vec.iter()) {
                *o += w * v;
            }
        }
    }

    fn run_attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let n_heads = self.cfg.n_heads;
        let n_kv = self.cfg.n_kv_heads;
        let head_dim = self.cfg.head_dim;
        let seq_len = kv_cache.seq_len() + 1;

        // Project Q, K, V from hidden state (no pre-norm on x in OLMo2)
        {
            let layer = &self.layers[layer_idx];
            let kq = self.kernel_for(&layer.attn_q)?;
            let kk = self.kernel_for(&layer.attn_k)?;
            let kv = self.kernel_for(&layer.attn_v)?;
            let hidden_copy = self.buf_hidden.clone();
            kq.gemv(&layer.attn_q, &hidden_copy, &mut self.buf_q)
                .map_err(ArchError::from)?;
            kk.gemv(&layer.attn_k, &hidden_copy, &mut self.buf_k)
                .map_err(ArchError::from)?;
            kv.gemv(&layer.attn_v, &hidden_copy, &mut self.buf_v)
                .map_err(ArchError::from)?;
        }

        // Per-head QK-norm
        {
            let q_norm = RmsNorm::new(
                self.layers[layer_idx].q_norm.weight.clone(),
                self.layers[layer_idx].q_norm.eps,
            );
            let k_norm = RmsNorm::new(
                self.layers[layer_idx].k_norm.weight.clone(),
                self.layers[layer_idx].k_norm.eps,
            );
            Self::apply_qk_norm(&mut self.buf_q, &q_norm, head_dim);
            Self::apply_qk_norm(&mut self.buf_k, &k_norm, head_dim);
        }

        // Apply RoPE
        for h in 0..n_heads {
            let off = h * head_dim;
            self.rope_table
                .apply(&mut self.buf_q[off..off + head_dim], position);
        }
        for h in 0..n_kv {
            let off = h * head_dim;
            self.rope_table
                .apply(&mut self.buf_k[off..off + head_dim], position);
        }

        // Store K/V in cache
        kv_cache.store_kv(layer_idx, &self.buf_k, &self.buf_v)?;

        let cached_keys = kv_cache.get_keys(layer_idx)?;
        let cached_vals = kv_cache.get_values(layer_idx)?;

        // Multi-head attention (GQA)
        let mut head_buf = vec![0.0f32; head_dim];
        let gqa_ratio = n_heads.checked_div(n_kv).unwrap_or(1).max(1);

        let mut attn_concat = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kv_head = h / gqa_ratio;
            let q_off = h * head_dim;
            let q_slice = &self.buf_q[q_off..q_off + head_dim];
            Self::sdpa(
                q_slice,
                cached_keys,
                cached_vals,
                kv_head,
                &mut head_buf,
                &mut self.buf_attn_scores,
                head_dim,
                seq_len,
            );
            attn_concat[q_off..q_off + head_dim].copy_from_slice(&head_buf);
        }

        // Output projection
        {
            let layer = &self.layers[layer_idx];
            let ko = self.kernel_for(&layer.attn_out)?;
            ko.gemv(&layer.attn_out, &attn_concat, &mut self.buf_attn_out)
                .map_err(ArchError::from)?;
        }

        // Post-attn RMSNorm
        {
            let post_norm = RmsNorm::new(
                self.layers[layer_idx].attn_post_norm.weight.clone(),
                self.layers[layer_idx].attn_post_norm.eps,
            );
            post_norm.forward(&mut self.buf_attn_out);
        }

        Ok(())
    }

    fn run_ffn(&mut self, layer_idx: usize) -> ArchResult<()> {
        {
            let layer = &self.layers[layer_idx];
            let kg = self.kernel_for(&layer.ffn_gate)?;
            let ku = self.kernel_for(&layer.ffn_up)?;
            let hidden_copy = self.buf_hidden.clone();
            kg.gemv(&layer.ffn_gate, &hidden_copy, &mut self.buf_gate)
                .map_err(ArchError::from)?;
            ku.gemv(&layer.ffn_up, &hidden_copy, &mut self.buf_up)
                .map_err(ArchError::from)?;
        }
        swiglu_inplace(&mut self.buf_gate, &self.buf_up.clone());
        {
            let layer = &self.layers[layer_idx];
            let kd = self.kernel_for(&layer.ffn_down)?;
            let gate_copy = self.buf_gate.clone();
            kd.gemv(&layer.ffn_down, &gate_copy, &mut self.buf_ffn_out)
                .map_err(ArchError::from)?;
        }
        // Post-FFN RMSNorm
        {
            let post_norm = RmsNorm::new(
                self.layers[layer_idx].ffn_post_norm.weight.clone(),
                self.layers[layer_idx].ffn_post_norm.eps,
            );
            post_norm.forward(&mut self.buf_ffn_out);
        }
        Ok(())
    }

    /// Core forward: run all layers and final norm, returning hidden state.
    fn run_layers(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(ArchError::ConfigMismatch {
                param: "tokens".to_string(),
                expected: "non-empty".to_string(),
                got: "empty".to_string(),
            });
        }

        let token = tokens[tokens.len() - 1];
        let position = kv_cache.seq_len();

        let x = self.embed_token(token)?;
        self.buf_hidden.copy_from_slice(&x);

        for layer_idx in 0..self.cfg.n_layers {
            // Save residual before attention
            let residual_attn = self.buf_hidden.clone();

            self.run_attention(layer_idx, position, kv_cache)?;

            // Residual add (attn)
            for (i, h) in self.buf_hidden.iter_mut().enumerate() {
                *h = residual_attn[i] + self.buf_attn_out[i];
            }

            // Save residual before FFN
            let residual_ffn = self.buf_hidden.clone();

            self.run_ffn(layer_idx)?;

            // Residual add (FFN)
            for (i, h) in self.buf_hidden.iter_mut().enumerate() {
                *h = residual_ffn[i] + self.buf_ffn_out[i];
            }
        }

        kv_cache.advance();

        // Final output norm
        let weight = self.output_norm.weight.clone();
        let eps = self.output_norm.eps;
        let out_norm = RmsNorm::new(weight, eps);
        let mut normed = self.buf_hidden.clone();
        out_norm.forward(&mut normed);
        Ok(normed)
    }
}

impl ForwardPass for Olmo2Forward {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let normed = self.run_layers(tokens, kv_cache)?;

        let vocab = self.cfg.vocab_size;
        let hidden = self.cfg.hidden_size;
        self.buf_logits.fill(0.0);
        for v in 0..vocab {
            let row = &self.output_weight[v * hidden..(v + 1) * hidden];
            self.buf_logits[v] = normed.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
        }
        Ok(self.buf_logits.clone())
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn max_context_length(&self) -> usize {
        self.cfg.max_context_length
    }

    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rms_norm::RmsNorm;
    use crate::olmo2::config::Olmo2Config;
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    struct NoopKvCache {
        seq_len: usize,
        keys: std::collections::HashMap<usize, Vec<f32>>,
        vals: std::collections::HashMap<usize, Vec<f32>>,
    }

    impl NoopKvCache {
        fn new() -> Self {
            Self {
                seq_len: 0,
                keys: Default::default(),
                vals: Default::default(),
            }
        }
    }

    impl crate::traits::KvCacheAccess for NoopKvCache {
        fn seq_len(&self) -> usize {
            self.seq_len
        }
        fn store_kv(
            &mut self,
            layer: usize,
            key: &[f32],
            value: &[f32],
        ) -> crate::error::ArchResult<()> {
            self.keys.insert(layer, key.to_vec());
            self.vals.insert(layer, value.to_vec());
            Ok(())
        }
        fn get_keys(&self, layer: usize) -> crate::error::ArchResult<&[f32]> {
            self.keys.get(&layer).map(|v| v.as_slice()).ok_or_else(|| {
                crate::error::ArchError::MissingTensor {
                    name: format!("keys layer {layer}"),
                }
            })
        }
        fn get_values(&self, layer: usize) -> crate::error::ArchResult<&[f32]> {
            self.vals.get(&layer).map(|v| v.as_slice()).ok_or_else(|| {
                crate::error::ArchError::MissingTensor {
                    name: format!("values layer {layer}"),
                }
            })
        }
        fn advance(&mut self) {
            self.seq_len += 1;
        }
    }

    fn make_zero_quant_tensor(rows: usize, cols: usize) -> QuantTensor {
        QuantTensor {
            data: vec![0u8; rows * cols * 4],
            tensor_type: GgufTensorType::F32,
            shape: vec![rows, cols],
        }
    }

    fn make_rms_norm(size: usize) -> RmsNorm {
        RmsNorm::new(vec![1.0f32; size], 1e-5)
    }

    fn make_forward(hidden: usize, intermediate: usize, vocab: usize) -> Olmo2Forward {
        let n_heads = 2usize;
        let n_kv = 2usize;
        let head_dim = hidden / n_heads;

        let cfg = Olmo2Config {
            n_layers: 1,
            hidden_size: hidden,
            n_heads,
            n_kv_heads: n_kv,
            intermediate_size: intermediate,
            vocab_size: vocab,
            max_context_length: 16,
            norm_eps: 1e-5,
            rope_freq_base: 500_000.0,
            head_dim,
        };

        let token_embd = vec![0.0f32; vocab * hidden];
        let output_weight = vec![0.0f32; vocab * hidden];
        let output_norm = make_rms_norm(hidden);

        let layer = Olmo2Layer {
            q_norm: make_rms_norm(head_dim),
            k_norm: make_rms_norm(head_dim),
            attn_post_norm: make_rms_norm(hidden),
            ffn_post_norm: make_rms_norm(hidden),
            attn_q: make_zero_quant_tensor(n_heads * head_dim, hidden),
            attn_k: make_zero_quant_tensor(n_kv * head_dim, hidden),
            attn_v: make_zero_quant_tensor(n_kv * head_dim, hidden),
            attn_out: make_zero_quant_tensor(hidden, n_heads * head_dim),
            ffn_gate: make_zero_quant_tensor(intermediate, hidden),
            ffn_up: make_zero_quant_tensor(intermediate, hidden),
            ffn_down: make_zero_quant_tensor(hidden, intermediate),
        };

        Olmo2Forward::new(cfg, token_embd, vec![layer], output_norm, output_weight, 16)
    }

    #[test]
    fn test_forward_returns_correct_vocab_size() {
        let vocab = 32;
        let mut fwd = make_forward(4, 8, vocab);
        let mut cache = NoopKvCache::new();
        let result = fwd.forward(&[0u32], &mut cache);
        assert!(result.is_ok(), "forward should succeed: {:?}", result);
        let logits = result.expect("logits");
        assert_eq!(logits.len(), vocab);
    }

    #[test]
    fn test_vocab_size_accessor() {
        let fwd = make_forward(4, 8, 100);
        assert_eq!(fwd.vocab_size(), 100);
    }

    #[test]
    fn test_hidden_size_accessor() {
        let fwd = make_forward(4, 8, 100);
        assert_eq!(fwd.hidden_size(), 4);
    }

    #[test]
    fn test_max_context_length_accessor() {
        let fwd = make_forward(4, 8, 100);
        assert_eq!(fwd.max_context_length(), 16);
    }
}

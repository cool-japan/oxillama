//! MiniCPM transformer forward pass.
//!
//! MiniCPM is LLaMA with a scaled-embedding input: after the embedding
//! lookup but before the first layer norm, the hidden vector is multiplied
//! by `embedding_scale = hidden_size / dim_model_base`.  Everything else
//! (RMSNorm, RoPE, GQA, SwiGLU) is identical to standard LLaMA.
//!
//! ## Forward per layer
//!
//! ```text
//!   residual = x
//!   x = rms_norm(x, attn_norm)
//!   q, k, v = x @ Wq, x @ Wk, x @ Wv
//!   q, k = rope(q, pos), rope(k, pos)
//!   attn_out = sdpa(q, k, v, kv_cache)
//!   x = attn_out @ Wo + residual
//!   residual = x
//!   x = rms_norm(x, ffn_norm)
//!   x = swiglu(x, Wgate, Wup, Wdown) + residual
//! ```

use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::error::{ArchError, ArchResult};
use crate::llama::softmax_inplace;
use crate::minicpm::config::MiniCpmConfig;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantKernel, QuantTensor};

// ── Layer storage ─────────────────────────────────────────────────────────────

/// A single MiniCPM (LLaMA-style) transformer layer.
pub struct MiniCpmLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
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
    /// FFN up projection (SwiGLU) `[intermediate_size, hidden_size]`.
    pub ffn_up: QuantTensor,
    /// FFN down projection `[hidden_size, intermediate_size]`.
    pub ffn_down: QuantTensor,
}

// ── Full model ────────────────────────────────────────────────────────────────

/// Loaded MiniCPM model capable of running forward passes.
pub struct MiniCpmForward {
    /// MiniCPM-specific hyperparameters.
    pub cfg: MiniCpmConfig,
    /// Dequantised token embedding table `[vocab_size * hidden_size]`.
    pub token_embd: Vec<f32>,
    /// All transformer layers.
    pub layers: Vec<MiniCpmLayer>,
    /// Final output RMSNorm.
    pub output_norm: RmsNorm,
    /// LM head weight `[vocab_size * hidden_size]` (dequantised f32).
    pub output_weight: Vec<f32>,
    /// Quantisation kernel dispatcher.
    pub dispatcher: KernelDispatcher,
    /// Precomputed RoPE table.
    rope_table: RopeTable,

    // ── Scratch buffers (reused across tokens) ────────────────────────────
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
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

impl MiniCpmForward {
    /// Construct a [`MiniCpmForward`] from pre-loaded components.
    pub fn new(
        cfg: MiniCpmConfig,
        token_embd: Vec<f32>,
        layers: Vec<MiniCpmLayer>,
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
            buf_norm: vec![0.0; hidden],
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

    /// Run attention for a single token, writing result into `buf_attn_out`.
    fn run_attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let n_heads = self.cfg.n_heads;
        let n_kv = self.cfg.n_kv_heads;
        let head_dim = self.cfg.head_dim;
        let seq_len = kv_cache.seq_len() + 1; // include current position

        // Project Q, K, V
        {
            let layer = &self.layers[layer_idx];
            let kernel_q = self.kernel_for(&layer.attn_q)?;
            let kernel_k = self.kernel_for(&layer.attn_k)?;
            let kernel_v = self.kernel_for(&layer.attn_v)?;
            kernel_q
                .gemv(&layer.attn_q, &self.buf_norm, &mut self.buf_q)
                .map_err(ArchError::from)?;
            kernel_k
                .gemv(&layer.attn_k, &self.buf_norm, &mut self.buf_k)
                .map_err(ArchError::from)?;
            kernel_v
                .gemv(&layer.attn_v, &self.buf_norm, &mut self.buf_v)
                .map_err(ArchError::from)?;
        }

        // Apply RoPE to each head of Q and K
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

        // Store current K and V in cache
        kv_cache.store_kv(layer_idx, &self.buf_k, &self.buf_v)?;

        let cached_keys = kv_cache.get_keys(layer_idx)?;
        let cached_vals = kv_cache.get_values(layer_idx)?;

        // Multi-head attention with GQA
        let mut head_buf = vec![0.0f32; head_dim];
        let gqa_ratio = n_heads.checked_div(n_kv).unwrap_or(1).max(1);

        // collect results per head into buf_attn_out
        self.buf_attn_out.fill(0.0);
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
            let out_off = h * head_dim;
            self.buf_attn_out[out_off..out_off + head_dim].copy_from_slice(&head_buf);
        }

        // Output projection: buf_attn_out → buf_hidden (reuse buf_norm as tmp)
        {
            let layer = &self.layers[layer_idx];
            let kernel_o = self.kernel_for(&layer.attn_out)?;
            // Write projected output into buf_norm (temporary), then copy to buf_attn_out
            let attn_out_copy = self.buf_attn_out.clone();
            kernel_o
                .gemv(&layer.attn_out, &attn_out_copy, &mut self.buf_norm)
                .map_err(ArchError::from)?;
        }
        self.buf_attn_out.copy_from_slice(&self.buf_norm);

        Ok(())
    }

    fn run_ffn(&mut self, layer_idx: usize) -> ArchResult<()> {
        {
            let layer = &self.layers[layer_idx];
            let kg = self.kernel_for(&layer.ffn_gate)?;
            let ku = self.kernel_for(&layer.ffn_up)?;
            let norm_copy = self.buf_norm.clone();
            kg.gemv(&layer.ffn_gate, &norm_copy, &mut self.buf_gate)
                .map_err(ArchError::from)?;
            ku.gemv(&layer.ffn_up, &norm_copy, &mut self.buf_up)
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
        Ok(())
    }

    /// Core forward pass returning the post-norm hidden state.
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

        // For simplicity, process the last token (single-token step mode).
        let token = tokens[tokens.len() - 1];
        let position = kv_cache.seq_len();

        // Token embedding + MiniCPM scaling
        let mut x = self.embed_token(token)?;
        let scale = self.cfg.embedding_scale;
        if (scale - 1.0).abs() > 1e-6 {
            for v in x.iter_mut() {
                *v *= scale;
            }
        }

        self.buf_hidden.copy_from_slice(&x);

        for layer_idx in 0..self.cfg.n_layers {
            // Pre-attn RMSNorm
            {
                let weight = self.layers[layer_idx].attn_norm.weight.clone();
                let eps = self.layers[layer_idx].attn_norm.eps;
                let norm = RmsNorm::new(weight, eps);
                norm.forward_to(&self.buf_hidden, &mut self.buf_norm);
            }

            self.run_attention(layer_idx, position, kv_cache)?;

            // Residual add
            for (h, &a) in self.buf_hidden.iter_mut().zip(self.buf_attn_out.iter()) {
                *h += a;
            }

            // Pre-FFN RMSNorm
            {
                let weight = self.layers[layer_idx].ffn_norm.weight.clone();
                let eps = self.layers[layer_idx].ffn_norm.eps;
                let norm = RmsNorm::new(weight, eps);
                norm.forward_to(&self.buf_hidden, &mut self.buf_norm);
            }

            self.run_ffn(layer_idx)?;

            // Residual add
            for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
                *h += f;
            }
        }

        kv_cache.advance();

        // Final RMSNorm
        let hidden_copy = self.buf_hidden.clone();
        let weight = self.output_norm.weight.clone();
        let eps = self.output_norm.eps;
        let out_norm = RmsNorm::new(weight, eps);
        let mut normed = hidden_copy;
        out_norm.forward(&mut normed);
        Ok(normed)
    }
}

impl ForwardPass for MiniCpmForward {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let normed = self.run_layers(tokens, kv_cache)?;

        // LM head projection
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
    use crate::minicpm::config::MiniCpmConfig;
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

    fn make_forward(hidden: usize, intermediate: usize, vocab: usize) -> MiniCpmForward {
        let cfg = MiniCpmConfig {
            n_layers: 1,
            hidden_size: hidden,
            n_heads: 2,
            n_kv_heads: 2,
            intermediate_size: intermediate,
            vocab_size: vocab,
            max_context_length: 16,
            norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            head_dim: hidden / 2,
            dim_model_base: hidden,
            embedding_scale: 1.0,
        };

        let token_embd = vec![0.0f32; vocab * hidden];
        let output_weight = vec![0.0f32; vocab * hidden];
        let output_norm = make_rms_norm(hidden);

        let head_dim = hidden / 2;
        let n_kv = 2;
        let layer = MiniCpmLayer {
            attn_norm: make_rms_norm(hidden),
            ffn_norm: make_rms_norm(hidden),
            attn_q: make_zero_quant_tensor(2 * head_dim, hidden),
            attn_k: make_zero_quant_tensor(n_kv * head_dim, hidden),
            attn_v: make_zero_quant_tensor(n_kv * head_dim, hidden),
            attn_out: make_zero_quant_tensor(hidden, 2 * head_dim),
            ffn_gate: make_zero_quant_tensor(intermediate, hidden),
            ffn_up: make_zero_quant_tensor(intermediate, hidden),
            ffn_down: make_zero_quant_tensor(hidden, intermediate),
        };

        MiniCpmForward::new(cfg, token_embd, vec![layer], output_norm, output_weight, 16)
    }

    #[test]
    fn test_forward_returns_correct_vocab_size() {
        let vocab = 32;
        let mut fwd = make_forward(4, 8, vocab);
        let mut cache = NoopKvCache::new();
        let result = fwd.forward(&[0u32], &mut cache);
        // With zero weights, forward will produce a vector of correct size
        // (zero logits is acceptable for stub testing)
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

    #[test]
    fn test_embedding_scale_applied() {
        // With scale != 1.0 the result should still be correct-sized
        let vocab = 16;
        let mut fwd = make_forward(4, 8, vocab);
        fwd.cfg.embedding_scale = 2.0;
        let mut cache = NoopKvCache::new();
        let result = fwd.forward(&[0u32], &mut cache);
        assert!(result.is_ok());
        assert_eq!(result.expect("logits").len(), vocab);
    }
}

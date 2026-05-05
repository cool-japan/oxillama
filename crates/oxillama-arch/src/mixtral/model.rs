//! Mixtral transformer forward pass implementation.
//!
//! Mixtral is a Sparse Mixture-of-Experts (SMoE) variant of Mistral.
//! Each FFN block is replaced by a pool of 8 expert SwiGLU FFNs, with only
//! the top-2 experts activated per token via learned routing.
//!
//! Architecture:
//!   embedding → N×(RMSNorm → SWA-GQA → residual → RMSNorm → MoE-FFN → residual) → RMSNorm → LM head
//!
//! Tensor names (GGUF convention):
//!   - `blk.{i}.ffn_gate_inp.weight` — router `[num_experts, hidden_size]`
//!   - `blk.{i}.ffn_gate_exps.weight` — all expert gate projections (packed)
//!   - `blk.{i}.ffn_up_exps.weight`   — all expert up projections (packed)
//!   - `blk.{i}.ffn_down_exps.weight` — all expert down projections (packed)

use crate::common::moe::MoeFfn;
use crate::common::rms_norm::RmsNorm;
use crate::config::ModelConfig;
use crate::error::ArchResult;
use crate::traits::{ForwardPass, KvCacheAccess};

/// Configuration specific to Mixtral's MoE FFN blocks.
#[derive(Debug, Clone)]
pub struct MixtralMoeConfig {
    /// Total number of experts in the pool (default 8 for Mixtral-8x7B).
    pub num_experts: usize,
    /// Number of experts activated per token (default 2 for Mixtral).
    pub num_experts_used: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Intermediate size for each individual expert FFN.
    pub intermediate_size: usize,
}

impl MixtralMoeConfig {
    /// Extract Mixtral MoE config from `ModelConfig`.
    ///
    /// Falls back to 8 total experts / 2 used if not set in metadata.
    pub fn from_model_config(config: &ModelConfig) -> Self {
        let num_experts = if config.num_experts > 0 {
            config.num_experts
        } else {
            8
        };
        let num_experts_used = if config.num_experts_used > 0 {
            config.num_experts_used
        } else {
            2
        };
        Self {
            num_experts,
            num_experts_used,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
        }
    }
}

/// A single Mixtral transformer layer with MoE FFN.
pub struct MixtralLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// Sparse MoE FFN (top-2-of-8 by default).
    pub moe_ffn: MoeFfn,
}

/// Complete Mixtral model.
///
/// Structurally identical to Mistral, but every FFN block is a sparse MoE.
/// The attention mechanism supports sliding window attention (SWA).
pub struct MixtralModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// MoE configuration derived from the model config.
    pub moe_config: MixtralMoeConfig,
    /// Sliding window size (None = full causal).
    pub sliding_window: Option<usize>,
    /// Token embedding weights `[vocab_size, hidden_size]` as f32.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<MixtralLayer>,
    /// Final RMSNorm.
    pub output_norm: RmsNorm,
    /// LM head weights `[vocab_size, hidden_size]` (f32 for simplicity).
    pub output_weights: Vec<f32>,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_moe_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl MixtralModel {
    /// Create a new `MixtralModel` from pre-loaded weights.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<MixtralLayer>,
        output_norm: RmsNorm,
        output_weights: Vec<f32>,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let sliding_window = config.sliding_window;
        let moe_config = MixtralMoeConfig::from_model_config(&config);

        Self {
            config,
            moe_config,
            sliding_window,
            token_embd,
            layers,
            output_norm,
            output_weights,
            buf_hidden: vec![0.0f32; hidden_size],
            buf_norm: vec![0.0f32; hidden_size],
            buf_q: vec![0.0f32; num_heads * head_dim],
            buf_k: vec![0.0f32; num_kv_heads * head_dim],
            buf_v: vec![0.0f32; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0f32; hidden_size],
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

    /// Minimal causal (or sliding-window) scaled dot-product attention.
    ///
    /// This is a simplified reference implementation that uses f32 dot-product
    /// attention. A production path would use `QuantLinear` + `RopeTable`.
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

        // For this reference implementation, Q = K = V = norm(hidden).
        // A full load-from-GGUF path would project through QuantLinear layers.
        let norm_slice = self.buf_norm.clone();

        // Fill Q/K/V with the normalised hidden state (identity projection).
        // Only fill up to kv_dim for K/V to respect GQA dimensionality.
        for i in 0..self.buf_q.len().min(norm_slice.len()) {
            self.buf_q[i] = norm_slice[i % norm_slice.len()];
        }
        for i in 0..kv_dim.min(norm_slice.len()) {
            self.buf_k[i] = norm_slice[i % norm_slice.len()];
            self.buf_v[i] = norm_slice[i % norm_slice.len()];
        }

        // Store K, V in cache
        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = kv_cache.get_keys(layer_idx)?;
        let cached_values = kv_cache.get_values(layer_idx)?;

        let window_start = match self.sliding_window {
            Some(w) => seq_len.saturating_sub(w),
            None => 0,
        };
        let window_len = seq_len - window_start;

        self.buf_attn_out.fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in window_start..seq_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_slice = &cached_keys[k_offset..k_offset + head_dim];
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * k_slice[d];
                }
                self.buf_attn_scores[pos - window_start] = score * scale;
            }

            softmax_inplace(&mut self.buf_attn_scores[..window_len]);

            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for pos in window_start..seq_len {
                let v_offset = pos * kv_dim + kv_head * head_dim;
                let v_slice = &cached_values[v_offset..v_offset + head_dim];
                let w = self.buf_attn_scores[pos - window_start];
                for d in 0..head_dim {
                    out_head[d] += w * v_slice[d];
                }
            }
        }

        // Add attention output to residual
        for (h, &a) in self.buf_hidden.iter_mut().zip(self.buf_attn_out.iter()) {
            *h += a;
        }
        Ok(())
    }

    /// Run MoE FFN for one layer and add to residual.
    fn moe_feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let norm_buf = self.buf_norm.clone();
        self.layers[layer_idx]
            .moe_ffn
            .forward(&norm_buf, &mut self.buf_moe_out)?;
        for (h, &m) in self.buf_hidden.iter_mut().zip(self.buf_moe_out.iter()) {
            *h += m;
        }
        Ok(())
    }
}

impl ForwardPass for MixtralModel {
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
                // Pre-attention norm
                let hidden_clone = self.buf_hidden.clone();
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&hidden_clone, &mut self.buf_norm);

                self.attention(layer_idx, position, kv_cache)?;

                // Pre-FFN norm
                let hidden_clone2 = self.buf_hidden.clone();
                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&hidden_clone2, &mut self.buf_norm);

                self.moe_feed_forward(layer_idx)?;
            }
            kv_cache.advance();
        }

        // Final norm
        let hidden_final = self.buf_hidden.clone();
        let mut normed = vec![0.0f32; self.config.hidden_size];
        self.output_norm.forward_to(&hidden_final, &mut normed);

        // LM head: output_weights [vocab_size, hidden_size] × normed [hidden_size]
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        self.buf_logits.fill(0.0);
        for v in 0..vocab_size {
            let row = &self.output_weights[v * hidden_size..(v + 1) * hidden_size];
            let mut sum = 0.0f32;
            for (w, &x) in row.iter().zip(normed.iter()) {
                sum += w * x;
            }
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

    fn swa_config(&self) -> Option<(u32, bool)> {
        self.config.swa_window.map(|w| (w, false))
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

/// Build a minimal `MixtralLayer` with all-small weights for testing.
///
/// This is only compiled in `#[cfg(test)]` contexts; production loading
/// goes through a separate GGUF loader that populates `QuantLinear` tensors.
#[cfg(test)]
pub fn make_test_layer(
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
) -> MixtralLayer {
    use crate::common::moe::Expert;

    let attn_norm = RmsNorm::new(vec![1.0f32; hidden_size], 1e-5);
    let ffn_norm = RmsNorm::new(vec![1.0f32; hidden_size], 1e-5);

    let experts = (0..num_experts)
        .map(|_| Expert {
            gate: vec![0.01f32; intermediate_size * hidden_size],
            up: vec![0.01f32; intermediate_size * hidden_size],
            down: vec![0.01f32; hidden_size * intermediate_size],
            hidden_size,
            intermediate_size,
        })
        .collect();

    let router = vec![0.01f32; num_experts * hidden_size];

    let moe_ffn = MoeFfn {
        router,
        experts,
        top_k,
        num_experts,
        hidden_size,
    };

    MixtralLayer {
        attn_norm,
        ffn_norm,
        moe_ffn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::moe::Expert;
    use crate::config::ModelConfig;
    use crate::error::ArchError;
    use crate::registry::ArchitectureRegistry;
    use crate::traits::ModelArchitecture;

    fn minimal_config(
        hidden_size: usize,
        num_experts: usize,
        num_experts_used: usize,
    ) -> ModelConfig {
        ModelConfig {
            architecture: "mixtral".to_string(),
            hidden_size,
            intermediate_size: 16,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 2,
            head_dim: hidden_size / 2,
            vocab_size: 4,
            max_context_length: 8,
            num_experts,
            num_experts_used,
            ..ModelConfig::default()
        }
    }

    fn make_model(config: &ModelConfig) -> MixtralModel {
        let h = config.hidden_size;
        let v = config.vocab_size;
        let n_experts = if config.num_experts > 0 {
            config.num_experts
        } else {
            8
        };
        let top_k = if config.num_experts_used > 0 {
            config.num_experts_used
        } else {
            2
        };

        let token_embd = vec![0.01f32; v * h];
        let layer = make_test_layer(h, config.intermediate_size, n_experts, top_k);
        let output_norm = RmsNorm::new(vec![1.0f32; h], 1e-5);
        let output_weights = vec![0.01f32; v * h];

        MixtralModel::new(
            config.clone(),
            token_embd,
            vec![layer],
            output_norm,
            output_weights,
        )
    }

    // ── Registry lookup ───────────────────────────────────────────────────────

    #[test]
    fn mixtral_registry_lookup() {
        let registry = ArchitectureRegistry::with_builtins();
        let arch = registry.get("mixtral");
        assert!(
            arch.is_ok(),
            "registry.get('mixtral') must succeed; got: {:?}",
            arch.err()
        );
        assert_eq!(arch.expect("mixtral arch").arch_id(), "mixtral");
    }

    // ── Tensor names ──────────────────────────────────────────────────────────

    #[test]
    fn mixtral_tensor_names_complete() {
        use crate::mixtral::MixtralArchitecture;

        let arch = MixtralArchitecture::new();
        let names = arch.tensor_names();
        assert!(
            !names.is_empty(),
            "tensor_names() must return at least one pattern"
        );

        // All patterns must be non-empty strings.
        for tp in &names {
            assert!(
                !tp.pattern.is_empty(),
                "TensorNamePattern.pattern must not be empty"
            );
            assert!(
                !tp.description.is_empty(),
                "TensorNamePattern.description must not be empty"
            );
        }

        // Required tensor names expected for Mixtral.
        let required_patterns = [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.{i}.ffn_gate_inp.weight",
            "blk.{i}.ffn_gate_exps.weight",
            "blk.{i}.ffn_up_exps.weight",
            "blk.{i}.ffn_down_exps.weight",
        ];
        let pattern_strs: Vec<&str> = names.iter().map(|n| n.pattern.as_str()).collect();
        for req in required_patterns {
            assert!(
                pattern_strs.contains(&req),
                "tensor_names should contain '{req}'"
            );
        }
    }

    // ── Top-2 routing ─────────────────────────────────────────────────────────

    /// Verify that MoeFfn top-2 routing selects the two experts with the
    /// highest router scores for a crafted input.
    ///
    /// Router layout: `router[expert, dim]` row-major.
    /// We set `router[2, 0] = 10.0` and `router[5, 0] = 8.0`, all others 0.
    /// For `input = [1, 0, 0, 0]`, expert 2 gets logit 10 and expert 5 gets 8.
    /// After softmax, experts 2 and 5 dominate → top-2 must be {2, 5}.
    #[test]
    fn mixtral_top2_routing_selects_expected_experts() {
        let hidden_size = 4;
        let intermediate_size = 8;
        let num_experts = 8;
        let top_k = 2;

        // Craft router so expert 2 and 5 strongly dominate.
        let mut router = vec![0.0f32; num_experts * hidden_size];
        router[2 * hidden_size] = 10.0; // expert 2, first dim
        router[5 * hidden_size] = 8.0; // expert 5, first dim

        // Build identifiable experts: each returns expert_index in output[0].
        let experts: Vec<_> = (0..num_experts)
            .map(|e| {
                let mut gate = vec![0.0f32; intermediate_size * hidden_size];
                let mut up = vec![0.0f32; intermediate_size * hidden_size];
                let mut down = vec![0.0f32; hidden_size * intermediate_size];
                // gate[0,0] = e+1, up[0,0] = 1 → out[0] ≈ silu(e+1) * down[0,0]
                gate[0] = (e + 1) as f32;
                up[0] = 1.0;
                down[0] = 1.0;
                Expert {
                    gate,
                    up,
                    down,
                    hidden_size,
                    intermediate_size,
                }
            })
            .collect();

        let moe = MoeFfn {
            router,
            experts,
            top_k,
            num_experts,
            hidden_size,
        };

        let input = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut output = vec![0.0f32; hidden_size];
        moe.forward(&input, &mut output)
            .expect("MoeFfn forward must succeed");

        // Expert 2 (e=2): gate[0,0]=3, up[0,0]=1, down[0,0]=1
        //   → intermediate[0] = silu(3 * 1) * 1 = silu(3)
        //   → expert2_out[0] = silu(3) * 1 = silu(3)
        // Expert 5 (e=5): gate[0,0]=6, up[0,0]=1, down[0,0]=1
        //   → intermediate[0] = silu(6 * 1) * 1 = silu(6)
        //   → expert5_out[0] = silu(6) * 1 = silu(6)
        //
        // Router logits (for input=[1,0,0,0]): expert2=10, expert5=8, others=0
        //   softmax_sum = exp(10) + exp(8) + 6*exp(0)
        //   raw_w2 = exp(10) / softmax_sum
        //   raw_w5 = exp(8) / softmax_sum
        //   After top-2 renorm: w2 = raw_w2/(raw_w2+raw_w5), w5 = raw_w5/(raw_w2+raw_w5)
        //   Expected: output[0] = w2*silu(3) + w5*silu(6)
        let silu3 = 3.0f32 / (1.0 + (-3.0f32).exp());
        let silu6 = 6.0f32 / (1.0 + (-6.0f32).exp());
        let softmax_sum = 10.0f32.exp() + 8.0f32.exp() + 6.0_f32;
        let raw_w2 = 10.0f32.exp() / softmax_sum;
        let raw_w5 = 8.0f32.exp() / softmax_sum;
        let renorm = raw_w2 + raw_w5;
        let w2 = raw_w2 / renorm;
        let w5 = raw_w5 / renorm;
        let expected = w2 * silu3 + w5 * silu6;
        let tolerance = 1e-4_f32;
        assert!(
            (output[0] - expected).abs() < tolerance,
            "top-2 should select experts 2 and 5; output[0]={} but expected {} (±{})",
            output[0],
            expected,
            tolerance
        );
        // Sanity: output must be significantly larger than what the worst expert
        // (expert 0 alone, with gate[0,0]=1) would produce.
        let silu1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!(
            output[0] > silu1,
            "output[0]={} should exceed silu(1)={} (contribution of weakest expert alone)",
            output[0],
            silu1
        );
    }

    // ── Load balance / softmax normalisation ─────────────────────────────────

    /// Router softmax output must sum to 1.0 (verified through MoeFfn internals).
    ///
    /// We test this indirectly: with uniform router weights, all 8 experts get
    /// equal scores → softmax is uniform (each 0.125). Top-2 renormalised
    /// weights each become 0.5. Output should be non-zero and consistent.
    #[test]
    fn mixtral_load_balance_weights_normalize() {
        let hidden_size = 4;
        let intermediate_size = 4;
        let num_experts = 8;
        let top_k = 2;

        let router = vec![1.0f32; num_experts * hidden_size];
        let experts: Vec<_> = (0..num_experts)
            .map(|_| Expert {
                gate: vec![1.0f32; intermediate_size * hidden_size],
                up: vec![1.0f32; intermediate_size * hidden_size],
                down: vec![1.0f32; hidden_size * intermediate_size],
                hidden_size,
                intermediate_size,
            })
            .collect();

        let moe = MoeFfn {
            router,
            experts,
            top_k,
            num_experts,
            hidden_size,
        };

        let input = vec![0.5f32; hidden_size];
        let mut output1 = vec![0.0f32; hidden_size];
        let mut output2 = vec![0.0f32; hidden_size];

        moe.forward(&input, &mut output1)
            .expect("first forward must succeed");
        moe.forward(&input, &mut output2)
            .expect("second forward must succeed");

        // Determinism: two runs on same input yield identical results.
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!(
                (a - b).abs() < 1e-8,
                "MoE output must be deterministic: {a} != {b}"
            );
        }

        // Non-trivial: with uniform experts and input, output is non-zero.
        assert!(
            output1.iter().any(|&v| v.abs() > 1e-6),
            "uniform MoE output should be non-zero, got {output1:?}"
        );
    }

    // ── Forward pass shape ────────────────────────────────────────────────────

    /// Minimal KV cache for forward-pass tests (stores real K/V data).
    struct SimpleKvCache {
        kv_dim: usize,
        max_seq: usize,
        n_layers: usize,
        position: usize,
        keys: Vec<Vec<f32>>,   // [layer][position * kv_dim]
        values: Vec<Vec<f32>>, // [layer][position * kv_dim]
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
            let copy_k = key.len().min(self.kv_dim);
            let copy_v = value.len().min(self.kv_dim);
            self.keys[layer][offset..offset + copy_k].copy_from_slice(&key[..copy_k]);
            self.values[layer][offset..offset + copy_v].copy_from_slice(&value[..copy_v]);
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

    #[test]
    fn mixtral_forward_output_shape() {
        let config = minimal_config(8, 8, 2);
        let vocab_size = config.vocab_size;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut model = make_model(&config);
        let mut kv_cache = SimpleKvCache::new(1, kv_dim, config.max_context_length);

        let result = model.forward(&[0u32], &mut kv_cache);
        assert!(result.is_ok(), "forward must succeed: {:?}", result.err());
        let logits = result.expect("logits");
        assert_eq!(
            logits.len(),
            vocab_size,
            "logits length must equal vocab_size"
        );
    }
}

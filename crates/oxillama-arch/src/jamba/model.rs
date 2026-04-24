//! Jamba hybrid (LLaMA × Mamba-2) model.
//!
//! Jamba interleaves standard LLaMA-style attention+FFN blocks with Mamba-2
//! SSM blocks.  Each layer type has its own per-sequence state:
//! - **Attention layers** carry an [`AttentionSequenceState`] (position counter).
//! - **SSM layers** carry an [`SsmLayerState`] (recurrent hidden vectors).
//!
//! The composite [`JambaSequenceState`] owns one entry per model layer.
//!
//! ## Forward pass
//!
//! Token by token:
//! 1. Embedding lookup.
//! 2. For each layer, dispatch to either `attention_forward` or `ssm_forward`.
//! 3. Final RMSNorm + LM head.
//!
//! Both layer type implementations are simplified stubs in this initial version —
//! they perform the correct linear algebra but do not use quantized weights.
//! Full GGUF loading (`load_jamba_from_gguf`) can be wired later.
//!
//! ## SSM state lifecycle
//!
//! **Important**: The `forward()` method on `JambaModel` is the *stateless*
//! entry-point used by the runtime when it manages per-request state externally
//! through [`SsmStatePool`](crate::common::sequence_state::SsmLayerState).
//! When called through the `ForwardPass` trait directly (e.g. by tests or the
//! reference path), it allocates a **throwaway** `SsmLayerState` per token and
//! per SSM layer.  This means SSM recurrent state is **not** carried across
//! token steps in the raw `forward()` path — only the runtime's
//! `allocate_sequence_state()` / `SequencePool` round-trip preserves state
//! between tokens.

use super::config::{JambaConfig, LayerKind};
use crate::common::rms_norm::RmsNorm;
use crate::common::sequence_state::{
    AttentionSequenceState, SequenceState, SequenceStateSnapshot, SsmLayerState,
};
use crate::error::{ArchError, ArchResult};
use crate::lora::LoraStack;
use crate::traits::{ForwardPass, KvCacheAccess};

// ─── Per-layer state ──────────────────────────────────────────────────────────

/// Per-layer state for one sequence slot in a Jamba model.
pub enum JambaLayerState {
    /// Attention layer: position counter only.
    Attention(AttentionSequenceState),
    /// SSM layer: recurrent hidden state.
    Ssm(SsmLayerState),
}

/// Full sequence state for a Jamba model (one per in-flight request).
///
/// Each entry in `layers` matches the corresponding layer kind in `JambaConfig`.
pub struct JambaSequenceState {
    /// Per-layer state (one entry per model layer).
    pub layers: Vec<JambaLayerState>,
    /// Current position (shared; advanced once per token).
    position: usize,
    /// Maximum capacity.
    max_capacity: usize,
}

impl JambaSequenceState {
    /// Construct a new `JambaSequenceState` from the model configuration.
    pub fn new(config: &JambaConfig, max_capacity: usize) -> Self {
        let layers: Vec<JambaLayerState> = config
            .layer_pattern()
            .into_iter()
            .map(|kind| match kind {
                LayerKind::Attention => {
                    JambaLayerState::Attention(AttentionSequenceState::new(max_capacity))
                }
                LayerKind::Ssm => {
                    JambaLayerState::Ssm(SsmLayerState::new(config.d_state, config.d_inner))
                }
            })
            .collect();

        Self {
            layers,
            position: 0,
            max_capacity,
        }
    }
}

impl SequenceState for JambaSequenceState {
    fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            match layer {
                JambaLayerState::Attention(s) => s.reset(),
                JambaLayerState::Ssm(s) => s.clear(),
            }
        }
    }

    fn step_position(&self) -> usize {
        self.position
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn capacity(&self) -> usize {
        self.max_capacity
    }

    fn snapshot_payload(&self) -> SequenceStateSnapshot {
        // For each layer: attention layers get an empty vec, SSM layers get their h.
        let ssm_states = self
            .layers
            .iter()
            .map(|l| match l {
                JambaLayerState::Attention(_) => Vec::new(),
                JambaLayerState::Ssm(s) => s.h.clone(),
            })
            .collect();
        SequenceStateSnapshot::Jamba {
            attention_position: self.position,
            ssm_states,
        }
    }

    fn restore_from_snapshot_payload(&mut self, snap: &SequenceStateSnapshot) {
        if let SequenceStateSnapshot::Jamba {
            attention_position,
            ssm_states,
        } = snap
        {
            self.position = *attention_position;
            for (layer, state) in self.layers.iter_mut().zip(ssm_states.iter()) {
                if let JambaLayerState::Ssm(s) = layer {
                    let copy_len = state.len().min(s.h.len());
                    s.h[..copy_len].copy_from_slice(&state[..copy_len]);
                }
            }
        }
    }
}

// ─── Layer weights ────────────────────────────────────────────────────────────

/// Weights for one Jamba attention block.
pub struct JambaAttentionLayerWeights {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Query projection `[hidden × hidden]` row-major.
    pub w_q: Vec<f32>,
    /// Key projection `[kv_dim × hidden]` row-major (kv_dim = num_kv_heads * head_dim).
    pub w_k: Vec<f32>,
    /// Value projection `[kv_dim × hidden]` row-major.
    pub w_v: Vec<f32>,
    /// Output projection `[hidden × hidden]` row-major.
    pub w_o: Vec<f32>,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN gate projection `[ffn_dim × hidden]` row-major (SwiGLU).
    pub w_gate: Vec<f32>,
    /// FFN up projection `[ffn_dim × hidden]` row-major.
    pub w_up: Vec<f32>,
    /// FFN down projection `[hidden × ffn_dim]` row-major.
    pub w_down: Vec<f32>,
}

/// Weights for one Jamba SSM block (Mamba-2 style).
pub struct JambaSsmLayerWeights {
    /// Pre-SSM RMSNorm.
    pub ssm_norm: RmsNorm,
    /// Combined gate + input projection `[2 * d_inner, hidden]` row-major.
    pub w_in_z: Vec<f32>,
    /// 1-D depthwise conv kernel `[d_inner × d_conv]` row-major.
    pub w_conv: Vec<f32>,
    /// Conv bias `[d_inner]`.
    pub b_conv: Vec<f32>,
    /// x → B projection `[d_state, d_inner]` row-major.
    pub w_b: Vec<f32>,
    /// x → C projection `[d_state, d_inner]` row-major.
    pub w_c: Vec<f32>,
    /// x → Δ projection `[d_inner, d_inner]` row-major.
    pub w_delta: Vec<f32>,
    /// Δ bias `[d_inner]`.
    pub b_delta: Vec<f32>,
    /// Log-parameterised A `[d_state × d_inner]` row-major.
    pub log_a: Vec<f32>,
    /// Skip-connection D `[d_inner]`.
    pub d_skip: Vec<f32>,
    /// Output projection `[hidden, d_inner]` row-major.
    pub w_out: Vec<f32>,
}

/// Per-layer weight storage for a Jamba model.
pub enum JambaLayerWeights {
    /// Attention block.
    Attention(JambaAttentionLayerWeights),
    /// SSM block.
    Ssm(JambaSsmLayerWeights),
}

// ─── Full model ────────────────────────────────────────────────────────────────

/// Complete Jamba model.
pub struct JambaModel {
    /// Model configuration.
    pub config: JambaConfig,
    /// Token embedding table `[vocab × hidden]` stored as f32.
    pub token_embd: Vec<f32>,
    /// Per-layer weights.
    pub layers: Vec<JambaLayerWeights>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head projection `[vocab × hidden]` stored as f32.
    pub lm_head: Vec<f32>,
    /// Optional LoRA adapter stack for this model instance.
    ///
    /// Set via [`ForwardPass::with_lora_stack`].  The stack's adapters must
    /// have `in_dim` and `out_dim` compatible with `config.hidden_size`.
    pub lora_stack: Option<LoraStack>,
}

impl JambaModel {
    /// Construct a `JambaModel`.
    pub fn new(
        config: JambaConfig,
        token_embd: Vec<f32>,
        layers: Vec<JambaLayerWeights>,
        output_norm: RmsNorm,
        lm_head: Vec<f32>,
    ) -> Self {
        Self {
            config,
            token_embd,
            layers,
            output_norm,
            lm_head,
            lora_stack: None,
        }
    }

    // ── Attention-layer forward ───────────────────────────────────────────────

    /// Run one attention block for a single token.
    ///
    /// Simplified dot-product attention without masking or RoPE — for
    /// correctness testing and architecture wiring.
    fn attention_forward(
        hidden: &[f32],
        w: &JambaAttentionLayerWeights,
        _kv_cache: &mut dyn KvCacheAccess,
        hidden_size: usize,
    ) -> ArchResult<Vec<f32>> {
        // --- Pre-norm ----------------------------------------------------------
        let mut normed = hidden.to_vec();
        w.attn_norm.forward(&mut normed);

        // --- Single-token attention (no masking needed) ------------------------
        // Q = w_q @ normed  [hidden]
        let q = gemv(&w.w_q, &normed, hidden_size, hidden_size);
        // K = w_k @ normed  [hidden] (kv_dim may equal hidden for simplicity)
        let k = gemv(&w.w_k, &normed, hidden_size, hidden_size);
        // V = w_v @ normed  [hidden]
        let v = gemv(&w.w_v, &normed, hidden_size, hidden_size);

        // Attention score: scalar (single token — no KV sequence)
        let scale = 1.0_f32 / (hidden_size as f32).sqrt().max(1.0);
        let score: f32 = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum::<f32>() * scale;

        // Single-element softmax (trivially 1.0 for a single token)
        let attended: Vec<f32> = v.iter().map(|vi| vi * score.tanh()).collect();

        // O projection.
        let attn_out = gemv(&w.w_o, &attended, hidden_size, hidden_size);

        // Residual.
        let after_attn: Vec<f32> = hidden
            .iter()
            .zip(attn_out.iter())
            .map(|(h, a)| h + a)
            .collect();

        // --- FFN (SwiGLU) ------------------------------------------------------
        let mut normed_ffn = after_attn.clone();
        w.ffn_norm.forward(&mut normed_ffn);

        let ffn_dim = w
            .w_gate
            .len()
            .checked_div(hidden_size)
            .unwrap_or(hidden_size);
        let gate = gemv(&w.w_gate, &normed_ffn, ffn_dim, hidden_size);
        let up = gemv(&w.w_up, &normed_ffn, ffn_dim, hidden_size);

        // SwiGLU: gate_silu * up
        let swiglu: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| silu(*g) * u)
            .collect();

        let ffn_out = gemv(&w.w_down, &swiglu, hidden_size, ffn_dim);

        let output: Vec<f32> = after_attn
            .iter()
            .zip(ffn_out.iter())
            .map(|(a, f)| a + f)
            .collect();

        Ok(output)
    }

    // ── SSM-layer forward ─────────────────────────────────────────────────────

    /// Run one SSM block for a single token, updating `layer_state`.
    fn ssm_forward(
        hidden: &[f32],
        w: &JambaSsmLayerWeights,
        layer_state: &mut SsmLayerState,
        hidden_size: usize,
    ) -> ArchResult<Vec<f32>> {
        use crate::mamba2::conv::conv1d_depthwise;
        use crate::mamba2::ssm::selective_scan_sequential;

        let d_inner = layer_state.d_inner;
        let d_state = layer_state.d_state;
        let d_conv = w.w_conv.len().checked_div(d_inner).unwrap_or(4);

        // Pre-norm.
        let mut normed = hidden.to_vec();
        w.ssm_norm.forward(&mut normed);

        // Gate + input projection.
        let z_and_y = gemv(&w.w_in_z, &normed, 2 * d_inner, hidden_size);
        let z_vec: Vec<f32> = z_and_y[..d_inner].to_vec();
        let y_in: Vec<f32> = z_and_y[d_inner..].to_vec();

        // Depthwise conv1d + SiLU.
        let y_conv = conv1d_depthwise(&y_in, &w.w_conv, &w.b_conv, 1, d_inner, d_conv);
        let y = &y_conv[..d_inner];

        // B, C, Δ.
        let b_vec = gemv(&w.w_b, y, d_state, d_inner);
        let c_vec = gemv(&w.w_c, y, d_state, d_inner);
        let d_raw = gemv(&w.w_delta, y, d_inner, d_inner);
        let delta: Vec<f32> = d_raw
            .iter()
            .zip(w.b_delta.iter())
            .map(|(d, b)| softplus(d + b))
            .collect();

        // Selective scan (single-step, seq_len=1).
        let ssm_out = selective_scan_sequential(
            y,
            &delta,
            &w.log_a,
            &b_vec,
            &c_vec,
            &w.d_skip,
            1, // seq_len = 1 (token by token)
            d_inner,
            d_state,
            layer_state,
        );

        // Gating: silu(z) * ssm_out.
        let gated: Vec<f32> = z_vec
            .iter()
            .zip(ssm_out.iter())
            .map(|(z, s)| silu(*z) * s)
            .collect();

        // Output projection.
        let block_out = gemv(&w.w_out, &gated, hidden_size, d_inner);

        // Residual.
        let output: Vec<f32> = hidden
            .iter()
            .zip(block_out.iter())
            .map(|(h, b)| h + b)
            .collect();

        Ok(output)
    }
}

// ─── ForwardPass impl ─────────────────────────────────────────────────────────

impl ForwardPass for JambaModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let seq_len = tokens.len();

        if seq_len == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "Jamba forward: empty token sequence".to_string(),
            });
        }

        let mut logits = vec![0.0f32; vocab];

        for &tok_id in tokens {
            let tok = tok_id as usize;
            if tok >= vocab {
                return Err(ArchError::InvalidConfig {
                    detail: format!("Jamba: token {tok} >= vocab_size {vocab}"),
                });
            }

            // Embedding lookup.
            let emb_off = tok * hidden_size;
            let mut hidden: Vec<f32> = self.token_embd[emb_off..emb_off + hidden_size].to_vec();

            // Per-layer dispatch.
            // We need both the layer weights and (for SSM layers) the ssm state.
            // We hold a single-step per-sequence state in a temporary JambaSequenceState
            // built here, since JambaModel owns the weights but not per-request state.
            // The runtime manages per-request states via SequencePool.
            //
            // For the `forward()` stub path (no external state), we allocate a
            // throwaway per-layer state. In production the runtime would pass state in.
            let mut temp_ssm_states: Vec<SsmLayerState> = self
                .layers
                .iter()
                .map(|layer| match layer {
                    JambaLayerWeights::Ssm(w) => {
                        let d_inner = w.w_out.len() / hidden_size.max(1);
                        let d_state = w.w_b.len() / d_inner.max(1);
                        SsmLayerState::new(d_state, d_inner)
                    }
                    JambaLayerWeights::Attention(_) => SsmLayerState::new(0, 0),
                })
                .collect();

            for (layer_idx, layer_w) in self.layers.iter().enumerate() {
                match layer_w {
                    JambaLayerWeights::Attention(w) => {
                        hidden = Self::attention_forward(&hidden, w, kv_cache, hidden_size)
                            .map_err(|e| ArchError::ForwardPassError {
                                layer: layer_idx,
                                message: format!("attention block: {e}"),
                            })?;
                    }
                    JambaLayerWeights::Ssm(w) => {
                        let ssm_state = &mut temp_ssm_states[layer_idx];
                        hidden =
                            Self::ssm_forward(&hidden, w, ssm_state, hidden_size).map_err(|e| {
                                ArchError::ForwardPassError {
                                    layer: layer_idx,
                                    message: format!("SSM block: {e}"),
                                }
                            })?;
                    }
                }
            }

            // Final norm + LM head.
            self.output_norm.forward(&mut hidden);
            for (v, lv) in logits.iter_mut().enumerate() {
                let row = &self.lm_head[v * hidden_size..(v + 1) * hidden_size];
                *lv = row.iter().zip(hidden.iter()).map(|(w, h)| w * h).sum();
            }
        }

        Ok(logits)
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

    fn allocate_sequence_state(
        &self,
        max_context_length: usize,
    ) -> Box<dyn crate::common::sequence_state::SequenceState> {
        Box::new(JambaSequenceState::new(&self.config, max_context_length))
    }

    /// Validate and attach a LoRA adapter stack to this model instance.
    ///
    /// # Validation
    ///
    /// Every adapter in `stack` that exposes a [`LoraDelta`] must have
    /// `in_dim == hidden_size` and `out_dim == hidden_size`.  If any delta
    /// in any adapter has mismatched dimensions this method returns
    /// [`ArchError::LoraIncompatible`].
    ///
    /// [`LoraDelta`]: crate::lora::LoraDelta
    fn with_lora_stack(&mut self, stack: LoraStack) -> ArchResult<()> {
        use crate::lora::TargetModule;

        let hidden = self.config.hidden_size;

        // Check every adapter for dimension compatibility.
        // We probe the attention target modules for layer 0 as a quick sanity
        // check — adapters targeting other layers with different dimensions
        // would also fail here if they cover layer 0.
        let probe_targets = [
            TargetModule::QueryProj,
            TargetModule::KeyProj,
            TargetModule::ValueProj,
            TargetModule::OutputProj,
        ];

        for adapter in stack.adapters() {
            for &target in &probe_targets {
                if let Some(delta) = adapter.delta(target, 0) {
                    if delta.in_dim != hidden {
                        return Err(ArchError::LoraIncompatible {
                            detail: format!(
                                "adapter in_dim {} != model hidden_size {} for {target}",
                                delta.in_dim, hidden
                            ),
                        });
                    }
                    if delta.out_dim != hidden {
                        return Err(ArchError::LoraIncompatible {
                            detail: format!(
                                "adapter out_dim {} != model hidden_size {} for {target}",
                                delta.out_dim, hidden
                            ),
                        });
                    }
                }
            }
        }

        self.lora_stack = Some(stack);
        Ok(())
    }
}

// ─── Builder helpers ──────────────────────────────────────────────────────────

/// Create a zero-weight `JambaAttentionLayerWeights` for the given dimensions.
pub fn make_zero_jamba_attn_layer(
    hidden_size: usize,
    ffn_dim: usize,
) -> JambaAttentionLayerWeights {
    JambaAttentionLayerWeights {
        attn_norm: RmsNorm::new(vec![1.0f32; hidden_size], 1e-5),
        w_q: vec![0.0f32; hidden_size * hidden_size],
        w_k: vec![0.0f32; hidden_size * hidden_size],
        w_v: vec![0.0f32; hidden_size * hidden_size],
        w_o: vec![0.0f32; hidden_size * hidden_size],
        ffn_norm: RmsNorm::new(vec![1.0f32; hidden_size], 1e-5),
        w_gate: vec![0.0f32; ffn_dim * hidden_size],
        w_up: vec![0.0f32; ffn_dim * hidden_size],
        w_down: vec![0.0f32; hidden_size * ffn_dim],
    }
}

/// Create a zero-weight `JambaSsmLayerWeights` for the given dimensions.
pub fn make_zero_jamba_ssm_layer(
    hidden_size: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
) -> JambaSsmLayerWeights {
    JambaSsmLayerWeights {
        ssm_norm: RmsNorm::new(vec![1.0f32; hidden_size], 1e-5),
        w_in_z: vec![0.0f32; 2 * d_inner * hidden_size],
        w_conv: vec![0.0f32; d_inner * d_conv],
        b_conv: vec![0.0f32; d_inner],
        w_b: vec![0.0f32; d_state * d_inner],
        w_c: vec![0.0f32; d_state * d_inner],
        w_delta: vec![0.0f32; d_inner * d_inner],
        b_delta: vec![0.0f32; d_inner],
        log_a: vec![0.0f32; d_state * d_inner],
        d_skip: vec![0.0f32; d_inner],
        w_out: vec![0.0f32; hidden_size * d_inner],
    }
}

/// Build a complete `JambaModel` with zero weights for unit testing.
pub fn build_zero_jamba_model(config: JambaConfig) -> JambaModel {
    let hidden_size = config.hidden_size;
    let ffn_dim = config.intermediate_size;
    let d_inner = config.d_inner;
    let d_state = config.d_state;
    let d_conv = config.d_conv;
    let vocab = config.vocab_size;

    let layers: Vec<JambaLayerWeights> = config
        .layer_pattern()
        .into_iter()
        .map(|kind| match kind {
            LayerKind::Attention => {
                JambaLayerWeights::Attention(make_zero_jamba_attn_layer(hidden_size, ffn_dim))
            }
            LayerKind::Ssm => JambaLayerWeights::Ssm(make_zero_jamba_ssm_layer(
                hidden_size,
                d_inner,
                d_state,
                d_conv,
            )),
        })
        .collect();

    let token_embd = vec![0.0f32; vocab * hidden_size];
    let output_norm = RmsNorm::new(vec![1.0f32; hidden_size], 1e-5);
    let lm_head = vec![0.0f32; vocab * hidden_size];

    JambaModel::new(config, token_embd, layers, output_norm, lm_head)
}

// ─── Stubs for full GGUF loading ──────────────────────────────────────────────

/// Load a Jamba model from a parsed GGUF file.
///
/// This is a stub that returns an error — full loading is wired when
/// the GGUF tensor-naming convention for Jamba is confirmed.
pub fn load_jamba_from_gguf(_model: &oxillama_gguf::GgufModel) -> ArchResult<JambaModel> {
    Err(ArchError::MissingTensor {
        name: "load_jamba_from_gguf: full loader not yet implemented; \
               use build_zero_jamba_model() for testing"
            .to_string(),
    })
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/// Row-major GEMV: `y = A @ x`, `A: [out × in]`, `x: [in]`, `y: [out]`.
fn gemv(a: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for (o, y_o) in y.iter_mut().enumerate() {
        let row_start = o * in_dim;
        let row_end = (row_start + in_dim).min(a.len());
        *y_o = a[row_start..row_end]
            .iter()
            .zip(x.iter().take(in_dim))
            .map(|(a_val, x_val)| a_val * x_val)
            .sum();
    }
    y
}

/// SiLU (swish) activation: `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: `log(1 + exp(x))`.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ArchResult;
    use crate::traits::KvCacheAccess;

    struct NullKv;
    impl KvCacheAccess for NullKv {
        fn seq_len(&self) -> usize {
            0
        }
        fn store_kv(&mut self, _: usize, _: &[f32], _: &[f32]) -> ArchResult<()> {
            Ok(())
        }
        fn get_keys(&self, _: usize) -> ArchResult<&[f32]> {
            Ok(&[])
        }
        fn get_values(&self, _: usize) -> ArchResult<&[f32]> {
            Ok(&[])
        }
        fn advance(&mut self) {}
    }

    fn tiny_config() -> JambaConfig {
        JambaConfig {
            n_layers: 4,
            attn_layer_period: 2,
            attn_layer_offset: 0,
            expert_count: 0,
            expert_top_k: 1,
            d_state: 4,
            d_inner: 8,
            d_conv: 2,
            hidden_size: 8,
            num_attention_heads: 2,
            num_kv_heads: 2,
            intermediate_size: 16,
            vocab_size: 32,
            max_context_length: 64,
            rms_norm_eps: 1e-5,
        }
    }

    /// `forward()` on an empty token sequence returns an error.
    #[test]
    fn jamba_forward_empty_tokens_errors() {
        let cfg = tiny_config();
        let mut model = build_zero_jamba_model(cfg);
        let mut kv = NullKv;
        assert!(
            model.forward(&[], &mut kv).is_err(),
            "empty tokens must return an error"
        );
    }

    /// `forward()` returns logits of the right size.
    #[test]
    fn jamba_forward_shape_correct() {
        let cfg = tiny_config();
        let vocab = cfg.vocab_size;
        let mut model = build_zero_jamba_model(cfg);
        let mut kv = NullKv;
        let logits = model
            .forward(&[1u32], &mut kv)
            .expect("forward must succeed");
        assert_eq!(logits.len(), vocab, "logits must have vocab_size elements");
    }

    /// All logits from a zero-weight model must be finite.
    #[test]
    fn jamba_forward_logits_finite() {
        let cfg = tiny_config();
        let mut model = build_zero_jamba_model(cfg);
        let mut kv = NullKv;
        let logits = model
            .forward(&[0u32], &mut kv)
            .expect("forward must succeed");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite"
        );
    }

    /// `vocab_size()`, `hidden_size()`, and `max_context_length()` are correct.
    #[test]
    fn jamba_forward_pass_metadata() {
        let cfg = tiny_config();
        let model = build_zero_jamba_model(cfg);
        assert_eq!(model.vocab_size(), 32);
        assert_eq!(model.hidden_size(), 8);
        assert_eq!(model.max_context_length(), 64);
    }

    // ─── LoRA stack validation tests ─────────────────────────────────────────

    use crate::lora::adapter::{LoraAdapterTrait, LoraDelta, TargetModule};
    use std::sync::Arc;

    /// A minimal in-memory adapter for testing dimension checks.
    struct DimAdapter {
        rank: usize,
        alpha: f32,
        delta: LoraDelta,
        target: TargetModule,
    }

    impl DimAdapter {
        fn new(rank: usize, in_dim: usize, out_dim: usize, target: TargetModule) -> Self {
            Self {
                rank,
                alpha: rank as f32,
                delta: LoraDelta::new(
                    vec![0.0f32; rank * in_dim],
                    vec![0.0f32; out_dim * rank],
                    rank,
                    in_dim,
                    out_dim,
                ),
                target,
            }
        }
    }

    impl LoraAdapterTrait for DimAdapter {
        fn rank(&self) -> usize {
            self.rank
        }
        fn alpha(&self) -> f32 {
            self.alpha
        }
        fn target_modules(&self) -> &[TargetModule] {
            std::slice::from_ref(&self.target)
        }
        fn delta(&self, target: TargetModule, layer: usize) -> Option<&LoraDelta> {
            if target == self.target && layer == 0 {
                Some(&self.delta)
            } else {
                None
            }
        }
    }

    /// `with_lora_stack` accepts a stack whose adapter dimensions match hidden_size.
    #[test]
    fn jamba_with_lora_stack_compatible_accepted() {
        let cfg = tiny_config(); // hidden_size = 8
        let mut model = build_zero_jamba_model(cfg);

        let adapter = DimAdapter::new(2, 8, 8, TargetModule::QueryProj);
        let mut stack = crate::lora::LoraStack::new();
        stack.push_adapter(Arc::new(adapter));

        let result = model.with_lora_stack(stack);
        assert!(
            result.is_ok(),
            "compatible adapter (in=8, out=8) must be accepted"
        );
        assert!(
            model.lora_stack.is_some(),
            "lora_stack field must be set after with_lora_stack"
        );
    }

    /// `with_lora_stack` rejects an adapter whose in_dim mismatches hidden_size.
    #[test]
    fn jamba_with_lora_stack_incompatible_dim_rejected() {
        let cfg = tiny_config(); // hidden_size = 8
        let mut model = build_zero_jamba_model(cfg);

        // Adapter with in_dim=16 but model hidden_size=8 → should be rejected.
        let adapter = DimAdapter::new(2, 16, 8, TargetModule::QueryProj);
        let mut stack = crate::lora::LoraStack::new();
        stack.push_adapter(Arc::new(adapter));

        let result = model.with_lora_stack(stack);
        assert!(result.is_err(), "mismatched in_dim must be rejected");
        match result {
            Err(ArchError::LoraIncompatible { ref detail }) => {
                assert!(
                    detail.contains("in_dim"),
                    "error must mention 'in_dim', got: {detail}"
                );
            }
            other => panic!("expected LoraIncompatible, got: {other:?}"),
        }
    }
}

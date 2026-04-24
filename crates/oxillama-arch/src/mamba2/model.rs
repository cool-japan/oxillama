//! Mamba-2 model implementation.
//!
//! Mamba-2 is a state-space sequence model (SSM) that uses selective scan
//! instead of attention. Each block processes tokens sequentially using
//! a recurrent hidden state per layer.
//!
//! ## Block forward pass (per layer)
//!
//! ```text
//! x      = rms_norm(hidden)
//! z      = x @ w_z          (gate)
//! y      = x @ w_in         (input projection)
//! y      = silu(conv1d(y, w_conv, b_conv))
//! B      = y @ w_B
//! C      = y @ w_C
//! d_raw  = y @ w_delta
//! delta  = softplus(d_raw + b_delta)
//! out    = selective_scan(y, delta, log_A, B, C, D)
//! out    = silu(z) * out     (gating)
//! hidden = hidden + out @ w_out
//! ```

use crate::common::rms_norm::RmsNorm;
use crate::common::sequence_state::{Mamba2SequenceState, SequenceState};
use crate::error::{ArchError, ArchResult};
use crate::mamba2::conv::conv1d_depthwise;
use crate::mamba2::ssm::selective_scan_sequential;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::GgufTensorType;
use oxillama_quant::{KernelDispatcher, QuantTensor};

// ─── Config ───────────────────────────────────────────────────────────────────

/// Mamba-2 model configuration.
#[derive(Debug, Clone)]
pub struct Mamba2Config {
    /// Hidden (model) dimension (d_model).
    pub d_model: usize,
    /// Number of SSM layers.
    pub n_layer: usize,
    /// SSM state dimension (typically 128 in full models; small in tests).
    pub d_state: usize,
    /// Convolution kernel width (typically 4).
    pub d_conv: usize,
    /// Expansion factor: `d_inner = d_model * expand`.
    pub expand: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl Mamba2Config {
    /// Compute the inner dimension.
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }
}

impl Mamba2Config {
    /// Parse a `Mamba2Config` from GGUF metadata.
    pub fn from_metadata(metadata: &oxillama_gguf::MetadataStore) -> Self {
        // Some GGUFs use "mamba2.*", others just "mamba.*"
        let d_model = metadata
            .get_u32("mamba2.d_model")
            .or_else(|_| metadata.get_u32("mamba.d_model"))
            .map(|v| v as usize)
            .unwrap_or(128);

        let n_layer = metadata
            .get_u32("mamba2.n_layer")
            .or_else(|_| metadata.get_u32("mamba.n_layer"))
            .or_else(|_| metadata.get_u32("mamba2.block_count"))
            .map(|v| v as usize)
            .unwrap_or(24);

        let d_state = metadata
            .get_u32("mamba2.d_state")
            .or_else(|_| metadata.get_u32("mamba.d_state"))
            .map(|v| v as usize)
            .unwrap_or(128);

        let d_conv = metadata
            .get_u32("mamba2.d_conv")
            .or_else(|_| metadata.get_u32("mamba.d_conv"))
            .map(|v| v as usize)
            .unwrap_or(4);

        let expand = metadata
            .get_u32("mamba2.expand")
            .or_else(|_| metadata.get_u32("mamba.expand"))
            .map(|v| v as usize)
            .unwrap_or(2);

        let vocab_size = metadata
            .get_u32("mamba2.vocab_size")
            .or_else(|_| metadata.get_u32("mamba.vocab_size"))
            .or_else(|_| metadata.get_u32("tokenizer.ggml.tokens.length"))
            .map(|v| v as usize)
            .unwrap_or(32000);

        let max_seq_len = metadata
            .get_u32("mamba2.context_length")
            .or_else(|_| metadata.get_u32("mamba.context_length"))
            .map(|v| v as usize)
            .unwrap_or(4096);

        Self {
            d_model,
            n_layer,
            d_state,
            d_conv,
            expand,
            vocab_size,
            max_seq_len,
        }
    }
}

// ─── Per-layer weights ─────────────────────────────────────────────────────────

/// Weights for one Mamba-2 SSM block.
///
/// All projection weights are stored as `f32` slices (pre-dequantised) to
/// keep the implementation straightforward. Real loaders would hold `QuantTensor`
/// and call dispatch kernels.
pub struct Mamba2LayerWeights {
    /// Combined gate+input projection `[2 * d_inner, d_model]` row-major.
    /// The first `d_inner` rows are the gate (z) projection,
    /// the next `d_inner` rows are the input (y) projection.
    pub w_in_z: Vec<f32>,
    /// 1-D depthwise conv kernel `[d_inner × d_conv]` row-major.
    pub w_conv: Vec<f32>,
    /// Conv bias `[d_inner]`.
    pub b_conv: Vec<f32>,
    /// x → B projection `[d_state, d_inner]` row-major.
    pub w_b: Vec<f32>,
    /// x → C projection `[d_state, d_inner]` row-major.
    pub w_c: Vec<f32>,
    /// x → Δ projection (dt) `[d_inner, d_inner]` row-major.
    pub w_delta: Vec<f32>,
    /// Δ bias `[d_inner]`.
    pub b_delta: Vec<f32>,
    /// Log-parameterised A `[d_state × d_inner]` row-major.
    pub log_a: Vec<f32>,
    /// Skip connection D `[d_inner]`.
    pub d_skip: Vec<f32>,
    /// Output projection `[d_model, d_inner]` row-major.
    pub w_out: Vec<f32>,
    /// Pre-block RMSNorm.
    pub norm: RmsNorm,
}

// ─── Full model ────────────────────────────────────────────────────────────────

/// Complete Mamba-2 model.
pub struct Mamba2Model {
    /// Model configuration.
    pub config: Mamba2Config,
    /// Token embedding table `[vocab_size, d_model]` stored as f32.
    pub token_embd: Vec<f32>,
    /// Per-layer SSM weights.
    pub layers: Vec<Mamba2LayerWeights>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head projection `[vocab_size, d_model]` stored as f32.
    pub lm_head: Vec<f32>,
    /// Recurrent state for all SSM layers.
    pub state: Mamba2SequenceState,
    /// Kernel dispatcher (kept for API compatibility with quantized paths).
    pub _dispatcher: KernelDispatcher,
}

impl Mamba2Model {
    /// Create a new `Mamba2Model` from pre-loaded weights.
    pub fn new(
        config: Mamba2Config,
        token_embd: Vec<f32>,
        layers: Vec<Mamba2LayerWeights>,
        output_norm: RmsNorm,
        lm_head: Vec<f32>,
    ) -> Self {
        let n_layer = config.n_layer;
        let d_state = config.d_state;
        let d_inner = config.d_inner();
        let max_seq = config.max_seq_len;
        let state = Mamba2SequenceState::new(n_layer, d_state, d_inner, max_seq);
        Self {
            config,
            token_embd,
            layers,
            output_norm,
            lm_head,
            state,
            _dispatcher: KernelDispatcher::new(),
        }
    }

    /// Reset all per-layer SSM hidden states and the position counter.
    pub fn reset_state(&mut self) {
        self.state.reset();
    }

    /// Run one Mamba-2 block for a single token `x` at the given layer.
    fn mamba_block(&mut self, layer_idx: usize, x: &[f32]) -> ArchResult<Vec<f32>> {
        let d_model = self.config.d_model;
        let d_inner = self.config.d_inner();
        let d_state = self.config.d_state;
        let d_conv = self.config.d_conv;

        let layer = &self.layers[layer_idx];

        // ── 1. RMSNorm ──────────────────────────────────────────────────────────
        // Normalise input token representation.
        let mut normed = x.to_vec();
        layer.norm.forward(&mut normed);

        // ── 2. Gate + input projection ──────────────────────────────────────────
        // w_in_z: [2*d_inner, d_model]
        // First d_inner rows → gate (z); next d_inner rows → input (y).
        let mut z_and_y = vec![0.0f32; 2 * d_inner];
        for (out_idx, z_or_y) in z_and_y.iter_mut().enumerate() {
            let row = &layer.w_in_z[out_idx * d_model..(out_idx + 1) * d_model];
            *z_or_y = row.iter().zip(normed.iter()).map(|(w, xi)| w * xi).sum();
        }
        let z_vec = z_and_y[..d_inner].to_vec();
        let y_in = z_and_y[d_inner..].to_vec();

        // ── 3. Depthwise conv1d + SiLU ──────────────────────────────────────────
        // y_in is treated as a single-token sequence for conv.
        let y_conv = conv1d_depthwise(
            &y_in,
            &layer.w_conv,
            &layer.b_conv,
            1, // seq_len = 1 (token by token)
            d_inner,
            d_conv,
        );
        // y_conv has shape [1 × d_inner]; extract the single token.
        let y = &y_conv[..d_inner];

        // ── 4. Compute B, C, Δ ────────────────────────────────────────────────
        // w_b: [d_state, d_inner], w_c: [d_state, d_inner], w_delta: [d_inner, d_inner]
        let mut b_vec = vec![0.0f32; d_state];
        for (s, bv) in b_vec.iter_mut().enumerate() {
            let row = &layer.w_b[s * d_inner..(s + 1) * d_inner];
            *bv = row.iter().zip(y.iter()).map(|(w, yi)| w * yi).sum();
        }

        let mut c_vec = vec![0.0f32; d_state];
        for (s, cv) in c_vec.iter_mut().enumerate() {
            let row = &layer.w_c[s * d_inner..(s + 1) * d_inner];
            *cv = row.iter().zip(y.iter()).map(|(w, yi)| w * yi).sum();
        }

        let mut delta_raw = vec![0.0f32; d_inner];
        for (i, dr) in delta_raw.iter_mut().enumerate() {
            let row = &layer.w_delta[i * d_inner..(i + 1) * d_inner];
            *dr = row.iter().zip(y.iter()).map(|(w, yi)| w * yi).sum();
            *dr += layer.b_delta[i];
        }

        // Softplus: log(1 + exp(x))
        let delta: Vec<f32> = delta_raw
            .iter()
            .map(|&x| if x > 20.0 { x } else { (1.0f32 + x.exp()).ln() })
            .collect();

        // ── 5. Selective scan (1-token step) ──────────────────────────────────
        let layer_state = &mut self.state.layers[layer_idx];
        let scan_out = selective_scan_sequential(
            y,
            &delta,
            &layer.log_a,
            &b_vec,
            &c_vec,
            &layer.d_skip,
            1, // seq_len = 1
            d_inner,
            d_state,
            layer_state,
        );

        // ── 6. Gate (SiLU(z) * scan_out) ──────────────────────────────────────
        let gated: Vec<f32> = scan_out
            .iter()
            .zip(z_vec.iter())
            .map(|(&o, &zi)| {
                let silu_z = zi / (1.0 + (-zi).exp());
                silu_z * o
            })
            .collect();

        // ── 7. Output projection ────────────────────────────────────────────────
        // w_out: [d_model, d_inner]
        let mut out = vec![0.0f32; d_model];
        for (j, ov) in out.iter_mut().enumerate() {
            let row = &layer.w_out[j * d_inner..(j + 1) * d_inner];
            *ov = row.iter().zip(gated.iter()).map(|(w, g)| w * g).sum();
        }

        Ok(out)
    }
}

impl ForwardPass for Mamba2Model {
    fn forward(
        &mut self,
        tokens: &[u32],
        _kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let d_model = self.config.d_model;
        let vocab = self.config.vocab_size;
        let seq_len = tokens.len();

        if seq_len == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "forward: empty token sequence".to_string(),
            });
        }

        // Embed all tokens.
        let mut logits = vec![0.0f32; vocab];

        for &tok_id in tokens {
            let tok = tok_id as usize;
            if tok >= vocab {
                return Err(ArchError::InvalidConfig {
                    detail: format!("token id {tok} out of range (vocab_size={vocab})"),
                });
            }

            // Embedding lookup.
            let emb_off = tok * d_model;
            let mut hidden: Vec<f32> = self.token_embd[emb_off..emb_off + d_model].to_vec();

            // Run SSM layers.
            let n_layers = self.config.n_layer;
            for layer_idx in 0..n_layers {
                let block_out = self.mamba_block(layer_idx, &hidden).map_err(|e| {
                    ArchError::ForwardPassError {
                        layer: layer_idx,
                        message: format!("Mamba-2 block: {e}"),
                    }
                })?;

                // Residual connection.
                for (h, b) in hidden.iter_mut().zip(block_out.iter()) {
                    *h += b;
                }
            }

            // Final norm + LM head (only last token used for logits).
            let mut last = hidden;
            self.output_norm.forward(&mut last);

            // LM head GEMV: logits[v] = dot(lm_head[v, :], last)
            for (v, lv) in logits.iter_mut().enumerate() {
                let row = &self.lm_head[v * d_model..(v + 1) * d_model];
                *lv = row.iter().zip(last.iter()).map(|(w, h)| w * h).sum();
            }

            self.state.advance();
        }

        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn max_context_length(&self) -> usize {
        self.config.max_seq_len
    }

    fn hidden_size(&self) -> usize {
        self.config.d_model
    }

    fn allocate_sequence_state(
        &self,
        max_context_length: usize,
    ) -> Box<dyn crate::common::sequence_state::SequenceState> {
        Box::new(Mamba2SequenceState::new(
            self.config.n_layer,
            self.config.d_state,
            self.config.d_inner(),
            max_context_length,
        ))
    }
}

// ─── Builder helpers ──────────────────────────────────────────────────────────

/// Create a zero-weight `Mamba2LayerWeights` for the given config.
///
/// Used in tests to construct structurally valid models quickly.
pub fn make_zero_mamba2_layer(cfg: &Mamba2Config) -> Mamba2LayerWeights {
    let d_model = cfg.d_model;
    let d_inner = cfg.d_inner();
    let d_state = cfg.d_state;
    let d_conv = cfg.d_conv;

    Mamba2LayerWeights {
        w_in_z: vec![0.0f32; 2 * d_inner * d_model],
        w_conv: vec![0.0f32; d_inner * d_conv],
        b_conv: vec![0.0f32; d_inner],
        w_b: vec![0.0f32; d_state * d_inner],
        w_c: vec![0.0f32; d_state * d_inner],
        w_delta: vec![0.0f32; d_inner * d_inner],
        b_delta: vec![0.0f32; d_inner],
        log_a: vec![0.0f32; d_state * d_inner],
        d_skip: vec![0.0f32; d_inner],
        w_out: vec![0.0f32; d_model * d_inner],
        norm: RmsNorm::new(vec![1.0f32; d_model], 1e-5),
    }
}

/// Construct a `Mamba2Model` from raw weights.
pub fn build_mamba2_model(
    config: Mamba2Config,
    token_embd: Vec<f32>,
    layers: Vec<Mamba2LayerWeights>,
    output_norm: RmsNorm,
    lm_head: Vec<f32>,
) -> Mamba2Model {
    Mamba2Model::new(config, token_embd, layers, output_norm, lm_head)
}

/// Load a Mamba-2 model from a parsed GGUF file (stub — use `build_mamba2_model` for testing).
pub fn load_mamba2_from_gguf(_model: &oxillama_gguf::GgufModel) -> ArchResult<Mamba2Model> {
    Err(ArchError::MissingTensor {
        name: "load_mamba2_from_gguf: full loader not yet implemented; \
               use build_mamba2_model() directly"
            .to_string(),
    })
}

// We keep QuantTensor in scope for the API; suppress the unused-import warning.
const _: fn() = || {
    let _ = QuantTensor::new(vec![], vec![], GgufTensorType::F32);
};

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
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

    fn tiny_config() -> Mamba2Config {
        Mamba2Config {
            d_model: 16,
            n_layer: 1,
            d_state: 8,
            d_conv: 4,
            expand: 1,
            vocab_size: 64,
            max_seq_len: 256,
        }
    }

    fn build_tiny_model() -> Mamba2Model {
        let cfg = tiny_config();
        let vocab = cfg.vocab_size;
        let d_model = cfg.d_model;

        let token_embd = vec![0.0f32; vocab * d_model];
        let layers = (0..cfg.n_layer)
            .map(|_| make_zero_mamba2_layer(&cfg))
            .collect();
        let output_norm = RmsNorm::new(vec![1.0f32; d_model], 1e-5);
        let lm_head = vec![0.0f32; vocab * d_model];

        build_mamba2_model(cfg, token_embd, layers, output_norm, lm_head)
    }

    #[test]
    fn forward_shape_correct() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let logits = model
            .forward(&[1u32], &mut kv)
            .expect("forward must succeed");
        assert_eq!(logits.len(), 64, "logits must have vocab_size=64 elements");
    }

    #[test]
    fn forward_all_finite() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let logits = model
            .forward(&[0u32], &mut kv)
            .expect("forward must succeed");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite"
        );
    }

    #[test]
    fn empty_tokens_returns_error() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let result = model.forward(&[], &mut kv);
        assert!(result.is_err(), "empty token sequence must return an error");
    }

    #[test]
    fn state_position_advances() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        model.forward(&[1u32, 2, 3], &mut kv).expect("forward");
        assert_eq!(
            model.state.step_position(),
            3,
            "state position must equal seq_len after forward"
        );
        model.reset_state();
        assert_eq!(
            model.state.step_position(),
            0,
            "state position must be 0 after reset"
        );
    }
}

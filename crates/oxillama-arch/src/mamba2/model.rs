//! Mamba-2 block and full-model forward pass.
//!
//! Mamba-2 is a state-space sequence model: no attention, no RoPE
//! (`llama_model_rope_type()` returns `LLAMA_ROPE_TYPE_NONE` for
//! `LLM_ARCH_MAMBA2`), and two recurrent tensors carried per layer — the SSM
//! state `h` and the convolution shift register.
//!
//! ## Block forward pass (per layer)
//!
//! Transcribed from `llm_build_mamba_base::build_mamba2_layer` in
//! `src/models/mamba-base.cpp`:
//!
//! ```text
//! cur    = rms_norm(hidden, attn_norm)
//!
//! zxBCdt = ssm_in @ cur                      // [2*d_inner + 2*n_group*d_state + n_head]
//! z      = zxBCdt[0 .. d_inner]
//! xBC    = zxBCdt[d_inner .. d_inner + conv_dim]
//! dt     = zxBCdt[d_inner + conv_dim ..]     // [n_head]
//!
//! xBC    = silu(conv1d(state ‖ xBC, ssm_conv1d) + ssm_conv1d.bias)
//! x      = xBC[0 .. d_inner]                        // head-major
//! B      = xBC[d_inner .. d_inner + n_group*d_state]
//! C      = xBC[d_inner + n_group*d_state .. conv_dim]
//!
//! y      = ssm_scan(h, x, dt + ssm_dt.bias, ssm_a, B, C)   // per-head A and dt
//! y      = y + x * ssm_d                                   // per-head D
//! y      = silu(z) * y                                     // swiglu_split(z, y)
//! y      = grouped_rms_norm(y, ssm_norm)                   // {d_inner/n_group, n_group}
//! out    = ssm_out @ y
//! hidden = hidden + out
//! ```
//!
//! ## What Mamba-2 is *not*
//!
//! There is no `ssm_x` projection and no `ssm_dt` **weight**: `B`, `C` and `dt`
//! all come out of the fused `ssm_in` projection, and `B`/`C` pass *through*
//! the convolution alongside `x`.  `ssm_a`, `ssm_d` and `ssm_dt.bias` are one
//! scalar per **head**, not per `(d_state, d_inner)` element.

use crate::common::attention::{validate_context_bounds, validate_token_ids};
use crate::common::rms_norm::RmsNorm;
use crate::common::sequence_state::{Mamba2SequenceState, SequenceState};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::mamba2::config::Mamba2Config;
use crate::mamba2::conv::conv1d_depthwise_stateful;
use crate::mamba2::ssm::{selective_scan_mamba2, Mamba2ScanDims};
use crate::mamba2::state::Mamba2ConvCache;
use crate::traits::{ForwardPass, KvCacheAccess};

// ─── Per-layer weights ─────────────────────────────────────────────────────────

/// Weights for one Mamba-2 SSM block.
///
/// Field names map 1:1 onto the GGUF tensors created by the `LLM_ARCH_MAMBA2`
/// branch of `llama_model::load_tensors` (`src/llama-model.cpp:4585`):
///
/// | Field | GGUF tensor | `ne` |
/// |---|---|---|
/// | `norm`     | `blk.N.attn_norm.weight`   | `{n_embd}` |
/// | `w_in`     | `blk.N.ssm_in.weight`      | `{n_embd, d_in_proj}` |
/// | `w_conv`   | `blk.N.ssm_conv1d.weight`  | `{d_conv, conv_dim}` |
/// | `b_conv`   | `blk.N.ssm_conv1d.bias`    | `{conv_dim}` |
/// | `dt_bias`  | `blk.N.ssm_dt.bias`        | `{n_head}` |
/// | `a`        | `blk.N.ssm_a`              | `{1, n_head}` |
/// | `d_skip`   | `blk.N.ssm_d`              | `{1, n_head}` |
/// | `ssm_norm` | `blk.N.ssm_norm.weight`    | `{d_inner/n_group, n_group}` |
/// | `w_out`    | `blk.N.ssm_out.weight`     | `{d_inner, n_embd}` |
#[derive(Debug, Clone)]
pub struct Mamba2LayerWeights {
    /// Pre-block RMSNorm (`blk.N.attn_norm.weight`).
    pub norm: RmsNorm,
    /// Fused `zxBCdt` input projection `[d_in_proj × d_model]` row-major.
    pub w_in: Vec<f32>,
    /// Depthwise conv kernel `[conv_dim × d_conv]` row-major.
    pub w_conv: Vec<f32>,
    /// Conv bias `[conv_dim]`.
    pub b_conv: Vec<f32>,
    /// Per-head Δ bias `[n_head]`.
    pub dt_bias: Vec<f32>,
    /// Per-head `A` `[n_head]`.
    ///
    /// Stored **as-is** from GGUF, where it already equals `-exp(A_log)`.
    pub a: Vec<f32>,
    /// Per-head skip connection `D` `[n_head]`.
    pub d_skip: Vec<f32>,
    /// Gated group-RMSNorm scale `[d_inner]`, laid out `[n_group][d_inner/n_group]`.
    pub ssm_norm: Vec<f32>,
    /// Output projection `[d_model × d_inner]` row-major.
    pub w_out: Vec<f32>,
}

impl Mamba2LayerWeights {
    /// Check every weight against the shapes `cfg` implies.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] naming the first offending tensor.
    pub fn validate(&self, cfg: &Mamba2Config, layer_idx: usize) -> ArchResult<()> {
        let check = |what: &str, got: usize, expected: usize| -> ArchResult<()> {
            if got == expected {
                Ok(())
            } else {
                Err(ArchError::InvalidShape {
                    name: format!("blk.{layer_idx}.{what}"),
                    expected: vec![expected],
                    got: vec![got],
                })
            }
        };

        check("attn_norm.weight", self.norm.weight.len(), cfg.d_model)?;
        check(
            "ssm_in.weight",
            self.w_in.len(),
            cfg.d_in_proj() * cfg.d_model,
        )?;
        check(
            "ssm_conv1d.weight",
            self.w_conv.len(),
            cfg.conv_dim() * cfg.d_conv,
        )?;
        check("ssm_conv1d.bias", self.b_conv.len(), cfg.conv_dim())?;
        check("ssm_dt.bias", self.dt_bias.len(), cfg.n_head)?;
        check("ssm_a", self.a.len(), cfg.n_head)?;
        check("ssm_d", self.d_skip.len(), cfg.n_head)?;
        check("ssm_norm.weight", self.ssm_norm.len(), cfg.d_inner)?;
        check(
            "ssm_out.weight",
            self.w_out.len(),
            cfg.d_model * cfg.d_inner,
        )?;
        Ok(())
    }
}

// ─── Full model ────────────────────────────────────────────────────────────────

/// Complete Mamba-2 model.
pub struct Mamba2Model {
    /// Mamba-2-specific hyper-parameters.
    pub config: Mamba2Config,
    /// Generic view of the same hyper-parameters, for the shared guards in
    /// [`crate::common::attention`].
    pub model_config: ModelConfig,
    /// Token embedding table `[vocab_size × d_model]` row-major.
    pub token_embd: Vec<f32>,
    /// Per-layer SSM weights.
    pub layers: Vec<Mamba2LayerWeights>,
    /// Final RMSNorm before the LM head.
    pub output_norm: RmsNorm,
    /// LM head projection `[vocab_size × d_model]` row-major.
    pub lm_head: Vec<f32>,
    /// Recurrent SSM hidden state for all layers.
    pub state: Mamba2SequenceState,
    /// Convolution shift registers for all layers.
    ///
    /// Held here rather than inside
    /// [`SsmLayerState`](crate::common::sequence_state::SsmLayerState) because
    /// that struct carries only `h`.  See `mamba2/state.rs` for the details.
    pub conv_state: Mamba2ConvCache,
}

/// Build the generic [`ModelConfig`] view of a [`Mamba2Config`].
fn model_config_for(cfg: &Mamba2Config) -> ModelConfig {
    ModelConfig {
        architecture: "mamba2".to_string(),
        hidden_size: cfg.d_model,
        num_layers: cfg.n_layer,
        vocab_size: cfg.vocab_size,
        max_context_length: cfg.max_seq_len,
        rms_norm_eps: cfg.rms_norm_eps,
        // Mamba-2 has no attention at all: LLM_ARCH_MAMBA2 is absent from both
        // RoPE tables and never builds a KV cache.
        num_attention_heads: 0,
        num_kv_heads: 0,
        head_dim: 0,
        intermediate_size: 0,
        ..ModelConfig::default()
    }
}

/// Dot a `[out_dim × in_dim]` row-major weight matrix with an `[in_dim]` vector.
///
/// Every length is checked first: a `zip` between a short weight row and the
/// input silently produces a truncated dot product, which is a wrong answer
/// rather than an error.
fn gemv(name: &str, w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> ArchResult<Vec<f32>> {
    if x.len() != in_dim {
        return Err(ArchError::InvalidShape {
            name: format!("{name}.input"),
            expected: vec![in_dim],
            got: vec![x.len()],
        });
    }
    let need = out_dim
        .checked_mul(in_dim)
        .ok_or_else(|| ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![usize::MAX],
            got: vec![w.len()],
        })?;
    if w.len() != need {
        return Err(ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![out_dim, in_dim],
            got: vec![w.len()],
        });
    }

    let mut out = vec![0.0f32; out_dim];
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += wv * xv;
        }
        *slot = acc;
    }
    Ok(out)
}

/// Grouped RMSNorm over `y`, in place.
///
/// llama.cpp reshapes `y` to `{d_inner/n_group, n_group}` and runs a plain
/// `LLM_NORM_RMS` over `ne[0]`, i.e. each group is normalised independently and
/// scaled by its own slice of `ssm_norm`:
///
/// ```text
/// y = ggml_reshape_4d(ctx0, y, d_inner / n_group, n_group, n_seq_tokens, n_seqs);
/// y = build_norm(y, model.layers[il].ssm_norm, NULL, LLM_NORM_RMS, il);
/// ```
///
/// # Errors
///
/// [`ArchError::InvalidShape`] when `y` or `weight` is not `d_inner` long, or
/// [`ArchError::InvalidConfig`] when `n_group` does not divide `d_inner`.
fn grouped_rms_norm(
    y: &mut [f32],
    weight: &[f32],
    n_group: usize,
    eps: f32,
    layer_idx: usize,
) -> ArchResult<()> {
    if n_group == 0 || !y.len().is_multiple_of(n_group) {
        return Err(ArchError::InvalidConfig {
            detail: format!(
                "blk.{layer_idx}.ssm_norm: n_group ({n_group}) must divide d_inner ({})",
                y.len()
            ),
        });
    }
    if weight.len() != y.len() {
        return Err(ArchError::InvalidShape {
            name: format!("blk.{layer_idx}.ssm_norm.weight"),
            expected: vec![y.len()],
            got: vec![weight.len()],
        });
    }

    let group_size = y.len() / n_group;
    if group_size == 0 {
        return Ok(());
    }
    for g in 0..n_group {
        let lo = g * group_size;
        let hi = lo + group_size;
        let mut sum_sq = 0.0f32;
        for v in &y[lo..hi] {
            sum_sq += v * v;
        }
        // An all-zero group with eps == 0 would give inv_rms == inf and then
        // 0 * inf == NaN; the limit of the normalisation is 0 there.
        let denom = (sum_sq / group_size as f32 + eps).sqrt();
        let inv_rms = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        for i in lo..hi {
            y[i] = y[i] * inv_rms * weight[i];
        }
    }
    Ok(())
}

/// SiLU: `x * sigmoid(x)`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Mamba2Model {
    /// Create a `Mamba2Model` from pre-loaded, dequantized weights.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] when `config` is internally inconsistent,
    /// or [`ArchError::InvalidShape`] when any weight — including
    /// `token_embd` and `lm_head` — disagrees with the config.  Validating the
    /// embedding table here is what keeps an over-estimated `vocab_size` from
    /// turning into an out-of-bounds slice during the first forward pass.
    pub fn new(
        config: Mamba2Config,
        token_embd: Vec<f32>,
        layers: Vec<Mamba2LayerWeights>,
        output_norm: RmsNorm,
        lm_head: Vec<f32>,
    ) -> ArchResult<Self> {
        config.validate()?;

        if layers.len() != config.n_layer {
            return Err(ArchError::InvalidShape {
                name: "mamba2.layers".to_string(),
                expected: vec![config.n_layer],
                got: vec![layers.len()],
            });
        }
        for (i, layer) in layers.iter().enumerate() {
            layer.validate(&config, i)?;
        }

        let embd_len = config
            .vocab_size
            .checked_mul(config.d_model)
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: format!(
                    "mamba2: vocab_size {} × d_model {} overflows",
                    config.vocab_size, config.d_model
                ),
            })?;
        if token_embd.len() != embd_len {
            return Err(ArchError::InvalidShape {
                name: "token_embd.weight".to_string(),
                expected: vec![config.vocab_size, config.d_model],
                got: vec![token_embd.len()],
            });
        }
        if lm_head.len() != embd_len {
            return Err(ArchError::InvalidShape {
                name: "output.weight".to_string(),
                expected: vec![config.vocab_size, config.d_model],
                got: vec![lm_head.len()],
            });
        }
        if output_norm.weight.len() != config.d_model {
            return Err(ArchError::InvalidShape {
                name: "output_norm.weight".to_string(),
                expected: vec![config.d_model],
                got: vec![output_norm.weight.len()],
            });
        }

        let state = Mamba2SequenceState::new(
            config.n_layer,
            config.d_state,
            config.d_inner,
            config.max_seq_len,
        );
        let conv_state = Mamba2ConvCache::new(config.n_layer, config.conv_dim(), config.d_conv);
        let model_config = model_config_for(&config);

        Ok(Self {
            config,
            model_config,
            token_embd,
            layers,
            output_norm,
            lm_head,
            state,
            conv_state,
        })
    }

    /// Reset all recurrent state: SSM hidden states, conv rings and position.
    pub fn reset_state(&mut self) {
        self.state.reset();
        self.conv_state.clear();
    }

    /// Look up one token's embedding row, bounds-checked.
    fn embed_token(&self, token_id: u32) -> ArchResult<Vec<f32>> {
        let d_model = self.config.d_model;
        let tok = token_id as usize;
        let off = tok
            .checked_mul(d_model)
            .ok_or_else(|| ArchError::ConfigMismatch {
                param: "token_id".to_string(),
                expected: format!("< vocab_size ({})", self.config.vocab_size),
                got: token_id.to_string(),
            })?;
        let row = self
            .token_embd
            .get(off..off.saturating_add(d_model))
            .ok_or_else(|| ArchError::ConfigMismatch {
                param: "token_id".to_string(),
                expected: format!(
                    "< token_embd rows ({})",
                    self.token_embd.len() / d_model.max(1)
                ),
                got: token_id.to_string(),
            })?;
        Ok(row.to_vec())
    }

    /// Run one Mamba-2 block for a single token at `layer_idx`.
    ///
    /// `x` is the residual-stream vector `[d_model]`; the returned vector is
    /// the block output to be *added* to it.
    fn mamba2_block(&mut self, layer_idx: usize, x: &[f32]) -> ArchResult<Vec<f32>> {
        let cfg = &self.config;
        let d_model = cfg.d_model;
        let d_inner = cfg.d_inner;
        let d_conv = cfg.d_conv;
        let n_head = cfg.n_head;
        let head_dim = cfg.head_dim();
        let n_group = cfg.n_group;
        let d_state = cfg.d_state;
        let conv_dim = cfg.conv_dim();
        let d_in_proj = cfg.d_in_proj();
        let bc_width = cfg.bc_width();
        let eps = cfg.rms_norm_eps;

        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: format!("mamba2: no weights for layer {layer_idx}"),
            })?;

        // ── 1. Pre-block RMSNorm ────────────────────────────────────────────
        if x.len() != d_model {
            return Err(ArchError::InvalidShape {
                name: format!("blk.{layer_idx}.input"),
                expected: vec![d_model],
                got: vec![x.len()],
            });
        }
        let mut normed = x.to_vec();
        layer.norm.forward(&mut normed);

        // ── 2. Fused zxBCdt projection ──────────────────────────────────────
        let zxbcdt = gemv(
            &format!("blk.{layer_idx}.ssm_in.weight"),
            &layer.w_in,
            &normed,
            d_in_proj,
            d_model,
        )?;
        let z = &zxbcdt[..d_inner];
        let xbc_in = &zxbcdt[d_inner..d_inner + conv_dim];
        let dt_raw = &zxbcdt[d_inner + conv_dim..];

        // ── 3. Depthwise conv1d (+ bias + SiLU) over x ‖ B ‖ C ───────────────
        let ring = self.conv_state.layer_mut(layer_idx)?;
        let xbc = conv1d_depthwise_stateful(
            xbc_in,
            &layer.w_conv,
            &layer.b_conv,
            ring,
            1, // one token per recurrent step
            conv_dim,
            d_conv,
        )?;

        // Re-borrow immutably: `layer_mut` above took &mut self.conv_state.
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: format!("mamba2: no weights for layer {layer_idx}"),
            })?;

        // ── 4. Split the conv output into x, B and C ─────────────────────────
        let x_ssm = &xbc[..d_inner];
        let b_mat = &xbc[d_inner..d_inner + bc_width];
        let c_mat = &xbc[d_inner + bc_width..conv_dim];

        // ── 5. Selective scan with per-head A and dt ─────────────────────────
        let layer_state =
            self.state
                .layers
                .get_mut(layer_idx)
                .ok_or_else(|| ArchError::InvalidConfig {
                    detail: format!("mamba2: no SSM state for layer {layer_idx}"),
                })?;
        let mut y = selective_scan_mamba2(
            x_ssm,
            dt_raw,
            &layer.dt_bias,
            &layer.a,
            b_mat,
            c_mat,
            &layer.d_skip,
            Mamba2ScanDims {
                seq_len: 1,
                n_head,
                head_dim,
                d_state,
                n_group,
            },
            layer_state,
        )?;

        // ── 6. Gate: swiglu_split(z, y) == silu(z) * y ──────────────────────
        for (yv, zv) in y.iter_mut().zip(z.iter()) {
            *yv *= silu(*zv);
        }

        // ── 7. Gated grouped RMSNorm ────────────────────────────────────────
        grouped_rms_norm(&mut y, &layer.ssm_norm, n_group, eps, layer_idx)?;

        // ── 8. Output projection ────────────────────────────────────────────
        gemv(
            &format!("blk.{layer_idx}.ssm_out.weight"),
            &layer.w_out,
            &y,
            d_model,
            d_inner,
        )
    }

    /// Run all blocks over `tokens`, returning the final hidden state
    /// (before `output_norm`).
    fn run_layers(&mut self, tokens: &[u32]) -> ArchResult<Vec<f32>> {
        let mut hidden = Vec::new();
        for &tok_id in tokens {
            hidden = self.embed_token(tok_id)?;

            for layer_idx in 0..self.config.n_layer {
                let block_out = self.mamba2_block(layer_idx, &hidden).map_err(|e| {
                    ArchError::ForwardPassError {
                        layer: layer_idx,
                        message: format!("Mamba-2 block: {e}"),
                    }
                })?;
                for (h, b) in hidden.iter_mut().zip(block_out.iter()) {
                    *h += b;
                }
            }

            self.state.advance();
        }
        Ok(hidden)
    }

    /// Shared entry guard for [`ForwardPass::forward`] and
    /// [`ForwardPass::embed`].
    ///
    /// Mamba-2 needs no per-token position (it has no RoPE and no KV cache), so
    /// the only positional quantity is the sequence offset used for the
    /// context-length bound.  The cache's `seq_len()` is the runtime's view of
    /// that offset; the model's own step position is used as a floor so that a
    /// stub cache which always reports 0 cannot defeat the guard.
    fn check_inputs(&self, tokens: &[u32], kv_cache: &dyn KvCacheAccess) -> ArchResult<()> {
        if tokens.is_empty() {
            return Err(ArchError::InvalidConfig {
                detail: "mamba2: empty token sequence".to_string(),
            });
        }
        let start_pos = kv_cache.seq_len().max(self.state.step_position());
        validate_context_bounds(&self.model_config, start_pos, tokens.len())?;
        validate_token_ids(&self.model_config, tokens)?;
        Ok(())
    }
}

impl ForwardPass for Mamba2Model {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.check_inputs(tokens, kv_cache)?;

        let mut last = self.run_layers(tokens)?;
        self.output_norm.forward(&mut last);

        gemv(
            "output.weight",
            &self.lm_head,
            &last,
            self.config.vocab_size,
            self.config.d_model,
        )
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

    /// Return the post-norm hidden state of the last token without projecting
    /// through the LM head.
    ///
    /// The returned vector has `d_model` elements — the hidden dimension, **not**
    /// `vocab_size`.
    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.check_inputs(tokens, kv_cache)?;

        let mut last = self.run_layers(tokens)?;
        self.output_norm.forward(&mut last);
        Ok(last)
    }

    /// Clear the SSM hidden state, the convolution shift registers and the
    /// step position.
    ///
    /// The trait default is a no-op, which for a recurrent model leaks the
    /// previous request's state into the next one.
    fn reset_sequence(&mut self) {
        self.reset_state();
    }

    fn allocate_sequence_state(
        &self,
        max_context_length: usize,
    ) -> Box<dyn crate::common::sequence_state::SequenceState> {
        Box::new(Mamba2SequenceState::new(
            self.config.n_layer,
            self.config.d_state,
            self.config.d_inner,
            max_context_length,
        ))
    }
}

// ─── Builder helpers ──────────────────────────────────────────────────────────

/// Create a zero-weight `Mamba2LayerWeights` for the given config.
///
/// Used in tests to construct structurally valid models quickly.
pub fn make_zero_mamba2_layer(cfg: &Mamba2Config) -> Mamba2LayerWeights {
    Mamba2LayerWeights {
        norm: RmsNorm::new(vec![1.0f32; cfg.d_model], cfg.rms_norm_eps),
        w_in: vec![0.0f32; cfg.d_in_proj() * cfg.d_model],
        w_conv: vec![0.0f32; cfg.conv_dim() * cfg.d_conv],
        b_conv: vec![0.0f32; cfg.conv_dim()],
        dt_bias: vec![0.0f32; cfg.n_head],
        a: vec![-1.0f32; cfg.n_head],
        d_skip: vec![0.0f32; cfg.n_head],
        ssm_norm: vec![1.0f32; cfg.d_inner],
        w_out: vec![0.0f32; cfg.d_model * cfg.d_inner],
    }
}

/// Construct a `Mamba2Model` from raw weights.
///
/// # Errors
///
/// See [`Mamba2Model::new`].
pub fn build_mamba2_model(
    config: Mamba2Config,
    token_embd: Vec<f32>,
    layers: Vec<Mamba2LayerWeights>,
    output_norm: RmsNorm,
    lm_head: Vec<f32>,
) -> ArchResult<Mamba2Model> {
    Mamba2Model::new(config, token_embd, layers, output_norm, lm_head)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba2::load_mamba2_from_gguf;
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
            d_model: 8,
            n_layer: 1,
            d_inner: 16,
            d_state: 4,
            d_conv: 4,
            n_head: 4,
            n_group: 2,
            vocab_size: 64,
            max_seq_len: 256,
            rms_norm_eps: 1e-5,
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
            .expect("tiny model must build")
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

    // ─── embed() tests ────────────────────────────────────────────────────────

    #[test]
    fn mamba2_embed_returns_correct_size() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let embedding = model.embed(&[1u32], &mut kv).expect("embed must succeed");
        assert_eq!(
            embedding.len(),
            model.config.d_model,
            "embed() must return d_model={} elements",
            model.config.d_model,
        );
    }

    #[test]
    fn mamba2_embed_no_nan() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let embedding = model
            .embed(&[0u32, 1, 2], &mut kv)
            .expect("embed must succeed");
        assert!(
            embedding.iter().all(|v| !v.is_nan()),
            "embed() output must not contain NaN"
        );
    }

    #[test]
    fn mamba2_embed_empty_returns_error() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let result = model.embed(&[], &mut kv);
        assert!(
            result.is_err(),
            "embed with empty tokens must return an error"
        );
    }

    // ─── Shape validation ─────────────────────────────────────────────────────

    #[test]
    fn build_rejects_token_embd_shorter_than_vocab() {
        let cfg = tiny_config();
        let layers = (0..cfg.n_layer)
            .map(|_| make_zero_mamba2_layer(&cfg))
            .collect();
        // Two rows short of vocab_size × d_model.
        let token_embd = vec![0.0f32; (cfg.vocab_size - 2) * cfg.d_model];
        let lm_head = vec![0.0f32; cfg.vocab_size * cfg.d_model];
        let err = build_mamba2_model(
            cfg,
            token_embd,
            layers,
            RmsNorm::new(vec![1.0f32; 8], 1e-5),
            lm_head,
        )
        .err()
        .expect("an over-estimated vocab_size must be rejected at load time");
        assert!(
            format!("{err}").contains("token_embd.weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gemv_rejects_short_weight_instead_of_truncating() {
        // 3 outputs × 4 inputs needs 12 weights; give it 10.
        let err = gemv("test.weight", &[1.0f32; 10], &[1.0, 1.0, 1.0, 1.0], 3, 4)
            .expect_err("a short weight matrix must be an error, not a short zip");
        assert!(format!("{err}").contains("test.weight"), "{err}");
    }

    #[test]
    fn grouped_rms_norm_normalises_each_group_independently() {
        // Two groups of two: [3, 4] and [0, 0].
        let mut y = vec![3.0f32, 4.0, 0.0, 0.0];
        let w = vec![1.0f32; 4];
        grouped_rms_norm(&mut y, &w, 2, 0.0, 0).expect("norm");
        // rms([3,4]) = sqrt((9+16)/2) = sqrt(12.5)
        let inv = 1.0f32 / 12.5f32.sqrt();
        assert!((y[0] - 3.0 * inv).abs() < 1e-6, "y[0] = {}", y[0]);
        assert!((y[1] - 4.0 * inv).abs() < 1e-6, "y[1] = {}", y[1]);
        // The zero group must stay zero (and must not become NaN through a
        // 0 * inf) and must not be affected by group 0.
        assert_eq!(y[2], 0.0, "zero group must not become {}", y[2]);
        assert_eq!(y[3], 0.0);
    }

    /// A group of all zeros must never produce NaN, whatever eps is.
    #[test]
    fn grouped_rms_norm_zero_group_is_not_nan() {
        for eps in [0.0f32, 1e-5] {
            let mut y = vec![0.0f32; 4];
            grouped_rms_norm(&mut y, &[1.0; 4], 2, eps, 0).expect("norm");
            assert!(
                y.iter().all(|v| v.is_finite()),
                "eps={eps} produced non-finite output {y:?}"
            );
        }
    }

    // ─── Round-trip loader test ───────────────────────────────────────────────

    #[test]
    fn mamba2_loader_round_trip() {
        let bytes = oxillama_gguf::test_utils::build_minimal_mamba2_gguf();
        let gguf_model =
            oxillama_gguf::GgufModel::from_bytes(bytes).expect("GGUF parse must succeed");

        let mut model =
            load_mamba2_from_gguf(&gguf_model).expect("load_mamba2_from_gguf must succeed");

        assert_eq!(model.config.d_model, 16, "d_model");
        assert_eq!(model.config.d_inner, 32, "d_inner = 2 * d_model");
        assert_eq!(model.config.vocab_size, 256, "vocab_size");
        assert_eq!(model.config.n_layer, 1, "n_layer");
        assert_eq!(model.config.n_group, 2, "ssm.group_count");
        assert_eq!(model.config.n_head, 4, "ssm.time_step_rank == n_head");
        assert_eq!(model.config.head_dim(), 8, "head_dim");

        let mut kv = NullKv;
        let logits = model.forward(&[1u32], &mut kv).expect("forward after load");
        assert_eq!(logits.len(), 256, "logit count == vocab_size");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite after load"
        );

        model.reset_state();
        let emb = model.embed(&[0u32], &mut kv).expect("embed after load");
        assert_eq!(emb.len(), 16, "embed len == d_model");
        assert!(
            emb.iter().all(|v| !v.is_nan()),
            "embed output must not contain NaN"
        );
    }
}

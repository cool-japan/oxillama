//! Rotary Position Embedding (RoPE).
//!
//! Applies rotary position embeddings to query and key tensors,
//! enabling the model to encode relative positional information.
//!
//! # Pair conventions
//!
//! llama.cpp uses **two mutually incompatible** rotation conventions, selected
//! per architecture by `llama_model_rope_type()`:
//!
//! | Convention | Rotated pair | `LLAMA_ROPE_TYPE_*` | [`RopeStyle`] |
//! |------------|--------------|---------------------|---------------|
//! | "normal"   | `(x[2i], x[2i+1])`      | `NORM` | [`RopeStyle::Norm`] |
//! | GPT-NeoX   | `(x[i], x[i + n/2])`    | `NEOX` | [`RopeStyle::Neox`] |
//!
//! The two are **not** interchangeable: `convert_hf_to_gguf.py`'s `permute()`
//! rewrites the Q/K weight rows of the `NORM` family so that the interleaved
//! pairing reproduces HuggingFace's half-split rotation.  Applying `Neox`
//! rotation to a `Norm` checkpoint (or vice versa) silently rotates the wrong
//! element pairs and destroys the model's positional signal.
//!
//! [`crate::config::rope_style_for_arch`] reproduces llama.cpp's per-architecture
//! table; [`crate::config::ModelConfig::rope_style`] is the convenience wrapper.
//!
//! # Frequency scaling
//!
//! - [`RopeScalingType::Standard`]: no scaling (default)
//! - [`RopeScalingType::Linear`]: divide all frequencies by `scaling_factor`
//! - [`RopeScalingType::Yarn`]: NTK-by-parts (YaRN paper), matching ggml's
//!   `rope_yarn` / `rope_yarn_ramp` exactly — high-frequency dimensions
//!   **extrapolate**, low-frequency dimensions **interpolate**
//! - [`RopeScalingType::LongRope`]: Phi-3.5 `longrope` / `su`, driven by the
//!   per-dimension `rope_factors_long` / `rope_factors_short` tensors
//! - [`RopeScalingType::Llama3`]: Llama-3.1/3.2 frequency-factor scaling; the
//!   official converter bakes this into a `rope_freqs.weight` tensor, which is
//!   consumed through [`RopeParams::with_freq_factors`]

use std::f32::consts::PI;

use oxillama_gguf::MetadataStore;

use crate::error::{ArchError, ArchResult};

/// RoPE element-pairing convention.
///
/// Mirrors `LLAMA_ROPE_TYPE_NORM` / `LLAMA_ROPE_TYPE_NEOX` in llama.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeStyle {
    /// GPT-NeoX convention: rotate `(x[i], x[i + head_dim/2])`.
    ///
    /// Used by qwen*/gemma*/phi*/falcon/grok/dbrx/stablelm/gptneox/olmo2/
    /// starcoder2/minicpm3/… — see [`crate::config::rope_style_for_arch`].
    ///
    /// This is the historical default of this crate and therefore the
    /// [`Default`], so architectures that have not yet declared a style keep
    /// their current behaviour.
    #[default]
    Neox,
    /// llama.cpp "normal" convention: rotate consecutive pairs `(x[2i], x[2i+1])`.
    ///
    /// Used by llama/mistral/command-r/deepseek2/granite/minicpm/starcoder/
    /// internlm2/yi/olmo/baichuan/… — these checkpoints have had their Q/K
    /// rows permuted at conversion time (`convert_hf_to_gguf.py::permute`).
    Norm,
}

/// RoPE frequency scaling strategy.
///
/// The string values are the ones written into `{arch}.rope.scaling.type` by
/// `gguf-py`'s `RopeScalingType` enum (`none` / `linear` / `yarn` / `longrope`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeScalingType {
    /// Standard RoPE — no scaling (GGUF `"none"`, or the key being absent).
    #[default]
    Standard,
    /// Linear frequency scaling: divide all frequencies by `rope_scaling_factor`.
    /// Extends context proportionally. GGUF `"linear"`.
    Linear,
    /// YaRN (NTK-by-parts): high-frequency dimensions extrapolate, low-frequency
    /// dimensions interpolate, with a smooth ramp between the two correction
    /// dimensions derived from `beta_fast` / `beta_slow`. GGUF `"yarn"`.
    Yarn,
    /// LongRoPE (Phi-3.5 `longrope` / `su`): per-dimension frequency factors
    /// shipped as the `rope_factors_long` / `rope_factors_short` tensors.
    /// GGUF `"longrope"`.
    LongRope,
    /// Llama-3.1 / 3.2 frequency-factor scaling (HF `rope_type = "llama3"`).
    ///
    /// The official converter pre-computes this into a `rope_freqs.weight`
    /// tensor and writes **no** scaling type, so most Llama-3.1 GGUFs parse as
    /// [`Self::Standard`] and must supply the tensor via
    /// [`RopeParams::with_freq_factors`].  This variant covers converters that
    /// do write `"llama3"` together with the `low_freq_factor` /
    /// `high_freq_factor` keys; the factors are then derived in-crate by
    /// [`llama3_freq_factors`].
    Llama3,
}

impl RopeScalingType {
    /// Parse a GGUF `{arch}.rope.scaling.type` string.
    ///
    /// # Errors
    ///
    /// Returns [`ArchError::NotSupported`] for a non-empty string that is not a
    /// recognised scaling type, rather than silently degrading to
    /// [`Self::Standard`] (which produces garbage past the training context).
    pub fn parse(value: &str) -> ArchResult<Self> {
        match value {
            "none" | "" => Ok(Self::Standard),
            "linear" => Ok(Self::Linear),
            "yarn" => Ok(Self::Yarn),
            // `su` is the pre-rename spelling still emitted by older Phi-3
            // conversions; llama.cpp maps both onto LLAMA_ROPE_SCALING_TYPE_LONGROPE.
            "longrope" | "su" => Ok(Self::LongRope),
            "llama3" => Ok(Self::Llama3),
            other => Err(ArchError::NotSupported {
                detail: format!(
                    "rope.scaling.type '{other}' is not a supported RoPE scaling strategy \
                     (expected one of: none, linear, yarn, longrope, su, llama3)"
                ),
            }),
        }
    }
}

/// Fully-resolved RoPE construction parameters.
///
/// This is the side-car that carries everything [`RopeTable`] needs beyond
/// `head_dim` / `max_seq_len` / `base`.  It exists as a separate struct rather
/// than as extra [`ModelConfig`](crate::config::ModelConfig) fields so that the
/// per-architecture `ModelConfig { .. }` literals keep compiling.
///
/// Build it with [`RopeParams::from_metadata`] (preferred — reads every YaRN
/// key) or [`RopeParams::from_config`] (when only a `ModelConfig` is at hand).
#[derive(Debug, Clone)]
pub struct RopeParams {
    /// Element-pairing convention.
    pub style: RopeStyle,
    /// Frequency scaling strategy.
    pub scaling_type: RopeScalingType,
    /// `{arch}.rope.scaling.factor` — the context-extension factor.
    /// llama.cpp's `rope_freq_scale = 1 / factor`.
    pub scaling_factor: f32,
    /// `{arch}.rope.scaling.original_context_length` (`n_ctx_orig_yarn`).
    pub original_context: u32,
    /// `{arch}.rope.scaling.yarn_beta_fast` — low correction dim (default 32).
    pub beta_fast: f32,
    /// `{arch}.rope.scaling.yarn_beta_slow` — high correction dim (default 1).
    pub beta_slow: f32,
    /// `{arch}.rope.scaling.attn_factor` (`hparams.rope_attn_factor`, default 1).
    pub attn_factor: f32,
    /// `{arch}.rope.scaling.yarn_ext_factor`. `None` means "derive from
    /// `scaling_type`" exactly as `llama_context` does: `1.0` for YaRN, `0.0`
    /// otherwise.
    pub ext_factor: Option<f32>,
    /// `{arch}.rope.scaling.yarn_log_multiplier` (`hparams.rope_yarn_log_mul`),
    /// DeepSeek-V2/V3's `mscale_all_dim`. `0.0` disables the special path.
    pub yarn_log_mul: f32,
    /// Per-rotation-pair frequency divisors, length `head_dim / 2`.
    ///
    /// This is ggml's `freq_factors` argument: `theta_i` becomes
    /// `theta_i / freq_factors[i]`.  It carries Llama-3.1's `rope_freqs.weight`
    /// and Phi-3.5's `rope_factors_long` / `rope_factors_short`.
    pub freq_factors: Option<Vec<f32>>,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            style: RopeStyle::default(),
            scaling_type: RopeScalingType::Standard,
            scaling_factor: 1.0,
            // llama.cpp falls back to n_ctx_train when n_ctx_orig_yarn is 0;
            // 0 here means "unset" and is resolved at table-build time.
            original_context: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.0,
            ext_factor: None,
            yarn_log_mul: 0.0,
            freq_factors: None,
        }
    }
}

impl RopeParams {
    /// Construct default parameters for a given pairing convention.
    pub fn new(style: RopeStyle) -> Self {
        Self {
            style,
            ..Self::default()
        }
    }

    /// Read every RoPE-related GGUF key for `arch`.
    ///
    /// Keys read (all optional except that an unrecognised
    /// `rope.scaling.type` is an error):
    ///
    /// | Key | Field |
    /// |-----|-------|
    /// | `{arch}.rope.scaling.type` | [`Self::scaling_type`] |
    /// | `{arch}.rope.scaling.factor` | [`Self::scaling_factor`] |
    /// | `{arch}.rope.scaling.original_context_length` | [`Self::original_context`] |
    /// | `{arch}.rope.scaling.yarn_beta_fast` | [`Self::beta_fast`] |
    /// | `{arch}.rope.scaling.yarn_beta_slow` | [`Self::beta_slow`] |
    /// | `{arch}.rope.scaling.attn_factor` | [`Self::attn_factor`] |
    /// | `{arch}.rope.scaling.yarn_ext_factor` | [`Self::ext_factor`] |
    /// | `{arch}.rope.scaling.yarn_log_multiplier` | [`Self::yarn_log_mul`] |
    ///
    /// The pairing convention comes from
    /// [`rope_style_for_arch`](crate::config::rope_style_for_arch).
    ///
    /// # Errors
    ///
    /// [`ArchError::NotSupported`] if `{arch}.rope.scaling.type` holds an
    /// unrecognised value.
    pub fn from_metadata(metadata: &MetadataStore, arch: &str) -> ArchResult<Self> {
        let scaling_type = match metadata.get_string(&format!("{arch}.rope.scaling.type")) {
            Ok(s) => RopeScalingType::parse(s)?,
            Err(_) => RopeScalingType::Standard,
        };

        let scaling_factor = metadata
            .get_f32(&format!("{arch}.rope.scaling.factor"))
            .unwrap_or(1.0);

        let original_context = metadata
            .get_u32(&format!("{arch}.rope.scaling.original_context_length"))
            .unwrap_or(0);

        let beta_fast = metadata
            .get_f32(&format!("{arch}.rope.scaling.yarn_beta_fast"))
            .unwrap_or(32.0);

        let beta_slow = metadata
            .get_f32(&format!("{arch}.rope.scaling.yarn_beta_slow"))
            .unwrap_or(1.0);

        let attn_factor = metadata
            .get_f32(&format!("{arch}.rope.scaling.attn_factor"))
            .unwrap_or(1.0);

        let ext_factor = metadata
            .get_f32(&format!("{arch}.rope.scaling.yarn_ext_factor"))
            .ok();

        let yarn_log_mul = metadata
            .get_f32(&format!("{arch}.rope.scaling.yarn_log_multiplier"))
            .unwrap_or(0.0);

        Ok(Self {
            style: crate::config::rope_style_for_arch(arch),
            scaling_type,
            scaling_factor,
            original_context,
            beta_fast,
            beta_slow,
            attn_factor,
            ext_factor,
            yarn_log_mul,
            freq_factors: None,
        })
    }

    /// Derive parameters from an already-parsed
    /// [`ModelConfig`](crate::config::ModelConfig).
    ///
    /// Only the fields `ModelConfig` actually carries are populated; YaRN's
    /// `beta_fast` / `beta_slow` / `attn_factor` fall back to the llama.cpp
    /// defaults.  Prefer [`Self::from_metadata`] when the `MetadataStore` is
    /// still available.
    pub fn from_config(config: &crate::config::ModelConfig) -> Self {
        Self {
            style: config.rope_style(),
            scaling_type: config.rope_scaling_type,
            scaling_factor: config.rope_scaling_factor,
            original_context: config.max_context_length as u32,
            ..Self::default()
        }
    }

    /// Attach per-rotation-pair frequency divisors (ggml's `freq_factors`).
    ///
    /// Pass the contents of `rope_freqs.weight` (Llama-3.1/3.2) or
    /// `rope_factors_long` / `rope_factors_short` (Phi-3.5 LongRoPE).  The slice
    /// must have `head_dim / 2` entries; [`RopeTable::new_with_params`] rejects
    /// any other length.
    #[must_use]
    pub fn with_freq_factors(mut self, factors: Vec<f32>) -> Self {
        self.freq_factors = Some(factors);
        self
    }

    /// Override the pairing convention.
    #[must_use]
    pub fn with_style(mut self, style: RopeStyle) -> Self {
        self.style = style;
        self
    }

    /// The effective YaRN extrapolation-mix factor.
    ///
    /// Mirrors `llama_context`: an explicit `yarn_ext_factor` wins, otherwise
    /// `1.0` for [`RopeScalingType::Yarn`] and `0.0` for everything else.
    pub fn effective_ext_factor(&self) -> f32 {
        self.ext_factor.unwrap_or(match self.scaling_type {
            RopeScalingType::Yarn => 1.0,
            _ => 0.0,
        })
    }

    /// llama.cpp's `rope_freq_scale` = `1 / factor`, forced to `1.0` when no
    /// scaling is in effect.
    pub fn freq_scale(&self) -> f32 {
        match self.scaling_type {
            RopeScalingType::Standard => 1.0,
            _ => {
                if self.scaling_factor > 0.0 {
                    1.0 / self.scaling_factor
                } else {
                    1.0
                }
            }
        }
    }

    /// The magnitude scale folded into `cos`/`sin`, matching
    /// `llama_context`'s `cparams.yarn_attn_factor` computation.
    fn resolved_attn_factor(&self) -> f32 {
        let mut attn = 1.0f32;
        if self.effective_ext_factor() != 0.0 {
            let freq_scale = self.freq_scale();
            let factor = if freq_scale > 0.0 {
                1.0 / freq_scale
            } else {
                1.0
            };
            let get_mscale = |scale: f32, mscale: f32| {
                if scale <= 1.0 {
                    1.0
                } else {
                    0.1 * mscale * scale.ln() + 1.0
                }
            };
            if self.yarn_log_mul != 0.0 {
                // DeepSeek-V2/V3: mscale vs mscale_all_dim.
                let mscale_all_dims = self.yarn_log_mul;
                let mscale = mscale_all_dims;
                attn = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);
            } else {
                attn = get_mscale(factor, 1.0);
            }
            // `rope_yarn` re-applies `1 + 0.1*ln(factor)`, so cancel it here
            // exactly as llama.cpp does.
            attn *= 1.0 / (1.0 + 0.1 * factor.ln());
        }
        attn * self.attn_factor
    }
}

/// Compute Llama-3.1/3.2 per-rotation-pair frequency factors.
///
/// This is a direct port of `convert_hf_to_gguf.py`'s
/// `generate_extra_tensors()` for `rope_type == "llama3"`, which is exactly what
/// the official converter bakes into `rope_freqs.weight`.
///
/// Returns `head_dim / 2` factors; frequency `i` is divided by `factors[i]`.
///
/// # Arguments
/// * `head_dim` – rotary dimension count (`n_rot`).
/// * `base` – RoPE base frequency (`rope_theta`).
/// * `factor` – HF `rope_scaling.factor` (Llama-3.1 uses 8.0).
/// * `low_freq_factor` – HF `rope_scaling.low_freq_factor` (default 1.0).
/// * `high_freq_factor` – HF `rope_scaling.high_freq_factor` (default 4.0).
/// * `original_context` – HF `original_max_position_embeddings` (default 8192).
pub fn llama3_freq_factors(
    head_dim: usize,
    base: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_context: u32,
) -> Vec<f32> {
    let half = head_dim / 2;
    let old_ctx = original_context as f32;
    let low_freq_wavelen = if low_freq_factor > 0.0 {
        old_ctx / low_freq_factor
    } else {
        f32::INFINITY
    };
    let high_freq_wavelen = if high_freq_factor > 0.0 {
        old_ctx / high_freq_factor
    } else {
        f32::INFINITY
    };

    (0..half)
        .map(|i| {
            let freq = base_freq(i, head_dim, base);
            let wavelen = 2.0 * PI / freq;
            if wavelen < high_freq_wavelen {
                1.0
            } else if wavelen > low_freq_wavelen {
                factor
            } else {
                let denom = (high_freq_factor - low_freq_factor).abs().max(1e-6);
                let smooth = (old_ctx / wavelen - low_freq_factor) / denom;
                1.0 / ((1.0 - smooth) / factor + smooth)
            }
        })
        .collect()
}

/// ggml's `ggml_rope_yarn_corr_dim`.
///
/// Solves `n_rot = 2π · x · base^(2·max_pos / n_dims)` for the rotation-pair
/// index `x`.
fn yarn_corr_dim(n_dims: usize, n_ctx_orig: u32, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * PI)).ln() / (2.0 * base.ln())
}

/// ggml's `ggml_rope_yarn_corr_dims`: the `[low, high]` ramp endpoints in
/// rotation-pair-index space.
fn yarn_corr_dims(
    n_dims: usize,
    n_ctx_orig: u32,
    base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> (f32, f32) {
    let start = yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, base).floor();
    let end = yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, base).ceil();
    (start.max(0.0), end.min(n_dims as f32 - 1.0))
}

/// ggml's `rope_yarn_ramp`, expressed in rotation-pair-index space
/// (ggml passes `i0` and divides by 2 internally).
///
/// Returns `1` for pair indices at or below `low` (**high** frequency →
/// extrapolate) and `0` at or above `high` (**low** frequency → interpolate).
fn yarn_ramp(low: f32, high: f32, pair_index: usize) -> f32 {
    let y = (pair_index as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// Compute the unscaled base frequency for rotation pair `i`.
#[inline]
fn base_freq(i: usize, head_dim: usize, base: f32) -> f32 {
    if head_dim == 0 {
        return 1.0;
    }
    1.0 / base.powf((2 * i) as f32 / head_dim as f32)
}

/// Per-pair theta coefficient (multiply by position to get the angle) and the
/// magnitude scale, following ggml's `rope_yarn` exactly.
fn yarn_theta_coeff(
    i: usize,
    head_dim: usize,
    base: f32,
    params: &RopeParams,
    corr: Option<(f32, f32)>,
) -> (f32, f32) {
    let ff = params
        .freq_factors
        .as_ref()
        .and_then(|f| f.get(i).copied())
        .filter(|v| *v != 0.0)
        .unwrap_or(1.0);

    let theta_extrap = base_freq(i, head_dim, base) / ff;
    let freq_scale = params.freq_scale();
    let theta_interp = freq_scale * theta_extrap;

    let ext_factor = params.effective_ext_factor();
    let mut mscale = params.resolved_attn_factor();

    if ext_factor != 0.0 {
        let (low, high) = corr.unwrap_or((0.0, 0.0));
        let ramp_mix = yarn_ramp(low, high, i) * ext_factor;
        let theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
        (theta, mscale)
    } else {
        (theta_interp, mscale)
    }
}

/// Precomputed RoPE frequency table.
///
/// Contains cos/sin values for each position and head dimension pair.
/// These are computed once during model loading and reused for every token.
///
/// The YaRN magnitude scale (`mscale`) is folded into the stored `cos`/`sin`
/// values, matching ggml's `rope_yarn`.
#[derive(Debug, Clone)]
pub struct RopeTable {
    /// Cosine values: [max_seq_len, head_dim / 2].
    pub cos: Vec<f32>,
    /// Sine values: [max_seq_len, head_dim / 2].
    pub sin: Vec<f32>,
    /// Half the head dimension (number of rotation pairs).
    pub half_dim: usize,
    /// Maximum precomputed sequence length.
    pub max_seq_len: usize,
    /// Element-pairing convention used by [`Self::apply`].
    style: RopeStyle,
    /// The magnitude scale folded into `cos`/`sin` (1.0 unless YaRN is active).
    mscale: f32,
}

impl RopeTable {
    /// Precompute the RoPE frequency table with a given scaling strategy.
    ///
    /// Uses the [`RopeStyle::Neox`] pairing convention.  Architectures in
    /// llama.cpp's `LLAMA_ROPE_TYPE_NORM` family **must** use
    /// [`Self::new_with_style`] or [`Self::new_with_params`] instead.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head.
    /// * `max_seq_len` - Maximum sequence length to precompute.
    /// * `base` - Base frequency (typically 10000.0).
    /// * `scaling_type` - Which scaling strategy to apply.
    /// * `scaling_factor` - Scaling multiplier (`1.0` = no scaling).
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        scaling_type: RopeScalingType,
        scaling_factor: f32,
    ) -> Self {
        Self::new_with_style(
            head_dim,
            max_seq_len,
            base,
            scaling_type,
            scaling_factor,
            RopeStyle::Neox,
        )
    }

    /// Precompute a RoPE table with an explicit pairing convention.
    ///
    /// Equivalent to [`Self::new_with_params`] with default YaRN parameters;
    /// the signature stays infallible because the only error source
    /// (`freq_factors` length) cannot arise here.
    pub fn new_with_style(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        scaling_type: RopeScalingType,
        scaling_factor: f32,
        style: RopeStyle,
    ) -> Self {
        let params = RopeParams {
            style,
            scaling_type,
            scaling_factor,
            ..RopeParams::default()
        };
        match Self::build(head_dim, max_seq_len, base, &params) {
            Ok(table) => table,
            // Unreachable: `freq_factors` is None, the sole error source.
            Err(_) => Self {
                cos: Vec::new(),
                sin: Vec::new(),
                half_dim: 0,
                max_seq_len: 0,
                style,
                mscale: 1.0,
            },
        }
    }

    /// Precompute a RoPE table from fully-resolved [`RopeParams`].
    ///
    /// This is the reference-faithful entry point: it implements ggml's
    /// `ggml_rope_cache_init` + `rope_yarn` including `freq_factors`, the
    /// correction-dimension ramp and the `mscale` magnitude correction.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] if `params.freq_factors` is present with a
    /// length other than `head_dim / 2`.
    pub fn new_with_params(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        params: &RopeParams,
    ) -> ArchResult<Self> {
        Self::build(head_dim, max_seq_len, base, params)
    }

    fn build(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        params: &RopeParams,
    ) -> ArchResult<Self> {
        let half_dim = head_dim / 2;

        if let Some(factors) = params.freq_factors.as_ref() {
            if factors.len() != half_dim {
                return Err(ArchError::InvalidShape {
                    name: "rope.freq_factors".to_string(),
                    expected: vec![half_dim],
                    got: vec![factors.len()],
                });
            }
        }

        // llama.cpp falls back to the training context when
        // `n_ctx_orig_yarn == 0`; `max_seq_len` is the best local proxy.
        let n_ctx_orig = if params.original_context > 0 {
            params.original_context
        } else {
            max_seq_len.max(1) as u32
        };

        let corr = if params.effective_ext_factor() != 0.0 && head_dim > 0 {
            Some(yarn_corr_dims(
                head_dim,
                n_ctx_orig,
                base,
                params.beta_fast,
                params.beta_slow,
            ))
        } else {
            None
        };

        let mut coeffs = Vec::with_capacity(half_dim);
        let mut mscale = 1.0f32;
        for i in 0..half_dim {
            let (theta_coeff, m) = yarn_theta_coeff(i, head_dim, base, params, corr);
            coeffs.push(theta_coeff);
            mscale = m;
        }

        let mut cos = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin = Vec::with_capacity(max_seq_len * half_dim);
        for pos in 0..max_seq_len {
            for &coeff in &coeffs {
                let theta = pos as f32 * coeff;
                cos.push(theta.cos() * mscale);
                sin.push(theta.sin() * mscale);
            }
        }

        Ok(Self {
            cos,
            sin,
            half_dim,
            max_seq_len,
            style: params.style,
            mscale,
        })
    }

    /// Create a standard (unscaled) NeoX RoPE table.
    pub fn new_standard(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        Self::new(head_dim, max_seq_len, base, RopeScalingType::Standard, 1.0)
    }

    /// Create a standard (unscaled) RoPE table with an explicit pairing style.
    pub fn new_standard_with_style(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        style: RopeStyle,
    ) -> Self {
        Self::new_with_style(
            head_dim,
            max_seq_len,
            base,
            RopeScalingType::Standard,
            1.0,
            style,
        )
    }

    /// The element-pairing convention this table rotates with.
    #[inline]
    pub fn style(&self) -> RopeStyle {
        self.style
    }

    /// The YaRN magnitude scale folded into `cos`/`sin` (1.0 when inactive).
    ///
    /// DeepSeek-V2/V3 additionally applies `mscale` to the attention softmax
    /// scale; architectures that need it read it from here.
    #[inline]
    pub fn mscale(&self) -> f32 {
        self.mscale
    }

    /// Apply RoPE to a single head vector in-place.
    ///
    /// **Never panics and never reads out of bounds.**  A `position` beyond
    /// [`Self::max_seq_len`], or a vector shorter than `2 * half_dim`, leaves
    /// `x` untouched.  Use [`Self::try_apply`] when the caller wants the
    /// overflow reported instead of ignored — that is the correct choice at the
    /// top of a `forward()` where an over-long prompt would otherwise silently
    /// lose its positional encoding.
    ///
    /// # Arguments
    /// * `x` - Head vector of length `head_dim` (only the first `2 * half_dim`
    ///   elements are rotated, so partial-rotary models can pass the full head).
    /// * `position` - Sequence position index.
    pub fn apply(&self, x: &mut [f32], position: usize) {
        // The error is intentionally discarded: this signature is used by every
        // architecture's inner attention loop and must stay infallible.
        let _ = self.try_apply(x, position);
    }

    /// Apply RoPE to a single head vector in-place, reporting bounds failures.
    ///
    /// # Errors
    ///
    /// * [`ArchError::ConfigMismatch`] if `position >= max_seq_len` — i.e. the
    ///   prompt is longer than the precomputed table.  Before this check
    ///   existed the index arithmetic ran off the end of `self.cos` and
    ///   aborted the process, which is reachable from HTTP-server input.
    /// * [`ArchError::InvalidShape`] if `x` is shorter than `2 * half_dim`.
    pub fn try_apply(&self, x: &mut [f32], position: usize) -> ArchResult<()> {
        let half = self.half_dim;
        if half == 0 {
            return Ok(());
        }
        if position >= self.max_seq_len {
            return Err(ArchError::ConfigMismatch {
                param: "rope.position".to_string(),
                expected: format!("< max_seq_len ({})", self.max_seq_len),
                got: position.to_string(),
            });
        }
        if x.len() < 2 * half {
            return Err(ArchError::InvalidShape {
                name: "rope.head_vector".to_string(),
                expected: vec![2 * half],
                got: vec![x.len()],
            });
        }

        let offset = position * half;
        let (Some(cos), Some(sin)) = (
            self.cos.get(offset..offset + half),
            self.sin.get(offset..offset + half),
        ) else {
            return Err(ArchError::InvalidShape {
                name: "rope.table".to_string(),
                expected: vec![self.max_seq_len * half],
                got: vec![self.cos.len()],
            });
        };

        match self.style {
            RopeStyle::Neox => {
                for i in 0..half {
                    let x0 = x[i];
                    let x1 = x[i + half];
                    let c = cos[i];
                    let s = sin[i];
                    x[i] = x0 * c - x1 * s;
                    x[i + half] = x0 * s + x1 * c;
                }
            }
            RopeStyle::Norm => {
                for i in 0..half {
                    let x0 = x[2 * i];
                    let x1 = x[2 * i + 1];
                    let c = cos[i];
                    let s = sin[i];
                    x[2 * i] = x0 * c - x1 * s;
                    x[2 * i + 1] = x0 * s + x1 * c;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_position_zero() {
        let table = RopeTable::new(4, 16, 10000.0, RopeScalingType::Standard, 1.0);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let original = x.clone();
        table.apply(&mut x, 0);

        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
        }
    }

    #[test]
    fn standard_and_linear_scale_1_match() {
        let t1 = RopeTable::new(64, 8, 10000.0, RopeScalingType::Standard, 1.0);
        let t2 = RopeTable::new(64, 8, 10000.0, RopeScalingType::Linear, 1.0);
        for (a, b) in t1.cos.iter().zip(t2.cos.iter()) {
            assert!((a - b).abs() < 1e-6, "cos mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn linear_scaling_extends_freq_range() {
        let standard = RopeTable::new(64, 8, 10000.0, RopeScalingType::Standard, 1.0);
        let linear = RopeTable::new(64, 8, 10000.0, RopeScalingType::Linear, 4.0);
        let differs = standard
            .cos
            .iter()
            .zip(linear.cos.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "Linear-4x should differ from standard");
    }

    #[test]
    fn yarn_produces_valid_values() {
        let yarn = RopeTable::new(64, 16, 10000.0, RopeScalingType::Yarn, 4.0);
        // With YaRN the mscale magnitude correction lifts |cos| slightly above 1.
        let bound = yarn.mscale().max(1.0) + 1e-6;
        for v in yarn.cos.iter().chain(yarn.sin.iter()) {
            assert!(v.abs() <= bound, "out of range: {v} (bound {bound})");
        }
    }

    #[test]
    fn apply_in_place_unchanged_shape() {
        let table = RopeTable::new_standard(8, 4, 10000.0);
        let mut x = vec![1.0f32; 8];
        table.apply(&mut x, 0);
        assert_eq!(x.len(), 8);
    }

    #[test]
    fn standard_base_frequency_matches_formula() {
        let table = RopeTable::new_standard(64, 4, 10000.0);
        // cos/sin at position 1 encode the raw frequency directly.
        let want = 1.0f32 / 10000.0f32.powf(6.0 / 64.0);
        let got = table.cos[table.half_dim + 3].acos();
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}

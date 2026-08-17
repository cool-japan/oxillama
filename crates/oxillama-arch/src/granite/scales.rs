//! The four scalar multipliers that make Granite different from LLaMA.
//!
//! Granite-3.x is LLaMA topology plus four hyper-parameters that llama.cpp's
//! `LLM_ARCH_GRANITE` reads in `llama_model::load_hparams`
//! (`src/llama-model.cpp`, the `case LLM_ARCH_GRANITE:` arm) and applies in
//! `src/models/granite.cpp`.  Reading them and dropping them produces a model
//! that runs, returns finite logits, and is wrong — which is exactly what this
//! crate did before this module existed.
//!
//! # GGUF keys (verified against `gguf-py/gguf/constants.py`)
//!
//! | Field | GGUF key | `constants.py` |
//! |-------|----------|----------------|
//! | [`GraniteScales::embedding`] | `granite.embedding_scale` | `EMBEDDING_SCALE = "{arch}.embedding_scale"` |
//! | [`GraniteScales::residual`]  | `granite.residual_scale`  | `RESIDUAL_SCALE = "{arch}.residual_scale"` |
//! | [`GraniteScales::attention`] | `granite.attention.scale` | `Attention.SCALE = "{arch}.attention.scale"` |
//! | [`GraniteScales::logit`]     | `granite.logit_scale`     | `LOGIT_SCALE = "{arch}.logit_scale"` |
//!
//! Note the third row: the attention multiplier lives under the
//! **`attention.` sub-namespace** (`llama-arch.cpp`:
//! `{ LLM_KV_ATTENTION_SCALE, "%s.attention.scale" }`), not `granite.attention_scale`.
//!
//! # Sentinel semantics
//!
//! llama.cpp guards three of the four on "is it non-zero", so a literal `0.0`
//! in the file means *no scaling*, never *multiply the branch by zero*:
//!
//! ```text
//! // src/llama-graph.cpp (build_inp_embd)
//! if (hparams.f_embedding_scale != 0.0f) { cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale); }
//! // src/models/granite.cpp (build_layer_ffn), at BOTH residual adds
//! if (hparams.f_residual_scale)          { cur = ggml_scale(ctx0, cur, hparams.f_residual_scale); }
//! // src/models/granite.cpp (build_attention_layer)
//! const float kq_scale = hparams.f_attention_scale == 0.0f
//!     ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
//! ```
//!
//! # `logit_scale` is a DIVISOR
//!
//! `src/models/granite.cpp` ends with
//!
//! ```text
//! // For Granite architectures - scale logits
//! cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
//! ```
//!
//! — a **reciprocal**.  `convert_hf_to_gguf.py`'s `GraniteModel` writes the raw
//! HF `logits_scaling` through `add_logit_scale()` with no inversion, and HF's
//! `GraniteForCausalLM` does `logits = logits / self.config.logits_scaling`.
//! All three agree: a larger `granite.logit_scale` makes the logits *smaller*.
//!
//! This is the opposite of `LLM_ARCH_COMMAND_R`, which multiplies by
//! `f_logit_scale` — and it is why [`GraniteModel`](super::GraniteModel) reads
//! the key itself instead of using `ModelConfig::logit_scale`, which the shared
//! parser fills for every architecture and whose only other consumer treats it
//! as a multiplier.

use oxillama_gguf::{MetadataStore, MetadataValue};

/// The GGUF key suffixes, spelled exactly as `gguf-py` writes them.
const KEY_EMBEDDING_SCALE: &str = "embedding_scale";
const KEY_RESIDUAL_SCALE: &str = "residual_scale";
const KEY_ATTENTION_SCALE: &str = "attention.scale";
const KEY_LOGIT_SCALE: &str = "logit_scale";
const KEY_ROPE_FINETUNED: &str = "rope.scaling.finetuned";

/// Read `{arch}.{suffix}` as an `f32`, accepting both float widths.
///
/// `MetadataValue::as_f32` only matches `Float32`; a writer that emitted these
/// as `Float64` would otherwise silently fall through to the default.
fn get_float(metadata: &MetadataStore, arch: &str, suffix: &str) -> Option<f32> {
    match metadata.get(&format!("{arch}.{suffix}"))? {
        MetadataValue::Float32(v) => Some(*v),
        MetadataValue::Float64(v) => Some(*v as f32),
        _ => None,
    }
}

/// Granite's four scalar multipliers, resolved from GGUF metadata.
///
/// Every field keeps llama.cpp's raw value including its sentinel, so the
/// accessors below are the only place the "0.0 means off" convention lives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraniteScales {
    /// `granite.embedding_scale` — multiplies the token embedding.
    /// `0.0` (or absent) disables the scaling.
    pub embedding: f32,
    /// `granite.residual_scale` — multiplies the branch output at **both**
    /// residual adds (post-attention and post-FFN).
    /// `0.0` (or absent) disables the scaling.
    pub residual: f32,
    /// `granite.attention.scale` — replaces `1/sqrt(head_dim)` as the softmax
    /// scale.  `0.0` (or absent) falls back to `1/sqrt(head_dim)`.
    pub attention: f32,
    /// `granite.logit_scale` — the final logits are **divided** by this.
    /// Absent, `0.0`, or non-finite means `1.0` (identity).
    pub logit: f32,
    /// `granite.rope.scaling.finetuned` — Granite uses this as an on/off switch
    /// for RoPE and llama.cpp **defaults it to `true`**:
    ///
    /// ```text
    /// // Granite uses rope_finetuned as a switch for rope, so default to true
    /// bool rope_finetuned = true;
    /// ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    /// ```
    pub rope_finetuned: bool,
}

impl Default for GraniteScales {
    /// The identity configuration: no scaling anywhere, RoPE on.
    ///
    /// This is what a GGUF that carries none of the four keys resolves to, and
    /// it makes Granite numerically identical to LLaMA — the correct behaviour,
    /// since llama.cpp's `hparams` defaults are `0.0` for all four.
    fn default() -> Self {
        Self {
            embedding: 0.0,
            residual: 0.0,
            attention: 0.0,
            logit: 1.0,
            rope_finetuned: true,
        }
    }
}

impl GraniteScales {
    /// Extract the multipliers from a GGUF metadata table.
    ///
    /// `arch` is the `general.architecture` string the keys are namespaced
    /// under — normally `"granite"`.  Missing keys keep their
    /// [`Default`] value, mirroring `ml.get_key(..)`'s behaviour for an
    /// architecture whose hparams start at zero.
    ///
    /// A `logit` of `0.0` or a non-finite `logit` is folded to `1.0`: llama.cpp
    /// marks the key *required* for `LLM_ARCH_GRANITE` and so never evaluates
    /// `1.0 / 0.0`, whereas this loader accepts checkpoints that omit it and
    /// must not hand back `inf` logits.
    pub fn from_metadata(metadata: &MetadataStore, arch: &str) -> Self {
        let logit = match get_float(metadata, arch, KEY_LOGIT_SCALE) {
            Some(v) if v != 0.0 && v.is_finite() => v,
            _ => 1.0,
        };
        let rope_finetuned = metadata
            .get(&format!("{arch}.{KEY_ROPE_FINETUNED}"))
            .and_then(MetadataValue::as_bool)
            .unwrap_or(true);

        Self {
            embedding: get_float(metadata, arch, KEY_EMBEDDING_SCALE).unwrap_or(0.0),
            residual: get_float(metadata, arch, KEY_RESIDUAL_SCALE).unwrap_or(0.0),
            attention: get_float(metadata, arch, KEY_ATTENTION_SCALE).unwrap_or(0.0),
            logit,
            rope_finetuned,
        }
    }

    /// Whether any of the four multipliers is actually active.
    ///
    /// Used only for diagnostics: a Granite checkpoint with all four inert is
    /// indistinguishable from a LLaMA one.
    pub fn is_identity(&self) -> bool {
        self.embedding == 0.0 && self.residual == 0.0 && self.attention == 0.0 && self.logit == 1.0
    }

    /// Apply `embedding_scale` to a freshly looked-up embedding row.
    ///
    /// Mirrors `build_inp_embd`'s `if (f_embedding_scale != 0.0f)` guard.
    pub fn scale_embedding(&self, x: &mut [f32]) {
        if self.embedding == 0.0 {
            return;
        }
        for v in x.iter_mut() {
            *v *= self.embedding;
        }
    }

    /// The factor a residual **branch** output is multiplied by before it is
    /// added back into the stream.
    ///
    /// `1.0` when the key is absent or zero, so the caller can always multiply
    /// unconditionally and stay bit-identical to the unscaled path
    /// (`x * 1.0 == x` for every finite `f32`).
    pub fn residual_factor(&self) -> f32 {
        if self.residual == 0.0 {
            1.0
        } else {
            self.residual
        }
    }

    /// The attention softmax scale for a model with `head_dim`-wide heads.
    ///
    /// Reproduces `build_attention_layer`'s
    /// `f_attention_scale == 0.0f ? 1.0f/sqrtf(n_embd_head) : f_attention_scale`.
    /// A zero `head_dim` yields `1.0` rather than `inf`.
    pub fn attention_scale(&self, head_dim: usize) -> f32 {
        if self.attention != 0.0 {
            return self.attention;
        }
        if head_dim == 0 {
            return 1.0;
        }
        1.0 / (head_dim as f32).sqrt()
    }

    /// Divide the final logits by `logit_scale`, in place.
    ///
    /// **Division**, matching `ggml_scale(ctx0, cur, 1.0f / f_logit_scale)`.
    /// The reciprocal is taken once rather than per element.
    pub fn scale_logits(&self, logits: &mut [f32]) {
        if self.logit == 1.0 {
            return;
        }
        let inv = 1.0 / self.logit;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(pairs: &[(&str, MetadataValue)]) -> MetadataStore {
        let mut s = MetadataStore::new();
        for (k, v) in pairs {
            s.insert((*k).to_string(), v.clone());
        }
        s
    }

    #[test]
    fn absent_keys_are_the_llama_identity() {
        let scales = GraniteScales::from_metadata(&MetadataStore::new(), "granite");
        assert!(scales.is_identity());
        assert!(
            scales.rope_finetuned,
            "Granite defaults rope_finetuned=true"
        );

        let mut x = vec![1.0f32, -2.0, 3.0];
        let before = x.clone();
        scales.scale_embedding(&mut x);
        assert_eq!(x, before, "no embedding key must leave the row untouched");

        assert!((scales.residual_factor() - 1.0).abs() < f32::EPSILON);
        assert!((scales.attention_scale(64) - 0.125).abs() < 1e-6);

        let mut logits = vec![4.0f32, -8.0];
        scales.scale_logits(&mut logits);
        assert_eq!(logits, vec![4.0, -8.0]);
    }

    /// The exact key spellings, taken from `gguf-py/gguf/constants.py`.
    ///
    /// `attention.scale` is the trap: it is not `attention_scale`.
    #[test]
    fn keys_use_the_gguf_py_spelling() {
        let s = store(&[
            ("granite.embedding_scale", MetadataValue::Float32(12.0)),
            ("granite.residual_scale", MetadataValue::Float32(0.22)),
            ("granite.attention.scale", MetadataValue::Float32(0.0078125)),
            ("granite.logit_scale", MetadataValue::Float32(16.0)),
        ]);
        let scales = GraniteScales::from_metadata(&s, "granite");
        assert!((scales.embedding - 12.0).abs() < 1e-6);
        assert!((scales.residual - 0.22).abs() < 1e-6);
        assert!((scales.attention - 0.0078125).abs() < 1e-9);
        assert!((scales.logit - 16.0).abs() < 1e-6);
        assert!(!scales.is_identity());

        // The wrong spelling must NOT be picked up.
        let wrong = store(&[("granite.attention_scale", MetadataValue::Float32(0.5))]);
        assert_eq!(
            GraniteScales::from_metadata(&wrong, "granite").attention,
            0.0,
            "`granite.attention_scale` is not a GGUF key; only `granite.attention.scale` is"
        );
    }

    /// `logit_scale` divides.  This is the direction llama.cpp uses
    /// (`ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale)`).
    #[test]
    fn logit_scale_divides_rather_than_multiplies() {
        let s = store(&[("granite.logit_scale", MetadataValue::Float32(2.0))]);
        let scales = GraniteScales::from_metadata(&s, "granite");
        let mut logits = vec![4.0f32, -8.0, 0.0];
        scales.scale_logits(&mut logits);
        assert_eq!(logits, vec![2.0, -4.0, 0.0], "logit_scale=2 must HALVE");
    }

    /// `1.0 / 0.0` must never reach the logits.
    #[test]
    fn zero_or_nonfinite_logit_scale_is_identity() {
        for value in [0.0f32, f32::INFINITY, f32::NAN] {
            let s = store(&[("granite.logit_scale", MetadataValue::Float32(value))]);
            let scales = GraniteScales::from_metadata(&s, "granite");
            assert_eq!(scales.logit, 1.0, "logit_scale={value} must fold to 1.0");
            let mut logits = vec![3.0f32];
            scales.scale_logits(&mut logits);
            assert_eq!(logits, vec![3.0]);
        }
    }

    /// A literal `0.0` means "off", not "multiply by zero".
    #[test]
    fn zero_sentinels_disable_rather_than_zero_the_branch() {
        let s = store(&[
            ("granite.embedding_scale", MetadataValue::Float32(0.0)),
            ("granite.residual_scale", MetadataValue::Float32(0.0)),
            ("granite.attention.scale", MetadataValue::Float32(0.0)),
        ]);
        let scales = GraniteScales::from_metadata(&s, "granite");

        let mut x = vec![5.0f32, -1.5];
        scales.scale_embedding(&mut x);
        assert_eq!(
            x,
            vec![5.0, -1.5],
            "embedding_scale=0 must not zero the row"
        );
        assert_eq!(scales.residual_factor(), 1.0);
        assert!((scales.attention_scale(16) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn attention_scale_overrides_one_over_sqrt_head_dim() {
        let s = store(&[("granite.attention.scale", MetadataValue::Float32(0.25))]);
        let scales = GraniteScales::from_metadata(&s, "granite");
        // 1/sqrt(64) would be 0.125; the explicit key wins.
        assert!((scales.attention_scale(64) - 0.25).abs() < 1e-6);
        // A zero head_dim must not divide by zero.
        let none = GraniteScales::default();
        assert_eq!(none.attention_scale(0), 1.0);
    }

    #[test]
    fn rope_finetuned_can_be_switched_off() {
        let s = store(&[("granite.rope.scaling.finetuned", MetadataValue::Bool(false))]);
        assert!(!GraniteScales::from_metadata(&s, "granite").rope_finetuned);
    }

    #[test]
    fn float64_metadata_is_accepted() {
        let s = store(&[("granite.embedding_scale", MetadataValue::Float64(12.0))]);
        let scales = GraniteScales::from_metadata(&s, "granite");
        assert!((scales.embedding - 12.0).abs() < 1e-6);
    }
}

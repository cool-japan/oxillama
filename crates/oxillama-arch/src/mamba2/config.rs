//! Mamba-2 hyper-parameters, parsed from GGUF metadata.
//!
//! The key names mirror `llama.cpp`'s `LLM_ARCH_MAMBA2` branch of
//! `llama_model::load_hparams` (`src/llama-model.cpp`, the `case
//! LLM_ARCH_MAMBA2` at line 1481):
//!
//! ```text
//! ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
//! ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
//! ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
//! ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);   // == n_head
//! ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
//! ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
//! ```
//!
//! The corresponding GGUF keys come from `src/llama-arch.cpp` lines 256-261 and
//! `gguf-py/gguf/constants.py` lines 211-216:
//! `{arch}.ssm.conv_kernel`, `{arch}.ssm.inner_size`, `{arch}.ssm.state_size`,
//! `{arch}.ssm.time_step_rank`, `{arch}.ssm.group_count`.
//!
//! # `time_step_rank` is the head count
//!
//! `build_mamba2_layer` reads `const int64_t n_head = hparams.ssm_dt_rank;` and
//! `head_dim = d_inner / n_head`.  `convert_hf_to_gguf.py::Mamba2Model`
//! confirms the writer side: `add_ssm_time_step_rank(self.d_inner // head_dim)`.
//! Mamba-**1**'s `dt_rank` (a low-rank projection width) is a different
//! quantity that happens to share the GGUF key.

use oxillama_gguf::MetadataStore;

use crate::error::{ArchError, ArchResult};

/// Mamba-2 model configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Mamba2Config {
    /// Hidden (model) dimension — GGUF `{arch}.embedding_length`.
    pub d_model: usize,
    /// Number of SSM blocks — GGUF `{arch}.block_count`.
    pub n_layer: usize,
    /// Inner SSM width — GGUF `{arch}.ssm.inner_size`.
    ///
    /// llama.cpp asserts `2 * n_embd == d_inner` for Mamba-2
    /// (`src/llama-model.cpp:4590`), but its own comment calls that a
    /// loader-side restriction — "only an expansion factor of 2 is supported
    /// for now" — rather than a property of the architecture.  [`Self::validate`]
    /// therefore **does not** enforce it: only the divisibility invariants that
    /// the kernels actually index by are checked.
    pub d_inner: usize,
    /// SSM state dimension — GGUF `{arch}.ssm.state_size`.
    pub d_state: usize,
    /// Convolution kernel width — GGUF `{arch}.ssm.conv_kernel`.
    pub d_conv: usize,
    /// Number of SSM heads — GGUF `{arch}.ssm.time_step_rank`.
    ///
    /// Mamba-2 has **one scalar `A` and one scalar `dt` per head**, not one per
    /// `(d_state, d_inner)` pair.
    pub n_head: usize,
    /// Number of B/C groups — GGUF `{arch}.ssm.group_count`
    /// (`LLM_KV_SSM_GROUP_COUNT`).
    pub n_group: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length — GGUF `{arch}.context_length`.
    pub max_seq_len: usize,
    /// RMSNorm epsilon — GGUF `{arch}.attention.layer_norm_rms_epsilon`.
    pub rms_norm_eps: f32,
}

impl Mamba2Config {
    /// Per-head channel width: `d_inner / n_head`.
    pub fn head_dim(&self) -> usize {
        self.d_inner.checked_div(self.n_head).unwrap_or(0)
    }

    /// Width of the tensor that flows through the depthwise convolution.
    ///
    /// `d_inner + 2 * n_group * d_state` — Mamba-2 pushes `x`, `B` **and** `C`
    /// through the same conv (`mamba-base.cpp`, `conv = ggml_reshape_3d(ctx0,
    /// conv, d_conv - 1, d_inner + 2*n_group*d_state, n_seqs)`).
    pub fn conv_dim(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }

    /// Output width of the fused `ssm_in` (`zxBCdt`) projection.
    ///
    /// `2*d_inner + 2*n_group*d_state + n_head`, matching
    /// `src/llama-model.cpp:4589`.
    pub fn d_in_proj(&self) -> usize {
        2 * self.d_inner + 2 * self.n_group * self.d_state + self.n_head
    }

    /// Total per-group channel count used by the gated `ssm_norm`.
    ///
    /// `ssm_norm.weight` has shape `{d_inner / n_group, n_group}`.
    pub fn norm_group_size(&self) -> usize {
        self.d_inner.checked_div(self.n_group).unwrap_or(0)
    }

    /// Number of `B`/`C` elements produced per token (`n_group * d_state`).
    pub fn bc_width(&self) -> usize {
        self.n_group * self.d_state
    }

    /// Check the divisibility invariants the Mamba-2 kernels rely on.
    ///
    /// `ggml_compute_forward_ssm_scan_f32` asserts `nh % ng == 0` and indexes
    /// heads as `d_inner / n_head` contiguous channels, so a config that fails
    /// these would index out of bounds rather than merely produce bad numbers.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] naming the violated invariant.
    pub fn validate(&self) -> ArchResult<()> {
        let bad = |detail: String| ArchError::InvalidConfig { detail };

        if self.d_model == 0 || self.n_layer == 0 || self.d_inner == 0 || self.d_state == 0 {
            return Err(bad(format!(
                "mamba2: d_model={}, n_layer={}, d_inner={}, d_state={} must all be non-zero",
                self.d_model, self.n_layer, self.d_inner, self.d_state
            )));
        }
        if self.d_conv == 0 {
            return Err(bad("mamba2: ssm.conv_kernel must be >= 1".to_string()));
        }
        if self.n_head == 0 {
            return Err(bad(
                "mamba2: ssm.time_step_rank (n_head) must be >= 1".to_string()
            ));
        }
        if self.n_group == 0 {
            return Err(bad(
                "mamba2: ssm.group_count (n_group) must be >= 1".to_string()
            ));
        }
        if !self.d_inner.is_multiple_of(self.n_head) {
            return Err(bad(format!(
                "mamba2: d_inner ({}) must be divisible by n_head ({})",
                self.d_inner, self.n_head
            )));
        }
        if !self.n_head.is_multiple_of(self.n_group) {
            return Err(bad(format!(
                "mamba2: n_head ({}) must be divisible by n_group ({}) \
                 (ggml_ssm_scan asserts nh % ng == 0)",
                self.n_head, self.n_group
            )));
        }
        if !self.d_inner.is_multiple_of(self.n_group) {
            return Err(bad(format!(
                "mamba2: d_inner ({}) must be divisible by n_group ({}) \
                 for the gated ssm_norm",
                self.d_inner, self.n_group
            )));
        }
        if self.vocab_size == 0 {
            return Err(bad("mamba2: vocab_size must be non-zero".to_string()));
        }
        if self.max_seq_len == 0 {
            return Err(bad("mamba2: context_length must be non-zero".to_string()));
        }
        Ok(())
    }

    /// Parse a `Mamba2Config` from GGUF metadata.
    ///
    /// Canonical llama.cpp keys are tried first.  The `mamba.*` prefix is
    /// accepted as an alias because Mamba-1 and Mamba-2 checkpoints share the
    /// `ssm.*` key layout, and the pre-0.1.4 `mamba2.d_model` / `mamba2.n_layer`
    /// spellings are still honoured so older fixtures keep loading.
    ///
    /// `vocab_size` is a *hint* here: the loader overrides it with the real
    /// `token_embd.weight` row count and rejects a mismatch.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] when the resulting hyper-parameters violate
    /// [`Self::validate`].
    pub fn from_metadata(metadata: &MetadataStore) -> ArchResult<Self> {
        let u = |keys: &[&str]| -> Option<usize> {
            keys.iter()
                .find_map(|k| metadata.get_u32(k).ok())
                .map(|v| v as usize)
        };
        let f =
            |keys: &[&str]| -> Option<f32> { keys.iter().find_map(|k| metadata.get_f32(k).ok()) };

        let d_model = u(&[
            "mamba2.embedding_length",
            "mamba.embedding_length",
            "mamba2.d_model",
            "mamba.d_model",
        ])
        .unwrap_or(128);

        let n_layer = u(&[
            "mamba2.block_count",
            "mamba.block_count",
            "mamba2.n_layer",
            "mamba.n_layer",
        ])
        .unwrap_or(24);

        let d_conv = u(&["mamba2.ssm.conv_kernel", "mamba.ssm.conv_kernel"]).unwrap_or(4);
        let d_state = u(&["mamba2.ssm.state_size", "mamba.ssm.state_size"]).unwrap_or(128);
        // `convert_hf_to_gguf.py::Mamba2Model` defaults d_inner to 2 * d_model.
        let d_inner = u(&["mamba2.ssm.inner_size", "mamba.ssm.inner_size"]).unwrap_or(2 * d_model);
        // head_dim defaults to 64 in the converter, so n_head = d_inner / 64.
        let n_head = u(&["mamba2.ssm.time_step_rank", "mamba.ssm.time_step_rank"])
            .unwrap_or_else(|| d_inner.div_ceil(64));
        let n_group = u(&["mamba2.ssm.group_count", "mamba.ssm.group_count"]).unwrap_or(1);

        let vocab_size = u(&[
            "mamba2.vocab_size",
            "mamba.vocab_size",
            "tokenizer.ggml.tokens.length",
        ])
        .unwrap_or(0);

        let max_seq_len = u(&["mamba2.context_length", "mamba.context_length"]).unwrap_or(4096);

        let rms_norm_eps = f(&[
            "mamba2.attention.layer_norm_rms_epsilon",
            "mamba.attention.layer_norm_rms_epsilon",
        ])
        .unwrap_or(1e-5);

        let cfg = Self {
            d_model,
            n_layer,
            d_inner,
            d_state,
            d_conv,
            n_head,
            n_group,
            // A zero here means "unknown"; the loader fills it from token_embd.
            vocab_size: vocab_size.max(1),
            max_seq_len,
            rms_norm_eps,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Mamba2Config {
        Mamba2Config {
            d_model: 16,
            n_layer: 1,
            d_inner: 32,
            d_state: 8,
            d_conv: 4,
            n_head: 4,
            n_group: 2,
            vocab_size: 256,
            max_seq_len: 512,
            rms_norm_eps: 1e-5,
        }
    }

    /// The derived widths must match `src/llama-model.cpp:4585`'s MAMBA2 block.
    #[test]
    fn derived_widths_match_llama_cpp() {
        let c = cfg();
        assert_eq!(c.head_dim(), 8, "head_dim = d_inner / n_head");
        assert_eq!(
            c.conv_dim(),
            32 + 2 * 2 * 8,
            "conv_dim = d_inner + 2*n_group*d_state"
        );
        assert_eq!(
            c.d_in_proj(),
            2 * 32 + 2 * 2 * 8 + 4,
            "d_in_proj = 2*d_inner + 2*n_group*d_state + n_head"
        );
        assert_eq!(c.norm_group_size(), 16, "ssm_norm group width");
        assert_eq!(c.bc_width(), 16, "n_group * d_state");
    }

    #[test]
    fn validate_rejects_indivisible_head_count() {
        let mut c = cfg();
        c.n_head = 5;
        let err = c.validate().expect_err("d_inner % n_head != 0 must fail");
        assert!(
            format!("{err}").contains("divisible by n_head"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_rejects_group_count_not_dividing_heads() {
        let mut c = cfg();
        c.n_group = 3;
        let err = c.validate().expect_err("n_head % n_group != 0 must fail");
        assert!(
            format!("{err}").contains("divisible by n_group"),
            "unexpected error: {err}"
        );
    }
}

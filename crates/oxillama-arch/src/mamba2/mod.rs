//! Mamba-2 selective-scan state-space model architecture.
//!
//! Mamba-2 is a purely recurrent sequence model: no self-attention, no KV cache
//! and **no RoPE at all** (`LLM_ARCH_MAMBA2` is absent from both RoPE tables in
//! `llama_model_rope_type()`, so it falls through to `LLAMA_ROPE_TYPE_NONE`).
//!
//! Registered under the GGUF architecture identifier `"mamba2"`.
//!
//! ## Sub-modules
//! - [`config`]: [`Mamba2Config`] and GGUF metadata parsing.
//! - [`conv`]: causal 1-D depthwise convolution, stateless and stateful.
//! - [`ssm`]: selective-scan primitives (Mamba-1 and Mamba-2 flavours).
//! - [`state`]: the per-layer convolution shift register.
//! - [`loader`][]: [`load_mamba2_from_gguf`].
//! - [`model`]: [`Mamba2Model`] and its `ForwardPass` impl.
//!
//! ## How Mamba-2 differs from Mamba-1
//!
//! | | Mamba-1 | Mamba-2 |
//! |---|---|---|
//! | `A` | `{d_state, d_inner}` | `{1, n_head}` — one scalar per head |
//! | `dt` | `ssm_dt` weight + bias, `{d_inner}` | bias only, `{n_head}` |
//! | `B`, `C` | projected from the post-conv activation via `ssm_x` | emitted by the fused `ssm_in` and passed **through** the conv |
//! | conv width | `d_inner` | `d_inner + 2*n_group*d_state` |
//! | `ssm_norm` | absent | gated group-RMSNorm on `y` before `ssm_out` |
//!
//! ## `blk.N.ssm_a` is `A`, not `log(A)`
//!
//! `convert_hf_to_gguf.py::Mamba2Model.modify_tensors` writes
//! `data_torch = -torch.exp(data_torch)` for `.A_log`, and
//! `ggml_compute_forward_ssm_scan_f32` uses the result directly
//! (`dA = expf(dt_soft_plus * A[h])`).  [`ssm::selective_scan_mamba2`]
//! therefore applies neither `exp` nor a negation to it.
//!
//! ## Sequence state
//!
//! Two tensors are carried per layer, mirroring llama.cpp's `get_s_l()` and
//! `get_r_l()`: the SSM hidden state (in
//! [`Mamba2SequenceState`](crate::common::sequence_state::Mamba2SequenceState))
//! and the convolution shift register (in [`state::Mamba2ConvCache`], because
//! `SsmLayerState` has no field for it).  Both are cleared by
//! [`ForwardPass::reset_sequence`].

pub mod config;
pub mod conv;
pub mod loader;
pub mod model;
pub mod ssm;
pub mod state;

pub use config::Mamba2Config;
pub use loader::load_mamba2_from_gguf;
pub use model::{build_mamba2_model, make_zero_mamba2_layer, Mamba2LayerWeights, Mamba2Model};
pub use state::{ConvRing, Mamba2ConvCache};

use crate::config::ModelConfig;
use crate::error::ArchResult;
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// Architecture plugin for Mamba-2 models.
///
/// Registered under the identifier `"mamba2"` (matching the GGUF
/// `general.architecture` value used in Mamba-2 GGUF files).
pub struct Mamba2Architecture;

impl Mamba2Architecture {
    /// Create a new `Mamba2Architecture` plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Mamba2Architecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for Mamba2Architecture {
    fn arch_id(&self) -> &str {
        "mamba2"
    }

    /// Not the loader entry point.
    ///
    /// [`ModelArchitecture::build`] receives only the tensor *metadata* table,
    /// which is not enough to dequantize weights.  Use
    /// [`ModelArchitecture::build_from_gguf`], which this architecture
    /// implements.
    fn build(
        &self,
        _config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Err(crate::error::ArchError::NotSupported {
            detail: "mamba2: build() only sees tensor metadata; \
                     use build_from_gguf() (or load_mamba2_from_gguf())"
                .to_string(),
        })
    }

    /// Route the registry straight at [`load_mamba2_from_gguf`].
    fn build_from_gguf(
        &self,
        model: &GgufModel,
        _config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(load_mamba2_from_gguf(model)?))
    }

    /// The `LLM_ARCH_MAMBA2` tensor set from `src/llama-arch.cpp:1388`.
    ///
    /// Note the lowercase `ssm_a` / `ssm_d` (they carry no `.weight` suffix),
    /// the `ssm_dt` **bias** with no matching weight, and the absence of any
    /// `ssm_x` / separate B/C projection.
    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        let p = |pattern: &str, description: &str, required: bool| TensorNamePattern {
            pattern: pattern.to_string(),
            description: description.to_string(),
            required,
        };
        vec![
            p("token_embd.weight", "Token embedding table", true),
            p("output_norm.weight", "Final RMSNorm scale", true),
            // TENSOR_NOT_REQUIRED in llama.cpp: falls back to tied token_embd.
            p(
                "output.weight",
                "LM head projection (tied to token_embd when absent)",
                false,
            ),
            p("blk.*.attn_norm.weight", "Pre-block RMSNorm scale", true),
            p(
                "blk.*.ssm_in.weight",
                "Fused zxBCdt projection [n_embd, 2*d_inner + 2*n_group*d_state + n_head]",
                true,
            ),
            p(
                "blk.*.ssm_conv1d.weight",
                "Depthwise conv kernel [d_conv, d_inner + 2*n_group*d_state]",
                true,
            ),
            p(
                "blk.*.ssm_conv1d.bias",
                "Depthwise conv bias [d_inner + 2*n_group*d_state]",
                true,
            ),
            p("blk.*.ssm_dt.bias", "Per-head Δ bias [n_head]", true),
            p("blk.*.ssm_a", "Per-head A = -exp(A_log) [n_head]", true),
            p("blk.*.ssm_d", "Per-head skip connection D [n_head]", true),
            p(
                "blk.*.ssm_norm.weight",
                "Gated group-RMSNorm scale [d_inner/n_group, n_group]",
                true,
            ),
            p(
                "blk.*.ssm_out.weight",
                "SSM output projection [d_inner, n_embd]",
                true,
            ),
        ]
    }
}

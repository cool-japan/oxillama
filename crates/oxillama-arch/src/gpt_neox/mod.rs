//! GPT-NeoX model architecture (EleutherAI GPT-NeoX-20B, the Pythia suite,
//! Dolly-v2, RedPajama-INCITE, …).
//!
//! GPT-NeoX introduced the **parallel residual** block later adopted by
//! StableLM, Falcon and PaLM.  This implementation is transcribed from
//! `~/work/refs/llama.cpp`:
//!
//! * `src/models/gptneox.cpp` — `llm_build_gptneox`, the graph.
//! * `src/llama-model.cpp` — `case LLM_ARCH_GPTNEOX:` in `load_hparams()`
//!   (`f_norm_eps`, `use_par_res`) and in `load_tensors()` (the tensor list).
//! * `convert_hf_to_gguf.py` — `class GPTNeoXModel` (metadata keys and the
//!   one-time QKV de-interleave).
//!
//! ## Key architectural features
//!
//! 1. **Parallel residual (selectable).**  `{arch}.use_parallel_residual`
//!    picks between `x = ffn(ffn_norm(x)) + x + attn(attn_norm(x))` and the
//!    sequential pre-norm block.  In the parallel form both norms read the
//!    *same* original residual stream.
//! 2. **Fused QKV.**  One `blk.{i}.attn_qkv` tensor per layer; its output
//!    splits into contiguous `Q | K | V` blocks.  There are no separate
//!    `attn_q`/`attn_k`/`attn_v` tensors.
//! 3. **Bias on every projection.**  QKV, output, FFN up, FFN down and all
//!    three LayerNorms carry a required bias.
//! 4. **Plain LayerNorm**, not RMSNorm.
//! 5. **Partial NeoX RoPE.**  Only the leading `{arch}.rope.dimension_count`
//!    (`rotary_pct × head_dim`, 0.25 for Pythia) elements of each head are
//!    rotated, and the frequency ladder is derived from that count — not from
//!    `head_dim`.
//! 6. **Gate-free GELU FFN.**
//!
//! ## Tensor naming convention (GGUF)
//!
//! See [`tensor_names`] for the full, citation-annotated list:
//!
//! - `token_embd.weight`
//! - `blk.{i}.attn_norm.weight` / `.bias`
//! - `blk.{i}.attn_qkv.weight` / `.bias`   (fused)
//! - `blk.{i}.attn_output.weight` / `.bias`
//! - `blk.{i}.ffn_norm.weight` / `.bias`
//! - `blk.{i}.ffn_up.weight` / `.bias`
//! - `blk.{i}.ffn_down.weight` / `.bias`
//! - `output_norm.weight` / `.bias`
//! - `output.weight`

mod loader;
mod model;
pub mod tensor_names;

pub use loader::load_gpt_neox_from_gguf;
#[cfg(test)]
pub use model::{f32_linear, make_test_layer};
pub use model::{
    validate_gpt_neox_shapes, validate_rotary_dims, GptNeoxLayer, GptNeoxLayerWeights,
    GptNeoxModel, DEFAULT_USE_PARALLEL_RESIDUAL,
};
pub use tensor_names::gpt_neox_tensor_name_patterns;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// GPT-NeoX architecture plugin.
pub struct GptNeoxArchitecture;

impl GptNeoxArchitecture {
    /// Create a new `GptNeoxArchitecture` instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GptNeoxArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for GptNeoxArchitecture {
    fn arch_id(&self) -> &str {
        "gptneox"
    }

    /// Metadata-only build.
    ///
    /// [`TensorStore`] carries tensor *descriptors*, not payload, so no weight
    /// can be materialised here.  The configuration is still validated so a
    /// caller that only has metadata gets a useful diagnostic, and the error
    /// then points at [`Self::build_from_gguf`] — which is what the registry
    /// and the runtime engine actually call.
    fn build(
        &self,
        config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        validate_gpt_neox_shapes(config)?;
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use GptNeoxArchitecture::build_from_gguf \
                   or load_gpt_neox_from_gguf for full loading)"
                .to_string(),
        })
    }

    fn build_from_gguf(
        &self,
        model: &GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(load_gpt_neox_from_gguf(model, config)?))
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        gpt_neox_tensor_name_patterns()
    }
}

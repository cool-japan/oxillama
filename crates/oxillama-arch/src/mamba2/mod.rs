//! Mamba-2 selective-scan state-space model architecture.
//!
//! Mamba-2 is a purely recurrent sequence model that does not use self-attention.
//! Each block applies a selective scan (SSM) with learned state-selection matrices
//! (B, C, Δ) that allow the model to selectively absorb or ignore information at
//! each step.
//!
//! Registered under the GGUF architecture identifier `"mamba2"`.
//!
//! ## Sub-modules
//! - [`conv`]: Causal 1-D depthwise convolution with SiLU activation.
//! - [`ssm`]: Sequential selective-scan primitive.
//! - [`model`]: `Mamba2Model` full transformer + `ForwardPass` impl.
//!
//! ## Key mathematical constraint
//! The `A` parameter is stored in GGUF as `log(A)`. The discrete-time transition
//! matrix is computed as `A_disc = exp(-Δ * exp(log_A))`, **not**
//! `exp(-Δ * log_A)`.
//!
//! ## Sequence state
//! Mamba-2 uses `common::sequence_state::Mamba2SequenceState` (from the E1 item)
//! to carry recurrent hidden states across tokens. The state is **arch-internal**
//! and not exposed through `KvCacheAccess`.

pub mod conv;
pub mod model;
pub mod ssm;

pub use model::{
    build_mamba2_model, load_mamba2_from_gguf, make_zero_mamba2_layer, Mamba2Config,
    Mamba2LayerWeights, Mamba2Model,
};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

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

    fn build(
        &self,
        _config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Err(ArchError::MissingTensor {
            name: "Mamba2Architecture::build() is not the loader entry point; \
                   call load_mamba2_from_gguf() instead"
                .to_string(),
        })
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        vec![
            TensorNamePattern {
                pattern: "token_embd.weight".to_string(),
                description: "Token embedding table".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.weight".to_string(),
                description: "Final RMSNorm scale".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ssm_in.weight".to_string(),
                description: "SSM combined gate+input projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ssm_conv1d.weight".to_string(),
                description: "1-D depthwise conv kernel".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ssm_conv1d.bias".to_string(),
                description: "1-D depthwise conv bias".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "blk.*.ssm_x.weight".to_string(),
                description: "x → B, C, Δ combined projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ssm_dt.weight".to_string(),
                description: "Δ projection weight".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ssm_dt.bias".to_string(),
                description: "Δ projection bias".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "blk.*.ssm_A".to_string(),
                description: "Log-parameterised A matrix".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ssm_D".to_string(),
                description: "Skip-connection D vector".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ssm_out.weight".to_string(),
                description: "SSM output projection".to_string(),
                required: true,
            },
        ]
    }
}

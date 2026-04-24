//! Jamba hybrid (LLaMA × Mamba-2) architecture.
//!
//! Jamba interleaves standard attention blocks with Mamba-2 SSM blocks on a
//! configurable period.  The first published version (AI21 Labs, 2024) uses a
//! 1:7 attention:SSM ratio with period 8.
//!
//! ## Sub-modules
//! - [`config`]: `JambaConfig`, `LayerKind` — configuration and layer layout.
//! - [`model`]: `JambaModel`, `JambaSequenceState`, builder helpers, stubs.
//!
//! ## Tensor naming (GGUF)
//!
//! Attention layers follow the LLaMA GGUF convention:
//! ```text
//! blk.{i}.attn_norm.weight
//! blk.{i}.attn_q.weight   blk.{i}.attn_k.weight  blk.{i}.attn_v.weight
//! blk.{i}.attn_output.weight
//! blk.{i}.ffn_norm.weight
//! blk.{i}.ffn_gate.weight  blk.{i}.ffn_up.weight  blk.{i}.ffn_down.weight
//! ```
//!
//! SSM layers follow the Mamba-2 GGUF convention:
//! ```text
//! blk.{i}.ssm_norm.weight
//! blk.{i}.ssm_in.weight   blk.{i}.ssm_conv1d.weight  blk.{i}.ssm_conv1d.bias
//! blk.{i}.ssm_x.weight    blk.{i}.ssm_dt.weight      blk.{i}.ssm_dt.bias
//! blk.{i}.ssm_A           blk.{i}.ssm_D              blk.{i}.ssm_out.weight
//! ```
//!
//! Global tensors:
//! ```text
//! token_embd.weight
//! output_norm.weight
//! output.weight
//! ```

pub mod config;
pub mod model;

pub use config::{JambaConfig, LayerKind};
pub use model::{
    build_zero_jamba_model, load_jamba_from_gguf, JambaLayerState, JambaLayerWeights, JambaModel,
    JambaSequenceState,
};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// Architecture plugin for Jamba hybrid models.
///
/// Registered under the identifier `"jamba"`.
pub struct JambaArchitecture;

impl JambaArchitecture {
    /// Create a new `JambaArchitecture` plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for JambaArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for JambaArchitecture {
    fn arch_id(&self) -> &str {
        "jamba"
    }

    fn build(
        &self,
        _config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        // Full loading goes through `load_jamba_from_gguf()`; this path
        // is a config-validation entry point.
        Err(ArchError::MissingTensor {
            name: "JambaArchitecture::build(): use load_jamba_from_gguf() for full loading"
                .to_string(),
        })
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        let mut patterns = vec![
            TensorNamePattern {
                pattern: "token_embd.weight".to_string(),
                description: "Token embedding matrix".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.weight".to_string(),
                description: "Final RMSNorm scale".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head / unembedding".to_string(),
                required: true,
            },
        ];

        // Attention layer tensors.
        for name in [
            "blk.{i}.attn_norm.weight",
            "blk.{i}.attn_q.weight",
            "blk.{i}.attn_k.weight",
            "blk.{i}.attn_v.weight",
            "blk.{i}.attn_output.weight",
            "blk.{i}.ffn_norm.weight",
            "blk.{i}.ffn_gate.weight",
            "blk.{i}.ffn_up.weight",
            "blk.{i}.ffn_down.weight",
        ] {
            patterns.push(TensorNamePattern {
                pattern: name.to_string(),
                description: "Jamba attention-block weight".to_string(),
                required: true,
            });
        }

        // SSM layer tensors.
        for (name, required) in [
            ("blk.{i}.ssm_norm.weight", true),
            ("blk.{i}.ssm_in.weight", true),
            ("blk.{i}.ssm_conv1d.weight", true),
            ("blk.{i}.ssm_conv1d.bias", false),
            ("blk.{i}.ssm_x.weight", true),
            ("blk.{i}.ssm_dt.weight", true),
            ("blk.{i}.ssm_dt.bias", false),
            ("blk.{i}.ssm_A", true),
            ("blk.{i}.ssm_D", true),
            ("blk.{i}.ssm_out.weight", true),
        ] {
            patterns.push(TensorNamePattern {
                pattern: name.to_string(),
                description: "Jamba SSM-block weight".to_string(),
                required,
            });
        }

        patterns
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelArchitecture;

    #[test]
    fn jamba_arch_id() {
        assert_eq!(JambaArchitecture::new().arch_id(), "jamba");
    }

    #[test]
    fn jamba_tensor_names_non_empty() {
        let arch = JambaArchitecture::new();
        assert!(!arch.tensor_names().is_empty());
    }

    #[test]
    fn jamba_tensor_names_include_token_embd() {
        let arch = JambaArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.contains("token_embd")),
            "should contain token_embd pattern"
        );
    }

    #[test]
    fn jamba_tensor_names_include_attn_and_ssm() {
        let arch = JambaArchitecture::new();
        let patterns: Vec<String> = arch.tensor_names().into_iter().map(|p| p.pattern).collect();
        assert!(
            patterns.iter().any(|p| p.contains("attn_q")),
            "should contain attn_q pattern"
        );
        assert!(
            patterns.iter().any(|p| p.contains("ssm_in")),
            "should contain ssm_in pattern"
        );
    }
}

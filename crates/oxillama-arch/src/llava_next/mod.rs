//! LLaVA-1.6 / LLaVA-NeXT architecture plugin.
//!
//! Registered under arch id `"llava16"`.
//!
//! ## Key differences from LLaVA-1.5
//!
//! LLaVA-1.5 encodes a single fixed 336×336 crop.  LLaVA-NeXT adds *anyres*
//! tiling: the image is divided into a variable grid of tiles (e.g. 2×2,
//! 1×3, 3×1) **plus** a global 336×336 thumbnail.  Each tile is processed
//! independently by the same CLIP vision encoder, and all tile features plus
//! the thumbnail features are concatenated before the multi-modal projector.
//! This yields a much higher effective resolution without changing the
//! per-tile encoder weights.
//!
//! ## Module layout
//!
//! - [`tiler`] — [`AnyresTileConfig`]: grid selection and image → tile splitting.
//! - [`model`] — [`LlavaNextModel`]: loads weights, runs the anyres pipeline.
//! - This file — [`LlavaNextArchitecture`]: trait impl for the plugin registry.

pub mod model;
pub mod tiler;

pub use model::LlavaNextModel;
pub use tiler::AnyresTileConfig;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// LLaVA-NeXT architecture plugin, registered under arch id `"llava16"`.
pub struct LlavaNextArchitecture;

impl LlavaNextArchitecture {
    /// Construct a new `LlavaNextArchitecture` plugin.
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlavaNextArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for LlavaNextArchitecture {
    fn arch_id(&self) -> &str {
        "llava16"
    }

    /// Validate configuration and signal that tensor loading must go through
    /// [`LlavaNextModel::load`].
    ///
    /// This path validates the config, then returns `Err(MissingTensor)` so
    /// that callers know to use the full `GgufModel`-based loader instead.
    fn build(
        &self,
        config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        if config.num_attention_heads == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "num_attention_heads".to_string(),
                expected: ">0".to_string(),
                got: "0".to_string(),
            });
        }
        if config.hidden_size == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "hidden_size".to_string(),
                expected: ">0".to_string(),
                got: "0".to_string(),
            });
        }

        // Full loading goes through LlavaNextModel::load(gguf, config).
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use LlavaNextModel::load for full loading)".to_string(),
        })
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        // LLaVA-NeXT uses the same tensor naming as LLaVA-1.5:
        // - Language backbone: identical to LLaMA
        // - Vision encoder: `v.*` prefix
        // - MM projector: `mm.0.*`, `mm.2.*`
        let mut patterns = vec![
            TensorNamePattern {
                pattern: "token_embd.weight".to_string(),
                description: "Token embedding matrix".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.weight".to_string(),
                description: "Final RMSNorm".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head / unembedding".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "mm.0.weight".to_string(),
                description: "MM projector FC1 weight".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "mm.0.bias".to_string(),
                description: "MM projector FC1 bias".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "mm.2.weight".to_string(),
                description: "MM projector FC2 weight".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "mm.2.bias".to_string(),
                description: "MM projector FC2 bias".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "v.patch_embd.weight".to_string(),
                description: "CLIP patch embedding weight".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "v.position_embd.weight".to_string(),
                description: "CLIP position embedding".to_string(),
                required: false,
            },
        ];

        // Per-layer LLaMA backbone tensors.
        let layer_tensors = [
            ("blk.{i}.attn_norm.weight", "Pre-attention RMSNorm"),
            ("blk.{i}.attn_q.weight", "Query projection"),
            ("blk.{i}.attn_k.weight", "Key projection"),
            ("blk.{i}.attn_v.weight", "Value projection"),
            ("blk.{i}.attn_output.weight", "Attention output projection"),
            ("blk.{i}.ffn_norm.weight", "Pre-FFN RMSNorm"),
            ("blk.{i}.ffn_gate.weight", "FFN gate projection"),
            ("blk.{i}.ffn_up.weight", "FFN up projection"),
            ("blk.{i}.ffn_down.weight", "FFN down projection"),
        ];

        for (pat, desc) in layer_tensors {
            patterns.push(TensorNamePattern {
                pattern: pat.to_string(),
                description: desc.to_string(),
                required: true,
            });
        }

        patterns
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::registry::ArchitectureRegistry;
    use oxillama_gguf::{MetadataStore, MetadataValue, TensorStore};

    fn make_config() -> ModelConfig {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("llava16".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal llava16 config should parse")
    }

    // ── arch_id ──────────────────────────────────────────────────────────────

    #[test]
    fn test_arch_id_is_llava16() {
        let arch = LlavaNextArchitecture::new();
        assert_eq!(arch.arch_id(), "llava16");
    }

    #[test]
    fn test_arch_default_has_same_id() {
        let arch = LlavaNextArchitecture;
        assert_eq!(arch.arch_id(), "llava16");
    }

    // ── tensor_names ─────────────────────────────────────────────────────────

    /// `tensor_names` must return at least one entry (required by the task spec).
    #[test]
    fn llava_next_tensor_names_non_empty() {
        let arch = LlavaNextArchitecture::new();
        let names = arch.tensor_names();
        assert!(
            !names.is_empty(),
            "tensor_names() should not be empty for llava16"
        );
    }

    #[test]
    fn test_tensor_names_contains_mm_projector() {
        let arch = LlavaNextArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.contains(&"mm.0.weight"),
            "mm.0.weight must be in tensor_names"
        );
        assert!(
            patterns.contains(&"mm.2.weight"),
            "mm.2.weight must be in tensor_names"
        );
    }

    #[test]
    fn test_tensor_names_contains_vision_patterns() {
        let arch = LlavaNextArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.starts_with("v.")),
            "tensor_names must contain vision encoder patterns starting with 'v.'"
        );
    }

    #[test]
    fn test_tensor_names_contains_backbone_patterns() {
        let arch = LlavaNextArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.contains(&"token_embd.weight"),
            "token_embd.weight must be present"
        );
        assert!(
            patterns.iter().any(|p| p.contains("blk.")),
            "per-layer blk.* patterns must be present"
        );
    }

    // ── build error paths ────────────────────────────────────────────────────

    #[test]
    fn test_build_with_zero_heads_returns_config_error() {
        let arch = LlavaNextArchitecture::new();
        let mut config = make_config();
        config.num_attention_heads = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(
            matches!(result, Err(ArchError::ConfigMismatch { .. })),
            "zero num_attention_heads must give ConfigMismatch"
        );
    }

    #[test]
    fn test_build_with_zero_hidden_size_returns_config_error() {
        let arch = LlavaNextArchitecture::new();
        let mut config = make_config();
        config.hidden_size = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(
            matches!(result, Err(ArchError::ConfigMismatch { .. })),
            "zero hidden_size must give ConfigMismatch"
        );
    }

    #[test]
    fn test_build_with_valid_config_returns_missing_tensor() {
        let arch = LlavaNextArchitecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "valid config should return MissingTensor (full load needs GgufModel)"
        );
    }

    // ── registry integration ─────────────────────────────────────────────────

    /// `ArchRegistry::get("llava16")` must return `Some` after registration.
    #[test]
    fn llava_next_registry_lookup() {
        let mut registry = ArchitectureRegistry::new();
        registry.register(Box::new(LlavaNextArchitecture::new()));
        let arch = registry.get("llava16");
        assert!(
            arch.is_ok(),
            "registry.get('llava16') must succeed after registration"
        );
        let arch = arch.expect("llava16 arch");
        assert_eq!(arch.arch_id(), "llava16");
    }

    /// With builtins, `"llava16"` should be present if `llava16` feature is active.
    #[test]
    fn llava_next_in_builtin_registry() {
        let registry = ArchitectureRegistry::with_builtins();
        assert!(
            registry.contains("llava16"),
            "ArchitectureRegistry::with_builtins() should contain 'llava16'"
        );
    }
}

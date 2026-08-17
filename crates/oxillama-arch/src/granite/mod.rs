//! Granite-3.x (IBM) architecture.
//!
//! GGUF `general.architecture` = `"granite"` — a **real** architecture id that
//! llama.cpp writes (`convert_hf_to_gguf.py`'s
//! `@ModelBase.register("GraniteForCausalLM")` sets
//! `model_arch = gguf.MODEL_ARCH.GRANITE`, and `gguf-py/gguf/constants.py` maps
//! `MODEL_ARCH.GRANITE` to the string `"granite"`), so unlike `yi` / `internlm3`
//! this plugin is genuinely reachable from `general.architecture`.
//!
//! # Topology
//!
//! LLaMA's, exactly: RMSNorm pre-normalisation, grouped-query attention with
//! NORM-convention RoPE, SwiGLU FFN, optionally tied input/output embeddings.
//! `src/llama-arch.cpp` gives `LLM_ARCH_GRANITE` the same twelve tensors as
//! `LLM_ARCH_LLAMA`; `src/llama-model.cpp` creates them in the same arm; and
//! `llama_model_rope_type` lists it under `LLAMA_ROPE_TYPE_NORM`.
//!
//! # What actually makes it Granite
//!
//! Four scalar multipliers — [`GraniteScales`] — that this crate did not read
//! at all before, and whose omission is silent: the model runs and returns
//! finite, wrong logits.
//!
//! | Key | Applied |
//! |-----|---------|
//! | `granite.embedding_scale` | multiplies the token embedding |
//! | `granite.residual_scale`  | multiplies the branch at **both** residual adds |
//! | `granite.attention.scale` | replaces `1/sqrt(head_dim)` as the softmax scale |
//! | `granite.logit_scale`     | **divides** the final logits |
//!
//! Plus `granite.rope.scaling.finetuned`, which Granite repurposes as an on/off
//! switch for RoPE and which defaults to `true`.
//!
//! See [`scales`] for the reference citations, the sentinel semantics, and why
//! `logit_scale` divides rather than multiplies.
//!
//! # Before this module
//!
//! `GraniteArchitecture::build()` returned
//! `MissingTensor { name: "token_embd.weight (use GraniteModel::from_gguf for
//! full loading)" }` — naming a `GraniteModel` type that did not exist anywhere
//! in the tree.  There was no loader, no forward pass, and no code path that
//! read any of the four multipliers.

mod loader;
mod model;
pub mod scales;

pub use loader::load_granite_from_gguf;
pub use model::{GraniteLayer, GraniteModel};
pub use scales::GraniteScales;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// Granite-3.x (IBM) dense decoder-only architecture plugin.
///
/// Registered under GGUF `general.architecture` = `"granite"`.
pub struct GraniteArchitecture;

impl GraniteArchitecture {
    /// Create a new Granite architecture plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GraniteArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for GraniteArchitecture {
    fn arch_id(&self) -> &str {
        "granite"
    }

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

        // `TensorStore` carries tensor *metadata* only — no payload and no
        // GGUF KV table, so neither the weights nor Granite's four multipliers
        // are reachable from here.  `build_from_gguf` is the real entry point.
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (TensorStore has no payload; use \
                   GraniteArchitecture::build_from_gguf or GraniteModel::from_gguf)"
                .to_string(),
        })
    }

    /// Build a runnable Granite model from a fully-loaded GGUF file.
    ///
    /// This is what lets the registry route `general.architecture = "granite"`
    /// to a working model instead of an error.
    fn build_from_gguf(
        &self,
        model: &GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(GraniteModel::from_gguf(model, config)?))
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
                description: "Final RMSNorm".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head / unembedding (may be tied to token_embd)".to_string(),
                required: false,
            },
        ];

        let layer_tensors = [
            ("blk.{i}.attn_norm.weight", "Pre-attention RMSNorm"),
            ("blk.{i}.attn_q.weight", "Query projection"),
            ("blk.{i}.attn_k.weight", "Key projection"),
            ("blk.{i}.attn_v.weight", "Value projection"),
            ("blk.{i}.attn_output.weight", "Attention output projection"),
            ("blk.{i}.ffn_norm.weight", "Pre-FFN RMSNorm"),
            ("blk.{i}.ffn_gate.weight", "FFN gate projection (SwiGLU)"),
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
            MetadataValue::String("granite".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal granite config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = GraniteArchitecture::new();
        assert_eq!(arch.arch_id(), "granite");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = GraniteArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = GraniteArchitecture::new();
        let names = arch.tensor_names();
        assert!(
            names.iter().any(|p| p.pattern.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_has_required_block_patterns() {
        let arch = GraniteArchitecture::new();
        let names = arch.tensor_names();
        let required_patterns = [
            "token_embd.weight",
            "output_norm.weight",
            "blk.{i}.attn_q.weight",
            "blk.{i}.ffn_gate.weight",
        ];
        for pat in required_patterns {
            assert!(
                names.iter().any(|p| p.pattern == pat),
                "missing required pattern: {pat}"
            );
        }
    }

    #[test]
    fn test_build_with_zero_heads_returns_config_error() {
        let arch = GraniteArchitecture::new();
        let mut config = make_config();
        config.num_attention_heads = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(result.is_err());
        assert!(matches!(result, Err(ArchError::ConfigMismatch { .. })));
    }

    /// The `build()` error must not name a type that does not exist.
    ///
    /// It used to read `"… (use GraniteModel::from_gguf for full loading)"`
    /// while no `GraniteModel` existed anywhere in the tree.
    #[test]
    fn build_error_names_only_real_entry_points() {
        let arch = GraniteArchitecture::new();
        let tensors = TensorStore::new();
        let err = arch
            .build(&make_config(), &tensors)
            .err()
            .expect("build() cannot succeed without tensor payload");
        match err {
            ArchError::MissingTensor { name } => {
                assert!(
                    name.contains("build_from_gguf") && name.contains("GraniteModel::from_gguf"),
                    "build() should point at the real loaders, got: {name}"
                );
            }
            other => panic!("expected MissingTensor, got {other}"),
        }
    }

    #[test]
    fn granite_in_registry() {
        let registry = ArchitectureRegistry::with_builtins();
        assert!(
            registry.contains("granite"),
            "granite must be present in the default registry"
        );
    }
}

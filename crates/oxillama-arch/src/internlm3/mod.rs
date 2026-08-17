//! InternLM3 architecture — a registry alias, **not** a GGUF architecture id.
//!
//! InternLM3 is dense decoder-only LLaMA topology: RMSNorm pre-normalisation,
//! grouped-query attention, NORM-convention RoPE, SwiGLU FFN, optionally tied
//! embeddings.
//!
//! # `general.architecture` is never `"internlm3"`
//!
//! Verified against the reference checkout:
//!
//! * `convert_hf_to_gguf.py`:
//!   ```python
//!   @ModelBase.register("InternLM3ForCausalLM")
//!   class InternLM3Model(TextModel):
//!       model_arch = gguf.MODEL_ARCH.LLAMA
//!   ```
//!   — the converter writes `general.architecture = "llama"` for every
//!   InternLM3 checkpoint.
//! * `gguf-py/gguf/constants.py` has no `MODEL_ARCH.INTERNLM3` and
//!   `src/llama-arch.cpp`'s `LLM_ARCH_NAMES` has no `"internlm3"` entry; the
//!   only InternLM id llama.cpp knows is `internlm2`, which is a **different**
//!   graph (fused `attn_qkv`, its own `llm_build_internlm2`) and is
//!   deliberately not aliased here.
//!
//! A real InternLM3 checkpoint therefore loads through
//! `crate::llama::load_llama_from_gguf`, and this plugin can only be reached
//! by an explicit `ArchitectureRegistry::get("internlm3")` — never by
//! architecture dispatch.  `crate::registry` documents the same thing at its
//! `register` call.
//!
//! # Why this is an error and not an implementation
//!
//! [`InternLm3Architecture::build`] used to return
//! `MissingTensor { name: "token_embd.weight (use InternLm3Model::from_gguf
//! for full loading)" }`, pointing at an `InternLm3Model` type that **did not
//! exist anywhere in the tree**.  Building one would have produced a duplicate
//! of `crate::llama::LlamaModel` that no checkpoint can route to, so both
//! entry points now return an accurate [`ArchError::NotSupported`] naming the
//! `llama` path instead.

mod model;

pub use model::InternLm3Architecture;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// The message both entry points report, kept in one place.
const UNREACHABLE_ARCH: &str =
    "architecture id 'internlm3' is not written by any GGUF converter: convert_hf_to_gguf.py \
     registers InternLM3ForCausalLM with model_arch = gguf.MODEL_ARCH.LLAMA, and llama.cpp has \
     no LLM_ARCH_INTERNLM3, so InternLM3 checkpoints ship as general.architecture = \"llama\" \
     and must be loaded through the llama path (`crate::llama::load_llama_from_gguf`). This \
     registry entry exists only for an explicit \
     `ArchitectureRegistry::get(\"internlm3\")` lookup";

impl ModelArchitecture for InternLm3Architecture {
    fn arch_id(&self) -> &str {
        "internlm3"
    }

    /// Always reports failure.
    ///
    /// The geometry checks run first so a genuinely malformed
    /// [`ModelConfig`] is still reported as such; a well-formed one then gets
    /// [`ArchError::NotSupported`] explaining that the id is unreachable.
    ///
    /// # Errors
    ///
    /// * [`ArchError::ConfigMismatch`] for a zero head count or hidden size.
    /// * [`ArchError::NotSupported`] otherwise, explaining that real
    ///   checkpoints load through the `llama` path.
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

        Err(ArchError::NotSupported {
            detail: UNREACHABLE_ARCH.to_string(),
        })
    }

    /// Also reports failure, with the same explanation.
    ///
    /// Overriding the default matters: without it the registry would report
    /// the generic `"architecture 'internlm3' has not implemented
    /// build_from_gguf()"`, which reads like an unfinished port rather than a
    /// deliberate alias.
    ///
    /// # Errors
    ///
    /// Always [`ArchError::NotSupported`].
    fn build_from_gguf(
        &self,
        _model: &GgufModel,
        _config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Err(ArchError::NotSupported {
            detail: UNREACHABLE_ARCH.to_string(),
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
            MetadataValue::String("internlm3".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal internlm3 config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = InternLm3Architecture::new();
        assert_eq!(arch.arch_id(), "internlm3");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = InternLm3Architecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = InternLm3Architecture::new();
        let names = arch.tensor_names();
        assert!(
            names.iter().any(|p| p.pattern.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_has_required_block_patterns() {
        let arch = InternLm3Architecture::new();
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
        let arch = InternLm3Architecture::new();
        let mut config = make_config();
        config.num_attention_heads = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(result.is_err());
        assert!(matches!(result, Err(ArchError::ConfigMismatch { .. })));
    }

    #[test]
    fn internlm3_in_registry() {
        let registry = ArchitectureRegistry::with_builtins();
        assert!(
            registry.contains("internlm3"),
            "internlm3 must be present in the default registry"
        );
    }

    /// Regression: `build()` used to point at `InternLm3Model::from_gguf`, and
    /// no `InternLm3Model` existed anywhere in the tree.
    #[test]
    fn build_does_not_name_a_nonexistent_type() {
        let arch = InternLm3Architecture::new();
        let tensors = TensorStore::new();
        let err = arch
            .build(&make_config(), &tensors)
            .err()
            .expect("the `internlm3` arch id is unreachable and must never build");
        match err {
            ArchError::NotSupported { detail } => {
                assert!(
                    !detail.contains("InternLm3Model"),
                    "must not reference a type that does not exist: {detail}"
                );
                assert!(
                    detail.contains("llama"),
                    "must point the caller at the llama path: {detail}"
                );
            }
            other => panic!("expected NotSupported, got {other}"),
        }
    }

    /// The registry's GGUF entry point must give the same accurate answer, not
    /// the trait default's "has not implemented build_from_gguf()".
    #[test]
    fn build_from_gguf_reports_the_same_reason() {
        let arch = InternLm3Architecture::new();
        let bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let model = oxillama_gguf::GgufModel::from_bytes(bytes).expect("test: parse fixture");
        let err = arch
            .build_from_gguf(&model, &make_config())
            .err()
            .expect("the `internlm3` arch id is unreachable and must never build");
        match err {
            ArchError::NotSupported { detail } => {
                assert!(
                    detail.contains("llama") && !detail.contains("has not implemented"),
                    "expected the alias explanation, got: {detail}"
                );
            }
            other => panic!("expected NotSupported, got {other}"),
        }
    }
}

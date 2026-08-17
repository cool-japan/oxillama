//! Yi architecture — a registry alias, **not** a GGUF architecture id.
//!
//! Yi (01.AI) is dense decoder-only LLaMA topology: RMSNorm pre-normalisation,
//! grouped-query attention, NORM-convention RoPE, SwiGLU FFN, optionally tied
//! embeddings.
//!
//! # `general.architecture` is never `"yi"`
//!
//! Verified against the reference checkout:
//!
//! * `gguf-py/gguf/constants.py` has **no** `MODEL_ARCH.YI`, and
//!   `src/llama-arch.cpp`'s `LLM_ARCH_NAMES` has **no** `"yi"` entry — the id
//!   does not exist in llama.cpp at all.
//! * `convert_hf_to_gguf.py` never mentions Yi.  Yi checkpoints declare
//!   `"architectures": ["LlamaForCausalLM"]` in their HF config, so they are
//!   converted by `LlamaModel` (`model_arch = gguf.MODEL_ARCH.LLAMA`) and the
//!   resulting GGUF says `general.architecture = "llama"`.
//!
//! A real Yi checkpoint therefore loads through
//! `crate::llama::load_llama_from_gguf`, and this plugin can only be reached
//! by an explicit `ArchitectureRegistry::get("yi")` — never by architecture
//! dispatch.  `crate::registry` documents the same thing at its `register`
//! call, and `crate::config::rope_style_for_arch` already lists `"yi"` under
//! "registry aliases whose checkpoints ship as `llama`".
//!
//! # Why this is an error and not an implementation
//!
//! [`YiArchitecture::build`] used to return
//! `MissingTensor { name: "token_embd.weight (use YiModel::from_gguf for full
//! loading)" }`, pointing at a `YiModel` type that **did not exist anywhere in
//! the tree**.  Building one would have produced a byte-for-byte duplicate of
//! `crate::llama::LlamaModel` that no checkpoint can route to, so both entry
//! points now return an accurate [`ArchError::NotSupported`] naming the `llama`
//! path instead.

mod model;

pub use model::YiArchitecture;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// The message both entry points report, kept in one place.
const UNREACHABLE_ARCH: &str = "architecture id 'yi' is not written by any GGUF converter: \
     llama.cpp has no LLM_ARCH_YI and convert_hf_to_gguf.py has no Yi entry, so Yi checkpoints \
     (HF `LlamaForCausalLM`) ship as general.architecture = \"llama\" and must be loaded through \
     the llama path (`crate::llama::load_llama_from_gguf`). This registry entry exists only for \
     an explicit `ArchitectureRegistry::get(\"yi\")` lookup";

impl ModelArchitecture for YiArchitecture {
    fn arch_id(&self) -> &str {
        "yi"
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
    /// the generic `"architecture 'yi' has not implemented build_from_gguf()"`,
    /// which reads like an unfinished port rather than a deliberate alias.
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
            MetadataValue::String("yi".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal yi config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = YiArchitecture::new();
        assert_eq!(arch.arch_id(), "yi");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = YiArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = YiArchitecture::new();
        let names = arch.tensor_names();
        assert!(
            names.iter().any(|p| p.pattern.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_has_required_block_patterns() {
        let arch = YiArchitecture::new();
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
        let arch = YiArchitecture::new();
        let mut config = make_config();
        config.num_attention_heads = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(result.is_err());
        assert!(matches!(result, Err(ArchError::ConfigMismatch { .. })));
    }

    #[test]
    fn yi_in_registry() {
        let registry = ArchitectureRegistry::with_builtins();
        assert!(
            registry.contains("yi"),
            "yi must be present in the default registry"
        );
    }

    /// Regression: `build()` used to point at `YiModel::from_gguf`, and no
    /// `YiModel` existed anywhere in the tree.
    #[test]
    fn build_does_not_name_a_nonexistent_type() {
        let arch = YiArchitecture::new();
        let tensors = TensorStore::new();
        let err = arch
            .build(&make_config(), &tensors)
            .err()
            .expect("the `yi` arch id is unreachable and must never build");
        match err {
            ArchError::NotSupported { detail } => {
                assert!(
                    !detail.contains("YiModel"),
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
        let arch = YiArchitecture::new();
        let bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let model = oxillama_gguf::GgufModel::from_bytes(bytes).expect("test: parse fixture");
        let err = arch
            .build_from_gguf(&model, &make_config())
            .err()
            .expect("the `yi` arch id is unreachable and must never build");
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

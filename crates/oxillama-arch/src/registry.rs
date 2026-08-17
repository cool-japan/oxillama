//! Architecture plugin registry.
//!
//! Model architectures register themselves here so the inference engine
//! can look up the correct implementation based on GGUF metadata.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{ArchError, ArchResult};
use crate::traits::ModelArchitecture;

/// Registry of model architecture plugins.
///
/// Maps architecture identifier strings (from GGUF `general.architecture`)
/// to their implementations.
///
/// # Aliases
///
/// Several implementations answer to more than one GGUF architecture id.  Some
/// of the ids this crate historically registered are not written by llama.cpp
/// at all — `mistral`, `mixtral`, `yi` and `internlm3` checkpoints all ship as
/// `"llama"`, and LLaVA GGUFs are `"llama"` plus a separate `clip`-arch
/// projector — so a registry keyed only on those names could never be selected
/// by `general.architecture`.  [`Self::register_with_aliases`] lets one
/// implementation own the real id **and** the historical name, so both lookups
/// resolve.
#[derive(Default)]
pub struct ArchitectureRegistry {
    architectures: HashMap<String, Arc<dyn ModelArchitecture>>,
}

impl ArchitectureRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            architectures: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with all built-in architectures.
    ///
    /// Implementations are registered under their own `arch_id()` plus any
    /// additional GGUF `general.architecture` string that routes to the *same*
    /// graph.  An alias is only added when the target loader genuinely handles
    /// that checkpoint: `gemma2` / `gemma3` are aliased because the engine
    /// already routes them into the Gemma loader, whereas `starcoder2`,
    /// `phi2`, `qwen2` and `internlm2` are deliberately **not** aliased — they
    /// are different graphs (different RoPE style and/or different tensor set)
    /// and silently routing them to a near-neighbour would produce garbage
    /// rather than an error.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        #[cfg(feature = "llama")]
        registry.register(Box::new(crate::llama::LlamaArchitecture::new()));

        #[cfg(feature = "qwen3")]
        registry.register(Box::new(crate::qwen3::Qwen3Architecture::new()));

        #[cfg(feature = "mistral")]
        registry.register(Box::new(crate::mistral::MistralArchitecture::new()));

        // The engine already routes `gemma2` / `gemma3` into this loader
        // (`build_forward_pass`); only the registry disagreed.
        #[cfg(feature = "gemma")]
        registry.register_with_aliases(
            Box::new(crate::gemma::GemmaArchitecture::new()),
            &["gemma2", "gemma3"],
        );

        // `phi` is the spelling the engine also accepts; `phi2` is a
        // genuinely different graph and is deliberately NOT aliased here.
        #[cfg(feature = "phi")]
        registry.register_with_aliases(Box::new(crate::phi::PhiArchitecture::new()), &["phi"]);

        #[cfg(feature = "command-r")]
        registry.register(Box::new(crate::command_r::CommandRArchitecture::new()));

        // NOTE: `starcoder2` is NOT aliased — it is a NeoX-RoPE architecture
        // while `starcoder` is NORM, so routing it here would silently rotate
        // the wrong element pairs.  It needs its own implementation.
        #[cfg(feature = "starcoder")]
        registry.register(Box::new(crate::starcoder::StarcoderArchitecture::new()));

        #[cfg(feature = "llava")]
        registry.register(Box::new(crate::llava::LlavaArchitecture::new()));

        #[cfg(feature = "llava16")]
        registry.register(Box::new(crate::llava_next::LlavaNextArchitecture::new()));

        #[cfg(feature = "falcon")]
        registry.register(Box::new(crate::falcon::FalconArchitecture::new()));

        #[cfg(feature = "minicpm")]
        registry.register(Box::new(crate::minicpm::MiniCpmArchitecture::new()));

        #[cfg(feature = "olmo2")]
        registry.register(Box::new(crate::olmo2::Olmo2Architecture::new()));

        #[cfg(feature = "granite")]
        registry.register(Box::new(crate::granite::GraniteArchitecture::new()));

        // NOTE: `yi`, `internlm3`, `mistral` and `mixtral` are not ids that
        // llama.cpp ever writes — those checkpoints all ship as `"llama"`.
        // They stay registered so existing callers keep resolving, but they can
        // only be reached by an explicit `get()`, never by
        // `general.architecture`.
        registry.register(Box::new(crate::yi::YiArchitecture::new()));
        registry.register(Box::new(crate::internlm3::InternLm3Architecture::new()));

        #[cfg(feature = "deepseek")]
        registry.register(Box::new(crate::deepseek::DeepSeekArchitecture::new()));

        #[cfg(feature = "dbrx")]
        registry.register(Box::new(crate::dbrx::DbrxArchitecture::new()));

        #[cfg(feature = "grok")]
        registry.register(Box::new(crate::grok::GrokArchitecture::new()));

        #[cfg(feature = "mamba2")]
        registry.register(Box::new(crate::mamba2::Mamba2Architecture::new()));

        #[cfg(feature = "jamba")]
        registry.register(Box::new(crate::jamba::JambaArchitecture::new()));

        #[cfg(feature = "qwen2-vl")]
        registry.register(Box::new(crate::qwen2_vl::Qwen2VlArchitecture::new()));

        #[cfg(feature = "mixtral")]
        registry.register(Box::new(crate::mixtral::MixtralArchitecture::new()));

        #[cfg(feature = "stablelm")]
        registry.register(Box::new(crate::stablelm::StablelmArchitecture::new()));

        #[cfg(feature = "gptneox")]
        registry.register(Box::new(crate::gpt_neox::GptNeoxArchitecture::new()));

        #[cfg(feature = "bloom")]
        registry.register(Box::new(crate::bloom::BloomArchitecture::new()));

        #[cfg(feature = "phimoe")]
        registry.register(Box::new(crate::phi_moe::PhiMoeArchitecture::new()));

        registry
    }

    /// Register a model architecture.
    ///
    /// If an architecture with the same ID is already registered, it is replaced.
    pub fn register(&mut self, arch: Box<dyn ModelArchitecture>) {
        self.register_with_aliases(arch, &[]);
    }

    /// Register a model architecture under its own id plus extra aliases.
    ///
    /// An alias that is already taken by a different implementation is **not**
    /// overwritten, so a dedicated implementation always wins over another
    /// architecture's alias regardless of registration order.
    pub fn register_with_aliases(&mut self, arch: Box<dyn ModelArchitecture>, aliases: &[&str]) {
        let shared: Arc<dyn ModelArchitecture> = Arc::from(arch);
        let primary = shared.arch_id().to_string();
        for alias in aliases {
            self.architectures
                .entry((*alias).to_string())
                .or_insert_with(|| Arc::clone(&shared));
        }
        self.architectures.insert(primary, shared);
    }

    /// Every GGUF architecture id that resolves to `arch_id`'s implementation.
    ///
    /// Includes `arch_id` itself when it is registered.  Order is unspecified.
    pub fn aliases_of(&self, arch_id: &str) -> Vec<&str> {
        let Some(target) = self.architectures.get(arch_id) else {
            return Vec::new();
        };
        self.architectures
            .iter()
            .filter(|(_, v)| Arc::ptr_eq(v, target))
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Look up an architecture by its identifier.
    ///
    /// # Errors
    ///
    /// [`ArchError::UnknownArchitecture`] when no implementation or alias
    /// claims `arch_id`.
    pub fn get(&self, arch_id: &str) -> ArchResult<&dyn ModelArchitecture> {
        self.architectures
            .get(arch_id)
            .map(|a| a.as_ref())
            .ok_or_else(|| ArchError::UnknownArchitecture {
                arch_id: arch_id.to_string(),
            })
    }

    /// Check if an architecture is registered.
    pub fn contains(&self, arch_id: &str) -> bool {
        self.architectures.contains_key(arch_id)
    }

    /// List all registered architecture IDs.
    pub fn list(&self) -> Vec<&str> {
        self.architectures.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the number of registered architectures.
    pub fn len(&self) -> usize {
        self.architectures.len()
    }

    /// Returns true if no architectures are registered.
    pub fn is_empty(&self) -> bool {
        self.architectures.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_registry_is_empty() {
        let reg = ArchitectureRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    /// Builtin implementations plus the 3 aliases (`gemma2`, `gemma3`, `phi`).
    ///
    /// `jamba` is feature-gated and is **not** in the crate's `default` list
    /// (its implementation is a non-functional stub — see `TODO.md`), so the
    /// expected count follows the feature.
    #[cfg(feature = "jamba")]
    const EXPECTED_BUILTINS: usize = 29;
    /// See [`EXPECTED_BUILTINS`].
    #[cfg(not(feature = "jamba"))]
    const EXPECTED_BUILTINS: usize = 28;

    #[test]
    fn test_with_builtins_registers_expected_architecture_ids() {
        let reg = ArchitectureRegistry::with_builtins();
        assert_eq!(
            reg.len(),
            EXPECTED_BUILTINS,
            "expected {EXPECTED_BUILTINS} builtin architectures + aliases, got {:?}",
            {
                let mut l = reg.list();
                l.sort_unstable();
                l
            }
        );
        assert!(!reg.is_empty());
    }

    /// `gemma2` / `gemma3` must resolve — the engine already routes them into
    /// the Gemma loader, so a registry that rejected them disagreed with the
    /// engine about which architectures exist.
    #[cfg(feature = "gemma")]
    #[test]
    fn test_gemma2_and_gemma3_resolve_to_the_gemma_implementation() {
        let reg = ArchitectureRegistry::with_builtins();
        for id in ["gemma", "gemma2", "gemma3"] {
            let arch = reg
                .get(id)
                .unwrap_or_else(|_| panic!("registry must resolve '{id}'"));
            assert_eq!(arch.arch_id(), "gemma");
        }
        let mut aliases = reg.aliases_of("gemma");
        aliases.sort_unstable();
        assert_eq!(aliases, vec!["gemma", "gemma2", "gemma3"]);
    }

    /// Architectures whose GGUF id exists but whose graph is genuinely
    /// different must NOT silently resolve to a near-neighbour.
    #[test]
    fn test_divergent_architectures_are_not_aliased() {
        let reg = ArchitectureRegistry::with_builtins();
        for id in ["starcoder2", "phi2", "qwen2", "internlm2"] {
            assert!(
                !reg.contains(id),
                "'{id}' is a different graph and must not resolve to a neighbour"
            );
        }
    }

    #[test]
    fn test_with_builtins_contains_all_expected_ids() {
        let reg = ArchitectureRegistry::with_builtins();
        let expected_ids = [
            "llama",
            "qwen3",
            "mistral",
            "gemma",
            "phi3",
            "command-r",
            "starcoder",
            "llava",
            "deepseek2",
            "dbrx",
            "grok",
            "mamba2",
        ];
        for id in expected_ids {
            assert!(
                reg.contains(id),
                "registry should contain architecture '{id}'"
            );
        }
    }

    #[test]
    fn test_get_known_architecture_succeeds() {
        let reg = ArchitectureRegistry::with_builtins();
        let arch = reg.get("llama");
        assert!(arch.is_ok(), "get('llama') should succeed");
        let arch = arch.expect("llama arch");
        assert_eq!(arch.arch_id(), "llama");
    }

    #[test]
    fn test_get_all_builtins_return_correct_ids() {
        let reg = ArchitectureRegistry::with_builtins();
        let ids = [
            "llama",
            "qwen3",
            "mistral",
            "gemma",
            "phi3",
            "command-r",
            "starcoder",
            "llava",
            "deepseek2",
            "dbrx",
            "grok",
            "mamba2",
        ];
        for id in ids {
            let arch = reg
                .get(id)
                .unwrap_or_else(|_| panic!("get('{id}') must succeed"));
            assert_eq!(
                arch.arch_id(),
                id,
                "arch_id() should match the registered key"
            );
        }
    }

    #[test]
    fn test_get_unknown_architecture_returns_error() {
        let reg = ArchitectureRegistry::with_builtins();
        let result = reg.get("nonexistent_arch_xyz");
        assert!(result.is_err(), "get with unknown id should return error");
        if let Err(ArchError::UnknownArchitecture { arch_id }) = result {
            assert_eq!(arch_id, "nonexistent_arch_xyz");
        } else {
            panic!("expected UnknownArchitecture error");
        }
    }

    #[test]
    fn test_contains_unknown_returns_false() {
        let reg = ArchitectureRegistry::with_builtins();
        assert!(!reg.contains("does_not_exist"));
    }

    #[test]
    fn test_list_returns_all_registered_ids() {
        let reg = ArchitectureRegistry::with_builtins();
        let mut listed = reg.list();
        listed.sort_unstable();
        let mut expected = vec![
            "gemma2",
            "gemma3",
            "phi",
            "bloom",
            "command-r",
            "dbrx",
            "deepseek2",
            "falcon",
            "gemma",
            "granite",
            "grok",
            "gptneox",
            "internlm3",
            #[cfg(feature = "jamba")]
            "jamba",
            "llama",
            "llava",
            "llava16",
            "mamba2",
            "minicpm",
            "mistral",
            "mixtral",
            "olmo2",
            "phi3",
            "phimoe",
            "qwen2vl",
            "qwen3",
            "stablelm",
            "starcoder",
            "yi",
        ];
        expected.sort_unstable();
        assert_eq!(listed, expected);
    }

    #[test]
    fn test_register_custom_architecture_and_retrieve() {
        use crate::error::ArchResult;
        use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
        use oxillama_gguf::TensorStore;

        struct DummyArch;

        impl ModelArchitecture for DummyArch {
            fn arch_id(&self) -> &str {
                "dummy-test-arch"
            }

            fn build(
                &self,
                _config: &crate::config::ModelConfig,
                _tensors: &TensorStore,
            ) -> ArchResult<Box<dyn ForwardPass>> {
                Err(ArchError::NotSupported {
                    detail: "dummy".to_string(),
                })
            }

            fn tensor_names(&self) -> Vec<TensorNamePattern> {
                vec![]
            }
        }

        let mut reg = ArchitectureRegistry::new();
        reg.register(Box::new(DummyArch));
        assert_eq!(reg.len(), 1);
        assert!(reg.contains("dummy-test-arch"));
        let arch = reg.get("dummy-test-arch").expect("should find dummy arch");
        assert_eq!(arch.arch_id(), "dummy-test-arch");
    }

    #[test]
    fn test_register_replaces_existing() {
        use crate::error::ArchResult;
        use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
        use oxillama_gguf::TensorStore;

        struct Arch1;
        struct Arch2;

        impl ModelArchitecture for Arch1 {
            fn arch_id(&self) -> &str {
                "replace-test"
            }
            fn build(
                &self,
                _c: &crate::config::ModelConfig,
                _t: &TensorStore,
            ) -> ArchResult<Box<dyn ForwardPass>> {
                Err(ArchError::NotSupported {
                    detail: "arch1".to_string(),
                })
            }
            fn tensor_names(&self) -> Vec<TensorNamePattern> {
                vec![]
            }
        }

        impl ModelArchitecture for Arch2 {
            fn arch_id(&self) -> &str {
                "replace-test"
            }
            fn build(
                &self,
                _c: &crate::config::ModelConfig,
                _t: &TensorStore,
            ) -> ArchResult<Box<dyn ForwardPass>> {
                Err(ArchError::NotSupported {
                    detail: "arch2".to_string(),
                })
            }
            fn tensor_names(&self) -> Vec<TensorNamePattern> {
                vec![]
            }
        }

        let mut reg = ArchitectureRegistry::new();
        reg.register(Box::new(Arch1));
        reg.register(Box::new(Arch2));
        // Length must still be 1 (replacement, not duplicate)
        assert_eq!(reg.len(), 1);
        assert!(reg.contains("replace-test"));
    }
}

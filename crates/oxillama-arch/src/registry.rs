//! Architecture plugin registry.
//!
//! Model architectures register themselves here so the inference engine
//! can look up the correct implementation based on GGUF metadata.

use std::collections::HashMap;

use crate::error::{ArchError, ArchResult};
use crate::traits::ModelArchitecture;

/// Registry of model architecture plugins.
///
/// Maps architecture identifier strings (from GGUF `general.architecture`)
/// to their implementations.
#[derive(Default)]
pub struct ArchitectureRegistry {
    architectures: HashMap<String, Box<dyn ModelArchitecture>>,
}

impl ArchitectureRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            architectures: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with all built-in architectures.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        #[cfg(feature = "llama")]
        registry.register(Box::new(crate::llama::LlamaArchitecture::new()));

        #[cfg(feature = "qwen3")]
        registry.register(Box::new(crate::qwen3::Qwen3Architecture::new()));

        #[cfg(feature = "mistral")]
        registry.register(Box::new(crate::mistral::MistralArchitecture::new()));

        #[cfg(feature = "gemma")]
        registry.register(Box::new(crate::gemma::GemmaArchitecture::new()));

        #[cfg(feature = "phi")]
        registry.register(Box::new(crate::phi::PhiArchitecture::new()));

        #[cfg(feature = "command-r")]
        registry.register(Box::new(crate::command_r::CommandRArchitecture::new()));

        #[cfg(feature = "starcoder")]
        registry.register(Box::new(crate::starcoder::StarcoderArchitecture::new()));

        #[cfg(feature = "llava")]
        registry.register(Box::new(crate::llava::LlavaArchitecture::new()));

        #[cfg(feature = "falcon")]
        registry.register(Box::new(crate::falcon::FalconArchitecture::new()));

        #[cfg(feature = "minicpm")]
        registry.register(Box::new(crate::minicpm::MiniCpmArchitecture::new()));

        #[cfg(feature = "olmo2")]
        registry.register(Box::new(crate::olmo2::Olmo2Architecture::new()));

        #[cfg(feature = "granite")]
        registry.register(Box::new(crate::granite::GraniteArchitecture::new()));

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

        registry
    }

    /// Register a model architecture.
    ///
    /// If an architecture with the same ID is already registered, it is replaced.
    pub fn register(&mut self, arch: Box<dyn ModelArchitecture>) {
        self.architectures.insert(arch.arch_id().to_string(), arch);
    }

    /// Look up an architecture by its identifier.
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

    #[test]
    fn test_with_builtins_has_twelve_architectures() {
        let reg = ArchitectureRegistry::with_builtins();
        // 18 previous + 1 new (jamba) = 19
        assert_eq!(reg.len(), 19, "expected 19 builtin architectures");
        assert!(!reg.is_empty());
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
            "command-r",
            "dbrx",
            "deepseek2",
            "falcon",
            "gemma",
            "granite",
            "grok",
            "internlm3",
            "jamba",
            "llama",
            "llava",
            "mamba2",
            "minicpm",
            "mistral",
            "olmo2",
            "phi3",
            "qwen3",
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

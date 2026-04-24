//! Cross-arch property-based fuzz harness.
//!
//! Exercises every registered architecture through `tensor_names()` and validates
//! structural invariants using synthetically generated `ModelConfig` values.

use oxillama_arch::{ArchitectureRegistry, ModelConfig};
use proptest::prelude::*;

prop_compose! {
    fn arb_model_config()
        (hidden_size in prop::sample::select(vec![64usize, 128, 256]),
         num_layers in 1usize..=4,
         num_attention_heads in prop::sample::select(vec![2usize, 4, 8]),
         vocab_size in prop::sample::select(vec![256usize, 512, 1024]))
        -> ModelConfig
    {
        let num_kv_heads = (num_attention_heads / 2).max(1);
        ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            vocab_size,
            ..Default::default()
        }
    }
}

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(32))]

    #[test]
    fn all_archs_have_valid_tensor_names(_config in arb_model_config()) {
        let registry = ArchitectureRegistry::with_builtins();
        for arch_id in registry.list() {
            let arch = registry.get(arch_id).expect("arch must exist after list()");
            let names = arch.tensor_names();
            prop_assert!(
                !names.is_empty(),
                "arch '{}' returned empty tensor_names()",
                arch_id
            );
            for tn in &names {
                prop_assert!(
                    !tn.pattern.is_empty(),
                    "arch '{}' has an empty tensor pattern (description: {})",
                    arch_id,
                    tn.description
                );
            }
        }
    }

    #[test]
    fn all_archs_have_valid_ids(_config in arb_model_config()) {
        let registry = ArchitectureRegistry::with_builtins();
        prop_assert!(!registry.is_empty(), "builtin registry must not be empty");
        for arch_id in registry.list() {
            prop_assert!(!arch_id.is_empty(), "registered arch_id must not be empty");
            let arch = registry.get(arch_id).expect("arch must exist after list()");
            prop_assert_eq!(
                arch.arch_id(),
                arch_id,
                "arch_id() must match the registry key"
            );
        }
    }

    #[test]
    fn all_archs_have_at_least_one_required_tensor(_config in arb_model_config()) {
        let registry = ArchitectureRegistry::with_builtins();
        for arch_id in registry.list() {
            let arch = registry.get(arch_id).expect("arch must exist after list()");
            let names = arch.tensor_names();
            let has_required = names.iter().any(|tn| tn.required);
            prop_assert!(
                has_required,
                "arch '{}' must declare at least one required tensor",
                arch_id
            );
        }
    }
}

#[test]
fn builtin_registry_contains_expected_architectures() {
    let registry = ArchitectureRegistry::with_builtins();
    let expected = ["llama", "qwen3", "mistral", "gemma", "phi3"];
    for id in expected {
        assert!(
            registry.contains(id),
            "builtin registry should contain architecture '{id}'"
        );
    }
}

#[test]
fn all_builtin_arch_ids_are_non_empty() {
    let registry = ArchitectureRegistry::with_builtins();
    assert!(!registry.is_empty());
    for id in registry.list() {
        assert!(!id.is_empty(), "arch id must be non-empty");
    }
}

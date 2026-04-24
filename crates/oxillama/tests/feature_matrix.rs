//! Integration tests: verify that feature combinations expose the expected
//! public API surface without requiring any external files or inference.

#[test]
fn test_gguf_types_accessible() {
    let _ = std::mem::size_of::<oxillama_gguf::GgufModel>();
    let _ = std::mem::size_of::<oxillama_gguf::MetadataStore>();
}

#[test]
fn test_quant_dispatcher_accessible() {
    let dispatcher = oxillama_quant::global_dispatcher();
    assert!(!dispatcher.supported_types().is_empty());
}

#[test]
fn test_arch_registry_accessible() {
    // Default (no architecture features) registry is empty; with_builtins reflects
    // whatever features are compiled in.
    let registry = oxillama_arch::ArchitectureRegistry::default();
    // An empty registry is valid — len() is still callable.
    let _ = registry.len();
}

#[test]
fn test_arch_registry_with_builtins() {
    let registry = oxillama_arch::ArchitectureRegistry::with_builtins();
    // At least 0 architectures (exact count depends on enabled features).
    assert!(registry.len() == registry.list().len());
}

#[test]
fn test_runtime_engine_metrics_accessible() {
    let _ = std::mem::size_of::<oxillama_runtime::EngineMetrics>();
    let _ = std::mem::size_of::<oxillama_runtime::MetricsSnapshot>();
}

#[test]
fn test_runtime_engine_config_default() {
    let config = oxillama_runtime::EngineConfig::default();
    // Verify the default is sensible — no panic, no model path.
    assert!(config.model_path.is_empty());
    assert!(config.num_threads > 0);
}

#[test]
fn test_meta_crate_reexports_gguf() {
    let _ = std::mem::size_of::<oxillama::gguf::GgufModel>();
}

#[test]
fn test_meta_crate_reexports_quant() {
    let _ = std::mem::size_of::<oxillama::quant::QuantTensor>();
}

#[test]
fn test_meta_crate_reexports_arch() {
    let _ = std::mem::size_of::<oxillama::arch::ModelConfig>();
}

#[test]
fn test_meta_crate_reexports_runtime() {
    let _ = std::mem::size_of::<oxillama::runtime::EngineConfig>();
}

#[test]
fn test_engine_metrics_snapshot_zero() {
    let metrics = oxillama_runtime::EngineMetrics::new();
    let snap = metrics.snapshot();
    assert_eq!(snap.tokens_generated, 0);
    assert_eq!(snap.tokens_prefilled, 0);
    assert_eq!(snap.requests_started, 0);
    assert_eq!(snap.requests_completed, 0);
}

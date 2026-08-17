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
    // 0 = auto (GEMV pool sized from `available_parallelism`); the pool itself
    // always ends up with at least one thread.
    assert_eq!(config.num_threads, 0);
    assert!(oxillama_quant::parallel::num_threads() > 0);
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

/// Regression test for the P4 defect: `oxillama`'s `default` feature set used
/// to be `["server", "bench", "simd-neon", "simd-avx2"]` — no architecture
/// feature of its own. Under `default-features` alone that "worked" only
/// incidentally, because `server` transitively pulls in every
/// `oxillama-runtime` architecture feature via `oxillama-server`'s own
/// dependency declaration. Any consumer who picked features *without* naming
/// `server` (e.g. `default-features = false, features = ["bench"]`) got a
/// crate that compiled cleanly and then failed at *runtime* with
/// `unsupported architecture: 'llama'` on a perfectly valid model, because
/// `oxillama-runtime::build_forward_pass` gates every forward-pass arm on a
/// runtime architecture feature.
///
/// This test asserts the invariant directly on `oxillama`'s own Cargo
/// features (not on what a downstream crate happens to re-enable
/// transitively): under the shipped `default` feature set, at least one
/// architecture feature must be active.
///
/// - Before the fix: run with the *old* default set reproduced explicitly —
///   `cargo nextest run -p oxillama --no-default-features --features
///   "server,bench,simd-neon,simd-avx2" test_default_features_include_at_least_one_architecture`
///   — fails, since none of `llama`/`qwen3`/.../`jamba` is named.
/// - After the fix (`"llama"` added to `oxillama`'s `default`): plain
///   `cargo nextest run -p oxillama` (default features) passes.
#[test]
fn test_default_features_include_at_least_one_architecture() {
    let has_arch = cfg!(feature = "llama")
        || cfg!(feature = "qwen3")
        || cfg!(feature = "mistral")
        || cfg!(feature = "gemma")
        || cfg!(feature = "phi")
        || cfg!(feature = "command-r")
        || cfg!(feature = "starcoder")
        || cfg!(feature = "llava")
        || cfg!(feature = "deepseek")
        || cfg!(feature = "dbrx")
        || cfg!(feature = "grok")
        || cfg!(feature = "mamba2")
        || cfg!(feature = "jamba");
    assert!(
        has_arch,
        "oxillama's active feature set names zero architecture features \
         (llama/qwen3/mistral/gemma/phi/command-r/starcoder/llava/deepseek/dbrx/grok/mamba2/jamba); \
         a build in this configuration compiles a runtime with zero forward-pass \
         arms and fails at runtime with 'unsupported architecture' on any model. \
         `oxillama`'s `default` feature list must name at least one architecture \
         (see crates/oxillama/Cargo.toml)."
    );
}

//! # OxiLLaMa
//!
//! **Pure Rust LLM inference engine — the sovereign alternative to llama.cpp.**
//!
//! This is the unified meta crate that re-exports the full OxiLLaMa API surface.
//! Each subcrate is available as a top-level module:
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`gguf`] | GGUF v3 parser and tensor loader |
//! | [`quant`] | Quantization kernels (25 formats, SIMD) |
//! | [`arch`] | Model architectures (8 models) |
//! | [`runtime`] | Inference engine, KV cache, sampling |
//! | [`server`] | OpenAI-compatible HTTP API (feature: `server`) |
//! | `bench` | Benchmark suite (feature: `bench`) |
//! | `gpu` | wgpu GPU backend (feature: `gpu`) |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxillama::runtime::{InferenceEngine, EngineConfig, SamplerConfig};
//!
//! let config = EngineConfig {
//!     model_path: "model.gguf".to_string(),
//!     ..Default::default()
//! };
//! let mut engine = InferenceEngine::new(config);
//! engine.load_model().expect("failed to load model");
//! engine.generate("Hello", 128, |tok| print!("{tok}")).expect("generation failed");
//! ```

/// GGUF v3 file format parser and tensor loader.
pub use oxillama_gguf as gguf;

/// Quantization kernels for all GGUF quantization types.
pub use oxillama_quant as quant;

/// Model architecture implementations.
pub use oxillama_arch as arch;

/// Inference runtime: engine, KV cache, sampling, tokenizer, speculative decoding.
pub use oxillama_runtime as runtime;

/// OpenAI-compatible HTTP API server.
#[cfg(feature = "server")]
pub use oxillama_server as server;

/// Benchmark suite: latency, throughput, memory estimation.
#[cfg(feature = "bench")]
pub use oxillama_bench as bench;

/// Optional wgpu GPU compute backend.
#[cfg(feature = "gpu")]
pub use oxillama_gpu as gpu;

// ── Recipe compilation checks ─────────────────────────────────────────────────
//
// Every `rust,no_run` fenced block in RECIPES.md is extracted and compiled by
// rustdoc when docs are built (`cargo test --doc -p oxillama`).  The `js,no_run`
// block for Recipe 6 is transparently ignored because it is not a `rust` fence.
//
// This module is `#[doc(hidden)]` so it does not appear on docs.rs but is still
// compiled and tested during `cargo test --doc`.
#[doc(hidden)]
#[doc = include_str!("../RECIPES.md")]
pub mod recipes_doc {}

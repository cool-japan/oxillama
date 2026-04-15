//! # oxillama-runtime
//!
//! Inference runtime for OxiLLaMa.
//!
//! Orchestrates the complete inference pipeline: model loading, tokenization,
//! forward pass execution, KV caching, and token sampling.

pub mod engine;
pub mod error;
pub mod flash_attention;
pub mod kv_cache;
pub mod lora_loader;
pub mod sampling;
pub mod scheduler;
pub mod speculative;
pub mod tokenizer_bridge;

pub use engine::{EngineConfig, InferenceEngine};
pub use error::{RuntimeError, RuntimeResult};
pub use flash_attention::{
    flash_attention, flash_attention_gqa, flash_attention_multi_head, FlashAttentionConfig,
};
pub use kv_cache::KvCache;
pub use lora_loader::apply_lora;
pub use sampling::chain::{SamplerChain, SamplerStage};
pub use sampling::grammar::{Grammar, GrammarError, GrammarState};
pub use sampling::{sample, Sampler, SamplerConfig};
pub use scheduler::{Scheduler, SchedulerConfig};
pub use speculative::{SpeculativeConfig, SpeculativeEngine};
// TokenizerBridge is always exported — when neither `tokenizer-wasm` nor
// `tokenizer-onig` is enabled the struct still exists but all methods return
// TokenizerNotAvailable.
pub use tokenizer_bridge::TokenizerBridge;

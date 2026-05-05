//! # oxillama-runtime
//!
//! Inference runtime for OxiLLaMa.
//!
//! Orchestrates the complete inference pipeline: model loading, tokenization,
//! forward pass execution, KV caching, and token sampling.

pub mod batched_attention;
pub mod beam_search;
pub mod embedding;
pub mod engine;
pub mod error;
pub mod flash_attention;
pub mod kv_cache;
pub mod kv_pool;
pub mod lora_loader;
pub mod metrics;
pub mod offload;
pub mod sampling;
pub mod scheduler;
pub mod sequence_pool;
pub mod snapshot;
pub mod speculative;
#[cfg(feature = "native-async")]
pub mod speculative_async;
pub mod tokenizer_bridge;
pub mod tool_dispatch;

pub use batched_attention::batched_flash_attention;
pub use beam_search::{
    beam_generate, BeamForwardPass, BeamHypothesis, BeamSearchConfig, EngineBeamAdapter,
};
pub use embedding::PoolingMode;
pub use engine::{EngineConfig, InferenceEngine, FLASH_ATTN_THRESHOLD};
pub use error::{RuntimeError, RuntimeResult};
pub use flash_attention::{
    flash_attention, flash_attention_forward, flash_attention_gqa, flash_attention_multi_head,
    FlashAttentionConfig,
};
pub use kv_cache::prefix::{CachedKvState, PrefixCacheConfig, PrefixKvCache};
pub use kv_cache::{BatchedKvView, KvCache, KvCacheSnapshot, KvSlot, VecBatchedKvView};
pub use kv_pool::KvCachePool;
pub use lora_loader::apply_lora;
pub use metrics::{EngineMetrics, MetricsSnapshot};
pub use offload::{
    FilePagerSource, LayerPager, MemoryPressureProbe, OffloadPolicy, PagerSource, ResidentTensor,
    TensorEntry, TensorId,
};
pub use oxillama_arch::lora::LoadedLora;
pub use oxillama_arch::LoraStack;
pub use sampling::advanced::{DryStage, EtaStage, TopAStage, TypicalPStage, XtcStage};
pub use sampling::chain::{LogitBias, SamplerChain, SamplerStage};
pub use sampling::grammar::{Grammar, GrammarError, GrammarState, JsonSchemaCompiler};
pub use sampling::{sample, Sampler, SamplerConfig};
pub use scheduler::{Scheduler, SchedulerConfig, MAX_DECODE_WAIT_MS, PREFILL_CHUNK};
pub use sequence_pool::{PoolError, PoolResult, SequencePool, SequenceSlot, SsmStatePool};
pub use speculative::{SpeculativeConfig, SpeculativeDeltaSync, SpeculativeEngine};
#[cfg(feature = "native-async")]
pub use speculative_async::{
    AsyncSpecConfig, RewindError, Rewindable, SpecStats, SpeculativeDecoder,
};
pub use tool_dispatch::{
    no_op_dispatcher, NoOpDispatcher, ToolCall, ToolCallDetector, ToolCallGrammar, ToolDispatcher,
    ToolResult,
};
// TokenizerBridge is always exported — when neither `tokenizer-wasm` nor
// `tokenizer-onig` is enabled the struct still exists but all methods return
// TokenizerNotAvailable.
pub use tokenizer_bridge::TokenizerBridge;

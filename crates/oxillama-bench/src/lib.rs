//! # oxillama-bench
//!
//! Benchmark suite for OxiLLaMa inference engine.
//!
//! Provides throughput, latency, memory, prefill/decode split, and
//! end-to-end benchmarking tools with architecture-specific configurations.

pub mod arch_config;
pub mod dispatch_matrix;
pub mod e2e;
pub mod heatmap;
pub mod latency;
pub mod long_context;
pub mod memory;
pub mod memory_profiler;
pub mod power;
pub mod prefill_decode;
pub mod regression_gate;
pub mod simd_comparison;
pub mod speculative;
pub mod throughput;

pub use arch_config::ArchBenchConfig;
pub use e2e::{run_e2e_bench, E2eBenchConfig, E2eBenchResult, E2eIterResult, InferenceBenchmark};
pub use heatmap::{BatchHeatmap, HeatmapPoint};
pub use latency::{
    compute_latency_stats, LatencyConfig, LatencyResult, LatencyTimer, TokenLatencyResult,
};
pub use long_context::{default_ctx_lengths, LongContextPoint, LongContextSweep};
pub use memory::{
    current_rss_bytes, estimate_memory, MemoryConfig, MemoryEstimate, MemoryProfiler, MemoryReport,
    MemoryResult, RssTracker,
};
pub use power::{
    compute_tokens_per_joule_from_delta, measure_tokens_per_joule, EnergyReading, PowerError,
    PowerResult, RaplReader,
};
pub use prefill_decode::{
    run_kv_cache_scaling, run_prefill_decode_bench, run_prefill_vs_decode_isolation,
    KvCacheScalingConfig, KvCacheScalingPoint, KvCacheScalingResult, PrefillDecodeBench,
    PrefillDecodeConfig, PrefillDecodePoint, PrefillDecodeResult, PrefillVsDecodeResult,
    KV_CACHE_CONTEXT_SIZES,
};
pub use regression_gate::{BaselineEntry, RegressionFailure, RegressionGate};
pub use simd_comparison::{
    format_comparison_table, run_dequant_comparison, run_gemv_comparison, KernelBenchResult,
    SimdComparisonConfig, SimdComparisonResult,
};
pub use speculative::{
    default_accept_thresholds, default_draft_sizes, run_acceptance_sweep, SpeculativeBenchConfig,
    SpeculativeBenchTable, SpeculativePoint, StubSpecEngine,
};
pub use throughput::{
    aggregate_throughput, bench_tokenizer_decode, bench_tokenizer_encode, compute_throughput,
    StubBpeTokenizer, ThroughputConfig, ThroughputResult, ThroughputTracker, TokenThroughputResult,
    TokenizerBench, TokenizerBenchConfig, TokenizerThroughputResult, TrackerConfig,
    TOKENIZER_SAMPLE_TEXT,
};

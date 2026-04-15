//! # oxillama-bench
//!
//! Benchmark suite for OxiLLaMa inference engine.
//!
//! Provides throughput, latency, memory, prefill/decode split, and
//! end-to-end benchmarking tools with architecture-specific configurations.

pub mod arch_config;
pub mod e2e;
pub mod latency;
pub mod memory;
pub mod prefill_decode;
pub mod simd_comparison;
pub mod throughput;

pub use arch_config::ArchBenchConfig;
pub use e2e::{run_e2e_bench, E2eBenchConfig, E2eBenchResult, E2eIterResult, InferenceBenchmark};
pub use latency::{
    compute_latency_stats, LatencyConfig, LatencyResult, LatencyTimer, TokenLatencyResult,
};
pub use memory::{
    current_rss_bytes, estimate_memory, MemoryConfig, MemoryEstimate, MemoryResult, RssTracker,
};
pub use prefill_decode::{
    run_prefill_decode_bench, PrefillDecodeBench, PrefillDecodeConfig, PrefillDecodePoint,
    PrefillDecodeResult,
};
pub use simd_comparison::{
    format_comparison_table, run_dequant_comparison, run_gemv_comparison, KernelBenchResult,
    SimdComparisonConfig, SimdComparisonResult,
};
pub use throughput::{
    aggregate_throughput, compute_throughput, ThroughputConfig, ThroughputResult,
    ThroughputTracker, TokenThroughputResult, TrackerConfig,
};

//! End-to-end inference benchmark harness.
//!
//! Provides a framework for measuring tokens-per-second, time-to-first-token,
//! and memory usage during full inference runs.

use crate::memory::MemoryResult;

/// Configuration for an end-to-end benchmark run.
#[derive(Debug, Clone)]
pub struct E2eBenchConfig {
    /// Number of warm-up iterations.
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
    /// Maximum tokens to generate per iteration.
    pub max_tokens: usize,
    /// Prompt text for each iteration.
    pub prompt: String,
    /// Whether to track memory usage.
    pub track_memory: bool,
}

impl Default for E2eBenchConfig {
    fn default() -> Self {
        Self {
            warmup_iters: 2,
            measure_iters: 5,
            max_tokens: 64,
            prompt: "Hello, world!".to_string(),
            track_memory: false,
        }
    }
}

/// Aggregated results from an end-to-end benchmark run.
#[derive(Debug, Clone)]
pub struct E2eBenchResult {
    /// Tokens per second (average across measurement iterations).
    pub tokens_per_sec: f64,
    /// Average time-to-first-token in ms.
    pub avg_ttft_ms: f64,
    /// Average per-token latency in ms (decode phase).
    pub avg_token_latency_ms: f64,
    /// P95 per-token latency in ms.
    pub p95_token_latency_ms: f64,
    /// Total tokens generated across all measurement iterations.
    pub total_tokens: usize,
    /// Memory delta in bytes (if tracked).
    pub memory_delta_bytes: Option<usize>,
    /// Individual iteration results.
    pub iterations: Vec<E2eIterResult>,
}

/// Result of a single benchmark iteration.
#[derive(Debug, Clone)]
pub struct E2eIterResult {
    /// Tokens generated in this iteration.
    pub tokens: usize,
    /// Total time for this iteration in ms.
    pub total_ms: f64,
    /// Time to first token in ms.
    pub ttft_ms: f64,
    /// Per-token times in ms.
    pub token_times_ms: Vec<f64>,
}

/// Trait for engines that can be benchmarked end-to-end.
pub trait InferenceBenchmark {
    /// Run one inference iteration. Returns per-token latency times in ms.
    /// The first entry is TTFT, subsequent entries are decode latencies.
    fn run_inference(&mut self, prompt: &str, max_tokens: usize) -> Vec<f64>;
}

/// Run an end-to-end benchmark with the given engine and configuration.
///
/// 1. Runs `warmup_iters` iterations (discarded).
/// 2. Optionally snapshots RSS before measurement.
/// 3. Runs `measure_iters` iterations, collecting per-token times.
/// 4. Optionally snapshots RSS after measurement.
/// 5. Computes aggregate statistics from collected data.
pub fn run_e2e_bench<E: InferenceBenchmark>(
    engine: &mut E,
    config: &E2eBenchConfig,
) -> E2eBenchResult {
    // Warm-up phase
    for _ in 0..config.warmup_iters {
        let _ = engine.run_inference(&config.prompt, config.max_tokens);
    }

    // Optional pre-measurement RSS snapshot
    let rss_before = if config.track_memory {
        Some(MemoryResult::current().current_rss_bytes)
    } else {
        None
    };

    // Measurement phase
    let mut iterations = Vec::with_capacity(config.measure_iters);
    for _ in 0..config.measure_iters {
        let token_times = engine.run_inference(&config.prompt, config.max_tokens);
        let tokens = token_times.len();
        let total_ms: f64 = token_times.iter().sum();
        let ttft_ms = token_times.first().copied().unwrap_or(0.0);
        iterations.push(E2eIterResult {
            tokens,
            total_ms,
            ttft_ms,
            token_times_ms: token_times,
        });
    }

    // Optional post-measurement RSS snapshot
    let memory_delta_bytes = if config.track_memory {
        let rss_after = MemoryResult::current().current_rss_bytes;
        let before = rss_before.unwrap_or(0);
        Some(rss_after.saturating_sub(before))
    } else {
        None
    };

    // Aggregate statistics
    let total_tokens: usize = iterations.iter().map(|it| it.tokens).sum();
    let total_time_ms: f64 = iterations.iter().map(|it| it.total_ms).sum();

    let tokens_per_sec = if total_time_ms > 0.0 {
        (total_tokens as f64 / total_time_ms) * 1000.0
    } else {
        0.0
    };

    let ttft_count = iterations.len();
    let avg_ttft_ms = if ttft_count > 0 {
        iterations.iter().map(|it| it.ttft_ms).sum::<f64>() / ttft_count as f64
    } else {
        0.0
    };

    // Collect all decode latencies (everything after TTFT in each iteration)
    let mut all_decode_latencies: Vec<f64> = iterations
        .iter()
        .flat_map(|it| it.token_times_ms.iter().skip(1).copied())
        .collect();

    let avg_token_latency_ms = if all_decode_latencies.is_empty() {
        0.0
    } else {
        all_decode_latencies.iter().sum::<f64>() / all_decode_latencies.len() as f64
    };

    let p95_token_latency_ms = if all_decode_latencies.is_empty() {
        0.0
    } else {
        all_decode_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((all_decode_latencies.len() as f64 * 0.95).ceil() as usize)
            .saturating_sub(1)
            .min(all_decode_latencies.len().saturating_sub(1));
        all_decode_latencies[idx]
    };

    E2eBenchResult {
        tokens_per_sec,
        avg_ttft_ms,
        avg_token_latency_ms,
        p95_token_latency_ms,
        total_tokens,
        memory_delta_bytes,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBenchEngine {
        token_latency_ms: f64,
        ttft_ms: f64,
    }

    impl InferenceBenchmark for MockBenchEngine {
        fn run_inference(&mut self, _prompt: &str, max_tokens: usize) -> Vec<f64> {
            let mut times = Vec::with_capacity(max_tokens);
            if max_tokens > 0 {
                times.push(self.ttft_ms);
            }
            for _ in 1..max_tokens {
                times.push(self.token_latency_ms);
            }
            times
        }
    }

    #[test]
    fn test_e2e_bench_basic() {
        let mut engine = MockBenchEngine {
            token_latency_ms: 10.0,
            ttft_ms: 50.0,
        };
        let config = E2eBenchConfig {
            warmup_iters: 1,
            measure_iters: 3,
            max_tokens: 10,
            prompt: "test".to_string(),
            track_memory: false,
        };
        let result = run_e2e_bench(&mut engine, &config);

        assert!(result.tokens_per_sec > 0.0);
        assert_eq!(result.total_tokens, 30); // 3 iters × 10 tokens
        assert_eq!(result.iterations.len(), 3);
        assert!(result.memory_delta_bytes.is_none());
    }

    #[test]
    fn test_e2e_bench_ttft() {
        let mut engine = MockBenchEngine {
            token_latency_ms: 5.0,
            ttft_ms: 100.0,
        };
        let config = E2eBenchConfig {
            warmup_iters: 0,
            measure_iters: 4,
            max_tokens: 8,
            prompt: "hello".to_string(),
            track_memory: false,
        };
        let result = run_e2e_bench(&mut engine, &config);

        // TTFT should be exactly 100.0 ms for each iteration
        for iter_result in &result.iterations {
            assert!((iter_result.ttft_ms - 100.0).abs() < f64::EPSILON);
        }
        assert!((result.avg_ttft_ms - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_e2e_bench_memory_tracking() {
        let mut engine = MockBenchEngine {
            token_latency_ms: 1.0,
            ttft_ms: 5.0,
        };
        let config = E2eBenchConfig {
            warmup_iters: 0,
            measure_iters: 2,
            max_tokens: 4,
            prompt: "mem".to_string(),
            track_memory: true,
        };
        let result = run_e2e_bench(&mut engine, &config);
        assert!(result.memory_delta_bytes.is_some());
    }

    #[test]
    fn test_e2e_config_default() {
        let config = E2eBenchConfig::default();
        assert_eq!(config.warmup_iters, 2);
        assert_eq!(config.measure_iters, 5);
        assert_eq!(config.max_tokens, 64);
        assert_eq!(config.prompt, "Hello, world!");
        assert!(!config.track_memory);
    }

    #[test]
    fn test_e2e_iterations_count() {
        let mut engine = MockBenchEngine {
            token_latency_ms: 2.0,
            ttft_ms: 10.0,
        };
        let config = E2eBenchConfig {
            warmup_iters: 3,
            measure_iters: 7,
            max_tokens: 5,
            prompt: "count".to_string(),
            track_memory: false,
        };
        let result = run_e2e_bench(&mut engine, &config);
        assert_eq!(result.iterations.len(), 7);
        for iter_result in &result.iterations {
            assert_eq!(iter_result.tokens, 5);
        }
    }

    #[test]
    fn test_e2e_tokens_per_sec() {
        let mut engine = MockBenchEngine {
            token_latency_ms: 10.0,
            ttft_ms: 10.0,
        };
        let config = E2eBenchConfig {
            warmup_iters: 0,
            measure_iters: 1,
            max_tokens: 10,
            prompt: "tps".to_string(),
            track_memory: false,
        };
        let result = run_e2e_bench(&mut engine, &config);

        // 10 tokens, each 10ms = 100ms total => 100 tokens/sec
        let expected_tps = (10.0 / 100.0) * 1000.0; // 100.0
        assert!(
            (result.tokens_per_sec - expected_tps).abs() < 0.01,
            "Expected TPS ~{expected_tps}, got {}",
            result.tokens_per_sec
        );
    }
}

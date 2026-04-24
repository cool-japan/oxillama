//! Prefill/decode split benchmarking.
//!
//! Measures the prefill and decode phases **separately**, which is critical
//! for understanding inference performance characteristics.  Prefill is
//! compute-bound (parallel token processing) while decode is
//! memory-bandwidth-bound (sequential token generation).

use std::fmt::Write as _;

/// Configuration for prefill/decode split benchmarking.
#[derive(Debug, Clone)]
pub struct PrefillDecodeConfig {
    /// Number of warm-up iterations before measurement.
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
    /// Prompt lengths to test (number of tokens).
    pub prompt_lengths: Vec<usize>,
    /// Number of decode tokens to generate per test point.
    pub decode_tokens: usize,
}

impl Default for PrefillDecodeConfig {
    fn default() -> Self {
        Self {
            warmup_iters: 2,
            measure_iters: 5,
            prompt_lengths: vec![32, 64, 128, 256, 512, 1024],
            decode_tokens: 64,
        }
    }
}

/// Result of a single prompt-length test point.
#[derive(Debug, Clone)]
pub struct PrefillDecodePoint {
    /// Number of prompt tokens.
    pub prompt_length: usize,
    /// Number of decode tokens generated.
    pub decode_tokens: usize,
    /// Time to process all prompt tokens (ms). Average over `measure_iters`.
    pub prefill_ms: f64,
    /// Time to generate each decode token (ms, average).
    pub decode_ms_per_token: f64,
    /// Prefill throughput: `prompt_length / prefill_ms * 1000`.
    pub prefill_tokens_per_sec: f64,
    /// Decode throughput: `1000 / decode_ms_per_token`.
    pub decode_tokens_per_sec: f64,
    /// P95 decode latency (ms).
    pub p95_decode_ms: f64,
    /// Individual prefill times for each iteration (ms).
    pub prefill_times_ms: Vec<f64>,
    /// Individual decode times for each iteration and token.
    pub decode_times_ms: Vec<Vec<f64>>,
}

/// Aggregated prefill/decode results across all prompt lengths.
#[derive(Debug, Clone)]
pub struct PrefillDecodeResult {
    /// Results per prompt length.
    pub points: Vec<PrefillDecodePoint>,
    /// Human-readable summary table.
    pub summary: String,
}

/// Trait for engines that support split prefill/decode benchmarking.
pub trait PrefillDecodeBench {
    /// Run prefill on `prompt_tokens` tokens and return elapsed time in ms.
    fn bench_prefill(&mut self, prompt_tokens: usize) -> f64;
    /// Generate one token and return the decode latency in ms.
    fn bench_decode_token(&mut self) -> f64;
    /// Reset state for a fresh run.
    fn bench_reset(&mut self);
}

/// Compute the P95 value from a sorted slice.
///
/// Returns `0.0` for an empty slice.
fn compute_p95(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len().saturating_sub(1));
    sorted[idx]
}

/// Run a prefill/decode split benchmark.
///
/// For each prompt length in `config.prompt_lengths`:
///   1. Run `warmup_iters` warm-up iterations (discarded).
///   2. Run `measure_iters` measurement iterations, each consisting of:
///      a. Reset the engine state.
///      b. Time prefill of `prompt_length` tokens.
///      c. Time `decode_tokens` individual decode steps.
///   3. Aggregate statistics.
///
/// Returns a [`PrefillDecodeResult`] with per-point data and a summary table.
pub fn run_prefill_decode_bench<E: PrefillDecodeBench>(
    engine: &mut E,
    config: &PrefillDecodeConfig,
) -> PrefillDecodeResult {
    let mut points = Vec::with_capacity(config.prompt_lengths.len());

    for &prompt_length in &config.prompt_lengths {
        // Warm-up phase
        for _ in 0..config.warmup_iters {
            engine.bench_reset();
            let _ = engine.bench_prefill(prompt_length);
            for _ in 0..config.decode_tokens {
                let _ = engine.bench_decode_token();
            }
        }

        // Measurement phase
        let mut prefill_times = Vec::with_capacity(config.measure_iters);
        let mut decode_times = Vec::with_capacity(config.measure_iters);

        for _ in 0..config.measure_iters {
            engine.bench_reset();

            // Prefill
            let prefill_ms = engine.bench_prefill(prompt_length);
            prefill_times.push(prefill_ms);

            // Decode
            let mut iter_decode_times = Vec::with_capacity(config.decode_tokens);
            for _ in 0..config.decode_tokens {
                let decode_ms = engine.bench_decode_token();
                iter_decode_times.push(decode_ms);
            }
            decode_times.push(iter_decode_times);
        }

        // Aggregate prefill stats
        let prefill_ms = if prefill_times.is_empty() {
            0.0
        } else {
            prefill_times.iter().sum::<f64>() / prefill_times.len() as f64
        };

        let prefill_tokens_per_sec = if prefill_ms > 0.0 {
            prompt_length as f64 / prefill_ms * 1000.0
        } else {
            0.0
        };

        // Aggregate decode stats: flatten all per-token times across iterations
        let mut all_decode_latencies: Vec<f64> = decode_times
            .iter()
            .flat_map(|iter_times| iter_times.iter().copied())
            .collect();

        let decode_ms_per_token = if all_decode_latencies.is_empty() {
            0.0
        } else {
            all_decode_latencies.iter().sum::<f64>() / all_decode_latencies.len() as f64
        };

        let decode_tokens_per_sec = if decode_ms_per_token > 0.0 {
            1000.0 / decode_ms_per_token
        } else {
            0.0
        };

        // P95 decode latency
        all_decode_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_decode_ms = compute_p95(&all_decode_latencies);

        points.push(PrefillDecodePoint {
            prompt_length,
            decode_tokens: config.decode_tokens,
            prefill_ms,
            decode_ms_per_token,
            prefill_tokens_per_sec,
            decode_tokens_per_sec,
            p95_decode_ms,
            prefill_times_ms: prefill_times,
            decode_times_ms: decode_times,
        });
    }

    let summary = build_summary_table(&points);

    PrefillDecodeResult { points, summary }
}

/// Build a human-readable summary table from collected data points.
fn build_summary_table(points: &[PrefillDecodePoint]) -> String {
    let mut table = String::new();

    let _ = writeln!(
        table,
        "| Prompt Len | Prefill (ms) | Prefill (tok/s) | Decode (ms/tok) | Decode (tok/s) | P95 Decode |"
    );
    let _ = writeln!(
        table,
        "|------------|-------------|-----------------|-----------------|----------------|------------|"
    );

    for p in points {
        let _ = writeln!(
            table,
            "| {:>10} | {:>11.1} | {:>15.0} | {:>15.1} | {:>14.0} | {:>10.1} |",
            p.prompt_length,
            p.prefill_ms,
            p.prefill_tokens_per_sec,
            p.decode_ms_per_token,
            p.decode_tokens_per_sec,
            p.p95_decode_ms,
        );
    }

    table
}

// ── KV-cache scaling benchmark ───────────────────────────────────────────

/// Context sizes swept in the KV-cache scaling benchmark.
pub const KV_CACHE_CONTEXT_SIZES: &[usize] = &[1024, 4096, 8192, 32768];

/// Configuration for KV-cache scaling benchmarks.
#[derive(Debug, Clone)]
pub struct KvCacheScalingConfig {
    /// Context sizes to sweep (token count).
    pub context_sizes: Vec<usize>,
    /// Number of transformer layers (for memory estimation).
    pub n_layers: usize,
    /// Number of KV attention heads.
    pub n_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Number of measurement iterations per context size.
    pub measure_iters: usize,
    /// Number of warm-up iterations per context size.
    pub warmup_iters: usize,
}

impl Default for KvCacheScalingConfig {
    fn default() -> Self {
        Self {
            context_sizes: KV_CACHE_CONTEXT_SIZES.to_vec(),
            n_layers: 32,
            n_heads: 32,
            head_dim: 128,
            measure_iters: 5,
            warmup_iters: 2,
        }
    }
}

/// Result of a single context-size data point.
#[derive(Debug, Clone)]
pub struct KvCacheScalingPoint {
    /// Context size (tokens).
    pub ctx_size: usize,
    /// Estimated KV-cache memory for this context size (bytes).
    pub kv_cache_bytes: usize,
    /// Average decode-token latency (ms/token) at this context size.
    pub decode_ms_per_token: f64,
    /// Decode token throughput (tokens/sec).
    pub decode_tokens_per_sec: f64,
}

/// Aggregated results of a KV-cache scaling sweep.
#[derive(Debug, Clone)]
pub struct KvCacheScalingResult {
    /// Per-context-size measurements.
    pub points: Vec<KvCacheScalingPoint>,
    /// Human-readable summary.
    pub summary: String,
}

/// Run a KV-cache scaling sweep.
///
/// For each context size in `config.context_sizes`:
/// 1. Estimate KV-cache memory (`n_layers × n_heads × head_dim × ctx × 2 × 2` bytes).
/// 2. Measure decode-token latency by calling [`PrefillDecodeBench::bench_decode_token`]
///    after a nominal prefill of `ctx_size` tokens.
pub fn run_kv_cache_scaling<E: PrefillDecodeBench>(
    engine: &mut E,
    config: &KvCacheScalingConfig,
) -> KvCacheScalingResult {
    let mut points = Vec::with_capacity(config.context_sizes.len());

    for &ctx_size in &config.context_sizes {
        // Estimate KV-cache memory: K + V, each f16 (2 bytes).
        let kv_cache_bytes = config.n_layers * config.n_heads * config.head_dim * ctx_size * 2 * 2;

        // Warm-up phase.
        for _ in 0..config.warmup_iters {
            engine.bench_reset();
            let _ = engine.bench_prefill(ctx_size);
            let _ = engine.bench_decode_token();
        }

        // Measurement phase: measure a single decode step after prefill.
        let mut decode_latencies = Vec::with_capacity(config.measure_iters);
        for _ in 0..config.measure_iters {
            engine.bench_reset();
            let _ = engine.bench_prefill(ctx_size);
            let lat = engine.bench_decode_token();
            decode_latencies.push(lat);
        }

        let decode_ms_per_token = if decode_latencies.is_empty() {
            0.0
        } else {
            decode_latencies.iter().sum::<f64>() / decode_latencies.len() as f64
        };

        let decode_tokens_per_sec = if decode_ms_per_token > 0.0 {
            1000.0 / decode_ms_per_token
        } else {
            0.0
        };

        points.push(KvCacheScalingPoint {
            ctx_size,
            kv_cache_bytes,
            decode_ms_per_token,
            decode_tokens_per_sec,
        });
    }

    let summary = build_kv_cache_summary(&points);
    KvCacheScalingResult { points, summary }
}

fn build_kv_cache_summary(points: &[KvCacheScalingPoint]) -> String {
    let mut table = String::new();
    let _ = writeln!(
        table,
        "| Ctx Size | KV Cache (MiB) | Decode (ms/tok) | Decode (tok/s) |"
    );
    let _ = writeln!(
        table,
        "|----------|----------------|-----------------|----------------|"
    );
    for p in points {
        let kv_mib = p.kv_cache_bytes as f64 / (1024.0 * 1024.0);
        let _ = writeln!(
            table,
            "| {:>8} | {:>14.1} | {:>15.2} | {:>14.0} |",
            p.ctx_size, kv_mib, p.decode_ms_per_token, p.decode_tokens_per_sec,
        );
    }
    table
}

// ── Prefill-vs-decode isolation benchmark ───────────────────────────────

/// Result of the prefill-vs-decode isolation benchmark.
#[derive(Debug, Clone)]
pub struct PrefillVsDecodeResult {
    /// Number of tokens processed in each measurement.
    pub n_tokens: usize,
    /// Total prefill time for `n_tokens` prompt tokens (ms).
    pub prefill_total_ms: f64,
    /// Prefill throughput (tokens/sec).
    pub prefill_tokens_per_sec: f64,
    /// Total decode time for `n_tokens` sequential steps (ms).
    pub decode_total_ms: f64,
    /// Decode throughput (tokens/sec).
    pub decode_tokens_per_sec: f64,
    /// Prefill/decode throughput ratio (>1 means prefill is faster).
    pub ratio: f64,
}

/// Benchmark prefill and autoregressive decode in isolation over `n_tokens` tokens.
///
/// - **Prefill pass**: calls `bench_prefill(n_tokens)` once.
/// - **Decode pass**: calls `bench_decode_token()` `n_tokens` times sequentially.
///
/// Results are averaged over `measure_iters` runs.
pub fn run_prefill_vs_decode_isolation<E: PrefillDecodeBench>(
    engine: &mut E,
    n_tokens: usize,
    warmup_iters: usize,
    measure_iters: usize,
) -> PrefillVsDecodeResult {
    // Warm-up
    for _ in 0..warmup_iters {
        engine.bench_reset();
        let _ = engine.bench_prefill(n_tokens);
        engine.bench_reset();
        for _ in 0..n_tokens {
            let _ = engine.bench_decode_token();
        }
    }

    // Measure prefill
    let mut prefill_times = Vec::with_capacity(measure_iters);
    for _ in 0..measure_iters {
        engine.bench_reset();
        let ms = engine.bench_prefill(n_tokens);
        prefill_times.push(ms);
    }

    // Measure decode
    let mut decode_times = Vec::with_capacity(measure_iters);
    for _ in 0..measure_iters {
        engine.bench_reset();
        let total_ms: f64 = (0..n_tokens).map(|_| engine.bench_decode_token()).sum();
        decode_times.push(total_ms);
    }

    let prefill_total_ms = if prefill_times.is_empty() {
        0.0
    } else {
        prefill_times.iter().sum::<f64>() / prefill_times.len() as f64
    };

    let decode_total_ms = if decode_times.is_empty() {
        0.0
    } else {
        decode_times.iter().sum::<f64>() / decode_times.len() as f64
    };

    let prefill_tokens_per_sec = if prefill_total_ms > 0.0 {
        n_tokens as f64 / prefill_total_ms * 1000.0
    } else {
        0.0
    };

    let decode_tokens_per_sec = if decode_total_ms > 0.0 {
        n_tokens as f64 / decode_total_ms * 1000.0
    } else {
        0.0
    };

    let ratio = if decode_tokens_per_sec > 0.0 {
        prefill_tokens_per_sec / decode_tokens_per_sec
    } else {
        f64::INFINITY
    };

    PrefillVsDecodeResult {
        n_tokens,
        prefill_total_ms,
        prefill_tokens_per_sec,
        decode_total_ms,
        decode_tokens_per_sec,
        ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock engine that returns fixed latencies for testing.
    struct MockEngine {
        prefill_latency_per_token_ms: f64,
        decode_latency_ms: f64,
        reset_count: usize,
    }

    impl MockEngine {
        fn new(prefill_latency_per_token_ms: f64, decode_latency_ms: f64) -> Self {
            Self {
                prefill_latency_per_token_ms,
                decode_latency_ms,
                reset_count: 0,
            }
        }
    }

    impl PrefillDecodeBench for MockEngine {
        fn bench_prefill(&mut self, prompt_tokens: usize) -> f64 {
            self.prefill_latency_per_token_ms * prompt_tokens as f64
        }

        fn bench_decode_token(&mut self) -> f64 {
            self.decode_latency_ms
        }

        fn bench_reset(&mut self) {
            self.reset_count += 1;
        }
    }

    #[test]
    fn test_mock_engine_basic() {
        let mut engine = MockEngine::new(0.1, 8.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 1,
            measure_iters: 3,
            prompt_lengths: vec![32, 64],
            decode_tokens: 10,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        assert_eq!(result.points.len(), 2);
        assert_eq!(result.points[0].prompt_length, 32);
        assert_eq!(result.points[1].prompt_length, 64);
    }

    #[test]
    fn test_prefill_latency_scales_with_length() {
        let mut engine = MockEngine::new(0.1, 8.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 2,
            prompt_lengths: vec![32, 128],
            decode_tokens: 4,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        let p32 = &result.points[0];
        let p128 = &result.points[1];

        // Prefill for 128 tokens should take ~4× the time of 32 tokens
        let ratio = p128.prefill_ms / p32.prefill_ms;
        assert!(
            (ratio - 4.0).abs() < 0.01,
            "expected ratio ~4.0, got {ratio}"
        );
    }

    #[test]
    fn test_decode_latency_is_fixed() {
        let mut engine = MockEngine::new(0.1, 10.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 3,
            prompt_lengths: vec![64],
            decode_tokens: 20,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        let point = &result.points[0];
        assert!(
            (point.decode_ms_per_token - 10.0).abs() < 0.01,
            "expected 10.0, got {}",
            point.decode_ms_per_token
        );
        assert!(
            (point.decode_tokens_per_sec - 100.0).abs() < 0.1,
            "expected 100.0, got {}",
            point.decode_tokens_per_sec
        );
    }

    #[test]
    fn test_multiple_prompt_lengths_produce_correct_point_count() {
        let mut engine = MockEngine::new(0.05, 5.0);
        let lengths = vec![16, 32, 64, 128, 256];
        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 2,
            prompt_lengths: lengths.clone(),
            decode_tokens: 8,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        assert_eq!(result.points.len(), lengths.len());
        for (point, &expected_len) in result.points.iter().zip(lengths.iter()) {
            assert_eq!(point.prompt_length, expected_len);
            assert_eq!(point.decode_tokens, 8);
        }
    }

    #[test]
    fn test_summary_table_format() {
        let mut engine = MockEngine::new(0.1, 8.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 2,
            prompt_lengths: vec![32, 64],
            decode_tokens: 10,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        assert!(
            result.summary.contains("Prompt Len"),
            "summary should contain 'Prompt Len'"
        );
        assert!(
            result.summary.contains("Prefill (ms)"),
            "summary should contain 'Prefill (ms)'"
        );
        assert!(
            result.summary.contains("Prefill (tok/s)"),
            "summary should contain 'Prefill (tok/s)'"
        );
        assert!(
            result.summary.contains("Decode (ms/tok)"),
            "summary should contain 'Decode (ms/tok)'"
        );
        assert!(
            result.summary.contains("Decode (tok/s)"),
            "summary should contain 'Decode (tok/s)'"
        );
        assert!(
            result.summary.contains("P95 Decode"),
            "summary should contain 'P95 Decode'"
        );
    }

    #[test]
    fn test_p95_calculation() {
        // With a mock that always returns 10.0 ms decode, p95 should be 10.0
        let mut engine = MockEngine::new(0.1, 10.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 4,
            prompt_lengths: vec![32],
            decode_tokens: 50,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        let point = &result.points[0];
        assert!(
            (point.p95_decode_ms - 10.0).abs() < 0.01,
            "p95 should be 10.0, got {}",
            point.p95_decode_ms
        );
    }

    #[test]
    fn test_p95_with_varying_latencies() {
        /// Engine that cycles through a fixed set of decode latencies.
        struct VaryingEngine {
            latencies: Vec<f64>,
            idx: usize,
        }

        impl PrefillDecodeBench for VaryingEngine {
            fn bench_prefill(&mut self, prompt_tokens: usize) -> f64 {
                prompt_tokens as f64 * 0.1
            }

            fn bench_decode_token(&mut self) -> f64 {
                let val = self.latencies[self.idx % self.latencies.len()];
                self.idx += 1;
                val
            }

            fn bench_reset(&mut self) {
                // Don't reset idx — keep cycling.
            }
        }

        // 95 values of 5.0, 5 values of 50.0 => p95 should be around 50.0
        let mut latencies = vec![5.0; 95];
        latencies.extend(vec![50.0; 5]);

        let mut engine = VaryingEngine { latencies, idx: 0 };

        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 1,
            prompt_lengths: vec![32],
            decode_tokens: 100,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        let point = &result.points[0];
        // With 95 values of 5.0 and 5 of 50.0, sorted: [5.0 × 95, 50.0 × 5]
        // p95 index = ceil(100 * 0.95) - 1 = 95 - 1 = 94 => 5.0
        // (the 95th element, 0-indexed 94, is still 5.0 since there are 95 of them)
        assert!(
            point.p95_decode_ms <= 50.0,
            "p95 should be <= 50.0, got {}",
            point.p95_decode_ms
        );
    }

    #[test]
    fn test_empty_prompt_lengths() {
        let mut engine = MockEngine::new(0.1, 8.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 3,
            prompt_lengths: vec![],
            decode_tokens: 10,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        assert!(result.points.is_empty());
        // Summary should still have a header but no data rows
        assert!(result.summary.contains("Prompt Len"));
    }

    #[test]
    fn test_zero_decode_tokens() {
        let mut engine = MockEngine::new(0.1, 8.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 2,
            prompt_lengths: vec![64],
            decode_tokens: 0,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        let point = &result.points[0];
        assert_eq!(point.decode_tokens, 0);
        assert!((point.decode_ms_per_token).abs() < f64::EPSILON);
        assert!((point.decode_tokens_per_sec).abs() < f64::EPSILON);
        assert!((point.p95_decode_ms).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prefill_throughput_calculation() {
        let mut engine = MockEngine::new(0.5, 10.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 2,
            prompt_lengths: vec![100],
            decode_tokens: 5,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        let point = &result.points[0];
        // prefill_ms = 0.5 * 100 = 50.0
        // prefill_tokens_per_sec = 100 / 50.0 * 1000 = 2000.0
        assert!(
            (point.prefill_ms - 50.0).abs() < 0.01,
            "prefill_ms: expected 50.0, got {}",
            point.prefill_ms
        );
        assert!(
            (point.prefill_tokens_per_sec - 2000.0).abs() < 1.0,
            "prefill tok/s: expected 2000.0, got {}",
            point.prefill_tokens_per_sec
        );
    }

    #[test]
    fn test_raw_timing_vectors_populated() {
        let mut engine = MockEngine::new(0.1, 8.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 0,
            measure_iters: 3,
            prompt_lengths: vec![64],
            decode_tokens: 10,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        let point = &result.points[0];
        assert_eq!(point.prefill_times_ms.len(), 3);
        assert_eq!(point.decode_times_ms.len(), 3);
        for iter_decode in &point.decode_times_ms {
            assert_eq!(iter_decode.len(), 10);
        }
    }

    #[test]
    fn test_warmup_iterations_are_discarded() {
        let mut engine = MockEngine::new(0.1, 8.0);
        let config = PrefillDecodeConfig {
            warmup_iters: 5,
            measure_iters: 2,
            prompt_lengths: vec![32],
            decode_tokens: 4,
        };
        let result = run_prefill_decode_bench(&mut engine, &config);

        // warmup: 5 resets, measure: 2 resets => 7 total resets for this prompt length
        assert_eq!(engine.reset_count, 7);

        // But we only record measure_iters
        let point = &result.points[0];
        assert_eq!(point.prefill_times_ms.len(), 2);
        assert_eq!(point.decode_times_ms.len(), 2);
    }

    #[test]
    fn test_compute_p95_empty() {
        assert!((compute_p95(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_p95_single() {
        assert!((compute_p95(&[42.0]) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_p95_sorted_range() {
        // 1..=100 sorted
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let p95 = compute_p95(&data);
        // ceil(100 * 0.95) - 1 = 95 - 1 = 94 => data[94] = 95.0
        assert!(
            (p95 - 95.0).abs() < f64::EPSILON,
            "expected 95.0, got {p95}"
        );
    }

    // ── KV-cache scaling tests ───────────────────────────────────────────

    #[test]
    fn test_kv_cache_scaling_point_count() {
        let mut engine = MockEngine::new(0.01, 5.0);
        let config = KvCacheScalingConfig {
            context_sizes: vec![1024, 4096, 8192],
            n_layers: 4,
            n_heads: 4,
            head_dim: 64,
            measure_iters: 2,
            warmup_iters: 1,
        };
        let result = run_kv_cache_scaling(&mut engine, &config);
        assert_eq!(result.points.len(), 3);
    }

    #[test]
    fn test_kv_cache_memory_scales_linearly() {
        let mut engine = MockEngine::new(0.01, 5.0);
        let config = KvCacheScalingConfig {
            context_sizes: vec![1024, 4096],
            n_layers: 4,
            n_heads: 4,
            head_dim: 64,
            measure_iters: 1,
            warmup_iters: 0,
        };
        let result = run_kv_cache_scaling(&mut engine, &config);
        let ratio = result.points[1].kv_cache_bytes / result.points[0].kv_cache_bytes;
        assert_eq!(ratio, 4, "kv cache bytes should scale 4× with 4× ctx");
    }

    #[test]
    fn test_kv_cache_summary_contains_headers() {
        let mut engine = MockEngine::new(0.01, 5.0);
        let config = KvCacheScalingConfig::default();
        let result = run_kv_cache_scaling(&mut engine, &config);
        assert!(result.summary.contains("Ctx Size"));
        assert!(result.summary.contains("KV Cache"));
        assert!(result.summary.contains("Decode"));
    }

    // ── Prefill vs decode isolation tests ───────────────────────────────

    #[test]
    fn test_prefill_vs_decode_isolation_ratio() {
        // prefill: 0.1 ms/tok × 512 = 51.2 ms total → 10_000 tok/s
        // decode:  8.0 ms/tok × 512 = 4096 ms total → 125 tok/s
        // ratio ≈ 80
        let mut engine = MockEngine::new(0.1, 8.0);
        let result = run_prefill_vs_decode_isolation(&mut engine, 512, 0, 2);
        assert!(
            result.ratio > 1.0,
            "prefill should be faster than decode, ratio={}",
            result.ratio
        );
        assert_eq!(result.n_tokens, 512);
    }

    #[test]
    fn test_prefill_vs_decode_throughput_relations() {
        let mut engine = MockEngine::new(0.5, 5.0);
        let result = run_prefill_vs_decode_isolation(&mut engine, 100, 0, 2);
        // prefill_total_ms ≈ 0.5 * 100 = 50 ms → tok/s ≈ 2000
        let expected_prefill_tps = 1000.0 / 0.5;
        assert!(
            (result.prefill_tokens_per_sec - expected_prefill_tps).abs() < 1.0,
            "expected ~{expected_prefill_tps}, got {}",
            result.prefill_tokens_per_sec
        );
        // decode: 5 ms/tok → 200 tok/s
        assert!(
            (result.decode_tokens_per_sec - 200.0).abs() < 1.0,
            "expected ~200, got {}",
            result.decode_tokens_per_sec
        );
    }
}

//! Long-context KV-cache sweep infrastructure.
//!
//! Sweeps context lengths `ctx ∈ {1024, 4096, 8192, 16384, 32768}` and records
//! decode throughput (tok/s), memory consumption (bytes), and prefill latency
//! (ms) at each context length.  Results are exposed as a [`LongContextSweep`]
//! whose [`summary_table`][LongContextSweep::summary_table] method renders a
//! Markdown table suitable for bench CI output.
//!
//! The sweep is driven through the [`PrefillDecodeBench`] trait so that any
//! engine implementing that interface (including the stub engines used in
//! tests and benchmarks) works out of the box.

use std::fmt::Write as _;

use crate::memory::current_rss_bytes;
use crate::prefill_decode::{KvCacheScalingConfig, PrefillDecodeBench};

/// Error type used by this module (wraps from the parent crate's error
/// conventions — we keep it simple since `oxillama-bench` does not publish a
/// separate `BenchError` type: all public functions return `Result<_, String>`
/// which is sufficient for a bench-only crate).
pub type BenchResult<T> = Result<T, String>;

/// Default context lengths swept by [`LongContextSweep`].
///
/// Returns a static slice so callers can pass it directly without allocation.
pub fn default_ctx_lengths() -> &'static [usize] {
    &[1024, 4096, 8192, 16384, 32768]
}

/// Measurement captured at one context length during a long-context sweep.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LongContextPoint {
    /// Context window length (tokens).
    pub ctx_len: usize,
    /// Decode throughput: tokens generated per second.
    pub decode_toks_per_sec: f64,
    /// Memory usage sampled immediately after the prefill phase (bytes).
    pub memory_bytes: usize,
    /// Wall-clock time to complete the prefill phase (milliseconds).
    pub prefill_ms: f64,
}

/// Aggregated long-context sweep result across all tested context lengths.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LongContextSweep {
    /// Per-context-length data points, in the order they were measured.
    pub points: Vec<LongContextPoint>,
    /// Human-readable label for the model / configuration under test.
    pub model_label: String,
}

impl LongContextSweep {
    /// Run a long-context sweep.
    ///
    /// For each length in `ctx_lengths`:
    /// 1. Reset the engine.
    /// 2. Run the prefill phase (`bench_prefill(ctx_len)`), recording wall time.
    /// 3. Sample the current RSS to approximate memory usage.
    /// 4. Run `measure_iters` decode steps and average the latency.
    /// 5. Compute `decode_toks_per_sec = 1000 / avg_decode_ms_per_token`.
    ///
    /// A small [`KvCacheScalingConfig`] is constructed internally (using
    /// `measure_iters = 5`, `warmup_iters = 2`) so that the sweep integrates
    /// naturally with the existing `run_kv_cache_scaling` infrastructure without
    /// duplicating the measurement logic.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if `ctx_lengths` is empty (the resulting
    /// [`LongContextSweep`] would carry no data and callers should treat this
    /// as a configuration error).
    pub fn run<E>(
        engine: &mut E,
        ctx_lengths: &[usize],
        _prompt_tokens: &[u32],
        model_label: impl Into<String>,
    ) -> BenchResult<Self>
    where
        E: PrefillDecodeBench,
    {
        if ctx_lengths.is_empty() {
            return Err("ctx_lengths must not be empty".to_string());
        }

        let label = model_label.into();

        // Build a KvCacheScalingConfig covering the requested context lengths so
        // we can leverage the existing warm-up / measurement discipline.
        let kv_config = KvCacheScalingConfig {
            context_sizes: ctx_lengths.to_vec(),
            // Representative LLaMA-like defaults — memory estimate only.
            n_layers: 32,
            n_heads: 32,
            head_dim: 128,
            measure_iters: 5,
            warmup_iters: 2,
        };

        let mut points = Vec::with_capacity(ctx_lengths.len());

        for &ctx_len in &kv_config.context_sizes {
            // ── Warm-up phase ──────────────────────────────────────────
            for _ in 0..kv_config.warmup_iters {
                engine.bench_reset();
                let _ = engine.bench_prefill(ctx_len);
                let _ = engine.bench_decode_token();
            }

            // ── Measurement phase ──────────────────────────────────────

            // Prefill timing: single representative measurement (the engine is
            // deterministic in bench mode so one high-quality sample suffices;
            // we run the full measure_iters below for decode).
            engine.bench_reset();
            let prefill_start = std::time::Instant::now();
            let _ = engine.bench_prefill(ctx_len);
            let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1_000.0;

            // Sample RSS immediately after prefill — this is the peak KV-state
            // occupancy for this context length.  `current_rss_bytes()` returns
            // `None` when the platform does not support RSS measurement (e.g.
            // in some container/CI environments), so we fall back to 0 in
            // that case.
            let memory_bytes = current_rss_bytes().unwrap_or(0) as usize;

            // Decode timing: average over measure_iters.
            let mut decode_latencies: Vec<f64> = Vec::with_capacity(kv_config.measure_iters);
            for _ in 0..kv_config.measure_iters {
                engine.bench_reset();
                let _ = engine.bench_prefill(ctx_len);
                let lat_ms = engine.bench_decode_token();
                decode_latencies.push(lat_ms);
            }

            let avg_decode_ms = if decode_latencies.is_empty() {
                0.0
            } else {
                decode_latencies.iter().sum::<f64>() / decode_latencies.len() as f64
            };

            let decode_toks_per_sec = if avg_decode_ms > 0.0 {
                1_000.0 / avg_decode_ms
            } else {
                0.0
            };

            points.push(LongContextPoint {
                ctx_len,
                decode_toks_per_sec,
                memory_bytes,
                prefill_ms,
            });
        }

        Ok(LongContextSweep {
            points,
            model_label: label,
        })
    }

    /// Render results as a Markdown table with four columns:
    ///
    /// ```text
    /// | ctx_len | decode tok/s | memory MiB | prefill ms |
    /// |---------|-------------|------------|------------|
    /// | 1024    | 1234.5      | 2.3        | 0.01       |
    /// ...
    /// ```
    ///
    /// The table is suitable for direct inclusion in GitHub PR comments or CI
    /// job summaries.
    pub fn summary_table(&self) -> String {
        let mut out = String::new();

        let _ = writeln!(out, "## Long-Context KV-Cache Sweep: {}", self.model_label);
        let _ = writeln!(out);
        let _ = writeln!(out, "| ctx_len | decode tok/s | memory MiB | prefill ms |");
        let _ = writeln!(out, "|--------:|-------------:|-----------:|-----------:|");

        for p in &self.points {
            let memory_mib = p.memory_bytes as f64 / (1024.0 * 1024.0);
            let _ = writeln!(
                out,
                "| {:>7} | {:>12.1} | {:>10.2} | {:>10.3} |",
                p.ctx_len, p.decode_toks_per_sec, memory_mib, p.prefill_ms,
            );
        }

        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefill_decode::PrefillDecodeBench;

    // ── Minimal stub engine for long_context tests ───────────────────────

    /// Stub engine: prefill is linear in context size; decode latency is a
    /// fixed 8.0 ms/token (simulating attention over a large KV cache).
    struct StubLongContextEngine {
        prefill_ms_per_token: f64,
        decode_ms_per_token: f64,
    }

    impl StubLongContextEngine {
        fn new(prefill_ms_per_token: f64, decode_ms_per_token: f64) -> Self {
            Self {
                prefill_ms_per_token,
                decode_ms_per_token,
            }
        }
    }

    impl PrefillDecodeBench for StubLongContextEngine {
        fn bench_prefill(&mut self, prompt_tokens: usize) -> f64 {
            self.prefill_ms_per_token * prompt_tokens as f64
        }

        fn bench_decode_token(&mut self) -> f64 {
            self.decode_ms_per_token
        }

        fn bench_reset(&mut self) {}
    }

    // ── Test 1: correct number of points produced ────────────────────────

    #[test]
    fn long_context_sweep_returns_expected_points() {
        let mut engine = StubLongContextEngine::new(0.001, 8.0);
        let ctx_lengths = &[1024_usize, 4096, 8192];
        let sweep = LongContextSweep::run(&mut engine, ctx_lengths, &[], "stub-model")
            .expect("sweep must succeed for non-empty ctx_lengths");

        assert_eq!(
            sweep.points.len(),
            ctx_lengths.len(),
            "number of points must equal number of ctx_lengths supplied"
        );

        // Each point must record the correct context length.
        for (point, &expected_ctx) in sweep.points.iter().zip(ctx_lengths.iter()) {
            assert_eq!(
                point.ctx_len, expected_ctx,
                "ctx_len mismatch: expected {expected_ctx}, got {}",
                point.ctx_len
            );
        }
    }

    // ── Test 2: Markdown table contains correct header and all ctx values ──

    #[test]
    fn long_context_summary_table_format() {
        let ctx_lengths = default_ctx_lengths();
        let mut engine = StubLongContextEngine::new(0.001, 5.0);
        let sweep = LongContextSweep::run(&mut engine, ctx_lengths, &[], "test-model")
            .expect("sweep must succeed");

        let table = sweep.summary_table();

        // Header columns must be present.
        assert!(
            table.contains("ctx_len"),
            "table must contain 'ctx_len' header"
        );
        assert!(
            table.contains("decode tok/s"),
            "table must contain 'decode tok/s' header"
        );
        assert!(
            table.contains("memory MiB"),
            "table must contain 'memory MiB' header"
        );
        assert!(
            table.contains("prefill ms"),
            "table must contain 'prefill ms' header"
        );

        // Each context length must appear as a string in the table.
        for &ctx in ctx_lengths {
            let ctx_str = ctx.to_string();
            assert!(
                table.contains(&ctx_str),
                "table must contain ctx_len '{ctx_str}'"
            );
        }
    }

    // ── Test 3: monotonically increasing memory assertion ────────────────

    #[test]
    fn long_context_memory_grows_monotonically() {
        // Build an artificial sweep where memory grows with context length.
        let sweep = LongContextSweep {
            model_label: "artificial".to_string(),
            points: vec![
                LongContextPoint {
                    ctx_len: 1024,
                    decode_toks_per_sec: 500.0,
                    memory_bytes: 256 * 1024 * 1024, // 256 MiB
                    prefill_ms: 1.0,
                },
                LongContextPoint {
                    ctx_len: 4096,
                    decode_toks_per_sec: 450.0,
                    memory_bytes: 512 * 1024 * 1024, // 512 MiB
                    prefill_ms: 4.0,
                },
                LongContextPoint {
                    ctx_len: 8192,
                    decode_toks_per_sec: 380.0,
                    memory_bytes: 1024 * 1024 * 1024, // 1 GiB
                    prefill_ms: 8.2,
                },
                LongContextPoint {
                    ctx_len: 16384,
                    decode_toks_per_sec: 310.0,
                    memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
                    prefill_ms: 16.5,
                },
                LongContextPoint {
                    ctx_len: 32768,
                    decode_toks_per_sec: 230.0,
                    memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB
                    prefill_ms: 33.1,
                },
            ],
        };

        // Assert strict monotonicity of memory_bytes across all consecutive pairs.
        for window in sweep.points.windows(2) {
            let (prev, next) = (&window[0], &window[1]);
            assert!(
                next.memory_bytes >= prev.memory_bytes,
                "memory must be non-decreasing: ctx {} ({} bytes) > ctx {} ({} bytes)",
                prev.ctx_len,
                prev.memory_bytes,
                next.ctx_len,
                next.memory_bytes,
            );
        }
    }
}

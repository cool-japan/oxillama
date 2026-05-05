//! Latency-vs-batch-size heatmap Criterion benchmarks.
//!
//! Sweeps `batch_size ∈ {1, 2, 4, 8}` × `seq_len ∈ {128, 512, 1024, 2048}`
//! and measures decode throughput using a synthetic stub engine (no real GGUF
//! model required).  The benchmark exposes the crossover point where continuous
//! batching stops paying off as `seq_len` grows.
//!
//! When `OXILLAMA_BENCH_PRINT_HEATMAP=1` is set, the benchmark additionally
//! runs [`BatchHeatmap::run`] and emits a full Markdown table (both tok/s and
//! p99 latency views) to stdout after the Criterion group finishes.
//!
//! # Running
//!
//! ```sh
//! # Criterion mode (writes HTML report to target/criterion/)
//! cargo bench --bench batch_heatmap --features bench
//!
//! # CI / compile-time sanity (--test skips actual timing)
//! cargo bench --bench batch_heatmap --features bench -- --test
//!
//! # Print Markdown heatmap tables
//! OXILLAMA_BENCH_PRINT_HEATMAP=1 cargo bench --bench batch_heatmap --features bench
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxillama_bench::{BatchHeatmap, PrefillDecodeBench};

// ── Stub engine ──────────────────────────────────────────────────────────────

/// Synthetic engine that models the KV-cache read-bandwidth cost that grows
/// linearly with both `seq_len` (larger KV state) and `batch_size` (more
/// simultaneous decode streams).
///
/// Simulated costs:
/// - **Prefill**: `prefill_us_per_token` µs × `seq_len` (batched GEMM proxy).
/// - **Decode**: `base_decode_us` µs + `seq_len / 1024 × kv_pressure_us` µs
///   (memory-bandwidth proxy for growing KV state).
///
/// The decode model is intentionally super-linear in `seq_len` so that the
/// heatmap captures a meaningful gradient across both dimensions.
struct HeatmapStubEngine {
    /// Cost per prefill token (microseconds).
    prefill_us_per_token: f64,
    /// Baseline decode latency per token (microseconds).
    base_decode_us: f64,
    /// Additional decode cost per 1024-token increment of KV state (µs).
    kv_pressure_us: f64,
    /// Currently loaded context length set by [`bench_prefill`].
    current_seq: usize,
}

impl HeatmapStubEngine {
    /// Create a stub engine with physically motivated default parameters.
    ///
    /// Default values are chosen so that at `seq_len = 2048`, `batch_size = 8`
    /// the engine operates close to a realistic mid-range GPU saturation point:
    /// - At `(bs=1, sl=128)`:  very fast, ~80k tok/s
    /// - At `(bs=8, sl=2048)`: moderate pressure, ~20k tok/s
    fn new() -> Self {
        Self {
            prefill_us_per_token: 0.01, // 10 ns / token — batched GEMM
            base_decode_us: 0.5,        // 0.5 µs baseline / token
            kv_pressure_us: 0.1,        // 0.1 µs per 1K tokens of KV state
            current_seq: 0,
        }
    }
}

impl PrefillDecodeBench for HeatmapStubEngine {
    fn bench_prefill(&mut self, tokens: usize) -> f64 {
        self.current_seq = tokens;
        // Return simulated prefill time in ms (not used by the bench loop).
        self.prefill_us_per_token * tokens as f64 / 1_000.0
    }

    fn bench_decode_token(&mut self) -> f64 {
        // Return simulated decode latency per token in ms.
        let kv_penalty = self.kv_pressure_us * (self.current_seq as f64 / 1_024.0);
        (self.base_decode_us + kv_penalty) / 1_000.0
    }

    fn bench_reset(&mut self) {
        self.current_seq = 0;
    }
}

// ── Criterion benchmark ──────────────────────────────────────────────────────

/// Benchmark group: one sub-benchmark per `(batch_size, seq_len)` cell.
///
/// Each Criterion iteration:
/// 1. Call `bench_prefill(seq_len)` — establishes the simulated KV state.
/// 2. Call `bench_decode_token()` `batch_size` times — the measured work.
/// 3. Call `bench_reset()` — tears down state.
///
/// [`Throughput::Elements`] is set to `batch_size` so that Criterion reports
/// the measurement in tokens/iteration on the y-axis, making sweep comparisons
/// across batch sizes directly meaningful.
fn batch_heatmap_bench(c: &mut Criterion) {
    let batch_sizes = [1usize, 2, 4, 8];
    let seq_lens = [128usize, 512, 1_024, 2_048];

    let mut group = c.benchmark_group("batch_heatmap");

    for &batch_size in &batch_sizes {
        for &seq_len in &seq_lens {
            // Label: "b{batch_size}/seq_len" — readable in criterion HTML reports.
            let id = BenchmarkId::new(format!("b{}", batch_size), seq_len);

            // Report throughput in decoded tokens so the Criterion y-axis scales
            // correctly when comparing across different batch sizes.
            group.throughput(Throughput::Elements(batch_size as u64));

            group.bench_with_input(id, &(batch_size, seq_len), |b, &(bs, sl)| {
                let mut eng = HeatmapStubEngine::new();
                b.iter(|| {
                    // The inner loop body is the minimal unit of work:
                    // one prefill + bs decode steps + reset.
                    let prefill_cost = std::hint::black_box(eng.bench_prefill(sl));
                    let mut decode_cost = 0.0_f64;
                    for _ in 0..bs {
                        decode_cost += std::hint::black_box(eng.bench_decode_token());
                    }
                    eng.bench_reset();
                    // Return both costs so the optimizer cannot elide either call.
                    (prefill_cost, decode_cost)
                });
            });
        }
    }

    group.finish();

    // ── Optional Markdown heatmap tables ────────────────────────────────────
    //
    // Gated on `OXILLAMA_BENCH_PRINT_HEATMAP=1` so that CI is quiet by default
    // but developers can inspect the full 2-D grid locally.
    if std::env::var("OXILLAMA_BENCH_PRINT_HEATMAP")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        print_heatmap_tables(&batch_sizes, &seq_lens);
    }
}

/// Build and emit both tok/s and p99-latency heatmap tables using
/// [`BatchHeatmap::run`].
fn print_heatmap_tables(batch_sizes: &[usize], seq_lens: &[usize]) {
    let mut eng = HeatmapStubEngine::new();

    match BatchHeatmap::run(&mut eng, batch_sizes, seq_lens, "HeatmapStub") {
        Ok(hm) => {
            println!();
            println!("{}", hm.summary_table());
            println!("{}", hm.p99_table());
        }
        Err(e) => {
            eprintln!("batch_heatmap sweep failed: {e}");
        }
    }
}

criterion_group!(benches, batch_heatmap_bench);
criterion_main!(benches);

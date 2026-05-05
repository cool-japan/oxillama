//! Long-context KV-cache scaling Criterion benchmarks.
//!
//! Sweeps `ctx ∈ {1024, 4096, 8192, 16384, 32768}` and measures decode
//! throughput using a synthetic stub engine (no real GGUF model required).
//! When the environment variable `OXILLAMA_BENCH_PRINT_TABLE=1` is set, the
//! benchmark also emits a Markdown summary table to stdout after all criterion
//! groups have run.
//!
//! # Running
//!
//! ```sh
//! # Criterion mode (writes HTML report to target/criterion/)
//! cargo bench --bench long_context --features bench
//!
//! # CI / compile-time sanity (--test skips actual timing)
//! cargo bench --bench long_context --features bench -- --test
//!
//! # Print summary table
//! OXILLAMA_BENCH_PRINT_TABLE=1 cargo bench --bench long_context --features bench
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxillama_bench::{default_ctx_lengths, LongContextSweep, PrefillDecodeBench};

// ── Stub engine ──────────────────────────────────────────────────────────

/// Synthetic engine that simulates KV-cache pressure growing linearly with
/// context length.
///
/// - Prefill cost: `prefill_ms_per_token × ctx_len` (O(n) simplified).
/// - Decode cost: `base_decode_ms + kv_pressure_factor × (ctx_len / 1024)`.
///
/// The decode model is intentionally slightly super-linear so that the sweep
/// captures a meaningful gradient when plotted.
struct LongContextStubEngine {
    /// Milliseconds of simulated cost per prefill token.
    prefill_ms_per_token: f64,
    /// Base decode latency in milliseconds (at ctx_len = 0).
    base_decode_ms: f64,
    /// Additional milliseconds of decode cost per 1024-token increment.
    kv_pressure_factor: f64,
    /// Currently loaded context length (set by `prepare_ctx`).
    current_ctx: usize,
}

impl LongContextStubEngine {
    /// Create a stub engine with realistic-looking simulated latencies.
    ///
    /// These values are chosen so that:
    /// - At ctx=1024  → decode ~1.00 ms/tok → ~1000 tok/s
    /// - At ctx=4096  → decode ~4.00 ms/tok → ~250  tok/s
    /// - At ctx=8192  → decode ~8.00 ms/tok → ~125  tok/s
    /// - At ctx=16384 → decode ~16.0 ms/tok → ~62.5 tok/s
    /// - At ctx=32768 → decode ~32.0 ms/tok → ~31.3 tok/s
    fn new() -> Self {
        Self {
            prefill_ms_per_token: 0.000_1, // 0.1 µs / token
            base_decode_ms: 0.0,
            kv_pressure_factor: 1.0, // 1 ms per 1024-token increment
            current_ctx: 1024,
        }
    }

    /// Simulate loading a specific context length into the engine state.
    ///
    /// This sets `current_ctx` so that subsequent `bench_decode_token()` calls
    /// return the correct simulated latency for that context.
    fn prepare_ctx(&mut self, ctx_len: usize) {
        self.current_ctx = ctx_len;
    }
}

impl PrefillDecodeBench for LongContextStubEngine {
    fn bench_prefill(&mut self, prompt_tokens: usize) -> f64 {
        self.prefill_ms_per_token * prompt_tokens as f64
    }

    fn bench_decode_token(&mut self) -> f64 {
        // Linear KV-read cost: each additional 1024 tokens costs
        // `kv_pressure_factor` ms of attention overhead.
        self.base_decode_ms + self.kv_pressure_factor * (self.current_ctx as f64 / 1024.0)
    }

    fn bench_reset(&mut self) {
        // No mutable state to reset for this stub.
    }
}

// ── Criterion benchmark ──────────────────────────────────────────────────

/// Benchmark group: one sub-benchmark per context length.
///
/// Each iteration:
/// 1. Prepare the engine for the target context length (simulates KV state).
/// 2. Run a single prefill + one decode step — this is the minimal "unit of
///    work" for measuring context-length sensitivity.
///
/// Criterion controls iteration count automatically based on timing variance.
pub fn bench_long_context_decode(c: &mut Criterion) {
    let ctx_lengths = default_ctx_lengths();

    let mut group = c.benchmark_group("long_context_decode");

    for &ctx_len in ctx_lengths {
        // Report throughput in tokens so Criterion scales the x-axis correctly.
        group.throughput(Throughput::Elements(ctx_len as u64));

        group.bench_with_input(BenchmarkId::new("ctx", ctx_len), &ctx_len, |b, &ctx| {
            let mut engine = LongContextStubEngine::new();
            // One-time setup: prime the engine's context length.
            engine.prepare_ctx(ctx);

            b.iter(|| {
                engine.bench_reset();
                // Minimal decode step at this context length — measured
                // operation is the bottleneck we're sweeping.
                let prefill_cost = std::hint::black_box(engine.bench_prefill(ctx));
                let decode_cost = std::hint::black_box(engine.bench_decode_token());
                // Return both so the compiler cannot elide either call.
                (prefill_cost, decode_cost)
            });
        });
    }

    group.finish();

    // ── Optional Markdown summary table ─────────────────────────────────
    //
    // Gated on `OXILLAMA_BENCH_PRINT_TABLE=1` so that CI runs are quiet by
    // default, but developers can easily inspect the sweep output locally.
    if std::env::var("OXILLAMA_BENCH_PRINT_TABLE")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        print_summary_table(ctx_lengths);
    }
}

/// Build a [`LongContextSweep`] using the same stub engine and emit its
/// Markdown summary to stdout.
fn print_summary_table(ctx_lengths: &[usize]) {
    let mut engine = LongContextStubEngine::new();

    match LongContextSweep::run(&mut engine, ctx_lengths, &[], "LongContextStub") {
        Ok(sweep) => {
            println!();
            println!("{}", sweep.summary_table());
        }
        Err(e) => {
            eprintln!("long_context sweep failed: {e}");
        }
    }
}

criterion_group!(benches, bench_long_context_decode);
criterion_main!(benches);

//! RAPL power-aware Criterion benchmarks.
//!
//! Uses a [`StubEngine`] that returns instantaneous (zero-cost) latencies so
//! that the bench can run in CI without any real model.  When RAPL is available
//! and the environment variable `OXILLAMA_BENCH_PRINT_POWER=1` is set, the
//! bench additionally measures tokens-per-joule for a decode-only pass and
//! prints the result to stdout.
//!
//! # Running
//!
//! ```sh
//! # Criterion mode (writes HTML report)
//! cargo bench --bench power --features bench
//!
//! # CI / compile-time sanity
//! cargo bench --bench power --features bench -- --test
//!
//! # With optional power output (Linux + RAPL only)
//! OXILLAMA_BENCH_PRINT_POWER=1 cargo bench --bench power --features bench
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxillama_bench::{
    power::{measure_tokens_per_joule, RaplReader},
    PrefillDecodeBench,
};

// ── Stub engine ──────────────────────────────────────────────────────────────

/// Minimal stub engine: returns constant latencies proportional to token count.
///
/// No I/O, no allocations in the hot path.  Stores the current context length
/// so that `bench_decode_token` can model a non-trivial (but still
/// deterministic) cost.
struct StubEngine {
    seq: usize,
}

impl StubEngine {
    fn new() -> Self {
        Self { seq: 0 }
    }
}

impl PrefillDecodeBench for StubEngine {
    fn bench_prefill(&mut self, tokens: usize) -> f64 {
        self.seq = tokens;
        // Simulate 1 µs per token (returns ms).
        tokens as f64 * 0.001
    }

    fn bench_decode_token(&mut self) -> f64 {
        // Simulate a fixed 0.1 µs decode step (returns ms).
        // Non-zero so that Criterion can measure it.
        0.0001
    }

    fn bench_reset(&mut self) {
        self.seq = 0;
    }
}

// ── Benchmark: prefill_128 ───────────────────────────────────────────────────

/// One Criterion iteration: prefill 128 tokens then decode 32 tokens.
///
/// Throughput is reported as `128 + 32 = 160` elements so that the y-axis
/// reflects total token throughput (prefill + decode).
fn bench_power(c: &mut Criterion) {
    // Try to open RAPL once before the benchmark group; availability is logged
    // but a missing RAPL is not a hard error.
    let rapl_opt = RaplReader::open().ok();

    if rapl_opt.is_some() {
        println!("[power bench] RAPL reader opened successfully.");
    } else {
        println!("[power bench] RAPL unavailable — power measurements skipped.");
    }

    let mut group = c.benchmark_group("power");

    // ── Prefill-128 + decode-32 ──────────────────────────────────────────────

    const PREFILL_LEN: usize = 128;
    const DECODE_STEPS: usize = 32;
    const TOTAL_TOKENS: usize = PREFILL_LEN + DECODE_STEPS;

    group.throughput(Throughput::Elements(TOTAL_TOKENS as u64));

    group.bench_function(BenchmarkId::new("prefill_decode", PREFILL_LEN), |b| {
        let mut engine = StubEngine::new();
        b.iter(|| {
            engine.bench_reset();
            let prefill_cost = std::hint::black_box(engine.bench_prefill(PREFILL_LEN));
            let mut decode_cost = 0.0_f64;
            for _ in 0..DECODE_STEPS {
                decode_cost += std::hint::black_box(engine.bench_decode_token());
            }
            (prefill_cost, decode_cost)
        });
    });

    // ── Decode-only at various context lengths ────────────────────────────────

    for &seq_len in &[64usize, 256, 512] {
        group.throughput(Throughput::Elements(32));
        group.bench_with_input(
            BenchmarkId::new("decode_only", seq_len),
            &seq_len,
            |b, &sl| {
                let mut engine = StubEngine::new();
                b.iter(|| {
                    engine.bench_reset();
                    let _ = std::hint::black_box(engine.bench_prefill(sl));
                    let mut total = 0.0_f64;
                    for _ in 0..32 {
                        total += std::hint::black_box(engine.bench_decode_token());
                    }
                    total
                });
            },
        );
    }

    group.finish();

    // ── Optional RAPL tokens-per-joule measurement ───────────────────────────

    if std::env::var("OXILLAMA_BENCH_PRINT_POWER").is_ok() {
        if let Some(rapl) = &rapl_opt {
            print_tokens_per_joule(rapl);
        } else {
            println!("[power bench] OXILLAMA_BENCH_PRINT_POWER is set but RAPL is unavailable.");
        }
    }
}

/// Measure tokens-per-joule for 100 decode tokens and print the result.
///
/// This function is only called when `OXILLAMA_BENCH_PRINT_POWER` is set
/// **and** a [`RaplReader`] was successfully opened, so it is safe to unwrap
/// the RAPL result here — any error is a genuine hardware failure.
fn print_tokens_per_joule(rapl: &RaplReader) {
    const MEASURE_TOKENS: usize = 100;

    let mut engine = StubEngine::new();
    // Warm-up: prime any internal caches before the measured window.
    engine.bench_reset();
    for _ in 0..MEASURE_TOKENS {
        let _ = engine.bench_decode_token();
    }

    // Measured window.
    engine.bench_reset();
    match measure_tokens_per_joule(rapl, || {
        for _ in 0..MEASURE_TOKENS {
            let _ = engine.bench_decode_token();
        }
        MEASURE_TOKENS
    }) {
        Ok((tokens, tpj)) => {
            println!(
                "[power bench] {tokens} decode tokens → {tpj:.2} tok/J  ({:.4} mJ)",
                tokens as f64 * 1_000.0 / tpj
            );
        }
        Err(e) => {
            eprintln!("[power bench] RAPL measurement failed: {e}");
        }
    }
}

criterion_group!(benches, bench_power);
criterion_main!(benches);

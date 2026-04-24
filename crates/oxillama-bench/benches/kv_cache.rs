//! KV-cache scaling and prefill/decode isolation Criterion benchmarks.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxillama_bench::{
    run_prefill_vs_decode_isolation, KvCacheScalingConfig, PrefillDecodeBench,
    KV_CACHE_CONTEXT_SIZES,
};

// ── Stub engine ─────────────────────────────────────────────────────────

/// Minimal stub engine: prefill cost is O(n_tokens) at 0.01 ms/token;
/// decode cost is a fixed 5 ms per token (simulating attention over KV state).
struct StubEngine {
    prefill_ms_per_token: f64,
    decode_ms_per_token: f64,
}

impl PrefillDecodeBench for StubEngine {
    fn bench_prefill(&mut self, prompt_tokens: usize) -> f64 {
        // Simulate O(n²) attention cost increasing with prompt length.
        self.prefill_ms_per_token * prompt_tokens as f64
    }

    fn bench_decode_token(&mut self) -> f64 {
        self.decode_ms_per_token
    }

    fn bench_reset(&mut self) {}
}

// ── KV-cache scaling curve ───────────────────────────────────────────────

pub fn bench_kv_cache_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_scaling");

    let config = KvCacheScalingConfig {
        context_sizes: KV_CACHE_CONTEXT_SIZES.to_vec(),
        n_layers: 32,
        n_heads: 32,
        head_dim: 128,
        measure_iters: 5,
        warmup_iters: 2,
    };

    for &ctx_size in &config.context_sizes {
        group.throughput(Throughput::Elements(ctx_size as u64));
        group.bench_with_input(
            BenchmarkId::new("decode_latency", ctx_size),
            &ctx_size,
            |b, &ctx| {
                let mut engine = StubEngine {
                    prefill_ms_per_token: 0.001,
                    // Simulate linear KV-read cost: 1 µs per 1024-token increment.
                    decode_ms_per_token: 0.001 * (ctx as f64 / 1024.0),
                };
                b.iter(|| {
                    engine.bench_reset();
                    let _ = engine.bench_prefill(ctx);
                    let _ = engine.bench_decode_token();
                });
            },
        );
    }

    group.finish();
}

// ── Prefill vs decode isolation ──────────────────────────────────────────

pub fn bench_prefill_vs_decode_isolation(c: &mut Criterion) {
    const N_TOKENS: usize = 512;

    let mut group = c.benchmark_group("prefill_vs_decode_isolation");
    group.throughput(Throughput::Elements(N_TOKENS as u64));

    group.bench_function("prefill_512", |b| {
        let mut engine = StubEngine {
            prefill_ms_per_token: 0.001,
            decode_ms_per_token: 5.0,
        };
        b.iter(|| {
            engine.bench_reset();
            let _ = engine.bench_prefill(N_TOKENS);
        });
    });

    group.bench_function("decode_512", |b| {
        let mut engine = StubEngine {
            prefill_ms_per_token: 0.001,
            decode_ms_per_token: 5.0,
        };
        b.iter(|| {
            engine.bench_reset();
            for _ in 0..N_TOKENS {
                let _ = engine.bench_decode_token();
            }
        });
    });

    // Convenience: run full isolation helper and report the ratio.
    group.bench_function("isolation_report", |b| {
        let mut engine = StubEngine {
            prefill_ms_per_token: 0.001,
            decode_ms_per_token: 5.0,
        };
        b.iter(|| {
            let _ = run_prefill_vs_decode_isolation(&mut engine, N_TOKENS, 0, 1);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_kv_cache_scaling,
    bench_prefill_vs_decode_isolation
);
criterion_main!(benches);

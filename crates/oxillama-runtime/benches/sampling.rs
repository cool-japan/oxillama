//! Benchmarks for OxiLLaMa sampling algorithms.
//!
//! Exercises each sampling strategy over a vocabulary of 32,000 tokens
//! (LLaMA-3 / Qwen3 vocabulary size) with deterministic pseudo-random logits.
//! No model loading is required — the benchmarks work entirely with raw logit
//! vectors.
//!
//! Run with:
//!   cargo bench -p oxillama-runtime --bench sampling
//!   cargo bench -p oxillama-runtime --bench sampling -- greedy --quick

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use oxillama_runtime::{sample, Sampler, SamplerConfig};

/// Default vocabulary size (LLaMA-3 / Qwen3).
const VOCAB_SIZE: usize = 32_000;

/// Generate reproducible pseudo-random logits without the `rand` crate.
///
/// Uses an LCG to produce values uniformly distributed in approximately
/// [-5.0, +5.0].
fn make_logits(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (h % 1000) as f32 / 100.0 - 5.0
        })
        .collect()
}

/// Build a token history of the given length using the same LCG.
fn make_token_history(len: usize, vocab_size: usize) -> Vec<u32> {
    (0..len)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_499);
            (h % vocab_size as u64) as u32
        })
        .collect()
}

// ─── Greedy (temperature = 0) ─────────────────────────────────────────────────

fn bench_greedy(c: &mut Criterion) {
    let logits = make_logits(VOCAB_SIZE);
    let config = SamplerConfig::greedy();

    c.bench_function("greedy_32k", |b| {
        b.iter_batched(
            || logits.clone(),
            |l| {
                let result = sample(std::hint::black_box(&l), &config, &[]);
                std::hint::black_box(result)
            },
            BatchSize::SmallInput,
        )
    });
}

// ─── Top-K (k = 40, temperature = 0.8) ───────────────────────────────────────

fn bench_top_k(c: &mut Criterion) {
    let logits = make_logits(VOCAB_SIZE);
    let config = SamplerConfig {
        temperature: 0.8,
        top_k: 40,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        seed: Some(42),
        ..SamplerConfig::default()
    };

    c.bench_function("top_k40_32k", |b| {
        b.iter_batched(
            || logits.clone(),
            |l| {
                let result = sample(std::hint::black_box(&l), &config, &[]);
                std::hint::black_box(result)
            },
            BatchSize::SmallInput,
        )
    });
}

// ─── Top-P / Nucleus (p = 0.9, temperature = 0.8) ────────────────────────────

fn bench_top_p(c: &mut Criterion) {
    let logits = make_logits(VOCAB_SIZE);
    let config = SamplerConfig {
        temperature: 0.8,
        top_k: 0,
        top_p: 0.9,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        seed: Some(42),
        ..SamplerConfig::default()
    };

    c.bench_function("top_p0.9_32k", |b| {
        b.iter_batched(
            || logits.clone(),
            |l| {
                let result = sample(std::hint::black_box(&l), &config, &[]);
                std::hint::black_box(result)
            },
            BatchSize::SmallInput,
        )
    });
}

// ─── Min-P (min_p = 0.05, temperature = 0.8) ─────────────────────────────────

fn bench_min_p(c: &mut Criterion) {
    let logits = make_logits(VOCAB_SIZE);
    let config = SamplerConfig {
        temperature: 0.8,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.05,
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        seed: Some(42),
        ..SamplerConfig::default()
    };

    c.bench_function("min_p0.05_32k", |b| {
        b.iter_batched(
            || logits.clone(),
            |l| {
                let result = sample(std::hint::black_box(&l), &config, &[]);
                std::hint::black_box(result)
            },
            BatchSize::SmallInput,
        )
    });
}

// ─── Repetition Penalty (history = 200 tokens, penalty = 1.1) ─────────────────

fn bench_repetition_penalty(c: &mut Criterion) {
    let logits = make_logits(VOCAB_SIZE);
    let history = make_token_history(200, VOCAB_SIZE);
    let config = SamplerConfig {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.0,
        repetition_penalty: 1.1,
        repetition_penalty_window: 200,
        seed: Some(42),
        ..SamplerConfig::default()
    };

    c.bench_function("rep_penalty_200hist_32k", |b| {
        b.iter_batched(
            || (logits.clone(), history.clone()),
            |(l, h)| {
                let result = sample(std::hint::black_box(&l), &config, std::hint::black_box(&h));
                std::hint::black_box(result)
            },
            BatchSize::SmallInput,
        )
    });
}

// ─── Mirostat v2 (tau = 5.0, eta = 0.1) ──────────────────────────────────────

fn bench_mirostat_v2(c: &mut Criterion) {
    let logits = make_logits(VOCAB_SIZE);
    let config = SamplerConfig {
        seed: Some(42),
        ..SamplerConfig::mirostat_v2(5.0, 0.1)
    };

    // Mirostat needs stateful sampling (mu tracks across calls).
    // We create the sampler inside the iter closure so each iteration gets
    // a fresh, independently-seeded sampler — this benchmarks cold-start cost.
    // For sustained throughput measure, use Sampler::new outside the closure.
    c.bench_function("mirostat_v2_32k", |b| {
        b.iter_batched(
            || (logits.clone(), Sampler::new(config.clone())),
            |(l, mut sampler)| {
                let result = sampler.sample(std::hint::black_box(&l), &[]);
                std::hint::black_box(result)
            },
            BatchSize::SmallInput,
        )
    });
}

// ─── Combined (top_k + top_p + min_p + rep_penalty) — realistic config ────────

fn bench_combined(c: &mut Criterion) {
    let logits = make_logits(VOCAB_SIZE);
    let history = make_token_history(64, VOCAB_SIZE);
    let config = SamplerConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.05,
        repetition_penalty: 1.1,
        repetition_penalty_window: 64,
        seed: Some(42),
        ..SamplerConfig::default()
    };

    c.bench_function("combined_default_32k", |b| {
        b.iter_batched(
            || (logits.clone(), history.clone()),
            |(l, h)| {
                let result = sample(std::hint::black_box(&l), &config, std::hint::black_box(&h));
                std::hint::black_box(result)
            },
            BatchSize::SmallInput,
        )
    });
}

// ─── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_greedy,
    bench_top_k,
    bench_top_p,
    bench_min_p,
    bench_repetition_penalty,
    bench_mirostat_v2,
    bench_combined,
);
criterion_main!(benches);

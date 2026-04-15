//! Criterion benchmarks for OxiLLaMa inference.
//!
//! These are lightweight smoke benchmarks that exercise the tokenize → sample
//! pipeline without loading a real GGUF model.  Full end-to-end benchmarks
//! require a model file and are intended to be run manually.
//!
//! Also benchmarks utility computations that don't require model loading.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

/// Simulate a minimal token-processing loop: hash each token ID with a
/// multiplicative mixer to avoid the optimizer collapsing the loop entirely.
fn simulate_token_loop(tokens: &[u32]) -> u32 {
    tokens.iter().fold(0u32, |acc, &t| {
        acc.wrapping_add(t.wrapping_mul(0x9E37_79B9))
    })
}

fn bench_token_loop(c: &mut Criterion) {
    const SEQ_LEN: usize = 64;
    let tokens: Vec<u32> = (0..SEQ_LEN as u32).collect();

    let mut group = c.benchmark_group("inference_smoke");
    group.throughput(Throughput::Elements(SEQ_LEN as u64));

    group.bench_function("token_hash_loop_64", |b| {
        b.iter(|| {
            let result = simulate_token_loop(&tokens);
            std::hint::black_box(result)
        });
    });

    group.finish();
}

/// Simulate sampler logic: argmax over a logit vector.
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn bench_argmax_sampler(c: &mut Criterion) {
    // Typical vocabulary size: 32 000.
    const VOCAB: usize = 32_000;
    // Pseudo-random logits using a simple LCG to avoid a real RNG dependency.
    let logits: Vec<f32> = (0..VOCAB)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            // Map to [-10, 10]
            (x as f32 / u32::MAX as f32) * 20.0 - 10.0
        })
        .collect();

    let mut group = c.benchmark_group("inference_smoke");
    group.throughput(Throughput::Elements(VOCAB as u64));

    group.bench_function("argmax_32k_vocab", |b| {
        b.iter(|| {
            let idx = argmax(&logits);
            std::hint::black_box(idx)
        });
    });

    group.finish();
}

/// Benchmark the cost of percentile computation on a realistic-sized timing array.
fn bench_percentile_computation(c: &mut Criterion) {
    let mut times: Vec<f64> = (0..1000).map(|i| 10.0 + (i as f64 % 5.0)).collect();

    c.bench_function("percentile/1000_elements", |b| {
        b.iter(|| {
            let mut sorted = times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p50_idx = (sorted.len() as f64 * 0.50) as usize;
            let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
            std::hint::black_box((sorted[p50_idx], sorted[p99_idx]));
        });
    });

    times.extend((0..9000).map(|i| 8.0 + (i as f64 % 7.0)));
    c.bench_function("percentile/10000_elements", |b| {
        b.iter(|| {
            let mut sorted = times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p50_idx = (sorted.len() as f64 * 0.50) as usize;
            let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
            std::hint::black_box((sorted[p50_idx], sorted[p99_idx]));
        });
    });
}

/// Benchmark softmax computation at typical vocabulary sizes.
fn bench_softmax(c: &mut Criterion) {
    let mut logits_32k: Vec<f32> = (0..32000).map(|i| (i as f32) * 0.001 - 16.0).collect();
    let mut logits_128k: Vec<f32> = (0..128000).map(|i| (i as f32) * 0.001 - 64.0).collect();

    c.bench_function("softmax/32k_vocab", |b| {
        b.iter(|| {
            let max_val = logits_32k.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in logits_32k.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for v in logits_32k.iter_mut() {
                    *v *= inv;
                }
            }
            std::hint::black_box(logits_32k[0]);
            // Reset for next iteration
            for (i, v) in logits_32k.iter_mut().enumerate() {
                *v = (i as f32) * 0.001 - 16.0;
            }
        });
    });

    c.bench_function("softmax/128k_vocab", |b| {
        b.iter(|| {
            let max_val = logits_128k
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in logits_128k.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for v in logits_128k.iter_mut() {
                    *v *= inv;
                }
            }
            std::hint::black_box(logits_128k[0]);
            for (i, v) in logits_128k.iter_mut().enumerate() {
                *v = (i as f32) * 0.001 - 64.0;
            }
        });
    });
}

criterion_group!(
    benches,
    bench_token_loop,
    bench_argmax_sampler,
    bench_percentile_computation,
    bench_softmax
);
criterion_main!(benches);

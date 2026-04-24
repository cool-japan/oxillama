//! Tokenizer throughput Criterion benchmark.
//!
//! Measures encode + decode rates using the `StubBpeTokenizer` harness
//! (no real model file needed in CI).

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxillama_bench::{
    bench_tokenizer_decode, bench_tokenizer_encode, StubBpeTokenizer, TokenizerBenchConfig,
    TOKENIZER_SAMPLE_TEXT,
};

const TOKEN_COUNT: u64 = 1024;

pub fn bench_tokenizer_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer_throughput");
    group.throughput(Throughput::Elements(TOKEN_COUNT));

    let config = TokenizerBenchConfig {
        repetitions: 1,
        warmup_iters: 5,
        measure_iters: 50,
    };

    // ── Encode ──────────────────────────────────────────────────────────
    group.bench_function("encode_1024_tokens", |b| {
        let mut tokenizer = StubBpeTokenizer;
        b.iter(|| {
            let result = bench_tokenizer_encode(&mut tokenizer, TOKENIZER_SAMPLE_TEXT, &config);
            result.tokens_per_sec
        });
    });

    // ── Decode ──────────────────────────────────────────────────────────
    group.bench_function("decode_1024_tokens", |b| {
        let ids: Vec<u32> = (0..TOKEN_COUNT as u32).collect();
        let mut tokenizer = StubBpeTokenizer;
        b.iter(|| {
            let result = bench_tokenizer_decode(&mut tokenizer, &ids, &config);
            result.tokens_per_sec
        });
    });

    // ── Direct encode throughput (Criterion-timed) ──────────────────────
    group.bench_function("encode_direct", |b| {
        let mut tokenizer = StubBpeTokenizer;
        use oxillama_bench::TokenizerBench;
        b.iter(|| tokenizer.encode(TOKENIZER_SAMPLE_TEXT));
    });

    // ── Direct decode throughput (Criterion-timed) ──────────────────────
    group.bench_function("decode_direct", |b| {
        let ids: Vec<u32> = (0..TOKEN_COUNT as u32).collect();
        let mut tokenizer = StubBpeTokenizer;
        use oxillama_bench::TokenizerBench;
        b.iter(|| tokenizer.decode(&ids));
    });

    group.finish();
}

criterion_group!(benches, bench_tokenizer_throughput);
criterion_main!(benches);

//! Speculative decoding acceptance-rate sweep — Criterion bench target.
//!
//! Sweeps the full `(draft_size, accept_threshold)` grid using a `StubSpecEngine`
//! so that the harness can run quickly in CI (`--test` mode) without a real GGUF.
//!
//! Set `OXILLAMA_BENCH_PRINT_SPEC=1` to also print the full Markdown table after
//! the Criterion run completes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxillama_bench::speculative::{
    default_accept_thresholds, default_draft_sizes, run_acceptance_sweep, StubSpecEngine,
};

fn bench_speculative(c: &mut Criterion) {
    let mut group = c.benchmark_group("speculative");

    // Reduce sample count so the full grid completes in reasonable time in CI.
    group.sample_size(10);

    let engine = StubSpecEngine::default();

    for &draft_size in default_draft_sizes() {
        for &threshold in default_accept_thresholds() {
            group.bench_with_input(
                BenchmarkId::new(format!("draft{}", draft_size), format!("{:.2}", threshold)),
                &(draft_size, threshold),
                |b, &(ds, t)| {
                    b.iter(|| run_acceptance_sweep(&engine, &engine, &[ds], &[t], 10, 1, "stub"));
                },
            );
        }
    }

    group.finish();

    // Optional: print the full 2-D Markdown table when the env-var is set.
    if std::env::var("OXILLAMA_BENCH_PRINT_SPEC").is_ok() {
        let table = run_acceptance_sweep(
            &engine,
            &engine,
            default_draft_sizes(),
            default_accept_thresholds(),
            100,
            3,
            "stub",
        );
        println!("\n## Speculative Decoding Speedup Table\n");
        println!("{}", table.summary_table());
    }
}

criterion_group!(benches, bench_speculative);
criterion_main!(benches);

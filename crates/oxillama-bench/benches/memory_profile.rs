//! Memory profiling Criterion benchmark.
//!
//! Runs the `MemoryProfiler` during a simulated inference loop and reports
//! peak, P50, and P99 RSS measurements via the `MemoryReport`.

use criterion::{criterion_group, criterion_main, Criterion};
use oxillama_bench::MemoryProfiler;

pub fn bench_memory_profile(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_profile");

    group.bench_function("profiler_record_100_samples", |b| {
        b.iter(|| {
            let mut profiler = MemoryProfiler::new();
            // Simulate sampling RSS 100 times during a benchmark run.
            for _ in 0..100 {
                profiler.record();
            }
            let report = profiler.report();
            assert_eq!(report.sample_count, 100);
            report
        });
    });

    group.bench_function("profiler_report_percentiles", |b| {
        // Pre-populate a profiler with 1000 samples.
        let mut profiler = MemoryProfiler::new();
        for _ in 0..1000 {
            profiler.record();
        }
        b.iter(|| {
            let p50 = profiler.percentile_bytes(50.0);
            let p99 = profiler.percentile_bytes(99.0);
            (p50, p99)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_memory_profile);
criterion_main!(benches);

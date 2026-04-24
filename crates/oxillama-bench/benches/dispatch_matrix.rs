//! Criterion benchmark harness for the cross-SIMD dispatch matrix.
//!
//! Run with:
//! ```
//! cargo bench -p oxillama-bench --bench dispatch_matrix
//! ```
//!
//! Feature flags:
//! - `--features simd-avx2` enables the AVX2 path
//! - `--features simd-avx512` enables the AVX-512 path
//! - `--features simd-neon` enables the ARM NEON path

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxillama_bench::dispatch_matrix;

fn bench_dispatch_matrix(c: &mut Criterion) {
    // ── Full matrix at 4096×4096 (one group per quant type) ───────────────
    let quant_types = ["Q4_0", "Q8_0", "Q4_K", "Q8_K"];
    let rows = 256;
    let cols = 256;

    let mut group = c.benchmark_group("dispatch_matrix");

    for &qt in &quant_types {
        for &simd in dispatch_matrix::detect_available_simd_paths().iter() {
            let id = BenchmarkId::new(qt, simd);
            // Approximate element count for throughput reporting
            group.throughput(Throughput::Elements((rows * cols) as u64));
            group.bench_with_input(id, &(qt, simd), |b, &(qt, simd)| {
                b.iter(|| dispatch_matrix::bench_single(qt, simd, rows, cols, 1))
            });
        }
    }

    group.finish();

    // ── scalar path only — thin smoke-test group ───────────────────────────
    let mut smoke = c.benchmark_group("dispatch_matrix_smoke");
    smoke.bench_function("q4_0_scalar_64x256", |b| {
        b.iter(|| dispatch_matrix::bench_single("Q4_0", "scalar", 64, 256, 1))
    });
    smoke.bench_function("q8_0_scalar_64x256", |b| {
        b.iter(|| dispatch_matrix::bench_single("Q8_0", "scalar", 64, 256, 1))
    });
    smoke.finish();
}

criterion_group!(benches, bench_dispatch_matrix);
criterion_main!(benches);

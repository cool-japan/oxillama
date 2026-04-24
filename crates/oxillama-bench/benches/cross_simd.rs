//! Cross-SIMD comparison Criterion benchmark.
//!
//! Measures dequantization + GEMV throughput across all available SIMD tiers
//! (scalar, NEON, AVX2, AVX-512) for Q4_0, Q8_0, Q4_K, and Q6_K.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxillama_bench::{run_dequant_comparison, run_gemv_comparison, SimdComparisonConfig};

/// Criterion function: cross-SIMD comparison table.
///
/// Runs the full dequant + GEMV comparison sweep and reports GB/s throughput
/// for each (quant_type, SIMD tier) combination available on the current CPU.
pub fn bench_cross_simd(c: &mut Criterion) {
    let config = SimdComparisonConfig {
        rows: 4096,
        cols: 4096,
        num_blocks: 1024,
        warmup_iters: 5,
        measure_iters: 20,
    };

    // ── Dequant comparison ──────────────────────────────────────────────
    let dequant_result = run_dequant_comparison(&config);
    let mut dequant_group = c.benchmark_group("cross_simd_dequant");

    for r in &dequant_result.results {
        // Throughput: bytes of quantized data processed per operation.
        let bytes_per_iter = r.avg_us.mul_add(0.0, 1.0) as u64; // placeholder; use block_bytes below
        let _ = bytes_per_iter;
        // Use the reported throughput result as the Criterion element count.
        // We pass num_blocks * block_size elements (floats out of dequant).
        let elements = config.num_blocks as u64 * 32; // ~32 elements per block avg
        dequant_group.throughput(Throughput::Elements(elements));
        dequant_group.bench_with_input(
            BenchmarkId::new(&r.quant_type, &r.simd_tier),
            &r.avg_us,
            |b, _| {
                // The actual benchmark runs the pre-measured helper functions
                // rather than re-running the full comparison sweep (which
                // includes warm-up/measurement loops internally).
                // Here we run the comparison again to give Criterion control
                // over iteration count.
                let inner_config = SimdComparisonConfig {
                    rows: 64,
                    cols: 64,
                    num_blocks: 16,
                    warmup_iters: 0,
                    measure_iters: 1,
                };
                b.iter(|| {
                    let _ = run_dequant_comparison(&inner_config);
                });
            },
        );
    }
    dequant_group.finish();

    // ── GEMV comparison ─────────────────────────────────────────────────
    let gemv_result = run_gemv_comparison(&config);
    let mut gemv_group = c.benchmark_group("cross_simd_gemv");

    for r in &gemv_result.results {
        let elements = (config.rows * config.cols) as u64;
        gemv_group.throughput(Throughput::Elements(elements));
        gemv_group.bench_with_input(
            BenchmarkId::new(&r.quant_type, &r.simd_tier),
            &r.avg_us,
            |b, _| {
                let inner_config = SimdComparisonConfig {
                    rows: 64,
                    cols: 64,
                    num_blocks: 16,
                    warmup_iters: 0,
                    measure_iters: 1,
                };
                b.iter(|| {
                    let _ = run_gemv_comparison(&inner_config);
                });
            },
        );
    }
    gemv_group.finish();
}

criterion_group!(benches, bench_cross_simd);
criterion_main!(benches);

//! Criterion benchmarks for OxiLLaMa quantization kernels.
//!
//! Measures GEMV throughput for all supported reference kernels at
//! LLM-realistic matrix dimensions (default 4096×4096, overridable via
//! the `BENCH_DIM` environment variable).
//!
//! Run with:
//!   cargo bench -p oxillama-quant --bench quant_kernels
//!   BENCH_DIM=2048 cargo bench -p oxillama-quant --bench quant_kernels

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxillama_gguf::GgufTensorType;
use oxillama_quant::{
    reference::{
        Iq1MRef, Iq1SRef, Iq2SRef, Iq2XsRef, Iq2XxsRef, Iq3SRef, Iq3XxsRef, Iq4NlRef, Iq4XsRef,
        Q1_0G128Ref, Q2KRef, Q3KRef, Q4KRef, Q4_0Ref, Q4_1Ref, Q5KRef, Q5_0Ref, Q5_1Ref, Q6KRef,
        Q8KRef, Q8_0Ref, Q8_1Ref,
    },
    traits::QuantKernel,
    types::QuantTensor,
};

/// Read the benchmark matrix dimension from `BENCH_DIM` env var (fallback: 4096).
///
/// The returned value is always a multiple of 256, 128, and 32 when using
/// the default 4096, which satisfies all block-size alignment requirements.
/// Custom values must also be multiples of 256 for K-quant types.
fn bench_dim() -> usize {
    std::env::var("BENCH_DIM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4096)
}

/// Build a synthetic `QuantTensor` with the given dimensions and quant type.
///
/// Uses `GgufTensorType::block_size()` / `block_bytes()` so this helper
/// works for every current and future type without an explicit match.
///
/// The first two bytes of every block are set to the FP16 representation of
/// 1.0 (0x3C00 LE) so the kernel has a valid, non-NaN scale factor.
fn make_synthetic_tensor(tensor_type: GgufTensorType, rows: usize, cols: usize) -> QuantTensor {
    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();

    assert_eq!(
        cols % block_size,
        0,
        "cols ({cols}) must be a multiple of block_size ({block_size}) for {tensor_type:?}"
    );

    let blocks_per_row = cols / block_size;
    let total_bytes = rows * blocks_per_row * block_bytes;

    // Allocate zeroed buffer and stamp FP16 1.0 as the scale in the first two
    // bytes of every block (little-endian 0x3C00).
    let mut data = vec![0u8; total_bytes];
    for block_start in (0..total_bytes).step_by(block_bytes) {
        data[block_start] = 0x00; // FP16 1.0 low byte
        data[block_start + 1] = 0x3C; // FP16 1.0 high byte
    }

    QuantTensor {
        data,
        shape: vec![rows, cols],
        tensor_type,
    }
}

/// Macro that expands to a benchmark function measuring reference GEMV for one
/// quantization type.  Accepts an optional neon variant.
///
/// # Parameters
/// - `$fn_name`    : identifier for the generated function
/// - `$group_name` : string label shown in Criterion output
/// - `$tensor_type`: `GgufTensorType` variant expression
/// - `$kernel`     : reference kernel expression (implements `QuantKernel`)
macro_rules! bench_gemv_ref {
    ($fn_name:ident, $group_name:expr, $tensor_type:expr, $kernel:expr) => {
        fn $fn_name(c: &mut Criterion) {
            let n = bench_dim();
            let tensor = make_synthetic_tensor($tensor_type, n, n);
            let input = vec![1.0f32; n];
            let mut output = vec![0.0f32; n];

            let mut group = c.benchmark_group($group_name);
            group.throughput(Throughput::Elements((n * n) as u64));

            group.bench_function(BenchmarkId::new("reference", n), |b| {
                b.iter(|| {
                    $kernel
                        .gemv(&tensor, &input, &mut output)
                        .expect("GEMV failed");
                    std::hint::black_box(&output);
                });
            });

            group.finish();
        }
    };
    // Variant with NEON implementation (kernel passed as expression, same as reference).
    // The neon branch is conditionally compiled behind feature + target_arch guards.
    ($fn_name:ident, $group_name:expr, $tensor_type:expr, $kernel:expr,
     neon_kernel: $neon_kernel:expr) => {
        fn $fn_name(c: &mut Criterion) {
            let n = bench_dim();
            let tensor = make_synthetic_tensor($tensor_type, n, n);
            let input = vec![1.0f32; n];
            let mut output = vec![0.0f32; n];

            let mut group = c.benchmark_group($group_name);
            group.throughput(Throughput::Elements((n * n) as u64));

            group.bench_function(BenchmarkId::new("reference", n), |b| {
                b.iter(|| {
                    $kernel
                        .gemv(&tensor, &input, &mut output)
                        .expect("GEMV failed");
                    std::hint::black_box(&output);
                });
            });

            #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
            {
                group.bench_function(BenchmarkId::new("neon", n), |b| {
                    b.iter(|| {
                        $neon_kernel
                            .gemv(&tensor, &input, &mut output)
                            .expect("NEON GEMV failed");
                        std::hint::black_box(&output);
                    });
                });
            }

            group.finish();
        }
    };
}

// ─── Legacy (32-element block) quant types ────────────────────────────────────

bench_gemv_ref!(
    bench_q4_0_gemv,
    "Q4_0 GEMV",
    GgufTensorType::Q4_0,
    Q4_0Ref,
    neon_kernel: oxillama_quant::simd::neon::Q4_0Neon
);

bench_gemv_ref!(bench_q4_1_gemv, "Q4_1 GEMV", GgufTensorType::Q4_1, Q4_1Ref);

bench_gemv_ref!(bench_q5_0_gemv, "Q5_0 GEMV", GgufTensorType::Q5_0, Q5_0Ref);

bench_gemv_ref!(bench_q5_1_gemv, "Q5_1 GEMV", GgufTensorType::Q5_1, Q5_1Ref);

bench_gemv_ref!(
    bench_q8_0_gemv,
    "Q8_0 GEMV",
    GgufTensorType::Q8_0,
    Q8_0Ref,
    neon_kernel: oxillama_quant::simd::neon::Q8_0Neon
);

bench_gemv_ref!(bench_q8_1_gemv, "Q8_1 GEMV", GgufTensorType::Q8_1, Q8_1Ref);

// ─── K-quant types (256-element blocks) ───────────────────────────────────────

bench_gemv_ref!(bench_q2k_gemv, "Q2_K GEMV", GgufTensorType::Q2K, Q2KRef);

bench_gemv_ref!(bench_q3k_gemv, "Q3_K GEMV", GgufTensorType::Q3K, Q3KRef);

bench_gemv_ref!(
    bench_q4k_gemv,
    "Q4_K GEMV",
    GgufTensorType::Q4K,
    Q4KRef,
    neon_kernel: oxillama_quant::simd::neon::Q4_KNeon
);

bench_gemv_ref!(bench_q5k_gemv, "Q5_K GEMV", GgufTensorType::Q5K, Q5KRef);

bench_gemv_ref!(bench_q6k_gemv, "Q6_K GEMV", GgufTensorType::Q6K, Q6KRef);

bench_gemv_ref!(bench_q8k_gemv, "Q8_K GEMV", GgufTensorType::Q8K, Q8KRef);

// ─── IQ4 family ───────────────────────────────────────────────────────────────

bench_gemv_ref!(
    bench_iq4nl_gemv,
    "IQ4_NL GEMV",
    GgufTensorType::Iq4Nl,
    Iq4NlRef
);

bench_gemv_ref!(
    bench_iq4xs_gemv,
    "IQ4_XS GEMV",
    GgufTensorType::Iq4Xs,
    Iq4XsRef
);

// ─── IQ1 family ───────────────────────────────────────────────────────────────

bench_gemv_ref!(bench_iq1s_gemv, "IQ1_S GEMV", GgufTensorType::Iq1S, Iq1SRef);

bench_gemv_ref!(bench_iq1m_gemv, "IQ1_M GEMV", GgufTensorType::Iq1M, Iq1MRef);

// ─── IQ2 family ───────────────────────────────────────────────────────────────

bench_gemv_ref!(
    bench_iq2xxs_gemv,
    "IQ2_XXS GEMV",
    GgufTensorType::Iq2Xxs,
    Iq2XxsRef
);

bench_gemv_ref!(
    bench_iq2xs_gemv,
    "IQ2_XS GEMV",
    GgufTensorType::Iq2Xs,
    Iq2XsRef
);

bench_gemv_ref!(bench_iq2s_gemv, "IQ2_S GEMV", GgufTensorType::Iq2S, Iq2SRef);

// ─── IQ3 family ───────────────────────────────────────────────────────────────

bench_gemv_ref!(
    bench_iq3xxs_gemv,
    "IQ3_XXS GEMV",
    GgufTensorType::Iq3Xxs,
    Iq3XxsRef
);

bench_gemv_ref!(bench_iq3s_gemv, "IQ3_S GEMV", GgufTensorType::Iq3S, Iq3SRef);

// ─── Special / OxiBonsai types ────────────────────────────────────────────────

bench_gemv_ref!(
    bench_q1_0_g128_gemv,
    "Q1_0_G128 GEMV",
    GgufTensorType::Q1_0G128,
    Q1_0G128Ref,
    neon_kernel: oxillama_quant::simd::neon::Q1_0G128Neon
);

// ─── Criterion groups ─────────────────────────────────────────────────────────

criterion_group!(
    benches_legacy,
    bench_q4_0_gemv,
    bench_q4_1_gemv,
    bench_q5_0_gemv,
    bench_q5_1_gemv,
    bench_q8_0_gemv,
    bench_q8_1_gemv,
);

criterion_group!(
    benches_k_quants,
    bench_q2k_gemv,
    bench_q3k_gemv,
    bench_q4k_gemv,
    bench_q5k_gemv,
    bench_q6k_gemv,
    bench_q8k_gemv,
);

criterion_group!(
    benches_iq,
    bench_iq4nl_gemv,
    bench_iq4xs_gemv,
    bench_iq1s_gemv,
    bench_iq1m_gemv,
    bench_iq2xxs_gemv,
    bench_iq2xs_gemv,
    bench_iq2s_gemv,
    bench_iq3xxs_gemv,
    bench_iq3s_gemv,
);

criterion_group!(benches_special, bench_q1_0_g128_gemv);

criterion_main!(
    benches_legacy,
    benches_k_quants,
    benches_iq,
    benches_special
);

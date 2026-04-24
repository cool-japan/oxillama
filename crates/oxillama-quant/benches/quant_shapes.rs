//! Shape-parametric quantisation benchmarks.
//!
//! Sweeps (quant_type, seq_len, hidden_size) to expose the decode vs prefill
//! throughput gap and let regressions get caught by shape, not just by format.
//!
//! Seq=1 models single-token decode (GEMV). Seq=64 and Seq=512 model prefill
//! batches (batched GEMV / GEMM-like).
//!
//! Each bench group name encodes: `shapes/<type>/seq<seq>`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxillama_gguf::GgufTensorType;
use oxillama_quant::{dispatch::KernelDispatcher, types::QuantTensor};

const SEQ_LENS: &[usize] = &[1, 64, 512];
const HIDDEN_SIZES: &[usize] = &[2048, 4096];

/// Format list: restrict to formats with scalar reference paths.
const BENCH_FORMATS: &[(GgufTensorType, &str)] = &[
    (GgufTensorType::Q4_0, "Q4_0"),
    (GgufTensorType::Q4K, "Q4_K"),
    (GgufTensorType::Q5K, "Q5_K"),
    (GgufTensorType::Q6K, "Q6_K"),
    (GgufTensorType::Q8_0, "Q8_0"),
    (GgufTensorType::Q8K, "Q8_K"),
];

/// Build a synthetic `QuantTensor` with valid FP16 scale bytes in every block.
fn make_tensor(tensor_type: GgufTensorType, rows: usize, cols: usize) -> QuantTensor {
    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();

    let blocks_per_row = cols.div_ceil(block_size);
    let total_bytes = rows * blocks_per_row * block_bytes;

    // Stamp FP16 1.0 (0x3C00 LE) as the scale in the first two bytes of every block.
    let mut data = vec![0u8; total_bytes];
    for block_start in (0..total_bytes).step_by(block_bytes) {
        if block_start + 1 < total_bytes {
            data[block_start] = 0x00;
            data[block_start + 1] = 0x3C;
        }
    }

    QuantTensor::new(data, vec![rows, cols], tensor_type)
}

fn bench_shapes(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::new();

    for (fmt, fmt_name) in BENCH_FORMATS {
        let block_size = fmt.block_size();

        let mut group = c.benchmark_group(format!("shapes/{fmt_name}"));

        for &seq_len in SEQ_LENS {
            for &hidden in HIDDEN_SIZES {
                // Round hidden up to block boundary.
                let n_blocks = hidden.div_ceil(block_size);
                let n_elem = n_blocks * block_size;

                let tensor = make_tensor(*fmt, seq_len, n_elem);
                let input = vec![1.0f32; n_elem];

                group.throughput(Throughput::Elements((seq_len * n_elem) as u64));
                group.bench_with_input(
                    BenchmarkId::new(format!("seq{seq_len}"), hidden),
                    &hidden,
                    |b, _| {
                        b.iter(|| {
                            let kernel = dispatcher
                                .get_kernel(*fmt)
                                .expect("kernel must exist for bench format");
                            let mut out = vec![0.0f32; seq_len];
                            kernel.gemv(&tensor, &input, &mut out).expect("GEMV failed");
                            std::hint::black_box(&out);
                        });
                    },
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches, bench_shapes);
criterion_main!(benches);

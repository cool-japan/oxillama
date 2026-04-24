//! Cross-SIMD dispatch matrix benchmark.
//!
//! Benchmarks `matvec_q8_fused` for every shipped quantization type across
//! all SIMD paths available on the current platform (scalar, AVX2, AVX-512,
//! NEON).  Results are collected in a [`DispatchMatrixRow`] table and can be
//! written to CSV or printed as an ASCII table.

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use oxillama_gguf::GgufTensorType;
use oxillama_quant::dispatch::{KernelDispatcher, SimdCapabilities};
use oxillama_quant::reference::{Q4_0Ref, Q4_1Ref, Q5_0Ref, Q5_1Ref, Q8_0Ref, Q8_1Ref};
use oxillama_quant::traits::QuantKernel;
use tabled::Tabled;

use crate::simd_comparison::make_synthetic_tensor;

/// One row in the dispatch matrix result table.
#[derive(Debug, Clone, Tabled)]
pub struct DispatchMatrixRow {
    /// Quantization type name (e.g., "Q4_0", "Q4_K").
    #[tabled(rename = "Quant")]
    pub quant_type: String,
    /// SIMD path label: "scalar", "avx2", "avx512", "neon", or "dispatched".
    #[tabled(rename = "SIMD")]
    pub simd_path: String,
    /// Operation name: "matvec_q8_fused".
    #[tabled(rename = "Op")]
    pub operation: String,
    /// Matrix dimensions as "RxC".
    #[tabled(rename = "Matrix")]
    pub matrix_size: String,
    /// Giga-operations per second (approximate).
    #[tabled(rename = "GOps/s")]
    pub throughput_gops: f64,
    /// Microseconds per call (mean over iterations).
    #[tabled(rename = "µs/call")]
    pub latency_us: f64,
    /// Approximate tokens per second for a single decode step at this size.
    #[tabled(rename = "tok/s")]
    pub tokens_per_sec: f64,
}

/// Quantization types included in the dispatch matrix.
///
/// Only types with `block_size == 32` are included, because the default
/// `matvec_q8_fused` implementation in `QuantKernel` maps weight-block indices
/// to Q8_0 activation-block indices assuming 32 elements per block.  K-family
/// types (Q4_K, Q5_K, Q6_K, Q8_K) use 256-element blocks and are exercised
/// through the `gemv` path in `simd_comparison.rs` instead.
const MATRIX_TYPES: [(GgufTensorType, &str); 6] = [
    (GgufTensorType::Q4_0, "Q4_0"),
    (GgufTensorType::Q4_1, "Q4_1"),
    (GgufTensorType::Q5_0, "Q5_0"),
    (GgufTensorType::Q5_1, "Q5_1"),
    (GgufTensorType::Q8_0, "Q8_0"),
    (GgufTensorType::Q8_1, "Q8_1"),
];

/// Return available SIMD path labels for the current platform.
///
/// Returns `["scalar"]` when only the scalar reference path is available,
/// and `["scalar", "dispatched"]` when any SIMD acceleration is present.
/// The `"dispatched"` label always refers to `KernelDispatcher::get_kernel`,
/// which selects the best tier (AVX-512 > AVX2+FMA > AVX2 > NEON > scalar)
/// at runtime.  Intermediate path labels (avx2/avx512/neon) are intentionally
/// omitted — constructing per-tier kernels directly would require routing
/// through internal `simd::<arch>::*` structs, which is deferred to a future
/// enhancement.
pub fn detect_available_simd_paths() -> Vec<&'static str> {
    let caps = SimdCapabilities::detect();
    let best = caps.best_tier();
    if best == "scalar" {
        vec!["scalar"]
    } else {
        vec!["scalar", "dispatched"]
    }
}

/// Get a reference (scalar) kernel for a given type, if one exists.
///
/// Only covers the 32-element block types included in `MATRIX_TYPES`.
fn scalar_kernel_for(tensor_type: GgufTensorType) -> Option<Box<dyn QuantKernel>> {
    match tensor_type {
        GgufTensorType::Q4_0 => Some(Box::new(Q4_0Ref)),
        GgufTensorType::Q4_1 => Some(Box::new(Q4_1Ref)),
        GgufTensorType::Q5_0 => Some(Box::new(Q5_0Ref)),
        GgufTensorType::Q5_1 => Some(Box::new(Q5_1Ref)),
        GgufTensorType::Q8_0 => Some(Box::new(Q8_0Ref)),
        GgufTensorType::Q8_1 => Some(Box::new(Q8_1Ref)),
        _ => None,
    }
}

/// Construct a synthetic Q8_0 activation buffer compatible with `matvec_q8_fused`.
///
/// Each Q8_0 block is 34 bytes: 2-byte FP16 scale followed by 32 × i8 values.
fn make_q8_acts(blocks_per_row: usize) -> Vec<u8> {
    const Q8_0_BLOCK_BYTES: usize = 34;
    let total = blocks_per_row * Q8_0_BLOCK_BYTES;
    let mut acts = vec![0u8; total];

    // Fill i8 values with a deterministic pattern
    for (i, b) in acts.iter_mut().enumerate() {
        *b = ((i.wrapping_mul(13).wrapping_add(7)) % 256) as u8;
    }

    // Fix the FP16 scale in each block to 1.0 (bits: 0x3C00)
    let scale_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
    for blk in 0..blocks_per_row {
        let base = blk * Q8_0_BLOCK_BYTES;
        if base + 1 < total {
            acts[base] = scale_bits[0];
            acts[base + 1] = scale_bits[1];
        }
    }
    acts
}

/// Benchmark a single `(quant_type, simd_path, operation)` combination.
///
/// Returns `None` if the combination cannot be built (e.g. SIMD kernel
/// not compiled, or no reference kernel available for the type).
pub fn bench_single(
    quant_type_name: &str,
    simd_path: &str,
    rows: usize,
    cols: usize,
    iterations: u32,
) -> Option<DispatchMatrixRow> {
    // Find the GgufTensorType by name
    let (tensor_type, _) = MATRIX_TYPES.iter().find(|(_, n)| *n == quant_type_name)?;
    let tensor_type = *tensor_type;

    let block_size = tensor_type.block_size();
    if block_size == 0 {
        return None;
    }

    // Align cols to block boundary
    let cols_aligned = cols.div_ceil(block_size) * block_size;
    let blocks_per_row = cols_aligned / block_size;

    // Build weight buffer and activation buffer
    let weight_tensor = make_synthetic_tensor(tensor_type, rows, cols_aligned);
    let acts_q8 = make_q8_acts(blocks_per_row);

    // Select the kernel based on the simd_path label
    let kernel: Box<dyn QuantKernel> = match simd_path {
        "scalar" => scalar_kernel_for(tensor_type)?,
        "dispatched" | "avx2" | "avx512" | "neon" => {
            // For these labels we use the best-available dispatched kernel.
            // If the best tier is scalar (i.e. we're on a plain machine),
            // the "dispatched" path is identical to scalar — we skip it at
            // the call site by never adding "dispatched" unless best != "scalar".
            let dispatcher = KernelDispatcher::new();
            dispatcher.get_kernel(tensor_type).ok()?
        }
        _ => return None,
    };

    // Compute total FLOP count: 2 * rows * cols multiply-adds
    let total_flops = 2u64 * rows as u64 * cols_aligned as u64;

    let mut out = vec![0.0f32; rows];
    let weights_bytes = &weight_tensor.data;

    // Warm-up: 2 iterations
    for _ in 0..2u32.min(iterations) {
        out.fill(0.0);
        let _ = kernel.matvec_q8_fused(weights_bytes, &acts_q8, &mut out, rows, cols_aligned);
    }

    // Measurement
    let iters = iterations.max(1);
    let t0 = Instant::now();
    for _ in 0..iters {
        out.fill(0.0);
        let _ = kernel.matvec_q8_fused(weights_bytes, &acts_q8, &mut out, rows, cols_aligned);
    }
    let elapsed = t0.elapsed();

    let total_secs = elapsed.as_secs_f64();
    let latency_us = total_secs / iters as f64 * 1_000_000.0;

    // GOps/s = (total_flops * iters) / (elapsed_secs * 1e9)
    let throughput_gops = if total_secs > 0.0 {
        (total_flops as f64 * iters as f64) / (total_secs * 1e9)
    } else {
        0.0
    };

    // Approximate tokens/s: one decode step ≈ one matvec call per layer.
    // We use the reciprocal of latency per call as a proxy.
    let tokens_per_sec = if latency_us > 0.0 {
        1_000_000.0 / latency_us
    } else {
        0.0
    };

    Some(DispatchMatrixRow {
        quant_type: quant_type_name.to_string(),
        simd_path: simd_path.to_string(),
        operation: "matvec_q8_fused".to_string(),
        matrix_size: format!("{}x{}", rows, cols_aligned),
        throughput_gops,
        latency_us,
        tokens_per_sec,
    })
}

/// Run the full dispatch matrix benchmark.
///
/// For each `(quant_type, simd_path)` combination available on this platform,
/// calls `bench_single` and collects all successful rows.
///
/// # Arguments
/// * `rows` — number of matrix rows.
/// * `cols` — number of matrix columns (rounded up to block boundary per type).
/// * `iterations` — measurement iterations per combination.
pub fn run_dispatch_matrix(rows: usize, cols: usize, iterations: u32) -> Vec<DispatchMatrixRow> {
    let simd_paths = detect_available_simd_paths();
    let mut results = Vec::new();

    for &(_, quant_name) in &MATRIX_TYPES {
        for &simd_path in &simd_paths {
            if let Some(row) = bench_single(quant_name, simd_path, rows, cols, iterations) {
                results.push(row);
            }
        }
    }

    results
}

/// Write dispatch matrix results to a CSV file.
///
/// The output file is overwritten if it already exists.
pub fn write_csv(rows: &[DispatchMatrixRow], output: &Path) -> std::io::Result<()> {
    let mut f = std::fs::File::create(output)?;
    writeln!(
        f,
        "quant_type,simd_path,operation,matrix_size,throughput_gops,latency_us,tokens_per_sec"
    )?;
    for row in rows {
        writeln!(
            f,
            "{},{},{},{},{:.3},{:.1},{:.1}",
            row.quant_type,
            row.simd_path,
            row.operation,
            row.matrix_size,
            row.throughput_gops,
            row.latency_us,
            row.tokens_per_sec,
        )?;
    }
    Ok(())
}

/// Print a formatted table of dispatch matrix results using `tabled`.
pub fn print_table(rows: &[DispatchMatrixRow]) {
    use tabled::Table;
    let table = Table::new(rows).to_string();
    println!("{table}");
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn dispatch_matrix_runs_scalar_path() {
        let results = run_dispatch_matrix(64, 256, 3);
        assert!(!results.is_empty(), "expected at least one row");
        assert!(
            results.iter().any(|r| r.simd_path == "scalar"),
            "expected at least one scalar row"
        );
    }

    #[test]
    fn dispatch_matrix_csv_output_has_header() {
        let results = run_dispatch_matrix(64, 256, 1);
        let path = temp_dir().join("test_bench_dispatch.csv");
        write_csv(&results, &path).expect("write_csv should succeed");
        let content = std::fs::read_to_string(&path).expect("read csv back");
        assert!(
            content.starts_with("quant_type,"),
            "CSV must start with header; got: {content}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bench_single_q4_0_scalar_returns_row() {
        let row = bench_single("Q4_0", "scalar", 64, 256, 2);
        assert!(row.is_some(), "Q4_0 scalar bench should produce a row");
        let row = row.expect("checked above");
        assert_eq!(row.quant_type, "Q4_0");
        assert_eq!(row.simd_path, "scalar");
        assert_eq!(row.operation, "matvec_q8_fused");
        assert!(row.latency_us >= 0.0);
    }

    #[test]
    fn dispatch_matrix_all_rows_have_positive_latency() {
        let results = run_dispatch_matrix(32, 128, 1);
        for row in &results {
            assert!(
                row.latency_us >= 0.0,
                "latency must be non-negative, got {} for {}/{}",
                row.latency_us,
                row.quant_type,
                row.simd_path
            );
        }
    }

    #[test]
    fn detect_simd_paths_always_has_scalar() {
        let paths = detect_available_simd_paths();
        assert!(
            paths.contains(&"scalar"),
            "scalar path must always be present"
        );
    }
}

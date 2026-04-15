//! Cross-SIMD comparison benchmarks.
//!
//! Measures dequantization and GEMV performance across all available SIMD
//! tiers (scalar reference, NEON, AVX2, AVX-512) for side-by-side comparison.

use std::fmt::Write as _;
use std::time::Instant;

use oxillama_gguf::GgufTensorType;
use oxillama_quant::dispatch::{KernelDispatcher, SimdCapabilities};
use oxillama_quant::reference::{Q4KRef, Q4_0Ref, Q6KRef, Q8_0Ref};
use oxillama_quant::traits::QuantKernel;
use oxillama_quant::types::QuantTensor;

/// Quantization types benchmarked in cross-SIMD comparisons.
const BENCH_TYPES: [(GgufTensorType, &str); 4] = [
    (GgufTensorType::Q4_0, "Q4_0"),
    (GgufTensorType::Q8_0, "Q8_0"),
    (GgufTensorType::Q4K, "Q4_K"),
    (GgufTensorType::Q6K, "Q6_K"),
];

/// Configuration for cross-SIMD comparison benchmarks.
#[derive(Debug, Clone)]
pub struct SimdComparisonConfig {
    /// Number of matrix rows for GEMV benchmarks.
    pub rows: usize,
    /// Number of matrix columns for GEMV benchmarks.
    pub cols: usize,
    /// Number of blocks for dequant benchmarks.
    pub num_blocks: usize,
    /// Number of warm-up iterations.
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
}

impl Default for SimdComparisonConfig {
    fn default() -> Self {
        Self {
            rows: 4096,
            cols: 4096,
            num_blocks: 1024,
            warmup_iters: 10,
            measure_iters: 100,
        }
    }
}

/// Result of a single kernel benchmark.
#[derive(Debug, Clone)]
pub struct KernelBenchResult {
    /// Quantization type name (e.g., "Q4_0").
    pub quant_type: String,
    /// SIMD tier name (e.g., "NEON", "AVX2", "scalar").
    pub simd_tier: String,
    /// Operation type ("dequant" or "gemv").
    pub operation: String,
    /// Average time per iteration in microseconds.
    pub avg_us: f64,
    /// Minimum time in microseconds.
    pub min_us: f64,
    /// Maximum time in microseconds.
    pub max_us: f64,
    /// Throughput in GB/s (bytes processed / time).
    pub throughput_gbs: f64,
}

/// Complete comparison result across all SIMD tiers.
#[derive(Debug)]
pub struct SimdComparisonResult {
    /// Detected CPU capabilities.
    pub capabilities: SimdCapabilities,
    /// Individual benchmark results.
    pub results: Vec<KernelBenchResult>,
}

// ── Synthetic data helpers ───────────────────────────────────────────────

/// Scale byte offset within a block for each benchmarked type.
///
/// Returns `(offset, len)` where `len` is always 2 (FP16).
fn scale_offset_for_type(tensor_type: GgufTensorType) -> (usize, usize) {
    match tensor_type {
        // Q4_0/Q8_0: scale is the first 2 bytes of the block.
        GgufTensorType::Q4_0 | GgufTensorType::Q8_0 => (0, 2),
        // Q4_K: first 2 bytes = d (scale), bytes 2..4 = dmin.
        GgufTensorType::Q4K => (0, 2),
        // Q6_K: scale (d) is the *last* 2 bytes of the 210-byte block.
        GgufTensorType::Q6K => (208, 2),
        _ => (0, 2),
    }
}

/// Create synthetic quantized block data for benchmarking.
fn make_synthetic_blocks(tensor_type: GgufTensorType, num_blocks: usize) -> Vec<u8> {
    let block_bytes = tensor_type.block_bytes();
    let total = block_bytes * num_blocks;
    let mut data = vec![0u8; total];

    // Fill with deterministic pseudo-random pattern
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((i.wrapping_mul(17).wrapping_add(31)) % 256) as u8;
    }

    // Fix scale bytes to valid f16 values (avoid NaN/Inf)
    let scale = half::f16::from_f32(0.5);
    let bits = scale.to_bits().to_le_bytes();
    let (offset, _len) = scale_offset_for_type(tensor_type);

    for blk in 0..num_blocks {
        let base = blk * block_bytes;
        if base + offset + 1 < total {
            data[base + offset] = bits[0];
            data[base + offset + 1] = bits[1];
        }
    }

    data
}

/// Create a synthetic `QuantTensor` suitable for GEMV benchmarking.
fn make_synthetic_tensor(tensor_type: GgufTensorType, rows: usize, cols: usize) -> QuantTensor {
    let block_size = tensor_type.block_size();
    // Round cols up to a multiple of block_size
    let cols_aligned = if block_size > 0 {
        cols.div_ceil(block_size) * block_size
    } else {
        cols
    };
    let blocks_per_row = cols_aligned.checked_div(block_size).unwrap_or(cols_aligned);
    let total_blocks = rows * blocks_per_row;
    let data = make_synthetic_blocks(tensor_type, total_blocks);
    QuantTensor::new(data, vec![rows, cols_aligned], tensor_type)
}

/// Create a deterministic pseudo-random f32 vector.
fn make_pseudo_random_vec(len: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; len];
    for (i, val) in v.iter_mut().enumerate() {
        // Simple LCG-like pattern producing values in [-1, 1]
        let bits = ((i.wrapping_mul(2654435761).wrapping_add(1013904223)) & 0xFFFF) as f32;
        *val = bits / 32768.0 - 1.0;
    }
    v
}

// ── Benchmark helpers ────────────────────────────────────────────────────

/// Get the reference (scalar) kernel for a given tensor type.
fn reference_kernel_for(tensor_type: GgufTensorType) -> Option<Box<dyn QuantKernel>> {
    match tensor_type {
        GgufTensorType::Q4_0 => Some(Box::new(Q4_0Ref)),
        GgufTensorType::Q8_0 => Some(Box::new(Q8_0Ref)),
        GgufTensorType::Q4K => Some(Box::new(Q4KRef)),
        GgufTensorType::Q6K => Some(Box::new(Q6KRef)),
        _ => None,
    }
}

/// Timing statistics from a benchmark run.
struct TimingStats {
    avg_us: f64,
    min_us: f64,
    max_us: f64,
}

/// Run a closure for warmup + measurement and collect timing stats.
fn bench_timing<F: FnMut()>(warmup_iters: usize, measure_iters: usize, mut f: F) -> TimingStats {
    // Warm-up
    for _ in 0..warmup_iters {
        f();
    }

    // Measurement
    let mut times_us = Vec::with_capacity(measure_iters);
    for _ in 0..measure_iters {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
    }

    let sum: f64 = times_us.iter().sum();
    let avg_us = if measure_iters > 0 {
        sum / measure_iters as f64
    } else {
        0.0
    };
    let min_us = times_us.iter().copied().fold(f64::INFINITY, f64::min);
    let max_us = times_us.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Guard against empty measurement
    let min_us = if min_us.is_infinite() { 0.0 } else { min_us };
    let max_us = if max_us.is_infinite() { 0.0 } else { max_us };

    TimingStats {
        avg_us,
        min_us,
        max_us,
    }
}

// ── Public API ───────────────────────────────────────────────────────────

/// Benchmark `dequant_block` across SIMD tiers for all supported quant types.
///
/// For each type, this benchmarks both the reference (scalar) kernel and the
/// best available dispatched kernel (which may be NEON, AVX2, or AVX-512
/// depending on the platform and enabled features).
pub fn run_dequant_comparison(config: &SimdComparisonConfig) -> SimdComparisonResult {
    let capabilities = SimdCapabilities::detect();
    let dispatcher = KernelDispatcher::new();
    let best_tier = capabilities.best_tier().to_string();

    let mut results = Vec::new();

    for &(tensor_type, name) in &BENCH_TYPES {
        let block_bytes = tensor_type.block_bytes();
        let block_size = tensor_type.block_size();
        let data = make_synthetic_blocks(tensor_type, config.num_blocks);
        let total_bytes = block_bytes * config.num_blocks;

        // ── Scalar reference ─────────────────────────────────────────
        if let Some(ref_kernel) = reference_kernel_for(tensor_type) {
            let mut output = vec![0.0f32; block_size];
            let stats = bench_timing(config.warmup_iters, config.measure_iters, || {
                for blk in 0..config.num_blocks {
                    let start_byte = blk * block_bytes;
                    let end_byte = start_byte + block_bytes;
                    if end_byte <= data.len() {
                        let _ = ref_kernel.dequant_block(&data[start_byte..end_byte], &mut output);
                    }
                }
            });

            let throughput_gbs = if stats.avg_us > 0.0 {
                total_bytes as f64 / (stats.avg_us * 1e-6) / 1e9
            } else {
                0.0
            };

            results.push(KernelBenchResult {
                quant_type: name.to_string(),
                simd_tier: "scalar".to_string(),
                operation: "dequant".to_string(),
                avg_us: stats.avg_us,
                min_us: stats.min_us,
                max_us: stats.max_us,
                throughput_gbs,
            });
        }

        // ── Dispatched (best available) ──────────────────────────────
        if let Ok(dispatched_kernel) = dispatcher.get_kernel(tensor_type) {
            // Only add dispatched if it differs from scalar
            let dispatched_tier = if best_tier == "scalar" {
                continue;
            } else {
                best_tier.clone()
            };

            let mut output = vec![0.0f32; block_size];
            let stats = bench_timing(config.warmup_iters, config.measure_iters, || {
                for blk in 0..config.num_blocks {
                    let start_byte = blk * block_bytes;
                    let end_byte = start_byte + block_bytes;
                    if end_byte <= data.len() {
                        let _ = dispatched_kernel
                            .dequant_block(&data[start_byte..end_byte], &mut output);
                    }
                }
            });

            let throughput_gbs = if stats.avg_us > 0.0 {
                total_bytes as f64 / (stats.avg_us * 1e-6) / 1e9
            } else {
                0.0
            };

            results.push(KernelBenchResult {
                quant_type: name.to_string(),
                simd_tier: dispatched_tier,
                operation: "dequant".to_string(),
                avg_us: stats.avg_us,
                min_us: stats.min_us,
                max_us: stats.max_us,
                throughput_gbs,
            });
        }
    }

    SimdComparisonResult {
        capabilities,
        results,
    }
}

/// Benchmark GEMV across SIMD tiers for all supported quant types.
///
/// For each type, this benchmarks both the reference (scalar) kernel and the
/// best available dispatched kernel.
pub fn run_gemv_comparison(config: &SimdComparisonConfig) -> SimdComparisonResult {
    let capabilities = SimdCapabilities::detect();
    let dispatcher = KernelDispatcher::new();
    let best_tier = capabilities.best_tier().to_string();

    let mut results = Vec::new();

    for &(tensor_type, name) in &BENCH_TYPES {
        let block_size = tensor_type.block_size();
        // Align cols to block_size
        let cols_aligned = if block_size > 0 {
            config.cols.div_ceil(block_size) * block_size
        } else {
            config.cols
        };
        let tensor = make_synthetic_tensor(tensor_type, config.rows, cols_aligned);
        let input = make_pseudo_random_vec(cols_aligned);
        let total_bytes = config.rows * cols_aligned * 4; // f32 output dimension

        // ── Scalar reference ─────────────────────────────────────────
        if let Some(ref_kernel) = reference_kernel_for(tensor_type) {
            let mut output = vec![0.0f32; config.rows];
            let stats = bench_timing(config.warmup_iters, config.measure_iters, || {
                output.fill(0.0);
                let _ = ref_kernel.gemv(&tensor, &input, &mut output);
            });

            let throughput_gbs = if stats.avg_us > 0.0 {
                total_bytes as f64 / (stats.avg_us * 1e-6) / 1e9
            } else {
                0.0
            };

            results.push(KernelBenchResult {
                quant_type: name.to_string(),
                simd_tier: "scalar".to_string(),
                operation: "gemv".to_string(),
                avg_us: stats.avg_us,
                min_us: stats.min_us,
                max_us: stats.max_us,
                throughput_gbs,
            });
        }

        // ── Dispatched (best available) ──────────────────────────────
        if let Ok(dispatched_kernel) = dispatcher.get_kernel(tensor_type) {
            let dispatched_tier = if best_tier == "scalar" {
                continue;
            } else {
                best_tier.clone()
            };

            let mut output = vec![0.0f32; config.rows];
            let stats = bench_timing(config.warmup_iters, config.measure_iters, || {
                output.fill(0.0);
                let _ = dispatched_kernel.gemv(&tensor, &input, &mut output);
            });

            let throughput_gbs = if stats.avg_us > 0.0 {
                total_bytes as f64 / (stats.avg_us * 1e-6) / 1e9
            } else {
                0.0
            };

            results.push(KernelBenchResult {
                quant_type: name.to_string(),
                simd_tier: dispatched_tier,
                operation: "gemv".to_string(),
                avg_us: stats.avg_us,
                min_us: stats.min_us,
                max_us: stats.max_us,
                throughput_gbs,
            });
        }
    }

    SimdComparisonResult {
        capabilities,
        results,
    }
}

/// Format comparison results as an ASCII table with box-drawing characters.
pub fn format_comparison_table(result: &SimdComparisonResult) -> String {
    let mut out = String::new();

    // Header line: detected tier
    let _ = writeln!(
        out,
        "CPU: {} (best tier: {})",
        describe_capabilities(&result.capabilities),
        result.capabilities.best_tier(),
    );
    let _ = writeln!(out);

    // Column widths
    const W_TYPE: usize = 8;
    const W_TIER: usize = 9;
    const W_OP: usize = 8;
    const W_AVG: usize = 10;
    const W_MIN: usize = 10;
    const W_GBS: usize = 10;

    // Top border
    let _ = writeln!(
        out,
        "╔{0:═<W_TYPE$}╦{0:═<W_TIER$}╦{0:═<W_OP$}╦{0:═<W_AVG$}╦{0:═<W_MIN$}╦{0:═<W_GBS$}╗",
        "",
    );

    // Header row
    let _ = writeln!(
        out,
        "║{:<W_TYPE$}║{:<W_TIER$}║{:<W_OP$}║{:<W_AVG$}║{:<W_MIN$}║{:<W_GBS$}║",
        " Type", " SIMD", " Op", " Avg (µs)", " Min (µs)", " GB/s",
    );

    // Header separator
    let _ = writeln!(
        out,
        "╠{0:═<W_TYPE$}╬{0:═<W_TIER$}╬{0:═<W_OP$}╬{0:═<W_AVG$}╬{0:═<W_MIN$}╬{0:═<W_GBS$}╣",
        "",
    );

    // Data rows
    for r in &result.results {
        let _ = writeln!(
            out,
            "║ {:<wt$}║ {:<ws$}║ {:<wo$}║ {:>wa$.1} ║ {:>wm$.1} ║ {:>wg$.2} ║",
            r.quant_type,
            r.simd_tier,
            r.operation,
            r.avg_us,
            r.min_us,
            r.throughput_gbs,
            wt = W_TYPE - 1,
            ws = W_TIER - 1,
            wo = W_OP - 1,
            wa = W_AVG - 2,
            wm = W_MIN - 2,
            wg = W_GBS - 2,
        );
    }

    // Bottom border
    let _ = writeln!(
        out,
        "╚{0:═<W_TYPE$}╩{0:═<W_TIER$}╩{0:═<W_OP$}╩{0:═<W_AVG$}╩{0:═<W_MIN$}╩{0:═<W_GBS$}╝",
        "",
    );

    out
}

/// Describe capabilities as a compact string.
fn describe_capabilities(caps: &SimdCapabilities) -> String {
    let mut features = Vec::new();
    if caps.avx512f {
        features.push("AVX-512F");
    }
    if caps.avx2 {
        features.push("AVX2");
    }
    if caps.fma {
        features.push("FMA");
    }
    if caps.neon {
        features.push("NEON");
    }
    if features.is_empty() {
        features.push("scalar only");
    }
    features.join(", ")
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> SimdComparisonConfig {
        SimdComparisonConfig {
            rows: 64,
            cols: 64,
            num_blocks: 8,
            warmup_iters: 1,
            measure_iters: 3,
        }
    }

    #[test]
    fn test_dequant_comparison_runs() {
        let config = small_config();
        let result = run_dequant_comparison(&config);
        // Should have at least one result per benchmarked type (scalar)
        assert!(
            !result.results.is_empty(),
            "dequant comparison should produce results"
        );
        for r in &result.results {
            assert_eq!(r.operation, "dequant");
            assert!(r.avg_us >= 0.0, "avg_us must be non-negative");
            assert!(r.min_us >= 0.0, "min_us must be non-negative");
            assert!(r.throughput_gbs >= 0.0, "throughput must be non-negative");
        }
    }

    #[test]
    fn test_gemv_comparison_runs() {
        let config = small_config();
        let result = run_gemv_comparison(&config);
        assert!(
            !result.results.is_empty(),
            "gemv comparison should produce results"
        );
        for r in &result.results {
            assert_eq!(r.operation, "gemv");
            assert!(r.avg_us >= 0.0);
            assert!(r.min_us >= 0.0);
        }
    }

    #[test]
    fn test_format_table_contains_header() {
        let result = SimdComparisonResult {
            capabilities: SimdCapabilities::detect(),
            results: vec![KernelBenchResult {
                quant_type: "Q4_0".to_string(),
                simd_tier: "scalar".to_string(),
                operation: "dequant".to_string(),
                avg_us: 123.4,
                min_us: 120.1,
                max_us: 130.0,
                throughput_gbs: 2.34,
            }],
        };
        let table = format_comparison_table(&result);
        assert!(table.contains("Type"), "table must contain 'Type' header");
        assert!(table.contains("SIMD"), "table must contain 'SIMD' header");
        assert!(table.contains("Op"), "table must contain 'Op' header");
        assert!(table.contains("Avg"), "table must contain 'Avg' header");
        assert!(table.contains("Min"), "table must contain 'Min' header");
        assert!(table.contains("GB/s"), "table must contain 'GB/s' header");
        assert!(table.contains("Q4_0"), "table must contain data row");
    }

    #[test]
    fn test_synthetic_blocks_valid_size() {
        for &(tensor_type, _name) in &BENCH_TYPES {
            let num_blocks = 16;
            let data = make_synthetic_blocks(tensor_type, num_blocks);
            let expected = tensor_type.block_bytes() * num_blocks;
            assert_eq!(
                data.len(),
                expected,
                "synthetic block size mismatch for {}",
                tensor_type.name(),
            );
        }
    }

    #[test]
    fn test_default_config() {
        let config = SimdComparisonConfig::default();
        assert_eq!(config.rows, 4096);
        assert_eq!(config.cols, 4096);
        assert_eq!(config.num_blocks, 1024);
        assert_eq!(config.warmup_iters, 10);
        assert_eq!(config.measure_iters, 100);
    }
}

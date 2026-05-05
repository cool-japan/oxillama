//! Latency-vs-batch-size heatmap benchmark infrastructure.
//!
//! Performs a 2-D sweep over `(batch_size × seq_len)` combinations and records
//! decode throughput (tok/s), p99 latency (ms), and memory usage (bytes) at
//! each grid point.  Results are aggregated into a [`BatchHeatmap`] whose
//! [`summary_table`][BatchHeatmap::summary_table] renders a Markdown table
//! suitable for CI output.
//!
//! The measurement discipline mirrors [`LongContextSweep`] (same warm-up /
//! measure philosophy) but adds a second dimension — batch size — which exposes
//! the crossover point where continuous batching stops paying off.
//!
//! # Design
//!
//! For each `(batch_size, seq_len)` cell:
//! 1. **Warm-up** — 3 rounds of: prefill `seq_len` tokens, decode `batch_size`
//!    tokens, reset.  Primes caches and JIT paths so first-measurement noise is
//!    excluded.
//! 2. **Measurement** — [`MEASURE_ROUNDS`] rounds of the same loop; each
//!    individual decode call is timed with [`std::time::Instant`] so we can
//!    extract a per-call latency distribution.
//! 3. **Aggregation** — derive `toks/s` from total tokens / total wall time;
//!    derive `p99_latency_ms` from the sorted per-call distribution.
//!
//! # Markdown output
//!
//! [`BatchHeatmap::summary_table`] emits a table where:
//! - **Rows** correspond to `seq_len` values.
//! - **Columns** correspond to `batch_size` values.
//! - **Cells** show `<toks/s> tok/s`.
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxillama_bench::{BatchHeatmap, PrefillDecodeBench};
//!
//! struct MyEngine;
//! impl PrefillDecodeBench for MyEngine {
//!     fn bench_prefill(&mut self, _tokens: usize) -> f64 { 0.0 }
//!     fn bench_decode_token(&mut self) -> f64 { 0.0 }
//!     fn bench_reset(&mut self) {}
//! }
//!
//! let mut engine = MyEngine;
//! let hm = BatchHeatmap::run(
//!     &mut engine,
//!     BatchHeatmap::default_batch_sizes(),
//!     BatchHeatmap::default_seq_lens(),
//!     "my-model",
//! ).unwrap();
//! println!("{}", hm.summary_table());
//! ```

use std::fmt::Write as _;

use crate::memory::current_rss_bytes;
use crate::prefill_decode::PrefillDecodeBench;

/// Convenience alias: all public functions in this module return
/// `Result<_, String>` (matching the existing `long_context` convention).
pub type HeatmapBenchResult<T> = Result<T, String>;

/// Number of measurement rounds used when sampling a single heatmap cell.
///
/// 20 rounds with `batch_size` decode calls per round gives a distribution of
/// `20 × batch_size` latency samples — enough for a stable p99 even at
/// `batch_size = 1`.
const MEASURE_ROUNDS: usize = 20;

/// Number of warm-up rounds before measurement.
const WARMUP_ROUNDS: usize = 3;

// ── Data types ───────────────────────────────────────────────────────────────

/// Single measured cell in the batch-size × seq-len heatmap grid.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HeatmapPoint {
    /// Number of tokens decoded in each synthetic batch.
    pub batch_size: usize,
    /// Number of prompt tokens used in the prefill phase for this cell.
    pub seq_len: usize,
    /// Sustained decode throughput for this `(batch_size, seq_len)` pair,
    /// in tokens per second.
    pub toks_per_sec: f64,
    /// 99th-percentile decode latency across all measured calls, in milliseconds.
    pub p99_latency_ms: f64,
    /// Process RSS sampled immediately after the warm-up phase (bytes).
    ///
    /// This captures the memory footprint *after* the KV-cache is warm for
    /// `seq_len` tokens, which is the baseline load imposed on real hardware.
    pub memory_bytes: usize,
}

/// Aggregated 2-D heatmap: all measured grid points plus axis labels.
#[derive(Debug, Clone)]
pub struct BatchHeatmap {
    /// All measured grid points in (batch_size, seq_len) scan order.
    pub points: Vec<HeatmapPoint>,
    /// Batch-size axis values (columns), in the order they were swept.
    pub batch_sizes: Vec<usize>,
    /// Sequence-length axis values (rows), in the order they were swept.
    pub seq_lens: Vec<usize>,
    /// Human-readable label for the model / configuration under test.
    pub model_label: String,
}

// ── Core implementation ──────────────────────────────────────────────────────

impl BatchHeatmap {
    /// Run a 2-D sweep over every `(batch_size, seq_len)` combination.
    ///
    /// The outer loop iterates over `batch_sizes` and the inner loop over
    /// `seq_lens`, so the CSV / table layout is batch-first.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` when either axis slice is empty — a heatmap with
    /// zero cells is a configuration error.
    pub fn run<E>(
        engine: &mut E,
        batch_sizes: &[usize],
        seq_lens: &[usize],
        model_label: impl Into<String>,
    ) -> HeatmapBenchResult<Self>
    where
        E: PrefillDecodeBench,
    {
        if batch_sizes.is_empty() {
            return Err("batch_sizes must not be empty".to_string());
        }
        if seq_lens.is_empty() {
            return Err("seq_lens must not be empty".to_string());
        }

        let label = model_label.into();
        let mut points = Vec::with_capacity(batch_sizes.len() * seq_lens.len());

        for &batch_size in batch_sizes {
            for &seq_len in seq_lens {
                let point = Self::measure_point(engine, batch_size, seq_len)?;
                points.push(point);
            }
        }

        Ok(Self {
            points,
            batch_sizes: batch_sizes.to_vec(),
            seq_lens: seq_lens.to_vec(),
            model_label: label,
        })
    }

    /// Measure a single `(batch_size, seq_len)` grid cell.
    ///
    /// Performs a warm-up phase ([`WARMUP_ROUNDS`] rounds) followed by a
    /// measurement phase ([`MEASURE_ROUNDS`] rounds).  Each round consists of:
    /// 1. Prefill `seq_len` prompt tokens.
    /// 2. Decode `batch_size` tokens, capturing the per-call wall time.
    /// 3. Reset the engine state.
    ///
    /// From the measurement phase we derive:
    /// - `toks_per_sec`: total tokens generated / total elapsed wall time.
    /// - `p99_latency_ms`: 99th percentile of the per-call latency distribution.
    /// - `memory_bytes`: RSS sampled once, immediately after the warm-up phase.
    fn measure_point<E>(
        engine: &mut E,
        batch_size: usize,
        seq_len: usize,
    ) -> HeatmapBenchResult<HeatmapPoint>
    where
        E: PrefillDecodeBench,
    {
        // ── Warm-up phase ────────────────────────────────────────────────────
        for _ in 0..WARMUP_ROUNDS {
            engine.bench_reset();
            let _ = engine.bench_prefill(seq_len);
            for _ in 0..batch_size {
                let _ = engine.bench_decode_token();
            }
        }

        // Sample RSS after warm-up so we capture the steady-state footprint.
        let memory_bytes = current_rss_bytes().unwrap_or(0) as usize;

        // ── Measurement phase ────────────────────────────────────────────────
        //
        // Collect per-call decode latencies so we can compute a full latency
        // distribution (not just the mean).  The capacity is pre-allocated to
        // avoid re-allocation noise during the hot loop.
        let total_samples = MEASURE_ROUNDS * batch_size;
        let mut call_latencies_ms: Vec<f64> = Vec::with_capacity(total_samples.max(1));

        let start_total = std::time::Instant::now();

        for _ in 0..MEASURE_ROUNDS {
            engine.bench_reset();
            let _ = engine.bench_prefill(seq_len);
            for _ in 0..batch_size {
                let call_start = std::time::Instant::now();
                let _ = engine.bench_decode_token();
                let elapsed_ms = call_start.elapsed().as_secs_f64() * 1_000.0;
                call_latencies_ms.push(elapsed_ms);
            }
        }

        let elapsed_total = start_total.elapsed().as_secs_f64();
        let total_tokens = MEASURE_ROUNDS * batch_size;

        // Guard against zero-elapsed (e.g. stub engines that return instantly).
        let toks_per_sec = if elapsed_total > 0.0 {
            total_tokens as f64 / elapsed_total
        } else {
            0.0
        };

        // Sort for percentile extraction.
        call_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p99_latency_ms = compute_p99(&call_latencies_ms);

        Ok(HeatmapPoint {
            batch_size,
            seq_len,
            toks_per_sec,
            p99_latency_ms,
            memory_bytes,
        })
    }

    // ── Summary rendering ────────────────────────────────────────────────────

    /// Render results as a Markdown table.
    ///
    /// Layout:
    /// - **Rows** = `seq_len` values.
    /// - **Columns** = `batch_size` values.
    /// - **Cells** = `<toks/s> tok/s` (rounded to nearest integer).
    ///
    /// Missing cells (if `points` is a partial result) are rendered as `—`.
    ///
    /// ```text
    /// ## Batch Heatmap: my-model
    ///
    /// | seq_len | batch=1 | batch=2 | batch=4 |
    /// |--------:|--------:|--------:|--------:|
    /// |     128 | 1234 tok/s | 2345 tok/s | 3456 tok/s |
    /// ...
    /// ```
    pub fn summary_table(&self) -> String {
        let mut out = String::new();

        let _ = writeln!(out, "## Batch Heatmap: {}", self.model_label);
        let _ = writeln!(out);

        // Header row
        out.push_str("| seq_len");
        for &bs in &self.batch_sizes {
            let _ = write!(out, " | batch={}", bs);
        }
        out.push_str(" |\n");

        // Separator row — right-align everything for readability
        out.push_str("|--------:");
        for _ in &self.batch_sizes {
            out.push_str("|--------:");
        }
        out.push_str("|\n");

        // Data rows — one row per seq_len
        for &sl in &self.seq_lens {
            let _ = write!(out, "| {:>7}", sl);
            for &bs in &self.batch_sizes {
                match self.lookup(bs, sl) {
                    Some(pt) => {
                        let _ = write!(out, " | {:.0} tok/s", pt.toks_per_sec);
                    }
                    None => {
                        out.push_str(" | —");
                    }
                }
            }
            out.push_str(" |\n");
        }

        out
    }

    /// Return the p99-latency-focused variant of the summary table.
    ///
    /// Same layout as [`summary_table`][Self::summary_table] but cells show
    /// `p99 = <ms> ms` instead of `toks/s`.  Useful for latency-SLO analysis.
    pub fn p99_table(&self) -> String {
        let mut out = String::new();

        let _ = writeln!(out, "## Batch Heatmap (p99 latency): {}", self.model_label);
        let _ = writeln!(out);

        out.push_str("| seq_len");
        for &bs in &self.batch_sizes {
            let _ = write!(out, " | batch={}", bs);
        }
        out.push_str(" |\n");

        out.push_str("|--------:");
        for _ in &self.batch_sizes {
            out.push_str("|--------:");
        }
        out.push_str("|\n");

        for &sl in &self.seq_lens {
            let _ = write!(out, "| {:>7}", sl);
            for &bs in &self.batch_sizes {
                match self.lookup(bs, sl) {
                    Some(pt) => {
                        let _ = write!(out, " | {:.2} ms", pt.p99_latency_ms);
                    }
                    None => {
                        out.push_str(" | —");
                    }
                }
            }
            out.push_str(" |\n");
        }

        out
    }

    // ── Axis defaults ────────────────────────────────────────────────────────

    /// Default batch-size axis for quick sweeps.
    ///
    /// Covers the range where most continuous-batching systems show diminishing
    /// returns: 1 (baseline), 2, 4, 8 (typical saturation for a single model
    /// shard on a mid-range GPU).
    pub fn default_batch_sizes() -> &'static [usize] {
        &[1, 2, 4, 8]
    }

    /// Default sequence-length axis for quick sweeps.
    ///
    /// Covers short (chat-style), medium (document), and long (context-heavy)
    /// prompt regimes without taking too long in CI.
    pub fn default_seq_lens() -> &'static [usize] {
        &[128, 512, 1_024, 2_048]
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Look up a cell by `(batch_size, seq_len)` coordinates.
    ///
    /// Returns `None` when no matching point exists (partial result sets).
    fn lookup(&self, batch_size: usize, seq_len: usize) -> Option<&HeatmapPoint> {
        self.points
            .iter()
            .find(|p| p.batch_size == batch_size && p.seq_len == seq_len)
    }
}

// ── Statistics helper ────────────────────────────────────────────────────────

/// Compute the 99th-percentile value from an already-sorted slice.
///
/// Returns `0.0` for an empty slice.  The percentile index uses
/// `ceil(n × 0.99) − 1` (saturated to `[0, n−1]`) which gives consistent
/// results for both small (n < 100) and large (n ≥ 100) sample counts.
fn compute_p99(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len().saturating_sub(1));
    sorted[idx]
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefill_decode::PrefillDecodeBench;

    // ── Minimal stub engine ──────────────────────────────────────────────────

    /// Stub engine where prefill and decode costs are proportional to
    /// `current_seq` so that measurements are deterministic and non-trivial.
    struct StubEngine {
        current_seq: usize,
    }

    impl StubEngine {
        fn new() -> Self {
            Self { current_seq: 0 }
        }
    }

    impl PrefillDecodeBench for StubEngine {
        fn bench_prefill(&mut self, tokens: usize) -> f64 {
            self.current_seq = tokens;
            tokens as f64 * 0.001
        }

        fn bench_decode_token(&mut self) -> f64 {
            self.current_seq as f64 * 0.000_01
        }

        fn bench_reset(&mut self) {
            self.current_seq = 0;
        }
    }

    // ── Test 1: correct number of points ─────────────────────────────────────

    #[test]
    fn heatmap_run_returns_all_points() {
        let mut eng = StubEngine::new();
        let hm = BatchHeatmap::run(&mut eng, &[1, 2], &[128, 512], "test")
            .expect("heatmap must succeed for valid inputs");
        // 2 batch_sizes × 2 seq_lens = 4 points
        assert_eq!(hm.points.len(), 4, "expected 4 grid points");
    }

    // ── Test 2: all axis combinations are represented ─────────────────────────

    #[test]
    fn heatmap_covers_all_combinations() {
        let batch_sizes = [1usize, 4, 8];
        let seq_lens = [128usize, 1024, 2048];
        let mut eng = StubEngine::new();
        let hm = BatchHeatmap::run(&mut eng, &batch_sizes, &seq_lens, "combo-test")
            .expect("heatmap must succeed");

        assert_eq!(
            hm.points.len(),
            batch_sizes.len() * seq_lens.len(),
            "expected {} points",
            batch_sizes.len() * seq_lens.len()
        );

        // Every (bs, sl) combination must appear exactly once.
        for &bs in &batch_sizes {
            for &sl in &seq_lens {
                assert!(
                    hm.points
                        .iter()
                        .any(|p| p.batch_size == bs && p.seq_len == sl),
                    "missing cell (batch_size={}, seq_len={})",
                    bs,
                    sl
                );
            }
        }
    }

    // ── Test 3: summary table contains expected headers ───────────────────────

    #[test]
    fn heatmap_summary_table_contains_headers() {
        let hm = BatchHeatmap {
            points: vec![HeatmapPoint {
                batch_size: 1,
                seq_len: 128,
                toks_per_sec: 500.0,
                p99_latency_ms: 2.0,
                memory_bytes: 0,
            }],
            batch_sizes: vec![1],
            seq_lens: vec![128],
            model_label: "header-test".into(),
        };
        let table = hm.summary_table();
        assert!(
            table.contains("seq_len"),
            "table must contain 'seq_len' header"
        );
        assert!(
            table.contains("batch=1"),
            "table must contain 'batch=1' header"
        );
        assert!(
            table.contains("128"),
            "table must contain the seq_len value"
        );
        assert!(table.contains("tok/s"), "table must contain 'tok/s' unit");
    }

    // ── Test 4: default params are non-empty ──────────────────────────────────

    #[test]
    fn heatmap_default_params_non_empty() {
        assert!(
            !BatchHeatmap::default_batch_sizes().is_empty(),
            "default_batch_sizes must be non-empty"
        );
        assert!(
            !BatchHeatmap::default_seq_lens().is_empty(),
            "default_seq_lens must be non-empty"
        );
    }

    // ── Test 5: empty batch_sizes returns Err ─────────────────────────────────

    #[test]
    fn heatmap_empty_batch_sizes_errors() {
        let mut eng = StubEngine::new();
        assert!(
            BatchHeatmap::run(&mut eng, &[], &[128], "test").is_err(),
            "empty batch_sizes should return Err"
        );
    }

    // ── Test 6: empty seq_lens returns Err ───────────────────────────────────

    #[test]
    fn heatmap_empty_seq_lens_errors() {
        let mut eng = StubEngine::new();
        assert!(
            BatchHeatmap::run(&mut eng, &[1], &[], "test").is_err(),
            "empty seq_lens should return Err"
        );
    }

    // ── Test 7: higher-throughput point renders correctly in the table ─────────

    #[test]
    fn heatmap_larger_batch_has_higher_throughput() {
        // Build a heatmap with known values: batch=8 has 8× the toks/s of batch=1.
        // This verifies that the table and lookup API correctly reflect supplied
        // throughput values without relying on non-deterministic wall-clock timing.
        let hm = BatchHeatmap {
            points: vec![
                HeatmapPoint {
                    batch_size: 1,
                    seq_len: 128,
                    toks_per_sec: 1_000.0,
                    p99_latency_ms: 1.0,
                    memory_bytes: 0,
                },
                HeatmapPoint {
                    batch_size: 8,
                    seq_len: 128,
                    toks_per_sec: 8_000.0,
                    p99_latency_ms: 0.5,
                    memory_bytes: 0,
                },
            ],
            batch_sizes: vec![1, 8],
            seq_lens: vec![128],
            model_label: "throughput-test".into(),
        };

        let pt_bs1 = hm.lookup(1, 128).expect("batch=1, seq=128 must exist");
        let pt_bs8 = hm.lookup(8, 128).expect("batch=8, seq=128 must exist");

        assert!(
            pt_bs8.toks_per_sec > pt_bs1.toks_per_sec,
            "batch=8 toks/s must exceed batch=1 toks/s in the heatmap"
        );
        let table = hm.summary_table();
        assert!(
            table.contains("8000"),
            "table must reflect batch=8 throughput"
        );
        assert!(
            table.contains("1000"),
            "table must reflect batch=1 throughput"
        );
    }

    // ── Test 8: p99 table header ──────────────────────────────────────────────

    #[test]
    fn heatmap_p99_table_contains_ms_unit() {
        let hm = BatchHeatmap {
            points: vec![HeatmapPoint {
                batch_size: 1,
                seq_len: 128,
                toks_per_sec: 500.0,
                p99_latency_ms: 2.5,
                memory_bytes: 0,
            }],
            batch_sizes: vec![1],
            seq_lens: vec![128],
            model_label: "p99-test".into(),
        };
        let table = hm.p99_table();
        assert!(table.contains("ms"), "p99 table must contain 'ms' unit");
        assert!(table.contains("p99"), "p99 table title must contain 'p99'");
    }

    // ── Test 9: lookup missing cell returns None ──────────────────────────────

    #[test]
    fn heatmap_lookup_missing_cell_returns_none() {
        let hm = BatchHeatmap {
            points: vec![],
            batch_sizes: vec![1],
            seq_lens: vec![128],
            model_label: "empty".into(),
        };
        assert!(
            hm.lookup(1, 128).is_none(),
            "lookup on empty points must return None"
        );
    }

    // ── Test 10: compute_p99 correctness ──────────────────────────────────────

    #[test]
    fn compute_p99_empty_slice() {
        assert!((compute_p99(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_p99_single_element() {
        assert!((compute_p99(&[42.0]) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_p99_sorted_hundred_elements() {
        // 1.0 .. 100.0 sorted: p99 = element at idx = ceil(100 * 0.99) - 1 = 98 = 99.0
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let p99 = compute_p99(&data);
        assert!(
            (p99 - 99.0).abs() < f64::EPSILON,
            "expected 99.0, got {p99}"
        );
    }

    // ── Test 11: run with default params succeeds ─────────────────────────────

    #[test]
    fn heatmap_run_default_params() {
        let mut eng = StubEngine::new();
        let hm = BatchHeatmap::run(
            &mut eng,
            BatchHeatmap::default_batch_sizes(),
            BatchHeatmap::default_seq_lens(),
            "default-params",
        )
        .expect("heatmap must succeed with default params");

        let expected_points =
            BatchHeatmap::default_batch_sizes().len() * BatchHeatmap::default_seq_lens().len();
        assert_eq!(
            hm.points.len(),
            expected_points,
            "expected {} points for default params",
            expected_points
        );
    }
}

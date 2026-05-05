//! CI regression gate: compare current benchmark results against a saved baseline.
//!
//! Provides a [`RegressionGate`] that loads a baseline JSON file, compares
//! incoming benchmark measurements against it, and emits a structured failure
//! list when any metric regresses beyond a configurable threshold.
//!
//! # Regression semantics
//!
//! | Metric | Direction | Regression condition |
//! |---|---|---|
//! | `toks_per_sec` | higher is better | `(baseline - current) / baseline > threshold` |
//! | `prefill_ms` | lower is better | `(current - baseline) / baseline > threshold` |
//! | `decode_ms_p99` | lower is better | `(current - baseline) / baseline > threshold` |
//!
//! New benchmarks (present in `current` but absent from `baseline`) are
//! silently skipped; they are not regressions.
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxillama_bench::regression_gate::{BaselineEntry, RegressionGate};
//! use std::path::Path;
//!
//! // Load baseline written by a previous CI run.
//! let gate = RegressionGate::from_file(Path::new("baseline.json"))
//!     .expect("failed to load baseline");
//!
//! let current = vec![
//!     BaselineEntry {
//!         name: "prefill_128".into(),
//!         toks_per_sec: 9500.0,
//!         prefill_ms: 12.0,
//!         decode_ms_p99: 1.1,
//!     },
//! ];
//!
//! if let Err(failures) = gate.check(&current) {
//!     eprintln!("{}", RegressionGate::format_report(&failures));
//!     std::process::exit(1);
//! }
//! ```

use std::path::Path;

// ── Data types ─────────────────────────────────────────────────────────────────

/// A single benchmark entry in the baseline (or current results) set.
///
/// The `name` field is the primary key used to match current results against
/// the stored baseline.  All three performance metrics must be present; use
/// `0.0` for metrics that are not applicable to a particular benchmark.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaselineEntry {
    /// Unique benchmark name (e.g. `"prefill_128"`, `"decode_greedy_64"`).
    pub name: String,
    /// Sustained decode throughput in tokens per second.
    pub toks_per_sec: f64,
    /// Prefill phase latency in milliseconds.
    pub prefill_ms: f64,
    /// P99 per-token decode latency in milliseconds.
    pub decode_ms_p99: f64,
}

/// A single detected regression.
///
/// Produced by [`RegressionGate::check`] for every metric that exceeds the
/// configured threshold.
#[derive(Debug, Clone)]
pub struct RegressionFailure {
    /// Benchmark name that regressed.
    pub name: String,
    /// The metric that regressed: one of `"toks_per_sec"`, `"prefill_ms"`,
    /// or `"decode_ms_p99"`.
    pub metric: String,
    /// Baseline (reference) value for this metric.
    pub baseline: f64,
    /// Current (measured) value for this metric.
    pub current: f64,
    /// How much worse the current value is, as a fraction of the baseline.
    ///
    /// Always positive when a regression is reported.  For `toks_per_sec`
    /// this is `(baseline - current) / baseline`; for latency metrics it is
    /// `(current - baseline) / baseline`.
    pub regression_pct: f64,
}

// ── RegressionGate ─────────────────────────────────────────────────────────────

/// CI regression gate: compares fresh benchmark results against a stored baseline.
///
/// Constructed via [`RegressionGate::new`] or [`RegressionGate::from_file`].
pub struct RegressionGate {
    /// Reference baseline entries keyed by benchmark name.
    pub baseline: Vec<BaselineEntry>,
    /// Maximum allowed regression fraction before flagging a failure.
    ///
    /// `0.05` means a 5% regression triggers a failure.
    pub threshold: f64,
}

impl RegressionGate {
    /// Create a new regression gate with the given baseline and threshold.
    ///
    /// # Arguments
    ///
    /// - `baseline`  — reference entries from a previous (good) run.
    /// - `threshold` — maximum allowed regression as a fraction, e.g. `0.05`
    ///   for 5%.
    pub fn new(baseline: Vec<BaselineEntry>, threshold: f64) -> Self {
        Self {
            baseline,
            threshold,
        }
    }

    /// Load a baseline from a JSON file produced by [`RegressionGate::save_baseline`].
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON is malformed.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read baseline file {}: {e}", path.display()))?;
        let baseline: Vec<BaselineEntry> = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("malformed baseline JSON in {}: {e}", path.display()))?;
        Ok(Self::new(baseline, 0.05))
    }

    /// Serialize `results` as a JSON baseline file at `path`.
    ///
    /// The file format is a JSON array of [`BaselineEntry`] objects,
    /// pretty-printed for human readability and VCS diff-friendliness.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or serialization fails.
    pub fn save_baseline(results: &[BaselineEntry], path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(results)
            .map_err(|e| anyhow::anyhow!("serialization failed: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| anyhow::anyhow!("cannot write baseline to {}: {e}", path.display()))?;
        Ok(())
    }

    /// Check `current` results against the stored baseline.
    ///
    /// For each entry in `current`:
    /// 1. Look up the matching baseline entry by `name`.  If no baseline entry
    ///    exists, the benchmark is new — skip it (not a regression).
    /// 2. For each metric, compute the regression fraction and compare against
    ///    [`self.threshold`][RegressionGate::threshold].
    /// 3. Collect all failures.
    ///
    /// Returns `Ok(())` when all checked metrics are within threshold;
    /// returns `Err(failures)` listing every metric that exceeded the threshold.
    pub fn check(&self, current: &[BaselineEntry]) -> Result<(), Vec<RegressionFailure>> {
        let mut failures: Vec<RegressionFailure> = Vec::new();

        for cur in current {
            // Find the matching baseline entry.
            let Some(base) = self.baseline.iter().find(|b| b.name == cur.name) else {
                // New benchmark — not a regression.
                continue;
            };

            // Check toks_per_sec (higher is better).
            if let Some(failure) = Self::check_higher_is_better(
                "toks_per_sec",
                &cur.name,
                base.toks_per_sec,
                cur.toks_per_sec,
                self.threshold,
            ) {
                failures.push(failure);
            }

            // Check prefill_ms (lower is better).
            if let Some(failure) = Self::check_lower_is_better(
                "prefill_ms",
                &cur.name,
                base.prefill_ms,
                cur.prefill_ms,
                self.threshold,
            ) {
                failures.push(failure);
            }

            // Check decode_ms_p99 (lower is better).
            if let Some(failure) = Self::check_lower_is_better(
                "decode_ms_p99",
                &cur.name,
                base.decode_ms_p99,
                cur.decode_ms_p99,
                self.threshold,
            ) {
                failures.push(failure);
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }

    /// Format a human-readable regression report suitable for CI log output.
    ///
    /// Produces a Markdown-style table listing each failing metric, its
    /// baseline vs current values, and the regression percentage.
    pub fn format_report(failures: &[RegressionFailure]) -> String {
        use std::fmt::Write as _;

        let mut out = String::new();
        let _ = writeln!(out, "## Benchmark Regression Report");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| Benchmark | Metric | Baseline | Current | Regression |"
        );
        let _ = writeln!(
            out,
            "|-----------|--------|----------|---------|------------|"
        );

        for f in failures {
            let _ = writeln!(
                out,
                "| {} | {} | {:.4} | {:.4} | +{:.2}% |",
                f.name,
                f.metric,
                f.baseline,
                f.current,
                f.regression_pct * 100.0,
            );
        }

        if failures.is_empty() {
            let _ = writeln!(out, "| — | — | — | — | No regressions |");
        }

        out
    }

    // ── Internal helpers ────────────────────────────────────────────────────────

    /// Check a metric where **higher is better** (e.g. throughput).
    ///
    /// Regression when `(baseline - current) / baseline > threshold`.
    fn check_higher_is_better(
        metric: &str,
        name: &str,
        baseline: f64,
        current: f64,
        threshold: f64,
    ) -> Option<RegressionFailure> {
        // Guard against zero/negative baselines to avoid NaN/Inf comparisons.
        if baseline <= 0.0 {
            return None;
        }
        let regression_pct = (baseline - current) / baseline;
        if regression_pct > threshold {
            Some(RegressionFailure {
                name: name.to_string(),
                metric: metric.to_string(),
                baseline,
                current,
                regression_pct,
            })
        } else {
            None
        }
    }

    /// Check a metric where **lower is better** (e.g. latency in ms).
    ///
    /// Regression when `(current - baseline) / baseline > threshold`.
    fn check_lower_is_better(
        metric: &str,
        name: &str,
        baseline: f64,
        current: f64,
        threshold: f64,
    ) -> Option<RegressionFailure> {
        // Guard against zero/negative baselines.
        if baseline <= 0.0 {
            return None;
        }
        let regression_pct = (current - baseline) / baseline;
        if regression_pct > threshold {
            Some(RegressionFailure {
                name: name.to_string(),
                metric: metric.to_string(),
                baseline,
                current,
                regression_pct,
            })
        } else {
            None
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_baseline(name: &str, toks: f64, prefill: f64, p99: f64) -> BaselineEntry {
        BaselineEntry {
            name: name.to_string(),
            toks_per_sec: toks,
            prefill_ms: prefill,
            decode_ms_p99: p99,
        }
    }

    // ── Test 1: within-threshold passes ───────────────────────────────────────

    #[test]
    fn regression_within_threshold_passes() {
        // Baseline: 10_000 tok/s, 10 ms prefill, 1 ms p99.
        // Current: 9_700 tok/s (−3%), 10.3 ms prefill (+3%), 1.03 ms p99 (+3%).
        // Threshold: 5%.  All metrics are within threshold → Ok(()).
        let gate = RegressionGate::new(vec![make_baseline("bench_a", 10_000.0, 10.0, 1.0)], 0.05);
        let current = vec![make_baseline("bench_a", 9_700.0, 10.3, 1.03)];
        assert!(
            gate.check(&current).is_ok(),
            "3% regression must pass a 5% threshold"
        );
    }

    // ── Test 2: above-threshold fails ─────────────────────────────────────────

    #[test]
    fn regression_above_threshold_fails() {
        // Current tok/s is 10% lower than baseline → regression.
        let gate = RegressionGate::new(vec![make_baseline("bench_b", 10_000.0, 10.0, 1.0)], 0.05);
        let current = vec![make_baseline("bench_b", 9_000.0, 10.0, 1.0)];
        let result = gate.check(&current);
        assert!(
            result.is_err(),
            "10% throughput drop must fail a 5% threshold"
        );
        let failures = result.expect_err("should have failures");
        assert!(!failures.is_empty());
        assert_eq!(failures[0].metric, "toks_per_sec");
        assert_eq!(failures[0].name, "bench_b");
    }

    // ── Test 3: report contains metric name and percentages ───────────────────

    #[test]
    fn gate_emits_per_metric_failure_messages() {
        let gate = RegressionGate::new(vec![make_baseline("decode_fast", 5_000.0, 5.0, 0.5)], 0.05);
        // Trigger a latency regression: p99 is 20% higher.
        let current = vec![make_baseline("decode_fast", 5_000.0, 5.0, 0.6)];
        let failures = gate.check(&current).expect_err("should fail");
        assert!(!failures.is_empty());

        let report = RegressionGate::format_report(&failures);
        assert!(
            report.contains("decode_ms_p99"),
            "report must name the regressed metric; got:\n{report}"
        );
        assert!(
            report.contains("decode_fast"),
            "report must name the benchmark; got:\n{report}"
        );
        // The regression percentage should appear numerically.
        assert!(
            report.contains('%'),
            "report must include a percentage symbol; got:\n{report}"
        );
    }

    // ── Test 4: new benchmark not in baseline is skipped gracefully ───────────

    #[test]
    fn gate_handles_missing_baseline_entry_gracefully() {
        // Baseline has "old_bench"; current has only "new_bench" (new).
        let gate = RegressionGate::new(vec![make_baseline("old_bench", 8_000.0, 8.0, 0.8)], 0.05);
        let current = vec![make_baseline("new_bench", 100.0, 100.0, 100.0)];
        assert!(
            gate.check(&current).is_ok(),
            "a benchmark absent from the baseline must not count as a regression"
        );
    }

    // ── Additional coverage ────────────────────────────────────────────────────

    #[test]
    fn gate_detects_latency_regression_only() {
        // tok/s unchanged, but prefill_ms is 20% higher (threshold 5%).
        let gate = RegressionGate::new(
            vec![make_baseline("latency_check", 1_000.0, 10.0, 1.0)],
            0.05,
        );
        let current = vec![make_baseline("latency_check", 1_000.0, 12.0, 1.0)];
        let failures = gate
            .check(&current)
            .expect_err("must fail on prefill regression");
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].metric, "prefill_ms");
    }

    #[test]
    fn gate_reports_multiple_metric_failures() {
        // Both tok/s and p99 regress by 20%.
        let gate = RegressionGate::new(vec![make_baseline("multi_fail", 1_000.0, 10.0, 1.0)], 0.05);
        let current = vec![make_baseline("multi_fail", 800.0, 10.0, 1.2)];
        let failures = gate.check(&current).expect_err("must fail on both metrics");
        assert_eq!(
            failures.len(),
            2,
            "expected exactly 2 failures, got {failures:?}"
        );
    }

    #[test]
    fn gate_format_report_empty_failures_shows_no_regressions() {
        let report = RegressionGate::format_report(&[]);
        assert!(
            report.contains("No regressions"),
            "empty failure list must say 'No regressions'; got:\n{report}"
        );
    }

    #[test]
    fn gate_save_and_load_baseline_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxillama_bench_regression_test_baseline.json");

        let entries = vec![
            make_baseline("bench_x", 12_345.0, 6.7, 0.89),
            make_baseline("bench_y", 500.0, 100.0, 5.0),
        ];

        RegressionGate::save_baseline(&entries, &path).expect("save_baseline must succeed");

        let gate = RegressionGate::from_file(&path).expect("from_file must succeed after save");

        assert_eq!(gate.baseline.len(), 2);
        assert_eq!(gate.baseline[0].name, "bench_x");
        assert!((gate.baseline[0].toks_per_sec - 12_345.0).abs() < 0.001);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gate_zero_baseline_metric_is_skipped_gracefully() {
        // A zero-baseline metric must not produce NaN/Inf comparisons.
        let gate = RegressionGate::new(vec![make_baseline("zero_base", 0.0, 0.0, 0.0)], 0.05);
        let current = vec![make_baseline("zero_base", 1000.0, 50.0, 5.0)];
        // Zero baseline → skipped → no failures
        assert!(
            gate.check(&current).is_ok(),
            "zero-baseline metrics must not trigger false regressions"
        );
    }

    #[test]
    fn gate_improvement_does_not_fail() {
        // Better-than-baseline values must never fail.
        let gate = RegressionGate::new(vec![make_baseline("fast_bench", 1_000.0, 10.0, 1.0)], 0.05);
        // 50% faster, 50% shorter latencies — all improvements.
        let current = vec![make_baseline("fast_bench", 1_500.0, 5.0, 0.5)];
        assert!(
            gate.check(&current).is_ok(),
            "improvements must never be reported as regressions"
        );
    }
}

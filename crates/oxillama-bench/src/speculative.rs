//! Speculative decoding acceptance-rate sweep benchmark.
//!
//! Simulates speculative decoding over a 2-D grid of (draft_size, accept_threshold)
//! configurations using any engine implementing [`PrefillDecodeBench`].
//!
//! The simulation is deterministic: acceptance is modelled as
//! `floor(draft_size × threshold)` accepted tokens per step, avoiding
//! real RNG dependencies that would make test assertions non-reproducible.

use std::fmt::Write as _;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::prefill_decode::PrefillDecodeBench;

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a single speculative decoding sweep point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeBenchConfig {
    /// Number of draft tokens generated per speculative step (gamma).
    pub draft_size: usize,
    /// Artificial acceptance probability in the range `0.0..=1.0`.
    ///
    /// Used to simulate the token-acceptance loop without a real LM.
    /// A value of `1.0` means all draft tokens are accepted; `0.0`
    /// means none are accepted (every step falls back to target model).
    pub accept_threshold: f32,
    /// Number of decode steps executed per trial.
    pub n_steps: usize,
    /// Number of independent repetitions per configuration point.
    pub n_trials: usize,
}

// ── Result types ─────────────────────────────────────────────────────────────

/// Measured result for one (draft_size, accept_threshold) grid cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativePoint {
    /// Draft token count (gamma) for this cell.
    pub draft_size: usize,
    /// Artificial acceptance probability for this cell.
    pub accept_threshold: f32,
    /// Baseline single-token decode throughput (tokens / second).
    pub baseline_toks_per_sec: f64,
    /// Speculative decode effective throughput (tokens / second).
    pub spec_toks_per_sec: f64,
    /// `spec_toks_per_sec / baseline_toks_per_sec`.
    pub speedup: f64,
    /// Mean number of accepted tokens per speculative step.
    pub mean_accepted: f64,
}

/// Full result table for a 2-D `(draft_size, accept_threshold)` sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeBenchTable {
    /// All individual measurement points (row-major, thresholds × draft_sizes).
    pub points: Vec<SpeculativePoint>,
    /// Unique draft sizes present in this sweep.
    pub draft_sizes: Vec<usize>,
    /// Unique acceptance thresholds present in this sweep.
    pub accept_thresholds: Vec<f32>,
    /// Human-readable label for the model or stub under test.
    pub model_label: String,
}

impl SpeculativeBenchTable {
    /// Look up the measurement for a specific `(draft_size, threshold)` pair.
    ///
    /// Returns `None` if the pair was not part of the sweep grid.
    pub fn lookup(&self, draft_size: usize, threshold: f32) -> Option<&SpeculativePoint> {
        self.points
            .iter()
            .find(|p| p.draft_size == draft_size && (p.accept_threshold - threshold).abs() < 1e-4)
    }

    /// Render a Markdown table where rows = accept_threshold and columns =
    /// draft_size, with the cell value being the speedup ratio.
    ///
    /// Example output:
    /// ```text
    /// | threshold \ draft |  1 |  2 |  4 |  8 |
    /// |-------------------|----|----|----|----|
    /// |              0.50 | …  | …  | …  | …  |
    /// ```
    pub fn summary_table(&self) -> String {
        let mut out = String::new();

        // Header row: "threshold \ draft | d1 | d2 | d4 | …"
        let _ = write!(out, "| threshold \\ draft |");
        for &ds in &self.draft_sizes {
            let _ = write!(out, " {:>6} |", ds);
        }
        let _ = writeln!(out);

        // Separator row
        let _ = write!(out, "|-------------------|");
        for _ in &self.draft_sizes {
            let _ = write!(out, "--------|");
        }
        let _ = writeln!(out);

        // Data rows: one per acceptance threshold
        for &threshold in &self.accept_thresholds {
            let _ = write!(out, "| {:>17.2} |", threshold);
            for &ds in &self.draft_sizes {
                match self.lookup(ds, threshold) {
                    Some(p) => {
                        let _ = write!(out, " {:>6.2} |", p.speedup);
                    }
                    None => {
                        let _ = write!(out, " {:>6} |", "n/a");
                    }
                }
            }
            let _ = writeln!(out);
        }

        out
    }

    /// Render the speedup as a 2-D grid with accept_threshold rows and
    /// draft_size columns.  Identical to [`summary_table`] but exposes a
    /// separate name for callers that distinguish the two concepts.
    pub fn speedup_grid(&self) -> String {
        self.summary_table()
    }
}

// ── Stub engine ──────────────────────────────────────────────────────────────

/// A minimal stub engine for the speculative bench that simulates work via
/// a tiny spin-loop rather than real model inference.
///
/// `prefill_ns` and `decode_ns` control how many nanoseconds are spent per
/// call so that timings remain non-zero and meaningful even in CI.
#[derive(Debug, Clone)]
pub struct StubSpecEngine {
    /// Nanoseconds to spin per `bench_prefill` call (divided by token count).
    pub prefill_ns_per_tok: u64,
    /// Nanoseconds to spin per `bench_decode_token` call.
    pub decode_ns: u64,
}

impl Default for StubSpecEngine {
    fn default() -> Self {
        Self {
            prefill_ns_per_tok: 200,
            decode_ns: 1_000,
        }
    }
}

impl StubSpecEngine {
    /// Busy-loop for `nanos` nanoseconds.
    fn spin_ns(nanos: u64) {
        if nanos == 0 {
            return;
        }
        let start = Instant::now();
        let target = std::time::Duration::from_nanos(nanos);
        while start.elapsed() < target {
            std::hint::spin_loop();
        }
    }
}

impl PrefillDecodeBench for StubSpecEngine {
    fn bench_prefill(&mut self, prompt_tokens: usize) -> f64 {
        let nanos = self.prefill_ns_per_tok * prompt_tokens as u64;
        let start = Instant::now();
        Self::spin_ns(nanos);
        start.elapsed().as_secs_f64() * 1_000.0
    }

    fn bench_decode_token(&mut self) -> f64 {
        let start = Instant::now();
        Self::spin_ns(self.decode_ns);
        start.elapsed().as_secs_f64() * 1_000.0
    }

    fn bench_reset(&mut self) {
        // Stateless stub — nothing to reset.
    }
}

// ── Default sweep parameters ─────────────────────────────────────────────────

/// Default draft sizes for a standard sweep: 1, 2, 4, 8.
pub fn default_draft_sizes() -> &'static [usize] {
    &[1, 2, 4, 8]
}

/// Default acceptance thresholds for a standard sweep: 0.5, 0.7, 0.85, 0.95.
pub fn default_accept_thresholds() -> &'static [f32] {
    &[0.5, 0.7, 0.85, 0.95]
}

// ── Core sweep logic ─────────────────────────────────────────────────────────

/// Compute the deterministic number of accepted draft tokens for one step.
///
/// Using `floor(draft_size × threshold)` gives a reproducible result that
/// does not depend on an external PRNG, keeping benchmarks and tests
/// deterministic across platforms.
///
/// At `threshold = 1.0` all `draft_size` tokens are accepted.
/// At `threshold = 0.0` zero tokens are accepted (pure fallback).
#[inline]
fn accepted_tokens(draft_size: usize, threshold: f32) -> usize {
    // For threshold == 1.0 we want all tokens accepted regardless of float
    // rounding, so clamp first.
    let clamped = threshold.clamp(0.0, 1.0);
    if (clamped - 1.0_f32).abs() < 1e-5 {
        draft_size
    } else {
        (draft_size as f32 * clamped).floor() as usize
    }
}

/// Measure the baseline (single-token decode) throughput.
///
/// Runs `n_steps` sequential `bench_decode_token` calls and returns the
/// throughput in tokens per second.  The engine is reset before each trial
/// and timing is accumulated across all `n_trials` repeats.
fn measure_baseline<E: PrefillDecodeBench>(engine: &mut E, n_steps: usize, n_trials: usize) -> f64 {
    let mut total_tokens = 0usize;
    let mut total_secs = 0.0_f64;

    for _ in 0..n_trials {
        engine.bench_reset();
        // Short prefill to set up a realistic KV-cache state.
        let _ = engine.bench_prefill(8);

        let start = Instant::now();
        for _ in 0..n_steps {
            let _ = engine.bench_decode_token();
        }
        total_secs += start.elapsed().as_secs_f64();
        total_tokens += n_steps;
    }

    if total_secs > 0.0 {
        total_tokens as f64 / total_secs
    } else {
        0.0
    }
}

/// Measure speculative decode throughput for a single (draft_size, threshold)
/// configuration.
///
/// Returns `(toks_per_sec, mean_accepted_per_step)`.
///
/// For each step the stub simulates:
/// 1. `draft_size` calls to `draft_engine.bench_decode_token()`.
/// 2. Acceptance: `accepted_tokens(draft_size, threshold)` tokens accepted.
/// 3. One call to `target_engine.bench_decode_token()` for the fallback token.
///
/// The "effective throughput" is `(accepted + 1) / elapsed_per_step` because
/// speculative decoding always emits at least the fallback target token.
fn measure_speculative<E: PrefillDecodeBench + Clone>(
    target_engine: &mut E,
    draft_engine: &mut E,
    draft_size: usize,
    threshold: f32,
    n_steps: usize,
    n_trials: usize,
) -> (f64, f64) {
    let tokens_per_step = accepted_tokens(draft_size, threshold) + 1;
    let mut total_tokens = 0usize;
    let mut total_secs = 0.0_f64;
    let mut total_accepted = 0usize;

    for _ in 0..n_trials {
        target_engine.bench_reset();
        draft_engine.bench_reset();
        let _ = target_engine.bench_prefill(8);
        let _ = draft_engine.bench_prefill(8);

        let start = Instant::now();
        for _ in 0..n_steps {
            // Draft phase: call draft engine draft_size times.
            for _ in 0..draft_size {
                let _ = draft_engine.bench_decode_token();
            }
            // Verification + one fallback target call (always required).
            let _ = target_engine.bench_decode_token();

            let accepted = accepted_tokens(draft_size, threshold);
            total_accepted += accepted;
            // The step emits `accepted` draft tokens + 1 target token.
            total_tokens += tokens_per_step;
        }
        total_secs += start.elapsed().as_secs_f64();
    }

    let spec_tps = if total_secs > 0.0 {
        total_tokens as f64 / total_secs
    } else {
        0.0
    };

    let mean_accepted = if n_steps * n_trials > 0 {
        total_accepted as f64 / (n_steps * n_trials) as f64
    } else {
        0.0
    };

    (spec_tps, mean_accepted)
}

/// Run a full 2-D acceptance-rate sweep over all
/// `draft_sizes × accept_thresholds` combinations.
///
/// Returns a [`SpeculativeBenchTable`] containing one [`SpeculativePoint`]
/// per grid cell.
///
/// # Arguments
///
/// * `target_engine` — engine simulating the large target model.
/// * `draft_engine`  — engine simulating the small draft model.
/// * `draft_sizes`   — values of gamma (tokens drafted per step) to sweep.
/// * `accept_thresholds` — artificial acceptance probabilities to sweep.
/// * `n_steps`       — decode steps per measurement trial.
/// * `n_trials`      — repetitions per grid cell (results are summed).
/// * `model_label`   — arbitrary label embedded in the returned table.
pub fn run_acceptance_sweep<E: PrefillDecodeBench + Clone>(
    target_engine: &E,
    draft_engine: &E,
    draft_sizes: &[usize],
    accept_thresholds: &[f32],
    n_steps: usize,
    n_trials: usize,
    model_label: &str,
) -> SpeculativeBenchTable {
    let mut points = Vec::with_capacity(draft_sizes.len() * accept_thresholds.len());

    // Measure baseline once per threshold (same engine, independent of draft_size).
    // We recompute per (draft_size, threshold) pair to keep the loop uniform,
    // but the baseline does not depend on draft_size so results are consistent.
    for &draft_size in draft_sizes {
        for &threshold in accept_thresholds {
            // Clone for baseline measurement (target-only decode path).
            let mut target_baseline = target_engine.clone();
            let baseline_tps = measure_baseline(&mut target_baseline, n_steps, n_trials);

            // Fresh clones for speculative measurement (draft + target).
            let mut target_spec = target_engine.clone();
            let mut draft_spec = draft_engine.clone();
            let (spec_tps, mean_accepted) = measure_speculative(
                &mut target_spec,
                &mut draft_spec,
                draft_size,
                threshold,
                n_steps,
                n_trials,
            );

            let speedup = if baseline_tps > 0.0 {
                spec_tps / baseline_tps
            } else {
                0.0
            };

            points.push(SpeculativePoint {
                draft_size,
                accept_threshold: threshold,
                baseline_toks_per_sec: baseline_tps,
                spec_toks_per_sec: spec_tps,
                speedup,
                mean_accepted,
            });
        }
    }

    SpeculativeBenchTable {
        points,
        draft_sizes: draft_sizes.to_vec(),
        accept_thresholds: accept_thresholds.to_vec(),
        model_label: model_label.to_string(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── accepted_tokens helper ────────────────────────────────────────────────

    #[test]
    fn accepted_tokens_full_accept() {
        // threshold 1.0 → all draft tokens accepted
        assert_eq!(accepted_tokens(4, 1.0), 4);
        assert_eq!(accepted_tokens(8, 1.0), 8);
        assert_eq!(accepted_tokens(1, 1.0), 1);
    }

    #[test]
    fn accepted_tokens_zero_accept() {
        // threshold 0.0 → no draft tokens accepted
        assert_eq!(accepted_tokens(4, 0.0), 0);
        assert_eq!(accepted_tokens(8, 0.0), 0);
    }

    #[test]
    fn accepted_tokens_partial() {
        // floor(4 × 0.5) = 2
        assert_eq!(accepted_tokens(4, 0.5), 2);
        // floor(8 × 0.75) = 6
        assert_eq!(accepted_tokens(8, 0.75), 6);
    }

    // ── run_acceptance_sweep ─────────────────────────────────────────────────

    #[test]
    fn spec_bench_runs_with_stub_engines() {
        let engine = StubSpecEngine::default();
        let table = run_acceptance_sweep(&engine, &engine, &[1, 2], &[0.5, 1.0], 5, 1, "stub");
        assert!(
            !table.points.is_empty(),
            "run_acceptance_sweep must return a non-empty table"
        );
    }

    #[test]
    fn acceptance_sweep_covers_grid() {
        let engine = StubSpecEngine::default();
        let draft_sizes = &[1, 2, 4];
        let thresholds = &[0.5_f32, 0.85, 1.0];
        let table = run_acceptance_sweep(&engine, &engine, draft_sizes, thresholds, 3, 1, "stub");
        assert_eq!(
            table.points.len(),
            draft_sizes.len() * thresholds.len(),
            "table must have draft_sizes.len() × thresholds.len() points"
        );
    }

    #[test]
    fn summary_table_renders_markdown() {
        let engine = StubSpecEngine::default();
        let table = run_acceptance_sweep(
            &engine,
            &engine,
            default_draft_sizes(),
            default_accept_thresholds(),
            3,
            1,
            "stub",
        );
        let rendered = table.summary_table();
        assert!(
            rendered.contains('|'),
            "summary_table must produce Markdown with pipe characters; got: {rendered}"
        );
        assert!(
            rendered.contains("threshold"),
            "summary_table must contain 'threshold' header"
        );
    }

    #[test]
    fn spec_speedup_above_one_when_accept_high() {
        // For speedup > 1 we need the draft engine to be much faster than the
        // target engine.  With threshold=1.0 and draft_size=4 each spec step
        // costs:  4 × draft_ns + 1 × target_ns  and produces 5 tokens.
        // Baseline costs: 1 × target_ns per token.
        //
        // speedup ≈ (5 / (4·draft_ns + target_ns)) / (1 / target_ns)
        //         = 5·target_ns / (4·draft_ns + target_ns)
        //
        // For speedup > 1:  5·T > 4·d + T  ⟹  4T > 4d  ⟹  T > d
        // So we need target_ns >> draft_ns.  Use target=20_000, draft=500.
        let target_engine = StubSpecEngine {
            prefill_ns_per_tok: 100,
            decode_ns: 20_000, // slow target
        };
        let draft_engine = StubSpecEngine {
            prefill_ns_per_tok: 50,
            decode_ns: 500, // fast draft (40× faster)
        };
        let table =
            run_acceptance_sweep(&target_engine, &draft_engine, &[4], &[1.0], 20, 2, "stub");
        let point = table.lookup(4, 1.0).expect("grid cell must exist");
        assert!(
            point.speedup > 1.0,
            "speedup should be > 1.0 when draft is 40× faster than target and threshold=1.0, got {}",
            point.speedup
        );
    }

    #[test]
    fn point_serializes_to_json() {
        let point = SpeculativePoint {
            draft_size: 4,
            accept_threshold: 0.85,
            baseline_toks_per_sec: 100.0,
            spec_toks_per_sec: 180.0,
            speedup: 1.8,
            mean_accepted: 3.4,
        };
        let json = serde_json::to_string(&point).expect("serialization must succeed");
        let back: SpeculativePoint =
            serde_json::from_str(&json).expect("deserialization must succeed");
        assert_eq!(back.draft_size, point.draft_size);
        assert!((back.speedup - point.speedup).abs() < 1e-6);
        assert!((back.accept_threshold - point.accept_threshold).abs() < 1e-5);
    }

    #[test]
    fn bench_handles_zero_accept_rate() {
        let engine = StubSpecEngine::default();
        let table = run_acceptance_sweep(&engine, &engine, &[4], &[0.0], 10, 1, "stub");
        let point = table.lookup(4, 0.0).expect("grid cell must exist");
        // mean_accepted should be 0 (no draft tokens accepted)
        assert!(
            point.mean_accepted < 1e-9,
            "mean_accepted must be 0 when threshold=0.0, got {}",
            point.mean_accepted
        );
        // speedup should be low (each step still costs draft_size+1 calls vs 1
        // for baseline), but spec_toks_per_sec > 0 because we still emit 1 target token.
        // The speedup will be < 1 when draft overhead exceeds the single accepted token.
        assert!(
            point.spec_toks_per_sec >= 0.0,
            "spec_toks_per_sec must be non-negative"
        );
    }

    #[test]
    fn bench_handles_full_accept() {
        let engine = StubSpecEngine::default();
        let table = run_acceptance_sweep(&engine, &engine, &[4], &[1.0], 10, 1, "stub");
        let point = table.lookup(4, 1.0).expect("grid cell must exist");
        // All 4 draft tokens accepted every step
        assert!(
            (point.mean_accepted - 4.0).abs() < 1e-9,
            "mean_accepted must equal draft_size when threshold=1.0, got {}",
            point.mean_accepted
        );
    }

    #[test]
    fn bench_with_single_grid_point() {
        let engine = StubSpecEngine::default();
        let table = run_acceptance_sweep(&engine, &engine, &[2], &[0.7], 5, 1, "stub");
        assert_eq!(
            table.points.len(),
            1,
            "single grid cell must produce exactly 1 point"
        );
        assert_eq!(table.draft_sizes, vec![2]);
        assert!((table.accept_thresholds[0] - 0.7_f32).abs() < 1e-5);
        assert_eq!(table.model_label, "stub");
    }
}

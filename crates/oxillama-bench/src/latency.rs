//! Latency benchmarking — per-token and time-to-first-token measurements.

use std::time::{Duration, Instant};

/// Configuration for latency measurement.
#[derive(Debug, Clone)]
pub struct LatencyConfig {
    /// Number of warm-up iterations discarded before measurement begins.
    pub warmup_iters: usize,
    /// Number of measurement iterations used to build the sample distribution.
    pub measure_iters: usize,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            warmup_iters: 3,
            measure_iters: 20,
        }
    }
}

/// Statistical summary of a latency measurement run.
#[derive(Debug, Clone)]
pub struct LatencyResult {
    /// Arithmetic mean latency in milliseconds.
    pub mean_ms: f64,
    /// Median (P50) latency in milliseconds.
    pub p50_ms: f64,
    /// P95 latency in milliseconds.
    pub p95_ms: f64,
    /// P99 latency in milliseconds.
    pub p99_ms: f64,
    /// Minimum observed latency in milliseconds.
    pub min_ms: f64,
    /// Maximum observed latency in milliseconds.
    pub max_ms: f64,
    /// All raw samples in sorted order (milliseconds).
    pub samples: Vec<f64>,
}

impl LatencyResult {
    /// Compute percentile statistics from a slice of [`Duration`] samples.
    ///
    /// Samples are converted to milliseconds and sorted in place.
    pub fn from_durations(durations: Vec<Duration>) -> Self {
        let mut samples: Vec<f64> = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1_000.0)
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples.len();
        let mean_ms = if n > 0 {
            samples.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        Self {
            mean_ms,
            p50_ms: samples.get(n / 2).copied().unwrap_or(0.0),
            p95_ms: samples.get((n * 95) / 100).copied().unwrap_or(0.0),
            p99_ms: samples.get((n * 99) / 100).copied().unwrap_or(0.0),
            min_ms: samples.first().copied().unwrap_or(0.0),
            max_ms: samples.last().copied().unwrap_or(0.0),
            samples,
        }
    }

    /// Warm up `f` for `config.warmup_iters` iterations, then measure it for
    /// `config.measure_iters` iterations and return the latency statistics.
    pub fn measure<F: FnMut()>(config: &LatencyConfig, mut f: F) -> Self {
        for _ in 0..config.warmup_iters {
            f();
        }
        let durations: Vec<Duration> = (0..config.measure_iters)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed()
            })
            .collect();
        Self::from_durations(durations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_result_from_durations_basic() {
        let durs: Vec<Duration> = (1..=10).map(Duration::from_millis).collect();
        let result = LatencyResult::from_durations(durs);
        assert_eq!(result.samples.len(), 10);
        assert!((result.min_ms - 1.0).abs() < 0.1, "min = {}", result.min_ms);
        assert!(
            (result.max_ms - 10.0).abs() < 0.1,
            "max = {}",
            result.max_ms
        );
    }

    #[test]
    fn test_latency_measure_runs() {
        let config = LatencyConfig {
            warmup_iters: 1,
            measure_iters: 5,
        };
        let mut counter = 0usize;
        let result = LatencyResult::measure(&config, || {
            counter += 1;
        });
        // warmup (1) + measure (5) = 6 total calls
        assert_eq!(counter, 6);
        assert_eq!(result.samples.len(), 5);
    }
}

/// Token-level latency statistics for inference monitoring.
#[derive(Debug, Clone)]
pub struct TokenLatencyResult {
    /// Average per-token latency in milliseconds (decode phase).
    pub avg_token_ms: f64,
    /// Median (P50) per-token latency in milliseconds.
    pub p50_token_ms: f64,
    /// P99 per-token latency in milliseconds.
    pub p99_token_ms: f64,
    /// Average time-to-first-token in milliseconds.
    pub avg_ttft_ms: f64,
}

/// Collect latency measurements from a series of timed token operations.
///
/// `token_times_ms` should contain per-token durations in milliseconds.
/// The first entry is treated as the time-to-first-token (TTFT).
pub fn compute_latency_stats(token_times_ms: &[f64]) -> TokenLatencyResult {
    if token_times_ms.is_empty() {
        return TokenLatencyResult {
            avg_token_ms: 0.0,
            p50_token_ms: 0.0,
            p99_token_ms: 0.0,
            avg_ttft_ms: 0.0,
        };
    }

    let avg_ttft_ms = token_times_ms[0];

    // Decode tokens (skip TTFT)
    let decode_times: Vec<f64> = if token_times_ms.len() > 1 {
        token_times_ms[1..].to_vec()
    } else {
        vec![token_times_ms[0]]
    };

    let avg_token_ms = decode_times.iter().sum::<f64>() / decode_times.len() as f64;

    let mut sorted = decode_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50_idx = (sorted.len() as f64 * 0.50) as usize;
    let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);

    TokenLatencyResult {
        avg_token_ms,
        p50_token_ms: sorted[p50_idx],
        p99_token_ms: sorted[p99_idx],
        avg_ttft_ms,
    }
}

/// A simple latency timer that collects per-token timings.
pub struct LatencyTimer {
    start: Instant,
    token_times: Vec<f64>,
    last_token: Instant,
}

impl LatencyTimer {
    /// Start a new latency timer.
    pub fn start() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            token_times: Vec::new(),
            last_token: now,
        }
    }

    /// Record that a token was generated.
    pub fn record_token(&mut self) {
        let now = Instant::now();
        let elapsed_ms = now.duration_since(self.last_token).as_secs_f64() * 1000.0;
        self.token_times.push(elapsed_ms);
        self.last_token = now;
    }

    /// Finish and compute statistics.
    pub fn finish(self) -> TokenLatencyResult {
        compute_latency_stats(&self.token_times)
    }

    /// Total elapsed time in milliseconds since start.
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

#[cfg(test)]
mod token_latency_tests {
    use super::*;

    #[test]
    fn test_compute_latency_stats_basic() {
        let times = vec![50.0, 10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.0, 11.0, 12.0];
        let result = compute_latency_stats(&times);

        assert!((result.avg_ttft_ms - 50.0).abs() < 0.01);
        assert!(result.avg_token_ms > 0.0);
        assert!(result.p50_token_ms > 0.0);
        assert!(result.p99_token_ms >= result.p50_token_ms);
    }

    #[test]
    fn test_compute_latency_stats_empty() {
        let result = compute_latency_stats(&[]);
        assert!((result.avg_token_ms).abs() < 0.01);
    }

    #[test]
    fn test_compute_latency_stats_single() {
        let result = compute_latency_stats(&[42.0]);
        assert!((result.avg_ttft_ms - 42.0).abs() < 0.01);
        assert!((result.avg_token_ms - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_latency_timer() {
        let mut timer = LatencyTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        timer.record_token();
        std::thread::sleep(std::time::Duration::from_millis(1));
        timer.record_token();
        let result = timer.finish();
        assert!(result.avg_ttft_ms > 0.0);
        assert_eq!(result.avg_ttft_ms, result.avg_ttft_ms); // NaN check
    }
}

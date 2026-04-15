//! Throughput benchmarking — tokens-per-second measurement.

use std::time::{Duration, Instant};

/// Configuration for throughput measurement.
#[derive(Debug, Clone)]
pub struct ThroughputConfig {
    /// How long to sustain the measurement window (seconds).
    pub duration_secs: f64,
    /// How long to run the warm-up before measurement begins (seconds).
    pub warmup_secs: f64,
}

impl Default for ThroughputConfig {
    fn default() -> Self {
        Self {
            duration_secs: 5.0,
            warmup_secs: 1.0,
        }
    }
}

/// Result of a throughput measurement run.
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    /// Average token generation rate (tokens / second).
    pub tokens_per_second: f64,
    /// Total number of tokens produced during the measurement window.
    pub total_tokens: u64,
    /// Actual wall-clock duration of the measurement window (seconds).
    pub duration_secs: f64,
    /// Estimated floating-point operations per second.
    ///
    /// Set to `0.0` unless the caller has computed and supplied the FLOP
    /// count.  Use [`with_flops`](ThroughputResult::with_flops) to attach it
    /// after construction.
    pub flops_per_second: f64,
}

impl ThroughputResult {
    /// Warm up `f` for `config.warmup_secs` seconds, then drive it for
    /// `config.duration_secs` seconds, accumulating the token counts that `f`
    /// returns.
    ///
    /// `f` must return the number of tokens produced by a single call so that
    /// the harness can accumulate them without knowing the internal batch size.
    pub fn measure<F: FnMut() -> u64>(config: &ThroughputConfig, mut f: F) -> Self {
        // Warm-up phase — discard results.
        let warmup_end = Instant::now() + Duration::from_secs_f64(config.warmup_secs);
        while Instant::now() < warmup_end {
            f();
        }

        // Measurement phase.
        let start = Instant::now();
        let end = start + Duration::from_secs_f64(config.duration_secs);
        let mut total_tokens = 0u64;
        while Instant::now() < end {
            total_tokens += f();
        }
        let elapsed = start.elapsed().as_secs_f64();

        let tokens_per_second = if elapsed > 0.0 {
            total_tokens as f64 / elapsed
        } else {
            0.0
        };

        Self {
            tokens_per_second,
            total_tokens,
            duration_secs: elapsed,
            flops_per_second: 0.0,
        }
    }

    /// Attach an estimated FLOP-per-second value computed by the caller.
    #[must_use]
    pub fn with_flops(mut self, flops: f64) -> Self {
        self.flops_per_second = flops;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throughput_measures_tokens() {
        let config = ThroughputConfig {
            duration_secs: 0.05,
            warmup_secs: 0.01,
        };
        let result = ThroughputResult::measure(&config, || 1u64);
        assert!(
            result.total_tokens > 0,
            "expected at least one token counted"
        );
        assert!(result.tokens_per_second > 0.0);
    }

    #[test]
    fn test_throughput_with_flops() {
        let r = ThroughputResult {
            tokens_per_second: 100.0,
            total_tokens: 500,
            duration_secs: 5.0,
            flops_per_second: 0.0,
        };
        let r2 = r.with_flops(1e12);
        assert!((r2.flops_per_second - 1e12).abs() < 1.0);
    }
}

/// Token-level throughput statistics for inference monitoring.
#[derive(Debug, Clone)]
pub struct TokenThroughputResult {
    /// Tokens generated per second.
    pub tokens_per_sec: f64,
    /// Total generation time in milliseconds.
    pub total_ms: f64,
    /// Time-to-first-token in milliseconds.
    pub ttft_ms: f64,
    /// Total number of tokens generated.
    pub num_tokens: usize,
}

/// Configuration for token-level throughput tracking.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Target number of tokens to generate.
    pub num_tokens: usize,
    /// Number of warm-up tokens to skip.
    pub warmup_tokens: usize,
    /// Number of repetitions for averaging.
    pub repetitions: usize,
}

/// Compute throughput from per-token timing data.
///
/// `token_times_ms` should contain per-token durations in milliseconds.
pub fn compute_throughput(token_times_ms: &[f64]) -> TokenThroughputResult {
    if token_times_ms.is_empty() {
        return TokenThroughputResult {
            tokens_per_sec: 0.0,
            total_ms: 0.0,
            ttft_ms: 0.0,
            num_tokens: 0,
        };
    }

    let total_ms: f64 = token_times_ms.iter().sum();
    let ttft_ms = token_times_ms[0];
    let num_tokens = token_times_ms.len();
    let tokens_per_sec = if total_ms > 0.0 {
        (num_tokens as f64 / total_ms) * 1000.0
    } else {
        0.0
    };

    TokenThroughputResult {
        tokens_per_sec,
        total_ms,
        ttft_ms,
        num_tokens,
    }
}

/// A throughput tracker that measures token generation rates.
pub struct ThroughputTracker {
    config: TrackerConfig,
    token_times: Vec<f64>,
    last_token: Instant,
    warmup_complete: bool,
    warmup_count: usize,
}

impl ThroughputTracker {
    /// Create a new throughput tracker with the given configuration.
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            token_times: Vec::new(),
            last_token: Instant::now(),
            warmup_complete: false,
            warmup_count: 0,
        }
    }

    /// Start measurement (call after prompt processing is done).
    pub fn start(&mut self) {
        self.last_token = Instant::now();
        self.warmup_count = 0;
        self.warmup_complete = self.config.warmup_tokens == 0;
        self.token_times.clear();
    }

    /// Record a generated token.
    pub fn record_token(&mut self) {
        let now = Instant::now();
        let elapsed_ms = now.duration_since(self.last_token).as_secs_f64() * 1000.0;
        self.last_token = now;

        if !self.warmup_complete {
            self.warmup_count += 1;
            if self.warmup_count >= self.config.warmup_tokens {
                self.warmup_complete = true;
            }
            return;
        }

        self.token_times.push(elapsed_ms);
    }

    /// Finish and compute results.
    pub fn finish(self) -> TokenThroughputResult {
        compute_throughput(&self.token_times)
    }
}

/// Compute the mean and standard deviation of throughput across repetitions.
pub fn aggregate_throughput(results: &[TokenThroughputResult]) -> (f64, f64) {
    if results.is_empty() {
        return (0.0, 0.0);
    }

    let rates: Vec<f64> = results.iter().map(|r| r.tokens_per_sec).collect();
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;
    let stddev = variance.sqrt();

    (mean, stddev)
}

#[cfg(test)]
mod token_throughput_tests {
    use super::*;

    #[test]
    fn test_compute_throughput_basic() {
        // 10 tokens at 10ms each = 100 tok/s
        let times = vec![10.0; 10];
        let result = compute_throughput(&times);

        assert!((result.tokens_per_sec - 100.0).abs() < 0.01);
        assert!((result.total_ms - 100.0).abs() < 0.01);
        assert_eq!(result.num_tokens, 10);
    }

    #[test]
    fn test_compute_throughput_empty() {
        let result = compute_throughput(&[]);
        assert!((result.tokens_per_sec).abs() < 0.01);
        assert_eq!(result.num_tokens, 0);
    }

    #[test]
    fn test_aggregate_throughput() {
        let results = vec![
            TokenThroughputResult {
                tokens_per_sec: 100.0,
                total_ms: 100.0,
                ttft_ms: 10.0,
                num_tokens: 10,
            },
            TokenThroughputResult {
                tokens_per_sec: 120.0,
                total_ms: 83.3,
                ttft_ms: 8.0,
                num_tokens: 10,
            },
        ];

        let (mean, stddev) = aggregate_throughput(&results);
        assert!((mean - 110.0).abs() < 0.01);
        assert!(stddev > 0.0);
    }

    #[test]
    fn test_throughput_tracker() {
        let config = TrackerConfig {
            num_tokens: 10,
            warmup_tokens: 2,
            repetitions: 1,
        };
        let mut tracker = ThroughputTracker::new(config);
        tracker.start();

        // Warmup tokens (should be skipped)
        for _ in 0..2 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            tracker.record_token();
        }

        // Measured tokens
        for _ in 0..5 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            tracker.record_token();
        }

        let result = tracker.finish();
        assert_eq!(result.num_tokens, 5);
        assert!(result.tokens_per_sec > 0.0);
    }
}

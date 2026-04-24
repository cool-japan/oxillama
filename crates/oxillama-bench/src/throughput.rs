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

// ── Tokenizer throughput helpers ─────────────────────────────────────────

/// A fixed 1024-token Lorem Ipsum sample text used as tokenizer benchmark input.
///
/// This is a deterministic ASCII string that approximates real prose density.
pub const TOKENIZER_SAMPLE_TEXT: &str = concat!(
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor ",
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud ",
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure ",
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. ",
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt ",
    "mollit anim id est laborum. ",
    "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque ",
    "laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi ",
    "architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas ",
    "sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione ",
    "voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit ",
    "amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut ",
    "labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis ",
    "nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea ",
    "commodi consequatur. Quis autem vel eum iure reprehenderit qui in ea voluptate velit ",
    "esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas ",
    "nulla pariatur. At vero eos et accusamus et iusto odio dignissimos ducimus qui ",
    "blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias ",
    "excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia ",
    "deserunt mollitia animi, id est laborum et dolorum fuga.",
);

/// Configuration for tokenizer throughput benchmarks.
#[derive(Debug, Clone)]
pub struct TokenizerBenchConfig {
    /// Number of times to repeat the sample text to reach the target token count.
    pub repetitions: usize,
    /// Number of warm-up passes before measurement.
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
}

impl Default for TokenizerBenchConfig {
    fn default() -> Self {
        Self {
            repetitions: 2,
            warmup_iters: 3,
            measure_iters: 20,
        }
    }
}

/// Throughput result for tokenizer encode/decode operations.
#[derive(Debug, Clone)]
pub struct TokenizerThroughputResult {
    /// Number of tokens processed.
    pub token_count: usize,
    /// Average time per iteration (ms).
    pub avg_ms: f64,
    /// Minimum time per iteration (ms).
    pub min_ms: f64,
    /// Maximum time per iteration (ms).
    pub max_ms: f64,
    /// Tokens per second.
    pub tokens_per_sec: f64,
    /// Megabytes of text per second (text_bytes / time).
    pub mb_per_sec: f64,
}

/// Trait for tokenizer implementations that can be benchmarked.
///
/// Implementations should not allocate results eagerly; the harness controls
/// timing.
pub trait TokenizerBench {
    /// Encode the input text and return the number of tokens produced.
    fn encode(&mut self, text: &str) -> usize;
    /// Decode a slice of token IDs back to text (returns byte length of result).
    fn decode(&mut self, token_ids: &[u32]) -> usize;
}

/// Measure encode throughput using the provided tokenizer.
pub fn bench_tokenizer_encode<T: TokenizerBench>(
    tokenizer: &mut T,
    text: &str,
    config: &TokenizerBenchConfig,
) -> TokenizerThroughputResult {
    let text_bytes = text.len();

    // Warm-up
    for _ in 0..config.warmup_iters {
        let _ = tokenizer.encode(text);
    }

    // Measurement
    let mut times_ms = Vec::with_capacity(config.measure_iters);
    let mut last_token_count = 0usize;
    for _ in 0..config.measure_iters {
        let start = std::time::Instant::now();
        last_token_count = tokenizer.encode(text);
        times_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    compute_tokenizer_stats(last_token_count, text_bytes, &times_ms)
}

/// Measure decode throughput using the provided tokenizer.
///
/// `token_ids` is the token sequence to decode on each iteration.
pub fn bench_tokenizer_decode<T: TokenizerBench>(
    tokenizer: &mut T,
    token_ids: &[u32],
    config: &TokenizerBenchConfig,
) -> TokenizerThroughputResult {
    let token_count = token_ids.len();

    // Warm-up
    for _ in 0..config.warmup_iters {
        let _ = tokenizer.decode(token_ids);
    }

    // Measurement
    let mut times_ms = Vec::with_capacity(config.measure_iters);
    let mut last_byte_count = 0usize;
    for _ in 0..config.measure_iters {
        let start = std::time::Instant::now();
        last_byte_count = tokenizer.decode(token_ids);
        times_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    compute_tokenizer_stats(token_count, last_byte_count, &times_ms)
}

fn compute_tokenizer_stats(
    token_count: usize,
    byte_count: usize,
    times_ms: &[f64],
) -> TokenizerThroughputResult {
    if times_ms.is_empty() {
        return TokenizerThroughputResult {
            token_count,
            avg_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            tokens_per_sec: 0.0,
            mb_per_sec: 0.0,
        };
    }

    let sum: f64 = times_ms.iter().sum();
    let avg_ms = sum / times_ms.len() as f64;
    let min_ms = times_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = times_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let min_ms = if min_ms.is_infinite() { 0.0 } else { min_ms };
    let max_ms = if max_ms.is_infinite() { 0.0 } else { max_ms };

    let tokens_per_sec = if avg_ms > 0.0 {
        token_count as f64 / avg_ms * 1_000.0
    } else {
        0.0
    };

    let mb_per_sec = if avg_ms > 0.0 {
        byte_count as f64 / avg_ms / 1_000.0 // bytes / ms / 1000 = MB/s
    } else {
        0.0
    };

    TokenizerThroughputResult {
        token_count,
        avg_ms,
        min_ms,
        max_ms,
        tokens_per_sec,
        mb_per_sec,
    }
}

/// A stub BPE tokenizer for benchmarking without a real model file.
///
/// Splits on whitespace; encodes each word-piece as a 16-bit hash.
/// Suitable only for measuring the benchmark harness overhead, not real
/// tokenizer performance.
pub struct StubBpeTokenizer;

impl TokenizerBench for StubBpeTokenizer {
    fn encode(&mut self, text: &str) -> usize {
        // Approximate BPE: split on whitespace and punctuation boundaries.
        text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .count()
    }

    fn decode(&mut self, token_ids: &[u32]) -> usize {
        // Each token ID maps to a 4-byte ASCII stub "tok\0".
        token_ids.len() * 4
    }
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;

    #[test]
    fn test_stub_bpe_encode_nonempty() {
        let mut tok = StubBpeTokenizer;
        let n = tok.encode("Hello world foo bar");
        assert!(n > 0, "expected non-zero token count");
    }

    #[test]
    fn test_stub_bpe_decode_byte_count() {
        let mut tok = StubBpeTokenizer;
        let ids = vec![1u32, 2, 3, 4];
        let bytes = tok.decode(&ids);
        assert_eq!(bytes, 16); // 4 ids × 4 bytes each
    }

    #[test]
    fn test_bench_tokenizer_encode_returns_result() {
        let mut tok = StubBpeTokenizer;
        let config = TokenizerBenchConfig {
            repetitions: 1,
            warmup_iters: 1,
            measure_iters: 3,
        };
        let result = bench_tokenizer_encode(&mut tok, TOKENIZER_SAMPLE_TEXT, &config);
        assert!(result.token_count > 0);
        assert!(result.avg_ms >= 0.0);
        assert!(result.tokens_per_sec >= 0.0);
    }

    #[test]
    fn test_bench_tokenizer_decode_returns_result() {
        let ids: Vec<u32> = (0..1024).collect();
        let mut tok = StubBpeTokenizer;
        let config = TokenizerBenchConfig {
            repetitions: 1,
            warmup_iters: 1,
            measure_iters: 3,
        };
        let result = bench_tokenizer_decode(&mut tok, &ids, &config);
        assert_eq!(result.token_count, 1024);
        assert!(result.avg_ms >= 0.0);
    }

    #[test]
    fn test_tokenizer_throughput_result_empty_times() {
        let result = compute_tokenizer_stats(100, 500, &[]);
        assert_eq!(result.token_count, 100);
        assert_eq!(result.avg_ms, 0.0);
        assert_eq!(result.tokens_per_sec, 0.0);
    }
}

//! Engine metrics — thread-safe counters for throughput and cache statistics.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Thread-safe metrics counters for an inference engine.
#[derive(Debug, Default)]
pub struct EngineMetrics {
    /// Total tokens generated (decode phase).
    pub tokens_generated: AtomicU64,
    /// Total tokens processed in prefill.
    pub tokens_prefilled: AtomicU64,
    /// Total KV cache hits (prefix/page cache returns a cached slot).
    pub kv_cache_hits: AtomicU64,
    /// Total KV cache misses.
    pub kv_cache_misses: AtomicU64,
    /// Total decode time in nanoseconds.
    pub decode_nanos: AtomicU64,
    /// Total prefill time in nanoseconds.
    pub prefill_nanos: AtomicU64,
    /// Number of requests started.
    pub requests_started: AtomicU64,
    /// Number of requests completed.
    pub requests_completed: AtomicU64,
}

impl EngineMetrics {
    /// Create an `Arc`-wrapped, zero-initialised metrics instance.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a single decode token and the time taken to produce it.
    pub fn record_decode_token(&self, elapsed: Duration) {
        self.tokens_generated.fetch_add(1, Ordering::Relaxed);
        self.decode_nanos
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record a prefill phase that processed `n_tokens` prompt tokens.
    pub fn record_prefill(&self, n_tokens: u64, elapsed: Duration) {
        self.tokens_prefilled.fetch_add(n_tokens, Ordering::Relaxed);
        self.prefill_nanos
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record a KV-cache hit (prefix or page reuse).
    pub fn record_kv_hit(&self) {
        self.kv_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a KV-cache miss (full prefill required).
    pub fn record_kv_miss(&self) {
        self.kv_cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a new inference request has started.
    pub fn record_request_start(&self) {
        self.requests_started.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that an inference request has completed.
    pub fn record_request_complete(&self) {
        self.requests_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns `(decode_tokens_per_sec, prefill_tokens_per_sec)`.
    ///
    /// Both values are 0.0 if no time has been recorded for that phase.
    pub fn throughput(&self) -> (f64, f64) {
        let decode_tokens = self.tokens_generated.load(Ordering::Relaxed);
        let decode_nanos = self.decode_nanos.load(Ordering::Relaxed);
        let prefill_tokens = self.tokens_prefilled.load(Ordering::Relaxed);
        let prefill_nanos = self.prefill_nanos.load(Ordering::Relaxed);

        let decode_tps = if decode_nanos == 0 {
            0.0_f64
        } else {
            decode_tokens as f64 / (decode_nanos as f64 * 1e-9)
        };

        let prefill_tps = if prefill_nanos == 0 {
            0.0_f64
        } else {
            prefill_tokens as f64 / (prefill_nanos as f64 * 1e-9)
        };

        (decode_tps, prefill_tps)
    }

    /// Returns KV cache hit rate in the range `[0.0, 1.0]`.
    ///
    /// Returns 0.0 when no lookups have been recorded.
    pub fn kv_cache_hit_rate(&self) -> f64 {
        let hits = self.kv_cache_hits.load(Ordering::Relaxed);
        let misses = self.kv_cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Returns a point-in-time snapshot of all counters.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let (decode_tps, prefill_tps) = self.throughput();
        MetricsSnapshot {
            tokens_generated: self.tokens_generated.load(Ordering::Relaxed),
            tokens_prefilled: self.tokens_prefilled.load(Ordering::Relaxed),
            decode_tokens_per_sec: decode_tps,
            prefill_tokens_per_sec: prefill_tps,
            kv_cache_hit_rate: self.kv_cache_hit_rate(),
            requests_started: self.requests_started.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.tokens_generated.store(0, Ordering::Relaxed);
        self.tokens_prefilled.store(0, Ordering::Relaxed);
        self.kv_cache_hits.store(0, Ordering::Relaxed);
        self.kv_cache_misses.store(0, Ordering::Relaxed);
        self.decode_nanos.store(0, Ordering::Relaxed);
        self.prefill_nanos.store(0, Ordering::Relaxed);
        self.requests_started.store(0, Ordering::Relaxed);
        self.requests_completed.store(0, Ordering::Relaxed);
    }
}

/// A point-in-time clone of [`EngineMetrics`] counters.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total tokens produced in the decode phase.
    pub tokens_generated: u64,
    /// Total tokens processed in the prefill phase.
    pub tokens_prefilled: u64,
    /// Decode throughput in tokens per second.
    pub decode_tokens_per_sec: f64,
    /// Prefill throughput in tokens per second.
    pub prefill_tokens_per_sec: f64,
    /// KV cache hit rate `[0.0, 1.0]`.
    pub kv_cache_hit_rate: f64,
    /// Number of requests started.
    pub requests_started: u64,
    /// Number of requests completed.
    pub requests_completed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_decode_token_increments() {
        let m = EngineMetrics::new();
        m.record_decode_token(Duration::from_secs(1));
        m.record_decode_token(Duration::from_secs(1));
        assert_eq!(m.tokens_generated.load(Ordering::Relaxed), 2);
        assert_eq!(m.decode_nanos.load(Ordering::Relaxed), 2_000_000_000);
    }

    #[test]
    fn test_throughput_decode() {
        let m = EngineMetrics::new();
        // 10 tokens in 1 second → 10 tok/s
        for _ in 0..10 {
            m.record_decode_token(Duration::from_millis(100));
        }
        let (decode_tps, prefill_tps) = m.throughput();
        assert!((decode_tps - 10.0).abs() < 0.1, "decode_tps={decode_tps}");
        assert_eq!(prefill_tps, 0.0);
    }

    #[test]
    fn test_snapshot_fields() {
        let m = EngineMetrics::new();
        m.record_prefill(5, Duration::from_millis(50));
        m.record_kv_hit();
        m.record_kv_miss();
        m.record_request_start();
        m.record_request_complete();

        let snap = m.snapshot();
        assert_eq!(snap.tokens_prefilled, 5);
        assert_eq!(snap.requests_started, 1);
        assert_eq!(snap.requests_completed, 1);
        assert!((snap.kv_cache_hit_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reset_clears_all() {
        let m = EngineMetrics::new();
        m.record_decode_token(Duration::from_millis(10));
        m.record_prefill(3, Duration::from_millis(5));
        m.record_kv_hit();
        m.reset();
        let snap = m.snapshot();
        assert_eq!(snap.tokens_generated, 0);
        assert_eq!(snap.tokens_prefilled, 0);
        assert_eq!(snap.decode_tokens_per_sec, 0.0);
        assert_eq!(snap.kv_cache_hit_rate, 0.0);
    }

    #[test]
    fn test_kv_cache_hit_rate_zero_when_no_lookups() {
        let m = EngineMetrics::new();
        assert_eq!(m.kv_cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_kv_cache_hit_rate_all_hits() {
        let m = EngineMetrics::new();
        m.record_kv_hit();
        m.record_kv_hit();
        assert!((m.kv_cache_hit_rate() - 1.0).abs() < 1e-9);
    }
}

//! Pure-Rust Prometheus-compatible metrics using lock-free atomics.
//!
//! No external prometheus crate is needed; all counters and gauges are
//! backed by `AtomicU64`.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free metrics store for the OxiLLaMa server.
///
/// All fields are atomic counters / gauges that can be safely shared
/// across request handlers without a mutex.
pub struct Metrics {
    // ── counters ────────────────────────────────────────────────
    /// Per-(endpoint, status) request counts.
    /// Flattened as pairs: (endpoint_idx, status_idx) → counter.
    request_counts: Vec<CounterEntry>,

    /// Total tokens generated across all requests.
    pub tokens_generated_total: AtomicU64,

    /// Total prompt tokens received across all requests.
    pub prompt_tokens_total: AtomicU64,

    // ── gauges ──────────────────────────────────────────────────
    /// Number of requests currently in flight.
    pub active_requests: AtomicU64,

    /// Current depth of the inference queue.
    pub queue_depth: AtomicU64,
}

/// Internal storage for a labelled counter.
struct CounterEntry {
    endpoint: &'static str,
    status: &'static str,
    value: AtomicU64,
}

/// Well-known endpoint labels.
const ENDPOINTS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
    "/health",
    "/metrics",
    "other",
];

/// Well-known status labels.
const STATUSES: &[&str] = &["2xx", "4xx", "5xx"];

impl Metrics {
    /// Create a fresh metrics store with all counters at zero.
    pub fn new() -> Self {
        let mut request_counts = Vec::with_capacity(ENDPOINTS.len() * STATUSES.len());
        for &ep in ENDPOINTS {
            for &st in STATUSES {
                request_counts.push(CounterEntry {
                    endpoint: ep,
                    status: st,
                    value: AtomicU64::new(0),
                });
            }
        }

        Self {
            request_counts,
            tokens_generated_total: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            queue_depth: AtomicU64::new(0),
        }
    }

    /// Increment the request counter for the given endpoint and status bucket.
    pub fn inc_request(&self, endpoint: &str, status_code: u16) {
        let ep = self.match_endpoint(endpoint);
        let st = status_bucket(status_code);
        for entry in &self.request_counts {
            if entry.endpoint == ep && entry.status == st {
                entry.value.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Render all metrics in Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(1024);

        // requests_total
        let _ = writeln!(
            out,
            "# HELP oxillama_requests_total Total HTTP requests by endpoint and status."
        );
        let _ = writeln!(out, "# TYPE oxillama_requests_total counter");
        for entry in &self.request_counts {
            let v = entry.value.load(Ordering::Relaxed);
            if v > 0 {
                let _ = writeln!(
                    out,
                    "oxillama_requests_total{{endpoint=\"{}\",status=\"{}\"}} {}",
                    entry.endpoint, entry.status, v
                );
            }
        }

        // tokens_generated_total
        let _ = writeln!(
            out,
            "# HELP oxillama_tokens_generated_total Total generated tokens."
        );
        let _ = writeln!(out, "# TYPE oxillama_tokens_generated_total counter");
        let _ = writeln!(
            out,
            "oxillama_tokens_generated_total {}",
            self.tokens_generated_total.load(Ordering::Relaxed)
        );

        // prompt_tokens_total
        let _ = writeln!(
            out,
            "# HELP oxillama_prompt_tokens_total Total prompt tokens received."
        );
        let _ = writeln!(out, "# TYPE oxillama_prompt_tokens_total counter");
        let _ = writeln!(
            out,
            "oxillama_prompt_tokens_total {}",
            self.prompt_tokens_total.load(Ordering::Relaxed)
        );

        // active_requests
        let _ = writeln!(
            out,
            "# HELP oxillama_active_requests Currently in-flight requests."
        );
        let _ = writeln!(out, "# TYPE oxillama_active_requests gauge");
        let _ = writeln!(
            out,
            "oxillama_active_requests {}",
            self.active_requests.load(Ordering::Relaxed)
        );

        // queue_depth
        let _ = writeln!(
            out,
            "# HELP oxillama_queue_depth Current inference queue depth."
        );
        let _ = writeln!(out, "# TYPE oxillama_queue_depth gauge");
        let _ = writeln!(
            out,
            "oxillama_queue_depth {}",
            self.queue_depth.load(Ordering::Relaxed)
        );

        out
    }

    /// Map a request path to a known endpoint label.
    fn match_endpoint(&self, path: &str) -> &'static str {
        for &ep in ENDPOINTS {
            if ep == path {
                return ep;
            }
        }
        "other"
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Map an HTTP status code to a bucket label.
fn status_bucket(code: u16) -> &'static str {
    match code {
        200..=299 => "2xx",
        400..=499 => "4xx",
        _ => "5xx",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_metrics_zeroed() {
        let m = Metrics::new();
        assert_eq!(m.tokens_generated_total.load(Ordering::Relaxed), 0);
        assert_eq!(m.prompt_tokens_total.load(Ordering::Relaxed), 0);
        assert_eq!(m.active_requests.load(Ordering::Relaxed), 0);
        assert_eq!(m.queue_depth.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_inc_request_counter() {
        let m = Metrics::new();
        m.inc_request("/health", 200);
        m.inc_request("/health", 200);
        m.inc_request("/health", 500);

        let rendered = m.render();
        assert!(rendered.contains(r#"oxillama_requests_total{endpoint="/health",status="2xx"} 2"#));
        assert!(rendered.contains(r#"oxillama_requests_total{endpoint="/health",status="5xx"} 1"#));
    }

    #[test]
    fn test_render_contains_all_sections() {
        let m = Metrics::new();
        m.tokens_generated_total.store(42, Ordering::Relaxed);
        m.prompt_tokens_total.store(100, Ordering::Relaxed);
        m.active_requests.store(3, Ordering::Relaxed);
        m.queue_depth.store(7, Ordering::Relaxed);

        let rendered = m.render();
        assert!(rendered.contains("oxillama_tokens_generated_total 42"));
        assert!(rendered.contains("oxillama_prompt_tokens_total 100"));
        assert!(rendered.contains("oxillama_active_requests 3"));
        assert!(rendered.contains("oxillama_queue_depth 7"));
    }

    #[test]
    fn test_unknown_endpoint_goes_to_other() {
        let m = Metrics::new();
        m.inc_request("/unknown/path", 200);

        let rendered = m.render();
        assert!(rendered.contains(r#"endpoint="other"#));
    }

    #[test]
    fn test_status_buckets() {
        assert_eq!(status_bucket(200), "2xx");
        assert_eq!(status_bucket(201), "2xx");
        assert_eq!(status_bucket(400), "4xx");
        assert_eq!(status_bucket(404), "4xx");
        assert_eq!(status_bucket(500), "5xx");
        assert_eq!(status_bucket(503), "5xx");
    }

    #[test]
    fn test_concurrent_increment() {
        use std::sync::Arc;
        let m = Arc::new(Metrics::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let m2 = Arc::clone(&m);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        m2.inc_request("/health", 200);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread should not panic");
        }
        let rendered = m.render();
        assert!(
            rendered.contains(r#"oxillama_requests_total{endpoint="/health",status="2xx"} 800"#)
        );
    }
}

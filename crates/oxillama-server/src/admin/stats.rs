//! Admin API statistics snapshot.

/// A snapshot of server-wide metrics for the admin stats endpoint.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AdminStats {
    /// Total requests received (across all endpoints).
    pub requests_total: u64,
    /// Total generated tokens across all requests.
    pub tokens_generated_total: u64,
    /// Total prompt tokens received.
    pub prompt_tokens_total: u64,
    /// Number of currently in-flight requests.
    pub active_requests: u64,
    /// Current depth of the inference queue.
    pub queue_depth: u64,
}

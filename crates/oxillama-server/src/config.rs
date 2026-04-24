//! Server configuration.

use serde::{Deserialize, Serialize};

/// Configuration for the OxiLLaMa API server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address to bind to.
    pub host: String,
    /// Port number.
    pub port: u16,
    /// Maximum concurrent requests.
    pub max_concurrent: usize,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Enable CORS headers.
    pub cors_enabled: bool,
    /// API keys for authentication (empty = no auth).
    pub api_keys: Vec<String>,
    /// Rate limit: maximum burst capacity (0.0 = no limit).
    pub rate_limit_capacity: f64,
    /// Rate limit: tokens per second refill rate.
    pub rate_limit_rate: f64,
    /// Maximum request body size in bytes (0 = no limit).
    pub body_limit_bytes: usize,
    /// Enable the /metrics Prometheus endpoint.
    pub metrics_enabled: bool,
    /// Enable structured request tracing middleware.
    pub structured_tracing: bool,

    // ── Router (multi-model pool) ─────────────────────────────────────────
    /// Maximum number of concurrently loaded models (0 = 1, single-model mode).
    pub router_capacity: usize,
    /// Memory budget for the model pool in MiB (0 = unlimited).
    pub router_mem_budget_mb: usize,
    /// Model IDs to pre-load at startup.
    pub router_preload: Vec<String>,

    // ── Admin API ─────────────────────────────────────────────────────────
    /// Bearer token required for all `/admin/*` routes.
    ///
    /// `None` = token-less mode (admin only accessible from loopback).
    pub admin_bearer_token: Option<String>,
    /// Address the admin interface is expected to listen on.
    /// Used for the startup safety check: non-loopback + no token → fatal error.
    pub admin_listen: String,

    // ── Batch disk spool ──────────────────────────────────────────────────
    /// Directory for disk-spooled batch jobs.
    /// Defaults to `$TMPDIR/oxillama_batch_spool`.
    pub batch_spool_dir: Option<String>,
    /// Maximum pending bytes across all queued batch jobs.
    pub batch_max_pending_bytes: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_concurrent: 64,
            timeout_secs: 300,
            cors_enabled: true,
            api_keys: Vec::new(),
            rate_limit_capacity: 0.0,
            rate_limit_rate: 10.0,
            body_limit_bytes: 10 * 1024 * 1024,
            metrics_enabled: true,
            structured_tracing: true,

            router_capacity: 1,
            router_mem_budget_mb: 0,
            router_preload: Vec::new(),

            admin_bearer_token: None,
            admin_listen: "127.0.0.1:8081".to_string(),

            batch_spool_dir: None,
            batch_max_pending_bytes: 1024 * 1024 * 1024, // 1 GiB
        }
    }
}

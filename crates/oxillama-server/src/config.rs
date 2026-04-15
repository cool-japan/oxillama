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
        }
    }
}

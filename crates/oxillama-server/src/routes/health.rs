//! GET /health handler.

use axum::Json;
use serde::Serialize;

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Server status.
    pub status: String,
    /// Server version.
    pub version: String,
}

/// Health check endpoint.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[cfg(test)]
mod tests {
    use crate::test_helpers::{build_test_app, get};

    /// Health endpoint always returns HTTP 200 with `{"status": "ok"}`,
    /// regardless of whether the inference worker is alive.
    #[tokio::test]
    async fn test_health_returns_200_with_ok_status() {
        let app = build_test_app().await;
        let (status, body) = get(app, "/health").await;
        assert_eq!(status.as_u16(), 200);
        assert_eq!(body["status"], "ok");
    }

    /// The version field should be present and non-empty.
    #[tokio::test]
    async fn test_health_includes_version() {
        let app = build_test_app().await;
        let (status, body) = get(app, "/health").await;
        assert_eq!(status.as_u16(), 200);
        let version = body["version"].as_str().expect("version must be a string");
        assert!(!version.is_empty(), "version string must not be empty");
    }
}

//! GET /health (liveness) and GET /ready (readiness) handlers.
//!
//! These map to the standard Kubernetes liveness/readiness distinction:
//! - `/health` — is the process alive at all? Always 200 as long as the
//!   HTTP server is answering requests, deliberately independent of
//!   whether the inference worker is currently able to serve traffic
//!   (used to decide whether to restart the container).
//! - `/ready` — can this instance actually serve inference requests right
//!   now? Reports 503 when the worker has died (D7's `worker_alive` flag),
//!   so a load balancer / orchestrator can stop routing traffic here
//!   without killing and restarting the process (D8 fix — previously
//!   there was no readiness endpoint at all, so `/health`'s unconditional
//!   200 was the only signal available, which could not distinguish "up"
//!   from "up but the sole inference worker silently died").

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

use crate::state::AppState;

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Server status.
    pub status: String,
    /// Server version.
    pub version: String,
}

/// Liveness check endpoint — always 200 while the process is up.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Readiness check response.
#[derive(Debug, Serialize)]
pub struct ReadyResponse {
    /// `"ready"` or `"not_ready"`.
    pub status: String,
    /// Whether the inference worker's main loop is currently running.
    pub worker_alive: bool,
    /// Number of requests currently queued/in-flight ahead of a new one.
    pub queue_depth: usize,
    /// Configured capacity of the inference request queue.
    pub queue_capacity: usize,
}

/// Readiness check endpoint.
///
/// Returns 503 when the inference worker is not alive (D7's
/// `worker_alive` flag), 200 otherwise. Also refreshes
/// `Metrics::queue_depth` as a side effect so `/metrics` reflects current
/// queue pressure without a separate polling task.
pub async fn ready(State(state): State<Arc<AppState>>) -> Response {
    let worker_alive = state.worker_alive.load(Ordering::Acquire);
    let queue_capacity = state.queue.max_capacity();
    let queue_depth = queue_capacity.saturating_sub(state.queue.capacity());

    state
        .metrics
        .queue_depth
        .store(queue_depth as u64, Ordering::Relaxed);

    let body = ReadyResponse {
        status: if worker_alive { "ready" } else { "not_ready" }.to_string(),
        worker_alive,
        queue_depth,
        queue_capacity,
    };

    let status = if worker_alive {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{build_test_app, get};
    use std::sync::atomic::AtomicBool;

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

    /// D8 regression: `/ready` returns 200 with `worker_alive: true` when
    /// the worker's liveness flag is set.
    #[tokio::test]
    async fn test_ready_returns_200_when_worker_alive() {
        let app = build_test_app().await;
        let (status, body) = get(app, "/ready").await;
        assert_eq!(
            status.as_u16(),
            200,
            "default test state has worker_alive=true: {body}"
        );
        assert_eq!(body["worker_alive"], true);
        assert_eq!(body["status"], "ready");
    }

    /// D8 regression: `/ready` returns 503 when `worker_alive` is false —
    /// this is exactly the case `/health` alone could never distinguish.
    #[tokio::test]
    async fn test_ready_returns_503_when_worker_dead() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<crate::queue::BatchRequest>(4);
        let state = crate::test_helpers::new_test_state(
            tx,
            "test-model",
            oxillama_runtime::sampling::SamplerConfig::default(),
            None,
            0,
        );
        state.worker_alive.store(false, Ordering::Release);
        let app = crate::app::build_app(std::sync::Arc::new(state));

        let (status, body) = get(app, "/ready").await;
        assert_eq!(
            status.as_u16(),
            503,
            "worker_alive=false must yield 503: {body}"
        );
        assert_eq!(body["worker_alive"], false);
        assert_eq!(body["status"], "not_ready");
    }

    /// `/ready` reports queue depth/capacity fields as numbers.
    #[tokio::test]
    async fn test_ready_reports_queue_depth() {
        let app = build_test_app().await;
        let (status, body) = get(app, "/ready").await;
        assert_eq!(status.as_u16(), 200);
        assert!(body["queue_capacity"].as_u64().is_some());
        assert!(body["queue_depth"].as_u64().is_some());
    }

    /// Sanity: constructing a fresh `AtomicBool` directly (not through
    /// `AppState`) still behaves as expected — guards against a
    /// regression where `Ordering` imports get shadowed incorrectly.
    #[test]
    fn worker_alive_atomic_bool_basic_semantics() {
        let flag = AtomicBool::new(true);
        assert!(flag.load(Ordering::Acquire));
        flag.store(false, Ordering::Release);
        assert!(!flag.load(Ordering::Acquire));
    }
}

//! Structured tracing middleware.
//!
//! Logs each HTTP request with structured JSON-friendly fields:
//! method, path, status, latency_ms, and a unique request-id.
//!
//! This is also where `/metrics` actually gets its data from (D14 fix):
//! previously no code path ever called `Metrics::inc_request`, so
//! `/metrics` reported zero requests forever regardless of traffic. This
//! middleware increments `active_requests` on entry (via an RAII guard so
//! a panic further down the stack still decrements it) and calls
//! `inc_request` once the response status is known.

use axum::{
    extract::{Request, State},
    http::HeaderValue,
    middleware::Next,
    response::Response,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::metrics::Metrics;

/// Global atomic counter for generating unique request IDs.
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique request ID from an atomic counter and the current
/// timestamp.  Not a true UUID but lightweight and unique within one
/// process lifetime.
fn generate_request_id() -> String {
    let seq = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{ts:x}-{seq:04x}")
}

/// RAII guard that decrements `Metrics::active_requests` on drop.
///
/// Using a guard (rather than an explicit decrement after `next.run`)
/// ensures the counter is still correctly decremented if a panic unwinds
/// through this middleware — `CatchPanicLayer` sits outside this layer in
/// `build_app_with_config`, so a panic here still unwinds through `Drop`
/// before being caught further out.
struct ActiveRequestGuard<'a> {
    metrics: &'a Metrics,
}

impl<'a> ActiveRequestGuard<'a> {
    fn new(metrics: &'a Metrics) -> Self {
        metrics.active_requests.fetch_add(1, Ordering::Relaxed);
        Self { metrics }
    }
}

impl Drop for ActiveRequestGuard<'_> {
    fn drop(&mut self) {
        self.metrics.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Middleware that adds structured request tracing and updates `Metrics`
/// (D14 fix).
///
/// Injects an `x-request-id` response header, logs each request with
/// `tracing::info!`, tracks `active_requests` for the duration of the
/// request, and increments the per-endpoint/status request counter that
/// `/metrics` and `GET /admin/stats` read from.
///
/// Note: `Metrics::queue_depth` is *not* updated here — it is refreshed by
/// `GET /ready` (see `routes::health::ready`), since queue depth is a
/// point-in-time property of the inference queue, not something meaningful
/// to attribute to any single HTTP request's lifecycle.
pub async fn tracing_middleware(
    State(metrics): State<Arc<Metrics>>,
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let request_id = generate_request_id();

    let _guard = ActiveRequestGuard::new(&metrics);

    let start = Instant::now();
    let mut response = next.run(request).await;
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    let status = response.status().as_u16();
    metrics.inc_request(&path, status);

    tracing::info!(
        method = %method,
        path = %path,
        status = status,
        latency_ms = latency_ms,
        request_id = %request_id,
        "request completed"
    );

    // Inject x-request-id header into the response.
    if let Ok(val) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", val);
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    fn traced_app() -> Router {
        let metrics = Arc::new(Metrics::new());
        Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                Arc::clone(&metrics),
                tracing_middleware,
            ))
    }

    #[tokio::test]
    async fn test_request_id_generated() {
        let app = traced_app();
        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .expect("request builder should succeed");

        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        assert!(resp.headers().contains_key("x-request-id"));
        let rid = resp.headers()["x-request-id"]
            .to_str()
            .expect("header should be valid string");
        assert!(!rid.is_empty());
    }

    #[tokio::test]
    async fn test_request_ids_are_unique() {
        let id1 = generate_request_id();
        let id2 = generate_request_id();
        assert_ne!(id1, id2, "consecutive request IDs must differ");
    }

    #[tokio::test]
    async fn test_response_status_preserved() {
        let app = traced_app();
        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .expect("request builder should succeed");

        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_404_still_traced() {
        let app = traced_app();
        let req = HttpRequest::builder()
            .uri("/nonexistent")
            .body(Body::empty())
            .expect("request builder should succeed");

        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        // x-request-id still present
        assert!(resp.headers().contains_key("x-request-id"));
    }

    #[tokio::test]
    async fn test_latency_is_non_negative() {
        // Integration verification — the tracing_middleware records latency.
        // We can't inspect the tracing output easily, but we verify the
        // middleware itself doesn't panic when computing latency on a fast
        // handler.
        let app = traced_app();
        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .expect("request builder should succeed");

        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// D14 regression: the middleware must actually increment
    /// `Metrics::inc_request` — previously nothing did, so `/metrics`
    /// reported zero requests regardless of traffic.
    #[tokio::test]
    async fn test_middleware_increments_request_counter() {
        let metrics = Arc::new(Metrics::new());
        let app = Router::new().route("/test", get(|| async { "ok" })).layer(
            middleware::from_fn_with_state(Arc::clone(&metrics), tracing_middleware),
        );

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .expect("request builder should succeed");
        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        assert_eq!(resp.status(), StatusCode::OK);

        assert_eq!(
            metrics.total_requests(),
            1,
            "the request must have been counted"
        );
    }

    /// D14 regression: `active_requests` returns to zero after the request
    /// completes (proving the RAII guard actually decrements, not just
    /// increments).
    #[tokio::test]
    async fn test_active_requests_returns_to_zero_after_completion() {
        let metrics = Arc::new(Metrics::new());
        let app = Router::new().route("/test", get(|| async { "ok" })).layer(
            middleware::from_fn_with_state(Arc::clone(&metrics), tracing_middleware),
        );

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .expect("request builder should succeed");
        let _resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");

        assert_eq!(
            metrics.active_requests.load(Ordering::Relaxed),
            0,
            "active_requests must return to zero once the request completes"
        );
    }
}

//! Structured tracing middleware.
//!
//! Logs each HTTP request with structured JSON-friendly fields:
//! method, path, status, latency_ms, and a unique request-id.

use axum::{extract::Request, http::HeaderValue, middleware::Next, response::Response};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

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

/// Middleware that adds structured request tracing.
///
/// Injects an `x-request-id` response header and logs each request with
/// `tracing::info!`.
pub async fn tracing_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let request_id = generate_request_id();

    let start = Instant::now();
    let mut response = next.run(request).await;
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    let status = response.status().as_u16();

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
        Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(middleware::from_fn(tracing_middleware))
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
}

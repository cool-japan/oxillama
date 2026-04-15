//! GET /metrics handler — Prometheus text exposition format.

use std::sync::Arc;

use axum::http::header;
use axum::response::{IntoResponse, Response};

use crate::metrics::Metrics;

/// Serve Prometheus-formatted metrics.
pub async fn metrics(metrics: Option<axum::extract::Extension<Arc<Metrics>>>) -> Response {
    let body = match metrics {
        Some(axum::extract::Extension(m)) => m.render(),
        None => String::new(),
    };

    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    fn metrics_app(m: Arc<Metrics>) -> Router {
        Router::new()
            .route("/metrics", get(metrics))
            .layer(axum::Extension(m))
    }

    #[tokio::test]
    async fn test_metrics_returns_200() {
        let m = Arc::new(Metrics::new());
        let app = metrics_app(m);

        let req = HttpRequest::builder()
            .uri("/metrics")
            .body(Body::empty())
            .expect("request builder should succeed");
        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_content_type() {
        let m = Arc::new(Metrics::new());
        let app = metrics_app(m);

        let req = HttpRequest::builder()
            .uri("/metrics")
            .body(Body::empty())
            .expect("request builder should succeed");
        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .expect("content-type header must exist");
        assert!(ct.to_str().unwrap_or("").contains("text/plain"));
    }

    #[tokio::test]
    async fn test_metrics_body_contains_counters() {
        let m = Arc::new(Metrics::new());
        m.tokens_generated_total
            .store(10, std::sync::atomic::Ordering::Relaxed);
        let app = metrics_app(m);

        let req = HttpRequest::builder()
            .uri("/metrics")
            .body(Body::empty())
            .expect("request builder should succeed");
        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("body should be readable");
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("oxillama_tokens_generated_total 10"));
    }

    #[tokio::test]
    async fn test_metrics_without_extension() {
        // When no Metrics extension is provided, endpoint still returns 200
        let app = Router::new().route("/metrics", get(metrics));

        let req = HttpRequest::builder()
            .uri("/metrics")
            .body(Body::empty())
            .expect("request builder should succeed");
        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_request_counter_reflects() {
        let m = Arc::new(Metrics::new());
        m.inc_request("/health", 200);
        m.inc_request("/health", 200);
        m.inc_request("/health", 500);

        let app = metrics_app(m);
        let req = HttpRequest::builder()
            .uri("/metrics")
            .body(Body::empty())
            .expect("request builder should succeed");
        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("body should be readable");
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains(r#"oxillama_requests_total{endpoint="/health",status="2xx"} 2"#));
    }
}

//! Request body-size limit configuration.
//!
//! Wraps `tower_http::limit::RequestBodyLimitLayer` with a configurable
//! maximum body size (default 10 MiB).

use tower_http::limit::RequestBodyLimitLayer;

/// Default body limit: 10 MiB.
pub const DEFAULT_BODY_LIMIT: usize = 10 * 1024 * 1024;

/// Create a [`RequestBodyLimitLayer`] with the given byte limit.
pub fn body_limit_layer(max_bytes: usize) -> RequestBodyLimitLayer {
    RequestBodyLimitLayer::new(max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;

    async fn echo_handler(body: axum::body::Bytes) -> String {
        format!("received {} bytes", body.len())
    }

    fn app_with_limit(limit: usize) -> Router {
        Router::new()
            .route("/upload", post(echo_handler))
            .layer(body_limit_layer(limit))
    }

    #[tokio::test]
    async fn test_default_limit_constant() {
        assert_eq!(DEFAULT_BODY_LIMIT, 10 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_small_body_passes() {
        let app = app_with_limit(1024);
        let body = vec![0u8; 512];
        let req = HttpRequest::builder()
            .method("POST")
            .uri("/upload")
            .body(Body::from(body))
            .expect("request builder should succeed");

        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_exact_limit_passes() {
        let app = app_with_limit(256);
        let body = vec![0u8; 256];
        let req = HttpRequest::builder()
            .method("POST")
            .uri("/upload")
            .body(Body::from(body))
            .expect("request builder should succeed");

        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_oversized_request_rejected() {
        let app = app_with_limit(128);
        let body = vec![0u8; 256];
        let req = HttpRequest::builder()
            .method("POST")
            .uri("/upload")
            .body(Body::from(body))
            .expect("request builder should succeed");

        let resp = app
            .oneshot(req)
            .await
            .expect("router should handle request");
        // tower-http returns 413 Payload Too Large
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn test_custom_limit_value() {
        // Verify layer creation with various sizes doesn't panic.
        let _layer = body_limit_layer(1);
        let _layer = body_limit_layer(1024 * 1024 * 100);
        let _layer = body_limit_layer(DEFAULT_BODY_LIMIT);
    }
}

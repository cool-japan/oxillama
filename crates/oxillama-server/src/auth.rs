//! Bearer-token authentication middleware.
//!
//! When API keys are configured in `ServerConfig`, every request must
//! carry a valid `Authorization: Bearer <key>` header. Unauthenticated
//! requests receive a 401 with an OpenAI-shaped error body.

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;

/// Shared set of valid API keys.
#[derive(Clone)]
pub struct ApiKeys(pub Arc<Vec<String>>);

/// Middleware function that validates Bearer tokens.
///
/// If `ApiKeys` is empty (no keys configured), all requests pass through.
/// Otherwise the `Authorization: Bearer <key>` header must match one of
/// the configured keys.
pub async fn auth_middleware(
    keys: Option<axum::extract::Extension<ApiKeys>>,
    request: Request,
    next: Next,
) -> Response {
    // If no keys configured, pass through
    let Some(axum::extract::Extension(api_keys)) = keys else {
        return next.run(request).await;
    };

    if api_keys.0.is_empty() {
        return next.run(request).await;
    }

    // Extract Authorization header
    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header[7..];
            if api_keys.0.iter().any(|k| k == token) {
                next.run(request).await
            } else {
                unauthorized_response("Invalid API key")
            }
        }
        Some(_) => unauthorized_response("Authorization header must use Bearer scheme"),
        None => unauthorized_response("Missing Authorization header"),
    }
}

fn unauthorized_response(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "authentication_error",
        }
    });
    (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request as HttpRequest, middleware, routing::get, Router};
    use tower::ServiceExt;

    async fn ok_handler() -> &'static str {
        "ok"
    }

    fn test_app(keys: Vec<String>) -> Router {
        let api_keys = ApiKeys(Arc::new(keys));
        Router::new()
            .route("/test", get(ok_handler))
            .layer(middleware::from_fn(auth_middleware))
            .layer(axum::Extension(api_keys.clone()))
    }

    #[tokio::test]
    async fn test_no_keys_passes_through() {
        let app = Router::new()
            .route("/test", get(ok_handler))
            .layer(middleware::from_fn(auth_middleware))
            .layer(axum::Extension(ApiKeys(Arc::new(vec![]))));

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_valid_key_passes() {
        let app = test_app(vec!["sk-test-key".to_string()]);
        let req = HttpRequest::builder()
            .uri("/test")
            .header("Authorization", "Bearer sk-test-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_invalid_key_returns_401() {
        let app = test_app(vec!["sk-test-key".to_string()]);
        let req = HttpRequest::builder()
            .uri("/test")
            .header("Authorization", "Bearer wrong-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_missing_header_returns_401() {
        let app = test_app(vec!["sk-test-key".to_string()]);
        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_non_bearer_scheme_returns_401() {
        let app = test_app(vec!["sk-test-key".to_string()]);
        let req = HttpRequest::builder()
            .uri("/test")
            .header("Authorization", "Basic dXNlcjpwYXNz")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}

//! Application builder — constructs the axum router with all routes.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use crate::auth::{auth_middleware, ApiKeys};
use crate::body_limit::body_limit_layer;
use crate::config::ServerConfig;
use crate::rate_limit::{rate_limit_middleware, RateLimiter};
use crate::routes;
use crate::state::AppState;
use crate::tracing_layer::tracing_middleware;

/// Build the OxiLLaMa API server router with shared state.
///
/// Mounts all OpenAI-compatible endpoints:
/// - `POST /v1/chat/completions`
/// - `POST /v1/completions`
/// - `POST /v1/embeddings`
/// - `GET /v1/models`
/// - `GET /health`
pub fn build_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/completions", post(routes::completions::completions))
        .route("/v1/embeddings", post(routes::embeddings::embeddings))
        .route("/v1/models", get(routes::models::list_models))
        .route("/health", get(routes::health::health))
        .with_state(state)
}

/// Build the app with full config (auth, rate limiting, CORS, body limit,
/// metrics, tracing).
pub fn build_app_with_config(state: Arc<AppState>, config: &ServerConfig) -> Router {
    let metrics = Arc::clone(&state.metrics);

    let mut app = Router::new()
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/completions", post(routes::completions::completions))
        .route("/v1/embeddings", post(routes::embeddings::embeddings))
        .route("/v1/models", get(routes::models::list_models))
        .route("/health", get(routes::health::health));

    // Metrics endpoint
    if config.metrics_enabled {
        app = app.route("/metrics", get(routes::metrics::metrics));
    }

    let mut app = app.with_state(state);

    // Metrics extension (shared with the metrics route handler)
    if config.metrics_enabled {
        app = app.layer(axum::Extension(metrics));
    }

    // Structured tracing middleware
    if config.structured_tracing {
        app = app.layer(axum::middleware::from_fn(tracing_middleware));
    }

    // Rate limiting layer
    if config.rate_limit_capacity > 0.0 {
        let limiter = RateLimiter::new(config.rate_limit_capacity, config.rate_limit_rate);
        app = app
            .layer(axum::middleware::from_fn(rate_limit_middleware))
            .layer(axum::Extension(limiter));
    }

    // Auth layer
    if !config.api_keys.is_empty() {
        let keys = ApiKeys(Arc::new(config.api_keys.clone()));
        app = app
            .layer(axum::middleware::from_fn(auth_middleware))
            .layer(axum::Extension(keys));
    }

    // Body-size limit (outermost layer)
    if config.body_limit_bytes > 0 {
        app = app.layer(body_limit_layer(config.body_limit_bytes));
    }

    app
}

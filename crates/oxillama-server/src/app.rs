//! Application builder — constructs the axum router with all routes.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use crate::admin;
use crate::auth::{auth_middleware, ApiKeys};
use crate::batch;
use crate::batch_spool;
use crate::body_limit::body_limit_layer;
use crate::config::ServerConfig;
use crate::rate_limit::{rate_limit_middleware, RateLimiter};
use crate::routes;
use crate::state::AppState;
use crate::tracing_layer::tracing_middleware;
use crate::ws::ws_handler;

/// Build the OxiLLaMa API server router with shared state.
///
/// Mounts all OpenAI-compatible endpoints and the admin API.
pub fn build_app(state: Arc<AppState>) -> Router {
    Router::new()
        // ── OpenAI inference ──────────────────────────────────────────────
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/completions", post(routes::completions::completions))
        .route("/v1/embeddings", post(routes::embeddings::embeddings))
        .route("/v1/models", get(routes::models::list_models))
        .route("/health", get(routes::health::health))
        .route("/v1/chat/ws", get(ws_handler))
        // ── Legacy in-memory batch API ────────────────────────────────────
        .route(
            "/v1/batches",
            post(batch::create_batch).get(batch::list_batches),
        )
        .route("/v1/batches/{id}", get(batch::get_batch))
        .route("/v1/batches/{id}/cancel", post(batch::cancel_batch))
        // ── Disk-spooled batch API ────────────────────────────────────────
        .route(
            "/v1/batch_jobs",
            post(batch_spool::routes::create_batch).get(batch_spool::routes::list_batches),
        )
        .route("/v1/batch_jobs/{id}", get(batch_spool::routes::get_batch))
        .route(
            "/v1/batch_jobs/{id}/output",
            get(batch_spool::routes::get_batch_output),
        )
        .route(
            "/v1/batch_jobs/{id}/cancel",
            post(batch_spool::routes::cancel_batch),
        )
        // ── Admin API ─────────────────────────────────────────────────────
        .route("/admin/models/load", post(admin::admin_load_model))
        .route("/admin/models/unload", post(admin::admin_unload_model))
        .route("/admin/models", get(admin::admin_list_models))
        .route("/admin/stats", get(admin::admin_stats))
        .route("/admin/health", get(admin::admin_health))
        .with_state(state)
}

/// Build the app with full config (auth, rate limiting, CORS, body limit,
/// metrics, tracing).
pub fn build_app_with_config(state: Arc<AppState>, config: &ServerConfig) -> Router {
    let metrics = Arc::clone(&state.metrics);

    // Admin auth config from server config.
    let admin_auth = crate::admin::AdminAuth {
        token: config.admin_bearer_token.clone(),
    };

    let mut app = Router::new()
        // ── OpenAI inference ──────────────────────────────────────────────
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/completions", post(routes::completions::completions))
        .route("/v1/embeddings", post(routes::embeddings::embeddings))
        .route("/v1/models", get(routes::models::list_models))
        .route("/health", get(routes::health::health))
        .route("/v1/chat/ws", get(ws_handler))
        // ── Legacy in-memory batch API ────────────────────────────────────
        .route(
            "/v1/batches",
            post(batch::create_batch).get(batch::list_batches),
        )
        .route("/v1/batches/{id}", get(batch::get_batch))
        .route("/v1/batches/{id}/cancel", post(batch::cancel_batch))
        // ── Disk-spooled batch API ────────────────────────────────────────
        .route(
            "/v1/batch_jobs",
            post(batch_spool::routes::create_batch).get(batch_spool::routes::list_batches),
        )
        .route("/v1/batch_jobs/{id}", get(batch_spool::routes::get_batch))
        .route(
            "/v1/batch_jobs/{id}/output",
            get(batch_spool::routes::get_batch_output),
        )
        .route(
            "/v1/batch_jobs/{id}/cancel",
            post(batch_spool::routes::cancel_batch),
        )
        // ── Admin API ─────────────────────────────────────────────────────
        .route("/admin/models/load", post(admin::admin_load_model))
        .route("/admin/models/unload", post(admin::admin_unload_model))
        .route("/admin/models", get(admin::admin_list_models))
        .route("/admin/stats", get(admin::admin_stats))
        .route("/admin/health", get(admin::admin_health));

    // Metrics endpoint
    if config.metrics_enabled {
        app = app.route("/metrics", get(routes::metrics::metrics));
    }

    let mut app = app.with_state(state);

    // Admin auth extension.
    app = app.layer(axum::Extension(admin_auth));

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

    // Auth layer — JWT takes priority over static bearer keys.
    #[cfg(feature = "jwt")]
    if let Some(jwt_config) = config.jwt.clone() {
        use crate::jwt_auth::{jwt_auth_middleware, JwtVerifier};
        let verifier = Arc::new(JwtVerifier::new(jwt_config));
        app = app.layer(axum::middleware::from_fn_with_state(
            verifier,
            jwt_auth_middleware,
        ));
    } else if !config.api_keys.is_empty() {
        let keys = ApiKeys(Arc::new(config.api_keys.clone()));
        app = app
            .layer(axum::middleware::from_fn(auth_middleware))
            .layer(axum::Extension(keys));
    }

    #[cfg(not(feature = "jwt"))]
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

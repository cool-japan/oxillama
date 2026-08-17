//! Application builder — constructs the axum router with all routes.

use std::sync::Arc;
use std::time::Duration;

use axum::middleware::{from_fn, from_fn_with_state};
use axum::routing::{get, post};
use axum::Extension;
use axum::Router;

use crate::admin;
use crate::auth::{auth_middleware, ApiKeys};
use crate::batch;
use crate::batch_spool;
use crate::body_limit::body_limit_layer;
use crate::config::ServerConfig;
use crate::error::ServerResult;
use crate::rate_limit::{per_key_rate_limit_middleware, rate_limit_middleware, RateLimiter};
use crate::routes;
use crate::state::AppState;
use crate::threads;
use crate::tracing_layer::tracing_middleware;
use crate::ws::ws_handler;

/// All non-admin, OpenAI-compatible + operational routes.
///
/// Shared by both [`build_app`] and [`build_app_with_config`] so the route
/// table itself cannot drift between the test-only builder and the
/// production one — only the *middleware stack* around it differs.
fn inference_routes() -> Router<Arc<AppState>> {
    Router::new()
        // ── OpenAI inference ──────────────────────────────────────────────
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/completions", post(routes::completions::completions))
        .route("/v1/embeddings", post(routes::embeddings::embeddings))
        .route("/v1/models", get(routes::models::list_models))
        // ── Liveness / readiness (D8) ───────────────────────────────────────
        .route("/health", get(routes::health::health))
        .route("/ready", get(routes::health::ready))
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
        // ── Files API ─────────────────────────────────────────────────────
        .route(
            "/v1/files",
            post(routes::files::create_file_handler).get(routes::files::list_files_handler),
        )
        .route(
            "/v1/files/{file_id}",
            get(routes::files::get_file_handler).delete(routes::files::delete_file_handler),
        )
        .route(
            "/v1/files/{file_id}/content",
            get(routes::files::get_file_content_handler),
        )
        // ── Assistants API ────────────────────────────────────────────────
        .route("/v1/threads", post(threads::routes::create_thread_handler))
        .route(
            "/v1/threads/{thread_id}",
            get(threads::routes::get_thread_handler),
        )
        .route(
            "/v1/threads/{thread_id}/messages",
            post(threads::routes::create_message_handler)
                .get(threads::routes::list_messages_handler),
        )
        .route(
            "/v1/threads/{thread_id}/runs",
            post(threads::routes::create_run_handler),
        )
        .route(
            "/v1/threads/{thread_id}/runs/{run_id}",
            get(threads::routes::get_run_handler),
        )
        .route(
            "/v1/threads/{thread_id}/runs/{run_id}/cancel",
            post(threads::routes::cancel_run_handler),
        )
        // ── Run Steps subresource ─────────────────────────────────────────
        .route(
            "/v1/threads/{thread_id}/runs/{run_id}/steps",
            get(threads::steps::list_steps_handler),
        )
        .route(
            "/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}",
            get(threads::steps::get_step_handler),
        )
        // ── Responses API ─────────────────────────────────────────────────
        .route(
            "/v1/responses",
            post(routes::responses::create_response).get(routes::responses::list_responses),
        )
        .route("/v1/responses/{id}", get(routes::responses::get_response))
}

/// The `/admin/*` route table, with NO authentication layer applied.
///
/// Callers must wrap this with [`authenticated_admin_routes`] before
/// merging it into a served router — mounting this directly (as the
/// pre-fix `build_app`/`build_app_with_config` both did, one via a missing
/// layer call and the other via an `Extension` injected without the
/// corresponding `from_fn` middleware) is exactly D1: a completely
/// unauthenticated admin API.
fn admin_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/admin/models/load", post(admin::admin_load_model))
        .route("/admin/models/unload", post(admin::admin_unload_model))
        .route("/admin/models", get(admin::admin_list_models))
        .route("/admin/stats", get(admin::admin_stats))
        .route("/admin/health", get(admin::admin_health))
        // ── LoRA registry ─────────────────────────────────────────────────
        .route(
            "/admin/loras",
            post(admin::admin_register_lora).get(admin::admin_list_loras),
        )
        .route(
            "/admin/loras/{name}",
            axum::routing::delete(admin::admin_unregister_lora),
        )
}

/// [`admin_routes`], with `admin_auth_middleware` actually applied (D1 fix).
///
/// This is the *only* way `/admin/*` should ever be mounted. The
/// middleware and its `Extension(AdminAuth)` are layered directly onto
/// this sub-router (not the whole app), so admin auth can never
/// accidentally apply to — or be bypassed by omission from — the
/// inference routes.
fn authenticated_admin_routes(token: Option<String>) -> Router<Arc<AppState>> {
    admin_routes()
        .layer(from_fn(admin::admin_auth_middleware))
        .layer(Extension(admin::AdminAuth { token }))
}

/// Build the OxiLLaMa API server router with shared state, **for tests
/// only**.
///
/// # Do not use this to serve real traffic
///
/// This applies admin auth (in token-less/loopback-only mode, so it is at
/// least not a silent full bypass) but skips every other config-driven
/// layer: no rate limiting, no body-size limit, no request timeout, no
/// concurrency limit, no CORS, no panic recovery, no structured tracing/
/// metrics wiring, and no API-key/JWT auth on inference routes.
///
/// Production code (the `serve` command in `oxillama-cli`) MUST call
/// [`build_app_with_config`] instead, which is where all of the above
/// (D1/D5/D7/D9/D13/D14) actually get applied. Historically the CLI called
/// *this* function, which is the linchpin defect (D13) that made every one
/// of those other fixes moot in production regardless of whether they
/// were individually correct — the middleware simply was never mounted.
#[doc(hidden)]
pub fn build_app(state: Arc<AppState>) -> Router {
    let admin = authenticated_admin_routes(None);
    inference_routes().merge(admin).with_state(state)
}

/// Build the production app: full route table, D1 admin auth, D5 load
/// shedding, D7 panic recovery, D9-bounded per-key rate limiting, and D14
/// metrics/tracing wiring — everything [`ServerConfig`] describes.
///
/// Returns `Err` if [`admin::ensure_admin_security`] rejects the
/// configured `admin_listen`/`admin_bearer_token` combination (D1/D11):
/// binding the admin API to a non-loopback address without a bearer token
/// is a startup-time configuration error, not something to silently allow
/// or `std::process::exit` out from under the caller.
///
/// ## Layering (innermost → outermost)
///
/// 1. Admin auth, scoped to `/admin/*` only (`authenticated_admin_routes`).
/// 2. Inference routes: per-key rate limiter (if `state.per_key_rate_limiter`
///    is set) → JWT-or-bearer-key auth (if configured).
/// 3. Merge admin routes into the inference router.
/// 4. `/metrics` route, if `config.metrics_enabled`.
/// 5. `.with_state(state)`.
/// 6. `Extension(metrics)`, if `config.metrics_enabled` (read by the
///    `/metrics` handler).
/// 7. [`tower_http::catch_panic::CatchPanicLayer`] — unconditional (D7):
///    a panic anywhere below this point (a handler, admin auth, per-key
///    rate limiting, …) becomes a 500 response instead of killing the
///    connection/task silently.
/// 8. [`tower::limit::ConcurrencyLimitLayer`], if `config.max_concurrent > 0`.
/// 9. [`tower_http::timeout::TimeoutLayer`], if `config.timeout_secs > 0`
///    (measured from before a request waits for a concurrency slot).
/// 10. [`tower_http::cors::CorsLayer::permissive`], if `config.cors_enabled`.
/// 11. Request body size limit, if `config.body_limit_bytes > 0`.
/// 12. Global token-bucket rate limiter, if `config.rate_limit_capacity > 0.0`.
/// 13. `tracing_middleware` (outermost), if `config.structured_tracing` —
///     must be outermost so it observes the *final* status code of every
///     response, including ones rejected by an inner layer (429 from rate
///     limiting, 413 from the body limit, 504 from the timeout, 500 from
///     a caught panic).
pub fn build_app_with_config(state: Arc<AppState>, config: &ServerConfig) -> ServerResult<Router> {
    admin::ensure_admin_security(&config.admin_listen, &config.admin_bearer_token)?;

    let metrics = Arc::clone(&state.metrics);

    // ── Route table ─────────────────────────────────────────────────────
    let mut inference = inference_routes();

    // Per-API-key rate limiter (applied before auth so a request that will
    // be rejected by auth anyway doesn't also need to pass rate limiting
    // first — order here only affects which 4xx a caller sees).
    if let Some(per_key_limiter) = state.per_key_rate_limiter.as_ref().cloned() {
        inference = inference.layer(from_fn_with_state(
            per_key_limiter,
            per_key_rate_limit_middleware,
        ));
    }

    // Auth layer — JWT takes priority over static bearer keys.
    #[cfg(feature = "jwt")]
    {
        if let Some(jwt_config) = config.jwt.clone() {
            use crate::jwt_auth::{jwt_auth_middleware, JwtVerifier};
            let verifier = Arc::new(JwtVerifier::new(jwt_config));
            inference = inference.layer(from_fn_with_state(verifier, jwt_auth_middleware));
        } else if !config.api_keys.is_empty() {
            let keys = ApiKeys(Arc::new(config.api_keys.clone()));
            inference = inference
                .layer(from_fn(auth_middleware))
                .layer(Extension(keys));
        }
    }
    #[cfg(not(feature = "jwt"))]
    {
        if !config.api_keys.is_empty() {
            let keys = ApiKeys(Arc::new(config.api_keys.clone()));
            inference = inference
                .layer(from_fn(auth_middleware))
                .layer(Extension(keys));
        }
    }

    let admin = authenticated_admin_routes(config.admin_bearer_token.clone());
    let mut app = inference.merge(admin);

    if config.metrics_enabled {
        app = app.route("/metrics", get(routes::metrics::metrics));
    }

    let mut app = app.with_state(Arc::clone(&state));

    if config.metrics_enabled {
        app = app.layer(Extension(Arc::clone(&metrics)));
    }

    // D7: catch panics anywhere in the stack below this layer and convert
    // them into a 500 response rather than letting them propagate as a
    // silently-dead connection.
    app = app.layer(tower_http::catch_panic::CatchPanicLayer::new());

    // D5: bound the number of requests handled concurrently.
    if config.max_concurrent > 0 {
        app = app.layer(tower::limit::ConcurrencyLimitLayer::new(
            config.max_concurrent,
        ));
    }

    // D5: bound how long a request may take end-to-end.
    if config.timeout_secs > 0 {
        app = app.layer(tower_http::timeout::TimeoutLayer::with_status_code(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            Duration::from_secs(config.timeout_secs),
        ));
    }

    if config.cors_enabled {
        app = app.layer(tower_http::cors::CorsLayer::permissive());
    }

    // D5: bound request body size (outermost of the shedding layers, so
    // an oversized body is rejected before it can influence anything else
    // downstream).
    if config.body_limit_bytes > 0 {
        app = app.layer(body_limit_layer(config.body_limit_bytes));
    }

    // D5: global token-bucket rate limiting.
    if config.rate_limit_capacity > 0.0 {
        let limiter = RateLimiter::new(config.rate_limit_capacity, config.rate_limit_rate);
        app = app
            .layer(from_fn(rate_limit_middleware))
            .layer(Extension(limiter));
    }

    // D14: outermost, so it observes the final status of every response
    // including ones short-circuited by an inner layer.
    if config.structured_tracing {
        app = app.layer(from_fn_with_state(metrics, tracing_middleware));
    }

    Ok(app)
}

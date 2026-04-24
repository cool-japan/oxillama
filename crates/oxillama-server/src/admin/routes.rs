//! Admin API route handlers.
//!
//! Endpoints (all mounted under `/admin`):
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | `/admin/models/load` | Background-load a model into the pool |
//! | POST | `/admin/models/unload` | Unload a model from the pool |
//! | GET | `/admin/models` | List pool contents |
//! | GET | `/admin/stats` | Server-wide request metrics |
//! | GET | `/admin/health` | Extended health check with pool readiness |
//!
//! Background load: `POST /admin/models/load` returns `202 Accepted` immediately
//! and spawns a `tokio::task::spawn_blocking` task that does the actual
//! `InferenceEngine::load_model()`.  The caller can poll `GET /admin/models`
//! and check for `status: "ready" | "loading" | "failed"`.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::admin::stats::AdminStats;
use crate::router::{ModelLoadStatus, ModelSpec};
use crate::state::AppState;

// ── Request / response types ─────────────────────────────────────────────────

/// Body for `POST /admin/models/load`.
#[derive(Debug, Deserialize)]
pub struct LoadModelBody {
    /// Stable model identifier (used in inference requests as `model` field).
    pub id: String,
    /// Filesystem path to the `.gguf` model file.
    pub path: String,
    /// Optional quantisation hint (informational).
    #[serde(default)]
    pub quant: Option<String>,
}

/// Body for `POST /admin/models/unload`.
#[derive(Debug, Deserialize)]
pub struct UnloadModelBody {
    /// Model identifier to unload.
    pub id: String,
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `POST /admin/models/load` — request a background model load.
///
/// Returns `202 Accepted` immediately. The actual load happens on a
/// `spawn_blocking` thread. Poll `GET /admin/models` for status.
pub async fn admin_load_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<LoadModelBody>,
) -> Response {
    let model_id = body.id.clone();
    let batch_id = format!("load_{}", Uuid::new_v4().as_simple());

    // Mark as loading in the pool (so GET /admin/models shows status=loading).
    if let Ok(mut pool) = state.model_pool.lock() {
        pool.mark_loading(model_id.clone());

        // Also register the spec in the loader so acquire() can find it.
        pool.loader_register(
            model_id.clone(),
            ModelSpec {
                path: std::path::PathBuf::from(&body.path),
                quant: body.quant.clone(),
            },
        );
    }

    // Spawn background load.
    let model_id_bg = model_id.clone();
    let path = body.path.clone();
    let state_bg = Arc::clone(&state);

    tokio::task::spawn(async move {
        tracing::info!(model_id = %model_id_bg, path, "background model load started");

        let model_id_bl = model_id_bg.clone();
        let path_bl = path.clone();

        let result = tokio::task::spawn_blocking(move || {
            use oxillama_runtime::engine::{EngineConfig, InferenceEngine};
            let cfg = EngineConfig {
                model_path: path_bl,
                ..EngineConfig::default()
            };
            let mut engine = InferenceEngine::new(cfg);
            engine.load_model()?;
            let mem_bytes = engine
                .model_config()
                .map(|_| 0usize) // real estimate in pool::estimate_mem_bytes
                .unwrap_or(0);
            Ok::<_, oxillama_runtime::RuntimeError>((engine, mem_bytes))
        })
        .await;

        match result {
            Ok(Ok((engine, mem_bytes))) => {
                if let Ok(mut pool) = state_bg.model_pool.lock() {
                    let _ = pool.mark_ready(&model_id_bl, engine, mem_bytes);
                }
                tracing::info!(model_id = %model_id_bl, "background model load succeeded");
            }
            Ok(Err(e)) => {
                if let Ok(mut pool) = state_bg.model_pool.lock() {
                    pool.mark_failed(&model_id_bl, e.to_string());
                }
                tracing::error!(model_id = %model_id_bl, error = %e, "background model load failed");
            }
            Err(e) => {
                if let Ok(mut pool) = state_bg.model_pool.lock() {
                    pool.mark_failed(&model_id_bl, e.to_string());
                }
                tracing::error!(model_id = %model_id_bl, error = %e, "spawn_blocking join error");
            }
        }
    });

    let body = serde_json::json!({
        "batch_id": batch_id,
        "model_id": model_id,
        "status": "loading",
        "message": "Model load initiated. Poll GET /admin/models for status.",
    });
    (StatusCode::ACCEPTED, Json(body)).into_response()
}

/// `POST /admin/models/unload` — synchronously unload a model.
pub async fn admin_unload_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<UnloadModelBody>,
) -> Response {
    let result = state
        .model_pool
        .lock()
        .ok()
        .and_then(|mut pool| pool.unload(&body.id).ok());

    match result {
        Some(()) => {
            let resp = serde_json::json!({ "model_id": body.id, "status": "unloaded" });
            (StatusCode::OK, Json(resp)).into_response()
        }
        None => {
            let err = serde_json::json!({
                "error": {
                    "message": format!("model '{}' is not loaded", body.id),
                    "type": "invalid_request_error",
                }
            });
            (StatusCode::NOT_FOUND, Json(err)).into_response()
        }
    }
}

/// `GET /admin/models` — list models in the pool.
pub async fn admin_list_models(State(state): State<Arc<AppState>>) -> Response {
    let models = state
        .model_pool
        .lock()
        .map(|pool| pool.list())
        .unwrap_or_default();

    let data: Vec<_> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "status": match m.status {
                    ModelLoadStatus::Loading => "loading",
                    ModelLoadStatus::Ready => "ready",
                    ModelLoadStatus::Failed => "failed",
                },
                "mem_bytes": m.mem_bytes,
                "last_used_secs_ago": m.last_used_secs,
                "inflight": m.inflight,
            })
        })
        .collect();

    let body = serde_json::json!({ "object": "list", "models": data });
    (StatusCode::OK, Json(body)).into_response()
}

/// `GET /admin/stats` — server-wide request metrics.
pub async fn admin_stats(State(state): State<Arc<AppState>>) -> Response {
    use std::sync::atomic::Ordering;

    let metrics = &state.metrics;
    let stats = AdminStats {
        requests_total: metrics.active_requests.load(Ordering::Relaxed),
        tokens_generated_total: metrics.tokens_generated_total.load(Ordering::Relaxed),
        prompt_tokens_total: metrics.prompt_tokens_total.load(Ordering::Relaxed),
        active_requests: metrics.active_requests.load(Ordering::Relaxed),
        queue_depth: metrics.queue_depth.load(Ordering::Relaxed),
    };

    let body = serde_json::json!({
        "requests_total": stats.requests_total,
        "tokens_generated_total": stats.tokens_generated_total,
        "prompt_tokens_total": stats.prompt_tokens_total,
        "active_requests": stats.active_requests,
        "queue_depth": stats.queue_depth,
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// `GET /admin/health` — extended health with pool readiness.
pub async fn admin_health(State(state): State<Arc<AppState>>) -> Response {
    let models = state
        .model_pool
        .lock()
        .map(|pool| pool.list())
        .unwrap_or_default();

    let loaded_count = models
        .iter()
        .filter(|m| m.status == ModelLoadStatus::Ready)
        .count();

    let loading_count = models
        .iter()
        .filter(|m| m.status == ModelLoadStatus::Loading)
        .count();

    let body = serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "pool": {
            "loaded": loaded_count,
            "loading": loading_count,
            "total": models.len(),
        }
    });
    (StatusCode::OK, Json(body)).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request, StatusCode};
    use axum::routing::{get, post};
    use axum::Router;
    use std::sync::Arc;
    use tower::ServiceExt as _;

    use crate::admin::auth::{admin_auth_middleware, AdminAuth};
    use crate::state::AppState;
    use crate::test_helpers::build_test_app_with_pool;

    async fn parse_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).unwrap_or(serde_json::json!(null))
    }

    fn make_admin_router(state: Arc<AppState>, token: Option<String>) -> Router {
        let auth = AdminAuth { token };
        // In axum, `.layer()` wraps from the outside in.  We want the order:
        //   request → auth_middleware → route handler
        // with the AdminAuth extension available to the middleware.
        //
        // Layers are applied in declaration order (outermost first):
        //   1. `Extension(auth)` injects AdminAuth into the request extensions.
        //   2. `from_fn(admin_auth_middleware)` runs next, can extract AdminAuth.
        Router::new()
            .route("/admin/models/load", post(super::admin_load_model))
            .route("/admin/models/unload", post(super::admin_unload_model))
            .route("/admin/models", get(super::admin_list_models))
            .route("/admin/stats", get(super::admin_stats))
            .route("/admin/health", get(super::admin_health))
            .layer(axum::middleware::from_fn(admin_auth_middleware))
            .layer(axum::Extension(auth))
            .with_state(state)
    }

    /// (a) admin_load_returns_202 — POST /admin/models/load returns 202.
    #[tokio::test]
    async fn admin_load_returns_202() {
        let state = build_test_app_with_pool().await;
        let app = make_admin_router(state, None);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/admin/models/load")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"id":"test","path":"/tmp/model.gguf"}"#))
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(
            resp.status(),
            StatusCode::ACCEPTED,
            "admin load should return 202"
        );
    }

    /// (b) admin_bearer_auth_rejects_missing_token — configure token; GET
    ///     /admin/models without auth; assert 401.
    #[tokio::test]
    async fn admin_bearer_auth_rejects_missing_token() {
        let state = build_test_app_with_pool().await;
        let app = make_admin_router(state, Some("secret-token".to_string()));

        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/models")
            .body(Body::empty())
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "missing token should yield 401"
        );
    }

    /// (c) admin_models_list_returns_json — GET /admin/models returns JSON
    ///     with a "models" array.
    #[tokio::test]
    async fn admin_models_list_returns_json() {
        let state = build_test_app_with_pool().await;
        let app = make_admin_router(state, None);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/models")
            .body(Body::empty())
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(resp.status(), StatusCode::OK);
        let json = parse_json(resp).await;
        assert!(
            json.get("models").is_some(),
            "response should have 'models' key: {json}"
        );
        assert!(
            json["models"].is_array(),
            "models should be an array: {json}"
        );
    }

    /// (d) admin_stats_returns_metrics — GET /admin/stats returns JSON with
    ///     "requests_total" key.
    #[tokio::test]
    async fn admin_stats_returns_metrics() {
        let state = build_test_app_with_pool().await;
        let app = make_admin_router(state, None);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/stats")
            .body(Body::empty())
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(resp.status(), StatusCode::OK);
        let json = parse_json(resp).await;
        assert!(
            json.get("requests_total").is_some(),
            "response should have 'requests_total': {json}"
        );
    }
}

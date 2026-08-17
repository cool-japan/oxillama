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
    // D1: reject model paths outside the configured allow-list (a no-op
    // when `allowed_model_dirs` is empty, i.e. not configured).
    if let Err(message) =
        crate::admin::path_guard::validate_model_path(&body.path, &state.allowed_model_dirs)
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                }
            })),
        )
            .into_response();
    }

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
        // D14 fix: this used to read `active_requests` (an in-flight
        // gauge) under the `requests_total` (cumulative counter) name.
        requests_total: metrics.total_requests(),
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

    // A missing (or unserialisable) `BackendInfo` reports the CPU shape rather
    // than `null`, so the key's type never varies across responses.
    let cpu_backend = || serde_json::json!({ "gpu_enabled": false });
    let backend = match state.backend_info() {
        Some(info) => serde_json::to_value(info).unwrap_or_else(|_| cpu_backend()),
        None => cpu_backend(),
    };

    let body = serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "pool": {
            "loaded": loaded_count,
            "loading": loading_count,
            "total": models.len(),
        },
        "backend": backend,
    });
    (StatusCode::OK, Json(body)).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::extract::ConnectInfo;
    use axum::http::{Method, Request, StatusCode};
    use axum::routing::{get, post};
    use axum::Router;
    use std::net::SocketAddr;
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

    /// Build a request that carries a loopback `ConnectInfo<SocketAddr>`
    /// extension, as axum supplies when bound with
    /// `into_make_service_with_connect_info::<SocketAddr>()`. Token-less
    /// admin auth (D1) now requires this to be present and loopback in
    /// order to allow the request through.
    fn loopback_request(method: Method, uri: &str) -> Request<Body> {
        let addr: SocketAddr = "127.0.0.1:54321".parse().expect("valid addr");
        Request::builder()
            .method(method)
            .uri(uri)
            .extension(ConnectInfo(addr))
            .body(Body::empty())
            .expect("build request")
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

        let mut req = loopback_request(Method::POST, "/admin/models/load");
        req.headers_mut()
            .insert("content-type", "application/json".parse().expect("hv"));
        *req.body_mut() = Body::from(r#"{"id":"test","path":"/tmp/model.gguf"}"#);

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

        let req = loopback_request(Method::GET, "/admin/models");

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

        let req = loopback_request(Method::GET, "/admin/stats");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(resp.status(), StatusCode::OK);
        let json = parse_json(resp).await;
        assert!(
            json.get("requests_total").is_some(),
            "response should have 'requests_total': {json}"
        );
    }

    /// (e) admin_health_reports_cpu_backend_by_default — an `AppState` that
    ///     was never given a `BackendInfo` must still expose a `backend`
    ///     object, reporting the CPU shape rather than omitting the key.
    #[tokio::test]
    async fn admin_health_reports_cpu_backend_by_default() {
        let state = build_test_app_with_pool().await;
        let app = make_admin_router(state, None);

        let req = loopback_request(Method::GET, "/admin/health");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(resp.status(), StatusCode::OK);
        let json = parse_json(resp).await;
        assert_eq!(json["status"], "ok", "existing keys must survive: {json}");
        assert!(
            json.get("pool").is_some(),
            "existing 'pool' key must survive: {json}"
        );
        assert_eq!(
            json["backend"]["gpu_enabled"],
            serde_json::json!(false),
            "default backend must report gpu_enabled=false: {json}"
        );
    }

    /// (f) admin_health_round_trips_backend_info — every `BackendInfo` field
    ///     set via `with_backend_info` reaches the `/admin/health` body.
    #[tokio::test]
    async fn admin_health_round_trips_backend_info() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let state = crate::test_helpers::new_test_state(
            tx,
            "test-model",
            oxillama_runtime::sampling::SamplerConfig::default(),
            None,
            0,
        )
        .with_backend_info(crate::state::BackendInfo {
            gpu_enabled: true,
            device_name: Some("Test Adapter".to_string()),
            backend: Some("Metal".to_string()),
            resident_tensors: Some(7),
            resident_bytes: Some(4096),
        });
        let app = make_admin_router(Arc::new(state), None);

        let req = loopback_request(Method::GET, "/admin/health");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(resp.status(), StatusCode::OK);
        let json = parse_json(resp).await;
        assert_eq!(json["backend"]["gpu_enabled"], serde_json::json!(true));
        assert_eq!(json["backend"]["device_name"], "Test Adapter");
        assert_eq!(json["backend"]["backend"], "Metal");
        assert_eq!(json["backend"]["resident_tensors"], serde_json::json!(7));
        assert_eq!(json["backend"]["resident_bytes"], serde_json::json!(4096));
    }

    /// D1 regression: with no token configured and no `ConnectInfo`
    /// extension present on the request (the exact shape of the previous
    /// vulnerability — `is_loopback` used to default to `true` in this
    /// case), the request must be rejected with 401, not allowed through.
    #[tokio::test]
    async fn admin_no_token_rejects_request_without_connect_info() {
        let state = build_test_app_with_pool().await;
        let app = make_admin_router(state, None);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/models")
            .body(Body::empty())
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "request with no verified peer address must be rejected, not treated as loopback"
        );
    }

    /// D1 regression: a genuinely remote peer cannot bypass token-less
    /// admin auth by spoofing `X-Forwarded-For: 127.0.0.1`.
    #[tokio::test]
    async fn admin_no_token_rejects_spoofed_forwarded_for() {
        let state = build_test_app_with_pool().await;
        let app = make_admin_router(state, None);

        let addr: SocketAddr = "203.0.113.7:1234".parse().expect("valid addr");
        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/models")
            .header("x-forwarded-for", "127.0.0.1")
            .extension(ConnectInfo(addr))
            .body(Body::empty())
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "spoofed X-Forwarded-For must not grant loopback trust to a remote peer"
        );
    }
}

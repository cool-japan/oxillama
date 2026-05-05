//! Admin LoRA registry endpoints.
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | `/admin/loras` | Load and register a LoRA adapter from a GGUF file |
//! | DELETE | `/admin/loras/{name}` | Unregister and drop a LoRA adapter |
//! | GET | `/admin/loras` | List all registered LoRA adapters |

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use crate::state::AppState;

// ── Request types ─────────────────────────────────────────────────────────────

/// Body for `POST /admin/loras`.
#[derive(Debug, Deserialize)]
pub struct RegisterLoraBody {
    /// Stable name used in inference requests (`lora: "name"`).
    pub name: String,
    /// Filesystem path to the LoRA GGUF adapter file.
    pub path: String,
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `POST /admin/loras` — load a LoRA adapter from a GGUF file and register it.
///
/// Blocks briefly while loading the adapter (adapter files are typically small,
/// < 1 GiB). Returns `200 OK` with the registered name on success, or
/// `400 Bad Request` if loading fails.
pub async fn admin_register_lora(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RegisterLoraBody>,
) -> Response {
    if body.name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "adapter name must not be empty",
                    "type": "invalid_request_error",
                }
            })),
        )
            .into_response();
    }

    let path = body.path.clone();
    let result = tokio::task::spawn_blocking(move || {
        oxillama_runtime::LoadedLora::load(&path).map_err(|e| format!("{e}"))
    })
    .await;

    match result {
        Ok(Ok(lora)) => {
            let name = body.name.clone();
            match state.loras.write() {
                Ok(mut registry) => {
                    registry.insert(name.clone(), Arc::new(lora));
                    let resp = serde_json::json!({
                        "name": name,
                        "status": "registered",
                    });
                    (StatusCode::OK, Json(resp)).into_response()
                }
                Err(_) => {
                    let err = serde_json::json!({
                        "error": {
                            "message": "LoRA registry lock poisoned",
                            "type": "internal_error",
                        }
                    });
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(err)).into_response()
                }
            }
        }
        Ok(Err(e)) => {
            let err = serde_json::json!({
                "error": {
                    "message": format!("failed to load LoRA adapter from path '{}': {e}", body.path),
                    "type": "invalid_request_error",
                }
            });
            (StatusCode::BAD_REQUEST, Json(err)).into_response()
        }
        Err(e) => {
            let err = serde_json::json!({
                "error": {
                    "message": format!("spawn_blocking join error: {e}"),
                    "type": "internal_error",
                }
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err)).into_response()
        }
    }
}

/// `DELETE /admin/loras/{name}` — unregister a LoRA adapter.
///
/// Returns `200 OK` if found and removed, `404 Not Found` otherwise.
pub async fn admin_unregister_lora(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Response {
    match state.loras.write() {
        Ok(mut registry) => {
            if registry.remove(&name).is_some() {
                let resp = serde_json::json!({ "name": name, "status": "unregistered" });
                (StatusCode::OK, Json(resp)).into_response()
            } else {
                let err = serde_json::json!({
                    "error": {
                        "message": format!("adapter '{name}' not found"),
                        "type": "invalid_request_error",
                    }
                });
                (StatusCode::NOT_FOUND, Json(err)).into_response()
            }
        }
        Err(_) => {
            let err = serde_json::json!({
                "error": {
                    "message": "LoRA registry lock poisoned",
                    "type": "internal_error",
                }
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err)).into_response()
        }
    }
}

/// `GET /admin/loras` — list all registered LoRA adapters.
pub async fn admin_list_loras(State(state): State<Arc<AppState>>) -> Response {
    match state.loras.read() {
        Ok(registry) => {
            let names: Vec<_> = registry.keys().cloned().collect();
            let resp = serde_json::json!({ "object": "list", "loras": names });
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(_) => {
            let err = serde_json::json!({
                "error": {
                    "message": "LoRA registry lock poisoned",
                    "type": "internal_error",
                }
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err)).into_response()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request, StatusCode};
    use axum::routing::{delete, get, post};
    use axum::Router;
    use std::sync::Arc;
    use tower::ServiceExt as _;

    use crate::state::AppState;
    use crate::test_helpers::build_test_app_with_pool;

    fn make_lora_router(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/admin/loras", post(super::admin_register_lora))
            .route("/admin/loras", get(super::admin_list_loras))
            .route("/admin/loras/{name}", delete(super::admin_unregister_lora))
            .with_state(state)
    }

    async fn parse_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).unwrap_or(serde_json::json!(null))
    }

    /// GET /admin/loras returns an empty list on a fresh state.
    #[tokio::test]
    async fn admin_list_loras_empty() {
        let state = build_test_app_with_pool().await;
        let app = make_lora_router(state);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/loras")
            .body(Body::empty())
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(resp.status(), StatusCode::OK);
        let json = parse_json(resp).await;
        assert!(json["loras"].is_array(), "loras must be an array: {json}");
        assert_eq!(
            json["loras"].as_array().unwrap().len(),
            0,
            "fresh state must have 0 loras"
        );
    }

    /// DELETE /admin/loras/{name} for an unknown name returns 404.
    #[tokio::test]
    async fn admin_unregister_unknown_lora_returns_404() {
        let state = build_test_app_with_pool().await;
        let app = make_lora_router(state);

        let req = Request::builder()
            .method(Method::DELETE)
            .uri("/admin/loras/nonexistent")
            .body(Body::empty())
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "unknown adapter must yield 404"
        );
    }

    /// POST /admin/loras with an empty name returns 400.
    #[tokio::test]
    async fn admin_register_lora_empty_name_returns_400() {
        let state = build_test_app_with_pool().await;
        let app = make_lora_router(state);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/admin/loras")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"","path":"/tmp/test.gguf"}"#))
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "empty adapter name must yield 400"
        );
    }

    /// POST /admin/loras with a non-existent path returns 400.
    #[tokio::test]
    async fn admin_register_lora_bad_path_returns_400() {
        let state = build_test_app_with_pool().await;
        let app = make_lora_router(state);

        let path = std::env::temp_dir().join("oxillama_lora_no_such_file.gguf");
        let _ = std::fs::remove_file(&path); // ensure it doesn't exist
        let body = serde_json::json!({
            "name": "test_adapter",
            "path": path.to_string_lossy()
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri("/admin/loras")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("build request");

        let resp = app.oneshot(req).await.expect("oneshot");
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "non-existent file must yield 400"
        );
    }

    /// Register then delete workflow: list shows 0 after delete.
    /// (Inserts a dummy LoadedLora directly into state to bypass file I/O.)
    #[tokio::test]
    async fn admin_delete_registered_lora_removes_it() {
        use oxillama_runtime::LoadedLora;
        let state = build_test_app_with_pool().await;

        // Insert a dummy LoadedLora directly (bypasses file I/O).
        {
            let dummy = Arc::new(LoadedLora {
                adapters: std::collections::HashMap::new(),
                rank: 8,
                alpha: 8.0,
            });
            state
                .loras
                .write()
                .expect("lock")
                .insert("test_delete_lora".to_string(), dummy);
        }

        let app = make_lora_router(Arc::clone(&state));

        // Verify it appears in the list.
        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/loras")
            .body(Body::empty())
            .expect("build GET");
        let resp = app.clone().oneshot(req).await.expect("oneshot");
        let json = parse_json(resp).await;
        let loras = json["loras"].as_array().expect("array");
        assert!(
            loras.iter().any(|v| v.as_str() == Some("test_delete_lora")),
            "adapter must appear in list before delete: {json}"
        );

        // Delete it.
        let req = Request::builder()
            .method(Method::DELETE)
            .uri("/admin/loras/test_delete_lora")
            .body(Body::empty())
            .expect("build DELETE");
        let resp = app.clone().oneshot(req).await.expect("oneshot");
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify it's gone.
        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/loras")
            .body(Body::empty())
            .expect("build GET");
        let resp = app.oneshot(req).await.expect("oneshot");
        let json = parse_json(resp).await;
        let loras = json["loras"].as_array().expect("array");
        assert!(
            !loras.iter().any(|v| v.as_str() == Some("test_delete_lora")),
            "adapter must be absent after delete: {json}"
        );
    }
}

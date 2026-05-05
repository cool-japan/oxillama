//! HTTP route handlers for the Run Steps subresource.
//!
//! | Method | Path | Handler |
//! |--------|------|---------|
//! | GET | `/v1/threads/:thread_id/runs/:run_id/steps`           | [`list_steps_handler`] |
//! | GET | `/v1/threads/:thread_id/runs/:run_id/steps/:step_id`  | [`get_step_handler`]   |

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::error::ServerError;
use crate::state::AppState;
use crate::threads::store::ThreadStore;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn require_store(state: &AppState) -> Option<Arc<ThreadStore>> {
    state.threads_store.clone()
}

fn assistants_unavailable_response() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "Assistants API is not enabled on this server",
            "type": "service_unavailable",
        }
    });
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

fn not_found_response(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "not_found_error",
        }
    });
    (StatusCode::NOT_FOUND, Json(body)).into_response()
}

fn server_error_response(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
        }
    });
    (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}

// ── GET /v1/threads/:thread_id/runs/:run_id/steps ─────────────────────────────

/// List all steps for a run, sorted by creation time ascending.
pub async fn list_steps_handler(
    State(state): State<Arc<AppState>>,
    Path((thread_id, run_id)): Path<(String, String)>,
) -> Response {
    let store = match require_store(&state) {
        Some(s) => s,
        None => return assistants_unavailable_response(),
    };

    let tid = thread_id.clone();
    let rid = run_id.clone();
    match tokio::task::spawn_blocking(move || store.list_steps(&tid, &rid)).await {
        Ok(Ok(steps)) => {
            let body = serde_json::json!({
                "object": "list",
                "data": steps,
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Ok(Err(ServerError::ThreadNotFound(_))) => {
            not_found_response(&format!("Thread '{thread_id}' not found"))
        }
        Ok(Err(ServerError::RunNotFound(_))) => {
            not_found_response(&format!("Run '{run_id}' not found"))
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── GET /v1/threads/:thread_id/runs/:run_id/steps/:step_id ───────────────────

/// Retrieve a single run step by ID.
pub async fn get_step_handler(
    State(state): State<Arc<AppState>>,
    Path((thread_id, run_id, step_id)): Path<(String, String, String)>,
) -> Response {
    let store = match require_store(&state) {
        Some(s) => s,
        None => return assistants_unavailable_response(),
    };

    let tid = thread_id.clone();
    let rid = run_id.clone();
    let sid = step_id.clone();
    match tokio::task::spawn_blocking(move || store.get_step(&tid, &rid, &sid)).await {
        Ok(Ok(step)) => (StatusCode::OK, Json(step)).into_response(),
        Ok(Err(ServerError::ThreadNotFound(_))) => {
            not_found_response(&format!("Thread '{thread_id}' not found"))
        }
        Ok(Err(ServerError::RunNotFound(_))) => {
            not_found_response(&format!("Run '{run_id}' not found"))
        }
        Ok(Err(ServerError::RunStepNotFound(_))) => {
            not_found_response(&format!("Step '{step_id}' not found"))
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::BatchRequest;
    use crate::queue::UsageStats;
    use crate::state::AppState;
    use crate::threads::queue::new_run_queue;
    use crate::threads::store::ThreadStore;
    use crate::threads::types::{Run, RunStatus, RunStep, RunStepStatus, RunStepType, Thread};
    use crate::threads::worker::spawn_run_worker;
    use axum::body::Body;
    use axum::http::Request;
    use axum::Router;
    use serde_json::{json, Value};
    use std::env::temp_dir;
    use std::sync::Arc;
    use tower::ServiceExt as _;
    use uuid::Uuid;

    fn make_store_and_dir(tag: &str) -> (Arc<ThreadStore>, std::path::PathBuf) {
        let id = Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_steps_route_test_{tag}_{id}"));
        let store = Arc::new(ThreadStore::new(dir.clone()).expect("ThreadStore::new"));
        (store, dir)
    }

    async fn build_live_router(store: Arc<ThreadStore>) -> Router {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<BatchRequest>(16);
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                if let BatchRequest::Generate { reply, .. } = req {
                    let _ = reply.send(Ok((
                        "mock response for steps test".to_string(),
                        UsageStats {
                            prompt_tokens: 3,
                            completion_tokens: 5,
                            total_tokens: 8,
                        },
                    )));
                }
            }
        });

        let (run_tx, run_rx) = new_run_queue();
        let store_c = Arc::clone(&store);
        let state = Arc::new(
            AppState::new(
                tx,
                "test-model".to_string(),
                oxillama_runtime::sampling::SamplerConfig::default(),
                None,
                0,
            )
            .with_threads(Arc::clone(&store), run_tx),
        );
        spawn_run_worker(store_c, run_rx, Arc::clone(&state));
        crate::app::build_app(state)
    }

    async fn post_json(app: Router, uri: &str, body: Value) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).expect("serialize")))
                    .expect("build request"),
            )
            .await
            .expect("router handled request");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("read body");
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(json!(null)),
        )
    }

    async fn get_json(app: Router, uri: &str) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(Body::empty())
                    .expect("build request"),
            )
            .await
            .expect("router handled request");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("read body");
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(json!(null)),
        )
    }

    /// After a run completes, a MessageCreation step should be recorded.
    #[tokio::test]
    async fn run_emits_message_creation_step() {
        let (store, _dir) = make_store_and_dir("emits_step");
        let app = build_live_router(Arc::clone(&store)).await;

        let (_, thread) = post_json(app.clone(), "/v1/threads", json!({})).await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/messages"),
            json!({ "role": "user", "content": "hello" }),
        )
        .await;

        let (_, run_body) = post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/runs"),
            json!({ "model": "test-model" }),
        )
        .await;
        let run_id = run_body["id"].as_str().expect("run id").to_string();

        // Poll until run is terminal.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let run = store.get_run(&thread_id, &run_id).expect("get run");
            if run.status.is_terminal() {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("run did not complete within deadline");
            }
        }

        // Now list steps.
        let steps = store.list_steps(&thread_id, &run_id).expect("list steps");
        assert!(
            !steps.is_empty(),
            "run should have produced at least one step"
        );
        assert_eq!(steps[0].step_type, RunStepType::MessageCreation);
    }

    /// `list_steps` via HTTP returns all steps in creation order.
    #[tokio::test]
    async fn step_list_route_returns_all_steps() {
        let (store, _dir) = make_store_and_dir("list_steps");

        // Manually create thread, run, and steps without going through a live worker.
        let thread = Thread {
            id: "thread_ls".to_string(),
            object: "thread".to_string(),
            created_at: 0,
            metadata: json!({}),
        };
        store.create_thread(&thread).expect("create thread");
        let run = Run {
            id: "run_ls".to_string(),
            object: "thread.run".to_string(),
            created_at: 0,
            thread_id: "thread_ls".to_string(),
            status: RunStatus::Completed,
            model: "test".to_string(),
            last_error: None,
        };
        store.create_run("thread_ls", &run).expect("create run");

        for i in 0..3_u32 {
            let step = RunStep {
                id: format!("step_{i}"),
                object: "thread.run.step".to_string(),
                run_id: "run_ls".to_string(),
                thread_id: "thread_ls".to_string(),
                step_type: RunStepType::MessageCreation,
                status: RunStepStatus::Completed,
                created_at: 1_000 + i as u64,
                completed_at: Some(2_000 + i as u64),
                failed_at: None,
                error: None,
                step_details: None,
            };
            store
                .append_step("thread_ls", "run_ls", &step)
                .expect("append step");
        }

        // Build a minimal app that serves the steps routes.
        let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);
        let (run_tx, _run_rx) = new_run_queue();
        let state = Arc::new(
            AppState::new(
                tx,
                "test-model".to_string(),
                oxillama_runtime::sampling::SamplerConfig::default(),
                None,
                0,
            )
            .with_threads(Arc::clone(&store), run_tx),
        );
        let app = crate::app::build_app(state);

        let (status, body) = get_json(app, "/v1/threads/thread_ls/runs/run_ls/steps").await;
        assert_eq!(status.as_u16(), 200);
        let data = body["data"].as_array().expect("data array");
        assert_eq!(data.len(), 3);
    }

    /// `get_step` by ID returns the matching step.
    #[tokio::test]
    async fn step_get_route_returns_correct_step() {
        let (store, _dir) = make_store_and_dir("get_step");

        let thread = Thread {
            id: "thread_gs".to_string(),
            object: "thread".to_string(),
            created_at: 0,
            metadata: json!({}),
        };
        store.create_thread(&thread).expect("create thread");
        let run = Run {
            id: "run_gs".to_string(),
            object: "thread.run".to_string(),
            created_at: 0,
            thread_id: "thread_gs".to_string(),
            status: RunStatus::Completed,
            model: "test".to_string(),
            last_error: None,
        };
        store.create_run("thread_gs", &run).expect("create run");

        let step = RunStep {
            id: "step_target".to_string(),
            object: "thread.run.step".to_string(),
            run_id: "run_gs".to_string(),
            thread_id: "thread_gs".to_string(),
            step_type: RunStepType::MessageCreation,
            status: RunStepStatus::Completed,
            created_at: 5_000,
            completed_at: Some(6_000),
            failed_at: None,
            error: None,
            step_details: None,
        };
        store
            .append_step("thread_gs", "run_gs", &step)
            .expect("append");

        let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);
        let (run_tx, _run_rx) = new_run_queue();
        let state = Arc::new(
            AppState::new(
                tx,
                "test-model".to_string(),
                oxillama_runtime::sampling::SamplerConfig::default(),
                None,
                0,
            )
            .with_threads(Arc::clone(&store), run_tx),
        );
        let app = crate::app::build_app(state);

        let (status, body) =
            get_json(app, "/v1/threads/thread_gs/runs/run_gs/steps/step_target").await;
        assert_eq!(status.as_u16(), 200);
        assert_eq!(body["id"], "step_target");
        assert_eq!(body["step_type"], "message_creation");
    }

    /// `get_step` for an unknown ID returns 404.
    #[tokio::test]
    async fn step_not_found_route_returns_404() {
        let (store, _dir) = make_store_and_dir("step_404");

        let thread = Thread {
            id: "thread_404".to_string(),
            object: "thread".to_string(),
            created_at: 0,
            metadata: json!({}),
        };
        store.create_thread(&thread).expect("create thread");
        let run = Run {
            id: "run_404".to_string(),
            object: "thread.run".to_string(),
            created_at: 0,
            thread_id: "thread_404".to_string(),
            status: RunStatus::Completed,
            model: "test".to_string(),
            last_error: None,
        };
        store.create_run("thread_404", &run).expect("create run");

        let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);
        let (run_tx, _run_rx) = new_run_queue();
        let state = Arc::new(
            AppState::new(
                tx,
                "test-model".to_string(),
                oxillama_runtime::sampling::SamplerConfig::default(),
                None,
                0,
            )
            .with_threads(Arc::clone(&store), run_tx),
        );
        let app = crate::app::build_app(state);

        let (status, _body) =
            get_json(app, "/v1/threads/thread_404/runs/run_404/steps/step_ghost").await;
        assert_eq!(status.as_u16(), 404);
    }
}

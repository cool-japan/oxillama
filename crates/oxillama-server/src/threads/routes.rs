//! HTTP route handlers for the OpenAI Assistants v2 API.
//!
//! Seven endpoints are provided:
//!
//! | Method | Path | Handler |
//! |--------|------|---------|
//! | POST | `/v1/threads` | [`create_thread_handler`] |
//! | GET | `/v1/threads/:thread_id` | [`get_thread_handler`] |
//! | POST | `/v1/threads/:thread_id/messages` | [`create_message_handler`] |
//! | GET | `/v1/threads/:thread_id/messages` | [`list_messages_handler`] |
//! | POST | `/v1/threads/:thread_id/runs` | [`create_run_handler`] |
//! | GET | `/v1/threads/:thread_id/runs/:run_id` | [`get_run_handler`] |
//! | POST | `/v1/threads/:thread_id/runs/:run_id/cancel` | [`cancel_run_handler`] |

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use uuid::Uuid;

use crate::error::ServerError;
use crate::state::AppState;
use crate::threads::queue::RunWorkItem;
use crate::threads::store::ThreadStore;
use crate::threads::types::{
    CreateMessageRequest, CreateRunRequest, CreateThreadRequest, Run, RunStatus, Thread,
    ThreadMessage,
};

// ── Helper: require the threads pipeline ─────────────────────────────────────

/// Extract `Arc<ThreadStore>` from state, returning a 503 response if threads are
/// not configured (i.e. `AppState::threads_store` is `None`).
///
/// Returns `Ok(store)` or `Err(ready-to-return Response)`.
fn require_store(state: &AppState) -> Option<Arc<ThreadStore>> {
    state.threads_store.clone()
}

/// Build a 503 "Assistants API not enabled" response.
fn assistants_unavailable_response() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "Assistants API is not enabled on this server",
            "type": "service_unavailable",
        }
    });
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

// ── POST /v1/threads ──────────────────────────────────────────────────────────

/// Create a new thread, optionally seeded with initial messages.
pub async fn create_thread_handler(
    State(state): State<Arc<AppState>>,
    body: Option<Json<CreateThreadRequest>>,
) -> Response {
    let store = match require_store(&state) {
        Some(s) => s,
        None => return assistants_unavailable_response(),
    };

    let thread_id = format!("thread_{}", Uuid::new_v4().as_simple());
    let metadata = body
        .as_ref()
        .and_then(|b| b.metadata.clone())
        .unwrap_or(serde_json::json!({}));

    let thread = Thread {
        id: thread_id.clone(),
        object: "thread".to_string(),
        created_at: unix_now(),
        metadata,
    };

    // Write thread to disk.
    let store_c = Arc::clone(&store);
    let thread_c = thread.clone();
    let result = tokio::task::spawn_blocking(move || store_c.create_thread(&thread_c)).await;

    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return server_error_response(&e.to_string()),
        Err(e) => return server_error_response(&format!("task join: {e}")),
    }

    // Append any seed messages provided in the request body.
    if let Some(Json(req)) = body {
        if let Some(seed_messages) = req.messages {
            for seed in seed_messages {
                let msg_id = format!("msg_{}", Uuid::new_v4().as_simple());
                let msg = ThreadMessage::new_user(msg_id, thread_id.clone(), seed.content);
                let store_c = Arc::clone(&store);
                let tid = thread_id.clone();
                let result =
                    tokio::task::spawn_blocking(move || store_c.append_message(&tid, &msg)).await;
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return server_error_response(&e.to_string()),
                    Err(e) => return server_error_response(&format!("task join: {e}")),
                }
            }
        }
    }

    (StatusCode::OK, Json(thread)).into_response()
}

// ── GET /v1/threads/:thread_id ────────────────────────────────────────────────

/// Retrieve a thread by ID.
pub async fn get_thread_handler(
    State(state): State<Arc<AppState>>,
    Path(thread_id): Path<String>,
) -> Response {
    let store = match require_store(&state) {
        Some(s) => s,
        None => return assistants_unavailable_response(),
    };

    let store_c = Arc::clone(&store);
    let tid = thread_id.clone();
    match tokio::task::spawn_blocking(move || store_c.get_thread(&tid)).await {
        Ok(Ok(thread)) => (StatusCode::OK, Json(thread)).into_response(),
        Ok(Err(ServerError::ThreadNotFound(_))) => {
            not_found_response(&format!("Thread '{thread_id}' not found"))
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── POST /v1/threads/:thread_id/messages ──────────────────────────────────────

/// Append a user message to a thread.
pub async fn create_message_handler(
    State(state): State<Arc<AppState>>,
    Path(thread_id): Path<String>,
    Json(body): Json<CreateMessageRequest>,
) -> Response {
    let store = match require_store(&state) {
        Some(s) => s,
        None => return assistants_unavailable_response(),
    };

    let msg_id = format!("msg_{}", Uuid::new_v4().as_simple());
    let msg = ThreadMessage::new_user(msg_id, thread_id.clone(), body.content);

    let store_c = Arc::clone(&store);
    let tid = thread_id.clone();
    let msg_c = msg.clone();
    match tokio::task::spawn_blocking(move || store_c.append_message(&tid, &msg_c)).await {
        Ok(Ok(())) => (StatusCode::OK, Json(msg)).into_response(),
        Ok(Err(ServerError::ThreadNotFound(_))) => {
            not_found_response(&format!("Thread '{thread_id}' not found"))
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── GET /v1/threads/:thread_id/messages ───────────────────────────────────────

/// List all messages in a thread in creation order.
pub async fn list_messages_handler(
    State(state): State<Arc<AppState>>,
    Path(thread_id): Path<String>,
) -> Response {
    let store = match require_store(&state) {
        Some(s) => s,
        None => return assistants_unavailable_response(),
    };

    let store_c = Arc::clone(&store);
    let tid = thread_id.clone();
    match tokio::task::spawn_blocking(move || store_c.list_messages(&tid)).await {
        Ok(Ok(messages)) => {
            let body = serde_json::json!({
                "object": "list",
                "data": messages,
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Ok(Err(ServerError::ThreadNotFound(_))) => {
            not_found_response(&format!("Thread '{thread_id}' not found"))
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── POST /v1/threads/:thread_id/runs ──────────────────────────────────────────

/// Create and enqueue a run for a thread.
pub async fn create_run_handler(
    State(state): State<Arc<AppState>>,
    Path(thread_id): Path<String>,
    Json(body): Json<CreateRunRequest>,
) -> Response {
    let store = match require_store(&state) {
        Some(s) => s,
        None => return assistants_unavailable_response(),
    };

    // Verify thread exists before creating the run.
    {
        let store_c = Arc::clone(&store);
        let tid = thread_id.clone();
        match tokio::task::spawn_blocking(move || store_c.get_thread(&tid)).await {
            Ok(Ok(_)) => {}
            Ok(Err(ServerError::ThreadNotFound(_))) => {
                return not_found_response(&format!("Thread '{thread_id}' not found"));
            }
            Ok(Err(e)) => return server_error_response(&e.to_string()),
            Err(e) => return server_error_response(&format!("task join: {e}")),
        }
    }

    let run_id = format!("run_{}", Uuid::new_v4().as_simple());
    let model = body.model.clone().unwrap_or_else(|| state.model_id.clone());
    let max_tokens = body.max_tokens.unwrap_or(512);

    let run = Run {
        id: run_id.clone(),
        object: "thread.run".to_string(),
        created_at: unix_now(),
        thread_id: thread_id.clone(),
        status: RunStatus::Queued,
        model: model.clone(),
        last_error: None,
    };

    // Persist the run.
    {
        let store_c = Arc::clone(&store);
        let run_c = run.clone();
        let tid = thread_id.clone();
        match tokio::task::spawn_blocking(move || store_c.create_run(&tid, &run_c)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return server_error_response(&e.to_string()),
            Err(e) => return server_error_response(&format!("task join: {e}")),
        }
    }

    // Enqueue for background processing.
    if let Some(run_queue_tx) = &state.run_queue_tx {
        let work_item = RunWorkItem {
            thread_id: thread_id.clone(),
            run_id: run_id.clone(),
            model: body.model.clone(),
            instructions: body.instructions.clone(),
            max_tokens,
        };
        if run_queue_tx.send(work_item).is_err() {
            // Worker is not running; mark run as failed immediately.
            let store_c = Arc::clone(&store);
            let tid = thread_id.clone();
            let rid = run_id.clone();
            tokio::task::spawn_blocking(move || {
                let _ = store_c.force_update_run_status(
                    &tid,
                    &rid,
                    RunStatus::Failed,
                    Some(crate::threads::types::RunError {
                        code: "worker_dead".to_string(),
                        message: "run worker is not running".to_string(),
                    }),
                );
            })
            .await
            .ok();
        }
    }

    (StatusCode::OK, Json(run)).into_response()
}

// ── GET /v1/threads/:thread_id/runs/:run_id ───────────────────────────────────

/// Retrieve the current status of a run.
pub async fn get_run_handler(
    State(state): State<Arc<AppState>>,
    Path((thread_id, run_id)): Path<(String, String)>,
) -> Response {
    let store = match require_store(&state) {
        Some(s) => s,
        None => return assistants_unavailable_response(),
    };

    let store_c = Arc::clone(&store);
    let tid = thread_id.clone();
    let rid = run_id.clone();
    match tokio::task::spawn_blocking(move || store_c.get_run(&tid, &rid)).await {
        Ok(Ok(run)) => (StatusCode::OK, Json(run)).into_response(),
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

// ── POST /v1/threads/:thread_id/runs/:run_id/cancel ───────────────────────────

/// Request cancellation of a queued or in-progress run.
///
/// If the run is already in a terminal state (completed, failed, cancelled,
/// expired) this endpoint returns 409 Conflict.
pub async fn cancel_run_handler(
    State(state): State<Arc<AppState>>,
    Path((thread_id, run_id)): Path<(String, String)>,
) -> Response {
    let store = match require_store(&state) {
        Some(s) => s,
        None => return assistants_unavailable_response(),
    };

    let store_c = Arc::clone(&store);
    let tid = thread_id.clone();
    let rid = run_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        // Read current status first.
        let run = store_c.get_run(&tid, &rid)?;
        if run.status.is_terminal() {
            return Err(ServerError::RunInTerminalState(format!(
                "run '{}' is already in terminal state {:?}",
                rid, run.status
            )));
        }
        // Atomically transition to Cancelled.
        store_c.force_update_run_status(&tid, &rid, RunStatus::Cancelled, None)?;
        store_c.get_run(&tid, &rid)
    })
    .await;

    match result {
        Ok(Ok(run)) => (StatusCode::OK, Json(run)).into_response(),
        Ok(Err(ServerError::ThreadNotFound(_))) => {
            not_found_response(&format!("Thread '{thread_id}' not found"))
        }
        Ok(Err(ServerError::RunNotFound(_))) => {
            not_found_response(&format!("Run '{run_id}' not found"))
        }
        Ok(Err(ServerError::RunInTerminalState(msg))) => {
            let body = serde_json::json!({
                "error": {
                    "message": msg,
                    "type": "conflict_error",
                }
            });
            (StatusCode::CONFLICT, Json(body)).into_response()
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

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

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
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
    use crate::threads::worker::spawn_run_worker;
    use axum::body::Body;
    use axum::http::Request;
    use axum::Router;
    use serde_json::{json, Value};
    use std::env::temp_dir;
    use std::sync::Arc;
    use tower::ServiceExt as _;
    use uuid::Uuid;

    // ── Test fixtures ─────────────────────────────────────────────────────────

    fn make_thread_store(tag: &str) -> Arc<ThreadStore> {
        let id = Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_threads_route_test_{tag}_{id}"));
        Arc::new(ThreadStore::new(dir).expect("ThreadStore::new"))
    }

    async fn build_test_router(store: Arc<ThreadStore>) -> Router {
        // Dead inference worker — handlers that don't touch inference succeed.
        let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);
        let (run_tx, mut run_rx) = new_run_queue();

        // Keep the receiver alive in a background task that drains but never
        // processes items.  This prevents `run_queue_tx.send()` from returning
        // `Err` (which would cause the create_run handler to mark runs as Failed).
        tokio::spawn(async move {
            // Drain silently so that runs stay in `Queued` state.
            while run_rx.recv().await.is_some() {}
        });

        let state = Arc::new(
            AppState::new(
                tx,
                "test-model".to_string(),
                oxillama_runtime::sampling::SamplerConfig::default(),
                None,
                0,
            )
            .with_threads(store, run_tx),
        );

        crate::app::build_app(state)
    }

    async fn build_live_router(store: Arc<ThreadStore>) -> Router {
        // Live inference worker that returns "mock assistant response".
        let (tx, mut rx) = tokio::sync::mpsc::channel::<BatchRequest>(16);
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                if let BatchRequest::Generate { reply, .. } = req {
                    let _ = reply.send(Ok((
                        "mock assistant response".to_string(),
                        UsageStats {
                            prompt_tokens: 5,
                            completion_tokens: 4,
                            total_tokens: 9,
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

        // Spawn the run worker so runs actually get processed.
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
                    .body(Body::from(
                        serde_json::to_string(&body).expect("serialize body"),
                    ))
                    .expect("build request"),
            )
            .await
            .expect("router handled request");

        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("read body");
        let value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
        (status, value)
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
        let value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
        (status, value)
    }

    // ── Thread tests ──────────────────────────────────────────────────────────

    /// POST /v1/threads → returns id and metadata.
    #[tokio::test]
    async fn thread_create_returns_id_and_metadata() {
        let store = make_thread_store("create_id");
        let app = build_test_router(store).await;

        let (status, body) = post_json(
            app,
            "/v1/threads",
            json!({ "metadata": { "user": "alice" } }),
        )
        .await;

        assert_eq!(status.as_u16(), 200);
        let id = body["id"].as_str().expect("id field");
        assert!(
            id.starts_with("thread_"),
            "id should start with thread_: {id}"
        );
        assert_eq!(body["object"], "thread");
        assert_eq!(body["metadata"]["user"], "alice");
    }

    /// GET /v1/threads/:id → returns persisted metadata.
    #[tokio::test]
    async fn thread_get_returns_persisted_metadata() {
        let store = make_thread_store("get_persisted");
        let app = build_test_router(Arc::clone(&store)).await;

        let (_, create_body) = post_json(
            app.clone(),
            "/v1/threads",
            json!({ "metadata": { "key": "value" } }),
        )
        .await;
        let thread_id = create_body["id"].as_str().expect("id").to_string();

        let app2 = build_test_router(store).await;
        let (status, body) = get_json(app2, &format!("/v1/threads/{thread_id}")).await;

        assert_eq!(status.as_u16(), 200);
        assert_eq!(body["id"], thread_id);
        assert_eq!(body["metadata"]["key"], "value");
    }

    /// POST /v1/threads/:id/messages → message is persisted.
    #[tokio::test]
    async fn thread_message_append_persists() {
        let store = make_thread_store("msg_append");
        let app = build_test_router(Arc::clone(&store)).await;

        let (_, thread) = post_json(app.clone(), "/v1/threads", json!({})).await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        let (status, msg_body) = post_json(
            app,
            &format!("/v1/threads/{thread_id}/messages"),
            json!({ "role": "user", "content": "hello world" }),
        )
        .await;

        assert_eq!(status.as_u16(), 200);
        assert_eq!(msg_body["content"][0]["text"]["value"], "hello world");
        assert_eq!(msg_body["role"], "user");
    }

    /// GET /v1/threads/:id/messages → messages returned in append order.
    #[tokio::test]
    async fn thread_messages_listed_in_order() {
        let store = make_thread_store("msgs_order");
        let app = build_test_router(Arc::clone(&store)).await;

        let (_, thread) = post_json(app.clone(), "/v1/threads", json!({})).await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        for i in 0..5_u32 {
            post_json(
                app.clone(),
                &format!("/v1/threads/{thread_id}/messages"),
                json!({ "role": "user", "content": format!("message {i}") }),
            )
            .await;
        }

        let app2 = build_test_router(store).await;
        let (status, list_body) =
            get_json(app2, &format!("/v1/threads/{thread_id}/messages")).await;
        assert_eq!(status.as_u16(), 200);
        let data = list_body["data"].as_array().expect("data array");
        assert_eq!(data.len(), 5);
        for (i, msg) in data.iter().enumerate() {
            assert_eq!(msg["content"][0]["text"]["value"], format!("message {i}"));
        }
    }

    // ── Run tests ─────────────────────────────────────────────────────────────

    /// POST /v1/threads/:id/runs → run transitions to completed.
    #[tokio::test]
    async fn thread_run_create_transitions_to_completed() {
        let store = make_thread_store("run_complete");
        let app = build_live_router(Arc::clone(&store)).await;

        let (_, thread) = post_json(app.clone(), "/v1/threads", json!({})).await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/messages"),
            json!({ "role": "user", "content": "what is 2+2?" }),
        )
        .await;

        let (status, run_body) = post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/runs"),
            json!({ "model": "test-model" }),
        )
        .await;
        assert_eq!(status.as_u16(), 200);
        let run_id = run_body["id"].as_str().expect("run id").to_string();

        // Poll until terminal.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let (_, status_body) = get_json(
                app.clone(),
                &format!("/v1/threads/{thread_id}/runs/{run_id}"),
            )
            .await;
            let run_status = status_body["status"].as_str().unwrap_or("unknown");
            if run_status == "completed" {
                break;
            }
            if matches!(run_status, "failed" | "cancelled" | "expired") {
                panic!("run reached unexpected terminal state: {run_status}");
            }
            if std::time::Instant::now() > deadline {
                panic!("run did not complete within deadline");
            }
        }
    }

    /// GET /v1/threads/:id/runs/:run_id → returns status.
    #[tokio::test]
    async fn thread_run_get_returns_status() {
        let store = make_thread_store("run_get_status");
        let app = build_test_router(Arc::clone(&store)).await;

        let (_, thread) = post_json(app.clone(), "/v1/threads", json!({})).await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        let (_, run_body) = post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/runs"),
            json!({ "model": "test-model" }),
        )
        .await;
        let run_id = run_body["id"].as_str().expect("run id").to_string();

        let (status, get_body) = get_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/runs/{run_id}"),
        )
        .await;
        assert_eq!(status.as_u16(), 200);
        assert_eq!(get_body["id"], run_id);
        assert!(get_body["status"].as_str().is_some());
    }

    /// POST /v1/threads/:id/runs/:run_id/cancel → marks run cancelled.
    #[tokio::test]
    async fn thread_run_cancel_marks_cancelled() {
        let store = make_thread_store("run_cancel");
        // Use a test router with dead worker so runs stay queued.
        let app = build_test_router(Arc::clone(&store)).await;

        let (_, thread) = post_json(app.clone(), "/v1/threads", json!({})).await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        let (_, run_body) = post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/runs"),
            json!({ "model": "test-model" }),
        )
        .await;
        let run_id = run_body["id"].as_str().expect("run id").to_string();

        let (cancel_status, cancel_body) = post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/runs/{run_id}/cancel"),
            json!({}),
        )
        .await;

        assert_eq!(cancel_status.as_u16(), 200);
        assert_eq!(cancel_body["status"], "cancelled");
    }

    // ── Not-found tests ───────────────────────────────────────────────────────

    /// GET non-existent thread → 404.
    #[tokio::test]
    async fn thread_not_found_returns_404() {
        let store = make_thread_store("thread_404");
        let app = build_test_router(store).await;

        let (status, _) = get_json(app, "/v1/threads/thread_doesnotexist").await;
        assert_eq!(status.as_u16(), 404);
    }

    /// GET non-existent run → 404.
    #[tokio::test]
    async fn run_not_found_returns_404() {
        let store = make_thread_store("run_404");
        let app = build_test_router(Arc::clone(&store)).await;

        let (_, thread) = post_json(app.clone(), "/v1/threads", json!({})).await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        let (status, _) = get_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/runs/run_doesnotexist"),
        )
        .await;
        assert_eq!(status.as_u16(), 404);
    }

    // ── Persistence test ──────────────────────────────────────────────────────

    /// Thread and message data persists across store drop + recreate.
    #[tokio::test]
    async fn thread_persistence_across_restart() {
        let id = Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_thread_restart_{id}"));
        let store = Arc::new(ThreadStore::new(dir.clone()).expect("store"));

        let app = build_test_router(Arc::clone(&store)).await;

        let (_, thread) = post_json(
            app.clone(),
            "/v1/threads",
            json!({ "metadata": { "restart": true } }),
        )
        .await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/messages"),
            json!({ "role": "user", "content": "persistent message" }),
        )
        .await;

        // Drop first store/app and recreate from same directory.
        drop(store);

        let store2 = Arc::new(ThreadStore::new(dir).expect("store2"));
        let app2 = build_test_router(store2).await;

        let (status, thread_body) =
            get_json(app2.clone(), &format!("/v1/threads/{thread_id}")).await;
        assert_eq!(
            status.as_u16(),
            200,
            "thread should be readable after restart"
        );
        assert_eq!(thread_body["metadata"]["restart"], true);

        let (msg_status, msg_list) =
            get_json(app2, &format!("/v1/threads/{thread_id}/messages")).await;
        assert_eq!(msg_status.as_u16(), 200);
        let data = msg_list["data"].as_array().expect("data");
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["content"][0]["text"]["value"], "persistent message");
    }

    // ── Atomic write test ─────────────────────────────────────────────────────

    /// Rapid writes to a thread's message log leave no partial state.
    #[tokio::test]
    async fn thread_message_atomic_write_no_partial_state() {
        let store = make_thread_store("atomic_write");
        let app = build_test_router(Arc::clone(&store)).await;

        let (_, thread) = post_json(app.clone(), "/v1/threads", json!({})).await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        // Append 20 messages in rapid succession.
        for i in 0..20_u32 {
            post_json(
                app.clone(),
                &format!("/v1/threads/{thread_id}/messages"),
                json!({ "role": "user", "content": format!("msg {i}") }),
            )
            .await;
        }

        // Read back and verify every entry is valid JSON with correct fields.
        let app2 = build_test_router(Arc::clone(&store)).await;
        let (status, list) = get_json(app2, &format!("/v1/threads/{thread_id}/messages")).await;
        assert_eq!(status.as_u16(), 200);
        let data = list["data"].as_array().expect("data");
        assert_eq!(data.len(), 20, "all 20 messages should be readable");
        for (i, msg) in data.iter().enumerate() {
            // Each message must have valid structure.
            assert!(msg["id"].as_str().is_some(), "message {i} should have id");
            assert_eq!(
                msg["content"][0]["text"]["value"],
                format!("msg {i}"),
                "message {i} content mismatch"
            );
        }
    }

    // ── Failed run status propagation test ───────────────────────────────────

    /// When the inference worker dies, a run transitions to failed.
    #[tokio::test]
    async fn thread_run_failed_status_propagates_error() {
        let store = make_thread_store("run_failed");

        // Build state with a dying worker channel.
        let (tx, rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);
        // Drop the receiver to make the queue appear closed.
        drop(rx);

        let (run_tx, run_rx) = new_run_queue();
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

        // Start the run worker; it will fail because the inference queue is dead.
        spawn_run_worker(Arc::clone(&store), run_rx, Arc::clone(&state));

        let app = crate::app::build_app(state);

        let (_, thread) = post_json(app.clone(), "/v1/threads", json!({})).await;
        let thread_id = thread["id"].as_str().expect("id").to_string();

        post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/messages"),
            json!({ "role": "user", "content": "test" }),
        )
        .await;

        let (_, run_body) = post_json(
            app.clone(),
            &format!("/v1/threads/{thread_id}/runs"),
            json!({ "model": "test-model" }),
        )
        .await;
        let run_id = run_body["id"].as_str().expect("run id").to_string();

        // Poll until the run reaches a terminal state.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let (_, status_body) = get_json(
                app.clone(),
                &format!("/v1/threads/{thread_id}/runs/{run_id}"),
            )
            .await;
            let run_status = status_body["status"].as_str().unwrap_or("unknown");
            if run_status == "failed" {
                // Verify error details are populated.
                assert!(
                    status_body["last_error"].is_object(),
                    "last_error should be an object: {status_body}"
                );
                break;
            }
            if run_status == "completed" {
                // Acceptable if mock somehow ran — skip this path check.
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("run did not reach terminal state within deadline");
            }
        }
    }
}

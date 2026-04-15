//! Test utilities for axum integration tests.
//!
//! Builds a test router backed by a dead inference worker channel so that
//! all route handlers can be exercised without loading a real GGUF model.
//! Handlers that send to the queue will receive a `WorkerDead` error once the
//! receiver is dropped; handlers that only read `AppState` metadata (health,
//! models) succeed normally.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt as _;

use crate::app::build_app;
use crate::queue::{BatchRequest, UsageStats};
use crate::state::AppState;
use oxillama_runtime::sampling::SamplerConfig;

/// Build an axum `Router` wired to a dead inference worker.
///
/// The mpsc receiver is immediately dropped after this function returns,
/// so any `queue.send(…)` call in a handler will see a closed channel and
/// return `ServerError::WorkerDead` (HTTP 503).
pub async fn build_test_app() -> axum::Router {
    let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);
    let state = Arc::new(AppState::new(
        tx,
        "test-model".to_string(),
        SamplerConfig::default(),
        None, // no vocab — grammar tests will hit ModelNotReady
        0,    // hidden_size irrelevant without a real model
    ));
    build_app(state)
}

/// POST JSON to `uri` on the given `app` and return `(StatusCode, Value)`.
pub async fn post_json(app: axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&body).expect("test body should be serializable"),
                ))
                .expect("request builder should succeed"),
        )
        .await
        .expect("router should handle the request");

    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body should be readable");
    let value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
    (status, value)
}

/// Build an axum `Router` wired to a live mock inference worker.
///
/// The mock worker processes [`BatchRequest`] messages as follows:
/// - `Generate`       → responds with `Ok("mock generated text")`
/// - `GenerateStream` → calls callback with `"mock "` then `"token"`, then `Ok(())`
/// - `Embed`          → responds with `Ok(vec![0.1_f32; 32])`
///
/// This allows success-path tests for route handlers without loading a real
/// GGUF model.
pub async fn build_live_test_app() -> axum::Router {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<BatchRequest>(16);

    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            match req {
                BatchRequest::Generate { reply, .. } => {
                    let usage = UsageStats {
                        prompt_tokens: 5,
                        completion_tokens: 3,
                        total_tokens: 8,
                    };
                    let _ = reply.send(Ok(("mock generated text".to_string(), usage)));
                }
                BatchRequest::GenerateStream {
                    mut callback,
                    reply,
                    ..
                } => {
                    // `blocking_send` inside the callback panics when called from
                    // an async context.  Move the callback invocations into a
                    // `spawn_blocking` thread so that `blocking_send` is safe.
                    let _ = tokio::task::spawn_blocking(move || {
                        callback("mock ");
                        callback("token");
                    })
                    .await;
                    let _ = reply.send(Ok(UsageStats {
                        prompt_tokens: 5,
                        completion_tokens: 2,
                        total_tokens: 7,
                    }));
                }
                BatchRequest::Embed { reply, .. } => {
                    let _ = reply.send(Ok(vec![0.1_f32; 32]));
                }
            }
        }
    });

    let state = Arc::new(AppState::new(
        tx,
        "test-model".to_string(),
        SamplerConfig::default(),
        None, // no vocab
        0,    // hidden_size irrelevant for mock
    ));
    build_app(state)
}

/// GET `uri` on the given `app` and return `(StatusCode, Value)`.
pub async fn get(app: axum::Router, uri: &str) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .expect("request builder should succeed"),
        )
        .await
        .expect("router should handle the request");

    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body should be readable");
    let value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
    (status, value)
}

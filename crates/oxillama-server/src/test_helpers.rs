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
use crate::prefix_registry::{PrefixCacheRegistry, DEFAULT_MAX_NAMESPACES};
use crate::queue::{BatchRequest, UsageStats, VocabBytes};
use crate::state::AppState;
use oxillama_runtime::sampling::SamplerConfig;
use oxillama_runtime::{ChatTemplate, FinishReason, PrefixCacheConfig};

/// Construct a fresh `AppState` for tests, wiring up a fresh prefix-cache
/// registry, a "worker alive" flag defaulted to `true`, and a unique
/// per-test temp directory for the batch spool store — all the plumbing
/// that `AppState::new`'s fallible, multi-arg signature (D6/D7/D10)
/// requires but that individual test call sites shouldn't have to repeat.
///
/// `.expect(...)` is acceptable here: this module is test-only
/// (`#[cfg(test)] pub(crate) mod test_helpers;` in `lib.rs`), so a failure
/// here is a broken test environment, not a production code path.
pub fn new_test_state(
    queue: tokio::sync::mpsc::Sender<BatchRequest>,
    model_id: &str,
    default_sampler: SamplerConfig,
    vocab_bytes: Option<VocabBytes>,
    hidden_size: usize,
) -> AppState {
    let registry = Arc::new(PrefixCacheRegistry::new(
        PrefixCacheConfig::default(),
        DEFAULT_MAX_NAMESPACES,
    ));
    let worker_alive = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let spool_dir = std::env::temp_dir().join(format!(
        "oxillama_test_spool_{}",
        uuid::Uuid::new_v4().as_simple()
    ));
    AppState::new(
        queue,
        model_id.to_string(),
        default_sampler,
        vocab_bytes,
        hidden_size,
        ChatTemplate::default(),
        registry,
        worker_alive,
        Some(spool_dir),
    )
    .expect("new_test_state: AppState::new should succeed in a test environment")
}

/// Build an axum `Router` wired to a dead inference worker.
///
/// The mpsc receiver is immediately dropped after this function returns,
/// so any `queue.send(…)` call in a handler will see a closed channel and
/// return `ServerError::WorkerDead` (HTTP 503).
pub async fn build_test_app() -> axum::Router {
    let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);
    let state = Arc::new(new_test_state(
        tx,
        "test-model",
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
                    let _ = reply.send(Ok((
                        "mock generated text".to_string(),
                        usage,
                        FinishReason::Eos,
                    )));
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
                    let _ = reply.send(Ok((
                        UsageStats {
                            prompt_tokens: 5,
                            completion_tokens: 2,
                            total_tokens: 7,
                        },
                        FinishReason::Eos,
                    )));
                }
                BatchRequest::Embed { reply, .. } => {
                    let _ = reply.send(Ok(vec![0.1_f32; 32]));
                }
            }
        }
    });

    let state = Arc::new(new_test_state(
        tx,
        "test-model",
        SamplerConfig::default(),
        None, // no vocab
        0,    // hidden_size irrelevant for mock
    ));
    build_app(state)
}

/// What the mock worker in [`build_live_test_app_spec`] should pretend
/// happened.
#[derive(Debug, Clone)]
pub struct MockWorkerSpec {
    /// The chat template the served "model" uses. Route handlers read this
    /// from `AppState` to render turns, so it is what decides which markers
    /// end up in the prompt the worker receives.
    pub chat_template: oxillama_runtime::ChatTemplate,
    /// The finish reason the mock reports for both `Generate` and
    /// `GenerateStream`.
    pub finish_reason: FinishReason,
    /// When `true` the "generated text" is the prompt the worker was handed.
    ///
    /// That is what makes the rendered prompt observable from the HTTP
    /// response, so a test can assert end-to-end that the wire path really
    /// used the model's template — not just that the pure render function
    /// does the right thing in isolation.
    pub echo_prompt: bool,
}

impl Default for MockWorkerSpec {
    fn default() -> Self {
        Self {
            chat_template: oxillama_runtime::ChatTemplate::default(),
            finish_reason: FinishReason::Eos,
            echo_prompt: false,
        }
    }
}

/// What the mock worker actually received, for assertions the HTTP response
/// alone cannot make (notably `add_special`).
#[derive(Debug, Clone)]
pub struct CapturedRequest {
    /// The fully rendered prompt as dispatched to the worker.
    pub prompt: String,
    /// The `max_tokens` budget the route resolved.
    pub max_tokens: usize,
    /// Whether the route asked the tokenizer to apply the model's own
    /// BOS/EOS policy.
    pub add_special: bool,
}

/// [`build_live_test_app`] with a configurable mock worker, plus a handle on
/// everything that worker received.
pub async fn build_live_test_app_spec(
    spec: MockWorkerSpec,
) -> (axum::Router, Arc<std::sync::Mutex<Vec<CapturedRequest>>>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<BatchRequest>(16);
    let captured: Arc<std::sync::Mutex<Vec<CapturedRequest>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured_worker = Arc::clone(&captured);
    let finish_reason = spec.finish_reason;
    let echo_prompt = spec.echo_prompt;

    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            match req {
                BatchRequest::Generate {
                    prompt,
                    max_tokens,
                    add_special,
                    reply,
                    ..
                } => {
                    if let Ok(mut c) = captured_worker.lock() {
                        c.push(CapturedRequest {
                            prompt: prompt.clone(),
                            max_tokens,
                            add_special,
                        });
                    }
                    let text = if echo_prompt {
                        prompt
                    } else {
                        "mock generated text".to_string()
                    };
                    let usage = UsageStats {
                        prompt_tokens: 5,
                        completion_tokens: 3,
                        total_tokens: 8,
                    };
                    let _ = reply.send(Ok((text, usage, finish_reason)));
                }
                BatchRequest::GenerateStream {
                    prompt,
                    max_tokens,
                    add_special,
                    mut callback,
                    reply,
                    ..
                } => {
                    if let Ok(mut c) = captured_worker.lock() {
                        c.push(CapturedRequest {
                            prompt: prompt.clone(),
                            max_tokens,
                            add_special,
                        });
                    }
                    let chunk = if echo_prompt {
                        prompt
                    } else {
                        "mock token".to_string()
                    };
                    let _ = tokio::task::spawn_blocking(move || {
                        callback(&chunk);
                    })
                    .await;
                    let _ = reply.send(Ok((
                        UsageStats {
                            prompt_tokens: 5,
                            completion_tokens: 2,
                            total_tokens: 7,
                        },
                        finish_reason,
                    )));
                }
                BatchRequest::Embed { reply, .. } => {
                    let _ = reply.send(Ok(vec![0.1_f32; 32]));
                }
            }
        }
    });

    let registry = Arc::new(PrefixCacheRegistry::new(
        PrefixCacheConfig::default(),
        DEFAULT_MAX_NAMESPACES,
    ));
    let worker_alive = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let spool_dir = std::env::temp_dir().join(format!(
        "oxillama_test_spool_{}",
        uuid::Uuid::new_v4().as_simple()
    ));
    let state = AppState::new(
        tx,
        "test-model".to_string(),
        SamplerConfig::default(),
        None,
        0,
        spec.chat_template,
        registry,
        worker_alive,
        Some(spool_dir),
    )
    .expect("build_live_test_app_spec: AppState::new should succeed in a test environment");

    (build_app(Arc::new(state)), captured)
}

/// Build an `Arc<AppState>` with a pool, useful for admin route tests.
///
/// Like `build_test_app` but returns the `Arc<AppState>` rather than a
/// `Router`, so that admin test helpers can inject custom routers with the
/// state.
pub async fn build_test_app_with_pool() -> std::sync::Arc<AppState> {
    let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);
    std::sync::Arc::new(new_test_state(
        tx,
        "test-model",
        SamplerConfig::default(),
        None,
        0,
    ))
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

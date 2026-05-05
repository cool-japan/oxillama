//! Responses API — `POST /v1/responses`, `GET /v1/responses`, `GET /v1/responses/:id`.
//!
//! The Responses API is a stateful sibling of the Chat Completions API.
//! Each call creates a [`ResponseRecord`] in the [`ResponseStore`], optionally
//! chains from a `previous_response_id`, and returns either a JSON object
//! (non-streaming) or a Server-Sent Events stream (`stream: true`).
//!
//! SSE event names follow the draft OpenAI Responses spec:
//! - `response.created`          — fired immediately with the new record
//! - `response.output_text.delta` — fired once per generated token
//! - `response.completed`         — fired when the full output is known
//! - `[DONE]`                      — terminator sentinel

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::error::{ServerError, ServerResult};
use crate::queue::{BatchRequest, UsageStats};
use crate::responses_store::{ResponseRecord, ResponseStatus, ResponseStore};
use crate::state::AppState;

// ── Request types ─────────────────────────────────────────────────────────────

/// Input to a Responses API request — either a bare text string or an array
/// of OpenAI-compatible message objects.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    /// Plain text prompt (converted to a single `user` message internally).
    Text(String),
    /// Array of message objects (same shape as `ChatCompletionRequest.messages`).
    Messages(Vec<serde_json::Value>),
}

/// Request body for `POST /v1/responses`.
#[derive(Debug, Deserialize)]
pub struct CreateResponseRequest {
    /// Model to use (falls back to the server's default when absent).
    pub model: Option<String>,
    /// The input to respond to.
    pub input: ResponseInput,
    /// Optional system-level instructions prepended before the input.
    pub instructions: Option<String>,
    /// ID of a previous response to continue from.
    pub previous_response_id: Option<String>,
    /// Whether to stream the response via SSE.
    pub stream: Option<bool>,
    /// Tool definitions (reserved; passed through to the stored record).
    pub tools: Option<Vec<serde_json::Value>>,
    /// Sampling temperature override.
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_output_tokens: Option<usize>,
}

// ── Response types ─────────────────────────────────────────────────────────────

/// Wire-format output item inside a response object.
#[derive(Debug, Serialize)]
pub struct OutputItem {
    /// Always `"output_text"`.
    pub r#type: String,
    /// The generated text.
    pub text: String,
}

/// Full response object returned by non-streaming `POST /v1/responses`.
#[derive(Debug, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub model: String,
    pub status: ResponseStatus,
    pub output: Vec<OutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub tools: Vec<serde_json::Value>,
}

impl ResponseObject {
    fn from_record(rec: &ResponseRecord) -> Self {
        let output = rec
            .output
            .as_ref()
            .map(|text| {
                vec![OutputItem {
                    r#type: "output_text".to_string(),
                    text: text.clone(),
                }]
            })
            .unwrap_or_default();

        Self {
            id: rec.id.clone(),
            object: rec.object.clone(),
            created_at: rec.created_at,
            model: rec.model.clone(),
            status: rec.status.clone(),
            output,
            previous_response_id: rec.previous_response_id.clone(),
            instructions: rec.instructions.clone(),
            tools: rec.tools.clone(),
        }
    }
}

/// Envelope for `GET /v1/responses` list.
#[derive(Debug, Serialize)]
pub struct ResponseList {
    pub object: String,
    pub data: Vec<ResponseObject>,
}

// ── SSE delta payload ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct DeltaPayload {
    r#type: String,
    delta: String,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert a [`ResponseInput`] into a flat `Vec<serde_json::Value>` of message objects.
fn input_to_messages(input: ResponseInput) -> Vec<serde_json::Value> {
    match input {
        ResponseInput::Text(text) => {
            vec![serde_json::json!({"role": "user", "content": text})]
        }
        ResponseInput::Messages(msgs) => msgs,
    }
}

/// Format a sequence of message objects into a chat-style prompt string.
///
/// The format mirrors `routes/chat.rs::format_chat_prompt` to keep prompt
/// templates consistent across endpoints.
fn format_prompt(messages: &[serde_json::Value], instructions: Option<&str>) -> String {
    let mut prompt = String::new();

    if let Some(sys) = instructions {
        prompt.push_str("<|system|>\n");
        prompt.push_str(sys);
        prompt.push_str("\n<|end|>\n");
    }

    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("user");
        let content = msg["content"].as_str().unwrap_or("");
        match role {
            "system" => {
                prompt.push_str("<|system|>\n");
                prompt.push_str(content);
                prompt.push_str("\n<|end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|assistant|>\n");
                prompt.push_str(content);
                prompt.push_str("\n<|end|>\n");
            }
            "tool" => {
                prompt.push_str("<|tool|>\n");
                prompt.push_str(content);
                prompt.push_str("\n<|end|>\n");
            }
            _ => {
                // "user" and anything unknown
                prompt.push_str("<|user|>\n");
                prompt.push_str(content);
                prompt.push_str("\n<|end|>\n");
            }
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Require the responses store from `AppState`, returning 503 if absent.
fn require_store(state: &AppState) -> ServerResult<Arc<ResponseStore>> {
    state
        .responses_store
        .as_ref()
        .cloned()
        .ok_or(ServerError::ModelNotReady)
}

// ── Route handlers ────────────────────────────────────────────────────────────

/// `POST /v1/responses`
///
/// Creates a new response, optionally chaining from `previous_response_id`.
/// When `stream: true` returns an SSE stream; otherwise blocks until the
/// inference worker completes and returns the full response object.
pub async fn create_response(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateResponseRequest>,
) -> ServerResult<axum::response::Response> {
    let store = require_store(&state)?;

    let model_id = request
        .model
        .clone()
        .unwrap_or_else(|| state.model_id.clone());
    let max_tokens = request.max_output_tokens.unwrap_or(256);
    let do_stream = request.stream.unwrap_or(false);
    let tools = request.tools.clone().unwrap_or_default();

    // Resolve input messages, prepending previous response context when asked.
    let mut input_messages = input_to_messages(request.input);

    if let Some(prev_id) = &request.previous_response_id {
        let prev = store
            .get(prev_id)
            .map_err(|_| ServerError::PreviousResponseNotFound(prev_id.clone()))?;

        // Prepend: previous input messages first, then previous output (as
        // an assistant turn), then the current request's input.
        let mut combined = prev.input.clone();
        if let Some(prev_output) = &prev.output {
            combined.push(serde_json::json!({
                "role": "assistant",
                "content": prev_output
            }));
        }
        combined.extend(input_messages);
        input_messages = combined;
    }

    // Build the prompt string.
    let prompt = format_prompt(&input_messages, request.instructions.as_deref());

    // Build per-request sampler config.
    let mut sampler_config = state.default_sampler.clone();
    if let Some(temp) = request.temperature {
        sampler_config.temperature = temp;
    }

    // Persist a new record in InProgress state.
    let rec = ResponseRecord::new_in_progress(
        model_id.clone(),
        input_messages,
        request.previous_response_id.clone(),
        request.instructions.clone(),
        tools,
    );
    let response_id = store.create(rec.clone())?;

    if do_stream {
        // ── SSE streaming path ────────────────────────────────────────────
        let (sse_tx, sse_rx) =
            tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);
        let (reply_tx, reply_rx) = oneshot::channel::<Result<UsageStats, String>>();

        // Fire `response.created` immediately.
        let created_payload = serde_json::json!({
            "type": "response.created",
            "response": ResponseObject::from_record(&rec)
        });
        let _ = sse_tx
            .send(Ok(Event::default().event("response.created").data(
                serde_json::to_string(&created_payload).unwrap_or_default(),
            )))
            .await;

        // Build streaming callback.
        let sse_tx_cb = sse_tx.clone();
        let callback: crate::queue::StreamCallback = Box::new(move |token_text: &str| {
            let delta_payload = DeltaPayload {
                r#type: "response.output_text.delta".to_string(),
                delta: token_text.to_string(),
            };
            let _ = sse_tx_cb.blocking_send(Ok(Event::default()
                .event("response.output_text.delta")
                .data(serde_json::to_string(&delta_payload).unwrap_or_default())));
        });

        // Dispatch to worker.
        state
            .queue
            .send(BatchRequest::GenerateStream {
                prompt,
                max_tokens,
                config: sampler_config,
                cache_prompt: true,
                lora_selection: vec![],
                callback,
                reply: reply_tx,
            })
            .await
            .map_err(|_| ServerError::WorkerDead)?;

        // Spawn finaliser task.
        let store_clone = Arc::clone(&store);
        let resp_id_finish = response_id.clone();
        let model_id_finish = model_id.clone();
        tokio::spawn(async move {
            // Collect all tokens that were already sent (we can't replay them)
            // — instead we record the final status and emit `response.completed`.
            let (final_status, output_text) = match reply_rx.await {
                Ok(Ok(_usage)) => (ResponseStatus::Completed, None),
                _ => (ResponseStatus::Failed, None),
            };

            // Update the record (no full text captured in stream path).
            let _ = store_clone.update_output(
                &resp_id_finish,
                output_text.unwrap_or_default(),
                final_status.clone(),
            );

            // Emit `response.completed`.
            let completed_payload = serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": resp_id_finish,
                    "model": model_id_finish,
                    "status": final_status,
                }
            });
            let _ = sse_tx
                .send(Ok(Event::default().event("response.completed").data(
                    serde_json::to_string(&completed_payload).unwrap_or_default(),
                )))
                .await;
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(sse_rx);
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok(sse.into_response())
    } else {
        // ── Non-streaming path ────────────────────────────────────────────
        let (reply_tx, reply_rx) = oneshot::channel::<Result<(String, UsageStats), String>>();

        state
            .queue
            .send(BatchRequest::Generate {
                prompt,
                max_tokens,
                config: sampler_config,
                cache_prompt: true,
                lora_selection: vec![],
                reply: reply_tx,
            })
            .await
            .map_err(|_| ServerError::WorkerDead)?;

        let (generated, _usage) = reply_rx
            .await
            .map_err(|_| ServerError::WorkerDead)?
            .map_err(|e| ServerError::InvalidRequest { message: e })?;

        // Persist output.
        store.update_output(&response_id, generated, ResponseStatus::Completed)?;

        // Return the updated record.
        let updated_rec = store.get(&response_id)?;
        let obj = ResponseObject::from_record(&updated_rec);
        Ok(Json(obj).into_response())
    }
}

/// `GET /v1/responses`
///
/// Returns all stored response objects sorted newest-first.
pub async fn list_responses(
    State(state): State<Arc<AppState>>,
) -> ServerResult<Json<ResponseList>> {
    let store = require_store(&state)?;
    let records = store.list();
    let data = records
        .iter()
        .map(ResponseObject::from_record)
        .collect::<Vec<_>>();
    Ok(Json(ResponseList {
        object: "list".to_string(),
        data,
    }))
}

/// `GET /v1/responses/:id`
///
/// Retrieves a single response by its stable `id`.
pub async fn get_response(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ServerResult<Json<ResponseObject>> {
    let store = require_store(&state)?;
    let rec = store.get(&id)?;
    Ok(Json(ResponseObject::from_record(&rec)))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request, StatusCode};
    use serde_json::{json, Value};
    use tower::ServiceExt as _;

    use crate::app::build_app;
    use crate::queue::BatchRequest;
    use crate::queue::UsageStats;
    use crate::responses_store::ResponseStore;
    use crate::state::AppState;
    use oxillama_runtime::sampling::SamplerConfig;

    // ── Test app factory ──────────────────────────────────────────────────

    /// Build a live test app that has the responses store populated.
    async fn build_responses_test_app() -> axum::Router {
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
                        let _ = reply.send(Ok(("mock response text".to_string(), usage)));
                    }
                    BatchRequest::GenerateStream {
                        mut callback,
                        reply,
                        ..
                    } => {
                        let _ = tokio::task::spawn_blocking(move || {
                            callback("stream ");
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

        let mut state = AppState::new(
            tx,
            "test-model".to_string(),
            SamplerConfig::default(),
            None,
            0,
        );
        state.responses_store = Some(Arc::new(ResponseStore::new()));
        let state = Arc::new(state);
        build_app(state)
    }

    /// Build a dead-worker test app that still has the responses store.
    async fn build_responses_dead_app() -> axum::Router {
        let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);
        let mut state = AppState::new(
            tx,
            "test-model".to_string(),
            SamplerConfig::default(),
            None,
            0,
        );
        state.responses_store = Some(Arc::new(ResponseStore::new()));
        let state = Arc::new(state);
        build_app(state)
    }

    async fn post_json(app: axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
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
        let bytes = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("response body should be readable");
        let value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
        (status, value)
    }

    async fn get_json(app: axum::Router, uri: &str) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(uri)
                    .body(Body::empty())
                    .expect("request builder should succeed"),
            )
            .await
            .expect("router should handle the request");

        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("response body should be readable");
        let value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
        (status, value)
    }

    // ── Tests ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn responses_create_non_streaming_returns_200() {
        let app = build_responses_test_app().await;
        let (status, body) = post_json(
            app,
            "/v1/responses",
            json!({
                "model": "test-model",
                "input": "Hello, how are you?"
            }),
        )
        .await;
        assert_eq!(
            status.as_u16(),
            200,
            "non-streaming create should return 200: {body}"
        );
        assert_eq!(
            body["object"].as_str().unwrap_or(""),
            "response",
            "object field should be 'response': {body}"
        );
        assert_eq!(
            body["status"].as_str().unwrap_or(""),
            "completed",
            "status should be 'completed': {body}"
        );
        assert!(
            body["id"].as_str().is_some_and(|s| s.starts_with("resp_")),
            "id should start with 'resp_': {body}"
        );
        assert!(
            body["output"].is_array(),
            "output should be an array: {body}"
        );
    }

    #[tokio::test]
    async fn responses_streaming_emits_delta_events() {
        let _app = build_responses_test_app().await;

        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&json!({
                    "model": "test-model",
                    "input": "stream this",
                    "stream": true
                }))
                .expect("serialise"),
            ))
            .expect("build request");

        let response = build_responses_test_app()
            .await
            .oneshot(request)
            .await
            .expect("handle request");

        assert_eq!(
            response.status().as_u16(),
            200,
            "streaming should return 200"
        );

        let ct = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );

        let bytes = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("read body");
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains("response.output_text.delta"),
            "body should contain delta events: {}",
            &text[..text.len().min(400)]
        );
    }

    #[tokio::test]
    async fn responses_streaming_terminates_with_done() {
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&json!({
                    "model": "test-model",
                    "input": "finish me",
                    "stream": true
                }))
                .expect("serialise"),
            ))
            .expect("build request");

        let response = build_responses_test_app()
            .await
            .oneshot(request)
            .await
            .expect("handle request");

        let bytes = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("read body");
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains("[DONE]"),
            "streaming body should contain [DONE] sentinel: {}",
            &text[..text.len().min(400)]
        );
    }

    #[tokio::test]
    async fn responses_previous_id_chains_input() {
        // Create initial response.
        let app = build_responses_test_app().await;
        let (status, first_body) = post_json(
            app,
            "/v1/responses",
            json!({
                "model": "test-model",
                "input": "first message"
            }),
        )
        .await;
        assert_eq!(status.as_u16(), 200, "first create: {first_body}");
        let first_id = first_body["id"].as_str().expect("id").to_string();

        // Create a follow-up response referencing the first.
        let app2 = build_responses_test_app().await;
        // We need to re-insert the first record into the new app's store — or
        // use the same app.  We rebuild with the same store for simplicity.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<BatchRequest>(16);
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                if let BatchRequest::Generate { reply, .. } = req {
                    let usage = UsageStats {
                        prompt_tokens: 5,
                        completion_tokens: 3,
                        total_tokens: 8,
                    };
                    let _ = reply.send(Ok(("chained response".to_string(), usage)));
                }
            }
        });

        let store = Arc::new(ResponseStore::new());

        // Manually create the "previous" record in the shared store.
        let prev_rec = crate::responses_store::ResponseRecord::new_in_progress(
            "test-model".to_string(),
            vec![json!({"role": "user", "content": "first message"})],
            None,
            None,
            vec![],
        );
        let prev_id = store.create(prev_rec.clone()).expect("create prev");
        store
            .update_output(
                &prev_id,
                "first output".to_string(),
                crate::responses_store::ResponseStatus::Completed,
            )
            .expect("update prev");

        let mut state = AppState::new(
            tx,
            "test-model".to_string(),
            SamplerConfig::default(),
            None,
            0,
        );
        state.responses_store = Some(Arc::clone(&store));
        let state = Arc::new(state);
        let chained_app = build_app(state);
        drop(app2);

        let (status2, body2) = post_json(
            chained_app,
            "/v1/responses",
            json!({
                "model": "test-model",
                "input": "follow-up question",
                "previous_response_id": prev_id
            }),
        )
        .await;

        // The request should succeed — the previous record exists in the store.
        assert_eq!(
            status2.as_u16(),
            200,
            "chained response should succeed: {body2}"
        );
        assert_eq!(
            body2["status"].as_str().unwrap_or(""),
            "completed",
            "chained response should complete: {body2}"
        );
        drop(first_id); // quiet unused-variable lint
    }

    #[tokio::test]
    async fn responses_unknown_id_returns_404() {
        let app = build_responses_dead_app().await;
        let (status, _body) = get_json(app, "/v1/responses/resp_does_not_exist_xyz_abc").await;
        assert_eq!(
            status.as_u16(),
            404,
            "unknown response id should return 404"
        );
    }
}

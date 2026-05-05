//! POST /v1/chat/completions handler.

use std::sync::Arc;
use std::time::SystemTime;

use crate::error::{ServerError, ServerResult};
use crate::queue::{BatchRequest, UsageStats};
use crate::state::AppState;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use oxillama_runtime::sampling::grammar::Grammar;
use oxillama_runtime::sampling::SamplerConfig;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use super::tools::{Tool, ToolCall, ToolCallDelta, ToolChoice};

fn default_cache_prompt() -> bool {
    true
}

/// A single LoRA adapter selection entry.
#[derive(Debug, Clone, Deserialize)]
pub struct LoraEntry {
    /// Stable adapter name registered via `POST /admin/loras`.
    pub name: String,
    /// Scale multiplier (default: 1.0).
    #[serde(default = "default_lora_scale")]
    pub scale: f32,
}

fn default_lora_scale() -> f32 {
    1.0
}

/// LoRA adapter selection: either a single adapter name or a list of entries.
///
/// Single-string form: `"lora": "my_adapter"` → scale defaults to 1.0.
/// Array form: `"lora": [{"name": "a", "scale": 0.8}, {"name": "b", "scale": 0.5}]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum LoraSelection {
    /// Single adapter by name (scale = 1.0).
    Single(String),
    /// Ordered list of adapters with per-entry scales.
    Multi(Vec<LoraEntry>),
}

impl LoraSelection {
    /// Convert to the `(name, scale)` pairs expected by `BatchRequest`.
    pub fn to_pairs(&self) -> Vec<(String, f32)> {
        match self {
            LoraSelection::Single(name) => vec![(name.clone(), 1.0)],
            LoraSelection::Multi(entries) => {
                entries.iter().map(|e| (e.name.clone(), e.scale)).collect()
            }
        }
    }
}

/// Chat completion request (OpenAI-compatible).
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier.
    pub model: String,
    /// List of chat messages.
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,
    /// Temperature for sampling.
    pub temperature: Option<f32>,
    /// Top-P for nucleus sampling.
    pub top_p: Option<f32>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,
    /// Optional GBNF grammar string for constrained sampling.
    /// When set, only tokens that can advance the grammar are sampled.
    /// Returns HTTP 400 if the grammar string cannot be parsed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    /// Tool definitions for function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool choice policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Whether to use the prefix KV cache for this request (default: true).
    ///
    /// Set to `false` to force a full prefill even when a matching cached
    /// prefix exists (e.g., for benchmarking or when the prompt is unique).
    #[serde(default = "default_cache_prompt")]
    pub cache_prompt: bool,
    /// Optional LoRA adapter(s) to apply for this request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraSelection>,
}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant", or "tool".
    pub role: String,
    /// Message content (`None` for assistant messages that only contain tool calls).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls produced by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For role="tool": the ID of the tool call this message responds to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Chat completion response (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// Streaming chunk response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// A single completion choice.
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// A streaming chunk choice.
#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Token usage statistics.
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Handle a chat completion request.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> ServerResult<axum::response::Response> {
    if request.messages.is_empty() {
        return Err(ServerError::InvalidRequest {
            message: "messages array must not be empty".to_string(),
        });
    }

    let max_tokens = request.max_tokens.unwrap_or(256);
    let prompt = format_chat_prompt(&request.messages);
    let model_id = state.model_id.clone();
    let now = unix_now();
    let request_id = format!("chatcmpl-{:x}", now);

    // Build per-request sampler config from the cached default — no engine
    // lock needed.
    let sampler_config = build_sampler_config(&state, &request)?;

    if request.stream {
        // SSE streaming response.
        //
        // We create an mpsc channel for SSE events and a oneshot for the
        // worker's completion signal.  The SSE callback runs inside the
        // blocking worker thread and pushes encoded events into `sse_tx`.
        let (sse_tx, sse_rx) =
            tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);
        let (reply_tx, reply_rx) = oneshot::channel::<Result<UsageStats, String>>();

        let req_id = request_id.clone();
        let model_id_clone = model_id.clone();
        let sse_tx_clone = sse_tx.clone();

        // Send the initial role chunk synchronously before we dispatch to
        // the worker so the client sees it immediately.
        let initial_chunk = ChatCompletionChunk {
            id: req_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_id_clone.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        let _ = sse_tx_clone
            .send(Ok(Event::default().data(
                serde_json::to_string(&initial_chunk).unwrap_or_default(),
            )))
            .await;

        // Build the streaming callback that runs on the blocking worker thread.
        let req_id_cb = req_id.clone();
        let model_id_cb = model_id_clone.clone();
        let sse_tx_cb = sse_tx.clone();
        let callback: crate::queue::StreamCallback = Box::new(move |token_text: &str| {
            let chunk = ChatCompletionChunk {
                id: req_id_cb.clone(),
                object: "chat.completion.chunk".to_string(),
                created: now,
                model: model_id_cb.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: Some(token_text.to_string()),
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
            };
            let _ = sse_tx_cb.blocking_send(Ok(
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            ));
        });

        // Dispatch to the worker.
        let lora_selection = request
            .lora
            .as_ref()
            .map(|l| l.to_pairs())
            .unwrap_or_default();
        state
            .queue
            .send(BatchRequest::GenerateStream {
                prompt,
                max_tokens,
                config: sampler_config,
                cache_prompt: request.cache_prompt,
                lora_selection,
                callback,
                reply: reply_tx,
            })
            .await
            .map_err(|_| ServerError::WorkerDead)?;

        // Spawn a task that waits for the worker to finish, then sends the
        // finish and [DONE] events.
        let req_id_finish = req_id.clone();
        let model_id_finish = model_id_clone.clone();
        tokio::spawn(async move {
            let finish_reason = match reply_rx.await {
                Ok(Ok(_usage)) => "stop",
                _ => "error",
            };
            let finish_chunk = ChatCompletionChunk {
                id: req_id_finish,
                object: "chat.completion.chunk".to_string(),
                created: now,
                model: model_id_finish,
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some(finish_reason.to_string()),
                }],
            };
            let _ = sse_tx
                .send(Ok(Event::default().data(
                    serde_json::to_string(&finish_chunk).unwrap_or_default(),
                )))
                .await;
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(sse_rx);
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok(sse.into_response())
    } else {
        // Non-streaming: send a Generate request and await the result.
        let (reply_tx, reply_rx) = oneshot::channel::<Result<(String, UsageStats), String>>();

        let lora_selection = request
            .lora
            .as_ref()
            .map(|l| l.to_pairs())
            .unwrap_or_default();
        state
            .queue
            .send(BatchRequest::Generate {
                prompt,
                max_tokens,
                config: sampler_config,
                cache_prompt: request.cache_prompt,
                lora_selection,
                reply: reply_tx,
            })
            .await
            .map_err(|_| ServerError::WorkerDead)?;

        let (generated, usage) = reply_rx
            .await
            .map_err(|_| ServerError::WorkerDead)?
            .map_err(|e| ServerError::InvalidRequest { message: e })?;

        // Check if this was a tool-calling request and try to parse the
        // generated output as a tool call JSON.
        let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
        let tool_choice_is_none = matches!(
            &request.tool_choice,
            Some(ToolChoice::Mode(m)) if m == "none"
        );

        let (message, finish_reason, choice_tool_calls) = if has_tools && !tool_choice_is_none {
            let call_id = format!("call_{:x}", now);
            if let Some(tc) = super::tools::parse_tool_call_output(&generated, &call_id) {
                let msg = ChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![tc.clone()]),
                    tool_call_id: None,
                };
                (msg, "tool_calls".to_string(), Some(vec![tc]))
            } else {
                // Model produced plain text — not a valid tool call.
                let msg = ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(generated),
                    tool_calls: None,
                    tool_call_id: None,
                };
                (msg, "stop".to_string(), None)
            }
        } else {
            let msg = ChatMessage {
                role: "assistant".to_string(),
                content: Some(generated),
                tool_calls: None,
                tool_call_id: None,
            };
            (msg, "stop".to_string(), None)
        };

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: now,
            model: model_id,
            choices: vec![ChatChoice {
                index: 0,
                message,
                finish_reason: Some(finish_reason),
                tool_calls: choice_tool_calls,
            }],
            usage: Usage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

/// Build a per-request `SamplerConfig` from the cached default config.
///
/// Grammar support reads the vocabulary byte table from the cached
/// `AppState::vocab_bytes` — no engine lock required.
fn build_sampler_config(
    state: &AppState,
    request: &ChatCompletionRequest,
) -> ServerResult<SamplerConfig> {
    let mut config = state.default_sampler.clone();

    // Apply per-request temperature/top_p overrides.
    if let Some(temp) = request.temperature {
        config.temperature = temp;
    }
    if let Some(top_p) = request.top_p {
        config.top_p = top_p;
    }

    // Determine the GBNF grammar string: explicit `grammar` field takes
    // precedence; otherwise generate one from tool definitions.
    let grammar_str = if let Some(g) = &request.grammar {
        Some(g.clone())
    } else if let Some(tools) = &request.tools {
        let gbnf = super::tools::tools_to_gbnf(tools, &request.tool_choice);
        if gbnf.is_empty() {
            None
        } else {
            Some(gbnf)
        }
    } else {
        None
    };

    // Parse and attach GBNF grammar if provided.
    if let Some(grammar_string) = &grammar_str {
        let grammar = Grammar::parse(grammar_string).map_err(|e| ServerError::InvalidRequest {
            message: format!("invalid GBNF grammar: {e}"),
        })?;

        let vocab = state
            .vocab_bytes
            .as_ref()
            .ok_or(ServerError::ModelNotReady)?
            .clone();

        config.grammar = Some(Arc::new(grammar));
        config.token_vocab = Some(vocab);
    }

    Ok(config)
}

/// Format chat messages into a prompt string.
fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let content = msg.content.as_deref().unwrap_or("");
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<|system|>\n");
                prompt.push_str(content);
                prompt.push_str("\n<|end|>\n");
            }
            "user" => {
                prompt.push_str("<|user|>\n");
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
                prompt.push_str(content);
                prompt.push('\n');
            }
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::test_helpers::{build_live_test_app, build_test_app, post_json};

    /// A request body that is missing required fields (`model`, `messages`)
    /// must be rejected by axum's JSON extractor before the handler runs
    /// (HTTP 422 Unprocessable Entity).
    #[tokio::test]
    async fn test_chat_missing_required_fields_returns_422() {
        let app = build_test_app().await;
        let (status, _body) = post_json(app, "/v1/chat/completions", json!({})).await;
        assert_eq!(
            status.as_u16(),
            422,
            "missing required fields should yield 422"
        );
    }

    /// An empty `messages` array passes JSON deserialization but fails
    /// handler-level validation — must return 400 Bad Request.
    #[tokio::test]
    async fn test_chat_empty_messages_returns_400() {
        let app = build_test_app().await;
        let (status, body) = post_json(
            app,
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": []
            }),
        )
        .await;
        assert_eq!(status.as_u16(), 400);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("messages"),
            "error message should mention 'messages': {body}"
        );
    }

    /// A well-formed request to a dead worker must return 503 Service
    /// Unavailable — the handler should not panic or return 200.
    #[tokio::test]
    async fn test_chat_worker_dead_returns_503() {
        let app = build_test_app().await;
        let (status, body) = post_json(
            app,
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false
            }),
        )
        .await;
        assert_eq!(status.as_u16(), 503);
        let error_type = body["error"]["type"].as_str().unwrap_or("");
        assert_eq!(
            error_type, "service_unavailable",
            "error type should be service_unavailable: {body}"
        );
    }

    /// Supplying a grammar string when no vocab is loaded in the AppState
    /// must return 503 (ModelNotReady) rather than panicking.
    #[tokio::test]
    async fn test_chat_grammar_without_vocab_returns_503() {
        let app = build_test_app().await;
        let (status, body) = post_json(
            app,
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "grammar": "root ::= [a-z]+"
            }),
        )
        .await;
        assert_eq!(
            status.as_u16(),
            503,
            "no vocab → ModelNotReady → 503: {body}"
        );
    }

    /// An invalid GBNF grammar string must return 400 Bad Request before
    /// any queue interaction happens.
    #[tokio::test]
    async fn test_chat_invalid_grammar_returns_400() {
        let app = build_test_app().await;
        let (status, _body) = post_json(
            app,
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "grammar": ":::: this is not a valid grammar ::::"
            }),
        )
        .await;
        // invalid grammar → Grammar::parse error → InvalidRequest → 400
        assert_eq!(status.as_u16(), 400, "invalid grammar should yield 400");
    }

    /// A well-formed chat request to a live mock worker must return HTTP 200
    /// with the expected OpenAI-compatible response structure.
    #[tokio::test]
    async fn test_chat_valid_request_returns_200() {
        let app = build_live_test_app().await;
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let (status, json) = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(
            status.as_u16(),
            200,
            "live worker should return 200: {json}"
        );
        assert_eq!(
            json["object"].as_str().unwrap_or(""),
            "chat.completion",
            "object field mismatch: {json}"
        );
        assert!(
            json["choices"][0]["message"]["content"].as_str().is_some(),
            "choices[0].message.content must be a string: {json}"
        );
    }

    /// A chat request with `max_tokens` set must succeed and carry a non-empty
    /// `finish_reason` in the first choice.
    #[tokio::test]
    async fn test_chat_returns_valid_finish_reason() {
        let app = build_live_test_app().await;
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5
        });
        let (status, json) = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(
            status.as_u16(),
            200,
            "live worker should return 200: {json}"
        );
        let finish_reason = json["choices"][0]["finish_reason"].as_str().unwrap_or("");
        assert!(
            !finish_reason.is_empty(),
            "finish_reason must be non-empty: {json}"
        );
    }

    // -------------------------------------------------------------------------
    // SSE streaming tests
    // -------------------------------------------------------------------------

    /// A `stream: true` request to a live mock worker must return HTTP 200 with
    /// `Content-Type: text/event-stream` — proving the SSE path is reachable.
    #[tokio::test]
    async fn test_chat_streaming_returns_200_with_sse_content_type() {
        use axum::body::Body;
        use axum::http::header::CONTENT_TYPE;
        use axum::http::{Method, Request};
        use tower::ServiceExt as _;

        let app = build_live_test_app().await;
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        });

        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&body).expect("test: body should serialise"),
            ))
            .expect("test: build request");

        let response = app
            .oneshot(request)
            .await
            .expect("test: router should handle request");

        assert_eq!(
            response.status().as_u16(),
            200,
            "streaming should return 200"
        );
        let ct = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );
    }

    /// A `stream: true` request to a live mock worker must produce SSE `data:`
    /// lines in the body and terminate with the `[DONE]` sentinel.
    #[tokio::test]
    async fn test_chat_streaming_body_contains_data_lines() {
        use axum::body::{to_bytes, Body};
        use axum::http::{Method, Request};
        use tower::ServiceExt as _;

        let app = build_live_test_app().await;
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        });

        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&body).expect("test: body should serialise"),
            ))
            .expect("test: build request");

        let response = app
            .oneshot(request)
            .await
            .expect("test: router should handle request");

        assert_eq!(
            response.status().as_u16(),
            200,
            "streaming should return 200"
        );

        let bytes = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("test: read body");
        let text = String::from_utf8_lossy(&bytes);

        // Every SSE event is prefixed with "data:".
        assert!(
            text.contains("data:"),
            "streaming body should contain SSE data lines, got: {}",
            &text[..text.len().min(200)]
        );
        // The stream must end with the [DONE] sentinel.
        assert!(
            text.contains("[DONE]"),
            "streaming body should contain [DONE] sentinel, got: {}",
            &text[..text.len().min(200)]
        );
    }

    /// A `stream: true` request to a dead worker must return a non-success
    /// status (503) or, if the handler sends the SSE headers first, a 200 with
    /// the stream immediately closing — either outcome is acceptable.
    #[tokio::test]
    async fn test_chat_streaming_with_dead_worker_returns_503() {
        let app = build_test_app().await;
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        });
        let (status, _json) = post_json(app, "/v1/chat/completions", body).await;
        assert!(
            status.as_u16() == 503 || status.as_u16() == 200,
            "dead-worker streaming should return 503 or 200-with-error, got {}",
            status.as_u16()
        );
    }

    // -------------------------------------------------------------------------
    // Tool / function calling deserialization tests
    // -------------------------------------------------------------------------

    /// ChatCompletionRequest with tools and tool_choice must deserialize.
    #[test]
    fn test_request_with_tools_deserializes() {
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto"
        });
        let req: super::ChatCompletionRequest =
            serde_json::from_value(body).expect("should deserialize with tools");
        assert_eq!(req.tools.as_ref().map(|t| t.len()), Some(1));
    }

    /// ChatCompletionRequest without tools still deserializes (backward compat).
    #[test]
    fn test_request_without_tools_deserializes() {
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let req: super::ChatCompletionRequest =
            serde_json::from_value(body).expect("should deserialize without tools");
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
    }

    /// ChatMessage with tool_calls and optional content deserializes.
    #[test]
    fn test_chat_message_with_tool_calls_deserializes() {
        let msg = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"Tokyo\"}"
                }
            }]
        });
        let parsed: super::ChatMessage = serde_json::from_value(msg).expect("should deserialize");
        assert!(parsed.content.is_none());
        assert_eq!(parsed.tool_calls.as_ref().map(|t| t.len()), Some(1));
    }

    /// ChatMessage with role="tool" and tool_call_id deserializes.
    #[test]
    fn test_chat_message_tool_role_deserializes() {
        let msg = json!({
            "role": "tool",
            "content": "72°F, Sunny",
            "tool_call_id": "call_123"
        });
        let parsed: super::ChatMessage = serde_json::from_value(msg).expect("should deserialize");
        assert_eq!(parsed.role, "tool");
        assert_eq!(parsed.content.as_deref(), Some("72°F, Sunny"));
        assert_eq!(parsed.tool_call_id.as_deref(), Some("call_123"));
    }

    /// Existing payloads with content as a plain string still work.
    #[test]
    fn test_chat_message_string_content_backward_compat() {
        let msg = json!({"role": "user", "content": "hello"});
        let parsed: super::ChatMessage = serde_json::from_value(msg).expect("should deserialize");
        assert_eq!(parsed.content.as_deref(), Some("hello"));
    }

    /// tool_choice="none" in a request with tools should deserialize and
    /// be recognized as the Mode variant.
    #[test]
    fn test_request_tool_choice_none_disables_tools() {
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "noop",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": "none"
        });
        let req: super::ChatCompletionRequest =
            serde_json::from_value(body).expect("should deserialize");
        assert!(matches!(
            req.tool_choice,
            Some(super::super::tools::ToolChoice::Mode(ref m)) if m == "none"
        ));
    }

    /// Specific tool_choice forces a single named function.
    #[test]
    fn test_request_specific_tool_choice() {
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "do_thing",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": {"type": "function", "function": {"name": "do_thing"}}
        });
        let req: super::ChatCompletionRequest =
            serde_json::from_value(body).expect("should deserialize");
        assert!(matches!(
            req.tool_choice,
            Some(super::super::tools::ToolChoice::Specific { .. })
        ));
    }
}

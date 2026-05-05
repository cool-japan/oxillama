//! POST /v1/completions handler.

use std::sync::Arc;
use std::time::SystemTime;

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::error::{ServerError, ServerResult};
use crate::queue::{BatchRequest, UsageStats};
use crate::state::AppState;

/// Text completion request (OpenAI-compatible).
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier.
    pub model: String,
    /// Input prompt text.
    pub prompt: String,
    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,
    /// Temperature for sampling.
    pub temperature: Option<f32>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,
}

/// Text completion response.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

/// A single completion choice.
#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Token usage statistics for the completions endpoint.
#[derive(Debug, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Handle a text completion request.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> ServerResult<Json<CompletionResponse>> {
    let max_tokens = request.max_tokens.unwrap_or(256);
    let prompt = request.prompt.clone();
    let model_id = state.model_id.clone();

    // Build a sampler config from the cached default, applying any
    // per-request temperature override.
    let mut config = state.default_sampler.clone();
    if let Some(temp) = request.temperature {
        config.temperature = temp;
    }

    let (reply_tx, reply_rx) = oneshot::channel::<Result<(String, UsageStats), String>>();

    state
        .queue
        .send(BatchRequest::Generate {
            prompt,
            max_tokens,
            config,
            cache_prompt: true,
            lora_selection: vec![],
            reply: reply_tx,
        })
        .await
        .map_err(|_| ServerError::WorkerDead)?;

    let (generated, usage) = reply_rx
        .await
        .map_err(|_| ServerError::WorkerDead)?
        .map_err(|e| ServerError::InvalidRequest { message: e })?;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let response = CompletionResponse {
        id: format!("cmpl-{:x}", now),
        object: "text_completion".to_string(),
        created: now,
        model: model_id,
        choices: vec![CompletionChoice {
            index: 0,
            text: generated,
            finish_reason: Some("stop".to_string()),
        }],
        usage: CompletionUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        },
    };

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::test_helpers::{build_live_test_app, build_test_app, post_json};

    /// A request body missing required fields (`model`, `prompt`) must be
    /// rejected with HTTP 422 by axum's JSON extractor.
    #[tokio::test]
    async fn test_completions_missing_required_fields_returns_422() {
        let app = build_test_app().await;
        let (status, _body) = post_json(app, "/v1/completions", json!({})).await;
        assert_eq!(
            status.as_u16(),
            422,
            "missing required fields should yield 422"
        );
    }

    /// A request with only `model` but no `prompt` must also fail validation.
    #[tokio::test]
    async fn test_completions_missing_prompt_returns_422() {
        let app = build_test_app().await;
        let (status, _body) =
            post_json(app, "/v1/completions", json!({"model": "test-model"})).await;
        assert_eq!(status.as_u16(), 422);
    }

    /// A well-formed request with a dead worker must return 503.
    #[tokio::test]
    async fn test_completions_worker_dead_returns_503() {
        let app = build_test_app().await;
        let (status, body) = post_json(
            app,
            "/v1/completions",
            json!({
                "model": "test-model",
                "prompt": "Once upon a time"
            }),
        )
        .await;
        assert_eq!(status.as_u16(), 503);
        assert_eq!(
            body["error"]["type"].as_str().unwrap_or(""),
            "service_unavailable"
        );
    }

    /// Temperature override is accepted by the deserializer (no 400 from
    /// unknown field rejection) — validation succeeds and the error comes
    /// from the dead worker, not from request parsing.
    #[tokio::test]
    async fn test_completions_with_temperature_override_fails_on_worker_not_parsing() {
        let app = build_test_app().await;
        let (status, _body) = post_json(
            app,
            "/v1/completions",
            json!({
                "model": "test-model",
                "prompt": "hi",
                "temperature": 0.7,
                "max_tokens": 32
            }),
        )
        .await;
        // Worker is dead → 503; request parsing must not have failed earlier
        assert_eq!(status.as_u16(), 503);
    }

    /// A well-formed completions request to a live mock worker must return
    /// HTTP 200 with the expected OpenAI-compatible `text_completion` object.
    #[tokio::test]
    async fn test_completions_valid_request_returns_200() {
        let app = build_live_test_app().await;
        let body = json!({
            "model": "test",
            "prompt": "hello world"
        });
        let (status, json) = post_json(app, "/v1/completions", body).await;
        assert_eq!(
            status.as_u16(),
            200,
            "live worker should return 200: {json}"
        );
        assert_eq!(
            json["object"].as_str().unwrap_or(""),
            "text_completion",
            "object field mismatch: {json}"
        );
        assert!(
            json["choices"][0]["text"].as_str().is_some(),
            "choices[0].text must be a string: {json}"
        );
    }

    /// Completions with `max_tokens` set must succeed with a live worker.
    #[tokio::test]
    async fn test_completions_with_max_tokens() {
        let app = build_live_test_app().await;
        let body = json!({
            "model": "test",
            "prompt": "test",
            "max_tokens": 10
        });
        let (status, json) = post_json(app, "/v1/completions", body).await;
        assert_eq!(
            status.as_u16(),
            200,
            "live worker + max_tokens should return 200: {json}"
        );
    }
}

//! POST /v1/embeddings handler.
//!
//! Implements the OpenAI `/v1/embeddings` API: tokenizes each input text,
//! runs the transformer forward pass up to the final RMSNorm (skipping the
//! LM-head projection), and returns L2-normalised `hidden_size`-dimensional
//! vectors.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::error::{ServerError, ServerResult};
use crate::queue::BatchRequest;
use crate::state::AppState;

/// Embedding request (OpenAI-compatible).
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// Model identifier.
    pub model: String,
    /// Input text(s) to embed.
    pub input: EmbeddingInput,
}

/// Input can be a single string or a list of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text input.
    Single(String),
    /// Batch of text inputs.
    Batch(Vec<String>),
}

/// Embedding response.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    /// Object type (`"list"`).
    pub object: String,
    /// Embedding data, one entry per input.
    pub data: Vec<EmbeddingData>,
    /// Model used.
    pub model: String,
    /// Token usage statistics.
    pub usage: EmbeddingUsage,
}

/// A single embedding result.
#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    /// Object type (`"embedding"`).
    pub object: String,
    /// Index in the input batch.
    pub index: usize,
    /// The L2-normalised embedding vector (length = model hidden_size).
    pub embedding: Vec<f32>,
}

/// Token usage statistics for an embedding request.
#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    /// Total prompt tokens across all inputs.
    pub prompt_tokens: usize,
    /// Same as `prompt_tokens` (embeddings have no completion tokens).
    pub total_tokens: usize,
}

/// Handle a POST /v1/embeddings request.
///
/// Sends one `BatchRequest::Embed` per input text through the inference
/// queue and awaits each result in sequence.  The worker serialises all
/// requests so they cannot race on the KV cache.
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> ServerResult<Json<EmbeddingResponse>> {
    let inputs: Vec<String> = match request.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Batch(v) => v,
    };

    if inputs.is_empty() {
        return Err(ServerError::InvalidRequest {
            message: "input must contain at least one string".to_string(),
        });
    }

    let model_id = state.model_id.clone();
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0usize;

    for (idx, text) in inputs.into_iter().enumerate() {
        // Approximate token count (whitespace-split word count as proxy).
        total_tokens += text.split_whitespace().count().max(1);

        let (reply_tx, reply_rx) = oneshot::channel::<Result<Vec<f32>, String>>();

        state
            .queue
            .send(BatchRequest::Embed {
                text,
                reply: reply_tx,
            })
            .await
            .map_err(|_| ServerError::WorkerDead)?;

        let embedding = reply_rx
            .await
            .map_err(|_| ServerError::WorkerDead)?
            .map_err(|e| ServerError::InvalidRequest { message: e })?;

        data.push(EmbeddingData {
            object: "embedding".to_string(),
            index: idx,
            embedding,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: model_id,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::test_helpers::{build_live_test_app, build_test_app, post_json};

    /// A request missing the required `input` field must be rejected with
    /// HTTP 422 by axum's JSON extractor.
    #[tokio::test]
    async fn test_embeddings_missing_input_returns_422() {
        let app = build_test_app().await;
        let (status, _body) =
            post_json(app, "/v1/embeddings", json!({"model": "test-model"})).await;
        assert_eq!(status.as_u16(), 422, "missing input field should yield 422");
    }

    /// A completely empty request body must also fail deserialization.
    #[tokio::test]
    async fn test_embeddings_empty_body_returns_422() {
        let app = build_test_app().await;
        let (status, _body) = post_json(app, "/v1/embeddings", json!({})).await;
        assert_eq!(status.as_u16(), 422);
    }

    /// A single-string input with a dead worker must return 503.
    #[tokio::test]
    async fn test_embeddings_single_input_worker_dead_returns_503() {
        let app = build_test_app().await;
        let (status, body) = post_json(
            app,
            "/v1/embeddings",
            json!({
                "model": "test-model",
                "input": "hello world"
            }),
        )
        .await;
        assert_eq!(status.as_u16(), 503);
        assert_eq!(
            body["error"]["type"].as_str().unwrap_or(""),
            "service_unavailable"
        );
    }

    /// A batch array input with a dead worker must also return 503 (the first
    /// item exhausts the channel send, returning WorkerDead before other
    /// items are processed).
    #[tokio::test]
    async fn test_embeddings_batch_input_worker_dead_returns_503() {
        let app = build_test_app().await;
        let (status, body) = post_json(
            app,
            "/v1/embeddings",
            json!({
                "model": "test-model",
                "input": ["foo", "bar", "baz"]
            }),
        )
        .await;
        assert_eq!(
            status.as_u16(),
            503,
            "batch dead worker should be 503: {body}"
        );
    }

    /// An empty batch array passes deserialization but is rejected by the
    /// handler with HTTP 400 (at least one string required).
    #[tokio::test]
    async fn test_embeddings_empty_batch_returns_400() {
        let app = build_test_app().await;
        let (status, body) = post_json(
            app,
            "/v1/embeddings",
            json!({
                "model": "test-model",
                "input": []
            }),
        )
        .await;
        assert_eq!(status.as_u16(), 400, "empty batch should yield 400: {body}");
    }

    /// A single-string input to a live mock worker must return HTTP 200 with
    /// the expected OpenAI-compatible embeddings response structure.
    #[tokio::test]
    async fn test_embeddings_single_text_returns_200() {
        let app = build_live_test_app().await;
        let body = json!({
            "model": "test",
            "input": "hello world"
        });
        let (status, json) = post_json(app, "/v1/embeddings", body).await;
        assert_eq!(
            status.as_u16(),
            200,
            "live worker should return 200: {json}"
        );
        assert_eq!(
            json["object"].as_str().unwrap_or(""),
            "list",
            "object field must be 'list': {json}"
        );
        let embedding = json["data"][0]["embedding"]
            .as_array()
            .expect("test: data[0].embedding must be an array");
        assert!(!embedding.is_empty(), "embedding vector must not be empty");
    }

    /// A batch of two strings must return HTTP 200 with two entries in `data`.
    #[tokio::test]
    async fn test_embeddings_batch_returns_200() {
        let app = build_live_test_app().await;
        let body = json!({
            "model": "test",
            "input": ["hello", "world"]
        });
        let (status, json) = post_json(app, "/v1/embeddings", body).await;
        assert_eq!(
            status.as_u16(),
            200,
            "live worker batch should return 200: {json}"
        );
        let data = json["data"]
            .as_array()
            .expect("test: data field must be an array");
        assert_eq!(
            data.len(),
            2,
            "two inputs must produce two embeddings: {json}"
        );
    }

    /// The embeddings response must carry the `model` field echoing the
    /// model identifier from `AppState`.
    #[tokio::test]
    async fn test_embeddings_response_has_model_field() {
        let app = build_live_test_app().await;
        let body = json!({
            "model": "test-model",
            "input": "test"
        });
        let (status, json) = post_json(app, "/v1/embeddings", body).await;
        assert_eq!(
            status.as_u16(),
            200,
            "live worker should return 200: {json}"
        );
        // AppState is built with "test-model" as the model_id, regardless of
        // what the request sends.
        assert_eq!(
            json["model"].as_str().unwrap_or(""),
            "test-model",
            "model field must match AppState model_id: {json}"
        );
    }
}

//! Request queue types for the continuous-batching inference worker.
//!
//! Instead of each HTTP handler holding the engine mutex directly, every
//! handler constructs a [`BatchRequest`] and sends it through a
//! `tokio::sync::mpsc::Sender`.  A single background worker receives these
//! requests one at a time and drives the `InferenceEngine`, eliminating
//! mutex contention across concurrent requests.

use std::sync::Arc;

use oxillama_runtime::sampling::SamplerConfig;
use tokio::sync::oneshot;

/// Vocabulary byte table: maps token ID to its UTF-8 byte sequence.
///
/// Used for grammar-constrained sampling.  Wrapped in `Arc` so it can be
/// cheaply shared between `AppState` and individual `SamplerConfig` instances.
pub type VocabBytes = Arc<Vec<(u32, Vec<u8>)>>;

/// Token usage statistics for a generation request.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UsageStats {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub completion_tokens: usize,
    /// Total tokens (prompt + completion).
    pub total_tokens: usize,
}

/// Callback invoked for each generated token during streaming.
///
/// The closure runs inside the blocking worker thread, so calling
/// `tokio::sync::mpsc::Sender::blocking_send` from within it is safe.
pub type StreamCallback = Box<dyn FnMut(&str) + Send>;

/// A single inference request dispatched to the worker task.
pub enum BatchRequest {
    /// Non-streaming generation: prompt → full response string.
    Generate {
        /// The formatted prompt to generate from.
        prompt: String,
        /// Maximum number of tokens to generate.
        max_tokens: usize,
        /// Per-request sampler configuration.
        config: SamplerConfig,
        /// Channel to send the result back to the caller.
        reply: oneshot::Sender<Result<(String, UsageStats), String>>,
    },

    /// Streaming generation: invokes `callback` for every decoded token.
    GenerateStream {
        /// The formatted prompt to generate from.
        prompt: String,
        /// Maximum number of tokens to generate.
        max_tokens: usize,
        /// Per-request sampler configuration.
        config: SamplerConfig,
        /// Called with each token text inside the blocking worker thread.
        callback: StreamCallback,
        /// Channel that receives `Ok(UsageStats)` once generation is complete, or
        /// `Err(message)` on failure.
        reply: oneshot::Sender<Result<UsageStats, String>>,
    },

    /// Embedding computation: text → L2-normalised vector.
    Embed {
        /// The text to embed.
        text: String,
        /// Channel to return the embedding vector (or an error message).
        reply: oneshot::Sender<Result<Vec<f32>, String>>,
    },
}

// Implement Debug manually because StreamCallback is not Debug.
impl std::fmt::Debug for BatchRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchRequest::Generate {
                prompt, max_tokens, ..
            } => f
                .debug_struct("Generate")
                .field("prompt_len", &prompt.len())
                .field("max_tokens", max_tokens)
                .finish(),
            BatchRequest::GenerateStream {
                prompt, max_tokens, ..
            } => f
                .debug_struct("GenerateStream")
                .field("prompt_len", &prompt.len())
                .field("max_tokens", max_tokens)
                .finish(),
            BatchRequest::Embed { text, .. } => f
                .debug_struct("Embed")
                .field("text_len", &text.len())
                .finish(),
        }
    }
}

/// Metadata about the loaded model, cached at startup so route handlers do
/// not need to hold a reference to the (now moved) engine.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    /// Default sampler configuration from the engine config.
    pub default_sampler: SamplerConfig,
    /// Vocabulary byte table for grammar-constrained sampling.
    ///
    /// `None` when no tokenizer is loaded (should not happen at serve time).
    pub vocab_bytes: Option<VocabBytes>,
    /// Hidden-state dimension for the embeddings endpoint.
    pub hidden_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    /// Round-trip a `BatchRequest::Generate` through an in-memory mpsc channel.
    ///
    /// This verifies that:
    /// 1. The variant can be constructed and sent without compile errors.
    /// 2. The oneshot reply channel delivers the result back to the caller.
    #[tokio::test]
    async fn test_generate_round_trip() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<BatchRequest>(8);

        let (reply_tx, reply_rx) = oneshot::channel::<Result<(String, UsageStats), String>>();

        tx.send(BatchRequest::Generate {
            prompt: "hello".to_string(),
            max_tokens: 16,
            config: SamplerConfig::default(),
            reply: reply_tx,
        })
        .await
        .expect("channel should accept the request");

        // Simulate a minimal worker: receive and immediately reply.
        let req = rx.recv().await.expect("worker should receive request");
        match req {
            BatchRequest::Generate {
                prompt,
                max_tokens,
                reply,
                ..
            } => {
                assert_eq!(prompt, "hello");
                assert_eq!(max_tokens, 16);
                let usage = UsageStats {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                };
                reply
                    .send(Ok(("world".to_string(), usage)))
                    .expect("reply should succeed");
            }
            other => panic!("unexpected variant: {other:?}"),
        }

        let result = reply_rx.await.expect("reply future should resolve");
        let (text, usage) = result.expect("should be Ok");
        assert_eq!(text, "world");
        assert_eq!(usage.total_tokens, 2);
    }

    /// Verify that the `Debug` implementation does not panic and includes
    /// the prompt length rather than the full text (privacy / log hygiene).
    #[test]
    fn test_debug_does_not_expose_full_prompt() {
        let (reply_tx, _reply_rx) = oneshot::channel::<Result<(String, UsageStats), String>>();
        let req = BatchRequest::Generate {
            prompt: "secret prompt contents".to_string(),
            max_tokens: 32,
            config: SamplerConfig::default(),
            reply: reply_tx,
        };
        let debug_str = format!("{req:?}");
        assert!(debug_str.contains("prompt_len"));
        assert!(!debug_str.contains("secret prompt contents"));
    }
}

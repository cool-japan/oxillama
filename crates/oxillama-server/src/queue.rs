//! Request queue types for the continuous-batching inference worker.
//!
//! Instead of each HTTP handler holding the engine mutex directly, every
//! handler constructs a [`BatchRequest`] and sends it through a
//! `tokio::sync::mpsc::Sender`.  A single background worker receives these
//! requests one at a time and drives the `InferenceEngine`, eliminating
//! mutex contention across concurrent requests.

use std::sync::Arc;

use oxillama_runtime::sampling::SamplerConfig;
use oxillama_runtime::FinishReason;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Error message used when a streaming request is shed by the worker
/// because the client had already disconnected (SSE receiver full/closed)
/// or explicitly cancelled before generation began (D3/D4).
pub const CANCELLED_MESSAGE: &str = "generation cancelled: client disconnected";

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

/// LoRA adapter selection for a single request.
///
/// Each entry is `(adapter_name, scale_multiplier)`.  The adapter must have
/// been pre-loaded via `POST /admin/loras` and registered in `AppState::loras`.
pub type LoraSelection = Vec<(String, f32)>;

/// What a completed non-streaming generation reports back to its caller:
/// the decoded text, token accounting, and **why generation stopped**.
///
/// The `FinishReason` used to be dropped inside the worker — `oxillama-runtime`
/// returned one from every `generate_*_detailed` call, but the reply channel
/// was typed `(String, UsageStats)` and threw it away, which is why every
/// OpenAI response hardcoded `finish_reason: "stop"` and never reported
/// `"length"` for a request that actually hit `max_tokens`.
pub type GenerateReply = Result<(String, UsageStats, FinishReason), String>;

/// The streaming counterpart of [`GenerateReply`]: no text (it was already
/// streamed token by token), but the same token accounting and finish reason
/// so the trailing SSE chunk can carry a truthful `finish_reason`.
pub type GenerateStreamReply = Result<(UsageStats, FinishReason), String>;

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
        /// Whether to look up and store the prompt's KV state in the prefix
        /// cache.  When `true` (default), the worker checks for a matching
        /// cached prefix and skips the redundant prefill if found.
        cache_prompt: bool,
        /// LoRA adapters to apply for this request.  Empty means no LoRA.
        lora_selection: LoraSelection,
        /// Whether prompt encoding may apply the model's own BOS/EOS policy.
        ///
        /// `true` for a raw `/v1/completions` prompt.  **`false`** whenever
        /// the caller pre-rendered a chat template that already emits a
        /// literal begin-of-text marker (`<|begin_of_text|>` for Llama-3,
        /// `<s>` for Mistral — see
        /// [`ChatTemplate::emits_literal_bos`](oxillama_runtime::ChatTemplate::emits_literal_bos)),
        /// or the model gets two BOS tokens.  ChatML/Alpaca emit no such
        /// marker and must keep this `true`, or the tokenizer's own
        /// `tokenizer.ggml.add_bos_token` policy is silently skipped.
        add_special: bool,
        /// Channel to send the result back to the caller.
        reply: oneshot::Sender<GenerateReply>,
    },

    /// Streaming generation: invokes `callback` for every decoded token.
    GenerateStream {
        /// The formatted prompt to generate from.
        prompt: String,
        /// Maximum number of tokens to generate.
        max_tokens: usize,
        /// Per-request sampler configuration.
        config: SamplerConfig,
        /// Whether to look up and store the prompt's KV state in the prefix
        /// cache.
        cache_prompt: bool,
        /// LoRA adapters to apply for this request.  Empty means no LoRA.
        lora_selection: LoraSelection,
        /// Whether prompt encoding may apply the model's own BOS/EOS policy.
        ///
        /// See [`BatchRequest::Generate::add_special`].
        add_special: bool,
        /// Cooperative cancellation signal (D3/D4).
        ///
        /// The HTTP layer cancels this token when the client's SSE/WS
        /// receiver can no longer accept events (a bounded channel is
        /// full or closed — the client stopped reading or disconnected).
        /// The worker checks it **before** starting generation: if already
        /// cancelled, the request is shed immediately without ever touching
        /// the engine.
        ///
        /// It does not (yet) preempt an in-progress decode loop. The
        /// mechanism to do so now exists —
        /// [`GenerationConfig::cancel_flag`](oxillama_runtime::GenerationConfig::cancel_flag)
        /// is checked once per decode iteration — but bridging this
        /// `CancellationToken` to that `Arc<AtomicBool>` is a separate
        /// change; see the server TODO. Until then an in-flight decode for
        /// an already-abandoned client still runs to completion, but the
        /// result is discarded rather than blocking the worker forever
        /// waiting on a full channel (which is what D3 actually fixes).
        cancel: CancellationToken,
        /// Called with each token text inside the blocking worker thread.
        callback: StreamCallback,
        /// Channel that receives token accounting plus the finish reason once
        /// generation is complete, or `Err(message)` on failure.
        reply: oneshot::Sender<GenerateStreamReply>,
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
                prompt,
                max_tokens,
                cache_prompt,
                lora_selection,
                add_special,
                ..
            } => f
                .debug_struct("Generate")
                .field("prompt_len", &prompt.len())
                .field("max_tokens", max_tokens)
                .field("cache_prompt", cache_prompt)
                .field("lora_count", &lora_selection.len())
                .field("add_special", add_special)
                .finish(),
            BatchRequest::GenerateStream {
                prompt,
                max_tokens,
                cache_prompt,
                lora_selection,
                add_special,
                ..
            } => f
                .debug_struct("GenerateStream")
                .field("prompt_len", &prompt.len())
                .field("max_tokens", max_tokens)
                .field("cache_prompt", cache_prompt)
                .field("lora_count", &lora_selection.len())
                .field("add_special", add_special)
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

        let (reply_tx, reply_rx) = oneshot::channel::<GenerateReply>();

        tx.send(BatchRequest::Generate {
            prompt: "hello".to_string(),
            max_tokens: 16,
            config: SamplerConfig::default(),
            cache_prompt: true,
            lora_selection: vec![],
            add_special: true,
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
                    .send(Ok(("world".to_string(), usage, FinishReason::Eos)))
                    .expect("reply should succeed");
            }
            other => panic!("unexpected variant: {other:?}"),
        }

        let result = reply_rx.await.expect("reply future should resolve");
        let (text, usage, finish_reason) = result.expect("should be Ok");
        assert_eq!(text, "world");
        assert_eq!(usage.total_tokens, 2);
        assert_eq!(finish_reason, FinishReason::Eos);
    }

    /// The reply channel must carry the `FinishReason` end to end — this is
    /// the field the OpenAI response layer needs in order to ever report
    /// `"length"` instead of an unconditional `"stop"`.
    #[tokio::test]
    async fn test_generate_reply_carries_max_tokens_finish_reason() {
        let (reply_tx, reply_rx) = oneshot::channel::<GenerateReply>();
        reply_tx
            .send(Ok((
                "truncated".to_string(),
                UsageStats::default(),
                FinishReason::MaxTokens,
            )))
            .expect("reply should succeed");
        let (_, _, finish_reason) = reply_rx
            .await
            .expect("reply future should resolve")
            .expect("should be Ok");
        assert_eq!(finish_reason.as_openai_str(), "length");
    }

    /// The streaming reply carries the same information minus the text.
    #[tokio::test]
    async fn test_generate_stream_reply_carries_finish_reason() {
        let (reply_tx, reply_rx) = oneshot::channel::<GenerateStreamReply>();
        reply_tx
            .send(Ok((UsageStats::default(), FinishReason::MaxTokens)))
            .expect("reply should succeed");
        let (_, finish_reason) = reply_rx
            .await
            .expect("reply future should resolve")
            .expect("should be Ok");
        assert_eq!(finish_reason.as_openai_str(), "length");
    }

    /// Verify that the `Debug` implementation does not panic and includes
    /// the prompt length rather than the full text (privacy / log hygiene).
    #[test]
    fn test_debug_does_not_expose_full_prompt() {
        let (reply_tx, _reply_rx) = oneshot::channel::<GenerateReply>();
        let req = BatchRequest::Generate {
            prompt: "secret prompt contents".to_string(),
            max_tokens: 32,
            config: SamplerConfig::default(),
            cache_prompt: true,
            lora_selection: vec![],
            add_special: true,
            reply: reply_tx,
        };
        let debug_str = format!("{req:?}");
        assert!(debug_str.contains("prompt_len"));
        assert!(!debug_str.contains("secret prompt contents"));
    }
}

//! Inference worker — drains the request queue on a dedicated blocking thread.
//!
//! There is exactly one worker per server process.  It owns the
//! `InferenceEngine` exclusively, which means no mutex is ever contested:
//! route handlers enqueue [`BatchRequest`] items and the worker processes
//! them sequentially, calling `engine.reset()` between requests to keep
//! the KV cache isolated across unrelated conversations.

use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use oxillama_runtime::engine::InferenceEngine;

use crate::queue::{BatchRequest, UsageStats};

/// Spawn the inference worker on a dedicated Tokio blocking thread.
///
/// The worker runs until the sending side of `rx` is dropped (i.e., the
/// server shuts down), then exits cleanly.
///
/// # Panics
///
/// Does **not** panic on individual request errors — those are reported
/// through the per-request `reply` oneshot channel.
pub fn spawn_inference_worker(engine: InferenceEngine, rx: mpsc::Receiver<BatchRequest>) {
    tokio::task::spawn_blocking(move || {
        run_worker(engine, rx);
    });
}

/// Blocking worker loop.  Must be called from a context that is allowed to
/// block (`spawn_blocking` or a dedicated `std::thread::spawn`).
fn run_worker(mut engine: InferenceEngine, mut rx: mpsc::Receiver<BatchRequest>) {
    tracing::info!("inference worker started");

    while let Some(req) = rx.blocking_recv() {
        debug!(req = ?req, "processing inference request");

        match req {
            // ----------------------------------------------------------------
            // Non-streaming generation
            // ----------------------------------------------------------------
            BatchRequest::Generate {
                prompt,
                max_tokens,
                config,
                reply,
            } => {
                engine.reset();
                let mut completion_tokens = 0usize;
                let prompt_tokens = engine.tokenize(&prompt).map(|t| t.len()).unwrap_or(0);
                let result = engine
                    .generate_with_config(&prompt, max_tokens, config, |_| {
                        completion_tokens += 1;
                    })
                    .map(|text| {
                        let usage = UsageStats {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                        };
                        (text, usage)
                    })
                    .map_err(|e| e.to_string());

                if reply.send(result).is_err() {
                    warn!("Generate reply channel closed before result was delivered");
                }
            }

            // ----------------------------------------------------------------
            // Streaming generation
            // ----------------------------------------------------------------
            BatchRequest::GenerateStream {
                prompt,
                max_tokens,
                config,
                mut callback,
                reply,
            } => {
                engine.reset();
                let mut completion_tokens = 0usize;
                let prompt_tokens = engine.tokenize(&prompt).map(|t| t.len()).unwrap_or(0);
                let result = engine
                    .generate_with_config(&prompt, max_tokens, config, |token| {
                        completion_tokens += 1;
                        callback(token);
                    })
                    .map(|_| UsageStats {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    })
                    .map_err(|e| e.to_string());

                if reply.send(result).is_err() {
                    warn!("GenerateStream reply channel closed before result was delivered");
                }
            }

            // ----------------------------------------------------------------
            // Embedding
            // ----------------------------------------------------------------
            BatchRequest::Embed { text, reply } => {
                // `embed()` calls `engine.reset()` internally; no need to
                // call it here, but we do so for consistency (idempotent).
                engine.reset();
                let result = engine.embed(&text).map_err(|e| e.to_string());

                if reply.send(result).is_err() {
                    warn!("Embed reply channel closed before result was delivered");
                }
            }
        }
    }

    error!("inference worker channel closed — no more requests can be processed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_runtime::{engine::EngineConfig, sampling::SamplerConfig};
    use tokio::sync::oneshot;

    /// A worker started with an unloaded engine should return an appropriate
    /// error (not panic) when it receives a `Generate` request.
    #[tokio::test]
    async fn test_worker_returns_error_for_unloaded_engine() {
        let engine = InferenceEngine::new(EngineConfig::default());
        let (tx, rx) = mpsc::channel::<BatchRequest>(4);

        spawn_inference_worker(engine, rx);

        let (reply_tx, reply_rx) =
            oneshot::channel::<Result<(String, crate::queue::UsageStats), String>>();
        tx.send(BatchRequest::Generate {
            prompt: "test".to_string(),
            max_tokens: 8,
            config: SamplerConfig::default(),
            reply: reply_tx,
        })
        .await
        .expect("send should succeed");

        let result = reply_rx.await.expect("reply future should resolve");
        assert!(
            result.is_err(),
            "unloaded engine should produce an error, got: {result:?}"
        );
    }

    /// An `Embed` request to an unloaded engine must return an error.
    #[tokio::test]
    async fn test_worker_embed_error_for_unloaded_engine() {
        let engine = InferenceEngine::new(EngineConfig::default());
        let (tx, rx) = mpsc::channel::<BatchRequest>(4);

        spawn_inference_worker(engine, rx);

        let (reply_tx, reply_rx) = oneshot::channel::<Result<Vec<f32>, String>>();
        tx.send(BatchRequest::Embed {
            text: "hello world".to_string(),
            reply: reply_tx,
        })
        .await
        .expect("send should succeed");

        let result = reply_rx.await.expect("reply future should resolve");
        assert!(
            result.is_err(),
            "unloaded engine embed should produce an error, got: {result:?}"
        );
    }
}

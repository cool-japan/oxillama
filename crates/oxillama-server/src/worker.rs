//! Inference worker — drains the request queue on a dedicated blocking thread.
//!
//! There is exactly one worker per server process.  It owns the
//! `InferenceEngine` exclusively, which means no mutex is ever contested:
//! route handlers enqueue [`BatchRequest`] items and the worker processes
//! them sequentially.
//!
//! ## Prefix KV cache
//!
//! On each `Generate`/`GenerateStream` request (when `cache_prompt` is true):
//! 1. Tokenize the prompt.
//! 2. Look up the longest matching prefix in `prefix_cache`.
//! 3. **Hit**: call `engine.prime_with_prefix(cached, restore_to, suffix)` to
//!    restore the KV cache and run only the suffix tokens through the forward
//!    pass, then call `engine.generate_with_logits` to decode.
//! 4. **Miss**: call `engine.reset()` then `engine.generate_with_config`.
//! 5. After generation (on hit or miss), if `cache_prompt` is true, store the
//!    full prompt's KV state in `prefix_cache` for future requests.
//!
//! ## Multi-LoRA
//!
//! When `lora_selection` is non-empty:
//! 1. Resolve adapter names against the `loras` registry.
//! 2. Push adapters onto the engine's LoRA stack.
//! 3. Call `engine.apply_lora_stack()`.
//! 4. Generate.
//! 5. Call `engine.unapply_all_loras()` in a scope guard so the stack is
//!    always cleared even on error.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use oxillama_runtime::engine::InferenceEngine;
use oxillama_runtime::{LoadedLora, PrefixKvCache};

use crate::queue::{BatchRequest, UsageStats};

// ── KV cache helpers ─────────────────────────────────────────────────────────

/// Attempt a prefix cache lookup and prime the engine's KV cache.
///
/// Returns `Some((prompt_tokens, initial_logits))` on a cache hit that
/// successfully primed the engine, `None` on a miss or error.
fn try_prefix_cache_hit(
    engine: &mut InferenceEngine,
    prompt: &str,
    prefix_cache: &Arc<Mutex<PrefixKvCache>>,
) -> Option<(Vec<u32>, Vec<f32>)> {
    let tokens = engine.tokenize(prompt).ok()?;
    if tokens.is_empty() {
        return None;
    }

    type PrefixHitData = (usize, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<u32>);

    // Clone the cached KV state to release the Mutex before calling into the
    // engine (which takes &mut self and thus cannot alias the cache guard).
    let (effective_match, cached_keys, cached_values, suffix_tokens): PrefixHitData = {
        let mut cache = prefix_cache.lock().ok()?;
        let (match_len, cached) = cache.lookup(&tokens)?;
        // Cap at tokens.len()-1 so we always process at least the last
        // prompt token to obtain fresh logits for the decode loop.
        let effective = match_len.min(tokens.len().saturating_sub(1));
        if effective == 0 {
            return None;
        }
        let suffix: Vec<u32> = tokens[effective..].to_vec();
        // Clone the cached KV data before dropping the lock guard.
        let keys = cached.keys().to_vec();
        let values = cached.values().to_vec();
        (effective, keys, values, suffix)
    };
    // Lock released here — safe to mutably borrow engine.

    // Reconstruct a temporary PrefixKvCache from the cloned data and obtain
    // a CachedKvState reference to pass into engine.prime_with_prefix.
    let mut scratch = PrefixKvCache::new(oxillama_runtime::PrefixCacheConfig {
        max_entries: 2,
        max_memory_bytes: usize::MAX,
        min_prefix_len: 1,
    });
    let snap = oxillama_runtime::CachedKvState::new(cached_keys, cached_values, effective_match);
    scratch.store_snapshot(&tokens[..effective_match], snap);

    let logits = if let Some((_m, cached)) = scratch.lookup(&tokens[..effective_match]) {
        engine
            .prime_with_prefix(cached, effective_match, &suffix_tokens)
            .ok()?
    } else {
        return None;
    };

    Some((tokens, logits))
}

/// Store the current KV state into the prefix cache after a successful
/// generation pass.
fn store_prefix_cache(
    engine: &mut InferenceEngine,
    prompt_tokens: &[u32],
    prefix_cache: &Arc<Mutex<PrefixKvCache>>,
) {
    if let Ok(mut cache) = prefix_cache.lock() {
        engine.store_kv_in_prefix_cache(prompt_tokens, &mut cache);
    }
}

// ── LoRA helpers ─────────────────────────────────────────────────────────────

/// Push the requested LoRA adapters onto the engine and apply the stack.
///
/// Returns the number of adapters applied (0 if `lora_selection` is empty or
/// any name is unknown — in the latter case a warning is emitted).
fn apply_lora_selection(
    engine: &mut InferenceEngine,
    lora_selection: &[(String, f32)],
    loras: &Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>,
) -> usize {
    if lora_selection.is_empty() {
        return 0;
    }

    let registry = match loras.read() {
        Ok(r) => r,
        Err(_) => {
            warn!("LoRA registry RwLock poisoned; skipping LoRA application");
            return 0;
        }
    };

    let mut applied = 0usize;
    for (name, scale) in lora_selection {
        match registry.get(name.as_str()) {
            Some(lora) => {
                engine.push_lora(Arc::clone(lora), *scale);
                applied += 1;
            }
            None => {
                warn!(adapter = %name, "unknown LoRA adapter name; skipping");
            }
        }
    }

    if applied > 0 {
        if let Err(e) = engine.apply_lora_stack() {
            warn!(error = %e, "apply_lora_stack failed; proceeding without LoRA");
            engine.unapply_all_loras();
            return 0;
        }
    }
    applied
}

// ── Spawn ─────────────────────────────────────────────────────────────────────

/// Spawn the inference worker on a dedicated Tokio blocking thread.
///
/// The worker runs until the sending side of `rx` is dropped (server shutdown).
pub fn spawn_inference_worker(
    engine: InferenceEngine,
    rx: mpsc::Receiver<BatchRequest>,
    prefix_cache: Arc<Mutex<PrefixKvCache>>,
    loras: Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>,
) {
    tokio::task::spawn_blocking(move || {
        run_worker(engine, rx, prefix_cache, loras);
    });
}

/// Blocking worker loop.
fn run_worker(
    mut engine: InferenceEngine,
    mut rx: mpsc::Receiver<BatchRequest>,
    prefix_cache: Arc<Mutex<PrefixKvCache>>,
    loras: Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>,
) {
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
                cache_prompt,
                lora_selection,
                reply,
            } => {
                let lora_count = apply_lora_selection(&mut engine, &lora_selection, &loras);

                let result = run_generate(
                    &mut engine,
                    &prompt,
                    max_tokens,
                    config,
                    cache_prompt,
                    &prefix_cache,
                    |_| {},
                );

                if lora_count > 0 {
                    engine.unapply_all_loras();
                }

                let result = result.map_err(|e| e.to_string());
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
                cache_prompt,
                lora_selection,
                mut callback,
                reply,
            } => {
                let lora_count = apply_lora_selection(&mut engine, &lora_selection, &loras);

                let result = run_generate(
                    &mut engine,
                    &prompt,
                    max_tokens,
                    config,
                    cache_prompt,
                    &prefix_cache,
                    |t| callback(t),
                );

                if lora_count > 0 {
                    engine.unapply_all_loras();
                }

                let result = result.map(|(_, usage)| usage).map_err(|e| e.to_string());

                if reply.send(result).is_err() {
                    warn!("GenerateStream reply channel closed before result was delivered");
                }
            }

            // ----------------------------------------------------------------
            // Embedding
            // ----------------------------------------------------------------
            BatchRequest::Embed { text, reply } => {
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

/// Core generation logic shared by streaming and non-streaming paths.
///
/// Returns `(text, UsageStats)` on success.
fn run_generate(
    engine: &mut InferenceEngine,
    prompt: &str,
    max_tokens: usize,
    config: oxillama_runtime::sampling::SamplerConfig,
    cache_prompt: bool,
    prefix_cache: &Arc<Mutex<PrefixKvCache>>,
    mut callback: impl FnMut(&str),
) -> Result<(String, UsageStats), oxillama_runtime::RuntimeError> {
    let mut completion_tokens = 0usize;

    // ── Attempt prefix cache hit ──────────────────────────────────────────
    let (text, prompt_token_count) = if cache_prompt {
        match try_prefix_cache_hit(engine, prompt, prefix_cache) {
            Some((prompt_tokens, initial_logits)) => {
                let pt_count = prompt_tokens.len();
                let text = engine.generate_with_logits(
                    &prompt_tokens,
                    initial_logits,
                    max_tokens,
                    config,
                    |t| {
                        completion_tokens += 1;
                        callback(t);
                    },
                )?;
                if cache_prompt {
                    store_prefix_cache(engine, &prompt_tokens, prefix_cache);
                }
                (text, pt_count)
            }
            None => {
                // Miss — fall through to full prefill.
                engine.reset();
                let pt_count = engine.tokenize(prompt).map(|t| t.len()).unwrap_or(0);
                let text = engine.generate_with_config(prompt, max_tokens, config, |t| {
                    completion_tokens += 1;
                    callback(t);
                })?;
                if cache_prompt {
                    // Tokenize again to store; tokenize is cheap.
                    if let Ok(tokens) = engine.tokenize(prompt) {
                        store_prefix_cache(engine, &tokens, prefix_cache);
                    }
                }
                (text, pt_count)
            }
        }
    } else {
        // Prefix cache disabled for this request.
        engine.reset();
        let pt_count = engine.tokenize(prompt).map(|t| t.len()).unwrap_or(0);
        let text = engine.generate_with_config(prompt, max_tokens, config, |t| {
            completion_tokens += 1;
            callback(t);
        })?;
        (text, pt_count)
    };

    let usage = UsageStats {
        prompt_tokens: prompt_token_count,
        completion_tokens,
        total_tokens: prompt_token_count + completion_tokens,
    };
    Ok((text, usage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_runtime::{engine::EngineConfig, sampling::SamplerConfig, PrefixCacheConfig};
    use tokio::sync::oneshot;

    type WorkerHandles = (
        mpsc::Sender<BatchRequest>,
        Arc<Mutex<PrefixKvCache>>,
        Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>,
    );

    fn make_worker() -> WorkerHandles {
        let engine = InferenceEngine::new(EngineConfig::default());
        let (tx, rx) = mpsc::channel::<BatchRequest>(4);
        let prefix_cache = Arc::new(Mutex::new(PrefixKvCache::new(PrefixCacheConfig::default())));
        let loras = Arc::new(RwLock::new(HashMap::new()));
        spawn_inference_worker(engine, rx, Arc::clone(&prefix_cache), Arc::clone(&loras));
        (tx, prefix_cache, loras)
    }

    /// An unloaded engine returns an error (not panic) for Generate.
    #[tokio::test]
    async fn test_worker_returns_error_for_unloaded_engine() {
        let (tx, _, _) = make_worker();

        let (reply_tx, reply_rx) =
            oneshot::channel::<Result<(String, crate::queue::UsageStats), String>>();
        tx.send(BatchRequest::Generate {
            prompt: "test".to_string(),
            max_tokens: 8,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
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

    /// An unloaded engine returns an error for Embed.
    #[tokio::test]
    async fn test_worker_embed_error_for_unloaded_engine() {
        let (tx, _, _) = make_worker();

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

    /// An unloaded engine returns an error for GenerateStream.
    #[tokio::test]
    async fn test_worker_generate_stream_error_for_unloaded_engine() {
        let (tx, _, _) = make_worker();

        let (reply_tx, reply_rx) = oneshot::channel::<Result<crate::queue::UsageStats, String>>();
        tx.send(BatchRequest::GenerateStream {
            prompt: "stream test".to_string(),
            max_tokens: 4,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
            callback: Box::new(|_| {}),
            reply: reply_tx,
        })
        .await
        .expect("send should succeed");

        let result = reply_rx.await.expect("reply future should resolve");
        assert!(
            result.is_err(),
            "unloaded GenerateStream should produce an error"
        );
    }

    /// Unknown LoRA adapter names are silently skipped; the request proceeds
    /// without LoRA (returning an engine error from the unloaded engine, not
    /// a LoRA-related panic).
    #[tokio::test]
    async fn test_worker_unknown_lora_name_does_not_panic() {
        let (tx, _, _) = make_worker();

        let (reply_tx, reply_rx) =
            oneshot::channel::<Result<(String, crate::queue::UsageStats), String>>();
        tx.send(BatchRequest::Generate {
            prompt: "test".to_string(),
            max_tokens: 4,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![("nonexistent_adapter".to_string(), 1.0)],
            reply: reply_tx,
        })
        .await
        .expect("send should succeed");

        // The worker should return an error (model not loaded), not panic.
        let result = reply_rx.await.expect("reply future should resolve");
        assert!(
            result.is_err(),
            "should return engine error, got: {result:?}"
        );
    }

    /// `cache_prompt = false` path: worker resets and calls generate_with_config.
    /// Returns an engine error on unloaded engine, not a cache-related error.
    #[tokio::test]
    async fn test_worker_cache_prompt_false_uses_full_prefill() {
        let (tx, _, _) = make_worker();

        let (reply_tx, reply_rx) =
            oneshot::channel::<Result<(String, crate::queue::UsageStats), String>>();
        tx.send(BatchRequest::Generate {
            prompt: "hello".to_string(),
            max_tokens: 8,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
            reply: reply_tx,
        })
        .await
        .expect("send should succeed");

        let result = reply_rx.await.expect("reply future should resolve");
        assert!(result.is_err(), "unloaded engine should error");
    }

    /// Multiple sequential requests complete without panicking (queue ordering).
    #[tokio::test]
    async fn test_worker_sequential_requests_do_not_panic() {
        let (tx, _, _) = make_worker();

        for _ in 0..3 {
            let (reply_tx, reply_rx) =
                oneshot::channel::<Result<(String, crate::queue::UsageStats), String>>();
            tx.send(BatchRequest::Generate {
                prompt: "hello".to_string(),
                max_tokens: 4,
                config: SamplerConfig::default(),
                cache_prompt: true,
                lora_selection: vec![],
                reply: reply_tx,
            })
            .await
            .expect("send should succeed");
            // Result is an error (model not loaded) — that's fine.
            let _ = reply_rx.await.expect("reply should resolve");
        }
    }
}

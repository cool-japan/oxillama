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

use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use oxillama_runtime::engine::InferenceEngine;
use oxillama_runtime::{FinishReason, GenerationConfig, LoadedLora};

use crate::prefix_registry::{cache_namespace, PrefixCacheRegistry};
use crate::queue::{BatchRequest, UsageStats, CANCELLED_MESSAGE};

// ── KV cache helpers ─────────────────────────────────────────────────────────

/// Attempt a prefix cache lookup and prime the engine's KV cache.
///
/// Returns `Some((prompt_tokens, initial_logits))` on a cache hit that
/// successfully primed the engine, `None` on a miss or error.
///
/// Looks up the cache for `namespace` only (D6 fix): a namespace is derived
/// from `(model_id, sorted lora_selection)`, so a request with a different
/// LoRA stack (or none at all) can never observe another namespace's cached
/// KV state, even if the token prefix matches exactly.
fn try_prefix_cache_hit(
    engine: &mut InferenceEngine,
    prompt: &str,
    registry: &PrefixCacheRegistry,
    namespace: &str,
    add_special: bool,
) -> Option<(Vec<u32>, Vec<f32>)> {
    // `add_special` must match what the miss path's `generate_detailed` would
    // use, or the cache-hit path prefills a *different* token sequence than
    // the one it looked up (an extra leading BOS for a Llama-3 chat prompt
    // that already renders `<|begin_of_text|>` itself), and the restored KV
    // state would belong to a prompt the caller never sent.
    let tokens = engine.tokenize_with(prompt, add_special, true).ok()?;
    if tokens.is_empty() {
        return None;
    }

    type PrefixHitData = (usize, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<u32>);

    // Clone the cached KV state to release the registry lock before calling
    // into the engine (which takes &mut self and thus cannot alias the
    // cache guard held inside `with_cache`).
    let hit: Option<PrefixHitData> = registry.with_cache(namespace, |cache| {
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
        Some((effective, keys, values, suffix))
    });
    let (effective_match, cached_keys, cached_values, suffix_tokens) = hit?;
    // Lock released here — safe to mutably borrow engine.

    // Reconstruct a temporary PrefixKvCache from the cloned data and obtain
    // a CachedKvState reference to pass into engine.prime_with_prefix.
    let mut scratch = oxillama_runtime::PrefixKvCache::new(oxillama_runtime::PrefixCacheConfig {
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

/// Store the current KV state into `namespace`'s prefix cache after a
/// successful generation pass.
fn store_prefix_cache(
    engine: &mut InferenceEngine,
    prompt_tokens: &[u32],
    registry: &PrefixCacheRegistry,
    namespace: &str,
) {
    registry.with_cache(namespace, |cache| {
        engine.store_kv_in_prefix_cache(prompt_tokens, cache);
    });
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
///
/// `prefix_registry` and `worker_alive` are shared with [`crate::state::AppState`]
/// (the same `Arc`s should be passed to both) so that `GET /ready` observes
/// the worker's actual liveness (D7) and the prefix cache is namespaced by
/// LoRA selection (D6) consistently between the HTTP layer and the worker.
pub fn spawn_inference_worker(
    engine: InferenceEngine,
    rx: mpsc::Receiver<BatchRequest>,
    model_id: String,
    prefix_registry: Arc<PrefixCacheRegistry>,
    loras: Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>,
    worker_alive: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        run_worker(engine, rx, model_id, prefix_registry, loras, worker_alive);
    })
}

/// Blocking worker loop.
///
/// D7 fix: each request is processed inside `catch_unwind` so that a panic
/// deep in the inference engine (an out-of-bounds tensor op, an assertion in
/// a quant kernel, …) cannot silently kill the worker thread forever while
/// `/health` keeps returning 200. On a caught panic the engine is reset and
/// the loop continues to the next request; `worker_alive` is only flipped to
/// `false` when the channel itself closes (i.e. the worker is intentionally
/// shutting down), not when an individual request panics.
fn run_worker(
    mut engine: InferenceEngine,
    mut rx: mpsc::Receiver<BatchRequest>,
    model_id: String,
    prefix_registry: Arc<PrefixCacheRegistry>,
    loras: Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>,
    worker_alive: Arc<AtomicBool>,
) {
    tracing::info!("inference worker started");
    worker_alive.store(true, Ordering::Release);

    while let Some(req) = rx.blocking_recv() {
        debug!(req = ?req, "processing inference request");

        // `req` carries a boxed `dyn FnMut` callback (`StreamCallback`),
        // which is not auto-`UnwindSafe` (trait objects don't propagate
        // auto traits unless named in the bound). We move `req` and `&mut
        // engine` into the closure entirely, so nothing outside can
        // observe a partially-mutated value if it panics — that's exactly
        // the situation `AssertUnwindSafe` is for.
        let result = catch_unwind(AssertUnwindSafe(|| {
            process_one_request(&mut engine, req, &model_id, &prefix_registry, &loras);
        }));

        if let Err(payload) = result {
            error!(
                panic_message = %panic_message(&payload),
                "inference worker caught a panic while processing a request; \
                 resetting engine state and continuing"
            );
            // The engine's internal state after an unwound panic is
            // unspecified (partial tensor writes, a KV cache left mid
            // update, ...) — reset it so the *next* request starts clean
            // rather than potentially observing corrupted state.
            engine.reset();
        }
    }

    worker_alive.store(false, Ordering::Release);
    error!("inference worker channel closed — no more requests can be processed");
}

/// Best-effort extraction of a human-readable message from a caught panic
/// payload (`&(dyn Any + Send)`), matching the two shapes `std::panic!`
/// actually produces (`&str` literals and owned `String`s).
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Dispatch a single [`BatchRequest`] to the engine. Called from inside
/// `catch_unwind` by `run_worker`.
fn process_one_request(
    engine: &mut InferenceEngine,
    req: BatchRequest,
    model_id: &str,
    prefix_registry: &Arc<PrefixCacheRegistry>,
    loras: &Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>,
) {
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
            add_special,
            reply,
        } => {
            // D3/D4: if the caller already dropped the reply receiver
            // (client disconnected before we even started), skip the work
            // entirely rather than burning a full generation on an
            // abandoned request.
            if reply.is_closed() {
                warn!("Generate request dropped before processing: reply channel closed");
                return;
            }

            let namespace = cache_namespace(model_id, &lora_selection);
            let lora_count = apply_lora_selection(engine, &lora_selection, loras);

            let result = run_generate(
                engine,
                &prompt,
                max_tokens,
                config,
                cache_prompt,
                add_special,
                prefix_registry,
                &namespace,
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
            add_special,
            cancel,
            mut callback,
            reply,
        } => {
            // D3/D4: shed if the client is already gone (reply dropped) or
            // has been marked cancelled (the SSE/WS producer hit a full or
            // closed channel before we started).
            if reply.is_closed() || cancel.is_cancelled() {
                warn!("GenerateStream request dropped before processing: client disconnected");
                let _ = reply.send(Err(CANCELLED_MESSAGE.to_string()));
                return;
            }

            let namespace = cache_namespace(model_id, &lora_selection);
            let lora_count = apply_lora_selection(engine, &lora_selection, loras);

            let result = run_generate(
                engine,
                &prompt,
                max_tokens,
                config,
                cache_prompt,
                add_special,
                prefix_registry,
                &namespace,
                |t| callback(t),
            );

            if lora_count > 0 {
                engine.unapply_all_loras();
            }

            let result = result
                .map(|(_, usage, finish_reason)| (usage, finish_reason))
                .map_err(|e| e.to_string());

            if reply.send(result).is_err() {
                warn!("GenerateStream reply channel closed before result was delivered");
            }
        }

        // ----------------------------------------------------------------
        // Embedding
        // ----------------------------------------------------------------
        BatchRequest::Embed { text, reply } => {
            if reply.is_closed() {
                warn!("Embed request dropped before processing: reply channel closed");
                return;
            }
            engine.reset();
            let result = engine.embed(&text).map_err(|e| e.to_string());
            if reply.send(result).is_err() {
                warn!("Embed reply channel closed before result was delivered");
            }
        }
    }
}

/// Core generation logic shared by streaming and non-streaming paths.
///
/// Returns `(text, UsageStats, FinishReason)` on success. `namespace` selects
/// which prefix-cache namespace (see [`crate::prefix_registry`]) is consulted
/// and updated — namespacing by `(model_id, lora_selection)` is the D6 fix
/// that prevents LoRA-contaminated cache hits.
///
/// Both branches now go through the `*_detailed` runtime entry points. The
/// plain `generate_with_config` / `generate_with_logits` wrappers return a
/// bare `String`, discarding the [`FinishReason`] the decode loop already
/// computed — which is precisely why every OpenAI response the server emitted
/// claimed `"stop"` even when generation had been truncated at `max_tokens`.
#[allow(clippy::too_many_arguments)]
fn run_generate(
    engine: &mut InferenceEngine,
    prompt: &str,
    max_tokens: usize,
    sampler: oxillama_runtime::sampling::SamplerConfig,
    cache_prompt: bool,
    add_special: bool,
    prefix_registry: &PrefixCacheRegistry,
    namespace: &str,
    mut callback: impl FnMut(&str),
) -> Result<(String, UsageStats, FinishReason), oxillama_runtime::RuntimeError> {
    let mut completion_tokens = 0usize;

    let gen_config = GenerationConfig {
        max_tokens,
        sampler,
        add_special,
        // Control-token text a chat template renders (`<|im_start|>`,
        // `<|eot_id|>`, …) must be recognised as single tokens rather than
        // split byte-by-byte.
        parse_special: true,
        ..GenerationConfig::default()
    };

    // ── Attempt prefix cache hit ──────────────────────────────────────────
    let outcome = if cache_prompt {
        match try_prefix_cache_hit(engine, prompt, prefix_registry, namespace, add_special) {
            Some((prompt_tokens, initial_logits)) => {
                let outcome = engine.generate_with_logits_detailed(
                    &prompt_tokens,
                    initial_logits,
                    &gen_config,
                    |t| {
                        completion_tokens += 1;
                        callback(t);
                    },
                )?;
                store_prefix_cache(engine, &prompt_tokens, prefix_registry, namespace);
                outcome
            }
            None => {
                // Miss — fall through to full prefill.
                engine.reset();
                let outcome = engine.generate_detailed(prompt, &gen_config, |t| {
                    completion_tokens += 1;
                    callback(t);
                })?;
                // Tokenize again to store; tokenize is cheap.
                if let Ok(tokens) = engine.tokenize_with(prompt, add_special, true) {
                    store_prefix_cache(engine, &tokens, prefix_registry, namespace);
                }
                outcome
            }
        }
    } else {
        // Prefix cache disabled for this request.
        engine.reset();
        engine.generate_detailed(prompt, &gen_config, |t| {
            completion_tokens += 1;
            callback(t);
        })?
    };

    // `generate_with_logits_detailed` reports `prompt_tokens` from the token
    // slice it was handed, and `generate_detailed` from its own encoding —
    // either way it is the authoritative count, so there is no need to
    // re-tokenize just to bill the request.
    let prompt_token_count = outcome.prompt_tokens;
    let usage = UsageStats {
        prompt_tokens: prompt_token_count,
        completion_tokens,
        total_tokens: prompt_token_count + completion_tokens,
    };
    Ok((outcome.text, usage, outcome.finish_reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix_registry::DEFAULT_MAX_NAMESPACES;
    use oxillama_runtime::{engine::EngineConfig, sampling::SamplerConfig, PrefixCacheConfig};
    use tokio::sync::oneshot;

    type WorkerHandles = (
        mpsc::Sender<BatchRequest>,
        Arc<PrefixCacheRegistry>,
        Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>,
        Arc<AtomicBool>,
    );

    fn make_worker() -> WorkerHandles {
        let engine = InferenceEngine::new(EngineConfig::default());
        let (tx, rx) = mpsc::channel::<BatchRequest>(4);
        let registry = Arc::new(PrefixCacheRegistry::new(
            PrefixCacheConfig::default(),
            DEFAULT_MAX_NAMESPACES,
        ));
        let loras = Arc::new(RwLock::new(HashMap::new()));
        let worker_alive = Arc::new(AtomicBool::new(false));
        spawn_inference_worker(
            engine,
            rx,
            "test-model".to_string(),
            Arc::clone(&registry),
            Arc::clone(&loras),
            Arc::clone(&worker_alive),
        );
        (tx, registry, loras, worker_alive)
    }

    /// An unloaded engine returns an error (not panic) for Generate.
    #[tokio::test]
    async fn test_worker_returns_error_for_unloaded_engine() {
        let (tx, _, _, _) = make_worker();

        let (reply_tx, reply_rx) = oneshot::channel::<crate::queue::GenerateReply>();
        tx.send(BatchRequest::Generate {
            prompt: "test".to_string(),
            max_tokens: 8,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
            add_special: true,
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
        let (tx, _, _, _) = make_worker();

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
        let (tx, _, _, _) = make_worker();

        let (reply_tx, reply_rx) = oneshot::channel::<crate::queue::GenerateStreamReply>();
        tx.send(BatchRequest::GenerateStream {
            prompt: "stream test".to_string(),
            max_tokens: 4,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
            add_special: true,
            cancel: tokio_util::sync::CancellationToken::new(),
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
        let (tx, _, _, _) = make_worker();

        let (reply_tx, reply_rx) = oneshot::channel::<crate::queue::GenerateReply>();
        tx.send(BatchRequest::Generate {
            prompt: "test".to_string(),
            max_tokens: 4,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![("nonexistent_adapter".to_string(), 1.0)],
            add_special: true,
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
        let (tx, _, _, _) = make_worker();

        let (reply_tx, reply_rx) = oneshot::channel::<crate::queue::GenerateReply>();
        tx.send(BatchRequest::Generate {
            prompt: "hello".to_string(),
            max_tokens: 8,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
            add_special: true,
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
        let (tx, _, _, _) = make_worker();

        for _ in 0..3 {
            let (reply_tx, reply_rx) = oneshot::channel::<crate::queue::GenerateReply>();
            tx.send(BatchRequest::Generate {
                prompt: "hello".to_string(),
                max_tokens: 4,
                config: SamplerConfig::default(),
                cache_prompt: true,
                lora_selection: vec![],
                add_special: true,
                reply: reply_tx,
            })
            .await
            .expect("send should succeed");
            // Result is an error (model not loaded) — that's fine.
            let _ = reply_rx.await.expect("reply should resolve");
        }
    }

    /// D3/D4 regression: a `Generate` request whose reply receiver has
    /// already been dropped (the caller gave up before the worker even
    /// looked at the request) is shed without attempting generation. We
    /// can't directly observe "no work was done" on an unloaded engine
    /// (both outcomes are silent), but we *can* prove the worker doesn't
    /// wedge on it and stays available for the next request — this is what
    /// D3 actually protects against (a slow/gone consumer parking the sole
    /// worker forever).
    #[tokio::test]
    async fn test_worker_sheds_generate_request_with_dropped_receiver() {
        let (tx, _, _, _) = make_worker();

        let (reply_tx, reply_rx) = oneshot::channel::<crate::queue::GenerateReply>();
        drop(reply_rx); // caller already gone

        tx.send(BatchRequest::Generate {
            prompt: "hello".to_string(),
            max_tokens: 4,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
            add_special: true,
            reply: reply_tx,
        })
        .await
        .expect("send should succeed");

        // Prove the worker is still alive and processes a subsequent
        // request — it must not have wedged on the dropped-receiver one.
        let (reply_tx2, reply_rx2) = oneshot::channel::<crate::queue::GenerateReply>();
        tx.send(BatchRequest::Generate {
            prompt: "hello again".to_string(),
            max_tokens: 4,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
            add_special: true,
            reply: reply_tx2,
        })
        .await
        .expect("send should succeed");
        let _ = reply_rx2.await.expect("worker should still be responsive");
    }

    /// D3/D4 regression: a `GenerateStream` request whose cancellation
    /// token is already cancelled before the worker picks it up is shed
    /// immediately with `CANCELLED_MESSAGE`, without ever calling into the
    /// (unloaded) engine.
    #[tokio::test]
    async fn test_worker_sheds_generate_stream_when_pre_cancelled() {
        let (tx, _, _, _) = make_worker();

        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();

        let (reply_tx, reply_rx) = oneshot::channel::<crate::queue::GenerateStreamReply>();
        tx.send(BatchRequest::GenerateStream {
            prompt: "stream test".to_string(),
            max_tokens: 4,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
            add_special: true,
            cancel,
            callback: Box::new(|_| {}),
            reply: reply_tx,
        })
        .await
        .expect("send should succeed");

        let result = reply_rx.await.expect("reply future should resolve");
        assert!(
            matches!(result, Err(ref msg) if msg == CANCELLED_MESSAGE),
            "pre-cancelled request should be shed with CANCELLED_MESSAGE, got: {result:?}"
        );
    }

    /// D7 regression: `worker_alive` flips to `true` once the worker loop
    /// starts and back to `false` only when the channel actually closes —
    /// it must not be flipped by an individual request failing (an
    /// unloaded-engine error is not a panic and must not be conflated with
    /// worker death).
    #[tokio::test]
    async fn test_worker_alive_flag_lifecycle() {
        let (tx, _, _, worker_alive) = make_worker();

        // Give the blocking worker thread a moment to start and set the
        // flag. Also exercise a normal (erroring, since unloaded) request
        // in between to prove an engine error does not flip the flag.
        for _ in 0..50 {
            if worker_alive.load(Ordering::Acquire) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            worker_alive.load(Ordering::Acquire),
            "worker_alive should become true once the worker loop starts"
        );

        let (reply_tx, reply_rx) = oneshot::channel::<crate::queue::GenerateReply>();
        tx.send(BatchRequest::Generate {
            prompt: "hello".to_string(),
            max_tokens: 4,
            config: SamplerConfig::default(),
            cache_prompt: false,
            lora_selection: vec![],
            add_special: true,
            reply: reply_tx,
        })
        .await
        .expect("send should succeed");
        let _ = reply_rx.await.expect("reply should resolve");
        assert!(
            worker_alive.load(Ordering::Acquire),
            "an ordinary engine error must not flip worker_alive to false"
        );

        // Closing the sender is what actually shuts the worker down.
        drop(tx);
        for _ in 0..50 {
            if !worker_alive.load(Ordering::Acquire) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            !worker_alive.load(Ordering::Acquire),
            "worker_alive should become false once the channel closes"
        );
    }
}

// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The shared autoregressive decode loop.
//!
//! Before 0.1.4 this loop existed in triplicate inside `engine/mod.rs`
//! (`generate`, `generate_with_config`, `generate_with_logits`), and each copy
//! carried the same four defects:
//!
//! * generation never stopped early, because the EOS check compared against a
//!   single id that was usually `None`;
//! * each token was decoded on its own, so any character spanning two tokens
//!   turned into `U+FFFD`;
//! * the context-length check ran *after* sampling and discarded the token it
//!   had just produced;
//! * the caller received a bare `String` and could not tell natural completion
//!   from truncation, which makes an OpenAI-compatible `finish_reason`
//!   impossible.
//!
//! There is now one loop, and it reports a [`FinishReason`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use oxillama_arch::traits::{ForwardPass, KvCacheAccess};

use crate::error::RuntimeResult;
use crate::kv_cache::KvCache;
use crate::metrics::EngineMetrics;
use crate::sampling::{Sampler, SamplerConfig};
use crate::stream_decode::{StopSequenceBuffer, Utf8StreamDecoder};
use crate::tokenizer_bridge::TokenizerBridge;

/// Why generation stopped.
///
/// Consumers building an OpenAI-compatible response map this with
/// [`FinishReason::as_openai_str`]: `Eos` and `Stopped` become `"stop"`, while
/// `MaxTokens` and `ContextFull` become `"length"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinishReason {
    /// An end-of-generation token was sampled.
    ///
    /// This covers the whole EOG set, not just `eos_token_id` — Llama-3-Instruct
    /// ends a turn with `<|eot_id|>` and Qwen with `<|im_end|>`.
    Eos,
    /// The requested `max_tokens` budget was exhausted.
    MaxTokens,
    /// The KV cache reached the model's context length.
    ContextFull,
    /// A caller-supplied stop sequence was produced.
    Stopped,
    /// The caller's [`GenerationConfig::cancel_flag`] was raised.
    ///
    /// The decode loop checks the flag once per iteration, so cancellation
    /// takes effect at the next token boundary rather than after the full
    /// `max_tokens` budget has been burned.
    Cancelled,
}

impl FinishReason {
    /// The OpenAI `finish_reason` string for this outcome.
    ///
    /// [`Self::Cancelled`] has no OpenAI counterpart — the vocabulary is
    /// `"stop"` / `"length"` / `"tool_calls"` / `"content_filter"` — so it
    /// maps to the `"cancelled"` string OxiLLaMa's own streaming endpoints
    /// already emit for an abandoned request, rather than being silently
    /// laundered into `"stop"` (which would tell a client the model chose to
    /// end the turn when in fact the turn was cut short).
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Eos | Self::Stopped => "stop",
            Self::MaxTokens | Self::ContextFull => "length",
            Self::Cancelled => "cancelled",
        }
    }

    /// `true` when the model chose to stop rather than being cut off.
    pub fn is_natural(self) -> bool {
        matches!(self, Self::Eos | Self::Stopped)
    }

    /// `true` when generation ended because the caller cancelled it.
    pub fn is_cancelled(self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl core::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Eos => "eos",
            Self::MaxTokens => "max_tokens",
            Self::ContextFull => "context_full",
            Self::Stopped => "stopped",
            Self::Cancelled => "cancelled",
        };
        f.write_str(s)
    }
}

/// Per-request generation settings.
///
/// Constructed with [`GenerationConfig::new`] and adjusted with the builder
/// methods; new fields will always have a sensible default so adding one is not
/// a breaking change for callers.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampler settings for this request.
    pub sampler: SamplerConfig,
    /// Stop sequences.
    ///
    /// Generation ends as soon as one is produced, and the stop text itself is
    /// excluded from the output.  Matching works across token boundaries: text
    /// that could still grow into a stop sequence is withheld from the callback
    /// until it is known not to, so a consumer never has to retract output.
    pub stop: Vec<String>,
    /// Whether control tokens appear in the streamed text.
    ///
    /// `false` (the default) matches llama.cpp's `--special` default: control
    /// tokens are consumed but not rendered.  Set to `true` to see them.
    pub render_special: bool,
    /// Whether prompt encoding applies the model's BOS/EOS policy.
    pub add_special: bool,
    /// Whether control tokens written in the prompt are recognised.
    pub parse_special: bool,
    /// Cooperative cancellation hook, checked once per decode iteration.
    ///
    /// `None` (the default) means "never cancel" and costs nothing: the loop
    /// only pays for an `Option::as_ref` test per token, no atomic load.
    /// When `Some`, a raised flag ends generation at the next token boundary
    /// with [`FinishReason::Cancelled`], returning everything produced so
    /// far rather than an error — so a caller that cancels at token 1 of a
    /// `max_tokens = 2048` request stops paying for the remaining 2047
    /// forward passes.
    ///
    /// Set it with [`GenerationConfig::with_cancel_flag`].
    pub cancel_flag: Option<Arc<AtomicBool>>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            sampler: SamplerConfig::default(),
            stop: Vec::new(),
            render_special: false,
            add_special: true,
            parse_special: true,
            cancel_flag: None,
        }
    }
}

impl GenerationConfig {
    /// A configuration generating at most `max_tokens` tokens.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Self::default()
        }
    }

    /// Use `sampler` for this request.
    pub fn with_sampler(mut self, sampler: SamplerConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Stop as soon as any of `stop` is generated.
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }

    /// Render control tokens into the streamed text.
    pub fn with_render_special(mut self, render_special: bool) -> Self {
        self.render_special = render_special;
        self
    }

    /// Stop generating as soon as `flag` is raised.
    ///
    /// See [`GenerationConfig::cancel_flag`].
    pub fn with_cancel_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel_flag = Some(flag);
        self
    }
}

/// Everything a caller needs to describe a completed generation.
#[derive(Debug, Clone)]
pub struct GenerationOutcome {
    /// The generated text, excluding any matched stop sequence.
    pub text: String,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// The generated token ids, in order.
    ///
    /// A terminating EOG token is *not* included — it was never emitted.
    pub generated_tokens: Vec<u32>,
    /// How many tokens the prompt occupied.
    pub prompt_tokens: usize,
    /// The stop sequence that matched, when `finish_reason` is
    /// [`FinishReason::Stopped`].
    pub stop_sequence: Option<String>,
}

impl GenerationOutcome {
    /// The number of generated tokens.
    pub fn completion_tokens(&self) -> usize {
        self.generated_tokens.len()
    }

    /// Total tokens billed for this request.
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens.len()
    }
}

/// Borrowed engine state the decode loop needs.
pub(crate) struct DecodeContext<'a> {
    /// The architecture-specific forward pass.
    pub forward_pass: &'a mut dyn ForwardPass,
    /// The KV cache being extended.
    pub kv_cache: &'a mut KvCache,
    /// Tokenizer used for EOG detection and detokenization.
    pub tokenizer: &'a TokenizerBridge,
    /// Live counters.
    pub metrics: &'a EngineMetrics,
}

/// Run prefill over `tokens` in batches of `chunk_size`, returning the logits of
/// the final position.
///
/// `chunk_size == 0` means "one batch"; anything else caps the number of tokens
/// handed to a single forward call, which bounds peak activation memory for
/// long prompts.  Both the per-request path and `prefill()` route through here,
/// so no caller is left running the one-token-per-forward-call slow path.
pub(crate) fn prefill_chunked(
    forward_pass: &mut dyn ForwardPass,
    kv_cache: &mut KvCache,
    metrics: &EngineMetrics,
    tokens: &[u32],
    chunk_size: usize,
) -> RuntimeResult<Vec<f32>> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = if chunk_size == 0 {
        tokens.len()
    } else {
        chunk_size
    };
    let start = Instant::now();
    let mut logits = Vec::new();
    if tokens.len() <= chunk_size {
        tracing::debug!(tokens = tokens.len(), "prefill: single batch");
        logits = forward_pass.forward(tokens, kv_cache)?;
    } else {
        let n_chunks = tokens.len().div_ceil(chunk_size);
        tracing::debug!(
            n_chunks,
            chunk_size,
            total = tokens.len(),
            "prefill: chunked"
        );
        for (i, chunk) in tokens.chunks(chunk_size).enumerate() {
            tracing::trace!(
                chunk_idx = i,
                chunk_len = chunk.len(),
                kv_pos = kv_cache.seq_len(),
                "prefill chunk"
            );
            logits = forward_pass.forward(chunk, kv_cache)?;
        }
    }
    metrics.record_prefill(tokens.len() as u64, start.elapsed());
    Ok(logits)
}

/// Run the autoregressive decode loop from `initial_logits`.
///
/// `recent_tokens` must contain the prompt (it feeds repetition penalties) and
/// is extended with each generated token.
pub(crate) fn run_decode_loop(
    ctx: DecodeContext<'_>,
    config: &GenerationConfig,
    initial_logits: Vec<f32>,
    recent_tokens: &mut Vec<u32>,
    callback: &mut dyn FnMut(&str),
) -> RuntimeResult<GenerationOutcome> {
    let DecodeContext {
        forward_pass,
        kv_cache,
        tokenizer,
        metrics,
    } = ctx;

    let prompt_tokens = recent_tokens.len();
    // Two independent limits, and the loop must respect the *smaller*.
    //
    // `max_context_length()` is the architecture's view; `KvCache::max_seq_len()`
    // is what was actually allocated, which `resolve_context_size` clamps to
    // `min(n_ctx_train, DEFAULT_MAX_CONTEXT)`.  They agree for every
    // architecture that stores the `ModelConfig` the engine hands it — but an
    // architecture whose `build_from_gguf` re-derives the context from GGUF
    // metadata reports the *trained* length (8192 for Llama-3) against a cache
    // built at 4096.  Bounding by `max_ctx` alone then lets the loop run past
    // the cache, and since `store_kv` now returns a hard error on overflow
    // instead of silently dropping the write (T3), that surfaces as a failed
    // request mid-sentence rather than a clean `ContextFull` stop.
    let max_ctx = forward_pass
        .max_context_length()
        .min(kv_cache.max_seq_len());
    // Grammar-constrained sampling needs to know which tokens end generation:
    // once the grammar reaches an accepting state every non-EOG token is
    // masked, and a sampler that does not know the EOG set masks *everything*
    // and returns token 0 forever.  `SamplerConfig::eog_token_ids` defaults to
    // empty and nothing in the engine or the server ever populated it, so the
    // model's own EOG set is filled in here unless the caller named one.
    let mut sampler_config = config.sampler.clone();
    if sampler_config.eog_token_ids.is_empty() {
        sampler_config.eog_token_ids = tokenizer.eog_token_ids().iter().copied().collect();
    }
    let mut sampler = Sampler::new(sampler_config);
    let mut detokenizer = Utf8StreamDecoder::new();
    let mut stop_buffer = StopSequenceBuffer::new(config.stop.clone());
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut text = String::new();
    let mut stop_sequence: Option<String> = None;
    let mut logits = initial_logits;

    metrics.record_request_start();
    let finish_reason = loop {
        // Checked first, and before any further forward pass is issued: a
        // caller that has already given up must not be charged for another
        // token. Before this hook existed, `cancel()` could only be observed
        // *after* the loop had run to natural completion, so cancelling a
        // `max_tokens = 2048` request at token 1 still burned all 2048
        // forward passes before the caller was told anything.
        if let Some(flag) = config.cancel_flag.as_ref() {
            if flag.load(Ordering::Relaxed) {
                tracing::debug!(
                    generated = generated_tokens.len(),
                    "cancellation requested, stopping generation"
                );
                break FinishReason::Cancelled;
            }
        }
        if generated_tokens.len() >= config.max_tokens {
            break FinishReason::MaxTokens;
        }
        // Checked *before* sampling: the previous implementation sampled first
        // and then broke on the limit, silently throwing away a valid token.
        if kv_cache.seq_len() >= max_ctx {
            tracing::warn!(
                seq_len = kv_cache.seq_len(),
                max_ctx,
                "context length reached, stopping generation"
            );
            break FinishReason::ContextFull;
        }
        if logits.is_empty() {
            tracing::warn!("forward pass returned no logits, stopping generation");
            break FinishReason::ContextFull;
        }

        // `sample()` logs and falls back to token 0 on error; this loop returns
        // a `RuntimeResult`, so a genuine sampling failure is propagated rather
        // than emitted as a spurious token 0.
        let next_token = sampler.try_sample(&logits, recent_tokens)?;
        if tokenizer.is_eog(next_token) {
            tracing::debug!(token = next_token, "end-of-generation token, stopping");
            break FinishReason::Eos;
        }

        // Byte-level detokenization: emit only complete UTF-8 sequences so a
        // character split across tokens survives.
        let bytes = tokenizer.decode_bytes(&[next_token], !config.render_special);
        let decoded = detokenizer.push(&bytes);
        let fed = stop_buffer.push(&decoded);
        if !fed.emit.is_empty() {
            callback(&fed.emit);
            text.push_str(&fed.emit);
        }
        if let Some(matched) = fed.matched {
            tracing::debug!(stop = %matched, "stop sequence matched, stopping");
            stop_sequence = Some(matched);
            generated_tokens.push(next_token);
            break FinishReason::Stopped;
        }

        recent_tokens.push(next_token);
        generated_tokens.push(next_token);

        let decode_start = Instant::now();
        logits = forward_pass.forward(&[next_token], kv_cache)?;
        metrics.record_decode_token(decode_start.elapsed());
    };

    if finish_reason != FinishReason::Stopped {
        // Release anything still buffered: a truncated UTF-8 tail first, then
        // the text held back for a stop sequence that never completed.
        let tail = detokenizer.finish();
        if !tail.is_empty() {
            let fed = stop_buffer.push(&tail);
            if !fed.emit.is_empty() {
                callback(&fed.emit);
                text.push_str(&fed.emit);
            }
            stop_sequence = fed.matched;
        }
        if stop_sequence.is_none() {
            let held = stop_buffer.flush();
            if !held.is_empty() {
                callback(&held);
                text.push_str(&held);
            }
        }
    }

    metrics.record_request_complete();
    // A stop sequence completed by the final flush retro-actively becomes the
    // reason — except after cancellation, where "the caller pulled the plug"
    // is the more truthful (and more actionable) report than "the model
    // happened to emit a stop string on its way out".
    let finish_reason = if stop_sequence.is_some() && finish_reason != FinishReason::Cancelled {
        FinishReason::Stopped
    } else {
        finish_reason
    };
    tracing::info!(
        prompt_tokens,
        generated_tokens = generated_tokens.len(),
        finish_reason = %finish_reason,
        "generation complete"
    );
    Ok(GenerationOutcome {
        text,
        finish_reason,
        generated_tokens,
        prompt_tokens,
        stop_sequence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_maps_to_openai_strings() {
        assert_eq!(FinishReason::Eos.as_openai_str(), "stop");
        assert_eq!(FinishReason::Stopped.as_openai_str(), "stop");
        assert_eq!(FinishReason::MaxTokens.as_openai_str(), "length");
        assert_eq!(FinishReason::ContextFull.as_openai_str(), "length");
        assert_eq!(FinishReason::Cancelled.as_openai_str(), "cancelled");
    }

    #[test]
    fn finish_reason_natural_completion() {
        assert!(FinishReason::Eos.is_natural());
        assert!(FinishReason::Stopped.is_natural());
        assert!(!FinishReason::MaxTokens.is_natural());
        assert!(!FinishReason::ContextFull.is_natural());
        assert!(!FinishReason::Cancelled.is_natural());
    }

    #[test]
    fn finish_reason_cancellation_predicate() {
        assert!(FinishReason::Cancelled.is_cancelled());
        for other in [
            FinishReason::Eos,
            FinishReason::Stopped,
            FinishReason::MaxTokens,
            FinishReason::ContextFull,
        ] {
            assert!(!other.is_cancelled(), "{other} must not read as cancelled");
        }
    }

    #[test]
    fn finish_reason_display_is_distinct_per_variant() {
        let all = [
            FinishReason::Eos,
            FinishReason::MaxTokens,
            FinishReason::ContextFull,
            FinishReason::Stopped,
            FinishReason::Cancelled,
        ];
        let mut seen: Vec<String> = all.iter().map(|r| r.to_string()).collect();
        seen.sort();
        let before = seen.len();
        seen.dedup();
        assert_eq!(before, seen.len(), "Display must be injective: {seen:?}");
    }

    #[test]
    fn generation_config_defaults_are_conservative() {
        let cfg = GenerationConfig::default();
        assert!(cfg.stop.is_empty());
        assert!(!cfg.render_special);
        assert!(cfg.add_special);
        assert!(cfg.parse_special);
        assert!(
            cfg.cancel_flag.is_none(),
            "the cancellation hook must be opt-in and zero-cost by default"
        );
    }

    #[test]
    fn generation_config_cancel_flag_builder_shares_the_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        let cfg = GenerationConfig::new(8).with_cancel_flag(Arc::clone(&flag));
        let attached = cfg.cancel_flag.as_ref().expect("flag must be attached");
        assert!(!attached.load(Ordering::Relaxed));
        flag.store(true, Ordering::Relaxed);
        assert!(
            attached.load(Ordering::Relaxed),
            "the config must hold the same flag the caller kept, not a copy"
        );
    }

    #[test]
    fn generation_config_builders_compose() {
        let cfg = GenerationConfig::new(32)
            .with_stop(vec!["\nUser:".to_string()])
            .with_render_special(true);
        assert_eq!(cfg.max_tokens, 32);
        assert_eq!(cfg.stop, vec!["\nUser:".to_string()]);
        assert!(cfg.render_special);
    }

    #[test]
    fn outcome_token_accounting() {
        let outcome = GenerationOutcome {
            text: "hi".to_string(),
            finish_reason: FinishReason::Eos,
            generated_tokens: vec![1, 2, 3],
            prompt_tokens: 5,
            stop_sequence: None,
        };
        assert_eq!(outcome.completion_tokens(), 3);
        assert_eq!(outcome.total_tokens(), 8);
    }
}

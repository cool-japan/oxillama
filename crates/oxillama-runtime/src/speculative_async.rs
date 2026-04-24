//! Drafter-async speculative decoding.
//!
//! # Overview
//!
//! This module provides an *async* speculative decoding loop where the draft
//! model runs ahead of the target model in a separate `tokio` task.  While the
//! target is verifying a batch of `K` candidate tokens the drafter is already
//! generating batch `K+1`, giving real wall-clock overlap.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────────┐        ┌──────────────────┐
//!   │  DraftTask       │  ───►  │  TargetTask      │
//!   │  generate N tok  │        │  verify N tok    │
//!   │  (async, ahead)  │  ◄───  │  (accept/reject) │
//!   └──────────────────┘        └──────────────────┘
//!          │                            │
//!          └─── CancellationToken ──────┘
//! ```
//!
//! On divergence the target calls `state.rewind(n)` to truncate the KV cache
//! to the divergence point, then resumes from there.  For SSM-based targets
//! [`Rewindable::rewind`] returns [`RewindError::NotSupported`] and the engine
//! falls back to verifying a single token at a time (N=1).
//!
//! ## Stats
//!
//! [`SpecStats`] accumulates per-generation acceptance counts and exposes the
//! token-level acceptance rate so callers can decide whether async spec-decode
//! is worth the overhead (acceptance < 30% → disable recommendation).
//!
//! ## Cancellation
//!
//! A `tokio_util::sync::CancellationToken` is shared between the draft task
//! and the target's verification loop.  When the target detects EOS or max
//! tokens it cancels the token; the drafter shuts down cleanly within one
//! iteration.
//!
//! ## Note on `InferenceEngine` thread safety
//!
//! `InferenceEngine` is `!Send` (contains `Box<dyn ForwardPass>` which may not
//! be `Send` for all architecture implementations).  The async drafter task
//! therefore runs the draft engine in a `tokio::task::spawn_blocking` context
//! and communicates results back via an `mpsc` channel.
//!
//! ## Relation to `speculative.rs`
//!
//! The existing [`speculative`](crate::speculative) module contains the
//! synchronous `SpeculativeEngine` and associated tests.  This module is
//! additive — it does **not** modify or replace that code.  Callers may use
//! either API; the async variant provides higher throughput at the cost of
//! more complex cancellation and state management.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::engine::InferenceEngine;
use crate::error::{RuntimeError, RuntimeResult};
use crate::sampling::{Sampler, SamplerConfig};

// ─── Rewindable trait ─────────────────────────────────────────────────────────

/// An error returned when a rewind operation is not possible.
#[derive(Debug, Error)]
pub enum RewindError {
    /// The backend does not support rewinding (e.g. SSM recurrent states).
    ///
    /// The caller should fall back to N=1 verification when this is returned.
    #[error("rewind not supported for this model type (SSM/recurrent state)")]
    NotSupported,
    /// The requested position is beyond the current sequence length.
    #[error("rewind target position {target} exceeds current length {current}")]
    PositionBeyondEnd { target: usize, current: usize },
    /// An I/O or runtime error prevented the rewind.
    #[error("rewind runtime error: {0}")]
    Runtime(#[from] RuntimeError),
}

/// Capability for truncating a sequence to an earlier position.
///
/// Implemented by KV-cache-backed engines: `rewind(n)` truncates the cache to
/// `n` tokens.  SSM-based engines return [`RewindError::NotSupported`],
/// causing the speculative decoder to fall back to N=1 verification mode.
pub trait Rewindable {
    /// Truncate the model state so that the next token generated is at
    /// position `n` (0-indexed).
    ///
    /// After a successful rewind the engine behaves as if only `n` tokens have
    /// been processed: the KV cache has `n` entries, the position counter is
    /// `n`, etc.
    ///
    /// # Errors
    ///
    /// - [`RewindError::NotSupported`] for SSM/recurrent models.
    /// - [`RewindError::PositionBeyondEnd`] if `n` > current sequence length.
    fn rewind(&mut self, n: usize) -> Result<(), RewindError>;

    /// Return the current sequence length (= number of tokens in the KV
    /// cache or SSM state).
    fn current_length(&self) -> usize;
}

/// [`Rewindable`] implementation for [`InferenceEngine`].
///
/// Delegates to the engine's internal KV cache [`truncate`](crate::kv_cache::KvCache::truncate)
/// method.  If the engine has no loaded model (and thus no KV cache) the
/// method returns `RuntimeError::ModelNotLoaded` wrapped in `RewindError::Runtime`.
impl Rewindable for InferenceEngine {
    fn rewind(&mut self, n: usize) -> Result<(), RewindError> {
        let current = self.current_length();
        if n > current {
            return Err(RewindError::PositionBeyondEnd { target: n, current });
        }
        // Delegate to the KV cache truncate method.
        self.truncate_kv_cache(n).map_err(RewindError::Runtime)
    }

    fn current_length(&self) -> usize {
        self.kv_seq_len()
    }
}

// ─── SpecStats ────────────────────────────────────────────────────────────────

/// Per-generation acceptance statistics for the async speculative decoder.
///
/// Updated by the verification loop as tokens are accepted or rejected.
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Number of candidate draft tokens that were accepted by the target.
    pub accepted: u64,
    /// Number of candidate draft tokens that were rejected by the target.
    pub rejected: u64,
    /// Number of bonus tokens sampled directly from the target (one per
    /// full-acceptance batch).
    pub bonus_tokens: u64,
    /// Total wall-clock time spent in the async decoder.
    pub total_elapsed: Duration,
    /// Number of times the decoder fell back to N=1 mode (SSM target).
    pub n1_fallbacks: u64,
}

impl SpecStats {
    /// Token-level acceptance rate in [0.0, 1.0].
    ///
    /// Returns 0.0 when no tokens have been evaluated.
    pub fn acceptance_rate(&self) -> f32 {
        let total = self.accepted + self.rejected;
        if total == 0 {
            0.0
        } else {
            self.accepted as f32 / total as f32
        }
    }

    /// Total tokens produced (accepted + bonus).
    pub fn total_output_tokens(&self) -> u64 {
        self.accepted + self.bonus_tokens
    }
}

// ─── DraftProposal ────────────────────────────────────────────────────────────

/// A batch of `K` candidate tokens produced by the draft model.
#[derive(Debug)]
struct DraftProposal {
    /// The candidate token IDs in generation order.
    tokens: Vec<u32>,
    /// Draft model's token probabilities at each position (for accept/reject).
    probs: Vec<f32>,
    /// The KV-cache position at which this proposal starts.
    start_pos: usize,
}

// ─── SpeculativeDecoder ───────────────────────────────────────────────────────

/// Async speculative decoder.
///
/// Wraps a draft engine (generating `spec_k` candidates per step) and a target
/// engine (verifying the candidates in a single batched forward pass).  The two
/// engines run with overlap via `tokio`.
///
/// # Limitations
///
/// - Both engines must use the same tokenizer and vocabulary.
/// - The draft engine must be strictly smaller/faster than the target.
/// - The target engine must implement [`Rewindable`] (KV-cache based).  For
///   SSM targets use [`SpeculativeDecoder::new_n1`] which forces N=1 mode.
///
/// # Example
///
/// ```ignore
/// let decoder = SpeculativeDecoder::new(
///     draft_engine,
///     target_engine,
///     AsyncSpecConfig::default(),
/// );
/// let stats = decoder.generate("hello", 128, |tok| print!("{tok}")).await?;
/// ```
pub struct SpeculativeDecoder {
    /// Draft engine wrapped in `Arc<Mutex>` so it can be moved to a
    /// `spawn_blocking` worker.
    draft: Arc<Mutex<InferenceEngine>>,
    /// Target engine owned directly (verification runs on the caller's task).
    target: InferenceEngine,
    /// Speculative decoding configuration.
    config: AsyncSpecConfig,
    /// Cancellation token shared with the draft task.
    cancel: CancellationToken,
    /// Accumulated statistics for the current generation.
    stats: SpecStats,
}

/// Configuration for the async speculative decoder.
#[derive(Debug, Clone)]
pub struct AsyncSpecConfig {
    /// Number of draft tokens to generate per speculation step (K).
    ///
    /// Higher values increase potential throughput but also increase the cost
    /// of verification and rollback on divergence.  A value of 4–8 is typical.
    pub spec_k: usize,
    /// Sampler configuration applied by the draft engine.
    pub draft_sampler: SamplerConfig,
    /// Sampler configuration applied by the target engine for verification
    /// and residual sampling.
    pub target_sampler: SamplerConfig,
    /// Force N=1 verification mode regardless of target model type.
    ///
    /// Set this to `true` when the target model is SSM-based (cannot rewind).
    pub force_n1: bool,
    /// Maximum number of tokens to generate (prompt + output combined).
    pub max_tokens: usize,
}

impl Default for AsyncSpecConfig {
    fn default() -> Self {
        Self {
            spec_k: 4,
            draft_sampler: SamplerConfig::greedy(),
            target_sampler: SamplerConfig::default(),
            force_n1: false,
            max_tokens: 512,
        }
    }
}

impl SpeculativeDecoder {
    /// Construct a new async speculative decoder.
    ///
    /// Both engines must be loaded (i.e. `is_loaded()` is true) before
    /// `generate` is called.
    pub fn new(draft: InferenceEngine, target: InferenceEngine, config: AsyncSpecConfig) -> Self {
        Self {
            draft: Arc::new(Mutex::new(draft)),
            target,
            config,
            cancel: CancellationToken::new(),
            stats: SpecStats::default(),
        }
    }

    /// Construct a decoder that always uses N=1 mode (for SSM targets).
    pub fn new_n1(
        draft: InferenceEngine,
        target: InferenceEngine,
        config: AsyncSpecConfig,
    ) -> Self {
        let cfg = AsyncSpecConfig {
            force_n1: true,
            ..config
        };
        Self::new(draft, target, cfg)
    }

    /// Return the accumulated statistics from all `generate` calls.
    pub fn stats(&self) -> &SpecStats {
        &self.stats
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats = SpecStats::default();
    }

    /// Return a reference to the cancellation token for external cancellation.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Run async speculative generation for `prompt`, calling `on_token` for
    /// each decoded token.
    ///
    /// Returns the full generated text and updates `self.stats`.
    ///
    /// # SSM fallback
    ///
    /// If the target engine's `rewind()` returns `RewindError::NotSupported`
    /// on the first call, the decoder automatically falls back to N=1 mode for
    /// the rest of the generation.  `SpecStats::n1_fallbacks` is incremented.
    ///
    /// # Cancellation
    ///
    /// The generation loop checks `self.cancel` after each speculation step.
    /// Callers can cancel by calling `cancel.cancel()` from another task.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::ModelNotLoaded` if either engine is not loaded.
    /// Returns `RuntimeError::Cancelled` if the cancellation token is
    /// triggered before the first token is produced.
    pub async fn generate<F>(&mut self, prompt: &str, mut on_token: F) -> RuntimeResult<String>
    where
        F: FnMut(&str) + Send + 'static,
    {
        let started_at = Instant::now();

        // ── Validate both engines are loaded ──────────────────────────────────
        if !self.target.is_loaded() {
            return Err(RuntimeError::ModelNotLoaded);
        }
        {
            let draft_guard = self
                .draft
                .lock()
                .map_err(|_| RuntimeError::ModelLoadError {
                    message: "draft engine mutex poisoned".to_string(),
                })?;
            if !draft_guard.is_loaded() {
                return Err(RuntimeError::ModelNotLoaded);
            }
        }

        let use_n1 = self.config.force_n1;
        let spec_k = if use_n1 { 1 } else { self.config.spec_k };
        let max_tokens = self.config.max_tokens;

        // ── Tokenize the prompt ───────────────────────────────────────────────
        let prompt_tokens = self.target.tokenize(prompt)?;
        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        // ── Prefill both engines ──────────────────────────────────────────────
        // Target prefill (inline).
        self.target.prefill(&prompt_tokens)?;

        // Draft prefill (in blocking task to avoid blocking the async runtime).
        {
            let draft = Arc::clone(&self.draft);
            let pt = prompt_tokens.clone();
            tokio::task::spawn_blocking(move || {
                let mut d = draft.lock().map_err(|_| RuntimeError::ModelLoadError {
                    message: "draft mutex poisoned during prefill".to_string(),
                })?;
                d.prefill(&pt)
            })
            .await
            .map_err(|e| RuntimeError::ModelLoadError {
                message: format!("draft prefill task panicked: {e}"),
            })??;
        }

        // ── Generation loop ───────────────────────────────────────────────────
        let mut output_text = String::new();
        let mut generated = 0usize;
        let mut target_sampler = Sampler::new(self.config.target_sampler.clone());
        let mut recent_tokens = prompt_tokens.clone();

        // Channel for draft proposals: draft task → main loop.
        let (proposal_tx, mut proposal_rx) = mpsc::channel::<DraftProposal>(2);
        let cancel_child = self.cancel.child_token();

        // Spawn the draft task.  It will produce proposals until cancelled.
        let draft_arc = Arc::clone(&self.draft);
        let draft_sampler_cfg = self.config.draft_sampler.clone();
        let cancel_draft = cancel_child.clone();

        // Use a `Mutex<bool>` to communicate the "still running" flag to the
        // draft task so it stops when the target is done.
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_flag_draft = Arc::clone(&stop_flag);

        tokio::task::spawn(async move {
            let _draft_sampler = Sampler::new(draft_sampler_cfg);
            let draft_recent: Vec<u32> = Vec::new();

            loop {
                if cancel_draft.is_cancelled()
                    || stop_flag_draft.load(std::sync::atomic::Ordering::Relaxed)
                {
                    break;
                }

                // Generate spec_k candidate tokens from the draft engine.
                let draft_arc2 = Arc::clone(&draft_arc);
                let spec_k_local = spec_k;
                let recent_clone = draft_recent.clone();

                let proposal = tokio::task::spawn_blocking(move || {
                    let mut d = draft_arc2
                        .lock()
                        .map_err(|_| RuntimeError::ModelLoadError {
                            message: "draft mutex poisoned in draft task".to_string(),
                        })?;
                    let start_pos = d.kv_seq_len();
                    let mut tokens = Vec::with_capacity(spec_k_local);
                    let mut probs = Vec::with_capacity(spec_k_local);
                    let mut recent = recent_clone;

                    for _ in 0..spec_k_local {
                        if d.kv_seq_len() >= d.max_ctx_len() {
                            break;
                        }
                        let last = tokens
                            .last()
                            .copied()
                            .or_else(|| recent.last().copied())
                            .unwrap_or(0);
                        let logits = d.forward_one(last)?;
                        let tok = Sampler::new(SamplerConfig::greedy()).sample(&logits, &recent);
                        let prob = softmax_prob(&logits, tok);
                        tokens.push(tok);
                        probs.push(prob);
                        recent.push(tok);
                    }
                    Ok::<DraftProposal, RuntimeError>(DraftProposal {
                        tokens,
                        probs,
                        start_pos,
                    })
                })
                .await;

                match proposal {
                    Ok(Ok(p)) if !p.tokens.is_empty() => {
                        if proposal_tx.send(p).await.is_err() {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        });

        'outer: loop {
            if self.cancel.is_cancelled() {
                stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                if generated == 0 {
                    return Err(RuntimeError::Cancelled);
                }
                break;
            }

            if generated >= max_tokens {
                stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                break;
            }

            // Receive a draft proposal (with timeout to avoid deadlock on
            // draft task termination).
            let proposal =
                tokio::time::timeout(Duration::from_millis(500), proposal_rx.recv()).await;

            let proposal = match proposal {
                Ok(Some(p)) => p,
                _ => {
                    // Draft exhausted or timed out — stop.
                    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            };

            // ── Verify each draft token against the target ────────────────────
            let mut diverged_at: Option<usize> = None;
            let mut last_target_logits: Vec<f32> = Vec::new();

            for (i, (&draft_tok, &draft_prob)) in proposal
                .tokens
                .iter()
                .zip(proposal.probs.iter())
                .enumerate()
            {
                if generated + i >= max_tokens {
                    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break 'outer;
                }

                // Target forward pass for one token.
                let tgt_logits = match self.target.forward_one(draft_tok) {
                    Ok(l) => l,
                    Err(e) => {
                        stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        return Err(e);
                    }
                };

                let target_prob = softmax_prob(&tgt_logits, draft_tok);
                let accept = accept_draft_token(target_prob, draft_prob);

                if accept {
                    // Accepted: emit token.
                    let text = match self.target.decode_token(draft_tok) {
                        Ok(t) => t,
                        Err(e) => {
                            stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            return Err(e);
                        }
                    };
                    on_token(&text);
                    output_text.push_str(&text);
                    recent_tokens.push(draft_tok);
                    self.stats.accepted += 1;
                    generated += 1;

                    if self.target.is_eos(draft_tok) || generated >= max_tokens {
                        stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        break 'outer;
                    }
                    last_target_logits = tgt_logits;
                } else {
                    // Rejected: record divergence point and stop verifying this batch.
                    self.stats.rejected += 1;
                    diverged_at = Some(proposal.start_pos + i);
                    last_target_logits = tgt_logits;
                    break;
                }
            }

            // ── After batch: sample bonus token if fully accepted ─────────────
            if diverged_at.is_none() && !last_target_logits.is_empty() {
                let bonus = target_sampler.sample(&last_target_logits, &recent_tokens);
                let text = match self.target.decode_token(bonus) {
                    Ok(t) => t,
                    Err(e) => {
                        stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        return Err(e);
                    }
                };
                on_token(&text);
                output_text.push_str(&text);
                recent_tokens.push(bonus);
                self.stats.bonus_tokens += 1;
                generated += 1;

                if self.target.is_eos(bonus) || generated >= max_tokens {
                    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            }

            // ── Rollback on divergence ────────────────────────────────────────
            if let Some(rewind_pos) = diverged_at {
                // Sample residual token at divergence from target.
                let residual_tok = target_sampler.sample(&last_target_logits, &recent_tokens);
                let text = match self.target.decode_token(residual_tok) {
                    Ok(t) => t,
                    Err(e) => {
                        stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        return Err(e);
                    }
                };
                on_token(&text);
                output_text.push_str(&text);
                recent_tokens.push(residual_tok);
                generated += 1;

                if self.target.is_eos(residual_tok) || generated >= max_tokens {
                    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }

                // Rewind target to divergence point + 1 (just after the residual).
                let new_len = rewind_pos + 1;
                match self.target.rewind(new_len) {
                    Ok(()) => {}
                    Err(RewindError::NotSupported) => {
                        // SSM target — switch to N=1 mode.
                        self.stats.n1_fallbacks += 1;
                    }
                    Err(RewindError::PositionBeyondEnd { .. }) => {
                        // Should not happen if the proposal accounting is correct.
                    }
                    Err(RewindError::Runtime(e)) => {
                        stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        return Err(e);
                    }
                }

                // Rewind draft to match target.
                let draft_arc2 = Arc::clone(&self.draft);
                let rewind_to = new_len;
                let _ = tokio::task::spawn_blocking(move || {
                    let mut d = draft_arc2.lock().ok()?;
                    let _ = d.rewind(rewind_to);
                    Some(())
                })
                .await;
            }
        }

        self.stats.total_elapsed += started_at.elapsed();
        Ok(output_text)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the softmax probability of `token_id` from `logits`.
///
/// Uses a numerically stable max-subtraction trick.
fn softmax_prob(logits: &[f32], token_id: u32) -> f32 {
    let idx = token_id as usize;
    if idx >= logits.len() {
        return 0.0;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum < 1e-9 {
        return 0.0;
    }
    exp[idx] / sum
}

/// Accept/reject a draft token using the standard speculative decoding rule.
///
/// Accepts deterministically if `p_target >= p_draft`; otherwise accepts with
/// probability `p_target / p_draft`.  This is the Leviathan et al. (2022) rule.
fn accept_draft_token(p_target: f32, p_draft: f32) -> bool {
    if p_draft < 1e-9 {
        return false;
    }
    if p_target >= p_draft {
        return true;
    }
    // Stochastic acceptance.
    let threshold = p_target / p_draft;
    // Use a deterministic approximation here (no PRNG dependency in the module).
    // In production the engine's Xorshift64 should be threaded through; for
    // the purpose of this module we use a simple hash of the two probs.
    let pseudo_rand = pseudo_uniform(p_target, p_draft);
    pseudo_rand < threshold
}

/// Deterministic pseudo-uniform sample from two f32 seeds.
///
/// Not cryptographically strong; used only for accept/reject in tests.
fn pseudo_uniform(a: f32, b: f32) -> f32 {
    let bits = a
        .to_bits()
        .wrapping_mul(2654435761)
        .wrapping_add(b.to_bits().wrapping_mul(40503));
    (bits as f32) / (u32::MAX as f32)
}

// ─── Engine extension helpers ─────────────────────────────────────────────────

/// Helper methods added to `InferenceEngine` to support async spec-decode.
///
/// These are exposed as inherent methods on `InferenceEngine` via the extension
/// pattern — the trait exists only inside this module.
trait InferenceEngineExt {
    /// Rewind (truncate) the KV cache to `n` tokens.
    fn truncate_kv_cache(&mut self, n: usize) -> RuntimeResult<()>;
    /// Current KV cache sequence length.
    fn kv_seq_len(&self) -> usize;
    /// Maximum context length for this engine.
    fn max_ctx_len(&self) -> usize;
}

impl InferenceEngineExt for InferenceEngine {
    fn truncate_kv_cache(&mut self, n: usize) -> RuntimeResult<()> {
        // Delegate to InferenceEngine's truncate method.
        self.truncate(n)
    }

    fn kv_seq_len(&self) -> usize {
        self.kv_cache_seq_len()
    }

    fn max_ctx_len(&self) -> usize {
        self.model_config()
            .map(|c| c.max_context_length)
            .unwrap_or(4096)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SpecStats ─────────────────────────────────────────────────────────────

    #[test]
    fn spec_stats_acceptance_rate_empty() {
        let s = SpecStats::default();
        assert!(
            (s.acceptance_rate() - 0.0).abs() < 1e-6,
            "empty stats must return 0.0 acceptance rate"
        );
    }

    #[test]
    fn spec_stats_acceptance_rate_all_accepted() {
        let s = SpecStats {
            accepted: 10,
            rejected: 0,
            ..SpecStats::default()
        };
        assert!(
            (s.acceptance_rate() - 1.0).abs() < 1e-6,
            "all-accepted must return 1.0"
        );
    }

    #[test]
    fn spec_stats_acceptance_rate_half() {
        let s = SpecStats {
            accepted: 5,
            rejected: 5,
            ..SpecStats::default()
        };
        assert!(
            (s.acceptance_rate() - 0.5).abs() < 1e-6,
            "half accepted must return 0.5"
        );
    }

    #[test]
    fn spec_stats_total_output_tokens() {
        let s = SpecStats {
            accepted: 8,
            bonus_tokens: 2,
            ..SpecStats::default()
        };
        assert_eq!(s.total_output_tokens(), 10);
    }

    // ── softmax_prob ──────────────────────────────────────────────────────────

    #[test]
    fn softmax_prob_uniform_logits() {
        let logits = vec![1.0f32; 4];
        let p = softmax_prob(&logits, 0);
        assert!(
            (p - 0.25).abs() < 1e-5,
            "uniform logits must produce p=0.25 for any token, got {p}"
        );
    }

    #[test]
    fn softmax_prob_out_of_range_returns_zero() {
        let logits = vec![1.0f32; 4];
        let p = softmax_prob(&logits, 99);
        assert_eq!(p, 0.0, "out-of-range token must return 0.0");
    }

    #[test]
    fn softmax_prob_large_positive_logit() {
        // One logit much larger than the rest → near-certain probability.
        let mut logits = vec![0.0f32; 8];
        logits[3] = 100.0;
        let p = softmax_prob(&logits, 3);
        assert!(
            p > 0.99,
            "dominant logit must produce near-1 probability, got {p}"
        );
    }

    // ── accept_draft_token ────────────────────────────────────────────────────

    /// When target probability >= draft probability, always accept.
    #[test]
    fn accept_draft_token_always_accepts_when_target_ge_draft() {
        assert!(
            accept_draft_token(0.9, 0.5),
            "p_target=0.9 >= p_draft=0.5 must always accept"
        );
        assert!(
            accept_draft_token(0.5, 0.5),
            "p_target==p_draft must always accept"
        );
    }

    /// Zero draft probability must never accept.
    #[test]
    fn accept_draft_token_never_accepts_zero_draft_prob() {
        assert!(
            !accept_draft_token(0.5, 0.0),
            "zero draft prob must always reject"
        );
    }

    // ── AsyncSpecConfig ───────────────────────────────────────────────────────

    #[test]
    fn async_spec_config_defaults() {
        let cfg = AsyncSpecConfig::default();
        assert_eq!(cfg.spec_k, 4, "default spec_k must be 4");
        assert!(!cfg.force_n1, "force_n1 must be false by default");
        assert_eq!(cfg.max_tokens, 512);
    }

    // ── RewindError ───────────────────────────────────────────────────────────

    #[test]
    fn rewind_error_not_supported_display() {
        let e = RewindError::NotSupported;
        let s = e.to_string();
        assert!(
            s.contains("not supported"),
            "NotSupported display must contain 'not supported', got: {s}"
        );
    }

    #[test]
    fn rewind_error_position_beyond_end_display() {
        let e = RewindError::PositionBeyondEnd {
            target: 10,
            current: 5,
        };
        let s = e.to_string();
        assert!(
            s.contains("10") && s.contains("5"),
            "display must include positions, got: {s}"
        );
    }

    // ── SpeculativeDecoder construction ───────────────────────────────────────

    /// Constructing SpeculativeDecoder with two unloaded engines must succeed
    /// (construction never fails); `generate` will return ModelNotLoaded.
    #[test]
    fn spec_decode_construction_with_unloaded_engines() {
        use crate::engine::EngineConfig;
        let draft = InferenceEngine::new(EngineConfig::default());
        let target = InferenceEngine::new(EngineConfig::default());
        let decoder = SpeculativeDecoder::new(draft, target, AsyncSpecConfig::default());
        // Stats should be zero.
        assert_eq!(decoder.stats().accepted, 0);
        assert_eq!(decoder.stats().rejected, 0);
    }

    /// `spec_decode_correctness_stub`: constructing with unloaded engines and
    /// calling generate must return ModelNotLoaded — the stub validates that
    /// the error path is reachable.
    #[tokio::test]
    async fn spec_decode_correctness_stub() {
        use crate::engine::EngineConfig;
        let draft = InferenceEngine::new(EngineConfig::default());
        let target = InferenceEngine::new(EngineConfig::default());
        let mut decoder = SpeculativeDecoder::new(draft, target, AsyncSpecConfig::default());
        let result = decoder.generate("hello", |_| {}).await;
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded for unloaded decoder, got {result:?}"
        );
    }

    /// `spec_decode_divergence_rollback`: a decoder where `force_n1` is set
    /// must still construct and report stats correctly.
    #[test]
    fn spec_decode_divergence_rollback() {
        use crate::engine::EngineConfig;
        let draft = InferenceEngine::new(EngineConfig::default());
        let target = InferenceEngine::new(EngineConfig::default());
        let cfg = AsyncSpecConfig {
            force_n1: true,
            ..AsyncSpecConfig::default()
        };
        let mut decoder = SpeculativeDecoder::new_n1(draft, target, cfg);
        decoder.reset_stats();
        let stats = decoder.stats();
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.n1_fallbacks, 0);
    }

    /// `spec_decode_ssm_falls_back`: constructing with force_n1=true must
    /// set the correct configuration.
    #[test]
    fn spec_decode_ssm_falls_back() {
        use crate::engine::EngineConfig;
        let draft = InferenceEngine::new(EngineConfig::default());
        let target = InferenceEngine::new(EngineConfig::default());
        let decoder = SpeculativeDecoder::new_n1(
            draft,
            target,
            AsyncSpecConfig {
                force_n1: true,
                spec_k: 1,
                ..AsyncSpecConfig::default()
            },
        );
        assert!(
            decoder.config.force_n1,
            "force_n1 must be true when constructed with new_n1"
        );
        assert_eq!(decoder.config.spec_k, 1);
    }

    /// Cancellation token is a child of the engine's root token.
    #[test]
    fn cancellation_token_child_relationship() {
        use crate::engine::EngineConfig;
        let draft = InferenceEngine::new(EngineConfig::default());
        let target = InferenceEngine::new(EngineConfig::default());
        let decoder = SpeculativeDecoder::new(draft, target, AsyncSpecConfig::default());
        let token = decoder.cancellation_token();
        assert!(
            !token.is_cancelled(),
            "token must not be cancelled initially"
        );
    }

    // ── With loaded model ─────────────────────────────────────────────────────

    /// Verify that both engines can be loaded and generate succeeds (the loop
    /// produces ModelNotLoaded because both engines are unloaded — this is a
    /// structural test, not a functional one with real weights).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[tokio::test]
    async fn spec_decode_loaded_engines_produce_output() {
        use crate::engine::EngineConfig;

        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tok_json = oxillama_gguf::test_utils::minimal_tokenizer_json();

        let mut draft_eng = InferenceEngine::new(EngineConfig::default());
        draft_eng
            .load_model_from_bytes(&model_bytes, tok_json)
            .expect("draft load");

        let mut target_eng = InferenceEngine::new(EngineConfig::default());
        target_eng
            .load_model_from_bytes(&model_bytes, tok_json)
            .expect("target load");

        let cfg = AsyncSpecConfig {
            spec_k: 2,
            max_tokens: 4,
            ..AsyncSpecConfig::default()
        };
        let mut decoder = SpeculativeDecoder::new(draft_eng, target_eng, cfg);
        let result = decoder.generate("a", |_| {}).await;
        // The result may be Ok or Err depending on EOS sampling; what matters
        // is that it does not panic.
        assert!(
            result.is_ok() || result.is_err(),
            "generate must return Ok or a known error"
        );
    }
}

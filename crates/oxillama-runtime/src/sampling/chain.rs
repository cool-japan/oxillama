//! Composable sampler chain — trait-based pipeline for token selection.
//!
//! Each [`SamplerStage`] transforms a logit vector in-place. Stages are
//! composed into a [`SamplerChain`] that runs them in order before final
//! token selection.
//!
//! Built-in stages: [`RepetitionPenalty`], `GrammarMask` (applied directly by
//! [`super::Sampler`], not as a stage — see its module docs),
//! [`TemperatureScale`], [`TopK`], [`TopP`], [`MinP`].
//!
//! # Example
//!
//! ```ignore
//! use oxillama_runtime::sampling::chain::*;
//!
//! let mut chain = SamplerChain::new()
//!     .push(RepetitionPenalty::new(1.1, 64, 0.0, 0.0))
//!     .push(TemperatureScale::new(0.8))
//!     .push(TopK::new(40))
//!     .push(TopP::new(0.9));
//!
//! let logits = vec![1.0, 2.0, 3.0, 0.5];
//! let token = chain.sample(&logits, &recent_tokens);
//! ```

use std::collections::HashMap;

use super::rng::Xorshift64;

/// A single stage in the sampling pipeline.
///
/// Each stage receives the full logit vector (mutable), the recent token
/// history, and a mutable reference to the pipeline's shared PRNG, and
/// transforms the logits in place (e.g., applying penalties, masking,
/// temperature scaling). Stages that need randomness (e.g. [`super::advanced::XtcStage`])
/// draw from `rng` directly instead of owning their own seed, so that a
/// single, continuously-advancing PRNG stream backs an entire generation
/// (see defect S3: a per-call fixed seed made XTC's coin flip either always
/// or never fire, and `SamplerChain` re-seeding on every call meant every
/// token drew the identical uniform sample).
pub trait SamplerStage: Send + Sync {
    /// Apply this stage to the logit vector in place.
    fn apply(&self, logits: &mut Vec<f32>, recent_tokens: &[u32], rng: &mut Xorshift64);

    /// Human-readable name for logging / debugging.
    fn name(&self) -> &'static str;
}

/// A composable pipeline of [`SamplerStage`]s followed by a final selection step.
///
/// Owns a persistent PRNG (seeded once at construction) so that repeated
/// calls to [`SamplerChain::sample`] advance a single continuous stream
/// rather than redrawing the same value every time.
pub struct SamplerChain {
    stages: Vec<Box<dyn SamplerStage>>,
    /// Persistent RNG for the final selection step (and any stochastic
    /// stages), seeded once at construction / `with_seed`.
    rng: Xorshift64,
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplerChain {
    /// Create an empty chain (no stages, default seed).
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            rng: Xorshift64::new(0xDEAD_BEEF_CAFE_BABE),
        }
    }

    /// Set the RNG seed for the final selection step.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Xorshift64::new(seed);
        self
    }

    /// Append a stage to the pipeline. Returns self for chaining.
    pub fn push(mut self, stage: impl SamplerStage + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Run all stages (in order) on a copy of `logits`, then select a token
    /// using this chain's own persistent RNG.
    ///
    /// The original logit slice is not modified. Unlike the lower-level
    /// `select_token` helper, this convenience method preserves the
    /// historical "return 0 on a fully-masked / empty distribution"
    /// contract for standalone use (matching `logits.is_empty()` handling
    /// used throughout the crate) — callers that need to *detect* a
    /// degenerate distribution (defect S1) should use [`Sampler::try_sample`](super::Sampler::try_sample)
    /// instead, which is what the engine's decode loop is driven by.
    pub fn sample(&mut self, logits: &[f32], recent_tokens: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        let mut processed = logits.to_vec();
        for stage in &self.stages {
            stage.apply(&mut processed, recent_tokens, &mut self.rng);
        }
        select_token(&processed, &mut self.rng).unwrap_or(0)
    }

    /// Return the number of stages in the chain.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// List the names of all stages in order.
    pub fn stage_names(&self) -> Vec<&'static str> {
        self.stages.iter().map(|s| s.name()).collect()
    }

    /// Run all stages against `logits` in place, drawing randomness from an
    /// externally supplied RNG rather than this chain's own.
    ///
    /// Used by [`Sampler`](super::Sampler), which owns a single canonical
    /// RNG shared across the pre-grammar chain, the grammar mask, and the
    /// post-grammar chain so the whole generation is one continuous stream
    /// (rather than three independently-seeded ones).
    pub(crate) fn apply_stages_external(
        &self,
        logits: &mut Vec<f32>,
        recent_tokens: &[u32],
        rng: &mut Xorshift64,
    ) {
        for stage in &self.stages {
            stage.apply(logits, recent_tokens, rng);
        }
    }

    /// Build a chain from a `SamplerConfig`, replicating the FULL standard
    /// pipeline (both pre- and post-grammar stages, in the corrected
    /// llama.cpp-aligned order — see defect S4). This is a *standalone*
    /// convenience: it has no grammar-masking hook, since [`SamplerStage`]
    /// only sees logits + recent tokens.  [`Sampler`](super::Sampler) does not use this
    /// method directly — it uses `SamplerChain::split_from_config` so a
    /// grammar mask can be applied at the correct point between the two
    /// halves.
    ///
    /// Order: logit-bias/bans → repetition penalty (+ frequency/presence) →
    /// DRY → top-K → typical-P → top-P → min-P → top-A → eta-cutoff → XTC →
    /// temperature → (softmax + weighted-random | greedy).
    ///
    /// llama.cpp applies temperature near the END of its chain
    /// (`top_k → typ_p → top_p → min_p → xtc → temp → dist`); the previous
    /// implementation applied temperature first, which shrinks the nucleus
    /// for `top_p` differently than the reference for the same `T < 1`.
    pub fn from_config(config: &super::SamplerConfig) -> Self {
        let mut chain = Self::new();
        if let Some(seed) = config.seed {
            chain = chain.with_seed(seed);
        }
        chain = push_pre_grammar_stages(chain, config);
        chain = push_post_grammar_stages(chain, config);
        chain
    }

    /// Build the pre-grammar and post-grammar halves of the standard
    /// pipeline separately, so a caller (namely [`Sampler`]) can apply a
    /// grammar mask between them.
    ///
    /// Pre-grammar: logit-bias/bans → repetition penalty (+ frequency /
    /// presence) → DRY. These all operate on logit-scale values and must
    /// see the *complete* candidate set before grammar masking removes
    /// tokens, matching the previous behaviour.
    ///
    /// Post-grammar: top-K → typical-P → top-P → min-P → top-A → eta-cutoff
    /// → XTC → temperature (see [`SamplerChain::from_config`] for why
    /// temperature is last). When `config.temperature <= 0.0` the
    /// post-grammar chain is just a [`GreedySelect`] stage (`Sampler` also
    /// short-circuits this case directly for performance — see defect S6 —
    /// but the chain built here remains correct if used standalone).
    pub(crate) fn split_from_config(config: &super::SamplerConfig) -> (Self, Self) {
        let mut pre = Self::new();
        let mut post = Self::new();
        if let Some(seed) = config.seed {
            pre = pre.with_seed(seed);
            post = post.with_seed(seed);
        }
        pre = push_pre_grammar_stages(pre, config);
        post = push_post_grammar_stages(post, config);
        (pre, post)
    }
}

fn push_pre_grammar_stages(mut chain: SamplerChain, config: &super::SamplerConfig) -> SamplerChain {
    // Logit-bias / banned-tokens must come first so bans and boosts are
    // visible to every downstream filtering stage.
    if !config.logit_bias.is_empty() || !config.banned_tokens.is_empty() {
        chain = chain.push(LogitBias::new(
            config.logit_bias.clone(),
            config.banned_tokens.clone(),
        ));
    }

    if config.repetition_penalty != 1.0
        || config.frequency_penalty != 0.0
        || config.presence_penalty != 0.0
    {
        chain = chain.push(RepetitionPenalty::new(
            config.repetition_penalty,
            config.repetition_penalty_window,
            config.frequency_penalty,
            config.presence_penalty,
        ));
    }

    if config.dry_multiplier != 0.0 {
        chain = chain.push(super::advanced::DryStage::new(
            config.dry_multiplier,
            config.dry_base,
            config.dry_allowed_length,
            Vec::new(), // sequence_breakers — not yet exposed on SamplerConfig
        ));
    }

    chain
}

fn push_post_grammar_stages(
    mut chain: SamplerChain,
    config: &super::SamplerConfig,
) -> SamplerChain {
    use super::advanced::{EtaStage, TopAStage, TypicalPStage, XtcStage};

    if config.temperature <= 0.0 {
        // Greedy: just push the greedy selector; nothing else matters.
        return chain.push(GreedySelect);
    }

    if config.top_k > 0 {
        chain = chain.push(TopK::new(config.top_k));
    }
    if config.typical_p < 1.0 {
        chain = chain.push(TypicalPStage::new(config.typical_p));
    }
    if config.top_p < 1.0 {
        chain = chain.push(TopP::new(config.top_p));
    }
    if config.min_p > 0.0 {
        chain = chain.push(MinP::new(config.min_p));
    }
    if config.top_a != 0.0 {
        chain = chain.push(TopAStage::new(config.top_a));
    }
    if config.eta_cutoff != 0.0 || config.epsilon_cutoff != 0.0 {
        chain = chain.push(EtaStage::new(config.eta_cutoff, config.epsilon_cutoff));
    }
    // `xtc_threshold == 0.0` is XTC's documented "disabled" sentinel
    // (`SamplerConfig::default()`'s doc comment: "Advanced stages (disabled
    // by default)"). Without the `> 0.0` guard this stage was pushed for
    // *every* default config (`xtc_threshold: 0.0, xtc_probability: 0.5`)
    // and paid a full softmax + `O(V log V)` sort every stochastic decode
    // step for zero behavioural effect: with `threshold == 0.0`,
    // `XtcStage::apply`'s cumulative-probability loop always satisfies
    // `cumulative >= self.threshold` on its first iteration, so `top_end`
    // is always `1` and the `if top_end < 2 { return; }` early-out fires
    // before anything is excluded — a pure-cost no-op, exactly the kind of
    // wasted full-vocab sort defect S6 exists to eliminate.
    if config.xtc_threshold > 0.0 && config.xtc_threshold < 1.0 && config.xtc_probability > 0.0 {
        chain = chain.push(XtcStage::new(config.xtc_threshold, config.xtc_probability));
    }
    if config.temperature != 1.0 {
        chain = chain.push(TemperatureScale::new(config.temperature));
    }

    chain
}

// ── Built-in stages ──────────────────────────────────────────────────────────

/// Repetition / frequency / presence penalty stage.
///
/// - `penalty` (classic repeat penalty): multiplicative, applied once per
///   *occurrence* of a token within the window (`logit /= penalty` when
///   positive, `logit *= penalty` when non-positive). `1.0` = no effect.
/// - `frequency_penalty` / `presence_penalty`: additive, applied once per
///   *distinct* token within the window, scaled by its occurrence count —
///   matches llama.cpp / OpenAI semantics:
///   `logit -= count * frequency_penalty + (count > 0 ? 1 : 0) * presence_penalty`.
pub struct RepetitionPenalty {
    penalty: f32,
    window: usize,
    frequency_penalty: f32,
    presence_penalty: f32,
}

impl RepetitionPenalty {
    /// Create a new repetition penalty stage.
    ///
    /// `penalty` of 1.0 = no multiplicative effect. `frequency_penalty` and
    /// `presence_penalty` of 0.0 = no additive effect. `window` bounds all
    /// three (matches llama.cpp's single `penalty_last_n`).
    pub fn new(penalty: f32, window: usize, frequency_penalty: f32, presence_penalty: f32) -> Self {
        Self {
            penalty,
            window,
            frequency_penalty,
            presence_penalty,
        }
    }
}

impl SamplerStage for RepetitionPenalty {
    fn apply(&self, logits: &mut Vec<f32>, recent_tokens: &[u32], _rng: &mut Xorshift64) {
        if recent_tokens.is_empty() {
            return;
        }
        if self.penalty == 1.0 && self.frequency_penalty == 0.0 && self.presence_penalty == 0.0 {
            return;
        }

        let start = recent_tokens.len().saturating_sub(self.window);
        let window = &recent_tokens[start..];

        if self.penalty != 1.0 {
            for &token in window {
                let idx = token as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.penalty;
                    } else {
                        logits[idx] *= self.penalty;
                    }
                }
            }
        }

        if self.frequency_penalty != 0.0 || self.presence_penalty != 0.0 {
            let mut counts: HashMap<u32, u32> = HashMap::new();
            for &token in window {
                *counts.entry(token).or_insert(0) += 1;
            }
            for (token, count) in counts {
                let idx = token as usize;
                if idx < logits.len() {
                    logits[idx] -= (count as f32) * self.frequency_penalty + self.presence_penalty;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "repetition_penalty"
    }
}

/// Temperature scaling stage.
pub struct TemperatureScale {
    temperature: f32,
}

impl TemperatureScale {
    /// Create a new temperature scaling stage.
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl SamplerStage for TemperatureScale {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], _rng: &mut Xorshift64) {
        if self.temperature <= 0.0 || self.temperature == 1.0 {
            return;
        }
        let inv = 1.0 / self.temperature;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }

    fn name(&self) -> &'static str {
        "temperature"
    }
}

/// Top-K filtering stage — keeps only the K highest logits (by rank); sets
/// the rest to `-inf`.
///
/// Uses `select_nth_unstable_by` for an O(n) average-case partition instead
/// of a full O(n log n) sort (defect S6), and masks *by rank* rather than by
/// comparing against a threshold value — the previous value-threshold
/// approach silently kept MORE than `k` tokens whenever multiple entries
/// tied the k-th largest value, since ties past the k-th kept slot matched
/// neither the "kept < k" branch nor the "< threshold" branch and were left
/// untouched (defect S3).
pub struct TopK {
    k: usize,
}

impl TopK {
    /// Create a new top-K stage with the given `k`.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl SamplerStage for TopK {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], _rng: &mut Xorshift64) {
        let n = logits.len();
        if self.k == 0 || self.k >= n {
            return;
        }

        let mut idx: Vec<usize> = (0..n).collect();
        idx.select_nth_unstable_by(self.k - 1, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = vec![false; n];
        for &i in &idx[..self.k] {
            keep[i] = true;
        }
        for (i, v) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "top_k"
    }
}

/// Top-P (nucleus) filtering stage — keeps smallest set with cumulative prob >= p.
pub struct TopP {
    p: f32,
}

impl TopP {
    /// Create a new top-P (nucleus) stage with the given probability threshold.
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl SamplerStage for TopP {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], _rng: &mut Xorshift64) {
        if self.p >= 1.0 {
            return;
        }
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let probs: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return;
        }

        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indices.len();
        for (i, &idx) in indices.iter().enumerate() {
            cumulative += probs[idx] / sum;
            if cumulative >= self.p {
                cutoff_idx = i + 1;
                break;
            }
        }

        for &idx in &indices[cutoff_idx..] {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn name(&self) -> &'static str {
        "top_p"
    }
}

/// Min-P filtering stage — removes tokens with prob < min_p * max_prob.
pub struct MinP {
    min_p: f32,
}

impl MinP {
    /// Create a new min-P stage with the given minimum probability ratio.
    pub fn new(min_p: f32) -> Self {
        Self { min_p }
    }
}

impl SamplerStage for MinP {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], _rng: &mut Xorshift64) {
        if self.min_p <= 0.0 {
            return;
        }
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let probs: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return;
        }
        let max_prob = probs.iter().fold(0.0f32, |a, &b| a.max(b)) / sum;
        let threshold = self.min_p * max_prob;
        for (i, v) in logits.iter_mut().enumerate() {
            if probs[i] / sum < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "min_p"
    }
}

/// Logit-bias stage — applies per-token additive biases and hard bans.
///
/// This stage must be positioned **before** temperature scaling, repetition
/// penalty, and any filtering stages so that bans and biases influence all
/// downstream steps uniformly.
///
/// Processing order:
/// 1. Banned tokens → `f32::NEG_INFINITY` (hard ban, cannot be overridden by bias).
/// 2. Logit biases are added to surviving logits.
pub struct LogitBias {
    /// Per-token additive biases.
    biases: HashMap<u32, f32>,
    /// Tokens that must never be sampled.
    banned: Vec<u32>,
}

impl LogitBias {
    /// Create a new logit-bias stage.
    ///
    /// `biases` maps token IDs to additive values (positive = boost,
    /// negative = suppress).  `banned` is the list of tokens to hard-ban.
    pub fn new(biases: HashMap<u32, f32>, banned: Vec<u32>) -> Self {
        Self { biases, banned }
    }

    /// Create a stage with only hard-banned tokens and no biases.
    pub fn banned_only(banned: Vec<u32>) -> Self {
        Self {
            biases: HashMap::new(),
            banned,
        }
    }

    /// Create a stage with only biases and no bans.
    pub fn biases_only(biases: HashMap<u32, f32>) -> Self {
        Self {
            biases,
            banned: Vec::new(),
        }
    }
}

impl SamplerStage for LogitBias {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], _rng: &mut Xorshift64) {
        for &token in &self.banned {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
        for (&token, &bias) in &self.biases {
            let idx = token as usize;
            if idx < logits.len() && logits[idx].is_finite() {
                logits[idx] += bias;
            }
        }
    }

    fn name(&self) -> &'static str {
        "logit_bias"
    }
}

/// Greedy selection stage — sets all logits except the max to -inf.
/// Use this as the final stage for deterministic (argmax) output.
pub struct GreedySelect;

impl SamplerStage for GreedySelect {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], _rng: &mut Xorshift64) {
        if logits.is_empty() {
            return;
        }
        let Some(max_idx) = super::rng::argmax(logits) else {
            // Nothing finite survived upstream stages; leave the (all -inf)
            // vector as-is rather than fabricating an arbitrary winner —
            // the caller (`select_token` / `Sampler::try_sample`) is
            // responsible for surfacing this as a real error.
            return;
        };
        for (i, v) in logits.iter_mut().enumerate() {
            if i != max_idx as usize {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "greedy"
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Final token selection: softmax + weighted random using the supplied RNG.
///
/// Returns `None` when every logit is non-finite (nothing survived
/// filtering) instead of silently defaulting to index 0 — see defect S1's
/// discussion of `argmax`, which had the same failure mode.
pub(crate) fn select_token(logits: &[f32], rng: &mut Xorshift64) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }

    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if !max_val.is_finite() {
        return None;
    }

    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return None;
    }

    // Check if only one token survived (common after greedy/aggressive filtering).
    let mut survivor_count = 0usize;
    let mut survivor_idx = 0u32;
    for (i, &e) in exps.iter().enumerate() {
        if e > 0.0 {
            survivor_count += 1;
            survivor_idx = i as u32;
            if survivor_count > 1 {
                break;
            }
        }
    }
    if survivor_count == 1 {
        return Some(survivor_idx);
    }

    // Use the open-interval draw (0, 1), never exactly 0.0: this walk is in
    // raw index order (not sorted by probability, per S6), so a draw of
    // exactly 0.0 would let the lowest-index token with *any* nonzero mass
    // win regardless of how small its true probability is.
    let r = rng.next_open01_f32();
    let mut cumulative = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        cumulative += e / sum;
        if r < cumulative {
            return Some(i as u32);
        }
    }

    Some((logits.len() - 1) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplerConfig;

    #[test]
    fn test_empty_chain_selects_token() {
        let mut chain = SamplerChain::new().with_seed(42);
        let logits = vec![1.0, 2.0, 3.0];
        let token = chain.sample(&logits, &[]);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_greedy_chain() {
        let mut chain = SamplerChain::new().push(GreedySelect);
        let logits = vec![1.0, 5.0, 3.0, 0.5];
        let token = chain.sample(&logits, &[]);
        assert_eq!(token, 1); // argmax
    }

    #[test]
    fn test_temperature_affects_distribution() {
        // Very cold temperature should always pick top
        let mut chain_cold = SamplerChain::new()
            .with_seed(42)
            .push(TemperatureScale::new(0.01));

        let logits = vec![3.0, 2.0, 1.0, 0.5];
        let token = chain_cold.sample(&logits, &[]);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_top_k_limits_candidates() {
        let mut chain = SamplerChain::new().push(TopK::new(1)).with_seed(42);
        let logits = vec![1.0, 5.0, 3.0];
        let token = chain.sample(&logits, &[]);
        assert_eq!(token, 1); // only top-1 survives
    }

    #[test]
    fn test_top_k_masks_ties_beyond_k() {
        // Four tokens tie for the top value; k=2 must leave EXACTLY 2 finite.
        let mut logits = vec![5.0f32, 5.0, 5.0, 5.0, 1.0];
        TopK::new(2).apply(&mut logits, &[], &mut Xorshift64::new(1));
        let finite = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(
            finite, 2,
            "TopK(2) must keep exactly 2 candidates even when values tie at the cutoff, got {finite}"
        );
    }

    #[test]
    fn test_repetition_penalty_reduces_repeated() {
        let mut chain = SamplerChain::new()
            .push(RepetitionPenalty::new(100.0, 64, 0.0, 0.0))
            .push(GreedySelect);
        let logits = vec![1.0, 5.0, 4.9, 1.0];
        // Without penalty, token 1 wins. With penalty on token 1, token 2 should win.
        let token = chain.sample(&logits, &[1]);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_frequency_penalty_scales_with_count() {
        // Token 1 appears 3 times in the window; frequency_penalty should
        // subtract proportionally more from it than from a token seen once.
        let mut logits = vec![5.0f32, 5.0, 5.0];
        RepetitionPenalty::new(1.0, 64, 1.0, 0.0).apply(
            &mut logits,
            &[1, 1, 1, 2],
            &mut Xorshift64::new(1),
        );
        assert!(
            logits[1] < logits[2],
            "token seen 3x should be penalised more than token seen 1x: {logits:?}"
        );
        assert_eq!(logits[0], 5.0, "token 0 never occurred, must be untouched");
    }

    #[test]
    fn test_presence_penalty_is_flat_regardless_of_count() {
        let mut logits = vec![5.0f32, 5.0, 5.0];
        RepetitionPenalty::new(1.0, 64, 0.0, 2.0).apply(
            &mut logits,
            &[1, 1, 1, 2],
            &mut Xorshift64::new(1),
        );
        // Presence penalty only cares whether the token appeared at all, so
        // tokens 1 (3x) and 2 (1x) should receive the SAME flat penalty.
        assert!(
            (logits[1] - logits[2]).abs() < 1e-6,
            "presence penalty must be flat regardless of occurrence count: {logits:?}"
        );
        assert_eq!(logits[0], 5.0);
    }

    #[test]
    fn test_chain_from_config_greedy() {
        let config = SamplerConfig::greedy();
        let mut chain = SamplerChain::from_config(&config);
        let logits = vec![1.0, 5.0, 3.0];
        assert_eq!(chain.sample(&logits, &[]), 1);
    }

    #[test]
    fn test_chain_from_config_default() {
        let config = SamplerConfig::default();
        let chain = SamplerChain::from_config(&config);
        assert!(!chain.is_empty());
        let names = chain.stage_names();
        assert!(names.contains(&"repetition_penalty"));
        assert!(names.contains(&"temperature"));
    }

    #[test]
    fn test_stage_order_matches_llama_cpp_reference() {
        // Defect S4: llama.cpp applies `top_k -> typ_p -> top_p -> min_p ->
        // xtc -> temp -> dist`, i.e. truncation happens BEFORE temperature.
        // Pin the order so a future change can't silently regress it.
        let config = SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            typical_p: 0.9,
            top_a: 0.1,
            eta_cutoff: 0.1,
            xtc_threshold: 0.5,
            xtc_probability: 0.5,
            ..SamplerConfig::default()
        };
        let chain = SamplerChain::from_config(&config);
        let names = chain.stage_names();
        let post_grammar_expected = [
            "top_k",
            "typical_p",
            "top_p",
            "min_p",
            "top_a",
            "eta",
            "xtc",
            "temperature",
        ];
        let pos = |name: &str| names.iter().position(|&n| n == name);
        let mut last = None;
        for &stage in &post_grammar_expected {
            let p =
                pos(stage).unwrap_or_else(|| panic!("stage {stage} missing from chain: {names:?}"));
            if let Some(prev) = last {
                assert!(
                    prev < p,
                    "stage order violated: expected {stage} after previous stage (names={names:?})"
                );
            }
            last = Some(p);
        }
        // temperature must be the LAST stage.
        assert_eq!(names.last(), Some(&"temperature"));
    }

    /// `xtc_threshold == 0.0` is XTC's documented "disabled" sentinel
    /// (`SamplerConfig::default()`'s "Advanced stages (disabled by
    /// default)" comment). The default config must NOT push an `XtcStage`
    /// into the chain: with `threshold == 0.0`, `XtcStage::apply` always
    /// resolves to a same-step no-op (see the comment in
    /// `push_post_grammar_stages`), so including it anyway pays a full
    /// vocab-sized softmax + `O(V log V)` sort per decode step for no
    /// behavioural effect — exactly the S6 performance defect.
    #[test]
    fn default_config_does_not_push_a_no_op_xtc_stage() {
        let chain = SamplerChain::from_config(&SamplerConfig::default());
        let names = chain.stage_names();
        assert!(
            !names.contains(&"xtc"),
            "default config (xtc_threshold=0.0) must not include XtcStage, got {names:?}"
        );
    }

    #[test]
    fn test_stage_names() {
        let chain = SamplerChain::new()
            .push(RepetitionPenalty::new(1.1, 64, 0.0, 0.0))
            .push(TemperatureScale::new(0.8))
            .push(TopK::new(40))
            .push(TopP::new(0.9))
            .push(MinP::new(0.05));
        let names = chain.stage_names();
        assert_eq!(
            names,
            vec![
                "repetition_penalty",
                "temperature",
                "top_k",
                "top_p",
                "min_p"
            ]
        );
    }

    #[test]
    fn test_empty_logits() {
        let mut chain = SamplerChain::new().push(GreedySelect);
        assert_eq!(chain.sample(&[], &[]), 0);
    }

    #[test]
    fn test_min_p_filters_low_prob() {
        let mut chain = SamplerChain::new().push(MinP::new(0.1)).push(GreedySelect);
        // One dominant token
        let logits = vec![10.0, -10.0, -10.0, -10.0];
        let token = chain.sample(&logits, &[]);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_top_p_nucleus() {
        let mut chain = SamplerChain::new().push(TopP::new(0.5)).with_seed(42);
        // One very dominant token
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let token = chain.sample(&logits, &[]);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_chain_len_and_is_empty() {
        let chain = SamplerChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);

        let chain = chain.push(GreedySelect);
        assert!(!chain.is_empty());
        assert_eq!(chain.len(), 1);
    }

    // ── LogitBias stage tests ─────────────────────────────────────────────────

    #[test]
    fn test_logit_bias_bans_token() {
        let mut chain = SamplerChain::new()
            .push(LogitBias::banned_only(vec![1]))
            .push(GreedySelect);
        // Token 1 would normally win (logit 5.0) but is banned.
        let logits = vec![1.0f32, 5.0, 3.0];
        let tok = chain.sample(&logits, &[]);
        assert_eq!(
            tok, 2,
            "banned token 1 should never win; token 2 (3.0) should"
        );
    }

    #[test]
    fn test_logit_bias_boosts_token() {
        let mut biases = HashMap::new();
        biases.insert(2u32, 100.0f32);
        let mut chain = SamplerChain::new()
            .push(LogitBias::biases_only(biases))
            .push(GreedySelect);
        let logits = vec![10.0f32, 10.0, 0.0]; // token 2 has lowest logit before bias
        let tok = chain.sample(&logits, &[]);
        assert_eq!(tok, 2, "large positive bias should make token 2 win");
    }

    #[test]
    fn test_logit_bias_ban_wins_over_positive_bias() {
        // A banned token should stay at -inf even if it also has a positive bias.
        let mut biases = HashMap::new();
        biases.insert(0u32, 999.0f32); // very large positive bias on token 0
        let mut chain = SamplerChain::new()
            .push(LogitBias::new(biases, vec![0])) // but also banned
            .push(GreedySelect);
        let logits = vec![10.0f32, 1.0, 1.0];
        let tok = chain.sample(&logits, &[]);
        // Token 0 is banned — the positive bias must NOT override the ban.
        assert_ne!(tok, 0, "ban must override positive bias");
    }

    #[test]
    fn test_from_config_includes_logit_bias_stage() {
        let mut biases = HashMap::new();
        biases.insert(0u32, -100.0f32);
        let config = SamplerConfig {
            temperature: 0.0,
            logit_bias: biases,
            ..SamplerConfig::greedy()
        };
        let chain = SamplerChain::from_config(&config);
        let names = chain.stage_names();
        assert!(
            names.contains(&"logit_bias"),
            "from_config should add logit_bias stage when bias map is non-empty"
        );
    }

    #[test]
    fn test_from_config_no_logit_bias_stage_when_empty() {
        let config = SamplerConfig::greedy();
        let chain = SamplerChain::from_config(&config);
        let names = chain.stage_names();
        assert!(
            !names.contains(&"logit_bias"),
            "from_config should NOT add logit_bias stage when both bias map and banned list are empty"
        );
    }

    #[test]
    fn test_chain_rng_advances_across_calls() {
        // Defect S3: `select_token(&processed, self.seed)` used to rebuild a
        // fresh xorshift state from `seed` on every call, so every token
        // drew the SAME uniform sample. A persistent per-chain RNG must
        // advance, producing a varied sequence over many draws.
        let mut chain = SamplerChain::new().with_seed(42);
        let logits = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            seen.insert(chain.sample(&logits, &[]));
        }
        assert!(
            seen.len() > 1,
            "persistent RNG must vary the draw across repeated calls, got {seen:?}"
        );
    }
}

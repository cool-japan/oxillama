//! Sampling strategies for next-token selection.
//!
//! Supports greedy, top-k, top-p (nucleus), min-p, temperature scaling,
//! repetition penalty, Mirostat v2, and GBNF grammar-constrained sampling.

pub mod advanced;
pub mod chain;
pub mod grammar;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use grammar::{apply_grammar_mask, Grammar, GrammarState};

/// Configuration for the sampling strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// Temperature for logit scaling (1.0 = no scaling, 0.0 = greedy).
    pub temperature: f32,
    /// Top-K: only consider the K most likely tokens (0 = disabled).
    pub top_k: usize,
    /// Top-P (nucleus): only consider tokens with cumulative probability <= p.
    pub top_p: f32,
    /// Min-P: minimum probability threshold relative to the top token.
    pub min_p: f32,
    /// Repetition penalty factor (1.0 = no penalty).
    pub repetition_penalty: f32,
    /// Number of recent tokens to consider for repetition penalty.
    pub repetition_penalty_window: usize,
    /// Random seed for reproducible sampling (None = random).
    pub seed: Option<u64>,
    /// Mirostat mode: 0 = disabled, 2 = Mirostat v2.
    pub mirostat: u8,
    /// Mirostat target surprise (tau). Controls coherence vs diversity.
    /// Lower = more coherent, higher = more diverse. Default: 5.0.
    pub mirostat_tau: f32,
    /// Mirostat learning rate (eta). How fast the algorithm adapts. Default: 0.1.
    pub mirostat_eta: f32,

    /// Optional GBNF grammar for constrained sampling.
    /// Logits for tokens that cannot advance the grammar are set to -∞.
    /// Skipped during serialization (not representable as JSON directly).
    #[serde(skip)]
    pub grammar: Option<Arc<Grammar>>,

    /// Pre-computed vocabulary `(token_id, byte_repr)` table used for grammar masking.
    /// Must be set when `grammar` is `Some`. Build via `TokenizerBridge::vocab_bytes()`.
    #[serde(skip)]
    #[allow(clippy::type_complexity)]
    pub token_vocab: Option<Arc<Vec<(u32, Vec<u8>)>>>,

    /// Per-token logit biases applied before top-k/top-p.
    ///
    /// Positive values increase a token's probability; negative values decrease it.
    /// For example, `logit_bias[token_id] = 5.0` strongly encourages that token,
    /// while `-100.0` effectively bans it (use `banned_tokens` for strict banning).
    ///
    /// Applied as: `logits[token_id] += bias` before the greedy / sampling steps.
    #[serde(default)]
    pub logit_bias: std::collections::HashMap<u32, f32>,

    /// Tokens that must never be generated.
    ///
    /// Their logits are set to `f32::NEG_INFINITY` before any other sampling
    /// step, including top-k/p filtering. This is a hard constraint — unlike
    /// a large negative `logit_bias`, a banned token will never be selected
    /// even if it is the only remaining candidate.
    #[serde(default)]
    pub banned_tokens: Vec<u32>,

    // ── Advanced sampler stages (v0.1.7 Track B) ─────────────────────────────
    /// DRY penalty multiplier (0.0 = disabled).
    ///
    /// Penalises tokens that would continue an n-gram already present in the
    /// recent context. Higher values apply stronger penalties.
    #[serde(default)]
    pub dry_multiplier: f32,

    /// DRY exponential base for match-length amplification (default = 1.75).
    ///
    /// Longer n-gram matches receive penalty `dry_multiplier * dry_base^(match_len - dry_allowed_length)`.
    #[serde(default = "dry_base_default")]
    pub dry_base: f32,

    /// Minimum match length (in tokens) before DRY applies any penalty (default = 2).
    #[serde(default = "dry_allowed_length_default")]
    pub dry_allowed_length: usize,

    /// XTC cumulative-probability threshold (0.0 = disabled; use ≥ 1.0 to disable).
    ///
    /// The "top set" is defined as the smallest set of tokens whose cumulative
    /// probability exceeds this threshold.
    #[serde(default)]
    pub xtc_threshold: f32,

    /// XTC exclusion probability — how often the top-set exclusion fires (default = 0.5).
    #[serde(default = "xtc_probability_default")]
    pub xtc_probability: f32,

    /// Locally-typical sampling budget (1.0 = disabled / passthrough).
    ///
    /// Keeps only tokens whose information content is closest to the distribution
    /// entropy until cumulative probability ≥ p.
    #[serde(default = "typical_p_default")]
    pub typical_p: f32,

    /// Top-A adaptive threshold multiplier (0.0 = disabled).
    ///
    /// Keeps tokens with `prob >= top_a * max_prob²`.
    #[serde(default)]
    pub top_a: f32,

    /// Eta-cutoff entropy-adaptive threshold (0.0 = disabled).
    ///
    /// Dynamic floor = `max(epsilon_cutoff, eta_cutoff / perplexity)`.
    #[serde(default)]
    pub eta_cutoff: f32,

    /// Epsilon hard-floor probability used together with `eta_cutoff` (0.0 = no floor).
    #[serde(default)]
    pub epsilon_cutoff: f32,
}

// Default-value helpers for serde.
fn dry_base_default() -> f32 {
    1.75
}
fn dry_allowed_length_default() -> usize {
    2
}
fn xtc_probability_default() -> f32 {
    0.5
}
fn typical_p_default() -> f32 {
    1.0
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.1,
            repetition_penalty_window: 64,
            seed: None,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            grammar: None,
            token_vocab: None,
            logit_bias: std::collections::HashMap::new(),
            banned_tokens: Vec::new(),
            // Advanced stages (disabled by default)
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            xtc_threshold: 0.0,
            xtc_probability: 0.5,
            typical_p: 1.0,
            top_a: 0.0,
            eta_cutoff: 0.0,
            epsilon_cutoff: 0.0,
        }
    }
}

impl SamplerConfig {
    /// Create a greedy sampling config (always pick the most likely token).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repetition_penalty_window: 0,
            seed: None,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            grammar: None,
            token_vocab: None,
            logit_bias: std::collections::HashMap::new(),
            banned_tokens: Vec::new(),
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            xtc_threshold: 0.0,
            xtc_probability: 0.5,
            typical_p: 1.0,
            top_a: 0.0,
            eta_cutoff: 0.0,
            epsilon_cutoff: 0.0,
        }
    }

    /// Create a Mirostat v2 config with the given target surprise.
    pub fn mirostat_v2(tau: f32, eta: f32) -> Self {
        Self {
            temperature: 1.0,
            mirostat: 2,
            mirostat_tau: tau,
            mirostat_eta: eta,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repetition_penalty_window: 0,
            seed: None,
            grammar: None,
            token_vocab: None,
            logit_bias: std::collections::HashMap::new(),
            banned_tokens: Vec::new(),
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            xtc_threshold: 0.0,
            xtc_probability: 0.5,
            typical_p: 1.0,
            top_a: 0.0,
            eta_cutoff: 0.0,
            epsilon_cutoff: 0.0,
        }
    }
}

/// Stateful sampler that maintains PRNG state across calls.
pub struct Sampler {
    config: SamplerConfig,
    rng: Xorshift64,
    /// Mirostat v2 running estimate of surprise (mu).
    /// Initialized to 2 * tau, updated after each sample.
    mirostat_mu: f32,
    /// Current grammar parse state (None when no grammar is configured).
    grammar_state: Option<GrammarState>,
}

impl Sampler {
    /// Create a new sampler with the given config.
    pub fn new(config: SamplerConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| {
            // Use a time-based seed when no explicit seed is provided.
            // This is deterministic enough for inference; not for crypto.
            let mut s = 0x517cc1b727220a95u64;
            // Mix in some bits from the stack address for entropy
            s ^= (&s as *const u64 as u64).wrapping_mul(0x9e3779b97f4a7c15);
            s ^ s.wrapping_shr(33)
        });
        let mirostat_mu = 2.0 * config.mirostat_tau;
        let grammar_state = config.grammar.as_ref().map(|g| g.initial_state());
        Self {
            config,
            rng: Xorshift64::new(seed),
            mirostat_mu,
            grammar_state,
        }
    }

    /// Sample a token ID from logits.
    pub fn sample(&mut self, logits: &[f32], recent_tokens: &[u32]) -> u32 {
        let token = if self.config.mirostat == 2 {
            self.sample_mirostat_v2(logits, recent_tokens)
        } else {
            sample_with_rng(
                logits,
                &self.config,
                recent_tokens,
                &mut self.rng,
                self.grammar_state.as_ref(),
            )
        };

        // Advance grammar state after token selection.
        // We look up the token bytes in the pre-built vocab table (binary search by id).
        if let Some(state) = &mut self.grammar_state {
            if let Some(vocab) = &self.config.token_vocab {
                if let Ok(idx) = vocab.binary_search_by_key(&token, |&(id, _)| id) {
                    let bytes = vocab[idx].1.clone();
                    // Silently ignore advance errors — the mask will catch a stuck state
                    // on the next step and -inf all invalid tokens.
                    let _ = state.advance(&bytes);
                }
            }
        }

        token
    }

    /// Reset the grammar state to the beginning (use for a new generation).
    pub fn reset_grammar(&mut self) {
        self.grammar_state = self.config.grammar.as_ref().map(|g| g.initial_state());
    }

    /// Returns true when the grammar (if any) is in a valid accepting state.
    pub fn grammar_complete(&self) -> bool {
        self.grammar_state
            .as_ref()
            .is_none_or(GrammarState::is_complete)
    }

    /// Mirostat v2 sampling.
    ///
    /// Adaptively controls the "surprise" of generated tokens to maintain
    /// a target perplexity level (tau). This produces more coherent text
    /// than fixed top-k/top-p by dynamically adjusting the token pool.
    fn sample_mirostat_v2(&mut self, logits: &[f32], recent_tokens: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        let mut processed = logits.to_vec();

        // Step 0: Apply logit bias and banned tokens — same order as
        // sample_with_rng so both code paths behave identically.
        apply_logit_bias_and_banned_tokens(&mut processed, &self.config);

        // Step 1: Apply repetition penalty
        apply_repetition_penalty(&mut processed, &self.config, recent_tokens);

        // Step 2: Apply grammar mask — BEFORE temperature and sorting.
        // Grammar masking must happen before any filtering so the constraint
        // is respected even in the greedy case.
        if let (Some(state), Some(vocab)) = (&self.grammar_state, &self.config.token_vocab) {
            apply_grammar_mask(&mut processed, state, vocab.as_ref());
        }

        // Step 3: Apply temperature
        if self.config.temperature > 0.0 && self.config.temperature != 1.0 {
            let inv_temp = 1.0 / self.config.temperature;
            for val in &mut processed {
                *val *= inv_temp;
            }
        }

        // Build sorted candidates with probabilities
        let mut candidates: Vec<(u32, f32)> = processed
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax to get probabilities
        softmax_candidates(&mut candidates);

        // Mirostat v2: filter tokens by surprise threshold
        // surprise(token) = -log2(prob)
        // Keep tokens where surprise <= mu
        let mu = self.mirostat_mu;
        candidates.retain(|&(_, p)| {
            if p <= 0.0 {
                return false;
            }
            let surprise = -p.log2();
            surprise <= mu
        });

        // Fallback: if all tokens filtered, keep the top one
        if candidates.is_empty() {
            let token = argmax(&processed);
            // Still update mu
            let top_prob = softmax_single_max(&processed);
            let surprise = if top_prob > 0.0 {
                -top_prob.log2()
            } else {
                self.config.mirostat_tau
            };
            self.mirostat_mu =
                mu - self.config.mirostat_eta * (surprise - self.config.mirostat_tau);
            return token;
        }

        // Re-normalize
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        if total > 0.0 && total != 1.0 {
            for (_, p) in &mut candidates {
                *p /= total;
            }
        }

        // Sample from filtered candidates
        let r = self.rng.next_f32();
        let mut cumulative = 0.0f32;
        let mut selected_idx = candidates[0].0;
        let mut selected_prob = candidates[0].1 * total; // original probability
        for &(idx, prob) in &candidates {
            cumulative += prob;
            if r < cumulative {
                selected_idx = idx;
                selected_prob = prob * total;
                break;
            }
        }

        // Update mu: mu' = mu - eta * (surprise - tau)
        let surprise = if selected_prob > 0.0 {
            -selected_prob.log2()
        } else {
            self.config.mirostat_tau
        };
        self.mirostat_mu = mu - self.config.mirostat_eta * (surprise - self.config.mirostat_tau);

        selected_idx
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Return the raw RNG state for snapshot/resume.
    pub fn rng_state(&self) -> u64 {
        self.rng.state_value()
    }

    /// Return the current mirostat mu value for snapshot/resume.
    pub fn mirostat_mu_value(&self) -> f32 {
        self.mirostat_mu
    }

    /// Restore the RNG state and mirostat mu (for resume).
    pub fn restore_rng_state(&mut self, state: u64, mu: f32) {
        self.rng = Xorshift64::from_state_value(state);
        self.mirostat_mu = mu;
    }
}

/// Sample a token ID from logits using the given configuration.
///
/// This is the stateless variant. Grammar state (if any in config) is ignored
/// because there is no place to persist it between calls. Use [`Sampler`] for
/// grammar-constrained generation.
///
/// # Arguments
/// * `logits` - Raw logits from the model (length = vocab_size).
/// * `config` - Sampling configuration.
/// * `recent_tokens` - Recent token history for repetition penalty.
///
/// # Returns
/// The selected token ID.
pub fn sample(logits: &[f32], config: &SamplerConfig, recent_tokens: &[u32]) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // For stateless API, create a one-shot RNG. Grammar state is not threaded
    // here — callers needing grammar must use `Sampler`.
    let seed = config.seed.unwrap_or(0xDEADBEEF_CAFEBABE);
    let mut rng = Xorshift64::new(seed);
    sample_with_rng(logits, config, recent_tokens, &mut rng, None)
}

/// Core sampling implementation with explicit RNG and optional grammar state.
fn sample_with_rng(
    logits: &[f32],
    config: &SamplerConfig,
    recent_tokens: &[u32],
    rng: &mut Xorshift64,
    grammar_state: Option<&GrammarState>,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let mut processed = logits.to_vec();

    // Step 0: Apply logit bias and banned tokens FIRST — before any other
    // transformation so that bans are absolute and biases influence all
    // downstream filtering steps (top-k, top-p, grammar masking, etc.).
    apply_logit_bias_and_banned_tokens(&mut processed, config);

    // Step 1: Apply repetition penalty
    apply_repetition_penalty(&mut processed, config, recent_tokens);

    // Step 2: Apply grammar mask — BEFORE the greedy shortcut.
    // This ensures grammar constraints are enforced even at temperature=0.
    if let (Some(state), Some(vocab)) = (grammar_state, &config.token_vocab) {
        apply_grammar_mask(&mut processed, state, vocab.as_ref());
    }

    // Step 3: Greedy shortcut (after grammar mask)
    if config.temperature <= 0.0 || config.top_k == 1 {
        return argmax(&processed);
    }

    // Step 4: Temperature scaling
    if config.temperature != 1.0 {
        let inv_temp = 1.0 / config.temperature;
        for val in &mut processed {
            *val *= inv_temp;
        }
    }

    // Step 5: Build sorted (index, logit) candidates
    let mut candidates: Vec<(u32, f32)> = processed
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Step 6: Top-K filtering
    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.truncate(config.top_k);
    }

    // Step 7: Softmax over remaining candidates
    softmax_candidates(&mut candidates);

    // Step 8: Min-P filtering (remove tokens with prob < min_p * max_prob)
    if config.min_p > 0.0 && !candidates.is_empty() {
        let max_prob = candidates[0].1; // already sorted descending by probability
        let threshold = config.min_p * max_prob;
        candidates.retain(|&(_, p)| p >= threshold);
    }

    // Step 9: Top-P (nucleus) filtering
    if config.top_p < 1.0 && !candidates.is_empty() {
        let mut cumulative = 0.0f32;
        let mut cutoff = candidates.len();
        for (i, &(_, prob)) in candidates.iter().enumerate() {
            cumulative += prob;
            if cumulative >= config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff);
    }

    // Step 10: Re-normalize after filtering
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    if total > 0.0 && total != 1.0 {
        for (_, p) in &mut candidates {
            *p /= total;
        }
    }

    // Step 11: Weighted random selection
    if candidates.is_empty() {
        return argmax(&processed);
    }
    if candidates.len() == 1 {
        return candidates[0].0;
    }

    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for &(idx, prob) in &candidates {
        cumulative += prob;
        if r < cumulative {
            return idx;
        }
    }

    // Fallback: return last candidate (rounding issues)
    candidates.last().map(|&(idx, _)| idx).unwrap_or(0)
}

/// Apply logit bias and banned-token masking to logits in-place.
///
/// Processing order:
/// 1. Banned tokens are set to `f32::NEG_INFINITY` unconditionally.
/// 2. Logit biases are added to the surviving logits.
///
/// Both operations are applied before repetition penalty, grammar masking,
/// and temperature / top-k / top-p filtering, so they influence all
/// downstream steps.
fn apply_logit_bias_and_banned_tokens(processed: &mut [f32], config: &SamplerConfig) {
    // Step A: hard-ban tokens.
    for &token in &config.banned_tokens {
        let idx = token as usize;
        if idx < processed.len() {
            processed[idx] = f32::NEG_INFINITY;
        }
    }

    // Step B: additive bias.
    for (&token, &bias) in &config.logit_bias {
        let idx = token as usize;
        if idx < processed.len() {
            // Do not modify already-banned tokens — a banned token must
            // remain at -inf even if a positive bias is also specified.
            if processed[idx].is_finite() {
                processed[idx] += bias;
            }
        }
    }
}

/// Apply repetition penalty to logits in-place.
fn apply_repetition_penalty(processed: &mut [f32], config: &SamplerConfig, recent_tokens: &[u32]) {
    if config.repetition_penalty == 1.0 || recent_tokens.is_empty() {
        return;
    }

    let window_start = recent_tokens
        .len()
        .saturating_sub(config.repetition_penalty_window);
    for &token in &recent_tokens[window_start..] {
        let idx = token as usize;
        if idx < processed.len() {
            if processed[idx] > 0.0 {
                processed[idx] /= config.repetition_penalty;
            } else {
                processed[idx] *= config.repetition_penalty;
            }
        }
    }
}

/// Compute softmax over candidates in-place (replaces logits with probabilities).
fn softmax_candidates(candidates: &mut [(u32, f32)]) {
    if candidates.is_empty() {
        return;
    }

    let max_logit = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for (_, logit) in candidates.iter_mut() {
        *logit = (*logit - max_logit).exp();
        sum += *logit;
    }

    if sum > 0.0 {
        for (_, prob) in candidates.iter_mut() {
            *prob /= sum;
        }
    }
}

/// Compute the softmax probability of the maximum logit (for fallback).
fn softmax_single_max(logits: &[f32]) -> f32 {
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
    if sum > 0.0 {
        1.0 / sum
    } else {
        0.0
    }
}

/// Return the index of the maximum value.
fn argmax(values: &[f32]) -> u32 {
    let mut max_idx = 0u32;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i as u32;
        }
    }
    max_idx
}

/// Simple xorshift64 PRNG — fast, small, seedable, no dependencies.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self {
            state: if seed == 0 { 0x517cc1b727220a95 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Return the raw internal state for snapshot/resume.
    pub(crate) fn state_value(&self) -> u64 {
        self.state
    }

    /// Reconstruct from a raw state value (for resume).
    pub(crate) fn from_state_value(state: u64) -> Self {
        Self {
            state: if state == 0 { 1 } else { state },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let config = SamplerConfig::greedy();
        let token = sample(&logits, &config, &[]);
        assert_eq!(token, 3); // index of 0.8
    }

    #[test]
    fn test_empty_logits() {
        let logits: Vec<f32> = vec![];
        let config = SamplerConfig::greedy();
        let token = sample(&logits, &config, &[]);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let config = SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        };
        let token = sample(&logits, &config, &[]);
        assert_eq!(token, 1); // argmax
    }

    #[test]
    fn test_top_k_1_is_greedy() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 1,
            ..SamplerConfig::default()
        };
        let token = sample(&logits, &config, &[]);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_seeded_determinism() {
        let logits = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: Some(42),
            ..SamplerConfig::default()
        };

        let mut sampler1 = Sampler::new(config.clone());
        let mut sampler2 = Sampler::new(config);

        // Same seed should produce same sequence
        for _ in 0..10 {
            let t1 = sampler1.sample(&logits, &[]);
            let t2 = sampler2.sample(&logits, &[]);
            assert_eq!(t1, t2, "seeded samplers should produce identical results");
        }
    }

    #[test]
    fn test_top_p_filters_low_prob() {
        // One token has overwhelming probability
        let logits = vec![100.0, 0.0, 0.0, 0.0, 0.0];
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            min_p: 0.0,
            seed: Some(123),
            ..SamplerConfig::default()
        };

        // With top_p=0.5, only the dominant token should remain
        let token = sample(&logits, &config, &[]);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_repetition_penalty() {
        // Token 1 has highest logit but is in recent history
        let logits = vec![1.0, 5.0, 4.9, 1.0];
        let config = SamplerConfig {
            temperature: 0.0,          // greedy after penalty
            repetition_penalty: 100.0, // severe penalty
            repetition_penalty_window: 64,
            ..SamplerConfig::greedy()
        };

        // Without penalty, token 1 wins
        let token_no_penalty = sample(&logits, &SamplerConfig::greedy(), &[]);
        assert_eq!(token_no_penalty, 1);

        // With penalty on token 1, token 2 (4.9) should win
        let token_with_penalty = sample(&logits, &config, &[1]);
        assert_eq!(token_with_penalty, 2);
    }

    #[test]
    fn test_sampling_distribution() {
        // Verify that with temperature sampling, we don't always pick argmax
        let logits = vec![2.0, 2.0, 2.0, 2.0]; // equal logits
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: Some(999),
            ..SamplerConfig::default()
        };

        let mut sampler = Sampler::new(config);
        let mut counts = [0u32; 4];
        for _ in 0..1000 {
            let t = sampler.sample(&logits, &[]);
            counts[t as usize] += 1;
        }

        // With equal logits, each token should get ~250 hits.
        // Allow generous margin (100-400).
        for (i, &count) in counts.iter().enumerate() {
            assert!(
                count > 100 && count < 400,
                "token {i} got {count} hits (expected ~250 for uniform distribution)"
            );
        }
    }

    #[test]
    fn test_min_p_filtering() {
        // One very likely token and several very unlikely ones
        let logits = vec![10.0, -10.0, -10.0, -10.0];
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.1, // require at least 10% of max prob
            seed: Some(42),
            ..SamplerConfig::default()
        };

        // The dominant token should always win after min_p filtering
        let mut sampler = Sampler::new(config);
        for _ in 0..100 {
            assert_eq!(sampler.sample(&logits, &[]), 0);
        }
    }

    #[test]
    fn test_xorshift_range() {
        let mut rng = Xorshift64::new(12345);
        for _ in 0..10000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "RNG produced {v} outside [0, 1)");
        }
    }

    #[test]
    fn test_mirostat_v2_basic() {
        // Mirostat v2 should produce valid tokens
        let logits = vec![3.0, 2.0, 1.0, 0.5, 0.1, -1.0, -2.0, -5.0];
        let config = SamplerConfig {
            seed: Some(42),
            ..SamplerConfig::mirostat_v2(5.0, 0.1)
        };
        let mut sampler = Sampler::new(config);

        for _ in 0..50 {
            let token = sampler.sample(&logits, &[]);
            assert!((token as usize) < logits.len());
        }
    }

    #[test]
    fn test_mirostat_v2_adapts_mu() {
        let logits = vec![5.0, 0.0, 0.0, 0.0];
        let config = SamplerConfig {
            seed: Some(123),
            ..SamplerConfig::mirostat_v2(3.0, 0.1)
        };
        let mut sampler = Sampler::new(config);
        let initial_mu = sampler.mirostat_mu;

        // After sampling, mu should change
        sampler.sample(&logits, &[]);
        assert!(
            (sampler.mirostat_mu - initial_mu).abs() > 1e-6,
            "mu should adapt after sampling"
        );
    }

    #[test]
    fn test_mirostat_v2_low_tau_prefers_top() {
        // Very low tau = very low target surprise = prefer high-probability tokens
        let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        let config = SamplerConfig {
            seed: Some(42),
            ..SamplerConfig::mirostat_v2(0.5, 0.1) // very low tau
        };
        let mut sampler = Sampler::new(config);

        let mut top_count = 0;
        for _ in 0..100 {
            if sampler.sample(&logits, &[]) == 0 {
                top_count += 1;
            }
        }
        // With tau=0.5, should almost always pick the top token
        assert!(
            top_count > 90,
            "low tau should strongly prefer top token, got {top_count}/100"
        );
    }

    #[test]
    fn test_mirostat_v2_deterministic_with_seed() {
        let logits = vec![2.0, 1.5, 1.0, 0.5];
        let config = SamplerConfig {
            seed: Some(777),
            ..SamplerConfig::mirostat_v2(5.0, 0.1)
        };

        let mut sampler1 = Sampler::new(config.clone());
        let mut sampler2 = Sampler::new(config);

        for _ in 0..20 {
            assert_eq!(
                sampler1.sample(&logits, &[]),
                sampler2.sample(&logits, &[]),
                "same seed should produce same sequence"
            );
        }
    }

    #[test]
    fn test_softmax_candidates_basic() {
        let mut candidates = vec![(0, 0.0f32), (1, 0.0), (2, 0.0)];
        softmax_candidates(&mut candidates);
        // Equal logits → equal probabilities
        for &(_, p) in &candidates {
            assert!((p - 1.0 / 3.0).abs() < 0.01, "expected ~0.333, got {p}");
        }
    }

    // ── Logit-bias / banned-tokens tests ──────────────────────────────────────

    #[test]
    fn banned_tokens_never_sampled() {
        // Only token 3 is allowed; all others are banned.
        let vocab_size = 5usize;
        let logits: Vec<f32> = (0..vocab_size).map(|i| i as f32).collect();

        let mut banned = Vec::new();
        for i in 0u32..vocab_size as u32 {
            if i != 3 {
                banned.push(i);
            }
        }
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: Some(42),
            banned_tokens: banned,
            ..SamplerConfig::default()
        };
        let mut sampler = Sampler::new(config);
        for _ in 0..50 {
            let tok = sampler.sample(&logits, &[]);
            assert_eq!(
                tok, 3,
                "only token 3 should ever be sampled when all others are banned"
            );
        }
    }

    #[test]
    fn positive_bias_increases_token_probability() {
        // Token 1 starts with a very low logit; add a large positive bias.
        // After bias, token 1 should dominate and be selected nearly always.
        let logits = vec![10.0f32, -20.0, -20.0, -20.0];
        let mut bias = std::collections::HashMap::new();
        bias.insert(1u32, 100.0f32); // huge positive bias on token 1

        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: Some(7),
            logit_bias: bias,
            ..SamplerConfig::default()
        };
        let mut sampler = Sampler::new(config);
        // With a +100 bias, token 1's effective logit = 80, far above token 0's 10.
        let tok = sampler.sample(&logits, &[]);
        assert_eq!(tok, 1, "large positive bias should make token 1 dominate");
    }

    #[test]
    fn negative_bias_decreases() {
        // Token 0 has the highest raw logit; apply a strongly negative bias.
        // Token 1 should win after bias.
        let logits = vec![100.0f32, 1.0, 0.5, 0.1];
        let mut bias = std::collections::HashMap::new();
        bias.insert(0u32, -200.0f32); // strong negative on the top token

        let config = SamplerConfig {
            temperature: 0.0, // greedy — picks strictly by highest logit after bias
            logit_bias: bias,
            ..SamplerConfig::greedy()
        };
        let tok = sample(&logits, &config, &[]);
        assert_eq!(
            tok, 1,
            "after large negative bias on token 0, token 1 should win"
        );
    }

    #[test]
    fn logit_bias_empty_config_no_op() {
        // Empty logit_bias and empty banned_tokens must not change sampling behaviour.
        let logits = vec![1.0f32, 2.0, 3.0, 0.5];
        let config_empty = SamplerConfig {
            temperature: 0.0,
            logit_bias: std::collections::HashMap::new(),
            banned_tokens: Vec::new(),
            ..SamplerConfig::greedy()
        };
        let tok = sample(&logits, &config_empty, &[]);
        // Greedy with no bias should still pick index 2 (value 3.0).
        assert_eq!(tok, 2, "empty logit_bias / banned_tokens should be a no-op");
    }

    // ── Grammar-constrained sampling tests ────────────────────────────────────

    #[test]
    fn test_grammar_constrained_yes_no() {
        let g = Grammar::parse(r#"root ::= "yes" | "no""#).unwrap();
        let state = g.initial_state();
        assert!(state.allows_token(b"yes"));
        assert!(state.allows_token(b"no"));
        assert!(!state.allows_token(b"maybe"));
    }

    #[test]
    fn test_grammar_sampler_masks_logits() {
        // Vocab: 0="maybe", 1="yes", 2="no"
        let vocab: Vec<(u32, Vec<u8>)> = vec![
            (0, b"maybe".to_vec()),
            (1, b"yes".to_vec()),
            (2, b"no".to_vec()),
        ];
        let g = Arc::new(Grammar::parse(r#"root ::= "yes" | "no""#).unwrap());
        let config = SamplerConfig {
            temperature: 0.0, // greedy — must pick grammar-compliant token
            grammar: Some(g),
            token_vocab: Some(Arc::new(vocab)),
            ..SamplerConfig::default()
        };

        // Give "maybe" the highest logit — grammar must mask it away
        let logits = vec![100.0f32, 1.0, 1.0];
        let mut sampler = Sampler::new(config);
        let tok = sampler.sample(&logits, &[]);
        // After masking, only "yes"(1) or "no"(2) remain
        assert!(tok == 1 || tok == 2, "expected yes(1) or no(2), got {tok}");
    }

    #[test]
    fn test_grammar_state_advances_through_sequence() {
        let vocab: Vec<(u32, Vec<u8>)> =
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())];
        let g = Arc::new(Grammar::parse(r#"root ::= "a" "b""#).unwrap());
        let config = SamplerConfig {
            temperature: 0.0,
            grammar: Some(g),
            token_vocab: Some(Arc::new(vocab)),
            ..SamplerConfig::default()
        };

        // Equal logits — grammar drives selection
        let logits = vec![1.0f32, 0.5, 0.5];
        let mut sampler = Sampler::new(config);

        // First step: only "a" is valid
        let tok1 = sampler.sample(&logits, &[]);
        assert_eq!(tok1, 0, "first token must be 'a' (id=0)");

        // Second step: only "b" is valid
        let tok2 = sampler.sample(&logits, &[0]);
        assert_eq!(tok2, 1, "second token must be 'b' (id=1)");

        assert!(
            sampler.grammar_complete(),
            "grammar should be complete after 'a' + 'b'"
        );
    }

    #[test]
    fn test_grammar_parse_roundtrip() {
        let g = Grammar::parse("root ::= [a-z]+ \":\" [0-9]+").unwrap();
        assert!(!g.rules.is_empty());
        assert_eq!(g.root, "root");
    }

    #[test]
    fn test_grammar_stuck_state_masks_all() {
        // A grammar that requires "x" — advancing with "y" must produce an error
        let g = Arc::new(Grammar::parse(r#"root ::= "x""#).unwrap());
        let mut state = g.initial_state();
        let result = state.advance(b"y");
        assert!(result.is_err(), "advancing with wrong bytes should error");
    }
}

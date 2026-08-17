//! Sampling strategies for next-token selection.
//!
//! Supports greedy, top-k, top-p (nucleus), min-p, temperature scaling,
//! repetition/frequency/presence penalties, DRY, XTC, typical-P, top-A,
//! eta-cutoff, Mirostat v1/v2, and GBNF grammar-constrained sampling.
//!
//! # Pipeline (defect S3 / S4)
//!
//! [`Sampler::sample`] routes through two [`chain::SamplerChain`]s built
//! once at construction (not rebuilt per token — see defect S6):
//!
//! 1. **Pre-grammar**: logit-bias/bans → repetition penalty (+ frequency /
//!    presence) → DRY.
//! 2. **Grammar mask** (if a grammar is configured) — applied directly by
//!    `Sampler`, not as a stage, since it needs the live, per-call
//!    [`grammar::GrammarState`] rather than static config.
//! 3. **Greedy shortcut**: if `temperature <= 0.0 || top_k == 1`, return the
//!    argmax immediately (matches llama.cpp and avoids the cost of the
//!    remaining stages entirely).
//! 4. **Post-grammar**: top-K → typical-P → top-P → min-P → top-A →
//!    eta-cutoff → XTC → temperature → final selection. This order matches
//!    llama.cpp's reference chain (`top_k → typ_p → top_p → min_p → xtc →
//!    temp → dist`), where truncation happens on the *untempered*
//!    distribution — see defect S4.
//!
//! Mirostat v1/v2 (`mirostat: 1 | 2`) bypass step 4 entirely and instead run
//! their own adaptive-perplexity selection (see `mirostat`).

pub mod advanced;
pub mod chain;
pub mod grammar;
mod mirostat;
pub mod rng;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{RuntimeError, RuntimeResult};
use chain::SamplerChain;
use grammar::{apply_grammar_mask, Grammar, GrammarState};
use rng::Xorshift64;

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
    /// Mirostat mode: 0 = disabled, 1 = Mirostat v1, 2 = Mirostat v2.
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

    /// End-of-generation token IDs (e.g. `</s>`, `<|im_end|>`, `<|endoftext|>`)
    /// consulted by grammar-constrained sampling.
    ///
    /// When a grammar is configured and its parse state becomes complete
    /// (`GrammarState::is_complete()`), these tokens are exempted from
    /// grammar masking so the model can actually terminate generation.
    /// Without this, every logit — including the caller's EOS token —
    /// would be masked once the grammar is satisfied (defect S1). Populate
    /// from the tokenizer's EOS/EOG token(s). Ignored when `grammar` is
    /// `None`.
    #[serde(default)]
    pub eog_token_ids: Vec<u32>,

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

    /// OpenAI-style frequency penalty: `logit[t] -= count(t) * frequency_penalty`.
    ///
    /// Applied once per *distinct* token in the repetition-penalty window,
    /// scaled by how many times it occurred (0.0 = disabled). Unlike
    /// `repetition_penalty` (multiplicative), this is additive and matches
    /// the OpenAI / llama.cpp `frequency_penalty` semantics.
    #[serde(default)]
    pub frequency_penalty: f32,

    /// OpenAI-style presence penalty: a flat `logit[t] -= presence_penalty`
    /// for every distinct token that appeared at least once in the window,
    /// regardless of how many times (0.0 = disabled).
    #[serde(default)]
    pub presence_penalty: f32,
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
            eog_token_ids: Vec::new(),
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
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
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
            eog_token_ids: Vec::new(),
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
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
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
            eog_token_ids: Vec::new(),
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
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// Create a Mirostat v1 config with the given target surprise.
    pub fn mirostat_v1(tau: f32, eta: f32) -> Self {
        Self {
            mirostat: 1,
            ..Self::mirostat_v2(tau, eta)
        }
    }
}

/// Stateful sampler that maintains PRNG state across calls.
pub struct Sampler {
    config: SamplerConfig,
    rng: Xorshift64,
    /// Mirostat running estimate of surprise (mu), shared by v1 and v2.
    /// Initialized to 2 * tau, updated after each sample.
    mirostat_mu: f32,
    /// Current grammar parse state (None when no grammar is configured).
    grammar_state: Option<GrammarState>,
    /// Stages that run before grammar masking (bias/bans, repetition +
    /// frequency/presence penalty, DRY). Built once at construction —
    /// see defect S6.
    pre_grammar_chain: SamplerChain,
    /// Stages that run after grammar masking (top-k, typical-p, top-p,
    /// min-p, top-a, eta, xtc, temperature). Built once at construction.
    post_grammar_chain: SamplerChain,
    /// Reusable scratch buffer for the processed logit vector, avoiding a
    /// fresh vocab-sized allocation on every call (defect S6).
    scratch: Vec<f32>,
}

impl Sampler {
    /// Create a new sampler with the given config.
    pub fn new(config: SamplerConfig) -> Self {
        let seed = config.seed.unwrap_or_else(rng::generate_seed);
        let mirostat_mu = 2.0 * config.mirostat_tau;
        let grammar_state = config.grammar.as_ref().map(|g| g.initial_state());
        let (pre_grammar_chain, post_grammar_chain) = SamplerChain::split_from_config(&config);
        Self {
            config,
            rng: Xorshift64::new(seed),
            mirostat_mu,
            grammar_state,
            pre_grammar_chain,
            post_grammar_chain,
            scratch: Vec::new(),
        }
    }

    /// Sample a token ID from logits.
    ///
    /// This is the infallible convenience wrapper around [`Sampler::try_sample`].
    /// A genuinely degenerate distribution (every logit non-finite — e.g.
    /// `banned_tokens` covers the entire vocabulary, or a grammar/EOG
    /// misconfiguration) is logged and falls back to token 0 as a
    /// last resort, documented explicitly rather than silently returned as
    /// if it were a real selection (see defect S1's discussion of the old
    /// `argmax` bug). Through the normal grammar path this should now be
    /// unreachable, since [`grammar::apply_grammar_mask`] refuses to mask
    /// the last surviving candidate.
    pub fn sample(&mut self, logits: &[f32], recent_tokens: &[u32]) -> u32 {
        match self.try_sample(logits, recent_tokens) {
            Ok(token) => token,
            Err(err) => {
                tracing::error!(
                    error = %err,
                    "sampler degenerated to a fully-masked distribution and fell back to token 0 -- \
                     check banned_tokens / logit_bias / grammar+eog_token_ids configuration"
                );
                0
            }
        }
    }

    /// Sample a token ID from logits, surfacing a degenerate ("every logit
    /// masked") distribution as an error instead of silently defaulting to
    /// token 0.
    pub fn try_sample(&mut self, logits: &[f32], recent_tokens: &[u32]) -> RuntimeResult<u32> {
        if logits.is_empty() {
            return Err(RuntimeError::SamplingError {
                message: "empty logits vector".to_string(),
            });
        }

        let token = if self.config.mirostat == 1 || self.config.mirostat == 2 {
            self.sample_mirostat(logits, recent_tokens)
        } else {
            self.sample_standard(logits, recent_tokens)?
        };

        // Advance grammar state after token selection.
        // We look up the token bytes in the pre-built vocab table (binary search by id).
        if let Some(state) = &mut self.grammar_state {
            if let Some(vocab) = &self.config.token_vocab {
                if let Ok(idx) = vocab.binary_search_by_key(&token, |&(id, _)| id) {
                    // Silently ignore advance errors — the mask will catch a stuck state
                    // on the next step and -inf all invalid tokens.
                    let _ = state.advance(&vocab[idx].1);
                }
            }
        }

        Ok(token)
    }

    /// Standard (non-mirostat) sampling pipeline — see the module docs for
    /// the full stage order.
    fn sample_standard(&mut self, logits: &[f32], recent_tokens: &[u32]) -> RuntimeResult<u32> {
        self.scratch.clear();
        self.scratch.extend_from_slice(logits);

        self.pre_grammar_chain.apply_stages_external(
            &mut self.scratch,
            recent_tokens,
            &mut self.rng,
        );

        if let (Some(state), Some(vocab)) = (&self.grammar_state, &self.config.token_vocab) {
            apply_grammar_mask(
                &mut self.scratch,
                state,
                vocab.as_ref(),
                &self.config.eog_token_ids,
            );
        }

        // Greedy shortcut — returns BEFORE any sort/softmax (defect S6).
        if self.config.temperature <= 0.0 || self.config.top_k == 1 {
            return rng::argmax(&self.scratch).ok_or_else(|| RuntimeError::SamplingError {
                message:
                    "no finite logits remain after masking (grammar / bans / logit_bias eliminated every candidate)"
                        .to_string(),
            });
        }

        self.post_grammar_chain.apply_stages_external(
            &mut self.scratch,
            recent_tokens,
            &mut self.rng,
        );

        chain::select_token(&self.scratch, &mut self.rng).ok_or_else(|| {
            RuntimeError::SamplingError {
                message: "no finite logits remain after the sampling pipeline".to_string(),
            }
        })
    }

    /// Mirostat v1/v2 sampling.
    ///
    /// Adaptively controls the "surprise" of generated tokens to maintain
    /// a target perplexity level (tau). This produces more coherent text
    /// than fixed top-k/top-p by dynamically adjusting the token pool.
    /// Mirostat always produces a valid token (it never needs the
    /// `try_sample` error path — see [`mirostat`]'s module docs).
    fn sample_mirostat(&mut self, logits: &[f32], recent_tokens: &[u32]) -> u32 {
        self.scratch.clear();
        self.scratch.extend_from_slice(logits);

        self.pre_grammar_chain.apply_stages_external(
            &mut self.scratch,
            recent_tokens,
            &mut self.rng,
        );

        if let (Some(state), Some(vocab)) = (&self.grammar_state, &self.config.token_vocab) {
            apply_grammar_mask(
                &mut self.scratch,
                state,
                vocab.as_ref(),
                &self.config.eog_token_ids,
            );
        }

        if self.config.temperature > 0.0 && self.config.temperature != 1.0 {
            let inv_temp = 1.0 / self.config.temperature;
            for val in &mut self.scratch {
                *val *= inv_temp;
            }
        }

        let mut mu = self.mirostat_mu;
        let token = if self.config.mirostat == 1 {
            mirostat::sample_v1(
                &self.scratch,
                &mut mu,
                self.config.mirostat_tau,
                self.config.mirostat_eta,
                &mut self.rng,
            )
        } else {
            mirostat::sample_v2(
                &self.scratch,
                &mut mu,
                self.config.mirostat_tau,
                self.config.mirostat_eta,
                &mut self.rng,
            )
        };
        self.mirostat_mu = mu;
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
/// grammar-constrained generation. Mirostat mode, if configured, runs with a
/// freshly-initialized `mu` on every call (also stateless) — for a
/// persistent, meaningfully-adapting `mu`, use [`Sampler`].
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

    let (pre_grammar_chain, post_grammar_chain) = SamplerChain::split_from_config(config);

    let mut scratch = logits.to_vec();
    pre_grammar_chain.apply_stages_external(&mut scratch, recent_tokens, &mut rng);

    if config.mirostat == 1 || config.mirostat == 2 {
        if config.temperature > 0.0 && config.temperature != 1.0 {
            let inv_temp = 1.0 / config.temperature;
            for v in &mut scratch {
                *v *= inv_temp;
            }
        }
        let mut mu = 2.0 * config.mirostat_tau;
        return if config.mirostat == 1 {
            mirostat::sample_v1(
                &scratch,
                &mut mu,
                config.mirostat_tau,
                config.mirostat_eta,
                &mut rng,
            )
        } else {
            mirostat::sample_v2(
                &scratch,
                &mut mu,
                config.mirostat_tau,
                config.mirostat_eta,
                &mut rng,
            )
        };
    }

    if config.temperature <= 0.0 || config.top_k == 1 {
        return rng::argmax(&scratch).unwrap_or(0);
    }

    post_grammar_chain.apply_stages_external(&mut scratch, recent_tokens, &mut rng);
    chain::select_token(&scratch, &mut rng).unwrap_or(0)
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

    // ── Mirostat v1 (defect S5) ────────────────────────────────────────────────

    #[test]
    fn test_mirostat_v1_is_no_longer_silently_ignored() {
        // Basic sanity check only: `mirostat: 1` must produce in-range
        // tokens. NOTE this assertion alone cannot distinguish "v1 actually
        // ran" from "silently fell through to standard sampling" — with
        // `SamplerConfig::mirostat_v1`'s underlying `top_k=0, top_p=1.0,
        // min_p=0.0`, unconstrained standard sampling would *also* produce
        // in-range tokens every time. The real dispatch regression test is
        // `test_mirostat_v1_actually_dispatches_not_falls_through` below,
        // which checks a side effect standard sampling cannot produce.
        let logits = vec![3.0, 2.0, 1.0, 0.5, 0.1, -1.0, -2.0, -5.0];
        let config = SamplerConfig {
            seed: Some(42),
            ..SamplerConfig::mirostat_v1(5.0, 0.1)
        };
        let mut sampler = Sampler::new(config);
        for _ in 0..50 {
            let token = sampler.sample(&logits, &[]);
            assert!((token as usize) < logits.len());
        }
    }

    /// Defect S5 regression (the actual dispatch bug): `mirostat: 1` used to
    /// fall through to ordinary top-k/p/min-p sampling with no warning.
    /// `mirostat_mu` is initialised to `2 * tau` and is *only* ever mutated
    /// inside `sample_mirostat` (see `sample_mirostat`'s `self.mirostat_mu =
    /// mu;`) — the standard pipeline never touches it. So if `mirostat: 1`
    /// silently fell through to `sample_standard`, `mirostat_mu_value()`
    /// would stay frozen at its initial value forever, which
    /// `test_mirostat_v1_is_no_longer_silently_ignored` above cannot detect
    /// but this test can.
    #[test]
    fn test_mirostat_v1_actually_dispatches_not_falls_through() {
        let logits = vec![3.0, 2.0, 1.0, 0.5, 0.1, -1.0, -2.0, -5.0];
        let config = SamplerConfig {
            seed: Some(42),
            ..SamplerConfig::mirostat_v1(5.0, 0.1)
        };
        let mut sampler = Sampler::new(config);
        let initial_mu = sampler.mirostat_mu_value();
        sampler.sample(&logits, &[]);
        assert!(
            (sampler.mirostat_mu_value() - initial_mu).abs() > 1e-6,
            "mirostat_mu must adapt after sampling with mirostat=1; if it \
             didn't move, `mirostat: 1` fell through to standard sampling \
             instead of running the v1 algorithm (defect S5)"
        );
    }

    #[test]
    fn test_mirostat_v1_deterministic_with_seed() {
        let logits = vec![2.0, 1.5, 1.0, 0.5, 0.2];
        let config = SamplerConfig {
            seed: Some(321),
            ..SamplerConfig::mirostat_v1(5.0, 0.1)
        };
        let mut sampler1 = Sampler::new(config.clone());
        let mut sampler2 = Sampler::new(config);
        for _ in 0..20 {
            assert_eq!(sampler1.sample(&logits, &[]), sampler2.sample(&logits, &[]));
        }
    }

    // ── Frequency / presence penalties (defect S5) ───────────────────────────

    #[test]
    fn test_frequency_penalty_applied_via_sampler() {
        // Token 1 repeats heavily in recent history; with a strong frequency
        // penalty it must lose to token 2 despite a slightly lower raw logit.
        //
        // NOTE: `repetition_penalty_window` gates ALL THREE penalty kinds
        // (classic repeat, frequency, presence) since they share one
        // window (matches llama.cpp's single `penalty_last_n`).
        // `SamplerConfig::greedy()` sets the window to 0 (appropriate when
        // only the classic repeat penalty, which defaults to 1.0/off, is in
        // play) -- tests that enable frequency/presence penalties on top of
        // `greedy()` must override the window explicitly.
        let logits = vec![1.0f32, 5.0, 4.9, 1.0];
        let config = SamplerConfig {
            temperature: 0.0,
            frequency_penalty: 2.0,
            repetition_penalty_window: 64,
            ..SamplerConfig::greedy()
        };
        let token = sample(&logits, &config, &[1, 1, 1, 1]);
        assert_eq!(
            token, 2,
            "heavy frequency penalty on token 1 should let token 2 win"
        );
    }

    #[test]
    fn test_presence_penalty_applied_via_sampler() {
        let logits = vec![1.0f32, 5.0, 4.9, 1.0];
        let config = SamplerConfig {
            temperature: 0.0,
            presence_penalty: 3.0,
            repetition_penalty_window: 64,
            ..SamplerConfig::greedy()
        };
        // Even a SINGLE occurrence should be enough for a flat presence
        // penalty of 3.0 to flip the winner from token 1 (5.0) to token 2 (4.9).
        let token = sample(&logits, &config, &[1]);
        assert_eq!(
            token, 2,
            "presence penalty should fire on a single occurrence"
        );
    }

    #[test]
    fn test_default_config_zero_penalties_no_op() {
        // Defaults must not change existing greedy behaviour.
        let logits = vec![1.0f32, 5.0, 4.9, 1.0];
        let token = sample(&logits, &SamplerConfig::greedy(), &[1, 1, 1]);
        assert_eq!(
            token, 1,
            "no frequency/presence penalty configured -> token 1 still wins"
        );
    }

    #[test]
    fn test_select_token_handles_equal_logits() {
        let logits = vec![0.0f32, 0.0, 0.0];
        let mut rng = Xorshift64::new(1);
        let selected = chain::select_token(&logits, &mut rng);
        assert!(matches!(selected, Some(idx) if (idx as usize) < logits.len()));
    }

    #[test]
    fn test_select_token_none_on_all_masked() {
        let logits = vec![f32::NEG_INFINITY; 4];
        let mut rng = Xorshift64::new(1);
        assert_eq!(
            chain::select_token(&logits, &mut rng),
            None,
            "select_token must report None (not silently pick 0) on a fully-masked input"
        );
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
        // After bias, token 1 should dominate. Exercised through the actual
        // stochastic weighted-draw path (`chain::select_token`, temperature
        // 1.0, top_k disabled) rather than the greedy shortcut, so this
        // covers the real S6 code path.
        //
        // This also doubles as a regression test for the `select_token`
        // open-interval draw fix: `select_token` walks candidates in raw
        // vocab-index order (not sorted by probability, per S6), so with
        // the old half-open `[0, 1)` draw, an RNG stream that ever produced
        // exactly `r == 0.0` would make the *lowest-index* token with any
        // nonzero probability mass win regardless of how astronomically
        // small its true probability was. Seed 7's first `next_f32()` draw
        // is exactly `0.0`, which used to make token 0 win here despite its
        // softmax probability being effectively zero next to token 1's.
        let logits = vec![10.0f32, -20.0, -20.0, -20.0];
        let mut bias = std::collections::HashMap::new();
        bias.insert(1u32, 100.0f32); // huge positive bias on token 1

        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0, // disable top-k filtering so the weighted draw actually runs
            seed: Some(7),
            logit_bias: bias,
            ..SamplerConfig::greedy()
        };
        let mut sampler = Sampler::new(config);
        // With a +100 bias, token 1's effective logit = 80, far above every
        // other token's (~10 and ~-20): softmax probability of token 1 is
        // within float epsilon of 1.0, so it must win the weighted draw too.
        let tok = sampler.sample(&logits, &[]);
        assert_eq!(
            tok, 1,
            "large positive bias should make token 1 dominate even under stochastic sampling"
        );
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

    // ── Defect S1: grammar-complete EOG handling, end to end ─────────────────

    #[test]
    fn test_s1_grammar_complete_allows_eog_and_avoids_token_zero_loop() {
        // Vocab: 0 = a non-EOG token that would otherwise win on raw logit
        // value, 1 = "h", 2 = "i" (the grammar requires BOTH, one per step,
        // so completion happens only after the second sample call), 3 = an
        // EOG token that is NOT part of the grammar's literal alphabet.
        let vocab: Vec<(u32, Vec<u8>)> = vec![
            (0, b"zzz".to_vec()),
            (1, b"h".to_vec()),
            (2, b"i".to_vec()),
            (3, b"<eos>".to_vec()),
        ];
        let g = Arc::new(Grammar::parse(r#"root ::= "h" "i""#).unwrap());
        let config = SamplerConfig {
            temperature: 0.0, // greedy -- exercises the exact argmax path from S1
            grammar: Some(g),
            token_vocab: Some(Arc::new(vocab)),
            eog_token_ids: vec![3],
            ..SamplerConfig::default()
        };
        let mut sampler = Sampler::new(config);

        // Step 1: only "h" (token 1) is grammar-valid; token 0 has the
        // highest raw logit but must lose to the grammar constraint.
        let logits = vec![100.0f32, 1.0, 1.0, 1.0];
        let tok1 = sampler.sample(&logits, &[]);
        assert_eq!(
            tok1, 1,
            "grammar must force token 1 ('h') despite token 0's higher logit"
        );
        assert!(
            !sampler.grammar_complete(),
            "grammar should not be complete yet"
        );

        // Step 2: only "i" (token 2) is grammar-valid; completes the grammar.
        let tok_mid = sampler.sample(&logits, &[1]);
        assert_eq!(tok_mid, 2, "second token must be 'i' (id=2)");

        // Step 3: grammar is now complete. Before the S1 fix, EVERY logit
        // (including the EOG token) was masked to -inf here, and argmax's
        // "return 0 on an all -inf vector" bug meant token 0 would be
        // emitted regardless of its actual logit.
        assert!(
            sampler.grammar_complete(),
            "grammar must be complete after 'h' + 'i'"
        );
        let tok2 = sampler.sample(&logits, &[1, 2]);
        assert_eq!(
            tok2, 3,
            "once the grammar is complete, the configured EOG token must become selectable"
        );

        // Drive several more steps: token 0 must never be emitted just
        // because the distribution degenerated to all -inf.
        for _ in 0..10 {
            let tok = sampler.sample(&logits, &[1, 2]);
            assert_ne!(
                tok, 0,
                "token 0 must never be emitted as a silent fallback for a fully-masked distribution"
            );
        }
    }

    #[test]
    fn test_try_sample_errors_instead_of_silently_returning_zero() {
        // Ban the entire vocabulary -- a genuinely unrecoverable
        // configuration. `try_sample` must surface this as an error rather
        // than `sample`'s documented last-resort fallback of token 0.
        let logits = vec![1.0f32, 2.0, 3.0];
        let config = SamplerConfig {
            temperature: 0.0,
            banned_tokens: vec![0, 1, 2],
            ..SamplerConfig::greedy()
        };
        let mut sampler = Sampler::new(config);
        let result = sampler.try_sample(&logits, &[]);
        assert!(
            result.is_err(),
            "banning the entire vocabulary must surface as an error from try_sample"
        );
    }

    // ── Defect S2: unseeded samplers must not collide ─────────────────────────

    #[test]
    fn test_unseeded_samplers_produce_different_rng_streams() {
        let config = SamplerConfig::default(); // seed: None
        let sampler_a = Sampler::new(config.clone());
        let sampler_b = Sampler::new(config);
        assert_ne!(
            sampler_a.rng_state(),
            sampler_b.rng_state(),
            "two unseeded samplers constructed back-to-back must not draw the same seed"
        );
    }

    #[test]
    fn test_many_unseeded_samplers_all_distinct() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let sampler = Sampler::new(SamplerConfig::default());
            assert!(
                seen.insert(sampler.rng_state()),
                "unseeded Sampler construction produced a duplicate RNG seed"
            );
        }
    }

    #[test]
    fn test_explicitly_seeded_sampler_still_reproducible() {
        let config = SamplerConfig {
            seed: Some(4242),
            ..SamplerConfig::default()
        };
        let a = Sampler::new(config.clone());
        let b = Sampler::new(config);
        assert_eq!(
            a.rng_state(),
            b.rng_state(),
            "an explicitly seeded sampler must remain fully reproducible"
        );
    }
}

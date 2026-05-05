//! Composable sampler chain — trait-based pipeline for token selection.
//!
//! Each [`SamplerStage`] transforms a logit vector in-place. Stages are
//! composed into a [`SamplerChain`] that runs them in order before final
//! token selection.
//!
//! Built-in stages: [`RepetitionPenalty`], `GrammarMask`,
//! [`TemperatureScale`], [`TopK`], [`TopP`], [`MinP`].
//!
//! # Example
//!
//! ```ignore
//! use oxillama_runtime::sampling::chain::*;
//!
//! let chain = SamplerChain::new()
//!     .push(RepetitionPenalty::new(1.1, 64))
//!     .push(TemperatureScale::new(0.8))
//!     .push(TopK::new(40))
//!     .push(TopP::new(0.9));
//!
//! let logits = vec![1.0, 2.0, 3.0, 0.5];
//! let token = chain.sample(&logits, &recent_tokens);
//! ```

use std::collections::HashSet;

/// A single stage in the sampling pipeline.
///
/// Each stage receives the full logit vector (mutable) and the recent token
/// history, and transforms the logits in place (e.g., applying penalties,
/// masking, temperature scaling).
pub trait SamplerStage: Send + Sync {
    /// Apply this stage to the logit vector in place.
    fn apply(&self, logits: &mut Vec<f32>, recent_tokens: &[u32]);

    /// Human-readable name for logging / debugging.
    fn name(&self) -> &'static str;
}

/// A composable pipeline of [`SamplerStage`]s followed by a final selection step.
pub struct SamplerChain {
    stages: Vec<Box<dyn SamplerStage>>,
    /// Seed for the final selection RNG.
    seed: u64,
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
            seed: 0xDEAD_BEEF_CAFE_BABE,
        }
    }

    /// Set the RNG seed for the final selection step.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Append a stage to the pipeline. Returns self for chaining.
    pub fn push(mut self, stage: impl SamplerStage + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Run all stages on a copy of the logits and select a token.
    ///
    /// The original logit slice is not modified.
    pub fn sample(&self, logits: &[f32], recent_tokens: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        let mut processed = logits.to_vec();

        for stage in &self.stages {
            stage.apply(&mut processed, recent_tokens);
        }

        // Final selection: softmax + weighted random
        select_token(&processed, self.seed)
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

    /// Build a chain from a `SamplerConfig`, replicating the standard pipeline.
    ///
    /// Pipeline order:
    /// logit-bias → repetition penalty → DRY → XTC → TypicalP → TopA → Eta
    ///   → temperature → top-K → min-P → top-P.
    ///
    /// Logit-bias must come first so that bans and boosts are visible to all
    /// downstream filtering stages. The five advanced stages are inserted after
    /// repetition penalty (they work on logit-scale values) but before temperature
    /// scaling (so they see the pre-temperature distribution shape).
    pub fn from_config(config: &super::SamplerConfig) -> Self {
        use super::advanced::{DryStage, EtaStage, TopAStage, TypicalPStage, XtcStage};

        let mut chain = Self::new();

        if let Some(seed) = config.seed {
            chain = chain.with_seed(seed);
        }

        // Insert logit-bias / banned-tokens stage first (before everything else).
        if !config.logit_bias.is_empty() || !config.banned_tokens.is_empty() {
            chain = chain.push(LogitBias::new(
                config.logit_bias.clone(),
                config.banned_tokens.clone(),
            ));
        }

        if config.repetition_penalty != 1.0 {
            chain = chain.push(RepetitionPenalty::new(
                config.repetition_penalty,
                config.repetition_penalty_window,
            ));
        }

        // ── Advanced stages (Track B, v0.1.7) ────────────────────────────────
        // Order: DRY → XTC → TypicalP → TopA → Eta
        if config.dry_multiplier != 0.0 {
            chain = chain.push(DryStage::new(
                config.dry_multiplier,
                config.dry_base,
                config.dry_allowed_length,
                Vec::new(), // sequence_breakers — not yet in SamplerConfig; extend later
            ));
        }

        if config.xtc_threshold < 1.0 && config.xtc_probability > 0.0 {
            let seed = config.seed.unwrap_or(0xDEAD_BEEF_CAFE_BABE);
            chain = chain.push(XtcStage::new(
                config.xtc_threshold,
                config.xtc_probability,
                seed,
            ));
        }

        if config.typical_p < 1.0 {
            chain = chain.push(TypicalPStage::new(config.typical_p));
        }

        if config.top_a != 0.0 {
            chain = chain.push(TopAStage::new(config.top_a));
        }

        if config.eta_cutoff != 0.0 || config.epsilon_cutoff != 0.0 {
            chain = chain.push(EtaStage::new(config.eta_cutoff, config.epsilon_cutoff));
        }
        // ─────────────────────────────────────────────────────────────────────

        if config.temperature <= 0.0 {
            // Greedy: just push the greedy selector
            chain = chain.push(GreedySelect);
            return chain;
        }

        if config.temperature != 1.0 {
            chain = chain.push(TemperatureScale::new(config.temperature));
        }

        if config.top_k > 0 {
            chain = chain.push(TopK::new(config.top_k));
        }

        if config.min_p > 0.0 {
            chain = chain.push(MinP::new(config.min_p));
        }

        if config.top_p < 1.0 {
            chain = chain.push(TopP::new(config.top_p));
        }

        chain
    }
}

// ── Built-in stages ──────────────────────────────────────────────────────────

/// Repetition penalty stage — penalizes recently generated tokens.
pub struct RepetitionPenalty {
    penalty: f32,
    window: usize,
}

impl RepetitionPenalty {
    /// Create a new repetition penalty stage.
    ///
    /// `penalty` of 1.0 = no effect. Values > 1.0 penalize repetition.
    /// `window` is the number of recent tokens to consider.
    pub fn new(penalty: f32, window: usize) -> Self {
        Self { penalty, window }
    }
}

impl SamplerStage for RepetitionPenalty {
    fn apply(&self, logits: &mut Vec<f32>, recent_tokens: &[u32]) {
        if self.penalty == 1.0 || recent_tokens.is_empty() {
            return;
        }
        let start = recent_tokens.len().saturating_sub(self.window);
        for &token in &recent_tokens[start..] {
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
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32]) {
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

/// Top-K filtering stage — keeps only the K highest logits; sets rest to -inf.
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
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32]) {
        if self.k == 0 || self.k >= logits.len() {
            return;
        }
        // Find the k-th largest value
        let mut sorted: Vec<f32> = logits.clone();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[self.k - 1];
        // Keep tokens at or above threshold, up to k
        let mut kept = 0usize;
        for v in logits.iter_mut() {
            if *v >= threshold && kept < self.k {
                kept += 1;
            } else if *v < threshold {
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
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32]) {
        if self.p >= 1.0 {
            return;
        }
        // Softmax first
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let probs: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return;
        }
        let probs: Vec<f32> = probs.iter().map(|&p| p / sum).collect();

        // Sort indices by probability descending
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find cutoff
        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indices.len();
        for (i, &idx) in indices.iter().enumerate() {
            cumulative += probs[idx];
            if cumulative >= self.p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Mask everything beyond cutoff
        let kept: HashSet<usize> = indices[..cutoff_idx].iter().copied().collect();
        for (i, v) in logits.iter_mut().enumerate() {
            if !kept.contains(&i) {
                *v = f32::NEG_INFINITY;
            }
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
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32]) {
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
/// Processing order (matches `sampling::mod::apply_logit_bias_and_banned_tokens`):
/// 1. Banned tokens → `f32::NEG_INFINITY` (hard ban, cannot be overridden by bias).
/// 2. Logit biases are added to surviving logits.
pub struct LogitBias {
    /// Per-token additive biases.
    biases: std::collections::HashMap<u32, f32>,
    /// Tokens that must never be sampled.
    banned: Vec<u32>,
}

impl LogitBias {
    /// Create a new logit-bias stage.
    ///
    /// `biases` maps token IDs to additive values (positive = boost,
    /// negative = suppress).  `banned` is the list of tokens to hard-ban.
    pub fn new(biases: std::collections::HashMap<u32, f32>, banned: Vec<u32>) -> Self {
        Self { biases, banned }
    }

    /// Create a stage with only hard-banned tokens and no biases.
    pub fn banned_only(banned: Vec<u32>) -> Self {
        Self {
            biases: std::collections::HashMap::new(),
            banned,
        }
    }

    /// Create a stage with only biases and no bans.
    pub fn biases_only(biases: std::collections::HashMap<u32, f32>) -> Self {
        Self {
            biases,
            banned: Vec::new(),
        }
    }
}

impl SamplerStage for LogitBias {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32]) {
        // Step 1: hard ban.
        for &token in &self.banned {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
        // Step 2: additive bias (skip already-banned slots).
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
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32]) {
        if logits.is_empty() {
            return;
        }
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        for (i, v) in logits.iter_mut().enumerate() {
            if i != max_idx {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "greedy"
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Final token selection: softmax + weighted random using xorshift64.
fn select_token(logits: &[f32], seed: u64) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // Softmax
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    if sum <= 0.0 {
        // All -inf — fallback to first non-NEG_INFINITY, or 0
        return logits
            .iter()
            .enumerate()
            .find(|(_, &v)| v > f32::NEG_INFINITY)
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Check if only one token survived (common after greedy/aggressive filtering)
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
        return survivor_idx;
    }

    // Weighted random selection via xorshift64
    let mut state = if seed == 0 {
        0x517c_c1b7_2722_0a95_u64
    } else {
        seed
    };
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    let r = (state >> 40) as f32 / (1u64 << 24) as f32;

    let mut cumulative = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        cumulative += e / sum;
        if r < cumulative {
            return i as u32;
        }
    }

    (logits.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplerConfig;

    #[test]
    fn test_empty_chain_selects_token() {
        let chain = SamplerChain::new().with_seed(42);
        let logits = vec![1.0, 2.0, 3.0];
        let token = chain.sample(&logits, &[]);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_greedy_chain() {
        let chain = SamplerChain::new().push(GreedySelect);
        let logits = vec![1.0, 5.0, 3.0, 0.5];
        let token = chain.sample(&logits, &[]);
        assert_eq!(token, 1); // argmax
    }

    #[test]
    fn test_temperature_affects_distribution() {
        // Very cold temperature should always pick top
        let chain_cold = SamplerChain::new()
            .with_seed(42)
            .push(TemperatureScale::new(0.01));

        let logits = vec![3.0, 2.0, 1.0, 0.5];
        let token = chain_cold.sample(&logits, &[]);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_top_k_limits_candidates() {
        let chain = SamplerChain::new().push(TopK::new(1)).with_seed(42);
        let logits = vec![1.0, 5.0, 3.0];
        let token = chain.sample(&logits, &[]);
        assert_eq!(token, 1); // only top-1 survives
    }

    #[test]
    fn test_repetition_penalty_reduces_repeated() {
        let chain = SamplerChain::new()
            .push(RepetitionPenalty::new(100.0, 64))
            .push(GreedySelect);
        let logits = vec![1.0, 5.0, 4.9, 1.0];
        // Without penalty, token 1 wins. With penalty on token 1, token 2 should win.
        let token = chain.sample(&logits, &[1]);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_chain_from_config_greedy() {
        let config = SamplerConfig::greedy();
        let chain = SamplerChain::from_config(&config);
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
    fn test_stage_names() {
        let chain = SamplerChain::new()
            .push(RepetitionPenalty::new(1.1, 64))
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
        let chain = SamplerChain::new().push(GreedySelect);
        assert_eq!(chain.sample(&[], &[]), 0);
    }

    #[test]
    fn test_min_p_filters_low_prob() {
        let chain = SamplerChain::new().push(MinP::new(0.1)).push(GreedySelect);
        // One dominant token
        let logits = vec![10.0, -10.0, -10.0, -10.0];
        let token = chain.sample(&logits, &[]);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_top_p_nucleus() {
        let chain = SamplerChain::new().push(TopP::new(0.5)).with_seed(42);
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
        let chain = SamplerChain::new()
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
        let mut biases = std::collections::HashMap::new();
        biases.insert(2u32, 100.0f32);
        let chain = SamplerChain::new()
            .push(LogitBias::biases_only(biases))
            .push(GreedySelect);
        let logits = vec![10.0f32, 10.0, 0.0]; // token 2 has lowest logit before bias
        let tok = chain.sample(&logits, &[]);
        assert_eq!(tok, 2, "large positive bias should make token 2 win");
    }

    #[test]
    fn test_logit_bias_ban_wins_over_positive_bias() {
        // A banned token should stay at -inf even if it also has a positive bias.
        let mut biases = std::collections::HashMap::new();
        biases.insert(0u32, 999.0f32); // very large positive bias on token 0
        let chain = SamplerChain::new()
            .push(LogitBias::new(biases, vec![0])) // but also banned
            .push(GreedySelect);
        let logits = vec![10.0f32, 1.0, 1.0];
        let tok = chain.sample(&logits, &[]);
        // Token 0 is banned — the positive bias must NOT override the ban.
        assert_ne!(tok, 0, "ban must override positive bias");
    }

    #[test]
    fn test_from_config_includes_logit_bias_stage() {
        let mut biases = std::collections::HashMap::new();
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
}

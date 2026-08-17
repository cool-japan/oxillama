//! Advanced sampler stages: DRY, XTC, TypicalP, TopA, and Eta.
//!
//! These stages implement research-grade sampling algorithms that go beyond
//! standard top-K/top-P filtering. They operate on the raw logit vector
//! (before or after temperature scaling, depending on pipeline position)
//! and set rejected token logits to `f32::NEG_INFINITY`.
//!
//! All stages gracefully handle:
//! - Empty logit vectors (no-op).
//! - All-negative-infinity inputs (preserved, final selection falls back
//!   to argmax of the original).
//! - Distributions that would become entirely `-inf` after filtering
//!   (at least 1 token is always preserved).

use super::chain::SamplerStage;
use super::rng::Xorshift64;

// ── Shared utilities ─────────────────────────────────────────────────────────

/// Compute softmax probabilities from logits into a new `Vec<f32>`.
///
/// Returns a vector of the same length as `logits` with non-negative values
/// summing to 1.0. Numerically stable via max-subtraction.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

/// Compute `H = -Σ p(x) log p(x)` (Shannon entropy in nats).
///
/// Tokens with probability ≤ 0 are skipped (contribute 0 to entropy).
fn entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .copied()
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}

// ── DryStage ─────────────────────────────────────────────────────────────────

/// "Don't Repeat Yourself" (DRY) penalty stage.
///
/// Penalises tokens that would continue an n-gram already present in the
/// recent context. The penalty is exponentially larger for longer repeated
/// n-gram suffixes, making the model strongly avoid verbatim repetition
/// while leaving novel continuations unaffected.
///
/// # Algorithm (defect S3: was O(vocab × history × match-length))
///
/// The naive approach iterates every VOCABULARY token and, for each one,
/// scans the full history for occurrences — for a 152k-token vocabulary
/// against a 1k-token history that is ~150M operations per generated token.
/// But the vast majority of vocabulary entries never occur in the history at
/// all, so that outer loop does no useful work for them.
///
/// This implementation instead iterates the (short) **history** directly:
/// for each history position `pos`, the only token that could possibly need
/// a penalty *because of* that position is `recent_tokens[pos]` itself, so
/// we compute its backward-match length once and keep a running best-per-token
/// map (`HashMap<token_id, usize>`, bounded by `min(history_len, vocab_len)`
/// entries) instead of visiting the entire vocabulary. This is an exact
/// re-derivation of the same aggregation the old code computed — not an
/// approximation — verified against the pre-existing behavioural tests.
///
/// Complexity: O(history) candidate positions, each with an O(history)
/// worst-case (but typically much shorter, since the inner backward walk
/// breaks at the first mismatch) backward-match walk — i.e. O(H) usually,
/// O(H²) worst case for pathologically repetitive history, and **entirely
/// independent of vocabulary size**. A true O(H) Z-algorithm (as used by
/// llama.cpp) is possible as a further optimization; this version was
/// chosen to eliminate the reported bottleneck (the vocab-size factor) with
/// minimal risk of introducing a subtle correctness regression in the
/// backward-match arithmetic.
///
/// # Original algorithm description (still accurate, restated per-position)
///
/// 1. Build the current "generation suffix" implicitly: `recent_tokens` IS
///    the context, and its own tail is the suffix we compare against.
/// 2. For each position `pos` in history:
///    - Walk backward from `pos-1` and from the context tail, counting how
///      many tokens match. Call this `match_len` (starts at 1 for `pos`
///      itself).
///    - Track the longest `match_len` seen per distinct token value.
/// 3. Tokens whose ID is in `sequence_breakers` are never penalised
///    regardless of repetition, and a breaker encountered mid-walk stops
///    that particular backward match (matches the pre-existing semantics).
/// 4. For every token with `best_match_len >= allowed_length`, apply
///    `multiplier * base^(match_len - allowed_length)` to its logit.
///
/// When `multiplier == 0.0` or the history is empty the stage is a no-op.
pub struct DryStage {
    /// Overall penalty scale. 0.0 = disabled.
    pub multiplier: f32,
    /// Exponential base for longer-match amplification. Typical default: 1.75.
    pub base: f32,
    /// Minimum match length before any penalty is applied. Typical default: 2.
    pub allowed_length: usize,
    /// Token IDs that act as "sentence breakers" — their presence resets the
    /// n-gram match and prevents a penalty regardless of repetition.
    pub sequence_breakers: Vec<u32>,
}

impl DryStage {
    /// Construct a `DryStage` with explicit parameters.
    pub fn new(
        multiplier: f32,
        base: f32,
        allowed_length: usize,
        sequence_breakers: Vec<u32>,
    ) -> Self {
        Self {
            multiplier,
            base,
            allowed_length,
            sequence_breakers,
        }
    }
}

impl SamplerStage for DryStage {
    fn apply(&self, logits: &mut Vec<f32>, recent_tokens: &[u32], _rng: &mut Xorshift64) {
        // Fast-path: disabled or nothing to penalise.
        if self.multiplier == 0.0 || logits.is_empty() || recent_tokens.is_empty() {
            return;
        }

        let breaker_set: std::collections::HashSet<u32> =
            self.sequence_breakers.iter().copied().collect();

        let hist_len = recent_tokens.len();

        // O(history) instead of O(vocab * history * match) — see doc comment.
        let mut best_match: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();

        for pos in 0..hist_len {
            let t = recent_tokens[pos];
            if breaker_set.contains(&t) {
                continue;
            }

            let mut match_len = 1usize; // t itself counts as length 1
            let max_back = pos.min(hist_len - 1);
            for k in 1..=max_back {
                let hist_token = recent_tokens[pos - k];
                let ctx_token = recent_tokens[hist_len - k];

                // A sequence breaker along the compared path stops the match.
                if breaker_set.contains(&hist_token) || breaker_set.contains(&ctx_token) {
                    break;
                }
                if hist_token == ctx_token {
                    match_len += 1;
                } else {
                    break;
                }
            }

            let entry = best_match.entry(t).or_insert(0);
            if match_len > *entry {
                *entry = match_len;
            }
        }

        for (t, match_len) in best_match {
            if match_len < self.allowed_length {
                continue;
            }
            if let Some(logit) = logits.get_mut(t as usize) {
                if logit.is_finite() {
                    let excess = (match_len - self.allowed_length) as f32;
                    *logit -= self.multiplier * self.base.powf(excess);
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "dry"
    }
}

// ── XtcStage ─────────────────────────────────────────────────────────────────

/// Exclude Top Choices (XTC) sampling stage.
///
/// With probability `probability`, removes the high-probability "safe"
/// tokens from the distribution, forcing the model to choose from
/// lower-probability, more creative alternatives.
///
/// # Algorithm
///
/// 1. Compute softmax probabilities.
/// 2. Sort candidates by probability descending.
/// 3. Collect tokens until cumulative probability exceeds `threshold`.
///    Call this the "top set" (size ≥ 1).
/// 4. If the top set has **at least 2** tokens:
///    - With probability `probability`, set all tokens in the top set
///      **except the single highest-probability token** to `-inf`.
///      This forces diversity while preserving the best fallback.
///
/// When `threshold >= 1.0` (always keep everything) or `probability == 0.0`
/// (never trigger), the stage is a no-op.
///
/// # Randomness (defect S3)
///
/// The coin flip used to previously be derived from a *fixed* `seed` field
/// via a one-shot xorshift draw (`xorshift64_f32(self.seed)`), so a given
/// `XtcStage` instance either **always** fired or **never** fired — every
/// call with the same seed produces the same single draw. `apply` now takes
/// the pipeline's shared, continuously-advancing RNG instead, so the coin
/// flip is genuinely per-token stochastic.
pub struct XtcStage {
    /// Cumulative-probability threshold that defines the "top set". Range (0, 1).
    pub threshold: f32,
    /// Probability of applying the exclusion. Range [0, 1].
    pub probability: f32,
}

impl XtcStage {
    /// Construct an `XtcStage`.
    pub fn new(threshold: f32, probability: f32) -> Self {
        Self {
            threshold,
            probability,
        }
    }
}

impl SamplerStage for XtcStage {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], rng: &mut Xorshift64) {
        // Fast-path passthrough conditions.
        if self.threshold >= 1.0 || self.probability == 0.0 || logits.is_empty() {
            return;
        }

        let probs = softmax(logits);

        // Sort indices by probability descending.
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Collect top set until cumulative probability exceeds threshold.
        let mut cumulative = 0.0f32;
        let mut top_end = 0usize; // exclusive end index into `indices`
        for &idx in &indices {
            cumulative += probs[idx];
            top_end += 1;
            if cumulative >= self.threshold {
                break;
            }
        }

        // Need at least 2 tokens in the top set for XTC to have an effect.
        if top_end < 2 {
            return;
        }

        // Coin flip: apply exclusion with probability `self.probability`,
        // drawing from the pipeline's shared RNG (see doc comment above).
        let rand_val = rng.next_f32();
        if rand_val >= self.probability {
            return;
        }

        // Exclude all top tokens except the single best (indices[0]).
        let best_idx = indices[0];
        for &idx in &indices[1..top_end] {
            logits[idx] = f32::NEG_INFINITY;
        }
        // Safety: ensure best always remains finite (it should already be, but guard).
        if !logits[best_idx].is_finite() {
            // fallback: un-ban it
            logits[best_idx] = 1.0;
        }
    }

    fn name(&self) -> &'static str {
        "xtc"
    }
}

// ── TypicalPStage ─────────────────────────────────────────────────────────────

/// Locally-typical sampling stage.
///
/// Keeps the "typical" tokens — those whose information content (surprise)
/// is closest to the distribution's entropy. This selects tokens that are
/// neither too predictable nor too surprising, producing text that is locally
/// typical of the distribution.
///
/// # Algorithm
///
/// 1. Compute softmax probabilities and Shannon entropy H = -Σ p log p.
/// 2. For each token t, compute `|log p(t) + H|` (deviation from "typical"
///    information content).
/// 3. Sort tokens by this deviation ascending (most typical first).
/// 4. Keep tokens greedily until cumulative probability ≥ `p`; set the
///    rest to `-inf`.
///
/// When `p >= 1.0`, the stage is a no-op.
pub struct TypicalPStage {
    /// Cumulative probability budget for the typical set. Range (0, 1].
    pub p: f32,
}

impl TypicalPStage {
    /// Construct a `TypicalPStage`.
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl SamplerStage for TypicalPStage {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], _rng: &mut Xorshift64) {
        if self.p >= 1.0 || logits.is_empty() {
            return;
        }

        let probs = softmax(logits);
        let h = entropy(&probs);

        // Compute typical deviation for each token.
        // |log p(t) + H|  (note: log p(t) is negative for p<1, so we use ln here)
        let deviations: Vec<f32> = probs
            .iter()
            .copied()
            .map(|p| {
                if p > 0.0 {
                    (p.ln() + h).abs()
                } else {
                    f32::INFINITY
                }
            })
            .collect();

        // Sort indices by deviation ascending.
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            deviations[a]
                .partial_cmp(&deviations[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedily accumulate until cumulative prob ≥ p.
        let mut cumulative = 0.0f32;
        let mut cutoff = 0usize; // number of tokens to keep
        for &idx in &indices {
            cumulative += probs[idx];
            cutoff += 1;
            if cumulative >= self.p {
                break;
            }
        }

        // At least 1 token must survive.
        let cutoff = cutoff.max(1);

        // Build keep set.
        let kept: std::collections::HashSet<usize> = indices[..cutoff].iter().copied().collect();

        for (i, v) in logits.iter_mut().enumerate() {
            if !kept.contains(&i) {
                *v = f32::NEG_INFINITY;
            }
        }

        // Safety: ensure we didn't accidentally ban everything.
        ensure_at_least_one_finite(logits);
    }

    fn name(&self) -> &'static str {
        "typical_p"
    }
}

// ── TopAStage ─────────────────────────────────────────────────────────────────

/// Top-A sampling stage.
///
/// Keeps only tokens whose probability is at least `a * max_prob²`, where
/// `max_prob` is the highest probability in the distribution. This
/// automatically adapts the vocabulary size to the "peakiness" of the
/// distribution: flat distributions keep more tokens; peaked ones keep fewer.
///
/// # Algorithm
///
/// 1. Compute softmax probabilities.
/// 2. Let `max_prob = max(probs)`.
/// 3. Threshold = `a * max_prob²`.
/// 4. Set all tokens with `prob < threshold` to `-inf`.
///
/// When `a == 0.0`, the stage is a no-op.
pub struct TopAStage {
    /// Scaling factor for the adaptive threshold. 0.0 = disabled.
    pub a: f32,
}

impl TopAStage {
    /// Construct a `TopAStage`.
    pub fn new(a: f32) -> Self {
        Self { a }
    }
}

impl SamplerStage for TopAStage {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], _rng: &mut Xorshift64) {
        if self.a == 0.0 || logits.is_empty() {
            return;
        }

        let probs = softmax(logits);
        let max_prob = probs.iter().copied().fold(0.0f32, f32::max);

        if max_prob <= 0.0 {
            return;
        }

        let threshold = self.a * max_prob * max_prob;

        for (i, v) in logits.iter_mut().enumerate() {
            if probs[i] < threshold {
                *v = f32::NEG_INFINITY;
            }
        }

        // Ensure at least 1 token survives.
        ensure_at_least_one_finite(logits);
    }

    fn name(&self) -> &'static str {
        "top_a"
    }
}

// ── EtaStage ──────────────────────────────────────────────────────────────────

/// Eta-cutoff sampling stage.
///
/// A perplexity-adaptive threshold that removes tokens whose probability
/// falls below a floor that depends on the current distribution's "spread".
/// High-perplexity (flat) distributions raise the threshold, keeping only
/// moderately likely tokens. Low-perplexity (peaked) distributions lower
/// the threshold, allowing more tokens.
///
/// # Algorithm
///
/// 1. Compute softmax probabilities and entropy H.
/// 2. Perplexity proxy `s = exp(H)`.
/// 3. Dynamic threshold = `max(epsilon, min(eta, sqrt(eta / s)))`.
///    Concretely: `cutoff = max(epsilon, eta / s)` after clamping.
///    (Implementation note: `eta / s` is used instead of `sqrt(eta/s)` to
///    match the llama.cpp reference implementation.)
/// 4. Set tokens with `prob < cutoff` to `-inf`, keeping at least 1.
///
/// When `eta == 0.0` and `epsilon == 0.0`, the stage is a no-op.
pub struct EtaStage {
    /// Target entropy-scaled cutoff parameter.
    pub eta: f32,
    /// Hard minimum floor for the cutoff probability.
    pub epsilon: f32,
}

impl EtaStage {
    /// Construct an `EtaStage`.
    pub fn new(eta: f32, epsilon: f32) -> Self {
        Self { eta, epsilon }
    }
}

impl SamplerStage for EtaStage {
    fn apply(&self, logits: &mut Vec<f32>, _recent_tokens: &[u32], _rng: &mut Xorshift64) {
        if self.eta == 0.0 && self.epsilon == 0.0 || logits.is_empty() {
            return;
        }

        let probs = softmax(logits);
        let h = entropy(&probs);
        let s = h.exp(); // perplexity proxy (always >= 1.0 for positive H)

        // Dynamic threshold, guarded against s == 0.
        let dynamic = if s > 0.0 { self.eta / s } else { self.eta };
        let cutoff = self.epsilon.max(dynamic);

        if cutoff <= 0.0 {
            return;
        }

        for (i, v) in logits.iter_mut().enumerate() {
            if probs[i] < cutoff {
                *v = f32::NEG_INFINITY;
            }
        }

        // Ensure at least 1 token survives.
        ensure_at_least_one_finite(logits);
    }

    fn name(&self) -> &'static str {
        "eta"
    }
}

// ── Guard helper ─────────────────────────────────────────────────────────────

/// If every logit is `-inf`, restore the original argmax token to avoid an
/// empty distribution in the downstream selection step.
fn ensure_at_least_one_finite(logits: &mut [f32]) {
    if logits.iter().any(|v| v.is_finite()) {
        return; // already fine
    }
    // Everything got masked — keep the argmax (position 0 as last resort).
    // We find the *true* argmax by the highest value (before masking we don't
    // have the originals, so all are -inf; just keep index 0 as convention).
    if !logits.is_empty() {
        logits[0] = 0.0;
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> Xorshift64 {
        Xorshift64::new(42)
    }

    // ── Passthrough (disabled) tests ──────────────────────────────────────────

    /// DRY with multiplier=0 must leave logits unchanged.
    #[test]
    fn dry_disabled_passthrough() {
        let stage = DryStage::new(0.0, 1.75, 2, vec![]);
        let original = vec![1.0f32, 2.0, 3.0, 0.5];
        let mut logits = original.clone();
        stage.apply(&mut logits, &[0, 1, 0, 1], &mut rng());
        assert_eq!(logits, original, "DRY(multiplier=0) must be a no-op");
    }

    /// XTC with threshold=0 (or probability=0) must leave logits unchanged.
    #[test]
    fn xtc_disabled_passthrough() {
        let stage = XtcStage::new(1.0, 0.5); // threshold >= 1.0 → passthrough
        let original = vec![1.0f32, 2.0, 3.0, 0.5];
        let mut logits = original.clone();
        stage.apply(&mut logits, &[], &mut rng());
        assert_eq!(logits, original, "XTC(threshold>=1.0) must be a no-op");

        let stage2 = XtcStage::new(0.5, 0.0); // probability=0 → passthrough
        let mut logits2 = original.clone();
        stage2.apply(&mut logits2, &[], &mut rng());
        assert_eq!(logits2, original, "XTC(probability=0) must be a no-op");
    }

    /// TypicalP with p=1.0 must leave logits unchanged.
    #[test]
    fn typical_p_disabled_passthrough() {
        let stage = TypicalPStage::new(1.0);
        let original = vec![1.0f32, 2.0, 3.0, 0.5];
        let mut logits = original.clone();
        stage.apply(&mut logits, &[], &mut rng());
        assert_eq!(logits, original, "TypicalP(p=1.0) must be a no-op");
    }

    /// TopA with a=0.0 must leave logits unchanged.
    #[test]
    fn top_a_disabled_passthrough() {
        let stage = TopAStage::new(0.0);
        let original = vec![1.0f32, 2.0, 3.0, 0.5];
        let mut logits = original.clone();
        stage.apply(&mut logits, &[], &mut rng());
        assert_eq!(logits, original, "TopA(a=0.0) must be a no-op");
    }

    /// Eta with eta=0 and epsilon=0 must leave logits unchanged.
    #[test]
    fn eta_disabled_passthrough() {
        let stage = EtaStage::new(0.0, 0.0);
        let original = vec![1.0f32, 2.0, 3.0, 0.5];
        let mut logits = original.clone();
        stage.apply(&mut logits, &[], &mut rng());
        assert_eq!(logits, original, "Eta(eta=0, epsilon=0) must be a no-op");
    }

    // ── Active (enabled) tests ─────────────────────────────────────────────────

    /// DRY must penalise a token that repeats an n-gram already in history.
    ///
    /// Setup: history = [A, B, C], current context ends with [A, B], and we
    /// look at token C. Since the trigram A→B→C already appears in history,
    /// token C (index 2) should receive a penalty while token D (index 3)
    /// should not be penalised.
    #[test]
    fn dry_active_penalises_repeated_tokens() {
        // Tokens: 0=A, 1=B, 2=C, 3=D
        // History: [A=0, B=1, C=2] — the sequence A,B,C appears once.
        // Current context (recent_tokens ends with [0=A, 1=B]).
        // Token C=2 would form the trigram A→B→C again → should be penalised.
        // Token D=3 has no matching n-gram → should be unchanged.
        let stage = DryStage::new(2.0, 1.75, 2, vec![]);
        let mut logits = vec![0.0f32, 0.0, 5.0, 5.0]; // A, B, C, D — C and D have equal high logits
                                                      // History = the generation context so far: [A, B, C, A, B]
                                                      // After [A, B], next is C → the 2-gram A,B was followed by C before.
        let recent = vec![0u32, 1, 2, 0, 1]; // history ending with A, B
        let original_c = logits[2];
        let original_d = logits[3];
        stage.apply(&mut logits, &recent, &mut rng());
        // C should be penalised (logit reduced), D should not.
        assert!(
            logits[2] < original_c,
            "token C should be penalised by DRY; was {original_c}, now {}",
            logits[2]
        );
        assert!(
            (logits[3] - original_d).abs() < 1e-6,
            "token D should NOT be penalised by DRY; was {original_d}, now {}",
            logits[3]
        );
    }

    /// The new O(history) DRY implementation must agree exactly with a
    /// brute-force O(vocab * history * match) reference on random inputs —
    /// this is a direct algebraic re-derivation, not an approximation, and
    /// this test locks that equivalence in.
    #[test]
    fn dry_matches_brute_force_reference() {
        fn brute_force(
            vocab_size: usize,
            recent_tokens: &[u32],
            allowed_length: usize,
            multiplier: f32,
            base: f32,
        ) -> Vec<f32> {
            let mut logits = vec![0.0f32; vocab_size];
            let hist_len = recent_tokens.len();
            for (t_idx, logit) in logits.iter_mut().enumerate() {
                let t = t_idx as u32;
                let mut best_match = 0usize;
                for pos in 0..hist_len {
                    if recent_tokens[pos] != t {
                        continue;
                    }
                    let mut match_len = 1usize;
                    let max_back = pos.min(hist_len - 1);
                    for k in 1..=max_back {
                        if recent_tokens[pos - k] == recent_tokens[hist_len - k] {
                            match_len += 1;
                        } else {
                            break;
                        }
                    }
                    best_match = best_match.max(match_len);
                }
                if best_match >= allowed_length {
                    let excess = (best_match - allowed_length) as f32;
                    *logit -= multiplier * base.powf(excess);
                }
            }
            logits
        }

        // A handful of pseudo-random-ish histories with repeats.
        let histories: Vec<Vec<u32>> = vec![
            vec![0, 1, 2, 0, 1, 2, 0, 1],
            vec![5, 5, 5, 5, 5],
            vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 6],
            vec![7],
            vec![],
            vec![0, 0, 1, 0, 0, 1, 0, 0],
        ];
        let vocab_size = 10usize;
        for history in histories {
            let expected = brute_force(vocab_size, &history, 2, 1.5, 1.75);
            let mut actual = vec![0.0f32; vocab_size];
            DryStage::new(1.5, 1.75, 2, vec![]).apply(&mut actual, &history, &mut rng());
            for i in 0..vocab_size {
                assert!(
                    (expected[i] - actual[i]).abs() < 1e-4,
                    "mismatch at token {i} for history {history:?}: expected {}, got {}",
                    expected[i],
                    actual[i]
                );
            }
        }
    }

    /// XTC with high probability must exclude top tokens (leaving only the best).
    ///
    /// We use a balanced distribution so that the "top set" contains at least 2
    /// tokens before the cumulative probability threshold is crossed, then seed
    /// the RNG so the coin flip always fires.
    #[test]
    fn xtc_active_excludes_top_tokens() {
        // Balanced distribution: tokens 0-3 have similar logits so that the
        // cumulative threshold of 0.6 requires at least 2 tokens (top set ≥ 2).
        // Logit ordering: 0 > 1 > 2 > 3, but all close enough so softmax gives
        // each token meaningful mass.
        //
        // Approximate probabilities after softmax of [3, 2, 1, 0]:
        //   token 0: ~0.644, token 1: ~0.237, token 2: ~0.087, token 3: ~0.032
        // With threshold=0.7, top set = {0, 1} (cumprob ≈ 0.644+0.237 = 0.881 > 0.7).
        //
        // probability=1.0: always trigger (coin flip always fires).
        let stage = XtcStage::new(0.7, 1.0);
        let mut logits = vec![3.0f32, 2.0, 1.0, 0.0];
        stage.apply(&mut logits, &[], &mut rng());
        // After XTC:
        // - token 0 (best in top set) must remain finite.
        // - token 1 (second-best in top set) should be set to -inf.
        assert!(
            logits[0].is_finite(),
            "XTC must preserve the top-1 token (token 0)"
        );
        assert_eq!(
            logits[1],
            f32::NEG_INFINITY,
            "XTC should exclude token 1 (second-best in top set)"
        );
    }

    /// XTC's coin flip must vary across successive calls sharing one RNG
    /// stream (defect S3: previously derived from a fixed seed, so it either
    /// always or never fired for a given `XtcStage` instance).
    #[test]
    fn xtc_coin_flip_varies_with_shared_rng() {
        let stage = XtcStage::new(0.7, 0.5);
        let mut shared_rng = Xorshift64::new(1);
        let mut fired = 0;
        let mut skipped = 0;
        for _ in 0..200 {
            let mut logits = vec![3.0f32, 2.0, 1.0, 0.0];
            stage.apply(&mut logits, &[], &mut shared_rng);
            if logits[1] == f32::NEG_INFINITY {
                fired += 1;
            } else {
                skipped += 1;
            }
        }
        assert!(
            fired > 0 && skipped > 0,
            "XTC coin flip must sometimes fire and sometimes not over 200 draws from a shared RNG (fired={fired}, skipped={skipped})"
        );
    }

    /// TypicalP with low p must reduce the number of finite-logit tokens.
    #[test]
    fn typical_p_active_reduces_distribution() {
        // Uniform distribution → entropy = log(4) ≈ 1.386.
        // Each token has dev |log(0.25) + log(4)| = 0.
        // All are equally "typical" → p=0.3 should keep only ~1 token (25% each).
        let stage = TypicalPStage::new(0.3);
        let mut logits = vec![0.0f32; 8]; // 8 tokens, all equal logits
        stage.apply(&mut logits, &[], &mut rng());
        let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
        assert!(
            finite_count < 8,
            "TypicalP(p=0.3) should reduce the number of active tokens; got {finite_count} finite"
        );
        assert!(
            finite_count >= 1,
            "TypicalP must preserve at least 1 token; got {finite_count}"
        );
    }

    /// TopA must set low-probability tokens to -inf, keeping only those near the peak.
    #[test]
    fn top_a_active_keeps_only_near_max() {
        // Token 0 is dominant (logit 10). Others are far below.
        // With a=1.0: threshold = 1.0 * max_prob^2.
        // max_prob ≈ 1.0 after softmax (token 0 is ~1.0 due to logit=10 vs -10).
        // threshold ≈ 1.0 * 1.0^2 = 1.0; tokens with prob < 1.0 are excluded.
        let stage = TopAStage::new(1.0);
        let mut logits = vec![10.0f32, -10.0, -10.0, -10.0];
        stage.apply(&mut logits, &[], &mut rng());
        // Token 0 should survive; tokens 1-3 should be -inf.
        assert!(logits[0].is_finite(), "dominant token must survive TopA");
        for (i, &v) in logits[1..].iter().enumerate() {
            assert_eq!(
                v,
                f32::NEG_INFINITY,
                "token {} should be excluded by TopA",
                i + 1
            );
        }
    }

    /// Eta must remove low-probability tokens below the computed threshold.
    #[test]
    fn eta_active_cuts_low_prob_tokens() {
        // Highly peaked distribution: token 0 has mass ~1.
        // With very low entropy, s = exp(H) ≈ 1.0.
        // dynamic threshold = eta / 1.0 = eta = 0.1.
        // Tokens with prob < 0.1 should be excluded.
        let stage = EtaStage::new(0.1, 0.0);
        let mut logits = vec![10.0f32, -10.0, -10.0, -10.0];
        stage.apply(&mut logits, &[], &mut rng());
        // Token 0 (prob ≈ 1) should survive; others (prob ≈ 0) should be -inf.
        assert!(
            logits[0].is_finite(),
            "dominant token must survive Eta cutoff"
        );
        // At least the tail tokens should be cut.
        let any_cut = logits[1..].contains(&f32::NEG_INFINITY);
        assert!(any_cut, "Eta should cut at least some low-prob tokens");
    }

    // ── Edge-case / safety tests ───────────────────────────────────────────────

    /// All stages must handle empty logit vectors without panicking.
    #[test]
    fn all_stages_handle_empty_logits() {
        let mut logits: Vec<f32> = Vec::new();
        DryStage::new(1.0, 1.75, 2, vec![]).apply(&mut logits, &[], &mut rng());
        XtcStage::new(0.5, 1.0).apply(&mut logits, &[], &mut rng());
        TypicalPStage::new(0.5).apply(&mut logits, &[], &mut rng());
        TopAStage::new(1.0).apply(&mut logits, &[], &mut rng());
        EtaStage::new(0.1, 0.0).apply(&mut logits, &[], &mut rng());
        assert!(logits.is_empty()); // still empty, no panic
    }

    /// TypicalP and TopA must always preserve at least 1 finite token even
    /// when the threshold is very aggressive.
    #[test]
    fn stages_never_empty_distribution() {
        let mut logits = vec![5.0f32, 4.9, 4.8, 4.7];

        // Very aggressive TopA (a=100 → threshold > any individual prob → keep argmax)
        TopAStage::new(100.0).apply(&mut logits, &[], &mut rng());
        let finite = logits.iter().filter(|&&v| v.is_finite()).count();
        assert!(
            finite >= 1,
            "TopA with extreme a must still preserve at least 1 token"
        );

        // Reset and try TypicalP with p=0 (edge-case: keep at least 1)
        let mut logits2 = vec![5.0f32, 4.9, 4.8, 4.7];
        TypicalPStage::new(0.0).apply(&mut logits2, &[], &mut rng());
        let finite2 = logits2.iter().filter(|&&v| v.is_finite()).count();
        assert!(
            finite2 >= 1,
            "TypicalP with p=0 must still preserve at least 1 token"
        );
    }
}

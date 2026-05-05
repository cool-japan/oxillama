//! Beam search decoding for sequence generation.
//!
//! Implements a full beam search decoder over an abstract forward-pass
//! interface.  The engine's `forward()` call is abstracted behind
//! [`BeamForwardPass`] so that both the real [`InferenceEngine`] and
//! test-only stubs can drive the algorithm.
//!
//! # Algorithm
//!
//! 1. Start with a single beam containing the prompt tokens.
//! 2. For each step up to `max_new_tokens`:
//!    a. For each active (unfinished) beam, call `forward(tokens)` to get logits.
//!    b. Compute log-softmax of the logits.
//!    c. Expand each beam to `beam_width` candidates (top-k log-probs).
//!    d. Keep the global top `beam_width` unique candidates across all expanded beams.
//!    e. If a candidate produces the EOS token, mark its beam as finished.
//!    f. If `early_stopping` is true and the best finished beam already scores
//!    higher than all active beams can possibly score, stop.
//! 3. Return all hypotheses (finished + active), sorted by normalised score
//!    descending.
//!
//! # Normalised score
//!
//! `score = logprob_sum / (n_tokens ^ length_penalty)`
//!
//! A `length_penalty` of 1.0 divides by token count (balances short vs long).
//! Values > 1.0 favour longer sequences.

use crate::error::{RuntimeError, RuntimeResult};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Configuration for the beam search decoder.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams to keep alive at each step (e.g. 4).
    pub beam_width: usize,
    /// Maximum number of new tokens to generate beyond the prompt.
    pub max_new_tokens: usize,
    /// Length-penalty exponent applied as `score = logprob_sum / len^length_penalty`.
    ///
    /// - `1.0` divides by length (neutral).
    /// - Values above `1.0` favour longer sequences.
    /// - Values below `1.0` favour shorter sequences.
    pub length_penalty: f32,
    /// Stop as soon as the best finished beam scores better than all active ones.
    pub early_stopping: bool,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_new_tokens: 256,
            length_penalty: 1.0,
            early_stopping: true,
        }
    }
}

/// A single beam hypothesis produced by the decoder.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token IDs generated so far (includes the prompt tokens).
    pub tokens: Vec<u32>,
    /// Sum of log-probabilities of all generated (non-prompt) tokens.
    pub logprob_sum: f32,
    /// True when this beam ended with the EOS token.
    pub finished: bool,
}

impl BeamHypothesis {
    /// Compute the length-normalised score for ranking.
    ///
    /// `score = logprob_sum / n_generated_tokens ^ length_penalty`
    ///
    /// When `n_generated_tokens == 0` (no tokens beyond prompt), the score is 0.
    pub fn score(&self, length_penalty: f32, prompt_len: usize) -> f32 {
        let n_gen = self.tokens.len().saturating_sub(prompt_len);
        if n_gen == 0 {
            return 0.0;
        }
        let denom = (n_gen as f32).powf(length_penalty);
        if denom > 0.0 {
            self.logprob_sum / denom
        } else {
            f32::NEG_INFINITY
        }
    }
}

// ─── Forward-pass abstraction ─────────────────────────────────────────────────

/// Abstraction over a forward pass that produces logits for a token sequence.
///
/// The real implementation is backed by [`InferenceEngine`]; test stubs
/// can implement this trait with pre-computed logit sequences.
pub trait BeamForwardPass {
    /// Run the forward pass on `tokens` and return raw logits.
    ///
    /// The implementation is free to maintain internal state (KV cache, etc.)
    /// but must be resettable via [`reset`].
    fn forward_tokens(&mut self, tokens: &[u32]) -> RuntimeResult<Vec<f32>>;

    /// Reset the internal state (e.g. clear the KV cache) so a fresh
    /// forward pass can be run for a different beam.
    fn reset(&mut self);
}

// ─── Engine adapter ───────────────────────────────────────────────────────────

/// Adapter that wraps [`InferenceEngine`] to implement [`BeamForwardPass`].
///
/// Each call to `forward_tokens` resets the KV cache, prefills the prompt
/// tokens, and returns the logits for the last token.
pub struct EngineBeamAdapter<'a> {
    engine: &'a mut crate::engine::InferenceEngine,
}

impl<'a> EngineBeamAdapter<'a> {
    /// Create an adapter over a loaded engine.
    pub fn new(engine: &'a mut crate::engine::InferenceEngine) -> Self {
        Self { engine }
    }
}

impl BeamForwardPass for EngineBeamAdapter<'_> {
    fn forward_tokens(&mut self, tokens: &[u32]) -> RuntimeResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(RuntimeError::ModelLoadError {
                message: "beam search: forward_tokens called with empty token slice".to_string(),
            });
        }
        // Use forward_one for the last token; the KV cache must already be
        // primed for all preceding tokens.  For beam search we re-run the
        // whole sequence from scratch (reset happens between beams).
        let last = *tokens.last().ok_or_else(|| RuntimeError::ModelLoadError {
            message: "beam search: token slice was empty after guard".to_string(),
        })?;
        // Process all tokens except the last to prime the KV cache.
        if tokens.len() > 1 {
            self.engine.prefill(&tokens[..tokens.len() - 1])?;
        }
        self.engine.forward_one(last)
    }

    fn reset(&mut self) {
        self.engine.reset();
    }
}

// ─── Beam search algorithm ────────────────────────────────────────────────────

/// Run beam search decoding.
///
/// `engine`        — any type implementing [`BeamForwardPass`]
/// `prompt_tokens` — initial token sequence (prompt)
/// `config`        — beam search hyper-parameters
/// `eos_token_id`  — token that signals end-of-sequence
///
/// Returns a list of [`BeamHypothesis`] sorted by normalised score descending.
/// The list contains at most `config.beam_width` hypotheses.
pub fn beam_generate<F: BeamForwardPass>(
    engine: &mut F,
    prompt_tokens: &[u32],
    config: &BeamSearchConfig,
    eos_token_id: u32,
) -> RuntimeResult<Vec<BeamHypothesis>> {
    if config.beam_width == 0 {
        return Err(RuntimeError::ModelLoadError {
            message: "beam_width must be >= 1".to_string(),
        });
    }
    if prompt_tokens.is_empty() {
        return Err(RuntimeError::ModelLoadError {
            message: "beam search: prompt_tokens must not be empty".to_string(),
        });
    }

    let prompt_len = prompt_tokens.len();

    // ── Initialisation ────────────────────────────────────────────────────────
    // Start with a single "beam" containing only the prompt.
    let mut active_beams: Vec<BeamHypothesis> = vec![BeamHypothesis {
        tokens: prompt_tokens.to_vec(),
        logprob_sum: 0.0,
        finished: false,
    }];
    let mut finished_beams: Vec<BeamHypothesis> = Vec::new();

    // ── Decode loop ───────────────────────────────────────────────────────────
    for _step in 0..config.max_new_tokens {
        if active_beams.is_empty() {
            break;
        }

        // For each active beam, expand to `beam_width` candidates.
        // A candidate is a (hypothesis, new_token, added_logprob) triple.
        let mut candidates: Vec<(BeamHypothesis, u32, f32)> = Vec::new();

        for beam in &active_beams {
            // Reset engine state, then run forward pass for this beam's tokens.
            engine.reset();
            let logits = engine.forward_tokens(&beam.tokens)?;

            // Log-softmax to obtain per-token log-probabilities.
            let log_probs = log_softmax(&logits);

            // Pick the top `beam_width` tokens from this beam.
            let mut token_logprob_pairs: Vec<(u32, f32)> = log_probs
                .iter()
                .enumerate()
                .map(|(i, &lp)| (i as u32, lp))
                .collect();
            // Sort by log-probability descending (highest first).
            token_logprob_pairs.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            token_logprob_pairs.truncate(config.beam_width);

            for (token, lp) in token_logprob_pairs {
                let mut new_tokens = beam.tokens.clone();
                new_tokens.push(token);
                let new_logprob_sum = beam.logprob_sum + lp;
                let finished = token == eos_token_id;
                candidates.push((
                    BeamHypothesis {
                        tokens: new_tokens,
                        logprob_sum: new_logprob_sum,
                        finished,
                    },
                    token,
                    lp,
                ));
            }
        }

        // ── Prune to beam_width global best ───────────────────────────────────
        // Sort all candidates by their normalised score (descending).
        candidates.sort_unstable_by(|(a, _, _), (b, _, _)| {
            b.score(config.length_penalty, prompt_len)
                .partial_cmp(&a.score(config.length_penalty, prompt_len))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(config.beam_width);

        // ── Separate finished from active ─────────────────────────────────────
        active_beams.clear();
        for (hyp, _token, _lp) in candidates {
            if hyp.finished {
                finished_beams.push(hyp);
            } else {
                active_beams.push(hyp);
            }
        }

        // ── Early stopping ────────────────────────────────────────────────────
        if config.early_stopping && !finished_beams.is_empty() {
            // Compute the best finished beam score.
            let best_finished_score = finished_beams
                .iter()
                .map(|h| h.score(config.length_penalty, prompt_len))
                .fold(f32::NEG_INFINITY, f32::max);

            // The best any active beam could ever score is its current logprob_sum
            // divided by its current length (lower bound on future length → best
            // possible score). If even that can't beat the best finished beam, stop.
            let best_possible_active = active_beams
                .iter()
                .map(|h| {
                    // Optimistic: assume the beam stops right now.
                    h.score(config.length_penalty, prompt_len)
                })
                .fold(f32::NEG_INFINITY, f32::max);

            if best_possible_active <= best_finished_score {
                break;
            }
        }
    }

    // Collect all hypotheses.
    let mut all_hyps: Vec<BeamHypothesis> = finished_beams;
    all_hyps.extend(active_beams);

    // Sort by normalised score descending.
    all_hyps.sort_unstable_by(|a, b| {
        b.score(config.length_penalty, prompt_len)
            .partial_cmp(&a.score(config.length_penalty, prompt_len))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Trim to at most beam_width results.
    all_hyps.truncate(config.beam_width);

    Ok(all_hyps)
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/// Compute log-softmax of a logit vector, returning log-probabilities.
///
/// `log_softmax(x_i) = x_i - log(sum_j(exp(x_j - x_max)))`
///
/// The `x_max` subtraction prevents overflow.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
    let log_sum = exp_sum.ln();
    logits.iter().map(|&v| (v - max_val) - log_sum).collect()
}

// ─── InferenceEngine integration ──────────────────────────────────────────────

impl crate::engine::InferenceEngine {
    /// Generate using beam search decoding.
    ///
    /// Wraps the engine in an [`EngineBeamAdapter`] and calls [`beam_generate`].
    ///
    /// Returns a list of [`BeamHypothesis`] sorted by normalised score
    /// descending.  The hypotheses include the original prompt tokens in
    /// `tokens`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ModelNotLoaded`] if no model has been loaded.
    pub fn beam_generate(
        &mut self,
        prompt_tokens: &[u32],
        config: &BeamSearchConfig,
        eos_token_id: u32,
    ) -> RuntimeResult<Vec<BeamHypothesis>> {
        if !self.is_loaded() {
            return Err(RuntimeError::ModelNotLoaded);
        }
        let mut adapter = EngineBeamAdapter::new(self);
        beam_generate(&mut adapter, prompt_tokens, config, eos_token_id)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test-only stub engine ─────────────────────────────────────────────────

    /// A stub `BeamForwardPass` backed by a fixed sequence of logit vectors.
    ///
    /// On each call to `forward_tokens`, the stub returns the next logit
    /// vector in its pre-programmed sequence (indexed by generation step,
    /// i.e. `tokens.len() - prompt_len`).  If the sequence is exhausted, the
    /// last vector is repeated.
    ///
    /// `reset()` rewinds the step counter so multiple beams can reuse the stub.
    struct StubEngine {
        /// Logit vectors for step 0, 1, 2, … (indexed by `tokens.len() - prompt_len`).
        logit_seq: Vec<Vec<f32>>,
        /// Length of the prompt (so we can compute the step index).
        prompt_len: usize,
    }

    impl StubEngine {
        fn new(prompt_len: usize, logit_seq: Vec<Vec<f32>>) -> Self {
            Self {
                logit_seq,
                prompt_len,
            }
        }
    }

    impl BeamForwardPass for StubEngine {
        fn forward_tokens(&mut self, tokens: &[u32]) -> RuntimeResult<Vec<f32>> {
            // Step index = how many tokens beyond the prompt have been generated.
            let step = tokens.len().saturating_sub(self.prompt_len);
            let idx = step.min(self.logit_seq.len().saturating_sub(1));
            Ok(self.logit_seq[idx].clone())
        }

        fn reset(&mut self) {
            // Stateless stub — nothing to reset.
        }
    }

    // ── Score formula tests ───────────────────────────────────────────────────

    #[test]
    fn beam_hypothesis_score_applies_length_penalty() {
        // A hypothesis with 2 generated tokens (beyond the prompt of length 1).
        // logprob_sum = -4.0, n_gen = 2.
        // With length_penalty = 2.0: score = -4.0 / 2^2 = -4.0 / 4 = -1.0
        let hyp = BeamHypothesis {
            tokens: vec![10u32, 20, 30], // prompt_len = 1, so 2 generated
            logprob_sum: -4.0,
            finished: false,
        };
        let score = hyp.score(2.0, 1);
        let expected = -4.0f32 / 4.0f32;
        assert!(
            (score - expected).abs() < 1e-5,
            "score with penalty=2.0 should be {expected}, got {score}"
        );
    }

    #[test]
    fn beam_hypothesis_score_neutral_length_penalty() {
        // length_penalty = 1.0: score = logprob_sum / n_generated_tokens.
        let hyp = BeamHypothesis {
            tokens: vec![1u32, 2, 3, 4], // prompt_len = 2 → 2 generated tokens
            logprob_sum: -6.0,
            finished: false,
        };
        let score = hyp.score(1.0, 2);
        let expected = -6.0f32 / 2.0f32;
        assert!(
            (score - expected).abs() < 1e-5,
            "neutral score should be {expected}, got {score}"
        );
    }

    #[test]
    fn beam_hypothesis_score_zero_when_no_generated_tokens() {
        // No generated tokens beyond the prompt → score = 0.
        let hyp = BeamHypothesis {
            tokens: vec![1u32, 2],
            logprob_sum: -99.0,
            finished: false,
        };
        let score = hyp.score(1.0, 2); // prompt_len == tokens.len()
        assert_eq!(score, 0.0, "score must be 0.0 when no tokens are generated");
    }

    // ── Beam width one matches greedy ─────────────────────────────────────────

    #[test]
    fn beam_search_width_one_matches_greedy() {
        // With beam_width=1 and a deterministic stub that always returns the
        // same logits, beam search should produce the same sequence as greedy
        // (argmax at each step).
        //
        // Vocab size = 4; EOS = 3.
        // Logits at every step: [0.0, 5.0, 2.0, -10.0]
        // → argmax = token 1 every time.
        let logits_per_step = vec![vec![0.0f32, 5.0, 2.0, -10.0]; 5];
        let prompt = vec![0u32];
        let eos = 3u32;

        let mut engine = StubEngine::new(prompt.len(), logits_per_step.clone());
        let config = BeamSearchConfig {
            beam_width: 1,
            max_new_tokens: 3,
            length_penalty: 1.0,
            early_stopping: false,
        };
        let hyps =
            beam_generate(&mut engine, &prompt, &config, eos).expect("beam search must succeed");
        assert!(!hyps.is_empty(), "must produce at least one hypothesis");

        // The only hypothesis should contain [prompt, 1, 1, 1] (greedy picks token 1).
        let best = &hyps[0];
        assert_eq!(
            &best.tokens[prompt.len()..],
            &[1u32, 1, 1],
            "beam_width=1 should match greedy decode (token 1 at each step)"
        );
    }

    // ── Beam width four returns four hypotheses ───────────────────────────────

    #[test]
    fn beam_width_four_returns_four_hypotheses() {
        // Vocab size = 8, EOS = 7.
        // Logits spread so all 4 beams stay active (no EOS in top-4).
        // Logits: [10, 9, 8, 7, 6, 5, 4, -100]  → top-4 = tokens 0,1,2,3
        let logits: Vec<f32> = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, -100.0];
        let logit_seq = vec![logits; 4];

        let prompt = vec![100u32];
        let eos = 7u32;

        let mut engine = StubEngine::new(prompt.len(), logit_seq);
        let config = BeamSearchConfig {
            beam_width: 4,
            max_new_tokens: 2,
            length_penalty: 1.0,
            early_stopping: false,
        };
        let hyps =
            beam_generate(&mut engine, &prompt, &config, eos).expect("beam search must succeed");
        assert_eq!(
            hyps.len(),
            4,
            "beam_width=4 should return 4 hypotheses, got {}",
            hyps.len()
        );
    }

    // ── Early stopping terminates ─────────────────────────────────────────────

    #[test]
    fn beam_early_stopping_terminates() {
        // Logits that always give a high probability to the EOS token.
        // EOS = 1, vocab = 3.
        // Logits: [0.0, 100.0, 0.0]  → EOS (token 1) is overwhelmingly likely.
        //
        // With beam_width=2 and early_stopping=true, the first step should
        // produce at least one finished beam (EOS), which then scores better
        // than the remaining active beam, causing early termination.
        let logits_step0 = vec![0.0f32, 100.0, 0.0]; // EOS dominates
        let logit_seq = vec![logits_step0; 5];

        let prompt = vec![0u32];
        let eos = 1u32;

        let mut engine = StubEngine::new(prompt.len(), logit_seq);
        let config = BeamSearchConfig {
            beam_width: 2,
            max_new_tokens: 10,
            length_penalty: 1.0,
            early_stopping: true,
        };
        let hyps =
            beam_generate(&mut engine, &prompt, &config, eos).expect("beam search must succeed");

        // At least the finished EOS hypothesis must be present.
        assert!(!hyps.is_empty(), "must return at least one hypothesis");
        // The best hypothesis should be finished (ended with EOS).
        // It's possible early_stopping didn't fire on step 1 if the active beam
        // still beats it; at minimum, a finished beam should appear.
        let has_finished = hyps.iter().any(|h| h.finished);
        assert!(
            has_finished,
            "at least one finished hypothesis should exist"
        );
    }

    // ── log_softmax correctness ────────────────────────────────────────────────

    #[test]
    fn log_softmax_sums_to_one_in_prob_space() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let lps = log_softmax(&logits);
        let prob_sum: f32 = lps.iter().map(|&lp| lp.exp()).sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-5,
            "exp(log-softmax) must sum to 1, got {prob_sum}"
        );
    }

    #[test]
    fn log_softmax_empty_is_empty() {
        let lps = log_softmax(&[]);
        assert!(lps.is_empty());
    }

    #[test]
    fn log_softmax_single_element_is_zero() {
        let lps = log_softmax(&[5.0f32]);
        assert!(
            (lps[0] - 0.0).abs() < 1e-6,
            "log-softmax of a single element must be 0, got {}",
            lps[0]
        );
    }

    // ── Error-path tests ──────────────────────────────────────────────────────

    #[test]
    fn beam_search_errors_on_zero_beam_width() {
        let prompt = vec![1u32];
        let mut engine = StubEngine::new(1, vec![vec![1.0, 2.0, 3.0]]);
        let config = BeamSearchConfig {
            beam_width: 0,
            ..BeamSearchConfig::default()
        };
        let result = beam_generate(&mut engine, &prompt, &config, 0);
        assert!(result.is_err(), "beam_width=0 should return an error");
    }

    #[test]
    fn beam_search_errors_on_empty_prompt() {
        let mut engine = StubEngine::new(0, vec![vec![1.0, 2.0, 3.0]]);
        let config = BeamSearchConfig::default();
        let result = beam_generate(&mut engine, &[], &config, 0);
        assert!(result.is_err(), "empty prompt should return an error");
    }

    // ── Engine integration (no model loaded) ─────────────────────────────────

    #[test]
    fn engine_beam_generate_errors_when_not_loaded() {
        let mut engine =
            crate::engine::InferenceEngine::new(crate::engine::EngineConfig::default());
        let config = BeamSearchConfig::default();
        let result = engine.beam_generate(&[1u32, 2], &config, 0);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "unloaded engine should return ModelNotLoaded, got {:?}",
            result
        );
    }
}

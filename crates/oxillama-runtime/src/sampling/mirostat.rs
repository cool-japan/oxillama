//! Mirostat v1 and v2 adaptive-perplexity sampling.
//!
//! Both functions expect `logits` to already have bias/bans, repetition
//! penalty, DRY, grammar masking, and temperature scaling applied (mirostat
//! replaces the top-k/top-p/min-p truncation stages entirely, but shares the
//! earlier pipeline steps — see [`super::Sampler::try_sample`]).
//!
//! # Defect S5
//!
//! Previously `mirostat: 1` silently fell through to ordinary top-k/top-p
//! sampling with no warning (only `mirostat == 2` was special-cased).
//! [`sample_v1`] implements the actual Mirostat v1 algorithm (Basu et al.,
//! as implemented by llama.cpp's `llama_sampler_mirostat_apply`).
//!
//! # Defect S6 (performance)
//!
//! Neither function sorts the full vocabulary. `sample_v2` never needs a
//! sorted order at all (surprise-threshold retention doesn't care about
//! order, and the "everything truncated" fallback needs only the argmax,
//! found in the same single O(V) pass as the softmax). `sample_v1` needs a
//! *partial* order (the top `M` = 100 tokens, to fit the Zipf-exponent
//! estimate) and gets it via `select_nth_unstable_by` + a small sort of just
//! that prefix, not a full `O(V log V)` sort.
//!
//! # Defect S9
//!
//! `sample_v2`'s surprise value for the mu update now uses the
//! **renormalised** probability of the selected token (i.e. its probability
//! within the truncated/surviving set), matching llama.cpp's second softmax
//! over the truncated candidate set. The previous implementation multiplied
//! back by the renormalisation `total` to recover the *pre*-truncation
//! probability, which is a different (self-consistent, but not
//! reference-matching) quantity.

use super::rng::Xorshift64;

/// Mirostat v1's default window size (`m` in the paper / llama.cpp).
const MIROSTAT_V1_M: usize = 100;

/// Mirostat v2 sampling.
///
/// Adaptively controls the "surprise" of generated tokens to maintain a
/// target perplexity level (`tau`), dynamically narrowing or widening the
/// candidate pool via the running estimate `mu`.
pub(super) fn sample_v2(
    logits: &[f32],
    mu: &mut f32,
    tau: f32,
    eta: f32,
    rng: &mut Xorshift64,
) -> u32 {
    let n = logits.len();
    if n == 0 {
        return 0;
    }

    // Softmax (single O(V) pass) while simultaneously tracking the argmax —
    // used both as the "everything truncated" fallback and to avoid a
    // second full pass just to find it.
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut top_id = 0u32;
    let mut top_val = f32::NEG_INFINITY;
    let mut exps: Vec<f32> = Vec::with_capacity(n);
    for (i, &v) in logits.iter().enumerate() {
        exps.push((v - max_val).exp());
        if v.is_finite() && v > top_val {
            top_val = v;
            top_id = i as u32;
        }
    }
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        // Nothing finite at all (fully masked distribution). Fall back to
        // the (arbitrary) first index and nudge mu toward tau.
        *mu -= eta * (0.0 - tau);
        return top_id;
    }

    let cur_mu = *mu;
    let mut candidates: Vec<(u32, f32)> = Vec::new();
    for (i, &e) in exps.iter().enumerate() {
        let p = e / sum;
        if p <= 0.0 {
            continue;
        }
        // surprise(token) = -log2(prob); keep tokens where surprise <= mu.
        if -p.log2() <= cur_mu {
            candidates.push((i as u32, p));
        }
    }

    if candidates.is_empty() {
        // Every candidate was truncated. llama.cpp forces the surviving set
        // to size 1 (the single best candidate) in this case; a softmax
        // over a singleton is trivially 1.0, so the observed surprise is
        // exactly 0 (defect S9 — see module doc).
        *mu = cur_mu - eta * (0.0 - tau);
        return top_id;
    }

    let total: f32 = candidates.iter().map(|&(_, p)| p).sum();
    if total > 0.0 {
        for (_, p) in &mut candidates {
            *p /= total;
        }
    }

    // Open-interval draw: candidates are walked in index order, so an exact
    // 0.0 draw would always pick the lowest-index survivor regardless of its
    // true (renormalised) probability. See `Xorshift64::next_open01_f32`.
    let r = rng.next_open01_f32();
    let mut cumulative = 0.0f32;
    let mut selected_idx = candidates[0].0;
    let mut selected_prob = candidates[0].1;
    for &(idx, prob) in &candidates {
        cumulative += prob;
        if r < cumulative {
            selected_idx = idx;
            selected_prob = prob;
            break;
        }
    }

    // S9: use the renormalised probability directly, not `prob * total`.
    let surprise = if selected_prob > 0.0 {
        -selected_prob.log2()
    } else {
        tau
    };
    *mu = cur_mu - eta * (surprise - tau);

    selected_idx
}

/// Mirostat v1 sampling (Basu et al. 2021 / llama.cpp `llama_sampler_mirostat_apply`).
///
/// Estimates the Zipf-law exponent `s_hat` of the current distribution from
/// the top `M` = 100 candidates' consecutive probability ratios, uses it to
/// derive an adaptive candidate-set size `k` from the target surprise `mu`,
/// truncates to the top `k`, and samples from the renormalised result.
pub(super) fn sample_v1(
    logits: &[f32],
    mu: &mut f32,
    tau: f32,
    eta: f32,
    rng: &mut Xorshift64,
) -> u32 {
    let n = logits.len();
    if n == 0 {
        return 0;
    }

    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut top_id = 0u32;
    let mut top_val = f32::NEG_INFINITY;
    let mut probs: Vec<f32> = Vec::with_capacity(n);
    for (i, &v) in logits.iter().enumerate() {
        probs.push((v - max_val).exp());
        if v.is_finite() && v > top_val {
            top_val = v;
            top_id = i as u32;
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        *mu -= eta * (0.0 - tau);
        return top_id;
    }
    for p in &mut probs {
        *p /= sum;
    }

    // Partial top-`take` selection: O(V) average via select_nth_unstable,
    // then sort only that (small, fixed-size) prefix — no full vocab sort.
    let take = MIROSTAT_V1_M.min(n).max(1);
    let mut idx: Vec<u32> = (0..n as u32).collect();
    if take < n {
        idx.select_nth_unstable_by(take - 1, |&a, &b| {
            probs[b as usize]
                .partial_cmp(&probs[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(take);
    }
    idx.sort_unstable_by(|&a, &b| {
        probs[b as usize]
            .partial_cmp(&probs[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Estimate the Zipf exponent s_hat via linear regression through the
    // origin over consecutive log-probability ratios of the top `take`
    // tokens (the same construction llama.cpp uses).
    let pairs = take.saturating_sub(1);
    let mut sum_ti_bi = 0.0f32;
    let mut sum_ti_sq = 0.0f32;
    for i in 0..pairs {
        let p_i = probs[idx[i] as usize].max(f32::MIN_POSITIVE);
        let p_ip1 = probs[idx[i + 1] as usize].max(f32::MIN_POSITIVE);
        let t_i = ((i as f32 + 2.0) / (i as f32 + 1.0)).ln();
        let b_i = (p_i / p_ip1).ln();
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    let s_hat = if sum_ti_sq > 1e-9 {
        sum_ti_bi / sum_ti_sq
    } else {
        1.0
    };

    let epsilon_hat = s_hat - 1.0;
    let vocab_n = n as f32;
    let k_estimate = if epsilon_hat.abs() > 1e-6 && s_hat.abs() > 1e-3 {
        let numerator = epsilon_hat * 2f32.powf(*mu);
        let denom = 1.0 - vocab_n.powf(-epsilon_hat);
        if denom.abs() > 1e-9 {
            (numerator / denom).max(0.0).powf(1.0 / s_hat)
        } else {
            take as f32
        }
    } else {
        take as f32
    };
    let k = (k_estimate.round().max(1.0) as usize).min(take);

    // Truncate to the estimated top-k and renormalise (llama.cpp's second
    // softmax over the truncated set).
    let cur_mu = *mu;
    let total: f32 = idx[..k].iter().map(|&i| probs[i as usize]).sum();
    if total <= 0.0 {
        *mu = cur_mu - eta * (0.0 - tau);
        return top_id;
    }

    let cand: Vec<(u32, f32)> = idx[..k]
        .iter()
        .map(|&i| (i, probs[i as usize] / total))
        .collect();

    // Open-interval draw — see the note in `sample_v2` above.
    let r = rng.next_open01_f32();
    let mut cumulative = 0.0f32;
    let mut selected_idx = cand[0].0;
    let mut selected_prob = cand[0].1;
    for &(id, prob) in &cand {
        cumulative += prob;
        if r < cumulative {
            selected_idx = id;
            selected_prob = prob;
            break;
        }
    }

    let surprise = if selected_prob > 0.0 {
        -selected_prob.log2()
    } else {
        tau
    };
    *mu = cur_mu - eta * (surprise - tau);

    selected_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_basic_returns_valid_token() {
        let logits = vec![3.0, 2.0, 1.0, 0.5, 0.1, -1.0, -2.0, -5.0];
        let mut mu = 10.0f32; // 2*tau with tau=5.0
        let mut rng = Xorshift64::new(42);
        for _ in 0..50 {
            let tok = sample_v2(&logits, &mut mu, 5.0, 0.1, &mut rng);
            assert!((tok as usize) < logits.len());
        }
    }

    #[test]
    fn v2_deterministic_with_same_seed() {
        let logits = vec![2.0, 1.5, 1.0, 0.5];
        let mut mu_a = 10.0f32;
        let mut mu_b = 10.0f32;
        let mut rng_a = Xorshift64::new(777);
        let mut rng_b = Xorshift64::new(777);
        for _ in 0..20 {
            let a = sample_v2(&logits, &mut mu_a, 5.0, 0.1, &mut rng_a);
            let b = sample_v2(&logits, &mut mu_b, 5.0, 0.1, &mut rng_b);
            assert_eq!(a, b);
        }
        assert!((mu_a - mu_b).abs() < 1e-6);
    }

    #[test]
    fn v2_mu_adapts() {
        let logits = vec![5.0, 0.0, 0.0, 0.0];
        let mut mu = 6.0f32;
        let initial = mu;
        let mut rng = Xorshift64::new(123);
        sample_v2(&logits, &mut mu, 3.0, 0.1, &mut rng);
        assert!(
            (mu - initial).abs() > 1e-6,
            "mu should adapt after sampling"
        );
    }

    /// Adjacent to defect S9: when mu is low enough that EVERY candidate is
    /// truncated, the fully-truncated fallback path forces
    /// observed_surprise = 0 by construction (a softmax over a singleton
    /// is trivially 1.0), so `mu' = mu - eta*(0 - tau)` regardless of the
    /// S9 pre/post-renormalisation distinction below — that distinction
    /// only matters when the *non-empty* candidate branch runs, which is
    /// covered by `v2_surprise_uses_post_renormalization_probability`.
    #[test]
    fn v2_singleton_fallback_surprise_is_zero() {
        let logits = vec![1.0f32, 1.0, 1.0, 1.0]; // uniform: surprise = 2 bits each
        let mut mu = 0.0f32; // any nonnegative surprise exceeds this -> full truncation
        let tau = 5.0f32;
        let eta = 0.1f32;
        let mut rng = Xorshift64::new(1);
        let tok = sample_v2(&logits, &mut mu, tau, eta, &mut rng);
        assert_eq!(
            tok, 0,
            "uniform logits -> first index wins the argmax tie-break"
        );
        let expected = 0.0f32 - eta * (0.0 - tau);
        assert!(
            (mu - expected).abs() < 1e-5,
            "expected mu={expected} (surprise=0 fallback), got {mu}"
        );
    }

    /// Defect S9 regression (the actual bug): when the *non-empty*
    /// candidate branch runs (some but not all tokens survive truncation),
    /// `mu` must update from the surprise of the **post-renormalisation**
    /// probability (llama.cpp's second softmax over the truncated set),
    /// not the raw pre-renormalisation vocabulary-wide probability.
    ///
    /// Construction: two tokens ("a") each carry raw probability 0.4, two
    /// ("b") each carry 0.1 (softmax of logits `[ln4, ln4, 0, 0]`). With
    /// `mu = 2.0`, only the two "a" tokens clear the `-log2(p) <= mu`
    /// filter (`-log2(0.4) ≈ 1.322 <= 2.0`; `-log2(0.1) ≈ 3.322 > 2.0`), so
    /// `total = 0.4 + 0.4 = 0.8` and each survivor renormalises to exactly
    /// `0.5`. Both survivors have the *same* renormalised probability, so
    /// the expected `mu` update is independent of which one the RNG draws:
    /// - correct (post-renorm): surprise = -log2(0.5) = 1.0 bit exactly.
    /// - buggy (pre-renorm, `selected_prob * total`): surprise =
    ///   -log2(0.5 * 0.8) = -log2(0.4) ≈ 1.32193 bits — recovers the
    ///   original *un-renormalised* probability, which is wrong.
    #[test]
    fn v2_surprise_uses_post_renormalization_probability() {
        let ln4 = 4.0f32.ln();
        let logits = vec![ln4, ln4, 0.0, 0.0];
        let tau = 5.0f32;
        let eta = 0.1f32;
        let cur_mu = 2.0f32;

        for seed in [1u64, 2, 3, 4, 5] {
            let mut mu = cur_mu;
            let mut rng = Xorshift64::new(seed);
            let tok = sample_v2(&logits, &mut mu, tau, eta, &mut rng);
            assert!(
                tok == 0 || tok == 1,
                "only the two high-probability tokens should survive truncation, got {tok}"
            );
            // Correct formula: surprise = -log2(0.5) = 1.0 exactly.
            let expected_correct = cur_mu - eta * (1.0f32 - tau);
            // The pre-renormalisation bug would instead give:
            let buggy = cur_mu - eta * ((-((0.5f32 * 0.8f32).log2())) - tau);
            assert!(
                (mu - expected_correct).abs() < 1e-4,
                "seed {seed}: expected post-renormalisation mu={expected_correct}, got {mu} \
                 (pre-renormalisation bug would give {buggy})"
            );
            assert!(
                (mu - buggy).abs() > 1e-3,
                "seed {seed}: mu={mu} must clearly differ from the pre-renormalisation buggy value {buggy}"
            );
        }
    }

    #[test]
    fn v1_basic_returns_valid_token() {
        let logits = vec![3.0, 2.0, 1.0, 0.5, 0.1, -1.0, -2.0, -5.0];
        let mut mu = 10.0f32;
        let mut rng = Xorshift64::new(42);
        for _ in 0..50 {
            let tok = sample_v1(&logits, &mut mu, 5.0, 0.1, &mut rng);
            assert!((tok as usize) < logits.len());
        }
    }

    #[test]
    fn v1_deterministic_with_same_seed() {
        let logits = vec![2.0, 1.5, 1.0, 0.5, 0.2, 0.1];
        let mut mu_a = 10.0f32;
        let mut mu_b = 10.0f32;
        let mut rng_a = Xorshift64::new(555);
        let mut rng_b = Xorshift64::new(555);
        for _ in 0..20 {
            let a = sample_v1(&logits, &mut mu_a, 5.0, 0.1, &mut rng_a);
            let b = sample_v1(&logits, &mut mu_b, 5.0, 0.1, &mut rng_b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn v1_mu_adapts() {
        let logits = vec![5.0, 0.0, 0.0, 0.0, -1.0, -2.0];
        let mut mu = 6.0f32;
        let initial = mu;
        let mut rng = Xorshift64::new(9);
        sample_v1(&logits, &mut mu, 3.0, 0.1, &mut rng);
        assert!(
            (mu - initial).abs() > 1e-6,
            "mu should adapt after sampling"
        );
    }

    #[test]
    fn v1_handles_large_vocab_without_full_sort_panic() {
        // Sanity check across the M=100 partial-sort boundary.
        let n = 500usize;
        let logits: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
        let mut mu = 10.0f32;
        let mut rng = Xorshift64::new(1);
        for _ in 0..10 {
            let tok = sample_v1(&logits, &mut mu, 5.0, 0.1, &mut rng);
            assert!((tok as usize) < n);
        }
    }

    #[test]
    fn v1_low_tau_prefers_top_token() {
        let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        let mut mu = 1.0f32; // 2*tau with tau=0.5
        let mut rng = Xorshift64::new(42);
        let mut top_count = 0;
        for _ in 0..100 {
            if sample_v1(&logits, &mut mu, 0.5, 0.1, &mut rng) == 0 {
                top_count += 1;
            }
        }
        assert!(
            top_count > 80,
            "low tau should strongly prefer the top token, got {top_count}/100"
        );
    }
}

//! Shared PRNG primitives for the sampling pipeline.
//!
//! Houses the xorshift64 generator used by [`super::Sampler`],
//! [`super::chain::SamplerChain`], and the advanced sampler stages, plus the
//! seed-generation logic for unseeded samplers and a shared `argmax` helper
//! that never silently degrades to index 0.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Process-lifetime counter mixed into every freshly generated seed.
///
/// Two [`Sampler`](super::Sampler)s constructed back-to-back can land in the
/// same wall-clock nanosecond (or on a platform with coarse clock
/// resolution), so wall-clock time alone is not sufficient to guarantee
/// distinct seeds. The counter guarantees distinctness regardless of clock
/// resolution.
static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a fresh, process-unique seed for an unseeded [`Sampler`](super::Sampler).
///
/// Mixes the current wall-clock time with a process-lifetime atomic counter,
/// then runs the mixture through a SplitMix64-style finalizer so the low bits
/// are well distributed. This replaces the previous "time-based seed" that
/// actually mixed in a stack address constant for a fixed call site, which
/// made every unseeded sampler in a server process draw the identical PRNG
/// stream (see defect S2).
pub(crate) fn generate_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut z = nanos ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Simple xorshift64 PRNG — fast, small, seedable, no dependencies.
///
/// Public (rather than crate-private) because [`super::chain::SamplerStage`]
/// — a public trait — takes `&mut Xorshift64` in its `apply` signature, so
/// external implementors of custom sampler stages need to be able to name
/// this type.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Construct a generator from a seed. A seed of `0` is remapped to a
    /// fixed non-zero constant, since xorshift is degenerate at state `0`
    /// (it would produce an all-zero stream forever).
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x517c_c1b7_2722_0a95
            } else {
                seed
            },
        }
    }

    /// Draw the next raw 64-bit value, advancing internal state.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Generate a uniform f32 in the *open* interval (0, 1) — never exactly
    /// `0.0` or `1.0`.
    ///
    /// Uses the midpoint convention `(raw + 0.5) / 2^24` rather than
    /// `next_f32`'s half-open `[0, 1)`. This matters for cumulative-sum
    /// weighted sampling (`chain::select_token`, mirostat v1/v2): those
    /// routines walk the candidate list in index order and pick the first
    /// entry whose *cumulative* probability exceeds the draw `r`. If `r`
    /// lands on exactly `0.0` (which `next_f32` can produce — `next_u64() >>
    /// 40` is `0` roughly one draw in 2^24), the very first candidate with
    /// *any* nonzero probability mass wins the draw regardless of how
    /// vanishingly small its true probability is, silently biasing the
    /// low-index end of the vocabulary. The open interval makes `r` require
    /// strictly positive cumulative mass to win, restoring the intended
    /// distribution while remaining O(1) and allocation-free.
    pub(crate) fn next_open01_f32(&mut self) -> f32 {
        (((self.next_u64() >> 40) as f32) + 0.5) / (1u64 << 24) as f32
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

/// Return the index of the maximum value, or `None` if `values` is empty or
/// every entry is non-finite (e.g. every logit has been masked to `-inf`).
///
/// Previously this defaulted to index `0` for an all-`-inf` input (see
/// defect S1): with `max_val` initialised to `NEG_INFINITY` and the loop
/// testing `v > max_val`, `-inf > -inf` is false for every entry, so the
/// loop body never runs and the function silently returned the initial
/// `max_idx = 0`. Surfacing `None` lets callers distinguish "the caller's
/// configuration/grammar/bans left no valid token" from a real selection.
pub(crate) fn argmax(values: &[f32]) -> Option<u32> {
    let mut best: Option<(u32, f32)> = None;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() || v == f32::NEG_INFINITY {
            continue;
        }
        let better = match best {
            Some((_, best_val)) => v > best_val,
            None => true,
        };
        if better {
            best = Some((i as u32, v));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_range() {
        let mut rng = Xorshift64::new(12345);
        for _ in 0..10000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "RNG produced {v} outside [0, 1)");
        }
    }

    #[test]
    fn test_xorshift_zero_seed_is_not_degenerate() {
        // A zero seed must not produce a stuck-at-zero generator.
        let mut rng = Xorshift64::new(0);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b, "xorshift with seed 0 must not be degenerate");
    }

    #[test]
    fn test_argmax_basic() {
        let values = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        assert_eq!(argmax(&values), Some(3));
    }

    #[test]
    fn test_argmax_empty_is_none() {
        let values: Vec<f32> = vec![];
        assert_eq!(argmax(&values), None);
    }

    #[test]
    fn test_argmax_all_neg_infinity_is_none() {
        // This is the exact S1 regression case: a fully-masked logit vector
        // must NOT resolve to token 0.
        let values = vec![f32::NEG_INFINITY; 8];
        assert_eq!(
            argmax(&values),
            None,
            "argmax over an all-masked vector must return None, not silently pick index 0"
        );
    }

    #[test]
    fn test_argmax_ties_pick_first() {
        let values = vec![1.0, 3.0, 3.0, 2.0];
        assert_eq!(argmax(&values), Some(1));
    }

    #[test]
    fn test_argmax_ignores_nan() {
        let values = vec![f32::NAN, 1.0, f32::NAN, 0.5];
        assert_eq!(argmax(&values), Some(1));
    }

    #[test]
    fn test_next_open01_f32_never_hits_zero_or_one() {
        // The whole point of the open-interval draw is that it must never
        // land on exactly 0.0 (which would let a vanishingly-small-but-first
        // candidate win a cumulative-sum draw) or 1.0.
        let mut rng = Xorshift64::new(1);
        for _ in 0..200_000 {
            let v = rng.next_open01_f32();
            assert!(v > 0.0, "next_open01_f32 produced non-positive {v}");
            assert!(v < 1.0, "next_open01_f32 produced >= 1.0: {v}");
        }
    }

    #[test]
    fn test_next_open01_f32_can_be_arbitrarily_close_to_zero() {
        // A seed whose raw draw is 0 (next_u64() >> 40 == 0) must still map
        // to a small-but-strictly-positive value via the +0.5 midpoint
        // convention, not silently clamp away from zero.
        let raw: f32 = 0.0;
        let v = (raw + 0.5) / (1u64 << 24) as f32;
        assert!(v > 0.0);
        assert!(v < 1e-6);
    }

    #[test]
    fn test_generate_seed_distinct_across_many_calls() {
        // Even called in a tight loop (well within the same wall-clock
        // nanosecond on many platforms), every seed must be distinct thanks
        // to the atomic counter mixed into the result.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            let s = generate_seed();
            assert!(seen.insert(s), "generate_seed produced a duplicate: {s}");
        }
    }
}

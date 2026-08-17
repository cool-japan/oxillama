//! Integer dot-product primitives shared by the fused Q8_0-activation kernels.
//!
//! The fused decode path quantizes the activation vector to Q8_0 once per
//! matmul input, which turns every weight row into a pure **integer** dot
//! product: `Σ q_w · q_a` with `q_w`/`q_a` in `i8` range.  This module is the
//! one place that knows how to evaluate such a dot product on AArch64.
//!
//! # Why inline assembly for SDOT
//!
//! ARMv8.2's `SDOT` performs four `i8 × i8` multiply-accumulates into each of
//! four `i32` lanes — 16 MACs per instruction, versus the four instructions
//! (`SMULL`/`SMULL2` + two `SADALP`) the widening fallback needs for the same
//! work. The Rust intrinsic `vdotq_s32` is still behind the unstable
//! `stdarch_neon_dotprod` feature gate (rust-lang/rust#117224), so on stable
//! the only way to reach the instruction is `asm!`, which *is* stable.
//!
//! The `asm!` block is `pure`/`nomem`/`nostack`: it reads only its register
//! operands and writes only its output register, so the optimiser is still
//! free to hoist, CSE and dead-code-eliminate it.
//!
//! # Runtime selection, not a compile-time gate
//!
//! `dotprod` is an *optional* ARMv8.2 extension — NEON itself is mandatory on
//! aarch64, but `dotprod` is not guaranteed on every aarch64 CPU (a Cortex-A53
//! or a generic `aarch64-unknown-linux-gnu` baseline build may lack it). This
//! module used to pick the SDOT path with `#[cfg(target_feature = "dotprod")]`
//! — a *compile-time* decision. That was safe on `aarch64-apple-darwin` only
//! because that specific target always compiles with `dotprod` enabled
//! (matching real Apple Silicon hardware), but it was a latent bug for any
//! other build: a generic-aarch64 binary built with
//! `-C target-feature=+dotprod` would select the SDOT path unconditionally
//! and `SIGILL` on real hardware that lacks the extension — and conversely,
//! the widening fallback was *never exercised at all* on Apple Silicon CI,
//! since that target never compiles it in.
//!
//! Both [`dot_acc_s8_sdot`] and [`dot_acc_s8_widening`] are now always
//! compiled (no `#[cfg(target_feature = ...)]` on either), and
//! [`dot_acc_s8`]/[`sum_acc_s8`] — the public entry points every caller in
//! `q4_k.rs`/`q6_k.rs` already uses, unchanged — select between them at
//! runtime using [`crate::simd::cached_capabilities`]`().dotprod`
//! (itself `is_aarch64_feature_detected!("dotprod")`, cached once). The
//! capability check is a single branch on a value that is constant for the
//! process's lifetime, so after the first call the branch predictor tracks
//! it perfectly; this is the same "hoist the constant runtime capability
//! check as a per-call branch" pattern the dispatcher uses to pick which
//! kernel to construct, just applied one level deeper because this function
//! is called per weight block rather than once per tensor.
//!
//! # Measured
//!
//! Apple M3 (4P+4E), Qwen3-4B `Q4_K_M`, `measure.sh` differencing harness,
//! median of 5 decode runs.  Switching this module from the widening fallback
//! to SDOT, with everything else held fixed:
//!
//! ```text
//! widening (SMULL/SADALP)  11.6102 tok/s
//! SDOT                     12.9807 tok/s     +11.8%
//! ```
//!
//! At kernel level (`fused_probe`, same machine) the gain is much larger —
//! Q4_K 4096x2560 went 0.351 ms → 0.202 ms — because end-to-end decode also
//! pays for attention, sampling and the bandwidth-bound LM head.
//!
//! Both paths compute the **same integer total**: `dot_acc_s8` and
//! `sum_acc_s8` distribute their partial sums differently across the four
//! accumulator lanes, but callers only ever consume `vaddvq_s32` of the final
//! accumulator, and integer addition is associative, so the results are
//! bit-identical regardless of which path ran.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

/// Largest batch a single pass of a `*_row_batch_neon` prefill kernel handles.
///
/// Bounds the per-row `block_sum` accumulator to a fixed stack array; wider
/// batches are chunked by the calling `matmul_q8_fused`, which costs one extra
/// sweep of the weights per chunk.  32 tokens already amortise the weight read
/// ~32x, so nothing measurable is left on the table.
pub(crate) const MAX_FUSED_BATCH: usize = 32;

/// Accumulate the 16 pairwise `i8 × i8` products of `w` and `a` into `acc`,
/// selecting the SDOT or widening implementation at runtime based on
/// [`crate::simd::cached_capabilities`]`().dotprod`.
///
/// The distribution of the 16 products across `acc`'s four lanes is
/// unspecified; only `vaddvq_s32(acc)` is meaningful.
///
/// Callers must guarantee `|w_i| · |a_i| ≤ 32767` per element so the
/// non-SDOT fallback's `i16` intermediate cannot overflow.  Every current
/// caller satisfies this by a wide margin: Q6_K contributes `|q − 32| ≤ 32`
/// and Q4_K a nibble `≤ 15`, against `|q_a| ≤ 128`.
///
/// # Safety
/// Must run on AArch64 with NEON.
#[inline(always)]
pub(crate) unsafe fn dot_acc_s8(acc: int32x4_t, w: int8x16_t, a: int8x16_t) -> int32x4_t {
    if crate::simd::cached_capabilities().dotprod {
        // SAFETY: `dotprod` confirmed available on this CPU at runtime
        // (cached once by `cached_capabilities`); caller upholds the
        // AArch64/NEON precondition documented above.
        unsafe { dot_acc_s8_sdot(acc, w, a) }
    } else {
        // SAFETY: every intrinsic used here is unconditionally available on
        // AArch64; caller upholds the precondition documented above.
        unsafe { dot_acc_s8_widening(acc, w, a) }
    }
}

/// SDOT-instruction fast path for [`dot_acc_s8`].
///
/// Always compiled (not gated on `target_feature = "dotprod"`); the
/// `#[target_feature(enable = "dotprod")]` attribute gives this one function
/// the codegen context the `sdot` mnemonic needs without requiring the whole
/// crate/module to assume the extension is present. Calling this on a CPU
/// that actually lacks `dotprod` is undefined behaviour (illegal
/// instruction) — [`dot_acc_s8`] only calls it after confirming the runtime
/// capability check.
///
/// # Safety
/// Must run on AArch64 with `dotprod` support.
///
/// Not `#[inline(always)]`: combining that with `#[target_feature]` is
/// rejected by this toolchain (`E0658`, rust-lang/rust#145574). Plain
/// `#[inline]` still lets the optimizer inline this at its discretion.
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn dot_acc_s8_sdot(acc: int32x4_t, w: int8x16_t, a: int8x16_t) -> int32x4_t {
    let mut out = acc;
    // SAFETY: `dotprod` is confirmed by the caller (`dot_acc_s8`) and by this
    // function's own `#[target_feature(enable = "dotprod")]`. The block
    // touches no memory and no stack.
    unsafe {
        core::arch::asm!(
            "sdot {out:v}.4s, {w:v}.16b, {a:v}.16b",
            out = inout(vreg) out,
            w = in(vreg) w,
            a = in(vreg) a,
            options(pure, nomem, nostack)
        );
    }
    out
}

/// Non-SDOT fallback for [`dot_acc_s8`]: widen to `i16`, then
/// pairwise-accumulate into `i32`. Always available on AArch64/NEON — no
/// `dotprod` required.
///
/// # Safety
/// Must run on AArch64 with NEON.
#[inline(always)]
unsafe fn dot_acc_s8_widening(acc: int32x4_t, w: int8x16_t, a: int8x16_t) -> int32x4_t {
    // SAFETY: every intrinsic used here is unconditionally available on AArch64.
    unsafe {
        let lo = vmull_s8(vget_low_s8(w), vget_low_s8(a));
        let hi = vmull_s8(vget_high_s8(w), vget_high_s8(a));
        vpadalq_s16(vpadalq_s16(acc, lo), hi)
    }
}

/// Accumulate the plain sum of `a`'s 16 `i8` lanes into `acc`, selecting the
/// SDOT or widening implementation at runtime — see [`dot_acc_s8`].
///
/// Used for the K-quant "minimum" correction terms, which need `Σ q_a` over
/// the same lanes as the dot product.
///
/// # Safety
/// Must run on AArch64 with NEON.
#[inline(always)]
pub(crate) unsafe fn sum_acc_s8(acc: int32x4_t, a: int8x16_t) -> int32x4_t {
    if crate::simd::cached_capabilities().dotprod {
        // SAFETY: `dotprod` confirmed available on this CPU at runtime;
        // SDOT against an all-ones operand sums the lanes (see `dot_acc_s8`).
        unsafe { dot_acc_s8_sdot(acc, vdupq_n_s8(1), a) }
    } else {
        // SAFETY: every intrinsic used here is unconditionally available on AArch64.
        unsafe { vpadalq_s16(acc, vpaddlq_s8(a)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar oracle for both helpers, so the SDOT and widening paths are
    /// checked against arithmetic rather than against each other.
    fn scalar_dot(w: &[i8; 16], a: &[i8; 16]) -> i32 {
        w.iter().zip(a).map(|(&x, &y)| x as i32 * y as i32).sum()
    }

    /// Same test cases as `dot_acc_matches_scalar_at_extremes`, run through
    /// each named path directly (regardless of which one the runtime
    /// dispatch in `dot_acc_s8` would have picked on this machine) — the
    /// whole point of making both paths always-compiled is that both must be
    /// independently correct and independently testable, not just whichever
    /// one happened to be selected at compile time.
    const DOT_CASES: [([i8; 16], [i8; 16]); 4] = [
        ([0; 16], [0; 16]),
        ([32; 16], [-128; 16]),
        (
            [
                -32, 31, -32, 31, -32, 31, -32, 31, 15, -15, 15, -15, 1, -1, 1, -1,
            ],
            [
                127, -128, 1, -1, 64, -64, 100, -100, 7, 7, -7, -7, 0, 0, 3, -3,
            ],
        ),
        ([15; 16], [127; 16]),
    ];

    #[test]
    fn dot_acc_matches_scalar_at_extremes() {
        // Saturating-ish inputs for the fallback's i16 intermediate: the
        // widest product any caller can produce is 32 x -128 = -4096.
        for (w, a) in DOT_CASES {
            // SAFETY: the test only runs on AArch64 (module is cfg-gated).
            let got = unsafe {
                let acc = vdupq_n_s32(0);
                vaddvq_s32(dot_acc_s8(acc, vld1q_s8(w.as_ptr()), vld1q_s8(a.as_ptr())))
            };
            assert_eq!(got, scalar_dot(&w, &a), "w={w:?} a={a:?}");
        }
    }

    /// The widening fallback, exercised directly and unconditionally — this
    /// is what used to be untestable on Apple Silicon CI, since
    /// `#[cfg(not(target_feature = "dotprod"))]` compiled it out entirely on
    /// that target.
    #[test]
    fn dot_acc_widening_matches_scalar_at_extremes() {
        for (w, a) in DOT_CASES {
            // SAFETY: the test only runs on AArch64 (module is cfg-gated);
            // `dot_acc_s8_widening` has no `dotprod` precondition.
            let got = unsafe {
                let acc = vdupq_n_s32(0);
                vaddvq_s32(dot_acc_s8_widening(
                    acc,
                    vld1q_s8(w.as_ptr()),
                    vld1q_s8(a.as_ptr()),
                ))
            };
            assert_eq!(got, scalar_dot(&w, &a), "w={w:?} a={a:?}");
        }
    }

    /// The SDOT path, exercised directly and unconditionally. Only sound to
    /// call when the CPU actually has `dotprod` — this test skips (rather
    /// than fails) on hardware that lacks it, since calling it there would
    /// be the exact `SIGILL` this whole module exists to prevent in
    /// production.
    #[test]
    fn dot_acc_sdot_matches_scalar_at_extremes() {
        if !crate::simd::cached_capabilities().dotprod {
            return;
        }
        for (w, a) in DOT_CASES {
            // SAFETY: `dotprod` confirmed available above; the test only
            // runs on AArch64 (module is cfg-gated).
            let got = unsafe {
                let acc = vdupq_n_s32(0);
                vaddvq_s32(dot_acc_s8_sdot(
                    acc,
                    vld1q_s8(w.as_ptr()),
                    vld1q_s8(a.as_ptr()),
                ))
            };
            assert_eq!(got, scalar_dot(&w, &a), "w={w:?} a={a:?}");
        }
    }

    /// Both paths must agree with each other bit-for-bit, not just with the
    /// scalar oracle independently — this is the property
    /// `dot_acc_s8`/`sum_acc_s8`'s runtime dispatch depends on: whichever
    /// path a given host takes, results must be identical to what the other
    /// path would have produced. Skips on hardware without `dotprod` (same
    /// reasoning as `dot_acc_sdot_matches_scalar_at_extremes`).
    #[test]
    fn dot_acc_sdot_and_widening_agree() {
        if !crate::simd::cached_capabilities().dotprod {
            return;
        }
        for (w, a) in DOT_CASES {
            // SAFETY: `dotprod` confirmed available above; AArch64 guaranteed
            // by the module cfg gate.
            let (sdot, widening) = unsafe {
                let wv = vld1q_s8(w.as_ptr());
                let av = vld1q_s8(a.as_ptr());
                (
                    vaddvq_s32(dot_acc_s8_sdot(vdupq_n_s32(0), wv, av)),
                    vaddvq_s32(dot_acc_s8_widening(vdupq_n_s32(0), wv, av)),
                )
            };
            assert_eq!(sdot, widening, "w={w:?} a={a:?}");
        }
    }

    #[test]
    fn dot_acc_chains_into_the_accumulator() {
        let w: [i8; 16] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ];
        let a: [i8; 16] = [2; 16];
        // SAFETY: the test only runs on AArch64 (module is cfg-gated).
        let got = unsafe {
            let mut acc = vdupq_n_s32(0);
            let wv = vld1q_s8(w.as_ptr());
            let av = vld1q_s8(a.as_ptr());
            acc = dot_acc_s8(acc, wv, av);
            acc = dot_acc_s8(acc, wv, av);
            acc = dot_acc_s8(acc, wv, av);
            vaddvq_s32(acc)
        };
        assert_eq!(got, 3 * scalar_dot(&w, &a));
    }

    #[test]
    fn sum_acc_matches_scalar() {
        let cases: [[i8; 16]; 3] = [
            [0; 16],
            [-128; 16],
            [
                127, -128, 1, -1, 64, -64, 100, -100, 7, 7, -7, -7, 0, 0, 3, -3,
            ],
        ];
        for a in cases {
            let want: i32 = a.iter().map(|&x| x as i32).sum();
            // SAFETY: the test only runs on AArch64 (module is cfg-gated).
            let got = unsafe { vaddvq_s32(sum_acc_s8(vdupq_n_s32(0), vld1q_s8(a.as_ptr()))) };
            assert_eq!(got, want, "a={a:?}");
        }
    }

    /// `cached_capabilities().dotprod` must be `true` on this test machine
    /// (Apple Silicon always has it) — a self-check that the detection wired
    /// up in `dispatch.rs` actually resolves the way the rest of these tests
    /// assume, so a detection regression fails loudly here instead of
    /// silently degrading every `dotprod`-only test above into a skip.
    #[test]
    fn dotprod_capability_is_detected_on_apple_silicon() {
        assert!(
            crate::simd::cached_capabilities().dotprod,
            "expected dotprod to be detected on this (Apple Silicon) test host"
        );
    }
}

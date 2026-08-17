//! Scaled-dot-product attention primitives shared by the decode and prefill
//! paths.
//!
//! # Why these are functions and not inline loops
//!
//! `llama::model` (one token) and `llama::batch` (a tile of tokens) both compute
//! `Q·K` and the value-weighted sum.  A batched prefill is only a safe default
//! if it is **bit-identical** to replaying the per-token loop — the CLI samples
//! at temperature, so one ULP is a different word — and floating-point addition
//! is not associative, so "the same maths" is not enough: the two paths must
//! accumulate in the same order.  Sharing one implementation makes that
//! structural rather than a thing to keep re-checking.
//!
//! # Why they are unrolled
//!
//! Attention was the last scalar, single-threaded stretch in the forward pass:
//! `rg 'rayon|par_iter|for_each_chunk' src/` used to hit only `qwen3/batch.rs`,
//! so GEMV ran on every core while attention — whose cost grows linearly with
//! context — ran on one, with a plain `for d in 0..head_dim` inner loop that
//! LLVM will not vectorise because each iteration depends on the previous
//! accumulator.
//!
//! [`dot_f32`] breaks that chain into four independent accumulators, which
//! rustc lowers to NEON `fmla` over `float32x4_t` (and to AVX on x86) without a
//! line of `unsafe` or a single intrinsic — this crate is `simd-neon`-enabled
//! but must also build for `wasm32`, so portable code that autovectorises is
//! worth more than an intrinsic that does not.  [`axpy_f32`] has no loop-carried
//! dependency to begin with and only needs the same 4-wide shape to line its
//! stores up.
//!
//! The head loop itself is parallelised by the callers via
//! [`oxillama_quant::parallel::for_each_chunk_init`]; one head's output is a
//! contiguous `head_dim` slice that no other head reads or writes, so handing
//! each to a worker changes no accumulation order either.

/// Dot product with four independent accumulators.
///
/// Sums `min(a.len(), b.len())` products.  The four partial sums are combined
/// as `(acc0 + acc2) + (acc1 + acc3)` and the `< 4` tail is folded in last, so
/// the result is a fixed function of the inputs — identical on every call site
/// and every thread, which is what the prefill/decode bit-identity rests on.
#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);

    let mut acc = [0.0f32; 4];
    let mut chunks = a.chunks_exact(4).zip(b.chunks_exact(4));
    for (x, y) in &mut chunks {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }

    let mut sum = (acc[0] + acc[2]) + (acc[1] + acc[3]);
    let tail = n - n % 4;
    for i in tail..n {
        sum += a[i] * b[i];
    }
    sum
}

/// `out[i] += w * v[i]` over `min(out.len(), v.len())` elements.
#[inline]
pub(crate) fn axpy_f32(out: &mut [f32], w: f32, v: &[f32]) {
    let n = out.len().min(v.len());
    let (out, v) = (&mut out[..n], &v[..n]);
    for (o, x) in out.chunks_exact_mut(4).zip(v.chunks_exact(4)) {
        o[0] += w * x[0];
        o[1] += w * x[1];
        o[2] += w * x[2];
        o[3] += w * x[3];
    }
    let tail = n - n % 4;
    for i in tail..n {
        out[i] += w * v[i];
    }
}

/// In-place softmax over a slice.
///
/// Subtracts the maximum first, so a row of large logits cannot overflow.
pub(crate) fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The unrolled dot must agree with the naive one to within fp32 noise.
    #[test]
    fn dot_matches_naive_sum() {
        for n in [0usize, 1, 3, 4, 5, 7, 8, 64, 128, 129] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.017 - 1.0).collect();
            let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.011).collect();
            let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let got = dot_f32(&a, &b);
            assert!(
                (naive - got).abs() <= 1e-4 * naive.abs().max(1.0),
                "n={n}: naive={naive}, unrolled={got}"
            );
        }
    }

    /// Two calls with the same inputs must return the *same bits*, which is
    /// what makes the batched prefill reproducible.
    #[test]
    fn dot_is_bitwise_reproducible() {
        let a: Vec<f32> = (0..131)
            .map(|i| ((i * 37) % 19) as f32 * 0.3 - 2.0)
            .collect();
        let b: Vec<f32> = (0..131)
            .map(|i| ((i * 11) % 23) as f32 * 0.7 - 5.0)
            .collect();
        assert_eq!(dot_f32(&a, &b).to_bits(), dot_f32(&a, &b).to_bits());
    }

    /// Mismatched lengths use the shorter one instead of panicking.
    #[test]
    fn dot_clamps_to_the_shorter_slice() {
        assert_eq!(dot_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0]), 3.0);
        assert_eq!(dot_f32(&[], &[1.0]), 0.0);
    }

    #[test]
    fn axpy_accumulates() {
        for n in [0usize, 1, 3, 4, 9, 64] {
            let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mut out = vec![1.0f32; n];
            axpy_f32(&mut out, 2.0, &v);
            for (i, o) in out.iter().enumerate() {
                assert!((o - (1.0 + 2.0 * i as f32)).abs() < 1e-6, "i={i}");
            }
        }
    }

    #[test]
    fn axpy_clamps_to_the_shorter_slice() {
        let mut out = vec![0.0f32; 2];
        axpy_f32(&mut out, 1.0, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn softmax_sums_to_one_and_preserves_order() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn softmax_survives_large_values_and_empty_input() {
        let mut x = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let mut empty: Vec<f32> = vec![];
        softmax_inplace(&mut empty);
    }
}

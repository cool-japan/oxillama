//! GELU (Gaussian Error Linear Unit) activation function.
//!
//! Used by GPT-style architectures such as StarCoder/GPT-BigCode.
//! This is the standard (non-gated) GELU with no gate projection,
//! unlike GeGLU/SwiGLU which multiply the activation by an up-projection.

/// GELU activation (tanh approximation).
///
/// `gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// This approximation closely matches the exact form
/// `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))` for all practical inputs.
#[inline]
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2 / π)
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Apply GELU in-place to every element of `x`.
///
/// After this call, `x[i] = gelu(x[i])` for all `i`.
pub fn gelu_inplace(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = gelu(*val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_zero() {
        // gelu(0) = 0 (identity at 0)
        assert!(
            gelu(0.0).abs() < 1e-6,
            "gelu(0) should be 0, got {}",
            gelu(0.0)
        );
    }

    #[test]
    fn test_gelu_positive_one() {
        // gelu(1.0) ≈ 0.8413
        let val = gelu(1.0);
        assert!(
            (val - 0.8413).abs() < 0.01,
            "gelu(1.0) = {val}, expected ~0.8413"
        );
    }

    #[test]
    fn test_gelu_negative_large() {
        // gelu(-10) ≈ 0 (large negative → near 0)
        let val = gelu(-10.0);
        assert!(val.abs() < 0.01, "gelu(-10) should be near 0, got {val}");
    }

    #[test]
    fn test_gelu_large_positive() {
        // gelu(10) ≈ 10 (large positive → approximately linear)
        let val = gelu(10.0);
        assert!(
            (val - 10.0).abs() < 0.01,
            "gelu(10) should be near 10, got {val}"
        );
    }

    #[test]
    fn test_gelu_inplace() {
        let mut x = vec![0.0f32, 1.0, -1.0, 2.0];
        gelu_inplace(&mut x);
        // x[0] = gelu(0) ≈ 0
        assert!(x[0].abs() < 1e-6);
        // x[1] = gelu(1) ≈ 0.8413
        assert!((x[1] - 0.8413).abs() < 0.01, "gelu(1.0) = {}", x[1]);
        // x[2] = gelu(-1) — slightly negative
        assert!(x[2] < 0.0, "gelu(-1) should be negative, got {}", x[2]);
        // x[3] = gelu(2) ≈ 1.9546
        assert!((x[3] - 1.9546).abs() < 0.05, "gelu(2.0) = {}", x[3]);
    }

    #[test]
    fn test_gelu_inplace_empty() {
        let mut x: Vec<f32> = vec![];
        gelu_inplace(&mut x); // Should not panic
    }
}

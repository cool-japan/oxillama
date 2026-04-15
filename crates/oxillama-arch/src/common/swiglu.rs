//! Gated activation functions (SwiGLU, GeGLU).
//!
//! SwiGLU (SiLU-gated) is used in LLaMA, Qwen3, Mistral, and most modern LLMs.
//! GeGLU (GELU-gated) is used in Gemma and some other architectures.

use super::gelu::gelu;

/// Apply SwiGLU activation: `silu(gate) * up`.
///
/// # Arguments
/// * `gate` - Gate projection output (will be modified in-place to hold result).
/// * `up` - Up projection output.
///
/// After this call, `gate[i] = silu(gate[i]) * up[i]`.
pub fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let sigmoid = 1.0 / (1.0 + (-*g).exp());
        *g = *g * sigmoid * u;
    }
}

/// Apply SiLU (Swish) activation in-place.
///
/// `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`
pub fn silu_inplace(x: &mut [f32]) {
    for val in x.iter_mut() {
        let sigmoid = 1.0 / (1.0 + (-*val).exp());
        *val *= sigmoid;
    }
}

/// Apply GeGLU activation: `gelu(gate) * up`.
///
/// GeGLU uses GELU (Gaussian Error Linear Unit) instead of SiLU as the
/// gating activation. Used in Gemma.
///
/// After this call, `gate[i] = gelu(gate[i]) * up[i]`.
pub fn geglu_inplace(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = gelu(*g) * u;
    }
}

/// Apply logit soft-capping in-place.
///
/// `soft_cap(x) = cap * tanh(x / cap)`
///
/// Used in Gemma 2 to prevent attention scores and final logits from
/// growing too large.
pub fn soft_cap_inplace(x: &mut [f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }
    let inv_cap = 1.0 / cap;
    for val in x.iter_mut() {
        *val = cap * (*val * inv_cap).tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_zero() {
        let mut x = vec![0.0];
        silu_inplace(&mut x);
        // silu(0) = 0 * 0.5 = 0.0
        assert!((x[0]).abs() < 1e-6);
    }

    #[test]
    fn test_swiglu() {
        let mut gate = vec![1.0, -1.0, 0.0];
        let up = vec![2.0, 3.0, 4.0];
        swiglu_inplace(&mut gate, &up);

        // silu(1.0) ≈ 0.7311, so gate[0] ≈ 0.7311 * 2.0 ≈ 1.4621
        assert!((gate[0] - 1.4621).abs() < 0.01);
        // silu(0.0) = 0.0, so gate[2] ≈ 0.0 * 4.0 = 0.0
        assert!((gate[2]).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_zero() {
        // gelu(0) = 0
        assert!(gelu(0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        // gelu(1.0) ≈ 0.8412
        let val = gelu(1.0);
        assert!(
            (val - 0.8412).abs() < 0.01,
            "gelu(1.0) = {val}, expected ~0.8412"
        );
    }

    #[test]
    fn test_geglu() {
        let mut gate = vec![1.0, 0.0];
        let up = vec![2.0, 3.0];
        geglu_inplace(&mut gate, &up);
        // gelu(1.0) ≈ 0.8412, so gate[0] ≈ 0.8412 * 2.0 ≈ 1.6824
        assert!(
            (gate[0] - 1.6824).abs() < 0.02,
            "geglu[0] = {}, expected ~1.6824",
            gate[0]
        );
        // gelu(0.0) = 0.0, so gate[1] = 0.0
        assert!(gate[1].abs() < 1e-6);
    }

    #[test]
    fn test_soft_cap() {
        let mut x = vec![100.0, -100.0, 0.0, 5.0];
        soft_cap_inplace(&mut x, 30.0);
        // tanh(100/30) ≈ 1.0, so x[0] ≈ 30.0
        assert!((x[0] - 30.0).abs() < 0.1, "soft_cap(100, 30) = {}", x[0]);
        // Symmetric
        assert!((x[1] + 30.0).abs() < 0.1);
        // Zero stays zero
        assert!(x[2].abs() < 1e-6);
        // Small values pass through approximately
        assert!((x[3] - 5.0).abs() < 0.5);
    }
}

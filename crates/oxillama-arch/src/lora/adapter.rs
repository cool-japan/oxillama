//! Public LoRA adapter trait, delta type, and target-module enum.
//!
//! This module defines the stable public API for composing LoRA adapters
//! through a [`LoraStack`](super::stack::LoraStack). Any adapter source
//! (GGUF-loaded, in-memory, synthetic) can implement [`LoraAdapterTrait`]
//! and be pushed onto the stack.

/// Identifies which linear projection a LoRA adapter targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetModule {
    /// Query projection (attn_q).
    QueryProj,
    /// Key projection (attn_k).
    KeyProj,
    /// Value projection (attn_v).
    ValueProj,
    /// Output projection (attn_o).
    OutputProj,
    /// Gate projection (ffn_gate).
    GateProj,
    /// Up projection (ffn_up).
    UpProj,
    /// Down projection (ffn_down).
    DownProj,
    /// Architecture-specific projection identified by a numeric tag.
    Custom(u32),
}

impl std::fmt::Display for TargetModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetModule::QueryProj => write!(f, "q_proj"),
            TargetModule::KeyProj => write!(f, "k_proj"),
            TargetModule::ValueProj => write!(f, "v_proj"),
            TargetModule::OutputProj => write!(f, "o_proj"),
            TargetModule::GateProj => write!(f, "gate_proj"),
            TargetModule::UpProj => write!(f, "up_proj"),
            TargetModule::DownProj => write!(f, "down_proj"),
            TargetModule::Custom(id) => write!(f, "custom_{id}"),
        }
    }
}

// ─── LoraDelta ────────────────────────────────────────────────────────────────

/// Raw A/B matrices for a LoRA delta at a single `(target, layer)`.
///
/// The weight delta is applied as:
///
/// ```text
/// Δ = (alpha / rank) * B @ A @ x
/// ```
///
/// where A has shape `[rank × in_dim]` (row-major) and B has shape
/// `[out_dim × rank]` (row-major).  The caller supplies the scalar
/// `scale = alpha / rank` when calling [`LoraDelta::apply`].
pub struct LoraDelta {
    /// Row-major A matrix: `[rank × in_dim]`.
    pub a: Vec<f32>,
    /// Row-major B matrix: `[out_dim × rank]`.
    pub b: Vec<f32>,
    /// Low-rank dimension.
    pub rank: usize,
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
}

impl LoraDelta {
    /// Construct a new [`LoraDelta`].
    pub fn new(a: Vec<f32>, b: Vec<f32>, rank: usize, in_dim: usize, out_dim: usize) -> Self {
        Self {
            a,
            b,
            rank,
            in_dim,
            out_dim,
        }
    }

    /// Compute `scale * B @ A @ x` and return the result vector of length `out_dim`.
    ///
    /// `x` must have length `in_dim`.
    pub fn apply(&self, x: &[f32], scale: f32) -> Vec<f32> {
        // Step 1: ax = A @ x  → [rank]
        let mut ax = vec![0.0f32; self.rank];
        for (r, ax_r) in ax.iter_mut().enumerate() {
            let row_start = r * self.in_dim;
            *ax_r = x
                .iter()
                .take(self.in_dim)
                .enumerate()
                .map(|(i, xi)| self.a[row_start + i] * xi)
                .sum();
        }

        // Step 2: bax = B @ ax  → [out_dim]
        let mut bax = vec![0.0f32; self.out_dim];
        for (o, bax_o) in bax.iter_mut().enumerate() {
            let row_start = o * self.rank;
            *bax_o = ax
                .iter()
                .enumerate()
                .map(|(r, axr)| self.b[row_start + r] * axr)
                .sum();
        }

        // Apply combined scale.
        bax.iter_mut().for_each(|v| *v *= scale);
        bax
    }
}

// ─── LoraAdapterTrait ────────────────────────────────────────────────────────

/// A single pluggable LoRA adapter.
///
/// Implement this trait to connect any adapter source (GGUF-loaded,
/// synthetic, merged) to [`LoraStack`](super::stack::LoraStack).
pub trait LoraAdapterTrait: Send + Sync {
    /// Low-rank dimension for this adapter.
    fn rank(&self) -> usize;
    /// Alpha value (scale numerator).
    fn alpha(&self) -> f32;
    /// The set of target modules this adapter covers.
    fn target_modules(&self) -> &[TargetModule];
    /// Returns the A/B delta for `(target, layer)`, or `None` when not covered.
    fn delta(&self, target: TargetModule, layer: usize) -> Option<&LoraDelta>;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TargetModule Display ──────────────────────────────────────────────────

    #[test]
    fn target_module_display_canonical_names() {
        assert_eq!(TargetModule::QueryProj.to_string(), "q_proj");
        assert_eq!(TargetModule::KeyProj.to_string(), "k_proj");
        assert_eq!(TargetModule::ValueProj.to_string(), "v_proj");
        assert_eq!(TargetModule::OutputProj.to_string(), "o_proj");
        assert_eq!(TargetModule::GateProj.to_string(), "gate_proj");
        assert_eq!(TargetModule::UpProj.to_string(), "up_proj");
        assert_eq!(TargetModule::DownProj.to_string(), "down_proj");
        assert_eq!(TargetModule::Custom(42).to_string(), "custom_42");
    }

    // ── LoraDelta::apply ──────────────────────────────────────────────────────

    /// Identity adapter (A = I, B = I, rank=4, in=out=4) should pass input through.
    #[test]
    fn lora_delta_identity_passes_input_through() {
        // A = identity [4×4], B = identity [4×4]
        let mut a = vec![0.0f32; 16];
        let mut b = vec![0.0f32; 16];
        for i in 0..4 {
            a[i * 4 + i] = 1.0;
            b[i * 4 + i] = 1.0;
        }
        let delta = LoraDelta::new(a, b, 4, 4, 4);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = delta.apply(&x, 1.0);
        for (r, xi) in result.iter().zip(x.iter()) {
            assert!((r - xi).abs() < 1e-5, "identity: r={r} xi={xi}");
        }
    }

    /// Zero A matrix gives zero output regardless of B.
    #[test]
    fn lora_delta_zero_a_gives_zero() {
        let a = vec![0.0f32; 8]; // rank=2, in=4
        let b = vec![1.0f32; 8]; // rank=2, out=4
        let delta = LoraDelta::new(a, b, 2, 4, 4);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = delta.apply(&x, 1.0);
        assert!(
            result.iter().all(|&v| v == 0.0),
            "zero A must give zero output"
        );
    }

    /// Scale factor is correctly applied.
    #[test]
    fn lora_delta_scale_multiplied() {
        // A = [[1.0, 0.0]], B = [[1.0], [0.0]]  (rank=1, in=2, out=2)
        let a = vec![1.0f32, 0.0]; // [1×2]
        let b = vec![1.0f32, 0.0]; // [2×1]
        let delta = LoraDelta::new(a, b, 1, 2, 2);
        let x = vec![2.0f32, 0.0];
        let base = delta.apply(&x, 1.0);
        let scaled = delta.apply(&x, 3.0);
        for (b, s) in base.iter().zip(scaled.iter()) {
            assert!((s - b * 3.0).abs() < 1e-5, "scaled must be 3x base");
        }
    }
}

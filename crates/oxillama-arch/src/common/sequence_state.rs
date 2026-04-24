//! Sequence-state abstractions for attention-based and SSM-based models.
//!
//! This module provides a [`SequenceState`] trait that generalises position tracking
//! for both transformer (attention) and state-space (Mamba-2) architectures. It is
//! **arch-internal** — the runtime crate consumes only [`KvCacheAccess`](crate::traits::KvCacheAccess).
//!
//! # Implementations
//! - [`AttentionSequenceState`] — wraps a token-position counter for
//!   attention-based decoders (LLaMA, Qwen3, etc.).
//! - [`Mamba2SequenceState`] — owns per-layer SSM hidden states for
//!   Mamba-2 selective-scan models.

// ─── Core trait ───────────────────────────────────────────────────────────────

/// Point-in-time capture of a sequence state for snapshot/resume.
///
/// This is arch-internal. The runtime crate wraps it in `SequenceStatePayload`
/// (in `snapshot.rs`) for actual wire serialization.
#[derive(Debug, Clone)]
pub enum SequenceStateSnapshot {
    /// Attention-based state: just the current position.
    Attention {
        /// The current token position.
        position: usize,
    },
    /// Mamba-2 SSM state: per-layer recurrent state + position.
    Mamba2 {
        /// Per-layer flattened SSM hidden states.
        ssm_states: Vec<Vec<f32>>,
        /// The current token position.
        position: usize,
    },
    /// Jamba hybrid: attention position + SSM states.
    Jamba {
        /// The shared token position (used by attention layers).
        attention_position: usize,
        /// Per-layer SSM states (only for SSM layers; attention layers have empty vecs).
        ssm_states: Vec<Vec<f32>>,
    },
}

/// Sequence-position tracking and state management for a running model.
///
/// Implemented by both attention-based models (wrapping a simple counter) and
/// state-space models (carrying a recurrent hidden state per SSM layer).
pub trait SequenceState {
    /// Reset the state to the beginning of a new sequence.
    ///
    /// For attention models this clears the position counter. For SSM models
    /// it also zeroes all per-layer recurrent state tensors.
    fn reset(&mut self);

    /// Return the current step / token position (0-indexed).
    fn step_position(&self) -> usize;

    /// Advance the position by one token.
    fn advance(&mut self);

    /// Return the maximum capacity (tokens) before the state wraps or errors.
    fn capacity(&self) -> usize;

    /// Snapshot the current sequence state for persistence.
    ///
    /// Default implementation captures only the step position (attention path).
    /// SSM implementations override this to also capture recurrent state.
    fn snapshot_payload(&self) -> SequenceStateSnapshot {
        SequenceStateSnapshot::Attention {
            position: self.step_position(),
        }
    }

    /// Restore state from a snapshot payload.
    ///
    /// Default implementation restores the step position for attention models.
    /// SSM implementations override this to also restore recurrent state.
    fn restore_from_snapshot_payload(&mut self, snap: &SequenceStateSnapshot) {
        if let SequenceStateSnapshot::Attention { position } = snap {
            self.reset();
            for _ in 0..*position {
                self.advance();
            }
        }
    }
}

// ─── Attention-based state ────────────────────────────────────────────────────

/// Sequence state for attention-based models.
///
/// Wraps a simple integer counter so that attention layers can query
/// the current token position for RoPE and KV-cache indexing.
///
/// # Example
///
/// ```
/// use oxillama_arch::common::sequence_state::{AttentionSequenceState, SequenceState};
///
/// let mut state = AttentionSequenceState::new(512);
/// state.advance();
/// state.advance();
/// assert_eq!(state.step_position(), 2);
/// state.reset();
/// assert_eq!(state.step_position(), 0);
/// ```
pub struct AttentionSequenceState {
    position: usize,
    max_capacity: usize,
}

impl AttentionSequenceState {
    /// Create a new [`AttentionSequenceState`] with the given maximum capacity.
    pub fn new(max_capacity: usize) -> Self {
        Self {
            position: 0,
            max_capacity,
        }
    }
}

impl SequenceState for AttentionSequenceState {
    fn reset(&mut self) {
        self.position = 0;
    }

    fn step_position(&self) -> usize {
        self.position
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn capacity(&self) -> usize {
        self.max_capacity
    }

    fn snapshot_payload(&self) -> SequenceStateSnapshot {
        SequenceStateSnapshot::Attention {
            position: self.position,
        }
    }

    fn restore_from_snapshot_payload(&mut self, snap: &SequenceStateSnapshot) {
        if let SequenceStateSnapshot::Attention { position } = snap {
            self.position = *position;
        }
    }
}

// ─── SSM per-layer state ──────────────────────────────────────────────────────

/// Recurrent hidden state for one Mamba-2 SSM layer.
///
/// Stores the `h` tensor of shape `[d_state × d_inner]` (row-major, where each
/// row corresponds to one inner dimension channel). This is updated by the
/// selective scan at every token step.
pub struct SsmLayerState {
    /// Flattened recurrent state: `[d_state × d_inner]` stored row-major.
    ///
    /// Access pattern: `h[s * d_inner + i]` = state dim `s`, inner dim `i`.
    pub h: Vec<f32>,
    /// Number of state dimensions (SSM state size).
    pub d_state: usize,
    /// Inner dimension (= d_model * expand).
    pub d_inner: usize,
}

impl SsmLayerState {
    /// Create a new [`SsmLayerState`] with all-zero initial state.
    pub fn new(d_state: usize, d_inner: usize) -> Self {
        Self {
            h: vec![0.0f32; d_state * d_inner],
            d_state,
            d_inner,
        }
    }

    /// Zero all elements of the hidden state.
    pub fn clear(&mut self) {
        self.h.fill(0.0);
    }
}

// ─── Mamba-2 sequence state ───────────────────────────────────────────────────

/// Sequence state for Mamba-2 (selective-scan SSM) models.
///
/// Owns one [`SsmLayerState`] per SSM layer. On [`reset`](SequenceState::reset)
/// all layer states are zeroed and the position counter resets to 0.
pub struct Mamba2SequenceState {
    /// Per-layer recurrent states (one per SSM block in the model).
    pub layers: Vec<SsmLayerState>,
    position: usize,
    max_capacity: usize,
}

impl Mamba2SequenceState {
    /// Create a new [`Mamba2SequenceState`] with all-zero initial hidden states.
    ///
    /// # Arguments
    /// * `n_layers`     – Number of SSM layers in the model.
    /// * `d_state`      – State dimension per layer.
    /// * `d_inner`      – Inner dimension per layer (d_model * expand).
    /// * `max_capacity` – Maximum tokens before wrapping.
    pub fn new(n_layers: usize, d_state: usize, d_inner: usize, max_capacity: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| SsmLayerState::new(d_state, d_inner))
            .collect();
        Self {
            layers,
            position: 0,
            max_capacity,
        }
    }
}

impl SequenceState for Mamba2SequenceState {
    fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    fn step_position(&self) -> usize {
        self.position
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn capacity(&self) -> usize {
        self.max_capacity
    }

    fn snapshot_payload(&self) -> SequenceStateSnapshot {
        let ssm_states = self.layers.iter().map(|l| l.h.clone()).collect();
        SequenceStateSnapshot::Mamba2 {
            ssm_states,
            position: self.position,
        }
    }

    fn restore_from_snapshot_payload(&mut self, snap: &SequenceStateSnapshot) {
        if let SequenceStateSnapshot::Mamba2 {
            ssm_states,
            position,
        } = snap
        {
            self.position = *position;
            for (layer, state) in self.layers.iter_mut().zip(ssm_states.iter()) {
                let copy_len = state.len().min(layer.h.len());
                layer.h[..copy_len].copy_from_slice(&state[..copy_len]);
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// (a) Advance 5 times → position = 5; reset → position = 0.
    #[test]
    fn attention_state_advance_and_reset() {
        let mut state = AttentionSequenceState::new(1024);
        for _ in 0..5 {
            state.advance();
        }
        assert_eq!(
            state.step_position(),
            5,
            "after 5 advances position must be 5"
        );
        state.reset();
        assert_eq!(state.step_position(), 0, "after reset position must be 0");
    }

    /// (b) Fill h with non-zero values, reset → all zeros.
    #[test]
    fn mamba_state_reset_clears_h() {
        let mut state = Mamba2SequenceState::new(3, 4, 8, 256);
        // Fill all layer states with non-zero values.
        for layer in &mut state.layers {
            layer.h.iter_mut().enumerate().for_each(|(i, v)| {
                *v = (i + 1) as f32;
            });
        }
        // Verify non-zero.
        for layer in &state.layers {
            assert!(
                layer.h.iter().any(|&v| v != 0.0),
                "h must be non-zero before reset"
            );
        }
        state.reset();
        // All layers must be zeroed.
        for (idx, layer) in state.layers.iter().enumerate() {
            assert!(
                layer.h.iter().all(|&v| v == 0.0),
                "layer {idx} h must be all-zero after reset"
            );
        }
        assert_eq!(state.step_position(), 0, "position must be 0 after reset");
    }

    /// (c) capacity() returns the value from construction.
    #[test]
    fn capacity_respected() {
        let attn = AttentionSequenceState::new(512);
        assert_eq!(attn.capacity(), 512, "AttentionSequenceState capacity");

        let mamba = Mamba2SequenceState::new(4, 8, 16, 1024);
        assert_eq!(mamba.capacity(), 1024, "Mamba2SequenceState capacity");
    }

    /// Extra: advance Mamba state and check position and layers are independent.
    #[test]
    fn mamba_state_advance_does_not_clear_h() {
        let mut state = Mamba2SequenceState::new(2, 2, 4, 64);
        // Set h to 1.0 in layer 0.
        state.layers[0].h.fill(1.0);
        state.advance();
        state.advance();
        assert_eq!(state.step_position(), 2);
        // h must be unchanged by advance.
        assert!(
            state.layers[0].h.iter().all(|&v| v == 1.0),
            "advance must not modify h"
        );
    }
}

//! Convolution shift state for Mamba-2.
//!
//! # Why this exists
//!
//! `llama.cpp` keeps **two** recurrent tensors per Mamba layer, not one:
//!
//! ```text
//! ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);   // [d_conv-1, conv_dim]
//! ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);   // [d_state, d_inner]
//! ```
//!
//! and `build_mamba2_layer` splices the conv state in front of the current
//! chunk before convolving:
//!
//! ```text
//! conv_x    = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);
//! last_conv = view(conv_x, d_conv - 1, ...);   // copied back into the cache
//! xBC       = ggml_ssm_conv(ctx0, conv_x, ssm_conv1d);
//! ```
//!
//! Without that state a decoder that steps one token at a time convolves a
//! length-1 signal against a width-`d_conv` kernel: every tap except the last
//! multiplies zero-padding, and the depthwise convolution collapses into a
//! scalar multiply.
//!
//! [`ConvRing`] is the Rust equivalent of `conv_states_all` for a single layer.
//!
//! # Ownership note
//!
//! This lives under `mamba2/` rather than in
//! [`crate::common::sequence_state`] because `SsmLayerState` there carries only
//! `h` / `d_state` / `d_inner`.  Adding a conv field to `SsmLayerState` (and a
//! matching field to `SequenceStateSnapshot::Mamba2`) is the proper home; until
//! that lands, the ring is held by the model.

use crate::error::{ArchError, ArchResult};

/// Per-layer depthwise-convolution shift register.
///
/// Holds the last `d_conv - 1` input vectors that entered the convolution, so
/// that position `t` sees positions `t-1 ..= t-(d_conv-1)` even when tokens are
/// fed one at a time.
///
/// Layout: `buf[slot * conv_dim + ch]`, where slot `0` is the **oldest**
/// retained position and slot `width - 1` is the immediately preceding token —
/// the same ordering as llama.cpp's `conv_x` prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvRing {
    buf: Vec<f32>,
    conv_dim: usize,
    width: usize,
}

impl ConvRing {
    /// Create a zeroed ring for a `conv_dim`-channel convolution of width
    /// `d_conv`.
    ///
    /// A `d_conv` of 0 or 1 yields a zero-width ring: a width-1 kernel is
    /// memoryless, so there is nothing to retain.
    pub fn new(conv_dim: usize, d_conv: usize) -> Self {
        let width = d_conv.saturating_sub(1);
        Self {
            buf: vec![0.0f32; width * conv_dim],
            conv_dim,
            width,
        }
    }

    /// Number of retained positions (`d_conv - 1`).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of channels per retained position.
    pub fn conv_dim(&self) -> usize {
        self.conv_dim
    }

    /// Zero the retained history, as at the start of a new sequence.
    pub fn clear(&mut self) {
        self.buf.fill(0.0);
    }

    /// Borrow the raw history buffer (`[width × conv_dim]`, oldest first).
    pub fn history(&self) -> &[f32] {
        &self.buf
    }

    /// Value of channel `ch` at retained slot `slot` (0 = oldest).
    ///
    /// Returns `0.0` for any out-of-range index, matching the zero-padding
    /// llama.cpp applies to a freshly cleared conv state.
    #[inline]
    pub fn get(&self, slot: usize, ch: usize) -> f32 {
        if slot >= self.width || ch >= self.conv_dim {
            return 0.0;
        }
        self.buf
            .get(slot * self.conv_dim + ch)
            .copied()
            .unwrap_or(0.0)
    }

    /// Replace the history with the last `width` positions of
    /// `[history ‖ x]`, where `x` is `[seq_len × conv_dim]` row-major.
    ///
    /// This is the Rust form of llama.cpp's "copy last (d_conv - 1) columns
    /// back into the state cache".
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] when `x` is not `seq_len * conv_dim` long.
    pub fn push_window(&mut self, x: &[f32], seq_len: usize) -> ArchResult<()> {
        let expected =
            seq_len
                .checked_mul(self.conv_dim)
                .ok_or_else(|| ArchError::InvalidShape {
                    name: "mamba2.conv_ring.input".to_string(),
                    expected: vec![usize::MAX],
                    got: vec![x.len()],
                })?;
        if x.len() != expected {
            return Err(ArchError::InvalidShape {
                name: "mamba2.conv_ring.input".to_string(),
                expected: vec![seq_len, self.conv_dim],
                got: vec![x.len()],
            });
        }
        if self.width == 0 {
            return Ok(());
        }

        let mut next = vec![0.0f32; self.width * self.conv_dim];
        for slot in 0..self.width {
            // Index into the concatenated `[history ‖ x]` array of length
            // `width + seq_len`, taking its last `width` positions.
            let idx = slot + seq_len;
            for ch in 0..self.conv_dim {
                next[slot * self.conv_dim + ch] = if idx < self.width {
                    self.buf[idx * self.conv_dim + ch]
                } else {
                    x[(idx - self.width) * self.conv_dim + ch]
                };
            }
        }
        self.buf = next;
        Ok(())
    }

    /// Copy the history out for a state snapshot.
    pub fn snapshot(&self) -> Vec<f32> {
        self.buf.clone()
    }

    /// Restore a history previously produced by [`Self::snapshot`].
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] when `data` has the wrong length.
    pub fn restore(&mut self, data: &[f32]) -> ArchResult<()> {
        if data.len() != self.buf.len() {
            return Err(ArchError::InvalidShape {
                name: "mamba2.conv_ring.snapshot".to_string(),
                expected: vec![self.width, self.conv_dim],
                got: vec![data.len()],
            });
        }
        self.buf.copy_from_slice(data);
        Ok(())
    }
}

/// All per-layer convolution rings for one Mamba-2 sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Mamba2ConvCache {
    /// One [`ConvRing`] per SSM block.
    pub layers: Vec<ConvRing>,
}

impl Mamba2ConvCache {
    /// Allocate `n_layer` zeroed rings.
    pub fn new(n_layer: usize, conv_dim: usize, d_conv: usize) -> Self {
        Self {
            layers: (0..n_layer)
                .map(|_| ConvRing::new(conv_dim, d_conv))
                .collect(),
        }
    }

    /// Zero every layer's history.
    pub fn clear(&mut self) {
        for ring in &mut self.layers {
            ring.clear();
        }
    }

    /// Capture every layer's history for a sequence-state snapshot.
    ///
    /// The counterpart of
    /// [`Mamba2SequenceState::snapshot_payload`](crate::common::sequence_state::Mamba2SequenceState),
    /// which today captures only `h`.  Once
    /// `SequenceStateSnapshot::Mamba2` grows a `conv_states` field this plugs
    /// straight into it.
    pub fn snapshot(&self) -> Vec<Vec<f32>> {
        self.layers.iter().map(|r| r.snapshot()).collect()
    }

    /// Restore histories previously produced by [`Self::snapshot`].
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] when the layer count or any per-layer
    /// history length disagrees with this cache.
    pub fn restore(&mut self, data: &[Vec<f32>]) -> ArchResult<()> {
        if data.len() != self.layers.len() {
            return Err(ArchError::InvalidShape {
                name: "mamba2.conv_cache.snapshot".to_string(),
                expected: vec![self.layers.len()],
                got: vec![data.len()],
            });
        }
        for (ring, saved) in self.layers.iter_mut().zip(data.iter()) {
            ring.restore(saved)?;
        }
        Ok(())
    }

    /// Mutable access to one layer's ring.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] when `layer_idx` is out of range.
    pub fn layer_mut(&mut self, layer_idx: usize) -> ArchResult<&mut ConvRing> {
        let n = self.layers.len();
        self.layers
            .get_mut(layer_idx)
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: format!("mamba2: conv ring for layer {layer_idx} (have {n} layers)"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_starts_zeroed_and_reports_geometry() {
        let ring = ConvRing::new(3, 4);
        assert_eq!(ring.width(), 3, "width = d_conv - 1");
        assert_eq!(ring.conv_dim(), 3);
        assert!(ring.history().iter().all(|&v| v == 0.0));
        assert_eq!(ring.get(0, 0), 0.0);
    }

    #[test]
    fn push_window_keeps_the_most_recent_positions() {
        // conv_dim = 2, d_conv = 3 -> width = 2.
        let mut ring = ConvRing::new(2, 3);
        // Feed three positions: [1,2], [3,4], [5,6].
        ring.push_window(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3)
            .expect("push_window");
        // Slot 0 (oldest retained) = position 1, slot 1 = position 2.
        assert_eq!(ring.get(0, 0), 3.0);
        assert_eq!(ring.get(0, 1), 4.0);
        assert_eq!(ring.get(1, 0), 5.0);
        assert_eq!(ring.get(1, 1), 6.0);
    }

    #[test]
    fn push_window_carries_old_history_when_chunk_is_short() {
        let mut ring = ConvRing::new(1, 4); // width = 3
        ring.push_window(&[1.0, 2.0, 3.0], 3).expect("seed");
        // One new position shifts the window by one.
        ring.push_window(&[4.0], 1).expect("step");
        assert_eq!(ring.get(0, 0), 2.0, "oldest slot after shift");
        assert_eq!(ring.get(1, 0), 3.0);
        assert_eq!(ring.get(2, 0), 4.0, "newest slot is the token just seen");
    }

    #[test]
    fn width_one_kernel_has_no_state() {
        let mut ring = ConvRing::new(4, 1);
        assert_eq!(ring.width(), 0);
        ring.push_window(&[1.0, 2.0, 3.0, 4.0], 1)
            .expect("width-0 push is a no-op");
        assert!(ring.history().is_empty());
    }

    #[test]
    fn push_window_rejects_wrong_length() {
        let mut ring = ConvRing::new(2, 3);
        let err = ring
            .push_window(&[1.0, 2.0, 3.0], 2)
            .expect_err("3 floats is not 2 positions of 2 channels");
        assert!(format!("{err}").contains("conv_ring.input"), "{err}");
    }

    #[test]
    fn clear_zeroes_history() {
        let mut ring = ConvRing::new(2, 3);
        ring.push_window(&[1.0, 2.0, 3.0, 4.0], 2).expect("push");
        ring.clear();
        assert!(ring.history().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn snapshot_round_trips() {
        let mut ring = ConvRing::new(2, 3);
        ring.push_window(&[1.0, 2.0, 3.0, 4.0], 2).expect("push");
        let snap = ring.snapshot();
        ring.clear();
        ring.restore(&snap).expect("restore");
        assert_eq!(ring.history(), snap.as_slice());
        assert!(ring.restore(&[0.0]).is_err(), "wrong length must error");
    }

    #[test]
    fn cache_snapshot_round_trips() {
        let mut cache = Mamba2ConvCache::new(2, 2, 3);
        cache
            .layer_mut(0)
            .expect("layer 0")
            .push_window(&[1.0, 2.0], 1)
            .expect("push");
        cache
            .layer_mut(1)
            .expect("layer 1")
            .push_window(&[3.0, 4.0], 1)
            .expect("push");

        let snap = cache.snapshot();
        cache.clear();
        assert!(cache.layers[0].history().iter().all(|&v| v == 0.0));

        cache.restore(&snap).expect("restore");
        assert_eq!(cache.layers[0].history(), snap[0].as_slice());
        assert_eq!(cache.layers[1].history(), snap[1].as_slice());

        assert!(
            cache.restore(&snap[..1]).is_err(),
            "a snapshot with the wrong layer count must error"
        );
    }

    #[test]
    fn cache_clear_and_bounds() {
        let mut cache = Mamba2ConvCache::new(2, 3, 4);
        cache
            .layer_mut(0)
            .expect("layer 0")
            .push_window(&[1.0; 3], 1)
            .expect("push");
        assert!(cache.layers[0].history().iter().any(|&v| v != 0.0));
        cache.clear();
        assert!(cache.layers[0].history().iter().all(|&v| v == 0.0));
        assert!(cache.layer_mut(9).is_err(), "out-of-range layer must error");
    }
}

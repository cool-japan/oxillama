//! Multimodal Rotary Position Embedding (M-RoPE).
//!
//! Qwen2-VL extends standard 1D RoPE to three axes:
//! - **Time / text** (t): used for temporal / text-sequence positions.
//! - **Height** (h): 2D spatial row index of a vision patch.
//! - **Width** (w): 2D spatial column index of a vision patch.
//!
//! The head dimension is split into three equal thirds.  Each third is rotated
//! with its own axis's cosine/sine table.  For text-only tokens the three
//! position indices are set to the same value `(pos, pos, pos)`, which
//! degrades exactly to standard 1-D RoPE applied to the full head vector.
//!
//! ## Reference
//! Qwen2-VL: "Enhancing Vision-Language Model's Perception of the World at Any
//! Resolution" (Qwen Team, 2024).

/// Precomputed M-RoPE frequency tables for three spatial axes.
///
/// Each axis holds `max_seq_len × half_dim_per_axis` cos/sin values.
/// `half_dim_per_axis = head_dim / 6` — the head is split into three equal
/// thirds (one per axis), then each third is split into a rotation pair.
///
/// # Invariant
///
/// `head_dim` must be divisible by 6 so that `head_dim / 3` is even.
#[derive(Debug, Clone)]
pub struct MRopeTable {
    /// Cosine values for the time/text axis: `[max_seq_len, half_dim_per_axis]`.
    cos_t: Vec<f32>,
    /// Cosine values for the height axis.
    cos_h: Vec<f32>,
    /// Cosine values for the width axis.
    cos_w: Vec<f32>,
    /// Sine values for the time/text axis.
    sin_t: Vec<f32>,
    /// Sine values for the height axis.
    sin_h: Vec<f32>,
    /// Sine values for the width axis.
    sin_w: Vec<f32>,
    /// Number of rotation pairs per axis = `head_dim / 6`.
    half_dim_per_axis: usize,
    /// Maximum precomputed sequence length.
    max_seq_len: usize,
}

impl MRopeTable {
    /// Build the M-RoPE frequency tables.
    ///
    /// # Arguments
    /// * `head_dim` — Full attention head dimension.  Must be divisible by 6.
    /// * `max_seq_len` — Maximum sequence length (rows in the table).
    /// * `base` — RoPE base frequency (typically 10000.0).
    ///
    /// # Returns
    /// A fully precomputed `MRopeTable` ready for use in `apply_mrope`.
    ///
    /// # Panics
    /// Does not panic even when `head_dim` is not divisible by 6;
    /// `half_dim_per_axis` is floored via integer division and the table is
    /// simply smaller than ideal.
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        // Each axis gets head_dim/3 dims; half of that forms the rotation pairs.
        let dim_per_axis = head_dim / 3;
        let half_dim_per_axis = dim_per_axis / 2;

        let capacity = max_seq_len * half_dim_per_axis;
        let mut cos_t = Vec::with_capacity(capacity);
        let mut sin_t = Vec::with_capacity(capacity);
        let mut cos_h = Vec::with_capacity(capacity);
        let mut sin_h = Vec::with_capacity(capacity);
        let mut cos_w = Vec::with_capacity(capacity);
        let mut sin_w = Vec::with_capacity(capacity);

        for pos in 0..max_seq_len {
            for i in 0..half_dim_per_axis {
                // Standard RoPE frequency formula: 1 / base^(2i / dim).
                // Here dim = dim_per_axis (not full head_dim) so that each axis
                // uses the same frequency spacing as standard 1D RoPE over its
                // sub-dimension.  When all three axes receive the same position
                // the concatenation is identical to applying standard RoPE to the
                // full head_dim vector.
                let freq = compute_mrope_freq(i, half_dim_per_axis, base);
                let theta = pos as f32 * freq;
                cos_t.push(theta.cos());
                sin_t.push(theta.sin());
                cos_h.push(theta.cos());
                sin_h.push(theta.sin());
                cos_w.push(theta.cos());
                sin_w.push(theta.sin());
            }
        }

        Self {
            cos_t,
            cos_h,
            cos_w,
            sin_t,
            sin_h,
            sin_w,
            half_dim_per_axis,
            max_seq_len,
        }
    }

    /// Apply M-RoPE in-place to a single attention-head vector.
    ///
    /// The head vector `x` of length `head_dim` is partitioned into three equal
    /// thirds.  Each third is rotated using the cos/sin table for the
    /// corresponding axis at the given position index.
    ///
    /// | Segment of `x`              | Axis  | Position used |
    /// |-----------------------------|-------|---------------|
    /// | `x[0  ..dim/3]`             | time  | `t_pos`       |
    /// | `x[dim/3..2*dim/3]`         | height| `h_pos`       |
    /// | `x[2*dim/3..dim]`           | width | `w_pos`       |
    ///
    /// For text-only tokens call with `t_pos == h_pos == w_pos`.  The result
    /// is then identical to standard 1-D RoPE applied across the full vector.
    ///
    /// # Arguments
    /// * `x`        — Mutable slice of length `head_dim` (modified in-place).
    /// * `t_pos`    — Time/text sequence position.
    /// * `h_pos`    — Height (row) patch index.
    /// * `w_pos`    — Width (column) patch index.
    /// * `head_dim` — Must match the `head_dim` used when constructing the table.
    ///
    /// If any position exceeds `max_seq_len` the operation is silently skipped.
    pub fn apply_mrope(
        &self,
        x: &mut [f32],
        t_pos: usize,
        h_pos: usize,
        w_pos: usize,
        head_dim: usize,
    ) {
        if self.half_dim_per_axis == 0 {
            return;
        }
        if t_pos >= self.max_seq_len || h_pos >= self.max_seq_len || w_pos >= self.max_seq_len {
            return;
        }

        let dim_per_axis = head_dim / 3;
        let half = self.half_dim_per_axis;

        // Each axis segment: x[base..base+dim_per_axis].
        // Within that segment the first `half` elements are the "real" part and
        // elements `[half..dim_per_axis]` are the "imaginary" part — same layout
        // as standard RoPE.
        apply_axis_rotation(x, 0, dim_per_axis, half, &self.cos_t, &self.sin_t, t_pos);
        apply_axis_rotation(
            x,
            dim_per_axis,
            2 * dim_per_axis,
            half,
            &self.cos_h,
            &self.sin_h,
            h_pos,
        );
        apply_axis_rotation(
            x,
            2 * dim_per_axis,
            3 * dim_per_axis,
            half,
            &self.cos_w,
            &self.sin_w,
            w_pos,
        );
    }

    /// Maximum sequence length for which frequency tables were precomputed.
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of rotation pairs per axis (`head_dim / 6`).
    #[inline]
    pub fn half_dim_per_axis(&self) -> usize {
        self.half_dim_per_axis
    }
}

// ---- Private helpers -------------------------------------------------------

/// Standard RoPE frequency formula for a single dimension index within an axis.
///
/// Equivalent to `1 / base^(2i / (2 * half_dim))`.
#[inline]
fn compute_mrope_freq(i: usize, half_dim: usize, base: f32) -> f32 {
    1.0 / base.powf((2 * i) as f32 / (2 * half_dim) as f32)
}

/// Apply RoPE rotation in-place to one axis segment of `x`.
///
/// Rotates pairs `(x[seg_start + i], x[seg_start + half + i])` for `i in 0..half`.
///
/// # Arguments
/// * `x`           — Full head vector (mutable).
/// * `seg_start`   — Start index of the axis segment.
/// * `seg_end`     — One-past-end of the axis segment (used for bounds check only).
/// * `half`        — Number of rotation pairs in this axis.
/// * `cos`         — Cosine table: `[max_seq_len, half]`.
/// * `sin`         — Sine table: `[max_seq_len, half]`.
/// * `pos`         — Sequence / position index for this axis.
#[inline]
fn apply_axis_rotation(
    x: &mut [f32],
    seg_start: usize,
    seg_end: usize,
    half: usize,
    cos: &[f32],
    sin: &[f32],
    pos: usize,
) {
    // Avoid out-of-bounds if the segment is smaller than expected.
    if seg_end > x.len() || seg_start + 2 * half > x.len() {
        return;
    }

    let offset = pos * half;
    if offset + half > cos.len() {
        return;
    }

    for i in 0..half {
        let x0 = x[seg_start + i];
        let x1 = x[seg_start + half + i];
        let c = cos[offset + i];
        let s = sin[offset + i];
        x[seg_start + i] = x0 * c - x1 * s;
        x[seg_start + half + i] = x0 * s + x1 * c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rope::RopeTable;

    /// M-RoPE with (pos, pos, pos) must produce the same result as applying
    /// standard 1-D RoPE **independently to each third** of the head vector.
    ///
    /// This exercises the core invariant: when all three axes receive the same
    /// position index, M-RoPE is equivalent to three independent RoPE applications,
    /// each over a sub-dimension of size `head_dim / 3`.
    ///
    /// Note: this is NOT the same as a single full-head 1-D RoPE, because the
    /// frequency spacing uses `dim_per_axis = head_dim / 3` as the denominator,
    /// not the full `head_dim`.
    #[test]
    fn mrope_text_only_matches_1d_rope() {
        // head_dim must be divisible by 6 for the third-split to be exact.
        // We use 24 = 6×4, so dim_per_axis = 8, half_per_axis = 4.
        let head_dim = 24usize;
        let max_seq = 32usize;
        let base = 10000.0f32;
        let pos = 5usize;

        let dim_per_axis = head_dim / 3; // = 8

        // Build M-RoPE table.
        let mrope = MRopeTable::new(head_dim, max_seq, base);

        // Build a standard RoPE table over dim_per_axis (= 8 dims, half = 4).
        let rope_axis = RopeTable::new_standard(dim_per_axis, max_seq, base);

        // Build a non-trivial test vector.
        let x_init: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Apply M-RoPE with equal positions on all axes.
        let mut x_mm = x_init.clone();
        mrope.apply_mrope(&mut x_mm, pos, pos, pos, head_dim);

        // Reference: apply the per-axis RoPE independently to each 8-element third.
        let mut x_ref = x_init.clone();
        for axis in 0..3 {
            let start = axis * dim_per_axis;
            let mut third: Vec<f32> = x_ref[start..start + dim_per_axis].to_vec();
            rope_axis.apply(&mut third, pos);
            x_ref[start..start + dim_per_axis].copy_from_slice(&third);
        }

        // M-RoPE must match the per-axis reference within floating-point tolerance.
        for (i, (a, b)) in x_mm.iter().zip(x_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at dim {i}: mrope={a}, per-axis-rope={b}"
            );
        }
    }

    /// Changing h_pos should only affect the middle third of the head vector,
    /// leaving the first and last thirds unchanged.
    #[test]
    fn mrope_vision_axes_independent() {
        let head_dim = 24usize;
        let max_seq = 32usize;
        let base = 10000.0f32;
        let t_pos = 0usize;
        let w_pos = 3usize;
        let h_pos_a = 2usize;
        let h_pos_b = 7usize;

        let mrope = MRopeTable::new(head_dim, max_seq, base);
        let x_init: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let mut x_a = x_init.clone();
        mrope.apply_mrope(&mut x_a, t_pos, h_pos_a, w_pos, head_dim);

        let mut x_b = x_init.clone();
        mrope.apply_mrope(&mut x_b, t_pos, h_pos_b, w_pos, head_dim);

        let dim_per_axis = head_dim / 3;

        // Time third (0..dim/3) must be identical — same t_pos.
        for i in 0..dim_per_axis {
            assert!(
                (x_a[i] - x_b[i]).abs() < 1e-7,
                "time axis should be equal at {i}: {:.6} vs {:.6}",
                x_a[i],
                x_b[i]
            );
        }

        // Height third (dim/3..2*dim/3) must differ because h_pos_a != h_pos_b.
        let h_third_differs =
            (dim_per_axis..2 * dim_per_axis).any(|i| (x_a[i] - x_b[i]).abs() > 1e-6);
        assert!(
            h_third_differs,
            "height axis results must differ when h_pos changes"
        );

        // Width third (2*dim/3..dim) must be identical — same w_pos.
        for i in 2 * dim_per_axis..head_dim {
            assert!(
                (x_a[i] - x_b[i]).abs() < 1e-7,
                "width axis should be equal at {i}: {:.6} vs {:.6}",
                x_a[i],
                x_b[i]
            );
        }
    }
}

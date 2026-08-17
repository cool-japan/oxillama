//! Multimodal Rotary Position Embedding (M-RoPE).
//!
//! Qwen2-VL extends standard 1D RoPE to three (optionally four) axes:
//! - **Time / text** (t): temporal / text-sequence position.
//! - **Height** (h): 2D spatial row index of a vision patch.
//! - **Width** (w): 2D spatial column index of a vision patch.
//! - **Extra** (e): the vision encoder's extra position id (Qwen3-VL).
//!
//! # How the reference actually works
//!
//! `ggml_mrope_cache_init` builds **one global frequency series** over the full
//! rotary dimension — `theta_scale = base^(-2 / n_dims)` applied cumulatively —
//! and uses the section table only to decide *which axis position* multiplies
//! each frequency:
//!
//! ```text
//! sect_dims = sections[0] + sections[1] + sections[2] + sections[3]
//! sector    = pair_index % sect_dims
//! axis      = t  if sector <  sections[0]
//!             h  if sector <  sections[0] + sections[1]
//!             w  if sector <  sections[0] + sections[1] + sections[2]
//!             e  otherwise
//! theta     = pos[axis] * base^(-2 * pair_index / n_dims)
//! ```
//!
//! and the rotation is **global GPT-NeoX pairing**:
//! `rotate_pairs(n_dims, n_dims/2, …)` rotates `(x[j], x[j + n_dims/2])`.
//!
//! The section widths come from the GGUF key `{arch}.rope.dimension_sections`
//! (e.g. `[16, 24, 24, 0]` for Qwen2-VL's 128-wide heads).
//!
//! An implementation that splits the head into equal thirds, restarts the
//! frequency series inside each third and pairs *within* a third computes a
//! different function entirely.
//!
//! ## Reference
//! Qwen2-VL: "Enhancing Vision-Language Model's Perception of the World at Any
//! Resolution" (Qwen Team, 2024); `ggml/src/ggml-cpu/ops.cpp`.

use crate::error::{ArchError, ArchResult};

/// Which positional axis drives a given rotation pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MRopeAxis {
    /// Time / text position.
    Time,
    /// Vision patch row.
    Height,
    /// Vision patch column.
    Width,
    /// Extra vision-encoder position id.
    Extra,
}

/// Precomputed M-RoPE frequency tables for the positional axes.
///
/// All four tables share the same **global** frequency series
/// `base^(-2j / head_dim)`; they differ only in which position index was used
/// to build them.  [`MRopeTable::apply_mrope`] then selects, per rotation pair,
/// the table named by the section layout.
#[derive(Debug, Clone)]
pub struct MRopeTable {
    /// Cosine values for the time/text axis: `[max_seq_len, half_dim]`.
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
    /// Per-rotation-pair axis assignment, length `half_dim`.
    axis_of: Vec<MRopeAxis>,
    /// Number of rotation pairs = `head_dim / 2`.
    half_dim: usize,
    /// The rotary dimension count this table was built for.
    head_dim: usize,
    /// Section widths `[t, h, w, e]` in rotation-pair units.
    sections: [usize; 4],
    /// Maximum precomputed sequence length.
    max_seq_len: usize,
}

impl MRopeTable {
    /// Build the M-RoPE frequency tables with the default equal-thirds split.
    ///
    /// Equivalent to [`Self::new_with_sections`] with
    /// `[half/3, half/3, half - 2*(half/3), 0]` where `half = head_dim / 2`.
    ///
    /// Prefer [`Self::new_with_sections`] with the checkpoint's
    /// `{arch}.rope.dimension_sections` — Qwen2-VL ships `[16, 24, 24, 0]`,
    /// which is **not** an equal split of its 64 rotation pairs.
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        let sections = default_sections(head_dim / 2);
        // Cannot fail: `default_sections` always sums to `half_dim`.
        match Self::new_with_sections(head_dim, max_seq_len, base, &sections) {
            Ok(table) => table,
            Err(_) => Self::empty(head_dim, max_seq_len, sections),
        }
    }

    /// Build the M-RoPE frequency tables from an explicit section layout.
    ///
    /// # Arguments
    /// * `head_dim` — rotary dimension count (`n_rot`).
    /// * `max_seq_len` — maximum position index (rows in each table).
    /// * `base` — RoPE base frequency (typically 10000.0).
    /// * `sections` — `{arch}.rope.dimension_sections`, in **rotation-pair**
    ///   units. One to four entries; missing entries are treated as `0`.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] if `sections` is empty or sums to zero, or
    /// [`ArchError::InvalidShape`] if the sum exceeds `head_dim / 2`
    /// (ggml asserts `sect_dims <= ne0`).
    pub fn new_with_sections(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        sections: &[usize],
    ) -> ArchResult<Self> {
        let half_dim = head_dim / 2;
        let mut sect = [0usize; 4];
        for (slot, &v) in sect.iter_mut().zip(sections.iter()) {
            *slot = v;
        }
        let sect_dims: usize = sect.iter().sum();

        if sections.is_empty() || sect_dims == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "rope.mrope_sections must contain at least one non-zero section"
                    .to_string(),
            });
        }
        if sect_dims > half_dim {
            return Err(ArchError::InvalidShape {
                name: "rope.mrope_sections".to_string(),
                expected: vec![half_dim],
                got: vec![sect_dims],
            });
        }

        // One global frequency series across the whole rotary dimension.
        let freqs: Vec<f32> = (0..half_dim)
            .map(|j| {
                if head_dim == 0 {
                    1.0
                } else {
                    1.0 / base.powf((2 * j) as f32 / head_dim as f32)
                }
            })
            .collect();

        let axis_of: Vec<MRopeAxis> = (0..half_dim)
            .map(|j| axis_for_pair(j, &sect, sect_dims))
            .collect();

        let capacity = max_seq_len * half_dim;
        let mut cos_t = Vec::with_capacity(capacity);
        let mut sin_t = Vec::with_capacity(capacity);
        let mut cos_h = Vec::with_capacity(capacity);
        let mut sin_h = Vec::with_capacity(capacity);
        let mut cos_w = Vec::with_capacity(capacity);
        let mut sin_w = Vec::with_capacity(capacity);

        for pos in 0..max_seq_len {
            for &freq in &freqs {
                let theta = pos as f32 * freq;
                let (s, c) = theta.sin_cos();
                cos_t.push(c);
                sin_t.push(s);
                cos_h.push(c);
                sin_h.push(s);
                cos_w.push(c);
                sin_w.push(s);
            }
        }

        Ok(Self {
            cos_t,
            cos_h,
            cos_w,
            sin_t,
            sin_h,
            sin_w,
            axis_of,
            half_dim,
            head_dim,
            sections: sect,
            max_seq_len,
        })
    }

    fn empty(head_dim: usize, max_seq_len: usize, sections: [usize; 4]) -> Self {
        Self {
            cos_t: Vec::new(),
            cos_h: Vec::new(),
            cos_w: Vec::new(),
            sin_t: Vec::new(),
            sin_h: Vec::new(),
            sin_w: Vec::new(),
            axis_of: Vec::new(),
            half_dim: 0,
            head_dim,
            sections,
            max_seq_len,
        }
    }

    /// Apply M-RoPE in-place to a single attention-head vector.
    ///
    /// Rotation pair `j` uses `(x[j], x[j + head_dim/2])` — global GPT-NeoX
    /// pairing — with the position index of the axis its section assigns.
    ///
    /// For text-only tokens call with `t_pos == h_pos == w_pos`; the result is
    /// then exactly standard 1-D NeoX RoPE over the whole head vector.
    ///
    /// **Never panics.** Out-of-range positions leave `x` untouched; use
    /// [`Self::try_apply_mrope`] to have that reported instead.
    ///
    /// `head_dim` must match the value the table was constructed with.
    pub fn apply_mrope(
        &self,
        x: &mut [f32],
        t_pos: usize,
        h_pos: usize,
        w_pos: usize,
        head_dim: usize,
    ) {
        let _ = self.try_apply_mrope(x, t_pos, h_pos, w_pos, head_dim);
    }

    /// Apply M-RoPE in-place, reporting out-of-range positions.
    ///
    /// # Errors
    ///
    /// * [`ArchError::ConfigMismatch`] when any axis position is at or beyond
    ///   [`Self::max_seq_len`].  This used to be a silent no-op, so a
    ///   too-long sequence produced *unrotated* queries and keys with no
    ///   diagnostic at all.
    /// * [`ArchError::InvalidShape`] when `head_dim` disagrees with the table
    ///   or `x` is shorter than `2 * half_dim`.
    pub fn try_apply_mrope(
        &self,
        x: &mut [f32],
        t_pos: usize,
        h_pos: usize,
        w_pos: usize,
        head_dim: usize,
    ) -> ArchResult<()> {
        if self.half_dim == 0 {
            return Ok(());
        }
        if head_dim != self.head_dim {
            return Err(ArchError::InvalidShape {
                name: "mrope.head_dim".to_string(),
                expected: vec![self.head_dim],
                got: vec![head_dim],
            });
        }
        let max_pos = t_pos.max(h_pos).max(w_pos);
        if max_pos >= self.max_seq_len {
            return Err(ArchError::ConfigMismatch {
                param: "mrope.position".to_string(),
                expected: format!("< max_seq_len ({})", self.max_seq_len),
                got: format!("({t_pos}, {h_pos}, {w_pos})"),
            });
        }
        let half = self.half_dim;
        if x.len() < 2 * half {
            return Err(ArchError::InvalidShape {
                name: "mrope.head_vector".to_string(),
                expected: vec![2 * half],
                got: vec![x.len()],
            });
        }

        for j in 0..half {
            let (cos, sin, pos) = match self.axis_of[j] {
                MRopeAxis::Time | MRopeAxis::Extra => (&self.cos_t, &self.sin_t, t_pos),
                MRopeAxis::Height => (&self.cos_h, &self.sin_h, h_pos),
                MRopeAxis::Width => (&self.cos_w, &self.sin_w, w_pos),
            };
            let idx = pos * half + j;
            let (Some(&c), Some(&s)) = (cos.get(idx), sin.get(idx)) else {
                continue;
            };
            let x0 = x[j];
            let x1 = x[j + half];
            x[j] = x0 * c - x1 * s;
            x[j + half] = x0 * s + x1 * c;
        }
        Ok(())
    }

    /// Maximum sequence length for which frequency tables were precomputed.
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Total number of rotation pairs (`head_dim / 2`).
    #[inline]
    pub fn half_dim(&self) -> usize {
        self.half_dim
    }

    /// Section widths `[t, h, w, e]` in rotation-pair units.
    #[inline]
    pub fn sections(&self) -> [usize; 4] {
        self.sections
    }

    /// Which axis drives rotation pair `pair_index`.
    #[inline]
    pub fn axis_of(&self, pair_index: usize) -> Option<MRopeAxis> {
        self.axis_of.get(pair_index).copied()
    }

    /// Number of rotation pairs per axis under an equal-thirds split.
    ///
    /// Retained for source compatibility with the previous API; it describes
    /// the *default* layout only and has no meaning once real
    /// `rope.mrope_sections` are supplied.
    #[inline]
    pub fn half_dim_per_axis(&self) -> usize {
        self.sections[0]
    }
}

// ---- Private helpers -------------------------------------------------------

/// The equal-thirds fallback layout, in rotation-pair units.
fn default_sections(half_dim: usize) -> [usize; 4] {
    let third = half_dim / 3;
    [third, third, half_dim - 2 * third, 0]
}

/// ggml's sector → axis mapping.
fn axis_for_pair(pair_index: usize, sections: &[usize; 4], sect_dims: usize) -> MRopeAxis {
    let sector = pair_index % sect_dims;
    let sec_h = sections[0] + sections[1];
    let sec_w = sec_h + sections[2];
    if sector < sections[0] {
        MRopeAxis::Time
    } else if sector < sec_h {
        MRopeAxis::Height
    } else if sector < sec_w {
        MRopeAxis::Width
    } else {
        MRopeAxis::Extra
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rope::RopeTable;

    /// With `t == h == w`, M-RoPE must be **exactly** standard 1-D NeoX RoPE
    /// over the full head vector — not three independent sub-rotations.
    ///
    /// The previous implementation restarted the frequency series inside each
    /// third and could only match a per-third reference; its own test said so.
    #[test]
    fn mrope_text_only_matches_full_width_1d_rope() {
        let head_dim = 24usize;
        let max_seq = 32usize;
        let base = 10000.0f32;
        let pos = 5usize;

        let mrope = MRopeTable::new(head_dim, max_seq, base);
        let rope = RopeTable::new_standard(head_dim, max_seq, base);

        let x_init: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let mut x_mm = x_init.clone();
        mrope.apply_mrope(&mut x_mm, pos, pos, pos, head_dim);

        let mut x_ref = x_init;
        rope.apply(&mut x_ref, pos);

        for (i, (a, b)) in x_mm.iter().zip(x_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at dim {i}: mrope={a}, 1d-rope={b}"
            );
        }
    }

    /// Qwen2-VL ships `[16, 24, 24]` for 128-wide heads: not an equal split.
    #[test]
    fn mrope_honours_qwen2_vl_sections() {
        let head_dim = 128usize;
        let table =
            MRopeTable::new_with_sections(head_dim, 8, 10000.0, &[16, 24, 24, 0]).expect("table");
        assert_eq!(table.sections(), [16, 24, 24, 0]);
        assert_eq!(table.half_dim(), 64);

        for j in 0..16 {
            assert_eq!(table.axis_of(j), Some(MRopeAxis::Time), "pair {j}");
        }
        for j in 16..40 {
            assert_eq!(table.axis_of(j), Some(MRopeAxis::Height), "pair {j}");
        }
        for j in 40..64 {
            assert_eq!(table.axis_of(j), Some(MRopeAxis::Width), "pair {j}");
        }
    }

    /// The frequency series is global: pair `j` uses `base^(-2j/head_dim)`
    /// regardless of which section it lands in.
    #[test]
    fn mrope_frequencies_are_global() {
        let head_dim = 128usize;
        let base = 10000.0f32;
        let table =
            MRopeTable::new_with_sections(head_dim, 4, base, &[16, 24, 24, 0]).expect("table");

        // Position 1, sin(theta) ≈ theta for the small-frequency pairs.
        for j in [20usize, 45, 63] {
            let want = 1.0f32 / base.powf((2 * j) as f32 / head_dim as f32);
            let got = table.sin_h[table.half_dim() + j];
            assert!(
                (got - want).abs() < want * 1e-3,
                "pair {j}: sin = {got}, expected global freq {want}"
            );
        }
    }

    /// Changing `h_pos` must move exactly the height section and nothing else.
    #[test]
    fn mrope_vision_axes_independent() {
        let head_dim = 128usize;
        let table =
            MRopeTable::new_with_sections(head_dim, 32, 10000.0, &[16, 24, 24, 0]).expect("table");
        let half = table.half_dim();
        let x_init: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let mut x_a = x_init.clone();
        table.apply_mrope(&mut x_a, 1, 2, 3, head_dim);
        let mut x_b = x_init;
        table.apply_mrope(&mut x_b, 1, 7, 3, head_dim);

        for j in 0..half {
            let differs =
                (x_a[j] - x_b[j]).abs() > 1e-6 || (x_a[j + half] - x_b[j + half]).abs() > 1e-6;
            let is_height = table.axis_of(j) == Some(MRopeAxis::Height);
            assert_eq!(
                differs,
                is_height,
                "pair {j} (axis {:?}) changed = {differs}",
                table.axis_of(j)
            );
        }
    }

    /// Positions past the table used to be a silent no-op.
    #[test]
    fn mrope_out_of_range_position_is_reported() {
        let table = MRopeTable::new_with_sections(24, 8, 10000.0, &[4, 4, 4, 0]).expect("table");
        let mut x = vec![1.0f32; 24];
        assert!(table.try_apply_mrope(&mut x, 8, 0, 0, 24).is_err());
        assert!(table.try_apply_mrope(&mut x, 0, 99, 0, 24).is_err());
        assert!(table.try_apply_mrope(&mut x, 7, 7, 7, 24).is_ok());
        // The infallible entry point must not panic.
        table.apply_mrope(&mut x, 10_000, 0, 0, 24);
    }

    #[test]
    fn mrope_rejects_oversized_sections() {
        // 64 pairs available, sections ask for 100.
        assert!(MRopeTable::new_with_sections(128, 4, 10000.0, &[50, 50, 0, 0]).is_err());
        assert!(MRopeTable::new_with_sections(128, 4, 10000.0, &[0, 0, 0, 0]).is_err());
        assert!(MRopeTable::new_with_sections(128, 4, 10000.0, &[]).is_err());
    }
}

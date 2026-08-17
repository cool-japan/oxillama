//! LLaMA's RoPE convention: rotate **consecutive** pairs of head values.
//!
//! # Why this is not [`RopeTable::apply`]
//!
//! ggml has two rotation layouts and llama.cpp picks one per architecture in
//! `llama_model_rope_type` (`src/llama-model.cpp`).  `LLM_ARCH_LLAMA` — which is
//! also the arch every Mistral and **Mixtral** checkpoint is converted to, see
//! `convert_hf_to_gguf.py`'s `@ModelBase.register("LlamaForCausalLM",
//! "MistralForCausalLM", "MixtralForCausalLM", ...)` → `model_arch =
//! gguf.MODEL_ARCH.LLAMA` — returns `LLAMA_ROPE_TYPE_NORM`:
//!
//! ```text
//! // ggml/src/ggml-cpu/ops.cpp, rotate_pairs<T>(n, n_offset, cache, src, dst, scale)
//! case GGML_ROPE_TYPE_NORMAL: rotate_pairs(n_dims, /*n_offset=*/1, ..., /*scale=*/1);
//! case GGML_ROPE_TYPE_NEOX:   rotate_pairs(n_dims, /*n_offset=*/n_dims/2, ...);   // scale = 2
//! ```
//!
//! With `scale = 1` and `n_offset = 1` the pair carrying angle `θ_j` is
//! `(x[2j], x[2j+1])`; with `scale = 2` and `n_offset = n_dims/2` it is
//! `(x[j], x[j + n_dims/2])`.  [`RopeTable::apply`] implements the second — the
//! NeoX split — which is what Qwen3, Gemma, Phi and friends need and what LLaMA
//! does **not**.
//!
//! The layout is not a free choice, because the converter bakes it into the
//! weights.  `convert_hf_to_gguf.py`'s `LlamaModel` sets `undo_permute = True`
//! and rewrites every `q_proj`/`k_proj` through
//!
//! ```python
//! weights.reshape(n_head, 2, dim // n_head // 2, ...).swapaxes(1, 2).reshape(...)
//! ```
//!
//! which turns HF's half-split row order into the interleaved order NORM wants.
//! Running the NeoX rotation over those permuted rows pairs each dimension with
//! the wrong partner and rotates it by the wrong angle: the logits stay finite,
//! so nothing errors, but the text is incoherent.
//!
//! # Why no new table
//!
//! [`RopeTable`]'s `cos`/`sin` are indexed `[position * half_dim + j]` by the
//! **pair index** `j`, and `j` is the same under both conventions — only which
//! two elements form pair `j` differs.  NORM therefore needs no extra
//! precomputation, just a different gather/scatter, and a llama model's table
//! stays byte-for-byte the table a Qwen3 model of the same geometry builds.

use crate::common::rope::RopeTable;

/// Apply NORM-convention RoPE to one head vector, in place.
///
/// `x` is one head (`head_dim` values) and `position` its absolute index in the
/// sequence.  Pair `j` is `(x[2j], x[2j+1])`, rotated by the angle `table`
/// precomputed for `(position, j)`.
///
/// Out-of-range positions and short vectors are **clamped**, not panicked on:
/// the number of pairs actually rotated is the smallest of the table's
/// `half_dim`, `x.len() / 2`, and what the table holds for `position`.  Head
/// dimensions past `2 * half_dim` are left untouched, mirroring ggml's "fill the
/// remaining channels with data from src" tail.
pub(crate) fn apply_rope_norm(table: &RopeTable, x: &mut [f32], position: usize) {
    let half = table.half_dim;
    if half == 0 {
        return;
    }
    let offset = position.saturating_mul(half);
    let available = table.cos.len().min(table.sin.len()).saturating_sub(offset);
    let pairs = half.min(x.len() / 2).min(available);
    if pairs == 0 {
        return;
    }

    let cos = &table.cos[offset..offset + pairs];
    let sin = &table.sin[offset..offset + pairs];
    for (j, (&c, &s)) in cos.iter().zip(sin.iter()).enumerate() {
        let x0 = x[2 * j];
        let x1 = x[2 * j + 1];
        x[2 * j] = x0 * c - x1 * s;
        x[2 * j + 1] = x0 * s + x1 * c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rope::RopeScalingType;

    fn table(head_dim: usize) -> RopeTable {
        RopeTable::new(head_dim, 16, 10000.0, RopeScalingType::Standard, 1.0)
    }

    /// Position 0 is the identity rotation under either convention.
    #[test]
    fn position_zero_is_identity() {
        let t = table(8);
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let before = x.clone();
        apply_rope_norm(&t, &mut x, 0);
        for (a, b) in x.iter().zip(before.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
        }
    }

    /// The NORM pairing is `(2j, 2j+1)`, **not** NeoX's `(j, j + half)`.
    ///
    /// A one-hot head isolates the pairing: setting only `x[0] = 1` must leave
    /// `x[1]` (NORM's partner of index 0) non-zero and `x[half]` (NeoX's
    /// partner) untouched.
    #[test]
    fn pairs_are_consecutive_not_half_split() {
        let head_dim = 8;
        let t = table(head_dim);
        let mut x = vec![0.0f32; head_dim];
        x[0] = 1.0;
        apply_rope_norm(&t, &mut x, 3);

        assert!(
            x[1].abs() > 1e-6,
            "NORM must rotate x[0] into its consecutive partner x[1], got {x:?}"
        );
        assert!(
            x[head_dim / 2].abs() < 1e-12,
            "NORM must NOT touch the NeoX partner x[{}], got {x:?}",
            head_dim / 2
        );
    }

    /// NORM and NeoX genuinely differ, so a mix-up is observable.
    #[test]
    fn norm_differs_from_the_neox_table_apply() {
        let t = table(8);
        let base: Vec<f32> = (1..=8).map(|v| v as f32).collect();

        let mut norm = base.clone();
        apply_rope_norm(&t, &mut norm, 5);

        let mut neox = base.clone();
        t.apply(&mut neox, 5);

        assert!(
            norm.iter().zip(&neox).any(|(a, b)| (a - b).abs() > 1e-4),
            "NORM and NeoX must not coincide: {norm:?} vs {neox:?}"
        );
    }

    /// Rotation is norm-preserving per pair, at every position.
    #[test]
    fn rotation_preserves_pair_magnitude() {
        let t = table(16);
        for position in [0usize, 1, 7, 15] {
            let mut x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.37 - 2.0).collect();
            let before = x.clone();
            apply_rope_norm(&t, &mut x, position);
            for j in 0..8 {
                let a = before[2 * j].hypot(before[2 * j + 1]);
                let b = x[2 * j].hypot(x[2 * j + 1]);
                assert!(
                    (a - b).abs() < 1e-4,
                    "pair {j} at position {position}: |before|={a}, |after|={b}"
                );
            }
        }
    }

    /// A position past the table, or a short head, is clamped rather than
    /// indexed out of bounds.
    #[test]
    fn out_of_range_position_and_short_head_do_not_panic() {
        let t = table(8);
        let mut x = vec![1.0f32; 8];
        apply_rope_norm(&t, &mut x, 10_000);
        let mut short = vec![1.0f32; 3];
        apply_rope_norm(&t, &mut short, 1);
        assert_eq!(short.len(), 3);
    }
}

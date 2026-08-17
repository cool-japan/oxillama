//! Q5_K super-block encoder (FP32 → 176 bytes per 256 weights).
//!
//! Byte-for-byte port of llama.cpp's `quantize_row_q5_K_ref`
//! (`ggml/src/ggml-quants.c`, commit `ba7e817ee`).
//!
//! Block layout, matching `block_q5_K`:
//!
//! | offset | size | field                                            |
//! |--------|------|--------------------------------------------------|
//! | 0      | 2    | `d`     — FP16 super-scale for the 6-bit scales   |
//! | 2      | 2    | `dmin`  — FP16 super-scale for the 6-bit mins     |
//! | 4      | 12   | `scales` — 8 × (6-bit scale, 6-bit min), packed   |
//! | 16     | 32   | `qh`    — bit 4 of each 5-bit code                |
//! | 48     | 128  | `qs`    — bits 0..3 of each code, split-half      |
//!
//! Q5_K reuses Q4_K's scale packing verbatim; only the code width (31 instead
//! of 15), the `make_qkx2_quants` search parameters, and the extra high-bit
//! plane differ.

use half::f16;

use super::helpers::{get_scale_min_k4, make_qkx2_quants, nearest_int};
use super::{QK_K, SUB_BLOCKS_32};

/// Bytes in one Q5_K super-block.
pub const Q5_K_BLOCK_BYTES: usize = 176;

/// Encode one 256-weight super-block, appending exactly 176 bytes to `out`.
pub(crate) fn encode_q5_k_block(x: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), QK_K);

    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; SUB_BLOCKS_32];
    let mut scales = [0.0f32; SUB_BLOCKS_32];

    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..SUB_BLOCKS_32 {
        let xs = &x[32 * j..32 * j + 32];
        let mut sum_x2 = 0.0f32;
        for &v in xs.iter() {
            sum_x2 += v * v;
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for (w, &v) in weights.iter_mut().zip(xs.iter()) {
            *w = av_x + v.abs();
        }
        scales[j] = make_qkx2_quants(
            32,
            31,
            xs,
            &weights,
            &mut l[32 * j..32 * j + 32],
            &mut mins[j],
            &mut laux,
            -0.5,
            0.1,
            15,
            false,
        );
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }

    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

    let mut packed = [0u8; 12];
    for j in 0..SUB_BLOCKS_32 {
        // Narrow to u8 first, clamp second — ggml's order; see the note in
        // `q4_k.rs` for why it is equivalent to `clamp(0, 63)` here.
        let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
        let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
        if j < 4 {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }

    let d = f16::from_f32(max_scale / 63.0);
    let dmin = f16::from_f32(max_min / 63.0);

    let d_f32 = d.to_f32();
    let dmin_f32 = dmin.to_f32();
    for j in 0..SUB_BLOCKS_32 {
        let (sc, m) = get_scale_min_k4(j, &packed);
        let dj = d_f32 * sc as f32;
        if dj == 0.0 {
            continue;
        }
        let dm = dmin_f32 * m as f32;
        for ii in 0..32 {
            let q = nearest_int((x[32 * j + ii] + dm) / dj);
            l[32 * j + ii] = q.clamp(0, 31) as u8;
        }
    }

    let mut qh = [0u8; QK_K / 8];
    let mut ql = [0u8; QK_K / 2];

    let mut m1: u8 = 1;
    let mut m2: u8 = 2;
    for (group, n) in (0..QK_K).step_by(64).enumerate() {
        for j in 0..32 {
            let mut l1 = l[n + j];
            if l1 > 15 {
                l1 -= 16;
                qh[j] |= m1;
            }
            let mut l2 = l[n + j + 32];
            if l2 > 15 {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[group * 32 + j] = l1 | (l2 << 4);
        }
        m1 <<= 2;
        m2 <<= 2;
    }

    out.extend_from_slice(&d.to_bits().to_le_bytes());
    out.extend_from_slice(&dmin.to_bits().to_le_bytes());
    out.extend_from_slice(&packed);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&ql);
}

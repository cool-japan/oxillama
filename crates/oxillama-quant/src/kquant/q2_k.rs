//! Q2_K super-block encoder (FP32 → 84 bytes per 256 weights).
//!
//! Byte-for-byte port of llama.cpp's `quantize_row_q2_K_ref`
//! (`ggml/src/ggml-quants.c`, commit `ba7e817ee`).
//!
//! Block layout, matching `block_q2_K`:
//!
//! | offset | size | field                                            |
//! |--------|------|--------------------------------------------------|
//! | 0      | 16   | `scales` — 16 × (4-bit scale, 4-bit min)          |
//! | 16     | 64   | `qs`     — 256 × 2-bit codes                      |
//! | 80     | 2    | `d`      — FP16 super-scale for the scales        |
//! | 82     | 2    | `dmin`   — FP16 super-scale for the mins          |

use half::f16;

use super::helpers::{make_qkx2_quants, nearest_int};
use super::{QK_K, SUB_BLOCKS_16};

/// Bytes in one Q2_K super-block.
pub const Q2_K_BLOCK_BYTES: usize = 84;

/// Encode one 256-weight super-block, appending exactly 84 bytes to `out`.
pub(crate) fn encode_q2_k_block(x: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), QK_K);

    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 16];
    let mut weights = [0.0f32; 16];
    let mut mins = [0.0f32; SUB_BLOCKS_16];
    let mut scales = [0.0f32; SUB_BLOCKS_16];

    const Q4SCALE: f32 = 15.0;

    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..SUB_BLOCKS_16 {
        let xs = &x[16 * j..16 * j + 16];
        for (w, &v) in weights.iter_mut().zip(xs.iter()) {
            *w = v.abs();
        }
        scales[j] = make_qkx2_quants(
            16,
            3,
            xs,
            &weights,
            &mut l[16 * j..16 * j + 16],
            &mut mins[j],
            &mut laux,
            -0.5,
            0.1,
            15,
            true,
        );
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }

    // Zero-initialised: ggml's `max_scale <= 0` branch explicitly zeroes the
    // scale nibbles, and the min nibbles are OR-ed in afterwards.
    let mut packed = [0u8; SUB_BLOCKS_16];
    let d = if max_scale > 0.0 {
        let iscale = Q4SCALE / max_scale;
        for (slot, &s) in packed.iter_mut().zip(scales.iter()) {
            *slot = nearest_int(iscale * s) as u8;
        }
        f16::from_f32(max_scale / Q4SCALE)
    } else {
        f16::from_f32(0.0)
    };
    let dmin = if max_min > 0.0 {
        let iscale = Q4SCALE / max_min;
        for (slot, &m) in packed.iter_mut().zip(mins.iter()) {
            *slot |= (nearest_int(iscale * m) as u8) << 4;
        }
        f16::from_f32(max_min / Q4SCALE)
    } else {
        f16::from_f32(0.0)
    };

    let d_f32 = d.to_f32();
    let dmin_f32 = dmin.to_f32();
    for j in 0..SUB_BLOCKS_16 {
        let dj = d_f32 * (packed[j] & 0xF) as f32;
        if dj == 0.0 {
            continue;
        }
        let dm = dmin_f32 * (packed[j] >> 4) as f32;
        for ii in 0..16 {
            let q = nearest_int((x[16 * j + ii] + dm) / dj);
            l[16 * j + ii] = q.clamp(0, 3) as u8;
        }
    }

    let mut qs = [0u8; QK_K / 4];
    for j in (0..QK_K).step_by(128) {
        for i in 0..32 {
            qs[j / 4 + i] =
                l[j + i] | (l[j + i + 32] << 2) | (l[j + i + 64] << 4) | (l[j + i + 96] << 6);
        }
    }

    out.extend_from_slice(&packed);
    out.extend_from_slice(&qs);
    out.extend_from_slice(&d.to_bits().to_le_bytes());
    out.extend_from_slice(&dmin.to_bits().to_le_bytes());
}

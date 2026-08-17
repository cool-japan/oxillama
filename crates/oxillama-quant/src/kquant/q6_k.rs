//! Q6_K super-block encoder (FP32 → 210 bytes per 256 weights).
//!
//! Byte-for-byte port of llama.cpp's `quantize_row_q6_K_ref`
//! (`ggml/src/ggml-quants.c`, commit `ba7e817ee`).
//!
//! Block layout, matching `block_q6_K`:
//!
//! | offset | size | field                                          |
//! |--------|------|------------------------------------------------|
//! | 0      | 128  | `ql`     — bits 0..3 of each 6-bit code         |
//! | 128    | 64   | `qh`     — bits 4..5 of each code               |
//! | 192    | 16   | `scales` — 16 × int8 sub-block scale            |
//! | 208    | 2    | `d`      — FP16 super-scale                     |
//!
//! Q6_K is symmetric: `w[16j + i] = d * scales[j] * (q - 32)`, no min offset.

use half::f16;

use super::helpers::{make_qx_quants, nearest_int, GROUP_MAX_EPS};
use super::{QK_K, SUB_BLOCKS_16};

/// Bytes in one Q6_K super-block.
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Encode one 256-weight super-block, appending exactly 210 bytes to `out`.
pub(crate) fn encode_q6_k_block(x: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), QK_K);

    let mut l = [0i8; QK_K];
    let mut scales = [0.0f32; SUB_BLOCKS_16];

    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;

    for ib in 0..SUB_BLOCKS_16 {
        let scale = make_qx_quants(
            16,
            32,
            &x[16 * ib..16 * ib + 16],
            &mut l[16 * ib..16 * ib + 16],
            1,
            None,
        );
        scales[ib] = scale;
        let abs_scale = scale.abs();
        if abs_scale > max_abs_scale {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if max_abs_scale < GROUP_MAX_EPS {
        // ggml `memset(&y[i], 0, sizeof(block_q6_K))` then writes fp16(0),
        // which is also all-zero: the whole 210-byte block is zeros.
        out.resize(out.len() + Q6_K_BLOCK_BYTES, 0);
        return;
    }

    let iscale = -128.0f32 / max_scale;
    let d = f16::from_f32(1.0 / iscale);

    let mut sc = [0i8; SUB_BLOCKS_16];
    for ib in 0..SUB_BLOCKS_16 {
        // ggml clamps the *int* to 127 and then narrows to int8_t; there is
        // no lower clamp, so a value below -128 wraps exactly as C does.
        sc[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
    }

    let d_f32 = d.to_f32();
    for j in 0..SUB_BLOCKS_16 {
        let dj = d_f32 * sc[j] as f32;
        if dj == 0.0 {
            // Leaves the first-pass codes from `make_qx_quants` in place.
            continue;
        }
        for ii in 0..16 {
            let q = nearest_int(x[16 * j + ii] / dj).clamp(-32, 31);
            l[16 * j + ii] = (q + 32) as i8;
        }
    }

    let mut ql = [0u8; QK_K / 2];
    let mut qh = [0u8; QK_K / 4];
    for (group, j) in (0..QK_K).step_by(128).enumerate() {
        let ql_off = group * 64;
        let qh_off = group * 32;
        for i in 0..32 {
            let l0 = l[j + i] as u8;
            let l1 = l[j + i + 32] as u8;
            let l2 = l[j + i + 64] as u8;
            let l3 = l[j + i + 96] as u8;
            ql[ql_off + i] = (l0 & 0xF) | ((l2 & 0xF) << 4);
            ql[ql_off + i + 32] = (l1 & 0xF) | ((l3 & 0xF) << 4);
            qh[qh_off + i] = (l0 >> 4) | ((l1 >> 4) << 2) | ((l2 >> 4) << 4) | ((l3 >> 4) << 6);
        }
    }

    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    for &v in sc.iter() {
        out.push(v as u8);
    }
    out.extend_from_slice(&d.to_bits().to_le_bytes());
}

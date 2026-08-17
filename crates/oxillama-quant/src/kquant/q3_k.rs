//! Q3_K super-block encoder (FP32 → 110 bytes per 256 weights).
//!
//! Byte-for-byte port of llama.cpp's `quantize_row_q3_K_ref`
//! (`ggml/src/ggml-quants.c`, commit `ba7e817ee`).
//!
//! Block layout, matching `block_q3_K`:
//!
//! | offset | size | field                                            |
//! |--------|------|--------------------------------------------------|
//! | 0      | 32   | `hmask`  — bit 2 of each 3-bit code               |
//! | 32     | 64   | `qs`     — bits 0..1 of each code                 |
//! | 96     | 12   | `scales` — 16 × 6-bit signed sub-block scale      |
//! | 108    | 2    | `d`      — FP16 super-scale                       |

use half::f16;

use super::helpers::{make_q3_quants, nearest_int};
use super::{QK_K, SUB_BLOCKS_16};

/// Bytes in one Q3_K super-block.
pub const Q3_K_BLOCK_BYTES: usize = 110;

/// Encode one 256-weight super-block, appending exactly 110 bytes to `out`.
pub(crate) fn encode_q3_k_block(x: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), QK_K);

    let mut l = [0i8; QK_K];
    let mut scales = [0.0f32; SUB_BLOCKS_16];

    let mut max_scale = 0.0f32;
    let mut amax = 0.0f32;
    for j in 0..SUB_BLOCKS_16 {
        scales[j] = make_q3_quants(
            16,
            4,
            &x[16 * j..16 * j + 16],
            &mut l[16 * j..16 * j + 16],
            true,
        );
        let scale = scales[j].abs();
        if scale > amax {
            amax = scale;
            max_scale = scales[j];
        }
    }

    // 6-bit scales, split across 12 bytes: low nibble in [0..8), high nibble
    // in [0..8) of the same byte, and the two spare bits in bytes 8..12.
    let mut packed = [0u8; 12];
    let d;
    if max_scale != 0.0 {
        let iscale = -32.0f32 / max_scale;
        for j in 0..SUB_BLOCKS_16 {
            // ggml narrows to int8_t *before* clamping, exactly as here.
            let mut lv = nearest_int(iscale * scales[j]) as i8;
            lv = lv.clamp(-32, 31) + 32;
            if j < 8 {
                packed[j] = (lv & 0xF) as u8;
            } else {
                packed[j - 8] |= ((lv & 0xF) as u8) << 4;
            }
            lv >>= 4;
            packed[j % 4 + 8] |= (lv as u8) << (2 * (j / 4));
        }
        d = f16::from_f32(1.0 / iscale);
    } else {
        d = f16::from_f32(0.0);
    }

    let d_f32 = d.to_f32();
    for j in 0..SUB_BLOCKS_16 {
        let mut sc: i8 = if j < 8 {
            (packed[j] & 0xF) as i8
        } else {
            (packed[j - 8] >> 4) as i8
        };
        sc = (sc | ((((packed[8 + j % 4] >> (2 * (j / 4))) & 3) << 4) as i8)) - 32;
        let dj = d_f32 * sc as f32;
        if dj == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let q = nearest_int(x[16 * j + ii] / dj).clamp(-4, 3);
            l[16 * j + ii] = (q + 4) as i8;
        }
    }

    // High bit plane: quant `j` contributes to byte `j % 32`, bit `j / 32`.
    let mut hmask = [0u8; QK_K / 8];
    let mut m = 0usize;
    let mut hm: u8 = 1;
    for code in l.iter_mut() {
        if *code > 3 {
            hmask[m] |= hm;
            *code -= 4;
        }
        m += 1;
        if m == QK_K / 8 {
            m = 0;
            hm <<= 1;
        }
    }

    let mut qs = [0u8; QK_K / 4];
    for j in (0..QK_K).step_by(128) {
        for i in 0..32 {
            let l0 = l[j + i] as u8;
            let l1 = l[j + i + 32] as u8;
            let l2 = l[j + i + 64] as u8;
            let l3 = l[j + i + 96] as u8;
            qs[j / 4 + i] = l0 | (l1 << 2) | (l2 << 4) | (l3 << 6);
        }
    }

    out.extend_from_slice(&hmask);
    out.extend_from_slice(&qs);
    out.extend_from_slice(&packed);
    out.extend_from_slice(&d.to_bits().to_le_bytes());
}

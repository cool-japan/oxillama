//! Q4_K super-block encoder (FP32 → 144 bytes per 256 weights).
//!
//! Byte-for-byte port of llama.cpp's `quantize_row_q4_K_ref`
//! (`ggml/src/ggml-quants.c`, commit `ba7e817ee`).
//!
//! Block layout, matching `block_q4_K`:
//!
//! | offset | size | field                                            |
//! |--------|------|--------------------------------------------------|
//! | 0      | 2    | `d`     — FP16 super-scale for the 6-bit scales   |
//! | 2      | 2    | `dmin`  — FP16 super-scale for the 6-bit mins     |
//! | 4      | 12   | `scales` — 8 × (6-bit scale, 6-bit min), packed   |
//! | 16     | 128  | `qs`    — 256 × 4-bit codes, split-half packed    |
//!
//! Decode: `w[32j + i] = (d * sc_j) * q - (dmin * m_j)`.

use half::f16;

use super::helpers::{get_scale_min_k4, make_qkx2_quants, nearest_int};
use super::{QK_K, SUB_BLOCKS_32};

/// Bytes in one Q4_K super-block.
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// Encode one 256-weight super-block, appending exactly 144 bytes to `out`.
pub(crate) fn encode_q4_k_block(x: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), QK_K);

    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; SUB_BLOCKS_32];
    let mut scales = [0.0f32; SUB_BLOCKS_32];

    let mut max_scale = 0.0f32; // deducting the min keeps every scale positive
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
            15,
            xs,
            &weights,
            &mut l[32 * j..32 * j + 32],
            &mut mins[j],
            &mut laux,
            -1.0,
            0.1,
            20,
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

    // `scales` is zero-initialised because the j >= 4 half OR-writes into
    // bytes the j < 4 half has already assigned.  The two halves must stay
    // in this order: byte `j-4` and byte `j` are `=`-assigned while j < 4
    // and only then `|=`-updated while j >= 4.
    let mut packed = [0u8; 12];
    for j in 0..SUB_BLOCKS_32 {
        // ggml narrows the rounded `int` to `uint8_t` and *then* clamps to
        // 63, which is not in general the same function as `clamp(0, 63)` on
        // the `i32` — a negative value wraps to >= 128 and would come out as
        // 63 rather than 0.  Here the two agree, because neither product can
        // be negative: `mins[j]` is `-min` with `min <= 0`, and
        // `make_qkx2_quants` never returns a negative scale (it returns
        // either `(max - min)/nmax > 0` or a weighted regression slope of `x`
        // on codes that are monotone non-decreasing in `x`).  A brute-force
        // search over 4M adversarial 32-weight rows — all-negative,
        // near-constant, spiked, heavy-tailed — found no negative scale, and
        // no golden case here distinguishes the two forms.  ggml's form is
        // kept anyway: it is what the reference does, so it stays right if
        // upstream ever admits a negative scale.
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

    // Second pass: re-derive the codes from the *quantized* scales, so the
    // encoder and the decoder agree on the reconstruction levels.  Sub-blocks
    // whose reconstructed scale is exactly zero are skipped, deliberately
    // leaving the first-pass codes from `make_qkx2_quants` in `l`.
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
            l[32 * j + ii] = q.clamp(0, 15) as u8;
        }
    }

    out.extend_from_slice(&d.to_bits().to_le_bytes());
    out.extend_from_slice(&dmin.to_bits().to_le_bytes());
    out.extend_from_slice(&packed);
    // Split-half nibble packing: byte `i` of a 64-weight group carries
    // weight `i` in the low nibble and weight `i + 32` in the high nibble.
    for j in (0..QK_K).step_by(64) {
        for i in 0..32 {
            out.push(l[j + i] | (l[j + i + 32] << 4));
        }
    }
}

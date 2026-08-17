//! Golden vectors for the AVX-512 fused Q8_0-activation GEMV kernels.
//!
//! # Why a *model* and not just the kernels
//!
//! This crate is developed on aarch64, where the AVX-512 kernels are
//! `cfg`-ed out and cannot be executed at all.  A test that only called the
//! kernels would therefore assert nothing here — the exact failure mode that
//! let seven quantization formats ship broken behind a green suite.
//!
//! So this file carries a **scalar model** of the lane arithmetic in
//! `simd/avx512/int_dot.rs` and `simd/avx512/fused.rs`: one Rust function per
//! intrinsic sequence, mirroring the widths, the operand signedness and the
//! order of the floating-point combination.  The model runs everywhere,
//! including on this ARM host, and is held against constants produced by
//! **executing llama.cpp's C** reference kernels.  On an x86-64 host with
//! AVX-512 the same tests additionally run the real intrinsics and require
//! them to agree with the model bit-for-bit.
//!
//! # Where the constants come from
//!
//! `scratchpad/golden/gen.c` links against llama.cpp's real
//! `ggml/src/ggml-cpu/quants.c` (the `*_generic` scalar entry points, built
//! straight from `~/work/refs/llama.cpp`) and prints the arrays below.  Two
//! extraction tricks are used:
//!
//! * **Integer core.**  Every generic kernel ends with
//!   `sumf += (d_x*d_y) * sumi [+ m_x * s_y]`.  With `d_x = d_y = 1` and
//!   `m_x = 0`, `sumf` *is* `sumi`, exactly: `|sumi| ≤ 32·31·128 = 126_976`,
//!   well inside binary32's exact-integer range.
//! * **Per-lane unpack.**  Dotting one weight block against 32 one-hot
//!   activation vectors recovers llama.cpp's own value for each unpacked
//!   weight, which pins the *order* of ggml's split-half nibble layout — the
//!   single most dangerous thing to get wrong in a hand-written SIMD unpack.
//!
//! A third set (`*_S_*`) uses real FP16 scales so the `f32` combination itself
//! — not only the integer core — is compared against llama.cpp.
//!
//! # Verification status
//!
//! Running the AVX-512 instructions themselves on AVX-512 hardware is
//! **pending**; nothing here has executed on such a machine.  See the
//! `simd/avx512` module docs.

// The model deliberately mirrors machine-word arithmetic; index-based loops
// and explicit casts are the point.
#![allow(clippy::needless_range_loop)]

/// Q8_0 activation block: 2-byte FP16 scale + 32 × i8.
const ACT_BYTES: usize = 34;

/// Read a little-endian FP16 at the start of `bytes`.
fn f16(bytes: &[u8]) -> f32 {
    half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

// ===========================================================================
// Scalar model of the AVX-512 lane arithmetic
// ===========================================================================
//
// Each function names the intrinsic sequence it mirrors.  Keep this module and
// `src/simd/avx512/{int_dot,fused}.rs` in step: a change to one without the
// other is exactly the drift these goldens exist to catch.
mod model {
    /// Mirrors `Bw::dot32_i8`:
    /// `_mm512_cvtepi8_epi16` ×2 → `_mm512_madd_epi16` → `_mm512_reduce_add_epi32`.
    ///
    /// `VPMADDWD` sums *adjacent pairs* into i32 lanes, which is reproduced
    /// literally here.  `Base::dot32_i8` instead widens to i32 and multiplies
    /// 16-wide; both are exact integer arithmetic over the same 32 products, so
    /// they agree bit-for-bit regardless of grouping (see
    /// [`pairing_does_not_change_the_integer_dot`]).
    pub fn dot32_i8(x: &[i8; 32], y: &[i8; 32]) -> i32 {
        let mut acc = 0i32;
        for k in 0..16 {
            let p = x[2 * k] as i32 * y[2 * k] as i32 + x[2 * k + 1] as i32 * y[2 * k + 1] as i32;
            acc += p;
        }
        acc
    }

    /// Mirrors `Bw::dot32_u8_i8`: `_mm512_cvtepu8_epi16` for the weights
    /// (zero-extend) and `_mm512_cvtepi8_epi16` for the activations
    /// (sign-extend), then the same `VPMADDWD` + reduce.
    pub fn dot32_u8_i8(x: &[u8; 32], y: &[i8; 32]) -> i32 {
        let mut acc = 0i32;
        for k in 0..16 {
            let p = x[2 * k] as i32 * y[2 * k] as i32 + x[2 * k + 1] as i32 * y[2 * k + 1] as i32;
            acc += p;
        }
        acc
    }

    /// Mirrors `Bw::sum32_i8`: `VPMADDWD` against a vector of ones, then reduce.
    pub fn sum32_i8(y: &[i8; 32]) -> i32 {
        let mut acc = 0i32;
        for k in 0..16 {
            acc += y[2 * k] as i32 + y[2 * k + 1] as i32;
        }
        acc
    }

    /// Mirrors `int_dot::expand_qh_bits_16`:
    /// `_mm512_maskz_set1_epi32(bits, 0x10)` puts `0x10` in the 32-bit lanes
    /// whose mask bit is set, and `_mm512_cvtepi32_epi8` (`VPMOVDB`) truncates
    /// lane `i` to byte `i`.
    pub fn expand_qh_bits_16(bits: u16) -> [u8; 16] {
        core::array::from_fn(|i| if (bits >> i) & 1 == 1 { 0x10 } else { 0x00 })
    }

    /// Mirrors `fused::unpack_q4_0`:
    /// `_mm_and_si128(raw, 0x0F)` → lanes 0..16, `_mm_srli_epi16(raw,4) & 0x0F`
    /// → lanes 16..32, joined by `_mm256_set_m128i(hi, lo)` (so `lo` occupies
    /// the low 128 bits = lanes 0..16), then `_mm256_sub_epi8(_, 8)`.
    pub fn unpack_q4_0(qs: &[u8]) -> [i8; 32] {
        let lo: [u8; 16] = core::array::from_fn(|i| qs[i] & 0x0F);
        let hi: [u8; 16] = core::array::from_fn(|i| (qs[i] >> 4) & 0x0F);
        core::array::from_fn(|i| {
            let v = if i < 16 { lo[i] } else { hi[i - 16] };
            (v as i8).wrapping_sub(8)
        })
    }

    /// Mirrors `fused::unpack_q5`: the same nibble split, `OR`-ed with
    /// [`expand_qh_bits_16`] of `qh`'s low 16 bits (low half) and high 16 bits
    /// (high half), joined by `_mm256_set_m128i`.  Result is the *unsigned*
    /// 5-bit quant, 0..=31.
    pub fn unpack_q5(qs: &[u8], qh: u32) -> [u8; 32] {
        let bits_lo = expand_qh_bits_16(qh as u16);
        let bits_hi = expand_qh_bits_16((qh >> 16) as u16);
        let lo: [u8; 16] = core::array::from_fn(|i| (qs[i] & 0x0F) | bits_lo[i]);
        let hi: [u8; 16] = core::array::from_fn(|i| ((qs[i] >> 4) & 0x0F) | bits_hi[i]);
        core::array::from_fn(|i| if i < 16 { lo[i] } else { hi[i - 16] })
    }

    /// Mirrors `unpack_q5` followed by `_mm256_sub_epi8(_, 16)` (Q5_0).
    pub fn unpack_q5_0(qs: &[u8], qh: u32) -> [i8; 32] {
        let q = unpack_q5(qs, qh);
        core::array::from_fn(|i| (q[i] as i8).wrapping_sub(16))
    }
}

// ===========================================================================
// Row models — the f32 combination, in the kernels' order
// ===========================================================================

/// The 32 `i8` quants of activation block `blk`.
fn act_quants(acts: &[u8], blk: usize) -> [i8; 32] {
    let b = &acts[blk * ACT_BYTES..(blk + 1) * ACT_BYTES];
    core::array::from_fn(|i| b[2 + i] as i8)
}

/// Mirrors `fused::row_i8_quants_impl` (Q8_0 with `qs_offset = 2`,
/// Q8_1 with `qs_offset = 4`).
fn row_i8_quants(
    row: &[u8],
    acts: &[u8],
    blocks: usize,
    n_cols: usize,
    block_bytes: usize,
    qs_offset: usize,
) -> f32 {
    let mut row_sum = 0.0f32;
    for blk in 0..blocks {
        let w = &row[blk * block_bytes..(blk + 1) * block_bytes];
        let a = &acts[blk * ACT_BYTES..(blk + 1) * ACT_BYTES];
        let scale = f16(w) * f16(a);
        let wq: [i8; 32] = core::array::from_fn(|i| w[qs_offset + i] as i8);
        let aq = act_quants(acts, blk);
        let valid = n_cols.saturating_sub(blk * 32).min(32);
        let sumi = if valid == 32 {
            model::dot32_i8(&wq, &aq)
        } else {
            (0..valid).map(|i| wq[i] as i32 * aq[i] as i32).sum()
        };
        row_sum += scale * sumi as f32;
    }
    row_sum
}

/// Mirrors `fused::row_q4_0_impl`.
fn row_q4_0(row: &[u8], acts: &[u8], blocks: usize, n_cols: usize) -> f32 {
    let mut row_sum = 0.0f32;
    for blk in 0..blocks {
        let w = &row[blk * 18..(blk + 1) * 18];
        let a = &acts[blk * ACT_BYTES..(blk + 1) * ACT_BYTES];
        let scale = f16(w) * f16(a);
        let wq = model::unpack_q4_0(&w[2..]);
        let aq = act_quants(acts, blk);
        let valid = n_cols.saturating_sub(blk * 32).min(32);
        let sumi: i32 = if valid == 32 {
            model::dot32_i8(&wq, &aq)
        } else {
            (0..valid).map(|i| wq[i] as i32 * aq[i] as i32).sum()
        };
        row_sum += scale * sumi as f32;
    }
    row_sum
}

/// Mirrors `fused::row_q5_0_impl`.
fn row_q5_0(row: &[u8], acts: &[u8], blocks: usize, n_cols: usize) -> f32 {
    let mut row_sum = 0.0f32;
    for blk in 0..blocks {
        let w = &row[blk * 22..(blk + 1) * 22];
        let a = &acts[blk * ACT_BYTES..(blk + 1) * ACT_BYTES];
        let scale = f16(w) * f16(a);
        let qh = u32::from_le_bytes([w[2], w[3], w[4], w[5]]);
        let wq = model::unpack_q5_0(&w[6..], qh);
        let aq = act_quants(acts, blk);
        let valid = n_cols.saturating_sub(blk * 32).min(32);
        let sumi: i32 = if valid == 32 {
            model::dot32_i8(&wq, &aq)
        } else {
            (0..valid).map(|i| wq[i] as i32 * aq[i] as i32).sum()
        };
        row_sum += scale * sumi as f32;
    }
    row_sum
}

/// Mirrors `fused::row_q5_1_impl`, including the exact association of the two
/// terms: `(d_w*d_a) * sumi + m_w * (d_a * suma)`.
fn row_q5_1(row: &[u8], acts: &[u8], blocks: usize, n_cols: usize) -> f32 {
    let mut row_sum = 0.0f32;
    for blk in 0..blocks {
        let w = &row[blk * 24..(blk + 1) * 24];
        let a = &acts[blk * ACT_BYTES..(blk + 1) * ACT_BYTES];
        let d_w = f16(w);
        let m_w = f16(&w[2..]);
        let d_a = f16(a);
        let qh = u32::from_le_bytes([w[4], w[5], w[6], w[7]]);
        let wq = model::unpack_q5(&w[8..], qh);
        let aq = act_quants(acts, blk);
        let valid = n_cols.saturating_sub(blk * 32).min(32);
        let (sumi, suma) = if valid == 32 {
            (model::dot32_u8_i8(&wq, &aq), model::sum32_i8(&aq))
        } else {
            let mut si = 0i32;
            let mut sa = 0i32;
            for i in 0..valid {
                si += wq[i] as i32 * aq[i] as i32;
                sa += aq[i] as i32;
            }
            (si, sa)
        };
        row_sum += (d_w * d_a) * sumi as f32 + m_w * (d_a * suma as f32);
    }
    row_sum
}

// ===========================================================================
// Golden constants — produced by executing llama.cpp's C
// ===========================================================================

#[rustfmt::skip]
#[allow(dead_code)]
mod goldens {
    include!("data/avx512_fused_goldens_data.rs");
}

/// A fixed Q8_0 activation block for the ragged-K tests.
///
/// These bytes need no golden of their own: the expected value is built from
/// `goldens::*_UNPACK` — llama.cpp's own per-lane weights — so the assertion
/// still rests on llama.cpp, not on this file.
const TAIL_ACTS: [u8; 34] = {
    let mut b = [0u8; 34];
    // FP16 1.0 = 0x3C00, so the f32 combination reduces to the integer sum.
    b[1] = 0x3C;
    let mut i = 0;
    while i < 32 {
        // A spread of positives, negatives and the i8 extremes.
        b[2 + i] = (((i as i32) * 23 - 128) as i8) as u8;
        i += 1;
    }
    b[2] = 0x80; // -128
    b[33] = 0x7F; // +127
    b
};

// ===========================================================================
// Integer-core tests (run on every host, including this aarch64 one)
// ===========================================================================

/// Per-block integer dot products for the `qs_offset = 2` formats.
fn block_sumi_i8(w: &[u8], acts: &[u8], blocks: usize, block_bytes: usize, off: usize) -> Vec<i32> {
    (0..blocks)
        .map(|blk| {
            let b = &w[blk * block_bytes..(blk + 1) * block_bytes];
            let wq: [i8; 32] = core::array::from_fn(|i| b[off + i] as i8);
            model::dot32_i8(&wq, &act_quants(acts, blk))
        })
        .collect()
}

#[test]
fn q8_0_integer_core_matches_llama_cpp() {
    use goldens::*;
    assert_eq!(
        block_sumi_i8(&Q8_0_A_W, &Q8_0_A_A, 2, 34, 2),
        Q8_0_A_SUMI.to_vec()
    );
    assert_eq!(
        block_sumi_i8(&Q8_0_B_W, &Q8_0_B_A, 3, 34, 2),
        Q8_0_B_SUMI.to_vec()
    );
    // All weights -1, all activations -128: the saturating corner.
    assert_eq!(
        block_sumi_i8(&Q8_0_X_W, &Q8_0_X_A, 1, 34, 2),
        Q8_0_X_SUMI.to_vec()
    );
}

#[test]
fn q4_0_integer_core_matches_llama_cpp() {
    use goldens::*;
    let sumi = |w: &[u8], a: &[u8], nb: usize| -> Vec<i32> {
        (0..nb)
            .map(|blk| {
                let b = &w[blk * 18..(blk + 1) * 18];
                model::dot32_i8(&model::unpack_q4_0(&b[2..]), &act_quants(a, blk))
            })
            .collect()
    };
    assert_eq!(sumi(&Q4_0_A_W, &Q4_0_A_A, 2), Q4_0_A_SUMI.to_vec());
    assert_eq!(sumi(&Q4_0_B_W, &Q4_0_B_A, 3), Q4_0_B_SUMI.to_vec());
    assert_eq!(sumi(&Q4_0_X_W, &Q4_0_X_A, 1), Q4_0_X_SUMI.to_vec());
}

#[test]
fn q5_0_integer_core_matches_llama_cpp() {
    use goldens::*;
    let sumi = |w: &[u8], a: &[u8], nb: usize| -> Vec<i32> {
        (0..nb)
            .map(|blk| {
                let b = &w[blk * 22..(blk + 1) * 22];
                let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                model::dot32_i8(&model::unpack_q5_0(&b[6..], qh), &act_quants(a, blk))
            })
            .collect()
    };
    assert_eq!(sumi(&Q5_0_A_W, &Q5_0_A_A, 2), Q5_0_A_SUMI.to_vec());
    assert_eq!(sumi(&Q5_0_B_W, &Q5_0_B_A, 3), Q5_0_B_SUMI.to_vec());
    assert_eq!(sumi(&Q5_0_X_W, &Q5_0_X_A, 1), Q5_0_X_SUMI.to_vec());
}

#[test]
fn q5_1_integer_core_matches_llama_cpp() {
    use goldens::*;
    let split = |w: &[u8], a: &[u8], nb: usize| -> (Vec<i32>, Vec<i32>) {
        let mut si = Vec::new();
        let mut sa = Vec::new();
        for blk in 0..nb {
            let b = &w[blk * 24..(blk + 1) * 24];
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let aq = act_quants(a, blk);
            si.push(model::dot32_u8_i8(&model::unpack_q5(&b[8..], qh), &aq));
            sa.push(model::sum32_i8(&aq));
        }
        (si, sa)
    };
    let (si, sa) = split(&Q5_1_A_W, &Q5_1_A_A, 2);
    assert_eq!(si, Q5_1_A_SUMI.to_vec());
    assert_eq!(sa, Q5_1_A_SUMA.to_vec());
    let (si, sa) = split(&Q5_1_B_W, &Q5_1_B_A, 3);
    assert_eq!(si, Q5_1_B_SUMI.to_vec());
    assert_eq!(sa, Q5_1_B_SUMA.to_vec());
    let (si, sa) = split(&Q5_1_X_W, &Q5_1_X_A, 1);
    assert_eq!(si, Q5_1_X_SUMI.to_vec());
    assert_eq!(sa, Q5_1_X_SUMA.to_vec());
}

/// Q8_1's quants are `i8` at byte offset 4 and its `s` field is unused, so its
/// integer core is Q8_0's with the block re-laid.  Rebuild the Q8_0 goldens as
/// Q8_1 blocks and require the same answers — this pins the *offset*, which is
/// the only thing that differs between the two kernels.
#[test]
fn q8_1_integer_core_matches_relaid_q8_0_goldens() {
    use goldens::*;
    let mut w = Vec::new();
    for blk in 0..2 {
        let b = &Q8_0_A_W[blk * 34..(blk + 1) * 34];
        w.extend_from_slice(&b[0..2]); // d
        w.extend_from_slice(&[0, 0]); // s — unused by this crate's Q8_1
        w.extend_from_slice(&b[2..34]); // quants
    }
    assert_eq!(block_sumi_i8(&w, &Q8_0_A_A, 2, 36, 4), Q8_0_A_SUMI.to_vec());
}

// ===========================================================================
// Per-lane unpack tests — these pin the weight ORDER
// ===========================================================================

#[test]
fn q4_0_unpack_order_matches_llama_cpp() {
    use goldens::*;
    let got = model::unpack_q4_0(&Q4_0_UNPACK_W[2..]);
    for i in 0..32 {
        assert_eq!(
            got[i] as i32, Q4_0_UNPACK[i],
            "lane {i}: split-half nibble order diverged from llama.cpp"
        );
    }
}

#[test]
fn q5_0_unpack_order_matches_llama_cpp() {
    use goldens::*;
    let qh = u32::from_le_bytes([
        Q5_0_UNPACK_W[2],
        Q5_0_UNPACK_W[3],
        Q5_0_UNPACK_W[4],
        Q5_0_UNPACK_W[5],
    ]);
    let got = model::unpack_q5_0(&Q5_0_UNPACK_W[6..], qh);
    for i in 0..32 {
        assert_eq!(
            got[i] as i32, Q5_0_UNPACK[i],
            "lane {i}: qh bit placement or nibble order diverged from llama.cpp"
        );
    }
}

#[test]
fn q5_1_unpack_order_matches_llama_cpp() {
    use goldens::*;
    let qh = u32::from_le_bytes([
        Q5_1_UNPACK_W[4],
        Q5_1_UNPACK_W[5],
        Q5_1_UNPACK_W[6],
        Q5_1_UNPACK_W[7],
    ]);
    let got = model::unpack_q5(&Q5_1_UNPACK_W[8..], qh);
    for i in 0..32 {
        assert_eq!(
            got[i] as i32, Q5_1_UNPACK[i],
            "lane {i}: qh bit placement or nibble order diverged from llama.cpp"
        );
    }
}

/// Negative control.
///
/// A golden that everything passes is a golden that tests nothing.  The
/// classic Q4_0 mistake is the *interleaved* layout — `lo → weight 2i`,
/// `hi → weight 2i+1` — which is what the AVX-512 Q4_0 module header used to
/// claim in prose.  llama.cpp uses split-half instead, so the interleaved
/// unpack must **disagree** with the goldens.  If this test ever starts
/// failing, the golden vectors have stopped discriminating between the two
/// layouts and every assertion above is worthless.
#[test]
fn interleaved_nibble_layout_is_rejected_by_the_goldens() {
    use goldens::*;
    let qs = &Q4_0_UNPACK_W[2..18];
    let interleaved: [i32; 32] = core::array::from_fn(|i| {
        let byte = qs[i / 2];
        let nib = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        nib as i32 - 8
    });
    assert_ne!(
        interleaved, Q4_0_UNPACK,
        "the interleaved layout reproduced llama.cpp's weights — these golden \
         bytes cannot distinguish the two nibble layouts, pick different ones"
    );

    // Same control for Q5_0's qh plane: swapping the two 16-bit halves must
    // change the answer, or the goldens do not pin the bit placement.
    let qh = u32::from_le_bytes([
        Q5_0_UNPACK_W[2],
        Q5_0_UNPACK_W[3],
        Q5_0_UNPACK_W[4],
        Q5_0_UNPACK_W[5],
    ]);
    let swapped = qh.rotate_left(16);
    let wrong = model::unpack_q5_0(&Q5_0_UNPACK_W[6..], swapped);
    let right: [i8; 32] = core::array::from_fn(|i| Q5_0_UNPACK[i] as i8);
    assert_ne!(
        wrong, right,
        "swapping qh's halves left the unpack unchanged — the golden block's \
         qh is too symmetric to pin bit placement"
    );
}

#[test]
fn q8_0_unpack_order_matches_llama_cpp() {
    use goldens::*;
    for i in 0..32 {
        assert_eq!(Q8_0_UNPACK_W[2 + i] as i8 as i32, Q8_0_UNPACK[i]);
    }
}

/// The `qh` expansion is the one bit-twiddle with no scalar twin in the kernel,
/// so pin it directly against the values llama.cpp implies: `unpack_q5` minus
/// its nibbles must be `0x10` exactly where a `qh` bit is set.
#[test]
fn qh_expansion_matches_llama_cpp_bit_placement() {
    use goldens::*;
    let qs = &Q5_0_UNPACK_W[6..22];
    let qh = u32::from_le_bytes([
        Q5_0_UNPACK_W[2],
        Q5_0_UNPACK_W[3],
        Q5_0_UNPACK_W[4],
        Q5_0_UNPACK_W[5],
    ]);
    for i in 0..16 {
        // Weight i uses qh bit i; weight i+16 uses qh bit i+16.
        let want_lo = (((qh >> i) & 1) as i32) * 16;
        let want_hi = (((qh >> (i + 16)) & 1) as i32) * 16;
        assert_eq!(Q5_0_UNPACK[i] + 16 - (qs[i] & 0x0F) as i32, want_lo);
        assert_eq!(
            Q5_0_UNPACK[i + 16] + 16 - (qs[i] >> 4) as i32,
            want_hi,
            "qh bit {} feeds weight {}",
            i + 16,
            i + 16
        );
    }
    // And the model's own expansion agrees with that placement.
    let lo = model::expand_qh_bits_16(qh as u16);
    let hi = model::expand_qh_bits_16((qh >> 16) as u16);
    for i in 0..16 {
        assert_eq!(lo[i] as i32, (((qh >> i) & 1) as i32) * 16);
        assert_eq!(hi[i] as i32, (((qh >> (i + 16)) & 1) as i32) * 16);
    }
}

/// `VPMADDWD` groups the 32 products into adjacent pairs; the AVX-512F-only
/// tier groups them into two halves of 16.  Integer addition is associative and
/// none of these sums can overflow `i32`, so the two groupings are the same
/// number — which is why one scalar model suffices for both tiers.
#[test]
fn pairing_does_not_change_the_integer_dot() {
    let x: [i8; 32] = core::array::from_fn(|i| (i as i32 * 17 - 128) as i8);
    let y: [i8; 32] = core::array::from_fn(|i| (127 - i as i32 * 9) as i8);
    let pairwise = model::dot32_i8(&x, &y);
    let halves: i32 = (0..16).map(|i| x[i] as i32 * y[i] as i32).sum::<i32>()
        + (16..32).map(|i| x[i] as i32 * y[i] as i32).sum::<i32>();
    assert_eq!(pairwise, halves);
    // Worst case magnitude stays far below i32::MAX.
    let worst: i32 = 32 * 127 * 128;
    assert!(pairwise.abs() <= worst && worst < i32::MAX / 2);
}

// ===========================================================================
// f32 combination tests — real FP16 scales, compared with llama.cpp's own f32
// ===========================================================================

/// llama.cpp's `sumf` and this model differ only in how `(d_x, d_y, sumi)` are
/// associated (`sumi*d_x*d_y` vs `(d_x*d_y)*sumi` for Q4_0), so a tight
/// relative bound — not bit equality — is the correct assertion.  A wrong
/// combination (a dropped `d_a`, a mis-signed min) fails it by orders of
/// magnitude.
fn assert_close_to_llama(got: f32, want_bits: u32, what: &str) {
    let want = f32::from_bits(want_bits);
    let tol = want.abs() * 1e-6;
    assert!(
        (got - want).abs() <= tol,
        "{what}: model {got} vs llama.cpp {want} (tolerance {tol})"
    );
}

#[test]
fn q4_0_f32_combination_matches_llama_cpp() {
    use goldens::*;
    let got = row_q4_0(&Q4_0_S_W, &Q4_0_S_A, 3, 96);
    assert_close_to_llama(got, Q4_0_S_SUMF_BITS, "Q4_0 scaled row");
}

#[test]
fn q5_0_f32_combination_matches_llama_cpp() {
    use goldens::*;
    let got = row_q5_0(&Q5_0_S_W, &Q5_0_S_A, 3, 96);
    assert_close_to_llama(got, Q5_0_S_SUMF_BITS, "Q5_0 scaled row");
}

#[test]
fn q8_0_f32_combination_matches_llama_cpp() {
    use goldens::*;
    let got = row_i8_quants(&Q8_0_S_W, &Q8_0_S_A, 3, 96, 34, 2);
    assert_close_to_llama(got, Q8_0_S_SUMF_BITS, "Q8_0 scaled row");
}

/// Q5_1's min term: `m_w * (d_a * Σa)`.  With `m_w = 0` in the golden blocks
/// the term vanishes, so re-run the same bytes with a non-zero `m` and check
/// the delta is exactly `m * Σ_blocks (d_a * suma)` — the shape ggml uses.
#[test]
fn q5_1_min_term_has_the_shape_llama_cpp_uses() {
    use goldens::*;
    let mut w = Q5_1_B_W.to_vec();
    let m = half::f16::from_f32(-0.75);
    for blk in 0..3 {
        w[blk * 24 + 2..blk * 24 + 4].copy_from_slice(&m.to_bits().to_le_bytes());
    }
    let with_min = row_q5_1(&w, &Q5_1_B_A, 3, 96);
    let without_min = row_q5_1(&Q5_1_B_W, &Q5_1_B_A, 3, 96);

    let mut delta = 0.0f32;
    for blk in 0..3 {
        let d_a = f16(&Q5_1_B_A[blk * ACT_BYTES..]);
        delta += m.to_f32() * (d_a * Q5_1_B_SUMA[blk] as f32);
    }
    assert!(
        (with_min - without_min - delta).abs() <= delta.abs() * 1e-6 + 1e-4,
        "min term shape wrong: {with_min} - {without_min} != {delta}"
    );
}

// ===========================================================================
// Ragged K — the tail path
// ===========================================================================

/// A tail is the same sum restricted to the in-bounds lanes.  Building the
/// expectation from the *golden* unpack (validated above against llama.cpp),
/// not from the kernel's own helper, is what keeps this from being a
/// self-parity test.
#[test]
fn q4_0_tail_uses_the_llama_cpp_lane_order() {
    use goldens::*;
    let aq = act_quants(&TAIL_ACTS, 0);
    for k in [1usize, 15, 16, 17, 31] {
        let want: i32 = (0..k).map(|i| Q4_0_UNPACK[i] * aq[i] as i32).sum();
        let mut w = Q4_0_UNPACK_W.to_vec();
        // Scale 1.0 so the f32 combination is the integer sum exactly.
        w[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        let got = row_q4_0(&w, &TAIL_ACTS, 1, k);
        assert_eq!(got, want as f32, "tail k={k}");
    }
}

/// Same for Q5_0, whose tail also has to re-derive the `qh` fifth bit
/// scalar-side.
#[test]
fn q5_0_tail_uses_the_llama_cpp_lane_order() {
    use goldens::*;
    let aq = act_quants(&TAIL_ACTS, 0);
    for k in [1usize, 16, 17, 31] {
        let want: i32 = (0..k).map(|i| Q5_0_UNPACK[i] * aq[i] as i32).sum();
        let mut w = Q5_0_UNPACK_W.to_vec();
        w[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        let got = row_q5_0(&w, &TAIL_ACTS, 1, k);
        assert_eq!(got, want as f32, "tail k={k}");
    }
}

// ===========================================================================
// The real intrinsics — SKIPPED unless the host has AVX-512
// ===========================================================================
//
// These are the only assertions in this file that touch the shipped kernels.
// On the aarch64 development host the whole block is `cfg`-ed out; on an x86-64
// host without AVX-512 each test returns early.  A future AVX-512 CI run is
// what turns the model above into a statement about the machine code.

#[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
mod intrinsics {
    use super::*;
    use oxillama_quant::QuantKernel;

    /// `true` when the AVX-512 kernels may be executed here.
    fn avx512() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
    }

    fn check(kernel: &dyn QuantKernel, w: &[u8], a: &[u8], n_cols: usize, want: f32) {
        let mut out = [0.0f32; 1];
        kernel
            .matvec_q8_fused(w, a, &mut out, 1, n_cols)
            .expect("fused matvec");
        assert_eq!(
            out[0].to_bits(),
            want.to_bits(),
            "{} n_cols={n_cols}: intrinsics {} vs model {want}",
            kernel.name(),
            out[0]
        );
    }

    #[test]
    fn q8_0_intrinsics_match_the_model() {
        if !avx512() {
            return; // Skipped: host has no AVX-512.
        }
        use goldens::*;
        let k = oxillama_quant::simd::avx512::Q8_0Avx512;
        check(
            &k,
            &Q8_0_S_W,
            &Q8_0_S_A,
            96,
            row_i8_quants(&Q8_0_S_W, &Q8_0_S_A, 3, 96, 34, 2),
        );
        check(
            &k,
            &Q8_0_S_W,
            &Q8_0_S_A,
            65,
            row_i8_quants(&Q8_0_S_W, &Q8_0_S_A, 3, 65, 34, 2),
        );
    }

    #[test]
    fn q4_0_intrinsics_match_the_model() {
        if !avx512() {
            return; // Skipped: host has no AVX-512.
        }
        use goldens::*;
        let k = oxillama_quant::simd::avx512::Q4_0Avx512;
        check(
            &k,
            &Q4_0_S_W,
            &Q4_0_S_A,
            96,
            row_q4_0(&Q4_0_S_W, &Q4_0_S_A, 3, 96),
        );
        check(
            &k,
            &Q4_0_S_W,
            &Q4_0_S_A,
            33,
            row_q4_0(&Q4_0_S_W, &Q4_0_S_A, 3, 33),
        );
    }

    #[test]
    fn q5_0_intrinsics_match_the_model() {
        if !avx512() {
            return; // Skipped: host has no AVX-512.
        }
        use goldens::*;
        let k = oxillama_quant::simd::avx512::Q5_0Avx512;
        check(
            &k,
            &Q5_0_S_W,
            &Q5_0_S_A,
            96,
            row_q5_0(&Q5_0_S_W, &Q5_0_S_A, 3, 96),
        );
        check(
            &k,
            &Q5_0_S_W,
            &Q5_0_S_A,
            70,
            row_q5_0(&Q5_0_S_W, &Q5_0_S_A, 3, 70),
        );
    }

    #[test]
    fn q5_1_intrinsics_match_the_model() {
        if !avx512() {
            return; // Skipped: host has no AVX-512.
        }
        use goldens::*;
        let k = oxillama_quant::simd::avx512::Q5_1Avx512;
        check(
            &k,
            &Q5_1_B_W,
            &Q5_1_B_A,
            96,
            row_q5_1(&Q5_1_B_W, &Q5_1_B_A, 3, 96),
        );
        check(
            &k,
            &Q5_1_B_W,
            &Q5_1_B_A,
            40,
            row_q5_1(&Q5_1_B_W, &Q5_1_B_A, 3, 40),
        );
    }

    #[test]
    fn q8_1_intrinsics_match_the_model() {
        if !avx512() {
            return; // Skipped: host has no AVX-512.
        }
        use goldens::*;
        // Re-lay the Q8_0 golden blocks as Q8_1 (d, s, 32 x i8).
        let mut w = Vec::new();
        for blk in 0..3 {
            let b = &Q8_0_S_W[blk * 34..(blk + 1) * 34];
            w.extend_from_slice(&b[0..2]);
            w.extend_from_slice(&[0, 0]);
            w.extend_from_slice(&b[2..34]);
        }
        let k = oxillama_quant::simd::avx512::Q8_1Avx512;
        check(
            &k,
            &w,
            &Q8_0_S_A,
            96,
            row_i8_quants(&w, &Q8_0_S_A, 3, 96, 36, 4),
        );
    }

    /// Every AVX-512/AVX2 kernel pair that could differ in the fused path.
    ///
    /// This table is the regression fence for the whole change: `dispatch.rs`
    /// checks AVX-512 *before* AVX2, so any row where AVX-512 falls back to a
    /// trait default that AVX2 overrides is a silent loss of the fused decode
    /// path when the `simd-avx512` feature is switched on.
    fn tiers() -> Vec<(
        &'static str,
        &'static dyn QuantKernel,
        &'static dyn QuantKernel,
    )> {
        use oxillama_quant::simd::{avx2, avx512};
        vec![
            ("Q8_0", &avx512::Q8_0Avx512, &avx2::Q8_0Avx2),
            ("Q8_1", &avx512::Q8_1Avx512, &avx2::Q8_1Avx2),
            ("Q5_0", &avx512::Q5_0Avx512, &avx2::Q5_0Avx2),
            ("Q5_1", &avx512::Q5_1Avx512, &avx2::Q5_1Avx2),
            ("Q4_0", &avx512::Q4_0Avx512, &avx2::Q4_0Avx2),
            ("Q2_K", &avx512::Q2_KAvx512, &avx2::Q2_KAvx2),
            ("Q3_K", &avx512::Q3_KAvx512, &avx2::Q3_KAvx2),
            ("Q4_K", &avx512::Q4_KAvx512, &avx2::Q4_KAvx2),
            ("Q5_K", &avx512::Q5_KAvx512, &avx2::Q5_KAvx2),
            ("Q6_K", &avx512::Q6_KAvx512, &avx2::Q6_KAvx2),
        ]
    }

    /// The gate is the deliverable: the AVX-512 kernels must advertise exactly
    /// the block counts their AVX2 counterparts do, or the fused path stays
    /// switched off and every kernel above is dead code.  This is the assertion
    /// that would have failed before this change for eight formats.
    #[test]
    fn avx512_gates_match_avx2_gates() {
        for (name, a512, a2) in tiers() {
            for k in [1usize, 32, 33, 255, 256, 257, 2560, 9728] {
                assert_eq!(
                    a512.q8_fused_acts_blocks(k),
                    a2.q8_fused_acts_blocks(k),
                    "{name}: AVX-512 gate diverged from AVX2 at K={k}"
                );
            }
        }
    }

    /// Wherever AVX2 overrides `matvec_q8_fused`, AVX-512 must too.  Detected
    /// behaviourally: the trait's default refuses any `block_size() != 32`
    /// kernel with a `KernelError`, so a K-quant that still returns `Err` here
    /// has no override.  Requires AVX2+FMA at runtime because the AVX2 kernels
    /// (and the AVX-512 K-quant delegations) execute AVX2 instructions.
    #[test]
    fn avx512_overrides_matvec_wherever_avx2_does() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
            || !std::arch::is_x86_feature_detected!("avx512f")
        {
            return; // Skipped: host cannot execute one of the two tiers.
        }
        for (name, a512, a2) in tiers() {
            let bs = a2.block_size();
            let n_cols = bs;
            let weights = vec![0u8; a2.block_bytes()];
            let acts = vec![0u8; bs.div_ceil(32) * 34];
            let mut out = [0.0f32; 1];
            let avx2_ok = a2
                .matvec_q8_fused(&weights, &acts, &mut out, 1, n_cols)
                .is_ok();
            let mut out = [0.0f32; 1];
            let avx512_ok = a512
                .matvec_q8_fused(&weights, &acts, &mut out, 1, n_cols)
                .is_ok();
            assert_eq!(
                avx2_ok, avx512_ok,
                "{name}: AVX2 override present={avx2_ok} but AVX-512 override \
                 present={avx512_ok} — the AVX-512 tier must be a superset"
            );
        }
    }

    /// The delegating K-quant kernels must produce exactly what the AVX2
    /// kernels produce — they *are* the AVX2 kernels, so this is bit equality,
    /// and it fails loudly if a delegation is ever pointed at the wrong struct.
    #[test]
    fn k_quant_delegation_is_bit_identical_to_avx2() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
            || !std::arch::is_x86_feature_detected!("avx512f")
        {
            return; // Skipped: host cannot execute one of the two tiers.
        }
        for (name, a512, a2) in tiers() {
            if a2.block_size() != 256 {
                continue;
            }
            let n_cols = 512usize;
            let blocks = 2usize;
            let weights: Vec<u8> = (0..blocks * a2.block_bytes())
                .map(|i| (i as u32 * 61 + 7) as u8)
                .collect();
            let acts: Vec<u8> = (0..blocks * 8 * 34)
                .map(|i| (i as u32 * 29 + 3) as u8)
                .collect();
            let mut want = [0.0f32; 1];
            let mut got = [0.0f32; 1];
            let w = a2.matvec_q8_fused(&weights, &acts, &mut want, 1, n_cols);
            let g = a512.matvec_q8_fused(&weights, &acts, &mut got, 1, n_cols);
            assert_eq!(w.is_ok(), g.is_ok(), "{name}: delegation status differs");
            if w.is_ok() {
                assert_eq!(
                    got[0].to_bits(),
                    want[0].to_bits(),
                    "{name}: delegated result {} != AVX2 {}",
                    got[0],
                    want[0]
                );
            }
        }
    }
}

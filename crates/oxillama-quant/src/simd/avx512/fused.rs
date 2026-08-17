//! AVX-512 fused Q8_0-activation GEMV (`QuantKernel::matvec_q8_fused`) for the
//! 32-weight block formats: Q4_0, Q5_0, Q5_1, Q8_0 and Q8_1.
//!
//! # Why this module exists
//!
//! `dispatch.rs` selects the AVX-512 kernel set *before* the AVX2 one.  Every
//! AVX-512 kernel used to inherit the trait's default `matvec_q8_fused` and
//! return `None` from `q8_fused_acts_blocks`, so switching the `simd-avx512`
//! feature on **removed** the fused decode path that AVX2 already had for eight
//! formats — a silent performance regression, not a gain.  This module supplies
//! the AVX-512 side so the AVX-512 tier is a strict superset of the AVX2 tier.
//!
//! # Numerics
//!
//! Every kernel here keeps a block's dot product in **exact `i32`** and applies
//! one `f32` scale per block, accumulating in block order:
//!
//! ```text
//! row_sum += (d_w * d_a) * (sumi as f32)                    // Q4_0/Q5_0/Q8_0/Q8_1
//! row_sum += (d_w * d_a) * (sumi as f32) + m_w * (d_a * suma as f32)  // Q5_1
//! ```
//!
//! That is exactly the shape of ggml's scalar reference
//! (`ggml_vec_dot_q4_0_q8_0_generic` and siblings in
//! `ggml/src/ggml-cpu/quants.c`), which is what lets
//! `tests/avx512_fused_goldens.rs` assert the integer core against constants
//! produced by *executing* llama.cpp's C.
//!
//! Two consequences worth stating plainly:
//!
//! * **Q8_0** is bit-identical to `Q8_0Avx2`'s fused kernel for *every* `K`,
//!   full blocks and ragged tail alike: both reduce the same exact integer
//!   products and apply one `f32` multiply per block, and AVX2's tail — an
//!   `f32` sum of `i8 × i8` products — is itself exact because every partial
//!   sum stays below `2^24`.
//! * **Q4_0** is bit-identical to `Q4_0Avx2` only when `K % 32 == 0`.  On a
//!   ragged last block AVX2 accumulates `row_sum += scale * (q*a) as f32`
//!   once *per element* while this kernel applies the scale once to the whole
//!   block's `i32` sum; the values agree to within one rounding, and the
//!   single-multiply form here is the more accurate one.  Do not write a
//!   cross-tier bit-equality test for Q4_0 at `K = 33` or `K = 257`.
//! * For Q5_0, Q5_1 and Q8_1 the AVX2 kernels convert each quant to `f32`
//!   first and accumulate with FMA; these AVX-512 kernels do not, so results
//!   can differ in the last bit(s).  Where they differ, the integer path is
//!   the more accurate one and is the one ggml itself uses.  Nothing in the
//!   decode path requires cross-tier bit-identity — the per-row parallel/serial
//!   bit-identity that `tests/gemv_parity.rs` does require is preserved,
//!   because rows never share an accumulator.
//!
//! # Tails
//!
//! `n_cols` need not be a multiple of 32.  The last block of a row is handled
//! by a scalar loop over the in-bounds lanes only — never by masking a
//! full-width load — so a caller that leaves garbage past `n_cols` in the
//! activation buffer cannot corrupt the result.  The scalar tail computes the
//! same `i32` sums and applies the same `f32` combination as the vector path.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::int_dot::{expand_qh_bits_16, have_avx512bw, Base, Bw, Int8Lanes};
use crate::simd::avx512::util::f16_to_f32;

/// Bytes per Q8_0 activation block: 2-byte FP16 scale + 32 × i8.
pub(crate) const Q8_0_ACT_BYTES: usize = 34;
/// Weights per block for every format in this module.
const BLOCK_SIZE: usize = 32;

/// Signature shared by every row kernel below:
/// `(row_weight_bytes, all_activation_bytes, blocks_per_row, n_cols) -> row dot`.
type RowFn = unsafe fn(&[u8], &[u8], usize, usize) -> f32;

// ---------------------------------------------------------------------------
// Shared plumbing
// ---------------------------------------------------------------------------

/// One format's row kernel in both AVX-512 tiers, plus its block stride.
///
/// Bundling them keeps the per-format entry points to a single line each and
/// puts the validation, the tier choice and the row loop in exactly one place —
/// so a future format cannot accidentally skip a bounds check or pick the BW
/// kernel without consulting `CPUID`.
#[derive(Clone, Copy)]
struct RowKernel {
    /// Requires `avx512f` **and** `avx512bw`.
    bw: RowFn,
    /// Requires `avx512f` only; bit-identical results, more instructions.
    base: RowFn,
    /// Bytes per weight block of this format.
    block_bytes: usize,
}

impl RowKernel {
    /// Validate, pick the tier, and accumulate one row per `out` slot.
    ///
    /// The checks and their error variants mirror the AVX2 kernels exactly, so
    /// a caller cannot tell the tiers apart by their failure modes.
    ///
    /// Rows are driven through [`crate::parallel::for_each_row`], which may
    /// spread them over threads but never splits a row — the per-row
    /// accumulation order, and therefore the result, is bit-identical to the
    /// serial loop.  (The AVX2 fused kernels still use a plain serial loop;
    /// this is one of the ways the AVX-512 tier is a superset rather than a
    /// copy.)
    fn matvec(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }

        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let row_bytes = blocks_per_row * self.block_bytes;
        let acts_needed = blocks_per_row * Q8_0_ACT_BYTES;

        if weights.len() < n_rows * row_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < acts_needed {
            return Err(QuantError::BufferTooSmall {
                needed: acts_needed,
                available: acts_q8.len(),
            });
        }

        let row_fn = if have_avx512bw() { self.bw } else { self.base };

        crate::parallel::for_each_row(out, n_rows, n_cols, |row, slot| {
            let start = row * row_bytes;
            // SAFETY: the length checks above cover `weights[start..][..row_bytes]`
            // and the activation buffer; `row_fn` was chosen from the cached
            // `CPUID` bits, and this kernel is only constructed after
            // `KernelDispatcher` confirmed `avx512f`.
            *slot += unsafe {
                row_fn(
                    &weights[start..start + row_bytes],
                    acts_q8,
                    blocks_per_row,
                    n_cols,
                )
            };
        });

        Ok(())
    }
}

/// FP16 scale of an activation block, and a pointer to its 32 `i8` quants.
///
/// # Safety
/// `a_block.len() >= Q8_0_ACT_BYTES`.
#[inline(always)]
unsafe fn act_block(a_block: &[u8]) -> (f32, *const i8) {
    (f16_to_f32(a_block), a_block.as_ptr().add(2) as *const i8)
}

// ---------------------------------------------------------------------------
// Q8_0 weights × Q8_0 activations
// ---------------------------------------------------------------------------

/// Q8_0 block: 2-byte FP16 scale + 32 × i8. Weight `i` is `q[i] * d`.
const Q8_0_BLOCK_BYTES: usize = 34;

/// One row of the Q8_0 × Q8_0 fused GEMV.
///
/// Scalar model: `tests/avx512_fused_goldens.rs::model::row_q8_0`.
///
/// # Safety
/// * `row.len() >= blocks_per_row * 34`, `acts.len() >= blocks_per_row * 34`.
/// * Caller's CPU must have the features `O` documents.
#[inline(always)]
unsafe fn row_q8_0_impl<O: Int8Lanes>(
    row: &[u8],
    acts: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    row_i8_quants_impl::<O>(row, acts, blocks_per_row, n_cols, Q8_0_BLOCK_BYTES, 2)
}

/// Shared body of the two "weights are already `i8`" formats (Q8_0 and Q8_1);
/// they differ only in block stride and the byte offset of the quants.
///
/// # Safety
/// See [`row_q8_0_impl`].
#[inline(always)]
unsafe fn row_i8_quants_impl<O: Int8Lanes>(
    row: &[u8],
    acts: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
    block_bytes: usize,
    qs_offset: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let w_block = &row[blk * block_bytes..(blk + 1) * block_bytes];
        let a_block = &acts[blk * Q8_0_ACT_BYTES..(blk + 1) * Q8_0_ACT_BYTES];

        let d_w = f16_to_f32(w_block);
        let (d_a, a_ptr) = act_block(a_block);
        let scale = d_w * d_a;

        let valid = n_cols.saturating_sub(blk * BLOCK_SIZE).min(BLOCK_SIZE);
        if valid == BLOCK_SIZE {
            let wv = _mm256_loadu_si256(w_block.as_ptr().add(qs_offset) as *const __m256i);
            let av = _mm256_loadu_si256(a_ptr as *const __m256i);
            row_sum += scale * O::dot32_i8(wv, av) as f32;
        } else if valid > 0 {
            let qs = w_block.as_ptr().add(qs_offset) as *const i8;
            let mut sumi = 0i32;
            for i in 0..valid {
                sumi += (*qs.add(i)) as i32 * (*a_ptr.add(i)) as i32;
            }
            row_sum += scale * sumi as f32;
        }
    }

    row_sum
}

/// AVX-512BW variant of [`row_q8_0_impl`].
///
/// # Safety
/// CPU must support `avx512f` **and** `avx512bw`; slice lengths as documented
/// on [`row_q8_0_impl`].
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn row_q8_0_bw(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q8_0_impl::<Bw>(row, acts, blocks_per_row, n_cols)
}

/// AVX-512F-only variant of [`row_q8_0_impl`].
///
/// # Safety
/// CPU must support `avx512f`; slice lengths as documented on [`row_q8_0_impl`].
#[target_feature(enable = "avx512f")]
unsafe fn row_q8_0_base(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q8_0_impl::<Base>(row, acts, blocks_per_row, n_cols)
}

/// Fused Q8_0 weight × Q8_0 activation GEMV, AVX-512.
///
/// Accumulates into `out` (the caller zeroes it when a fresh GEMV is wanted),
/// matching [`crate::traits::QuantKernel::matvec_q8_fused`]'s contract.
pub(crate) fn matvec_q8_0(
    weights: &[u8],
    acts_q8: &[u8],
    out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) -> QuantResult<()> {
    RowKernel {
        bw: row_q8_0_bw,
        base: row_q8_0_base,
        block_bytes: Q8_0_BLOCK_BYTES,
    }
    .matvec(weights, acts_q8, out, n_rows, n_cols)
}

// ---------------------------------------------------------------------------
// Q8_1 weights × Q8_0 activations
// ---------------------------------------------------------------------------

/// Q8_1 block: FP16 `d`, FP16 `s`, then 32 × i8.  `s` is the pre-summed
/// `d * Σq` ggml uses when *both* operands are Q8_1; this crate's Q8_1 GEMV and
/// reference kernel reconstruct `w = d * q` and ignore `s`, so the fused kernel
/// does too.
const Q8_1_BLOCK_BYTES: usize = 36;

/// One row of the Q8_1 × Q8_0 fused GEMV.
///
/// Scalar model: `tests/avx512_fused_goldens.rs::model::row_q8_1`.
///
/// # Safety
/// * `row.len() >= blocks_per_row * 36`, `acts.len() >= blocks_per_row * 34`.
/// * Caller's CPU must have the features `O` documents.
#[inline(always)]
unsafe fn row_q8_1_impl<O: Int8Lanes>(
    row: &[u8],
    acts: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    row_i8_quants_impl::<O>(row, acts, blocks_per_row, n_cols, Q8_1_BLOCK_BYTES, 4)
}

/// AVX-512BW variant of [`row_q8_1_impl`].
///
/// # Safety
/// CPU must support `avx512f` **and** `avx512bw`; slice lengths as documented
/// on [`row_q8_1_impl`].
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn row_q8_1_bw(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q8_1_impl::<Bw>(row, acts, blocks_per_row, n_cols)
}

/// AVX-512F-only variant of [`row_q8_1_impl`].
///
/// # Safety
/// CPU must support `avx512f`; slice lengths as documented on [`row_q8_1_impl`].
#[target_feature(enable = "avx512f")]
unsafe fn row_q8_1_base(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q8_1_impl::<Base>(row, acts, blocks_per_row, n_cols)
}

/// Fused Q8_1 weight × Q8_0 activation GEMV, AVX-512.
pub(crate) fn matvec_q8_1(
    weights: &[u8],
    acts_q8: &[u8],
    out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) -> QuantResult<()> {
    RowKernel {
        bw: row_q8_1_bw,
        base: row_q8_1_base,
        block_bytes: Q8_1_BLOCK_BYTES,
    }
    .matvec(weights, acts_q8, out, n_rows, n_cols)
}

// ---------------------------------------------------------------------------
// Q4_0 weights × Q8_0 activations
// ---------------------------------------------------------------------------

/// Q4_0 block: 2-byte FP16 scale + 16 packed nibble bytes.
const Q4_0_BLOCK_BYTES: usize = 18;

/// Unpack the 16 nibble bytes of a Q4_0 block into 32 lanes of `q - 8`.
///
/// ggml's Q4_0 uses the **split-half** layout: `qs[j] & 0x0F` is weight `j`
/// (`j < 16`) and `qs[j] >> 4` is weight `j + 16`.  Concatenating the low
/// nibbles below the high nibbles therefore yields the 32 weights already in
/// order — no lane permutation, which is why `_mm256_set_m128i` suffices.
///
/// Scalar model: `tests/avx512_fused_goldens.rs::model::unpack_q4_0`.
///
/// # Safety
/// `qs` must point at 16 readable bytes; CPU must support `avx512f` (which
/// implies AVX2 in LLVM's feature graph — see `int_dot.rs`'s module doc).
#[inline(always)]
unsafe fn unpack_q4_0(qs: *const u8) -> __m256i {
    let raw = _mm_loadu_si128(qs as *const __m128i);
    let mask = _mm_set1_epi8(0x0F);
    // Low nibbles are weights 0..16; high nibbles are weights 16..32.
    let lo = _mm_and_si128(raw, mask);
    let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
    // `q - 8` lands in -8..=7, so the subtraction is exact in i8 lanes.
    _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8))
}

/// Scalar twin of [`unpack_q4_0`] for the ragged last block.
#[inline(always)]
fn q4_0_weight(qs: &[u8], i: usize) -> i32 {
    if i < 16 {
        (qs[i] & 0x0F) as i32 - 8
    } else {
        (qs[i - 16] >> 4) as i32 - 8
    }
}

/// One row of the Q4_0 × Q8_0 fused GEMV.
///
/// Scalar model: `tests/avx512_fused_goldens.rs::model::row_q4_0`.
///
/// # Safety
/// * `row.len() >= blocks_per_row * 18`, `acts.len() >= blocks_per_row * 34`.
/// * Caller's CPU must have the features `O` documents.
#[inline(always)]
unsafe fn row_q4_0_impl<O: Int8Lanes>(
    row: &[u8],
    acts: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let w_block = &row[blk * Q4_0_BLOCK_BYTES..(blk + 1) * Q4_0_BLOCK_BYTES];
        let a_block = &acts[blk * Q8_0_ACT_BYTES..(blk + 1) * Q8_0_ACT_BYTES];

        let d_w = f16_to_f32(w_block);
        let (d_a, a_ptr) = act_block(a_block);
        let scale = d_w * d_a;

        let valid = n_cols.saturating_sub(blk * BLOCK_SIZE).min(BLOCK_SIZE);
        if valid == BLOCK_SIZE {
            let wv = unpack_q4_0(w_block.as_ptr().add(2));
            let av = _mm256_loadu_si256(a_ptr as *const __m256i);
            row_sum += scale * O::dot32_i8(wv, av) as f32;
        } else if valid > 0 {
            let qs = &w_block[2..];
            let mut sumi = 0i32;
            for i in 0..valid {
                sumi += q4_0_weight(qs, i) * (*a_ptr.add(i)) as i32;
            }
            row_sum += scale * sumi as f32;
        }
    }

    row_sum
}

/// AVX-512BW variant of [`row_q4_0_impl`].
///
/// # Safety
/// CPU must support `avx512f` **and** `avx512bw`; slice lengths as documented
/// on [`row_q4_0_impl`].
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn row_q4_0_bw(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q4_0_impl::<Bw>(row, acts, blocks_per_row, n_cols)
}

/// AVX-512F-only variant of [`row_q4_0_impl`].
///
/// # Safety
/// CPU must support `avx512f`; slice lengths as documented on [`row_q4_0_impl`].
#[target_feature(enable = "avx512f")]
unsafe fn row_q4_0_base(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q4_0_impl::<Base>(row, acts, blocks_per_row, n_cols)
}

/// Fused Q4_0 weight × Q8_0 activation GEMV, AVX-512.
pub(crate) fn matvec_q4_0(
    weights: &[u8],
    acts_q8: &[u8],
    out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) -> QuantResult<()> {
    RowKernel {
        bw: row_q4_0_bw,
        base: row_q4_0_base,
        block_bytes: Q4_0_BLOCK_BYTES,
    }
    .matvec(weights, acts_q8, out, n_rows, n_cols)
}

// ---------------------------------------------------------------------------
// Q5_0 weights × Q8_0 activations
// ---------------------------------------------------------------------------

/// Q5_0 block: FP16 `d`, 4-byte `qh` bit plane, 16 packed nibble bytes.
const Q5_0_BLOCK_BYTES: usize = 22;

/// Unpack a Q5_0/Q5_1 block's nibbles and `qh` plane into 32 lanes of the
/// unsigned 5-bit quant `q ∈ 0..=31`, in weight order.
///
/// Split-half layout again: nibble `j` low → weight `j`, nibble `j` high →
/// weight `j + 16`, with `qh` bit `j` supplying weight `j`'s bit 4 and `qh`
/// bit `j + 16` supplying weight `j + 16`'s.
///
/// Scalar model: `tests/avx512_fused_goldens.rs::model::unpack_q5`.
///
/// # Safety
/// `qs` must point at 16 readable bytes; CPU must support `avx512f`.
#[inline(always)]
unsafe fn unpack_q5(qs: *const u8, qh: u32) -> __m256i {
    let raw = _mm_loadu_si128(qs as *const __m128i);
    let mask = _mm_set1_epi8(0x0F);
    let lo_nib = _mm_and_si128(raw, mask);
    let hi_nib = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
    // Bit j of `qh` -> byte j of the low half; bit j+16 -> byte j of the high
    // half.  `expand_qh_bits_16` yields 0x10 / 0x00 per byte.
    let lo = _mm_or_si128(lo_nib, expand_qh_bits_16(qh as u16));
    let hi = _mm_or_si128(hi_nib, expand_qh_bits_16((qh >> 16) as u16));
    _mm256_set_m128i(hi, lo)
}

/// Scalar twin of [`unpack_q5`] for the ragged last block: the unsigned 5-bit
/// quant of weight `i`.
#[inline(always)]
fn q5_quant(qs: &[u8], qh: u32, i: usize) -> i32 {
    if i < 16 {
        ((qs[i] & 0x0F) | ((((qh >> i) & 1) as u8) << 4)) as i32
    } else {
        let j = i - 16;
        ((qs[j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)) as i32
    }
}

/// One row of the Q5_0 × Q8_0 fused GEMV.  `w = d * (q - 16)`.
///
/// Scalar model: `tests/avx512_fused_goldens.rs::model::row_q5_0`.
///
/// # Safety
/// * `row.len() >= blocks_per_row * 22`, `acts.len() >= blocks_per_row * 34`.
/// * Caller's CPU must have the features `O` documents.
#[inline(always)]
unsafe fn row_q5_0_impl<O: Int8Lanes>(
    row: &[u8],
    acts: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let w_block = &row[blk * Q5_0_BLOCK_BYTES..(blk + 1) * Q5_0_BLOCK_BYTES];
        let a_block = &acts[blk * Q8_0_ACT_BYTES..(blk + 1) * Q8_0_ACT_BYTES];

        let d_w = f16_to_f32(w_block);
        let qh = u32::from_le_bytes([w_block[2], w_block[3], w_block[4], w_block[5]]);
        let (d_a, a_ptr) = act_block(a_block);
        let scale = d_w * d_a;

        let valid = n_cols.saturating_sub(blk * BLOCK_SIZE).min(BLOCK_SIZE);
        if valid == BLOCK_SIZE {
            // `q - 16` lands in -16..=15, exact in i8 lanes.
            let wv = _mm256_sub_epi8(unpack_q5(w_block.as_ptr().add(6), qh), _mm256_set1_epi8(16));
            let av = _mm256_loadu_si256(a_ptr as *const __m256i);
            row_sum += scale * O::dot32_i8(wv, av) as f32;
        } else if valid > 0 {
            let qs = &w_block[6..];
            let mut sumi = 0i32;
            for i in 0..valid {
                sumi += (q5_quant(qs, qh, i) - 16) * (*a_ptr.add(i)) as i32;
            }
            row_sum += scale * sumi as f32;
        }
    }

    row_sum
}

/// AVX-512BW variant of [`row_q5_0_impl`].
///
/// # Safety
/// CPU must support `avx512f` **and** `avx512bw`; slice lengths as documented
/// on [`row_q5_0_impl`].
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn row_q5_0_bw(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q5_0_impl::<Bw>(row, acts, blocks_per_row, n_cols)
}

/// AVX-512F-only variant of [`row_q5_0_impl`].
///
/// # Safety
/// CPU must support `avx512f`; slice lengths as documented on [`row_q5_0_impl`].
#[target_feature(enable = "avx512f")]
unsafe fn row_q5_0_base(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q5_0_impl::<Base>(row, acts, blocks_per_row, n_cols)
}

/// Fused Q5_0 weight × Q8_0 activation GEMV, AVX-512.
pub(crate) fn matvec_q5_0(
    weights: &[u8],
    acts_q8: &[u8],
    out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) -> QuantResult<()> {
    RowKernel {
        bw: row_q5_0_bw,
        base: row_q5_0_base,
        block_bytes: Q5_0_BLOCK_BYTES,
    }
    .matvec(weights, acts_q8, out, n_rows, n_cols)
}

// ---------------------------------------------------------------------------
// Q5_1 weights × Q8_0 activations
// ---------------------------------------------------------------------------

/// Q5_1 block: FP16 `d`, FP16 `m`, 4-byte `qh` plane, 16 packed nibble bytes.
const Q5_1_BLOCK_BYTES: usize = 24;

/// One row of the Q5_1 × Q8_0 fused GEMV.  `w = d * q + m`, so the block
/// contributes `(d·d_a)·Σ(q·a) + m·(d_a·Σa)` — the same decomposition
/// `ggml_vec_dot_q5_1_q8_1_generic` uses, with `d_a·Σa` computed exactly in
/// f32 here instead of being read back from an FP16 `s` field.
///
/// Scalar model: `tests/avx512_fused_goldens.rs::model::row_q5_1`.
///
/// # Safety
/// * `row.len() >= blocks_per_row * 24`, `acts.len() >= blocks_per_row * 34`.
/// * Caller's CPU must have the features `O` documents.
#[inline(always)]
unsafe fn row_q5_1_impl<O: Int8Lanes>(
    row: &[u8],
    acts: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let w_block = &row[blk * Q5_1_BLOCK_BYTES..(blk + 1) * Q5_1_BLOCK_BYTES];
        let a_block = &acts[blk * Q8_0_ACT_BYTES..(blk + 1) * Q8_0_ACT_BYTES];

        let d_w = f16_to_f32(w_block);
        let m_w = f16_to_f32(&w_block[2..]);
        let qh = u32::from_le_bytes([w_block[4], w_block[5], w_block[6], w_block[7]]);
        let (d_a, a_ptr) = act_block(a_block);

        let valid = n_cols.saturating_sub(blk * BLOCK_SIZE).min(BLOCK_SIZE);
        let (sumi, suma) = if valid == BLOCK_SIZE {
            let wv = unpack_q5(w_block.as_ptr().add(8), qh);
            let av = _mm256_loadu_si256(a_ptr as *const __m256i);
            (O::dot32_u8_i8(wv, av), O::sum32_i8(av))
        } else if valid > 0 {
            let qs = &w_block[8..];
            let mut sumi = 0i32;
            let mut suma = 0i32;
            for i in 0..valid {
                let a = (*a_ptr.add(i)) as i32;
                sumi += q5_quant(qs, qh, i) * a;
                suma += a;
            }
            (sumi, suma)
        } else {
            (0, 0)
        };

        row_sum += (d_w * d_a) * sumi as f32 + m_w * (d_a * suma as f32);
    }

    row_sum
}

/// AVX-512BW variant of [`row_q5_1_impl`].
///
/// # Safety
/// CPU must support `avx512f` **and** `avx512bw`; slice lengths as documented
/// on [`row_q5_1_impl`].
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn row_q5_1_bw(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q5_1_impl::<Bw>(row, acts, blocks_per_row, n_cols)
}

/// AVX-512F-only variant of [`row_q5_1_impl`].
///
/// # Safety
/// CPU must support `avx512f`; slice lengths as documented on [`row_q5_1_impl`].
#[target_feature(enable = "avx512f")]
unsafe fn row_q5_1_base(row: &[u8], acts: &[u8], blocks_per_row: usize, n_cols: usize) -> f32 {
    row_q5_1_impl::<Base>(row, acts, blocks_per_row, n_cols)
}

/// Fused Q5_1 weight × Q8_0 activation GEMV, AVX-512.
pub(crate) fn matvec_q5_1(
    weights: &[u8],
    acts_q8: &[u8],
    out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) -> QuantResult<()> {
    RowKernel {
        bw: row_q5_1_bw,
        base: row_q5_1_base,
        block_bytes: Q5_1_BLOCK_BYTES,
    }
    .matvec(weights, acts_q8, out, n_rows, n_cols)
}

/// Number of Q8_0 activation blocks a 32-weight format consumes for `n_cols`.
///
/// Identical to what the AVX2 kernels advertise, which is the point: the
/// dispatch gate must not change when the AVX-512 tier is selected.
#[inline]
pub(crate) fn acts_blocks_32(n_cols: usize) -> Option<usize> {
    Some(n_cols.div_ceil(BLOCK_SIZE))
}

// ---------------------------------------------------------------------------
// K-quant delegation to the AVX2 tier
// ---------------------------------------------------------------------------

/// Weights per K-quant block.
const K_BLOCK_SIZE: usize = 256;
/// Q8_0 activation blocks a K-quant block consumes: `256 / 32`.
const K_ACT_BLOCKS: usize = 8;

/// Gate for a K-quant kernel whose fused body is the AVX2 one: `ceil(K/256)*8`,
/// exactly what `Q2_KAvx2`/`Q3_KAvx2`/`Q4_KAvx2`/`Q6_KAvx2` advertise.
///
/// Unconditional, precisely because the AVX2 gate is unconditional: the two
/// must be indistinguishable, and a test pins that
/// (`avx512_gates_match_avx2_gates`).  The soundness check for the
/// architecturally-impossible "AVX-512F without AVX2+FMA" CPU lives in
/// [`delegate_to_avx2`], which refuses with an error rather than executing
/// instructions the CPU lacks.
#[inline]
pub(crate) fn delegated_acts_blocks_k(n_cols: usize) -> Option<usize> {
    Some(n_cols.div_ceil(K_BLOCK_SIZE) * K_ACT_BLOCKS)
}

/// Run an AVX2 fused kernel on behalf of its AVX-512 sibling.
///
/// This is an **explicit delegation**, not a fallthrough: `dispatch.rs` hands
/// K-quant tensors to the AVX-512 kernel, and without this the fused decode
/// path that AVX2 has would simply vanish whenever `simd-avx512` is enabled.
/// Every AVX-512F CPU also implements AVX2+FMA (LLVM even encodes the
/// implication), so delegating is never slower than the AVX2 tier the user
/// would otherwise have got; it just does not add 512-bit width on top.
///
/// The `avx2_usable()` check keeps the delegation *sound* rather than merely
/// probable: the AVX2 kernels are `#[target_feature(enable = "avx2,fma")]`, and
/// those are independently reported `CPUID` bits.
pub(crate) fn delegate_to_avx2(
    avx2_kernel: &dyn crate::traits::QuantKernel,
    weights: &[u8],
    acts_q8: &[u8],
    out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) -> QuantResult<()> {
    if !crate::simd::cached_capabilities().avx2_usable() {
        return Err(QuantError::KernelError {
            message: format!(
                "the AVX-512 {} kernel delegates matvec_q8_fused to the AVX2 kernel, \
                 but this CPU reports avx512f without avx2+fma",
                avx2_kernel.name()
            ),
        });
    }
    avx2_kernel.matvec_q8_fused(weights, acts_q8, out, n_rows, n_cols)
}

// ---------------------------------------------------------------------------
// Tests — these execute AVX-512 instructions and are therefore SKIPPED on any
// host without AVX-512F (including every aarch64 machine, where the whole
// module is `cfg`-ed out).  A future x86-64 CI run with an AVX-512 host is what
// exercises them; see the crate README's verification-status section.
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
mod tests {
    use super::*;
    use crate::reference::{Q4_0Ref, Q5_0Ref, Q5_1Ref, Q8_0Ref, Q8_1Ref};
    use crate::traits::QuantKernel;

    /// Build one Q8_0 activation block from `scale` and 32 quants.
    fn act(scale: f32, q: &[i8; 32]) -> Vec<u8> {
        let mut b = half::f16::from_f32(scale).to_bits().to_le_bytes().to_vec();
        b.extend(q.iter().map(|&v| v as u8));
        b
    }

    fn ramp() -> [i8; 32] {
        core::array::from_fn(|i| (i as i32 * 7 - 100) as i8)
    }

    /// Dequantize the weights and dot them with the dequantized activations —
    /// the definition the fused kernels approximate.
    fn expected(kernel: &dyn QuantKernel, w: &[u8], a: &[u8], n_cols: usize) -> f32 {
        let bs = kernel.block_size();
        let bb = kernel.block_bytes();
        let mut scratch = vec![0.0f32; bs];
        let mut sum = 0.0f32;
        for blk in 0..n_cols.div_ceil(bs) {
            kernel
                .dequant_block(&w[blk * bb..(blk + 1) * bb], &mut scratch)
                .expect("dequant");
            let ab = &a[blk * Q8_0_ACT_BYTES..(blk + 1) * Q8_0_ACT_BYTES];
            let d_a = half::f16::from_bits(u16::from_le_bytes([ab[0], ab[1]])).to_f32();
            for i in 0..(n_cols - blk * bs).min(bs) {
                sum += scratch[i] * (ab[2 + i] as i8 as f32 * d_a);
            }
        }
        sum
    }

    /// A `matvec_*` entry point of this module.
    type MatvecFn = fn(&[u8], &[u8], &mut [f32], usize, usize) -> QuantResult<()>;

    fn run(f: MatvecFn, reference: &dyn QuantKernel, w: &[u8], a: &[u8], n_cols: usize) {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return; // Skipped: host has no AVX-512.
        }
        let mut out = vec![0.0f32; 1];
        f(w, a, &mut out, 1, n_cols).expect("fused matvec");
        let want = expected(reference, w, a, n_cols);
        let tol = want.abs().max(1.0) * 1e-4;
        assert!(
            (out[0] - want).abs() <= tol,
            "fused={} expected={} n_cols={n_cols}",
            out[0],
            want
        );
    }

    fn weights(bytes: usize, blocks: usize, scale_lanes: &[(usize, f32)]) -> Vec<u8> {
        let mut w = vec![0u8; bytes * blocks];
        for (i, b) in w.iter_mut().enumerate() {
            *b = (i as u32 * 37 + 11) as u8;
        }
        for blk in 0..blocks {
            for &(off, v) in scale_lanes {
                let bits = half::f16::from_f32(v).to_bits().to_le_bytes();
                w[blk * bytes + off] = bits[0];
                w[blk * bytes + off + 1] = bits[1];
            }
        }
        w
    }

    #[test]
    fn q8_0_fused_matches_dequantized_dot() {
        let w = weights(34, 2, &[(0, 0.125)]);
        let a = [act(0.25, &ramp()), act(-0.5, &ramp())].concat();
        run(matvec_q8_0, &Q8_0Ref, &w, &a, 64);
        run(matvec_q8_0, &Q8_0Ref, &w, &a, 33);
    }

    #[test]
    fn q8_1_fused_matches_dequantized_dot() {
        let w = weights(36, 2, &[(0, 0.125), (2, 0.0)]);
        let a = [act(0.25, &ramp()), act(-0.5, &ramp())].concat();
        run(matvec_q8_1, &Q8_1Ref, &w, &a, 64);
        run(matvec_q8_1, &Q8_1Ref, &w, &a, 33);
    }

    #[test]
    fn q4_0_fused_matches_dequantized_dot() {
        let w = weights(18, 2, &[(0, 0.125)]);
        let a = [act(0.25, &ramp()), act(-0.5, &ramp())].concat();
        run(matvec_q4_0, &Q4_0Ref, &w, &a, 64);
        run(matvec_q4_0, &Q4_0Ref, &w, &a, 33);
    }

    #[test]
    fn q5_0_fused_matches_dequantized_dot() {
        let w = weights(22, 2, &[(0, 0.125)]);
        let a = [act(0.25, &ramp()), act(-0.5, &ramp())].concat();
        run(matvec_q5_0, &Q5_0Ref, &w, &a, 64);
        run(matvec_q5_0, &Q5_0Ref, &w, &a, 33);
    }

    #[test]
    fn q5_1_fused_matches_dequantized_dot() {
        let w = weights(24, 2, &[(0, 0.125), (2, -0.75)]);
        let a = [act(0.25, &ramp()), act(-0.5, &ramp())].concat();
        run(matvec_q5_1, &Q5_1Ref, &w, &a, 64);
        run(matvec_q5_1, &Q5_1Ref, &w, &a, 33);
    }

    /// The AVX-512F-only tier must return the same `i32` as the BW tier.  Runs
    /// both explicitly rather than trusting the runtime pick, so a BW host
    /// still exercises the fallback.
    #[test]
    fn bw_and_base_tiers_agree() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return; // Skipped: host has no AVX-512.
        }
        let w = weights(18, 3, &[(0, 0.125)]);
        let a = [act(0.25, &ramp()), act(-0.5, &ramp()), act(1.5, &ramp())].concat();
        // SAFETY: `avx512f` confirmed above; the BW variant additionally needs
        // `avx512bw`, checked before it is called.
        let base = unsafe { row_q4_0_base(&w, &a, 3, 96) };
        if std::arch::is_x86_feature_detected!("avx512bw") {
            let bw = unsafe { row_q4_0_bw(&w, &a, 3, 96) };
            assert_eq!(bw.to_bits(), base.to_bits(), "BW and base tiers diverged");
        }
    }
}

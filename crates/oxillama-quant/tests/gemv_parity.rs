//! GEMV parity tests at the shapes an actual decode step touches.
//!
//! Two independent properties are checked for every quantization type in
//! [`PARITY_TYPES`] — the two that dominate a Qwen3-4B `Q4_K_M` forward pass
//! (Q4_K for every projection, Q6_K for `attn_v`/`ffn_down` and the tied
//! `token_embd` LM head) plus the seven whose nibble/bit layouts were realigned
//! with GGML (Q4_0, Q4_1, Q5_K, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0):
//!
//! 1. **Dispatched vs reference** — the kernel [`KernelDispatcher`] actually
//!    selects (NEON on aarch64, AVX2 on x86_64, scalar elsewhere) must agree
//!    with the scalar reference to within fp32 accumulation-order noise.
//! 2. **Parallel vs serial, bit-exact** — [`parallel::for_each_row`] splits the
//!    output rows across threads but never splits a row, so the multi-threaded
//!    result must be *bit-identical* to evaluating each row on its own.  Row
//!    counts deliberately include values that are not multiples of the SIMD
//!    lane count, the 256-weight block size, or the thread count.

use half::f16;
use oxillama_gguf::GgufTensorType;
use oxillama_quant::reference::{
    Iq4NlRef, Iq4XsRef, Q4KRef, Q4_0Ref, Q4_1Ref, Q5KRef, Q6KRef, Tq1_0Ref, Tq2_0Ref,
};
use oxillama_quant::{KernelDispatcher, QuantKernel, QuantTensor};

/// Weights per Q4_K / Q6_K block.
///
/// The fused and batched Q8-activation paths below are specific to the
/// 256-weight K-quants, so they keep using this constant directly; the
/// dispatch/parallel parity sweep uses [`block_size`] instead.
const K_BLOCK: usize = 256;
/// Bytes per Q4_K block: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs).
const Q4_K_BYTES: usize = 144;
/// Bytes per Q6_K block: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d).
const Q6_K_BYTES: usize = 210;
/// Bytes per Q2_K block: 16 (scales) + 64 (qs) + 2 (d) + 2 (dmin).
const Q2_K_BYTES: usize = 84;
/// Bytes per Q3_K block: 32 (hmask) + 64 (qs) + 12 (scales) + 2 (d).
const Q3_K_BYTES: usize = 110;

/// Every type covered by the dispatched-vs-reference and parallel-vs-serial
/// parity sweep.
///
/// The seven non-K-quant entries are exactly the formats whose layouts were
/// realigned with GGML.  Without them the sweep could not detect a tier that
/// still decodes with the old convention: the per-format unit tests inside
/// `simd/*` mostly compare against the *same* reference, but they do not cover
/// the partial-block, wide-K, and parallel-split shapes this file does.
const PARITY_TYPES: &[GgufTensorType] = &[
    GgufTensorType::Q4K,
    GgufTensorType::Q6K,
    GgufTensorType::Q4_0,
    GgufTensorType::Q4_1,
    GgufTensorType::Q5K,
    GgufTensorType::Iq4Nl,
    GgufTensorType::Iq4Xs,
    GgufTensorType::Tq1_0,
    GgufTensorType::Tq2_0,
];

/// Row counts exercised by the parity sweep.
///
/// `1`/`2`/`7` are below every parallel threshold; `63`/`65` straddle
/// [`oxillama_quant::parallel::PARALLEL_ROW_THRESHOLD`]; `1023`/`1025` are
/// coprime with the 4- and 8-lane NEON/AVX2 widths and with any plausible
/// thread count; `1024`/`2560` are real Qwen3-4B projection widths.
const ROW_COUNTS: &[usize] = &[1, 2, 7, 63, 65, 1023, 1024, 1025, 2560];

/// Deterministic xorshift64* stream — keeps the tests reproducible without
/// pulling a PRNG dependency into `oxillama-quant`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }

    /// Uniform in `[-1, 1)`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 8_388_608.0 - 1.0
    }
}

/// Build `n_blocks` well-formed Q4_K blocks with pseudo-random payloads.
///
/// Every byte pattern is a legal Q4_K block, so random `scales`/`qs` are a
/// valid — and unusually harsh — parity input: they exercise all 16 nibble
/// values and the full 6-bit scale/min packing.  `d` and `dmin` are kept in a
/// realistic range so the accumulated dot products do not saturate fp32.
fn make_q4_k(n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
    let mut data = Vec::with_capacity(n_blocks * Q4_K_BYTES);
    for _ in 0..n_blocks {
        let d = f16::from_f32(rng.next_f32() * 0.05);
        let dmin = f16::from_f32(rng.next_f32() * 0.02);
        data.extend_from_slice(&d.to_bits().to_le_bytes());
        data.extend_from_slice(&dmin.to_bits().to_le_bytes());
        for _ in 0..12 + 128 {
            data.push(rng.next_u8());
        }
    }
    data
}

/// Build `n_blocks` well-formed Q6_K blocks with pseudo-random payloads.
fn make_q6_k(n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
    let mut data = Vec::with_capacity(n_blocks * Q6_K_BYTES);
    for _ in 0..n_blocks {
        // ql (128) + qh (64) + scales (16)
        for _ in 0..128 + 64 + 16 {
            data.push(rng.next_u8());
        }
        let d = f16::from_f32(rng.next_f32() * 0.02);
        data.extend_from_slice(&d.to_bits().to_le_bytes());
    }
    data
}

/// Build `n_blocks` well-formed Q2_K blocks with pseudo-random payloads.
///
/// Like Q6_K (and unlike Q4_K), Q2_K stores its FP16 scales *after* the
/// quantized payload — `scales`/`qs` first, then a trailing `d`/`dmin` pair —
/// so it needs its own builder rather than `make_simple`'s "scales first"
/// layout or `make_trailing_scale`'s single-trailing-value layout.
fn make_q2_k(n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
    let mut data = Vec::with_capacity(n_blocks * Q2_K_BYTES);
    for _ in 0..n_blocks {
        // scales (16) + qs (64)
        for _ in 0..16 + 64 {
            data.push(rng.next_u8());
        }
        let d = f16::from_f32(rng.next_f32() * 0.05);
        let dmin = f16::from_f32(rng.next_f32() * 0.02);
        data.extend_from_slice(&d.to_bits().to_le_bytes());
        data.extend_from_slice(&dmin.to_bits().to_le_bytes());
    }
    data
}

/// Build `n_blocks` blocks of a non-K format with pseudo-random payloads.
///
/// Every byte pattern is legal for all of these formats — the nibble tables,
/// 6-bit scale packings, 2-bit fields and base-3 fixed-point bytes are total
/// functions of the input byte — so random payloads are a valid and unusually
/// harsh parity input.  `d` (and `dmin`) are kept small so the accumulated dot
/// products stay well inside fp32 range even for IQ4_NL/IQ4_XS, whose lookup
/// table reaches ±127 and whose sub-scale reaches ±32.
fn make_simple(n_blocks: usize, rng: &mut Rng, scales: &[f32], payload: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(n_blocks * (2 * scales.len() + payload));
    for _ in 0..n_blocks {
        for &s in scales {
            let v = f16::from_f32(rng.next_f32() * s);
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for _ in 0..payload {
            data.push(rng.next_u8());
        }
    }
    data
}

/// Build `n_blocks` blocks whose FP16 scale is stored *after* the payload.
fn make_trailing_scale(n_blocks: usize, rng: &mut Rng, payload: usize, scale: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(n_blocks * (payload + 2));
    for _ in 0..n_blocks {
        for _ in 0..payload {
            data.push(rng.next_u8());
        }
        let d = f16::from_f32(rng.next_f32() * scale);
        data.extend_from_slice(&d.to_bits().to_le_bytes());
    }
    data
}

fn block_bytes(ty: GgufTensorType) -> usize {
    match ty {
        GgufTensorType::Q6K => Q6_K_BYTES,
        GgufTensorType::Q2K => Q2_K_BYTES,
        GgufTensorType::Q3K => Q3_K_BYTES,
        GgufTensorType::Q4_0 => 18,
        GgufTensorType::Q4_1 => 20,
        GgufTensorType::Q5K => 176,
        GgufTensorType::Iq4Nl => 18,
        GgufTensorType::Iq4Xs => 136,
        GgufTensorType::Tq1_0 => 54,
        GgufTensorType::Tq2_0 => 66,
        _ => Q4_K_BYTES,
    }
}

/// Weights per block for `ty`.
fn block_size(ty: GgufTensorType) -> usize {
    match ty {
        GgufTensorType::Q4_0 | GgufTensorType::Q4_1 | GgufTensorType::Iq4Nl => 32,
        _ => K_BLOCK,
    }
}

fn make_rows(ty: GgufTensorType, n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
    match ty {
        GgufTensorType::Q6K => make_q6_k(n_blocks, rng),
        GgufTensorType::Q2K => make_q2_k(n_blocks, rng),
        // hmask (32) + qs (64) + scales (12) + trailing d
        GgufTensorType::Q3K => make_trailing_scale(n_blocks, rng, 32 + 64 + 12, 0.02),
        // d + 16 nibble bytes
        GgufTensorType::Q4_0 => make_simple(n_blocks, rng, &[0.05], 16),
        // d + m + 16 nibble bytes
        GgufTensorType::Q4_1 => make_simple(n_blocks, rng, &[0.05, 0.02], 16),
        // d + dmin + 12 scales + 32 qh + 128 qs
        GgufTensorType::Q5K => make_simple(n_blocks, rng, &[0.05, 0.02], 12 + 32 + 128),
        // d + 16 nibble bytes; kvalues reach ±127 so d is scaled down
        GgufTensorType::Iq4Nl => make_simple(n_blocks, rng, &[0.002], 16),
        // d + scales_h + scales_l + 128 nibble bytes; |dl| ≤ 32·|d| and
        // |kvalue| ≤ 127, so d is scaled down further still
        GgufTensorType::Iq4Xs => make_simple(n_blocks, rng, &[0.0002], 2 + 4 + 128),
        // 48 qs + 4 qh + trailing d
        GgufTensorType::Tq1_0 => make_trailing_scale(n_blocks, rng, 48 + 4, 0.05),
        // 64 qs + trailing d
        GgufTensorType::Tq2_0 => make_trailing_scale(n_blocks, rng, 64, 0.05),
        _ => make_q4_k(n_blocks, rng),
    }
}

fn dispatched(ty: GgufTensorType) -> Box<dyn QuantKernel> {
    KernelDispatcher::new()
        .get_kernel(ty)
        .expect("dispatched kernel available")
}

fn reference(ty: GgufTensorType) -> Box<dyn QuantKernel> {
    match ty {
        GgufTensorType::Q6K => Box::new(Q6KRef),
        GgufTensorType::Q4_0 => Box::new(Q4_0Ref),
        GgufTensorType::Q4_1 => Box::new(Q4_1Ref),
        GgufTensorType::Q5K => Box::new(Q5KRef),
        GgufTensorType::Iq4Nl => Box::new(Iq4NlRef),
        GgufTensorType::Iq4Xs => Box::new(Iq4XsRef),
        GgufTensorType::Tq1_0 => Box::new(Tq1_0Ref),
        GgufTensorType::Tq2_0 => Box::new(Tq2_0Ref),
        _ => Box::new(Q4KRef),
    }
}

/// Relative error between the dispatched and reference GEMV results.
///
/// Both kernels sum the same products in a different order, so the residual is
/// pure fp32 accumulation noise.  It is normalised by the largest output
/// magnitude in the *vector* rather than per element: with pseudo-random
/// weights an individual row can cancel to near zero, and dividing a
/// full-magnitude rounding error by that near-zero result would measure
/// cancellation, not kernel disagreement.
fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = a
        .iter()
        .chain(b.iter())
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(1e-6);
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale
}

/// Dispatched kernel agrees with the scalar reference at decode shapes.
fn assert_dispatch_matches_reference(ty: GgufTensorType, n_rows: usize, n_cols: usize, tol: f32) {
    let mut rng = Rng::new(0xC001_1AEA_u64 ^ (n_rows as u64) << 8 ^ n_cols as u64);
    let blocks_per_row = n_cols.div_ceil(block_size(ty));
    let data = make_rows(ty, n_rows * blocks_per_row, &mut rng);
    let tensor = QuantTensor::new(data, vec![n_rows, n_cols], ty);
    let input: Vec<f32> = (0..n_cols).map(|_| rng.next_f32()).collect();

    let mut got = vec![0.0f32; n_rows];
    let mut want = vec![0.0f32; n_rows];
    dispatched(ty)
        .gemv(&tensor, &input, &mut got)
        .expect("dispatched gemv");
    reference(ty)
        .gemv(&tensor, &input, &mut want)
        .expect("reference gemv");

    let err = max_rel_err(&got, &want);
    assert!(
        err <= tol,
        "{ty:?} {n_rows}x{n_cols}: max relative error {err} exceeds {tol}"
    );
}

/// Parallel row splitting is bit-identical to per-row serial evaluation.
///
/// The single-row tensors fall under [`oxillama_quant::parallel::should_parallelize`],
/// so they always take the serial path regardless of the pool width, giving an
/// exact oracle for the multi-threaded full-tensor GEMV.
fn assert_parallel_bit_identical(ty: GgufTensorType, n_rows: usize, n_cols: usize) {
    let mut rng = Rng::new(0x5EED_5EED_u64 ^ (n_rows as u64) << 16 ^ n_cols as u64);
    let bb = block_bytes(ty);
    let blocks_per_row = n_cols.div_ceil(block_size(ty));
    let row_bytes = blocks_per_row * bb;
    let data = make_rows(ty, n_rows * blocks_per_row, &mut rng);
    let input: Vec<f32> = (0..n_cols).map(|_| rng.next_f32()).collect();

    let kernel = dispatched(ty);

    let tensor = QuantTensor::new(data.clone(), vec![n_rows, n_cols], ty);
    let mut parallel_out = vec![0.0f32; n_rows];
    kernel
        .gemv(&tensor, &input, &mut parallel_out)
        .expect("parallel gemv");

    for row in 0..n_rows {
        let row_data = data[row * row_bytes..(row + 1) * row_bytes].to_vec();
        let row_tensor = QuantTensor::new(row_data, vec![1, n_cols], ty);
        let mut serial_out = [0.0f32; 1];
        kernel
            .gemv(&row_tensor, &input, &mut serial_out)
            .expect("serial gemv");
        assert_eq!(
            parallel_out[row].to_bits(),
            serial_out[0].to_bits(),
            "{ty:?} {n_rows}x{n_cols}: row {row} differs between the parallel \
             ({}) and serial ({}) paths — row splitting must never change the \
             accumulation order",
            parallel_out[row],
            serial_out[0]
        );
    }
}

/// Q4_K projections: `attn_q`, `attn_k`, `attn_o`, `ffn_gate`, `ffn_up`.
#[test]
fn q4_k_dispatch_matches_reference_at_decode_shapes() {
    for &rows in ROW_COUNTS {
        assert_dispatch_matches_reference(GgufTensorType::Q4K, rows, 2560, 1e-4);
    }
}

/// Q6_K: `attn_v`, `ffn_down`, and the tied `token_embd` LM head.
#[test]
fn q6_k_dispatch_matches_reference_at_decode_shapes() {
    for &rows in ROW_COUNTS {
        assert_dispatch_matches_reference(GgufTensorType::Q6K, rows, 2560, 1e-4);
    }
}

/// `ffn_down` is the one projection whose K is the 9728-wide FFN dimension.
#[test]
fn wide_k_dispatch_matches_reference() {
    assert_dispatch_matches_reference(GgufTensorType::Q4K, 129, 9728, 1e-4);
    assert_dispatch_matches_reference(GgufTensorType::Q6K, 129, 9728, 1e-4);
}

/// A K that is not a whole number of 256-weight blocks still matches.
#[test]
fn partial_trailing_block_dispatch_matches_reference() {
    for &cols in &[64usize, 255, 257, 700] {
        assert_dispatch_matches_reference(GgufTensorType::Q4K, 65, cols, 1e-4);
        assert_dispatch_matches_reference(GgufTensorType::Q6K, 65, cols, 1e-4);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-tier parity for every format in `PARITY_TYPES`
//
// The `golden_vectors` suite pins the scalar reference *and* the dispatched
// kernel to GGML, but only at one block and only through `dequant_block` /
// single-block `gemv`.  These sweeps cover the shapes where a tier's fast path
// and its scalar tail diverge: partial trailing blocks, K far wider than one
// block, and row counts that cross the parallel threshold.
// ─────────────────────────────────────────────────────────────────────────────

/// Every parity type agrees with its scalar reference at decode row counts.
#[test]
fn all_types_dispatch_match_reference_at_decode_shapes() {
    for &ty in PARITY_TYPES {
        for &rows in ROW_COUNTS {
            assert_dispatch_matches_reference(ty, rows, 2560, 1e-4);
        }
    }
}

/// Every parity type agrees with its scalar reference at a wide K.
#[test]
fn all_types_dispatch_match_reference_at_wide_k() {
    for &ty in PARITY_TYPES {
        assert_dispatch_matches_reference(ty, 129, 9728, 1e-4);
    }
}

/// Every parity type agrees with its scalar reference on partial trailing
/// blocks — the shapes where a SIMD tier falls back to a hand-written scalar
/// tail, which is exactly where a layout fix is easiest to miss.
#[test]
fn all_types_dispatch_match_reference_on_partial_blocks() {
    for &ty in PARITY_TYPES {
        // Cover both a 32-weight and a 256-weight block boundary, plus values
        // that straddle each and a non-multiple of any SIMD lane width.
        for &cols in &[17usize, 31, 33, 63, 64, 255, 257, 700] {
            assert_dispatch_matches_reference(ty, 65, cols, 1e-4);
        }
    }
}

/// Row splitting never changes the result for any parity type.
#[test]
fn all_types_parallel_rows_are_bit_identical() {
    for &ty in PARITY_TYPES {
        for &rows in ROW_COUNTS {
            assert_parallel_bit_identical(ty, rows, 2560);
        }
    }
}

#[test]
fn q4_k_parallel_rows_are_bit_identical() {
    for &rows in ROW_COUNTS {
        assert_parallel_bit_identical(GgufTensorType::Q4K, rows, 2560);
    }
}

#[test]
fn q6_k_parallel_rows_are_bit_identical() {
    for &rows in ROW_COUNTS {
        assert_parallel_bit_identical(GgufTensorType::Q6K, rows, 2560);
    }
}

/// The `ffn_gate`/`ffn_up` row count (9728) is not a multiple of any plausible
/// thread count, so the last band is short — still bit-identical.
#[test]
fn ffn_row_count_parallel_is_bit_identical() {
    assert_parallel_bit_identical(GgufTensorType::Q4K, 9728, 2560);
}

/// A slice of the tied Q6_K LM head: 151936 rows is too big for a unit test,
/// but 4099 rows is coprime with every lane and thread count and takes the
/// same code path.
#[test]
fn lm_head_slice_parallel_is_bit_identical() {
    assert_parallel_bit_identical(GgufTensorType::Q6K, 4099, 2560);
}

/// The GEMV pool always reports a usable width, and `0` is rejected as a
/// request rather than producing a zero-thread pool.
#[test]
fn gemv_pool_width_is_sane() {
    assert!(!oxillama_quant::parallel::set_num_threads(0));
    let width = oxillama_quant::parallel::num_threads();
    assert!(width >= 1, "GEMV pool must have at least one thread");
}

// ───────────────────────────────────────────────────────────────────────────
// Fused Q8_0-activation GEMV
// ───────────────────────────────────────────────────────────────────────────
//
// The decode hot path quantizes each activation vector to Q8_0 once per matmul
// input and then feeds `matvec_q8_fused`, so the activation side of every dot
// product is the Q8_0 *reconstruction* of the f32 vector.  The fused result is
// therefore NOT expected to match the f32-activation GEMV bit for bit; it is
// expected to match it to within the activation quantization error, and the
// tolerance below is derived from that error rather than guessed.
//
// ## Tolerance derivation
//
// Q8_0 encodes a 32-element block with `d = amax/127` and `q_i =
// round(x_i/d)`, so each reconstructed activation carries an error
//
//     e_i = q_i*d - x_i,        |e_i| <= d/2 = amax/254
//
// which behaves as a uniform random variable on [-d/2, d/2]: `std(e) =
// d/sqrt(12)`.  A row dot product `Σ w_i x_i` over K terms then sees an
// absolute error `Σ w_i e_i` whose standard deviation is
//
//     sigma_err = sqrt(K) * w_rms * d / sqrt(12)
//
// while the dot product itself has magnitude
//
//     sigma_out = sqrt(K) * w_rms * x_rms.
//
// The weight statistics cancel, leaving a *relative* error that depends only
// on the activation distribution:
//
//     sigma_err / sigma_out = d / (sqrt(12) * x_rms)
//                           = amax / (127 * sqrt(12) * x_rms).
//
// `Rng::next_f32` is uniform on [-1, 1): `x_rms = 1/sqrt(3) = 0.577` and the
// per-block `amax` is the max of 32 uniforms, ~0.97.  Hence
//
//     sigma_err / sigma_out ~ 0.97 / (127 * 3.464 * 0.577) = 3.8e-3.
//
// `max_rel_err` normalises the worst row error by the worst output magnitude.
// Both are extreme-value statistics over the same number of rows drawn from
// the same Gaussian shape, so the ~3.5-sigma extreme-value factor divides out
// and the expected ratio stays ~3.8e-3.  Two further terms are strictly
// smaller and folded into the margin: the FP16 rounding of `d` (relative
// 2^-11 = 4.9e-4) and fp32 accumulation noise (~1e-4 at these K).
//
// 2.0e-2 is that 3.8e-3 with a 5x margin — tight enough that a genuine kernel
// bug (a mis-indexed sub-block, a dropped `Σq_a` min correction, a wrong
// nibble half) blows straight through it, since those produce O(1) relative
// errors, not O(1e-2).
const FUSED_TOL: f32 = 2.0e-2;

/// Q8_0 activation block bytes: 2-byte FP16 scale + 32 int8 values.
const Q8_0_BYTES: usize = 34;

/// Fused GEMV over Q8_0 activations agrees with the f32-activation GEMV.
///
/// Skips silently when the dispatched kernel does not opt into the fused path
/// (`q8_fused_acts_blocks` is `None`) — that is the documented "stay on the
/// plain GEMV" state, not a failure.
fn assert_fused_matches_gemv(ty: GgufTensorType, n_rows: usize, n_cols: usize) {
    let kernel = dispatched(ty);
    let Some(n_blocks) = kernel.q8_fused_acts_blocks(n_cols) else {
        return;
    };

    let mut rng = Rng::new(0xF05E_D000_u64 ^ (n_rows as u64) << 8 ^ n_cols as u64);
    let blocks_per_row = n_cols.div_ceil(K_BLOCK);
    let data = make_rows(ty, n_rows * blocks_per_row, &mut rng);
    let tensor = QuantTensor::new(data.clone(), vec![n_rows, n_cols], ty);
    let input: Vec<f32> = (0..n_cols).map(|_| rng.next_f32()).collect();

    let mut want = vec![0.0f32; n_rows];
    kernel
        .gemv(&tensor, &input, &mut want)
        .expect("f32-activation gemv");

    let mut acts = Vec::new();
    oxillama_quant::quantize_activations_q8_0_into(&input, n_blocks, &mut acts);
    assert_eq!(
        acts.len(),
        n_blocks * Q8_0_BYTES,
        "activation buffer must be exactly the length the kernel promised"
    );

    let mut got = vec![0.0f32; n_rows];
    kernel
        .matvec_q8_fused(&data, &acts, &mut got, n_rows, n_cols)
        .expect("fused gemv");

    let err = max_rel_err(&got, &want);
    assert!(
        err <= FUSED_TOL,
        "{ty:?} fused {n_rows}x{n_cols}: max relative error {err} exceeds \
         {FUSED_TOL} (Q8_0 activation quantization alone predicts ~3.8e-3)"
    );
}

/// Fused row splitting is bit-identical to evaluating each row on its own.
fn assert_fused_parallel_bit_identical(ty: GgufTensorType, n_rows: usize, n_cols: usize) {
    let kernel = dispatched(ty);
    let Some(n_blocks) = kernel.q8_fused_acts_blocks(n_cols) else {
        return;
    };

    let mut rng = Rng::new(0xB177_1DEA_u64 ^ (n_rows as u64) << 16 ^ n_cols as u64);
    let bb = block_bytes(ty);
    let blocks_per_row = n_cols.div_ceil(block_size(ty));
    let row_bytes = blocks_per_row * bb;
    let data = make_rows(ty, n_rows * blocks_per_row, &mut rng);
    let input: Vec<f32> = (0..n_cols).map(|_| rng.next_f32()).collect();

    let mut acts = Vec::new();
    oxillama_quant::quantize_activations_q8_0_into(&input, n_blocks, &mut acts);

    let mut parallel_out = vec![0.0f32; n_rows];
    kernel
        .matvec_q8_fused(&data, &acts, &mut parallel_out, n_rows, n_cols)
        .expect("parallel fused gemv");

    for row in 0..n_rows {
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
        let mut serial_out = [0.0f32; 1];
        kernel
            .matvec_q8_fused(row_data, &acts, &mut serial_out, 1, n_cols)
            .expect("serial fused gemv");
        assert_eq!(
            parallel_out[row].to_bits(),
            serial_out[0].to_bits(),
            "{ty:?} fused {n_rows}x{n_cols}: row {row} differs between the \
             parallel ({}) and serial ({}) paths",
            parallel_out[row],
            serial_out[0]
        );
    }
}

/// `matvec_q8_fused` accumulates; it must never overwrite.
fn assert_fused_accumulates(ty: GgufTensorType, n_cols: usize) {
    let kernel = dispatched(ty);
    let Some(n_blocks) = kernel.q8_fused_acts_blocks(n_cols) else {
        return;
    };

    let mut rng = Rng::new(0xACC0_0000);
    let blocks_per_row = n_cols.div_ceil(K_BLOCK);
    let data = make_rows(ty, 3 * blocks_per_row, &mut rng);
    let input: Vec<f32> = (0..n_cols).map(|_| rng.next_f32()).collect();
    let mut acts = Vec::new();
    oxillama_quant::quantize_activations_q8_0_into(&input, n_blocks, &mut acts);

    let mut fresh = vec![0.0f32; 3];
    kernel
        .matvec_q8_fused(&data, &acts, &mut fresh, 3, n_cols)
        .expect("fresh fused");

    let seed = [1.5f32, -2.25, 0.75];
    let mut seeded = seed.to_vec();
    kernel
        .matvec_q8_fused(&data, &acts, &mut seeded, 3, n_cols)
        .expect("seeded fused");

    for row in 0..3 {
        assert_eq!(
            seeded[row].to_bits(),
            (seed[row] + fresh[row]).to_bits(),
            "{ty:?} fused row {row} must add to the existing accumulator"
        );
    }
}

/// Decode-shape parity: the two K values a Qwen3-4B layer actually uses.
#[test]
fn fused_matches_gemv_at_decode_shapes() {
    for &ty in &[GgufTensorType::Q4K, GgufTensorType::Q6K] {
        for &rows in &[1usize, 7, 65, 1025, 2560, 4099] {
            assert_fused_matches_gemv(ty, rows, 2560);
        }
        // `ffn_down`'s K, and the `ffn_gate`/`ffn_up` row count.
        assert_fused_matches_gemv(ty, 1025, 9728);
        assert_fused_matches_gemv(ty, 9728, 2560);
    }
}

/// A K that is not a whole number of 256-weight blocks: the fused kernels'
/// masked tail path must agree with the f32 GEMV's masked tail path.
#[test]
fn fused_matches_gemv_with_ragged_k() {
    for &ty in &[GgufTensorType::Q4K, GgufTensorType::Q6K] {
        for &cols in &[32usize, 64, 255, 257, 700, 2561] {
            assert_fused_matches_gemv(ty, 65, cols);
        }
    }
}

#[test]
fn fused_parallel_rows_are_bit_identical() {
    for &ty in &[GgufTensorType::Q4K, GgufTensorType::Q6K] {
        for &rows in &[1usize, 7, 65, 1025, 4099] {
            assert_fused_parallel_bit_identical(ty, rows, 2560);
        }
        assert_fused_parallel_bit_identical(ty, 1025, 9728);
    }
}

#[test]
fn fused_accumulates_into_output() {
    for &ty in &[GgufTensorType::Q4K, GgufTensorType::Q6K] {
        assert_fused_accumulates(ty, 2560);
        assert_fused_accumulates(ty, 700);
    }
}

/// Zero-padding a ragged activation vector must be equivalent to quantizing an
/// explicitly zero-padded vector — this is what lets a kernel ask for
/// `ceil(K/256)*8` blocks when K is not a multiple of 32.
#[test]
fn activation_padding_matches_explicit_zero_padding() {
    let mut rng = Rng::new(0x0AD0_0AD0_u64);
    for &(len, n_blocks) in &[(2560usize, 80usize), (700, 24), (1, 8), (33, 8), (0, 8)] {
        let input: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
        let mut got = Vec::new();
        oxillama_quant::quantize_activations_q8_0_into(&input, n_blocks, &mut got);

        let mut padded = input.clone();
        padded.resize(n_blocks * 32, 0.0);
        let want = oxillama_quant::quantize_f32_to_q8_0(&padded).expect("padded quantize");
        assert_eq!(got, want, "len={len} n_blocks={n_blocks}");
    }
}

/// Reusing one buffer across differently sized activations must leave no
/// stale bytes behind — the decode path relies on exactly this.
#[test]
fn activation_buffer_reuse_is_stateless() {
    let mut rng = Rng::new(0x5C4A_7C48);
    let mut shared = Vec::new();
    for &(len, n_blocks) in &[(9728usize, 304usize), (2560, 80), (4096, 128), (2560, 80)] {
        let input: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
        oxillama_quant::quantize_activations_q8_0_into(&input, n_blocks, &mut shared);

        let mut fresh = Vec::new();
        oxillama_quant::quantize_activations_q8_0_into(&input, n_blocks, &mut fresh);
        assert_eq!(shared, fresh, "len={len} n_blocks={n_blocks}");
    }
}

/// The fused assertions above return early when the dispatched kernel opts
/// out, so this pins the gate itself: wherever the NEON Q4_K kernel is the one
/// in use, it must advertise the fused path with the exact block count its row
/// kernel indexes (`ceil(K/256) * 8`).  Without this, silently losing the
/// override would turn every `assert_fused_*` into a no-op that still passes.
#[test]
fn fused_dispatch_gate_is_active_where_expected() {
    let kernel = dispatched(GgufTensorType::Q4K);
    if kernel.name() == "Q4_K_NEON" {
        assert_eq!(kernel.q8_fused_acts_blocks(2560), Some(80));
        assert_eq!(kernel.q8_fused_acts_blocks(9728), Some(304));
        // Ragged K still rounds up to whole 256-weight weight blocks.
        assert_eq!(kernel.q8_fused_acts_blocks(257), Some(16));
    }
}

/// Companion to [`fused_dispatch_gate_is_active_where_expected`] for Q6_K,
/// which reached the fused path only after its row kernel was rewritten from a
/// scalar body into real NEON.
#[test]
fn fused_dispatch_gate_covers_q6_k() {
    let kernel = dispatched(GgufTensorType::Q6K);
    if kernel.name() == "Q6_K_NEON" {
        assert_eq!(kernel.q8_fused_acts_blocks(2560), Some(80));
        assert_eq!(kernel.q8_fused_acts_blocks(9728), Some(304));
        assert_eq!(kernel.q8_fused_acts_blocks(257), Some(16));
    }
}

/// AVX2 companion to [`fused_dispatch_gate_is_active_where_expected`] /
/// [`fused_dispatch_gate_covers_q6_k`].
///
/// The AVX2 Q4_K/Q6_K kernels already had working `matvec_q8_fused` bodies,
/// but nothing overrode `q8_fused_acts_blocks` to advertise them, so every
/// `assert_fused_*`/`assert_batch_*` helper in this file silently skipped on
/// x86_64 (the `let Some(n_blocks) = ... else { return; }` gate at the top
/// of each) even with `simd-avx2` enabled — the fused path was dead code on
/// that tier and this whole suite was, without knowing it, only exercising
/// NEON. This pins the AVX2 gate itself so a regression there is caught
/// even though it cannot be observed on this aarch64 host.
///
/// Cannot be name-gated like the NEON tests above: `Iq2XxsAvx2`-style AVX2
/// kernels suffix their `name()`, but `Q4_KAvx2`/`Q6_KAvx2` return the bare
/// `"Q4_K"`/`"Q6_K"` — identical to `Q4KRef`/`Q6KRef`'s `name()`. Gating on
/// the name would silently no-op (not fail) on a build without `simd-avx2`,
/// which defeats the point; gating on the Cargo feature and runtime CPU
/// detection instead makes the assertion fire whenever the AVX2 kernel is
/// actually the one in use.
#[test]
fn fused_dispatch_gate_covers_avx2_k_quants() {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        // AVX-512 is checked before AVX2 in the dispatcher, so on a host with
        // avx512f (and the feature enabled) the AVX-512 kernel is the one that
        // answers.  It used to leave `q8_fused_acts_blocks` at the trait
        // default, which made enabling `simd-avx512` *remove* the fused path;
        // it now advertises the identical gate (`Q4_KAvx512`/`Q6_KAvx512`
        // delegate their `matvec_q8_fused` to the AVX2 kernels on purpose —
        // see `simd/avx512/fused.rs`).  So the assertions below hold on both
        // tiers and only the `name()` check has to tolerate either kernel;
        // `tests/avx512_fused_goldens.rs::avx512_gates_match_avx2_gates` pins
        // the tier-to-tier equality directly.

        let q4k = dispatched(GgufTensorType::Q4K);
        assert_eq!(q4k.name(), "Q4_K", "expected an AVX2/AVX-512 Q4_K kernel");
        assert_eq!(q4k.q8_fused_acts_blocks(2560), Some(80));
        assert_eq!(q4k.q8_fused_acts_blocks(9728), Some(304));
        assert_eq!(q4k.q8_fused_acts_blocks(257), Some(16));

        let q6k = dispatched(GgufTensorType::Q6K);
        assert_eq!(q6k.name(), "Q6_K", "expected an AVX2/AVX-512 Q6_K kernel");
        assert_eq!(q6k.q8_fused_acts_blocks(2560), Some(80));
        assert_eq!(q6k.q8_fused_acts_blocks(9728), Some(304));
        assert_eq!(q6k.q8_fused_acts_blocks(257), Some(16));
    }
}

// ── Batched fused matmul (prefill) ────────────────────────────────────────
//
// `matmul_q8_fused` is the prefill kernel: it hoists the weight decode out of
// the token loop so a prompt reads the weight matrix once for `m` tokens
// instead of once per token.  The contract that makes that safe to switch on
// by default is *bit-identity*: every `(row, token)` output must equal, to the
// last mantissa bit, what `matvec_q8_fused` produces for that token alone.
// Anything weaker would let a batched prefill and a sequential prefill diverge
// into different sampled tokens.

/// Batch sizes exercised: 1 (degenerate), 2, 3 (odd), 8, 16 (typical prefill
/// tiles), 32 (`MAX_FUSED_BATCH`) and 33 / 40 (which force the kernel's
/// internal chunking, so the chunk seam is covered too).
const BATCH_SIZES: &[usize] = &[1, 2, 3, 8, 16, 32, 33, 40];

/// Quantize `m` activation vectors into one contiguous buffer the batched
/// kernel can consume, and return them alongside the per-vector f32 copies.
fn make_batch_acts(
    n_cols: usize,
    n_blocks: usize,
    m: usize,
    rng: &mut Rng,
) -> (Vec<f32>, Vec<u8>, Vec<Vec<u8>>) {
    let rows: Vec<f32> = (0..m * n_cols).map(|_| rng.next_f32()).collect();

    let mut batch = Vec::new();
    oxillama_quant::quantize_activations_q8_0_batch_into(
        &rows, m, n_cols, n_cols, n_blocks, &mut batch,
    );

    let per_vector: Vec<Vec<u8>> = (0..m)
        .map(|t| {
            let mut acts = Vec::new();
            oxillama_quant::quantize_activations_q8_0_into(
                &rows[t * n_cols..(t + 1) * n_cols],
                n_blocks,
                &mut acts,
            );
            acts
        })
        .collect();

    (rows, batch, per_vector)
}

/// The batched activation encoder must produce exactly the concatenation of
/// the per-vector encoder's output — no shared scales, no cross-token state.
fn assert_batch_acts_are_concatenated(ty: GgufTensorType, n_cols: usize, m: usize) {
    let kernel = dispatched(ty);
    let Some(n_blocks) = kernel.q8_fused_acts_blocks(n_cols) else {
        return;
    };
    let mut rng = Rng::new(0x4C7A_0001_u64 ^ (m as u64) << 24 ^ n_cols as u64);
    let (_, batch, per_vector) = make_batch_acts(n_cols, n_blocks, m, &mut rng);

    assert_eq!(batch.len(), m * n_blocks * Q8_0_BYTES);
    let stride = n_blocks * Q8_0_BYTES;
    for (t, one) in per_vector.iter().enumerate() {
        assert_eq!(
            &batch[t * stride..(t + 1) * stride],
            one.as_slice(),
            "batched activation encoding of vector {t} differs from the \
             single-vector encoding"
        );
    }
}

/// Batched fused matmul is **bit-identical** to `m` separate fused GEMVs.
fn assert_batch_matches_matvec(ty: GgufTensorType, n_rows: usize, n_cols: usize, m: usize) {
    let kernel = dispatched(ty);
    let Some(n_blocks) = kernel.q8_fused_acts_blocks(n_cols) else {
        return;
    };

    let mut rng =
        Rng::new(0x8A7C_0BA7_u64 ^ (n_rows as u64) << 20 ^ (m as u64) << 8 ^ n_cols as u64);
    let blocks_per_row = n_cols.div_ceil(K_BLOCK);
    let data = make_rows(ty, n_rows * blocks_per_row, &mut rng);
    let (_, batch, per_vector) = make_batch_acts(n_cols, n_blocks, m, &mut rng);

    // Feature-major [n_rows][m] accumulator, exactly as the trait documents.
    let mut got = vec![0.0f32; n_rows * m];
    kernel
        .matmul_q8_fused(&data, &batch, &mut got, n_rows, n_cols, m)
        .expect("batched fused matmul");

    for (t, acts) in per_vector.iter().enumerate() {
        let mut want = vec![0.0f32; n_rows];
        kernel
            .matvec_q8_fused(&data, acts, &mut want, n_rows, n_cols)
            .expect("per-token fused gemv");
        for row in 0..n_rows {
            assert_eq!(
                got[row * m + t].to_bits(),
                want[row].to_bits(),
                "{ty:?} batched {n_rows}x{n_cols} m={m}: (row {row}, token {t}) \
                 batched={} sequential={}",
                got[row * m + t],
                want[row]
            );
        }
    }
}

/// `matmul_q8_fused` accumulates into `out`, like its single-vector sibling.
fn assert_batch_accumulates(ty: GgufTensorType, n_cols: usize, m: usize) {
    let kernel = dispatched(ty);
    let Some(n_blocks) = kernel.q8_fused_acts_blocks(n_cols) else {
        return;
    };
    let n_rows = 3usize;
    let mut rng = Rng::new(0xACC0_1234_u64 ^ (m as u64) << 8 ^ n_cols as u64);
    let blocks_per_row = n_cols.div_ceil(K_BLOCK);
    let data = make_rows(ty, n_rows * blocks_per_row, &mut rng);
    let (_, batch, _) = make_batch_acts(n_cols, n_blocks, m, &mut rng);

    let mut fresh = vec![0.0f32; n_rows * m];
    kernel
        .matmul_q8_fused(&data, &batch, &mut fresh, n_rows, n_cols, m)
        .expect("batched fused matmul");

    let mut seeded: Vec<f32> = (0..n_rows * m).map(|i| i as f32 * 0.25 + 1.0).collect();
    let seed = seeded.clone();
    kernel
        .matmul_q8_fused(&data, &batch, &mut seeded, n_rows, n_cols, m)
        .expect("batched fused matmul (seeded)");

    for i in 0..n_rows * m {
        assert_eq!(
            seeded[i].to_bits(),
            (seed[i] + fresh[i]).to_bits(),
            "batched matmul must accumulate at slot {i}"
        );
    }
}

#[test]
fn batched_activation_encoding_is_per_vector() {
    for &m in BATCH_SIZES {
        for &n_cols in &[256usize, 2560, 2561] {
            assert_batch_acts_are_concatenated(GgufTensorType::Q4K, n_cols, m);
            assert_batch_acts_are_concatenated(GgufTensorType::Q6K, n_cols, m);
        }
    }
}

#[test]
fn q4_k_batched_matmul_is_bit_identical_to_sequential() {
    for &m in BATCH_SIZES {
        for &n_rows in &[1usize, 7, 65, 1024] {
            assert_batch_matches_matvec(GgufTensorType::Q4K, n_rows, 2560, m);
        }
    }
}

#[test]
fn q6_k_batched_matmul_is_bit_identical_to_sequential() {
    for &m in BATCH_SIZES {
        for &n_rows in &[1usize, 7, 65, 1024] {
            assert_batch_matches_matvec(GgufTensorType::Q6K, n_rows, 2560, m);
        }
    }
}

/// Ragged K drives the scalar tail block inside both batched kernels.
///
/// Q2_K/Q3_K are included here (and in
/// [`batched_matmul_accumulates_into_output`]) rather than in their own
/// dedicated tests: neither has a native batched AVX2 kernel, so
/// `matmul_q8_fused` runs through `traits.rs`'s default per-token
/// `matvec_q8_fused` delegation. These two sweeps are what pins that the
/// `q8_fused_acts_blocks` gate fix in `traits.rs` actually exercises that
/// fallback correctly for them, instead of silently skipping.
#[test]
fn batched_matmul_is_bit_identical_with_ragged_k() {
    for &m in &[1usize, 5, 16, 33] {
        for &n_cols in &[257usize, 300, 2561] {
            assert_batch_matches_matvec(GgufTensorType::Q2K, 65, n_cols, m);
            assert_batch_matches_matvec(GgufTensorType::Q3K, 65, n_cols, m);
            assert_batch_matches_matvec(GgufTensorType::Q4K, 65, n_cols, m);
            assert_batch_matches_matvec(GgufTensorType::Q6K, 65, n_cols, m);
        }
    }
}

/// Real Qwen3-4B projection shapes at a realistic prefill tile.
#[test]
fn batched_matmul_matches_at_qwen3_shapes() {
    for &(n_rows, n_cols) in &[
        (4096usize, 2560usize),
        (1024, 2560),
        (2560, 4096),
        (9728, 2560),
    ] {
        assert_batch_matches_matvec(GgufTensorType::Q4K, n_rows, n_cols, 16);
    }
    assert_batch_matches_matvec(GgufTensorType::Q6K, 2560, 9728, 16);
}

#[test]
fn batched_matmul_accumulates_into_output() {
    for &m in &[1usize, 8, 33] {
        assert_batch_accumulates(GgufTensorType::Q2K, 2560, m);
        assert_batch_accumulates(GgufTensorType::Q3K, 2560, m);
        assert_batch_accumulates(GgufTensorType::Q4K, 2560, m);
        assert_batch_accumulates(GgufTensorType::Q6K, 2560, m);
    }
}

//! Golden-vector conformance tests for the GPU-side dequantisation helpers.
//!
//! # Why this file exists
//!
//! This is the GPU-crate sibling of
//! `crates/oxillama-quant/tests/golden_vectors.rs`.  Every constant and block
//! builder below is copied byte-for-byte from that file so that **both**
//! crates are pinned to the same upstream oracle (the verbatim
//! `dequantize_row_*` bodies from `llama.cpp/ggml/src/ggml-quants.c`) instead
//! of being pinned to each other.  Comparing the GPU dequant helpers against
//! the CPU reference kernels would not catch a bug that was present in both
//! (which is exactly what happened here: the GPU kernels copied the CPU
//! kernels' pre-fix, wrong nibble/qh/trit layout and passed every
//! self-consistency test that existed at the time).
//!
//! See the module doc on the CPU file for the full rationale of the input
//! patterns used (`nib(k) = (k & 0xF) | ((15 - (k & 0xF)) << 4)`, etc.).
//!
//! # Scope
//!
//! This module tests the CPU-side `dequant_*_to_f32` helpers that every GPU
//! kernel in this crate calls before uploading weights to the device (see
//! each kernel's module doc: "Dequantise `weight_bytes` to f32 on the CPU
//! ... Upload ... Dispatch the generic f32 GEMV shader").  The actual WGSL
//! shader (`gemv_f32.wgsl`) is a format-agnostic dot product over an
//! already-dequantised f32 buffer, so it carries none of the format-specific
//! layout logic and needs no separate golden test.  Real-hardware coverage
//! (dispatching an actual compute pipeline and reading back results) lives in
//! `tests/cpu_gpu_cross_check.rs`.

#![allow(clippy::excessive_precision)]

use half::f16;

// ─────────────────────────────────────────────────────────────────────────────
// Expected outputs — copied from oxillama-quant/tests/golden_vectors.rs,
// which derives them from upstream GGML. See that file for the full
// derivation and hand-worked spot checks.
// ─────────────────────────────────────────────────────────────────────────────

const EXPECT_Q4_0: [f32; 32] = [
    -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5,
    3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0,
];

const EXPECT_Q4_1: [f32; 32] = [
    0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75,
    7.75, 7.25, 6.75, 6.25, 5.75, 5.25, 4.75, 4.25, 3.75, 3.25, 2.75, 2.25, 1.75, 1.25, 0.75, 0.25,
];

const EXPECT_Q5_0: [f32; 32] = [
    0.0, 0.5, 1.0, 1.5, -6.0, -5.5, -5.0, -4.5, 4.0, 4.5, 5.0, 5.5, -2.0, -1.5, -1.0, -0.5, 7.5,
    7.0, 6.5, 6.0, -2.5, -3.0, -3.5, -4.0, 3.5, 3.0, 2.5, 2.0, -6.5, -7.0, -7.5, -8.0,
];

const EXPECT_Q5_K: [f32; 256] = [
    6.75, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 10.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75,
    6.25, 6.75, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 10.75, 3.25, 3.75, 4.25, 4.75, 5.25,
    5.75, 6.25, 13.5, 28.5, 11.5, 10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 20.5, 3.5, 2.5, 1.5, 0.5, -0.5,
    -1.5, 13.5, 28.5, 11.5, 10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 20.5, 3.5, 2.5, 1.5, 0.5, -0.5, -1.5,
    -1.75, -0.25, 25.25, 2.75, 4.25, 5.75, 7.25, 8.75, 10.25, 11.75, 37.25, 14.75, 16.25, 17.75,
    19.25, 20.75, -1.75, -0.25, 25.25, 2.75, 4.25, 5.75, 7.25, 8.75, 10.25, 11.75, 37.25, 14.75,
    16.25, 17.75, 19.25, 20.75, 28.0, 26.0, 24.0, 54.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0,
    38.0, 4.0, 2.0, 0.0, -2.0, 28.0, 26.0, 24.0, 54.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0,
    38.0, 4.0, 2.0, 0.0, -2.0, -0.5, 0.0, 0.5, 1.0, 9.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 13.5,
    6.0, 6.5, 7.0, -0.5, 0.0, 0.5, 1.0, 9.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 13.5, 6.0, 6.5,
    7.0, 21.5, 20.0, 18.5, 17.0, 15.5, 38.0, 12.5, 11.0, 9.5, 8.0, 6.5, 5.0, 3.5, 26.0, 0.5, -1.0,
    21.5, 20.0, 18.5, 17.0, 15.5, 38.0, 12.5, 11.0, 9.5, 8.0, 6.5, 5.0, 3.5, 26.0, 0.5, -1.0, -1.5,
    1.0, 3.5, 6.0, 8.5, 11.0, 53.5, 16.0, 18.5, 21.0, 23.5, 26.0, 28.5, 31.0, 73.5, 36.0, -1.5,
    1.0, 3.5, 6.0, 8.5, 11.0, 53.5, 16.0, 18.5, 21.0, 23.5, 26.0, 28.5, 31.0, 73.5, 36.0, 50.5,
    47.0, 43.5, 40.0, 36.5, 33.0, 29.5, 82.0, 22.5, 19.0, 15.5, 12.0, 8.5, 5.0, 1.5, 54.0, 50.5,
    47.0, 43.5, 40.0, 36.5, 33.0, 29.5, 82.0, 22.5, 19.0, 15.5, 12.0, 8.5, 5.0, 1.5, 54.0,
];

const EXPECT_TQ2_0: [f32; 256] = [
    -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

const EXPECT_IQ4_NL: [f32; 32] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0, 113.0, 89.0, 69.0, 53.0, 38.0, 25.0, 13.0, 1.0, -10.0, -22.0, -35.0, -49.0, -65.0,
    -83.0, -104.0, -127.0,
];

const EXPECT_IQ4_XS: [f32; 256] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63.5, -52.0, -41.5, -32.5,
    -24.5, -17.5, -11.0, -5.0, 0.5, 6.5, 12.5, 19.0, 26.5, 34.5, 44.5, 56.5, 56.5, 44.5, 34.5,
    26.5, 19.0, 12.5, 6.5, 0.5, -5.0, -11.0, -17.5, -24.5, -32.5, -41.5, -52.0, -63.5, -127.0,
    -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0,
    113.0, 113.0, 89.0, 69.0, 53.0, 38.0, 25.0, 13.0, 1.0, -10.0, -22.0, -35.0, -49.0, -65.0,
    -83.0, -104.0, -127.0, -190.5, -156.0, -124.5, -97.5, -73.5, -52.5, -33.0, -15.0, 1.5, 19.5,
    37.5, 57.0, 79.5, 103.5, 133.5, 169.5, 169.5, 133.5, 103.5, 79.5, 57.0, 37.5, 19.5, 1.5, -15.0,
    -33.0, -52.5, -73.5, -97.5, -124.5, -156.0, -190.5, -254.0, -208.0, -166.0, -130.0, -98.0,
    -70.0, -44.0, -20.0, 2.0, 26.0, 50.0, 76.0, 106.0, 138.0, 178.0, 226.0, 226.0, 178.0, 138.0,
    106.0, 76.0, 50.0, 26.0, 2.0, -20.0, -44.0, -70.0, -98.0, -130.0, -166.0, -208.0, -254.0,
    -317.5, -260.0, -207.5, -162.5, -122.5, -87.5, -55.0, -25.0, 2.5, 32.5, 62.5, 95.0, 132.5,
    172.5, 222.5, 282.5, 282.5, 222.5, 172.5, 132.5, 95.0, 62.5, 32.5, 2.5, -25.0, -55.0, -87.5,
    -122.5, -162.5, -207.5, -260.0, -317.5, -381.0, -312.0, -249.0, -195.0, -147.0, -105.0, -66.0,
    -30.0, 3.0, 39.0, 75.0, 114.0, 159.0, 207.0, 267.0, 339.0, 339.0, 267.0, 207.0, 159.0, 114.0,
    75.0, 39.0, 3.0, -30.0, -66.0, -105.0, -147.0, -195.0, -249.0, -312.0, -381.0, -444.5, -364.0,
    -290.5, -227.5, -171.5, -122.5, -77.0, -35.0, 3.5, 45.5, 87.5, 133.0, 185.5, 241.5, 311.5,
    395.5, 395.5, 311.5, 241.5, 185.5, 133.0, 87.5, 45.5, 3.5, -35.0, -77.0, -122.5, -171.5,
    -227.5, -290.5, -364.0, -444.5,
];

const TQ1_0_GOLDEN_QS: [u8; 48] = [
    0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45,
    0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C,
    0x94, 0xBB, 0x31, 0x94, 0xBB, 0x31, 0x94, 0xBB, 0x31, 0x94, 0xBB, 0x31, 0x94, 0xBB, 0x31, 0x94,
];

const TQ1_0_GOLDEN_QH: [u8; 4] = [0x30, 0x92, 0xBB, 0x30];

/// Q4_K / Q5_K packed 6-bit scales+mins.  Decodes to
/// `sc = [1, 2, 3, 4, 1, 3, 5, 7]`, `mn = [5, 6, 7, 8, 2, 4, 6, 8]`.
const SCALES_12: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 0x21, 0x43, 0x65, 0x87];

// ─────────────────────────────────────────────────────────────────────────────
// Block builders — identical to oxillama-quant/tests/golden_vectors.rs.
// ─────────────────────────────────────────────────────────────────────────────

/// Canonical layout-discriminating nibble byte: `lo = k & 0xF`, `hi = 15 - lo`.
fn nib(k: usize) -> u8 {
    let l = (k & 0xF) as u8;
    l | ((15 - l) << 4)
}

/// Little-endian `f16` bytes for an exactly representable scale.
fn f16le(v: f32) -> [u8; 2] {
    f16::from_f32(v).to_le_bytes()
}

fn block_q4_0() -> Vec<u8> {
    let mut b = f16le(0.5).to_vec();
    b.extend((0..16).map(nib));
    b
}

fn block_q4_1() -> Vec<u8> {
    let mut b = f16le(0.5).to_vec();
    b.extend_from_slice(&f16le(0.25));
    b.extend((0..16).map(nib));
    b
}

fn block_q5_0() -> Vec<u8> {
    let mut b = f16le(0.5).to_vec();
    b.extend_from_slice(&[0x0F, 0x0F, 0x0F, 0x0F]);
    b.extend((0..16).map(nib));
    b
}

fn block_q5_k() -> Vec<u8> {
    let mut b = f16le(0.5).to_vec();
    b.extend_from_slice(&f16le(0.25));
    b.extend_from_slice(&SCALES_12);
    // qh[l] = 1 << (l & 7): exactly one 5th-bit source bit per byte, and the
    // eight bit positions are individually identifiable in the output.
    b.extend((0..32).map(|l: usize| 1u8 << (l & 7)));
    b.extend((0..128).map(nib));
    b
}

/// TQ2_0 block whose 2-bit codes are `l % 3` in the first 32 bytes and
/// `(l + 1) % 3` in the last 32, where `l` is the *digit* index.
fn block_tq2_0() -> Vec<u8> {
    let mut b = vec![0x24u8; 32];
    b.extend_from_slice(&[0x49u8; 32]);
    b.extend_from_slice(&f16le(0.5));
    b
}

fn block_tq1_0() -> Vec<u8> {
    let mut b = TQ1_0_GOLDEN_QS.to_vec();
    b.extend_from_slice(&TQ1_0_GOLDEN_QH);
    b.extend_from_slice(&f16le(1.0));
    b
}

fn block_iq4_nl() -> Vec<u8> {
    let mut b = f16le(1.0).to_vec();
    b.extend((0..16).map(nib));
    b
}

/// IQ4_XS block with `ls[i] = 32 + i`, i.e. sub-scale `ls - 32 = i`.
fn block_iq4_xs() -> Vec<u8> {
    let mut b = f16le(0.5).to_vec();
    b.extend_from_slice(&0xAAAAu16.to_le_bytes());
    b.extend_from_slice(&[0x10, 0x32, 0x54, 0x76]);
    b.extend((0..128).map(nib));
    b
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — call the GPU crate's own CPU-side `dequant_*_to_f32` helpers
// directly (one block, one row) and compare against the upstream oracle.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_q4_0() {
    let block = block_q4_0();
    let got = super::q4_0::dequant_q4_0_to_f32(&block, 1, 32).expect("dequant_q4_0_to_f32");
    assert_eq!(got, EXPECT_Q4_0, "Q4_0 GPU dequant vs GGML golden vector");
}

#[test]
fn golden_q4_1() {
    let block = block_q4_1();
    let got = super::q4_1::dequant_q4_1_to_f32(&block, 1, 32).expect("dequant_q4_1_to_f32");
    assert_eq!(got, EXPECT_Q4_1, "Q4_1 GPU dequant vs GGML golden vector");
}

#[test]
fn golden_iq4_nl() {
    let block = block_iq4_nl();
    let got = super::iq4_nl::dequant_iq4_nl_to_f32(&block, 1, 32).expect("dequant_iq4_nl_to_f32");
    assert_eq!(
        got, EXPECT_IQ4_NL,
        "IQ4_NL GPU dequant vs GGML golden vector"
    );
}

#[test]
fn golden_iq4_xs() {
    let block = block_iq4_xs();
    let got = super::iq4_xs::dequant_iq4_xs_to_f32(&block, 1, 256).expect("dequant_iq4_xs_to_f32");
    assert_eq!(
        got, EXPECT_IQ4_XS,
        "IQ4_XS GPU dequant vs GGML golden vector"
    );
}

#[test]
fn golden_q5_k() {
    let block = block_q5_k();
    let got = super::q5_k::dequant_q5_k_to_f32(&block, 1, 256).expect("dequant_q5_k_to_f32");
    assert_eq!(got, EXPECT_Q5_K, "Q5_K GPU dequant vs GGML golden vector");
}

#[test]
fn golden_tq2_0() {
    let block = block_tq2_0();
    let got = super::tq2_0::dequant_tq2_0_to_f32(&block, 1, 256).expect("dequant_tq2_0_to_f32");
    assert_eq!(got, EXPECT_TQ2_0, "TQ2_0 GPU dequant vs GGML golden vector");
}

#[test]
fn golden_tq1_0() {
    let block = block_tq1_0();
    let got = super::tq1_0::dequant_tq1_0_to_f32(&block, 1, 256).expect("dequant_tq1_0_to_f32");
    let expected: Vec<f32> = (0..256).map(|i: usize| (i % 3) as f32 - 1.0).collect();
    assert_eq!(got, expected, "TQ1_0 GPU dequant vs GGML golden vector");
}

/// The all-`+1` block is the cheapest discriminator for the base-3 decode —
/// see the sibling CPU golden test for the full derivation.
#[test]
fn golden_tq1_0_all_plus_one() {
    let mut block = vec![0xFFu8; 48];
    block.extend_from_slice(&[253u8; 4]);
    block.extend_from_slice(&f16le(1.0));
    let got = super::tq1_0::dequant_tq1_0_to_f32(&block, 1, 256).expect("dequant_tq1_0_to_f32");
    let expected = vec![1.0f32; 256];
    assert_eq!(
        got, expected,
        "TQ1_0 all-+1 GPU dequant vs GGML golden vector"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Controls — formats that were already correct.  Must pass before and after
// any change in this pass, which is what shows the oracle itself is right
// rather than merely different.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_q5_0_control() {
    let block = block_q5_0();
    let got = super::q5_0::dequant_q5_0_to_f32(&block, 1, 32).expect("dequant_q5_0_to_f32");
    assert_eq!(got, EXPECT_Q5_0, "Q5_0 GPU dequant vs GGML golden vector");
}

/// Multi-block regression for the [`TQ1_0_QS_BYTES`]-cancellation bug found
/// while porting the CPU fix: the original GPU decoder computed
/// `col = out_idx - weight_base`, which always cancels back to `0..256`
/// regardless of `blk`, so every block past the first silently overwrote
/// columns `0..256` of the row instead of `blk*256..blk*256+256`.  Two
/// distinct blocks concatenated into one 512-column row must decode to two
/// distinct 256-element halves.
#[test]
fn golden_tq1_0_multi_block_does_not_alias() {
    let block_a = block_tq1_0(); // decodes to (i % 3) - 1
    let mut block_b = vec![0xFFu8; 48];
    block_b.extend_from_slice(&[253u8; 4]);
    block_b.extend_from_slice(&f16le(1.0)); // decodes to all +1.0

    let mut weight_bytes = block_a.clone();
    weight_bytes.extend_from_slice(&block_b);

    let got =
        super::tq1_0::dequant_tq1_0_to_f32(&weight_bytes, 1, 512).expect("dequant_tq1_0_to_f32");

    let expected_a: Vec<f32> = (0..256).map(|i: usize| (i % 3) as f32 - 1.0).collect();
    let expected_b = vec![1.0f32; 256];

    assert_eq!(got[..256], expected_a[..], "block 0 (columns 0..256)");
    assert_eq!(got[256..], expected_b[..], "block 1 (columns 256..512)");
}

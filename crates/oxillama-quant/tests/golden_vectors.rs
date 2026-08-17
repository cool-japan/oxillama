//! Golden-vector conformance tests against upstream llama.cpp / GGML.
//!
//! # Why this file exists
//!
//! Every SIMD tier in this crate is parity-tested against the *scalar
//! reference*, and every round-trip test in `quantize.rs` pairs this crate's
//! encoder with this crate's decoder.  Both families of test are
//! **self-consistent**: a systematic disagreement with GGML — a permuted
//! nibble layout, a permuted `qh` bit assignment, a different scale
//! convention — passes them all.  Only a comparison against values produced
//! by *upstream's own algorithm* can detect that class of bug.
//!
//! Every expected array below was produced by running the verbatim
//! `dequantize_row_*` / `quantize_row_*_ref` bodies extracted from
//! `llama.cpp/ggml/src/ggml-quants.c` on the exact input blocks constructed
//! here.  Each test also carries the closed-form hand derivation in its doc
//! comment so a reviewer can re-check the numbers without a C toolchain.
//!
//! # Why the inputs look the way they do
//!
//! The inputs are chosen to be **layout-discriminating**.  A block of
//! `[0x88; 16]` or `[0xFF; 16]` (as used by several pre-existing unit tests)
//! decodes identically under *any* nibble permutation and therefore proves
//! nothing.  The canonical pattern used here is
//!
//! ```text
//! qs[k] = (k & 0xF) | ((15 - (k & 0xF)) << 4)
//! ```
//!
//! so byte `k` carries low nibble `k & 0xF` and high nibble `15 - (k & 0xF)`:
//! every nibble value appears, and the low/high halves are mirror images, so
//! swapping or interleaving them changes the output at almost every index.
//!
//! All scales are exactly representable in both `f16` and `f32` (`0.5`,
//! `0.25`, `1.0`), and all quantised values are small integers, so every
//! expected value is *exact*: the assertions use `==`, not a tolerance.
//!
//! # Controls
//!
//! `Q5_0`, `Q4_K`, `Q6_K` and `Q8_0` are included as controls.  Their kernels
//! were already correct, so their golden tests must pass **before and after**
//! any change — that is what shows the oracle itself is right rather than
//! merely different.

#![allow(clippy::excessive_precision)]

use half::f16;
use oxillama_gguf::GgufTensorType;
use oxillama_quant::reference::{
    Iq4NlRef, Iq4XsRef, Q4KRef, Q4_0Ref, Q4_1Ref, Q5KRef, Q5_0Ref, Q6KRef, Q8_0Ref, Tq1_0Ref,
    Tq2_0Ref,
};
use oxillama_quant::{KernelDispatcher, QuantKernel, QuantTensor};

// ─────────────────────────────────────────────────────────────────────────────
// Expected outputs — generated from upstream GGML, see module doc.
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

const EXPECT_Q8_0: [f32; 32] = [
    -8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
];

const EXPECT_Q4_K: [f32; 256] = [
    -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75,
    6.25, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25,
    5.75, 6.25, 13.5, 12.5, 11.5, 10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5, -0.5,
    -1.5, 13.5, 12.5, 11.5, 10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5, -0.5, -1.5,
    -1.75, -0.25, 1.25, 2.75, 4.25, 5.75, 7.25, 8.75, 10.25, 11.75, 13.25, 14.75, 16.25, 17.75,
    19.25, 20.75, -1.75, -0.25, 1.25, 2.75, 4.25, 5.75, 7.25, 8.75, 10.25, 11.75, 13.25, 14.75,
    16.25, 17.75, 19.25, 20.75, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0,
    6.0, 4.0, 2.0, 0.0, -2.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0,
    4.0, 2.0, 0.0, -2.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
    6.5, 7.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
    21.5, 20.0, 18.5, 17.0, 15.5, 14.0, 12.5, 11.0, 9.5, 8.0, 6.5, 5.0, 3.5, 2.0, 0.5, -1.0, 21.5,
    20.0, 18.5, 17.0, 15.5, 14.0, 12.5, 11.0, 9.5, 8.0, 6.5, 5.0, 3.5, 2.0, 0.5, -1.0, -1.5, 1.0,
    3.5, 6.0, 8.5, 11.0, 13.5, 16.0, 18.5, 21.0, 23.5, 26.0, 28.5, 31.0, 33.5, 36.0, -1.5, 1.0,
    3.5, 6.0, 8.5, 11.0, 13.5, 16.0, 18.5, 21.0, 23.5, 26.0, 28.5, 31.0, 33.5, 36.0, 50.5, 47.0,
    43.5, 40.0, 36.5, 33.0, 29.5, 26.0, 22.5, 19.0, 15.5, 12.0, 8.5, 5.0, 1.5, -2.0, 50.5, 47.0,
    43.5, 40.0, 36.5, 33.0, 29.5, 26.0, 22.5, 19.0, 15.5, 12.0, 8.5, 5.0, 1.5, -2.0,
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

const EXPECT_Q6_K: [f32; 256] = [
    128.0, 60.0, -8.0, -76.0, 112.0, 44.0, -24.0, -92.0, 96.0, 28.0, -40.0, -108.0, 80.0, 12.0,
    -56.0, -124.0, 112.0, 52.5, -7.0, -66.5, 98.0, 38.5, -21.0, -80.5, 84.0, 24.5, -35.0, -94.5,
    70.0, 10.5, -49.0, -108.5, 96.0, 93.0, 90.0, 87.0, 36.0, 33.0, 30.0, 27.0, -24.0, -27.0, -30.0,
    -33.0, -84.0, -87.0, -90.0, -93.0, 80.0, 77.5, 75.0, 72.5, 30.0, 27.5, 25.0, 22.5, -20.0,
    -22.5, -25.0, -27.5, -70.0, -72.5, -75.0, -77.5, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0,
    48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0,
    13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0,
    12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -16.0, -7.5, 1.0, 9.5, -14.0, -5.5, 3.0, 11.5, -12.0, -3.5,
    5.0, 13.5, -10.0, -1.5, 7.0, 15.5, -32.0, -31.0, -30.0, -29.0, -12.0, -11.0, -10.0, -9.0, 8.0,
    9.0, 10.0, 11.0, 28.0, 29.0, 30.0, 31.0, -48.0, -46.5, -45.0, -43.5, -18.0, -16.5, -15.0,
    -13.5, 12.0, 13.5, 15.0, 16.5, 42.0, 43.5, 45.0, 46.5, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0,
    18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 77.5, 75.0, 72.5, 70.0, 67.5, 65.0,
    62.5, 60.0, 57.5, 55.0, 52.5, 50.0, 47.5, 45.0, 42.5, 40.0, -51.0, -54.0, -57.0, -60.0, -63.0,
    -66.0, -69.0, -72.0, -75.0, -78.0, -81.0, -84.0, -87.0, -90.0, -93.0, -96.0, -59.5, -63.0,
    -66.5, -70.0, -73.5, -77.0, -80.5, -84.0, -87.5, -91.0, -94.5, -98.0, -101.5, -105.0, -108.5,
    -112.0,
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

const EXPECT_Q4_0_ENC_ROUNDTRIP: [f32; 32] = [
    -5.0, -5.0, -4.375, -4.375, -3.75, -3.75, -3.75, -3.125, -3.125, -2.5, -2.5, -2.5, -1.875,
    -1.875, -1.25, -1.25, -1.25, -0.625, -0.625, 0.0, 0.0, 0.0, 0.625, 0.625, 1.25, 1.25, 1.25,
    1.875, 1.875, 2.5, 2.5, 2.5,
];

const TQ1_0_GOLDEN_QS: [u8; 48] = [
    0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45,
    0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C, 0xCF, 0x45, 0x6C,
    0x94, 0xBB, 0x31, 0x94, 0xBB, 0x31, 0x94, 0xBB, 0x31, 0x94, 0xBB, 0x31, 0x94, 0xBB, 0x31, 0x94,
];

const TQ1_0_GOLDEN_QH: [u8; 4] = [0x30, 0x92, 0xBB, 0x30];

const TQ2_0_GOLDEN_ENC_QS: [u8; 64] = [
    0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18,
    0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61,
    0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86,
    0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18, 0x61, 0x86, 0x18,
];

const Q4_0_GOLDEN_ENC_QS: [u8; 16] = [
    0x60, 0x70, 0x71, 0x81, 0x82, 0x82, 0x92, 0x93, 0xA3, 0xA4, 0xA4, 0xB4, 0xB5, 0xC5, 0xC6, 0xC6,
];

// ─────────────────────────────────────────────────────────────────────────────
// Block builders — byte-for-byte mirrors of the C driver that produced the
// arrays above.
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

/// Q4_K / Q5_K packed 6-bit scales+mins.
///
/// `get_scale_min_k4` decodes these to
/// `sc = [1, 2, 3, 4, 1, 3, 5, 7]`, `mn = [5, 6, 7, 8, 2, 4, 6, 8]`:
/// bytes 0..4 give `sc[0..4]`, bytes 4..8 give `mn[0..4]`, and the low/high
/// nibbles of bytes 8..12 give `sc[4..8]` / `mn[4..8]` (the 2-bit high parts
/// are all zero because bytes 0..8 are all below 64).  No sub-block scale or
/// min is zero, so every sub-block contributes.
const SCALES_12: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 0x21, 0x43, 0x65, 0x87];

/// Decoded `sc` from [`SCALES_12`].
const SC: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 1.0, 3.0, 5.0, 7.0];
/// Decoded `mn` from [`SCALES_12`].
const MN: [f32; 8] = [5.0, 6.0, 7.0, 8.0, 2.0, 4.0, 6.0, 8.0];

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

fn block_q8_0() -> Vec<u8> {
    let mut b = f16le(0.5).to_vec();
    b.extend((0..32).map(|j: i32| (j - 16) as i8 as u8));
    b
}

fn block_q4_k() -> Vec<u8> {
    let mut b = f16le(0.5).to_vec();
    b.extend_from_slice(&f16le(0.25));
    b.extend_from_slice(&SCALES_12);
    b.extend((0..128).map(nib));
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

fn block_q6_k() -> Vec<u8> {
    let mut b: Vec<u8> = (0..128).map(nib).collect();
    b.extend((0..64).map(|k| k as u8));
    b.extend((0..16).map(|i: i32| (i - 8) as i8 as u8));
    b.extend_from_slice(&f16le(0.5));
    b
}

/// TQ2_0 block whose 2-bit codes are `l % 3` in the first 32 bytes and
/// `(l + 1) % 3` in the last 32, where `l` is the *digit* index.
///
/// Byte 0x24 = `0b00_10_01_00` → digits `l = 0,1,2,3` are `0,1,2,0`.
/// Byte 0x49 = `0b01_00_10_01` → digits `l = 0,1,2,3` are `1,2,0,1`.
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
///
/// `scales_l` nibbles are `0,1,2,3,4,5,6,7` (low nibble = even sub-block) and
/// every 2-bit `scales_h` field is `0b10`, so `ls[i] = i | (2 << 4) = 32 + i`.
fn block_iq4_xs() -> Vec<u8> {
    let mut b = f16le(0.5).to_vec();
    b.extend_from_slice(&0xAAAAu16.to_le_bytes());
    b.extend_from_slice(&[0x10, 0x32, 0x54, 0x76]);
    b.extend((0..128).map(nib));
    b
}

// ─────────────────────────────────────────────────────────────────────────────
// Assertion helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Assert a kernel's `dequant_block` reproduces GGML exactly, for both the
/// scalar reference and the kernel [`KernelDispatcher`] actually selects on
/// this host (NEON on aarch64, AVX2/AVX-512 on x86_64, scalar elsewhere).
fn assert_golden(
    name: &str,
    ty: GgufTensorType,
    reference: &dyn QuantKernel,
    block: &[u8],
    expected: &[f32],
) {
    let dispatched = KernelDispatcher::new()
        .get_kernel(ty)
        .expect("dispatched kernel available");

    for (tier, kernel) in [("reference", reference), ("dispatched", &*dispatched)] {
        let mut got = vec![f32::NAN; expected.len()];
        kernel
            .dequant_block(block, &mut got)
            .expect("dequant_block failed");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g,
                e,
                "{name} [{tier}]: weight[{i}] = {g}, GGML says {e}\n\
                 got      = {:?}\nexpected = {:?}",
                &got[..expected.len().min(40)],
                &expected[..expected.len().min(40)]
            );
        }
    }
}

/// Assert `gemv` pairs weight `i` with activation `i`.
///
/// The activation vector is the ramp `x[i] = i`, which is the cheapest input
/// that makes the dot product sensitive to *ordering*: any permutation of the
/// weights changes the result.  A constant activation vector would not.
fn assert_golden_gemv(
    name: &str,
    ty: GgufTensorType,
    reference: &dyn QuantKernel,
    block: &[u8],
    expected: &[f32],
) {
    let n = expected.len();
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let want: f32 = expected.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
    let scale = want.abs().max(1.0);

    let dispatched = KernelDispatcher::new()
        .get_kernel(ty)
        .expect("dispatched kernel available");
    let tensor = QuantTensor::new(block.to_vec(), vec![1, n], ty);

    for (tier, kernel) in [("reference", reference), ("dispatched", &*dispatched)] {
        let mut got = [0.0f32; 1];
        kernel.gemv(&tensor, &input, &mut got).expect("gemv failed");
        let err = (got[0] - want).abs() / scale;
        assert!(
            err <= 1e-5,
            "{name} [{tier}]: gemv against ramp = {}, GGML layout gives {want} (rel err {err})",
            got[0]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_0 — split-half nibble layout
// ─────────────────────────────────────────────────────────────────────────────

/// Upstream `dequantize_row_q4_0` writes
/// `y[j] = ((qs[j] & 0xF) - 8) * d` and `y[j + 16] = ((qs[j] >> 4) - 8) * d`.
///
/// With `d = 0.5` and `qs[j] = j | ((15 - j) << 4)`:
/// * `y[j]      = (j - 8) * 0.5`         → `-4.0, -3.5, …, 3.5`
/// * `y[j + 16] = (15 - j - 8) * 0.5`    → `3.5, 3.0, …, -4.0`
///
/// The interleaved layout (`y[2j]`, `y[2j+1]`) would instead give
/// `-4.0, 3.5, -3.5, 3.0, …`.
#[test]
fn golden_q4_0() {
    let expected: Vec<f32> = (0..16)
        .map(|j| (j as f32 - 8.0) * 0.5)
        .chain((0..16).map(|j| (7.0 - j as f32) * 0.5))
        .collect();
    assert_eq!(expected, EXPECT_Q4_0, "hand derivation vs GGML output");
    assert_golden(
        "Q4_0",
        GgufTensorType::Q4_0,
        &Q4_0Ref,
        &block_q4_0(),
        &EXPECT_Q4_0,
    );
    assert_golden_gemv(
        "Q4_0",
        GgufTensorType::Q4_0,
        &Q4_0Ref,
        &block_q4_0(),
        &EXPECT_Q4_0,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_1
// ─────────────────────────────────────────────────────────────────────────────

/// Upstream `dequantize_row_q4_1`:
/// `y[j] = (qs[j] & 0xF) * d + m`, `y[j + 16] = (qs[j] >> 4) * d + m`.
///
/// With `d = 0.5`, `m = 0.25`: `y[j] = 0.5 j + 0.25`,
/// `y[j + 16] = 0.5 (15 - j) + 0.25`.
#[test]
fn golden_q4_1() {
    let expected: Vec<f32> = (0..16)
        .map(|j| 0.5 * j as f32 + 0.25)
        .chain((0..16).map(|j| 0.5 * (15.0 - j as f32) + 0.25))
        .collect();
    assert_eq!(expected, EXPECT_Q4_1, "hand derivation vs GGML output");
    assert_golden(
        "Q4_1",
        GgufTensorType::Q4_1,
        &Q4_1Ref,
        &block_q4_1(),
        &EXPECT_Q4_1,
    );
    assert_golden_gemv(
        "Q4_1",
        GgufTensorType::Q4_1,
        &Q4_1Ref,
        &block_q4_1(),
        &EXPECT_Q4_1,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// IQ4_NL
// ─────────────────────────────────────────────────────────────────────────────

/// Upstream `dequantize_row_iq4_nl`:
/// `y[j] = d * kvalues_iq4nl[qs[j] & 0xF]`,
/// `y[j + 16] = d * kvalues_iq4nl[qs[j] >> 4]`.
///
/// With `d = 1.0` this is the `kvalues` table forwards in `y[0..16]` and
/// backwards in `y[16..32]`.
#[test]
fn golden_iq4_nl() {
    const KV: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];
    let expected: Vec<f32> = KV.iter().copied().chain(KV.iter().rev().copied()).collect();
    assert_eq!(expected, EXPECT_IQ4_NL, "hand derivation vs GGML output");
    assert_golden(
        "IQ4_NL",
        GgufTensorType::Iq4Nl,
        &Iq4NlRef,
        &block_iq4_nl(),
        &EXPECT_IQ4_NL,
    );
    assert_golden_gemv(
        "IQ4_NL",
        GgufTensorType::Iq4Nl,
        &Iq4NlRef,
        &block_iq4_nl(),
        &EXPECT_IQ4_NL,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// IQ4_XS
// ─────────────────────────────────────────────────────────────────────────────

/// Upstream `dequantize_row_iq4_xs` applies the split-half layout *per 32-weight
/// sub-block*: for sub-block `ib` with `dl = d * (ls - 32)`,
/// `y[32 ib + j] = dl * kv[qs[16 ib + j] & 0xF]` and
/// `y[32 ib + 16 + j] = dl * kv[qs[16 ib + j] >> 4]` for `j` in `0..16`.
///
/// With `d = 0.5`, `ls[ib] - 32 = ib`, and `qs[k] = nib(k)` (so within every
/// sub-block the low nibbles run `0..16` and the high nibbles `15..0`):
/// `y[32 ib + j] = 0.5 ib * kv[j]`, `y[32 ib + 16 + j] = 0.5 ib * kv[15 - j]`.
#[test]
fn golden_iq4_xs() {
    const KV: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];
    let mut expected = vec![0.0f32; 256];
    for ib in 0..8 {
        let dl = 0.5 * ib as f32;
        for j in 0..16 {
            expected[32 * ib + j] = dl * KV[j];
            expected[32 * ib + 16 + j] = dl * KV[15 - j];
        }
    }
    assert_eq!(expected, EXPECT_IQ4_XS, "hand derivation vs GGML output");
    assert_golden(
        "IQ4_XS",
        GgufTensorType::Iq4Xs,
        &Iq4XsRef,
        &block_iq4_xs(),
        &EXPECT_IQ4_XS,
    );
    assert_golden_gemv(
        "IQ4_XS",
        GgufTensorType::Iq4Xs,
        &Iq4XsRef,
        &block_iq4_xs(),
        &EXPECT_IQ4_XS,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Q5_K — `qh` bit assignment
// ─────────────────────────────────────────────────────────────────────────────

/// Upstream `dequantize_row_q5_K` walks the block in four 64-weight groups
/// with `u1 = 1, u2 = 2` doubled twice per group, so for group `g`:
/// the low-nibble half takes its 5th bit from `qh` bit `2g` and the
/// high-nibble half from `qh` bit `2g + 1`.
///
/// Hence `qh[0]`'s eight bits feed weights 0, 32, 64, 96, 128, 160, 192, 224.
/// The buggy mapping (`bit g` / `bit g + 4`) instead feeds
/// 0, 64, 128, 192, 32, 96, 160, 224 — agreeing only on bits 0 and 7.
///
/// With `qh[l] = 1 << (l & 7)`, `qs[k] = nib(k)`, `d = 0.5`, `dmin = 0.25`:
/// * `y[64g + l]      = 0.5 sc[2g]   * ((l % 16)      + 16·[l & 7 == 2g])     - 0.25 mn[2g]`
/// * `y[64g + 32 + l] = 0.5 sc[2g+1] * ((15 - l % 16) + 16·[l & 7 == 2g + 1]) - 0.25 mn[2g+1]`
///
/// Spot check `y[0]`: `g = 0`, `l = 0`, `l & 7 == 0 == 2g` so the 5th bit is
/// set: `0.5·1·(0 + 16) - 0.25·5 = 8 - 1.25 = 6.75`.
#[test]
fn golden_q5_k() {
    let mut expected = vec![0.0f32; 256];
    for g in 0..4usize {
        for l in 0..32usize {
            let lo_bump = if l & 7 == 2 * g { 16.0 } else { 0.0 };
            let hi_bump = if l & 7 == 2 * g + 1 { 16.0 } else { 0.0 };
            expected[64 * g + l] = 0.5 * SC[2 * g] * ((l % 16) as f32 + lo_bump) - 0.25 * MN[2 * g];
            expected[64 * g + 32 + l] =
                0.5 * SC[2 * g + 1] * ((15 - (l % 16)) as f32 + hi_bump) - 0.25 * MN[2 * g + 1];
        }
    }
    assert_eq!(expected, EXPECT_Q5_K, "hand derivation vs GGML output");
    assert_golden(
        "Q5_K",
        GgufTensorType::Q5K,
        &Q5KRef,
        &block_q5_k(),
        &EXPECT_Q5_K,
    );
    assert_golden_gemv(
        "Q5_K",
        GgufTensorType::Q5K,
        &Q5KRef,
        &block_q5_k(),
        &EXPECT_Q5_K,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// TQ2_0 — digit-major decode order
// ─────────────────────────────────────────────────────────────────────────────

/// Upstream `dequantize_row_tq2_0` iterates `j` (32-byte group), then the
/// *digit* `l` in `0..4`, then `m` in `0..32`, emitting sequentially.  So
/// `y[128·(j/32) + 32 l + m] = (((qs[j + m] >> 2l) & 3) - 1) * d`: digit `l`
/// of byte `m` lands 32 weights apart, not 1 apart.
///
/// With `qs[m < 32] = 0x24` (digits `0,1,2,0`) and `qs[m >= 32] = 0x49`
/// (digits `1,2,0,1`), and `d = 0.5`, the whole block is piecewise constant in
/// runs of 32:
/// `-0.5, 0.0, 0.5, -0.5` then `0.0, 0.5, -0.5, 0.0`.
/// The buggy `y[4m + l]` order would instead repeat `-0.5, 0.0, 0.5, -0.5`
/// every four weights.
#[test]
fn golden_tq2_0() {
    let runs = [-0.5f32, 0.0, 0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let expected: Vec<f32> = (0..256).map(|i: usize| runs[i / 32]).collect();
    assert_eq!(expected, EXPECT_TQ2_0, "hand derivation vs GGML output");
    assert_golden(
        "TQ2_0",
        GgufTensorType::Tq2_0,
        &Tq2_0Ref,
        &block_tq2_0(),
        &EXPECT_TQ2_0,
    );
    assert_golden_gemv(
        "TQ2_0",
        GgufTensorType::Tq2_0,
        &Tq2_0Ref,
        &block_tq2_0(),
        &EXPECT_TQ2_0,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// TQ1_0 — base-3 fixed-point decode, digit-major order
// ─────────────────────────────────────────────────────────────────────────────

/// TQ1_0 is not a plain base-3 packing.  `quantize_row_tq1_0_ref` builds
/// `q = Σ xi_n · 3^(4-n)` in `0..243` and then stores
/// `ceil(q · 256 / 243)`; `dequantize_row_tq1_0` recovers digit `n` with
/// `xi = ((u16)(u8)(byte · 3^n) · 3) >> 8`.  Decoding the stored byte with
/// naive `% 3` / `/ 3` arithmetic yields *different values*, not merely a
/// different order.
///
/// The block below is upstream's own encoding of `t(i) = (i % 3) - 1` for
/// `i` in `0..256` (so `amax = 1` and `d = 1.0`).  Because both the encoder's
/// and the decoder's index maps are the identity, a correct decoder must
/// return `t` exactly.
///
/// Worked example for `qs[0] = 0x45 = 69`.  It packs
/// `x[0], x[32], x[64], x[96], x[128] = -1, +1, 0, -1, +1`, i.e.
/// `xi = 0, 2, 1, 0, 2` → `q = ((((0·3+2)·3+1)·3+0)·3+2) = 65` →
/// `ceil(65·256/243) = 69`.  Decoding:
/// * `n=0`: `69·1 = 69`,   `(69·3)>>8 = 0`   → `-1`
/// * `n=1`: `69·3 = 207`,  `(207·3)>>8 = 2`  → `+1`
/// * `n=2`: `69·9 = 621 → 109 (mod 256)`, `(109·3)>>8 = 1` → `0`
/// * `n=3`: `69·27 = 1863 → 71`,  `(71·3)>>8 = 0`  → `-1`
/// * `n=4`: `69·81 = 5589 → 213`, `(213·3)>>8 = 2` → `+1`
///
/// The old `% 3` decoder gives `-1, 0, 0, -1, -1` for the same byte.
#[test]
fn golden_tq1_0() {
    let expected: Vec<f32> = (0..256).map(|i: usize| (i % 3) as f32 - 1.0).collect();
    assert_golden(
        "TQ1_0",
        GgufTensorType::Tq1_0,
        &Tq1_0Ref,
        &block_tq1_0(),
        &expected,
    );
    assert_golden_gemv(
        "TQ1_0",
        GgufTensorType::Tq1_0,
        &Tq1_0Ref,
        &block_tq1_0(),
        &expected,
    );
}

/// The all-`+1` block is the cheapest discriminator for the base-3 decode.
///
/// `q = 242` (five trits of value 2) → stored `ceil(242·256/243) = 255`.
/// Upstream decodes `255` to `+1, +1, +1, +1, +1`; the naive `% 3` decoder
/// gives `-1, 0, 0, -1, -1`.  The `qh` bytes pack four trits shifted up one
/// position: `q = ((((0·3+2)·3+2)·3+2)·3+2)·3 = 240` → `ceil(240·256/243) = 253`.
#[test]
fn golden_tq1_0_all_plus_one() {
    let mut block = vec![0xFFu8; 48];
    block.extend_from_slice(&[253u8; 4]);
    block.extend_from_slice(&f16le(1.0));
    let expected = vec![1.0f32; 256];
    assert_golden(
        "TQ1_0 all +1",
        GgufTensorType::Tq1_0,
        &Tq1_0Ref,
        &block,
        &expected,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Controls — formats that were already correct.
// ─────────────────────────────────────────────────────────────────────────────

/// Q5_0 already used the split-half layout.  `qh = 0x0F0F0F0F` sets bits
/// 0..4 and 8..12 (the low half's 5th bits) and bits 16..20 and 24..28
/// (the high half's, read as bit `j + 16`).
#[test]
fn golden_q5_0_control() {
    assert_golden(
        "Q5_0",
        GgufTensorType::Q5_0,
        &Q5_0Ref,
        &block_q5_0(),
        &EXPECT_Q5_0,
    );
    assert_golden_gemv(
        "Q5_0",
        GgufTensorType::Q5_0,
        &Q5_0Ref,
        &block_q5_0(),
        &EXPECT_Q5_0,
    );
}

/// Q8_0 has no packing at all: `y[j] = qs[j] * d`.
#[test]
fn golden_q8_0_control() {
    let expected: Vec<f32> = (0..32).map(|j| (j as f32 - 16.0) * 0.5).collect();
    assert_eq!(expected, EXPECT_Q8_0, "hand derivation vs GGML output");
    assert_golden(
        "Q8_0",
        GgufTensorType::Q8_0,
        &Q8_0Ref,
        &block_q8_0(),
        &EXPECT_Q8_0,
    );
    assert_golden_gemv(
        "Q8_0",
        GgufTensorType::Q8_0,
        &Q8_0Ref,
        &block_q8_0(),
        &EXPECT_Q8_0,
    );
}

/// Q4_K already used `l` / `l + 32` within each 64-weight group — the same
/// convention Q4_0 must adopt, one level up.
/// `y[64g + l] = 0.5 sc[2g] (l % 16) - 0.25 mn[2g]`,
/// `y[64g + 32 + l] = 0.5 sc[2g+1] (15 - l % 16) - 0.25 mn[2g+1]`.
#[test]
fn golden_q4_k_control() {
    let mut expected = vec![0.0f32; 256];
    for g in 0..4usize {
        for l in 0..32usize {
            expected[64 * g + l] = 0.5 * SC[2 * g] * (l % 16) as f32 - 0.25 * MN[2 * g];
            expected[64 * g + 32 + l] =
                0.5 * SC[2 * g + 1] * (15 - (l % 16)) as f32 - 0.25 * MN[2 * g + 1];
        }
    }
    assert_eq!(expected, EXPECT_Q4_K, "hand derivation vs GGML output");
    assert_golden(
        "Q4_K",
        GgufTensorType::Q4K,
        &Q4KRef,
        &block_q4_k(),
        &EXPECT_Q4_K,
    );
    assert_golden_gemv(
        "Q4_K",
        GgufTensorType::Q4K,
        &Q4KRef,
        &block_q4_k(),
        &EXPECT_Q4_K,
    );
}

/// Q6_K already used `l` / `l + 32` / `l + 64` / `l + 96`.
#[test]
fn golden_q6_k_control() {
    assert_golden(
        "Q6_K",
        GgufTensorType::Q6K,
        &Q6KRef,
        &block_q6_k(),
        &EXPECT_Q6_K,
    );
    assert_golden_gemv(
        "Q6_K",
        GgufTensorType::Q6K,
        &Q6KRef,
        &block_q6_k(),
        &EXPECT_Q6_K,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoders
// ─────────────────────────────────────────────────────────────────────────────

/// Input used for both Q4_0 encoder tests: a ramp whose extreme is negative,
/// which is what makes the *signed* `d = max / -8` convention observable.
///
/// `x[j] = (j - 20) / 4` runs `-5.00 … +2.75`.  The largest magnitude is
/// `|x[0]| = 5`, and `max` keeps its sign, so `d = -5 / -8 = 0.625` — exactly
/// representable in `f16`.  Under the rejected `amax / 7` convention the scale
/// would be `5/7 ≈ 0.714`, a different value and different bytes.
fn q4_0_encoder_input() -> Vec<f32> {
    (0..32).map(|j| (j as f32 - 20.0) * 0.25).collect()
}

/// Both Q4_0 encoders must emit exactly the bytes `quantize_row_q4_0_ref`
/// emits.
///
/// Upstream packs `qs[j] = xi(x[j]) | (xi(x[j + 16]) << 4)` where
/// `xi(v) = min(15, trunc(v / d + 8.5))`.  Spot check `qs[0] = 0x60`:
/// `x[0] = -5`, `-5 / 0.625 = -8`, `trunc(-8 + 8.5) = 0`;
/// `x[16] = -1`, `-1 / 0.625 = -1.6`, `trunc(-1.6 + 8.5) = 6`.
#[test]
fn golden_q4_0_encoder_bytes() {
    let src = q4_0_encoder_input();
    let mut want = f16le(0.625).to_vec();
    want.extend_from_slice(&Q4_0_GOLDEN_ENC_QS);

    let quant = oxillama_quant::quantize_f32_to_q4_0(&src).expect("oxillama-quant encoder");
    assert_eq!(quant, want, "oxillama-quant quantize_f32_to_q4_0 vs GGML");

    let gguf = oxillama_gguf::quantize_on_load::encode_q4_0(&src);
    assert_eq!(gguf, want, "oxillama-gguf encode_q4_0 vs GGML");

    // Explicit cross-encoder identity, as required even if both were wrong in
    // the same way.
    assert_eq!(quant, gguf, "the two Q4_0 encoders disagree");
}

/// Encode → decode must reproduce upstream's own round-trip, including the
/// split-half layout.
#[test]
fn golden_q4_0_encoder_round_trip() {
    let src = q4_0_encoder_input();
    let block = oxillama_quant::quantize_f32_to_q4_0(&src).expect("encoder");
    assert_golden(
        "Q4_0 round-trip",
        GgufTensorType::Q4_0,
        &Q4_0Ref,
        &block,
        &EXPECT_Q4_0_ENC_ROUNDTRIP,
    );
}

/// The two encoders agree byte-for-byte on a wide range of inputs, not just
/// the golden one.
#[test]
fn q4_0_encoders_agree_on_random_input() {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 8_388_608.0 - 1.0
    };
    for case in 0..64 {
        let scale = 1e-3f32 * (1 << (case % 12)) as f32;
        let src: Vec<f32> = (0..128).map(|_| next() * scale).collect();
        let a = oxillama_quant::quantize_f32_to_q4_0(&src).expect("encoder");
        let b = oxillama_gguf::quantize_on_load::encode_q4_0(&src);
        assert_eq!(a, b, "encoders disagree on case {case}");
    }
}

/// TQ2_0: upstream's `quantize_row_tq2_0_ref` output for `t(i) = (i % 3) - 1`
/// must decode back to `t` under the corrected decoder.
///
/// This pins the *encoder's* index map (`qs[j + m]` digit `n` ← `x[m + 32n]`)
/// against the decoder's, which is what the digit-major order exists to
/// satisfy.
#[test]
fn golden_tq2_0_upstream_encoding_round_trip() {
    let mut block = TQ2_0_GOLDEN_ENC_QS.to_vec();
    block.extend_from_slice(&f16le(1.0));
    let expected: Vec<f32> = (0..256).map(|i: usize| (i % 3) as f32 - 1.0).collect();
    assert_golden(
        "TQ2_0 upstream encoding",
        GgufTensorType::Tq2_0,
        &Tq2_0Ref,
        &block,
        &expected,
    );
}

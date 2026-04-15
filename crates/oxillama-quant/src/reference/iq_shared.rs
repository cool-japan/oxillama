//! Shared constants for IQ4_NL and IQ4_XS quantization kernels.
//!
//! The non-linear 4-bit lookup table `KVALUES_IQ4NL` is used by both
//! IQ4_NL and IQ4_XS to map 4-bit nibble indices (0..=15) to signed
//! integer weights before scaling by the block's FP16 delta.

/// Non-linear 4-bit quantization lookup table for IQ4_NL / IQ4_XS.
///
/// Values from llama.cpp `ggml-quants.c` — these 16 signed integers
/// approximate a non-uniform distribution that minimises reconstruction
/// error relative to uniform Q4_0 for typical LLM weight distributions.
///
/// Index 0 encodes the most-negative value (-127) and index 15 the
/// most-positive (113).  A 4-bit nibble in a block payload is used
/// directly as an index into this table.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

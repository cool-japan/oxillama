//! IQ1S grid codebook — 2048 entries of packed 8×i8 weight values.
//
//! Each `u64` encodes 8 signed weight values (one byte each, little-endian).
//! The bytes must be reinterpreted as `i8` (0xff = -1, 0x00 = 0, 0x01 = 1).
//! Source: llama.cpp `ggml-common.h` (GGML_COMMON_IMPL_C guard),
//!         `iq1s_grid[NGRID_IQ1S]` where `NGRID_IQ1S = 2048`.
//!
//! The table is split across two sub-modules (`data_a` and `data_b`) to keep
//! each source file under 2000 lines, then reassembled here as a single
//! compile-time constant using a const initializer block.

mod data_a;
mod data_b;

/// IQ1S codebook — 2048 entries of packed 8×i8 weight values (stored as u64).
///
/// To extract the signed byte values:
/// ```text
/// let raw: [u8; 8] = IQ1S_GRID[idx].to_le_bytes();
/// let weights: [i8; 8] = raw.map(|b| b as i8);
/// ```
pub const IQ1S_GRID: [u64; 2048] = {
    let mut arr = [0u64; 2048];
    let mut i = 0usize;
    while i < 1024 {
        arr[i] = data_a::IQ1S_GRID_A[i];
        i += 1;
    }
    while i < 2048 {
        arr[i] = data_b::IQ1S_GRID_B[i - 1024];
        i += 1;
    }
    arr
};

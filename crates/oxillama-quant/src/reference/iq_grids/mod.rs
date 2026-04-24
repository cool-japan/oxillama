//! Shared grid (codebook) tables for IQ2/IQ3 importance quantization kernels.
//!
//! These tables are derived from llama.cpp's `ggml-common.h` (under the
//! `GGML_COMMON_IMPL` guard).  They are embedded as compile-time constants
//! so the compiler can emit them in read-only memory without any heap
//! allocation.
//!
//! # Table descriptions
//!
//! | Name             | Type  | Size | Description                                    |
//! |------------------|-------|------|------------------------------------------------|
//! [`KMASK_IQ2XS`]   | u8    | 8    | Bit-mask for each of the 8 weight signs        |
//! [`KSIGNS_IQ2XS`]  | u8    | 128  | 7-bit → 8-sign byte lookup                     |
//! [`IQ2XXS_GRID`]   | u64   | 256  | IQ2_XXS codebook — 8 weight magnitudes per u64 |
//! [`IQ2XS_GRID`]    | u64   | 512  | IQ2_XS  codebook — 8 weight magnitudes per u64 |
//! [`IQ2S_GRID`]     | u64   | 1024 | IQ2_S   codebook — 8 weight magnitudes per u64 |
//! [`IQ3XXS_GRID`]   | u32   | 256  | IQ3_XXS codebook — 4 weight magnitudes per u32 |
//! [`IQ3S_GRID`]     | u32   | 512  | IQ3_S   codebook — 4 weight magnitudes per u32 |
//!
//! ## Decoding grid entries
//!
//! Both grids store unsigned weight magnitudes packed as individual bytes.
//! To extract the bytes in Rust, call `.to_le_bytes()` on each entry:
//!
//! ```text
//! let weights_u8: [u8; 8] = IQ2XXS_GRID[idx].to_le_bytes();
//! let weights_u8: [u8; 8] = IQ2XS_GRID[idx].to_le_bytes();
//! let weights_u8: [u8; 8] = IQ2S_GRID[idx].to_le_bytes();
//! let weights_u8: [u8; 4] = IQ3XXS_GRID[idx].to_le_bytes();
//! let weights_u8: [u8; 4] = IQ3S_GRID[idx].to_le_bytes();
//! ```
//!
//! Signs are then applied from [`KSIGNS_IQ2XS`] indexed by a 7-bit value
//! extracted from the block header, combined with [`KMASK_IQ2XS`] bit-tests.

pub mod iq2_s;
pub mod iq2_xs;
pub mod iq2_xxs;
pub mod iq3_s;
pub mod iq3_xxs;
pub mod signs;

pub use iq2_s::IQ2S_GRID;
pub use iq2_xs::IQ2XS_GRID;
pub use iq2_xxs::IQ2XXS_GRID;
pub use iq3_s::IQ3S_GRID;
pub use iq3_xxs::IQ3XXS_GRID;
pub use signs::{KMASK_IQ2XS, KSIGNS_IQ2XS};

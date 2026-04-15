//! Reference (naive) implementations of quantization kernels.
//!
//! These are pure scalar Rust implementations used for:
//! - Correctness verification against SIMD-optimized kernels.
//! - Fallback on platforms without SIMD support.
//! - Property-based testing baselines.
//!
//! Each quantization type has its own submodule here.

pub mod bf16;
pub mod f16;
pub mod f32;
pub mod iq1_m;
pub mod iq1_s;
pub mod iq1s_grid;
pub mod iq1s_table_hi;
pub mod iq1s_table_lo;
pub mod iq2_s;
pub mod iq2_xs;
pub mod iq2_xxs;
pub mod iq2s_table;
pub mod iq3_s;
pub mod iq3_xxs;
pub mod iq4_nl;
pub mod iq4_xs;
pub mod iq_grids;
pub mod iq_shared;
pub mod q1_0_g128;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod q8_1;
pub mod q8_k;
pub mod tq1_0;
pub mod tq2_0;

pub use bf16::Bf16Ref;
pub use f16::F16Ref;
pub use f32::F32Ref;
pub use iq1_m::Iq1MRef;
pub use iq1_s::Iq1SRef;
pub use iq2_s::Iq2SRef;
pub use iq2_xs::Iq2XsRef;
pub use iq2_xxs::Iq2XxsRef;
pub use iq3_s::Iq3SRef;
pub use iq3_xxs::Iq3XxsRef;
pub use iq4_nl::Iq4NlRef;
pub use iq4_xs::Iq4XsRef;
pub use q1_0_g128::Q1_0G128Ref;
pub use q2_k::Q2KRef;
pub use q3_k::Q3KRef;
pub use q4_0::Q4_0Ref;
pub use q4_1::Q4_1Ref;
pub use q4_k::Q4KRef;
pub use q5_0::Q5_0Ref;
pub use q5_1::Q5_1Ref;
pub use q5_k::Q5KRef;
pub use q6_k::Q6KRef;
pub use q8_0::Q8_0Ref;
pub use q8_1::Q8_1Ref;
pub use q8_k::Q8KRef;
pub use tq1_0::Tq1_0Ref;
pub use tq2_0::Tq2_0Ref;

//! Platform-specific SIMD quantization kernels.
//!
//! This module provides a cached runtime capability detection entry point
//! and re-exports each platform's submodule behind its appropriate feature
//! and target gates.
//!
//! # Runtime detection
//!
//! [`cached_capabilities`] returns a `&'static SimdCapabilities` that is
//! initialised exactly once (via [`std::sync::OnceLock`]) on first access.
//! Subsequent calls return the cached value with zero overhead.
//!
//! # Sub-modules
//!
//! | Module | Feature flag | Target |
//! |--------|-------------|--------|
//! | `avx2` | `simd-avx2` | `x86_64` |

use std::sync::OnceLock;

use crate::dispatch::SimdCapabilities;

/// Cached SIMD capability detection result.
///
/// Populated on the first call to [`cached_capabilities`] and reused for
/// all subsequent calls.
static CACHED_CAPS: OnceLock<SimdCapabilities> = OnceLock::new();

/// Return the detected SIMD capabilities for this CPU, lazily computed once.
///
/// Uses [`std::sync::OnceLock`] so the underlying detection (`CPUID` on
/// x86_64, compile-time flag on aarch64) runs at most once per process.
pub fn cached_capabilities() -> &'static SimdCapabilities {
    CACHED_CAPS.get_or_init(SimdCapabilities::detect)
}

/// AVX2+FMA kernels (x86_64 only, `simd-avx2` feature).
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
pub mod avx2;

/// AVX-512 kernels (x86_64 only, `simd-avx512` feature).
#[cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]
pub mod avx512;

/// AArch64 NEON kernels (`simd-neon` feature).
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
pub mod neon;

/// oxiblas-backed GEMM kernels for F32, F16, and BF16.
///
/// Always available — no CPU feature gate required since oxiblas is a
/// workspace dependency for all configurations.
pub mod float_gemm;

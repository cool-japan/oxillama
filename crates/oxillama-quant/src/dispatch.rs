//! Runtime kernel selection and dispatch.
//!
//! Selects the best available [`QuantKernel`] implementation for a given
//! quantization type based on compile-time feature flags and runtime
//! CPU feature detection.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use oxillama_gguf::GgufTensorType;

use crate::error::{QuantError, QuantResult};
use crate::reference::{
    Bf16Ref, F16Ref, F32Ref, Iq1MRef, Iq1SRef, Iq2SRef, Iq2XsRef, Iq2XxsRef, Iq3SRef, Iq3XxsRef,
    Iq4NlRef, Iq4XsRef, Q1_0G128Ref, Q2KRef, Q3KRef, Q4KRef, Q4_0Ref, Q4_1Ref, Q5KRef, Q5_0Ref,
    Q5_1Ref, Q6KRef, Q8KRef, Q8_0Ref, Q8_1Ref, Tq1_0Ref, Tq2_0Ref,
};
#[cfg(any(
    all(feature = "simd-avx512", target_arch = "x86_64"),
    all(feature = "simd-avx2", target_arch = "x86_64"),
    all(feature = "simd-neon", target_arch = "aarch64"),
))]
use crate::simd;
use crate::simd::float_gemm::{Bf16OxiblasKernel, F16OxiblasKernel, F32OxiblasKernel};
use crate::traits::QuantKernel;

/// Detected CPU SIMD capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    /// x86_64 AVX2 support.
    pub avx2: bool,
    /// x86_64 AVX-512F support.
    pub avx512f: bool,
    /// x86_64 FMA support (usually paired with AVX2).
    pub fma: bool,
    /// ARM NEON support.
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect CPU SIMD capabilities at runtime.
    pub fn detect() -> Self {
        Self {
            avx2: Self::detect_avx2(),
            avx512f: Self::detect_avx512f(),
            fma: Self::detect_fma(),
            neon: Self::detect_neon(),
        }
    }

    /// Returns the best available SIMD tier name for display.
    pub fn best_tier(&self) -> &'static str {
        if self.avx512f {
            "AVX-512"
        } else if self.avx2 && self.fma {
            "AVX2+FMA"
        } else if self.avx2 {
            "AVX2"
        } else if self.neon {
            "NEON"
        } else {
            "scalar"
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect_avx2() -> bool {
        false
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_avx512f() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect_avx512f() -> bool {
        false
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_fma() -> bool {
        std::arch::is_x86_feature_detected!("fma")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect_fma() -> bool {
        false
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_neon() -> bool {
        // NEON is mandatory on aarch64
        true
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn detect_neon() -> bool {
        false
    }
}

/// Kernel dispatcher — selects and caches the best kernel for each quant type.
///
/// The dispatcher checks (in order):
/// 1. Platform-specific SIMD kernels (AVX-512, AVX2, NEON) if features enabled.
/// 2. Portable SIMD kernels.
/// 3. Reference (naive) scalar kernels.
#[derive(Debug)]
pub struct KernelDispatcher {
    /// Detected CPU SIMD capabilities.
    pub capabilities: SimdCapabilities,
}

impl Default for KernelDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelDispatcher {
    /// Create a new dispatcher with runtime CPU feature detection.
    pub fn new() -> Self {
        Self {
            capabilities: SimdCapabilities::detect(),
        }
    }

    /// Get the best available kernel for the given quantization type.
    ///
    /// Currently returns reference (scalar) kernels for all types.
    /// When SIMD features are enabled and the CPU supports them,
    /// this will return optimized SIMD kernels instead.
    ///
    /// Returns an error if the quantization type is not yet implemented.
    pub fn get_kernel(&self, tensor_type: GgufTensorType) -> QuantResult<Box<dyn QuantKernel>> {
        // 1. AVX-512 path
        #[cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]
        if simd::cached_capabilities().avx512f {
            match tensor_type {
                GgufTensorType::Q4_0 => return Ok(Box::new(simd::avx512::Q4_0Avx512)),
                GgufTensorType::Q4_1 => return Ok(Box::new(simd::avx512::Q4_1Avx512)),
                GgufTensorType::Q8_0 => return Ok(Box::new(simd::avx512::Q8_0Avx512)),
                GgufTensorType::Q8_1 => return Ok(Box::new(simd::avx512::Q8_1Avx512)),
                GgufTensorType::Q2K => return Ok(Box::new(simd::avx512::Q2_KAvx512)),
                GgufTensorType::Q3K => return Ok(Box::new(simd::avx512::Q3_KAvx512)),
                GgufTensorType::Q4K => return Ok(Box::new(simd::avx512::Q4_KAvx512)),
                GgufTensorType::Q1_0G128 => return Ok(Box::new(simd::avx512::Q1_0G128Avx512)),
                GgufTensorType::Q5K => return Ok(Box::new(simd::avx512::Q5_KAvx512)),
                GgufTensorType::Q5_1 => return Ok(Box::new(simd::avx512::Q5_1Avx512)),
                GgufTensorType::Q6K => return Ok(Box::new(simd::avx512::Q6_KAvx512)),
                GgufTensorType::Tq1_0 => return Ok(Box::new(simd::avx512::Tq1_0Avx512)),
                GgufTensorType::Tq2_0 => return Ok(Box::new(simd::avx512::Tq2_0Avx512)),
                GgufTensorType::Q5_0 => return Ok(Box::new(simd::avx512::Q5_0Avx512)),
                GgufTensorType::Q8K => return Ok(Box::new(simd::avx512::Q8_KAvx512)),
                GgufTensorType::Iq2Xxs => return Ok(Box::new(simd::avx512::Iq2XxsAvx512)),
                GgufTensorType::Iq2Xs => return Ok(Box::new(simd::avx512::Iq2XsAvx512)),
                GgufTensorType::Iq3S => return Ok(Box::new(simd::avx512::Iq3SAvx512)),
                GgufTensorType::Iq4Xs => return Ok(Box::new(simd::avx512::Iq4XsAvx512)),
                _ => {}
            }
        }

        // 2. AVX2 path
        #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
        if simd::cached_capabilities().avx2 {
            match tensor_type {
                GgufTensorType::Q4_0 => return Ok(Box::new(simd::avx2::Q4_0Avx2)),
                GgufTensorType::Q5_0 => return Ok(Box::new(simd::avx2::Q5_0Avx2)),
                GgufTensorType::Q8_0 => return Ok(Box::new(simd::avx2::Q8_0Avx2)),
                GgufTensorType::Q4K => return Ok(Box::new(simd::avx2::Q4_KAvx2)),
                GgufTensorType::Q5K => return Ok(Box::new(simd::avx2::Q5_KAvx2)),
                GgufTensorType::Q6K => return Ok(Box::new(simd::avx2::Q6_KAvx2)),
                GgufTensorType::Q1_0G128 => return Ok(Box::new(simd::avx2::Q1_0G128Avx2)),
                GgufTensorType::Q2K => return Ok(Box::new(simd::avx2::Q2_KAvx2)),
                GgufTensorType::Q3K => return Ok(Box::new(simd::avx2::Q3_KAvx2)),
                GgufTensorType::Q4_1 => return Ok(Box::new(simd::avx2::Q4_1Avx2)),
                GgufTensorType::Q5_1 => return Ok(Box::new(simd::avx2::Q5_1Avx2)),
                GgufTensorType::Q8_1 => return Ok(Box::new(simd::avx2::Q8_1Avx2)),
                GgufTensorType::Iq1S => return Ok(Box::new(simd::avx2::Iq1SAvx2)),
                GgufTensorType::Iq1M => return Ok(Box::new(simd::avx2::Iq1MAvx2)),
                GgufTensorType::Iq2Xs => return Ok(Box::new(simd::avx2::Iq2XsAvx2)),
                GgufTensorType::Iq2Xxs => return Ok(Box::new(simd::avx2::Iq2XxsAvx2)),
                GgufTensorType::Iq2S => return Ok(Box::new(simd::avx2::Iq2SAvx2)),
                GgufTensorType::Iq3Xxs => return Ok(Box::new(simd::avx2::Iq3XxsAvx2)),
                GgufTensorType::Iq3S => return Ok(Box::new(simd::avx2::Iq3SAvx2)),
                GgufTensorType::Iq4Nl => return Ok(Box::new(simd::avx2::Iq4NlAvx2)),
                GgufTensorType::Iq4Xs => return Ok(Box::new(simd::avx2::Iq4XsAvx2)),
                GgufTensorType::Q8K => return Ok(Box::new(simd::avx2::Q8_KAvx2)),
                GgufTensorType::Tq1_0 => return Ok(Box::new(simd::avx2::Tq1_0Avx2)),
                GgufTensorType::Tq2_0 => return Ok(Box::new(simd::avx2::Tq2_0Avx2)),
                _ => {}
            }
        }

        // 3. NEON path
        #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
        if simd::cached_capabilities().neon {
            match tensor_type {
                GgufTensorType::Q4_0 => return Ok(Box::new(simd::neon::Q4_0Neon)),
                GgufTensorType::Q4_1 => return Ok(Box::new(simd::neon::Q4_1Neon)),
                GgufTensorType::Q5_0 => return Ok(Box::new(simd::neon::Q5_0Neon)),
                GgufTensorType::Q5_1 => return Ok(Box::new(simd::neon::Q5_1Neon)),
                GgufTensorType::Q8_0 => return Ok(Box::new(simd::neon::Q8_0Neon)),
                GgufTensorType::Q8_1 => return Ok(Box::new(simd::neon::Q8_1Neon)),
                GgufTensorType::Q2K => return Ok(Box::new(simd::neon::Q2_KNeon)),
                GgufTensorType::Q3K => return Ok(Box::new(simd::neon::Q3_KNeon)),
                GgufTensorType::Q4K => return Ok(Box::new(simd::neon::Q4_KNeon)),
                GgufTensorType::Q5K => return Ok(Box::new(simd::neon::Q5_KNeon)),
                GgufTensorType::Q1_0G128 => return Ok(Box::new(simd::neon::Q1_0G128Neon)),
                GgufTensorType::Q6K => return Ok(Box::new(simd::neon::Q6_KNeon)),
                GgufTensorType::Q8K => return Ok(Box::new(simd::neon::Q8_KNeon)),
                GgufTensorType::Iq1S => return Ok(Box::new(simd::neon::Iq1SNeon)),
                GgufTensorType::Iq1M => return Ok(Box::new(simd::neon::Iq1MNeon)),
                GgufTensorType::Iq2S => return Ok(Box::new(simd::neon::Iq2SNeon)),
                GgufTensorType::Iq2Xxs => return Ok(Box::new(simd::neon::Iq2XxsNeon)),
                GgufTensorType::Iq2Xs => return Ok(Box::new(simd::neon::Iq2XsNeon)),
                GgufTensorType::Iq3Xxs => return Ok(Box::new(simd::neon::Iq3XxsNeon)),
                GgufTensorType::Iq3S => return Ok(Box::new(simd::neon::Iq3SNeon)),
                GgufTensorType::Iq4Xs => return Ok(Box::new(simd::neon::Iq4XsNeon)),
                GgufTensorType::Iq4Nl => return Ok(Box::new(simd::neon::Iq4NlNeon)),
                GgufTensorType::Tq1_0 => return Ok(Box::new(simd::neon::Tq1_0Neon)),
                GgufTensorType::Tq2_0 => return Ok(Box::new(simd::neon::Tq2_0Neon)),
                _ => {}
            }
        }

        // 4. oxiblas-backed float kernels (always available, no CPU feature gate needed)
        match tensor_type {
            GgufTensorType::F32 => return Ok(Box::new(F32OxiblasKernel)),
            GgufTensorType::F16 => return Ok(Box::new(F16OxiblasKernel)),
            GgufTensorType::Bf16 => return Ok(Box::new(Bf16OxiblasKernel)),
            _ => {}
        }

        // 5. Scalar reference fallback
        self.get_reference_kernel(tensor_type)
    }

    /// Get the reference (scalar) kernel for a given type.
    fn get_reference_kernel(
        &self,
        tensor_type: GgufTensorType,
    ) -> QuantResult<Box<dyn QuantKernel>> {
        match tensor_type {
            GgufTensorType::F32 => Ok(Box::new(F32Ref)),
            GgufTensorType::F16 => Ok(Box::new(F16Ref)),
            GgufTensorType::Bf16 => Ok(Box::new(Bf16Ref)),
            GgufTensorType::Q4_0 => Ok(Box::new(Q4_0Ref)),
            GgufTensorType::Q4_1 => Ok(Box::new(Q4_1Ref)),
            GgufTensorType::Q5_0 => Ok(Box::new(Q5_0Ref)),
            GgufTensorType::Q5_1 => Ok(Box::new(Q5_1Ref)),
            GgufTensorType::Q8_0 => Ok(Box::new(Q8_0Ref)),
            GgufTensorType::Q8_1 => Ok(Box::new(Q8_1Ref)),
            GgufTensorType::Q2K => Ok(Box::new(Q2KRef)),
            GgufTensorType::Q3K => Ok(Box::new(Q3KRef)),
            GgufTensorType::Q4K => Ok(Box::new(Q4KRef)),
            GgufTensorType::Q5K => Ok(Box::new(Q5KRef)),
            GgufTensorType::Q6K => Ok(Box::new(Q6KRef)),
            GgufTensorType::Q8K => Ok(Box::new(Q8KRef)),
            GgufTensorType::Q1_0G128 => Ok(Box::new(Q1_0G128Ref)),
            GgufTensorType::Iq1S => Ok(Box::new(Iq1SRef)),
            GgufTensorType::Iq1M => Ok(Box::new(Iq1MRef)),
            GgufTensorType::Iq2Xxs => Ok(Box::new(Iq2XxsRef)),
            GgufTensorType::Iq2Xs => Ok(Box::new(Iq2XsRef)),
            GgufTensorType::Iq2S => Ok(Box::new(Iq2SRef)),
            GgufTensorType::Iq3Xxs => Ok(Box::new(Iq3XxsRef)),
            GgufTensorType::Iq3S => Ok(Box::new(Iq3SRef)),
            GgufTensorType::Iq4Nl => Ok(Box::new(Iq4NlRef)),
            GgufTensorType::Iq4Xs => Ok(Box::new(Iq4XsRef)),
            GgufTensorType::Tq1_0 => Ok(Box::new(Tq1_0Ref)),
            GgufTensorType::Tq2_0 => Ok(Box::new(Tq2_0Ref)),
            _ => Err(QuantError::UnsupportedType {
                quant_type: tensor_type.name().to_string(),
            }),
        }
    }

    /// Check if a kernel is available for the given quantization type.
    pub fn is_supported(&self, tensor_type: GgufTensorType) -> bool {
        matches!(
            tensor_type,
            GgufTensorType::F32
                | GgufTensorType::F16
                | GgufTensorType::Bf16
                | GgufTensorType::Q4_0
                | GgufTensorType::Q4_1
                | GgufTensorType::Q5_0
                | GgufTensorType::Q5_1
                | GgufTensorType::Q8_0
                | GgufTensorType::Q8_1
                | GgufTensorType::Q2K
                | GgufTensorType::Q3K
                | GgufTensorType::Q4K
                | GgufTensorType::Q5K
                | GgufTensorType::Q6K
                | GgufTensorType::Q8K
                | GgufTensorType::Q1_0G128
                | GgufTensorType::Iq1S
                | GgufTensorType::Iq1M
                | GgufTensorType::Iq2Xxs
                | GgufTensorType::Iq2Xs
                | GgufTensorType::Iq2S
                | GgufTensorType::Iq3Xxs
                | GgufTensorType::Iq3S
                | GgufTensorType::Iq4Nl
                | GgufTensorType::Iq4Xs
                | GgufTensorType::Tq1_0
                | GgufTensorType::Tq2_0
        )
    }

    /// List all currently supported quantization types.
    pub fn supported_types(&self) -> Vec<GgufTensorType> {
        vec![
            GgufTensorType::F32,
            GgufTensorType::F16,
            GgufTensorType::Bf16,
            GgufTensorType::Q2K,
            GgufTensorType::Q3K,
            GgufTensorType::Q4_0,
            GgufTensorType::Q4_1,
            GgufTensorType::Q4K,
            GgufTensorType::Q5_0,
            GgufTensorType::Q5_1,
            GgufTensorType::Q5K,
            GgufTensorType::Q6K,
            GgufTensorType::Q8_0,
            GgufTensorType::Q8_1,
            GgufTensorType::Q8K,
            GgufTensorType::Q1_0G128,
            GgufTensorType::Iq1S,
            GgufTensorType::Iq1M,
            GgufTensorType::Iq2Xxs,
            GgufTensorType::Iq2Xs,
            GgufTensorType::Iq2S,
            GgufTensorType::Iq3Xxs,
            GgufTensorType::Iq3S,
            GgufTensorType::Iq4Nl,
            GgufTensorType::Iq4Xs,
            GgufTensorType::Tq1_0,
            GgufTensorType::Tq2_0,
        ]
    }
}

/// Cached kernel dispatcher — singleton per process.
///
/// Returns `Arc<dyn QuantKernel>` instead of `Box<dyn QuantKernel>`,
/// allowing zero-allocation kernel lookup after the first call for each type.
pub struct CachedDispatcher {
    inner: KernelDispatcher,
    cache: Mutex<HashMap<GgufTensorType, Arc<dyn QuantKernel>>>,
}

impl CachedDispatcher {
    /// Create a new cached dispatcher with runtime CPU feature detection.
    pub fn new() -> Self {
        Self {
            inner: KernelDispatcher::new(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a cached kernel for the given tensor type.
    ///
    /// The first call for each type allocates; subsequent calls return
    /// a clone of the `Arc` (cheap reference count bump).
    pub fn get_kernel(&self, tensor_type: GgufTensorType) -> QuantResult<Arc<dyn QuantKernel>> {
        // Fast path: check if already cached
        {
            let cache = self.cache.lock().map_err(|_| QuantError::Internal {
                message: "kernel cache lock poisoned".to_string(),
            })?;
            if let Some(kernel) = cache.get(&tensor_type) {
                return Ok(Arc::clone(kernel));
            }
        }

        // Slow path: create and cache
        let kernel: Arc<dyn QuantKernel> = self.inner.get_kernel(tensor_type)?.into();
        let mut cache = self.cache.lock().map_err(|_| QuantError::Internal {
            message: "kernel cache lock poisoned".to_string(),
        })?;
        cache
            .entry(tensor_type)
            .or_insert_with(|| Arc::clone(&kernel));
        Ok(kernel)
    }

    /// Access the underlying capabilities.
    pub fn capabilities(&self) -> &SimdCapabilities {
        &self.inner.capabilities
    }

    /// Check if a type is supported (delegates to inner).
    pub fn is_supported(&self, tensor_type: GgufTensorType) -> bool {
        self.inner.is_supported(tensor_type)
    }

    /// List supported types (delegates to inner).
    pub fn supported_types(&self) -> Vec<GgufTensorType> {
        self.inner.supported_types()
    }
}

impl Default for CachedDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global cached dispatcher instance.
static GLOBAL_DISPATCHER: OnceLock<CachedDispatcher> = OnceLock::new();

/// Get the global cached dispatcher singleton.
///
/// The dispatcher is initialized on first call with runtime SIMD detection.
/// Subsequent calls return a reference to the same instance.
pub fn global_dispatcher() -> &'static CachedDispatcher {
    GLOBAL_DISPATCHER.get_or_init(CachedDispatcher::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let caps = SimdCapabilities::detect();
        // Just verify it doesn't panic and returns something
        let tier = caps.best_tier();
        assert!(!tier.is_empty());
    }

    #[test]
    fn test_dispatcher_returns_all_supported() {
        let dispatcher = KernelDispatcher::new();
        for tensor_type in dispatcher.supported_types() {
            let kernel = dispatcher.get_kernel(tensor_type);
            assert!(kernel.is_ok(), "failed to get kernel for {:?}", tensor_type);
        }
    }

    #[test]
    fn test_dispatcher_unsupported() {
        let dispatcher = KernelDispatcher::new();
        // TQ1_0 and TQ2_0 are now supported
        assert!(dispatcher.is_supported(GgufTensorType::Tq1_0));
        assert!(dispatcher.is_supported(GgufTensorType::Tq2_0));
        // Experimental packed types are not supported
        assert!(!dispatcher.is_supported(GgufTensorType::Q4_0_4_4));
        // All IQ types are now supported
        assert!(dispatcher.is_supported(GgufTensorType::Iq1S));
        assert!(dispatcher.is_supported(GgufTensorType::Iq4Nl));
    }

    #[test]
    fn test_cached_dispatcher_returns_same_kernel() {
        let dispatcher = CachedDispatcher::new();
        let k1 = dispatcher.get_kernel(GgufTensorType::Q4_0).expect("k1");
        let k2 = dispatcher.get_kernel(GgufTensorType::Q4_0).expect("k2");
        assert_eq!(k1.name(), k2.name());
        assert!(
            Arc::ptr_eq(&k1, &k2),
            "second call should return cached Arc"
        );
    }

    #[test]
    fn test_global_dispatcher_singleton() {
        let d1 = global_dispatcher();
        let d2 = global_dispatcher();
        assert!(
            std::ptr::eq(d1, d2),
            "global_dispatcher should return same instance"
        );
    }

    #[test]
    fn test_cached_dispatcher_all_types() {
        let dispatcher = CachedDispatcher::new();
        for t in dispatcher.supported_types() {
            let k = dispatcher.get_kernel(t);
            assert!(k.is_ok(), "cached dispatch failed for {:?}", t);
        }
    }
}

//! # oxillama-gpu
//!
//! Optional wgpu-based GPU compute backend for OxiLLaMa.
//!
//! ## Feature flags
//!
//! | Feature | Description | Default |
//! |---------|-------------|---------|
//! | `gpu`   | Enable wgpu device init, buffer helpers, and WGSL shaders | No |
//!
//! When `gpu` is **disabled** (the default) this crate still compiles and all
//! public types are available.  [`GpuContext::try_init`] returns `None` and
//! [`GpuDispatcher::has_gpu`] returns `false`.
//!
//! ## Quick start
//!
//! ```rust
//! use oxillama_gpu::{GpuDispatcher, GpuContext};
//!
//! let dispatcher = GpuDispatcher::new();
//! if dispatcher.has_gpu() {
//!     println!("GPU available — will use hardware acceleration");
//! } else {
//!     println!("No GPU — CPU fallback active");
//! }
//! ```

pub mod buffer;
pub mod context;
pub mod error;
pub mod kernels;

pub use context::GpuContext;
pub use context::GpuDeviceInfo;
pub use error::{GpuError, GpuResult};
pub use kernels::{
    batched_gemv_f32, supports_f16, BatchedGemvConfig, BatchedGpuKernel, F16AccumulatorConfig,
    FusedAttentionKernel, GpuKernel, Iq2SGpuKernel, Iq2XxsGpuKernel, Iq3SGpuKernel,
    Iq3XxsGpuKernel, Iq4XsGpuKernel, Q1_0_G128GpuKernel, Q2_KGpuKernel, Q3_KGpuKernel,
    Q4_0GpuKernel, Q4_1GpuKernel, Q4_KGpuKernel, Q5_0GpuKernel, Q5_1GpuKernel, Q5_KGpuKernel,
    Q6_KGpuKernel, Q8_0GpuKernel, Q8_1GpuKernel, Q8_KGpuKernel, TiledGemmKernel,
};
#[cfg(any(feature = "gpu", test))]
pub use kernels::{dequant_q4_0_to_f16, dequant_q8_0_to_f16};
#[cfg(feature = "gpu")]
pub use kernels::{f16_gemv, upload_f16};

use oxillama_gguf::GgufTensorType;

/// Central dispatcher that holds an optional [`GpuContext`] and vends
/// GPU kernels for supported tensor types.
///
/// Construct with [`GpuDispatcher::new`].  The dispatcher performs GPU
/// initialisation exactly once at construction time.  Kernel dispatch is
/// then `O(1)` (a simple `match`).
pub struct GpuDispatcher {
    ctx: Option<GpuContext>,
}

impl GpuDispatcher {
    /// Create a new dispatcher.  Attempts to initialise a GPU context; stores
    /// `None` if no GPU is available or the `gpu` feature is disabled.
    pub fn new() -> Self {
        Self {
            ctx: GpuContext::try_init(),
        }
    }

    /// Returns `true` if a GPU context was successfully initialised.
    pub fn has_gpu(&self) -> bool {
        self.ctx.is_some()
    }

    /// Return a GPU kernel for the given tensor type, or `None` if:
    /// - No GPU is available (`has_gpu() == false`).
    /// - The tensor type has no GPU kernel implementation.
    pub fn get_kernel(&self, tensor_type: GgufTensorType) -> Option<Box<dyn GpuKernel>> {
        // No context → no kernel.
        self.ctx.as_ref()?;

        match tensor_type {
            GgufTensorType::Q2K => Some(Box::new(Q2_KGpuKernel)),
            GgufTensorType::Q3K => Some(Box::new(Q3_KGpuKernel)),
            GgufTensorType::Q4_0 => Some(Box::new(Q4_0GpuKernel)),
            GgufTensorType::Q4_1 => Some(Box::new(Q4_1GpuKernel)),
            GgufTensorType::Q4K => Some(Box::new(Q4_KGpuKernel)),
            GgufTensorType::Q5_0 => Some(Box::new(Q5_0GpuKernel)),
            GgufTensorType::Q5_1 => Some(Box::new(Q5_1GpuKernel)),
            GgufTensorType::Q5K => Some(Box::new(Q5_KGpuKernel)),
            GgufTensorType::Q6K => Some(Box::new(Q6_KGpuKernel)),
            GgufTensorType::Q8_0 => Some(Box::new(Q8_0GpuKernel)),
            GgufTensorType::Q8_1 => Some(Box::new(Q8_1GpuKernel)),
            GgufTensorType::Q8K => Some(Box::new(Q8_KGpuKernel)),
            GgufTensorType::Q1_0G128 => Some(Box::new(Q1_0_G128GpuKernel)),
            GgufTensorType::Iq4Xs => Some(Box::new(Iq4XsGpuKernel)),
            GgufTensorType::Iq2Xxs => Some(Box::new(Iq2XxsGpuKernel)),
            GgufTensorType::Iq2S => Some(Box::new(Iq2SGpuKernel)),
            GgufTensorType::Iq3Xxs => Some(Box::new(Iq3XxsGpuKernel)),
            GgufTensorType::Iq3S => Some(Box::new(Iq3SGpuKernel)),
            _ => None,
        }
    }

    /// Return a reference to the underlying [`GpuContext`], if one exists.
    pub fn context(&self) -> Option<&GpuContext> {
        self.ctx.as_ref()
    }

    /// Create a dispatcher selecting a GPU adapter by name substring
    /// (case-insensitive).  Falls back to no-GPU if no adapter matches.
    pub fn with_device_name(name: &str) -> Self {
        Self {
            ctx: GpuContext::try_init_with_name(name),
        }
    }

    /// Create a dispatcher selecting a GPU adapter by index.
    /// Falls back to no-GPU if the index is out of bounds.
    pub fn with_device_index(index: usize) -> Self {
        Self {
            ctx: GpuContext::try_init_with_index(index),
        }
    }

    /// Enumerate available GPU adapters.
    pub fn enumerate_devices() -> Vec<GpuDeviceInfo> {
        GpuContext::enumerate_devices()
    }
}

impl Default for GpuDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic smoke tests (always run, even without GPU) ─────────────────────

    #[test]
    fn test_gpu_context_try_init_no_crash() {
        // Must not panic regardless of whether a GPU is present.
        let _ctx = GpuContext::try_init();
    }

    #[test]
    fn test_gpu_dispatcher_new_no_crash() {
        let dispatcher = GpuDispatcher::new();
        // has_gpu() may be false in CI — that is fine.
        let _ = dispatcher.has_gpu();
    }

    #[test]
    fn test_gpu_dispatcher_default_no_crash() {
        let _dispatcher = GpuDispatcher::default();
    }

    #[test]
    fn test_gpu_dispatcher_no_kernel_for_f32() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::F32);
        assert!(kernel.is_none(), "F32 should not have a GPU kernel");
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q4k_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q4K);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q4K should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(kernel.is_none(), "Q4K should not have a kernel without GPU");
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q5k_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q5K);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q5K should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(kernel.is_none(), "Q5K should not have a kernel without GPU");
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q6k_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q6K);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q6K should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(kernel.is_none(), "Q6K should not have a kernel without GPU");
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q2k_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q2K);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q2K should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(kernel.is_none(), "Q2K should not have a kernel without GPU");
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q3k_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q3K);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q3K should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(kernel.is_none(), "Q3K should not have a kernel without GPU");
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q8k_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q8K);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q8K should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(kernel.is_none(), "Q8K should not have a kernel without GPU");
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_iq4xs_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Iq4Xs);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Iq4Xs should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Iq4Xs should not have a kernel without GPU"
            );
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_iq2xxs_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Iq2Xxs);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Iq2Xxs should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Iq2Xxs should not have a kernel without GPU"
            );
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_iq2s_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Iq2S);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Iq2S should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Iq2S should not have a kernel without GPU"
            );
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_iq3xxs_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Iq3Xxs);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Iq3Xxs should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Iq3Xxs should not have a kernel without GPU"
            );
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_iq3s_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Iq3S);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Iq3S should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Iq3S should not have a kernel without GPU"
            );
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q4_1_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q4_1);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q4_1 should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Q4_1 should not have a kernel without GPU"
            );
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q5_0_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q5_0);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q5_0 should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Q5_0 should not have a kernel without GPU"
            );
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q5_1_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q5_1);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q5_1 should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Q5_1 should not have a kernel without GPU"
            );
        }
    }

    #[test]
    fn test_gpu_dispatcher_kernel_for_q8_1_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q8_1);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q8_1 should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Q8_1 should not have a kernel without GPU"
            );
        }
    }

    #[test]
    fn test_gpu_error_display() {
        let e = GpuError::NoAdapter;
        assert!(!e.to_string().is_empty(), "error message must not be empty");
    }

    #[test]
    fn test_gpu_error_buffer_size() {
        let e = GpuError::BufferSize {
            expected: 32,
            got: 16,
        };
        let msg = e.to_string();
        assert!(msg.contains("32"), "message should mention expected=32");
        assert!(msg.contains("16"), "message should mention got=16");
    }

    #[test]
    fn test_gpu_error_device_request() {
        let e = GpuError::DeviceRequest("timeout".to_owned());
        assert!(e.to_string().contains("timeout"));
    }

    #[test]
    fn test_gpu_error_unsupported_type() {
        let e = GpuError::UnsupportedType {
            name: "Q6K".to_owned(),
        };
        assert!(e.to_string().contains("Q6K"));
    }

    #[test]
    fn test_gpu_error_shader_compilation() {
        let e = GpuError::ShaderCompilation {
            detail: "parse error".to_owned(),
        };
        assert!(e.to_string().contains("parse error"));
    }

    #[test]
    fn test_gpu_error_buffer_map() {
        let e = GpuError::BufferMap {
            detail: "lost".to_owned(),
        };
        assert!(e.to_string().contains("lost"));
    }

    // ── GPU-available tests (skip gracefully when no GPU) ────────────────────

    /// When a GPU is available, Q4_0 and Q8_0 kernels must be returned.
    #[test]
    fn test_gpu_dispatcher_kernels_when_gpu_present() {
        let dispatcher = GpuDispatcher::new();
        if !dispatcher.has_gpu() {
            return; // CI — no GPU
        }
        assert!(
            dispatcher.get_kernel(GgufTensorType::Q4_0).is_some(),
            "Q4_0 kernel must be available when GPU is present"
        );
        assert!(
            dispatcher.get_kernel(GgufTensorType::Q8_0).is_some(),
            "Q8_0 kernel must be available when GPU is present"
        );
    }

    /// Full end-to-end Q4_0 GEMV: GPU result must match CPU dequant+dot to
    /// within 1e-4 absolute tolerance.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q4_0_matches_cpu() {
        use crate::kernels::q4_0::Q4_0GpuKernel;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return, // skip if no GPU
        };

        // Two Q4_0 blocks (rows=2, cols=32).
        // Nibble layout: 0x88 → lo=0, hi=8 after -8 bias, so all values are 0.
        // But let's make the first nibble of each row non-zero.
        let make_block = |scale: f32, first_nibble: u8| -> Vec<u8> {
            let mut nibbles = [0x88u8; 16];
            nibbles[0] = first_nibble; // lo byte of pair 0
            let mut block = Vec::with_capacity(18);
            let d_bits = half::f16::from_f32(scale).to_bits();
            block.extend_from_slice(&d_bits.to_le_bytes());
            block.extend_from_slice(&nibbles);
            block
        };

        // Row 0: scale=1.0, first nibble lo=0xA (10-8=2), hi=0x8 (0)
        // Row 1: scale=0.5, first nibble lo=0x6 (6-8=-2), hi=0x8 (0)
        let mut weight_bytes = Vec::new();
        weight_bytes.extend_from_slice(&make_block(1.0, 0x8A)); // lo=A=10→+2, hi=8→0
        weight_bytes.extend_from_slice(&make_block(0.5, 0x86)); // lo=6→-2, hi=8→0

        // input: all 1.0 except index 0 = 3.0
        let mut input = vec![1.0f32; 32];
        input[0] = 3.0;

        // CPU reference: row0 = 2.0*3.0 + 0 = 6.0; row1 = -1.0*3.0 = -3.0
        // (scale*lo * input[0], rest are 0)
        let expected = [6.0f32, -3.0f32];

        let mut output = vec![0.0f32; 2];
        let kernel = Q4_0GpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, 2, 32)
            .expect("Q4_0 GPU GEMV");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}"
            );
        }
    }

    /// Full end-to-end Q8_0 GEMV.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q8_0_matches_cpu() {
        use crate::kernels::q8_0::Q8_0GpuKernel;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let make_block = |scale: f32, first_val: i8| -> Vec<u8> {
            let mut vals = [0i8; 32];
            vals[0] = first_val;
            let mut block = Vec::with_capacity(34);
            let d_bits = half::f16::from_f32(scale).to_bits();
            block.extend_from_slice(&d_bits.to_le_bytes());
            for &v in &vals {
                block.push(v as u8);
            }
            block
        };

        // Row 0: scale=2.0, q[0]=3  → weight[0][0] = 6.0
        // Row 1: scale=1.0, q[0]=-4 → weight[1][0] = -4.0
        let mut weight_bytes = Vec::new();
        weight_bytes.extend_from_slice(&make_block(2.0, 3));
        weight_bytes.extend_from_slice(&make_block(1.0, -4));

        let mut input = vec![0.0f32; 32];
        input[0] = 1.5;

        // row0 = 6.0*1.5 = 9.0; row1 = -4.0*1.5 = -6.0
        let expected = [9.0f32, -6.0f32];

        let mut output = vec![0.0f32; 2];
        let kernel = Q8_0GpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, 2, 32)
            .expect("Q8_0 GPU GEMV");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}"
            );
        }
    }

    // ── Q1_0_G128 GPU tests ─────────────────────────────────────────────────

    #[test]
    fn test_gpu_dispatcher_kernel_for_q1_0_g128_when_gpu() {
        let dispatcher = GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(GgufTensorType::Q1_0G128);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q1_0G128 should have a GPU kernel when GPU is present"
            );
        } else {
            assert!(
                kernel.is_none(),
                "Q1_0G128 should not have a kernel without GPU"
            );
        }
    }

    /// Full end-to-end Q1_0_G128 GEMV: GPU result must match CPU dequant+dot.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q1_0_g128_matches_cpu() {
        use crate::kernels::q1_0_g128::Q1_0_G128GpuKernel;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return, // skip if no GPU
        };

        let make_block = |scale: f32, sign_bits: &[u8; 16]| -> Vec<u8> {
            let mut block = Vec::with_capacity(18);
            let d_bits = half::f16::from_f32(scale).to_bits();
            block.extend_from_slice(&d_bits.to_le_bytes());
            block.extend_from_slice(sign_bits);
            block
        };

        // Row 0: scale=2.0, all bits=1 → all weights = +2.0
        // Row 1: scale=1.0, all bits=0 → all weights = -1.0
        let mut weight_bytes = Vec::new();
        weight_bytes.extend_from_slice(&make_block(2.0, &[0xFF; 16]));
        weight_bytes.extend_from_slice(&make_block(1.0, &[0x00; 16]));

        // input: all 1.0
        let input = vec![1.0f32; 128];

        // row0 = sum(2.0 * 1.0) for 128 weights = 256.0
        // row1 = sum(-1.0 * 1.0) for 128 weights = -128.0
        let expected = [256.0f32, -128.0f32];

        let mut output = vec![0.0f32; 2];
        let kernel = Q1_0_G128GpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, 2, 128)
            .expect("Q1_0_G128 GPU GEMV");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-1,
                "row {i}: got {got}, expected {want}"
            );
        }
    }

    // ── Device selection tests ───────────────────────────────────────────────

    #[test]
    fn test_enumerate_devices_no_panic() {
        let devices = GpuDispatcher::enumerate_devices();
        // May be empty in CI — just checking it doesn't panic.
        let _ = devices.len();
    }

    #[test]
    fn test_enumerate_devices_from_context_no_panic() {
        let devices = GpuContext::enumerate_devices();
        let _ = devices.len();
    }

    #[test]
    fn test_try_init_with_name_nonexistent_returns_none() {
        let ctx = GpuContext::try_init_with_name("__nonexistent_gpu_xyz_999__");
        assert!(ctx.is_none(), "Non-matching name pattern must return None");
    }

    #[test]
    fn test_try_init_with_index_out_of_bounds_returns_none() {
        let ctx = GpuContext::try_init_with_index(9999);
        assert!(ctx.is_none(), "Out-of-bounds index must return None");
    }

    #[test]
    fn test_dispatcher_with_device_name_nonexistent() {
        let dispatcher = GpuDispatcher::with_device_name("__nonexistent_gpu_xyz_999__");
        assert!(
            !dispatcher.has_gpu(),
            "Non-matching device name must yield no GPU"
        );
    }

    #[test]
    fn test_dispatcher_with_device_index_out_of_bounds() {
        let dispatcher = GpuDispatcher::with_device_index(9999);
        assert!(
            !dispatcher.has_gpu(),
            "Out-of-bounds index must yield no GPU"
        );
    }

    #[test]
    fn test_gpu_device_info_debug() {
        let info = GpuDeviceInfo {
            name: "Test GPU".to_owned(),
            backend: "Vulkan".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
        };
        let debug_str = format!("{info:?}");
        assert!(debug_str.contains("Test GPU"));
        assert!(debug_str.contains("Vulkan"));
    }

    #[test]
    fn test_gpu_device_info_clone() {
        let info = GpuDeviceInfo {
            name: "GPU".to_owned(),
            backend: "Metal".to_owned(),
            device_type: "IntegratedGpu".to_owned(),
        };
        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.backend, info.backend);
        assert_eq!(cloned.device_type, info.device_type);
    }
}

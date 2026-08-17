// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU offload policy and the device-resident Q4_0 kernel wrapper.
//!
//! # What is actually offloaded
//!
//! Exactly one thing: **Q4_0 weight matrices, through
//! `oxillama_gpu::gemv_q4_0_resident`**.  That kernel is the only GPU entry
//! point in `oxillama-gpu` that uploads its weights once (at model load) and
//! reuses a cached compute pipeline per call.  Every other GPU kernel in that
//! crate rebuilds its pipeline — and re-uploads dequantised f32 weights — on
//! every invocation, which measures *slower* than the CPU kernels this runtime
//! already dispatches, so none of them are wired here.
//!
//! # Where the offload applies
//!
//! Kernel substitution happens through
//! [`ForwardPass::remap_quant_kernels`], whose contract covers the stored
//! per-projection kernels read by **single-token decode**.  Explicitly out of
//! that contract, and therefore *not* GPU-accelerated:
//!
//! * tiled multi-token prefill (`llama`/`qwen3` re-dispatch from the model's
//!   own `KernelDispatcher` per call rather than reading the stored `Arc`),
//! * token-embedding row lookup (same reason),
//! * MoE expert and router weights (no architecture exposes them to the
//!   visitor).
//!
//! A caller that needs GPU residency to hold for prompt processing must drive
//! prefill one token at a time.

use crate::error::{RuntimeError, RuntimeResult};
use oxillama_arch::traits::ForwardPass;

#[cfg(feature = "gpu")]
use oxillama_gguf::GgufTensorType;
#[cfg(feature = "gpu")]
use oxillama_quant::{QuantKernel, QuantResult, QuantTensor};
#[cfg(feature = "gpu")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Minimum `rows * cols` a weight tensor must have before it is worth keeping
/// on the GPU, when [`GpuOptions::min_weight_elems`] is `None`.
///
/// A device-resident GEMV still pays a per-call input upload plus a blocking
/// readback, so small projections lose to the CPU kernel even with the weights
/// already on the device.  This threshold is a placeholder chosen to admit
/// roughly "4096 × 256 and larger"; it is expected to be re-tuned from
/// benchmark data rather than treated as a measured constant.
pub const DEFAULT_GPU_MIN_WEIGHT_ELEMS: usize = 1_048_576;

/// Bytes one Q4_0 block occupies **on the device** after
/// [`oxillama_gpu::Q4_0Resident::upload`]: a 4-byte f32 scale (widened from the
/// f16 scale in the GGUF block) plus the 16 nibble bytes copied verbatim.
///
/// This is deliberately not the 18-byte host-side block size — the scale is
/// widened during upload.
#[cfg(feature = "gpu")]
const DEVICE_BYTES_PER_Q4_0_BLOCK: u64 = 20;

/// Weights per Q4_0 block.
#[cfg(feature = "gpu")]
const Q4_0_BLOCK_SIZE: usize = 32;

/// Whether the engine should offload any weights to a GPU.
///
/// `Off` is the default and keeps the runtime byte-for-byte on the CPU path.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum GpuPolicy {
    /// No GPU is initialised and no kernel is remapped.
    #[default]
    Off,
    /// Offload eligible weights according to the given options.
    On(GpuOptions),
}

/// Knobs for [`GpuPolicy::On`].
///
/// The all-`None` default means "auto-select an adapter, offload every
/// eligible tensor, use the built-in size threshold".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuOptions {
    /// Which adapter to bind, or `None` to let wgpu pick the
    /// highest-performance one.
    pub device: Option<GpuDeviceSelector>,
    /// How many transformer layers to place on the GPU.
    ///
    /// * `None` — every transformer layer **plus the LM head**.
    /// * `Some(n)` — the **last** `n` layers, i.e. layer indices
    ///   `>= num_layers - n`, matching llama.cpp's `--n-gpu-layers`.  The LM
    ///   head counts as one extra layer beyond the transformer stack, so it is
    ///   included only when `n > num_layers` (`n == num_layers + 1` or more
    ///   means "everything").  `Some(0)` offloads nothing.
    pub n_gpu_layers: Option<usize>,
    /// Minimum `rows * cols` for a tensor to be routed to the GPU.
    ///
    /// `None` uses [`DEFAULT_GPU_MIN_WEIGHT_ELEMS`].
    pub min_weight_elems: Option<usize>,
}

/// How to pick the GPU adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuDeviceSelector {
    /// Position in `oxillama_gpu::GpuContext::enumerate_devices` order.
    Index(usize),
    /// Case-insensitive substring of the adapter name.
    Name(String),
}

/// What the GPU actually ended up holding after a model load.
///
/// `resident_tensors == 0` is a successful outcome, not an error: it means a
/// device was found but nothing matched the eligibility rules (wrong quant
/// type, below the size threshold, or excluded by `n_gpu_layers`).  Callers
/// surface this so a user can tell "no GPU" from "GPU, but idle".
#[derive(Debug, Clone, PartialEq)]
pub struct GpuStatus {
    /// Adapter name reported by wgpu.
    pub device_name: String,
    /// Adapter backend (`Metal`, `Vulkan`, `Dx12`, …).
    pub backend: String,
    /// Tensors whose weights now live on the device.
    pub resident_tensors: usize,
    /// Device bytes those tensors occupy (quantised, not dequantised).
    pub resident_bytes: u64,
    /// Tensors left on the CPU, for any reason including upload failure.
    pub cpu_tensors: usize,
    /// Subset of `cpu_tensors` that was eligible but failed to upload.
    pub upload_failures: usize,
}

/// Whether a kernel site is covered by `n_gpu_layers` for a model with
/// `num_layers` transformer blocks.
///
/// `layer` is `None` for the LM head.  See [`GpuOptions::n_gpu_layers`] for the
/// contract this implements; the addition form (`idx + n >= num_layers`) is
/// used instead of `idx >= num_layers - n` so that `n > num_layers` cannot
/// underflow.
#[cfg(any(feature = "gpu", test))]
pub(crate) fn site_eligible(
    layer: Option<usize>,
    n_gpu_layers: Option<usize>,
    num_layers: usize,
) -> bool {
    match n_gpu_layers {
        None => true,
        Some(n) => match layer {
            Some(idx) => idx.saturating_add(n) >= num_layers,
            None => n > num_layers,
        },
    }
}

/// Apply `policy` to a freshly built forward pass.
///
/// Returns `Ok(None)` for [`GpuPolicy::Off`].  For [`GpuPolicy::On`] this is
/// `activate` when the `gpu` feature is on, and a hard
/// [`RuntimeError::GpuUnavailable`] otherwise — asking for a GPU and silently
/// getting a CPU run is exactly the failure mode this refuses.
///
/// Only kernels reachable through [`ForwardPass::remap_quant_kernels`] are
/// affected; see this module's documentation for what that excludes.
pub(crate) fn apply_gpu_policy(
    forward_pass: &mut dyn ForwardPass,
    policy: &GpuPolicy,
    num_layers: usize,
) -> RuntimeResult<Option<GpuStatus>> {
    match policy {
        GpuPolicy::Off => Ok(None),
        GpuPolicy::On(opts) => {
            #[cfg(feature = "gpu")]
            {
                activate(forward_pass, opts, num_layers).map(Some)
            }
            #[cfg(not(feature = "gpu"))]
            {
                let _ = (forward_pass, opts, num_layers);
                Err(RuntimeError::GpuUnavailable {
                    reason: "OxiLLaMa was built without the `gpu` feature; rebuild with \
                             --features gpu"
                        .to_string(),
                })
            }
        }
    }
}

/// Human-readable list of the adapters wgpu can see, for error messages.
#[cfg(feature = "gpu")]
fn available_devices() -> String {
    let devices = oxillama_gpu::GpuContext::enumerate_devices();
    if devices.is_empty() {
        return "none".to_string();
    }
    devices
        .iter()
        .map(|d| format!("{} ({})", d.name, d.backend))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Bind the adapter `selector` names, or the highest-performance one when it is
/// `None`.
///
/// # Errors
///
/// [`RuntimeError::GpuUnavailable`]; for an explicit selector the reason lists
/// the adapters that *do* exist, so the user can correct the flag.
#[cfg(feature = "gpu")]
fn init_context(selector: Option<&GpuDeviceSelector>) -> RuntimeResult<oxillama_gpu::GpuContext> {
    let context = match selector {
        None => oxillama_gpu::GpuContext::try_init(),
        Some(GpuDeviceSelector::Index(index)) => {
            oxillama_gpu::GpuContext::try_init_with_index(*index)
        }
        Some(GpuDeviceSelector::Name(name)) => oxillama_gpu::GpuContext::try_init_with_name(name),
    };
    // The reason is built lazily: `available_devices` re-enumerates adapters,
    // which is far too expensive to pay on the success path.
    context.ok_or_else(|| RuntimeError::GpuUnavailable {
        reason: match selector {
            None => "no compatible GPU adapter is available on this host".to_string(),
            Some(GpuDeviceSelector::Index(index)) => format!(
                "no GPU device at index {index}; available devices: {}",
                available_devices()
            ),
            Some(GpuDeviceSelector::Name(name)) => format!(
                "no GPU device name contains {name:?}; available devices: {}",
                available_devices()
            ),
        },
    })
}

/// Bind a GPU device and move every eligible Q4_0 weight onto it.
///
/// A tensor is eligible when all of the following hold:
/// its site passes [`site_eligible`], its type is `Q4_0`, its shape is
/// two-dimensional, and `rows * cols` reaches
/// [`GpuOptions::min_weight_elems`].  Everything else keeps the CPU kernel it
/// already had, as does any tensor whose upload fails.
///
/// # Errors
///
/// [`RuntimeError::GpuUnavailable`] when no adapter matching
/// [`GpuOptions::device`] can be initialised.  Upload failures are *not*
/// errors: they are counted into [`GpuStatus::upload_failures`] and the model
/// still runs, on the CPU.
#[cfg(feature = "gpu")]
pub(crate) fn activate(
    forward_pass: &mut dyn ForwardPass,
    opts: &GpuOptions,
    num_layers: usize,
) -> RuntimeResult<GpuStatus> {
    let ctx = Arc::new(init_context(opts.device.as_ref())?);

    let min_elems = opts
        .min_weight_elems
        .unwrap_or(DEFAULT_GPU_MIN_WEIGHT_ELEMS);
    let n_gpu_layers = opts.n_gpu_layers;

    let mut resident_tensors = 0usize;
    let mut resident_bytes = 0u64;
    let mut cpu_tensors = 0usize;
    let mut upload_failures = 0usize;

    forward_pass.remap_quant_kernels(&mut |site, kernel| {
        let weight = site.weight;
        let eligible = site_eligible(site.layer, n_gpu_layers, num_layers)
            && weight.tensor_type == GgufTensorType::Q4_0
            && weight.shape.len() == 2;
        if !eligible {
            cpu_tensors += 1;
            return kernel;
        }
        let (rows, cols) = (weight.shape[0], weight.shape[1]);
        if rows.saturating_mul(cols) < min_elems {
            cpu_tensors += 1;
            return kernel;
        }

        let bytes = weight.data.as_slice();
        match oxillama_gpu::Q4_0Resident::upload(&ctx, bytes, rows, cols) {
            Ok(resident) => {
                resident_tensors += 1;
                resident_bytes += rows as u64
                    * cols.div_ceil(Q4_0_BLOCK_SIZE) as u64
                    * DEVICE_BYTES_PER_Q4_0_BLOCK;
                Arc::new(GpuResidentQ4_0Kernel {
                    ctx: Arc::clone(&ctx),
                    resident,
                    expected_ptr: bytes.as_ptr() as usize,
                    expected_len: bytes.len(),
                    cpu: kernel,
                    fell_back: AtomicBool::new(false),
                })
            }
            Err(err) => {
                upload_failures += 1;
                cpu_tensors += 1;
                tracing::warn!(
                    role = site.role,
                    layer = ?site.layer,
                    rows,
                    cols,
                    error = %err,
                    "Q4_0 weight upload failed; this tensor stays on the CPU"
                );
                kernel
            }
        }
    });

    let info = ctx.device_info();
    let status = GpuStatus {
        device_name: info.name.clone(),
        backend: info.backend.clone(),
        resident_tensors,
        resident_bytes,
        cpu_tensors,
        upload_failures,
    };
    tracing::info!(
        device = %status.device_name,
        backend = %status.backend,
        resident_tensors = status.resident_tensors,
        resident_mib = status.resident_bytes / (1024 * 1024),
        cpu_tensors = status.cpu_tensors,
        upload_failures = status.upload_failures,
        "GPU offload activated"
    );
    Ok(status)
}

/// A [`QuantKernel`] that answers `gemv` from device-resident Q4_0 weights and
/// delegates everything else to the CPU kernel it replaced.
///
/// One instance is bound to exactly one [`QuantTensor`]: `resident` holds that
/// tensor's bytes already on the device, so serving a *different* tensor
/// through this kernel would silently compute the wrong matrix.  `expected_ptr`
/// / `expected_len` pin the pairing — see [`Self::serves`].
#[cfg(feature = "gpu")]
#[allow(non_camel_case_types)]
struct GpuResidentQ4_0Kernel {
    /// The device this kernel dispatches to; shared with every other wrapper
    /// built by the same [`activate`] call.
    ctx: Arc<oxillama_gpu::GpuContext>,
    /// The uploaded weights.
    resident: oxillama_gpu::Q4_0Resident,
    /// Start address of the tensor payload this kernel was built for, stored as
    /// an integer so the struct stays automatically `Send`/`Sync`.
    expected_ptr: usize,
    /// Length of that payload.
    expected_len: usize,
    /// The kernel this one replaced; every non-`gemv` method is its.
    cpu: Arc<dyn QuantKernel>,
    /// Set once the device path has failed; keeps the failure from being
    /// retried (and re-logged) on every subsequent token.
    fell_back: AtomicBool,
}

#[cfg(feature = "gpu")]
impl GpuResidentQ4_0Kernel {
    /// Whether `tensor` is the tensor whose bytes are on the device.
    ///
    /// Pointer identity is a sound test here because [`QuantTensor`] clones
    /// share their payload — `SharedBytes` copies its cached start pointer
    /// verbatim into the clone — so a copy of the bound tensor still matches,
    /// while an unrelated tensor cannot.  Architecture code pairs kernels and
    /// tensors 1:1, so this should never fail; it exists so that a future
    /// violation degrades to a slower answer rather than a wrong one.
    fn serves(&self, tensor: &QuantTensor) -> bool {
        let bytes = tensor.data.as_slice();
        bytes.as_ptr() as usize == self.expected_ptr && bytes.len() == self.expected_len
    }
}

#[cfg(feature = "gpu")]
impl QuantKernel for GpuResidentQ4_0Kernel {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        self.cpu.dequant_block(block, output)
    }

    fn gemv(
        &self,
        quant_matrix: &QuantTensor,
        input: &[f32],
        output: &mut [f32],
    ) -> QuantResult<()> {
        if self.fell_back.load(Ordering::Relaxed) || !self.serves(quant_matrix) {
            return self.cpu.gemv(quant_matrix, input, output);
        }
        match oxillama_gpu::gemv_q4_0_resident(&self.ctx, &self.resident, input, output) {
            Ok(()) => Ok(()),
            Err(err) => {
                if !self.fell_back.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        rows = self.resident.rows(),
                        cols = self.resident.cols(),
                        error = %err,
                        "device-resident Q4_0 GEMV failed; this weight stays on the CPU \
                         for the rest of the process"
                    );
                }
                self.cpu.gemv(quant_matrix, input, output)
            }
        }
    }

    fn gemm(
        &self,
        quant_matrix: &QuantTensor,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> QuantResult<()> {
        self.cpu.gemm(quant_matrix, input, output, m, n, k)
    }

    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        self.cpu
            .matvec_q8_fused(weights, acts_q8, out, n_rows, n_cols)
    }

    fn matmul_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
        m: usize,
    ) -> QuantResult<()> {
        self.cpu
            .matmul_q8_fused(weights, acts_q8, out, n_rows, n_cols, m)
    }

    /// Delegated verbatim, which is only safe while no Q4_0 CPU kernel opts
    /// into the fused path.
    ///
    /// This method is the dispatch gate `QuantLinear::q8_fused_blocks` reads:
    /// a `Some` here routes decode through `matvec_q8_fused` and the device
    /// path is never reached.  Every Q4_0 kernel in `oxillama-quant`
    /// (reference, AVX2, AVX-512, NEON) deliberately leaves it at the trait
    /// default of `None`, so `gemv` is the live path.
    /// `gpu_wrapped_q4_0_kernels_stay_on_the_gemv_path` in this module's tests
    /// fails the moment that stops being true.
    fn q8_fused_acts_blocks(&self, n_cols: usize) -> Option<usize> {
        self.cpu.q8_fused_acts_blocks(n_cols)
    }

    fn block_size(&self) -> usize {
        self.cpu.block_size()
    }

    fn block_bytes(&self) -> usize {
        self.cpu.block_bytes()
    }

    fn name(&self) -> &'static str {
        "q4_0_gpu_resident"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_policy_defaults_to_off() {
        assert_eq!(GpuPolicy::default(), GpuPolicy::Off);
    }

    #[test]
    fn gpu_options_default_is_fully_automatic() {
        let opts = GpuOptions::default();
        assert_eq!(opts.device, None);
        assert_eq!(opts.n_gpu_layers, None);
        assert_eq!(opts.min_weight_elems, None);
    }

    /// `None` means "every layer plus the LM head".
    #[test]
    fn all_layers_eligible_when_n_gpu_layers_is_none() {
        for layer in 0..4 {
            assert!(site_eligible(Some(layer), None, 4));
        }
        assert!(site_eligible(None, None, 4), "LM head must be included");
    }

    /// `Some(n)` selects the LAST `n` transformer layers, llama.cpp-style.
    #[test]
    fn some_n_selects_the_last_n_layers() {
        assert!(!site_eligible(Some(0), Some(2), 4));
        assert!(!site_eligible(Some(1), Some(2), 4));
        assert!(site_eligible(Some(2), Some(2), 4));
        assert!(site_eligible(Some(3), Some(2), 4));
    }

    /// The LM head counts as one layer beyond the stack.
    #[test]
    fn lm_head_needs_more_layers_than_the_model_has() {
        assert!(!site_eligible(None, Some(4), 4), "n == num_layers excludes");
        assert!(site_eligible(None, Some(5), 4), "n > num_layers includes");
        assert!(site_eligible(None, Some(99), 4));
    }

    #[test]
    fn zero_layers_offloads_nothing() {
        assert!(!site_eligible(Some(0), Some(0), 4));
        assert!(!site_eligible(Some(3), Some(0), 4));
        assert!(!site_eligible(None, Some(0), 4));
    }

    /// The wrapper delegates `q8_fused_acts_blocks` verbatim, so if a Q4_0 CPU
    /// kernel ever starts returning `Some` the decode path silently stops
    /// calling `gemv` and the GPU goes dead while still reporting resident
    /// tensors.  Assert the invariant the delegation relies on.
    #[test]
    fn gpu_wrapped_q4_0_kernels_stay_on_the_gemv_path() {
        let kernel = oxillama_quant::global_dispatcher()
            .get_kernel(oxillama_gguf::GgufTensorType::Q4_0)
            .expect("Q4_0 kernel must be dispatchable");
        for n_cols in [32usize, 64, 4096] {
            assert_eq!(
                kernel.q8_fused_acts_blocks(n_cols),
                None,
                "the dispatched Q4_0 kernel opted into the fused path; \
                 GpuResidentQ4_0Kernel::q8_fused_acts_blocks must now return None \
                 or decode will never reach the device"
            );
        }
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::*;
        use oxillama_arch::error::ArchResult;
        use oxillama_arch::traits::KvCacheAccess;
        use oxillama_gguf::GgufTensorType;

        /// A model with no kernel bindings — enough to exercise `activate`'s
        /// device-selection path without any weights.
        struct NoKernelsModel;

        impl ForwardPass for NoKernelsModel {
            fn forward(
                &mut self,
                _tokens: &[u32],
                _kv_cache: &mut dyn KvCacheAccess,
            ) -> ArchResult<Vec<f32>> {
                Ok(Vec::new())
            }

            fn vocab_size(&self) -> usize {
                1
            }

            fn max_context_length(&self) -> usize {
                1
            }

            fn hidden_size(&self) -> usize {
                1
            }
        }

        /// Deterministic Q4_0 payload: `rows × (cols / 32)` blocks of an f16
        /// scale plus 16 nibble bytes.  `seed` shifts the nibble pattern so two
        /// tensors built here hold genuinely different weights.
        fn q4_0_bytes(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
            let blocks = rows * (cols / 32);
            let mut out = Vec::with_capacity(blocks * 18);
            for b in 0..blocks {
                let scale = half::f16::from_f32(0.05);
                out.extend_from_slice(&scale.to_bits().to_le_bytes());
                for i in 0..16u8 {
                    let lo = (i.wrapping_add(seed).wrapping_add(b as u8)) & 0x0F;
                    let hi = (i.wrapping_mul(3).wrapping_add(seed)) & 0x0F;
                    out.push(lo | (hi << 4));
                }
            }
            out
        }

        fn cpu_q4_0_kernel() -> Arc<dyn QuantKernel> {
            oxillama_quant::global_dispatcher()
                .get_kernel(GgufTensorType::Q4_0)
                .expect("Q4_0 kernel must be dispatchable")
        }

        /// Wrap `tensor` for `ctx`.  The upload is `expect`ed rather than
        /// skipped on failure: a caller only reaches here after `try_init`
        /// succeeded, and a silent skip would let the device assertions in
        /// these tests pass without ever touching a device.
        fn wrap(
            ctx: &Arc<oxillama_gpu::GpuContext>,
            tensor: &QuantTensor,
            cpu: Arc<dyn QuantKernel>,
        ) -> GpuResidentQ4_0Kernel {
            let bytes = tensor.data.as_slice();
            let resident =
                oxillama_gpu::Q4_0Resident::upload(ctx, bytes, tensor.shape[0], tensor.shape[1])
                    .expect("uploading a well-formed Q4_0 tensor must succeed");
            GpuResidentQ4_0Kernel {
                ctx: Arc::clone(ctx),
                resident,
                expected_ptr: bytes.as_ptr() as usize,
                expected_len: bytes.len(),
                cpu,
                fell_back: AtomicBool::new(false),
            }
        }

        fn input_vector(cols: usize) -> Vec<f32> {
            (0..cols)
                .map(|i| ((i % 7) as f32 - 3.0) / 8.0)
                .collect::<Vec<_>>()
        }

        /// The wrapper must be shareable across the decode threads that hold
        /// the model; a wgpu bump that made `Q4_0Resident` non-`Sync` would
        /// otherwise only show up at the `Arc<dyn QuantKernel>` coercion.
        #[test]
        fn wrapper_is_send_and_sync() {
            const fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<GpuResidentQ4_0Kernel>();
        }

        /// The device path must reproduce the CPU kernel's answer.
        #[test]
        fn resident_gemv_matches_the_cpu_kernel() {
            let Some(ctx) = oxillama_gpu::GpuContext::try_init() else {
                return;
            };
            let ctx = Arc::new(ctx);
            let (rows, cols) = (4usize, 64usize);
            let tensor = QuantTensor::new(
                q4_0_bytes(rows, cols, 1),
                vec![rows, cols],
                GgufTensorType::Q4_0,
            );
            let cpu = cpu_q4_0_kernel();
            let wrapper = wrap(&ctx, &tensor, Arc::clone(&cpu));

            let input = input_vector(cols);
            let mut cpu_out = vec![0.0f32; rows];
            cpu.gemv(&tensor, &input, &mut cpu_out)
                .expect("CPU gemv must succeed");
            let mut gpu_out = vec![0.0f32; rows];
            wrapper
                .gemv(&tensor, &input, &mut gpu_out)
                .expect("wrapped gemv must succeed");

            // Matching numbers alone would also be produced by a silent CPU
            // fallback, so assert the device path was the one that answered.
            assert!(
                !wrapper.fell_back.load(Ordering::Relaxed),
                "the device path errored and fell back; the comparison below \
                 would then be CPU-vs-CPU and prove nothing"
            );
            for (row, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
                assert!(
                    (g - c).abs() < 1e-3,
                    "row {row}: device {g} vs CPU {c} (tolerance 1e-3)"
                );
            }
            assert_eq!(wrapper.name(), "q4_0_gpu_resident");
        }

        /// Handed a tensor it was not built for, the wrapper must answer for
        /// *that* tensor via the CPU kernel rather than returning the resident
        /// matrix's product.
        #[test]
        fn foreign_tensor_falls_back_to_the_cpu_kernel() {
            let Some(ctx) = oxillama_gpu::GpuContext::try_init() else {
                return;
            };
            let ctx = Arc::new(ctx);
            let (rows, cols) = (4usize, 64usize);
            let bound = QuantTensor::new(
                q4_0_bytes(rows, cols, 1),
                vec![rows, cols],
                GgufTensorType::Q4_0,
            );
            let foreign = QuantTensor::new(
                q4_0_bytes(rows, cols, 9),
                vec![rows, cols],
                GgufTensorType::Q4_0,
            );
            let cpu = cpu_q4_0_kernel();
            let wrapper = wrap(&ctx, &bound, Arc::clone(&cpu));

            assert!(!wrapper.serves(&foreign), "test setup: tensors must differ");

            let input = input_vector(cols);
            let mut expected = vec![0.0f32; rows];
            cpu.gemv(&foreign, &input, &mut expected)
                .expect("CPU gemv must succeed");
            let mut actual = vec![0.0f32; rows];
            wrapper
                .gemv(&foreign, &input, &mut actual)
                .expect("wrapped gemv must succeed");
            assert_eq!(
                actual, expected,
                "a foreign tensor must be served bit-identically by the CPU kernel"
            );

            // A clone shares the payload pointer, so it is still the bound tensor.
            assert!(wrapper.serves(&bound.clone()));
        }

        /// Everything except `gemv` and `name` is the CPU kernel's answer.
        #[test]
        fn non_gemv_methods_delegate_to_the_cpu_kernel() {
            let Some(ctx) = oxillama_gpu::GpuContext::try_init() else {
                return;
            };
            let ctx = Arc::new(ctx);
            let (rows, cols) = (4usize, 64usize);
            let tensor = QuantTensor::new(
                q4_0_bytes(rows, cols, 3),
                vec![rows, cols],
                GgufTensorType::Q4_0,
            );
            let cpu = cpu_q4_0_kernel();
            let wrapper = wrap(&ctx, &tensor, Arc::clone(&cpu));

            assert_eq!(wrapper.block_size(), cpu.block_size());
            assert_eq!(wrapper.block_bytes(), cpu.block_bytes());
            for n_cols in [32usize, 64, 4096] {
                assert_eq!(
                    wrapper.q8_fused_acts_blocks(n_cols),
                    cpu.q8_fused_acts_blocks(n_cols)
                );
            }

            let block = &tensor.data.as_slice()[..cpu.block_bytes()];
            let mut from_wrapper = vec![0.0f32; cpu.block_size()];
            let mut from_cpu = vec![0.0f32; cpu.block_size()];
            wrapper
                .dequant_block(block, &mut from_wrapper)
                .expect("dequant must succeed");
            cpu.dequant_block(block, &mut from_cpu)
                .expect("dequant must succeed");
            assert_eq!(from_wrapper, from_cpu);
        }

        /// Naming a device that does not exist must fail loudly and tell the
        /// user what they could have named instead.
        #[test]
        fn unknown_device_name_reports_the_available_devices() {
            let mut model = NoKernelsModel;
            let opts = GpuOptions {
                device: Some(GpuDeviceSelector::Name("__nonexistent_xyz__".to_string())),
                ..GpuOptions::default()
            };
            let result = activate(&mut model, &opts, 4);
            match result {
                Err(RuntimeError::GpuUnavailable { reason }) => {
                    assert!(
                        reason.contains("__nonexistent_xyz__"),
                        "reason must quote the requested name, got: {reason}"
                    );
                    assert!(
                        reason.contains("available devices:"),
                        "reason must list the alternatives, got: {reason}"
                    );
                }
                other => panic!("expected GpuUnavailable, got {other:?}"),
            }
        }

        /// An out-of-range index fails the same way.
        #[test]
        fn out_of_range_device_index_is_rejected() {
            let mut model = NoKernelsModel;
            let opts = GpuOptions {
                device: Some(GpuDeviceSelector::Index(9999)),
                ..GpuOptions::default()
            };
            assert!(matches!(
                activate(&mut model, &opts, 4),
                Err(RuntimeError::GpuUnavailable { .. })
            ));
        }

        /// A model that exposes no kernel sites still yields a status, so a
        /// caller can distinguish "no GPU" from "GPU holding nothing".
        #[test]
        fn a_model_without_kernel_sites_still_reports_status() {
            if oxillama_gpu::GpuContext::try_init().is_none() {
                return;
            }
            let mut model = NoKernelsModel;
            let status = activate(&mut model, &GpuOptions::default(), 4)
                .expect("activation on the default adapter must succeed");
            assert_eq!(status.resident_tensors, 0);
            assert_eq!(status.resident_bytes, 0);
            assert_eq!(status.cpu_tensors, 0);
            assert_eq!(status.upload_failures, 0);
            assert!(!status.device_name.is_empty());
            assert!(!status.backend.is_empty());
        }
    }
}

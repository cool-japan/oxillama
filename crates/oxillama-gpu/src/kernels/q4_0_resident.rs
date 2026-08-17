//! Q4_0 device-resident GEMV — the fix for T3 of the numerical-correctness
//! pass: cached compute pipelines, quantised-byte upload, and in-shader
//! dequantisation.
//!
//! # What this replaces
//!
//! [`crate::kernels::GpuKernel::gemv`] (`kernels/q4_0.rs`) does, on
//! **every** call:
//! 1. `dequant_q4_0_to_f32` — dequantise every weight on the CPU.
//! 2. `upload_f32` — upload the *dequantised* f32 matrix (4 bytes/weight —
//!    4× the quantised byte count) to the GPU.
//! 3. `create_shader_module` → `create_bind_group_layout` →
//!    `create_pipeline_layout` → `create_compute_pipeline` — rebuild the
//!    entire compute pipeline from WGSL source text.
//!
//! For a decode loop calling GEMV once per token, that is full CPU
//! dequantisation plus full shader recompilation on every single token —
//! slower than just running the matching CPU kernel. This module fixes all
//! three:
//!
//! 1. [`Q4_0Resident::upload`] uploads the **quantised** bytes once: an f32
//!    scale per block (converted from f16 on the CPU — negligible, one
//!    value per 32 weights) plus the 16-byte nibble section copied verbatim
//!    (no per-weight unpacking; it is already 4-`u32`-word aligned).
//! 2. [`gemv_q4_0_resident`] dequantises **in the WGSL shader**
//!    (`shaders/gemv_q4_0_resident.wgsl`) using the same split-half nibble
//!    convention the rest of this pass fixed (see `kernels::q4_0`'s module
//!    doc) — weight `i` is byte `i`'s low nibble, weight `i + 16` is its
//!    high nibble.
//! 3. The compute pipeline is built once per [`crate::GpuContext`] via
//!    [`crate::GpuContext::get_or_create_pipeline`] and reused on every
//!    subsequent call.
//!
//! Per-call cost is now: upload `input` (the actual token-varying data),
//! dispatch, and read back `output` — no shader compilation, no CPU-side
//! weight dequantisation, no re-upload of the weight matrix.
//!
//! # Integration (for the `oxillama-runtime` / `oxillama-cli` owners)
//!
//! See the crate-level `TODO.md` §4 / the T2 section of this pass's report
//! for the full wiring instructions. In short: call
//! `Q4_0Resident::upload` once when a Q4_0 tensor is loaded (model-load
//! time, not per-token), keep the returned handle alongside the tensor for
//! the lifetime of the model, and call `gemv_q4_0_resident` per token
//! instead of `Q4_0GpuKernel::gemv`.
//!
//! The shape of that lifecycle — upload once, call many times, fall back to
//! CPU on any `Err` — is compile-checked here so it cannot silently drift
//! from this doc comment:
//!
//! ```
//! use oxillama_gpu::{gemv_q4_0_resident, GpuContext, Q4_0Resident};
//!
//! fn load_and_run(weight_bytes: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
//!     let mut output = vec![0.0f32; rows];
//!
//!     // Model-load time, once per Q4_0 tensor: acquire a context and
//!     // upload the *quantised* bytes. Either step can fail (no adapter, no
//!     // `gpu` feature, malformed input) — the caller always has a CPU
//!     // fallback (`oxillama_quant::reference::Q4_0Ref` or equivalent).
//!     let Some(ctx) = GpuContext::try_init() else {
//!         return cpu_fallback_gemv(weight_bytes, rows, cols, input, &mut output);
//!     };
//!     let resident = match Q4_0Resident::upload(&ctx, weight_bytes, rows, cols) {
//!         Ok(r) => r,
//!         Err(_) => return cpu_fallback_gemv(weight_bytes, rows, cols, input, &mut output),
//!     };
//!
//!     // Per token thereafter: reuse `ctx` and `resident` — no re-upload,
//!     // no pipeline rebuild.
//!     if gemv_q4_0_resident(&ctx, &resident, input, &mut output).is_err() {
//!         return cpu_fallback_gemv(weight_bytes, rows, cols, input, &mut output);
//!     }
//!     output
//! }
//! # fn cpu_fallback_gemv(_: &[u8], _: usize, _: usize, _: &[f32], out: &mut [f32]) -> Vec<f32> {
//! #     out.to_vec()
//! # }
//! # fn main() {}
//! ```

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};

/// Weights per Q4_0 block.
#[cfg(feature = "gpu")]
const Q4_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q4_0 block: 2 (f16 scale) + 16 (nibbles).
#[cfg(any(feature = "gpu", test))]
const Q4_0_BLOCK_BYTES: usize = 18;
/// Bytes of nibble data per block (always 4-`u32`-word aligned).
#[cfg(feature = "gpu")]
const Q4_0_QS_BYTES: usize = 16;

/// Device-resident Q4_0 weights.
///
/// Uploaded once via [`Q4_0Resident::upload`]; every [`gemv_q4_0_resident`]
/// call reuses the same device buffers.  Rows are still each split into
/// `blocks_per_row` Q4_0 blocks internally, but the caller only ever sees
/// `rows` / `cols` (weight-space) dimensions.
pub struct Q4_0Resident {
    #[cfg(feature = "gpu")]
    qs_buf: wgpu::Buffer,
    #[cfg(feature = "gpu")]
    scales_buf: wgpu::Buffer,
    rows: usize,
    cols: usize,
    #[cfg(feature = "gpu")]
    blocks_per_row: usize,
}

impl Q4_0Resident {
    /// Upload `weight_bytes` (raw Q4_0 block bytes, row-major: `rows` rows
    /// of `cols.div_ceil(32)` blocks each) to the device once.
    ///
    /// Returns `Err(GpuError::NoAdapter)` when the `gpu` feature is absent.
    pub fn upload(
        ctx: &GpuContext,
        weight_bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> GpuResult<Self> {
        #[cfg(feature = "gpu")]
        {
            upload_impl(ctx, weight_bytes, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }

    /// Number of weight rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of weight columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

#[cfg(feature = "gpu")]
fn upload_impl(
    ctx: &GpuContext,
    weight_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> GpuResult<Q4_0Resident> {
    let blocks_per_row = cols.div_ceil(Q4_0_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q4_0_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let n_blocks = rows * blocks_per_row;
    let mut scales = vec![0.0f32; n_blocks];
    let mut qs_bytes = vec![0u8; n_blocks * Q4_0_QS_BYTES];

    for b in 0..n_blocks {
        let off = b * Q4_0_BLOCK_BYTES;
        let block = &weight_bytes[off..off + Q4_0_BLOCK_BYTES];
        scales[b] = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        qs_bytes[b * Q4_0_QS_BYTES..(b + 1) * Q4_0_QS_BYTES]
            .copy_from_slice(&block[2..2 + Q4_0_QS_BYTES]);
    }

    let qs_words: &[u32] = bytemuck::cast_slice(&qs_bytes);
    let qs_buf = crate::buffer::upload_u32(&ctx.device, "q4_0-resident-qs", qs_words);
    let scales_buf = crate::buffer::upload_f32(&ctx.device, "q4_0-resident-scales", &scales);

    Ok(Q4_0Resident {
        qs_buf,
        scales_buf,
        rows,
        cols,
        blocks_per_row,
    })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg(feature = "gpu")]
struct Params {
    rows: u32,
    cols: u32,
    blocks_per_row: u32,
    _pad: u32,
}

/// GEMV against device-resident Q4_0 weights (see the module doc).
///
/// Returns `Err(GpuError::NoAdapter)` when the `gpu` feature is absent.
pub fn gemv_q4_0_resident(
    ctx: &GpuContext,
    weights: &Q4_0Resident,
    input: &[f32],
    output: &mut [f32],
) -> GpuResult<()> {
    #[cfg(feature = "gpu")]
    {
        gemv_impl(ctx, weights, input, output)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (ctx, weights, input, output);
        Err(GpuError::NoAdapter)
    }
}

#[cfg(feature = "gpu")]
fn gemv_impl(
    ctx: &GpuContext,
    weights: &Q4_0Resident,
    input: &[f32],
    output: &mut [f32],
) -> GpuResult<()> {
    use crate::buffer::{create_output_f32, download_f32, upload_f32, upload_uniform};
    use crate::context::CachedPipeline;
    use wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    };

    if output.len() < weights.rows {
        return Err(GpuError::BufferSize {
            expected: weights.rows,
            got: output.len(),
        });
    }
    if input.len() < weights.cols {
        return Err(GpuError::BufferSize {
            expected: weights.cols,
            got: input.len(),
        });
    }

    // Pipeline built once per context, reused on every call thereafter.
    let cached = ctx.get_or_create_pipeline("q4_0-resident-gemv", |device| {
        const WGSL: &str = include_str!("../shaders/gemv_q4_0_resident.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("gemv_q4_0_resident"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q4_0-resident-bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_rw(3),
                bgl_uniform(4),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("q4_0-resident-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q4_0-resident-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        CachedPipeline {
            bind_group_layout,
            pipeline,
        }
    });

    // Per-call buffers: only the token-varying input and the output.
    let input_buf = upload_f32(&ctx.device, "q4_0-resident-input", input);
    let output_buf = create_output_f32(&ctx.device, "q4_0-resident-output", weights.rows);

    let params = Params {
        rows: weights.rows as u32,
        cols: weights.cols as u32,
        blocks_per_row: weights.blocks_per_row as u32,
        _pad: 0,
    };
    let params_buf = upload_uniform(&ctx.device, "q4_0-resident-params", &params);

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q4_0-resident-bg"),
        layout: &cached.bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: weights.qs_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: weights.scales_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: input_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: output_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let dispatch_x = (weights.rows as u32).div_ceil(64);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q4_0-resident-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q4_0-resident-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }
    ctx.queue.submit([encoder.finish()]);

    let result = download_f32(&ctx.device, &ctx.queue, &output_buf, weights.rows)?;
    output[..weights.rows].copy_from_slice(&result[..weights.rows]);

    Ok(())
}

// ─── Bind-group layout entry helpers ─────────────────────────────────────────

#[cfg(feature = "gpu")]
fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(feature = "gpu")]
fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(feature = "gpu")]
fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Q4_0 block: 2-byte f16 scale + 16 nibble bytes.
    fn make_q4_0_block(scale: f32, nibbles: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q4_0_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        block.extend_from_slice(nibbles);
        block
    }

    /// `Q4_0Resident::upload` must never panic regardless of GPU
    /// availability: it succeeds when a real adapter is present (`gpu`
    /// feature + hardware) and is simply unreachable otherwise, since
    /// `GpuContext::try_init` itself returns `None` without a real adapter.
    #[test]
    fn test_resident_upload_no_panic_without_gpu() {
        let block = make_q4_0_block(1.0, &[0x88u8; 16]);
        if let Some(ctx) = GpuContext::try_init() {
            let resident = Q4_0Resident::upload(&ctx, &block, 1, 32);
            assert!(resident.is_ok(), "upload should succeed with real GPU");
        }
    }

    /// End-to-end: upload + resident GEMV must match the CPU dequant+dot
    /// reference, [`crate::kernels::Q4_0GpuKernel`]'s existing (non-resident)
    /// GPU path, AND `oxillama_quant::reference::Q4_0Ref` — the independent,
    /// upstream-pinned oracle used by `tests/cpu_gpu_cross_check.rs`.  The
    /// first two comparisons alone would not catch a layout bug shared by
    /// this crate's own `dequant_q4_0_to_f32` and the legacy path (that is
    /// exactly the class of bug this whole pass fixed); the third comparison
    /// closes that gap for the resident path specifically, since the
    /// dispatcher-driven cross-check test does not reach `gemv_q4_0_resident`
    /// (it only exercises `GpuDispatcher::get_kernel`, which never returns a
    /// `Q4_0Resident`).
    #[cfg(feature = "gpu")]
    #[test]
    fn test_resident_gemv_matches_cpu_and_legacy_gpu_path() {
        use crate::kernels::q4_0::dequant_q4_0_to_f32;
        use crate::kernels::{GpuKernel, Q4_0GpuKernel};
        use oxillama_quant::reference::Q4_0Ref;
        use oxillama_quant::{QuantKernel, QuantTensor};

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return, // no adapter in this environment
        };

        let rows = 6;
        let cols = 64; // 2 blocks per row
        let mut weight_bytes = Vec::with_capacity(rows * 2 * Q4_0_BLOCK_BYTES);
        for r in 0..rows {
            for blk in 0..2 {
                let scale = 0.05 + (r * 2 + blk) as f32 * 0.01;
                let nibbles: [u8; 16] =
                    std::array::from_fn(|i| ((r * 7 + blk * 3 + i * 5 + 11) & 0xFF) as u8);
                weight_bytes.extend_from_slice(&make_q4_0_block(scale, &nibbles));
            }
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1) - 3.2).collect();

        // CPU reference.
        let f32_weights = dequant_q4_0_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
        let cpu_expected: Vec<f32> = (0..rows)
            .map(|r| {
                f32_weights[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum()
            })
            .collect();

        // Resident path.
        let resident = Q4_0Resident::upload(&ctx, &weight_bytes, rows, cols).expect("upload");
        assert_eq!(resident.rows(), rows);
        assert_eq!(resident.cols(), cols);
        let mut resident_out = vec![0.0f32; rows];
        gemv_q4_0_resident(&ctx, &resident, &input, &mut resident_out).expect("resident gemv");

        for (i, (&got, &want)) in resident_out.iter().zip(cpu_expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "resident row {i}: got {got}, cpu expected {want}"
            );
        }

        // Legacy (non-resident, per-call pipeline rebuild) path — must agree.
        let legacy_kernel = Q4_0GpuKernel;
        let mut legacy_out = vec![0.0f32; rows];
        legacy_kernel
            .gemv(&ctx, &weight_bytes, &input, &mut legacy_out, rows, cols)
            .expect("legacy gpu gemv");

        for (i, (&resident_v, &legacy_v)) in resident_out.iter().zip(legacy_out.iter()).enumerate()
        {
            assert!(
                (resident_v - legacy_v).abs() < 1e-2,
                "row {i}: resident={resident_v}, legacy={legacy_v}"
            );
        }

        // Independent oracle: `oxillama_quant::reference::Q4_0Ref` lives in a
        // different crate and was not derived from this crate's own
        // `dequant_q4_0_to_f32`, so this is a genuine cross-check rather than
        // a self-comparison.
        let tensor = QuantTensor::new(
            weight_bytes.clone(),
            vec![rows, cols],
            oxillama_gguf::GgufTensorType::Q4_0,
        );
        let mut oracle_out = vec![0.0f32; rows];
        Q4_0Ref
            .gemv(&tensor, &input, &mut oracle_out)
            .expect("independent oracle gemv");

        for (i, (&resident_v, &oracle_v)) in resident_out.iter().zip(oracle_out.iter()).enumerate()
        {
            assert!(
                (resident_v - oracle_v).abs() < 1e-2,
                "row {i}: resident={resident_v}, independent oracle={oracle_v}"
            );
        }
    }

    /// The pipeline is built once and reused: two GEMV calls on the same
    /// context must not increase the cached pipeline count.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_resident_pipeline_is_cached_across_calls() {
        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let block = make_q4_0_block(1.0, &[0x9Au8; 16]);
        let resident = Q4_0Resident::upload(&ctx, &block, 1, 32).expect("upload");
        let input = vec![1.0f32; 32];
        let mut out = vec![0.0f32; 1];

        gemv_q4_0_resident(&ctx, &resident, &input, &mut out).expect("first gemv");
        let count_after_first = ctx.cached_pipeline_count();
        gemv_q4_0_resident(&ctx, &resident, &input, &mut out).expect("second gemv");
        let count_after_second = ctx.cached_pipeline_count();

        assert_eq!(
            count_after_first, count_after_second,
            "pipeline count must not grow across repeated calls"
        );
        assert!(
            count_after_first >= 1,
            "at least the q4_0-resident pipeline must be cached"
        );
    }

    #[test]
    fn test_resident_upload_buffer_too_small_errors() {
        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };
        let result = Q4_0Resident::upload(&ctx, &[0u8; 4], 1, 32);
        assert!(result.is_err(), "should fail on too-small input");
    }
}

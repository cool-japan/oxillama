//! IQ4_NL GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the IQ4_NL block
//!      format: 32 weights per 18-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! IQ4_NL block layout (18 bytes per 32 weights):
//! - bytes  0-1: d (f16 little-endian scale)
//! - bytes  2-17: 16 nibble-bytes encoding 32 four-bit weight indices
//!   Low nibble  = weight[2i], High nibble = weight[2i+1]
//!
//! Dequantisation: `w = d * KVALUES_IQ4NL[nibble]`
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// IQ4_NL GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Iq4NlGpuKernel;

impl GpuKernel for Iq4NlGpuKernel {
    fn gemv(
        &self,
        ctx: &GpuContext,
        weight_bytes: &[u8],
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> GpuResult<()> {
        #[cfg(feature = "gpu")]
        {
            gpu_gemv_iq4_nl(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── IQ4_NL block constants ───────────────────────────────────────────────────

/// Weights per IQ4_NL block.
#[cfg(any(feature = "gpu", test))]
const IQ4_NL_BLOCK_SIZE: usize = 32;
/// Bytes per IQ4_NL block: 2 (FP16 scale) + 16 (nibble data).
#[cfg(any(feature = "gpu", test))]
const IQ4_NL_BLOCK_BYTES: usize = 18;

/// Non-linear 4-bit quantization lookup table (KVALUES_IQ4NL).
///
/// Matches the reference implementation in `oxillama-quant/src/reference/iq_shared.rs`.
#[cfg(any(feature = "gpu", test))]
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Dequantise all IQ4_NL blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq4_nl_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(IQ4_NL_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * IQ4_NL_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * IQ4_NL_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + IQ4_NL_BLOCK_BYTES];

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let nibbles = &block[2..IQ4_NL_BLOCK_BYTES];
            let weight_base = blk * IQ4_NL_BLOCK_SIZE;

            for (i, &byte) in nibbles.iter().enumerate().take(IQ4_NL_BLOCK_SIZE / 2) {
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;

                let col0 = weight_base + i * 2;
                let col1 = col0 + 1;

                if col0 < cols {
                    f32_weights[row * cols + col0] = d * KVALUES_IQ4NL[lo] as f32;
                }
                if col1 < cols {
                    f32_weights[row * cols + col1] = d * KVALUES_IQ4NL[hi] as f32;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_iq4_nl(
    ctx: &GpuContext,
    weight_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) -> GpuResult<()> {
    use crate::buffer::{create_output_f32, download_f32, upload_f32, upload_uniform};
    use bytemuck::{Pod, Zeroable};
    use wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    };

    if output.len() < rows {
        return Err(GpuError::BufferSize {
            expected: rows,
            got: output.len(),
        });
    }
    if input.len() < cols {
        return Err(GpuError::BufferSize {
            expected: cols,
            got: input.len(),
        });
    }

    let f32_weights = dequant_iq4_nl_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "iq4_nl-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "iq4_nl-input", input);
    let output_buf = create_output_f32(&ctx.device, "iq4_nl-output", rows);

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Params {
        rows: u32,
        cols: u32,
    }
    let params = Params {
        rows: rows as u32,
        cols: cols as u32,
    };
    let params_buf = upload_uniform(&ctx.device, "iq4_nl-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_iq4_nl"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("iq4_nl-bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("iq4_nl-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("iq4_nl-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("iq4_nl-bg"),
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: weight_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: input_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let dispatch_x = rows.div_ceil(64) as u32;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("iq4_nl-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("iq4_nl-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }
    ctx.queue.submit([encoder.finish()]);

    let result = download_f32(&ctx.device, &ctx.queue, &output_buf, rows)?;
    output[..rows].copy_from_slice(&result[..rows]);

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

    /// Build an IQ4_NL block from a scale and 16 nibble bytes.
    fn make_iq4_nl_block(scale: f32, nibble_bytes: [u8; 16]) -> Vec<u8> {
        let mut block = vec![0u8; IQ4_NL_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block[2..IQ4_NL_BLOCK_BYTES].copy_from_slice(&nibble_bytes);
        block
    }

    #[test]
    fn test_dequant_iq4_nl_zero_scale() {
        // d = 0 → all weights must be 0 regardless of nibbles.
        let block = make_iq4_nl_block(0.0, [0xAAu8; 16]);
        let result = dequant_iq4_nl_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "weight[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn test_dequant_iq4_nl_index8_value1() {
        // KVALUES_IQ4NL[8] = 1, nibble 0x88 → lo=8, hi=8
        // d=1.0 → all weights = 1.0
        let block = make_iq4_nl_block(1.0, [0x88u8; 16]);
        let result = dequant_iq4_nl_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 1.0).abs() < 0.01, "weight[{i}] = {v}, expected 1.0");
        }
    }

    #[test]
    fn test_dequant_iq4_nl_all_max_index() {
        // nibble 0xFF → lo=hi=15, KVALUES[15] = 113. d=1.0 → all = 113.0
        let block = make_iq4_nl_block(1.0, [0xFFu8; 16]);
        let result = dequant_iq4_nl_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 113.0).abs() < 0.1, "weight[{i}] = {v}, expected 113.0");
        }
    }

    #[test]
    fn test_dequant_iq4_nl_two_rows() {
        // Row 0: d=0, all zero. Row 1: d=1, nibble 0x88 → all 1.0
        let block0 = make_iq4_nl_block(0.0, [0x88u8; 16]);
        let block1 = make_iq4_nl_block(1.0, [0x88u8; 16]);
        let mut data = Vec::new();
        data.extend_from_slice(&block0);
        data.extend_from_slice(&block1);
        let result = dequant_iq4_nl_to_f32(&data, 2, 32).expect("dequant");
        for &v in &result[..32] {
            assert!(v.abs() < 1e-6, "row0 weight expected 0, got {v}");
        }
        for (i, &v) in result[32..].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 0.01,
                "row1 weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4_nl_too_small() {
        assert!(
            dequant_iq4_nl_to_f32(&[0u8; 4], 1, 32).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_iq4_nl_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Iq4NlGpuKernel;
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_iq4_nl_matches_cpu_reference() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return, // skip if no GPU
        };

        let rows = 64;
        let cols = 32;

        let mut weight_bytes = Vec::with_capacity(rows * IQ4_NL_BLOCK_BYTES);
        for r in 0..rows {
            let d_val = 0.01 + r as f32 * 0.001;
            let nibbles: [u8; 16] = std::array::from_fn(|i| ((r * 5 + i * 3 + 7) & 0xFF) as u8);
            let block = make_iq4_nl_block(d_val, nibbles);
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1) - 1.6).collect();

        let f32_weights = dequant_iq4_nl_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                f32_weights[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum()
            })
            .collect();

        let mut result = vec![0.0f32; rows];
        let kernel = Iq4NlGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut result, rows, cols)
            .expect("GPU GEMV IQ4_NL");

        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

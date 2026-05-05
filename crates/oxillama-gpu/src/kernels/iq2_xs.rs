//! IQ2_XS GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the IQ2_XS block
//!      format: 256 weights per 74-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! IQ2_XS block layout (74 bytes per 256 weights):
//! - bytes  0-1:   d (f16 little-endian scale)
//! - bytes  2-65:  `qs[32]` — 32 × u16 little-endian.
//!   Each u16: lower 9 bits = grid index into `IQ2XS_GRID[512]`,
//!   upper 7 bits = sign selector index into `KSIGNS_IQ2XS[128]`.
//! - bytes 66-73:  `scales[8]` — 8 bytes, low nibble → `db[0]`, high nibble → `db[1]`.
//!
//! Scale formula:
//!   `db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25`  (groups l=0,1)
//!   `db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25`   (groups l=2,3)
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// IQ2_XS GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Iq2XsGpuKernel;

impl GpuKernel for Iq2XsGpuKernel {
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
            gpu_gemv_iq2_xs(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── IQ2_XS block constants ───────────────────────────────────────────────────

/// Weights per IQ2_XS block.
#[cfg(any(feature = "gpu", test))]
const IQ2XS_BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_XS block: 2 (d) + 64 (32 × u16 qs) + 8 (scales).
#[cfg(any(feature = "gpu", test))]
const IQ2XS_BLOCK_BYTES: usize = 74;
/// Number of super-blocks per IQ2_XS block.
#[cfg(any(feature = "gpu", test))]
const IQ2XS_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
#[cfg(any(feature = "gpu", test))]
const IQ2XS_SUPER_BLOCK_SIZE: usize = IQ2XS_BLOCK_SIZE / IQ2XS_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block (4 groups × 8 = 32 weights).
#[cfg(any(feature = "gpu", test))]
const IQ2XS_GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
#[cfg(any(feature = "gpu", test))]
const IQ2XS_WEIGHTS_PER_GROUP: usize = 8;
/// Byte offset of `qs` (after 2-byte FP16 scale).
#[cfg(any(feature = "gpu", test))]
const IQ2XS_QS_OFFSET: usize = 2;
/// Byte offset of `scales` (after d + qs = 2 + 64 = 66).
#[cfg(any(feature = "gpu", test))]
const IQ2XS_SCALES_OFFSET: usize = 66;

/// Dequantise all IQ2_XS blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq2_xs_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    use super::iq_grids::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

    let blocks_per_row = cols.div_ceil(IQ2XS_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * IQ2XS_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * IQ2XS_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + IQ2XS_BLOCK_BYTES];

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs_bytes = &block[IQ2XS_QS_OFFSET..IQ2XS_SCALES_OFFSET];
            let scales = &block[IQ2XS_SCALES_OFFSET..IQ2XS_BLOCK_BYTES];

            for (ib32, &scale_byte) in scales.iter().enumerate().take(IQ2XS_N_SUPERBLOCKS) {
                // Groups 0-1 use db0, groups 2-3 use db1.
                let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
                let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;

                let weight_base = blk * IQ2XS_BLOCK_SIZE + ib32 * IQ2XS_SUPER_BLOCK_SIZE;

                for l in 0..IQ2XS_GROUPS_PER_SUPER {
                    // u16 at position 4*ib32 + l (in u16 units = 8*ib32 + 2*l bytes).
                    let byte_pos = 8 * ib32 + 2 * l;
                    let qs_val =
                        u16::from_le_bytes([qs_bytes[byte_pos], qs_bytes[byte_pos + 1]]) as usize;

                    let grid_idx = qs_val & 511;
                    let sign_idx = qs_val >> 9;

                    let magnitudes: [u8; 8] = IQ2XS_GRID[grid_idx].to_le_bytes();
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];

                    // Groups 0-1 use db0, groups 2-3 use db1.
                    let dl = if l < 2 { db0 } else { db1 };

                    let group_base = weight_base + l * IQ2XS_WEIGHTS_PER_GROUP;
                    for j in 0..IQ2XS_WEIGHTS_PER_GROUP {
                        let col = group_base + j;
                        if col < cols {
                            let mag = magnitudes[j] as f32;
                            let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                                -1.0_f32
                            } else {
                                1.0_f32
                            };
                            f32_weights[row * cols + col] = dl * mag * sign;
                        }
                    }
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_iq2_xs(
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

    let f32_weights = dequant_iq2_xs_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "iq2_xs-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "iq2_xs-input", input);
    let output_buf = create_output_f32(&ctx.device, "iq2_xs-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "iq2_xs-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_iq2_xs"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("iq2_xs-bgl"),
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
            label: Some("iq2_xs-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("iq2_xs-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("iq2_xs-bg"),
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
            label: Some("iq2_xs-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("iq2_xs-pass"),
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

    fn make_zero_iq2_xs_block(scale: f32) -> Vec<u8> {
        let mut block = vec![0u8; IQ2XS_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_dequant_iq2_xs_zero_scale() {
        // d = 0 → all weights = 0.
        let block = make_zero_iq2_xs_block(0.0);
        let result = dequant_iq2_xs_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, 0.0, "weight[{i}] expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_iq2_xs_grid0_all_positive() {
        // All-zero qs → grid_idx=0, sign_idx=0, scales=0.
        // IQ2XS_GRID[0] = 0x0808080808080808 → all magnitudes = 8.
        // KSIGNS_IQ2XS[0] = 0 → all positive.
        // db0 = d * (0.5 + 0) * 0.25 = d * 0.125.
        // weight = d * 0.125 * 8 = d.
        let d = 2.0_f32;
        let block = make_zero_iq2_xs_block(d);
        let result = dequant_iq2_xs_to_f32(&block, 1, 256).expect("dequant");
        let expected = d * 0.125 * 8.0;
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq2_xs_high_nibble_scale() {
        // scales[0] = 0x20 → low nibble = 0, high nibble = 2.
        // db[0] = d * 0.125 (groups l=0,1), db[1] = d * 0.625 (groups l=2,3)
        let d = 1.0_f32;
        let mut block = make_zero_iq2_xs_block(d);
        block[IQ2XS_SCALES_OFFSET] = 0x20;
        let result = dequant_iq2_xs_to_f32(&block, 1, 256).expect("dequant");
        // Groups 0-1 (weights 0..15): db[0] * 8 = 1.0
        let expected_lo = d * 0.125 * 8.0;
        for (i, &v) in result.iter().enumerate().take(16) {
            assert!(
                (v - expected_lo).abs() < 1e-4,
                "out[{i}]={v}, expected {expected_lo}"
            );
        }
        // Groups 2-3 (weights 16..31): db[1] * 8 = 1.0 * 0.625 * 8 = 5.0
        let expected_hi = d * 0.625 * 8.0;
        for (i, &v) in result.iter().enumerate().take(32).skip(16) {
            assert!(
                (v - expected_hi).abs() < 1e-4,
                "out[{i}]={v}, expected {expected_hi}"
            );
        }
    }

    #[test]
    fn test_dequant_iq2_xs_too_small() {
        assert!(
            dequant_iq2_xs_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_iq2_xs_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Iq2XsGpuKernel;
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_iq2_xs_matches_cpu_reference() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 64;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * IQ2XS_BLOCK_BYTES);
        for r in 0..rows {
            let d_val = 0.01 + r as f32 * 0.001;
            let mut block = vec![0u8; IQ2XS_BLOCK_BYTES];
            let d_le = half::f16::from_f32(d_val).to_le_bytes();
            block[0] = d_le[0];
            block[1] = d_le[1];
            // Vary the scales: set some nibbles for interesting values.
            for sb in 0..IQ2XS_N_SUPERBLOCKS {
                let lo = ((r + sb * 2) % 16) as u8;
                let hi = ((r + sb * 2 + 3) % 16) as u8;
                block[IQ2XS_SCALES_OFFSET + sb] = lo | (hi << 4);
            }
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let f32_weights = dequant_iq2_xs_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Iq2XsGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut result, rows, cols)
            .expect("GPU GEMV IQ2_XS");

        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

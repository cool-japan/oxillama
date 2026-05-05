//! IQ1_S GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the IQ1_S block
//!      format: 256 weights per 50-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! IQ1_S block layout (50 bytes per 256 weights):
//! - bytes  0-1:   d (f16 little-endian global scale)
//! - bytes  2-33:  `qs[32]` — 32 bytes (lower 8 bits of each 11-bit grid index)
//! - bytes 34-49:  `qh[8]`  — 8 × u16 LE sub-block headers
//!
//! Sub-block decode (8 sub-blocks of 32 weights each):
//!   `dl = d * (2 * scale_bits + 1)` where `scale_bits = (qh[ib] >> 12) & 7`
//!   `delta = if qh[ib] & 0x8000 != 0 { -0.125 } else { 0.125 }`
//!   11-bit grid index: `grid_idx = qs[4*ib + l] | (((qh[ib] >> (3*l)) & 7) << 8)`
//!   Grid byte → i8 signed: `y[j] = dl * (grid_i8[j] as f32 + delta)`
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// IQ1_S GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Iq1SGpuKernel;

impl GpuKernel for Iq1SGpuKernel {
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
            gpu_gemv_iq1_s(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── IQ1_S block constants ────────────────────────────────────────────────────

/// Weights per IQ1_S block.
#[cfg(any(feature = "gpu", test))]
const IQ1S_BLOCK_SIZE: usize = 256;
/// Bytes per IQ1_S block: 2 (FP16 d) + 32 (qs) + 16 (qh as 8 × u16).
#[cfg(any(feature = "gpu", test))]
const IQ1S_BLOCK_BYTES: usize = 50;
/// Byte offset where `qs` begins.
#[cfg(any(feature = "gpu", test))]
const IQ1S_QS_OFFSET: usize = 2;
/// Byte offset where `qh` (8 × u16 LE) begins.
#[cfg(any(feature = "gpu", test))]
const IQ1S_QH_OFFSET: usize = 34;
/// Number of sub-blocks per IQ1_S block.
#[cfg(any(feature = "gpu", test))]
const IQ1S_N_SUBBLOCKS: usize = 8;
/// Weights per sub-block.
#[cfg(any(feature = "gpu", test))]
const IQ1S_SUB_BLOCK_SIZE: usize = IQ1S_BLOCK_SIZE / IQ1S_N_SUBBLOCKS; // 32
/// Number of groups per sub-block (each group = 8 weights).
#[cfg(any(feature = "gpu", test))]
const IQ1S_GROUPS_PER_SUB: usize = 4;
/// Weights per group.
#[cfg(any(feature = "gpu", test))]
const IQ1S_WEIGHTS_PER_GROUP: usize = 8;
/// Delta constant (0.125).
#[cfg(any(feature = "gpu", test))]
const IQ1S_DELTA: f32 = 0.125;

/// Dequantise all IQ1_S blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq1_s_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    use super::iq1s_grid::IQ1S_GRID;

    let blocks_per_row = cols.div_ceil(IQ1S_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * IQ1S_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * IQ1S_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + IQ1S_BLOCK_BYTES];

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[IQ1S_QS_OFFSET..IQ1S_QH_OFFSET];
            let qh_bytes = &block[IQ1S_QH_OFFSET..IQ1S_BLOCK_BYTES];

            for ib in 0..IQ1S_N_SUBBLOCKS {
                let qh_val = u16::from_le_bytes([qh_bytes[ib * 2], qh_bytes[ib * 2 + 1]]);

                // Scale: d × (2 × scale_bits + 1), scale_bits from bits 12-14.
                let scale_bits = ((qh_val >> 12) & 0x7) as f32;
                let dl = d * (2.0 * scale_bits + 1.0);

                // Delta sign from bit 15.
                let delta = if qh_val & 0x8000 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                };

                let qs_base = ib * IQ1S_GROUPS_PER_SUB;
                let output_base = blk * IQ1S_BLOCK_SIZE + ib * IQ1S_SUB_BLOCK_SIZE;

                for l in 0..IQ1S_GROUPS_PER_SUB {
                    // 11-bit grid index: lower 8 bits from qs, upper 3 bits from qh.
                    let upper_bits = ((qh_val >> (3 * l as u16)) & 0x7) as usize;
                    let grid_idx = (qs[qs_base + l] as usize) | (upper_bits << 8);

                    let grid_raw = IQ1S_GRID[grid_idx].to_le_bytes();

                    let group_base = output_base + l * IQ1S_WEIGHTS_PER_GROUP;
                    for (j, &grid_byte) in grid_raw.iter().enumerate().take(IQ1S_WEIGHTS_PER_GROUP)
                    {
                        let col = group_base + j;
                        if col < cols {
                            let gv = grid_byte as i8 as f32;
                            f32_weights[row * cols + col] = dl * (gv + delta);
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
fn gpu_gemv_iq1_s(
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

    let f32_weights = dequant_iq1_s_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "iq1_s-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "iq1_s-input", input);
    let output_buf = create_output_f32(&ctx.device, "iq1_s-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "iq1_s-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_iq1_s"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("iq1_s-bgl"),
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
            label: Some("iq1_s-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("iq1_s-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("iq1_s-bg"),
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
            label: Some("iq1_s-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("iq1_s-pass"),
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

    fn make_zero_iq1s_block(scale: f32) -> Vec<u8> {
        let mut block = vec![0u8; IQ1S_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_dequant_iq1_s_zero_scale() {
        // d=0 → all weights = 0.
        let block = make_zero_iq1s_block(0.0);
        let result = dequant_iq1_s_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, 0.0, "weight[{i}] expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_iq1_s_grid0_positive_delta() {
        // Grid index 0 → IQ1S_GRID[0] = 0xffffffffffffffff → all bytes = 0xff = i8(-1).
        // qh = 0x0000 → scale_bits=0 → dl = d * 1.0; bit 15=0 → delta = +0.125.
        // Expected: d * (-1.0 + 0.125) = d * (-0.875).
        let d = 2.0_f32;
        let block = make_zero_iq1s_block(d);
        let result = dequant_iq1_s_to_f32(&block, 1, 256).expect("dequant");
        let expected = d * (-1.0 + IQ1S_DELTA);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1_s_scale_bits() {
        // qh[0] = 0x3000 → bits 12-14 = 0x3 → scale_bits = 3
        // dl = d * (2*3+1) = d * 7; grid[0]→-1; delta=+0.125
        // Expected for sub-block 0: d * 7 * (-0.875)
        let d = 1.0_f32;
        let mut block = make_zero_iq1s_block(d);
        block[IQ1S_QH_OFFSET] = 0x00;
        block[IQ1S_QH_OFFSET + 1] = 0x30; // qh[0] = 0x3000
        let result = dequant_iq1_s_to_f32(&block, 1, 256).expect("dequant");
        let expected_sb0 = d * 7.0 * (-1.0 + IQ1S_DELTA);
        for (i, &v) in result.iter().enumerate().take(32) {
            assert!(
                (v - expected_sb0).abs() < 1e-4,
                "weight[{i}] = {v}, expected {expected_sb0}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1_s_negative_delta() {
        // qh[0] bit 15 = 1 → delta = -0.125.
        let d = 1.0_f32;
        let mut block = make_zero_iq1s_block(d);
        block[IQ1S_QH_OFFSET] = 0x00;
        block[IQ1S_QH_OFFSET + 1] = 0x80; // qh[0] = 0x8000
        let result = dequant_iq1_s_to_f32(&block, 1, 256).expect("dequant");
        let expected_sb0 = d * 1.0 * (-1.0 - IQ1S_DELTA);
        for (i, &v) in result.iter().enumerate().take(32) {
            assert!(
                (v - expected_sb0).abs() < 1e-4,
                "weight[{i}] = {v}, expected {expected_sb0}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1_s_too_small() {
        assert!(
            dequant_iq1_s_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_iq1_s_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Iq1SGpuKernel;
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_iq1_s_matches_cpu_reference() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 32;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * IQ1S_BLOCK_BYTES);
        for r in 0..rows {
            let d_val = 0.02 + r as f32 * 0.001;
            let mut block = vec![0u8; IQ1S_BLOCK_BYTES];
            let d_le = half::f16::from_f32(d_val).to_le_bytes();
            block[0] = d_le[0];
            block[1] = d_le[1];
            // Set some varying scale bits in qh headers.
            for ib in 0..IQ1S_N_SUBBLOCKS {
                let scale_bits = ((r + ib) % 8) as u16;
                let sign_bit: u16 = if (r + ib) % 2 == 0 { 0x8000 } else { 0 };
                let qh_val: u16 = (scale_bits << 12) | sign_bit;
                let qh_le = qh_val.to_le_bytes();
                block[IQ1S_QH_OFFSET + ib * 2] = qh_le[0];
                block[IQ1S_QH_OFFSET + ib * 2 + 1] = qh_le[1];
            }
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let f32_weights = dequant_iq1_s_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Iq1SGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut result, rows, cols)
            .expect("GPU GEMV IQ1_S");

        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

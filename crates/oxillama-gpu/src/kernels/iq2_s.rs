//! IQ2_S GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the IQ2_S block
//!      format: 256 weights per 82-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! IQ2_S block layout (82 bytes per 256 weights):
//! - bytes  0-1:   d (f16 little-endian scale)
//! - bytes  2-65:  qs\[64\] — 64 raw bytes: first 32 = grid base indices, next 32 = sign masks
//! - bytes  66-73: qh\[8\] — 8 bytes, one per super-block (high bits for grid index)
//! - bytes  74-81: scales\[8\] — 8 bytes, one per super-block
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// IQ2_S GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Iq2SGpuKernel;

impl GpuKernel for Iq2SGpuKernel {
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
            gpu_gemv_iq2_s(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── IQ2_S block constants ────────────────────────────────────────────────────

/// Weights per IQ2_S block.
#[cfg(any(feature = "gpu", test))]
const IQ2S_BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_S block: 2 + 64 + 8 + 8 = 82.
#[cfg(any(feature = "gpu", test))]
const IQ2S_BLOCK_BYTES: usize = 82;
/// Number of super-blocks per IQ2_S block (QK_K/32 = 8).
#[cfg(any(feature = "gpu", test))]
const IQ2S_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
#[cfg(any(feature = "gpu", test))]
const IQ2S_SUPER_BLOCK_SIZE: usize = IQ2S_BLOCK_SIZE / IQ2S_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block.
#[cfg(any(feature = "gpu", test))]
const IQ2S_GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
#[cfg(any(feature = "gpu", test))]
const IQ2S_WEIGHTS_PER_GROUP: usize = 8;
/// Byte offset of qs region within a block.
#[cfg(any(feature = "gpu", test))]
const IQ2S_QS_OFFSET: usize = 2;
/// Total bytes in the qs region.
#[cfg(any(feature = "gpu", test))]
const IQ2S_QS_BYTES: usize = 64;
/// Byte boundary within qs where sign bytes begin.
#[cfg(any(feature = "gpu", test))]
const IQ2S_SIGNS_IN_QS: usize = 32;
/// Byte offset of qh region within a block.
#[cfg(any(feature = "gpu", test))]
const IQ2S_QH_OFFSET: usize = 66;
/// Byte offset of scales region within a block.
#[cfg(any(feature = "gpu", test))]
const IQ2S_SCALES_OFFSET: usize = 74;

/// Dequantise all IQ2_S blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq2_s_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    use super::iq_grids::{IQ2S_GRID, KMASK_IQ2XS};

    let blocks_per_row = cols.div_ceil(IQ2S_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * IQ2S_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * IQ2S_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + IQ2S_BLOCK_BYTES];

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[IQ2S_QS_OFFSET..IQ2S_QS_OFFSET + IQ2S_QS_BYTES];
            let qs_base = &qs[..IQ2S_SIGNS_IN_QS];
            let qs_signs = &qs[IQ2S_SIGNS_IN_QS..];
            let qh = &block[IQ2S_QH_OFFSET..IQ2S_SCALES_OFFSET];
            let scales = &block[IQ2S_SCALES_OFFSET..IQ2S_BLOCK_BYTES];

            for ib32 in 0..IQ2S_N_SUPERBLOCKS {
                let scale_byte = scales[ib32];
                let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
                let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                let qh_byte = qh[ib32] as u16;

                let weight_base = blk * IQ2S_BLOCK_SIZE + ib32 * IQ2S_SUPER_BLOCK_SIZE;

                for l in 0..IQ2S_GROUPS_PER_SUPER {
                    let base_idx = qs_base[4 * ib32 + l] as u16;
                    let shift = 8u16.saturating_sub(2 * l as u16);
                    let high_bit = (qh_byte << shift) & 0x300;
                    let grid_idx = (base_idx | high_bit) as usize;

                    let signs_byte = qs_signs[4 * ib32 + l];
                    let magnitudes: [u8; 8] = IQ2S_GRID[grid_idx].to_le_bytes();
                    let dl = if l < 2 { db0 } else { db1 };

                    let group_base = weight_base + l * IQ2S_WEIGHTS_PER_GROUP;
                    for j in 0..IQ2S_WEIGHTS_PER_GROUP {
                        let col = group_base + j;
                        if col < cols {
                            let mag = magnitudes[j] as f32;
                            let sign = if signs_byte & KMASK_IQ2XS[j] != 0 {
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
fn gpu_gemv_iq2_s(
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

    let f32_weights = dequant_iq2_s_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "iq2_s-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "iq2_s-input", input);
    let output_buf = create_output_f32(&ctx.device, "iq2_s-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "iq2_s-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_iq2_s"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("iq2_s-bgl"),
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
            label: Some("iq2_s-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("iq2_s-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("iq2_s-bg"),
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
            label: Some("iq2_s-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("iq2_s-pass"),
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

    /// Build a zero IQ2_S block with given scale.
    /// All-zero → grid_idx=0, IQ2S_GRID[0]=0x0808080808080808, mags=8, signs=positive.
    /// scales=0 → db = d * 0.5 * 0.25.
    fn make_zero_block(scale: f32) -> Vec<u8> {
        let mut block = vec![0u8; IQ2S_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_dequant_zero_scale() {
        let block = make_zero_block(0.0);
        let result = dequant_iq2_s_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, 0.0, "weight[{i}] expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_grid0_all_positive() {
        // IQ2S_GRID[0] = 0x0808080808080808 → mags all 8.
        // db = d * (0.5 + 0) * 0.25 = d * 0.125; weight = d * 0.125 * 8 = d.
        let d = 2.0_f32;
        let block = make_zero_block(d);
        let result = dequant_iq2_s_to_f32(&block, 1, 256).expect("dequant");
        let expected = d * 0.125 * 8.0;
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_too_small() {
        assert!(
            dequant_iq2_s_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Iq2SGpuKernel;
    }

    #[test]
    fn test_dequant_sign_applied() {
        // qs_signs[0] = 1 → bit 0 set → weight[0] of first group negated.
        // qs_signs start at block[2 + IQ2S_SIGNS_IN_QS] = block[2 + 32] = block[34].
        let d = 1.0_f32;
        let mut block = make_zero_block(d);
        block[2 + IQ2S_SIGNS_IN_QS] = 1;
        let result = dequant_iq2_s_to_f32(&block, 1, 256).expect("dequant");
        let dl = d * 0.5 * 0.25;
        let mag = 8.0_f32;
        assert!(
            (result[0] - (-dl * mag)).abs() < 1e-5,
            "weight[0]={}, expected {}",
            result[0],
            -dl * mag
        );
        assert!(
            (result[1] - (dl * mag)).abs() < 1e-5,
            "weight[1]={}, expected {}",
            result[1],
            dl * mag
        );
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_iq2_s_matches_cpu() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 64;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * IQ2S_BLOCK_BYTES);
        for r in 0..rows {
            let d_val = 0.01 + r as f32 * 0.001;
            let mut block = vec![0u8; IQ2S_BLOCK_BYTES];
            let d_le = half::f16::from_f32(d_val).to_le_bytes();
            block[0] = d_le[0];
            block[1] = d_le[1];
            // Vary the scale nibbles for each super-block
            for ib32 in 0..IQ2S_N_SUPERBLOCKS {
                let low_nibble = ((r + ib32 * 3) % 15) as u8;
                let high_nibble = ((r + ib32 * 7 + 5) % 15) as u8;
                block[IQ2S_SCALES_OFFSET + ib32] = (high_nibble << 4) | low_nibble;
            }
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let f32_weights = dequant_iq2_s_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                f32_weights[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum()
            })
            .collect();

        let mut output = vec![0.0f32; rows];
        let kernel = Iq2SGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("GPU GEMV IQ2_S");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

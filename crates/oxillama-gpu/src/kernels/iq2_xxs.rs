//! IQ2_XXS GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the IQ2_XXS block
//!      format: 256 weights per 66-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! IQ2_XXS block layout (66 bytes per 256 weights):
//! - bytes  0-1: d (f16 little-endian scale)
//! - bytes  2-65: `qs[32]` — 32 × u16, stored as 64 raw bytes (little-endian).
//!   The 256 weights are divided into 8 super-blocks of 32 weights each.
//!   Each super-block occupies 8 bytes (2 × u32):
//!   - `aux32[0]` → 4 grid indices (1 byte each)
//!   - `aux32[1]` → 4-bit scale (bits 28-31), 4 × 7-bit sign selectors
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// IQ2_XXS GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Iq2XxsGpuKernel;

impl GpuKernel for Iq2XxsGpuKernel {
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
            gpu_gemv_iq2_xxs(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── IQ2_XXS block constants ──────────────────────────────────────────────────

/// Weights per IQ2_XXS block.
#[cfg(any(feature = "gpu", test))]
const IQ2XXS_BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_XXS block.
#[cfg(any(feature = "gpu", test))]
const IQ2XXS_BLOCK_BYTES: usize = 66;
/// Number of super-blocks per IQ2_XXS block.
#[cfg(any(feature = "gpu", test))]
const IQ2XXS_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
#[cfg(any(feature = "gpu", test))]
const IQ2XXS_SUPER_BLOCK_SIZE: usize = IQ2XXS_BLOCK_SIZE / IQ2XXS_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block.
#[cfg(any(feature = "gpu", test))]
#[allow(dead_code)]
const IQ2XXS_GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
#[cfg(any(feature = "gpu", test))]
const IQ2XXS_WEIGHTS_PER_GROUP: usize = 8;

/// Dequantise all IQ2_XXS blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq2_xxs_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    use super::iq_grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

    let blocks_per_row = cols.div_ceil(IQ2XXS_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * IQ2XXS_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * IQ2XXS_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + IQ2XXS_BLOCK_BYTES];

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..IQ2XXS_BLOCK_BYTES];

            for ib32 in 0..IQ2XXS_N_SUPERBLOCKS {
                let base = ib32 * 8;
                let aux32_0 =
                    u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
                let aux32_1 =
                    u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);

                let scale_factor = (aux32_1 >> 28) as f32;
                let db = d * (0.5 + scale_factor) * 0.25;
                let aux8: [u8; 4] = aux32_0.to_le_bytes();

                let weight_base = blk * IQ2XXS_BLOCK_SIZE + ib32 * IQ2XXS_SUPER_BLOCK_SIZE;

                for (l, &grid_byte) in aux8.iter().enumerate() {
                    let grid_idx = grid_byte as usize;
                    let magnitudes: [u8; 8] = IQ2XXS_GRID[grid_idx].to_le_bytes();

                    let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];

                    let group_base = weight_base + l * IQ2XXS_WEIGHTS_PER_GROUP;
                    for j in 0..IQ2XXS_WEIGHTS_PER_GROUP {
                        let col = group_base + j;
                        if col < cols {
                            let mag = magnitudes[j] as f32;
                            let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                                -1.0_f32
                            } else {
                                1.0_f32
                            };
                            f32_weights[row * cols + col] = db * mag * sign;
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
fn gpu_gemv_iq2_xxs(
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

    let f32_weights = dequant_iq2_xxs_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "iq2_xxs-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "iq2_xxs-input", input);
    let output_buf = create_output_f32(&ctx.device, "iq2_xxs-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "iq2_xxs-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_iq2_xxs"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("iq2_xxs-bgl"),
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
            label: Some("iq2_xxs-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("iq2_xxs-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("iq2_xxs-bg"),
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
            label: Some("iq2_xxs-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("iq2_xxs-pass"),
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

    /// Build a zero IQ2_XXS block with given scale.
    /// All-zero qs → grid_idx=0 for all groups → IQ2XXS_GRID[0] = 0x0808080808080808 → all mags=8.
    /// aux32[1]=0 → scale_bits=0 → db = d * 0.5 * 0.25, sign_idx=0 → all positive.
    fn make_zero_block(scale: f32) -> Vec<u8> {
        let mut block = vec![0u8; IQ2XXS_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_dequant_zero_scale() {
        let block = make_zero_block(0.0);
        let result = dequant_iq2_xxs_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, 0.0, "weight[{i}] expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_grid0_all_positive() {
        // d=2.0, all qs=0 → grid_idx=0, mags=8, signs=positive, scale_bits=0.
        // db = 2.0 * (0.5 + 0) * 0.25 = 0.25; weight = 0.25 * 8 = 2.0.
        let block = make_zero_block(2.0);
        let result = dequant_iq2_xxs_to_f32(&block, 1, 256).expect("dequant");
        let expected = 2.0 * 0.5 * 0.25 * 8.0;
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
            dequant_iq2_xxs_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Iq2XxsGpuKernel;
    }

    #[test]
    fn test_dequant_two_rows() {
        let block0 = make_zero_block(0.0); // all zeros
        let block1 = make_zero_block(2.0); // uniform positive
        let mut data = Vec::new();
        data.extend_from_slice(&block0);
        data.extend_from_slice(&block1);
        let result = dequant_iq2_xxs_to_f32(&data, 2, 256).expect("dequant");
        // Row 0: all 0
        for &v in &result[..256] {
            assert_eq!(v, 0.0, "row0 expected 0, got {v}");
        }
        // Row 1: expected = 2.0 * 0.125 * 8 = 2.0
        let expected = 2.0 * 0.5 * 0.25 * 8.0;
        for (i, &v) in result[256..].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "row1 weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_iq2_xxs_matches_cpu() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 64;
        let cols = 256;

        // Build varied weight blocks: each row uses a different scale.
        let mut weight_bytes = Vec::with_capacity(rows * IQ2XXS_BLOCK_BYTES);
        for r in 0..rows {
            let d_val = 0.01 + r as f32 * 0.001;
            let mut block = vec![0u8; IQ2XXS_BLOCK_BYTES];
            let d_le = half::f16::from_f32(d_val).to_le_bytes();
            block[0] = d_le[0];
            block[1] = d_le[1];
            // Vary the scale bits of aux32[1] for each super-block
            for ib32 in 0..IQ2XXS_N_SUPERBLOCKS {
                let base = 2 + ib32 * 8;
                // aux32[1] lower 28 bits = sign selectors (0 = all positive)
                // bits 28-31 = scale multiplier 0..15
                let scale_nibble = ((r + ib32 * 3) % 16) as u8;
                block[base + 4] = 0;
                block[base + 5] = 0;
                block[base + 6] = 0;
                block[base + 7] = scale_nibble << 4; // bits 28-31
            }
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        // CPU reference.
        let f32_weights = dequant_iq2_xxs_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Iq2XxsGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("GPU GEMV IQ2_XXS");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

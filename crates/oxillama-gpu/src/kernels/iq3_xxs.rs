//! IQ3_XXS GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the IQ3_XXS block
//!      format: 256 weights per 98-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! IQ3_XXS block layout (98 bytes per 256 weights):
//! - bytes  0-1:  d (f16 little-endian scale)
//! - bytes  2-65: qs_grid\[64\] — 64 bytes: 8 bytes per super-block (2 per group × 4 groups)
//! - bytes  66-97: qs_signs\[32\] — 32 bytes: 4 bytes per super-block (1 u32 per sb)
//!   aux32 = u32::from_le_bytes(qs_signs[4*ib32..4*ib32+4]):
//!   - bits 28-31: 4-bit scale multiplier
//!   - bits [7*l .. 7*l+6] for l in 0..4: 7-bit sign selector per group
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// IQ3_XXS GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Iq3XxsGpuKernel;

impl GpuKernel for Iq3XxsGpuKernel {
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
            gpu_gemv_iq3_xxs(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── IQ3_XXS block constants ──────────────────────────────────────────────────

/// Weights per IQ3_XXS block.
#[cfg(any(feature = "gpu", test))]
const IQ3XXS_BLOCK_SIZE: usize = 256;
/// Bytes per IQ3_XXS block: 2 + 96 = 98.
#[cfg(any(feature = "gpu", test))]
const IQ3XXS_BLOCK_BYTES: usize = 98;
/// Number of super-blocks per IQ3_XXS block.
#[cfg(any(feature = "gpu", test))]
const IQ3XXS_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
#[cfg(any(feature = "gpu", test))]
const IQ3XXS_SUPER_BLOCK_SIZE: usize = IQ3XXS_BLOCK_SIZE / IQ3XXS_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block.
#[cfg(any(feature = "gpu", test))]
const IQ3XXS_GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
#[cfg(any(feature = "gpu", test))]
const IQ3XXS_WEIGHTS_PER_GROUP: usize = 8;
/// Offset within qs where sign/scale data starts (QK_K/4 = 64 bytes into qs).
#[cfg(any(feature = "gpu", test))]
const IQ3XXS_SIGNS_OFFSET: usize = 64;

/// Dequantise all IQ3_XXS blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq3_xxs_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    use super::iq_grids::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

    let blocks_per_row = cols.div_ceil(IQ3XXS_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * IQ3XXS_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * IQ3XXS_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + IQ3XXS_BLOCK_BYTES];

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..IQ3XXS_BLOCK_BYTES];
            let qs_grid = &qs[..IQ3XXS_SIGNS_OFFSET];
            let qs_signs = &qs[IQ3XXS_SIGNS_OFFSET..];

            for ib32 in 0..IQ3XXS_N_SUPERBLOCKS {
                let signs_base = ib32 * 4;
                let aux32 = u32::from_le_bytes([
                    qs_signs[signs_base],
                    qs_signs[signs_base + 1],
                    qs_signs[signs_base + 2],
                    qs_signs[signs_base + 3],
                ]);

                let scale_bits = (aux32 >> 28) as f32;
                let db = d * (0.5 + scale_bits) * 0.5;

                let grid_base = ib32 * 8;
                let weight_base = blk * IQ3XXS_BLOCK_SIZE + ib32 * IQ3XXS_SUPER_BLOCK_SIZE;

                for l in 0..IQ3XXS_GROUPS_PER_SUPER {
                    let g1 = qs_grid[grid_base + 2 * l] as usize;
                    let g2 = qs_grid[grid_base + 2 * l + 1] as usize;
                    let mags1: [u8; 4] = IQ3XXS_GRID[g1].to_le_bytes();
                    let mags2: [u8; 4] = IQ3XXS_GRID[g2].to_le_bytes();

                    let sign_idx = ((aux32 >> (7 * l)) & 0x7F) as usize;
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];

                    let group_base = weight_base + l * IQ3XXS_WEIGHTS_PER_GROUP;
                    for j in 0..4 {
                        // First 4 weights from grid1.
                        let col0 = group_base + j;
                        if col0 < cols {
                            let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                                -1.0_f32
                            } else {
                                1.0_f32
                            };
                            f32_weights[row * cols + col0] = db * mags1[j] as f32 * sign1;
                        }
                        // Second 4 weights from grid2.
                        let col1 = group_base + j + 4;
                        if col1 < cols {
                            let sign2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0_f32
                            } else {
                                1.0_f32
                            };
                            f32_weights[row * cols + col1] = db * mags2[j] as f32 * sign2;
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
fn gpu_gemv_iq3_xxs(
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

    let f32_weights = dequant_iq3_xxs_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "iq3_xxs-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "iq3_xxs-input", input);
    let output_buf = create_output_f32(&ctx.device, "iq3_xxs-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "iq3_xxs-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_iq3_xxs"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("iq3_xxs-bgl"),
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
            label: Some("iq3_xxs-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("iq3_xxs-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("iq3_xxs-bg"),
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
            label: Some("iq3_xxs-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("iq3_xxs-pass"),
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

    /// Build a zero IQ3_XXS block with given scale.
    /// All-zero → grid_idx=0 for both grids, IQ3XXS_GRID[0]=0x04040404 → mags=4.
    /// aux32=0 → scale_bits=0 → db = d * 0.5 * 0.5, signs all positive.
    fn make_zero_block(scale: f32) -> Vec<u8> {
        let mut block = vec![0u8; IQ3XXS_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_dequant_zero_scale() {
        let block = make_zero_block(0.0);
        let result = dequant_iq3_xxs_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, 0.0, "weight[{i}] expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_grid0_all_positive() {
        // IQ3XXS_GRID[0] = 0x04040404 → mags all 4.
        // db = d * 0.5 * 0.5 = d * 0.25; weight = d * 0.25 * 4 = d.
        let d = 2.0_f32;
        let block = make_zero_block(d);
        let result = dequant_iq3_xxs_to_f32(&block, 1, 256).expect("dequant");
        let expected = d * 0.25 * 4.0;
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
            dequant_iq3_xxs_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Iq3XxsGpuKernel;
    }

    #[test]
    fn test_dequant_sign_flip() {
        // KSIGNS_IQ2XS[1] = 129 = 0b10000001 → weight[0] and weight[7] negated.
        // Set aux32 = 1 for first super-block (sign_idx=1 for group 0).
        let d = 1.0_f32;
        let mut block = make_zero_block(d);
        // qs_signs is at block[2 + IQ3XXS_SIGNS_OFFSET] = block[2 + 64] = block[66].
        block[2 + IQ3XXS_SIGNS_OFFSET] = 1;
        let result = dequant_iq3_xxs_to_f32(&block, 1, 256).expect("dequant");
        let db = d * 0.25 * 4.0;
        // weight[0]: sign_byte bit 0 set → negative
        assert!(
            (result[0] - (-db)).abs() < 1e-5,
            "weight[0]={}, expected {}",
            result[0],
            -db
        );
        // weight[1]: bit 1 not set → positive
        assert!(
            (result[1] - db).abs() < 1e-5,
            "weight[1]={}, expected {}",
            result[1],
            db
        );
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_iq3_xxs_matches_cpu() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 64;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * IQ3XXS_BLOCK_BYTES);
        for r in 0..rows {
            let d_val = 0.01 + r as f32 * 0.001;
            let mut block = vec![0u8; IQ3XXS_BLOCK_BYTES];
            let d_le = half::f16::from_f32(d_val).to_le_bytes();
            block[0] = d_le[0];
            block[1] = d_le[1];
            // Vary scale bits in qs_signs for each super-block.
            for ib32 in 0..IQ3XXS_N_SUPERBLOCKS {
                // aux32 bits 28-31 = scale nibble; keep signs = 0 for simplicity.
                let scale_nibble = ((r + ib32 * 2) % 16) as u32;
                let aux32: u32 = scale_nibble << 28;
                let aux32_bytes = aux32.to_le_bytes();
                let signs_base = 2 + IQ3XXS_SIGNS_OFFSET + ib32 * 4;
                block[signs_base] = aux32_bytes[0];
                block[signs_base + 1] = aux32_bytes[1];
                block[signs_base + 2] = aux32_bytes[2];
                block[signs_base + 3] = aux32_bytes[3];
            }
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let f32_weights = dequant_iq3_xxs_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Iq3XxsGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("GPU GEMV IQ3_XXS");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

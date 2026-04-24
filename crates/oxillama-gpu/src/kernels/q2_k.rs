//! Q2_K GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q2_K block
//!      format: 256 weights per 84-byte super-block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! Q2_K block layout (84 bytes per 256 weights):
//! - bytes  0-15: scales (16 bytes) — 16 sub-blocks, each byte encodes lo4=scale, hi4=min
//! - bytes 16-79: qs (64 bytes) — 256 × 2-bit weights packed (4 per byte)
//! - bytes 80-81: d  (f16 little-endian super-block scale)
//! - bytes 82-83: dmin (f16 little-endian super-block minimum)
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q2_K GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
#[allow(non_camel_case_types)]
pub struct Q2_KGpuKernel;

impl GpuKernel for Q2_KGpuKernel {
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
            gpu_gemv_q2_k(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q2_K block constants ─────────────────────────────────────────────────────

/// Weights per Q2_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q2_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q2_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q2_K_BLOCK_BYTES: usize = 84;

/// Dequantise all Q2_K blocks to a flat f32 buffer.
///
/// Q2_K layout:
/// - Bytes  0-15: scales (16 bytes): lo4 = scale, hi4 = min for each sub-block.
/// - Bytes 16-79: qs (64 bytes): 256 × 2-bit packed values (4 per byte).
/// - Bytes 80-81: d   (f16 super-block scale).
/// - Bytes 82-83: dmin (f16 super-block minimum).
///
/// Weight formula: `w = d * scale_i * q - dmin * min_i`
/// where q is 2-bit (0..3).
#[cfg(any(feature = "gpu", test))]
fn dequant_q2_k_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q2_K_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q2_K_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * Q2_K_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + Q2_K_BLOCK_BYTES];

            let scales = &block[0..16];
            let qs = &block[16..80];
            let d = half::f16::from_bits(u16::from_le_bytes([block[80], block[81]])).to_f32();
            let dmin = half::f16::from_bits(u16::from_le_bytes([block[82], block[83]])).to_f32();

            // Process in 2 groups of 128 weights each, matching Q2KRef layout.
            let mut is: usize = 0; // sub-block index
            let mut qs_off: usize = 0;
            let mut weight_off: usize = 0;

            for _group in 0..2 {
                for shift in (0..8).step_by(2) {
                    // First 16 weights of this sub-block.
                    let sc_byte = scales[is];
                    let dl = d * (sc_byte & 0x0F) as f32;
                    let ml = dmin * (sc_byte >> 4) as f32;
                    is += 1;

                    for l in 0..16 {
                        let col = blk * Q2_K_BLOCK_SIZE + weight_off + l;
                        if col < cols {
                            let q = (qs[qs_off + l] >> shift) & 3;
                            f32_weights[row * cols + col] = dl * q as f32 - ml;
                        }
                    }
                    weight_off += 16;

                    // Second 16 weights of this sub-block.
                    let sc_byte = scales[is];
                    let dl = d * (sc_byte & 0x0F) as f32;
                    let ml = dmin * (sc_byte >> 4) as f32;
                    is += 1;

                    for l in 0..16 {
                        let col = blk * Q2_K_BLOCK_SIZE + weight_off + l;
                        if col < cols {
                            let q = (qs[qs_off + 16 + l] >> shift) & 3;
                            f32_weights[row * cols + col] = dl * q as f32 - ml;
                        }
                    }
                    weight_off += 16;
                }
                qs_off += 32;
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q2_k(
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

    let f32_weights = dequant_q2_k_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "q2_k-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q2_k-input", input);
    let output_buf = create_output_f32(&ctx.device, "q2_k-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q2_k-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q2_k"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q2_k-bgl"),
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
            label: Some("q2_k-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q2_k-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q2_k-bg"),
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
            label: Some("q2_k-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q2_k-pass"),
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

    /// Build a Q2_K block (84 bytes) for testing.
    fn make_q2_k_block(d: f32, dmin: f32, scales: &[u8; 16], qs: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q2_K_BLOCK_BYTES);
        block.extend_from_slice(scales);
        block.extend_from_slice(qs);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block
    }

    #[test]
    fn test_dequant_q2_k_zeros() {
        // d=0, dmin=0 → all weights = 0
        let block = make_q2_k_block(0.0, 0.0, &[0; 16], &[0; 64]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let result = dequant_q2_k_to_f32(&data, 2, 256).expect("dequant should succeed");
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q2_k_uniform() {
        // d=1.0, dmin=0.0, all scales=0x01 (scale=1, min=0), all qs=0xFF (2-bit = 3)
        // Weight = 1.0 * 1 * 3 - 0 = 3.0
        let scales = [0x01u8; 16];
        let qs = [0xFFu8; 64];
        let block = make_q2_k_block(1.0, 0.0, &scales, &qs);
        let result = dequant_q2_k_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 3.0).abs() < 0.01, "weight[{i}] = {v}, expected 3.0");
        }
    }

    #[test]
    fn test_dequant_q2_k_with_min() {
        // d=2.0, dmin=1.0, all scales=0x11 (scale=1, min=1), all qs=0x00 (2-bit=0)
        // Weight = 2.0 * 1 * 0 - 1.0 * 1 = -1.0
        let scales = [0x11u8; 16];
        let qs = [0x00u8; 64];
        let block = make_q2_k_block(2.0, 1.0, &scales, &qs);
        let result = dequant_q2_k_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 0.01,
                "weight[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_q2_k_too_small() {
        assert!(
            dequant_q2_k_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_q2_k_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Q2_KGpuKernel;
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q2_k_matches_cpu() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return, // skip if no GPU
        };

        // 64 rows × 256 cols: one block per row (84 bytes each).
        let rows = 64;
        let cols = 256;

        // Build weight bytes: varied scale/min bytes, varied qs
        let mut weight_bytes = Vec::with_capacity(rows * Q2_K_BLOCK_BYTES);
        for r in 0..rows {
            let mut scales = [0u8; 16];
            let mut qs = [0u8; 64];
            for (i, s) in scales.iter_mut().enumerate() {
                // scale nibble = 1..5, min nibble = 0..3
                *s = (((r + i + 1) % 5 + 1) as u8) | (((r + i) % 4) as u8) << 4;
            }
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((r * 7 + i * 3 + 5) & 0xFF) as u8;
            }
            let d_val = 0.01 + (r as f32) * 0.001;
            let dmin_val = 0.005 + (r as f32) * 0.0005;
            let block = make_q2_k_block(d_val, dmin_val, &scales, &qs);
            weight_bytes.extend_from_slice(&block);
        }

        // Varied input vector.
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        // CPU reference: dequant then dot.
        let f32_weights = dequant_q2_k_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Q2_KGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("GPU GEMV Q2_K");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

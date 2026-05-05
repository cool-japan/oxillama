//! TQ2_0 GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the TQ2_0 block
//!      format: 256 weights per 66-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! TQ2_0 block layout (66 bytes per 256 weights):
//! - bytes  0-63: `qs[64]` — 64 bytes of 2-bit packed ternary codes (4 per byte)
//! - bytes 64-65: d (f16 little-endian scale)
//!
//! Ternary encoding: each 2-bit code maps as `0→-1, 1→0, 2→+1`.
//! Weight formula: `w = d * (q_2bit - 1)`
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// TQ2_0 GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Tq2_0GpuKernel;

impl GpuKernel for Tq2_0GpuKernel {
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
            gpu_gemv_tq2_0(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── TQ2_0 block constants ────────────────────────────────────────────────────

/// Weights per TQ2_0 block.
#[cfg(any(feature = "gpu", test))]
const TQ2_0_BLOCK_SIZE: usize = 256;
/// Bytes per TQ2_0 block: 64 (qs) + 2 (d).
#[cfg(any(feature = "gpu", test))]
const TQ2_0_BLOCK_BYTES: usize = 66;

/// Dequantise all TQ2_0 blocks to a flat f32 buffer.
///
/// TQ2_0 layout: 64 bytes of 2-bit ternary codes (4 per byte), then 2-byte FP16 scale.
/// Ternary mapping: code 0 → -1, code 1 → 0, code 2 → +1.
#[cfg(any(feature = "gpu", test))]
fn dequant_tq2_0_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(TQ2_0_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * TQ2_0_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * TQ2_0_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + TQ2_0_BLOCK_BYTES];

            // qs: first 64 bytes, d: bytes[64..66]
            let qs = &block[0..64];
            let d = half::f16::from_le_bytes([block[64], block[65]]).to_f32();

            let weight_base = blk * TQ2_0_BLOCK_SIZE;

            for (i, &byte) in qs.iter().enumerate() {
                let v0 = (byte & 3) as i32 - 1;
                let v1 = ((byte >> 2) & 3) as i32 - 1;
                let v2 = ((byte >> 4) & 3) as i32 - 1;
                let v3 = ((byte >> 6) & 3) as i32 - 1;

                let base = weight_base + i * 4;
                if base < cols {
                    f32_weights[row * cols + base] = d * v0 as f32;
                }
                if base + 1 < cols {
                    f32_weights[row * cols + base + 1] = d * v1 as f32;
                }
                if base + 2 < cols {
                    f32_weights[row * cols + base + 2] = d * v2 as f32;
                }
                if base + 3 < cols {
                    f32_weights[row * cols + base + 3] = d * v3 as f32;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_tq2_0(
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

    let f32_weights = dequant_tq2_0_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "tq2_0-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "tq2_0-input", input);
    let output_buf = create_output_f32(&ctx.device, "tq2_0-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "tq2_0-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_tq2_0"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tq2_0-bgl"),
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
            label: Some("tq2_0-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("tq2_0-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("tq2_0-bg"),
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
            label: Some("tq2_0-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("tq2_0-pass"),
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

    /// Pack 4 × 2-bit ternary codes into one byte.
    fn pack_2bit(v0: u8, v1: u8, v2: u8, v3: u8) -> u8 {
        (v0 & 3) | ((v1 & 3) << 2) | ((v2 & 3) << 4) | ((v3 & 3) << 6)
    }

    fn make_tq2_0_block(scale: f32, qs: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(TQ2_0_BLOCK_BYTES);
        block.extend_from_slice(qs);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block
    }

    #[test]
    fn test_dequant_tq2_0_zero_scale() {
        // d = 0 → all weights = 0 regardless of ternary codes.
        let qs = [pack_2bit(2, 2, 2, 2); 64];
        let block = make_tq2_0_block(0.0, &qs);
        let result = dequant_tq2_0_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "weight[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn test_dequant_tq2_0_all_positive() {
        // All 2-bit codes = 2 → ternary +1 → weight = d * 1.0 = 1.0
        let qs = [pack_2bit(2, 2, 2, 2); 64];
        let block = make_tq2_0_block(1.0, &qs);
        let result = dequant_tq2_0_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-3, "weight[{i}] = {v}, expected 1.0");
        }
    }

    #[test]
    fn test_dequant_tq2_0_all_negative() {
        // All 2-bit codes = 0 → ternary -1 → weight = d * (-1.0) = -1.0
        let qs = [0x00u8; 64];
        let block = make_tq2_0_block(1.0, &qs);
        let result = dequant_tq2_0_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-3,
                "weight[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_tq2_0_mixed() {
        // Byte 0: codes 0,1,2,0 → ternary -1,0,+1,-1 → d*[-1,0,1,-1]
        // d = 2.0
        let mut qs = [0x55u8; 64]; // code 1 → ternary 0
        qs[0] = pack_2bit(0, 1, 2, 0);
        let block = make_tq2_0_block(2.0, &qs);
        let result = dequant_tq2_0_to_f32(&block, 1, 256).expect("dequant");
        assert!((result[0] - (-2.0)).abs() < 1e-3, "got {}", result[0]);
        assert!(result[1].abs() < 1e-3, "got {}", result[1]);
        assert!((result[2] - 2.0).abs() < 1e-3, "got {}", result[2]);
        assert!((result[3] - (-2.0)).abs() < 1e-3, "got {}", result[3]);
        // Remaining: code 1 → ternary 0 → 0.0
        for &v in &result[4..] {
            assert!(v.abs() < 1e-5, "tail weight expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_tq2_0_too_small() {
        assert!(
            dequant_tq2_0_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_tq2_0_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Tq2_0GpuKernel;
    }

    /// End-to-end GPU GEMV: dequant+dot must match within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_tq2_0_matches_cpu_reference() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 32;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * TQ2_0_BLOCK_BYTES);
        for r in 0..rows {
            let mut qs = [0u8; 64];
            for (i, byte) in qs.iter_mut().enumerate() {
                let v0 = ((r + i) % 3) as u8;
                let v1 = ((r + i + 1) % 3) as u8;
                let v2 = ((r + i + 2) % 3) as u8;
                let v3 = (i % 2) as u8;
                *byte = pack_2bit(v0, v1, v2, v3);
            }
            let block = make_tq2_0_block(0.5 + r as f32 * 0.01, &qs);
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let f32_weights = dequant_tq2_0_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Tq2_0GpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut result, rows, cols)
            .expect("GPU GEMV TQ2_0");

        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

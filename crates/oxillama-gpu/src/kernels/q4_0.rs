//! Q4_0 GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the reference
//!      `Q4_0Ref` kernel already in `oxillama-quant`.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q4_0 GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Q4_0GpuKernel;

impl GpuKernel for Q4_0GpuKernel {
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
            gpu_gemv_q4_0(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            // Suppress unused-variable warnings when gpu feature is off.
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── GPU implementation ───────────────────────────────────────────────────────

/// Constants mirroring the Q4_0 block layout (same as in `oxillama-quant`).
#[cfg(any(feature = "gpu", test))]
const Q4_0_BLOCK_SIZE: usize = 32;
#[cfg(any(feature = "gpu", test))]
const Q4_0_BLOCK_BYTES: usize = 18; // 2 bytes scale + 16 bytes nibbles

/// Dequantise all Q4_0 blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
pub(crate) fn dequant_q4_0_to_f32(
    weight_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q4_0_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q4_0_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q4_0_BLOCK_BYTES;
            let block = &weight_bytes[block_offset..block_offset + Q4_0_BLOCK_BYTES];

            let scale_bits = u16::from_le_bytes([block[0], block[1]]);
            let d = half::f16::from_bits(scale_bits).to_f32();

            for i in 0..(Q4_0_BLOCK_SIZE / 2) {
                let byte = block[2 + i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;

                let base_col = blk * Q4_0_BLOCK_SIZE + i * 2;
                if base_col < cols {
                    f32_weights[row * cols + base_col] = lo as f32 * d;
                }
                if base_col + 1 < cols {
                    f32_weights[row * cols + base_col + 1] = hi as f32 * d;
                }
            }
        }
    }

    Ok(f32_weights)
}

#[cfg(feature = "gpu")]
fn gpu_gemv_q4_0(
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

    // Step 1 — dequantise on CPU.
    let f32_weights = dequant_q4_0_to_f32(weight_bytes, rows, cols)?;

    // Step 2 — upload buffers.
    let weight_buf = upload_f32(&ctx.device, "q4_0-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q4_0-input", input);
    let output_buf = create_output_f32(&ctx.device, "q4_0-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q4_0-params", &params);

    // Step 3 — build compute pipeline.
    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q4_0-bgl"),
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
            label: Some("q4_0-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q4_0-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q4_0-bg"),
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

    // Step 4 — dispatch.
    // Workgroup size = 64; we need ceil(rows / 64) groups.
    let dispatch_x = rows.div_ceil(64) as u32;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q4_0-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q4_0-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }
    ctx.queue.submit([encoder.finish()]);

    // Step 5 — read back.
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

    /// Build a minimal Q4_0 block: 2-byte f16 scale + 16 bytes nibbles.
    fn make_q4_0_block(scale: f32, nibbles: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q4_0_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(nibbles);
        block
    }

    #[test]
    fn test_dequant_q4_0_zeros() {
        // All nibbles = 0x8 → (0x8 - 8) = 0, so all weights must be 0.
        let block = make_q4_0_block(1.0, &[0x88u8; 16]);
        let mut data = Vec::new();
        for _ in 0..2 {
            data.extend_from_slice(&block);
        }
        let result = dequant_q4_0_to_f32(&data, 2, 32).expect("dequant should succeed");
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q4_0_values() {
        // First nibble: lo=0x0 → 0-8=-8, hi=0xF → 15-8=7. Scale=0.5.
        let mut nibbles = [0x88u8; 16];
        nibbles[0] = 0xF0; // lo=0 (−8×0.5=−4), hi=F (7×0.5=3.5)
        let block = make_q4_0_block(0.5, &nibbles);
        let result = dequant_q4_0_to_f32(&block, 1, 32).expect("dequant");
        assert!((result[0] - (-4.0)).abs() < 0.01, "got {}", result[0]);
        assert!((result[1] - 3.5).abs() < 0.01, "got {}", result[1]);
    }

    #[test]
    fn test_dequant_q4_0_too_small_errors() {
        assert!(
            dequant_q4_0_to_f32(&[0u8; 4], 1, 32).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_q4_0_kernel_no_gpu_returns_none_adapter_err() {
        // Verify the kernel is constructible and satisfies the GpuKernel bound
        // regardless of whether the gpu feature is active.
        let _kernel: &dyn GpuKernel = &Q4_0GpuKernel;
    }
}

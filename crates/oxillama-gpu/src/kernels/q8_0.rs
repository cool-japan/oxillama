//! Q8_0 GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q8_0 block
//!      format: 2-byte f16 scale + 32 × i8 quantised values = 34 bytes/block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q8_0 GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Q8_0GpuKernel;

impl GpuKernel for Q8_0GpuKernel {
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
            gpu_gemv_q8_0(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q8_0 block constants ─────────────────────────────────────────────────────

/// Weights per Q8_0 block.
#[cfg(any(feature = "gpu", test))]
const Q8_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q8_0 block: 2 (scale) + 32 (i8 values).
#[cfg(any(feature = "gpu", test))]
const Q8_0_BLOCK_BYTES: usize = 34;

/// Dequantise all Q8_0 blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_q8_0_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q8_0_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q8_0_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q8_0_BLOCK_BYTES;
            let block = &weight_bytes[block_offset..block_offset + Q8_0_BLOCK_BYTES];

            let scale_bits = u16::from_le_bytes([block[0], block[1]]);
            let d = half::f16::from_bits(scale_bits).to_f32();

            for i in 0..Q8_0_BLOCK_SIZE {
                let col = blk * Q8_0_BLOCK_SIZE + i;
                if col < cols {
                    let q = block[2 + i] as i8;
                    f32_weights[row * cols + col] = q as f32 * d;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q8_0(
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

    // Fast path: f16 accumulator when adapter supports SHADER_F16.
    if crate::kernels::supports_f16(ctx) {
        use crate::kernels::f16_accumulator::{dequant_q8_0_to_f16, f16_gemv};
        let f16_weights = dequant_q8_0_to_f16(weight_bytes, rows, cols)?;
        return f16_gemv(ctx, &f16_weights, input, output, rows, cols);
    }

    // Step 1 — dequantise on CPU.
    let f32_weights = dequant_q8_0_to_f32(weight_bytes, rows, cols)?;

    // Step 2 — upload buffers.
    let weight_buf = upload_f32(&ctx.device, "q8_0-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q8_0-input", input);
    let output_buf = create_output_f32(&ctx.device, "q8_0-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q8_0-params", &params);

    // Step 3 — build compute pipeline.
    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q8_0"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q8_0-bgl"),
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
            label: Some("q8_0-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q8_0-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q8_0-bg"),
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
    let dispatch_x = rows.div_ceil(64) as u32;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q8_0-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q8_0-pass"),
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

    fn make_q8_0_block(scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_0_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &v in values {
            block.push(v as u8);
        }
        block
    }

    #[test]
    fn test_dequant_q8_0_zeros() {
        let block = make_q8_0_block(1.0, &[0i8; 32]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let result = dequant_q8_0_to_f32(&data, 2, 32).expect("dequant");
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q8_0_values() {
        let mut values = [0i8; 32];
        values[0] = 10;
        values[1] = -5;
        let block = make_q8_0_block(0.5, &values);
        let result = dequant_q8_0_to_f32(&block, 1, 32).expect("dequant");
        assert!((result[0] - 5.0).abs() < 0.01, "got {}", result[0]);
        assert!((result[1] - (-2.5)).abs() < 0.01, "got {}", result[1]);
    }

    #[test]
    fn test_dequant_q8_0_too_small_errors() {
        assert!(
            dequant_q8_0_to_f32(&[0u8; 4], 1, 32).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_q8_0_kernel_constructible() {
        let _kernel: &dyn GpuKernel = &Q8_0GpuKernel;
    }

    /// Verify that `dequant_q8_0_to_f16` produces the same element count as
    /// `dequant_q8_0_to_f32`, confirming the f16 path covers every weight.
    #[test]
    fn test_f16_path_element_count_matches_f32() {
        use crate::kernels::f16_accumulator::dequant_q8_0_to_f16;

        let block = make_q8_0_block(1.0, &[0i8; 32]);
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&block);
        }
        let (rows, cols) = (4, 32);
        let f16_count = dequant_q8_0_to_f16(&data, rows, cols)
            .expect("f16 dequant")
            .len();
        let f32_count = dequant_q8_0_to_f32(&data, rows, cols)
            .expect("f32 dequant")
            .len();
        assert_eq!(
            f16_count, f32_count,
            "f16 and f32 paths must produce same element count"
        );
    }
}

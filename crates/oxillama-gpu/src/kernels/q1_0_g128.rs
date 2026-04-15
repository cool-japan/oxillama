//! Q1_0_G128 GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q1_0_G128
//!      reference kernel from `oxillama-quant`.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! Q1_0_G128 block format (18 bytes per 128 weights):
//!   - 2 bytes: FP16 scale (d)
//!   - 16 bytes: 128 sign bits (bit=1 → +d, bit=0 → −d)
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q1_0_G128 GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
#[allow(non_camel_case_types)]
pub struct Q1_0_G128GpuKernel;

impl GpuKernel for Q1_0_G128GpuKernel {
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
            gpu_gemv_q1_0_g128(ctx, weight_bytes, input, output, rows, cols)
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

/// Block size for Q1_0_G128: 128 weights per block.
#[cfg(any(feature = "gpu", test))]
const Q1_0_G128_BLOCK_SIZE: usize = 128;
/// Bytes per Q1_0_G128 block: 2 (FP16 scale) + 16 (128 sign bits) = 18 bytes.
#[cfg(any(feature = "gpu", test))]
const Q1_0_G128_BLOCK_BYTES: usize = 18;

/// Dequantise all Q1_0_G128 blocks to a flat f32 buffer.
///
/// Each block: 2-byte FP16 scale `d`, then 16 bytes of sign bits.
/// bit=1 → +d, bit=0 → −d.
#[cfg(any(feature = "gpu", test))]
fn dequant_q1_0_g128_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q1_0_G128_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q1_0_G128_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q1_0_G128_BLOCK_BYTES;
            let block = &weight_bytes[block_offset..block_offset + Q1_0_G128_BLOCK_BYTES];

            let scale_bits = u16::from_le_bytes([block[0], block[1]]);
            let d = half::f16::from_bits(scale_bits).to_f32();

            // 128 sign bits packed in 16 bytes (little-endian bit order)
            for byte_idx in 0..16 {
                let byte = block[2 + byte_idx];
                for bit_idx in 0..8u32 {
                    let weight_idx = blk * Q1_0_G128_BLOCK_SIZE + byte_idx * 8 + bit_idx as usize;
                    if weight_idx < cols {
                        let bit = (byte >> bit_idx) & 1;
                        f32_weights[row * cols + weight_idx] = if bit == 1 { d } else { -d };
                    }
                }
            }
        }
    }

    Ok(f32_weights)
}

#[cfg(feature = "gpu")]
fn gpu_gemv_q1_0_g128(
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
    let f32_weights = dequant_q1_0_g128_to_f32(weight_bytes, rows, cols)?;

    // Step 2 — upload buffers.
    let weight_buf = upload_f32(&ctx.device, "q1_0_g128-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q1_0_g128-input", input);
    let output_buf = create_output_f32(&ctx.device, "q1_0_g128-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q1_0_g128-params", &params);

    // Step 3 — build compute pipeline.
    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q1_0_g128-bgl"),
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
            label: Some("q1_0_g128-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q1_0_g128-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q1_0_g128-bg"),
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
            label: Some("q1_0_g128-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q1_0_g128-pass"),
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

    /// Build a minimal Q1_0_G128 block: 2-byte FP16 scale + 16 bytes sign bits.
    fn make_q1_block(scale: f32, bits: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q1_0_G128_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(bits);
        block
    }

    #[test]
    fn test_block_constants() {
        assert_eq!(Q1_0_G128_BLOCK_SIZE, 128);
        assert_eq!(Q1_0_G128_BLOCK_BYTES, 18);
    }

    #[test]
    fn test_dequant_all_positive() {
        // All bits = 1 → all weights = +d
        let block = make_q1_block(2.0, &[0xFF; 16]);
        let result = dequant_q1_0_g128_to_f32(&block, 1, 128).expect("dequant should succeed");
        for &v in &result {
            assert!((v - 2.0).abs() < 0.01, "expected +2.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_all_negative() {
        // All bits = 0 → all weights = -d
        let block = make_q1_block(3.0, &[0x00; 16]);
        let result = dequant_q1_0_g128_to_f32(&block, 1, 128).expect("dequant should succeed");
        for &v in &result {
            assert!((v - (-3.0)).abs() < 0.01, "expected -3.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_mixed_bits() {
        // First byte = 0b10101010: bits 1,3,5,7 are set → +d at indices 1,3,5,7
        //                          bits 0,2,4,6 are clear → -d at indices 0,2,4,6
        let mut bits = [0x00u8; 16];
        bits[0] = 0xAA; // 0b10101010
        let block = make_q1_block(1.0, &bits);
        let result = dequant_q1_0_g128_to_f32(&block, 1, 128).expect("dequant should succeed");

        // Index 0: bit 0 = 0 → -1.0
        assert!(
            (result[0] - (-1.0)).abs() < 0.01,
            "idx 0: got {}",
            result[0]
        );
        // Index 1: bit 1 = 1 → +1.0
        assert!((result[1] - 1.0).abs() < 0.01, "idx 1: got {}", result[1]);
        // Index 2: bit 2 = 0 → -1.0
        assert!(
            (result[2] - (-1.0)).abs() < 0.01,
            "idx 2: got {}",
            result[2]
        );
        // Index 3: bit 3 = 1 → +1.0
        assert!((result[3] - 1.0).abs() < 0.01, "idx 3: got {}", result[3]);

        // Remaining bytes are 0 → -1.0
        for &v in &result[8..128] {
            assert!((v - (-1.0)).abs() < 0.01, "expected -1.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_multi_row() {
        // Two rows, each 128 columns
        let block_pos = make_q1_block(1.5, &[0xFF; 16]); // all +1.5
        let block_neg = make_q1_block(1.5, &[0x00; 16]); // all -1.5
        let mut data = Vec::new();
        data.extend_from_slice(&block_pos);
        data.extend_from_slice(&block_neg);

        let result = dequant_q1_0_g128_to_f32(&data, 2, 128).expect("dequant");
        // Row 0: all +1.5
        for &v in &result[..128] {
            assert!((v - 1.5).abs() < 0.01, "row 0: expected +1.5, got {v}");
        }
        // Row 1: all -1.5
        for &v in &result[128..256] {
            assert!((v - (-1.5)).abs() < 0.01, "row 1: expected -1.5, got {v}");
        }
    }

    #[test]
    fn test_dequant_too_small_errors() {
        assert!(
            dequant_q1_0_g128_to_f32(&[0u8; 4], 1, 128).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_q1_0_g128_kernel_trait_object() {
        let _kernel: &dyn GpuKernel = &Q1_0_G128GpuKernel;
    }
}

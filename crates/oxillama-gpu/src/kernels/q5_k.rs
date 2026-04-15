//! Q5_K GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q5_K block
//!      format: 256 weights per 176-byte super-block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q5_K GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
#[allow(non_camel_case_types)]
pub struct Q5_KGpuKernel;

impl GpuKernel for Q5_KGpuKernel {
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
            gpu_gemv_q5_k(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q5_K block constants ─────────────────────────────────────────────────────

/// Weights per Q5_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q5_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q5_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q5_K_BLOCK_BYTES: usize = 176;
/// Number of sub-blocks inside each Q5_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q5_K_NUM_SUB_BLOCKS: usize = 8;
/// Weights per sub-block.
#[cfg(any(feature = "gpu", test))]
const Q5_K_SUB_BLOCK_SIZE: usize = 32;

/// Extract the 6-bit scale and min values for each of the 8 sub-blocks
/// from the 12-byte packed `scales_and_mins` array.
///
/// Same packing scheme as Q4_K.
#[cfg(any(feature = "gpu", test))]
fn unpack_q5_k_scales(scales_and_mins: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for i in 0..4 {
        scales[i] = scales_and_mins[i] & 0x3F;
    }
    for i in 0..4 {
        mins[i] = scales_and_mins[4 + i] & 0x3F;
    }
    for i in 0..4 {
        scales[4 + i] = (scales_and_mins[8 + i] & 0x0F) | ((scales_and_mins[i] >> 6) << 4);
    }
    for i in 0..4 {
        mins[4 + i] = (scales_and_mins[8 + i] >> 4) | ((scales_and_mins[4 + i] >> 6) << 4);
    }

    (scales, mins)
}

/// Dequantise all Q5_K blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_q5_k_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q5_K_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q5_K_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * Q5_K_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + Q5_K_BLOCK_BYTES];

            // Bytes 0-1: d (f16 super-block scale)
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            // Bytes 2-3: dmin (f16 super-block minimum)
            let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            // Bytes 4-15: scales_and_mins (12 bytes packed)
            let scales_and_mins = &block[4..16];
            let (sc, m) = unpack_q5_k_scales(scales_and_mins);

            // Bytes 16-143: qs (128 bytes, low 4-bit nibbles for 256 values)
            let qs = &block[16..144];
            // Bytes 144-175: qh (32 bytes, 1 high bit per value)
            let qh = &block[144..176];

            for j in 0..Q5_K_NUM_SUB_BLOCKS {
                let scale_val = d * sc[j] as f32;
                let min_val = dmin * m[j] as f32;

                for k in 0..Q5_K_SUB_BLOCK_SIZE {
                    let idx = j * Q5_K_SUB_BLOCK_SIZE + k;
                    let col = blk * Q5_K_BLOCK_SIZE + idx;
                    if col >= cols {
                        break;
                    }

                    // Low 4 bits from qs.
                    let byte_idx = idx / 2;
                    let lo_nibble = if idx % 2 == 0 {
                        qs[byte_idx] & 0x0F
                    } else {
                        (qs[byte_idx] >> 4) & 0x0F
                    };

                    // High bit from qh.
                    let qh_byte = qh[idx / 8];
                    let qh_bit = (qh_byte >> (idx % 8)) & 1;

                    let quant_val = lo_nibble as u32 | ((qh_bit as u32) << 4);

                    f32_weights[row * cols + col] = scale_val * quant_val as f32 - min_val;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q5_k(
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

    let f32_weights = dequant_q5_k_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "q5_k-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q5_k-input", input);
    let output_buf = create_output_f32(&ctx.device, "q5_k-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q5_k-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q5_k"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q5_k-bgl"),
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
            label: Some("q5_k-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q5_k-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q5_k-bg"),
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
            label: Some("q5_k-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q5_k-pass"),
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

    /// Build a minimal Q5_K super-block (176 bytes) for testing.
    fn make_q5_k_block(
        d: f32,
        dmin: f32,
        scales: &[u8; 8],
        mins: &[u8; 8],
        qs: &[u8; 128],
        qh: &[u8; 32],
    ) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q5_K_BLOCK_BYTES);

        let d_bits = half::f16::from_f32(d).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());

        let dmin_bits = half::f16::from_f32(dmin).to_bits();
        block.extend_from_slice(&dmin_bits.to_le_bytes());

        // Pack scales_and_mins (12 bytes)
        let mut packed = [0u8; 12];
        for i in 0..4 {
            packed[i] = (scales[i] & 0x3F) | ((scales[4 + i] >> 4) << 6);
        }
        for i in 0..4 {
            packed[4 + i] = (mins[i] & 0x3F) | ((mins[4 + i] >> 4) << 6);
        }
        for i in 0..4 {
            packed[8 + i] = (scales[4 + i] & 0x0F) | ((mins[4 + i] & 0x0F) << 4);
        }
        block.extend_from_slice(&packed);

        block.extend_from_slice(qs);
        block.extend_from_slice(qh);

        block
    }

    #[test]
    fn test_dequant_q5_k_zeros() {
        let block = make_q5_k_block(1.0, 1.0, &[0; 8], &[0; 8], &[0; 128], &[0; 32]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let result = dequant_q5_k_to_f32(&data, 2, 256).expect("dequant should succeed");
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q5_k_values() {
        // Sub-block 0: scale=2, min=1; d=0.5, dmin=0.25
        // nibble[0] lo = 5, qh bit 0 = 1 → quant = 5 | (1<<4) = 21
        // weight = 0.5*2*21 - 0.25*1 = 21.0 - 0.25 = 20.75
        let mut scales = [0u8; 8];
        scales[0] = 2;
        let mut mins = [0u8; 8];
        mins[0] = 1;
        let mut qs = [0u8; 128];
        qs[0] = 0x05; // nibble[0] lo = 5, nibble[1] hi = 0
        let mut qh = [0u8; 32];
        qh[0] = 0x01; // bit 0 set → qh_bit for idx=0 is 1

        let block = make_q5_k_block(0.5, 0.25, &scales, &mins, &qs, &qh);
        let result = dequant_q5_k_to_f32(&block, 1, 256).expect("dequant");

        let expected_0 = 0.5 * 2.0 * 21.0 - 0.25 * 1.0; // 20.75
        assert!(
            (result[0] - expected_0).abs() < 0.01,
            "got {}, expected {expected_0}",
            result[0]
        );

        // nibble[1] hi = 0, qh bit 1 = 0 → quant = 0
        // weight = 0.5*2*0 - 0.25*1 = -0.25
        let expected_1 = 0.5 * 2.0 * 0.0 - 0.25 * 1.0;
        assert!(
            (result[1] - expected_1).abs() < 0.01,
            "got {}, expected {expected_1}",
            result[1]
        );
    }

    #[test]
    fn test_dequant_q5_k_too_small() {
        assert!(
            dequant_q5_k_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_q5_k_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Q5_KGpuKernel;
    }
}

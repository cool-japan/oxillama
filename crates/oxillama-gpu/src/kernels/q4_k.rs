//! Q4_K GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q4_K block
//!      format: 256 weights per 144-byte super-block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q4_K GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
#[allow(non_camel_case_types)]
pub struct Q4_KGpuKernel;

impl GpuKernel for Q4_KGpuKernel {
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
            gpu_gemv_q4_k(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q4_K block constants ─────────────────────────────────────────────────────

/// Weights per Q4_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q4_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q4_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q4_K_BLOCK_BYTES: usize = 144;
/// Number of sub-blocks inside each Q4_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q4_K_NUM_SUB_BLOCKS: usize = 8;
/// Weights per sub-block.
#[cfg(any(feature = "gpu", test))]
const Q4_K_SUB_BLOCK_SIZE: usize = 32;

/// Extract the 6-bit scale and min values for each of the 8 sub-blocks
/// from the 12-byte packed `scales_and_mins` array.
///
/// Returns `(scales[8], mins[8])` as `u8` values (0..63).
#[cfg(any(feature = "gpu", test))]
fn unpack_q4_k_scales(scales_and_mins: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    // Low 4 bits of scales[0..4] are in bytes 0..4 lower nibble.
    // Low 4 bits of mins[0..4]   are in bytes 0..4 upper nibble.
    // Low 4 bits of scales[4..8] are in bytes 4..8 lower nibble.
    // Low 4 bits of mins[4..8]   are in bytes 4..8 upper nibble.
    for i in 0..4 {
        scales[i] = scales_and_mins[i] & 0x3F;
        mins[i] = (scales_and_mins[i] >> 6) | ((scales_and_mins[i + 4] >> 4) & 0x0C);
    }

    // Wait — the packing scheme from the spec is:
    // Low 4 bits of scales 0-3: bytes[0..4] (lower nibble of each)
    //   Actually, let me re-read the spec more carefully.
    //
    // The llama.cpp Q4_K scale packing:
    //   bytes 0..3:  low 6 bits → scales[0..4] lower 6 bits
    //   Actually, the standard packing is:
    //     scales[i] = (bytes[i] & 0x3F)           for i < 4
    //     mins[i]   = (bytes[4+i] & 0x3F)         for i < 4
    //     scales[4+i] = (bytes[8+i] & 0x0F) | ((bytes[i] >> 6) << 4)   for i < 4
    //     mins[4+i]   = (bytes[8+i] >> 4)   | ((bytes[4+i] >> 6) << 4) for i < 4

    // Reset and redo properly following llama.cpp reference:
    scales = [0u8; 8];
    mins = [0u8; 8];

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

/// Dequantise all Q4_K blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_q4_k_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q4_K_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q4_K_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * Q4_K_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + Q4_K_BLOCK_BYTES];

            // Bytes 0-1: d (f16 super-block scale)
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            // Bytes 2-3: dmin (f16 super-block minimum)
            let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            // Bytes 4-15: scales_and_mins (12 bytes packed)
            let scales_and_mins = &block[4..16];
            let (sc, m) = unpack_q4_k_scales(scales_and_mins);

            // Bytes 16-143: qs (128 bytes, 4-bit nibbles for 256 values)
            let qs = &block[16..144];

            for j in 0..Q4_K_NUM_SUB_BLOCKS {
                let scale_val = d * sc[j] as f32;
                let min_val = dmin * m[j] as f32;

                for k in 0..Q4_K_SUB_BLOCK_SIZE {
                    let idx = j * Q4_K_SUB_BLOCK_SIZE + k;
                    let col = blk * Q4_K_BLOCK_SIZE + idx;
                    if col >= cols {
                        break;
                    }

                    // Each byte in qs holds two 4-bit nibbles.
                    let byte_idx = idx / 2;
                    let nibble = if idx.is_multiple_of(2) {
                        qs[byte_idx] & 0x0F
                    } else {
                        (qs[byte_idx] >> 4) & 0x0F
                    };

                    f32_weights[row * cols + col] = scale_val * nibble as f32 - min_val;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q4_k(
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

    let f32_weights = dequant_q4_k_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "q4_k-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q4_k-input", input);
    let output_buf = create_output_f32(&ctx.device, "q4_k-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q4_k-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q4_k"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q4_k-bgl"),
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
            label: Some("q4_k-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q4_k-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q4_k-bg"),
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
            label: Some("q4_k-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q4_k-pass"),
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

    /// Build a minimal Q4_K super-block (144 bytes) for testing.
    ///
    /// `d` and `dmin` are the f16 super-block scale and minimum.
    /// `scales` and `mins` are the 8 per-sub-block 6-bit values.
    /// `qs` is the 128-byte nibble array.
    fn make_q4_k_block(
        d: f32,
        dmin: f32,
        scales: &[u8; 8],
        mins: &[u8; 8],
        qs: &[u8; 128],
    ) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q4_K_BLOCK_BYTES);

        // Bytes 0-1: d (f16)
        let d_bits = half::f16::from_f32(d).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());

        // Bytes 2-3: dmin (f16)
        let dmin_bits = half::f16::from_f32(dmin).to_bits();
        block.extend_from_slice(&dmin_bits.to_le_bytes());

        // Bytes 4-15: scales_and_mins (12 bytes packed)
        let mut packed = [0u8; 12];
        // Low 6 bits of scales[0..4] → bytes[0..4]
        for i in 0..4 {
            packed[i] = (scales[i] & 0x3F) | ((scales[4 + i] >> 4) << 6);
        }
        // Low 6 bits of mins[0..4] → bytes[4..8]
        for i in 0..4 {
            packed[4 + i] = (mins[i] & 0x3F) | ((mins[4 + i] >> 4) << 6);
        }
        // Low 4 bits of scales[4..8] and mins[4..8] → bytes[8..12]
        for i in 0..4 {
            packed[8 + i] = (scales[4 + i] & 0x0F) | ((mins[4 + i] & 0x0F) << 4);
        }
        block.extend_from_slice(&packed);

        // Bytes 16-143: qs (128 bytes)
        block.extend_from_slice(qs);

        block
    }

    #[test]
    fn test_dequant_q4_k_zeros() {
        // All nibbles = 0, scales = 0, mins = 0 → all weights = 0.
        let block = make_q4_k_block(1.0, 1.0, &[0; 8], &[0; 8], &[0; 128]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let result = dequant_q4_k_to_f32(&data, 2, 256).expect("dequant should succeed");
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q4_k_values() {
        // Sub-block 0: scale=2, min=1; d=0.5, dmin=0.25
        // nibble[0] = 5 → weight = 0.5*2*5 - 0.25*1 = 5.0 - 0.25 = 4.75
        let mut scales = [0u8; 8];
        scales[0] = 2;
        let mut mins = [0u8; 8];
        mins[0] = 1;
        let mut qs = [0u8; 128];
        qs[0] = 0x05; // nibble[0] lo = 5, nibble[1] hi = 0

        let block = make_q4_k_block(0.5, 0.25, &scales, &mins, &qs);
        let result = dequant_q4_k_to_f32(&block, 1, 256).expect("dequant");

        let expected_0 = 0.5 * 2.0 * 5.0 - 0.25 * 1.0; // 4.75
        assert!(
            (result[0] - expected_0).abs() < 0.01,
            "got {}, expected {expected_0}",
            result[0]
        );

        // nibble[1] = 0 → weight = 0.5*2*0 - 0.25*1 = -0.25
        let expected_1 = 0.5 * 2.0 * 0.0 - 0.25 * 1.0;
        assert!(
            (result[1] - expected_1).abs() < 0.01,
            "got {}, expected {expected_1}",
            result[1]
        );
    }

    #[test]
    fn test_dequant_q4_k_too_small() {
        assert!(
            dequant_q4_k_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_q4_k_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Q4_KGpuKernel;
    }
}

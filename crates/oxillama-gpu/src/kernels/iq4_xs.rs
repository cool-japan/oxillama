//! IQ4_XS GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the IQ4_XS block
//!      format: 256 weights per 136-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! IQ4_XS block layout (136 bytes per 256 weights):
//! - bytes  0-1: d (f16 little-endian delta)
//! - bytes  2-3: scales_h — 2-byte u16 holding 2-bit high parts of the 8 sub-scales
//! - bytes  4-7: scales_l — 4 bytes holding 4-bit low parts of the 8 sub-scales (2 per byte)
//! - bytes 8-135: nibbles — 128 bytes encoding 256 four-bit weight indices
//!
//! Dequantisation:
//!   ls_low  = (scales_l[i/2] >> (4 * (i & 1))) & 0x0F
//!   ls_high = (scales_h_u16 >> (2 * i)) as u8 & 0x03
//!   ls      = ls_low | (ls_high << 4)            [6-bit, 0..63]
//!   ls_signed = ls wrapping_sub 32               [-32..31]
//!   w = d * ls_signed * KVALUES_IQ4NL\[nibble\]
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// IQ4_XS GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Iq4XsGpuKernel;

impl GpuKernel for Iq4XsGpuKernel {
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
            gpu_gemv_iq4_xs(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── IQ4_XS block constants ───────────────────────────────────────────────────

/// Weights per IQ4_XS block.
#[cfg(any(feature = "gpu", test))]
const IQ4_XS_BLOCK_SIZE: usize = 256;
/// Bytes per IQ4_XS block.
#[cfg(any(feature = "gpu", test))]
const IQ4_XS_BLOCK_BYTES: usize = 136;
/// Number of sub-blocks per IQ4_XS block.
#[cfg(any(feature = "gpu", test))]
const IQ4_XS_N_SUPERBLOCKS: usize = 8;
/// Weights per sub-block.
#[cfg(any(feature = "gpu", test))]
const IQ4_XS_SUB_BLOCK_SIZE: usize = IQ4_XS_BLOCK_SIZE / IQ4_XS_N_SUPERBLOCKS; // 32

/// Non-linear 4-bit quantization lookup table (KVALUES_IQ4NL).
///
/// Matches the reference implementation in `oxillama-quant/src/reference/iq_shared.rs`.
#[cfg(any(feature = "gpu", test))]
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Unpack the signed sub-scale for sub-block `i`.
///
/// Returns a value in the range [-32, 31].
#[cfg(any(feature = "gpu", test))]
#[inline]
fn unpack_sub_scale(scales_h_u16: u16, scales_l: &[u8], i: usize) -> i32 {
    let ls_low: u8 = (scales_l[i / 2] >> (4 * (i & 1))) & 0x0F;
    let ls_high: u8 = (scales_h_u16 >> (2 * i)) as u8 & 0x03;
    let ls: u8 = ls_low | (ls_high << 4);
    (ls as i32).wrapping_sub(32)
}

/// Dequantise all IQ4_XS blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq4_xs_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(IQ4_XS_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * IQ4_XS_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * IQ4_XS_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + IQ4_XS_BLOCK_BYTES];

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let scales_h_u16 = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let nibbles = &block[8..136];

            for sub in 0..IQ4_XS_N_SUPERBLOCKS {
                let ls_signed = unpack_sub_scale(scales_h_u16, scales_l, sub);
                let scale = d * ls_signed as f32;

                // 16 nibble-bytes per sub-block → 32 weights.
                let nibble_offset = sub * (IQ4_XS_SUB_BLOCK_SIZE / 2);
                let weight_offset = blk * IQ4_XS_BLOCK_SIZE + sub * IQ4_XS_SUB_BLOCK_SIZE;

                for i in 0..(IQ4_XS_SUB_BLOCK_SIZE / 2) {
                    let byte = nibbles[nibble_offset + i];
                    let lo = (byte & 0x0F) as usize;
                    let hi = ((byte >> 4) & 0x0F) as usize;

                    let col0 = weight_offset + i * 2;
                    let col1 = col0 + 1;

                    if col0 < cols {
                        f32_weights[row * cols + col0] = scale * KVALUES_IQ4NL[lo] as f32;
                    }
                    if col1 < cols {
                        f32_weights[row * cols + col1] = scale * KVALUES_IQ4NL[hi] as f32;
                    }
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_iq4_xs(
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

    let f32_weights = dequant_iq4_xs_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "iq4_xs-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "iq4_xs-input", input);
    let output_buf = create_output_f32(&ctx.device, "iq4_xs-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "iq4_xs-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_iq4_xs"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("iq4_xs-bgl"),
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
            label: Some("iq4_xs-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("iq4_xs-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("iq4_xs-bg"),
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
            label: Some("iq4_xs-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("iq4_xs-pass"),
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

    /// Build an IQ4_XS block (136 bytes) for testing.
    ///
    /// * `scale`     — FP16 delta `d`.
    /// * `sub_scales` — 8 unsigned sub-scale values in [0, 63].
    /// * `nibbles`   — 128-byte nibble data.
    fn make_iq4_xs_block(scale: f32, sub_scales: [u8; 8], nibbles: [u8; 128]) -> Vec<u8> {
        let mut block = vec![0u8; IQ4_XS_BLOCK_BYTES];

        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];

        let mut scales_h_u16: u16 = 0;
        let mut scales_l = [0u8; 4];
        for i in 0..IQ4_XS_N_SUPERBLOCKS {
            let v = sub_scales[i] & 0x3F;
            let ls_low = v & 0x0F;
            let ls_high = (v >> 4) & 0x03;
            if i & 1 == 0 {
                scales_l[i / 2] |= ls_low;
            } else {
                scales_l[i / 2] |= ls_low << 4;
            }
            scales_h_u16 |= (ls_high as u16) << (2 * i);
        }
        let sh = scales_h_u16.to_le_bytes();
        block[2] = sh[0];
        block[3] = sh[1];
        block[4..8].copy_from_slice(&scales_l);
        block[8..136].copy_from_slice(&nibbles);
        block
    }

    #[test]
    fn test_dequant_iq4_xs_zero_scale() {
        // All sub_scales = 32 → ls_signed = 0 → all weights = 0.
        let block = make_iq4_xs_block(1.0, [32u8; 8], [0xAAu8; 128]);
        let result = dequant_iq4_xs_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "weight[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn test_dequant_iq4_xs_unit_subscale() {
        // d=1.0, sub_scale[0]=33 → ls_signed=1.
        // All nibbles = 0x88 → lo=hi=8 → KVALUES[8]=1
        // First 32 weights = 1.0 * 1 * 1 = 1.0; rest = 0.
        let mut sub_scales = [32u8; 8];
        sub_scales[0] = 33;
        let block = make_iq4_xs_block(1.0, sub_scales, [0x88u8; 128]);
        let result = dequant_iq4_xs_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate().take(32) {
            assert!((v - 1.0).abs() < 0.01, "weight[{i}] = {v}, expected 1.0");
        }
        for (i, &v) in result.iter().enumerate().skip(32) {
            assert!(v.abs() < 1e-6, "weight[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn test_dequant_iq4_xs_two_rows() {
        // Row 0: all sub_scales=32 → weights=0. Row 1: all=33 → weights=1.0
        let block0 = make_iq4_xs_block(1.0, [32u8; 8], [0x88u8; 128]);
        let block1 = make_iq4_xs_block(1.0, [33u8; 8], [0x88u8; 128]);
        let mut data = Vec::new();
        data.extend_from_slice(&block0);
        data.extend_from_slice(&block1);
        let result = dequant_iq4_xs_to_f32(&data, 2, 256).expect("dequant");
        for &v in &result[..256] {
            assert!(v.abs() < 1e-6, "row0 weight expected 0, got {v}");
        }
        for (i, &v) in result[256..].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 0.01,
                "row1 weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4_xs_too_small() {
        assert!(
            dequant_iq4_xs_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_iq4_xs_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Iq4XsGpuKernel;
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_iq4_xs_matches_cpu() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return, // skip if no GPU
        };

        // 64 rows × 256 cols: one block per row (136 bytes each).
        let rows = 64;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * IQ4_XS_BLOCK_BYTES);
        for r in 0..rows {
            // Vary sub_scales around 33 (ls_signed=1) for non-zero weights.
            let sub_scales: [u8; 8] = std::array::from_fn(|i| {
                // Scale values centred around 33 in [1..63]
                (((r + i * 7 + 33) % 63) + 1) as u8
            });
            let nibbles: [u8; 128] = std::array::from_fn(|i| ((r * 5 + i * 3 + 7) & 0xFF) as u8);
            let d_val = 0.01 + (r as f32) * 0.001;
            let block = make_iq4_xs_block(d_val, sub_scales, nibbles);
            weight_bytes.extend_from_slice(&block);
        }

        // Varied input vector.
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        // CPU reference: dequant then dot.
        let f32_weights = dequant_iq4_xs_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Iq4XsGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("GPU GEMV IQ4_XS");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

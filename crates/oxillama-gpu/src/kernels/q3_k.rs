//! Q3_K GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q3_K block
//!      format: 256 weights per 110-byte super-block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! Q3_K block layout (110 bytes per 256 weights):
//! - bytes  0-31: hmask (32 bytes) — 1 bit per weight; if set subtract 0, else subtract 4
//! - bytes 32-95: qs (64 bytes) — lower 2 bits of each 3-bit quant (4 per byte)
//! - bytes 96-107: scales (12 bytes) — 16 sub-block scales, 6-bit unsigned packed
//! - bytes 108-109: d (f16 little-endian super-block scale)
//!
//! Q3_K is a symmetric format; no minimum offset.
//! Weight formula: `w = d * scale_i * (q_lo - (hmask_bit ? 0 : 4))`
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q3_K GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
#[allow(non_camel_case_types)]
pub struct Q3_KGpuKernel;

impl GpuKernel for Q3_KGpuKernel {
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
            gpu_gemv_q3_k(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q3_K block constants ─────────────────────────────────────────────────────

/// Weights per Q3_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q3_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q3_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q3_K_BLOCK_BYTES: usize = 110;

/// Decode 16 signed 6-bit scales from the 12-byte packed representation.
///
/// Returns an array of 16 f32 scale values in the range [-32.0, 31.0].
/// Matches the Q3KRef reference implementation exactly.
#[cfg(any(feature = "gpu", test))]
fn decode_q3_k_scales(scales_raw: &[u8]) -> [f32; 16] {
    let mut sc = [0u32; 16];

    // Scales 0..3: lower 6 bits of bytes 0..3
    for j in 0..4 {
        sc[j] = (scales_raw[j] & 0x3F) as u32;
    }
    // Scales 4..7: lower 6 bits of bytes 4..7
    for j in 0..4 {
        sc[4 + j] = (scales_raw[4 + j] & 0x3F) as u32;
    }
    // Scales 8..11: low 4 bits from bytes 8..11, high 2 bits from upper bits of bytes 0..3
    for j in 0..4 {
        let lo = (scales_raw[8 + j] & 0x0F) as u32;
        let hi = ((scales_raw[j] >> 6) & 0x03) as u32;
        sc[8 + j] = lo | (hi << 4);
    }
    // Scales 12..15: high 4 bits from bytes 8..11, high 2 bits from upper bits of bytes 4..7
    for j in 0..4 {
        let lo = ((scales_raw[8 + j] >> 4) & 0x0F) as u32;
        let hi = ((scales_raw[4 + j] >> 6) & 0x03) as u32;
        sc[12 + j] = lo | (hi << 4);
    }

    // Convert to signed: subtract 32 to get range -32..31.
    let mut result = [0.0f32; 16];
    for i in 0..16 {
        result[i] = (sc[i] as i32 - 32) as f32;
    }
    result
}

/// Dequantise all Q3_K blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_q3_k_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q3_K_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q3_K_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * Q3_K_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + Q3_K_BLOCK_BYTES];

            let hmask = &block[0..32];
            let qs = &block[32..96];
            let scales_raw = &block[96..108];
            let d = half::f16::from_bits(u16::from_le_bytes([block[108], block[109]])).to_f32();

            let sc = decode_q3_k_scales(scales_raw);

            // Dequantise following Q3KRef layout:
            // Two groups of 128 weights each. `m` is a rotating bit selector for hmask.
            let mut is: usize = 0; // scale index
            let mut weight_off: usize = 0;
            let mut m: u8 = 1; // hmask bit selector

            for group in 0..2 {
                let qs_base = group * 32;

                for shift in (0..8).step_by(2) {
                    // Two sub-blocks of 16 weights each.
                    for n in 0..2 {
                        let dl = d * sc[is];
                        is += 1;

                        for l in 0..16 {
                            let col = blk * Q3_K_BLOCK_SIZE + weight_off + l;
                            if col < cols {
                                let qs_idx = qs_base + n * 16 + l;
                                let q_lo = ((qs[qs_idx] >> shift) & 3) as i32;
                                let subtract = if hmask[n * 16 + l] & m != 0 { 0 } else { 4 };
                                f32_weights[row * cols + col] = dl * (q_lo - subtract) as f32;
                            }
                        }
                        weight_off += 16;
                    }
                    m <<= 1;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q3_k(
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

    let f32_weights = dequant_q3_k_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "q3_k-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q3_k-input", input);
    let output_buf = create_output_f32(&ctx.device, "q3_k-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q3_k-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q3_k"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q3_k-bgl"),
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
            label: Some("q3_k-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q3_k-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q3_k-bg"),
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
            label: Some("q3_k-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q3_k-pass"),
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

    /// Build a Q3_K block (110 bytes) for testing.
    fn make_q3_k_block(d: f32, scales: &[u8; 12], hmask: &[u8; 32], qs: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q3_K_BLOCK_BYTES);
        block.extend_from_slice(hmask);
        block.extend_from_slice(qs);
        block.extend_from_slice(scales);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block
    }

    #[test]
    fn test_dequant_q3_k_zeros() {
        // d=0 → all weights = 0
        let block = make_q3_k_block(0.0, &[0; 12], &[0; 32], &[0; 64]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let result = dequant_q3_k_to_f32(&data, 2, 256).expect("dequant should succeed");
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q3_k_hmask_set_q3() {
        // hmask all set → subtract 0. All q_lo = 3.
        // All 16 scales = signed +1 (raw 33). Weight = d * 1 * 3 = 3.0
        let hmask = [0xFFu8; 32];
        let qs = [0xFFu8; 64];
        // All 16 scales = signed +1: encoding from reference
        let scales: [u8; 12] = [
            0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0x11, 0x11, 0x11, 0x11,
        ];
        let block = make_q3_k_block(1.0, &scales, &hmask, &qs);
        let result = dequant_q3_k_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 3.0).abs() < 0.01, "weight[{i}] = {v}, expected 3.0");
        }
    }

    #[test]
    fn test_dequant_q3_k_hmask_clear() {
        // hmask all clear → subtract 4. All q_lo = 0.
        // All scales = signed +1. Weight = d * 1 * (0 - 4) = -4.0
        let hmask = [0x00u8; 32];
        let qs = [0x00u8; 64];
        let scales: [u8; 12] = [
            0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0x11, 0x11, 0x11, 0x11,
        ];
        let block = make_q3_k_block(1.0, &scales, &hmask, &qs);
        let result = dequant_q3_k_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-4.0)).abs() < 0.01,
                "weight[{i}] = {v}, expected -4.0"
            );
        }
    }

    #[test]
    fn test_dequant_q3_k_too_small() {
        assert!(
            dequant_q3_k_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_q3_k_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Q3_KGpuKernel;
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q3_k_matches_cpu() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return, // skip if no GPU
        };

        // 64 rows × 256 cols: one block per row (110 bytes each).
        let rows = 64;
        let cols = 256;

        // All 16 scales = signed +1 (raw 33); varied hmask and qs.
        let scales: [u8; 12] = [
            0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0x11, 0x11, 0x11, 0x11,
        ];

        let mut weight_bytes = Vec::with_capacity(rows * Q3_K_BLOCK_BYTES);
        for r in 0..rows {
            let mut hmask = [0u8; 32];
            let mut qs = [0u8; 64];
            for (i, h) in hmask.iter_mut().enumerate() {
                *h = ((r * 7 + i * 3 + 1) & 0xFF) as u8;
            }
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((r * 11 + i * 5 + 3) & 0xFF) as u8;
            }
            let d_val = 0.01 + (r as f32) * 0.001;
            let block = make_q3_k_block(d_val, &scales, &hmask, &qs);
            weight_bytes.extend_from_slice(&block);
        }

        // Varied input vector.
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        // CPU reference: dequant then dot.
        let f32_weights = dequant_q3_k_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Q3_KGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("GPU GEMV Q3_K");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

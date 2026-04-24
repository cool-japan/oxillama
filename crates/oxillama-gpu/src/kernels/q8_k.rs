//! Q8_K GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q8_K block
//!      format: 256 weights per 292-byte super-block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! Q8_K block layout (292 bytes per 256 weights):
//! - bytes  0-3:   d  (f32 little-endian super-block scale)
//! - bytes  4-259: qs (256 bytes) — 256 × signed int8 quantised values
//! - bytes 260-291: bsums (32 bytes) — 16 × int16 block sums (unused in dequant)
//!
//! Weight formula: `w = d * qs[i]` where `qs[i]` is signed int8.
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q8_K GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
#[allow(non_camel_case_types)]
pub struct Q8_KGpuKernel;

impl GpuKernel for Q8_KGpuKernel {
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
            gpu_gemv_q8_k(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q8_K block constants ─────────────────────────────────────────────────────

/// Weights per Q8_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q8_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q8_K super-block: 4 (f32 d) + 256 (int8) + 32 (int16 bsums).
#[cfg(any(feature = "gpu", test))]
const Q8_K_BLOCK_BYTES: usize = 292;

/// Dequantise all Q8_K blocks to a flat f32 buffer.
///
/// Q8_K layout:
/// - Bytes  0-3:   d (f32 scale).
/// - Bytes  4-259: qs (256 × signed int8).
/// - Bytes 260-291: bsums (unused).
///
/// Weight formula: `w = d * qs[i]`.
#[cfg(any(feature = "gpu", test))]
fn dequant_q8_k_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q8_K_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q8_K_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * Q8_K_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + Q8_K_BLOCK_BYTES];

            // f32 scale at bytes 0-3.
            let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            // Signed int8 values at bytes 4-259.
            let qs = &block[4..260];
            // bsums at bytes 260-291 — not needed for dequant.

            for (i, &q) in qs.iter().enumerate() {
                let col = blk * Q8_K_BLOCK_SIZE + i;
                if col < cols {
                    f32_weights[row * cols + col] = d * (q as i8) as f32;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q8_k(
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

    let f32_weights = dequant_q8_k_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "q8_k-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q8_k-input", input);
    let output_buf = create_output_f32(&ctx.device, "q8_k-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q8_k-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q8_k"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q8_k-bgl"),
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
            label: Some("q8_k-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q8_k-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q8_k-bg"),
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
            label: Some("q8_k-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q8_k-pass"),
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

    /// Build a Q8_K block (292 bytes) for testing.
    fn make_q8_k_block(d: f32, qs: &[i8; 256]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_K_BLOCK_BYTES);
        block.extend_from_slice(&d.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        // bsums: 16 × int16 = 32 bytes, zeroed
        block.extend_from_slice(&[0u8; 32]);
        block
    }

    #[test]
    fn test_dequant_q8_k_zeros() {
        let block = make_q8_k_block(0.0, &[0; 256]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let result = dequant_q8_k_to_f32(&data, 2, 256).expect("dequant should succeed");
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q8_k_positive() {
        // d=0.25, all qs=40 → w = 0.25 * 40 = 10.0
        let block = make_q8_k_block(0.25, &[40; 256]);
        let result = dequant_q8_k_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 10.0).abs() < 0.01, "weight[{i}] = {v}, expected 10.0");
        }
    }

    #[test]
    fn test_dequant_q8_k_negative() {
        // d=1.0, all qs=-100 → w = -100.0
        let block = make_q8_k_block(1.0, &[-100; 256]);
        let result = dequant_q8_k_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-100.0)).abs() < 0.01,
                "weight[{i}] = {v}, expected -100.0"
            );
        }
    }

    #[test]
    fn test_dequant_q8_k_too_small() {
        assert!(
            dequant_q8_k_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_q8_k_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Q8_KGpuKernel;
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q8_k_matches_cpu() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return, // skip if no GPU
        };

        // 64 rows × 256 cols: one block per row (292 bytes each).
        let rows = 64;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * Q8_K_BLOCK_BYTES);
        for r in 0..rows {
            let mut qs = [0i8; 256];
            for (i, q) in qs.iter_mut().enumerate() {
                let raw = ((r * 3 + i * 7 + 13) & 0xFF) as i16 - 128;
                *q = raw.clamp(-128, 127) as i8;
            }
            let d_val = 0.01 + (r as f32) * 0.001;
            let block = make_q8_k_block(d_val, &qs);
            weight_bytes.extend_from_slice(&block);
        }

        // Varied input vector.
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        // CPU reference: dequant then dot.
        let f32_weights = dequant_q8_k_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Q8_KGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("GPU GEMV Q8_K");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

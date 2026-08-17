//! Q6_K GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q6_K block
//!      format: 256 weights per 210-byte super-block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q6_K GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
#[allow(non_camel_case_types)]
pub struct Q6_KGpuKernel;

impl GpuKernel for Q6_KGpuKernel {
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
            gpu_gemv_q6_k(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q6_K block constants ─────────────────────────────────────────────────────

/// Weights per Q6_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q6_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q6_K super-block.
#[cfg(any(feature = "gpu", test))]
const Q6_K_BLOCK_BYTES: usize = 210;

/// Dequantise all Q6_K blocks to a flat f32 buffer.
///
/// Q6_K layout (210 bytes per 256 weights):
/// - Bytes   0-127: ql (128 bytes, low 4-bit values)
/// - Bytes 128-191: qh (64 bytes, 2-bit high values)
/// - Bytes 192-207: scales (16 × i8 per-sub-block scales)
/// - Bytes 208-209: d (f16 super-block scale)
///
/// weight = d * scales[j] * (ql_4bit | (qh_2bit << 4) - 32)
#[cfg(any(feature = "gpu", test))]
fn dequant_q6_k_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q6_K_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q6_K_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * Q6_K_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + Q6_K_BLOCK_BYTES];

            // Bytes 0-127: ql (128 bytes)
            let ql = &block[0..128];
            // Bytes 128-191: qh (64 bytes)
            let qh = &block[128..192];
            // Bytes 192-207: scales (16 × i8)
            let scales = &block[192..208];
            // Bytes 208-209: d (f16)
            let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();

            // Upstream `dequantize_row_q6_K` (and this crate's CPU
            // reference, `oxillama_quant::reference::q6_k`) processes the
            // block in TWO groups of 128 weights, each producing four
            // 32-weight quarters `l`, `l+32`, `l+64`, `l+96` per inner index
            // `l` in `0..32`, reading `ql`/`qh` at DIFFERENT byte offsets and
            // bit shifts per quarter, and non-contiguous scale indices
            // (`is`, `is+2`, `is+4`, `is+6` where `is = l/16`).  That is NOT
            // equivalent to a flat `qs[idx/2]` nibble / `qs[idx/4]` 2-bit
            // mapping over a linear 16-weight sub-block index — the previous
            // implementation here used the latter and was caught diverging
            // from the CPU oracle by `tests/cpu_gpu_cross_check.rs` on real
            // hardware.
            for group in 0..2usize {
                let ql_off = group * 64;
                let qh_off = group * 32;
                let sc_off = group * 8;
                let out_off = group * 128;

                for l in 0..32usize {
                    let is = l / 16; // sub-block index within group: 0 or 1

                    let q1 = ((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4))
                        as i32
                        - 32;
                    let q3 =
                        ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4))
                        as i32
                        - 32;

                    let s0 = scales[sc_off + is] as i8 as f32;
                    let s1 = scales[sc_off + is + 2] as i8 as f32;
                    let s2 = scales[sc_off + is + 4] as i8 as f32;
                    let s3 = scales[sc_off + is + 6] as i8 as f32;

                    let cols_out = [
                        (out_off + l, d * s0 * q1 as f32),
                        (out_off + l + 32, d * s1 * q2 as f32),
                        (out_off + l + 64, d * s2 * q3 as f32),
                        (out_off + l + 96, d * s3 * q4 as f32),
                    ];
                    for (local_idx, weight) in cols_out {
                        let col = blk * Q6_K_BLOCK_SIZE + local_idx;
                        if col < cols {
                            f32_weights[row * cols + col] = weight;
                        }
                    }
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q6_k(
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

    let f32_weights = dequant_q6_k_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "q6_k-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q6_k-input", input);
    let output_buf = create_output_f32(&ctx.device, "q6_k-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q6_k-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q6_k"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q6_k-bgl"),
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
            label: Some("q6_k-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q6_k-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q6_k-bg"),
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
            label: Some("q6_k-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q6_k-pass"),
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

    /// Build a minimal Q6_K super-block (210 bytes) for testing.
    fn make_q6_k_block(d: f32, scales: &[i8; 16], ql: &[u8; 128], qh: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q6_K_BLOCK_BYTES);

        // Bytes 0-127: ql
        block.extend_from_slice(ql);
        // Bytes 128-191: qh
        block.extend_from_slice(qh);
        // Bytes 192-207: scales (16 × i8)
        for &s in scales {
            block.push(s as u8);
        }
        // Bytes 208-209: d (f16)
        let d_bits = half::f16::from_f32(d).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());

        block
    }

    #[test]
    fn test_dequant_q6_k_zeros() {
        // All ql=0, qh=0, scales=0 → all weights = d*0*(0-32) = 0 (scale is 0)
        let block = make_q6_k_block(1.0, &[0; 16], &[0; 128], &[0; 64]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let result = dequant_q6_k_to_f32(&data, 2, 256).expect("dequant should succeed");
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q6_k_values() {
        // Sub-block 0, weight 0: ql nibble = 5, qh 2bit = 1
        // quant = 5 | (1<<4) = 21, then -32 = -11
        // weight = d * scale[0] * -11 = 0.5 * 2 * -11 = -11.0
        let mut scales = [0i8; 16];
        scales[0] = 2;
        let mut ql = [0u8; 128];
        ql[0] = 0x05; // nibble[0] lo = 5, nibble[1] hi = 0
        let mut qh = [0u8; 64];
        qh[0] = 0x01; // 2-bit value at position 0 = 1

        let block = make_q6_k_block(0.5, &scales, &ql, &qh);
        let result = dequant_q6_k_to_f32(&block, 1, 256).expect("dequant");

        let expected_0 = 0.5 * 2.0 * (21.0 - 32.0); // -11.0
        assert!(
            (result[0] - expected_0).abs() < 0.01,
            "got {}, expected {expected_0}",
            result[0]
        );

        // Weight 1: ql nibble = 0, qh 2bit at idx=1 → (qh[0] >> 2) & 3 = 0
        // quant = 0 | 0 = 0, then -32 = -32
        // weight = 0.5 * 2 * -32 = -32.0
        let expected_1 = 0.5 * 2.0 * (0.0 - 32.0);
        assert!(
            (result[1] - expected_1).abs() < 0.01,
            "got {}, expected {expected_1}",
            result[1]
        );
    }

    /// Discriminates the old buggy flat `ql[idx/2]`/`qh[idx/4]` mapping from
    /// the correct four-quarter group mapping: for weight index 32 (the
    /// first weight of the SECOND quarter of group 0), the two mappings
    /// read different `ql` bytes even though they agree on the scale index.
    #[test]
    fn test_dequant_q6_k_second_quarter_byte_mapping() {
        let mut scales = [0i8; 16];
        scales[2] = 1; // sub-block 2 → second-quarter scale for l in 0..16
        let mut ql = [0u8; 128];
        ql[32] = 0x0B; // correct source: ql[ql_off + l + 32] for l=0 → weight[32]=11-32=-21
        ql[16] = 0x07; // old buggy source: ql[idx/2] for idx=32 → would give 7-32=-25
        let qh = [0u8; 64];

        let block = make_q6_k_block(1.0, &scales, &ql, &qh);
        let result = dequant_q6_k_to_f32(&block, 1, 256).expect("dequant");

        assert!(
            (result[32] - (-21.0)).abs() < 0.01,
            "weight[32] = {}, expected -21.0",
            result[32]
        );
    }

    #[test]
    fn test_dequant_q6_k_too_small() {
        assert!(
            dequant_q6_k_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_q6_k_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Q6_KGpuKernel;
    }
}

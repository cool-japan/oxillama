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
pub(crate) fn dequant_q5_k_to_f32(
    weight_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> GpuResult<Vec<f32>> {
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

            // Upstream `block_q5_K` is `{ d; dmin; scales[12]; qh[32]; qs[128] }`
            // — `qh` precedes `qs`.  The previous implementation here read
            // them in the opposite order (a second, independent bug from the
            // nibble/qh-bit mapping bug fixed below).
            //
            // Bytes 16-47: qh (32 bytes, 1 high bit per value)
            let qh = &block[16..48];
            // Bytes 48-175: qs (128 bytes, low 4-bit nibbles for 256 values)
            let qs = &block[48..176];

            // Upstream `dequantize_row_q5_K` walks the block in four
            // 64-weight groups.  Within group `g`, the low-nibble half (32
            // weights) takes its 5th bit from `qh[l]` bit `2g`, and the
            // high-nibble half takes its 5th bit from `qh[l]` bit `2g + 1` —
            // so the eight bits of `qh[0]` feed weights 0, 32, 64, 96, 128,
            // 160, 192, 224 (see `oxillama_quant::reference::q5_k`'s module
            // doc, which this must match exactly).  This is *not* the same
            // as reading `qh[idx / 8]` bit `idx % 8` per output index.
            let mut is = 0usize;
            let mut qs_offset = 0usize;
            let mut out_offset = 0usize;

            for group in 0..(Q5_K_NUM_SUB_BLOCKS / 2) {
                let d1 = d * sc[is] as f32;
                let m1 = dmin * m[is] as f32;
                let d2 = d * sc[is + 1] as f32;
                let m2 = dmin * m[is + 1] as f32;

                // Low nibbles → first 32 weights of this group.
                for l in 0..Q5_K_SUB_BLOCK_SIZE {
                    let col = blk * Q5_K_BLOCK_SIZE + out_offset + l;
                    if col < cols {
                        let qh_bit = (qh[l] >> (2 * group)) & 1;
                        let q = ((qs[qs_offset + l] & 0x0F) | (qh_bit << 4)) as f32;
                        f32_weights[row * cols + col] = d1 * q - m1;
                    }
                }

                // High nibbles → next 32 weights of this group.
                for l in 0..Q5_K_SUB_BLOCK_SIZE {
                    let col = blk * Q5_K_BLOCK_SIZE + out_offset + Q5_K_SUB_BLOCK_SIZE + l;
                    if col < cols {
                        let qh_bit = (qh[l] >> (2 * group + 1)) & 1;
                        let q = (((qs[qs_offset + l] >> 4) & 0x0F) | (qh_bit << 4)) as f32;
                        f32_weights[row * cols + col] = d2 * q - m2;
                    }
                }

                is += 2;
                qs_offset += Q5_K_SUB_BLOCK_SIZE;
                out_offset += 2 * Q5_K_SUB_BLOCK_SIZE;
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
    ///
    /// Field order matches upstream `block_q5_K`:
    /// `{ d; dmin; scales[12]; qh[32]; qs[128] }`.
    fn make_q5_k_block(
        d: f32,
        dmin: f32,
        scales: &[u8; 8],
        mins: &[u8; 8],
        qh: &[u8; 32],
        qs: &[u8; 128],
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

        block.extend_from_slice(qh);
        block.extend_from_slice(qs);

        block
    }

    #[test]
    fn test_dequant_q5_k_zeros() {
        let block = make_q5_k_block(1.0, 1.0, &[0; 8], &[0; 8], &[0; 32], &[0; 128]);
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

        let block = make_q5_k_block(0.5, 0.25, &scales, &mins, &qh, &qs);
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

    /// Discriminates the old buggy `qh[idx/8] bit idx%8` mapping from the
    /// correct `qh[l] bit 2*group(+1)` mapping upstream
    /// `dequantize_row_q5_K` uses.  `test_dequant_q5_k_values` above only
    /// exercises `idx` 0 and 1, where both mappings happen to agree — this
    /// test picks an index (32) where they disagree, so it would have failed
    /// against the pre-fix kernel.
    #[test]
    fn test_dequant_q5_k_group_qh_bit_mapping() {
        // Only qh[0] bit 1 is set.  Under the correct mapping that is the
        // 5th-bit source for group 0's HIGH-nibble half at l=0, i.e.
        // weight[32].  Under the old per-index mapping, weight[32] (idx=32)
        // would instead read qh[32/8]=qh[4] bit 32%8=0, which is unset, and
        // stay 0 — so this test distinguishes the two.
        let mut scales = [0u8; 8];
        scales[0] = 1;
        scales[1] = 1;
        let mins = [0u8; 8];
        let qs = [0u8; 128];
        let mut qh = [0u8; 32];
        qh[0] = 0x02;

        let block = make_q5_k_block(1.0, 0.0, &scales, &mins, &qh, &qs);
        let result = dequant_q5_k_to_f32(&block, 1, 256).expect("dequant");

        assert!(
            (result[32] - 16.0).abs() < 0.01,
            "weight[32] = {}, expected 16.0",
            result[32]
        );
        for (i, &v) in result.iter().enumerate() {
            if i == 32 {
                continue;
            }
            assert!(v.abs() < 0.01, "weight[{i}] = {v}, expected 0.0");
        }
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

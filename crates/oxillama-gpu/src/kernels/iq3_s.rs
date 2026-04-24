//! IQ3_S GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the IQ3_S block
//!      format: 256 weights per 110-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! IQ3_S block layout (110 bytes per 256 weights):
//! - bytes  0-1:    d (f16 little-endian scale)
//! - bytes  2-65:   qs\[64\] — 64 bytes: grid base indices (8 per super-block)
//! - bytes  66-73:  qh\[8\] — high bits for grid indices (1 byte per super-block)
//! - bytes  74-105: signs\[32\] — per-group sign masks (4 bytes per super-block)
//! - bytes  106-109: scales\[4\] — per-pair-of-super-blocks nibbles
//!
//! Scale formula: `d * (1 + 2 * nibble)` (different from IQ2/IQ3_XXS!).
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// IQ3_S GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Iq3SGpuKernel;

impl GpuKernel for Iq3SGpuKernel {
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
            gpu_gemv_iq3_s(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── IQ3_S block constants ────────────────────────────────────────────────────

/// Weights per IQ3_S block.
#[cfg(any(feature = "gpu", test))]
const IQ3S_BLOCK_SIZE: usize = 256;
/// Bytes per IQ3_S block: 2 + 64 + 8 + 32 + 4 = 110.
#[cfg(any(feature = "gpu", test))]
const IQ3S_BLOCK_BYTES: usize = 110;
/// Number of super-blocks per IQ3_S block.
#[cfg(any(feature = "gpu", test))]
const IQ3S_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
#[cfg(any(feature = "gpu", test))]
const IQ3S_SUPER_BLOCK_SIZE: usize = IQ3S_BLOCK_SIZE / IQ3S_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block.
#[cfg(any(feature = "gpu", test))]
const IQ3S_GROUPS_PER_SUPER: usize = 4;
/// Byte offset of qs within the block.
#[cfg(any(feature = "gpu", test))]
const IQ3S_QS_OFFSET: usize = 2;
/// Size of qs region (QK_K/4 = 64 bytes).
#[cfg(any(feature = "gpu", test))]
const IQ3S_QS_BYTES: usize = 64;
/// Byte offset of qh within the block.
#[cfg(any(feature = "gpu", test))]
const IQ3S_QH_OFFSET: usize = 66;
/// Size of qh region (QK_K/32 = 8 bytes).
#[cfg(any(feature = "gpu", test))]
const IQ3S_QH_BYTES: usize = 8;
/// Byte offset of signs within the block.
#[cfg(any(feature = "gpu", test))]
const IQ3S_SIGNS_OFFSET: usize = 74;
/// Size of signs region (QK_K/8 = 32 bytes).
#[cfg(any(feature = "gpu", test))]
const IQ3S_SIGNS_BYTES: usize = 32;
/// Byte offset of scales within the block.
#[cfg(any(feature = "gpu", test))]
const IQ3S_SCALES_OFFSET: usize = 106;

/// Dequantise a single IQ3_S super-block into `out` (32 floats).
///
/// `qs_sb` — the 8-byte qs slice for this super-block (already sliced to [8*ib32..8*ib32+8]).
/// `qh_byte` — the qh byte for this super-block (provides high bits for grid indices).
/// `signs_sb` — 4-byte slice: one sign-mask byte per group.
/// `db` — the scale value for this super-block.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq3s_superblock(
    qs_sb: &[u8],
    qh_byte: u8,
    signs_sb: &[u8],
    db: f32,
    out: &mut [f32],
    col_base: usize,
    cols: usize,
) {
    use super::iq_grids::{IQ3S_GRID, KMASK_IQ2XS};

    let qh = qh_byte as usize;

    for l in 0..IQ3S_GROUPS_PER_SUPER {
        // 9-bit grid indices for the two entries in this group.
        let qs0 = qs_sb[2 * l] as usize;
        let qs1 = qs_sb[2 * l + 1] as usize;

        let shift0 = 8usize.saturating_sub(2 * l);
        let shift1 = 7usize.saturating_sub(2 * l);
        let idx1 = qs0 | ((qh << shift0) & 256);
        let idx2 = qs1 | ((qh << shift1) & 256);

        let grid1: [u8; 4] = IQ3S_GRID[idx1].to_le_bytes();
        let grid2: [u8; 4] = IQ3S_GRID[idx2].to_le_bytes();

        let sign_byte = signs_sb[l];

        let group_base = l * 8;
        for j in 0..4 {
            let col0 = col_base + group_base + j;
            if col0 < cols {
                let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                out[group_base + j] = db * grid1[j] as f32 * sign1;
            }

            let col1 = col_base + group_base + j + 4;
            if col1 < cols {
                let sign2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                out[group_base + j + 4] = db * grid2[j] as f32 * sign2;
            }
        }
    }
}

/// Dequantise all IQ3_S blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq3_s_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(IQ3S_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * IQ3S_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * IQ3S_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + IQ3S_BLOCK_BYTES];

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[IQ3S_QS_OFFSET..IQ3S_QS_OFFSET + IQ3S_QS_BYTES];
            let qh = &block[IQ3S_QH_OFFSET..IQ3S_QH_OFFSET + IQ3S_QH_BYTES];
            let signs = &block[IQ3S_SIGNS_OFFSET..IQ3S_SIGNS_OFFSET + IQ3S_SIGNS_BYTES];
            let scales = &block[IQ3S_SCALES_OFFSET..IQ3S_BLOCK_BYTES];

            let row_weight_base = row * cols;

            // Process super-blocks in pairs (ib32 += 2).
            let mut ib32 = 0usize;
            while ib32 < IQ3S_N_SUPERBLOCKS {
                let pair = ib32 / 2;
                let scale_byte = scales[pair];
                let db1 = d * (1.0 + 2.0 * (scale_byte & 0xf) as f32);
                let db2 = d * (1.0 + 2.0 * (scale_byte >> 4) as f32);

                // First super-block in pair.
                let col_base1 = blk * IQ3S_BLOCK_SIZE + ib32 * IQ3S_SUPER_BLOCK_SIZE;
                let sb_end1 = (col_base1 + IQ3S_SUPER_BLOCK_SIZE).min(cols);
                let out_slice1 =
                    &mut f32_weights[row_weight_base + col_base1..row_weight_base + sb_end1];
                // Need a temporary buffer for the full 32-element super-block.
                let mut tmp1 = [0.0f32; 32];
                dequant_iq3s_superblock(
                    &qs[8 * ib32..8 * ib32 + 8],
                    qh[ib32],
                    &signs[4 * ib32..4 * ib32 + 4],
                    db1,
                    &mut tmp1,
                    col_base1,
                    cols,
                );
                let copy_len = out_slice1.len();
                out_slice1.copy_from_slice(&tmp1[..copy_len]);

                // Second super-block in pair.
                let ib32b = ib32 + 1;
                let col_base2 = blk * IQ3S_BLOCK_SIZE + ib32b * IQ3S_SUPER_BLOCK_SIZE;
                let sb_end2 = (col_base2 + IQ3S_SUPER_BLOCK_SIZE).min(cols);
                if col_base2 < cols {
                    let out_slice2 =
                        &mut f32_weights[row_weight_base + col_base2..row_weight_base + sb_end2];
                    let mut tmp2 = [0.0f32; 32];
                    dequant_iq3s_superblock(
                        &qs[8 * ib32b..8 * ib32b + 8],
                        qh[ib32b],
                        &signs[4 * ib32b..4 * ib32b + 4],
                        db2,
                        &mut tmp2,
                        col_base2,
                        cols,
                    );
                    let copy_len = out_slice2.len();
                    out_slice2.copy_from_slice(&tmp2[..copy_len]);
                }

                ib32 += 2;
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_iq3_s(
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

    let f32_weights = dequant_iq3_s_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "iq3_s-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "iq3_s-input", input);
    let output_buf = create_output_f32(&ctx.device, "iq3_s-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "iq3_s-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_iq3_s"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("iq3_s-bgl"),
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
            label: Some("iq3_s-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("iq3_s-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("iq3_s-bg"),
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
            label: Some("iq3_s-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("iq3_s-pass"),
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

    /// Build a zero IQ3_S block with given scale.
    /// All-zero → qs=0, qh=0, signs=0, scales=0.
    /// IQ3S_GRID[0] = 0x01010101 → magnitudes = [1,1,1,1].
    /// scales=0 → db = d * (1 + 2*0) = d * 1.0.
    fn make_zero_block(scale: f32) -> Vec<u8> {
        let mut block = vec![0u8; IQ3S_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_dequant_zero_scale() {
        let block = make_zero_block(0.0);
        let result = dequant_iq3_s_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, 0.0, "weight[{i}] expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_grid0_all_positive() {
        // IQ3S_GRID[0] = 0x01010101 → mags all 1.
        // db = d * (1 + 2*0) = d; weight = d * 1 = d.
        let d = 2.0_f32;
        let block = make_zero_block(d);
        let result = dequant_iq3_s_to_f32(&block, 1, 256).expect("dequant");
        let expected = d;
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_too_small() {
        assert!(
            dequant_iq3_s_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Iq3SGpuKernel;
    }

    #[test]
    fn test_dequant_scale_nibble() {
        // scales[0] = 0x12 → low nibble = 2, high nibble = 1.
        // db1 = d * (1 + 2*2) = d * 5.0  (super-block 0)
        // db2 = d * (1 + 2*1) = d * 3.0  (super-block 1)
        // IQ3S_GRID[0] = [1,1,1,1].
        let d = 1.0_f32;
        let mut block = make_zero_block(d);
        block[IQ3S_SCALES_OFFSET] = 0x12;
        let result = dequant_iq3_s_to_f32(&block, 1, 256).expect("dequant");

        // Super-block 0 (weights 0..31): db1 * 1.0 = 5.0
        for (i, &v) in result.iter().enumerate().take(32) {
            assert!((v - 5.0).abs() < 1e-4, "out[{i}]={v}, expected 5.0");
        }
        // Super-block 1 (weights 32..63): db2 * 1.0 = 3.0
        for (i, &v) in result.iter().enumerate().take(64).skip(32) {
            assert!((v - 3.0).abs() < 1e-4, "out[{i}]={v}, expected 3.0");
        }
    }

    #[test]
    fn test_dequant_sign_applied() {
        // signs[0] = 1 → bit 0 set → weight[0] negated.
        let d = 1.0_f32;
        let mut block = make_zero_block(d);
        block[IQ3S_SIGNS_OFFSET] = 1;
        let result = dequant_iq3_s_to_f32(&block, 1, 256).expect("dequant");
        // db = d * 1.0 = 1.0, magnitude = 1
        assert!(
            (result[0] - (-1.0_f32)).abs() < 1e-5,
            "weight[0]={}, expected -1.0",
            result[0]
        );
        assert!(
            (result[1] - 1.0_f32).abs() < 1e-5,
            "weight[1]={}, expected 1.0",
            result[1]
        );
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_iq3_s_matches_cpu() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 64;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * IQ3S_BLOCK_BYTES);
        for r in 0..rows {
            let d_val = 0.01 + r as f32 * 0.001;
            let mut block = vec![0u8; IQ3S_BLOCK_BYTES];
            let d_le = half::f16::from_f32(d_val).to_le_bytes();
            block[0] = d_le[0];
            block[1] = d_le[1];
            // Vary the scale nibbles for each pair of super-blocks.
            for pair in 0..4 {
                let low_nibble = ((r + pair * 3) % 15) as u8;
                let high_nibble = ((r + pair * 7 + 5) % 15) as u8;
                block[IQ3S_SCALES_OFFSET + pair] = (high_nibble << 4) | low_nibble;
            }
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let f32_weights = dequant_iq3_s_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Iq3SGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("GPU GEMV IQ3_S");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

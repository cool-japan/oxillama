//! IQ1_M GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the IQ1_M block
//!      format: 256 weights per 56-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! IQ1_M block layout (56 bytes per 256 weights):
//! - bytes  0-31:  `qs[32]` — 32 bytes of quantized indices (lower 8 bits)
//! - bytes 32-47:  `qh[16]` — 16 bytes of sub-block headers
//! - bytes 48-55:  `scales[8]` — 8 bytes of packed sub-block scales
//!
//! Global scale `d` is reconstructed from `scales` as four u16 words:
//!   `d_bits = (sc[0]>>12) | ((sc[1]>>8)&0x00f0) | ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000)`
//!
//! Per-sub-block scales: sc_pair = sc[ib/2]; shift = 6*(ib%2):
//!   `dl1 = d * (2 * ((sc_pair >> shift)   & 7) + 1)` — groups 0-1
//!   `dl2 = d * (2 * ((sc_pair >> shift+3) & 7) + 1)` — groups 2-3
//!
//! Grid indices from qs + qh, delta from qh bits 3 and 7.
//! Grid bytes are signed i8: `y[j] = dl * (grid_i8[j] + delta[l])`
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// IQ1_M GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Iq1MGpuKernel;

impl GpuKernel for Iq1MGpuKernel {
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
            gpu_gemv_iq1_m(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── IQ1_M block constants ────────────────────────────────────────────────────

/// Weights per IQ1_M block.
#[cfg(any(feature = "gpu", test))]
const IQ1M_BLOCK_SIZE: usize = 256;
/// Bytes per IQ1_M block: 32 (qs) + 16 (qh) + 8 (scales).
#[cfg(any(feature = "gpu", test))]
const IQ1M_BLOCK_BYTES: usize = 56;
/// Byte offset where `qs` begins.
#[cfg(any(feature = "gpu", test))]
const IQ1M_QS_OFFSET: usize = 0;
/// Byte offset where `qh` begins.
#[cfg(any(feature = "gpu", test))]
const IQ1M_QH_OFFSET: usize = 32;
/// Byte offset where `scales` begins.
#[cfg(any(feature = "gpu", test))]
const IQ1M_SCALES_OFFSET: usize = 48;
/// Number of sub-blocks per IQ1_M block.
#[cfg(any(feature = "gpu", test))]
const IQ1M_N_SUBBLOCKS: usize = 8;
/// Weights per sub-block.
#[cfg(any(feature = "gpu", test))]
const IQ1M_SUB_BLOCK_SIZE: usize = IQ1M_BLOCK_SIZE / IQ1M_N_SUBBLOCKS; // 32
/// Number of groups per sub-block.
#[cfg(any(feature = "gpu", test))]
const IQ1M_GROUPS_PER_SUB: usize = 4;
/// Weights per group.
#[cfg(any(feature = "gpu", test))]
const IQ1M_WEIGHTS_PER_GROUP: usize = 8;
/// Delta constant (0.125).
#[cfg(any(feature = "gpu", test))]
const IQ1M_DELTA: f32 = 0.125;

/// Reconstruct the global FP16 scale `d` from the IQ1_M scales field (8 bytes).
///
/// Interpretation: 4 × u16 LE, upper nibble of each encodes 4 bits of the FP16 value.
/// `d_bits = (sc[0]>>12) | ((sc[1]>>8)&0x00f0) | ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000)`
#[cfg(any(feature = "gpu", test))]
#[inline]
fn reconstruct_d(scales: &[u8]) -> f32 {
    let sc0 = u16::from_le_bytes([scales[0], scales[1]]);
    let sc1 = u16::from_le_bytes([scales[2], scales[3]]);
    let sc2 = u16::from_le_bytes([scales[4], scales[5]]);
    let sc3 = u16::from_le_bytes([scales[6], scales[7]]);
    let d_bits: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    half::f16::from_bits(d_bits).to_f32()
}

/// Dequantise all IQ1_M blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_iq1_m_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    use super::iq1s_grid::IQ1S_GRID;

    let blocks_per_row = cols.div_ceil(IQ1M_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * IQ1M_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * IQ1M_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + IQ1M_BLOCK_BYTES];

            let qs = &block[IQ1M_QS_OFFSET..IQ1M_QH_OFFSET];
            let qh = &block[IQ1M_QH_OFFSET..IQ1M_SCALES_OFFSET];
            let scales = &block[IQ1M_SCALES_OFFSET..IQ1M_BLOCK_BYTES];

            let d = reconstruct_d(scales);

            // The scales field as 4 × u16 for per-sub-block scales.
            let sc: [u16; 4] = [
                u16::from_le_bytes([scales[0], scales[1]]),
                u16::from_le_bytes([scales[2], scales[3]]),
                u16::from_le_bytes([scales[4], scales[5]]),
                u16::from_le_bytes([scales[6], scales[7]]),
            ];

            for ib in 0..IQ1M_N_SUBBLOCKS {
                let sc_pair = sc[ib / 2];
                let sc_shift_base = 6 * (ib % 2);

                // dl1 for groups 0..2, dl2 for groups 2..4.
                let dl1 = d * (2.0 * (((sc_pair >> sc_shift_base) & 0x7) as f32) + 1.0);
                let dl2 = d * (2.0 * (((sc_pair >> (sc_shift_base + 3)) & 0x7) as f32) + 1.0);

                let qs_base = ib * IQ1M_GROUPS_PER_SUB;
                let qh_base = ib * 2;

                let qh0 = qh[qh_base] as usize;
                let qh1 = qh[qh_base + 1] as usize;

                let idx: [usize; 4] = [
                    (qs[qs_base] as usize) | ((qh0 << 8) & 0x700),
                    (qs[qs_base + 1] as usize) | ((qh0 << 4) & 0x700),
                    (qs[qs_base + 2] as usize) | ((qh1 << 8) & 0x700),
                    (qs[qs_base + 3] as usize) | ((qh1 << 4) & 0x700),
                ];

                // Delta signs from qh bits 3 and 7.
                let delta: [f32; 4] = [
                    if qh[qh_base] & 0x08 != 0 {
                        -IQ1M_DELTA
                    } else {
                        IQ1M_DELTA
                    },
                    if qh[qh_base] & 0x80 != 0 {
                        -IQ1M_DELTA
                    } else {
                        IQ1M_DELTA
                    },
                    if qh[qh_base + 1] & 0x08 != 0 {
                        -IQ1M_DELTA
                    } else {
                        IQ1M_DELTA
                    },
                    if qh[qh_base + 1] & 0x80 != 0 {
                        -IQ1M_DELTA
                    } else {
                        IQ1M_DELTA
                    },
                ];

                let output_base = blk * IQ1M_BLOCK_SIZE + ib * IQ1M_SUB_BLOCK_SIZE;

                for l in 0..IQ1M_GROUPS_PER_SUB {
                    let dl = if l < 2 { dl1 } else { dl2 };
                    let grid_raw = IQ1S_GRID[idx[l]].to_le_bytes();
                    let group_base = output_base + l * IQ1M_WEIGHTS_PER_GROUP;
                    for (j, &grid_byte) in grid_raw.iter().enumerate().take(IQ1M_WEIGHTS_PER_GROUP)
                    {
                        let col = group_base + j;
                        if col < cols {
                            let gv = grid_byte as i8 as f32;
                            f32_weights[row * cols + col] = dl * (gv + delta[l]);
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
fn gpu_gemv_iq1_m(
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

    let f32_weights = dequant_iq1_m_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "iq1_m-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "iq1_m-input", input);
    let output_buf = create_output_f32(&ctx.device, "iq1_m-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "iq1_m-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_iq1_m"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("iq1_m-bgl"),
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
            label: Some("iq1_m-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("iq1_m-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("iq1_m-bg"),
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
            label: Some("iq1_m-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("iq1_m-pass"),
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

    /// Build a zero IQ1_M block (d=0 via zero scales → d_bits=0 → d=0.0).
    fn make_zero_iq1m_block() -> Vec<u8> {
        vec![0u8; IQ1M_BLOCK_BYTES]
    }

    /// Build an IQ1_M block with a specific FP16 scale encoded in the scales field.
    ///
    /// The reconstruct_d function reads:
    ///   d_bits = (sc[0]>>12) | ((sc[1]>>8)&0x00f0) | ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000)
    /// So we encode d_bits nibbles:
    ///   sc[0][15:12] = d_bits[3:0]   → sc[0] = (d_bits & 0x000f) << 12
    ///   sc[1][15:12] = d_bits[7:4]   → sc[1] = (d_bits & 0x00f0) << 8
    ///   sc[2][15:12] = d_bits[11:8]  → sc[2] = (d_bits & 0x0f00) << 4
    ///   sc[3][15:12] = d_bits[15:12] → sc[3] =  d_bits & 0xf000
    fn make_scaled_iq1m_block(d: f32) -> Vec<u8> {
        let mut block = vec![0u8; IQ1M_BLOCK_BYTES];
        let d_bits = half::f16::from_f32(d).to_bits();
        let sc0: u16 = (d_bits & 0x000f) << 12;
        let sc1: u16 = (d_bits & 0x00f0) << 8;
        let sc2: u16 = (d_bits & 0x0f00) << 4;
        let sc3: u16 = d_bits & 0xf000;
        block[48..50].copy_from_slice(&sc0.to_le_bytes());
        block[50..52].copy_from_slice(&sc1.to_le_bytes());
        block[52..54].copy_from_slice(&sc2.to_le_bytes());
        block[54..56].copy_from_slice(&sc3.to_le_bytes());
        block
    }

    #[test]
    fn test_dequant_iq1_m_zero_scale() {
        // d=0 → all weights = 0.
        let block = make_zero_iq1m_block();
        let result = dequant_iq1_m_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, 0.0, "weight[{i}] expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_iq1_m_reconstruct_d_roundtrip() {
        let d_in = 2.0_f32;
        let block = make_scaled_iq1m_block(d_in);
        let scales = &block[IQ1M_SCALES_OFFSET..IQ1M_BLOCK_BYTES];
        let d_out = reconstruct_d(scales);
        assert!(
            (d_out - d_in).abs() < 1e-2,
            "d_out={d_out}, expected {d_in}"
        );
    }

    #[test]
    fn test_dequant_iq1_m_nonzero_scale() {
        // Grid index 0 → all bytes 0xff = -1. sc bits=0 → dl1=dl2=d*1.
        // delta = +0.125. Expected per weight: d_actual * (-0.875).
        let d = 1.0_f32;
        let block = make_scaled_iq1m_block(d);
        let result = dequant_iq1_m_to_f32(&block, 1, 256).expect("dequant");
        let d_actual = {
            let scales = &block[IQ1M_SCALES_OFFSET..IQ1M_BLOCK_BYTES];
            reconstruct_d(scales)
        };
        let expected = d_actual * (-1.0 + IQ1M_DELTA);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-3,
                "weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1_m_too_small() {
        assert!(
            dequant_iq1_m_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_iq1_m_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Iq1MGpuKernel;
    }

    /// End-to-end GPU GEMV: result must match CPU dequant+dot to within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_iq1_m_matches_cpu_reference() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 32;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * IQ1M_BLOCK_BYTES);
        for r in 0..rows {
            let d_val = 0.5 + r as f32 * 0.05;
            let block = make_scaled_iq1m_block(d_val);
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let f32_weights = dequant_iq1_m_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                f32_weights[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum()
            })
            .collect();

        let mut result = vec![0.0f32; rows];
        let kernel = Iq1MGpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut result, rows, cols)
            .expect("GPU GEMV IQ1_M");

        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

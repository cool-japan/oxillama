//! Q5_1 GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q5_1 block
//!      format: 2-byte f16 scale (d), 2-byte f16 minimum (m), 4-byte high-bits
//!      (qh), 16 bytes of lower 4 bits packed (qs) = 24 bytes/block, 32/block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! Q5_1 weight formula:
//!   `q5 = qs_nibble | (qh_bit << 4)` (5-bit unsigned, 0..31)
//!   `w   = d * q5 + m`               (asymmetric: explicit minimum)
//!
//! The 32 high bits in `qh` are laid out as:
//!   bit `i`      → high bit of weight `i`       (i in 0..16, lo nibble path)
//!   bit `i + 16` → high bit of weight `i + 16`  (i in 0..16, hi nibble path)
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q5_1 GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Q5_1GpuKernel;

impl GpuKernel for Q5_1GpuKernel {
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
            gpu_gemv_q5_1(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q5_1 block constants ─────────────────────────────────────────────────────

/// Weights per Q5_1 block.
#[cfg(any(feature = "gpu", test))]
const Q5_1_BLOCK_SIZE: usize = 32;

/// Bytes per Q5_1 block: 2 (scale) + 2 (min) + 4 (qh) + 16 (qs) = 24.
#[cfg(any(feature = "gpu", test))]
const Q5_1_BLOCK_BYTES: usize = 24;

/// Dequantise all Q5_1 blocks in `weight_bytes` into a flat f32 buffer.
///
/// Block layout (24 bytes, 32 weights):
/// - `[0..2]`   f16 LE scale `d`
/// - `[2..4]`   f16 LE minimum `m`
/// - `[4..8]`   4 bytes: `qh` (u32 LE) — 32 high bits
/// - `[8..24]`  16 bytes: `qs` — lower 4 bits of each weight (2 per byte)
///
/// Decoding for weight pair at byte `i` (0..16):
/// ```text
///   lo_nibble = qs[i] & 0x0F
///   hi_nibble = (qs[i] >> 4) & 0x0F
///   hi_bit_lo  = (qh >> i) & 1
///   hi_bit_hi  = (qh >> (i + 16)) & 1
///   q0 = lo_nibble | (hi_bit_lo << 4)   (5-bit unsigned)
///   q1 = hi_nibble | (hi_bit_hi << 4)   (5-bit unsigned)
///   w[i]      = d * q0 + m
///   w[i + 16] = d * q1 + m
/// ```
#[cfg(any(feature = "gpu", test))]
pub(crate) fn dequant_q5_1_to_f32(
    weight_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q5_1_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q5_1_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q5_1_BLOCK_BYTES;
            let block = &weight_bytes[block_offset..block_offset + Q5_1_BLOCK_BYTES];

            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let m = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
            let qs = &block[8..24];

            let base_row = row * cols;
            let base_blk = blk * Q5_1_BLOCK_SIZE;

            for (i, &qs_byte) in qs.iter().enumerate() {
                let lo_nibble = qs_byte & 0x0F;
                let hi_nibble = (qs_byte >> 4) & 0x0F;

                let hi_bit_lo = ((qh >> i) & 1) as u8;
                let hi_bit_hi = ((qh >> (i + 16)) & 1) as u8;

                let q0 = (lo_nibble | (hi_bit_lo << 4)) as f32;
                let q1 = (hi_nibble | (hi_bit_hi << 4)) as f32;

                let col0 = base_blk + i;
                let col1 = base_blk + i + 16;

                if col0 < cols {
                    f32_weights[base_row + col0] = d * q0 + m;
                }
                if col1 < cols {
                    f32_weights[base_row + col1] = d * q1 + m;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q5_1(
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
    let f32_weights = dequant_q5_1_to_f32(weight_bytes, rows, cols)?;

    // Step 2 — upload buffers.
    let weight_buf = upload_f32(&ctx.device, "q5_1-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q5_1-input", input);
    let output_buf = create_output_f32(&ctx.device, "q5_1-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q5_1-params", &params);

    // Step 3 — build compute pipeline.
    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q5_1"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q5_1-bgl"),
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
            label: Some("q5_1-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q5_1-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q5_1-bg"),
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

    // Step 4 — dispatch (workgroup size = 64).
    let dispatch_x = rows.div_ceil(64) as u32;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q5_1-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q5_1-pass"),
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

    /// Build a Q5_1 block: scale (f16) + min (f16) + qh (u32 LE) + 16 qs bytes.
    fn make_q5_1_block(d: f32, m: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q5_1_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    // ── 1. zero block ──────────────────────────────────────────────────────────

    /// d=0, m=0, qh=0, qs=0 → w = 0 * 0 + 0 = 0 for all weights.
    #[test]
    fn test_dequant_q5_1_zero_block() {
        let block = make_q5_1_block(0.0, 0.0, 0, &[0u8; 16]);
        let result = dequant_q5_1_to_f32(&block, 1, 32).expect("dequant");
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0.0, got {v}");
        }
    }

    /// Minimum-only: d=0, m=3.0, any q → w = 0 * q + 3.0 = 3.0.
    #[test]
    fn test_dequant_q5_1_min_only() {
        let block = make_q5_1_block(0.0, 3.0, 0, &[0u8; 16]);
        let result = dequant_q5_1_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 3.0).abs() < 1e-4, "weight[{i}] expected 3.0, got {v}");
        }
    }

    // ── 2. buffer underflow error ──────────────────────────────────────────────

    /// 4-byte buffer is too small for one Q5_1 block (24 bytes).
    #[test]
    fn test_dequant_q5_1_buffer_underflow_error() {
        let result = dequant_q5_1_to_f32(&[0u8; 4], 1, 32);
        assert!(result.is_err(), "must error on too-small buffer");
        match result {
            Err(GpuError::BufferSize { expected, got }) => {
                assert_eq!(expected, Q5_1_BLOCK_BYTES);
                assert_eq!(got, 4);
            }
            _ => panic!("wrong error variant"),
        }
    }

    // ── 3. matches-scalar-reference ───────────────────────────────────────────

    /// qh=0xFFFFFFFF, qs=0 → q5=16 for all → w = d*16 + m.
    #[test]
    fn test_dequant_q5_1_matches_scalar_reference() {
        let d = 1.0f32;
        let m = 2.0f32;
        let block = make_q5_1_block(d, m, 0xFFFF_FFFFu32, &[0u8; 16]);
        let result = dequant_q5_1_to_f32(&block, 1, 32).expect("dequant");
        let expected = d * 16.0 + m; // = 18.0
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "weight[{i}] expected {expected}, got {v}"
            );
        }
    }

    /// Maximum quant: qh=0xFFFFFFFF, qs=0xFF → q5=31 → w = d*31 + m.
    #[test]
    fn test_dequant_q5_1_all_max() {
        let d = 1.0f32;
        let m = 0.0f32;
        let block = make_q5_1_block(d, m, 0xFFFF_FFFFu32, &[0xFFu8; 16]);
        let result = dequant_q5_1_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 31.0).abs() < 1e-4,
                "weight[{i}] expected 31.0, got {v}"
            );
        }
    }

    // ── 4. dispatcher coherence ────────────────────────────────────────────────

    #[test]
    fn test_q5_1_dispatcher_returns_none_without_gpu() {
        let _kernel: &dyn GpuKernel = &Q5_1GpuKernel;
        let dispatcher = crate::GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(oxillama_gguf::GgufTensorType::Q5_1);
        if dispatcher.has_gpu() {
            assert!(
                kernel.is_some(),
                "Q5_1 kernel must be present when GPU is available"
            );
        } else {
            assert!(kernel.is_none(), "Q5_1 kernel must be absent without GPU");
        }
    }

    // ── 5. two-block roundtrip ─────────────────────────────────────────────────

    /// Two Q5_1 blocks — each row must decode independently.
    #[test]
    fn test_q5_1_two_block_roundtrip() {
        // Block A: d=1.0, m=0.0, qh=0, qs=0 → q5=0 → w = 1.0*0+0 = 0.0
        let block_a = make_q5_1_block(1.0, 0.0, 0, &[0u8; 16]);
        // Block B: d=0.5, m=1.0, qh=0xFFFFFFFF, qs=0xFF → q5=31 → w=0.5*31+1.0=16.5
        let block_b = make_q5_1_block(0.5, 1.0, 0xFFFF_FFFFu32, &[0xFFu8; 16]);

        let mut data = Vec::new();
        data.extend_from_slice(&block_a);
        data.extend_from_slice(&block_b);

        let result = dequant_q5_1_to_f32(&data, 2, 32).expect("two-block dequant");
        assert_eq!(result.len(), 64, "must produce 2 * BLOCK_SIZE values");

        for &v in &result[..32] {
            assert!(v.abs() < 1e-4, "row0 weight: expected 0.0, got {v}");
        }
        for &v in &result[32..] {
            assert!(
                (v - 16.5).abs() < 1e-3,
                "row1 weight: expected 16.5, got {v}"
            );
        }
    }

    /// Scale-plus-min with mixed nibble pattern.
    #[test]
    fn test_dequant_q5_1_scale_min_mixed_nibbles() {
        // d=0.5, m=0.25; first byte qs[0]=0x12 → lo=2, hi=1; qh=0 → hi_bits=0
        // w[0] = 0.5*2 + 0.25 = 1.25; w[1] ... wait, layout:
        // w[0] = d*q0+m where q0 = lo_nibble | 0 = 2; w[0] = 0.5*2+0.25 = 1.25
        // w[16] = d*q1+m where q1 = hi_nibble | 0 = 1; w[16] = 0.5*1+0.25 = 0.75
        let mut qs = [0u8; 16];
        qs[0] = 0x12; // lo=2, hi=1
        let block = make_q5_1_block(0.5, 0.25, 0, &qs);
        let result = dequant_q5_1_to_f32(&block, 1, 32).expect("dequant");
        assert!(
            (result[0] - 1.25).abs() < 1e-4,
            "w[0] expected 1.25, got {}",
            result[0]
        );
        assert!(
            (result[16] - 0.75).abs() < 1e-4,
            "w[16] expected 0.75, got {}",
            result[16]
        );
    }

    /// GPU GEMV end-to-end: result matches CPU dequant+dot.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q5_1_matches_cpu() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 4;
        let cols = 32;
        let mut weight_bytes = Vec::new();
        for r in 0..rows {
            let qh: u32 = (r as u32)
                .wrapping_mul(0x55AA_BB77)
                .wrapping_add(0x1234_5678);
            let mut qs = [0u8; 16];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((r * 13 + i * 7 + 5) & 0xFF) as u8;
            }
            let d_val = 0.1 + r as f32 * 0.05;
            let m_val = 0.05 + r as f32 * 0.01;
            weight_bytes.extend_from_slice(&make_q5_1_block(d_val, m_val, qh, &qs));
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1) - 1.5).collect();
        let f32_weights = dequant_q5_1_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Q5_1GpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("Q5_1 GPU GEMV");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}"
            );
        }
    }
}

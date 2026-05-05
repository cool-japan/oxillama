//! Q5_0 GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q5_0 block
//!      format: 2-byte f16 scale (d), 4-byte high-bits (qh), 16 bytes of
//!      lower 4 bits packed (qs) = 22 bytes/block, 32 weights/block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! Q5_0 weight formula:
//!   `q5 = qs_nibble | (qh_bit << 4)` (5-bit unsigned, 0..31)
//!   `w   = d * (q5 - 16)`            (symmetric: bias of 16)
//!
//! The 32 high bits in `qh` are laid out as follows:
//!   bit `i`      → high bit of weight `i`       (i in 0..16, lo nibble path)
//!   bit `i + 16` → high bit of weight `i + 16`  (i in 0..16, hi nibble path)
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q5_0 GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Q5_0GpuKernel;

impl GpuKernel for Q5_0GpuKernel {
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
            gpu_gemv_q5_0(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q5_0 block constants ─────────────────────────────────────────────────────

/// Weights per Q5_0 block.
#[cfg(any(feature = "gpu", test))]
const Q5_0_BLOCK_SIZE: usize = 32;

/// Bytes per Q5_0 block: 2 (scale) + 4 (qh) + 16 (qs) = 22.
#[cfg(any(feature = "gpu", test))]
const Q5_0_BLOCK_BYTES: usize = 22;

/// Dequantise all Q5_0 blocks in `weight_bytes` into a flat f32 buffer.
///
/// Block layout (22 bytes, 32 weights):
/// - `[0..2]`   f16 LE scale `d`
/// - `[2..6]`   4 bytes: `qh` (u32 LE) — 32 high bits
/// - `[6..22]`  16 bytes: `qs` — lower 4 bits of each weight (2 per byte)
///
/// Decoding for weight pair at byte `i` (0..16):
/// ```text
///   lo_nibble = qs[i] & 0x0F
///   hi_nibble = (qs[i] >> 4) & 0x0F
///   hi_bit_lo  = (qh >> i) & 1
///   hi_bit_hi  = (qh >> (i + 16)) & 1
///   q0 = (lo_nibble | (hi_bit_lo << 4)) as i32 - 16
///   q1 = (hi_nibble | (hi_bit_hi << 4)) as i32 - 16
///   w[i]      = d * q0
///   w[i + 16] = d * q1
/// ```
#[cfg(any(feature = "gpu", test))]
pub(crate) fn dequant_q5_0_to_f32(
    weight_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q5_0_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q5_0_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q5_0_BLOCK_BYTES;
            let block = &weight_bytes[block_offset..block_offset + Q5_0_BLOCK_BYTES];

            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
            let qs = &block[6..22];

            let base_row = row * cols;
            let base_blk = blk * Q5_0_BLOCK_SIZE;

            for (i, &qs_byte) in qs.iter().enumerate() {
                let lo_nibble = qs_byte & 0x0F;
                let hi_nibble = (qs_byte >> 4) & 0x0F;

                let hi_bit_lo = ((qh >> i) & 1) as u8;
                let hi_bit_hi = ((qh >> (i + 16)) & 1) as u8;

                let q0 = (lo_nibble | (hi_bit_lo << 4)) as i32 - 16;
                let q1 = (hi_nibble | (hi_bit_hi << 4)) as i32 - 16;

                let col0 = base_blk + i;
                let col1 = base_blk + i + 16;

                if col0 < cols {
                    f32_weights[base_row + col0] = d * q0 as f32;
                }
                if col1 < cols {
                    f32_weights[base_row + col1] = d * q1 as f32;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q5_0(
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
    let f32_weights = dequant_q5_0_to_f32(weight_bytes, rows, cols)?;

    // Step 2 — upload buffers.
    let weight_buf = upload_f32(&ctx.device, "q5_0-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q5_0-input", input);
    let output_buf = create_output_f32(&ctx.device, "q5_0-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q5_0-params", &params);

    // Step 3 — build compute pipeline.
    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q5_0"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q5_0-bgl"),
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
            label: Some("q5_0-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q5_0-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q5_0-bg"),
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
            label: Some("q5_0-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q5_0-pass"),
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

    /// Build a Q5_0 block: 2-byte f16 scale + 4-byte qh (u32 LE) + 16 qs bytes.
    fn make_q5_0_block(d: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q5_0_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    // ── 1. zero block ──────────────────────────────────────────────────────────

    /// d=0, qh=0, qs=0 → all weights = 0 * (0 - 16) = 0.
    #[test]
    fn test_dequant_q5_0_zero_block() {
        let block = make_q5_0_block(0.0, 0, &[0u8; 16]);
        let result = dequant_q5_0_to_f32(&block, 1, 32).expect("dequant");
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0.0, got {v}");
        }
    }

    // ── 2. buffer underflow error ──────────────────────────────────────────────

    /// 4-byte buffer is too small for a Q5_0 block (22 bytes).
    #[test]
    fn test_dequant_q5_0_buffer_underflow_error() {
        let result = dequant_q5_0_to_f32(&[0u8; 4], 1, 32);
        assert!(result.is_err(), "must error on too-small buffer");
        match result {
            Err(GpuError::BufferSize { expected, got }) => {
                assert_eq!(expected, Q5_0_BLOCK_BYTES);
                assert_eq!(got, 4);
            }
            _ => panic!("wrong error variant"),
        }
    }

    // ── 3. matches-scalar-reference ───────────────────────────────────────────

    /// Check a specific Q5_0 block with known values against the formula.
    ///
    /// All 5-bit quants = 16 (lo=0, hi=1) → q-16 = 0 → weight = 0.
    /// qh = 0xFFFFFFFF (all bits set), qs = all 0 (nibbles 0).
    #[test]
    fn test_dequant_q5_0_matches_scalar_reference() {
        // q5 = 0 | (1 << 4) = 16; w = d * (16 - 16) = 0
        let block = make_q5_0_block(1.0, 0xFFFF_FFFFu32, &[0u8; 16]);
        let result = dequant_q5_0_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-5, "weight[{i}] expected 0.0, got {v}");
        }
    }

    /// All 5-bit quants = 0 → q-16 = -16 → weight = d * (-16).
    #[test]
    fn test_dequant_q5_0_all_min() {
        // qh=0, qs=0 → q5 = 0 | 0 = 0; w = 2.0 * (0-16) = -32.0
        let block = make_q5_0_block(2.0, 0, &[0u8; 16]);
        let result = dequant_q5_0_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-32.0)).abs() < 1e-4,
                "weight[{i}] expected -32.0, got {v}"
            );
        }
    }

    /// All 5-bit quants = 31 → q-16 = 15 → weight = d * 15.
    #[test]
    fn test_dequant_q5_0_all_max() {
        // qh=0xFFFFFFFF, qs=0xFF → nibbles all 15, high bit 1 → q=31; w=1.0*15=15.0
        let block = make_q5_0_block(1.0, 0xFFFF_FFFFu32, &[0xFFu8; 16]);
        let result = dequant_q5_0_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 15.0).abs() < 1e-4,
                "weight[{i}] expected 15.0, got {v}"
            );
        }
    }

    // ── 4. dispatcher returns Some/None consistently ───────────────────────────

    #[test]
    fn test_q5_0_dispatcher_returns_none_without_gpu() {
        let _kernel: &dyn GpuKernel = &Q5_0GpuKernel;
        let dispatcher = crate::GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(oxillama_gguf::GgufTensorType::Q5_0);
        if dispatcher.has_gpu() {
            assert!(kernel.is_some(), "Q5_0 kernel must be present when GPU is available");
        } else {
            assert!(kernel.is_none(), "Q5_0 kernel must be absent without GPU");
        }
    }

    // ── 5. two-block roundtrip ─────────────────────────────────────────────────

    /// Two back-to-back Q5_0 blocks; each row decodes independently.
    #[test]
    fn test_q5_0_two_block_roundtrip() {
        // Block A: d=1.0, qh=0, qs=0 → q5=0 → w=-16.0 for all weights
        let block_a = make_q5_0_block(1.0, 0, &[0u8; 16]);
        // Block B: d=0.5, qh=0xFFFFFFFF, qs=0xFF → q5=31 → w=0.5*15=7.5 for all
        let block_b = make_q5_0_block(0.5, 0xFFFF_FFFFu32, &[0xFFu8; 16]);

        let mut data = Vec::new();
        data.extend_from_slice(&block_a);
        data.extend_from_slice(&block_b);

        let result = dequant_q5_0_to_f32(&data, 2, 32).expect("two-block dequant");
        assert_eq!(result.len(), 64, "must produce 2 * BLOCK_SIZE values");

        for &v in &result[..32] {
            assert!(
                (v - (-16.0)).abs() < 1e-4,
                "row0 weight: expected -16.0, got {v}"
            );
        }
        for &v in &result[32..] {
            assert!(
                (v - 7.5).abs() < 1e-4,
                "row1 weight: expected 7.5, got {v}"
            );
        }
    }

    /// Mixed high-bit pattern: alternating qh bits.
    #[test]
    fn test_dequant_q5_0_alternating_qh() {
        // qh = 0xAAAAAAAA: bits 1,3,5,... set.
        // For i=0: hi_bit_lo = (qh >> 0) & 1 = 0; hi_bit_hi = (qh >> 16) & 1 = 0
        // For i=1: hi_bit_lo = (qh >> 1) & 1 = 1; hi_bit_hi = (qh >> 17) & 1 = 1
        // qs all 0 (nibbles 0).
        let qh: u32 = 0xAAAA_AAAAu32;
        let block = make_q5_0_block(1.0, qh, &[0u8; 16]);
        let result = dequant_q5_0_to_f32(&block, 1, 32).expect("dequant");

        // Verify first 4 weights manually:
        // i=0: lo_nibble=0, hi_bit_lo=(qh>>0)&1=0 → q5=0 → w=-16
        // i=1: lo_nibble=0, hi_bit_lo=(qh>>1)&1=1 → q5=16 → w=0
        assert!((result[0] - (-16.0)).abs() < 1e-4, "w[0]={}", result[0]);
        assert!((result[1] - 0.0).abs() < 1e-4, "w[1]={}", result[1]);
        // i=0: hi_nibble=0, hi_bit_hi=(qh>>16)&1=0 → q5=0 → w=-16 → result[16]
        assert!((result[16] - (-16.0)).abs() < 1e-4, "w[16]={}", result[16]);
    }

    /// GPU GEMV end-to-end: result matches CPU dequant+dot.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q5_0_matches_cpu() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 4;
        let cols = 32;
        let mut weight_bytes = Vec::new();
        for r in 0..rows {
            let qh: u32 = ((r as u32 * 0x12345678).wrapping_add(0xABCD_EF01)) & 0xFFFF_FFFF;
            let mut qs = [0u8; 16];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((r * 7 + i * 11 + 3) & 0xFF) as u8;
            }
            let d_val = 0.1 + r as f32 * 0.05;
            weight_bytes.extend_from_slice(&make_q5_0_block(d_val, qh, &qs));
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1) - 1.5).collect();
        let f32_weights = dequant_q5_0_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Q5_0GpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("Q5_0 GPU GEMV");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}"
            );
        }
    }
}

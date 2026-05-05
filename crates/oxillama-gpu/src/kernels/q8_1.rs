//! Q8_1 GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q8_1 block
//!      format: 2-byte f16 scale (d), 2-byte f16 sum (s, unused for dequant),
//!      32 × i8 signed values = 36 bytes/block, 32 weights/block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! Q8_1 weight formula: `w = d * qs[i]` where `qs[i]` is signed int8.
//!
//! The `sum` field `s` (block[2..4]) equals `d * Σ qs[i]` — it is stored as
//! an optimisation for GEMM but is NOT needed for plain dequant/GEMV.
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q8_1 GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Q8_1GpuKernel;

impl GpuKernel for Q8_1GpuKernel {
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
            gpu_gemv_q8_1(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q8_1 block constants ─────────────────────────────────────────────────────

/// Weights per Q8_1 block.
#[cfg(any(feature = "gpu", test))]
const Q8_1_BLOCK_SIZE: usize = 32;

/// Bytes per Q8_1 block: 2 (scale) + 2 (sum) + 32 (i8 values) = 36.
#[cfg(any(feature = "gpu", test))]
const Q8_1_BLOCK_BYTES: usize = 36;

/// Dequantise all Q8_1 blocks in `weight_bytes` into a flat f32 buffer.
///
/// Block layout (36 bytes, 32 weights):
/// - `[0..2]`   f16 LE scale `d`
/// - `[2..4]`   f16 LE sum `s` (`d * Σ qs`) — ignored for dequant
/// - `[4..36]`  32 signed bytes (`i8`) quantised values
///
/// Dequant: `w[i] = d * qs[i]`
#[cfg(any(feature = "gpu", test))]
pub(crate) fn dequant_q8_1_to_f32(
    weight_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q8_1_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q8_1_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q8_1_BLOCK_BYTES;
            let block = &weight_bytes[block_offset..block_offset + Q8_1_BLOCK_BYTES];

            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            // block[2..4] = sum — skip (not needed for per-weight dequant).
            let qs = &block[4..36];

            let base = row * cols + blk * Q8_1_BLOCK_SIZE;
            for i in 0..Q8_1_BLOCK_SIZE {
                let col = blk * Q8_1_BLOCK_SIZE + i;
                if col < cols {
                    f32_weights[base + i] = d * (qs[i] as i8) as f32;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q8_1(
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
    let f32_weights = dequant_q8_1_to_f32(weight_bytes, rows, cols)?;

    // Step 2 — upload buffers.
    let weight_buf = upload_f32(&ctx.device, "q8_1-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q8_1-input", input);
    let output_buf = create_output_f32(&ctx.device, "q8_1-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q8_1-params", &params);

    // Step 3 — build compute pipeline.
    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q8_1"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q8_1-bgl"),
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
            label: Some("q8_1-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q8_1-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q8_1-bg"),
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
            label: Some("q8_1-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q8_1-pass"),
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

    /// Build a Q8_1 block: 2-byte f16 scale + 2-byte f16 sum + 32 i8 values.
    fn make_q8_1_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_1_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(d).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        // sum = d * Σ qs  (stored as f16, not used for dequant but must be valid bytes)
        let s: f32 = d * qs.iter().map(|&q| q as f32).sum::<f32>();
        let s_bits = half::f16::from_f32(s).to_bits();
        block.extend_from_slice(&s_bits.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    // ── 1. zero block ──────────────────────────────────────────────────────────

    /// d=0, all qs=0 → w = 0 * 0 = 0 for all weights.
    #[test]
    fn test_dequant_q8_1_zero_block() {
        let block = make_q8_1_block(0.0, &[0i8; 32]);
        let result = dequant_q8_1_to_f32(&block, 1, 32).expect("dequant");
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0.0, got {v}");
        }
    }

    // ── 2. buffer underflow error ──────────────────────────────────────────────

    /// 4-byte buffer is too small for one Q8_1 block (36 bytes).
    #[test]
    fn test_dequant_q8_1_buffer_underflow_error() {
        let result = dequant_q8_1_to_f32(&[0u8; 4], 1, 32);
        assert!(result.is_err(), "must error on too-small buffer");
        match result {
            Err(GpuError::BufferSize { expected, got }) => {
                assert_eq!(expected, Q8_1_BLOCK_BYTES);
                assert_eq!(got, 4);
            }
            _ => panic!("wrong error variant"),
        }
    }

    // ── 3. matches-scalar-reference ───────────────────────────────────────────

    /// Known positive i8 values: d=0.5, all qs=10 → w = 0.5 * 10 = 5.0.
    #[test]
    fn test_dequant_q8_1_matches_scalar_reference_positive() {
        let block = make_q8_1_block(0.5, &[10i8; 32]);
        let result = dequant_q8_1_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-4, "weight[{i}] expected 5.0, got {v}");
        }
    }

    /// Known negative i8 values: d=2.0, all qs=-5 → w = 2.0 * (-5) = -10.0.
    #[test]
    fn test_dequant_q8_1_matches_scalar_reference_negative() {
        let block = make_q8_1_block(2.0, &[-5i8; 32]);
        let result = dequant_q8_1_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-10.0)).abs() < 1e-4,
                "weight[{i}] expected -10.0, got {v}"
            );
        }
    }

    /// Mixed i8 values cross-check: compute expected manually and compare.
    #[test]
    fn test_dequant_q8_1_matches_scalar_reference_mixed() {
        let d = 0.25f32;
        let mut qs = [0i8; 32];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i16 * 7 - 64).clamp(-128, 127)) as i8;
        }
        let block = make_q8_1_block(d, &qs);
        let result = dequant_q8_1_to_f32(&block, 1, 32).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            let expected = d * qs[i] as f32;
            assert!(
                (v - expected).abs() < 1e-5,
                "weight[{i}]: got {v}, expected {expected}"
            );
        }
    }

    // ── 4. dispatcher coherence ────────────────────────────────────────────────

    #[test]
    fn test_q8_1_dispatcher_returns_none_without_gpu() {
        let _kernel: &dyn GpuKernel = &Q8_1GpuKernel;
        let dispatcher = crate::GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(oxillama_gguf::GgufTensorType::Q8_1);
        if dispatcher.has_gpu() {
            assert!(kernel.is_some(), "Q8_1 kernel must be present when GPU is available");
        } else {
            assert!(kernel.is_none(), "Q8_1 kernel must be absent without GPU");
        }
    }

    // ── 5. two-block roundtrip ─────────────────────────────────────────────────

    /// Two Q8_1 blocks — each row's 32 values must decode independently.
    #[test]
    fn test_q8_1_two_block_roundtrip() {
        // Block A: d=1.0, all qs=3 → w = 3.0
        let block_a = make_q8_1_block(1.0, &[3i8; 32]);
        // Block B: d=0.5, all qs=-6 → w = -3.0
        let block_b = make_q8_1_block(0.5, &[-6i8; 32]);

        let mut data = Vec::new();
        data.extend_from_slice(&block_a);
        data.extend_from_slice(&block_b);

        // 2 rows × 32 cols — one block per row.
        let result = dequant_q8_1_to_f32(&data, 2, 32).expect("two-block dequant");
        assert_eq!(result.len(), 64, "must produce 2 * BLOCK_SIZE values");

        for &v in &result[..32] {
            assert!((v - 3.0).abs() < 1e-4, "row0 weight: expected 3.0, got {v}");
        }
        for &v in &result[32..] {
            assert!(
                (v - (-3.0)).abs() < 1e-4,
                "row1 weight: expected -3.0, got {v}"
            );
        }
    }

    /// i8 extremes: d=1.0, qs[0]=127, qs[1]=-128 → w[0]=127.0, w[1]=-128.0.
    #[test]
    fn test_dequant_q8_1_i8_extremes() {
        let mut qs = [0i8; 32];
        qs[0] = 127;
        qs[1] = -128;
        let block = make_q8_1_block(1.0, &qs);
        let result = dequant_q8_1_to_f32(&block, 1, 32).expect("dequant");
        assert!((result[0] - 127.0).abs() < 1e-4, "w[0]={}", result[0]);
        assert!((result[1] - (-128.0)).abs() < 1e-4, "w[1]={}", result[1]);
        for &v in &result[2..] {
            assert!(v.abs() < 1e-4, "expected 0.0, got {v}");
        }
    }

    /// GPU GEMV end-to-end: result matches CPU dequant+dot.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q8_1_matches_cpu() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 4;
        let cols = 32;
        let mut weight_bytes = Vec::new();
        for r in 0..rows {
            let mut qs = [0i8; 32];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((r as i16 * 7 + i as i16 * 3 - 50).clamp(-128, 127)) as i8;
            }
            let d_val = 0.1 + r as f32 * 0.05;
            weight_bytes.extend_from_slice(&make_q8_1_block(d_val, &qs));
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1) - 1.5).collect();
        let f32_weights = dequant_q8_1_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Q8_1GpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, rows, cols)
            .expect("Q8_1 GPU GEMV");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}"
            );
        }
    }
}

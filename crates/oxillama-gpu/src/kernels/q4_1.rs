//! Q4_1 GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the Q4_1 block
//!      format: 2-byte f16 scale (d), 2-byte f16 minimum (m), 16 bytes of
//!      4-bit nibbles (2 per byte) = 20 bytes/block, 32 weights/block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! Q4_1 weight formula: `w = d * nibble + m` where nibble is 4-bit unsigned
//! (0..15) — no sign bias.
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Q4_1 GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Q4_1GpuKernel;

impl GpuKernel for Q4_1GpuKernel {
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
            gpu_gemv_q4_1(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            // Suppress unused-variable warnings when gpu feature is off.
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Q4_1 block constants ─────────────────────────────────────────────────────

/// Weights per Q4_1 block.
#[cfg(any(feature = "gpu", test))]
const Q4_1_BLOCK_SIZE: usize = 32;

/// Bytes per Q4_1 block: 2 (scale f16) + 2 (min f16) + 16 (nibbles) = 20.
#[cfg(any(feature = "gpu", test))]
const Q4_1_BLOCK_BYTES: usize = 20;

/// Dequantise all Q4_1 blocks in `weight_bytes` into a flat f32 buffer.
///
/// Block layout (20 bytes, 32 weights):
/// - `[0..2]`   f16 LE scale  `d`
/// - `[2..4]`   f16 LE minimum `m`
/// - `[4..20]`  16 bytes of packed 4-bit nibbles, lo nibble first.
///
/// Dequant: `w[i] = d * nibble[i] + m`  (nibble unsigned, 0..15)
#[cfg(any(feature = "gpu", test))]
pub(crate) fn dequant_q4_1_to_f32(
    weight_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(Q4_1_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q4_1_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q4_1_BLOCK_BYTES;
            let block = &weight_bytes[block_offset..block_offset + Q4_1_BLOCK_BYTES];

            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let m = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();

            // 16 nibble bytes, each encodes 2 weights: lo nibble = w[2i], hi = w[2i+1].
            for i in 0..(Q4_1_BLOCK_SIZE / 2) {
                let byte = block[4 + i];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;

                let base_col = blk * Q4_1_BLOCK_SIZE + i * 2;
                if base_col < cols {
                    f32_weights[row * cols + base_col] = d * lo + m;
                }
                if base_col + 1 < cols {
                    f32_weights[row * cols + base_col + 1] = d * hi + m;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_q4_1(
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
    let f32_weights = dequant_q4_1_to_f32(weight_bytes, rows, cols)?;

    // Step 2 — upload buffers.
    let weight_buf = upload_f32(&ctx.device, "q4_1-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "q4_1-input", input);
    let output_buf = create_output_f32(&ctx.device, "q4_1-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "q4_1-params", &params);

    // Step 3 — build compute pipeline.
    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_q4_1"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("q4_1-bgl"),
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
            label: Some("q4_1-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("q4_1-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("q4_1-bg"),
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
            label: Some("q4_1-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("q4_1-pass"),
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

    /// Build a single Q4_1 block: 2-byte f16 scale + 2-byte f16 min + 16 nibble bytes.
    fn make_q4_1_block(d: f32, m: f32, nibbles: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q4_1_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
        block.extend_from_slice(nibbles);
        block
    }

    // ── 1. zero block ──────────────────────────────────────────────────────────

    /// A block with d=0, m=0 and all nibbles=0 must dequant to all zeros.
    #[test]
    fn test_dequant_q4_1_zero_block() {
        let block = make_q4_1_block(0.0, 0.0, &[0u8; 16]);
        let result = dequant_q4_1_to_f32(&block, 1, 32).expect("dequant should succeed");
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0.0, got {v}");
        }
    }

    /// Minimum-only: d=0, m=5 → every weight = 0 * nibble + 5 = 5.
    #[test]
    fn test_dequant_q4_1_min_only() {
        let block = make_q4_1_block(0.0, 5.0, &[0u8; 16]);
        let result = dequant_q4_1_to_f32(&block, 1, 32).expect("dequant");
        for &v in &result {
            assert!((v - 5.0).abs() < 1e-4, "expected 5.0, got {v}");
        }
    }

    // ── 2. buffer underflow error ──────────────────────────────────────────────

    /// A 4-byte buffer is too small for even one Q4_1 block (20 bytes).
    #[test]
    fn test_dequant_q4_1_buffer_underflow_error() {
        let result = dequant_q4_1_to_f32(&[0u8; 4], 1, 32);
        assert!(result.is_err(), "must error on too-small buffer");
        match result {
            Err(GpuError::BufferSize { expected, got }) => {
                assert_eq!(expected, Q4_1_BLOCK_BYTES, "expected full block size");
                assert_eq!(got, 4);
            }
            _ => panic!("wrong error variant"),
        }
    }

    /// Exact-match scenario: known d, m, nibbles → check specific weights.
    #[test]
    fn test_dequant_q4_1_known_values() {
        // d=1.0, m=0.0; first byte = 0x30 → lo=0, hi=3.
        // weight[0] = 1.0*0 + 0 = 0.0; weight[1] = 1.0*3 + 0 = 3.0.
        let mut nibbles = [0x00u8; 16];
        nibbles[0] = 0x30; // lo=0, hi=3
        let block = make_q4_1_block(1.0, 0.0, &nibbles);
        let result = dequant_q4_1_to_f32(&block, 1, 32).expect("dequant");
        assert!((result[0] - 0.0).abs() < 1e-5, "weight[0]={}", result[0]);
        assert!((result[1] - 3.0).abs() < 1e-5, "weight[1]={}", result[1]);
        // Remaining nibbles are 0 → d*0+m = 0.0
        for &v in &result[2..] {
            assert!(v.abs() < 1e-5, "expected 0.0, got {v}");
        }
    }

    // ── 3. matches-scalar-reference ───────────────────────────────────────────

    /// Cross-check: dequant output from our GPU-path function must match a
    /// manually computed reference for a known test block (d=0.5, m=1.0).
    #[test]
    fn test_dequant_q4_1_matches_scalar_reference() {
        // d=0.5, m=1.0, nibbles: byte_i = i (i in 0..16)
        //   byte 0 = 0x00: lo=0 → w[0] = 0.5*0+1.0 = 1.0; hi=0 → w[1] = 1.0
        //   byte 1 = 0x01: lo=1 → w[2] = 1.5;             hi=0 → w[3] = 1.0
        //   ...
        let mut nibbles = [0u8; 16];
        for (i, n) in nibbles.iter_mut().enumerate() {
            *n = i as u8; // lo = i & 0xF = i (safe since i < 16), hi = 0
        }
        let block = make_q4_1_block(0.5, 1.0, &nibbles);
        let result = dequant_q4_1_to_f32(&block, 1, 32).expect("dequant");
        assert_eq!(result.len(), 32);

        for i in 0..16usize {
            let lo = (nibbles[i] & 0x0F) as f32;
            let hi = ((nibbles[i] >> 4) & 0x0F) as f32;
            let expected_lo = 0.5 * lo + 1.0;
            let expected_hi = 0.5 * hi + 1.0;
            assert!(
                (result[i * 2] - expected_lo).abs() < 1e-5,
                "weight[{}]: got {}, expected {expected_lo}",
                i * 2,
                result[i * 2],
            );
            assert!(
                (result[i * 2 + 1] - expected_hi).abs() < 1e-5,
                "weight[{}]: got {}, expected {expected_hi}",
                i * 2 + 1,
                result[i * 2 + 1],
            );
        }
    }

    // ── 4. dispatcher returns Some without real GPU ────────────────────────────

    /// The kernel struct satisfies `GpuKernel` — constructible at any time.
    #[test]
    fn test_q4_1_dispatcher_returns_none_without_gpu() {
        // Without an adapter the dispatcher wraps None, so get_kernel returns None.
        // But the kernel type itself must always be constructible.
        let _kernel: &dyn GpuKernel = &Q4_1GpuKernel;
        // Additionally verify the dispatcher is coherent.
        let dispatcher = crate::GpuDispatcher::new();
        let kernel = dispatcher.get_kernel(oxillama_gguf::GgufTensorType::Q4_1);
        if dispatcher.has_gpu() {
            assert!(kernel.is_some(), "Q4_1 kernel must be present when GPU is available");
        } else {
            assert!(kernel.is_none(), "Q4_1 kernel must be absent without GPU");
        }
    }

    // ── 5. two-block roundtrip ─────────────────────────────────────────────────

    /// Concatenate two Q4_1 blocks and verify the output has 2 × BLOCK_SIZE
    /// elements and both blocks decoded correctly.
    #[test]
    fn test_q4_1_two_block_roundtrip() {
        // Block A: d=1.0, m=0.0, all nibbles=0xAA (lo=A=10, hi=A=10)
        let block_a = make_q4_1_block(1.0, 0.0, &[0xAAu8; 16]);
        // Block B: d=2.0, m=1.0, all nibbles=0x55 (lo=5, hi=5)
        let block_b = make_q4_1_block(2.0, 1.0, &[0x55u8; 16]);

        let mut data = Vec::new();
        data.extend_from_slice(&block_a);
        data.extend_from_slice(&block_b);

        // 2 rows × 32 cols — one block per row.
        let result = dequant_q4_1_to_f32(&data, 2, 32).expect("two-block dequant");
        assert_eq!(result.len(), 2 * 32, "must have 2*BLOCK_SIZE elements");

        // Row 0: d=1.0, m=0.0, nibble=10 → w = 10.0
        for &v in &result[..32] {
            assert!((v - 10.0).abs() < 1e-4, "row0 weight: expected 10.0, got {v}");
        }
        // Row 1: d=2.0, m=1.0, nibble=5 → w = 2.0*5 + 1.0 = 11.0
        for &v in &result[32..] {
            assert!((v - 11.0).abs() < 1e-4, "row1 weight: expected 11.0, got {v}");
        }
    }

    /// All-max nibbles: d=1.0, m=0.0, all nibbles=0xFF (lo=15, hi=15) → weight = 15.
    #[test]
    fn test_dequant_q4_1_max_nibble() {
        let block = make_q4_1_block(1.0, 0.0, &[0xFFu8; 16]);
        let result = dequant_q4_1_to_f32(&block, 1, 32).expect("dequant");
        for &v in &result {
            assert!((v - 15.0).abs() < 1e-4, "expected 15.0, got {v}");
        }
    }

    /// Scale-and-min combined: nibbles=0x88 (8), d=0.5, m=0.25 → w = 0.5*8+0.25 = 4.25.
    #[test]
    fn test_dequant_q4_1_scale_and_min() {
        let block = make_q4_1_block(0.5, 0.25, &[0x88u8; 16]);
        let result = dequant_q4_1_to_f32(&block, 1, 32).expect("dequant");
        for &v in &result {
            assert!((v - 4.25).abs() < 1e-4, "expected 4.25, got {v}");
        }
    }

    /// GPU GEMV end-to-end: when GPU is available, result must match CPU ref.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_q4_1_matches_cpu() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let make_block = |d: f32, m: f32, pattern: u8| -> Vec<u8> {
            make_q4_1_block(d, m, &[pattern; 16])
        };

        // Row 0: d=1.0, m=0.0, nibbles=0xAA (10) → weights all 10.0
        // Row 1: d=0.5, m=1.0, nibbles=0x22 (2)  → weights all 2.0
        let mut weight_bytes = Vec::new();
        weight_bytes.extend_from_slice(&make_block(1.0, 0.0, 0xAA));
        weight_bytes.extend_from_slice(&make_block(0.5, 1.0, 0x22));

        let input = vec![1.0f32; 32];

        // CPU ref via dequant_q4_1_to_f32 + dot.
        let f32_weights = dequant_q4_1_to_f32(&weight_bytes, 2, 32).expect("cpu dequant");
        let expected: Vec<f32> = (0..2)
            .map(|r| f32_weights[r * 32..(r + 1) * 32].iter().sum::<f32>())
            .collect();

        let mut output = vec![0.0f32; 2];
        let kernel = Q4_1GpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut output, 2, 32)
            .expect("Q4_1 GPU GEMV");

        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}"
            );
        }
    }
}

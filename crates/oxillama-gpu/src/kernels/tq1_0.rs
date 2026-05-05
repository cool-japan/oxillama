//! TQ1_0 GPU kernel.
//!
//! Strategy:
//!   1. Dequantise `weight_bytes` to f32 on the CPU using the TQ1_0 block
//!      format: 256 weights per 54-byte block.
//!   2. Upload the dequantised f32 matrix and the input vector to the GPU.
//!   3. Dispatch the generic f32 GEMV shader (`gemv_f32.wgsl`).
//!   4. Read back the output.
//!
//! TQ1_0 block layout (54 bytes per 256 weights):
//! - bytes  0-47:  `qs[48]` — 48 bytes of base-3 packed ternary values (5 per byte = 240 values)
//! - bytes 48-51:  `qh[4]`  — 4 bytes of 2-bit ternary codes (4 per byte = 16 values)
//! - bytes 52-53:  d (f16 little-endian scale)
//!
//! Ternary encoding: `{0→-1, 1→0, 2→+1}` (i.e. digit - 1).
//! Weight formula: `w = d * ternary_value`
//!
//! When the `gpu` feature is absent the kernel is a ZST and `gemv` returns
//! `Err(GpuError::NoAdapter)`.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// TQ1_0 GPU kernel — dequantises on CPU, dispatches f32 GEMV on GPU.
pub struct Tq1_0GpuKernel;

impl GpuKernel for Tq1_0GpuKernel {
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
            gpu_gemv_tq1_0(ctx, weight_bytes, input, output, rows, cols)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, weight_bytes, input, output, rows, cols);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── TQ1_0 block constants ────────────────────────────────────────────────────

/// Weights per TQ1_0 block.
#[cfg(any(feature = "gpu", test))]
const TQ1_0_BLOCK_SIZE: usize = 256;
/// Bytes per TQ1_0 block: 48 (qs) + 4 (qh) + 2 (d) = 54.
#[cfg(any(feature = "gpu", test))]
const TQ1_0_BLOCK_BYTES: usize = 54;
/// Number of base-3 packed qs bytes.
#[cfg(any(feature = "gpu", test))]
const TQ1_0_QS_BYTES: usize = 48;
/// Number of 2-bit ternary qh bytes.
#[cfg(any(feature = "gpu", test))]
const TQ1_0_QH_BYTES: usize = 4;
/// Byte offset of `qh`.
#[cfg(any(feature = "gpu", test))]
const TQ1_0_QH_OFFSET: usize = TQ1_0_QS_BYTES; // 48
/// Byte offset of `d` (FP16 scale).
#[cfg(any(feature = "gpu", test))]
const TQ1_0_D_OFFSET: usize = TQ1_0_QS_BYTES + TQ1_0_QH_BYTES; // 52

/// Decode a single `qs` byte into 5 ternary values (-1, 0, or +1).
///
/// The byte encodes 5 base-3 digits: `v[i] = (q / 3^i) % 3 - 1`.
#[cfg(any(feature = "gpu", test))]
#[inline]
fn decode_qs_byte(byte: u8) -> [i8; 5] {
    let mut q = byte as u16;
    let mut out = [0i8; 5];
    for v in &mut out {
        *v = (q % 3) as i8 - 1;
        q /= 3;
    }
    out
}

/// Decode a single `qh` byte into 4 ternary values (-1, 0, or +1).
///
/// Each pair of bits encodes one value: `(bits & 3) - 1`.
#[cfg(any(feature = "gpu", test))]
#[inline]
fn decode_qh_byte(byte: u8) -> [i8; 4] {
    [
        (byte & 0x03) as i8 - 1,
        ((byte >> 2) & 0x03) as i8 - 1,
        ((byte >> 4) & 0x03) as i8 - 1,
        ((byte >> 6) & 0x03) as i8 - 1,
    ]
}

/// Dequantise all TQ1_0 blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
fn dequant_tq1_0_to_f32(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<f32>> {
    let blocks_per_row = cols.div_ceil(TQ1_0_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * TQ1_0_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f32_weights = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let offset = (row * blocks_per_row + blk) * TQ1_0_BLOCK_BYTES;
            let block = &weight_bytes[offset..offset + TQ1_0_BLOCK_BYTES];

            let d = half::f16::from_le_bytes([block[TQ1_0_D_OFFSET], block[TQ1_0_D_OFFSET + 1]])
                .to_f32();

            let weight_base = blk * TQ1_0_BLOCK_SIZE;
            let mut out_idx = weight_base;

            // Decode qs: 48 bytes → 240 ternary values
            for &qs_byte in block.iter().take(TQ1_0_QS_BYTES) {
                let vals = decode_qs_byte(qs_byte);
                for &v in &vals {
                    let col = out_idx - weight_base;
                    if col < cols {
                        f32_weights[row * cols + col] = d * v as f32;
                    }
                    out_idx += 1;
                }
            }

            // Decode qh: 4 bytes → 16 ternary values
            for qh_idx in 0..TQ1_0_QH_BYTES {
                let vals = decode_qh_byte(block[TQ1_0_QH_OFFSET + qh_idx]);
                for &v in &vals {
                    let col = out_idx - weight_base;
                    if col < cols {
                        f32_weights[row * cols + col] = d * v as f32;
                    }
                    out_idx += 1;
                }
            }
        }
    }

    Ok(f32_weights)
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_gemv_tq1_0(
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

    let f32_weights = dequant_tq1_0_to_f32(weight_bytes, rows, cols)?;

    let weight_buf = upload_f32(&ctx.device, "tq1_0-weights", &f32_weights);
    let input_buf = upload_f32(&ctx.device, "tq1_0-input", input);
    let output_buf = create_output_f32(&ctx.device, "tq1_0-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "tq1_0-params", &params);

    const WGSL: &str = include_str!("../shaders/gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f32_tq1_0"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tq1_0-bgl"),
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
            label: Some("tq1_0-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("tq1_0-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("tq1_0-bg"),
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
            label: Some("tq1_0-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("tq1_0-pass"),
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

    /// Encode 5 ternary values (-1, 0, +1) into a single qs byte (base-3).
    fn encode_qs(vals: [i8; 5]) -> u8 {
        let mut byte: u8 = 0;
        let mut multiplier: u8 = 1;
        for &v in &vals {
            let encoded = (v + 1) as u8; // -1→0, 0→1, +1→2
            byte = byte.wrapping_add(encoded.wrapping_mul(multiplier));
            multiplier = multiplier.wrapping_mul(3);
        }
        byte
    }

    /// Encode 4 ternary values (-1, 0, +1) into a single qh byte (2-bit codes).
    fn encode_qh(vals: [i8; 4]) -> u8 {
        let mut byte: u8 = 0;
        for (i, &v) in vals.iter().enumerate() {
            let encoded = (v + 1) as u8;
            byte |= encoded << (i * 2);
        }
        byte
    }

    fn make_tq1_0_block(scale: f32, qs: &[u8; 48], qh: &[u8; 4]) -> Vec<u8> {
        let mut block = Vec::with_capacity(TQ1_0_BLOCK_BYTES);
        block.extend_from_slice(qs);
        block.extend_from_slice(qh);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block
    }

    #[test]
    fn test_dequant_tq1_0_zero_scale() {
        // d=0 → all weights = 0 regardless of ternary values.
        let qs = [encode_qs([1, 1, -1, 0, 1]); 48];
        let qh = [encode_qh([1, -1, 0, 1]); 4];
        let block = make_tq1_0_block(0.0, &qs, &qh);
        let result = dequant_tq1_0_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-7, "weight[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn test_dequant_tq1_0_all_positive() {
        // d=1.0, all ternary +1 → all weights = 1.0
        let qs = [encode_qs([1, 1, 1, 1, 1]); 48];
        let qh = [encode_qh([1, 1, 1, 1]); 4];
        let block = make_tq1_0_block(1.0, &qs, &qh);
        let result = dequant_tq1_0_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-3, "weight[{i}] = {v}, expected 1.0");
        }
    }

    #[test]
    fn test_dequant_tq1_0_all_negative() {
        // d=1.0, all ternary -1 → all weights = -1.0
        let qs = [encode_qs([-1, -1, -1, -1, -1]); 48];
        let qh = [encode_qh([-1, -1, -1, -1]); 4];
        let block = make_tq1_0_block(1.0, &qs, &qh);
        let result = dequant_tq1_0_to_f32(&block, 1, 256).expect("dequant");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-3,
                "weight[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_tq1_0_too_small() {
        assert!(
            dequant_tq1_0_to_f32(&[0u8; 4], 1, 256).is_err(),
            "should fail on too-small input"
        );
    }

    #[test]
    fn test_tq1_0_kernel_trait_bound() {
        let _kernel: &dyn GpuKernel = &Tq1_0GpuKernel;
    }

    #[test]
    fn test_decode_roundtrip_qs() {
        // Verify decode_qs_byte inverts encode_qs.
        for a in -1i8..=1 {
            for b in -1i8..=1 {
                let vals = [a, b, 1, -1, 0];
                let encoded = encode_qs(vals);
                let decoded = decode_qs_byte(encoded);
                assert_eq!(vals, decoded, "qs roundtrip failed for {vals:?}");
            }
        }
    }

    #[test]
    fn test_decode_roundtrip_qh() {
        // Verify decode_qh_byte inverts encode_qh.
        for a in -1i8..=1 {
            for b in -1i8..=1 {
                let vals = [a, b, -1, 1];
                let encoded = encode_qh(vals);
                let decoded = decode_qh_byte(encoded);
                assert_eq!(vals, decoded, "qh roundtrip failed for {vals:?}");
            }
        }
    }

    /// End-to-end GPU GEMV: dequant+dot must match within 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_gemv_tq1_0_matches_cpu_reference() {
        let ctx = match crate::context::GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let rows = 32;
        let cols = 256;

        let mut weight_bytes = Vec::with_capacity(rows * TQ1_0_BLOCK_BYTES);
        for r in 0..rows {
            let mut qs = [0u8; 48];
            for (i, byte) in qs.iter_mut().enumerate() {
                let pattern: [i8; 5] = match (r + i) % 3 {
                    0 => [-1, 0, 1, -1, 0],
                    1 => [1, 1, -1, 0, 0],
                    _ => [0, -1, 1, 1, -1],
                };
                *byte = encode_qs(pattern);
            }
            let mut qh = [0u8; 4];
            for (i, byte) in qh.iter_mut().enumerate() {
                let pattern: [i8; 4] = match (r + i) % 2 {
                    0 => [1, -1, 0, 1],
                    _ => [-1, 0, 1, -1],
                };
                *byte = encode_qh(pattern);
            }
            let block = make_tq1_0_block(0.5 + r as f32 * 0.01, &qs, &qh);
            weight_bytes.extend_from_slice(&block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let f32_weights = dequant_tq1_0_to_f32(&weight_bytes, rows, cols).expect("cpu dequant");
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
        let kernel = Tq1_0GpuKernel;
        kernel
            .gemv(&ctx, &weight_bytes, &input, &mut result, rows, cols)
            .expect("GPU GEMV TQ1_0");

        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "row {i}: got {got}, expected {want}, diff {}",
                (got - want).abs()
            );
        }
    }
}

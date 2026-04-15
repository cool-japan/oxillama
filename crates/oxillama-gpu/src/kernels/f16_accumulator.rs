//! Half-precision (f16) accumulation path for GPU kernels.
//!
//! Provides f16 dequantization utilities and a WGSL compute shader that
//! operates entirely in f16, halving memory bandwidth vs the f32 path.
//! Falls back to f32 if the GPU does not support shader-f16.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};

// ─── Q4_0 / Q8_0 block constants ─────────────────────────────────────────────

/// Weights per Q4_0 block.
#[cfg(any(feature = "gpu", test))]
const Q4_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q4_0 block: 2 (f16 scale) + 16 (nibble pairs for 32 weights).
#[cfg(any(feature = "gpu", test))]
const Q4_0_BLOCK_BYTES: usize = 18;

/// Weights per Q8_0 block.
#[cfg(any(feature = "gpu", test))]
const Q8_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (i8 quantised values).
#[cfg(any(feature = "gpu", test))]
const Q8_0_BLOCK_BYTES: usize = 34;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the f16 accumulator.
#[derive(Debug, Clone, Default)]
pub struct F16AccumulatorConfig {
    /// Whether to force f32 fallback even if f16 is available.
    pub force_f32: bool,
}

// ─── Feature detection ────────────────────────────────────────────────────────

/// Check if the GPU context supports f16 shader operations.
///
/// Returns `true` if the device supports the `SHADER_F16` feature.
/// Always returns `false` when the `gpu` feature is disabled.
pub fn supports_f16(_ctx: &GpuContext) -> bool {
    #[cfg(feature = "gpu")]
    {
        _ctx.device.features().contains(wgpu::Features::SHADER_F16)
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

// ─── f16 dequantisation ──────────────────────────────────────────────────────

/// Dequantize Q4_0 blocks to f16 (half the memory of f32 path).
///
/// Returns a `Vec<u16>` where each element is the bit-representation of an
/// `f16` weight value, suitable for GPU upload as packed u32 pairs.
#[cfg(any(feature = "gpu", test))]
pub fn dequant_q4_0_to_f16(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<u16>> {
    let blocks_per_row = cols.div_ceil(Q4_0_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q4_0_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f16_weights = vec![0u16; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q4_0_BLOCK_BYTES;
            let block = &weight_bytes[block_offset..block_offset + Q4_0_BLOCK_BYTES];

            let scale_bits = u16::from_le_bytes([block[0], block[1]]);
            let scale_f32 = half::f16::from_bits(scale_bits).to_f32();

            for i in 0..(Q4_0_BLOCK_SIZE / 2) {
                let byte = block[2 + i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;

                let base_col = blk * Q4_0_BLOCK_SIZE + i * 2;
                if base_col < cols {
                    f16_weights[row * cols + base_col] =
                        half::f16::from_f32(lo as f32 * scale_f32).to_bits();
                }
                if base_col + 1 < cols {
                    f16_weights[row * cols + base_col + 1] =
                        half::f16::from_f32(hi as f32 * scale_f32).to_bits();
                }
            }
        }
    }

    Ok(f16_weights)
}

/// Dequantize Q8_0 blocks to f16.
///
/// Returns a `Vec<u16>` where each element is the bit-representation of an
/// `f16` weight value, suitable for GPU upload as packed u32 pairs.
#[cfg(any(feature = "gpu", test))]
pub fn dequant_q8_0_to_f16(weight_bytes: &[u8], rows: usize, cols: usize) -> GpuResult<Vec<u16>> {
    let blocks_per_row = cols.div_ceil(Q8_0_BLOCK_SIZE);
    let expected_bytes = rows * blocks_per_row * Q8_0_BLOCK_BYTES;
    if weight_bytes.len() < expected_bytes {
        return Err(GpuError::BufferSize {
            expected: expected_bytes,
            got: weight_bytes.len(),
        });
    }

    let mut f16_weights = vec![0u16; rows * cols];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q8_0_BLOCK_BYTES;
            let block = &weight_bytes[block_offset..block_offset + Q8_0_BLOCK_BYTES];

            let scale_bits = u16::from_le_bytes([block[0], block[1]]);
            let scale_f32 = half::f16::from_bits(scale_bits).to_f32();

            for i in 0..Q8_0_BLOCK_SIZE {
                let col = blk * Q8_0_BLOCK_SIZE + i;
                if col < cols {
                    let q = block[2 + i] as i8;
                    f16_weights[row * cols + col] =
                        half::f16::from_f32(q as f32 * scale_f32).to_bits();
                }
            }
        }
    }

    Ok(f16_weights)
}

// ─── GPU buffer upload ────────────────────────────────────────────────────────

/// Upload f16 data (as `u16` bits) to a GPU storage buffer.
///
/// The buffer is created with `STORAGE | COPY_SRC` usage, suitable for
/// read-only shader storage bindings.
#[cfg(feature = "gpu")]
pub fn upload_f16(device: &wgpu::Device, label: &str, data: &[u16]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

// ─── Mixed-precision GEMV ─────────────────────────────────────────────────────

/// Pack f16 values (as u16 bits) into u32 pairs for the WGSL shader.
///
/// Each u32 stores two adjacent f16 values as `(hi << 16) | lo`, matching
/// the layout expected by `unpack2x16float()`.
#[cfg(any(feature = "gpu", test))]
fn pack_f16_to_u32(f16_data: &[u16]) -> Vec<u32> {
    let packed_len = f16_data.len().div_ceil(2);
    let mut packed = Vec::with_capacity(packed_len);

    let mut i = 0;
    while i + 1 < f16_data.len() {
        let lo = f16_data[i] as u32;
        let hi = f16_data[i + 1] as u32;
        packed.push(lo | (hi << 16));
        i += 2;
    }
    // Handle odd trailing element.
    if i < f16_data.len() {
        packed.push(f16_data[i] as u32);
    }

    packed
}

/// Create a GEMV pipeline using f16 weights with f32 input/output.
///
/// The shader loads packed f16 weights (as u32), unpacks via
/// `unpack2x16float()`, multiplies by f32 input, and accumulates in f32.
/// This is a "mixed precision" path that halves weight-memory bandwidth.
///
/// Returns `Err(GpuError::NoAdapter)` when the `gpu` feature is absent.
#[cfg(feature = "gpu")]
pub fn f16_gemv(
    ctx: &GpuContext,
    weight_f16: &[u16],
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

    // Pack f16 weights into u32 pairs for the shader.
    let packed_weights = pack_f16_to_u32(weight_f16);

    // Upload buffers.
    let weight_buf = upload_f16_packed(&ctx.device, "f16-weights", &packed_weights);
    let input_buf = upload_f32(&ctx.device, "f16-input", input);
    let output_buf = create_output_f32(&ctx.device, "f16-output", rows);

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
    let params_buf = upload_uniform(&ctx.device, "f16-params", &params);

    // Build compute pipeline using the f16 GEMV shader.
    const WGSL: &str = include_str!("../shaders/gemv_f16.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemv_f16"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("f16-bgl"),
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
            label: Some("f16-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("f16-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("f16-bg"),
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

    // Dispatch — workgroup size 256.
    let dispatch_x = rows.div_ceil(256) as u32;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("f16-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("f16-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }
    ctx.queue.submit([encoder.finish()]);

    // Read back results.
    let result = download_f32(&ctx.device, &ctx.queue, &output_buf, rows)?;
    output[..rows].copy_from_slice(&result[..rows]);

    Ok(())
}

/// Non-gpu fallback — always returns `Err(GpuError::NoAdapter)`.
#[cfg(not(feature = "gpu"))]
pub fn f16_gemv(
    _ctx: &GpuContext,
    _weight_f16: &[u16],
    _input: &[f32],
    _output: &mut [f32],
    _rows: usize,
    _cols: usize,
) -> GpuResult<()> {
    Err(GpuError::NoAdapter)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Upload packed u32 data to a GPU storage buffer (STORAGE | COPY_SRC).
#[cfg(feature = "gpu")]
fn upload_f16_packed(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

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

    // ── Q4_0 → f16 dequantisation ───────────────────────────────────────────

    /// Build a minimal Q4_0 block: 2-byte f16 scale + 16 bytes nibbles.
    fn make_q4_0_block(scale: f32, nibbles: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q4_0_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(nibbles);
        block
    }

    #[test]
    fn test_dequant_q4_0_to_f16_zeros() {
        // All nibbles = 0x88 → both lo and hi = (8 - 8) = 0.
        let block = make_q4_0_block(1.0, &[0x88u8; 16]);
        let result =
            dequant_q4_0_to_f16(&block, 1, Q4_0_BLOCK_SIZE).expect("dequant should succeed");
        assert_eq!(result.len(), Q4_0_BLOCK_SIZE);
        for bits in &result {
            let val = half::f16::from_bits(*bits).to_f32();
            assert!(val.abs() < 1e-3, "expected zero, got {val}");
        }
    }

    #[test]
    fn test_dequant_q4_0_to_f16_known_values() {
        // scale = 2.0, first nibble byte = 0x9A → lo = (0xA - 8) = 2, hi = (0x9 - 8) = 1
        // expected: weight[0] = 2 * 2.0 = 4.0, weight[1] = 1 * 2.0 = 2.0
        let mut nibbles = [0x88u8; 16];
        nibbles[0] = 0x9A;
        let block = make_q4_0_block(2.0, &nibbles);
        let result =
            dequant_q4_0_to_f16(&block, 1, Q4_0_BLOCK_SIZE).expect("dequant should succeed");

        let w0 = half::f16::from_bits(result[0]).to_f32();
        let w1 = half::f16::from_bits(result[1]).to_f32();
        assert!((w0 - 4.0).abs() < 0.05, "expected ~4.0, got {w0}");
        assert!((w1 - 2.0).abs() < 0.05, "expected ~2.0, got {w1}");
    }

    #[test]
    fn test_dequant_q4_0_to_f16_negative() {
        // scale = 1.0, nibble byte = 0x35 → lo = (5 - 8) = -3, hi = (3 - 8) = -5
        let mut nibbles = [0x88u8; 16];
        nibbles[0] = 0x35;
        let block = make_q4_0_block(1.0, &nibbles);
        let result =
            dequant_q4_0_to_f16(&block, 1, Q4_0_BLOCK_SIZE).expect("dequant should succeed");

        let w0 = half::f16::from_bits(result[0]).to_f32();
        let w1 = half::f16::from_bits(result[1]).to_f32();
        assert!((w0 - (-3.0)).abs() < 0.05, "expected ~-3.0, got {w0}");
        assert!((w1 - (-5.0)).abs() < 0.05, "expected ~-5.0, got {w1}");
    }

    #[test]
    fn test_dequant_q4_0_to_f16_buffer_too_small() {
        let data = vec![0u8; 10]; // Too small for even one block
        let result = dequant_q4_0_to_f16(&data, 1, Q4_0_BLOCK_SIZE);
        assert!(result.is_err());
        match result {
            Err(GpuError::BufferSize { expected, got }) => {
                assert_eq!(expected, Q4_0_BLOCK_BYTES);
                assert_eq!(got, 10);
            }
            other => panic!("expected BufferSize error, got {other:?}"),
        }
    }

    // ── Q8_0 → f16 dequantisation ───────────────────────────────────────────

    /// Build a minimal Q8_0 block: 2-byte f16 scale + 32 × i8 quants.
    fn make_q8_0_block(scale: f32, quants: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_0_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &q in quants {
            block.push(q as u8);
        }
        block
    }

    #[test]
    fn test_dequant_q8_0_to_f16_zeros() {
        let quants = [0i8; 32];
        let block = make_q8_0_block(1.0, &quants);
        let result =
            dequant_q8_0_to_f16(&block, 1, Q8_0_BLOCK_SIZE).expect("dequant should succeed");
        assert_eq!(result.len(), Q8_0_BLOCK_SIZE);
        for bits in &result {
            let val = half::f16::from_bits(*bits).to_f32();
            assert!(val.abs() < 1e-3, "expected zero, got {val}");
        }
    }

    #[test]
    fn test_dequant_q8_0_to_f16_known_values() {
        // scale = 0.5, quants[0] = 10 → 10 * 0.5 = 5.0
        // quants[1] = -4 → -4 * 0.5 = -2.0
        let mut quants = [0i8; 32];
        quants[0] = 10;
        quants[1] = -4;
        let block = make_q8_0_block(0.5, &quants);
        let result =
            dequant_q8_0_to_f16(&block, 1, Q8_0_BLOCK_SIZE).expect("dequant should succeed");

        let w0 = half::f16::from_bits(result[0]).to_f32();
        let w1 = half::f16::from_bits(result[1]).to_f32();
        assert!((w0 - 5.0).abs() < 0.05, "expected ~5.0, got {w0}");
        assert!((w1 - (-2.0)).abs() < 0.05, "expected ~-2.0, got {w1}");
    }

    #[test]
    fn test_dequant_q8_0_to_f16_buffer_too_small() {
        let data = vec![0u8; 20]; // Too small for one Q8_0 block
        let result = dequant_q8_0_to_f16(&data, 1, Q8_0_BLOCK_SIZE);
        assert!(result.is_err());
        match result {
            Err(GpuError::BufferSize { expected, got }) => {
                assert_eq!(expected, Q8_0_BLOCK_BYTES);
                assert_eq!(got, 20);
            }
            other => panic!("expected BufferSize error, got {other:?}"),
        }
    }

    // ── Cross-check: f16 vs f32 dequant ─────────────────────────────────────

    #[test]
    fn test_dequant_q4_0_f16_matches_f32() {
        // Build a block with varied nibbles and check f16 values are close to f32.
        let nibbles: [u8; 16] = [
            0x9A, 0x35, 0xBC, 0x71, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
            0x88, 0x88,
        ];
        let block = make_q4_0_block(0.25, &nibbles);

        let f16_vals =
            dequant_q4_0_to_f16(&block, 1, Q4_0_BLOCK_SIZE).expect("f16 dequant should succeed");

        // Manually dequant to f32 for comparison.
        let scale = half::f16::from_f32(0.25).to_f32();
        let expected_f32: Vec<f32> = {
            let mut out = vec![0.0f32; Q4_0_BLOCK_SIZE];
            for i in 0..16 {
                let byte = nibbles[i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                out[i * 2] = lo as f32 * scale;
                out[i * 2 + 1] = hi as f32 * scale;
            }
            out
        };

        for (idx, &bits) in f16_vals.iter().enumerate() {
            let f16_val = half::f16::from_bits(bits).to_f32();
            let diff = (f16_val - expected_f32[idx]).abs();
            assert!(
                diff < 0.01,
                "index {idx}: f16={f16_val}, f32={}, diff={diff}",
                expected_f32[idx]
            );
        }
    }

    // ── Config / feature detection ──────────────────────────────────────────

    #[test]
    fn test_f16_accumulator_config_default() {
        let config = F16AccumulatorConfig::default();
        assert!(!config.force_f32);
    }

    #[test]
    fn test_f16_accumulator_config_force_f32() {
        let config = F16AccumulatorConfig { force_f32: true };
        assert!(config.force_f32);
    }

    #[test]
    fn test_supports_f16_without_gpu() {
        // Without actual GPU hardware, supports_f16 should return false.
        // We can only test this when GpuContext is available.
        if let Some(ctx) = GpuContext::try_init() {
            // Result depends on hardware — just verify it doesn't panic.
            let _ = supports_f16(&ctx);
        }
    }

    // ── Packing ─────────────────────────────────────────────────────────────

    #[test]
    fn test_pack_f16_to_u32_even() {
        let data: Vec<u16> = vec![0x3C00, 0x4000, 0x4200, 0x4400]; // 1.0, 2.0, 3.0, 4.0
        let packed = pack_f16_to_u32(&data);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0] & 0xFFFF, 0x3C00);
        assert_eq!(packed[0] >> 16, 0x4000);
        assert_eq!(packed[1] & 0xFFFF, 0x4200);
        assert_eq!(packed[1] >> 16, 0x4400);
    }

    #[test]
    fn test_pack_f16_to_u32_odd() {
        let data: Vec<u16> = vec![0x3C00, 0x4000, 0x4200];
        let packed = pack_f16_to_u32(&data);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0] & 0xFFFF, 0x3C00);
        assert_eq!(packed[0] >> 16, 0x4000);
        // Trailing element: just the low 16 bits.
        assert_eq!(packed[1], 0x4200);
    }

    #[test]
    fn test_pack_f16_to_u32_empty() {
        let data: Vec<u16> = vec![];
        let packed = pack_f16_to_u32(&data);
        assert!(packed.is_empty());
    }

    #[test]
    fn test_pack_f16_to_u32_single() {
        let data: Vec<u16> = vec![0xBE00]; // -1.5
        let packed = pack_f16_to_u32(&data);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0xBE00);
    }
}

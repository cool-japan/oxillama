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

/// Ternary digits packed per `qs` byte (base-3, 3^5 = 243 ≤ 256).
#[cfg(any(feature = "gpu", test))]
const TQ1_0_QS_DIGITS: usize = 5;
/// Ternary digits packed per `qh` byte (base-3, shifted up one trit).
#[cfg(any(feature = "gpu", test))]
const TQ1_0_QH_DIGITS: usize = 4;

/// Powers of three used by the fixed-point base-3 decode, as `u8` exactly as
/// upstream declares them (`static const uint8_t pow3[6]`).
#[cfg(any(feature = "gpu", test))]
const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Recover ternary digit `digit` from a packed TQ1_0 byte.
///
/// TQ1_0 is **not** a plain base-3 packing: `quantize_row_tq1_0_ref` builds
/// the 5-digit value most-significant-digit-first,
/// `q = Σ_n xi_n · 3^(4-n)` in `0..=242`, and stores the scaled ceiling
/// `ceil(q · 256 / 243)`.  This is a literal port of upstream's decode:
///
/// ```c
/// uint8_t q  = x[i].qs[j + m] * pow3[n];   // uint8_t: wraps mod 256
/// int16_t xi = ((uint16_t) q * 3) >> 8;    // leading base-3 digit
/// *y++ = (float) (xi - 1) * d;
/// ```
///
/// Decoding the stored byte with naive `% 3` / `/ 3` arithmetic (the
/// previous implementation here) produces *different values*, not merely a
/// different order — see `oxillama_quant::reference::tq1_0`'s module doc for
/// the worked example.  `qh` uses the same scheme with the four digits
/// shifted up one trit.
#[cfg(any(feature = "gpu", test))]
#[inline]
fn decode_trit(byte: u8, digit: usize) -> i8 {
    let q = byte.wrapping_mul(POW3[digit]);
    ((((q as u16) * 3) >> 8) as i8) - 1
}

/// Dequantise all TQ1_0 blocks to a flat f32 buffer.
#[cfg(any(feature = "gpu", test))]
pub(crate) fn dequant_tq1_0_to_f32(
    weight_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> GpuResult<Vec<f32>> {
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
            // Block-local output index, 0..256.  Kept separate from
            // `weight_base` (unlike the previous implementation, which
            // computed `col = out_idx - weight_base` and therefore always
            // canceled back to `0..256` regardless of `blk` — silently
            // aliasing every block past the first onto columns `0..256`).
            let mut local_idx = 0usize;

            // Decode qs: 48 bytes → 240 ternary values, digit-major within
            // each group.  48 = one 32-byte group (bytes 0..32, digit-major
            // over 32 → 160 values) + one 16-byte group (bytes 32..48,
            // digit-major over 16 → 80 values), matching upstream's
            // `sizeof(qs) - sizeof(qs) % 32` split.
            let mut j = 0usize;
            let qs_head = TQ1_0_QS_BYTES - TQ1_0_QS_BYTES % 32; // 32
            while j < qs_head {
                for digit in 0..TQ1_0_QS_DIGITS {
                    for m in 0..32 {
                        let col = weight_base + local_idx;
                        if col < cols {
                            let v = decode_trit(block[j + m], digit);
                            f32_weights[row * cols + col] = d * v as f32;
                        }
                        local_idx += 1;
                    }
                }
                j += 32;
            }
            while j < TQ1_0_QS_BYTES {
                for digit in 0..TQ1_0_QS_DIGITS {
                    for m in 0..16 {
                        let col = weight_base + local_idx;
                        if col < cols {
                            let v = decode_trit(block[j + m], digit);
                            f32_weights[row * cols + col] = d * v as f32;
                        }
                        local_idx += 1;
                    }
                }
                j += 16;
            }

            // Decode qh: 4 bytes → 16 ternary values, also digit-major.
            for digit in 0..TQ1_0_QH_DIGITS {
                for m in 0..TQ1_0_QH_BYTES {
                    let col = weight_base + local_idx;
                    if col < cols {
                        let v = decode_trit(block[TQ1_0_QH_OFFSET + m], digit);
                        f32_weights[row * cols + col] = d * v as f32;
                    }
                    local_idx += 1;
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

    /// Encode 5 ternary values into one `qs` byte — literal port of the inner
    /// loop of upstream `quantize_row_tq1_0_ref`.  `vals[n]` becomes digit
    /// `n`, i.e. the value [`decode_trit`] recovers with
    /// `decode_trit(byte, n)`.  Digits accumulate most-significant-first and
    /// the result is the fixed-point ceiling `ceil(q · 256 / 243)`, which is
    /// what makes the `(·3) >> 8` decode exact (see [`decode_trit`]'s doc).
    fn encode_qs(vals: [i8; 5]) -> u8 {
        let mut q: u8 = 0;
        for &v in &vals {
            q = q * 3 + (v + 1) as u8;
        }
        ((q as u16) * 256).div_ceil(243) as u8
    }

    /// Encode 4 ternary values into one `qh` byte — the same base-3 scheme
    /// as [`encode_qs`] but with the four digits shifted up one position
    /// (`q *= 3`), leaving the least-significant trit unused, matching
    /// upstream's `qh` encode loop.
    fn encode_qh(vals: [i8; 4]) -> u8 {
        let mut q: u8 = 0;
        for &v in &vals {
            q = q * 3 + (v + 1) as u8;
        }
        q *= 3;
        ((q as u16) * 256).div_ceil(243) as u8
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
        // Verify decode_trit(.., n) for n in 0..5 inverts encode_qs.
        for a in -1i8..=1 {
            for b in -1i8..=1 {
                let vals = [a, b, 1, -1, 0];
                let encoded = encode_qs(vals);
                let decoded: [i8; 5] = std::array::from_fn(|n| decode_trit(encoded, n));
                assert_eq!(vals, decoded, "qs roundtrip failed for {vals:?}");
            }
        }
    }

    #[test]
    fn test_decode_roundtrip_qh() {
        // Verify decode_trit(.., n) for n in 0..4 inverts encode_qh.
        for a in -1i8..=1 {
            for b in -1i8..=1 {
                let vals = [a, b, -1, 1];
                let encoded = encode_qh(vals);
                let decoded: [i8; 4] = std::array::from_fn(|n| decode_trit(encoded, n));
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

//! Batched GEMV — process multiple input vectors against a single matrix.
//!
//! `Y[batch, row] = Σ_col A[row, col] × X[batch, col]`
//!
//! This is the key operation for efficient prefill: instead of running GEMV
//! once per token, we batch all prompt tokens into a single dispatch.

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernels::GpuKernel;

/// Batched GEMV configuration.
#[derive(Debug, Clone)]
pub struct BatchedGemvConfig {
    /// Number of matrix rows (output dimension per vector).
    pub rows: usize,
    /// Number of matrix columns (input dimension per vector).
    pub cols: usize,
    /// Batch size (number of input vectors).
    pub batch_size: usize,
}

/// Execute a batched f32 GEMV on the GPU.
///
/// Computes `Y[b, r] = Σ_c(A[r, c] × X[b, c])` for all batch indices `b`.
///
/// # Arguments
/// * `ctx` — GPU context (device + queue)
/// * `matrix_f32` — Row-major matrix A of shape `[rows, cols]`
/// * `vectors_f32` — Row-major input vectors X of shape `[batch_size, cols]`
/// * `config` — Batch dimensions
///
/// # Returns
/// Output vectors Y of shape `[batch_size, rows]` as `Vec<f32>`.
pub fn batched_gemv_f32(
    ctx: &GpuContext,
    matrix_f32: &[f32],
    vectors_f32: &[f32],
    config: &BatchedGemvConfig,
) -> GpuResult<Vec<f32>> {
    #[cfg(feature = "gpu")]
    {
        gpu_batched_gemv_f32(ctx, matrix_f32, vectors_f32, config)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (ctx, matrix_f32, vectors_f32, config);
        Err(GpuError::NoAdapter)
    }
}

#[cfg(feature = "gpu")]
fn gpu_batched_gemv_f32(
    ctx: &GpuContext,
    matrix_f32: &[f32],
    vectors_f32: &[f32],
    config: &BatchedGemvConfig,
) -> GpuResult<Vec<f32>> {
    use crate::buffer::{create_output_f32, download_f32, upload_f32, upload_uniform};
    use bytemuck::{Pod, Zeroable};
    use wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    };

    let BatchedGemvConfig {
        rows,
        cols,
        batch_size,
    } = *config;

    // ── dimension validation ─────────────────────────────────────────────
    let expected_matrix = rows * cols;
    if matrix_f32.len() != expected_matrix {
        return Err(GpuError::BufferSize {
            expected: expected_matrix,
            got: matrix_f32.len(),
        });
    }
    let expected_vectors = batch_size * cols;
    if vectors_f32.len() != expected_vectors {
        return Err(GpuError::BufferSize {
            expected: expected_vectors,
            got: vectors_f32.len(),
        });
    }
    if rows == 0 || cols == 0 || batch_size == 0 {
        return Ok(vec![0.0f32; batch_size * rows]);
    }

    // ── upload buffers ───────────────────────────────────────────────────
    let matrix_buf = upload_f32(&ctx.device, "batched-gemv-matrix", matrix_f32);
    let vectors_buf = upload_f32(&ctx.device, "batched-gemv-vectors", vectors_f32);
    let output_len = batch_size * rows;
    let output_buf = create_output_f32(&ctx.device, "batched-gemv-output", output_len);

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Params {
        rows: u32,
        cols: u32,
        batch_size: u32,
        _pad: u32,
    }
    let params = Params {
        rows: rows as u32,
        cols: cols as u32,
        batch_size: batch_size as u32,
        _pad: 0,
    };
    let params_buf = upload_uniform(&ctx.device, "batched-gemv-params", &params);

    // ── shader + pipeline ────────────────────────────────────────────────
    const WGSL: &str = include_str!("../shaders/batched_gemv_f32.wgsl");
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("batched_gemv_f32"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("batched-gemv-bgl"),
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
            label: Some("batched-gemv-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("batched-gemv-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("batched-gemv-bg"),
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: matrix_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: vectors_buf.as_entire_binding(),
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

    // ── dispatch ─────────────────────────────────────────────────────────
    // workgroup_size(64,1,1) → dispatch (ceil(rows/64), batch_size, 1)
    let dispatch_x = rows.div_ceil(64) as u32;
    let dispatch_y = batch_size as u32;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched-gemv-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("batched-gemv-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    ctx.queue.submit([encoder.finish()]);

    // ── read back ────────────────────────────────────────────────────────
    download_f32(&ctx.device, &ctx.queue, &output_buf, output_len)
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

// ─── BatchedGpuKernel trait ──────────────────────────────────────────────────

/// Trait for GPU kernels that support batched GEMV.
///
/// A batched GEMV multiplies one quantised matrix against multiple input
/// vectors in a single GPU dispatch.  This is the hot path during prefill.
pub trait BatchedGpuKernel: GpuKernel {
    /// Compute batched GEMV: `output[b*rows + r] = Σ_c weight[r*cols+c] × vectors[b*cols+c]`.
    ///
    /// - `quant_data` — raw quantised weight bytes for the matrix.
    /// - `vectors`    — flattened `[batch_size, cols]` f32 input.
    /// - `rows`       — number of matrix rows (output dimension per vector).
    /// - `cols`       — number of matrix columns (input dimension per vector).
    /// - `batch_size` — number of input vectors.
    ///
    /// Returns a flat `[batch_size, rows]` output.
    fn batched_gemv(
        &self,
        ctx: &GpuContext,
        quant_data: &[u8],
        vectors: &[f32],
        rows: usize,
        cols: usize,
        batch_size: usize,
    ) -> GpuResult<Vec<f32>>;
}

// ─── Q4_0 batched GEMV ──────────────────────────────────────────────────────

use crate::kernels::q4_0::Q4_0GpuKernel;

impl BatchedGpuKernel for Q4_0GpuKernel {
    fn batched_gemv(
        &self,
        ctx: &GpuContext,
        quant_data: &[u8],
        vectors: &[f32],
        rows: usize,
        cols: usize,
        batch_size: usize,
    ) -> GpuResult<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            // Dequantise Q4_0 on CPU, then dispatch batched f32 GEMV.
            let f32_weights = crate::kernels::q4_0::dequant_q4_0_to_f32(quant_data, rows, cols)?;
            let config = BatchedGemvConfig {
                rows,
                cols,
                batch_size,
            };
            batched_gemv_f32(ctx, &f32_weights, vectors, &config)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ctx, quant_data, vectors, rows, cols, batch_size);
            Err(GpuError::NoAdapter)
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference: Y = A × X  (row-major, standard matmul).
    #[cfg(feature = "gpu")]
    fn cpu_batched_gemv(
        matrix: &[f32],
        vectors: &[f32],
        rows: usize,
        cols: usize,
        batch: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * rows];
        for b in 0..batch {
            for r in 0..rows {
                let mut acc = 0.0f32;
                for c in 0..cols {
                    acc += matrix[r * cols + c] * vectors[b * cols + c];
                }
                out[b * rows + r] = acc;
            }
        }
        out
    }

    #[test]
    fn test_batched_gemv_no_gpu_graceful() {
        // Without a GPU this must not panic.
        let _ctx = GpuContext::try_init();
        // If we do have a GPU, the function tests below will cover it.
    }

    #[cfg(feature = "gpu")]
    fn try_gpu_ctx() -> Option<GpuContext> {
        GpuContext::try_init()
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_batched_gemv_identity_batch1() {
        let ctx = match try_gpu_ctx() {
            Some(c) => c,
            None => return,
        };
        // 4×4 identity matrix, batch_size=1, vector=[1,2,3,4]
        let rows = 4;
        let cols = 4;
        let batch = 1;
        #[rustfmt::skip]
        let matrix = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let vectors = vec![1.0, 2.0, 3.0, 4.0];
        let config = BatchedGemvConfig {
            rows,
            cols,
            batch_size: batch,
        };
        let result = batched_gemv_f32(&ctx, &matrix, &vectors, &config)
            .expect("batched GEMV should succeed");
        assert_eq!(result.len(), batch * rows);
        for (i, (&got, &want)) in result.iter().zip(vectors.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "element {i}: got {got}, expected {want}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_batched_gemv_identity_batch4() {
        let ctx = match try_gpu_ctx() {
            Some(c) => c,
            None => return,
        };
        let rows = 4;
        let cols = 4;
        let batch = 4;
        #[rustfmt::skip]
        let matrix = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let vectors = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 0.5, 0.5, 0.5, -1.0, -2.0, -3.0, -4.0,
        ];
        let config = BatchedGemvConfig {
            rows,
            cols,
            batch_size: batch,
        };
        let result = batched_gemv_f32(&ctx, &matrix, &vectors, &config)
            .expect("batched GEMV should succeed");
        assert_eq!(result.len(), batch * rows);
        // Identity → output == input
        for (i, (&got, &want)) in result.iter().zip(vectors.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "element {i}: got {got}, expected {want}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_batched_gemv_known_values() {
        let ctx = match try_gpu_ctx() {
            Some(c) => c,
            None => return,
        };
        // 2×3 matrix, batch_size=2
        let rows = 2;
        let cols = 3;
        let batch = 2;
        #[rustfmt::skip]
        let matrix = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let vectors = vec![
            1.0, 0.0, 0.0, // batch 0: selects column 0
            0.0, 1.0, 0.0, // batch 1: selects column 1
        ];
        let expected = cpu_batched_gemv(&matrix, &vectors, rows, cols, batch);
        // batch 0: [1, 4]   batch 1: [2, 5]
        let config = BatchedGemvConfig {
            rows,
            cols,
            batch_size: batch,
        };
        let result = batched_gemv_f32(&ctx, &matrix, &vectors, &config)
            .expect("batched GEMV should succeed");
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "element {i}: got {got}, expected {want}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_batched_gemv_batch1_matches_single() {
        let ctx = match try_gpu_ctx() {
            Some(c) => c,
            None => return,
        };
        // 3×4 random-ish matrix, single vector
        let rows = 3;
        let cols = 4;
        #[rustfmt::skip]
        let matrix = vec![
            0.5, -1.0, 2.0, 0.3,
            1.0,  0.0, -0.5, 1.2,
            -0.3, 0.7, 0.1, -0.9,
        ];
        let vector = vec![1.0, 2.0, 3.0, 4.0];

        // Single GEMV via GpuKernel trait
        let kernel = Q4_0GpuKernel;
        // We can't use Q4_0 here directly (needs quant data). Just use CPU ref.
        let expected = cpu_batched_gemv(&matrix, &vector, rows, cols, 1);

        // Batched with batch_size=1
        let config = BatchedGemvConfig {
            rows,
            cols,
            batch_size: 1,
        };
        let result =
            batched_gemv_f32(&ctx, &matrix, &vector, &config).expect("batched GEMV should succeed");
        let _ = kernel; // suppress unused warning

        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "element {i}: got {got}, expected {want}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_batched_gemv_various_batch_sizes() {
        let ctx = match try_gpu_ctx() {
            Some(c) => c,
            None => return,
        };
        let rows = 8;
        let cols = 16;
        // Simple matrix: A[r,c] = (r * cols + c) as f32 * 0.01
        let matrix: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.01).collect();

        for batch_size in [1, 2, 4, 8] {
            let vectors: Vec<f32> = (0..batch_size * cols)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
                .collect();
            let expected = cpu_batched_gemv(&matrix, &vectors, rows, cols, batch_size);
            let config = BatchedGemvConfig {
                rows,
                cols,
                batch_size,
            };
            let result = batched_gemv_f32(&ctx, &matrix, &vectors, &config)
                .unwrap_or_else(|e| panic!("batch_size={batch_size}: {e}"));
            assert_eq!(result.len(), expected.len(), "batch_size={batch_size}");
            for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-3,
                    "batch_size={batch_size} element {i}: got {got}, expected {want}"
                );
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_batched_gemv_dimension_validation_matrix() {
        let ctx = match try_gpu_ctx() {
            Some(c) => c,
            None => return,
        };
        let config = BatchedGemvConfig {
            rows: 4,
            cols: 4,
            batch_size: 1,
        };
        // Matrix too short
        let result = batched_gemv_f32(&ctx, &[1.0; 10], &[1.0; 4], &config);
        assert!(result.is_err(), "should reject matrix with wrong size");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_batched_gemv_dimension_validation_vectors() {
        let ctx = match try_gpu_ctx() {
            Some(c) => c,
            None => return,
        };
        let config = BatchedGemvConfig {
            rows: 4,
            cols: 4,
            batch_size: 2,
        };
        // Vectors too short (need 2*4=8, only 4)
        let result = batched_gemv_f32(&ctx, &[1.0; 16], &[1.0; 4], &config);
        assert!(result.is_err(), "should reject vectors with wrong size");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_q4_0_batched_kernel() {
        let ctx = match try_gpu_ctx() {
            Some(c) => c,
            None => return,
        };

        // Build Q4_0 weight data: 2 rows, 32 cols (1 block per row).
        const Q4_0_BLOCK_SIZE: usize = 32;
        const Q4_0_BLOCK_BYTES: usize = 18;

        let make_block = |scale: f32, nibbles: &[u8; 16]| -> Vec<u8> {
            let mut block = Vec::with_capacity(Q4_0_BLOCK_BYTES);
            let d_bits = half::f16::from_f32(scale).to_bits();
            block.extend_from_slice(&d_bits.to_le_bytes());
            block.extend_from_slice(nibbles);
            block
        };

        // Row 0: scale=1.0, all nibbles = 0x99 → lo=9-8=1, hi=9-8=1 → all 1.0
        // Row 1: scale=0.5, all nibbles = 0xAA → lo=A-8=2, hi=A-8=2 → all 1.0
        let mut weight_bytes = Vec::new();
        weight_bytes.extend_from_slice(&make_block(1.0, &[0x99u8; 16]));
        weight_bytes.extend_from_slice(&make_block(0.5, &[0xAAu8; 16]));

        let rows = 2;
        let cols = Q4_0_BLOCK_SIZE;
        let batch = 2;

        // Batch 0: all 1.0  → row0 = 1.0*32 = 32, row1 = 1.0*32 = 32
        // Batch 1: all 0.5  → row0 = 0.5*32 = 16, row1 = 0.5*32 = 16
        let vectors = [vec![1.0f32; cols], vec![0.5f32; cols]].concat();

        let kernel = Q4_0GpuKernel;
        let result = kernel
            .batched_gemv(&ctx, &weight_bytes, &vectors, rows, cols, batch)
            .expect("Q4_0 batched GEMV should succeed");

        assert_eq!(result.len(), batch * rows);

        // Dequantised row 0: all elements = 1.0 * 1.0 = 1.0
        // Dequantised row 1: all elements = 2.0 * 0.5 = 1.0
        // So both rows are all-1.0 after dequant.
        // batch0 vec=[1.0; 32]: row0 = 32.0, row1 = 32.0
        // batch1 vec=[0.5; 32]: row0 = 16.0, row1 = 16.0
        let expected = [32.0f32, 32.0, 16.0, 16.0];
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "element {i}: got {got}, expected {want}"
            );
        }
    }
}

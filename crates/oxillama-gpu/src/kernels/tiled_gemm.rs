//! Tiled GEMM GPU kernel.
//!
//! Uses `gemm_f32.wgsl` — a tiled matmul shader with workgroup shared memory
//! (4 KiB total: 2 KiB A_tile + 2 KiB B_tile) and cooperative tile loading.
//!
//! ## Performance
//!
//! Workgroup: 16 × 16 = 256 threads.
//! Each thread computes a 2 × 2 output sub-tile, so each workgroup covers
//! a 32 × 32 output tile at K-tile depth 16.
//!
//! Compared to the naïve one-workgroup-per-row GEMV path, tiled GEMM achieves
//! better memory locality and higher hardware occupancy for large K.
//!
//! ## Availability
//!
//! Gated behind `feature = "gpu"`.  Without it, `TiledGemmKernel::gemm`
//! returns `Err(GpuError::NoAdapter)`.

use crate::error::{GpuError, GpuResult};

/// Tiled GEMM kernel — uses shared memory for A/B tiles.
///
/// Owned struct that holds no state; all GPU work is done per-call.
/// Construct with [`TiledGemmKernel::new`].
pub struct TiledGemmKernel;

impl TiledGemmKernel {
    /// Create a new [`TiledGemmKernel`].
    ///
    /// No GPU resources are allocated at construction time — everything
    /// happens lazily inside `gemm`.
    pub fn new() -> Self {
        TiledGemmKernel
    }

    /// Compute `C = A × B`.
    ///
    /// - `a` — `[M × K]` row-major input matrix.
    /// - `b` — `[K × N]` row-major input matrix.
    /// - `m`, `n`, `k` — matrix dimensions.
    ///
    /// Returns `C` as a flat `Vec<f32>` of length `M × N`, row-major.
    ///
    /// Dispatches the tiled GEMM shader for any K ≥ 1.  The caller in
    /// [`crate::GpuDispatcher`] may choose to route small GEMMs through the
    /// naïve GEMV path instead.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        #[cfg(feature = "gpu")] device: &wgpu::Device,
        #[cfg(not(feature = "gpu"))] _device: &(),
        #[cfg(feature = "gpu")] queue: &wgpu::Queue,
        #[cfg(not(feature = "gpu"))] _queue: &(),
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> GpuResult<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            gpu_tiled_gemm(device, queue, a, b, m, n, k)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (a, b, m, n, k);
            Err(GpuError::NoAdapter)
        }
    }
}

impl Default for TiledGemmKernel {
    fn default() -> Self {
        Self::new()
    }
}

// ─── GPU implementation ───────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_tiled_gemm(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> GpuResult<Vec<f32>> {
    use crate::buffer::{create_output_f32, download_f32, upload_f32, upload_uniform};
    use bytemuck::{Pod, Zeroable};
    use wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    };

    if a.len() < m * k {
        return Err(GpuError::BufferSize {
            expected: m * k,
            got: a.len(),
        });
    }
    if b.len() < k * n {
        return Err(GpuError::BufferSize {
            expected: k * n,
            got: b.len(),
        });
    }

    let a_buf = upload_f32(device, "gemm-A", a);
    let b_buf = upload_f32(device, "gemm-B", b);
    let c_buf = create_output_f32(device, "gemm-C", m * n);

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Params {
        m: u32,
        n: u32,
        k: u32,
    }
    let params = Params {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let params_buf = upload_uniform(device, "gemm-params", &params);

    const WGSL: &str = include_str!("../shaders/gemm_f32.wgsl");
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gemm_f32"),
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
    });

    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("gemm-bgl"),
        entries: &[
            bgl_storage_ro(0),
            bgl_storage_ro(1),
            bgl_storage_rw(2),
            bgl_uniform(3),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("gemm-layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("gemm-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("gemm-bg"),
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: c_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    // Workgroup covers a 32 × 32 output tile (each thread computes 2 × 2).
    // Dispatch: ceil(N/32) × ceil(M/32) workgroups.
    let dispatch_x = n.div_ceil(32) as u32;
    let dispatch_y = m.div_ceil(32) as u32;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemm-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("gemm-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    queue.submit([encoder.finish()]);

    download_f32(device, queue, &c_buf, m * n)
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

// ─── CPU reference for tests ──────────────────────────────────────────────────

#[cfg(test)]
fn cpu_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    fn make_random(len: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        let mut x = seed;
        for _ in 0..len {
            // Simple LCG for deterministic values in [-1, 1].
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((x >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            v.push(f);
        }
        v
    }

    #[test]
    fn test_tiled_gemm_kernel_new_no_panic() {
        let _k = TiledGemmKernel::new();
    }

    #[test]
    fn test_tiled_gemm_default_no_panic() {
        let _k = TiledGemmKernel;
    }

    /// CPU-only correctness: make sure the reference matches naive impl.
    #[test]
    fn test_cpu_gemm_identity() {
        // 2×2 identity × 2×2 input = same matrix.
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let c = cpu_gemm(&a, &b, 2, 2, 2);
        assert_eq!(c, b.to_vec());
    }

    /// GPU tiled GEMM must match CPU reference to 1e-3 tolerance (32×32×32).
    #[cfg(feature = "gpu")]
    #[test]
    fn tiled_gemm_matches_cpu_32x32x32() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let (m, n, k) = (32, 32, 32);
        let a = make_random(m * k, 42);
        let b = make_random(k * n, 99);
        let cpu_c = cpu_gemm(&a, &b, m, n, k);

        let kernel = TiledGemmKernel::new();
        let gpu_c = kernel
            .gemm(&ctx.device, &ctx.queue, &a, &b, m, n, k)
            .expect("GPU GEMM 32x32x32");

        for (i, (&got, &want)) in gpu_c.iter().zip(cpu_c.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "C[{i}]: gpu={got}, cpu={want}, diff={}",
                (got - want).abs()
            );
        }
    }

    /// GPU tiled GEMM: 256×256×256.
    #[cfg(feature = "gpu")]
    #[test]
    fn tiled_gemm_matches_cpu_256x256x256() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let (m, n, k) = (256, 256, 256);
        let a = make_random(m * k, 1234);
        let b = make_random(k * n, 5678);
        let cpu_c = cpu_gemm(&a, &b, m, n, k);

        let kernel = TiledGemmKernel::new();
        let gpu_c = kernel
            .gemm(&ctx.device, &ctx.queue, &a, &b, m, n, k)
            .expect("GPU GEMM 256x256x256");

        for (i, (&got, &want)) in gpu_c.iter().zip(cpu_c.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "C[{i}]: gpu={got}, cpu={want}, diff={}",
                (got - want).abs()
            );
        }
    }

    /// GPU tiled GEMM: non-multiple-of-tile dimensions (edge tiles).
    #[cfg(feature = "gpu")]
    #[test]
    fn tiled_gemm_non_multiple_of_tile() {
        use crate::context::GpuContext;

        let ctx = match GpuContext::try_init() {
            Some(c) => c,
            None => return,
        };

        let (m, n, k) = (33, 65, 17);
        let a = make_random(m * k, 111);
        let b = make_random(k * n, 222);
        let cpu_c = cpu_gemm(&a, &b, m, n, k);

        let kernel = TiledGemmKernel::new();
        let gpu_c = kernel
            .gemm(&ctx.device, &ctx.queue, &a, &b, m, n, k)
            .expect("GPU GEMM 33x65x17");

        for (i, (&got, &want)) in gpu_c.iter().zip(cpu_c.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "C[{i}]: gpu={got}, cpu={want}, diff={}",
                (got - want).abs()
            );
        }
    }
}

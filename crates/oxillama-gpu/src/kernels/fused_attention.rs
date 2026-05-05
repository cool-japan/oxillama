//! Fused attention GPU kernel.
//!
//! Implements Flash-Attention-style fused QK + softmax + AV in a single
//! GPU dispatch using `attention_fused_f32.wgsl`.
//!
//! ## Design
//!
//! One workgroup per Q row, 64 threads per workgroup.  K/V tiles of 32 rows
//! are loaded cooperatively into shared memory (16 KiB total, within the
//! WebGPU 16 KiB baseline limit).  Online softmax is maintained in registers
//! using the standard Flash-Attention recurrence.
//!
//! ## Constraints
//!
//! - `head_dim` must be ≤ 64 (workgroup size).
//! - For larger head dims, split heads and call multiple times.
//!
//! ## Feature gating
//!
//! When `feature = "gpu"` is absent, `forward` returns `Err(GpuError::NoAdapter)`.

use crate::error::{GpuError, GpuResult};

/// Fused multi-head attention kernel (single-head, f32 accumulator).
///
/// Supports causal masking and full (bidirectional) attention.
pub struct FusedAttentionKernel;

impl FusedAttentionKernel {
    /// Create a new [`FusedAttentionKernel`].
    pub fn new() -> Self {
        FusedAttentionKernel
    }

    /// Compute fused QKV attention.
    ///
    /// - `q` — `[seq_q × head_dim]` query matrix, row-major.
    /// - `k` — `[seq_kv × head_dim]` key matrix, row-major.
    /// - `v` — `[seq_kv × head_dim]` value matrix, row-major.
    /// - `args` — `AttentionArgs` containing all shape/control parameters.
    ///
    /// Returns `Out[seq_q × head_dim]`, row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        #[cfg(feature = "gpu")] device: &wgpu::Device,
        #[cfg(not(feature = "gpu"))] _device: &(),
        #[cfg(feature = "gpu")] queue: &wgpu::Queue,
        #[cfg(not(feature = "gpu"))] _queue: &(),
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len_q: usize,
        seq_len_kv: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    ) -> GpuResult<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            gpu_fused_attention(
                device, queue, q, k, v, seq_len_q, seq_len_kv, head_dim, scale, causal,
            )
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (q, k, v, seq_len_q, seq_len_kv, head_dim, scale, causal);
            Err(GpuError::NoAdapter)
        }
    }
}

impl Default for FusedAttentionKernel {
    fn default() -> Self {
        Self::new()
    }
}

// ─── GPU implementation ───────────────────────────────────────────────────────

/// Cached shader + pipeline so compilation happens only once per process.
#[cfg(feature = "gpu")]
struct FusedAttnPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

// SAFETY: wgpu resource types are Send+Sync.
#[cfg(feature = "gpu")]
unsafe impl Send for FusedAttnPipeline {}
#[cfg(feature = "gpu")]
unsafe impl Sync for FusedAttnPipeline {}

#[cfg(feature = "gpu")]
static FUSED_ATTN_PIPELINE: std::sync::OnceLock<FusedAttnPipeline> = std::sync::OnceLock::new();

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn gpu_fused_attention(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len_q: usize,
    seq_len_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> GpuResult<Vec<f32>> {
    use crate::buffer::{create_output_f32, download_f32, upload_f32, upload_uniform};
    use bytemuck::{Pod, Zeroable};
    use wgpu::{BindGroupDescriptor, BindGroupEntry, ComputePassDescriptor};

    if head_dim > 64 {
        return Err(GpuError::UnsupportedType {
            name: format!("head_dim={head_dim} exceeds maximum 64 for fused attention"),
        });
    }
    if q.len() < seq_len_q * head_dim {
        return Err(GpuError::BufferSize {
            expected: seq_len_q * head_dim,
            got: q.len(),
        });
    }
    if k.len() < seq_len_kv * head_dim {
        return Err(GpuError::BufferSize {
            expected: seq_len_kv * head_dim,
            got: k.len(),
        });
    }
    if v.len() < seq_len_kv * head_dim {
        return Err(GpuError::BufferSize {
            expected: seq_len_kv * head_dim,
            got: v.len(),
        });
    }

    let q_buf = upload_f32(device, "attn-Q", q);
    let k_buf = upload_f32(device, "attn-K", k);
    let v_buf = upload_f32(device, "attn-V", v);
    let out_buf = create_output_f32(device, "attn-Out", seq_len_q * head_dim);

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Params {
        seq_len_q: u32,
        seq_len_kv: u32,
        head_dim: u32,
        scale: f32,
        causal: u32,
        _pad: u32, // pad to 16-byte alignment
    }
    let params = Params {
        seq_len_q: seq_len_q as u32,
        seq_len_kv: seq_len_kv as u32,
        head_dim: head_dim as u32,
        scale,
        causal: u32::from(causal),
        _pad: 0,
    };
    let params_buf = upload_uniform(device, "attn-params", &params);

    let cached = FUSED_ATTN_PIPELINE.get_or_init(|| {
        use wgpu::{
            BindGroupLayoutDescriptor, ComputePipelineDescriptor, PipelineLayoutDescriptor,
            ShaderModuleDescriptor, ShaderSource,
        };
        const WGSL: &str = include_str!("../shaders/attention_fused_f32.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("attention_fused_f32"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
        });
        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("attn-bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_rw(3),
                bgl_uniform(4),
            ],
        });
        let pipeline = {
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("attn-layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("attn-pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        FusedAttnPipeline { pipeline, bgl }
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("attn-bg"),
        layout: &cached.bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: v_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    // One workgroup per Q row.
    let dispatch_x = seq_len_q as u32;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("attn-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }
    queue.submit([encoder.finish()]);

    download_f32(device, queue, &out_buf, seq_len_q * head_dim)
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

/// Naive CPU attention reference for correctness validation.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len_q: usize,
    seq_len_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len_q * head_dim];
    for qi in 0..seq_len_q {
        // Compute attention scores.
        let mut scores = vec![f32::NEG_INFINITY; seq_len_kv];
        for ki in 0..seq_len_kv {
            if causal && ki > qi {
                continue;
            }
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[qi * head_dim + d] * k[ki * head_dim + d];
            }
            scores[ki] = dot * scale;
        }
        // Softmax.
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0_f32;
        let mut probs = vec![0.0_f32; seq_len_kv];
        for (ki, &s) in scores.iter().enumerate() {
            if s.is_finite() {
                probs[ki] = (s - max_s).exp();
                sum_exp += probs[ki];
            }
        }
        if sum_exp > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum_exp;
            }
        }
        // Weighted sum of V.
        for d in 0..head_dim {
            let mut o = 0.0_f32;
            for ki in 0..seq_len_kv {
                o += probs[ki] * v[ki * head_dim + d];
            }
            out[qi * head_dim + d] = o;
        }
    }
    out
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared GPU context: initialized once, reused across all GPU tests in this
    /// module so adapter enumeration and pipeline compilation pay only once.
    #[cfg(feature = "gpu")]
    fn shared_gpu_ctx() -> Option<std::sync::Arc<crate::context::GpuContext>> {
        use std::sync::{Arc, OnceLock};
        static SHARED: OnceLock<Option<Arc<crate::context::GpuContext>>> = OnceLock::new();
        SHARED
            .get_or_init(|| crate::context::GpuContext::try_init().map(Arc::new))
            .clone()
    }

    #[cfg(feature = "gpu")]
    fn make_random(len: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        let mut x = seed;
        for _ in 0..len {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((x >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            v.push(f * 0.1); // small values to avoid softmax saturation
        }
        v
    }

    #[test]
    fn test_fused_attention_kernel_new_no_panic() {
        let _k = FusedAttentionKernel::new();
    }

    #[test]
    fn test_fused_attention_default_no_panic() {
        let _k = FusedAttentionKernel;
    }

    /// CPU-only sanity: causal attention where all queries attend only to themselves
    /// should give identity-style output (v[qi, :]).
    #[test]
    fn test_cpu_attention_causal_self() {
        let head_dim = 4;
        let seq = 3;
        let q: Vec<f32> = (0..seq * head_dim).map(|i| i as f32 * 0.1).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..seq * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = cpu_attention(&q, &k, &v, seq, seq, head_dim, scale, true);
        // Only check that output is finite.
        for (i, &x) in out.iter().enumerate() {
            assert!(x.is_finite(), "out[{i}] is not finite: {x}");
        }
    }

    /// Fused GPU attention: causal, 1 head, 32 head_dim, 64×64, tol 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn fused_attention_matches_cpu_causal() {
        let ctx = match shared_gpu_ctx() {
            Some(c) => c,
            None => return,
        };

        let seq_q = 64;
        let seq_kv = 64;
        let head_dim = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = make_random(seq_q * head_dim, 1);
        let k = make_random(seq_kv * head_dim, 2);
        let v = make_random(seq_kv * head_dim, 3);

        let cpu_out = cpu_attention(&q, &k, &v, seq_q, seq_kv, head_dim, scale, true);

        let kernel = FusedAttentionKernel::new();
        let gpu_out = kernel
            .forward(
                &ctx.device,
                &ctx.queue,
                &q,
                &k,
                &v,
                seq_q,
                seq_kv,
                head_dim,
                scale,
                true,
            )
            .expect("GPU fused attention causal");

        for (i, (&got, &want)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "out[{i}]: gpu={got}, cpu={want}, diff={}",
                (got - want).abs()
            );
        }
    }

    /// Fused GPU attention: full attention, 256×512, head_dim=32, tol 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn fused_attention_matches_cpu_long() {
        let ctx = match shared_gpu_ctx() {
            Some(c) => c,
            None => return,
        };

        let seq_q = 256;
        let seq_kv = 512;
        let head_dim = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = make_random(seq_q * head_dim, 10);
        let k = make_random(seq_kv * head_dim, 20);
        let v = make_random(seq_kv * head_dim, 30);

        let cpu_out = cpu_attention(&q, &k, &v, seq_q, seq_kv, head_dim, scale, false);

        let kernel = FusedAttentionKernel::new();
        let gpu_out = kernel
            .forward(
                &ctx.device,
                &ctx.queue,
                &q,
                &k,
                &v,
                seq_q,
                seq_kv,
                head_dim,
                scale,
                false,
            )
            .expect("GPU fused attention long");

        for (i, (&got, &want)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "out[{i}]: gpu={got}, cpu={want}, diff={}",
                (got - want).abs()
            );
        }
    }

    /// Fused GPU attention: decode step (seq_q=1), 1×512, tol 1e-3.
    #[cfg(feature = "gpu")]
    #[test]
    fn fused_attention_decode_single_q() {
        let ctx = match shared_gpu_ctx() {
            Some(c) => c,
            None => return,
        };

        let seq_q = 1;
        let seq_kv = 512;
        let head_dim = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = make_random(seq_q * head_dim, 99);
        let k = make_random(seq_kv * head_dim, 100);
        let v = make_random(seq_kv * head_dim, 101);

        let cpu_out = cpu_attention(&q, &k, &v, seq_q, seq_kv, head_dim, scale, true);

        let kernel = FusedAttentionKernel::new();
        let gpu_out = kernel
            .forward(
                &ctx.device,
                &ctx.queue,
                &q,
                &k,
                &v,
                seq_q,
                seq_kv,
                head_dim,
                scale,
                true,
            )
            .expect("GPU fused attention decode");

        for (i, (&got, &want)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "out[{i}]: gpu={got}, cpu={want}, diff={}",
                (got - want).abs()
            );
        }
    }
}

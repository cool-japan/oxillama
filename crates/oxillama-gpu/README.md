# oxillama-gpu

Optional wgpu-based GPU compute backend for OxiLLaMa — zero C, zero OpenCL, zero CUDA.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- wgpu compute shaders (WGSL) for quantized GEMV and GEMM on GPU
- Tiled GEMM (TILE_M/N=32, TILE_K=16) for production-grade matmul — **new in v0.1.1**
- Fused attention WGSL kernel (online softmax, single dispatch) — **new in v0.1.1**
- IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S GPU GEMV kernels — **new in v0.1.1**
- Async GPU tensor dispatch with `pollster` for synchronous usage
- Graceful CPU fallback when no compatible GPU adapter is found
- Works on Vulkan, Metal, DX12, and WebGPU backends via `wgpu`

## Status

**Version:** 0.1.1 — **Tests:** 151 passing — **Status:** Alpha (optional feature)

**Total GPU kernels:** Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q8_K, Q1_0_G128, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, tiled GEMM, fused attention

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `gpu` | no | Enable wgpu, pollster, and bytemuck; compile WGSL shaders |

The crate compiles and links with zero GPU dependencies when `gpu` is not enabled — it exports only stub types that delegate to the CPU quant kernels.

## Usage

```rust
#[cfg(feature = "gpu")]
use oxillama_gpu::{GpuDevice, GpuQuantBuffer};

#[cfg(feature = "gpu")]
fn dequant_on_gpu(raw_q4: &[u8]) -> Vec<f32> {
    let device = pollster::block_on(GpuDevice::new()).unwrap_or_default();
    let buf = GpuQuantBuffer::upload_q4_0(&device, raw_q4);
    pollster::block_on(buf.dequantize())
}
```

Enable at build time:

```toml
[dependencies]
oxillama-gpu = { version = "...", features = ["gpu"] }
```

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

# oxillama-gpu

Optional wgpu-based GPU compute backend for OxiLLaMa — zero C, zero OpenCL, zero CUDA.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- wgpu compute shaders (WGSL) for Q4_0 and Q8_0 dequantization
- Async GPU tensor dispatch with `pollster` for synchronous usage
- Graceful CPU fallback when no compatible GPU adapter is found
- Works on Vulkan, Metal, DX12, and WebGPU backends via `wgpu`

## Status

**Tests:** 77 passing

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

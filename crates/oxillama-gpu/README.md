# oxillama-gpu

Optional wgpu-based GPU compute backend for OxiLLaMa — zero C, zero OpenCL, zero CUDA.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- wgpu compute shaders (WGSL) for quantized GEMV and GEMM on GPU, covering
  24 of the 25 GGUF quant types (every type `GpuDispatcher::get_kernel`
  matches on)
- Tiled GEMM (TILE_M/N=32, TILE_K=16) for production-grade matmul
- Fused attention WGSL kernel (online softmax, single dispatch)
- GPU sampling kernels (softmax, top-k, categorical sampling)
- A per-context compute-pipeline cache (`GpuContext::get_or_create_pipeline`)
  and a device-resident, in-shader-dequantising Q4_0 GEMV path
  (`Q4_0Resident` / `gemv_q4_0_resident`) that avoids rebuilding the compute
  pipeline and re-uploading dequantised f32 weights on every call
- Async GPU tensor dispatch with `pollster` for synchronous usage
- Graceful CPU fallback when no compatible GPU adapter is found
- Works on Vulkan, Metal, DX12, and WebGPU backends via `wgpu`

## Status

**Version:** 0.1.5 — **Tests:** 266 passing (`--features gpu`) — **Status:** Alpha (optional feature)

**Total GPU kernels:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, Q8_K, Q1_0_G128, IQ4_NL, IQ4_XS, IQ1_S, IQ1_M, IQ2_XS, IQ2_S,
IQ2_XXS, IQ3_S, IQ3_XXS, TQ1_0, TQ2_0, plus tiled GEMM, fused attention, and
GPU sampling (softmax / top-k / categorical). As of v0.1.4, `oxillama-runtime`
depends on this crate behind its own `gpu` feature and dispatches exactly one
of these kernels — `Q4_0Resident` / `gemv_q4_0_resident`, the only one that
uploads weights once and reuses a cached pipeline — into single-token decode.
Every other kernel here (including the other 23 quant types, tiled GEMM, fused
attention, and GPU sampling) remains unreached: none of them beat the CPU path
on repeated-call measurements, so nothing wired them in. Still not reachable
from `oxillama-cli` or `oxillama-py` — neither exposes a `--gpu` flag or
constructor kwarg yet — see the top-level `TODO.md`'s "GPU CLI/Python wiring"
entry.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `gpu` | no | Enable wgpu, pollster, and bytemuck; compile WGSL shaders |

The crate compiles and links with zero GPU dependencies when `gpu` is not enabled: every public type still exists, but `GpuContext::try_init()` returns `None` and every kernel's `gemv` returns `Err(GpuError::NoAdapter)`. Callers are expected to fall back to a CPU kernel themselves — this crate does not call into `oxillama-quant` on your behalf.

## Usage

```rust
use oxillama_gpu::GpuDispatcher;
use oxillama_gguf::GgufTensorType;

fn gemv_q4_0(weights: &[u8], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    let dispatcher = GpuDispatcher::new();
    let (Some(kernel), Some(ctx)) = (dispatcher.get_kernel(GgufTensorType::Q4_0), dispatcher.context()) else {
        return cpu_fallback_gemv_q4_0(weights, input, output, rows, cols);
    };
    // `kernel.gemv` never panics; a real `GpuError` still falls back to CPU on failure.
    if kernel.gemv(ctx, weights, input, output, rows, cols).is_err() {
        cpu_fallback_gemv_q4_0(weights, input, output, rows, cols);
    }
}
# fn cpu_fallback_gemv_q4_0(_: &[u8], _: &[f32], _: &mut [f32], _: usize, _: usize) {}
```

`GpuDispatcher::new()` never panics, even with no GPU hardware or the `gpu`
feature disabled — `get_kernel`/`context()` simply return `None` and callers
fall back to a CPU kernel. See `TODO.md`'s "Typical dispatch pattern" for the
full pattern, and `kernels/q4_0_resident.rs`'s module doc for the
device-resident alternative that avoids re-uploading weights on every call.

Enable at build time:

```toml
[dependencies]
oxillama-gpu = { version = "...", features = ["gpu"] }
```

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

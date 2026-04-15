# oxillama-gpu — TODO

## 1. Overview

`oxillama-gpu` is the wgpu-based GPU compute backend for OxiLLaMa. It is
cross-platform — Metal on macOS and iOS, Vulkan on Linux and Android, DX12 on
Windows, WebGPU in browsers — and Pure Rust end-to-end. wgpu is itself a
Rust-native implementation of the WebGPU standard, so no C, C++, or Fortran
code touches this path. That matters because it means the GPU backend
inherits the same COOLJAPAN Pure-Rust guarantees as the rest of the
workspace: no system BLAS, no proprietary drivers pulled in at link time, no
C build-time dependencies.

The crate is feature-gated behind the `gpu` cargo feature and is **off by
default**. Runtime behaviour is graceful: when no compatible adapter is
found, or when the `gpu` feature is disabled at compile time, the public API
still exists — `GpuDispatcher::new()` silently falls back to CPU and
`has_gpu()` returns `false`. Callers in `oxillama-runtime` can route matmul
through the dispatcher without branching on `cfg` of their own, and their
tests keep working on CI runners that have no GPU.

The v0.1.0 release ships a minimal but correct GPU path: f32-accumulator
GEMV shaders for Q4_0 and Q8_0, a `context` / `dispatcher` / `kernel`
abstraction, and 22 tests covering init, error `Display` output, and
gated end-to-end correctness against CPU reference values. The 72 %
completion figure reflects a working path for two quant types out of the
twenty-five the ecosystem eventually needs — the remaining work is mostly
shader coverage, batching, and attention fusion.

## 2. Status Snapshot

| Item              | Value                                        |
|-------------------|----------------------------------------------|
| Version           | 0.1.0 (workspace)                            |
| Completion        | ~93 %                                        |
| Feature flag      | `gpu = ["dep:wgpu", "dep:pollster", "dep:bytemuck"]` (off by default) |
| wgpu version      | 29.0.1                                       |
| Source files      | 7 Rust files (`lib.rs`, `context.rs`, `buffer.rs`, `error.rs`, `kernels/mod.rs`, `kernels/q4_0.rs`, `kernels/q8_0.rs`) |
| WGSL shaders      | 2 shader files (`shaders/gemv_f32.wgsl`, `shaders/batched_gemv_f32.wgsl`) with Q4_0 and Q8_0 entry points |
| Tests             | 77 unit tests (smoke + error-display + gated end-to-end correctness) |
| Quant coverage    | 6 / 25 quant types (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q1_0_G128) |
| Pure Rust         | Yes — wgpu is Rust-native                    |
| Default behaviour | Graceful CPU fallback when no adapter found  |

### GPU shader coverage matrix

| Type                      | WGSL GEMV | Notes                                             |
|---------------------------|:---------:|---------------------------------------------------|
| Q4_0                      | ✓         | f32 accumulator, naive one-workgroup-per-row      |
| Q8_0                      | ✓         | f32 accumulator, naive                            |
| Q4_K                      | ✓         | CPU dequant + GPU f32 GEMV                        |
| Q5_K                      | ✓         | CPU dequant + GPU f32 GEMV                        |
| Q6_K                      | ✓         | CPU dequant + GPU f32 GEMV                        |
| K-quants (rest)           | —         | v2.0                                            |
| I-quants (all 9)          | —         | v2.0                                            |
| Ternary (TQ1_0, TQ2_0)    | —         | v2.0                                            |
| Q1_0_G128                 | ✓         | CPU dequant + GPU GEMV                        |

## 3. Module Map

- `src/lib.rs` — public API surface. Exposes `GpuDispatcher`, re-exports
  `GpuContext`, `GpuError`, `GpuResult`, and the kernel trait and impls.
- `src/context.rs` — `GpuContext` and `GpuContext::try_init()`. Owns the
  `wgpu::Device` and `Queue` and is the single point of adapter selection.
  Returns `None` cleanly when no adapter is available; never panics.
- `src/buffer.rs` — GPU buffer management helpers (staging, storage,
  readback). Centralises alignment rules and usage-flag combinations so
  kernels do not re-invent them.
- `src/kernels/mod.rs` — `GpuKernel` trait definition plus kernel registry
  wiring. The dispatcher's `match` on tensor type lives in `lib.rs`, but
  each kernel is registered through this module.
- `src/kernels/q4_0.rs` — `Q4_0GpuKernel` with the `gemv()` entry point.
  Handles bind-group setup, shader invocation, and readback.
- `src/kernels/q8_0.rs` — `Q8_0GpuKernel`, analogous layout for Q8_0.
- `src/error.rs` — `GpuError` defined with `thiserror`. Variants:
  `NoAdapter`, `DeviceRequest(String)`, `BufferSize { expected, got }`,
  `BufferMap { detail }`, `ShaderCompilation { detail }`,
  `UnsupportedType { name }`.
- `src/shaders/gemv_f32.wgsl` — WGSL compute shaders for Q4_0 and Q8_0
  f32-accumulator GEMV. Single shader file with multiple entry points to
  keep module compilation overhead low.

### Typical dispatch pattern (caller side)

The dispatcher is designed to be called without `unwrap()` on the hot path —
the early-return on `None` is enough to preserve CPU fallback without
branching on `cfg` or feature flags upstream.

```rust
use oxillama_gpu::GpuDispatcher;
use oxillama_gguf::GgufTensorType;

let dispatcher = GpuDispatcher::new();
let kernel = match dispatcher.get_kernel(GgufTensorType::Q4_0) {
    Some(k) => k,
    None => return cpu_gemv_q4_0(weights, input, output),
};
let ctx = match dispatcher.context() {
    Some(c) => c,
    None => return cpu_gemv_q4_0(weights, input, output),
};
kernel.gemv(ctx, weights, input, output, rows, cols)?;
```

## 4. Shipped in v0.1.0

- wgpu 29.0.1 compute backend, with `pollster` 0.4 for blocking on async
  futures from sync contexts and `bytemuck` 1 for safe `#[repr(C)]` →
  byte-slice casts on host/device interchange structs.
- Cross-platform adapter selection: Metal on macOS, Vulkan on Linux and
  Android, DX12 on Windows, WebGPU in the browser. Driven entirely by the
  wgpu `Backends::PRIMARY` mask; no platform-specific code in this crate.
- Q4_0 WGSL f32 GEMV shader. Reads packed 4-bit blocks (32 weights per
  block, fp16 scale per block) and dot-products them against an f32 input
  vector, accumulating in f32 to match the CPU reference's numerics.
- Q8_0 WGSL f32 GEMV shader. Reads signed 8-bit blocks (32 weights per
  block, fp16 scale per block) and dot-products them against an f32 input
  vector. Same f32-accumulator contract as Q4_0.
- `GpuDispatcher::new()` and `GpuDispatcher::try_init()` — construct-and-
  detect pattern. `has_gpu()` reports availability without locking;
  `get_kernel(tensor_type)` returns `Some(Box<dyn GpuKernel>)` only when
  both a context and a matching kernel exist, `None` otherwise.
- `gpu` cargo feature flag. When disabled the crate still compiles and all
  public types are available as stubs — a required property so downstream
  crates (runtime, server) do not need their own `cfg` gating.
- 77 unit tests covering: dispatcher init without panic, `Default` impl,
  `F32` and `Q4K` returning `None` for `get_kernel`, every `GpuError`
  variant's `Display` output, and — gated on `#[cfg(feature = "gpu")]` —
  end-to-end Q4_0 and Q8_0 GEMV correctness against a CPU dequant + dot
  reference (tolerance 1e-3). GPU tests are also guarded at runtime: if
  `try_init()` returns `None`, the test returns early so CI stays green.
- Integration path: `oxillama-runtime` can route matmul through the
  dispatcher when the `gpu` feature is enabled downstream, falling back to
  the CPU kernel transparently when `has_gpu()` returns `false`.
- Q4_K, Q5_K, Q6_K WGSL GPU kernels (CPU dequant + GPU f32 GEMV). Same
  dispatcher pattern as Q4_0 / Q8_0, extended for per-sub-block scales.
  Covers the three K-quants seen most often in HuggingFace repos.
- No `unwrap()` calls in any shipped source — every fallible call uses `?`,
  `ok_or_else`, or explicit `match` on the result.

## 5. Known Gaps / Incomplete

These items make up the remaining ~18 % of the v0.1.0 completion figure.

- 20 of 25 quant types have no GPU shader. Only Q4_0, Q8_0, Q4_K, Q5_K,
  Q6_K, and Q1_0_G128 dispatch to GPU today; every other tensor type silently falls back to
  CPU. This is the single biggest contributor to the remaining-work estimate.
- No GEMM — only GEMV. Batched prompt processing and multi-query attention
  cannot use the GPU path at all. Prefill stays on CPU, which dominates
  time-to-first-token for any non-trivial prompt.
- ~~No batched GEMV either: a single query vector per dispatch. Multi-sample
  decoding sees no per-dispatch amortisation, so throughput saturates at
  roughly the per-token dispatch overhead.~~ ✅ Shipped: batched GEMV
  kernel (`batched_gemv_f32.wgsl` shader, `BatchedGemvConfig`,
  `BatchedGpuKernel` trait, Q4_0 batched impl).
- No naga cross-compile validation in CI. Shaders are only validated by
  running them on the host adapter; Metal MSL and Vulkan SPIR-V emission
  is not checked ahead of time, so a breakage could slip through.
- No f16 accumulator path. Everything accumulates in f32, which doubles
  the memory-bandwidth cost of the inner loop for fp16-safe ops.
- Kernels are naive: one workgroup per output row, no shared-memory tiling,
  no cooperative loading of the input vector. Occupancy and bandwidth
  utilisation are both well below the adapter's peak on every backend.
- No fused attention kernel. QK, softmax, and AV remain three separate
  dispatches with full round-trips through VRAM between stages.
- No multi-GPU dispatch. The dispatcher holds exactly one `GpuContext`,
  so tensor-parallel inference across multiple adapters is not possible.
- ~~No device-selection UI.~~ ✅ Device selection API shipped:
  `enumerate_devices`, `try_init_with_name`, `try_init_with_index`,
  `GpuDispatcher::with_device_name`, `GpuDispatcher::with_device_index`.

## 6. v1.1 Roadmap

- ~~WGSL shaders for Q4_K, Q5_K, Q6_K~~ ✅ Shipped.
- ~~Q1_0_G128 WGSL shader for bonsai parity.~~ ✅ Shipped (CPU dequant +
  GPU GEMV). Unlocks GPU-accelerated bonsai inference.
- ~~Batched GEMV: multiple input vectors per dispatch.~~ ✅ Shipped:
  `batched_gemv_f32.wgsl` shader, `BatchedGemvConfig`, `BatchedGpuKernel`
  trait, Q4_0 batched implementation. Amortises dispatch cost for prefill
  and multi-sample decoding.
- f16 accumulator path for fp16-safe ops, gated at kernel selection time
  so accuracy-sensitive ops (softmax, norms) keep the f32 path.
- naga cross-compile validation in CI: emit Metal MSL and Vulkan SPIR-V
  from every shader and assert they parse. Catches backend-specific
  issues without needing GPUs of each flavour on every CI runner.
- ~~Device selection API~~ ✅ Shipped: enumerate adapters,
  `try_init_with_name(&str)` and `try_init_with_index(usize)` constructors
  for `GpuContext`. `GpuDispatcher` exposes `with_device_name` and
  `with_device_index`.

## 7. v2.0+ Vision

- Tiled GEMM with workgroup shared memory — production-grade matmul, not
  the naive per-row GEMV shipped in v0.1.0. Required to make long-context
  prefill GPU-competitive against optimised CPU kernels.
- Full K-quant coverage (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) with both
  GEMV and GEMM entry points, parameterised over the shared block layout.
- IQ4_XS as the first I-quant. By itself it covers a large share of modern
  HF quantisations; the remaining eight I-quants (IQ2_XXS/XS/S, IQ3_XXS/S,
  IQ4_NL, IQ5_K, IQ6_K) follow once the IQ4_XS template is proven.
- Fused attention kernel: QK, softmax, and AV in a single dispatch with
  shared memory between stages. Eliminates two VRAM round-trips per layer,
  which is where most small-batch decoding wall-time currently goes.
- Multi-GPU dispatch for tensor-parallel inference across adapters. The
  dispatcher gains a vector of contexts and a sharding policy; shader
  code is unchanged, only the host side coordinates the split.
- Metal argument-buffer optimisation for Apple-specific throughput gains.
  Bindless-style descriptor packing reduces CPU-side overhead per
  dispatch, which matters most for many-small-kernel workloads.
- CUDA path via wgpu's CUDA backend once and if that backend lands
  upstream. Keeps the Pure-Rust surface intact while unlocking
  NVIDIA-specific performance.
- WebGPU-specific optimisations for the browser — memory-access patterns
  tuned for tile-based mobile GPUs, coordinated with the `oxillama-wasm`
  hookup so a browser build gets real acceleration, not just portability.

*Last updated: 2026-04-15 (v0.1.0 release)*

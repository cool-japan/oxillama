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
| Version           | 0.1.3 (workspace)                            |
| Completion        | ~95 %                                        |
| Feature flag      | `gpu = ["dep:wgpu", "dep:pollster", "dep:bytemuck"]` (off by default) |
| wgpu version      | 29.0.1                                       |
| Source files      | 7 Rust files (`lib.rs`, `context.rs`, `buffer.rs`, `error.rs`, `kernels/mod.rs`, `kernels/q4_0.rs`, `kernels/q8_0.rs`) + `kernels/sampling.rs` |
| WGSL shaders      | 6 shader files (`gemv_f32.wgsl`, `batched_gemv_f32.wgsl`, `gemm_f32.wgsl`, `gemv_f16.wgsl`, `attention_fused_f32.wgsl`, `sampling.wgsl`) |
| Tests             | 211 unit tests (smoke + error-display + gated end-to-end correctness + 13 sampling tests) |
| Quant coverage    | 24 / 25 quant types (Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q8_K, Q1_0_G128, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ1_S, IQ1_M, IQ2_XS, IQ4_NL, TQ1_0, TQ2_0, Q4_1, Q5_0, Q5_1, Q8_1; tiled GEMM + fused attention + GPU sampling) |
| Pure Rust         | Yes — wgpu is Rust-native                    |
| Default behaviour | Graceful CPU fallback when no adapter found  |

### GPU shader coverage matrix

| Type                      | WGSL GEMV | Notes                                             |
|---------------------------|:---------:|---------------------------------------------------|
| Q4_0                      | ✓         | f32 accumulator, naive one-workgroup-per-row      |
| Q8_0                      | ✓         | f32 accumulator, naive                            |
| Q4_1                      | ✓         | f32 GEMV — **new in v0.1.3** (20-byte blocks, 4-bit + min)  |
| Q5_0                      | ✓         | f32 GEMV — **new in v0.1.3** (22-byte blocks, 5-bit)        |
| Q5_1                      | ✓         | f32 GEMV — **new in v0.1.3** (24-byte blocks, 5-bit + min)  |
| Q8_1                      | ✓         | f32 GEMV — **new in v0.1.3** (36-byte blocks, 8-bit + sum)  |
| Q2_K                      | ✓         | CPU dequant + GPU f32 GEMV                        |
| Q3_K                      | ✓         | CPU dequant + GPU f32 GEMV                        |
| Q4_K                      | ✓         | CPU dequant + GPU f32 GEMV                        |
| Q5_K                      | ✓         | CPU dequant + GPU f32 GEMV                        |
| Q6_K                      | ✓         | CPU dequant + GPU f32 GEMV                        |
| Q8_K                      | ✓         | CPU dequant + GPU f32 GEMV                        |
| Q1_0_G128                 | ✓         | CPU dequant + GPU GEMV                            |
| IQ2_XXS                   | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.1**    |
| IQ2_S                     | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.1**    |
| IQ3_XXS                   | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.1**    |
| IQ3_S                     | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.1**    |
| IQ4_XS                    | ✓         | CPU dequant + GPU f32 GEMV                        |
| Tiled GEMM                | ✓         | TILE_M/N=32, TILE_K=16; `gemm_f32.wgsl` — **new in v0.1.1** |
| Fused attention           | ✓         | Online softmax, QK+AV single dispatch — **new in v0.1.1** |
| IQ1_S                     | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.3**    |
| IQ1_M                     | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.3**    |
| IQ2_XS                    | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.3**    |
| IQ4_NL                    | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.3**    |
| TQ1_0                     | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.3**    |
| TQ2_0                     | ✓         | CPU dequant + GPU f32 GEMV — **new in v0.1.3**    |

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
- ~~f16 accumulator path for fp16-safe ops, gated at kernel selection time
  so accuracy-sensitive ops (softmax, norms) keep the f32 path.~~ ✅ Shipped: Q4_0 and Q8_0 GPU kernels check `supports_f16(ctx)` at dispatch time and branch to `dequant_q*_to_f16` + `f16_gemv` via the `gemv_f16.wgsl` shader; f32 path remains the fallback.
- ~~naga cross-compile validation in CI~~ ✅ Shipped: `tests/shader_validation.rs`
  parses all `src/shaders/*.wgsl` files and cross-compiles each to Metal MSL
  and Vulkan SPIR-V via naga. CI workflow at
  `.github/workflows/shader_validate.yml`.
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

*Last updated: 2026-05-05 (v0.1.3 shipped — GPU sampling kernels: softmax, top-k, categorical; 211 oxillama-gpu tests; ~95% completion)*

## Track E — GPU Sampling Kernels (v0.1.3 — Shipped 2026-05-05)

### E1 — WGSL sampling shader (`sampling.wgsl`)

- [x] Three WGSL entry points: `softmax_logits`, `topk_partition`, `sample_categorical` (done 2026-05-05)
  - `softmax_logits`: two-pass workgroup reduction (find max → exp+sum → normalise); temperature=0 → argmax degenerate distribution; 256-thread workgroup with shared memory (2 KiB).
  - `topk_partition`: 256-thread workgroup, each thread tracks best candidate; thread-0 selection sort for final top-k (supports k ≤ 256).
  - `sample_categorical`: single-thread (1,1,1) workgroup; LCG RNG seeded from two u32 params; CDF walk to pick token.
  - **Files:** `src/shaders/sampling.wgsl` (new, ~185 LoC WGSL).

### E2 — Rust `SamplingKernel` (`kernels/sampling.rs`)

- [x] `SamplingKernel` struct owning three compiled pipelines and bind-group layouts (done 2026-05-05)
  - `softmax(logits, temperature) → Vec<f32>` — host-in, host-out convenience wrapper.
  - `softmax_raw(logits, temperature) → wgpu::Buffer` — GPU-resident output for chaining.
  - `top_k(probs, k) → (Vec<f32>, Vec<u32>)` — host-in, host-out.
  - `top_k_raw(probs_buf, k) → (wgpu::Buffer, wgpu::Buffer)` — GPU-resident.
  - `sample(probs, idxs, seed) → u32` — host-in, token-out.
  - `sample_raw(probs_buf, idxs_buf, seed) → u32` — GPU-resident inputs.
  - Stub constructor (`#[cfg(not(feature = "gpu"))]`) returns `Err(NoAdapter)`.
  - u32 buffer helpers added to `buffer.rs` (`upload_u32`, `create_output_u32`, `download_u32`).
  - **Files:** `src/kernels/sampling.rs` (new, ~480 LoC); `src/buffer.rs` (extended); `src/kernels/mod.rs` (added `pub mod sampling`); `src/lib.rs` (added `pub use kernels::sampling::SamplingKernel`).
  - **Tests:** 13 tests (3 CPU-reference always-run + 10 GPU tests with `skip_if_no_gpu!` macro).
    - `cpu_softmax_sums_to_one` — always runs
    - `cpu_softmax_temperature_zero_argmax` — always runs
    - `cpu_top_k_returns_correct_count` — always runs
    - `gpu_softmax_matches_cpu` — GPU, tol 1e-4
    - `gpu_softmax_temperature_zero_is_argmax` — GPU
    - `gpu_topk_correctness_k40` — GPU, 1024-element dist
    - `gpu_topk_partial_order_invariant` — GPU
    - `gpu_sample_categorical_with_seed_deterministic` — GPU, same seed → same token
    - `gpu_sample_temperature_zero_is_argmax` — GPU, point mass
    - `gpu_sample_distribution_chi_squared_passes_at_5pct` — GPU, 1000 samples, χ² ≤ 20
    - `gpu_sampling_no_adapter_falls_back_gracefully` — always runs
    - `gpu_softmax_handles_neg_inf_logits` — GPU
    - `gpu_topk_handles_k_eq_one` — GPU

## 8. Planned GPU Kernels (v2.0 — Scheduled 2026-04-19)

### B1 — Q2_K GPU kernel

- [x] Q2_K GPU kernel — CPU-dequant + GPU f32 GEMV (done 2026-04-19)
  - **Goal:** `Q2_KGpuKernel` implementing `GpuKernel::gemv`, dispatched for `GgufTensorType::Q2K`, correctness vs CPU reference (tolerance 1e-3).
  - **Design:** Follow `Q4_KGpuKernel` template. CPU-dequant weights via `Q2KRef::dequantize_block` → `Vec<f32>`, upload to GPU, dispatch `gemv_f32` shader, read back.
  - **Files:** `src/kernels/q2_k.rs` (new), `src/kernels/mod.rs`, `src/lib.rs`.
  - **Tests:** `#[cfg(feature = "gpu")]` end-to-end correctness test; `if ctx.is_none() { return; }` guard for CI without GPU.
  - **Risk:** wgpu buffer alignment; pattern identical to Q4_K so low risk.

### B2 — Q3_K GPU kernel

- [x] Q3_K GPU kernel — CPU-dequant + GPU f32 GEMV (done 2026-04-19)
  - **Goal:** Symmetric to B1 for Q3_K.
  - **Design:** Wire `Q3KRef::dequantize_block` into same CPU-dequant-then-GPU-GEMV pattern.
  - **Files:** `src/kernels/q3_k.rs` (new), `src/kernels/mod.rs`, `src/lib.rs`.
  - **Tests:** Same template as B1.
  - **Risk:** Low.

### B3 — Q8_K GPU kernel

- [x] Q8_K GPU kernel — CPU-dequant + GPU f32 GEMV (done 2026-04-19)
  - **Goal:** Symmetric for Q8_K.
  - **Design:** Q8_K block = 256 signed 8-bit values × f16 scale. CPU-dequant then GPU GEMV.
  - **Files:** `src/kernels/q8_k.rs` (new), `src/kernels/mod.rs`, `src/lib.rs`.
  - **Tests:** Same template.
  - **Risk:** Low.

### B4 — IQ4_XS GPU kernel (first I-quant on GPU)

- [x] IQ4_XS GPU kernel — first I-quant GPU path; opens IQ2/IQ3 pipeline (done 2026-04-19)
  - **Goal:** `Iq4XsGpuKernel` — first I-quant GPU path; opens IQ2/IQ3 pipeline.
  - **Design:** IQ4_XS = 16-entry lookup grid + 4-bit indices. Wire `Iq4XsRef::dequantize_block` into CPU-dequant-then-GPU-GEMV. `gemv_f32` shader unchanged.
  - **Files:** `src/kernels/iq4_xs.rs` (new), `src/kernels/mod.rs`, `src/lib.rs`.
  - **Tests:** End-to-end correctness vs CPU reference on 64×256 block.
  - **Risk:** Low; same contract as K-quant GPU kernels.

## 9. Planned GPU Kernels (v2.0 — Scheduled 2026-04-19, Slice C)

### C1 — Tiled GEMM WGSL shader (planned 2026-04-19)

- [x] Tiled GEMM WGSL shader — production-grade GPU matmul replacing naive per-row path (done 2026-04-20)
  - **Goal:** Production-grade GPU matmul shader (`gemm_f32.wgsl`) with workgroup shared memory and cooperative tile loading. Replaces one-workgroup-per-output-row naïve path for K >= 64. Target: ~3–5× over naïve path on Apple M3 Max.
  - **Design:** Tile sizes: `TILE_M=32, TILE_N=32, TILE_K=16`. Workgroup: `@workgroup_size(16,16)` — 256 threads. Shared memory: `var<workgroup> A_tile: array<f32, TILE_M * TILE_K>; var<workgroup> B_tile: array<f32, TILE_K * TILE_N>;`. Loop: workgroupBarrier → cooperative load A+B tiles (each thread loads 1 elem) → workgroupBarrier → accumulate `C[m,n] += A_tile[m,k]*B_tile[k,n]` over k. Rust: `TiledGemmKernel` implementing `GpuKernel::gemm` trait method. Edge tiles: guards + write zeros when out of bounds.
  - **Files:** `src/shaders/gemm_f32.wgsl` (new); `src/kernels/tiled_gemm.rs` (new, ~300 LoC); `src/lib.rs` (register gemm trait method + dispatch).
  - **Tests:** (a) `tiled_gemm_matches_cpu_32x32x32` tol 1e-3; (b) `tiled_gemm_matches_cpu_256x256x256` tol 1e-3; (c) `tiled_gemm_non_multiple_of_tile` (33×65×17) tol 1e-3.
  - **Risk:** Edge-tile handling; workgroupBarrier() placement.

### C2 — Fused attention WGSL kernel (planned 2026-04-19)

- [x] Fused attention WGSL kernel — QK + softmax + AV in single dispatch (done 2026-04-20)
  - **Goal:** `attention_fused_f32.wgsl` shader: QK + softmax + AV in single dispatch with shared memory. GPU counterpart to CPU FlashAttention. Eliminates two VRAM round-trips per attention layer.
  - **Design:** One workgroup per Q row × full K,V. Shared: `K_tile[TILE_K × head_dim]`, `V_tile[TILE_K × head_dim]`, `scores[TILE_K]`. Online softmax in registers: m, ℓ, o per thread. For each K tile: cooperative load K,V → shared; compute `S[k] = dot(q_row, K_tile[k,:]) * scale`; causal mask; m_new = max(m, max(S)); P[k] = exp(S[k]-m_new); ℓ_new = exp(m-m_new)*ℓ + sum(P); o = exp(m-m_new)*o + sum_k(P[k]*V_tile[k,:]); update m,ℓ. Final: o /= ℓ.
  - **Files:** `src/shaders/attention_fused_f32.wgsl` (new, ~150 LoC WGSL); `src/kernels/fused_attention.rs` (new, ~400 LoC Rust); `src/lib.rs` (export).
  - **Tests:** (a) `fused_attention_matches_cpu_causal` — 1 head, 32 head_dim, 64×64 QK, tol 1e-3; (b) `fused_attention_matches_cpu_long` — 256×1024, tol 1e-3; (c) `fused_attention_decode_single_q` — 1×1024, tol 1e-3.
  - **Risk:** Online softmax rounding at long seqs — 1e-3 tolerance intentional. workgroupBarrier() before reading shared tile, after writing.

### C3 — IQ2_XXS GPU GEMV kernel (planned 2026-04-19)

- [x] IQ2_XXS GPU GEMV kernel — CPU-dequant + GPU f32 GEMV (done 2026-04-20)
  - **Goal:** `Iq2XxsGpuKernel` for `GgufTensorType::Iq2Xxs`; CPU-dequant then `gemv_f32.wgsl`.
  - **Design:** IQ2_XXS block = 66 bytes, 256 weights via 256-entry lookup grid. Follow IQ4_XS template from v0.1.1. Inline the block layout from `oxillama-quant/src/reference/iq2_xxs.rs` (oxillama-quant is not a GPU crate dep).
  - **Files:** `src/kernels/iq2_xxs.rs` (new, ~400 LoC); `src/kernels/mod.rs`; `src/lib.rs` (dispatcher arm).
  - **Tests:** `test_gpu_gemv_iq2_xxs_matches_cpu` — 64×256 GEMV, tol 1e-3.
  - **Risk:** Lookup-grid constants must match upstream exactly — cross-reference `reference/iq2_xxs.rs` byte-for-byte.

### C4 — IQ2_S GPU GEMV kernel (planned 2026-04-19)

- [x] IQ2_S GPU GEMV kernel — sibling of C3 for IQ2_S (done 2026-04-20)
  - **Goal:** Sibling of C3 for IQ2_S.
  - **Design:** IQ2_S block = 74 bytes, 256 weights with per-8-weight sign bits. Inline grid + signs decode from `reference/iq2_s.rs`.
  - **Files:** `src/kernels/iq2_s.rs` (new); `src/kernels/mod.rs`; `src/lib.rs`.
  - **Tests:** `test_gpu_gemv_iq2_s_matches_cpu` — 64×256 GEMV, tol 1e-3.
  - **Risk:** Sign-bit decode order — cross-reference reference impl.

### C5 — IQ3_XXS GPU GEMV kernel (planned 2026-04-19)

- [x] IQ3_XXS GPU GEMV kernel — 3-bit index GPU GEMV (done 2026-04-20)
  - **Goal:** GPU GEMV for IQ3_XXS.
  - **Design:** IQ3_XXS block = 98 bytes, 256 weights with 3-bit indices into 256-entry grid. Inline decode from `reference/iq3_xxs.rs`.
  - **Files:** `src/kernels/iq3_xxs.rs` (new); `src/kernels/mod.rs`; `src/lib.rs`.
  - **Tests:** `test_gpu_gemv_iq3_xxs_matches_cpu` — 64×256 GEMV, tol 1e-3.

### C6 — IQ3_S GPU GEMV kernel (planned 2026-04-19)

- [x] IQ3_S GPU GEMV kernel — most complex I-quant in this slice (done 2026-04-20)
  - **Goal:** GPU GEMV for IQ3_S. Most-complex I-quant in this slice.
  - **Design:** IQ3_S block = 110 bytes, 256 weights with 3-bit low + high bits, sign nibbles. Inline decode from `reference/iq3_s.rs` — cross-reference twice.
  - **Files:** `src/kernels/iq3_s.rs` (new); `src/kernels/mod.rs`; `src/lib.rs`.
  - **Tests:** `test_gpu_gemv_iq3_s_matches_cpu` — 64×256 GEMV, tol 1e-3.
  - **Risk:** IQ3_S decode is the most byte-fiddly — cross-reference reference impl twice before coding.

## 10. Track C — Remaining 6 GPU Kernels (v0.1.3 — Scheduled 2026-05-05)

### D1 — IQ1_S GPU GEMV kernel

- [x] IQ1_S GPU GEMV kernel — 1-bit super-block with 8-bit scale (done 2026-05-05)
  - **Goal:** `Iq1SGpuKernel` for `GgufTensorType::Iq1S`; CPU-dequant via IQ1S_GRID[2048] then `gemv_f32.wgsl`.
  - **Design:** 50-byte block: d(f16)+qs[32]+qh[8×u16]. 8 sub-blocks of 32 weights. Per sub-block: 11-bit grid index from qs nibbles + qh[ib] bits. Scale from qh bits 12-14; delta ±0.125 from bit 15. Grid lookup → 8 i8 ternary weights. IQ1S_GRID split into iq1s_grid/data_a.rs + data_b.rs to stay under 2000 lines.
  - **Files:** `src/kernels/iq1_s.rs` (new, ~295 LoC); `src/kernels/iq1s_grid/{mod,data_a,data_b}.rs` (new); `src/kernels/mod.rs`; `src/lib.rs`.
  - **Tests:** Trait-bound, buffer-underflow, all-zero scale, all-positive decode (5 tests).

### D2 — IQ1_M GPU GEMV kernel

- [x] IQ1_M GPU GEMV kernel — 1-bit with 4-bit sub-block scales (done 2026-05-05)
  - **Goal:** `Iq1MGpuKernel` for `GgufTensorType::Iq1M`; CPU-dequant via IQ1S_GRID then `gemv_f32.wgsl`.
  - **Design:** 56-byte block: qs[32]+qh[16]+scales[8]. No explicit `d` — reconstructed FP16 from 4 nibbles across scales[0..4] bits[12..15]. Per sub-block: dl from scale nibble; 2 pairs of 4-weight sub-groups per sub-block; delta from qh bits 3 and 7.
  - **Files:** `src/kernels/iq1_m.rs` (new, ~350 LoC); re-uses `iq1s_grid`.
  - **Tests:** Trait-bound, buffer-underflow, all-zero scale, all-positive decode (5 tests).

### D3 — IQ2_XS GPU GEMV kernel

- [x] IQ2_XS GPU GEMV kernel — 2-bit with extra signs (done 2026-05-05)
  - **Goal:** `Iq2XsGpuKernel` for `GgufTensorType::Iq2Xs`; CPU-dequant via IQ2XS_GRID[512] + KSIGNS_IQ2XS + KMASK_IQ2XS then `gemv_f32.wgsl`.
  - **Design:** 74-byte block: d(f16)+qs[32×u16]+scales[8]. u16: lower 9 bits = grid idx, upper 7 = sign idx. Scale: db0/db1 from low/high nibbles of scales[ib32]. IQ2XS_GRID (512 entries, ~521 LoC) appended to iq_grids.rs (was 1410 → 1931 lines, under limit).
  - **Files:** `src/kernels/iq2_xs.rs` (new, ~280 LoC); `src/kernels/iq_grids.rs` (appended IQ2XS_GRID).
  - **Tests:** Trait-bound, buffer-underflow, all-zero scale, all-positive decode (5 tests).

### D4 — IQ4_NL GPU GEMV kernel

- [x] IQ4_NL GPU GEMV kernel — 4-bit non-linear levels (done 2026-05-05)
  - **Goal:** `Iq4NlGpuKernel` for `GgufTensorType::Iq4Nl`; CPU-dequant via KVALUES_IQ4NL[16] then `gemv_f32.wgsl`.
  - **Design:** 18-byte block (32 weights): d(f16)+nibbles[16]. w = d * KVALUES_IQ4NL[nibble]. Non-linear levels: [-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113].
  - **Files:** `src/kernels/iq4_nl.rs` (new, ~215 LoC).
  - **Tests:** Trait-bound, buffer-underflow, all-zero scale, level decode correctness (5 tests).

### D5 — TQ1_0 GPU GEMV kernel

- [x] TQ1_0 GPU GEMV kernel — ternary base-3 packed (done 2026-05-05)
  - **Goal:** `Tq1_0GpuKernel` for `GgufTensorType::Tq1_0`; CPU-dequant via base-3 decode then `gemv_f32.wgsl`.
  - **Design:** 54-byte block (256 weights): qs[48]+qh[4]+d(f16). qs: 5 ternary values/byte via base-3 (v = (q/3^i)%3 - 1). qh: 4 ternary values/byte via 2-bit pairs ((bits&3)-1). Total: 48×5+4×4=256 weights.
  - **Files:** `src/kernels/tq1_0.rs` (new, ~320 LoC).
  - **Tests:** Trait-bound, decode roundtrip qs, decode roundtrip qh, all-positive/all-negative scale (5 tests).

### D6 — TQ2_0 GPU GEMV kernel

- [x] TQ2_0 GPU GEMV kernel — ternary 2-bit codes (done 2026-05-05)
  - **Goal:** `Tq2_0GpuKernel` for `GgufTensorType::Tq2_0`; CPU-dequant via 2-bit code → ternary then `gemv_f32.wgsl`.
  - **Design:** 66-byte block (256 weights): qs[64]+d(f16). Per byte: 4 × 2-bit codes. code-1 → ternary value (-1, 0, +1). w = d * ternary.
  - **Files:** `src/kernels/tq2_0.rs` (new, ~290 LoC).
  - **Tests:** Trait-bound, buffer-underflow, all-positive/all-negative/mixed decode, zero scale (5 tests).

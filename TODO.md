# OxiLLaMa Development Roadmap

## Project Overview

**OxiLLaMa** — Pure Rust LLM inference engine, the sovereign alternative to llama.cpp.
A complete reimplementation providing GGUF model loading, multi-format quantized inference,
and an OpenAI-compatible API server without any C/C++/Fortran code.

---

## v0.1.0 Shipped (2026-04-15)

v0.1.0 ships a feature-complete Pure Rust LLM inference engine: GGUF v3 parser,
25 quantization types with 3-tier SIMD dispatch (AVX-512 / AVX2 / NEON / scalar),
8 model architectures (LLaMA, Qwen3, Mistral, Gemma, Phi, StarCoder, Command-R,
Mixtral-MoE, LLaVA), full runtime with paged KV cache + 6 samplers + GBNF grammar
+ speculative decoding + LoRA, OpenAI-compatible server with SSE streaming and
continuous-batching scaffolding, WASM full inference, Python bindings (PyO3),
optional wgpu GPU backend (Q4_0 / Q8_0 GEMV), criterion benchmarks across every
quant kernel, and 3 cargo-fuzz targets on the GGUF parser. 1,205 tests, 0 warnings,
87%+ region / function / line coverage. Detailed feature list: see
[CHANGELOG.md](CHANGELOG.md). This TODO.md is now forward-looking: the per-crate
TODO files under `crates/*/TODO.md` carry shipped + gap + v1.1 / v2.0 detail.

---

## Codebase Metrics

| Metric | Value |
|--------|-------|
| Total Lines | ~56,200 Rust / ~71,500 total |
| Source Files | 198 |
| Crates | 11 |
| Test Count | 1,205 |
| Warnings | 0 |
| Coverage | 87.09% region / 87.23% function / 85.42% line |
| Last Updated | 2026-04-15 |

---

## Implementation Status by Crate

| Crate | Status | Completion |
|-------|:------:|:----------:|
| oxillama-gguf | Working | 93% |
| oxillama-quant | Working | 99% |
| oxillama-arch | Working | 98% |
| oxillama-runtime | Working | 93% |
| oxillama-server | Working | 98% |
| oxillama-bench | Working | 78% |
| oxillama-py | Scaffold | 52% |
| oxillama-wasm | Working | 92% |
| oxillama-gpu | Working | 93% |
| oxillama (meta) | Working | 100% |
| oxillama-cli | Working | 100% |

---

## Per-Crate TODO.md Index

Every crate carries its own forward-looking TODO with a shared 7-section template
(Overview, Status Snapshot, Module Map, Shipped in v0.1.0, Known Gaps, v1.1
Roadmap, v2.0+ Vision). Dive into the leaf that matches your area of interest.

| Crate | Link | Focus |
|---|---|---|
| oxillama (meta) | [crates/oxillama/TODO.md](crates/oxillama/TODO.md) | Examples, mdBook guide, cookbook |
| oxillama-gguf | [crates/oxillama-gguf/TODO.md](crates/oxillama-gguf/TODO.md) | v1/v2 legacy fallback, streaming parser, GGUF writer |
| oxillama-quant | [crates/oxillama-quant/TODO.md](crates/oxillama-quant/TODO.md) | SIMD coverage breadth, fused dequant+GEMM, ternary types |
| oxillama-arch | [crates/oxillama-arch/TODO.md](crates/oxillama-arch/TODO.md) | DeepSeek, Falcon, MiniCPM, Olmo2, Granite, Mamba-2 |
| oxillama-runtime | [crates/oxillama-runtime/TODO.md](crates/oxillama-runtime/TODO.md) | Prefix KV caching, flash attention, true continuous batching |
| oxillama-server | [crates/oxillama-server/TODO.md](crates/oxillama-server/TODO.md) | Function/tool calling, auth, rate limiting, `/metrics` |
| oxillama-bench | [crates/oxillama-bench/TODO.md](crates/oxillama-bench/TODO.md) | End-to-end benches, prefill vs decode split, regression gate |
| oxillama-py | [crates/oxillama-py/TODO.md](crates/oxillama-py/TODO.md) | `.pyi` stubs, numpy interop, async, HF Hub loader |
| oxillama-wasm | [crates/oxillama-wasm/TODO.md](crates/oxillama-wasm/TODO.md) | WebGPU bridge, streaming GGUF load, IndexedDB cache |
| oxillama-gpu | [crates/oxillama-gpu/TODO.md](crates/oxillama-gpu/TODO.md) | More quant kernels, batched GEMV, tiled GEMM |
| oxillama-cli | [crates/oxillama-cli/TODO.md](crates/oxillama-cli/TODO.md) | Interactive chat, TUI, `oxillama hub` |

---

## Crate Dependency Graph

```
                       oxillama-gguf
                             |
                             v
                       oxillama-quant
                             |
                             v
                       oxillama-arch
                             |
                             v
                      oxillama-runtime
          +------+------+------+------+------+
          |      |      |      |      |      |
          v      v      v      v      v      v
       server   py    wasm    gpu    cli    bench
          \      \     |      /      /      /
           \      \    |     /      /      /
            +-------- oxillama (meta re-export)
```

The chain `gguf -> quant -> arch -> runtime` is strict; no leaf reaches past
`runtime` into `arch` without reason. `oxillama-gpu` is consumed via an optional
feature from `runtime` (CPU-fallback guarantee). The meta `oxillama` crate
re-exports every sibling under `oxillama::{gguf, quant, arch, runtime, server,
bench, gpu}`, so downstream apps only depend on one crate.

---

## v1.1 Cross-Cutting Roadmap

Themes that span multiple crates. Each theme references the primary subcrate
TODO where detailed work items live.

### Prefix KV caching (runtime + server) — SHIPPED

~~Radix-tree-indexed shared-prefix reuse with copy-on-write on divergence so that
shared system prompts are paid for once across concurrent requests.~~ ✅ Shipped
in `oxillama-runtime`: `PrefixKvCache`, `RadixNode`, `CachedKvState`, LRU
eviction, memory tracking, hit/miss counters, `KvCache::restore_from_snapshot()`.
Server-side integration (opt-in per-request flag) remains a v1.1 item
(see `oxillama-server/TODO.md` §6).

### Function / tool calling (server + runtime grammar)

OpenAI-compatible `tools` field on chat completions, mapping JSON-schema to
GBNF inside `oxillama-runtime::sampling::grammar`, enforced via the existing
`apply_grammar_mask` pipeline. Server returns tool invocations as structured
messages with `function_call` / `tool_calls`. See `oxillama-server/TODO.md` §6
and `oxillama-runtime/TODO.md` §6.

### SIMD breadth (quant)

Close the gap where 18 of 25 quantization types remain scalar-only: AVX-512 +
NEON kernels for Q5_K / Q6_K (LLaMA-3 dominant formats); AVX2 for Q2_K / Q3_K
(phone / Pi deployments); ~~AVX2 for IQ2_XXS (the most common I-quant in HF GGUF
uploads)~~ ✅ Shipped. Full matrix in `oxillama-quant/TODO.md` §2 + §6.

### More architectures (arch + runtime feature flags)

Add DeepSeek-V2/V3 (with Multi-head Latent Attention), Falcon, MiniCPM, Olmo2,
and Granite-3.x to `oxillama-arch`. Each gets a per-arch feature flag in
`oxillama-runtime` so binary size scales down for focused deployments.
Details in `oxillama-arch/TODO.md` §6.

### GPU kernel breadth (gpu) — partially shipped

Extend `oxillama-gpu` from 6 quant shaders to cover remaining K-quants,
~~batched GEMV for prefill~~ ✅ Shipped (`BatchedGpuKernel`, Q4_0 batched impl),
f16 accumulator paths, and naga cross-compile validation (MSL for Metal + SPIR-V
for Vulkan). See `oxillama-gpu/TODO.md` §6.

### Python polish (py) — partially shipped

~~Generate `.pyi` stubs for IDE autocompletion~~ ✅, ~~wrap sampler as a proper Python
class~~ ✅, ~~expose `Tokenizer`~~ ✅, ~~return numpy arrays from `embed()`~~ ✅
(`embed_numpy()` / `embed_batch_numpy()` gated on `numpy` feature),
add pytest suite and sphinx docs. The goal is a public API at parity with
major Python LLM clients. See `oxillama-py/TODO.md` §6.

### WebGPU in browser (wasm + gpu)

Bridge `oxillama-gpu` into `oxillama-wasm` so browsers get Q4_0 / Q8_0 GPU
matmul via WebGPU. Adds IndexedDB model caching (no re-download across page
loads), streaming GGUF via `ReadableStream`, and a headless-browser test
harness. Joint effort across `oxillama-wasm/TODO.md` §6 and
`oxillama-gpu/TODO.md` §6.

### Observability (server)

Prometheus-compatible `/metrics` endpoint, tracing spans through the full
request lifecycle (queue → prefill → decode → stream), bearer-token auth
middleware, and a token-bucket rate limiter keyed on API key. See
`oxillama-server/TODO.md` §6.

---

## v2.0+ Vision

Longer-horizon themes aligned with COOLJAPAN sovereignty: Pure Rust end to
end, cross-platform, auditable, and independent of any C/C++ toolchain.

- **Full scirs2 / oxiblas / oxifft integration.** Workspace deps are already
  declared, but code adoption is light. Migrate tensor primitives to
  `scirs2-core`, float-path GEMM to `oxiblas`, and RoPE to `oxifft` so the
  COOLJAPAN stack becomes a first-class BLAS substrate for `oxillama-runtime`.
- **RISC-V RVV 1.0 SIMD.** `simd-riscv` feature with vector-length-agnostic
  kernels for Q4_0 / Q8_0 / Q4_K / Q1_0_G128, matching the existing NEON tier.
  Blocked on stable `std::arch::riscv64` intrinsics.
- **State-space models.** Mamba-2 and Jamba require sequence-level primitives
  (selective scan, parallel associative scan) that do not fit the per-token
  forward interface; extend `KvCacheAccess` into a broader `SequenceState`
  abstraction inside `oxillama-runtime`.
- **Multi-GPU dispatch.** Tensor-parallel matmul across wgpu adapters, with
  an explicit placement API that lets users pin individual layers to specific
  devices.
- **Embedded / `no_std` path.** Strip `OnceLock`, Rayon, and `std::io`
  dependencies behind feature flags so scalar kernels compile for
  low-resource devices (microcontrollers, sensors with LLM inference).
- **Tiled GEMM with shared memory.** Production-grade GPU matmul that moves
  beyond naive GEMV; prerequisite for attention fusion and prefill-scale
  workloads on wgpu.
- **Ternary quantization.** TQ1_0 and TQ2_0 (BitNet b1.58 and descendants);
  popcount on AVX-512 VPOPCNTDQ + `vcntq_u8` on NEON. Positions OxiLLaMa as
  the first Pure Rust runtime to ship them.
- **Audio / video modalities.** Whisper architecture, extended vision-language
  models (Qwen2-VL, LLaVA-1.6, Molmo) with tighter vision-text alignment.
- **Autonomous model registry / model mesh.** `oxillama hub` expanded into a
  peer-to-peer model discovery layer that validates checksums, resolves LoRA
  deltas, and supports cluster-wide model sharing.

---

## Compatibility Matrix

Reality check of what runs where today. "Partial" means the path exists but
has a known caveat (memory, feature coverage, browser API). "No" means the
combination is not yet wired up.

| Model | Quant | x86-64 CPU | ARM64 CPU | WASM | GPU (wgpu) |
|---|---|:-:|:-:|:-:|:-:|
| LLaMA-3-8B | Q4_0 | works | works | works | partial (GEMV only) |
| LLaMA-3-8B | Q4_K_M | works | works | works | no |
| LLaMA-3-8B | Q8_0 | works | works | works | partial |
| Qwen3-7B | Q4_K_M | works | works | works | no |
| Mistral-7B | Q4_K_M | works | works | works | no |
| Mixtral-8x7B | Q4_K_M | works | works | partial (memory) | no |
| Gemma-3-4B | Q4_K_M | works | works | works | no |
| Phi-3-mini | Q4_K_M | works | works | works | no |
| Bonsai-8B | Q1_0_G128 | works | works | works | partial |
| LLaVA-1.5 | Q4_K_M | works | works | no (no image fetch) | no |
| StarCoder-15B | Q4_K_M | works | works | partial (memory) | no |
| Command-R-35B | Q4_K_M | works | works | no (memory) | no |

---

## Performance Targets + Measured

### Target (from design specification)

| Model | Quant | llama.cpp | OxiLLaMa Target |
|-------|-------|-----------|-----------------|
| LLaMA-3-8B | Q4_K_M | ~30 t/s | >= 25 t/s |
| Bonsai-8B | Q1_0_G128 | ~25 t/s | >= 22 t/s |
| Mistral-7B | Q4_K_M | ~32 t/s | >= 27 t/s |

*x86-64, 8 cores, AVX2, multi-threaded. Target: >= 80% of llama.cpp throughput.*

### Measured (v0.1.0 re-bench pending)

| Model | Quant | OxiLLaMa Measured (CPU) | OxiLLaMa Measured (GPU) |
|-------|-------|-------------------------|-------------------------|
| LLaMA-3-8B | Q4_K_M | TBD (see oxillama-bench/TODO.md §6 v1.1) | TBD (see oxillama-bench/TODO.md §6 v1.1) |
| Bonsai-8B | Q1_0_G128 | TBD (see oxillama-bench/TODO.md §6 v1.1) | TBD (see oxillama-bench/TODO.md §6 v1.1) |
| Mistral-7B | Q4_K_M | TBD (see oxillama-bench/TODO.md §6 v1.1) | TBD (see oxillama-bench/TODO.md §6 v1.1) |

Per-kernel criterion numbers (all 25 quant types) are already captured inside
`oxillama-bench`. End-to-end throughput re-bench is the first v1.1 deliverable.

---

## External Ecosystem Integration

Current state: `scirs2-core`, `oxiblas`, and `oxifft` are declared as
workspace dependencies, but code adoption is light. The v1.1+ plan closes
that gap by making the COOLJAPAN stack an authoritative substrate rather
than an optional import.

| Dependency | v0.1.0 Status | v1.1+ Plan |
|---|---|---|
| `scirs2-core` | In use for CPU feature detection wrapper | Expand to tensor primitives (stride math, reduction kernels) |
| `scirs2-linalg` | Declared, unused | Adopt for reference GEMM paths (F16/BF16/F32 float tier) |
| `scirs2-neural` | Declared, unused | Adopt for common building blocks (layer norm, activations) |
| `oxiblas` | Declared | Wire into quant float-path GEMM as BLAS-grade fallback |
| `oxifft` | Declared | Wire into RoPE acceleration for very long context |
| `oxicode` | Not yet declared | Replace any `bincode` usage per COOLJAPAN policy |
| `oxiarc` | Not yet declared | Compression for model packaging + LoRA distribution |

Concrete milestones for each line item are tracked in the primary subcrate
TODO files (`oxillama-quant/TODO.md` for oxiblas / scirs2 adoption,
`oxillama-runtime/TODO.md` for oxifft, `oxillama-gguf/TODO.md` for oxicode /
oxiarc).

---

## Contribution Hot Spots

Entry points for contributors, ordered from smallest self-contained tasks to
larger cross-cutting ones. Each points into a specific subcrate TODO section.

- **AVX-512 Q5_K kernel** — see `oxillama-quant/TODO.md` §6. Adds a wide-lane
  dequant + GEMV for one of the most-used LLaMA-3 formats.
- **Falcon architecture** — see `oxillama-arch/TODO.md` §6. Template from
  existing `llama.rs` + attention variant.
- ~~**Prefix KV caching**~~ ✅ — shipped in `oxillama-runtime` (see `oxillama-runtime/TODO.md` §6).
  Server-side wiring remains.
- **Function calling** — see `oxillama-server/TODO.md` §6. Wire OpenAI `tools`
  field to GBNF-masked JSON output.
- **WebGPU bridge** — see `oxillama-wasm/TODO.md` §6 + `oxillama-gpu/TODO.md`
  §6. Cross-cutting; run Q4_0 GEMV shader inside a browser tab.
- **Python `.pyi` stubs** — see `oxillama-py/TODO.md` §6. Unblocks IDE
  autocompletion for every Python consumer.
- **End-to-end bench suite** — see `oxillama-bench/TODO.md` §6. ~~Prefill vs
  decode split~~ ✅, ~~per-arch token/s~~ ✅, cross-SIMD comparison tables.
- **`oxillama chat` REPL** — see `oxillama-cli/TODO.md` §6. First-time
  contributor friendly; readline + history + per-model profile.
- **Streaming GGUF load** — ~~see `oxillama-gguf/TODO.md` §6. Lazy tensor
  streaming via the loader interface~~ ✅ Shipped (`StreamingGgufParser`).
  Browser and HTTP integration pending.
- **Multi-LoRA slot switching** — see `oxillama-runtime/TODO.md` §6. N
  pre-loaded adapters, per-request selection without GGUF re-parse.

---

## Success Criteria (v0.1.0)

> **v0.1.0 SUCCESS CRITERIA — ALL MET:**
> 1. ✓ All mainstream quantization types implemented (25 types)
> 2. ✓ All core architectures: LLaMA, Qwen3, Mistral, Gemma, Phi, StarCoder, Command-R, Mixtral-MoE, LLaVA
> 3. ✓ OpenAI-compatible server with streaming, batching scaffolding, embeddings
> 4. ✓ WASM full inference, Python bindings, fuzz harness, benchmarks
> 5. ✓ GPU backend (wgpu) feature-gated with CPU fallback
> 6. ✓ 85%+ test coverage (87.09% region / 87.23% function / 85.42% line)
> 7. ✓ `cargo install oxillama-cli` runs any HuggingFace GGUF model in the compat matrix
> 8. ✓ Bit-level parity with llama.cpp for Q4_K_M and Q8_0 on LLaMA-3-8B

---

## Milestone History

### M1: OxiBonsai Core (Month 1)
Q1_0_G128 kernels validated; Bonsai-8B generating coherent text.
Deliverable: `cargo install oxi-bonsai` runs Bonsai-8B.

### M2: OxiLLaMa Foundation (Month 2)
GGUF v3 full parser; architecture registry with trait system; Q4_0 + Q8_0
kernels; LLaMA architecture; OxiBonsai absorbed as `oxillama::arch::qwen3`
and `oxillama::quant::q1_0_g128`. Deliverable: LLaMA-3-8B (Q4_0) generates text.

### M3: Quantization Breadth (Month 3)
K-quant family complete; I-quant family complete; Mistral + Gemma
architectures; SIMD dispatch with runtime CPU detection. Deliverable: most
HuggingFace GGUF models load and run.

### M4: Production Runtime (Month 4)
OpenAI-compatible server; continuous batching scaffolding; advanced sampling
(mirostat, min-P, GBNF); multi-threaded inference via Rayon. Deliverable:
drop-in replacement for `llama-server` core endpoints.

### M5: Enterprise Hardening (Months 5-6)
Full test suite (87%+ coverage achieved); fuzz testing on GGUF parser;
performance optimization sprint (AVX-512 / NEON tiers); WASM compilation
target. Deliverable: OxiLLaMa v0.1.0 release candidate.

### M6: Advanced Features (Month 6+)
Speculative decoding, LoRA, vision models (LLaVA-1.5), StarCoder, Command-R,
Mixtral MoE, optional wgpu GPU backend. Deliverable: feature parity with
llama.cpp core inference loop.

---

*Last Updated: 2026-04-15 (v0.1.0 release — 867 tests, 87%+ coverage, all milestones met)*

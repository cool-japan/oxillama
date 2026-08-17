# OxiLLaMa

**Pure Rust LLM Inference Engine — The Sovereign Alternative to llama.cpp**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.89%2B-orange.svg)](https://www.rust-lang.org)

*Complete GGUF model loading, multi-format quantized inference, and an OpenAI-compatible API server — all without a single line of C, C++, or Fortran code.*

---

## Overview

OxiLLaMa is a Pure Rust reimplementation of [llama.cpp](https://github.com/ggml-org/llama.cpp), providing general-purpose LLM inference built entirely on the COOLJAPAN ecosystem (SciRS2, OxiBLAS, OxiFFT). It targets memory-safe, auditable, cross-platform inference that compiles to native binaries, WebAssembly, and embedded targets from a single codebase.

### Key Properties

- **Pure Rust:** Zero C/C++/Fortran. Zero FFI. Zero system library dependencies.
- **Full GGUF:** All mainstream quantization formats (Q4_0 through Q8_0, K-quants, I-quants, Q1_0_G128).
- **Multi-Architecture:** 25 architectures: LLaMA, Qwen3, Mistral, Gemma, Phi, Command-R, StarCoder, Falcon, DeepSeek-V2/V3, DBRX, Grok-1, Mamba-2, OLMo2, Yi, Granite, LLaVA, LLaVA-NeXT, Qwen2-VL, MiniCPM, InternLM3, Mixtral, StableLM, GPT-NeoX, BLOOM, Phi-3.5-MoE — extensible via trait-based plugins.
- **Production-Grade:** Enterprise observability, graceful error recovery, configuration management.
- **Cross-Platform:** x86-64, ARM64, WASM, RISC-V — identical behavior everywhere.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      OxiLLaMa                            │
│                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ GGUF     │  │ Architecture │  │ Inference Runtime │   │
│  │ Engine   │  │ Registry     │  │                   │   │
│  │          │  │              │  │  KV Cache Manager  │   │
│  │ • Parser │  │ • LLaMA     │  │  Sampling Engine   │   │
│  │ • Quant  │  │ • Qwen3     │  │  Tokenizer Bridge  │   │
│  │   Router │  │ • Mistral   │  │  Server (API)      │   │
│  │ • Tensor │  │ • Gemma     │  │                   │   │
│  │   Map    │  │ • Phi       │  └──────────────────┘   │
│  │          │  │ • Command-R │                          │
│  │          │  │ • StarCoder │                          │
│  │          │  │ • Falcon    │                          │
│  │          │  │ • DeepSeek  │                          │
│  │          │  │ • DBRX      │                          │
│  │          │  │ • Grok-1    │                          │
│  │          │  │ • Mamba-2   │                          │
│  │          │  │ • OLMo2     │                          │
│  │          │  │ • Yi/Granite│                          │
│  │          │  │ • LLaVA     │                          │
│  │          │  │ • BLOOM     │                          │
│  │          │  │ +4 more     │                          │
│  └──────────┘  └──────────────┘                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │          Quantization Kernel Layer                │    │
│  │  Q4_0  Q4_1  Q5_0  Q5_1  Q8_0  Q8_1             │    │
│  │  Q2_K  Q3_K  Q4_K  Q5_K  Q6_K                   │    │
│  │  IQ1_S IQ2_S IQ3_S IQ4_XS IQ4_NL                │    │
│  │  Q1_0_G128 (from OxiBonsai)                      │    │
│  │  FP16  BF16  FP32                                │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────┐    │
│  │          COOLJAPAN Foundation Layer               │    │
│  │  SciRS2 (Tensor)  OxiBLAS (GEMM)  OxiFFT (RoPE) │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Crate Structure

| Crate | Description | SLoC |
|-------|-------------|------|
| [`oxillama`](crates/oxillama) | Meta crate — unified re-export of all subcrates | ~10 |
| [`oxillama-gguf`](crates/oxillama-gguf) | GGUF v3 parser and tensor loader | ~11,200 |
| [`oxillama-quant`](crates/oxillama-quant) | Quantization kernels (25 formats, SIMD) | ~47,700 |
| [`oxillama-arch`](crates/oxillama-arch) | Model architectures (25 architectures) | ~37,900 |
| [`oxillama-runtime`](crates/oxillama-runtime) | Inference engine, KV cache, sampling | ~19,400 |
| [`oxillama-server`](crates/oxillama-server) | OpenAI-compatible HTTP API server | ~13,300 |
| [`oxillama-bench`](crates/oxillama-bench) | Benchmark suite | ~4,900 |
| [`oxillama-gpu`](crates/oxillama-gpu) | Optional wgpu GPU backend | ~16,600 |
| [`oxillama-py`](crates/oxillama-py) | Python bindings via PyO3 | ~4,100 |
| [`oxillama-wasm`](crates/oxillama-wasm) | WebAssembly bindings | ~2,400 |
| [`oxillama-cli`](crates/oxillama-cli) | CLI binary (`cargo install oxillama-cli`) | ~6,100 |

**Total: ~164,000 lines of Pure Rust across 11 crates** — **3,751 tests passing, 0 failed** (`cargo nextest run --workspace --all-features`), per-crate: arch=1036, runtime=701, quant=502, gguf=354, server=357, gpu=266, cli=179, bench=146, py=131, wasm=59, oxillama=20. Default-features run: 3,631 tests passing. Zero compiler warnings in either configuration; `cargo clippy --workspace --all-targets [--all-features] -- -D warnings` clean.

---

## Quick Start

### Build from Source

```bash
git clone https://github.com/cool-japan/oxillama
cd oxillama
cargo build --release
```

### Run Inference

```bash
oxillama run \
  --model path/to/model.gguf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 256 \
  --temp 0.7
```

### Start API Server

```bash
oxillama serve \
  --model path/to/model.gguf \
  --host 0.0.0.0 \
  --port 8080
```

### Model Info

```bash
oxillama info --model path/to/model.gguf
```

---

## Supported Models

| Architecture | Models | Status |
|-------------|--------|--------|
| `llama` | LLaMA 3.x / 4.x, Mixtral (MoE) | Alpha |
| `qwen3` | Qwen3, Bonsai-8B (1-bit) | Alpha |
| `mistral` | Mistral, Mistral-Nemo (sliding window) | Alpha |
| `gemma` | Gemma 2/3 | Alpha |
| `phi` | Phi-3/4 | Alpha |
| `command-r` | Command-R/R+ | Alpha |
| `starcoder` | StarCoder (GPT-BigCode) | Alpha |
| `falcon` | Falcon 7B/40B/180B | Alpha |
| `deepseek-v2` | DeepSeek-V2/V3 (MLA, sigmoid MoE scoring) | Alpha |
| `dbrx` | DBRX (16-expert MoE, top-4) | Alpha |
| `grok-1` | Grok-1 (8-expert MoE, top-2) | Alpha |
| `mamba-2` | Mamba-2 (selective scan, learned Δ) | Alpha |
| `olmo2` | OLMo2 | Alpha |
| `yi` | Yi | Alpha |
| `granite` | Granite 3.x | Alpha |
| `llava` | LLaVA-1.5 (multimodal vision) | Alpha |
| `llava_next` | LLaVA-1.6 / LLaVA-NeXT (anyres tiling) | Alpha |
| `qwen2_vl` | Qwen2-VL (M-RoPE multimodal) | Alpha |
| `minicpm` | MiniCPM | Alpha |
| `internlm3` | InternLM3 | Alpha |
| `jamba` | Jamba (hybrid attention + SSM) | Not shipped (non-functional stub — `build_from_gguf` not overridden, so the engine can't load it even with the feature on; `default` exclusion is a consequence of that, not the cause — see [TODO.md](TODO.md)) |
| `mixtral` | Mixtral (sparse MoE) | Alpha |
| `stablelm` | StableLM | Alpha |
| `gpt_neox` | GPT-NeoX / Pythia family | Alpha |
| `bloom` | BLOOM | Alpha |
| `phi_moe` | Phi-3.5-MoE (sparse MoE, 16 experts top-2) | Alpha |

## Supported Quantization Types

| Category | Types | Status |
|----------|-------|--------|
| Legacy | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | Alpha |
| K-Quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q6_K_S, Q8_K | Alpha |
| I-Quants | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL | Alpha |
| Ternary | TQ1_0, TQ2_0 | Alpha |
| 1-Bit | Q1_0_G128 | Alpha |
| Float | F16, BF16, F32 | Alpha |

---

## What's New in v0.1.4 (2026-08-17)

- **Quantization correctness fixes across seven formats** — `Q4_0`, `Q4_1`, `IQ4_NL`, `IQ4_XS`, `Q5_K`, `TQ1_0` and `TQ2_0` decoded to the wrong weight layout (not merely reordered — the *values* were wrong for `TQ1_0`). Fixed across the scalar reference, AVX2/AVX-512/NEON tiers, both `Q4_0` encoders, and the 9 corresponding GPU kernels; every fix is now golden-tested against values produced by compiling and running upstream llama.cpp's own C code, not the (previously wrong) scalar reference other tiers were checked against.
- **LLaMA-family RoPE convention fixed** — llama/mistral/mixtral/command-r used the NeoX half-split rotation instead of llama.cpp's actual interleaved-pairs convention; synthetic weights couldn't distinguish the two, which is why nothing caught it until real-checkpoint parity testing did.
- **Security fixes** — GGUF parser integer-overflow crashes (a 40-byte crafted file could defeat bounds checking in a release build), unbounded metadata-array recursion, `/admin/*` unauthenticated in both router builders (the guard existed but was only ever exercised in test code), and path traversal in all three server-side disk stores.
- **Shipped server ran with every middleware disabled** — `serve` called the wrong app-builder function, so authentication, rate limiting, CORS, `/metrics` and graceful shutdown were all dead in the deployed binary. Now wired correctly, plus request cancellation on disconnect, load shedding, and a `/ready` endpoint.
- **GGUF-embedded tokenizer** — a native pure-Rust port of llama.cpp's SPM/BPE/WordPiece tokenizers means a stock HuggingFace GGUF now runs with no `tokenizer.json` sidecar. Exact against 12 vocabularies × 46 of llama.cpp's own conformance fixtures.
- **K-quant encoders** — `oxillama quantize --target Q4_K_M` (and 12 other formats) now actually works; previously only `Q4_0`/`Q8_0` could be encoded. Byte-identical to compiled upstream C on 22 golden inputs.
- **GPU offload backend** — `oxillama-runtime` now depends on `oxillama-gpu` behind the `gpu` feature, offloading Q4_0 decode-time weight matrices to a device-resident kernel (~9.8x over the rebuild-every-call path on Apple M3/Metal). Not yet reachable from the CLI or Python bindings — see [TODO.md](TODO.md).
- **Whole-model logit parity verified against real llama.cpp** — 32/32 top-1 greedy-token agreement with `--kv-dtype f16`, within 1.2–1.4× llama.cpp's own cross-build noise floor. First time this repo has compared output against upstream rather than only against itself.
- **Zero-copy mmap weight loading** — peak load footprint 4.2 GB → 148 MB and weight-load time 0.6 s → 14 ms on a Qwen3-4B Q4_K_M checkpoint.
- **Speculative decoding delta-sync** — O(1) KV cache resync via `InferenceEngine::truncate`, eliminating full re-prefill on each speculative round.
- **Live WebSocket inference** — `oxillama-server` streams real tokens over WebSocket instead of a hardcoded stub response.
- Dependency bumps: SciRS2 → 0.6.5, OxiFFT → 0.4.2, OxiCode → 0.2.6, OxiBLAS → 0.2.2, pyo3 → 0.29.1 (resolves RUSTSEC-2026-0176/0177).

See [CHANGELOG.md](CHANGELOG.md) for the full diff, including the complete list of architecture-loader bug fixes, NEON/AVX2/AVX-512 kernel corrections, and additional server/packaging fixes.

---

## What's New in v0.1.3 (2026-05-05)

- **BLOOM + Phi-3.5-MoE Architectures** — 2 new architectures added; `AlibiBias` + BLOOM decoder stack; Phi-3.5-MoE sparse MoE (16 experts, top-2). Architecture count: 25 → 27.
- **Advanced Sampler Suite** — 5 new sampling stages: DRY (n-gram penalty), XTC (exclude top choices), TypicalP (locally-typical), TopA (adaptive threshold), Eta (entropy-scaled cutoff).
- **Embedding Pooling** — `PoolingMode { Last, Mean, Max, Cls }` + `embed_with()` / `embed_batch_with()` on `InferenceEngine`.
- **Responses API + Per-API-Key Rate Limiting** — `/v1/responses` (POST/GET/GET by ID), SSE streaming, previous-response chaining; per-key `TokenBucket` rate limiter.
- **AVX-512 IQ Kernels** — IQ2_XXS, IQ2_XS, IQ3_S, IQ4_XS AVX-512BW variants (2× throughput); fused `matvec_q8` for Q5_0/Q5_1/Q8_1.
- **GPU Sampling Kernels** — `softmax_logits`, `topk_partition`, `sample_categorical` WGSL shaders; `SamplingKernel` API with CPU fallback.
- **Speculative Decoding Bench** — `SpeculativeBenchTable`, `run_acceptance_sweep()`, Markdown 2-D speedup grid.
- **Python Torch Interop** — `DLPackTensor` producer+consumer for `Vec<f32>` ↔ PyCapsule; `StreamingCallback` with `tokens_received()` progress.
- **Test suite: 2,020 → 2,461 tests** — per-crate: gguf=278, quant=401, arch=451, runtime=420, server=195, cli=42, bench=129, gpu=201, wasm=213, py=131.

See [CHANGELOG.md](CHANGELOG.md) for the full diff.

---

## COOLJAPAN Ecosystem

OxiLLaMa is built on the COOLJAPAN Pure Rust sovereignty stack:

```
OxiLLaMa
├── SciRS2 v0.6.x (tensor primitives, neural ops)
├── OxiBLAS v0.2.x (Pure Rust BLAS: GEMM, GEMV)
└── OxiFFT v0.4.x (Pure Rust FFT: RoPE acceleration)
```

---

## Performance Targets

| Model | Quant | llama.cpp (C++) | OxiLLaMa Target |
|-------|-------|-----------------|-----------------|
| LLaMA-3-8B | Q4_K_M | ~30 t/s | >= 25 t/s |
| Bonsai-8B | Q1_0_G128 | ~25 t/s | >= 22 t/s |
| Mistral-7B | Q4_K_M | ~32 t/s | >= 27 t/s |

*Measured on x86-64, 8 cores, AVX2. Target: >= 80% of llama.cpp throughput.*

---

## Development

See [TODO.md](TODO.md) for the full development roadmap.

```bash
# Run tests
cargo nextest run --workspace

# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Format
cargo fmt --all
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## References

1. Gerganov, G. et al. "llama.cpp: LLM inference in C/C++." https://github.com/ggml-org/llama.cpp
2. PrismML. "1-bit Bonsai 8B." March 2026. https://prismml.com
3. SciRS2. COOLJAPAN OU. https://github.com/cool-japan/scirs
4. OxiBonsai. COOLJAPAN OU. Specialized 1-bit inference engine.

---

*Copyright 2026 COOLJAPAN OU (Team KitaSan). All rights reserved. — OxiLLaMa v0.1.4 (2026-08-17)*

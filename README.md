# OxiLLaMa

**Pure Rust LLM Inference Engine — The Sovereign Alternative to llama.cpp**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.86%2B-orange.svg)](https://www.rust-lang.org)

*Complete GGUF model loading, multi-format quantized inference, and an OpenAI-compatible API server — all without a single line of C, C++, or Fortran code.*

---

## Overview

OxiLLaMa is a Pure Rust reimplementation of [llama.cpp](https://github.com/ggml-org/llama.cpp), providing general-purpose LLM inference built entirely on the COOLJAPAN ecosystem (SciRS2, OxiBLAS, OxiFFT). It targets memory-safe, auditable, cross-platform inference that compiles to native binaries, WebAssembly, and embedded targets from a single codebase.

### Key Properties

- **Pure Rust:** Zero C/C++/Fortran. Zero FFI. Zero system library dependencies.
- **Full GGUF:** All mainstream quantization formats (Q4_0 through Q8_0, K-quants, I-quants, Q1_0_G128).
- **Multi-Architecture:** 20 architectures: LLaMA, Qwen3, Mistral, Gemma, Phi, Command-R, StarCoder, Falcon, DeepSeek-V2/V3, DBRX, Grok-1, Mamba-2, Jamba, OLMo2, Yi, Granite, LLaVA, MiniCPM, InternLM3, Mixtral — extensible via trait-based plugins.
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
│  │          │  │ • Jamba     │                          │
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
| [`oxillama-gguf`](crates/oxillama-gguf) | GGUF v3 parser and tensor loader | ~3,100 |
| [`oxillama-quant`](crates/oxillama-quant) | Quantization kernels (25 formats, SIMD) | ~20,500 |
| [`oxillama-arch`](crates/oxillama-arch) | Model architectures (20 architectures) | ~7,500 |
| [`oxillama-runtime`](crates/oxillama-runtime) | Inference engine, KV cache, sampling | ~5,400 |
| [`oxillama-server`](crates/oxillama-server) | OpenAI-compatible HTTP API server | ~1,500 |
| [`oxillama-bench`](crates/oxillama-bench) | Benchmark suite | ~640 |
| [`oxillama-gpu`](crates/oxillama-gpu) | Optional wgpu GPU backend | ~870 |
| [`oxillama-py`](crates/oxillama-py) | Python bindings via PyO3 | ~1,100 |
| [`oxillama-wasm`](crates/oxillama-wasm) | WebAssembly bindings | ~150 |
| [`oxillama-cli`](crates/oxillama-cli) | CLI binary (`cargo install oxillama-cli`) | ~430 |

**Total: ~87,400 lines of Pure Rust across 11 crates** (as of v0.1.1, 1,898 tests passing)

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
| `minicpm` | MiniCPM | Alpha |
| `internlm3` | InternLM3 | Alpha |
| `jamba` | Jamba (hybrid attention + SSM) | Alpha |
| `mixtral` | Mixtral (sparse MoE) | Alpha |

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

## What's New in v0.1.1 (2026-04-24)

- **FlashAttention tiled CPU kernel** — BQ=BK=64 blocking, online softmax, rayon per-head parallelism; removes the full N×N attention matrix allocation.
- **True continuous batching** — per-request KV slot allocation, `BatchedKvView` trait; enables heterogeneous request lengths with zero padding waste.
- **Fused dequant+GEMM** — Q4_0 and Q4_K AVX2 + NEON paths skip the scratch buffer entirely; measured ~12% throughput gain on Q4_K_M decode.
- **oxiblas float GEMM fallback** — F16/BF16/F32 tensor paths now route through OxiBLAS GEMM rather than naive loops.
- **Tiled GEMM WGSL shader** — TILE_M/N=32, TILE_K=16, shared memory cooperative load; replaces naive GPU matmul for prefill workloads.
- **Fused attention WGSL kernel** — QK + softmax + AV in a single GPU dispatch, eliminating intermediate buffer round-trips.
- **4 new GPU GEMV kernels** — IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S quant GEMV on wgpu; GPU now covers 10 quantization types.
- **5 new architectures** — DBRX (16-expert MoE, top-4), Grok-1 (8-expert MoE, top-2), DeepSeek-V3 sigmoid-with-bias MoE scoring, Mamba-2 (selective scan, learned Δ), plus OLMo2, Yi, Granite, MiniCPM, InternLM3.
- **SequenceState trait** — arch-internal SSM abstraction that generalises the KV cache slot interface to state-space models.

See [CHANGELOG.md](CHANGELOG.md) for the full diff.

---

## COOLJAPAN Ecosystem

OxiLLaMa is built on the COOLJAPAN Pure Rust sovereignty stack:

```
OxiLLaMa
├── SciRS2 v0.4.x (tensor primitives, neural ops)
├── OxiBLAS v0.2.x (Pure Rust BLAS: GEMM, GEMV)
├── OxiFFT v0.2.x (Pure Rust FFT: RoPE acceleration)
└── MeCrab (Japanese tokenization)
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

*Copyright 2026 COOLJAPAN OU (Team KitaSan). All rights reserved. — OxiLLaMa v0.1.1 (2026-04-24)*

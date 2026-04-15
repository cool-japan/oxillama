# Changelog

All notable changes to OxiLLaMa are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
OxiLLaMa uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-15

### Added

#### Core
- GGUF v3 binary format parser (`oxillama-gguf`): magic, version, KV metadata, tensor info, mmap loading
- 25 quantization types (`oxillama-quant`): F32/F16/BF16 pass-through; legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1; K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K; I-quants IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS; Q1_0_G128
- SIMD-accelerated kernels: AVX-512 → AVX2 → NEON → scalar dispatch for Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q1_0_G128

#### Architectures (`oxillama-arch`)
- LLaMA 2/3/4 with GQA + RoPE
- Qwen3 with attention bias
- Mistral with sliding-window attention
- Gemma 2/3 with GeGLU, post-norm, logit soft-capping
- Phi-3/4 with merged QKV and partial RoPE
- StarCoder (GPT-BigCode) with MQA, LayerNorm, GELU, absolute position embeddings
- Command-R/R+ with logit scaling and optional Q/K norms
- Mixtral-MoE: LLaMA extended with sparse MoE FFN routing (Mixtral-7B/8x7B compatible)
- LLaVA-1.5 multimodal: full CLIP ViT-L/14 encoder (patch extraction, CLS token, position embeddings, N transformer layers, post-LN) + MmProjector 2-layer MLP

#### Runtime (`oxillama-runtime`)
- Paged KV cache with block-level memory management
- Sampling: greedy argmax, top-K, top-P (nucleus), temperature scaling, min-P, mirostat-v2, repetition penalty, seeded RNG
- GBNF grammar-constrained sampling
- Rayon row-parallel GEMV (feature-gated, scalar fallback for WASM)
- Continuous batching (`BatchRequest` queue + worker task; KV-cache reset between requests)
- Speculative decoding (`SpeculativeEngine`): draft/target model pair, token-level accept/reject, KV-cache resync
- LoRA adapter loading: `LoadedLora` from GGUF, `apply_lora()` runtime API, `QuantLinear` LoRA field

#### Server (`oxillama-server`)
- OpenAI-compatible HTTP API: `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`
- SSE streaming with `data:` lines and `[DONE]` sentinel
- llama.cpp CLI flag aliases: `-n/--n-predict`, `--temperature`, `-c/--n-ctx`, `--seed`, `--repeat-penalty`, `--min-p`
- `bench` subcommand for throughput measurement

#### Bindings
- Python bindings (`oxillama-py`): `Engine`, `SpeculativeEngine`, `LoadedLora` via PyO3 0.24; GIL-releasing inference; streaming callback; maturin wheel
- WebAssembly bindings (`oxillama-wasm`): `InferenceEngine::load_model_from_bytes()`, GGUF parsing exposed via wasm-bindgen

#### GPU Backend (`oxillama-gpu`)
- wgpu 29.0.1 compute backend (feature-gated `gpu = ["dep:wgpu"]`, off by default)
- Q4_0 + Q8_0 WGSL f32 GEMV shaders; `GpuDispatcher::try_init()` with graceful CPU fallback

#### Quality
- 1,205 tests, 0 warnings
- 87%+ test coverage (region/function/line)
- 3 cargo-fuzz targets for GGUF parser
- Criterion benchmarks: all 25 quant types + sampling pipeline

#### Project Structure
- `oxillama` meta crate: unified re-export of all subcrates (`oxillama::gguf`, `oxillama::quant`, etc.)
- `oxillama-cli` binary crate: CLI moved from workspace root to `crates/oxillama-cli/`

[0.1.0]: https://github.com/cool-japan/oxillama/releases/tag/v0.1.0

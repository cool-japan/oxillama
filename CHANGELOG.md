# Changelog

All notable changes to OxiLLaMa are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
OxiLLaMa uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-05-03

### Added

#### Server Prefix-KV Cache Wiring (`oxillama-server` + `oxillama-runtime`)
- **`InferenceEngine::prime_with_prefix`**: restores KV cache from a `CachedKvState` snapshot then forward-passes suffix tokens, returning initial logits for the decode loop — skips re-prefilling shared system prompts on cache hits
- **`InferenceEngine::generate_with_logits`**: decode-only loop starting from pre-computed initial logits; used after a prefix-cache hit to avoid a second prefill pass
- **`InferenceEngine::store_kv_in_prefix_cache`**: public helper that snapshots current KV state into a `PrefixKvCache` without exposing the mutable KV reference across crate boundaries
- **`CachedKvState::new`**: public constructor enabling reconstruction from cloned K/V buffers after the `Mutex` guard is released
- **Server-side `PrefixKvCache` wiring**: `AppState` now holds `prefix_cache: Arc<Mutex<PrefixKvCache>>`; the worker thread looks up the longest matching token prefix, restores KV state on hit, generates via `generate_with_logits`, and stores the post-generation KV state; full-prefill fallback on miss
- **Per-request `cache_prompt` flag**: new `cache_prompt: bool` field on `ChatCompletionRequest` (default `true`) and `BatchRequest::Generate` / `GenerateStream`; setting `false` disables caching for that request
- **Type aliases for complex worker types**: `PrefixHitData` and `WorkerHandles` suppress clippy "very complex type" lint without losing information

#### Server Multi-LoRA Registry + Per-Request Adapter Selection (`oxillama-server` + `oxillama-arch`)
- **`InferenceEngine::unapply_all_loras`**: clears `lora_stack` and calls `ForwardPass::unapply_all_loras()` — properly reverts all `QuantLinear.lora` fields set by `apply_lora_stack()`, restoring the base model weights for the next request
- **`ForwardPass::unapply_all_loras`**: new default no-op trait method; overridden in LLaMA, LLaVA, and Command-R model impls to iterate all linear layers and call `clear_lora()`
- **`AppState::loras` registry**: `loras: Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>` field on `AppState`; `spawn_inference_worker` accepts the registry Arc and resolves adapter names inside the worker
- **Per-request LoRA selection**: new `lora_selection: Vec<(String, f32)>` on `BatchRequest::Generate` / `GenerateStream`; chat route parses `"lora": "name"` (string form with scale 1.0) or `"lora": [{"name": "...", "scale": 0.8}]` (array form), resolves names → 400 on unknown
- **Admin LoRA CRUD endpoints**: `POST /admin/loras` (load GGUF + register), `DELETE /admin/loras/{name}` (unregister, 404 if not found), `GET /admin/loras` (list registered names); backed by `admin/loras.rs` (~220 LoC, 5 tests)
- **`LoraSelection` public type alias**: re-exported from `oxillama_server` for use in downstream crates

#### AVX-512 K-Quant Kernels (`oxillama-quant`)
- **`Q2_KAvx512`** (`simd/avx512/q2_k.rs`, ~700 LoC): full AVX-512F dequant + GEMV kernel for Q2_K super-block format (84 bytes, 256 weights, 2-bit packed with scale-of-scales); 2× wider lanes vs the AVX2 path via `_mm512_*` intrinsics; registered in `dispatch.rs`
- **`Q3_KAvx512`** (`simd/avx512/q3_k.rs`, ~830 LoC): full AVX-512F dequant + GEMV kernel for Q3_K (110 bytes, 3-bit packed from 32-byte `qs` + 8-byte `hmask`, 6-bit scale array); signed [-4..3] weight reconstruction; registered in `dispatch.rs`
- AVX-512 coverage table now includes Q2_K and Q3_K (previously scalar + AVX2 only)

#### GPU Legacy Quad Kernels (`oxillama-gpu`)
- **`Q4_1GpuKernel`** (`kernels/q4_1.rs`, ~370 LoC): WGSL GEMV kernel for Q4_1 blocks (20 bytes: 2-byte `d` + 2-byte `m` + 16 nibble bytes); registered in `GpuDispatcher`
- **`Q5_0GpuKernel`** (`kernels/q5_0.rs`, ~410 LoC): WGSL GEMV kernel for Q5_0 blocks (22 bytes: 2-byte `d` + 4-byte `qh` high bits + 16-byte `qs`); 5-bit unpacking with separate high-bit array
- **`Q5_1GpuKernel`** (`kernels/q5_1.rs`, ~430 LoC): WGSL GEMV kernel for Q5_1 blocks (24 bytes: 2-byte `d` + 2-byte `m` + 4-byte `qh` + 16-byte `qs`)
- **`Q8_1GpuKernel`** (`kernels/q8_1.rs`, ~360 LoC): WGSL GEMV kernel for Q8_1 blocks (36 bytes: 2-byte `d` + 2-byte `sum` + 32 signed-byte `qs`)
- GPU dispatcher now covers 18 quantization types (was 14); Q4_1/Q5_0/Q5_1/Q8_1 cover ~85% of community-quantized HuggingFace GGUF uploads

#### Quality
- **1,873 tests passing**, up from 1,825 in v0.1.2; 0 warnings maintained

## [0.1.2] - 2026-04-25

### Added

#### Session Persistence (`oxillama-runtime`)
- **Conversation save/resume** (`session.rs`): `Session::save()` and `Session::load()` serialize the full conversation history via `oxicode` (COOLJAPAN Pure Rust codec); SHA-256 KV sidecar validates integrity on load; schema version guard rejects incompatible session files
- **`/save` and `/load` slash commands**: interactive CLI commands to persist and restore conversation sessions across process restarts

#### HuggingFace Hub Integration (`oxillama-cli`)
- **`oxillama hub pull/list/rm` subcommands**: download models directly from HuggingFace Hub (`hf-hub 0.5`), list cached models, and remove cached entries; uses `ureq` with `rustls` for Pure Rust TLS and the `directories` crate for platform-appropriate cache paths

#### TUI Chat Mode (`oxillama-cli`)
- **Full-screen TUI chat** (`ratatui 0.30` + `crossterm 0.29`): scrollable chat history, input line, live streaming via `spawn_blocking` + `mpsc` channel for async token delivery without blocking the TUI event loop; 6 unit tests covering layout, input handling, and message rendering

#### New Architecture Loaders (`oxillama-arch`)
- **DBRX GGUF loader**: support for Databricks DBRX mixture-of-experts architecture
- **Grok-1 GGUF loader**: support for xAI Grok-1 architecture
- **Mamba-2 GGUF loader**: support for state-space model architecture with `embed()` override for Mamba-2-specific token embedding logic

#### KV Cache API Extensions (`oxillama-arch`)
- **`KvCacheAccess` trait extensions**: new `kv_dim()`, `for_each_key()`, and `for_each_value()` methods on the `KvCacheAccess` trait; `PagedKvCache` fully implements multi-page support for all three methods
- **`BatchedKvView` + `KvSlot`**: moved to `oxillama-arch/traits.rs` for cross-crate reuse
- **`ForwardPass::forward_batched`**: default implementation added to the trait; LLaMA provides a concrete optimized implementation

#### Quality
- **2,020 tests passing**, up from 1,979 in v0.1.1

## [0.1.1] - 2026-04-24

### Added

#### GGUF Loader Hardening (`oxillama-gguf`)
- **Partial-download resume** (`resume.rs`): `GgufModel::resume()` reads an adjacent `.oxiresume` sidecar checkpoint, validates the last-valid byte offset, and provides a `ResumeHandle::finish()` path once the download completes — survives interrupted HuggingFace pulls without re-downloading
- **Sharded multi-file loading** (`sharded.rs`): `ShardedGgufModel::load_sharded()` auto-discovers all HuggingFace-named sibling shards (`<base>-NNNNN-of-MMMMM.gguf`) from a single shard path and presents a unified logical model
- **Quantize-on-the-fly** (`quantize_on_load.rs`): optional pass that dequantizes and re-quantizes tensors to a target format during load, eliminating a separate conversion step for deployment

#### Runtime Snapshot/Resume (`oxillama-runtime`)
- **`EngineSnapshot`** (`snapshot.rs`): captures the full KV-cache and sampler RNG state into a byte blob via `InferenceEngine::snapshot()`; `InferenceEngine::resume()` validates the model fingerprint and restores from the blob, enabling session persistence across process restarts
- **Oxicode serialization**: `EngineSnapshot` is serialized with `oxicode` (COOLJAPAN Pure Rust codec) rather than `bincode`, in compliance with the workspace serialization policy

#### Facade Examples & Cookbook (`oxillama`)
- **`examples/load_and_generate.rs`**: end-to-end example: load a GGUF, configure the sampler, and stream tokens to stdout
- **`examples/lora_apply.rs`**: demonstrates hot-swapping two LoRA adapters on a running engine without model reload
- **`examples/speculative.rs`**: shows the `SpeculativeEngine` API with a 1B draft model and 70B target
- **`RECIPES.md`**: 8-recipe task-oriented cookbook covering generation, serving, LoRA, speculative decoding, snapshot/resume, WASM browser chat, partial-download resume, and sharded model loading

#### Quantization (`oxillama-quant`)
- **AVX2 kernels**: Q4_1, Q5_0, Q5_1, Q8_1 — 4 new legacy-quant AVX2 dot-product kernels, completing full AVX2 coverage for all legacy quantization types
- **NEON kernels**: Q4_1, Q5_0, Q5_1, Q8_1, Q2_K, Q3_K — 6 new Apple Silicon NEON kernels; combined with IQ/TQ additions gives near-complete NEON coverage across all quantization families
- **NEON AArch64 kernels**: IQ2_XXS, IQ2_XS, IQ3_S, IQ4_XS, IQ4_NL, TQ1_0, TQ2_0, IQ1_S, IQ1_M, IQ3_XXS, IQ2_S — all 11 IQ types now have Apple Silicon NEON acceleration
- **AVX-512 kernels**: TQ1_0, TQ2_0, Q5_0, Q8_K — AVX-512 coverage extended to 10 types

#### GPU Backend (`oxillama-gpu`)
- **Q2_K GEMV shader**: WGSL compute shader for Q2_K dequant + dot-product on GPU
- **Q3_K GEMV shader**: WGSL compute shader for Q3_K dequant + dot-product on GPU
- **Q8_K GEMV shader**: WGSL compute shader for Q8_K dequant + dot-product on GPU
- **IQ4_XS GEMV shader**: WGSL compute shader for IQ4_XS dequant + dot-product on GPU
- **Async WebGPU bridge** (`gpu_bridge.rs`): `initWebGpuDevice()`, `webgpuDequantQ4_0Async()`, `webgpuGemvAsync()` using `wasm_bindgen_futures::JsFuture` for real GPU dispatch in browsers with WebGPU support

#### Architectures (`oxillama-arch`)
- **Multi-head Latent Attention (MLA)**: Low-rank KV compression primitive (`MlaLayer`) with decoupled RoPE — reduces KV-cache memory footprint by up to 93% vs standard MHA
- **DeepSeek-V2 architecture**: Full `DeepSeekV2Model` with MLA attention, DeepSeekMoE sparse FFN routing (N shared experts + top-K routed experts), 3-bit/8-bit quantized expert dispatch

#### Developer Experience
- **oxillama (meta)**: `openai_server.rs` example showing programmatic server startup; `python_bridge.rs` example documenting Rust↔Python API parity
- **oxillama-cli**: Colorized output via `colored` (cyan/bold key labels, green banners); `indicatif` spinner progress bar during model loading; 5 integration smoke tests in `tests/cli_smoke.rs`

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

[0.1.1]: https://github.com/cool-japan/oxillama/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxillama/releases/tag/v0.1.0

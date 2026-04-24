# oxillama

Unified meta crate that re-exports the full OxiLLaMa API surface.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxillama = "0.1.1"
```

Then use any subcrate through the unified namespace:

```rust,no_run
use oxillama::gguf::GgufModel;
use oxillama::quant::QuantKernel;
use oxillama::arch::ForwardPass;
use oxillama::runtime::{InferenceEngine, EngineConfig};
```

## Modules

| Module | Crate | Description |
|--------|-------|-------------|
| `gguf` | oxillama-gguf | GGUF v3 parser and tensor loader |
| `quant` | oxillama-quant | Quantization kernels (25 formats) |
| `arch` | oxillama-arch | Model architectures (20 architectures) |
| `runtime` | oxillama-runtime | Inference engine, KV cache, sampling |
| `server` | oxillama-server | OpenAI-compatible HTTP API (feature: `server`) |
| `bench` | oxillama-bench | Benchmark suite (feature: `bench`) |
| `gpu` | oxillama-gpu | wgpu GPU backend (feature: `gpu`) |

## Documentation

- **[RECIPES.md](RECIPES.md)** — 8 task-oriented code recipes (load & generate, serve, LoRA, speculative decoding, snapshot/resume, WASM, partial-download resume, sharded model loading)

## Tests

Meta-crate test suite: `feature_matrix`, `error_types`, `recipes_doctest` — **19 passing**.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `server` | yes | Enable OpenAI-compatible server |
| `bench` | yes | Enable benchmark suite |
| `gpu` | no | Enable wgpu GPU backend |
| `simd-avx2` | no | AVX2 SIMD kernels |
| `simd-avx512` | no | AVX-512 SIMD kernels |
| `simd-neon` | no | ARM NEON SIMD kernels |
| `llama` | no | LLaMA architecture |
| `qwen3` | no | Qwen3 architecture |
| `mistral` | no | Mistral architecture |
| `gemma` | no | Gemma architecture |
| `phi` | no | Phi architecture |
| `command-r` | no | Command-R architecture |
| `starcoder` | no | StarCoder architecture |
| `deepseek` | no | DeepSeek architecture |
| `dbrx` | yes | DBRX architecture |
| `grok` | yes | Grok-1 architecture |
| `mamba2` | yes | Mamba-2 SSM architecture |
| `llava` | no | LLaVA multimodal (requires `llama`) |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

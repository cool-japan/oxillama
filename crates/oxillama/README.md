# oxillama

Unified meta crate that re-exports the full OxiLLaMa API surface.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxillama = "0.1.0"
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
| `arch` | oxillama-arch | Model architectures (8 models) |
| `runtime` | oxillama-runtime | Inference engine, KV cache, sampling |
| `server` | oxillama-server | OpenAI-compatible HTTP API (feature: `server`) |
| `bench` | oxillama-bench | Benchmark suite (feature: `bench`) |
| `gpu` | oxillama-gpu | wgpu GPU backend (feature: `gpu`) |

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
| `llava` | no | LLaVA multimodal (requires `llama`) |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

# oxillama

Unified meta crate that re-exports the full OxiLLaMa API surface.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxillama = "0.1.4"
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
| `arch` | oxillama-arch | Model architectures (26 architectures) |
| `runtime` | oxillama-runtime | Inference engine, KV cache, sampling |
| `server` | oxillama-server | OpenAI-compatible HTTP API (feature: `server`) |
| `bench` | oxillama-bench | Benchmark suite (feature: `bench`) |
| `gpu` | oxillama-gpu | wgpu GPU backend (feature: `gpu`) |

## Documentation

- **[RECIPES.md](RECIPES.md)** — 8 task-oriented code recipes (load & generate, serve, LoRA, speculative decoding, snapshot/resume, WASM, partial-download resume, sharded model loading)

## Tests

Meta-crate test suite: `feature_matrix`, `error_types`, `recipes_doctest` — **20 passing**.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `server` | yes | Enable OpenAI-compatible server |
| `bench` | no | Enable benchmark suite — depend on `oxillama-bench` directly, or enable this explicitly; dropped from `default` because it pulled the full Criterion stack (plotters, rayon, regex) plus `tabled`/`sysinfo` into every downstream `cargo add oxillama` |
| `gpu` | no | Enable wgpu GPU backend (device-resident Q4_0 decode offload; no `--gpu` CLI flag exists yet in `oxillama-cli`) |
| `simd-avx2` | yes | AVX2 SIMD kernels |
| `simd-avx512` | no | AVX-512 SIMD kernels |
| `simd-neon` | yes | ARM NEON SIMD kernels |
| `llama` | yes | LLaMA architecture — `default` names at least one architecture feature so a caller who drops `server` doesn't get a crate that compiles cleanly and then fails at *runtime* with "unsupported architecture" |
| `qwen3` | no | Qwen3 architecture |
| `mistral` | no | Mistral architecture |
| `gemma` | no | Gemma architecture |
| `phi` | no | Phi architecture |
| `command-r` | no | Command-R architecture |
| `starcoder` | no | StarCoder architecture |
| `deepseek` | no | DeepSeek architecture |
| `dbrx` | no | DBRX architecture |
| `grok` | no | Grok-1 architecture |
| `mamba2` | no | Mamba-2 SSM architecture |
| `jamba` | no | Jamba hybrid attention+SSM architecture |
| `llava` | no | LLaVA multimodal (requires `llama`) |

> **Note:** `deepseek`, `dbrx`, `grok`, `mamba2`, `jamba`, and `llava` currently
> forward only to `oxillama-arch` (`Cargo.toml`: e.g. `jamba = ["oxillama-arch/jamba"]`)
> — `oxillama-runtime` has no matching feature for any of the six (verified: absent
> from `oxillama-runtime`'s `[features]`), so enabling one of these flags compiles
> the architecture module but `InferenceEngine` still has no dispatch path to run
> it. Only `llama`/`qwen3`/`mistral`/`gemma`/`phi`/`command-r`/`starcoder` forward
> to both `oxillama-arch` and `oxillama-runtime` and are runnable end-to-end today.

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

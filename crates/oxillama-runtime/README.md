# oxillama-runtime

Full inference runtime for transformer LLMs — KV cache, sampling, tokenizer, and advanced decoding.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- **InferenceEngine**: single-batch and continuous-batch forward pass over any architecture
- **Paged KV cache**: block-manager for efficient memory reuse across multiple requests
- **Sampling pipeline**: greedy, top-K, top-P (nucleus), min-P, temperature, repetition penalty, mirostat v1/v2, grammar-constrained (GBNF)
- **Tokenizer bridge**: HuggingFace `tokenizers` with `onig` (native) or `unstable_wasm` (pure-Rust) regex backends
- **LoRA adapters**: load and hot-swap rank-decomposition adapters at runtime
- **Speculative decoding**: draft-model + verifier pipeline (`SpeculativeEngine`)

## Key Types

| Type | Description |
|------|-------------|
| `InferenceEngine` | Main engine; wraps model + cache + sampler |
| `SamplerConfig` | Builder for all sampling hyper-parameters |
| `PagedKvCache` | Block-paged KV store with eviction policy |
| `SpeculativeEngine` | Draft+target model pair for speculative decoding |
| `LoadedLora` | In-memory LoRA adapter ready to apply |
| `RuntimeError` | Unified error wrapping `ArchError`, `GgufError`, `QuantError` |

## Usage

```rust
use oxillama_runtime::{InferenceEngine, SamplerConfig, RuntimeResult};

fn generate(model_path: &str, prompt: &str) -> RuntimeResult<String> {
    let engine = InferenceEngine::from_gguf(model_path)?;

    let sampler = SamplerConfig::builder()
        .temperature(0.8)
        .top_p(0.95)
        .max_new_tokens(256)
        .build();

    let output = engine.generate(prompt, &sampler)?;
    Ok(output)
}
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `llama` | yes | LLaMA 2/3/4 architecture |
| `qwen3` | yes | Qwen3 architecture |
| `mistral` | yes | Mistral / Mixtral architecture |
| `gemma` | yes | Gemma 2/3 architecture |
| `phi` | yes | Phi-3/4 architecture |
| `command-r` | yes | Command-R architecture |
| `starcoder` | yes | StarCoder 2 architecture |
| `tokenizer-onig` | yes | HF tokenizers with Oniguruma regex (recommended for desktop) |
| `tokenizer-wasm` | no | HF tokenizers with pure-Rust regex (required for WASM) |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

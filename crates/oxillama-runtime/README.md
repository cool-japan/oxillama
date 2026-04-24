# oxillama-runtime

Full inference runtime for transformer LLMs — KV cache, sampling, tokenizer, and advanced decoding.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.1 — **Tests:** 343 passing — **Completion:** ~98% — **Status:** Alpha

## What It Provides

- **InferenceEngine**: single-batch and continuous-batch forward pass over any architecture
- **Paged KV cache**: block-manager for efficient memory reuse across multiple requests
- **FlashAttention**: tiled CPU kernel (`BQ=BK=64`, online softmax, causal masking) dispatched above `FLASH_ATTN_THRESHOLD=512` tokens
- **Continuous batching**: `BatchedKvView` trait + `KvSlot` struct + `VecBatchedKvView` for true per-request KV slot isolation
- **Sampling pipeline**: greedy, top-K, top-P (nucleus), min-P, temperature, repetition penalty, mirostat v1/v2, grammar-constrained (GBNF)
- **Tokenizer bridge**: HuggingFace `tokenizers` with `onig` (native) or `unstable_wasm` (pure-Rust) regex backends
- **LoRA adapters**: load and hot-swap rank-decomposition adapters at runtime via `LoraStack`
- **Speculative decoding**: draft-model + verifier pipeline (`SpeculativeEngine`) with delta-sync KV resync

## Key Types

| Type | Description |
|------|-------------|
| `InferenceEngine` | Main engine; wraps model + cache + sampler |
| `SamplerConfig` | Builder for all sampling hyper-parameters |
| `PagedKvCache` | Block-paged KV store with eviction policy |
| `SpeculativeEngine` | Draft+target model pair for speculative decoding |
| `LoadedLora` | In-memory LoRA adapter ready to apply |
| `Grammar` / `GrammarState` | GBNF grammar parser and logit-mask state machine |
| `Scheduler` | Continuous-batching scheduler with prefill priority and chunked prefill |
| `BatchedKvView` | Trait for per-request KV slot access (continuous batching) |
| `KvSlot` | Per-request KV slot allocated from `VecBatchedKvView` |
| `RuntimeError` | Unified error wrapping `ArchError`, `GgufError`, `QuantError` |
| `EngineSnapshot` / `ModelFingerprint` | Session snapshot and resume via oxicode — **new in v0.1.1** |
| `ToolDispatcher` / `ToolCallDetector` / `ToolCall` / `NoOpDispatcher` | Tool/function-calling trait and helpers — **new in v0.1.1** |
| `SpeculativeDecoder` / `AsyncSpecConfig` / `SpecStats` | Async speculative decoding pipeline — **new in v0.1.1** |
| `PrefixKvCache` / `PrefixCacheConfig` | Prompt-prefix KV cache with radix-tree lookup — **new in v0.1.1** |
| `KvCachePool` | Pooled KV cache allocator for multi-request reuse — **new in v0.1.1** |
| `EngineMetrics` / `MetricsSnapshot` | Prometheus-compatible lock-free counters — **new in v0.1.1** |
| `SequencePool` / `SsmStatePool` | Attention and SSM sequence state pools — **new in v0.1.1** |

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
| `falcon` | yes | TII Falcon (passes through to `oxillama-arch`) |
| `minicpm` | yes | MiniCPM scaled embedding (passes through to `oxillama-arch`) |
| `olmo2` | yes | Allen AI OLMo2 (passes through to `oxillama-arch`) |
| `granite` | yes | IBM Granite 3.x (passes through to `oxillama-arch`) |
| `deepseek` | yes | DeepSeek-V2/V3 MLA+MoE (passes through to `oxillama-arch`) |
| `dbrx` | yes | Databricks DBRX (passes through to `oxillama-arch`) |
| `grok` | yes | xAI Grok-1 (passes through to `oxillama-arch`) |
| `mamba2` | yes | Mamba-2 selective-scan SSM (passes through to `oxillama-arch`) |
| `jamba` | yes | Hybrid attention+SSM (passes through to `oxillama-arch`, enables `mamba2`) |
| `tokenizer-wasm` | yes | HF tokenizers with pure-Rust regex (required for WASM) |
| `tokenizer-onig` | no | HF tokenizers with Oniguruma regex (native desktop alternative) |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

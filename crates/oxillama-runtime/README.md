# oxillama-runtime

Full inference runtime for transformer LLMs — KV cache, sampling, tokenizer, and advanced decoding.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.2 — **Tests:** 370 passing — **Completion:** ~98% — **Status:** Alpha

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
| `RuntimeError` | Unified error wrapping `ArchError`, `GgufError`, `QuantError` |
| `EngineSnapshot` / `ModelFingerprint` | Session snapshot and resume via oxicode — **new in v0.1.1** |
| `ToolDispatcher` / `ToolCallDetector` / `ToolCall` / `NoOpDispatcher` | Tool/function-calling trait and helpers — **new in v0.1.1** |
| `SpeculativeDecoder` / `AsyncSpecConfig` / `SpecStats` | Async speculative decoding pipeline — **new in v0.1.1** |
| `PrefixKvCache` / `PrefixCacheConfig` | Prompt-prefix KV cache with radix-tree lookup — **new in v0.1.1** |
| `KvCachePool` | Pooled KV cache allocator for multi-request reuse — **new in v0.1.1** |
| `EngineMetrics` / `MetricsSnapshot` | Prometheus-compatible lock-free counters — **new in v0.1.1** |
| `SequencePool` / `SsmStatePool` | Attention and SSM sequence state pools — **new in v0.1.1** |
| `KvCacheAccess` | Trait extension: `kv_dim`, `for_each_key`, `for_each_value` with contiguous defaults and `PagedKvCache` multi-page overrides — **new in v0.1.2** |
| `BatchedKvView` / `KvSlot` | Moved to `oxillama-arch/traits.rs`; re-exported from `oxillama-runtime` for backwards compatibility — **new in v0.1.2** |
| `ForwardPass::forward_batched` | Default impl on `ForwardPass` trait; LLaMA proof-of-concept continuous-batch forward — **new in v0.1.2** |

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
| `tokenizer-wasm` | yes | HF tokenizers with pure-Rust regex (required for WASM) |
| `tokenizer-onig` | no | HF tokenizers with Oniguruma regex (native desktop alternative) |
| `parallel` | no | Multi-threaded tensor ops via rayon |
| `native-async` | no | Tokio-backed async engine API |
| `mmap` | no | Memory-mapped model file loading |
| `offload` | no | Tensor offload to secondary storage |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

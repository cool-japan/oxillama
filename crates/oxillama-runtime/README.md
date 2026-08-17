# oxillama-runtime

Full inference runtime for transformer LLMs — KV cache, sampling, tokenizer, and advanced decoding.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.5 — **Tests:** 701 passing — **Status:** Alpha

## What It Provides

- **InferenceEngine**: single-batch and continuous-batch forward pass over any architecture
- **Lazy-growth KV cache**: blocks are allocated as the sequence advances rather than eagerly at the full context (1024 MiB → 64 MiB for a 20-token conversation on Llama-3-8B)
- **Selectable KV storage dtype**: `KvCacheDtype::F16` halves the cache (`EngineConfig::kv_dtype`, `oxillama run|serve --kv-dtype f16`); attention still accumulates in f32. Default stays `F32` — llama.cpp defaults to f16, OxiLLaMa does not, so upgrading never changes existing output
- **Sampling pipeline**: greedy, top-K, top-P (nucleus), min-P, temperature, repetition penalty, mirostat v1/v2, grammar-constrained (GBNF)
- **Tokenizer bridge**: HuggingFace `tokenizers` with `onig` (native) or `unstable_wasm` (pure-Rust) regex backends
- **LoRA adapters**: load and hot-swap rank-decomposition adapters at runtime via `LoraStack`
- **Speculative decoding**: draft-model + verifier pipeline (`SpeculativeEngine`) with delta-sync KV resync

## Key Types

| Type | Description |
|------|-------------|
| `InferenceEngine` | Main engine; wraps model + cache + sampler |
| `SamplerConfig` | Builder for all sampling hyper-parameters |
| `PagedKvCache` | Block-paged KV store with eviction policy — implemented and unit-tested, **not yet wired into the inference path** |
| `SpeculativeEngine` | Draft+target model pair for speculative decoding |
| `LoadedLora` | In-memory LoRA adapter ready to apply |
| `Grammar` / `GrammarState` | GBNF grammar parser and logit-mask state machine |
| `Scheduler` | Continuous-batching scheduler with prefill priority and chunked prefill — implemented and unit-tested, **no caller**; the server runs one sequential worker |
| `RuntimeError` | Unified error wrapping `ArchError`, `GgufError`, `QuantError` |
| `EngineSnapshot` / `ModelFingerprint` | Session snapshot and resume via oxicode — **new in v0.1.1** |
| `ToolDispatcher` / `ToolCallDetector` / `ToolCall` / `NoOpDispatcher` | Tool/function-calling trait and helpers — **new in v0.1.1** |
| `SpeculativeDecoder` / `AsyncSpecConfig` / `SpecStats` | Async speculative decoding pipeline — **new in v0.1.1** |
| `PrefixKvCache` / `PrefixCacheConfig` | Prompt-prefix KV cache with radix-tree lookup — **new in v0.1.1** |
| `KvCachePool` | Pooled KV cache allocator for multi-request reuse — implemented and unit-tested, **not integrated** |
| `EngineMetrics` / `MetricsSnapshot` | Prometheus-compatible lock-free counters — **new in v0.1.1** |
| `SequencePool` / `SsmStatePool` | Attention and SSM sequence state pools — implemented and unit-tested, **no caller** |
| `KvCacheAccess` | Trait extension: `kv_dim`, `for_each_key`, `for_each_value` with contiguous defaults and `PagedKvCache` multi-page overrides — **new in v0.1.2** |
| `KvCacheDtype` | KV storage element type (`F32` / `F16`); `KvCache::with_dtype`, `EngineConfig::kv_dtype`. Every architecture reads the cache through `oxillama_arch::common::fetch_keys`/`fetch_values`, so F16 is reachable end-to-end — **new in v0.1.4** |
| `BatchedKvView` / `KvSlot` | Moved to `oxillama-arch/traits.rs`; re-exported from `oxillama-runtime` for backwards compatibility — **new in v0.1.2** |
| `ForwardPass::forward_batched` | Default impl on `ForwardPass` trait; LLaMA proof-of-concept continuous-batch forward — **new in v0.1.2** |
| `GpuPolicy` / `GpuOptions` / `GpuStatus` / `InferenceEngine::gpu_status()` | GPU offload policy (behind the `gpu` feature); offloads Q4_0 decode-time weight matrices to a device-resident kernel via `EngineConfig::gpu`. Default `GpuPolicy::Off`, CPU behavior byte-for-byte unchanged. Not yet exposed through a CLI flag or Python kwarg — **new in v0.1.4** |

## Usage

```rust
use oxillama_runtime::{EngineConfig, InferenceEngine, RuntimeResult, SamplerConfig};

fn generate(model_path: &str, prompt: &str) -> RuntimeResult<String> {
    let mut engine = InferenceEngine::new(EngineConfig {
        model_path: model_path.to_string(),
        sampler: SamplerConfig {
            temperature: 0.8,
            top_p: 0.95,
            ..SamplerConfig::default()
        },
        ..EngineConfig::default()
    });
    engine.load_model()?;

    // The callback receives each decoded chunk; chunks are always complete
    // UTF-8 sequences.  `generate` does NOT reset the KV cache — call
    // `engine.reset()` first if the prompt should not continue the previous
    // sequence.
    engine.generate(prompt, 256, |_chunk| {})
}
```

`generate_detailed` takes a `GenerationConfig` (stop sequences, per-call
sampler, special-token rendering) and returns a `GenerationOutcome` carrying
the text, a `FinishReason`, and token counts.

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
| `parallel` | yes | Multi-threaded tensor ops via rayon |
| `simd-neon` | no | ARM NEON SIMD passthrough to `oxillama-quant` / `oxillama-arch` |
| `simd-avx2` | no | AVX2 SIMD passthrough to `oxillama-quant` / `oxillama-arch` |
| `simd-avx512` | no | AVX-512 SIMD passthrough to `oxillama-quant` / `oxillama-arch` |
| `native-async` | yes | Tokio-backed async engine API |
| `mmap` | yes | Memory-mapped model file loading |
| `offload` | yes | Tensor offload to secondary storage |
| `gpu` | no | Device-resident Q4_0 decode offload via `oxillama-gpu` (wgpu); pulls in the wgpu stack, off by default — **new in v0.1.4** |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

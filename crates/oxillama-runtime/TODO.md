# oxillama-runtime — TODO

## 1. Overview

`oxillama-runtime` is the inference orchestration layer of OxiLLaMa. It
composes `oxillama-arch` forward passes into end-to-end text generation,
threading together the KV cache, samplers, tokenizer, LoRA application, and
the continuous-batching scheduler.

Responsibilities:

- Load a GGUF model (via `oxillama-gguf`) and instantiate the correct
  architecture (via `oxillama-arch`).
- Run the prefill / decode loop with a growing KV cache.
- Sample the next token (greedy, top-K/P, min-P, mirostat-v2, GBNF-masked).
- Drive speculative decoding against a draft/target pair.
- Accept and apply LoRA adapters loaded from GGUF.
- Schedule many concurrent sequences via a queue + worker pattern.
- Provide pooled-hidden-state embeddings for retrieval and reranking use
  cases (single and batched).

Public surface exposed at crate root: `InferenceEngine`, `EngineConfig`,
`SpeculativeEngine`, `SpeculativeConfig`, `Sampler`, `SamplerConfig`,
`Grammar`, `GrammarState`, `KvCache`, `Scheduler`, `SchedulerConfig`,
`TokenizerBridge`, `apply_lora`, `RuntimeError` / `RuntimeResult`.

All errors flow through `RuntimeError` (`thiserror`), which wraps
`ArchError`, `GgufError`, `QuantError`, `GrammarError`, and `std::io::Error`.
Pure Rust end-to-end: `scirs2-*` for tensor primitives, `oxiblas` for GEMM,
`oxifft` for RoPE where applicable. No C / C++ / Fortran / FFI on the
default feature set. The optional `tokenizer-onig` feature is the only
non-pure-Rust path, and it is off by default; the pure-Rust
`tokenizer-wasm` backend covers both native and wasm32 targets.

Dependency position: `gguf -> quant -> arch -> runtime -> server / bench`.
Downstream crates (`oxillama-server`, `oxillama-py`, `oxillama-wasm`,
`oxillama-cli`) consume `runtime` directly and should not reach past it
into `arch` unless they have a concrete reason.

## 2. Status Snapshot

| Field | Value |
|---|---|
| Version | 0.1.0 |
| Completion | ~93% |
| Source files | 15 (`src/**/*.rs`) |
| Largest file | `src/engine.rs` (~1.4K lines, under 2000-line policy) |
| Default tokenizer backend | `tokenizer-wasm` (pure Rust, wasm32-safe) |
| Alternate backend | `tokenizer-onig` (C regex, native only) |
| Per-arch features | `llama`, `qwen3`, `mistral`, `gemma`, `phi`, `command-r`, `starcoder` |
| Default features | all 7 archs + `tokenizer-wasm` |
| Deps | `oxillama-gguf`, `oxillama-quant`, `oxillama-arch`, `half`, `thiserror`, `tracing`, `rayon`, `serde`, `serde_json`, optional `tokenizers` |
| Dev deps | `criterion`, `oxillama-gguf/test-utils` |
| Benches | `sampling` (criterion) |
| `unwrap()` in production | 0 (policy: zero unwrap outside tests) |

## 3. Module Map

| Path | Role |
|---|---|
| `src/lib.rs` | Module declarations and public re-exports |
| `src/engine.rs` | `InferenceEngine`, `EngineConfig`; load, prefill, decode, `generate`, `generate_with_config`, `embed`, `embed_batch`, `forward_one`, `tokenize`, `decode_token`, `apply_lora_adapters`, `reset`, `is_eos` |
| `src/kv_cache/mod.rs` | `KvCache` (contiguous pre-allocated) implementing `KvCacheAccess`; `stored_len` invariant that lets attention read back K/V written in the same forward pass |
| `src/kv_cache/paged.rs` | `PagedKvCache` with 16-token pages, `get_keys_into`, `get_values_into`, `iter_keys`, `iter_values`, `shrink_to_fit`, `memory_bytes` |
| `src/sampling/mod.rs` | Stateless `sample()`, stateful `Sampler`, `SamplerConfig`, `Xorshift64` PRNG; greedy, top-K, top-P, min-P, temperature, repetition penalty, mirostat-v2 |
| `src/sampling/grammar/mod.rs` | Re-exports |
| `src/sampling/grammar/parser.rs` | GBNF grammar parser |
| `src/sampling/grammar/machine.rs` | `Grammar`, `GrammarState`, `apply_grammar_mask()` for logit masking |
| `src/sampling/grammar/error.rs` | `GrammarError` (`thiserror`) |
| `src/speculative.rs` | `SpeculativeEngine`, `SpeculativeConfig`; draft/target accept-reject, residual sampling, KV cache re-sync |
| `src/scheduler.rs` | `Scheduler`, `SchedulerConfig`, `Sequence`, `SeqState`, `ScheduledBatch`; prefill priority, decode round-robin, chunked prefill |
| `src/tokenizer_bridge.rs` | `TokenizerBridge` over HuggingFace `tokenizers` (onig or unstable_wasm); stub returns `TokenizerNotAvailable` with neither feature |
| `src/sampling/chain.rs` | `SamplerStage` trait, `SamplerChain`, built-in stages (composable pipeline) |
| `src/lora_loader.rs` | `apply_lora()` — parses LoRA GGUF via `oxillama-arch::lora::LoadedLora` and patches engine linear layers |
| `src/error.rs` | `RuntimeError` + `RuntimeResult<T>`; `#[from]` conversions for Arch, GGUF, Quant, Grammar, I/O |

## 4. Shipped in v0.1.0

Inference engine

- `InferenceEngine` with load + prefill + single-token decode cycle
- `generate()` streaming callback API and `generate_with_config()` override
- `embed()` and `embed_batch()` for pooled-hidden-state embeddings
- `forward_one()`, `tokenize()`, `decode_token()`, `is_eos()` primitives
- `reset()` to clear KV cache between generations
- `load_model_from_bytes()` for in-memory GGUF (wasm32 / tests)
- Integration tests covering all 8 supported architectures via
  `oxillama-gguf::test_utils::build_minimal_*_gguf()` fixtures

KV cache

- Contiguous `KvCache`: one K + one V buffer per layer, pre-allocated
- Paged `PagedKvCache`: 16-token pages, on-demand allocation, zero-copy
  single-page fast path, explicit `get_*_into(&mut Vec<f32>)` for multi-page
- Release-check bug fix for the `stored_len` vs `seq_len` invariant so
  same-step attention reads include the just-written entry

Sampling

- Greedy (argmax), top-K, top-P (nucleus), min-P, temperature scaling
- Repetition penalty with configurable window
- Mirostat v2 (adaptive surprise control, tau + eta + mu update)
- Seeded `Xorshift64` PRNG for deterministic replay
- GBNF grammar: parser, `GrammarState` machine, `apply_grammar_mask()`
  zeroes (to `-inf`) any logit whose token bytes would violate the grammar;
  applied before greedy shortcut so constraints hold at temperature = 0

Speculative decoding

- `SpeculativeEngine` owning draft + target `InferenceEngine`
- Standard accept-reject rule: accept when `p_target >= p_draft`, else
  accept with probability `p_target / p_draft`
- Residual sampling on rejection: `max(0, p_target - p_draft) / Z`
- Bonus-token sampling on full-acceptance branch
- KV cache resync by re-prefilling draft from accepted history

LoRA

- `apply_lora()` runtime API parses LoRA GGUF into `LoadedLora`, patches
  each `QuantLinear` whose tensor name matches an adapter entry
- Correction applied after main GEMV on every subsequent forward

Tokenizer bridge

- `TokenizerBridge` wraps HuggingFace `tokenizers`
- Native build: `tokenizer-onig` (Oniguruma C regex)
- WASM / pure-Rust build: `tokenizer-wasm` (fancy-regex via
  `tokenizers/unstable_wasm`) — default
- `vocab_bytes()` pre-computes `(id, bytes)` table used by grammar masking
- Neither feature: compile-time stub returns `TokenizerNotAvailable`

Scheduler (continuous-batching scaffolding)

- `Scheduler` with waiting queue + active list
- `Sequence` states: `Waiting`, `Prefilling`, `Decoding`, `Finished`,
  `Preempted`
- `schedule()` produces a `ScheduledBatch` (prefill priority, decode
  round-robin, chunked prefill bounded by `max_prefill_tokens`)
- `drain_finished()` returns sequences that hit EOS / max-tokens

Performance

- `rayon` row-parallel GEMV (feature-gated via `oxillama-quant/parallel`;
  scalar fallback on WASM)
- Batch prefill with a configurable chunk size to cap forward-pass memory

Composable sampler chain

- Composable sampler chain: `SamplerStage` trait + `SamplerChain` builder
  with 6 built-in stages (RepetitionPenalty, TemperatureScale, TopK, TopP,
  MinP, GreedySelect), `SamplerChain::from_config()` convenience builder

Cached vocabulary

- Cached vocabulary: `TokenizerBridge::vocab_bytes_cached()` with
  `OnceLock` — compute once, reuse across generation steps

## 5. Known Gaps / Incomplete

- ~~No prefix / radix-tree KV caching — shared prompt prefixes are
  re-processed on every request.~~ ✅ Shipped: `PrefixKvCache` with
  `RadixNode` radix tree, `CachedKvState`, LRU eviction, memory tracking,
  hit/miss counters, and `KvCache::restore_from_snapshot()`.
- No flash-attention-style tiled kernel — attention materializes full
  score matrices, so memory bandwidth dominates at long context.
- Scheduler resets the KV cache between requests in practice; per-request
  KV slots are not yet wired through, so the scheduler is effectively
  serialized rather than truly continuous.
- Single `LoadedLora` per engine — no multi-LoRA hot-swap / per-request
  adapter selection.
- No CPU offload / lazy paging — entire model must fit in RAM.
- Speculative decoding has no automatic draft-model selection heuristic,
  and KV resync re-prefills the full accepted history each round (the
  delta-sync optimisation is explicitly marked TODO in `speculative.rs`).
- `PagedKvCache` has no pool / free-list across engines; each engine owns
  its pages independently.
- No public metrics surface (tokens/sec, prefill vs decode split,
  cache-hit rate); downstream crates must scrape `tracing` events.

## 6. v1.1 Roadmap

- ~~Prefix KV caching: radix-tree-indexed shared-prefix reuse across
  requests, with copy-on-write on divergence. Integrates with the
  scheduler so system prompts are paid for once.~~ ✅ Shipped:
  `PrefixKvCache`, `RadixNode`, `CachedKvState`, LRU eviction, memory
  tracking, hit/miss counters, `KvCache::restore_from_snapshot()`.
- Block-tiled flash-style attention kernel in `oxillama-quant` consumed
  through `KvCacheAccess`; reduces intermediate memory and speeds up
  long-context prefill. Scalar reference + SIMD-fast paths gated by the
  existing `simd-avx2` / `simd-avx512` / `simd-neon` features.
- True continuous batching: per-request KV slots inside `PagedKvCache`
  (no reset between requests), scheduler threads decode tokens from
  multiple active sequences in one forward pass.
- Multi-LoRA slot switching: N pre-loaded adapters, per-request selection
  without re-parsing GGUF; LRU eviction on slot pressure.
- Delta KV resync for speculative decoding: process only newly accepted
  tokens on the draft side instead of re-prefilling all history.

## 7. v2.0+ Vision

- Paged attention v2: variable block sizes, better per-sequence packing,
  first-class support inside `oxillama-arch` attention kernels, and a
  block-table view that the scheduler can hand to the forward pass
  without a contiguous copy.
- Chunked prefill with scheduler fairness: long prompts no longer starve
  short ones sharing the engine; fair-share budget per sequence per
  scheduler tick.
- State-space model runtime: Mamba-2 / Jamba require sequence-level
  primitives (selective scan, parallel associative scan) that do not fit
  the current per-token forward interface — extend `KvCacheAccess` into a
  broader `SequenceState` abstraction.
- RoPE extrapolation: YaRN, LongRoPE, and dynamic NTK scaling to extend
  context beyond training length without re-training; gated by metadata
  from the GGUF so behaviour stays reproducible.
- DRAFTER-style async speculative decoding: overlap draft and target
  forward passes across threads; requires `SpeculativeEngine` to own two
  independent executors and a shared accept/reject channel.
- Tool-invocation callbacks: OpenAI-style function-calling hooks driven by
  GBNF-constrained JSON output plus a runtime dispatcher; the server
  crate surfaces them on `/v1/chat/completions`.
- CPU / disk offload: lazy tensor paging, on-demand dequant of cold
  layers, pinned hot-layer set; pairs with the GPU crate's staging buffer
  and with `oxillama-gguf` mmap ranges.
- Automatic draft-model selection: given a target GGUF, pick the best
  compatible draft (same tokenizer, compatible vocab, smaller variant) or
  synthesise a quantised draft in-process.
- Streaming / resumable generation state: serialise a `(tokens, KV, RNG,
  sampler state, grammar state)` snapshot so a session can be paused,
  persisted, and resumed across process restarts.

*Last updated: 2026-04-15 (v0.1.0 release)*

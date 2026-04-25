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
| Version | 0.1.1 |
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
- [x] ~~No flash-attention-style tiled kernel — attention materializes full
  score matrices, so memory bandwidth dominates at long context.~~
  **A1 — FlashAttention tiled CPU kernel** ✅ **Done (2026-04-20)**
  - **Shipped:** `flash_attention_forward(q, k, v, num_heads, head_dim, softmax_scale, causal_mask) -> RuntimeResult<Vec<f32>>` in `flash_attention.rs`. Seq-major layout `[seq_len, num_heads, head_dim]`. Internally transposes to head-major, dispatches per-head via `rayon::par_chunks_mut`, uses `BQ=64 / BK=64` tile loop with online softmax. Causal mask uses absolute positions with `q_offset = seq_len_kv - seq_len_q` for decode-step asymmetry.
  - **Constants:** `FLASH_ATTN_THRESHOLD = 512` in `engine.rs` (exported from `lib.rs`).
  - **Tests (5 new):** `flash_matches_reference_causal_short` (tol 1e-5), `flash_matches_reference_causal_long` 512×1024 (tol 1e-4), `flash_matches_reference_non_causal` (tol 1e-5), `flash_determinism` (bit-equal), `flash_single_token_decode` seq_len_q=1 seq_len_kv=1024 (tol 1e-5).
- [x] ~~Scheduler resets the KV cache between requests in practice; per-request
  KV slots are not yet wired through, so the scheduler is effectively
  serialized rather than truly continuous.~~
  **A2 — Per-request KV slot continuous batching** ✅ **Done (2026-04-20)**
  - **Shipped:** `BatchedKvView` trait + `KvSlot` struct + `VecBatchedKvView` impl in `kv_cache/mod.rs`; `batched_flash_attention<V: BatchedKvView>` in new `batched_attention.rs` (~351 LoC); `slot_id: Option<usize>` field on `Sequence` in `scheduler.rs`.
  - **Note:** `ForwardPass::forward_batched` is now implemented in `oxillama-arch/src/traits.rs` (default returns `NotSupported`; LLaMA overrides with a real per-slot attention impl). `BatchedKvView` + `KvSlot` moved to `oxillama-arch/src/traits.rs` and re-exported from runtime. `kv_dim()`, `for_each_key()`, `for_each_value()` added to `KvCacheAccess` with defaults (contiguous path) and `PagedKvCache` overrides (multi-page path). `batched_attention.rs` wires through `forward_batched`.
  - **Tests (2 new):** `batched_kv_view_basic` (trait correctness), `batched_flash_decode_matches_serial` (batched output == two serial calls, tol 1e-5).
- Single `LoadedLora` per engine — no multi-LoRA hot-swap / per-request
  ✅ **Done**: `LoraStack` integrated into `InferenceEngine`; `push_lora`, `pop_lora`, `clear_loras`, `apply_lora_stack` support hot-swap without restart
  adapter selection.
- No CPU offload / lazy paging — entire model must fit in RAM.
- ~~Speculative decoding has no delta-sync optimisation; KV resync
  re-prefills the full accepted history each round.~~ ✅ Shipped:
  `SpeculativeDeltaSync` checkpoints verified KV state and restores
  on rejection, wired into `SpeculativeEngine::generate`.
- ~~`PagedKvCache` has no pool / free-list across engines.~~ ✅ Shipped:
  `KvCachePool` free-list pool with `alloc`/`free`/`page`/`page_mut`.
- ~~No public metrics surface (tokens/sec, prefill vs decode split,
  cache-hit rate).~~ ✅ Shipped: `EngineMetrics` + `MetricsSnapshot`
  wired into `InferenceEngine`; `engine.metrics()` exposes live counters.

## 6. v1.1 Roadmap

- ~~Prefix KV caching: radix-tree-indexed shared-prefix reuse across
  requests, with copy-on-write on divergence. Integrates with the
  scheduler so system prompts are paid for once.~~ ✅ Shipped:
  `PrefixKvCache`, `RadixNode`, `CachedKvState`, LRU eviction, memory
  tracking, hit/miss counters, `KvCache::restore_from_snapshot()`.
- ~~Delta KV resync for speculative decoding.~~ ✅ Shipped:
  `SpeculativeDeltaSync` with `checkpoint`/`restore`, wired into
  `SpeculativeEngine`; draft only re-runs corrected token on rejection.
- ~~Public metrics surface.~~ ✅ Shipped: `EngineMetrics`/`MetricsSnapshot`
  in `src/metrics.rs`; `InferenceEngine::metrics()` / `metrics_snapshot()`.
- ~~KV cache pool / free-list.~~ ✅ Shipped: `KvCachePool` in
  `src/kv_pool.rs` with `alloc`/`free`/`page`/`page_mut`.

## 7. v2.0+ Vision

- Paged attention v2: variable block sizes, better per-sequence packing,
  first-class support inside `oxillama-arch` attention kernels, and a
  block-table view that the scheduler can hand to the forward pass
  without a contiguous copy.
- [x] Chunked-prefill scheduler fairness ✅ **Done (2026-04-20)**
  - **Goal:** A single 32K-token prefill no longer blocks short decode requests. Prefill split into `PREFILL_CHUNK = 512` token chunks; scheduler interleaves decode steps after each chunk.
  - **Shipped:**
    - `Sequence` extended with `prefill_progress: usize`, `prefill_total: usize`, `last_emit_time: Instant` in `scheduler.rs`.
    - `PREFILL_CHUNK = 512` and `MAX_DECODE_WAIT_MS = 100` constants exported from `scheduler.rs` / `lib.rs`.
    - `advance_prefill(n)`, `prefill_fraction()`, `decode_wait_exceeded()` methods on `Sequence`.
    - `append_token` refreshes `last_emit_time`.
    - `forward_prefill(tokens, pos_start)` and `forward_decode(token, pos)` added to `InferenceEngine`.
    - `KvCache::truncate(n)` added for speculative decoding rollback.
  - **Tests (8 new in scheduler, 5 new in engine):**
    - `chunked_prefill_reports_progress`, `chunked_prefill_kv_matches_singleshot`, `decode_wait_exceeded_false_initially`, `advance_prefill_is_independent_of_prompt_pos`, `append_token_refreshes_last_emit_time`, `prefill_fraction_one_for_empty_prompt`, `prefill_fairness_constants` (scheduler).
    - `test_forward_prefill_errors_when_not_loaded`, `test_forward_prefill_empty_slice_errors`, `test_forward_decode_errors_when_not_loaded`, `test_forward_prefill_returns_logits_after_load`, `test_forward_decode_returns_logits_after_load`, `chunked_prefill_kv_matches_singleshot` (engine, feature-gated).
- [x] SSM runtime bridge — polymorphic sequence-state pool ✅ **Done (2026-04-20)**
  - **Goal:** Mamba-2 (and any future SSM) can use a pooled state slot; the engine stays arch-agnostic.
  - **Shipped:** New `crates/oxillama-runtime/src/sequence_pool.rs` (~420 LoC):
    - `SequenceSlot { state: Box<dyn SequenceState>, position: usize, request_id: u64 }` with `step()`, `reset()`.
    - `SsmStatePool` free-list pool with `alloc(request_id)`, `release(idx)`, `slot(idx)`, `slot_mut(idx)`, `capacity()`, `free_count()`, `used_count()`.
    - `SequencePool` enum: `KvBased(KvCachePool)` / `Ssm(SsmStatePool)` with `alloc_kv`, `free_kv`, `alloc_ssm`, `release_ssm`, `ssm_slot`, `ssm_slot_mut`, `is_kv_based`, `is_ssm`.
    - `PoolError` (`Exhausted`, `InvalidSlot`) via `thiserror`.
    - Exported from `lib.rs`: `PoolError`, `PoolResult`, `SequencePool`, `SequenceSlot`, `SsmStatePool`.
  - **Tests (11 new):** `sequence_slot_position_advances`, `sequence_slot_reset_clears_position`, `sequence_pool_allocate_release`, `ssm_pool_exhaustion_returns_error`, `ssm_pool_double_release_errors`, `ssm_pool_release_resets_state`, `sequence_pool_kv_based_alloc_free`, `sequence_pool_kv_rejects_ssm_ops`, `sequence_pool_ssm_alloc_release`, `sequence_pool_ssm_rejects_kv_ops`, `mixed_pool_isolation`, `ssm_pool_out_of_range_slot_errors`, `slot_reset_on_eos_for_ssm`.
  - **Note:** Full arch integration (`ForwardPass::allocate_sequence_state`) requires arch subagent; the pool primitives are wired and tested.
- ~~RoPE extrapolation: YaRN, LongRoPE, and dynamic NTK scaling to extend
  context beyond training length without re-training; gated by metadata
  from the GGUF so behaviour stays reproducible.~~ **[DONE 2026-04-16]**
  YaRN + linear scaling implemented in `oxillama-arch` (`RopeScalingType`,
  `ModelConfig::rope_scaling_type/factor`, GGUF metadata read).
- [x] Drafter-async speculative decoding ✅ **Done (2026-04-20)**
  - **Goal:** Draft model runs ahead of target model in a separate tokio task; target verifies N candidates; on divergence, rollback to divergence point.
  - **Shipped:** New `crates/oxillama-runtime/src/speculative_async.rs` (~760 LoC):
    - `Rewindable` trait with `rewind(n)` / `current_length()`.
    - `RewindError` enum: `NotSupported`, `PositionBeyondEnd`, `Runtime`.
    - `SpecStats` with `accepted`, `rejected`, `bonus_tokens`, `total_elapsed`, `n1_fallbacks`, `acceptance_rate()`, `total_output_tokens()`.
    - `AsyncSpecConfig` with `spec_k`, `draft_sampler`, `target_sampler`, `force_n1`, `max_tokens`.
    - `SpeculativeDecoder::new` / `new_n1` / `generate(prompt, on_token)` async method.
    - `CancellationToken` shared between draft task and verification loop.
    - SSM fallback: `RewindError::NotSupported` increments `n1_fallbacks` and continues N=1.
    - `KvCache::truncate(n)` added to support `Rewindable::rewind`.
    - `InferenceEngine::truncate(n)` / `kv_cache_seq_len()` helper methods added to engine.
    - `tokio` + `tokio-util` added to workspace and runtime `Cargo.toml`.
  - **Exported from `lib.rs`:** `AsyncSpecConfig`, `RewindError`, `Rewindable`, `SpecStats`, `SpeculativeDecoder`.
  - **Tests (15 new):** `spec_stats_acceptance_rate_empty/all_accepted/half`, `spec_stats_total_output_tokens`, `softmax_prob_*` (3), `accept_draft_token_*` (2), `async_spec_config_defaults`, `rewind_error_*_display` (2), `spec_decode_construction_with_unloaded_engines`, `spec_decode_correctness_stub`, `spec_decode_divergence_rollback`, `spec_decode_ssm_falls_back`, `cancellation_token_child_relationship`, `spec_decode_loaded_engines_produce_output` (feature-gated).
- [x] Tool-invocation runtime callbacks ✅ **Done (2026-04-20)**
  - **Goal:** Incremental tool-call detection as tokens arrive, JSON parsing, dispatcher invocation.
  - **Shipped:** `crates/oxillama-runtime/src/tool_dispatch.rs` (~530 LoC):
    - `ToolDispatcher` trait (`Send + Sync`), `ToolResult { Ok(Value), Err(String) }`.
    - `ToolCallGrammar` enum: `Llama3`, `Qwen`, `Mistral`, `Custom { open, close }` with `open_delimiter()` / `close_delimiter()` accessors.
    - `ToolCall { name: String, args: Value }`.
    - `ToolCallDetector` state machine (`Idle` → `Capturing`) with `new(grammar)`, `feed(token_text) -> Option<ToolCall>`, `reset()`.
    - `NoOpDispatcher` + `no_op_dispatcher() -> Arc<dyn ToolDispatcher>` helper.
    - OpenAI compat: `"arguments"` key accepted as alias for `"args"`.
    - Exported from `lib.rs`: `no_op_dispatcher`, `NoOpDispatcher`, `ToolCall`, `ToolCallDetector`, `ToolCallGrammar`, `ToolDispatcher`, `ToolResult`.
  - **Tests (13):** All from original spec plus grammar-delimiter accessors and reset test.
- [x] CPU / disk offload: lazy tensor paging, pinned hot-layer set ✅ **Done (2026-04-24)**
  - **Shipped:** `src/offload/{mod,policy,pager,pressure}.rs` (~660 LoC of pager logic):
    - `OffloadPolicy` enum (None / Budget / PinnedHotSet) with `#[derive(Default)]` via `#[default]` on `None`.
    - `LayerPager` — LRU weight pager with `RwLock`-protected resident map, `Mutex`-protected LRU queue, `AtomicU64` byte counter, pinned tensor set. `acquire()` has fast path (read-lock only) and slow path (evict → load from source → write-lock).
    - `PagerSource` trait (`read_bytes_at`, `total_size_bytes`) for `dyn` dispatch.
    - `FilePagerSource` (seek+read, always available) + `MmapPagerSource` (behind `mmap` feature).
    - `MemoryPressureProbe` with Linux `/proc` parsing; macOS/other returns `None`.
    - `OffloadPolicy::None` (default) wires to `layer_pager: None` in `InferenceEngine` — existing inference path unchanged.
    - `EngineConfig::with_offload(policy)` builder method; `InferenceEngine::layer_pager()` / `set_layer_pager()` inspection hooks.
    - 3 new `RuntimeError` variants: `OffloadEof`, `TensorNotFound`, `LockPoisoned`.
  - **Tests (15 new in `offload/pager.rs`, 6 in `policy.rs`, 6 in `pressure.rs`):** budget eviction, pinned survival, correct bytes, unknown tensor error, double acquire, `FilePagerSource` correctness + EOF error, resident count tracking, `is_pinned`, strict budget, `TensorId` display.
  - **Deferred:** linear-layer per-GEMM integration lives in `oxillama-arch/src/common/linear.rs` (out of R1 scope); the `layer_pager` field on `InferenceEngine` is the integration hook for that follow-up.
  - **Feature:** `offload = []` (always-on by default); `mmap` feature now also gates `dep:memmap2` for `MmapPagerSource`.
- Automatic draft-model selection: given a target GGUF, pick the best
  compatible draft (same tokenizer, compatible vocab, smaller variant) or
  synthesise a quantised draft in-process.
- [x] Runtime snapshot/resume generation state via oxicode (2-crate slice: runtime + arch) ✅ **Done (2026-04-24)**
  - **Goal:** Serialise a live inference session into a portable opaque blob and deserialise deterministically. Captures: tokens_generated, kv_cache_state, sampler_rng_state, sampler_config, grammar_state, arch_id, model_fingerprint, context_position. Enables crash recovery, chat session persistence across restarts, fleet migration of in-progress responses.
  - **Scope note:** Primarily `oxillama-runtime`, plus additive default-method additions to `SequenceState` trait in `oxillama-arch` (`snapshot()`/`restore()` — no existing signatures change, all 18 shipped archs remain binary-compatible).
  - **Design:**
    - `src/snapshot.rs` (~600 LoC): `EngineSnapshot { version: u32, arch_id, model_fingerprint: ModelFingerprint, tokens, kv_state, sampler_state, grammar_state, engine_config }`. Magic header `b"OXISNAP1"`. Serialised via oxicode (never bincode). `ModelFingerprint { file_size, mtime_secs, head_hash, tail_hash, probe_size=8MB }` — bounded O(constant) probe matching G1's design.
    - KV snapshot: `KvSnapshotPayload { layout: KvLayout, per_layer: Vec<LayerKvBlob> }` where `KvLayout` handles Contiguous, Paged, and SSM variants. `SequenceState` trait gets new additive default methods `snapshot()`/`restore()`.
    - Sampler snapshot: expose `Xorshift64::state()`/`from_state()`, preserve mirostat-v2 `mu`.
    - Grammar snapshot: serialise `{ grammar_source: String, current_state_id: u32 }`, reparse on resume.
    - API: `InferenceEngine::snapshot(&self) -> RuntimeResult<Vec<u8>>`, `InferenceEngine::resume(snapshot: &[u8], model_path: &Path) -> RuntimeResult<Self>`.
    - New errors: `RuntimeError::SnapshotIncompatible { detail }`, `RuntimeError::ModelFingerprintMismatch { expected, found, detail }`.
  - **Files:** `src/snapshot.rs`, `src/engine.rs` (+80 LoC), `src/error.rs` (2 new variants), `src/kv_cache/mod.rs` (+100 LoC), `src/kv_cache/paged.rs`, `src/sampling/mod.rs` (Xorshift64 accessors), `src/sampling/grammar/machine.rs` (GrammarState::from_state_id), `crates/oxillama-arch/src/common/sequence_state.rs` (additive trait methods + Mamba2/Jamba overrides), `Cargo.toml` (oxicode dep), `tests/snapshot.rs`.
  - **Tests:** `snapshot_roundtrip_small`, `snapshot_rejects_wrong_model_fingerprint`, `snapshot_rejects_incompatible_version`, `snapshot_preserves_mirostat_mu`, `snapshot_preserves_grammar_state`, `snapshot_ssm_roundtrip`, `snapshot_paged_kv_roundtrip`, `snapshot_cross_process_determinism`.

*Last updated: 2026-04-24 (v0.1.2)*

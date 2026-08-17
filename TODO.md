# OxiLLaMa Development Roadmap

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [x] ~~`oxillama-runtime`: `crates/oxillama-runtime/src/speculative.rs:28` — implement delta-KV-cache sync so draft model processes only newly accepted tokens instead of full history reprefill on each round~~ ✅ **Done (2026-06-13)**
  - **Shipped:** `SpeculativeEngine::generate` now keeps the draft KV cache aligned with the committed context and resyncs via an `O(1)` `InferenceEngine::truncate` + a single `forward_one` for the newly committed token, replacing the per-round full re-prefill. The duplicate "re-forward the last committed token" pattern was eliminated by carrying `draft_next_logits` / `target_next_logits` across rounds (this also corrected the rejection residual to use the proper per-position draft distribution). `SpeculativeDeltaSync` was repurposed into a truncation-based rollback manager that records reuse statistics (`verified_len`, `tokens_reused`, `rounds`).
  - **Tests:** gold test `test_delta_sync_matches_full_reprefill_kv_state` asserts the delta-synced KV state is **byte-identical** to a full re-prefill; plus rollback/record/reset unit tests and a cache-alignment end-to-end test. All 438 `oxillama-runtime` tests pass; `cargo clippy --workspace --all-targets -- -D warnings` clean.

## Known gaps (added 2026-08-03)

- [x] ~~`oxillama-arch`: tied-embedding LM-head fallback missing in 11 of 16 GGUF loaders.~~ ✅ **Done (2026-08-05)** — `common::loader::load_lm_head(model, "output.weight", "token_embd.weight")` is the shared fallback; `llama` (and therefore `llava`/`llava_next`), `mistral`, `phi`, `phi_moe`, `command_r`, `starcoder` and the newly-written `falcon`/`gptneox`/`stablelm`/`olmo2`/`minicpm` loaders all route through it. `llama`'s `tensor_names()` also marks `output.weight` `required: false` so a validator cannot reject a tied Llama-3.2-1B/3B.
- [x] ~~`oxillama-runtime` / `oxillama-cli`: the tokenizer is only ever built from an HF `tokenizer.json`, so a self-contained single-file GGUF cannot be run at all.~~ ✅ **Done (2026-08-05)** — `crates/oxillama-runtime/src/gguf_vocab/` is a native pure-Rust port of llama.cpp's tokenizers (SPM bigram-merge, byte-level BPE, WordPiece) reading `tokenizer.ggml.{model,pre,tokens,scores,token_type,merges,*_token_id,add_bos_token}`, with the GGUF vocabulary preferred over a sidecar. Conformance: 12 vocabularies × 46 cases exact against llama.cpp's own `.inp`/`.out` fixtures (`tests/gguf_tokenizer.rs`); `bert-bge` is 44/46, the two failures being accent stripping that needs Unicode NFD. `oxillama tokenize`/`detokenize` use it too. Note SPM in GGUF is **not** Unigram/Viterbi — llama.cpp runs a greedy highest-score bigram merge, because LLaMA's SentencePiece model was trained in BPE mode.

## Decode speed (2026-08-03): 0.346 → 6.08 tok/s measured on Apple M3, Qwen3-4B Q4_K_M

Shipped: (a) `simd-neon`/`simd-avx2` are now default features of `oxillama-quant` AND named explicitly on `oxillama-arch`'s dependency edge (the workspace table pins `oxillama-quant = { default-features = false }`, so the arch edge is what actually reaches consumers); `oxillama-arch`/`oxillama-runtime` gained `simd-*` passthrough features they never had. (b) All 122 serial per-row GEMV loops (reference/neon/avx2/avx512) now route through `oxillama_quant::parallel::for_each_row`, which splits output rows over a dedicated rayon pool (never the global pool); rows are never split internally, so results are bit-identical to serial at any thread count (`tests/gemv_parity.rs`, 9 tests). (c) `EngineConfig::num_threads` is finally consumed; `0 = auto` (`available_parallelism()`) is the new default — a pinned 4 costs ~20% on 4P+4E. Fixed row bands were tried and measured WORSE than rayon adaptive splitting (5.08/4.34 vs 6.37 tok/s); numbers are in a comment in `parallel.rs` so nobody re-tries it.

Remaining headroom (all compute-bound; bandwidth ceiling ~22 tok/s for this model on M3, decode is at ~15 GB/s of ~100):

- [x] ~~`oxillama-arch`: route the forward path through the fused Q8-activation kernels~~ ✅ **Done (2026-08-04, see next section)** — incl. the real-NEON rewrite of `fused_q6_k_q8_0_row_neon` (1.89x vs the Q6_K gemv it used to lose to by 7.5x).
- [x] ~~`oxillama-quant`: no SDOT anywhere~~ ✅ **Done (2026-08-04)** — via stable `asm!("sdot …")` in `simd/neon/int_dot.rs`, because `vdotq_s32` is still unstable on rustc 1.95 (`stdarch_neon_dotprod`, rust-lang/rust#117224) even though `dotprod` is in the default target features.
- [x] ~~`oxillama-runtime`/`oxillama-arch`: prefill is not batched~~ ✅ **Done (2026-08-04)** — 16-token tiles, 9.34 → 28.77 prompt tok/s (3.08x) on a 921-token prompt.
- [x] ~~`oxillama-arch` load path: `data.to_vec()` mmap defeat + 1.556 GB f32 token_embd~~ ✅ **Done (2026-08-04)** — RSS 6.736 → 2.674 GB, peak footprint 4.204 GB → 148 MB, weight-load 0.60 s → 0.014 s.
- [ ] `oxillama-quant`: `simd/float_gemm.rs` (F32/F16/BF16) is the one kernel family whose row loop was NOT converted to `for_each_row` (irrelevant for K-quant GGUFs — all F32 tensors in real models are 1-D norms — but convert it for consistency).
- [x] ~~`oxillama-arch` per-token allocs~~ ✅ **Done (2026-08-04)** — proj_out reuse, logits `mem::take`, per-layer `Arc<dyn QuantKernel>` resolved at load; measured ≈ noise (decode is SDOT/bandwidth-bound now), kept as cleanup. Deliberately left: `embed()`'s 10 KB `buf_hidden.clone()` (read by the batched-prefill `copy_from_slice`) and the 1 remaining per-token dispatch in `qwen3/embedding.rs`.

## Decode/prefill/memory (2026-08-04): decode 6.34 → 12.98 tok/s (2.05x, ~37x cumulative), prefill 9.3 → 28.8 prompt tok/s (3.08x), RSS 6.74 → 2.67 GB

Shipped (on top of 2026-08-03; all medians-of-5 on Apple M3, Qwen3-4B Q4_K_M, interleaved A/B under load so contention cancels):

- **Fused Q8 activations + SDOT** (`oxillama-quant`, `oxillama-arch/qwen3`): activations quantized to Q8_0 once per matmul input (`quantize_activations_q8_0_into`, ~3 µs at K=2560, <1% of the matmul it feeds) and shared across q/k/v and gate/up; per-kernel opt-in gate `QuantKernel::q8_fused_acts_blocks` (only kernels measured faster than their own gemv return `Some`); `fused_q6_k_q8_0_row_neon` rewritten into real NEON (one Q8_0 block = exactly two Q6_K 16-weight sub-blocks → integer-exact per-block reduction). SDOT reached through `asm!` with a `vmull_s8`/`vpadalq_s16` fallback that computes the identical integer total. Kernel-level fused-vs-gemv after SDOT: ffn_gate/up 3.2x, attn_q 5.3x, ffn_down 5.6x, LM head 2.9x. Stage medians: 6.3427 → 8.7293 (Q4_K fused) → 11.6102 (+Q6_K) → 12.9807 (+SDOT). Q8_0 tensors stay on gemv (none exist in this checkpoint); non-qwen3 archs stay on the f32-activation gemv until each is benchmarked (adoption recipe in the `forward_q8_fused` rustdoc).
- **Batched prefill** (`qwen3/batch.rs`): 16-token tiles through new feature-major `QuantKernel::matmul_q8_fused` batched kernels that are **bit-identical to sequential** (same per-(row,token) K accumulation via shared helpers; m∈{33,40} tests cross the MAX_FUSED_BATCH=32 chunk seam). Intra-tile K/V staged and replayed through the ordinary `store_kv`/`advance` order, so `KvCacheAccess` and every cache impl are untouched. Batching matmuls alone gained ZERO wall-clock (ablation: −35% CPU, ±0 s) because single-threaded O(position) attention was the bottleneck — attention now threads over the same dedicated pool via new `parallel::for_each_chunk_init` (pool/no-splitting/bit-identity invariants unchanged). Tile sweep 1/8/16/32 = 9.3/26.9/28.8/28.7 prompt tok/s, recorded in the PREFILL_TILE comment. Decode never enters the tile path; LoRA and non-fused kernels fall back per-token.
- **mmap-preserving load** (`oxillama-gguf::bytes`, 10 arch loaders): `SharedBytes` (Arc-backed, pointer-cached byte view; `unsafe trait ByteOwner` with documented stable-address/no-mutation contract) replaces `Vec<u8>` in `QuantTensor::data` (pre-1.0 API break; `QuantTensor::new(Vec…)` kept for the 240+ existing call sites). token_embd stays quantized, one row dequantized per token (`qwen3/embedding.rs`; bit-identical-to-bulk-dequant test); tied LM head shares the same payload so tying stays free. RSS 6.736 → 2.674 GB (~2.38 GB of which is the clean evictable file mapping), peak footprint 4.204 GB → 148 MB, page reclaims 413k → 166k, weight-load 0.604 s → 0.014 s; mmap first-touch faults cost nothing measurable on decode.
- **Proof obligations**: generation byte-identical at seed 12345 before/after prefill+memory+allocs steps (5-token AND 921-token prompts) and across 1/4/8 threads (md5-equal); fused path tolerance 2.0e-2 derived from Q8_0 quantization statistics in `gemv_parity.rs` (23 tests), and an independent f64 scalar oracle (written from the ggml spec, no oxillama code reused) measured max rel err ≤ 2.1e-7 on real shapes. Workspace 2444/2444 nextest, clippy `-D warnings` clean, wasm32 arch build still rayon-free, net unsafe lines −33.

Follow-ups discovered:

- [ ] **PRE-EXISTING bug (found in review, not introduced, not yet fixed):** `PagedKvCache::get_keys` bounds its slice by `self.seq_len`, but the per-token qwen3 attention reads position `seq_len` *before* `advance()` — the SEQUENTIAL path would panic (OOB) on a paged cache. The new batched path range-checks and returns `ArchError::ForwardPassError` instead. Needs its own fix + test.
- [ ] Latent silent-zero: `quantize_activations_q8_0_batch_into` falls back to `&[]` on an undersized `rows` slice → all-zero activations instead of an error (kernels return `QuantError::BufferTooSmall` in the same situation). Caller sizing in `qwen3/batch.rs` is correct today; make it error anyway.
- [ ] Wire fused Q8 + batched prefill into mistral/gemma/… (llama shipped in v0.1.4: `llama/batch.rs` + `forward_q8_fused` call sites through `llama/model.rs`) — mechanical per the rustdoc recipe, but each remaining arch needs a benchmark model before flipping a numerics-changing default.
- [ ] NEON dot/AXPY for the now-threaded but still-scalar attention inner loops — the next prefill win.
- [ ] Threading-heuristic nit: `matmul_q8_fused`'s >32-token chunk loop passes full `m` (not `chunk`) into the `should_parallelize` work estimate — correctness unaffected, sizing slightly inflated.
- [ ] Remaining `to_vec()` on load paths, deliberately left: `GgufModel::from_bytes` (borrowed-slice API must own) and the `oxillama-wasm` streaming loader — neither is the mmap path.

## Project Overview

**OxiLLaMa** — Pure Rust LLM inference engine, the sovereign alternative to llama.cpp.
A complete reimplementation providing GGUF model loading, multi-format quantized inference,
and an OpenAI-compatible API server without any C/C++/Fortran code.

---

## v0.1.4 Shipped (2026-08-17)

v0.1.4 theme: **Correctness and security hardening against real llama.cpp, GPU offload backend, GGUF-embedded tokenizer, and K-quant encoders.**

**Quantization correctness** (`oxillama-quant`, `oxillama-gpu`, `oxillama-arch`):
Seven quantization formats (`Q4_0`, `Q4_1`, `IQ4_NL`, `IQ4_XS`, `Q5_K`, `TQ1_0`, `TQ2_0`) decoded to the wrong weight layout across the scalar reference and every SIMD tier — invisible because SIMD tiers were parity-tested against the (also wrong) scalar reference. Fixed across scalar/AVX2/AVX-512/NEON and the 9 corresponding GPU kernels, now golden-tested against values from compiling and running upstream llama.cpp's own C. GPU `Q4_K`/`Q6_K` used a flat index instead of upstream's group-based one. The LLaMA-family RoPE convention (llama/mistral/mixtral/command-r) used the NeoX half-split instead of llama.cpp's actual interleaved-pairs convention. Multiple architecture loader bugs fixed (StarCoder, Command-R, Gemma, Phi, Gemma-2, BLOOM, ALiBi, DBRX/Grok, OLMo2/MiniCPM, DeepSeek-V3) — see CHANGELOG.md for the full per-architecture list.

**Security fixes** (`oxillama-gguf`, `oxillama-server`):
GGUF parser integer-overflow crashes (a 40-byte crafted file defeated bounds checking in a release build) and unbounded metadata-array recursion. `/admin/*` was unauthenticated in both router builders — the guard existed but was only ever exercised inside test code. Path traversal in all three server-side disk stores via unescaped request IDs. `serve` called the wrong app-builder function, so authentication, rate limiting, CORS and graceful shutdown were all dead in the deployed server — now fixed, plus request cancellation on disconnect and load shedding.

**GGUF-embedded tokenizer** (`oxillama-runtime/src/gguf_vocab/`):
Native pure-Rust SPM/BPE/WordPiece tokenizers read straight from GGUF metadata — a stock HuggingFace GGUF now runs with no `tokenizer.json` sidecar. Exact against 12 vocabularies × 46 of llama.cpp's own conformance fixtures.

**K-quant encoders** (`oxillama-quant/src/kquant/`):
`oxillama quantize --target Q4_K_M` (and 12 other formats) works for the first time; previously only `Q4_0`/`Q8_0` could be encoded. Byte-identical to compiled upstream C on 22 golden inputs.

**GPU offload backend** (`oxillama-runtime`, `oxillama-gpu`, `oxillama-server`):
`oxillama-runtime` now depends on `oxillama-gpu` behind the `gpu` feature (`GpuPolicy`/`EngineConfig::gpu`/`InferenceEngine::gpu_status()`), offloading Q4_0 decode-time weight matrices to a device-resident kernel. `/admin/health` reports backend status. Not yet wired to a `--gpu` CLI flag or Python kwarg — see "GPU CLI/Python wiring" below.

**Whole-model logit parity vs real llama.cpp** (first time, not self-comparison):
32/32 top-1 greedy-token agreement with `--kv-dtype f16`, within 1.2–1.4× llama.cpp's own cross-build noise floor, over 2 models × 2 prompts × 8 teacher-forced steps.

**Speculative decoding delta-KV-cache sync** (`oxillama-runtime`):
`SpeculativeEngine::generate` now keeps the draft KV cache aligned via `O(1)` `InferenceEngine::truncate` + single `forward_one` per committed token, replacing per-round full re-prefill.

**WebSocket live inference** (`oxillama-server`):
`ws_handler` + `handle_socket` stream real inference tokens over WebSocket instead of a hardcoded stub token list.

**Zero-copy mmap weight loading, threaded GEMV, NEON SDOT** (`oxillama-gguf`, `oxillama-quant`):
Peak load footprint 4.2 GB → 148 MB, weight-load time 0.6 s → 14 ms on Qwen3-4B Q4_K_M.

**CLI `--model-id` override, GGUF magic-byte regression test, dependency modernization**:
`--model-id` flag on `serve`/`chat`. Explicit regression test for `GGUF_MAGIC` byte-order correctness. SciRS2 → 0.6.5, OxiFFT → 0.4.2, OxiCode → 0.2.6, OxiBLAS → 0.2.2, pyo3 → 0.29.1 (fixes RUSTSEC-2026-0176 + RUSTSEC-2026-0177).

**3,751 tests passing, 0 failed** (`--all-features`; 3,631 on default features), **0 compiler warnings**, `cargo deny check bans` clean, `cargo audit` 0 vulnerabilities. See [CHANGELOG.md](CHANGELOG.md) for full details.

---

## v0.1.3 Shipped (2026-05-05)

v0.1.3 theme: **End-to-end feature completeness** across all crates: new architectures (BLOOM, Phi-3.5-MoE, Mixtral, StableLM, GPT-NeoX, LLaVA-1.6, Qwen2-VL), advanced sampler suite (DRY/XTC/TypicalP/TopA/Eta), `/v1/responses` + per-API-key rate limiting, AVX-512 IQ kernels + fused legacy matvec, GPU sampling kernels, speculative decoding bench + Python torch interop, server productionization (prefix-KV cache, multi-LoRA, Assistants API, Files store, Batch API, CLI tools, power benchmarks, DLPack interop).

**BLOOM + Phi-3.5-MoE architectures** (`oxillama-arch`):
`AlibiBias` primitive (`common/alibi.rs`) with slope formula matching transformers/modeling_bloom.py.
`BloomArchitecture` (`"bloom"`) with ALiBi positional bias (no RoPE), pre-LayerNorm, GELU FFN, MHA; bias terms on all projections. `PhiMoeArchitecture` (`"phimoe"`) reusing Phi-3 merged QKV + partial RoPE + `SparseTopKMoe` router (16 experts, top-2). Test fixtures `build_minimal_bloom_gguf` + `build_minimal_phi_moe_gguf`. Arch count: 25 → 27.

**Advanced sampler suite + embedding pooling** (`oxillama-runtime`):
`DryStage` (n-gram DRY penalty, exponential growth), `XtcStage` (exclude top-choices with probability), `TypicalPStage` (locally-typical sampling via Shannon entropy), `TopAStage` (adaptive `a × max_prob²` threshold), `EtaStage` (entropy-scaled cutoff combining typical + epsilon). 9 new `SamplerConfig` fields, all `#[serde(default)]`, byte-identical defaults. `PoolingMode { Last, Mean, Max, Cls }` + `pool_hidden_states()` + `embed_with()` / `embed_batch_with()` API.

**`/v1/responses` + per-API-key rate limiting** (`oxillama-server`):
`ResponseStore` (in-memory `Arc<RwLock<HashMap>>`) with atomic create/get/update/list. `POST /v1/responses` (non-streaming + SSE `response.created / output_text.delta / completed / [DONE]`), `GET /v1/responses`, `GET /v1/responses/:id`. `previous_response_id` chains prior response into context. `PerKeyRateLimiter` with lazy per-key `TokenBucket` insertion, override map, `per_key_rate_limit_middleware`.

**AVX-512 IQ kernels + fused legacy matvec** (`oxillama-quant`):
`Iq2XxsAvx512`, `Iq2XsAvx512`, `Iq3SAvx512`, `Iq4XsAvx512` — `_mm512_permutexvar_epi8` grid lookup (AVX-512BW); 2× per-iter throughput over AVX2; runtime-guarded, auto-skip on non-AVX-512. `matvec_q8_fused` override for Q5_0/Q5_1/Q8_1 on both AVX2 and NEON paths; scalar parity oracles in reference tier.

**GPU sampling kernels** (`oxillama-gpu`):
`sampling.wgsl` — `softmax_logits` (256-thread shared-memory reduction, temperature scaling, temp=0 argmax path), `topk_partition` (workgroup cooperative top-k ≤ 256), `sample_categorical` (LCG RNG + CDF walk). `SamplingKernel` struct with `softmax()`, `top_k()`, `sample()` Rust wrappers; graceful `NoAdapter` fallback.

**Speculative decoding bench + Python torch interop** (`oxillama-bench` + `oxillama-py`):
`SpeculativeBenchConfig`, `SpeculativePoint`, `SpeculativeBenchTable` + `run_acceptance_sweep()` with deterministic acceptance simulation; `summary_table()` / `speedup_grid()` Markdown output; Criterion bench `benches/speculative.rs`. `torch_helper.py` pure-Python bridge: `Engine.logits_torch()` / `embeddings_torch()` via lazy `torch.from_dlpack(capsule)` (no Rust torch dependency).

2,235 tests, 0 warnings. See [CHANGELOG.md](CHANGELOG.md) for full details.

---

## v0.1.1 Shipped (2026-04-24)

v0.1.1 ships on top of v0.1.0's foundation: FlashAttention tiled CPU kernel
(BQ=BK=64, online softmax, rayon per-head), true continuous batching
(per-request KV slots, `BatchedKvView` trait), fused dequant+GEMM for Q4_0
and Q4_K (AVX2 + NEON, no scratch buffer), oxiblas float GEMM fallback for
F16/BF16/F32 tensors, tiled GEMM WGSL shader (TILE_M/N=32, TILE_K=16, shared
memory cooperative), fused attention WGSL kernel (single-dispatch QK+softmax+AV),
IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S GPU GEMV kernels (+4 GPU kernels), DBRX
(16-expert MoE, top-4), Grok-1 (8-expert MoE, top-2), DeepSeek-V3 sigmoid MoE
scoring, Mamba-2 (selective scan, learned Δ), Jamba (hybrid SSM+attention),
OLMo2, Yi, Granite, MiniCPM, InternLM3, and the `SequenceState` trait for SSM
abstraction. GGUF loader hardening: partial-download resume (`GgufModel::resume`
+ `.oxiresume` sidecar), sharded multi-file loading (`ShardedGgufModel`),
quantize-on-the-fly pass. Runtime snapshot/resume (`EngineSnapshot`, oxicode
serialization). Facade examples (load_and_generate, lora_apply, speculative) +
`RECIPES.md` cookbook (8 recipes). 1,662 tests, 0 warnings, 87%+ coverage.
Detailed feature list: see [CHANGELOG.md](CHANGELOG.md).
This TODO.md is forward-looking: the per-crate TODO files under `crates/*/TODO.md`
carry shipped + gap + v1.1 / v2.0 detail.

## v0.1.0 Shipped (2026-04-15)

v0.1.0 ships a feature-complete Pure Rust LLM inference engine: GGUF v3 parser,
25 quantization types with 3-tier SIMD dispatch (AVX-512 / AVX2 / NEON / scalar),
8 model architectures (LLaMA, Qwen3, Mistral, Gemma, Phi, StarCoder, Command-R,
Mixtral-MoE, LLaVA), full runtime with paged KV cache + 6 samplers + GBNF grammar
+ speculative decoding + LoRA, OpenAI-compatible server with SSE streaming and
continuous-batching scaffolding, WASM full inference, Python bindings (PyO3),
optional wgpu GPU backend (Q4_0 / Q8_0 GEMV), criterion benchmarks across every
quant kernel, and 3 cargo-fuzz targets on the GGUF parser. 1,205 tests, 0 warnings,
87%+ region / function / line coverage. Detailed feature list: see
[CHANGELOG.md](CHANGELOG.md).

---

## Codebase Metrics

| Metric | Value |
|--------|-------|
| Total Lines | ~185,000 Rust (src/) / ~235,000 total (incl. tests/benches/examples) |
| Source Files | 594 Rust files / 680 total |
| Crates | 11 |
| Test Count | 3,751 passing, 0 failed (`--all-features`); 3,631 passing, 0 failed (default features) |
| Warnings | 0 (`cargo clippy --workspace --all-targets [--all-features] -- -D warnings` clean) |
| Coverage | 87.09% region / 87.23% function / 85.42% line |
| Last Updated | 2026-08-17 |

---

## Implementation Status by Crate

| Crate | Status | Completion |
|-------|:------:|:----------:|
| oxillama-gguf | Working | 93% |
| oxillama-quant | Working | 100% |
| oxillama-arch | Working | 99% |
| oxillama-runtime | Working | 94% |
| oxillama-server | Working | 99% |
| oxillama-bench | Working | 88% |
| oxillama-py | Scaffold | 84% |
| oxillama-wasm | Working | 97% |
| oxillama-gpu | Working | 93% |
| oxillama (meta) | Working | 100% |
| oxillama-cli | Working | 100% |

---

## Per-Crate TODO.md Index

Every crate carries its own forward-looking TODO with a shared 7-section template
(Overview, Status Snapshot, Module Map, Shipped in v0.1.0, Known Gaps, v1.1
Roadmap, v2.0+ Vision). Dive into the leaf that matches your area of interest.

| Crate | Link | Focus |
|---|---|---|
| oxillama (meta) | [crates/oxillama/TODO.md](crates/oxillama/TODO.md) | Examples, mdBook guide, cookbook |
| oxillama-gguf | [crates/oxillama-gguf/TODO.md](crates/oxillama-gguf/TODO.md) | v1/v2 legacy fallback, streaming parser, GGUF writer |
| oxillama-quant | [crates/oxillama-quant/TODO.md](crates/oxillama-quant/TODO.md) | SIMD coverage breadth, ~~fused dequant+GEMM~~ ✅ Shipped v0.1.1, ternary types |
| oxillama-arch | [crates/oxillama-arch/TODO.md](crates/oxillama-arch/TODO.md) | ~~DeepSeek, Falcon, MiniCPM, Olmo2, Granite, Mamba-2~~ ✅ Shipped v0.1.1; audio/video modalities next |
| oxillama-runtime | [crates/oxillama-runtime/TODO.md](crates/oxillama-runtime/TODO.md) | ~~flash attention~~ ✅ ~~true continuous batching~~ ✅ Shipped v0.1.1; prefix KV server wiring, multi-LoRA |
| oxillama-server | [crates/oxillama-server/TODO.md](crates/oxillama-server/TODO.md) | Function/tool calling, auth, rate limiting, `/metrics` |
| oxillama-bench | [crates/oxillama-bench/TODO.md](crates/oxillama-bench/TODO.md) | End-to-end benches, prefill vs decode split, regression gate |
| oxillama-py | [crates/oxillama-py/TODO.md](crates/oxillama-py/TODO.md) | `.pyi` stubs, numpy interop, async, HF Hub loader |
| oxillama-wasm | [crates/oxillama-wasm/TODO.md](crates/oxillama-wasm/TODO.md) | WebGPU bridge, streaming GGUF load, IndexedDB cache |
| oxillama-gpu | [crates/oxillama-gpu/TODO.md](crates/oxillama-gpu/TODO.md) | ~~batched GEMV~~ ✅ ~~tiled GEMM~~ ✅ ~~fused attention~~ ✅ ~~IQ2/IQ3 kernels~~ ✅ Shipped v0.1.1; K-quant GPU coverage, naga MSL/SPIR-V |
| oxillama-cli | [crates/oxillama-cli/TODO.md](crates/oxillama-cli/TODO.md) | Interactive chat, TUI, `oxillama hub` |

---

## Crate Dependency Graph

```
                       oxillama-gguf
                             |
                             v
                       oxillama-quant
                             |
                             v
                       oxillama-arch
                             |
                             v
                      oxillama-runtime
          +------+------+------+------+------+
          |      |      |      |      |      |
          v      v      v      v      v      v
       server   py    wasm    gpu    cli    bench
          \      \     |      /      /      /
           \      \    |     /      /      /
            +-------- oxillama (meta re-export)
```

The chain `gguf -> quant -> arch -> runtime` is strict; no leaf reaches past
`runtime` into `arch` without reason. `oxillama-gpu` is consumed via an optional
feature from `runtime` (CPU-fallback guarantee). The meta `oxillama` crate
re-exports every sibling under `oxillama::{gguf, quant, arch, runtime, server,
bench, gpu}`, so downstream apps only depend on one crate.

---

## v1.1 Cross-Cutting Roadmap

Themes that span multiple crates. Each theme references the primary subcrate
TODO where detailed work items live.

### Prefix KV caching (runtime + server) — FULLY SHIPPED ✅ v0.1.3

~~Radix-tree-indexed shared-prefix reuse with copy-on-write on divergence so that
shared system prompts are paid for once across concurrent requests.~~ ✅ Fully
shipped: runtime `PrefixKvCache` (v0.1.2) + server-side wiring (v0.1.3):
`engine.prime_with_prefix`, `generate_with_logits`, `store_kv_in_prefix_cache`,
per-request `cache_prompt` flag, `AppState::prefix_cache` Arc, and worker-side
hit/miss/store logic.

### Function / tool calling (server + runtime grammar)

OpenAI-compatible `tools` field on chat completions, mapping JSON-schema to
GBNF inside `oxillama-runtime::sampling::grammar`, enforced via the existing
`apply_grammar_mask` pipeline. Server returns tool invocations as structured
messages with `function_call` / `tool_calls`. See `oxillama-server/TODO.md` §6
and `oxillama-runtime/TODO.md` §6.

### SIMD breadth (quant)

Close the gap where 18 of 25 quantization types remain scalar-only: AVX-512 +
NEON kernels for Q5_K / Q6_K (LLaMA-3 dominant formats); ~~AVX2 for Q2_K / Q3_K
(phone / Pi deployments)~~ ✅ Shipped in v0.1.1; ~~AVX-512 for Q2_K / Q3_K~~ ✅
Shipped in v0.1.3; ~~AVX2 for IQ2_XXS (the most common I-quant in HF GGUF
uploads)~~ ✅ Shipped. Full matrix in `oxillama-quant/TODO.md` §2 + §6.

### More architectures (arch + runtime feature flags) — SHIPPED

~~Add DeepSeek-V2/V3 (with Multi-head Latent Attention), Falcon, MiniCPM, Olmo2,
and Granite-3.x to `oxillama-arch`. Each gets a per-arch feature flag in
`oxillama-runtime` so binary size scales down for focused deployments.~~ ✅ Shipped
in v0.1.1: Falcon, DeepSeek-V2/V3 (MLA + sigmoid MoE scoring), DBRX (16-expert,
top-4), Grok-1 (8-expert, top-2), Mamba-2 (selective scan, learned Δ), Jamba
(hybrid SSM+attention), OLMo2, Yi, Granite, MiniCPM, InternLM3. 20 architectures total.
Details in `oxillama-arch/TODO.md` §6.

### GPU kernel breadth (gpu) — partially shipped

Extend `oxillama-gpu` from 6 quant shaders to cover remaining K-quants,
~~batched GEMV for prefill~~ ✅ Shipped (`BatchedGpuKernel`, Q4_0 batched impl),
~~IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S GPU GEMV kernels~~ ✅ Shipped in v0.1.1 (+4 kernels,
now 14 quant types on GPU), ~~tiled GEMM WGSL shader (TILE_M/N=32, TILE_K=16,
shared memory cooperative)~~ ✅ Shipped in v0.1.1, ~~fused attention WGSL kernel
(single-dispatch QK+softmax+AV)~~ ✅ Shipped in v0.1.1,
~~Q4_1/Q5_0/Q5_1/Q8_1 legacy quad GPU kernels~~ ✅ Shipped in v0.1.3 (now 24
quant types dispatched by `GpuDispatcher::get_kernel`, covers ~85% of
community HuggingFace uploads),
f16 accumulator paths, and naga cross-compile validation (MSL for Metal + SPIR-V
for Vulkan). See `oxillama-gpu/TODO.md` §6.

### GPU CLI/Python wiring (cli, py) — not started

`oxillama-runtime` now depends on `oxillama-gpu` behind the `gpu` feature
(`GpuPolicy`/`GpuOptions`/`GpuStatus`, wired through `EngineConfig::gpu` and
`InferenceEngine::gpu_status()`), but nothing above it actually turns the
knob on:

* **CLI**: `crates/oxillama-cli/src/cli_args.rs` already has
  `gpu_policy_from_flags(gpu, device, n_gpu_layers)` and `print_gpu_banner`,
  both unit-tested, but neither `run` nor `serve` defines `--gpu` /
  `--gpu-device` / `--n-gpu-layers` clap arguments or calls them — both
  functions are dead code outside `#[cfg(test)]` (only visible with
  `--all-features`, since both now carry `#[cfg_attr(not(feature = "gpu"),
  allow(dead_code))]`). Needs: clap arg definitions with `requires = "gpu"`
  on the two sub-flags (per the existing doc comment's stated contract),
  threading into the `EngineConfig` built by `run`/`serve`, and a
  `print_gpu_banner(engine.gpu_status())` call after a successful
  `load_model()`.
* **Python**: `crates/oxillama-py/src/engine.rs`'s `PyEngineConfig::to_rust()`
  hardcodes `gpu: GpuPolicy::Off` (see the comment there) — there is no
  constructor kwarg to turn GPU offload on, and no `GpuStatus` wrapper or
  `Engine.gpu_status()` accessor. `GpuUnavailableError` exists in the Rust
  exception hierarchy and is exported from the Python package already, so
  the error-handling half is ready; only the config/status surface is
  missing.

Found while fixing an `oxillama-py` build break on 2026-08-14 (v0.1.4 branch)
caused by these fields/variants landing in `oxillama-runtime` without the
downstream crates catching up.

### Python polish (py) — partially shipped

~~Generate `.pyi` stubs for IDE autocompletion~~ ✅, ~~wrap sampler as a proper Python
class~~ ✅, ~~expose `Tokenizer`~~ ✅, ~~return numpy arrays from `embed()`~~ ✅
(`embed_numpy()` / `embed_batch_numpy()` gated on `numpy` feature),
add pytest suite and sphinx docs. The goal is a public API at parity with
major Python LLM clients. See `oxillama-py/TODO.md` §6.

### WebGPU in browser (wasm + gpu)

Bridge `oxillama-gpu` into `oxillama-wasm` so browsers get Q4_0 / Q8_0 GPU
matmul via WebGPU. Adds IndexedDB model caching (no re-download across page
loads), streaming GGUF via `ReadableStream`, and a headless-browser test
harness. Joint effort across `oxillama-wasm/TODO.md` §6 and
`oxillama-gpu/TODO.md` §6.

### Observability (server)

Prometheus-compatible `/metrics` endpoint, tracing spans through the full
request lifecycle (queue → prefill → decode → stream), bearer-token auth
middleware, and a token-bucket rate limiter keyed on API key. See
`oxillama-server/TODO.md` §6.

---

## v2.0+ Vision

Longer-horizon themes aligned with COOLJAPAN sovereignty: Pure Rust end to
end, cross-platform, auditable, and independent of any C/C++ toolchain.

- **Full scirs2 / oxiblas / oxifft integration.** Workspace deps are already
  declared, but code adoption is light. Migrate tensor primitives to
  `scirs2-core`, float-path GEMM to `oxiblas`, and RoPE to `oxifft` so the
  COOLJAPAN stack becomes a first-class BLAS substrate for `oxillama-runtime`.
- **RISC-V RVV 1.0 SIMD.** `simd-riscv` feature with vector-length-agnostic
  kernels for Q4_0 / Q8_0 / Q4_K / Q1_0_G128, matching the existing NEON tier.
  Blocked on stable `std::arch::riscv64` intrinsics.
- ~~**State-space models.**~~ ✅ Shipped v0.1.1: Mamba-2 (selective scan, learned Δ)
  and `SequenceState` trait (arch-internal SSM abstraction) both shipped.
  **Jamba (hybrid SSM+attention) is NOT shipped** — the claim was withdrawn in
  0.1.4 and the `jamba` feature removed from `oxillama-arch`'s `default` list.
  What exists is a non-functional stub: the SSM `C` matrix is hard-coded to
  zeros (so the recurrent branch is only the `D` skip), `attention_forward`
  ignores the KV cache, applies no RoPE, does no head split and replaces softmax
  attention with `v * tanh(score)`, the SSM state is re-allocated inside the
  per-token loop, `ssm_dt_norm`/`ssm_b_norm`/`ssm_c_norm` are never loaded, and
  there is no MoE despite `expert_count` in the config. `load_jamba_from_gguf`
  now returns `NotSupported` naming those gaps rather than loading a model that
  produces plausible-looking wrong output. A real implementation must follow
  llama.cpp's `llm_build_jamba` (per-layer hybrid switch
  `recurrent_layer_arr[i] = n_head_kv(i) == 0`) and needs a
  `build_minimal_jamba_gguf` fixture, which was never written.
  Parallel associative scan remains on the roadmap.
- **Multi-GPU dispatch.** Tensor-parallel matmul across wgpu adapters, with
  an explicit placement API that lets users pin individual layers to specific
  devices.
- **Embedded / `no_std` path.** Strip `OnceLock`, Rayon, and `std::io`
  dependencies behind feature flags so scalar kernels compile for
  low-resource devices (microcontrollers, sensors with LLM inference).
- ~~**Tiled GEMM with shared memory.**~~ ✅ Shipped v0.1.1: TILE_M/N=32, TILE_K=16
  cooperative WGSL shader; fused attention kernel also shipped. Multi-GPU
  dispatch and Metal/Vulkan naga cross-compile remain on the roadmap.
- **Ternary quantization.** TQ1_0 and TQ2_0 (BitNet b1.58 and descendants);
  popcount on AVX-512 VPOPCNTDQ + `vcntq_u8` on NEON. Positions OxiLLaMa as
  the first Pure Rust runtime to ship them.
- **Audio / video modalities.** Whisper architecture, extended vision-language
  models (Qwen2-VL, LLaVA-1.6, Molmo) with tighter vision-text alignment.
- **Autonomous model registry / model mesh.** `oxillama hub` expanded into a
  peer-to-peer model discovery layer that validates checksums, resolves LoRA
  deltas, and supports cluster-wide model sharing.

---

## Compatibility Matrix

Reality check of what runs where today. "Partial" means the path exists but
has a known caveat (memory, feature coverage, browser API). "No" means the
combination is not yet wired up.

GPU column note: `GpuDispatcher::get_kernel` (`crates/oxillama-gpu`) dispatches
a GEMV kernel for all K-quant types (Q2_K-Q8_K), not just Q4_0/Q8_0 — the
K-quant rows below were previously marked "no", which understated kernel
coverage. They are marked "partial (GEMV only)" rather than "works" because
having a per-tensor kernel is not the same claim as "the model runs
end-to-end on GPU"; that broader integration has not been verified here.

| Model | Quant | x86-64 CPU | ARM64 CPU | WASM | GPU (wgpu) |
|---|---|:-:|:-:|:-:|:-:|
| LLaMA-3-8B | Q4_0 | works | works | works | partial (GEMV only) |
| LLaMA-3-8B | Q4_K_M | works | works | works | partial (GEMV only) |
| LLaMA-3-8B | Q8_0 | works | works | works | partial |
| Qwen3-7B | Q4_K_M | works | works | works | partial (GEMV only) |
| Mistral-7B | Q4_K_M | works | works | works | partial (GEMV only) |
| Mixtral-8x7B | Q4_K_M | works | works | partial (memory) | partial (GEMV only) |
| Gemma-3-4B | Q4_K_M | works | works | works | partial (GEMV only) |
| Phi-3-mini | Q4_K_M | works | works | works | partial (GEMV only) |
| Bonsai-8B | Q1_0_G128 | works | works | works | partial |
| LLaVA-1.5 | Q4_K_M | text-only | text-only | text-only | not reachable |
| StarCoder-15B | Q4_K_M | works | works | partial (memory) | partial (GEMV only) |
| Command-R-35B | Q4_K_M | works | works | no (memory) | not reachable |

**Read the GPU column as "kernel exists", not "usable".** `oxillama-gpu` ships
24 dispatching quant kernels (Q2_K–Q8_K included) plus tiled GEMM, fused
attention and GPU sampling, and its Q4_0 resident path measures ~9.8x over the
non-resident one on Apple M3/Metal. `oxillama-runtime` now depends on it
behind the `gpu` feature (`EngineConfig::gpu` / `GpuPolicy`, wired through
`gpu_backend::apply_gpu_policy` into the Q4_0 resident kernel only — see that
module's doc comment for exactly which call sites are covered), but neither
the CLI nor the Python bindings expose a way to turn it on yet — see "GPU
CLI/Python wiring" above — so no shipped entry point reaches it.

**LLaVA is text-only in practice.** The CLIP tower, projector, prompt splicer
and embedding-injection path are implemented and unit-tested, but they have
never been exercised against a real mmproj GGUF, and the CLI has no
`--mmproj <path>` flag — which is how llama.cpp selects the multimodal path
(LLaVA checkpoints ship as `general.architecture = "llama"` plus a separate
`clip`-arch mmproj file, so the vision path cannot be selected from the main
GGUF alone).

---

## Performance Targets + Measured

### Target (from design specification)

| Model | Quant | llama.cpp | OxiLLaMa Target |
|-------|-------|-----------|-----------------|
| LLaMA-3-8B | Q4_K_M | ~30 t/s | >= 25 t/s |
| Bonsai-8B | Q1_0_G128 | ~25 t/s | >= 22 t/s |
| Mistral-7B | Q4_K_M | ~32 t/s | >= 27 t/s |

*x86-64, 8 cores, AVX2, multi-threaded. Target: >= 80% of llama.cpp throughput.*

### Measured (v0.1.1 end-to-end re-bench pending)

| Model | Quant | OxiLLaMa Measured (CPU) | OxiLLaMa Measured (GPU) |
|-------|-------|-------------------------|-------------------------|
| LLaMA-3-8B | Q4_K_M | TBD (see oxillama-bench/TODO.md §6 v1.1) | TBD (see oxillama-bench/TODO.md §6 v1.1) |
| Bonsai-8B | Q1_0_G128 | TBD (see oxillama-bench/TODO.md §6 v1.1) | TBD (see oxillama-bench/TODO.md §6 v1.1) |
| Mistral-7B | Q4_K_M | TBD (see oxillama-bench/TODO.md §6 v1.1) | TBD (see oxillama-bench/TODO.md §6 v1.1) |

Per-kernel criterion numbers (all 25 quant types) are already captured inside
`oxillama-bench`. End-to-end throughput re-bench is the first v1.1 deliverable.

---

## External Ecosystem Integration

Current state: `scirs2-core`, `oxiblas`, and `oxifft` are declared as
workspace dependencies, but code adoption is light. The v1.1+ plan closes
that gap by making the COOLJAPAN stack an authoritative substrate rather
than an optional import.

| Dependency | v0.1.1 Status | v1.1+ Plan |
|---|---|---|
| `scirs2-core` | In use for CPU feature detection wrapper | Expand to tensor primitives (stride math, reduction kernels) |
| `scirs2-linalg` | Declared, unused | Adopt for reference GEMM paths (F16/BF16/F32 float tier) |
| `scirs2-neural` | Declared, unused | Adopt for common building blocks (layer norm, activations) |
| `oxiblas` | ✅ Wired v0.1.1 | Float GEMM fallback (F16/BF16/F32) shipping; fused dequant+GEMM for Q4_0/Q4_K also shipped |
| `oxifft` | Declared | Wire into RoPE acceleration for very long context |
| `oxicode` | Not yet declared | Replace any `bincode` usage per COOLJAPAN policy |
| `oxiarc` | Not yet declared | Compression for model packaging + LoRA distribution |

Concrete milestones for each line item are tracked in the primary subcrate
TODO files (`oxillama-quant/TODO.md` for oxiblas / scirs2 adoption,
`oxillama-runtime/TODO.md` for oxifft, `oxillama-gguf/TODO.md` for oxicode /
oxiarc).

---

## Contribution Hot Spots

Entry points for contributors, ordered from smallest self-contained tasks to
larger cross-cutting ones. Each points into a specific subcrate TODO section.

- **AVX-512 Q5_K kernel** — see `oxillama-quant/TODO.md` §6. Adds a wide-lane
  dequant + GEMV for one of the most-used LLaMA-3 formats.
- ~~**Falcon architecture**~~ ✅ Shipped v0.1.1.
- ~~**Prefix KV caching**~~ ✅ — shipped in `oxillama-runtime` (see `oxillama-runtime/TODO.md` §6).
  Server-side wiring remains.
- **Function calling** — see `oxillama-server/TODO.md` §6. Wire OpenAI `tools`
  field to GBNF-masked JSON output.
- **WebGPU bridge** — see `oxillama-wasm/TODO.md` §6 + `oxillama-gpu/TODO.md`
  §6. Cross-cutting; run Q4_0 GEMV shader inside a browser tab.
- **Python `.pyi` stubs** — see `oxillama-py/TODO.md` §6. Unblocks IDE
  autocompletion for every Python consumer.
- **End-to-end bench suite** — see `oxillama-bench/TODO.md` §6. ~~Prefill vs
  decode split~~ ✅, ~~per-arch token/s~~ ✅, cross-SIMD comparison tables.
- **`oxillama chat` REPL** — see `oxillama-cli/TODO.md` §6. First-time
  contributor friendly; readline + history + per-model profile.
- **Streaming GGUF load** — ~~see `oxillama-gguf/TODO.md` §6. Lazy tensor
  streaming via the loader interface~~ ✅ Shipped (`StreamingGgufParser`).
  Browser and HTTP integration pending.
- **Multi-LoRA slot switching** — see `oxillama-runtime/TODO.md` §6. N
  pre-loaded adapters, per-request selection without GGUF re-parse.

---

## Success Criteria (v0.1.0)

> **v0.1.0 SUCCESS CRITERIA:**
> 1. ✓ All mainstream quantization types implemented (25 types)
> 2. ✓ All core architectures: LLaMA, Qwen3, Mistral, Gemma, Phi, StarCoder, Command-R, Mixtral-MoE, LLaVA
> 3. ✓ OpenAI-compatible server with streaming, batching scaffolding, embeddings
> 4. ✓ WASM full inference, Python bindings, fuzz harness, benchmarks
> 5. ✓ GPU backend (wgpu) feature-gated with CPU fallback
> 6. ✓ 85%+ test coverage (87.09% region / 87.23% function / 85.42% line)
> 7. ✓ A stock HuggingFace GGUF runs with no sidecar. Verified by execution:
>    `oxillama run --model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --prompt "The
>    capital of France is"` → *" Paris, which is situated in the north-central
>    part of the country. It lies on both"*. The GGUF-embedded vocabulary
>    loader lives in `crates/oxillama-runtime/src/gguf_vocab/` and reads
>    `tokenizer.ggml.{model,pre,tokens,scores,token_type,merges,*_token_id}`.
> 8. **Partly verified — do not read this as end-to-end parity.** What is
>    checked against llama.cpp today:
>    - `oxillama-quant/tests/golden_vectors.rs` — expected values produced by
>      extracting upstream's `dequantize_row_*` bodies, compiling them, and
>      running them on the same bytes the Rust test builds. Covers the seven
>      formats whose layout was wrong (Q4_0, Q4_1, IQ4_NL, IQ4_XS, Q5_K,
>      TQ1_0, TQ2_0) plus Q5_0/Q8_0/Q4_K/Q6_K as controls.
>    - `oxillama-gpu/src/kernels/golden_tests.rs` — the same oracle for the
>      GPU kernels, so CPU and GPU are pinned to upstream rather than to each
>      other.
>    - `oxillama-runtime/tests/gguf_tokenizer.rs` — 12 vocabularies × 46 cases
>      against llama.cpp's own `.inp`/`.out` fixtures, exact.
>    - **Whole-model logit parity (2026-08-05).** Full-vocabulary float32
>      logits were compared against llama.cpp `ba7e817e` (ggml 0.9.7) built
>      CPU-only with every matmul fast path off (`GGML_CPU_REPACK`,
>      `GGML_LLAMAFILE`, `GGML_BLAS`, `GGML_ACCELERATE` all OFF). Dumps came
>      from `oxillama run --dump-logits <DIR>` (`crates/oxillama-cli/src/dump_logits.rs`).
>      Repeatable as a test:
>      `crates/oxillama-cli/tests/llamacpp_logit_parity.rs` embeds the golden
>      llama.cpp token ids and per-step margins and re-runs the whole check
>      through the real binary; it skips unless
>      `OXILLAMA_PARITY_LLAMA3_GGUF` / `OXILLAMA_PARITY_QWEN3_GGUF` point at
>      the checkpoints (they are far too large to commit).
>
>    **Exactly what that verified:** 2 architectures (`llama`, `qwen3`) ×
>    2 prompts × 8 teacher-forced decode steps = 32 logit vectors of width
>    128256 / 151936, on `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` and
>    `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`. CPU only. Q4_K_M only. Prompts of
>    5 and 58 tokens; max sequence position 65. Greedy sampling only.
>    Measured against llama.cpp's *own* cross-kernel spread (two builds of the
>    same commit differ by up to 0.626 logits, min cosine 0.999017, max KL
>    0.008846 nats, and flip 1 of 29 greedy tokens):
>
>    | metric | llama.cpp vs llama.cpp | OxiLLaMa vs llama.cpp |
>    |---|---|---|
>    | max abs logit diff | 0.62574 | 0.89055 |
>    | max mean abs diff | 0.134356 | 0.188168 |
>    | max KL(ref‖ours) | 0.008846 | 0.010844 |
>    | min cosine | 0.999017 | 0.998673 |
>    | min top-20 overlap | 18/20 | 17/20 |
>    | top-1 agreement | 28/29 | **32/32** |
>
>    With the KV cache dtype matched to llama.cpp's (`--kv-dtype f16`), all
>    32/32 steps pick the same greedy token, all 26/26 "decisive" steps (those
>    whose reference top1−top2 margin exceeds the 0.626 cross-kernel spread)
>    match, and all 4/4 free-running greedy continuations are token-for-token
>    identical to llama.cpp's. At OxiLLaMa's default `--kv-dtype f32`, 31/32
>    steps and 3/4 continuations match; the single flip is qwen3/p2 step 4,
>    where the reference's own margin is 0.1246 logits — the same step, and the
>    same replacement token, at which llama.cpp's default CPU build diverges
>    from this reference. OxiLLaMa's dumps are byte-identical across processes
>    and across `-t 1` vs `-t 8`.
>
>    What is still **not** checked, and must not be inferred from the above:
>    bit-exactness (0/32 steps are byte-identical, and llama.cpp is not
>    byte-identical to itself either); any architecture other than `llama` and
>    `qwen3`; any quantization type other than Q4_K_M end-to-end; contexts
>    beyond 65 positions (so no long-context RoPE / YaRN path); more than 8
>    decode steps; EOG handling (no run emitted one); chat templates;
>    non-greedy sampling; the server, GPU, LoRA, speculative or batched paths;
>    and any CPU other than Apple Silicon arm64.
>
>    **Known tokenizer divergence found by this pass:**
>    `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` carries neither
>    `tokenizer.ggml.pre` nor `tokenizer.ggml.add_bos_token`. llama.cpp then
>    falls back to `LLAMA_VOCAB_PRE_TYPE_DEFAULT` with `add_bos = false`;
>    OxiLLaMa infers `PreType::Llama3` from the special tokens and prepends BOS
>    `128000`. The prompt body tokenizes identically — only the BOS differs —
>    but it is enough to change the continuation, which is why criterion 7's
>    recorded output differs from llama.cpp's on the same file. The behaviour
>    is deliberate (canonical Llama-3 does prepend BOS) and is now documented
>    on `gguf_vocab::pretok::implies_add_bos`; a user-facing
>    `--no-add-bos`/`--add-special` override on `run` is still open. Qwen3
>    tokenizes identically to llama.cpp, BOS included (i.e. none).

---

## Milestone History

### M1: OxiBonsai Core (Month 1)
Q1_0_G128 kernels validated; Bonsai-8B generating coherent text.
Deliverable: `cargo install oxi-bonsai` runs Bonsai-8B.

### M2: OxiLLaMa Foundation (Month 2)
GGUF v3 full parser; architecture registry with trait system; Q4_0 + Q8_0
kernels; LLaMA architecture; OxiBonsai absorbed as `oxillama::arch::qwen3`
and `oxillama::quant::q1_0_g128`. Deliverable: LLaMA-3-8B (Q4_0) generates text.

### M3: Quantization Breadth (Month 3)
K-quant family complete; I-quant family complete; Mistral + Gemma
architectures; SIMD dispatch with runtime CPU detection. Deliverable: most
HuggingFace GGUF models load and run.

### M4: Production Runtime (Month 4)
OpenAI-compatible server; continuous batching scaffolding; advanced sampling
(mirostat, min-P, GBNF); multi-threaded inference via Rayon. Deliverable:
drop-in replacement for `llama-server` core endpoints.

### M5: Enterprise Hardening (Months 5-6)
Full test suite (87%+ coverage achieved); fuzz testing on GGUF parser;
performance optimization sprint (AVX-512 / NEON tiers); WASM compilation
target. Deliverable: OxiLLaMa v0.1.0 release candidate.

### M6: Advanced Features (Month 6+)
Speculative decoding, LoRA, vision models (LLaVA-1.5), StarCoder, Command-R,
Mixtral MoE, optional wgpu GPU backend. Deliverable: feature parity with
llama.cpp core inference loop.

---

*Last Updated: 2026-08-17 (v0.1.4 release — 3,751 tests, 87%+ coverage, 25 architectures, 25 quant formats, 24 GPU kernels)*

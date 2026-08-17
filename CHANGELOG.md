# Changelog

All notable changes to OxiLLaMa are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
OxiLLaMa uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - Unreleased

## [0.1.4] - 2026-08-17

### Added

- **GGUF-embedded tokenizer** (`oxillama-runtime/src/gguf_vocab/`): a native
  pure-Rust port of llama.cpp's SPM (greedy bigram merge, *not* Unigram),
  byte-level BPE and WordPiece tokenizers. A stock HuggingFace GGUF now runs
  with no `tokenizer.json` sidecar — previously impossible. Conformance: 12
  vocabularies × 46 cases exact against llama.cpp's own fixtures. EOS/BOS and
  the end-of-generation set now come from GGUF metadata rather than probing
  three hard-coded strings, so generation stops naturally instead of always
  running to `max_tokens`.
- **GGUF loaders for Falcon, GPT-NeoX, StableLM, OLMo2 and MiniCPM**, which
  previously could only be built from pre-materialized `Vec<f32>` weights, and
  a real Mixtral block (its attention was a stub computing `Q = K = V =
  norm(hidden)`). `build_forward_pass` now consults the architecture registry
  and falls back to the direct loaders, so more than the original seven
  architectures are reachable.
- **Vision pipeline for LLaVA / LLaVA-NeXT / Qwen2-VL**: CLIP tower, projector,
  `<image>` prompt splicer and embedding injection. Previously `encode_image`
  had no callers and `LlavaModel::forward` ignored the vision tower entirely.
  Still not exercised against a real mmproj GGUF and the CLI has no `--mmproj`
  flag — see TODO.md.
- Stop sequences, `FinishReason`, and `generate_detailed` returning a
  `GenerationOutcome` with token counts.
- `deny.toml`, so the banned-crate policy is actually enforced.
- **K-quant encoders** (`oxillama-quant/src/kquant/`): byte-for-byte ports of
  llama.cpp's `quantize_row_q{2,3,4,5,6}_K_ref` and their helpers
  (`nearest_int`'s round-half-to-even bit trick, `make_qx_quants`,
  `make_q3_quants`, `make_qkx2_quants`), plus the legacy `Q5_0`/`Q5_1`
  encoders llama.cpp uses as K-quant shape fallbacks. `oxillama quantize
  --target` now accepts 13 formats (Q4_0/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K_S,M,L/
  Q4_K_S,M/Q5_K_S,M/Q6_K) with a literal port of `llama_tensor_get_type`'s
  per-tensor mixture selection — verified to reproduce llama.cpp's exact
  per-tensor type for all 291 + 398 tensors of the two real Q4_K_M
  checkpoints — and new `--allow-requantize`, `--force-requantize`, `--pure`
  and `--dry-run` flags. Encoders are asserted byte-identical to compiled
  upstream C on 22 golden inputs (including real Llama-3 weight slices), and a
  full requantize of Qwen3-4B produces token-for-token identical greedy output
  to the source model. Previously only `Q4_0`/`Q8_0` could be encoded, so
  `quantize --target Q4_K_M` could not work at all.
- **Streaming GGUF writer**: `GgufWriter::declare_tensor` +
  `into_data_writer`/`into_file_data_writer` + `write_tensor`(`_from_chunks`)
  compute all tensor offsets up front and stream data straight to disk — peak
  memory is one chunk, measured 11.3 MiB RSS while writing a 4 GiB logical
  tensor (previously the writer buffered the entire model, ~4.5 GB for an 8B
  quantize). The buffered `add_tensor` API is unchanged and byte-identical;
  `convert` and `quantize` both use the streaming path.
- **F16 KV cache is now reachable.** `oxillama_arch::common::{fetch_keys,
  fetch_values}` return `Cow<[f32]>` — a zero-copy borrow on contiguous f32
  caches (pointer-identity-tested) and a gather via `for_each_key` otherwise.
  All 21 `&[f32]` KV borrow sites across every architecture family were
  migrated, so `KvCacheDtype::F16` no longer errors: `EngineConfig::kv_dtype`
  and a `--kv-dtype {f32,f16}` flag on `run`/`serve` expose it, KV bytes halve
  exactly (`memory_bytes()`-verified), and f16 storage is bit-exact against a
  pre-rounded f32 reference. Default stays f32 (llama.cpp defaults to f16 —
  deliberately not flipped this wave; note that `--kv-dtype f16` is the
  configuration that matches llama.cpp's greedy output 32/32).
- **AVX-512 fused Q8 path**: native 512-bit kernels for Q4_0/Q5_0/Q5_1/Q8_0/
  Q8_1 (AVX-512BW integer-dot tier with an AVX-512F-only fallback, runtime
  `avx512bw` detection) and explicit, documented delegation to AVX2 for the
  five K-quants. Previously enabling `simd-avx512` *removed* the fused path
  AVX2 had, because no AVX-512 kernel overrode `matvec_q8_fused`.
  `simd-avx512` now implies `simd-avx2`, so the five IQ formats with no
  AVX-512 kernel fall back to AVX2 instead of scalar. Validated by a
  line-by-line scalar model against compiled-upstream goldens and by running
  the full x86_64 suite under Rosetta 2; no AVX-512 instruction has executed
  on real hardware yet and no speedup is claimed.
- **MoE fused-Q8 path**: `QuantExpert::forward_fused`/`QuantMoeFfn` (shared by
  Mixtral, DBRX, LLaMA-MoE, Grok and DeepSeek's routed+shared experts)
  quantize the activation vector to Q8_0 once per token and reuse it across
  all selected experts, mirroring the dense path's gating — with bit-for-bit
  parity tests against fresh-per-expert quantization.
- **Real end-to-end benchmark** (`oxillama-bench`, `--bench real_e2e`): model
  load wall-time, prefill tok/s, decode tok/s and peak RSS against a real GGUF
  given via `OXILLAMA_BENCH_MODEL`; skips cleanly when unset.
- **`oxillama run --dump-logits <DIR>`** (debug) writes the full per-step
  logit vector plus a manifest, with `--prompt-tokens`/`--force-tokens` for
  teacher-forced comparison protocols; the env-gated
  `llamacpp_logit_parity.rs` test replays the llama.cpp parity check against
  the real checkpoints.
- **Python input validation**: `EngineConfig`/`SamplerConfig` constructors now
  raise `ValueError` on empty model paths, zero `context_size`, negative/NaN
  `temperature`, out-of-range `top_p`/`min_p`, non-positive
  `repetition_penalty` and invalid `mirostat` modes, with 19 new tests
  (previously invalid values passed through to the engine).
- **Real cancellation**: `GenerationConfig::cancel_flag`
  (`Option<Arc<AtomicBool>>`, zero-cost when unset) is checked every decode
  iteration and breaks with the new `FinishReason::Cancelled`. Python's
  `CancellationToken` is wired to it — `cancel()` now actually stops
  generation (a real-model cancellation test dropped from tens of minutes to
  2.35 s) — and is finally importable: `from oxillama_py import
  CancellationToken` previously raised `ImportError` because `__init__.py`
  never re-exported the class.
- **`oxillama run --add-bos <auto|always|never>`** (default `auto`,
  behavior unchanged): `never` reproduces llama.cpp's tokenization of
  pre-`tokenizer.ggml.pre` Llama-3 GGUFs exactly (verified against the
  recorded llama.cpp token ids); `always` force-inserts BOS.

#### GPU Offload Backend — Q4_0 Decode (`oxillama-runtime`, `oxillama-gpu`, `oxillama-server`)
- **`oxillama-runtime` now depends on `oxillama-gpu`** behind the `gpu` feature. New `gpu_backend` module: `GpuPolicy` (`Off` / `On(GpuOptions)`), `GpuOptions` (adapter selector, `n_gpu_layers`, tensor-size threshold), `GpuDeviceSelector` (`Index`/`Name`), `GpuStatus` (device name, backend, resident tensor/byte counts, upload-failure count) — wired through `EngineConfig::gpu` (default `Off`, CPU behavior byte-for-byte unchanged) and `InferenceEngine::gpu_status()`. Offloads exactly one thing: Q4_0 weight matrices through `oxillama_gpu::gemv_q4_0_resident`, the only kernel in `oxillama-gpu` that uploads weights once at load and reuses a cached pipeline — every other GPU kernel there rebuilds its pipeline (and re-uploads dequantized weights) per call, which measures slower than the CPU path, so none of them are wired here. Applies only to single-token decode, via `ForwardPass::remap_quant_kernels` (implemented for LLaMA and Qwen3); tiled multi-token prefill, embedding-row lookup, and MoE expert/router weights are explicitly out of scope. `GpuPolicy::On` is a hard requirement, not a hint: a model load fails with the new `RuntimeError::GpuUnavailable { reason }` rather than silently falling back to CPU when no device can be bound.
- **`GET /admin/health`** (`oxillama-server`) now reports a `"backend"` object (`BackendInfo`: `gpu_enabled`, `device_name`, `backend`, `resident_tensors`, `resident_bytes`) — the CPU shape (`gpu_enabled: false`) when no backend was reported, populated via `AppState::with_backend_info()` after the CLI loads the engine. `ModelLoader` gained a `default_gpu: GpuPolicy` field, propagated into every pooled model's `EngineConfig`.
- **`oxillama-cli`**: `gpu_policy_from_flags` / `print_gpu_banner` helpers added to `cli_args.rs` (unit-tested) as forward-prep. Neither `run` nor `serve` defines the `--gpu` / `--gpu-device` / `--n-gpu-layers` clap arguments yet, so nothing above the runtime layer can turn this on — see TODO.md's "GPU CLI/Python wiring" entry.

#### GPU Exception Plumbing (`oxillama-py`)
- **`GpuUnavailableError`**: new Python exception mapped from `RuntimeError::GpuUnavailable`, exported from `oxillama_py` (`__all__`) and declared in `__init__.pyi`. The Python bindings do not yet expose a constructor kwarg for GPU offload — `PyEngineConfig::to_rust()` hardcodes `gpu: GpuPolicy::Off` so bindings stay on the CPU path by default — so this lands the exception-handling half of the surface ahead of the config half; see TODO.md's "GPU CLI/Python wiring" entry.

#### Speculative Decoding — Delta KV-Cache Sync (`oxillama-runtime`)
- **Delta-sync speculative decoding** (`src/speculative.rs`): replaced the prior `O(context)` draft-cache rebuild after every round with an `O(1)` truncation-based delta sync. The draft KV cache is now kept aligned with the committed context across rounds; each round retains the verified prefix in place (via `InferenceEngine::truncate`) and recomputes only the single newly committed token. Per-round draft cost drops from `O(context)` to `O(1)` plus one forward pass, independent of context length.
- **`SpeculativeDeltaSync`**: new sync primitive that performs the truncation and records reuse statistics.
- **Carry-across logits**: `draft_next_logits` / `target_next_logits` are now threaded across rounds rather than recovered by replaying the last committed token; eliminates a redundant forward pass per round.
- **Clean prefill invariant**: prompt prefill now feeds all but the final prompt token then runs one `forward_one` call, ensuring both caches are exactly `prompt_tokens.len()` positions deep at round 0 with no duplicated token.
- **Per-position draft distribution for residual sampling**: rejection now uses the draft distribution at the rejected position (`draft_step_logits[i]`) rather than the global last-step distribution, making residual sampling statistically correct.
- `SpeculativeEngine::generate` now calls `self.delta_sync.reset()` at the top of every generation to clear stale sync state.

#### WebSocket Inference Integration (`oxillama-server`)
- **Live inference via WebSocket** (`src/ws.rs`): `handle_socket` now dispatches to the real inference worker queue (`state.queue`) using `BatchRequest::GenerateStream` instead of returning a hardcoded stub token list.
- Streaming tokens flow through an `mpsc::channel<String>` (capacity 32) from a `StreamCallback` closure into the WebSocket drain loop; the `oneshot` reply channel delivers final `UsageStats` for accurate `prompt_tokens` / `completion_tokens` in the `done` event.
- **`format_ws_prompt`**: new helper that formats a `Vec<WsMessage>` into a Phi-3 chat prompt string (`<|system|>` / `<|user|>` / `<|assistant|>` turns) consistent with `chat.rs::format_chat_prompt`.
- Error propagation: unavailable worker, generation failure, and dropped reply channel each produce a typed `WsEvent::Error` frame before the socket closes.

#### CLI `--model-id` Override (`oxillama-cli`)
- **`--model-id <ID>`** flag added to both `serve` and `chat` subcommands (`src/main.rs`): allows callers to override the model identifier reported to API clients, defaulting to the GGUF file stem when omitted.
- Four new CLI unit tests: `serve_model_id_override_parses`, `serve_model_id_defaults_none`, `chat_model_id_override_parses`, `chat_model_id_defaults_none`.

#### GGUF Magic Byte Regression Test (`oxillama-gguf`)
- **Regression test for issue #1** (`src/header.rs`): three-case file-on-disk test that writes minimal GGUF payloads to `std::env::temp_dir()` and validates: (1) correct `b"GGUF"` magic parses without error, (2) the transposed-nibble constant `0x46475547` is rejected with `GgufError::InvalidMagic`, (3) all-zero magic is rejected. Temp files are cleaned up after each case.

#### WebAssembly Worker Clarification (`oxillama-wasm`)
- **`parse_worker_message` Generate dispatch** (`src/worker.rs`): the stub placeholder response has been replaced with a `WorkerOutMessage::Error` that directs callers to `WasmEngine.generate()` or the top-level `generate()` export, clarifying that `parse_worker_message` is a stateless message-routing helper with no loaded model.
- New test `generate_dispatch_produces_error_variant_with_wasm_engine_directive` covering the dispatch logic at the Rust type level.

### Fixed — Correctness

- **Quantization weight layout for seven formats.** `Q4_0`, `Q4_1`, `IQ4_NL`
  and `IQ4_XS` decoded byte *j*'s nibbles to weights `2j`/`2j+1` where GGML
  writes `j`/`j+block/2`; `Q5_K` read the 5th bit from permuted `qh` positions
  (`0,64,128,192,…` instead of `0,32,64,96,…`); `TQ1_0`/`TQ2_0` decoded
  digit-minor instead of digit-major, and `TQ1_0`'s `qh` used four 2-bit
  fields instead of upstream's base-3 fixed-point scheme, making its *values*
  wrong and not merely their order. Fixed across the scalar reference and the
  AVX2/AVX-512/NEON tiers, in both `Q4_0` encoders, and in the 9 corresponding
  GPU kernels. The bugs were invisible because every SIMD tier was
  parity-tested against the (wrong) scalar reference; `oxillama-quant/tests/
  golden_vectors.rs` and `oxillama-gpu/src/kernels/golden_tests.rs` now pin
  both to values produced by compiling and running upstream's own
  `dequantize_row_*` bodies.
- **GPU `Q4_K` and `Q6_K`** used a flat index mapping instead of upstream's
  group-based one — found only once the CPU/GPU cross-check test existed
  (relative error 17%–296%).
- **LLaMA-family RoPE convention.** llama.cpp assigns `LLAMA_ROPE_TYPE_NORM`
  (interleaved pairs `2j`/`2j+1`) to llama/mistral/mixtral/command-r and
  `convert_hf_to_gguf.py` bakes that layout into the weights via
  `undo_permute`; every architecture here used the NeoX half-split. Synthetic
  weights cannot distinguish the two, which is why no test caught it.
- **NEON tail accumulation** counted each remainder element four times in 11
  GEMV kernels (`vdupq_n_f32` broadcast before `vaddvq_f32`); **AVX2 `Q4_1`
  GEMV** dropped the `m` term entirely; **AVX-512 `IQ2_XXS`/`IQ2_XS`** reduced
  undefined register lanes (`_mm512_castps256_ps512` is a shuffle against
  `_mm256_undefined_ps`, not a zero-extension).
- **`QuantKernel::matvec_q8_fused`'s default** indexed a 32-element scratch
  buffer with a `block_size()`-sized loop, panicking for 12 of 27 dispatched
  kernels.
- **Architecture loaders that could not load a real checkpoint:** StarCoder
  required `attn_out.weight` (GGUF spells it `attn_output.weight`); Command-R
  required an `ffn_norm` that Cohere checkpoints do not contain and used the
  sequential LLaMA block instead of Cohere's single-norm parallel block with
  LayerNorm; Gemma sized `buf_attn_out` as `hidden_size` while indexing it by
  `num_heads * head_dim`, panicking on Gemma-2-9B; Phi read a
  `rope.partial_rotary_factor` key no converter writes, so half of every head
  received no positional signal; Gemma-2 soft-capping was silently disabled by
  two misspelled metadata keys; BLOOM omitted `token_embd_norm`, its defining
  feature; ALiBi slopes were wrong for non-power-of-two head counts; DBRX and
  Grok called `kv_cache.advance()` once per *layer*; OLMo2 and MiniCPM
  discarded every prompt token but the last; DeepSeek-V3 used the *biased*
  router score as the combination weight.
- **Grammar-constrained sampling** emitted token 0 forever once the grammar
  completed (every token masked to `-inf`, `argmax` returning index 0) and
  masked EOS at every step. Unseeded samplers derived their "random" seed from
  a stack address, so every request in a process drew an identical stream.
- **`oxillama bench` reported inconsistent token counts run-to-run.** `run_benchmark` built its `GenerationConfig` from `SamplerConfig::default()` rather than the engine's own configured sampler; the default's `seed: None` draws a fresh, time-based RNG seed on every call (`sampling::rng::generate_seed`), so back-to-back runs against the same seeded engine reported different completion-token counts for reasons unrelated to throughput. Now built from `engine.config().sampler.clone()`, matching `run`/`serve`. The same unseeded-sampler trap was also present in an `oxillama-runtime` engine test (`sequence_isolation`), which relied on `temperature: 0.0` alone for determinism — that only skips `TemperatureScale`'s rescaling step, not the RNG draw itself, so `top_k`/`top_p` could still land on a random token; fixed by switching to `SamplerConfig::greedy()` (`top_k: 1`), which short-circuits `select_token` with no RNG draw at all.

### Fixed — Security

- **GGUF parser, remotely crashable two ways.** `BinaryReader::check` computed
  `pos + n` and overflowed `usize`, so a declared string length of `u64::MAX`
  wrapped and *passed* the bounds check — a 40-byte crafted file defeated it
  in a release build. Unbounded metadata-array recursion aborted the process
  from 12 input bytes per nesting level. Also fixed: unchecked `u64`
  arithmetic on tensor offsets/sizes, allocation bombs from unvalidated
  `n_dims`/string lengths, `align_up` overflow reachable from
  `oxillama info`, and silent duplicate-tensor collapse.
- **`/admin/*` was unauthenticated in both router builders** — the guard
  existed but was only ever applied inside `#[cfg(test)]` code — and its
  handlers take an attacker-supplied filesystem path for
  `InferenceEngine::load_model` and `LoadedLora::load`. Loopback detection now
  fails closed when `ConnectInfo` is absent, and model paths are checked
  against a configured allow-list.
- **Path traversal in all three disk stores.** Request IDs went straight into
  `PathBuf::join`; axum percent-decodes path parameters after routing, so
  `..%2F` traversed and `%2Ftmp%2Fx` (an absolute component) discarded the
  base entirely. `FilesStore::delete` then called `remove_dir_all` on the
  result.
- API-key comparison is now constant-time; the per-key rate-limit map is
  bounded; `overflow-checks` is enabled for the crates that parse untrusted
  bytes.

### Fixed — Shipped server ran with every middleware disabled

The `serve` binary called `build_app()` rather than `build_app_with_config()`,
so authentication, rate limiting, body-size limits, CORS, `/metrics`,
structured tracing and graceful shutdown were all dead in the deployed server.
`serve` is now wired to `build_app_with_config`, binds with
`into_make_service_with_connect_info::<SocketAddr>()` (required for the admin
loopback check) and shuts down via `shutdown_signal()`. Also added: request
cancellation on client disconnect, load shedding (429 instead of parking on a
full queue), per-request panic recovery with worker liveness tracking, a
`/ready` endpoint, and a prefix-KV-cache namespace keyed by
`(model_id, lora selection)` — previously a LoRA request's KV state could be
served to a non-LoRA request, since `cache_prompt` defaults to true.

### Fixed — Server

- **Chat endpoints rendered every model with a fabricated template.**
  `format_chat_prompt()` (and four sibling copies — five sites total,
  including the batch spooler) hardcoded a `<|system|>…<|end|>` format no
  model was trained on; Qwen3 imitated the fake marker as literal text
  (`"Hello! <|end|#>"` in `message.content`). The CLI's model-family
  detection and rendering (Llama-3 / ChatML / Mistral / Alpaca, from GGUF
  metadata and vocab special tokens) moved into
  `oxillama_runtime::chat_template`; all server routes now resolve the
  template once at model load and render with it (Qwen3 now returns
  `"Hello! 😊"`, `chat_template=chatml` in the logs). The prefix-cache lookup
  tokenizes with the same `add_special` policy as the decode path, so a cache
  hit can no longer restore KV state for a differently-tokenized prompt.
- **`finish_reason` was hardcoded to `"stop"`.** The worker discarded the
  `FinishReason` the decode loop computed. Now plumbed through chat
  (non-streaming and the trailing SSE chunk), completions, WebSocket and
  batch records; hitting `max_tokens` reports `"length"`.
- Pre-existing latent breakage fixed along the way: the `bench` feature
  failed to compile (an `EngineConfig` construction missed the new
  `kv_dtype` field), `cargo test --doc` failed in two places (`RECIPES.md`
  and a Python `Example::` block rustdoc compiled as Rust), and the
  `FinishReason` `.pyi` stub listed four variants.
- **`rate_limit.rs` poison guard** (`oxillama-server`, `src/rate_limit.rs`): test-only `buckets.read().unwrap()` replaced with `.unwrap_or_else(|e| e.into_inner())` to avoid panicking if a previous test poisoned the lock.

### Fixed — Packaging

- The Python wheel shipped a 127-byte maturin shim and none of the package —
  `python-source` was missing from `pyproject.toml`. The wheel now contains 14
  files instead of 5, and a `py.typed` marker makes the `.pyi` stubs visible to
  mypy and pyright (without it, PEP 561 requires type checkers to ignore them).
- `oxillama-wasm` did not compile with `--no-default-features`; the meta crate's
  default feature set contained no architecture, so some feature selections
  produced a runtime with zero forward-pass arms that failed at load time.

### Fixed — Benchmarks & Python bindings

- `oxillama-bench`'s `long_context` benchmark was missing the `harness = false`
  `[[bench]]` entry that its other 11 targets all have, so it silently ran
  under the default `libtest` harness instead of Criterion and reported 0
  measurements (`cargo bench --bench long_context -- --test` compiled and
  exited 0 without ever invoking Criterion). Added the entry to `Cargo.toml`;
  also fixed the bench file's doc comment, which referenced a nonexistent
  `--features bench` flag (this crate has no `[features]` table). Verified:
  `cargo bench -p oxillama-bench --bench long_context -- --test` now reports
  real measurements for all five context lengths.
- `oxillama-py`'s hub download (`Engine.from_hub()` /
  `oxillama_py.hub.load_from_hub()`) regressed to showing no progress bar
  during this same release's `hf-hub` 1.0 migration (see Dependencies below) —
  restored by adding an optional `indicatif` dependency (gated on the `hub`
  feature) and a `PullProgress` handler in `hub.rs`, mirroring `oxillama-cli`'s
  existing implementation. Not exercised against a live download in this
  pass (no network call was made) — the code path mirrors the CLI's own
  handler, which is equally untested at that level.

### Verified — whole-model logit parity vs llama.cpp

- Full-vocabulary logits compared against llama.cpp `ba7e817e` built CPU-only
  with all matmul fast paths off, over 2 models × 2 prompts × 8 teacher-forced
  greedy steps (32 vectors). With `--kv-dtype f16` (llama.cpp's own KV dtype):
  top-1 agreement 32/32, max KL 0.0108 nats, min cosine 0.99867, and all 4
  free-running greedy continuations token-for-token identical — i.e. within
  1.2–1.4× llama.cpp's *own* cross-build noise floor (two builds of the same
  commit differ by up to 0.626 logits and flip 1 of 29 greedy tokens).
  Bit-exactness is not achieved and not achievable (llama.cpp is 0/29
  byte-identical against itself). Scope and limits are recorded in TODO.md
  success criterion 8; the reference tooling lives outside the repo.
- The pass also pinned down a real tokenizer divergence: on Llama-3 GGUFs
  lacking `tokenizer.ggml.pre`, llama.cpp falls back to `add_bos = false`
  while OxiLLaMa infers Llama-3 and prepends BOS 128000. Deliberately
  unchanged (canonical Llama-3 prepends BOS); documented on
  `gguf_vocab::pretok::implies_add_bos`.

### Performance

- **KV cache**: lazy block growth replaces eager full-context allocation —
  1,162 MB → 154 MB of anonymous memory for a 20-token conversation on
  Llama-3-8B (peak RSS 5,788 → 4,757 MB). `clear()` no longer memsets the
  whole buffer: 356,903,038 ns → 6 ns per call.
- **LLaMA decode 1.87x, prefill ~2.2x** via the fused Q8 GEMV path (previously
  reachable only from Qwen3), tiled batched prefill, a quantized embedding
  table instead of a 2.10 GB f32 one, threaded attention, and load-time kernel
  resolution. Measured under heavy load-average contention; the absolute
  numbers are noise-dominated, the ratios are paired.
- The AVX2 fused-Q8 dispatch gate was opened for 8 kernels whose
  `matvec_q8_fused` bodies were previously unreachable dead code.
- **GPU Q4_0 resident path** — pipeline caching plus one-time quantized weight
  upload — measures ~9.8x over the rebuild-every-call path on Apple M3/Metal.
- Sampler top-k uses `select_nth_unstable_by` rather than a full-vocabulary
  sort, and reuses a scratch buffer instead of two vocab-sized allocations per
  token.
- **Zero-copy mmap weight loading** (`oxillama-gguf`, 10 architecture loaders): a new `SharedBytes`/`ByteOwner` abstraction lets `QuantTensor` hold an `Arc`-backed view straight into the mmap instead of a private `Vec<u8>` copy; `GgufModel::tensor_bytes` and the dbrx, deepseek, gemma, grok, llama, mistral, phi, starcoder, qwen2_vl and qwen3 loaders were migrated off `.to_vec()`. On Qwen3-4B Q4_K_M (Apple M3): peak footprint 4.204 GB → 148 MB, weight-load wall time 0.604 s → 0.014 s, page reclaims 413k → 166k; decode throughput unaffected.
- **Threaded GEMV via a dedicated rayon pool**: all 122 serial per-row GEMV loops (reference/NEON/AVX2/AVX-512 tiers) now route through `oxillama_quant::parallel::for_each_row`, which splits output rows over a pool separate from the global rayon pool; rows are never split internally, so results are bit-identical to serial at any thread count (`tests/gemv_parity.rs`). `EngineConfig::num_threads` is finally consumed, with `0` (auto, `available_parallelism()`) as the new default — a pinned 4 threads cost ~20% versus auto on a 4P+4E part.
- **NEON SDOT integer dot products** (`oxillama-quant`): reached through stable `asm!("sdot …")`, since the `vdotq_s32` intrinsic is still gated behind unstable `stdarch_neon_dotprod` (rust-lang/rust#117224) even though `dotprod` is a default AArch64 target feature; a pure-NEON widening fallback covers targets without it, proven to compute the identical integer total. +11.8% decode on Apple M3 (Qwen3-4B Q4_K_M) with everything else held fixed.
- **NEON fused-Q8-activation dispatch gate** opened for Q4_K (+38%: 6.34 → 8.73 tok/s) and Q6_K (+21%: 10.72 → 12.98 tok/s), the NEON counterpart to the AVX2 gate-opening above. Included: `fused_q6_k_q8_0_row_neon`, previously a scalar body despite its name and measured 7.5x *slower* than Q6_K's own plain NEON GEMV, rewritten into real SDOT-based NEON and now 1.89x faster at `ffn_down` (2560×9728) and 1.22x faster at the bandwidth-bound tied LM head.
- **Qwen3 decode 0.346 → 12.98 tok/s (37.5x), prefill 9.34 → 28.8 prompt tok/s (3.08x)** on Apple M3 Q4_K_M — the combined effect of zero-copy mmap loading, threaded GEMV, NEON SDOT, the NEON fused-Q8 gate above and 16-token-tile batched prefill. Generation verified byte-identical at a fixed seed before/after and across 1/4/8 threads (md5-equal); fused-path tolerance is derived from Q8_0 quantization statistics and cross-checked against an independent f64 scalar oracle (max relative error ≤ 2.1e-7 on real shapes).

### Changed

- The workspace dependency on `oxillama-arch` now sets
  `default-features = false`; a new `all-architectures` umbrella feature keeps
  the architecture list in exactly one place, and `oxillama-cli`,
  `oxillama-py` and `oxillama-runtime` name it explicitly instead of relying
  on feature unification. A missing architecture feature is a *run-time*
  "unsupported architecture" rejection, not a compile error — the manifests
  now carry comments saying exactly that, and
  `engine/tests.rs::registry_routing` makes the requirement executable.
- `[profile.release]`: added `overflow-checks = true` as defence in depth. An earlier unchecked integer overflow in the GGUF parser had turned into a bounds-check bypass in release builds; the arithmetic sites were fixed with `checked_*`, and this flag catches the next one before it ships. Measured cost on a Qwen3-4B-Instruct Q4_K_M generation (128 tokens, interleaved A/B runs to control for system load): roughly 10-20% additional CPU time (`user` time), depending on measurement method — wall-clock time on this run was too noisy (concurrent background builds) to use directly.

#### CLI Module Split (`oxillama-cli`)
- `main.rs` split into `run_cmd.rs` and `serve_cmd.rs` (799 lines moved out), keeping the file under the workspace's 2000-line-per-file policy. No behavior change.

- **`QuantTensor::data` changed type from `Vec<u8>` to `SharedBytes`** (`oxillama-quant`) — a pre-1.0 API break for any caller constructing the struct by field literal or pattern-matching its `data` field directly; `QuantTensor::new(Vec<u8>, …)` is unchanged and still accepts a `Vec<u8>`, wrapping it internally, so the 240+ existing call sites that construct via `new` are unaffected. Enables the zero-copy mmap loading above.

#### Paged KV Cache Error Messages (`oxillama-runtime`)
- Multi-page access errors in `PagedKvCache::get_keys()` and `get_values()` (`src/kv_cache/paged.rs`) now include the method names and explicit alternatives (`get_keys_into(&mut buf)` / `for_each_key()`) rather than generic "use get_keys_into() for multi-page access" messages. Error message wording clarified to explain the structural constraint (no interior mutability under `Send + Sync`).

#### KV Cache Snapshot Doc (`oxillama-runtime`)
- `KvCacheSnapshot` doc comment updated (`src/kv_cache/mod.rs`) to reflect that speculative decoding uses the `O(1)` `KvCache::truncate` primitive rather than snapshot/restore, and that snapshot/restore remains the mechanism for engine-level `kv_snapshot` / `kv_restore` helpers.

#### Quantization CLI Doc (`oxillama-cli`)
- `quantize.rs` module-level doc and `as_tensor_type` doc updated: table header changed from "Encoder available" to "Status"; K-quant rows now say "not encodable — returns typed `Err`" and the explanation clarifies that dequantization kernels exist but the inverse FP32 → K-quant encoder does not.

### Dependencies

- `scirs2-core` / `scirs2-linalg` / `scirs2-neural`: `0.4.3` → `0.6.5`
- `oxifft`: `0.3` → `0.4.2`
- `oxicode`: `0.2.2` → `0.2.6`
- `oxiblas`: `0.2.1` → `0.2.2`
- `fancy-regex`: `0.17` → `0.19.0`, and moved from a hardcoded per-crate version in
  `oxillama-runtime/Cargo.toml` to `[workspace.dependencies]` (`fancy-regex.workspace = true`) —
  it was the only dependency in the workspace not managed at that level. `tokenizers` still pulls
  its own internal `fancy-regex 0.17.0` transitively, so both versions now coexist in the resolved
  graph (harmless — `cargo deny check bans` treats this as a non-blocking `multiple-versions` warning).
- `pyo3`: pinned to explicit patch `0.29.1` (was caret `0.29`); `numpy`: pinned to explicit patch `0.29.0` (was caret `0.29`)
- Declared-version resync: `tokio` 1.52.3→1.53.1, `serde` 1.0.228→1.0.229, `serde_json` 1.0.150→1.0.151, `thiserror` 2.0.18→2.0.19, `anyhow` 1.0.103→1.0.104, `clap` 4.6.1→4.6.5, `clap_complete` 4.6.6→4.6.8, `tokio-stream` 0.1.18→0.1.19, `tokio-util` 0.7.18→0.7.19, `futures-util` 0.3.32→0.3.33, `bytemuck` 1.25.0→1.25.2, `naga`/`wgpu` 29.0.3→29.0.4, `sysinfo` 0.39.5→0.39.6, `toml` 1.1.2→1.1.4, plus explicit patch pins for previously caret-only `colored`, `uuid`, `ratatui`, `crossterm`, `indicatif` — cosmetic only, caret ranges already resolved to these versions in `Cargo.lock`.
- `ureq`: added `default-features = false` (keeping only `native-tls`). Drops `flate2`/`miniz_oxide`/`rustls` from the `oxillama-gguf --features http` build path. Compile-verified only — native-tls-only HTTP behavior (no more automatic gzip response decoding) has not been exercised against a live server.
- `wgpu`/`naga`: `29.0.4` → `30.0.0`; `pollster`: `0.4.0` → `1.0.1` (`oxillama-gpu`, optional `gpu` feature). `wgpu::RequestAdapterOptions` gained a new `apply_limit_buckets` field (set to `false` — it only coarsens reported adapter limits as an anti-fingerprinting measure for untrusted content, which does not apply to a local inference engine); `BufferSlice::get_mapped_range()` now returns `Result<BufferView, MapRangeError>` instead of panicking internally, propagated through the existing `GpuError` type at both call sites in `buffer.rs`. Verified with real GPU execution (`cargo nextest -p oxillama-gpu --all-features`: 266 passed; default-features: 227 passed), not just a compile check.
- `hf-hub`: `0.5.0` → `1.0.0` (`oxillama-cli`/`oxillama-py`, optional `hub` feature) — a from-scratch rewrite. Dropped `ureq` entirely for a `reqwest`/tokio core plus a new mandatory `hf-xet` (HuggingFace Xet storage protocol) and `bon` (builder macro) dependency; the old `api::sync::{ApiBuilder, ApiError, ApiRepo}` / top-level `Cache`/`Repo`/`RepoType` types are gone, replaced by `HFClient`/`HFClientBuilder::build_sync()`/`HFClientSync`/`HFRepositorySync`/`HFError`/`split_id`. Both `crates/oxillama-cli/src/hub.rs` and `crates/oxillama-py/src/hub.rs` are migrated onto the new API, staying synchronous via the `blocking` feature (no async introduced at either call site); on-disk cache layout (`models--{org}--{name}/blobs|snapshots/`) is unchanged. `oxillama-cli`'s `--force` re-download path now uses hf-hub's native `.force_download()` instead of ~30 lines of manual cache-eviction code that reached into hf-hub's internal blob layout — a correctness improvement, not just a port (it also fixes a latent panic on non-ASCII filenames in the progress-bar truncation logic, found while porting it to the new `ProgressHandler` callback API that replaced 0.5's built-in indicatif renderer). **Behavior change, fixed within this same release:** `oxillama-py`'s hub download briefly lost its progress bar (0.5 always rendered one by default); restored by adding `indicatif` to `oxillama-py/Cargo.toml` — see "Fixed — Benchmarks & Python bindings" above. See "Security / Supply chain" below for the `lz4_flex`/`aws-lc-sys` consequences of this bump.
- `base64`: `0.22.1` → `0.23.1` (`oxillama-server`) — no code changes required, API unchanged for the `Engine` calls this workspace uses.
- Second round of declared-version resync (cosmetic, `cargo upgrade` with incompatible bumps ignored): `thiserror` 2.0.19→2.0.20, `uuid` 1.24.0→1.24.1, `futures-util` 0.3.33→0.3.34, `clap` 4.6.5→4.6.6, `clap_complete` 4.6.8→4.6.9, `clap_mangen` 0.3.0→0.3.3, `blake3` 1.8.5→1.8.6, `ureq` 3.3.0→3.4.0, `pyo3` 0.29.1→0.29.2, `wasm-bindgen` 0.2.126→0.2.127, `wasm-bindgen-futures` 0.4.76→0.4.77, `js-sys` 0.3.103→0.3.104, `rand` 0.10.1→0.10.2 — caret ranges already resolved to these versions in `Cargo.lock`, req-string-only.
- `rand_core` remains hard-pinned at `=0.6` (see the existing `Cargo.toml` comment: `rsa` 0.9's `CryptoRngCore`/`OsRng` used by `oxillama-server` JWT RS256 tests need it); `cargo upgrade` correctly skipped it.
- Fixed a `cargo publish` packaging hazard (release-check §2.16 Check 1): 7 intra-workspace `[dev-dependencies]` — `oxillama-gguf` in `oxillama-arch`/`oxillama-bench`/`oxillama-cli`/`oxillama-quant`/`oxillama-runtime`, `oxillama-quant` in `oxillama-gpu`/`oxillama-py` — were declared `{ workspace = true, ... }`, which carries the root `[workspace.dependencies]` entry's explicit `version` into the packaged manifest. A dev-dependency with a version is *not* stripped when the crate is packaged, so `cargo publish` would have failed on every one of these 7 crates the moment its sibling dev-dependency wasn't yet on crates.io at the matching version. Converted all 7 to version-less `{ path = "../<crate>", default-features = false, ... }` (explicit `default-features = false` because a bare path dependency doesn't inherit the workspace entry's own `default-features = false`) — path-only dev-dependencies are dropped from the published manifest entirely. `deny.toml`'s `[bans]` gained `allow-wildcard-paths = true` to match: a version-less path dependency is a "wildcard" to `cargo-deny`, and this is the documented exception for exactly this pattern, not a loosening of the `wildcards = "deny"` protection against genuine unpinned crates.io deps. Verified with `cargo publish --dry-run --allow-dirty` from all 11 crate directories in topological order: `oxillama-gguf` (no internal deps) dry-run passes cleanly; every other crate's dry-run still fails, but now *only* on its `[dependencies]` (not `[dev-dependencies]`) siblings not yet existing on crates.io at 0.1.4 — the expected, unavoidable state for a workspace whose members have never been published at this version — confirming the dev-dependency packaging hazard itself is resolved.

### Security / Supply chain

- Added `deny.toml`: enforces the COOLJAPAN never-use → oxi* substitution table via `cargo-deny` (previously unconfigured, so `cargo deny check bans` passed vacuously). `flate2`/`miniz_oxide` were initially banned with a scoped `wrappers` exception for their one known path through `hf-hub` 0.5.0's `ureq` backend; the `hf-hub` 1.0.0 bump (above) removed `ureq` from the graph entirely, so both are now plain (unconditional) bans again, matching every other never-use entry.
- `deny.toml`: new scoped `wrappers = ["xet-core-structures"]` exception for `lz4_flex`, introduced by the same `hf-hub` 1.0.0 bump — `hf-hub`'s new mandatory `hf-xet` dependency (Xet storage protocol) has no feature to disable it, and `xet-core-structures` uses `lz4_flex` (itself Pure Rust; the deviation is from the substitution table, not the Pure-Rust policy) for chunk compression. Reachable only through the optional `hub` feature (off by default in both `oxillama-cli` and `oxillama-py`).
- `deny.toml`: documented (not banned — no `oxi*` substitute exists) that `hf-hub` 1.0.0's `rustls-tls` feature pulls in `aws-lc-sys`/`aws-lc-rs` (rustls's default crypto provider), which compiles C at build time via `aws-lc-sys`'s build script — new versus 0.5.0's `ureq`+`native-tls`, which built no C. Tolerated under "feature-gate FFI off by default" because it is reachable only through the same opt-in `hub` feature; a stock build compiles no C. See `deny.toml` for the full reasoning, including the one point inferred from manifest reading rather than empirically re-verified (whether `reqwest`'s own default features would pull the same crypto provider even without `hf-hub`'s `rustls-tls` selected).
- `.cargo/audit.toml`: added `RUSTSEC-2024-0436` (`paste`, via `tokenizers`) and `RUSTSEC-2026-0173` (`proc-macro-error2`, via `tabled`/`oxillama-bench`) to the ignore list — both unmaintained-crate warnings, proc-macro/build-time only, no runtime exposure. `RUSTSEC-2023-0071` (`rsa` Marvin Attack, optional `jwt` feature) remains ignored, unchanged.
- Removed `.github/workflows/shader_validate.yml.disabled` — a dormant, `push`/`pull_request`-triggered workflow one rename away from running; `.github/workflows/` now contains only the two permitted publish workflows.

### Documentation

- Corrected claims that did not match the code: subsystems marked "✅ Shipped"
  with zero callers (`PagedKvCache`, `KvCachePool`, `Scheduler`,
  `SequencePool`), the GPU compatibility column (kernels existed but no crate depended on
  `oxillama-gpu` at the time — since closed within this same release cycle by
  the GPU Offload Backend entry under Added), LLaVA's "works" row (text-only in practice), a
  `oxillama-runtime` README usage example calling an API that does not exist,
  feature-default tables that were inverted, and a "MeCrab" dependency that is
  not in any manifest.

- README.md: removed the fabricated "MeCrab (Japanese tokenization)" line from the COOLJAPAN ecosystem diagram (no such dependency exists anywhere in the workspace); corrected stale SciRS2/OxiFFT version claims; fixed the Rust version badge (1.86+ → 1.89, matching `rust-version`); removed the stale v0.1.4 test-count claim (`2,243 tests`, internally contradictory against the v0.1.3 section's `2,461`) — the fresh count is now recorded under Quality below.
- CLAUDE.md: corrected the feature-flag table — `simd-avx2` and `simd-neon` are default-enabled (`oxillama-quant`, `oxillama`, `oxillama-cli` all declare them in `default = [...]`); only `simd-avx512` is genuinely opt-in.
- TODO.md: corrected the GPU compatibility matrix (K-quant rows Q2_K-Q8_K were marked `no`; `GpuDispatcher::get_kernel` dispatches kernels for all of them — relabeled `partial (GEMV only)` to match the existing vocabulary used for Q4_0/Q8_0, since kernel-exists is not the same claim as model-runs-end-to-end-on-GPU) and, at the time of that audit, marked two Success Criteria claims as false: GGUF-embedded-tokenizer support (a stock Q4_K_M model failed without an external `--tokenizer`/`tokenizer.json`) and llama.cpp output comparison (no such test existed, contradicting the "bit-level parity" claim). Both gaps were closed within this same release cycle — see the GGUF-embedded tokenizer entry under Added and the whole-model logit parity verification above.
- A second documentation pass (per-crate README.md/TODO.md refresh against actual source, current as of this release) found and corrected further drift, and surfaced two real (not documentation-only) gaps left open rather than silently patched:
  - `oxillama-arch`'s `jamba` feature is missing from its own `all-architectures` aggregate in `Cargo.toml`, so — despite `oxillama-arch`, `oxillama-runtime`, and `oxillama`'s READMEs historically listing Jamba as a working default-on architecture — a stock `default-features` build cannot load a Jamba GGUF at all; only `--features jamba` (or `--all-features`) can. **Correction (later in this same release cycle):** this is not the one-line fix it first appeared to be. `JambaArchitecture` does not override `ModelArchitecture::build_from_gguf()`, so it falls through to the trait default (`ArchError::NotSupported`); the engine's fallback direct-loader match is gated on `oxillama-runtime` feature flags that don't include `jamba` either. Adding `jamba` to `all-architectures` would register the arch id without making it loadable. Left excluded, matching `registry.rs`'s own "not in the default list" comment and README's "non-functional stub" wording — see `oxillama-arch/TODO.md` Known Gaps for the full analysis.
  - `oxillama-bench`'s `long_context` benchmark is missing the `harness = false` `[[bench]]` entry that its other 11 targets all have, so it silently runs under the default `libtest` harness instead of Criterion and reports 0 measurements. Confirmed empirically (`cargo bench --bench long_context -- --test`). **Fixed later in this same release cycle** — see "Fixed — Benchmarks & Python bindings" above.
  - `oxillama-gguf`'s README Quick Start example and half of `oxillama-quant`'s README Key Types/Usage section referenced APIs that do not exist in either crate (`GgufFile::open`, `.get_str`, `dispatch_dequantize`, `oxillama_gguf::GgmlType`, `DequantizeSlice`, among others) — neither example would have compiled. Rewritten against the real API surface.
  - `oxillama-server`'s README undersold the shipped HTTP surface (Files, Assistants/Threads, and Responses APIs, plus the LoRA admin endpoints, had zero README coverage) and mischaracterized two shipped features as incomplete: the WebSocket route (`ws.rs`) was documented as a "stub token stream" but is fully wired through the same worker queue as the SSE chat route, and the persistent batch-jobs endpoint (`/v1/batch_jobs`, disk-spooled, survives restarts) was undocumented entirely in favor of the older in-memory `/v1/batches`, which does *not* survive a restart despite a Features bullet claiming otherwise.
  - Per-crate SLoC figures in the root README's Crate Structure table were stale from an early version of the workspace (some off by more than 5x — e.g. `oxillama-arch` listed as ~7,500 lines against an actual 37,984); replaced with fresh `tokei` measurements. The workspace-wide "~164,000 lines" total was independently re-derived from the same per-crate figures and confirmed already correct.

### Quality

- 3,631 tests passing (default features), 3,751 passing (`--all-features`), 0 failed, 0 skipped-unexpectedly, 0 compiler warnings in either run (`cargo nextest run --workspace` / `--all-features --workspace`).
- `cargo test --doc --workspace`: all doctests pass.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean, in both default and `--all-features` configurations.
- `cargo deny check bans`: clean — no banned crates, per `deny.toml`'s COOLJAPAN never-use → oxi* substitution table.

## [0.1.3] - 2026-05-05

### Added

#### BLOOM + Phi-3.5-MoE Architectures (`oxillama-arch`)
- **`AlibiBias`** (`src/common/alibi.rs`, ~200 LoC): `AlibiBias::new(num_heads)` computing `m_h = 2^(-8*(h+1)/num_heads)` slope-per-head bias; `apply(scores, seq_q, seq_k)` adds ALiBi bias matrix in-place; `slopes()` accessor for testing; 2 tests including geometric-sequence slope invariant
- **`BloomArchitecture`** (`src/bloom/{mod,model,config}.rs`, ~900 LoC): arch id `"bloom"`, gated `bloom` feature (included in default); ALiBi positional bias (no RoPE), pre-LayerNorm, GELU FFN, MHA; bias terms on all attention and FFN projections; tensor names: `blk.{L}.{attn_norm,attn_qkv,attn_output,ffn_norm,ffn_up,ffn_down}.{weight,bias}`; 5 tests including `bloom_no_rope_present` and ALiBi slope reference match
- **`PhiMoeArchitecture`** (`src/phi_moe/{mod,model,config}.rs`, ~700 LoC): arch id `"phimoe"`, gated `phimoe` feature; Phi-3 merged QKV + partial RoPE (first 25% of head_dim) reusing `phi/attention` path; sparse MoE FFN (16 experts, top-2) via `common/moe.rs::SparseTopKMoe`; tensor names: Phi-3 attention layout + `blk.{L}.ffn_{gate,up,down}_exps.weight` / `ffn_gate_inp.weight`; 5 tests
- **GGUF test fixtures**: `build_minimal_bloom_gguf()` (1-layer, hidden=64, 8 heads) + `build_minimal_phi_moe_gguf()` (1-layer, 4 experts, top-2) added to `oxillama-gguf/src/test_utils.rs`
- Arch count: 25 → **27**; `bloom = []`, `phimoe = []` added to default features

#### Advanced Sampler Suite + Embedding Pooling (`oxillama-runtime`)
- **`DryStage`** (`sampling/advanced.rs`): "Don't Repeat Yourself" n-gram penalty; `multiplier * base^(match_len - allowed_length)` subtracted from logit; `sequence_breakers` list prevents cross-sentence penalties; passthrough when `multiplier == 0.0`
- **`XtcStage`**: Exclude Top Choices — collects cumulative-probability top set; with probability `xtc_probability` zeroes all but the single best token in the top set; passthrough when `threshold >= 1.0` or `probability == 0.0`
- **`TypicalPStage`**: locally-typical sampling via Shannon entropy H; sorts tokens by `|ln p(t) + H|` ascending; keeps until cumulative probability ≥ p; passthrough when `p >= 1.0`
- **`TopAStage`**: adaptive threshold `a * max_prob²`; keeps only tokens whose softmax prob exceeds the threshold; passthrough when `a == 0.0`
- **`EtaStage`**: entropy-scaled cutoff `max(epsilon, eta / exp(H))`; combines typical + epsilon into a perplexity-adaptive floor; passthrough when both fields are 0.0
- **`SamplerConfig` extended** with 9 new fields (`dry_multiplier`, `dry_base`, `dry_allowed_length`, `xtc_threshold`, `xtc_probability`, `typical_p`, `top_a`, `eta_cutoff`, `epsilon_cutoff`), all `#[serde(default)]`; existing output byte-identical when all at defaults
- **5 new stages registered** in `SamplerChain::from_config()` after `LogitBias`/`RepetitionPenalty`, before `Temperature`; order: DRY → XTC → TypicalP → TopA → Eta
- **`PoolingMode { Last, Mean, Max, Cls }`** (`src/embedding.rs`, ~250 LoC): serde; `pool_hidden_states(states, seq_len, hidden_size, mode) -> RuntimeResult<Vec<f32>>`
- **`embed_with(text, mode)`** + **`embed_batch_with(texts, mode)`** on `InferenceEngine`; existing `embed()` / `embed_batch()` delegate to `PoolingMode::Last` (zero behaviour change)
- 20 new tests (10 sampler + 8 pooling + 2 edge-case safety)

#### Responses API + Per-API-Key Rate Limiting (`oxillama-server`)
- **`ResponseStore`** (`src/responses_store.rs`, ~280 LoC): `Arc<RwLock<HashMap<String, ResponseRecord>>>` with `create()`, `get()` (404 on miss), `update_output()`, `list()` (descending by `created_at`); 4 unit tests
- **Five route handlers** (`src/routes/responses.rs`): `POST /v1/responses` (non-streaming + SSE; `response.created` / `response.output_text.delta` / `response.completed` / `[DONE]` event names; `previous_response_id` chains prior record into prompt), `GET /v1/responses`, `GET /v1/responses/:id`; 5 integration tests
- **`PerKeyRateLimiter`** (`src/rate_limit.rs` extension, ~250 LoC): lazy per-key `TokenBucket` map (`Arc<RwLock<HashMap<String, Mutex<TokenBucket>>>>`) — read lock on subsequent hits, write lock only on first-seen key; `with_overrides(map)` for per-key capacity/rate; `per_key_rate_limit_middleware` reads `Authorization: Bearer` or `X-Api-Key`; anonymous requests fall through to global limiter; 5 unit tests
- **`ServerConfig.per_key_rate_limits`**: optional override map `HashMap<String, (f64, f64)>` (capacity, rate)
- **Error variants**: `ResponseNotFound(String)` (HTTP 404), `PreviousResponseNotFound(String)` (HTTP 404)
- **`AppState`** extended with `responses_store: Option<Arc<ResponseStore>>`, `per_key_rate_limiter: Option<Arc<PerKeyRateLimiter>>`; builder methods `with_responses_store()`, `with_per_key_rate_limiter()`

#### AVX-512 IQ Kernels + Fused Legacy Matvec (`oxillama-quant`)
- **`Iq2XxsAvx512`**, **`Iq2XsAvx512`**, **`Iq3SAvx512`**, **`Iq4XsAvx512`** (`simd/avx512/{iq2_xxs,iq2_xs,iq3_s,iq4_xs}.rs`, ~900 LoC): mirror AVX2 templates with `__m512i`; `_mm512_permutexvar_epi8` for grid lookup (AVX-512BW); 2× per-iter throughput; runtime-guarded via `is_x86_feature_detected!("avx512bw")`; tests auto-skip on non-AVX-512 hosts; 8 new tests
- **`matvec_q8_fused` for Q5_0/Q5_1/Q8_1** (AVX2 + NEON + scalar reference): single-pass dequant+dot in registers with no scratch f32; Q5_0 signed high-bit reconstruction, Q5_1 affine (`d` + `m` bias), Q8_1 with `s` precomputed-sum correction; 6 new tests (tol 1e-5 on 64×1024 GEMV)

#### GPU Sampling Kernels (`oxillama-gpu`)
- **`sampling.wgsl`** (`src/shaders/sampling.wgsl`, ~185 LoC WGSL): `softmax_logits` (256-thread shared-memory reduction with max+sum two-pass, temperature scaling, temp=0 argmax degenerate path); `topk_partition` (workgroup cooperative selection, k ≤ 256); `sample_categorical` (LCG RNG seeded from host u32 pair + CDF walk)
- **`SamplingKernel`** (`src/kernels/sampling.rs`, ~480 LoC): `softmax(logits, temp) -> GpuBuffer`, `top_k(probs, k) -> (vals, idxs)`, `sample(probs, idxs, seed) -> u32`; GPU-resident chaining variants (`_raw`); graceful `Err(GpuError::NoAdapter)` when no GPU; 10 new tests (CPU-reference always-run + GPU tests auto-skip via `skip_if_no_gpu!` macro)

#### Speculative Decoding Bench + Python Torch Interop (`oxillama-bench` + `oxillama-py`)
- **`SpeculativeBenchTable`** (`src/speculative.rs`, ~480 LoC): `SpeculativePoint { draft_size, accept_threshold, baseline_toks_per_sec, spec_toks_per_sec, speedup, mean_accepted }` (serde); `run_acceptance_sweep()` with deterministic floor-based acceptance; `summary_table()` / `speedup_grid()` Markdown 2-D grids; `default_draft_sizes()` `&[1,2,4,8]`, `default_accept_thresholds()` `&[0.5,0.7,0.85,0.95]`; Criterion bench `benches/speculative.rs` with `OXILLAMA_BENCH_PRINT_SPEC=1` gate; 8 new tests
- **`torch_helper.py`** (`python/oxillama_py/torch_helper.py`, ~155 LoC): `Engine.logits_torch(text) -> torch.Tensor` and `Engine.embeddings_torch(text) -> torch.Tensor` via lazy `import torch` + `torch.from_dlpack(self.logits_dlpack(...))`; monkey-patched onto `Engine` class at module load; graceful `ImportError` with helpful message when `torch` absent; no Rust-level `torch` dependency; type stubs updated in `__init__.pyi`; 8+ Python tests in `test_torch_interop.py` (skipped when `torch` unavailable)

#### Mixtral + StableLM + GPT-NeoX Architectures (`oxillama-arch`)
- **`MixtralArchitecture`** (`src/mixtral/{mod.rs,model.rs}`, ~400 LoC): arch id `"mixtral"`, gated `mixtral` feature; sparse top-2-of-8 MoE FFN reusing `common/moe.rs`; sliding window attention + RMSNorm from Mistral path; tensor names: `blk.{i}.ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`, `ffn_gate_inp.weight` (router); 6 tests including routing correctness and load-balance softmax normalization
- **`StablelmArchitecture`** (`src/stablelm/{mod.rs,model.rs,config.rs}`, ~700 LoC): arch id `"stablelm"`, gated `stablelm` feature; parallel attention+FFN block (`out = residual + attn_out + ffn_out`); partial RoPE on first `partial_rotary_factor` (default 25%) of head_dim; LayerNorm with bias (`common/layer_norm.rs`); 4 tests
- **`GptNeoxArchitecture`** (`src/gpt_neox/{mod.rs,model.rs}`, ~650 LoC): arch id `"gptneox"`, gated `gptneox` feature; parallel residual `x + attn(ln1(x)) + ffn(ln2(x))`; learned-bias LayerNorm; partial RoPE; 4 tests
- Arch count: 22 → **25**; `mixtral = []`, `stablelm = []`, `gptneox = []` added to default features

#### Logit-Bias + JSON-Schema → GBNF + Beam Search (`oxillama-runtime`)
- **`SamplerConfig.logit_bias: HashMap<u32, f32>`** and **`.banned_tokens: Vec<u32>`** (`sampling/mod.rs`): applied as the first step in the sampler chain before temperature/top-k; banned tokens set to `f32::NEG_INFINITY`, biases are additive
- **`SamplerStep::LogitBias`** (`sampling/chain.rs`): inserted before temperature scaling in `SamplerChain::from_config()`
- **`JsonSchemaCompiler::compile(schema_json) -> GrammarResult<Grammar>`** (`sampling/grammar/json_schema.rs`, ~600 LoC): JSON Schema subset → GBNF Grammar; supports `type` (all 7 types), `properties`+`required`, `enum`, `items`, `minimum`/`maximum`, `minLength`/`maxLength`, literal `pattern`; nested schemas promoted to named rules; 6 tests
- **`BeamSearchConfig { beam_width, max_new_tokens, length_penalty, early_stopping }`** and **`BeamHypothesis { tokens, logprob_sum, finished }`** (`beam_search.rs`, ~450 LoC): numerically stable log-softmax, global top-k pruning, length-penalty normalized scoring; **`InferenceEngine::beam_generate()`** convenience wrapper; 4 tests

#### Files Store + Run Steps + Run Streaming (`oxillama-server`)
- **`FilesStore`** (`src/files_store.rs`, ~300 LoC): atomic temp-rename writes; directory layout `{root}/{file_id}/{meta.json,data.bin}`; `FilePurpose` (assistants/batch/fine-tune); `create_with_limit()` for testable size limits; 8 unit tests
- **Five route handlers** (`src/routes/files.rs`): `POST /v1/files` (multipart, max 512 MiB), `GET /v1/files`, `GET /v1/files/:id`, `GET /v1/files/:id/content`, `DELETE /v1/files/:id`
- **`RunStep`** (`threads/types.rs`): `RunStepType` (MessageCreation / ToolCalls), `RunStepStatus`, `MessageCreationStepDetails`; stored at `runs/<run_id>/steps/<step_id>.json`
- **`ThreadStore` step methods** (`threads/store.rs`): `append_step()`, `list_steps()`, `get_step()`, `update_step_status()`; 4 unit tests
- **Step route handlers** (`threads/steps.rs`): `GET /v1/threads/:id/runs/:run_id/steps`, `GET /v1/threads/:id/runs/:run_id/steps/:step_id`; 4 integration tests
- **`RunEvent` SSE stream** (`threads/stream.rs`): `Created/InProgress/MessageDelta/Completed/Failed` events; `tokio::sync::broadcast` channel; `build_run_sse_stream()` via `stream::unfold`; activated when `CreateRunRequest.stream = true`; 5 unit tests
- **Worker emits steps**: `spawn_run_worker` creates `MessageCreation` step as `InProgress` before generation, marks `Completed` after
- **Error variants**: `FileNotFound` (404), `FileTooLarge` (413), `FileStoreError` (500), `RunStepNotFound` (404)
- **`AppState`** extended with `files_store: Option<Arc<FilesStore>>`, `run_event_tx_broadcast: Option<Arc<Sender<RunEvent>>>`

#### CLI Subcommands: quantize + convert + verify + tokenize (`oxillama-cli`)
- **`oxillama quantize <input.gguf> <output.gguf> --target <TYPE>`** (`src/quantize.rs`): re-quantizes GGUF tensors to Q4_0 or Q8_0 (K-quants refused with clear error); 2 tests
- **`oxillama convert <input.safetensors> <output.gguf>`** (`src/convert.rs`): wraps `SafetensorsConverter::from_bytes()` + writes synthesised GGUF; 2 tests
- **`oxillama verify <model.gguf> [--sha256 <hex>]`** (`src/verify.rs`): checks magic, version (1–3), parse, tensor bounds, optional SHA256; 4 tests
- **`oxillama tokenize`** / **`oxillama detokenize`** (`src/tokenize.rs`): encode text → token IDs; decode IDs → text; 2 tests

#### Power/Watt Benchmarks + CI Regression Gate (`oxillama-bench`)
- **`RaplReader`** (`src/power.rs`, ~260 LoC): scans `/sys/class/powercap/intel-rapl:<N>` (top-level domains); reads `energy_uj` + `max_energy_range_uj`; `compute_delta_uj()` handles wraparound; `measure_tokens_per_joule()` wrapper; `cfg(target_os = "linux")` gated with graceful `NoRapl` fallback; 12 tests
- **`RegressionGate`** (`src/regression_gate.rs`, ~280 LoC): `BaselineEntry { name, toks_per_sec, prefill_ms, decode_ms_p99 }`; hard-fails on metric regression above `threshold` (default 5%); skips new benchmarks not in baseline; `from_file()`/`save_baseline()` JSON I/O; `format_report()` Markdown table; 10 tests
- **Criterion bench target** (`benches/power.rs`): `StubEngine` + `RaplReader`; `OXILLAMA_BENCH_PRINT_POWER=1` env gate

#### Hub-Aware Snapshots + DLPack Interop (`oxillama-py`)
- **`HubOrigin { repo_id, filename, sha256 }`** field added to `EngineSnapshotMeta` (`src/snapshot.rs`): `restore()` re-downloads from hub if `model_path` is missing and `hub_origin` is set; SHA256 verified after download; `from_snapshot_with_hub()` classmethod; 5 Rust tests + 8 Python tests
- **`vec_to_dlpack()` / `dlpack_to_vec()`** (`src/dlpack.rs`, ~280 LoC): full DLPack v0.8 C struct layout (`DLDevice/kCPU`, `DLDataType/f32`, `DLTensor`, `DLManagedTensor`); `ManagedTensorState` owns `Vec<f32>`+`Vec<i64>` with `extern "C"` deleter; `PyCapsule` with name `"dltensor"`; 5 Rust tests + 8 Python tests
- **`PyEngine::logits_dlpack()`**, **`embeddings_dlpack()`** added to engine API; type stubs updated

#### LLaVA-1.6 / LLaVA-NeXT Anyres Tiling (`oxillama-arch`)
- **`AnyresTileConfig`** (`src/llava_next/tiler.rs`): `select_grid(img_w, img_h) -> (n_cols, n_rows)` via fill-fraction minimisation across `grid_pinpoints`; `split_into_tiles(pixels, img_w, img_h)` bilinear-resizes the image into a variable NxM tile grid plus a global-view thumbnail
- **`LlavaNextModel`** (`src/llava_next/model.rs`): reuses `ClipEncoder` + `MmProjector` from LLaVA-1.5; `encode_image()` splits → per-tile CLIP → concat → project; text-only `ForwardPass` fallback
- **`LlavaNextArchitecture`** registered under arch id `"llava16"` behind `llava16` feature (included in default features); arch count updated to 22
- 6 new tests: grid selection (2×2 pinpoint), tile count (4+1 thumbnail), tile dimensions, registry lookup, tensor names, clip-encode feature count

#### Remote GGUF HTTP Range + Safetensors Bridge (`oxillama-gguf`)
- **`HttpRangeSource`** (`src/http_source.rs`, ~300 LoC): `Source` trait implementation using `ureq 3.x` HTTP range requests (`Range: bytes=N-M`); 128 KiB warm cache for repeated small reads; `GgufModel::from_url(url)` entry point; gated behind `http` feature flag; network-dependent tests `#[ignore]`d
- **`SafetensorsConverter`** (`src/safetensors.rs`, ~350 LoC): `load(path)` and `from_bytes(bytes)` parse the 8-byte LE `header_size` prefix, UTF-8 JSON metadata, and raw tensor data; dtype map: `F32→F32`, `F16→F16`, `BF16→Bf16`, `I8→Q8_0`, others error with `UnsupportedDtype`; builds synthetic GGUF v3 byte buffer
- **`ureq = "3.3.0"`** added to workspace dependencies
- 7 new safetensors tests (header parse, dtype mapping, roundtrip, error cases)

#### IQ1_S / IQ1_M / IQ2_XS / IQ4_NL / TQ1_0 / TQ2_0 GPU Kernels (`oxillama-gpu`)
- **6 new GPU kernel files**: `iq1_s.rs`, `iq1_m.rs`, `iq2_xs.rs`, `iq4_nl.rs`, `tq1_0.rs`, `tq2_0.rs` — each implements inline CPU dequant + dispatch to `gemv_f32.wgsl`
- **`iq1s_grid/` split**: IQ1_S_GRID[2048] split across `iq1s_grid/{mod,data_a,data_b}.rs` to respect the 2000-line file limit
- GPU quant coverage increases from 18 → **24 types** (~95% of community HuggingFace uploads)
- 6 new tests (one per kernel: dequant output shape + finite values)

#### Python Native Async Engine (`oxillama-py`)
- **`AsyncEngine`** class (`python/oxillama_py/__init__.py`): pure-Python asyncio bridge using `ThreadPoolExecutor` + `asyncio.run_in_executor`; `async generate(prompt, max_tokens, temperature, ...) -> str` and `async stream(prompt, ...) -> AsyncIterator[str]` via `queue.Queue` + sentinel pattern
- **`PyEngine::async_engine()`**: Rust method returning an `AsyncEngine` instance wrapping `self`
- **Type stubs updated**: `__init__.pyi` extended with `AsyncEngine` class and `Engine.async_engine()` method
- **36 new Python tests** (`python/tests/test_async_engine.py`): coroutine/asyncgenfunction type checks, mock-engine functional tests, stream completion, exception propagation (32 pass, 4 skip without native extension)

#### Fused Dequant+GEMV for Q2_K / Q3_K (`oxillama-quant`)
- **`Q2_KAvx2::matvec_q8_fused`**: `fused_q2k_q8_0_row_avx2` unsafe fn; formula `(dl × Σ(q2_i × q8_i) − ml × Σ(q8_i)) × d_a` with 256-weight super-block (8 Q8_0 input blocks); eliminates scratch dequant buffer
- **`Q3_KAvx2::matvec_q8_fused`**: `fused_q3k_q8_0_row_avx2` unsafe fn; symmetric format (no `min` term), `dl × Σ(q3_i × q8_i) × d_a`
- **`Q2_KNeon::matvec_q8_fused`** and **`Q3_KNeon::matvec_q8_fused`**: ARM NEON paths via `vmull_s16` / `vmlal_s16` / `vaddvq_s32`
- **Reference scalar overrides**: `Q2_KRef` and `Q3_KRef` `matvec_q8_fused` overrides corrected to match 256-weight super-block layout (default trait impl was broken for 8-blocks-per-super-block formats)
- 8 new tests (2 AVX2 + 2 NEON + 2 reference correctness + 2 multi-row)

#### Latency-vs-Batch-Size Heatmap Bench (`oxillama-bench`)
- **`HeatmapPoint`** (`src/heatmap.rs`): `{ batch_size, seq_len, toks_per_sec, p99_latency_ms, memory_bytes }` with serde support
- **`BatchHeatmap`**: `run<E: PrefillDecodeBench>(engine, batch_sizes, seq_lens, label)` sweeps a 2-D grid; `summary_table()` (toks/s grid) + `p99_table()` (latency grid) Markdown output; `lookup(batch_size, seq_len)` point accessor
- **`default_batch_sizes()`**: `&[1, 2, 4, 8]`; **`default_seq_lens()`**: `&[128, 512, 1024, 2048]`
- **Criterion bench target** (`benches/batch_heatmap.rs`): `HeatmapStubEngine` + `BenchmarkId::new(format!("b{}", batch_size), seq_len)` naming; `OXILLAMA_BENCH_PRINT_HEATMAP=1` env gate prints tables to stdout
- 14 new unit tests (grid coverage, table headers, p99 unit label, lookup missing cell, monotonicity, error cases)

#### AVX-512 SIMD Completeness for Legacy Quant Types (`oxillama-quant`)
- **`Q4_1Avx512`** (`simd/avx512/q4_1.rs`, ~280 LoC): AVX-512F dequant + GEMV kernel for Q4_1 18-byte blocks (`d` f16, `m` f16, 8 nibble bytes); FMA path `result = d * nibble + m` using `_mm512_fmadd_ps`; 3 tests (dequant, 64×1024 GEMV, partial-block GEMV)
- **`Q5_1Avx512`** (`simd/avx512/q5_1.rs`, ~310 LoC): AVX-512F kernel for Q5_1 (high-bit array + low nibbles, unsigned 0–31 values with `m` bias instead of sign bias); 3 tests
- **`Q8_1Avx512`** (`simd/avx512/q8_1.rs`, ~250 LoC): AVX-512F kernel for Q8_1 36-byte blocks (offset +4 to qs array vs +2 in Q8_0, `s` precomputed partial sum at offset +2); 3 tests
- Dispatch table updated; all 11 legacy quant types now have a full four-tier SIMD ladder (AVX-512 → AVX2 → NEON → scalar)

#### Long-Context KV-Cache Scaling Bench (`oxillama-bench`)
- **`LongContextSweep` / `LongContextPoint`** (`src/long_context.rs`, ~220 LoC): helper structs wrapping `run_kv_cache_scaling` across a configurable context sweep; `summary_table()` emits a Markdown table with columns `ctx_len | decode tok/s | memory MiB | prefill ms`
- **`default_ctx_lengths()`**: returns `&[1024, 4096, 8192, 16384, 32768]`
- **Criterion bench target** (`benches/long_context.rs`): sweeps with `LongContextStubEngine` simulating linear KV-read cost growth; `BenchmarkId::new("ctx", ctx_len)` naming; `OXILLAMA_BENCH_PRINT_TABLE=1` env gate prints Markdown summary to stdout; 3 unit tests

#### OpenAI Assistants API Subset (`oxillama-server`)
- **Thread persistence** (`src/threads/store.rs`, ~400 LoC): `ThreadStore` with atomic writes via temp-file + rename; directory layout `{root}/{thread_id}/{meta.json, messages.jsonl, runs/{run_id}/status.json}`
- **Run worker** (`src/threads/worker.rs`, ~250 LoC): `spawn_run_worker` drains the run queue, formats thread messages as chat prompt, dispatches `BatchRequest::Generate`, appends assistant message, transitions `queued → in_progress → completed` (or `failed`)
- **Seven route handlers** (`src/threads/routes.rs`): `POST/GET /v1/threads`, `POST/GET /v1/threads/:id/messages`, `POST/GET /v1/threads/:id/runs`, `GET /v1/threads/:id/runs/:run_id`, `POST /v1/threads/:id/runs/:run_id/cancel`
- **OpenAI v2 serde types** (`src/threads/types.rs`): `Thread`, `ThreadMessage`, `Run`, `RunStatus` (Queued/InProgress/Completed/Cancelled/Failed/Expired), `MessageRole`, `ContentBlock`, `TextContent`, `RunError`
- **`AppState` extensions**: `threads_store: Option<Arc<ThreadStore>>`, `run_queue_tx: Option<RunQueueSender>`, `with_threads()` builder mirroring `with_batch_pipeline()`
- **Error variants**: `ThreadNotFound`, `RunNotFound`, `RunInTerminalState` → HTTP 404/409
- 12 new integration tests including persistence-across-restart and atomic-write-no-partial-state

#### Qwen2-VL Multimodal Architecture with M-RoPE (`oxillama-arch`)
- **`MRopeTable`** (`src/common/mrope.rs`, ~270 LoC): three-axis (time/height/width) cos/sin tables partitioning `head_dim` into thirds; `apply_mrope(x, t_pos, h_pos, w_pos)` for per-head rotation; text-only tokens use `(pos, pos, pos)` yielding three independent per-axis RoPE rotations
- **Qwen2-VL vision encoder** (`src/qwen2_vl/vision.rs`, ~250 LoC): native ViT with patch size 14, no CLS token, 2D spatial RoPE, dynamic resolution (native aspect ratio + window attention of 8×8 patches), outputs one feature vector per patch
- **`MmMerger`**: 2×2 spatial patch → 1 LLM token compression via reshape + linear projection
- **`Qwen2VlModel`** (`src/qwen2_vl/model.rs`, ~700 LoC): full forward pass for multimodal + text-only inputs; M-RoPE applied per layer
- **`Qwen2VlArchitecture`** registered under arch id `"qwen2vl"` behind `qwen2-vl` feature (included in default features); arch count updated to 21
- **`ModelConfig` extensions**: `vision_config: Option<VisionConfig>`, `rope_dimensions: Option<[usize; 3]>` for M-RoPE axis split
- **`build_minimal_qwen2vl_gguf()`** test fixture in `oxillama-gguf`
- 8 new tests (M-RoPE axis independence, tensor name coverage, forward shape, dynamic resolution, MM merger compression, registry lookup, text-only fallback)

#### SpeculativeEngine Snapshot/Restore (`oxillama-runtime` + `oxillama-py`)
- **`SpeculativeEngineSnapshot`** (`src/snapshot.rs`, +~280 LoC): magic `b"OXISPEC1"`, wraps `target_snapshot + draft_snapshot + num_speculative + spec_seed + accepted_tokens + rng_state`; `encode()`/`decode()`/`fingerprint()` methods
- **`SpeculativeEngine::snapshot()`**, `snapshot_to_file()`, `resume()`, `resume_from_file()`: full snapshot/restore cycle reusing per-engine `InferenceEngine::snapshot()`
- **`RuntimeError::SpecSnapshotIncompatible`** variant for magic/version mismatch
- **Python bindings**: `PySpeculativeEngine::snapshot(path)`, `snapshot_bytes()`, `restore()` classmethod, pickle-compatible `__reduce__` returning `(restore, (path, target, draft))` tuple (replaces prior pickle-refusal hook)
- **Type stubs updated**: `__init__.pyi` extended with `snapshot`, `snapshot_bytes`, `restore` signatures
- **`python/tests/test_speculative_snapshot.py`**: 13 pure-Python method-existence tests + 3 model-gated integration tests
- 4 Rust tests (roundtrip, wrong-magic rejection, truncated rejection, accepted-history preservation)

#### WASM SIMD128 Diagnostics + Service-Worker Model Cache (`oxillama-wasm`)
- **`getSimd128Status()`** (`src/simd_check.rs`, ~80 LoC): JS function returning `{ compiled_with, runtime_detected, user_agent }`; `compiled_with` / `runtime_detected` resolved at compile time via `cfg!(target_feature = "simd128")`
- **`getServiceWorkerScript(options_json)`** (`src/service_worker.rs`, ~280 LoC): generates a self-contained cache-first JS service worker script string intercepting `/models/*.gguf` fetches from the Cache Storage API; `ServiceWorkerOptions { gguf_path_prefix, cache_name }` with serde roundtrip
- **`registerServiceWorker(script_url)`**: calls `navigator.serviceWorker.register()` via `js_sys::Reflect`, returns `js_sys::Promise`
- **`examples/service_worker_demo.html`**: demo page showing registration + SIMD status UI
- **`tests/mobile_matrix_doc.md`**: manual test matrix for iOS Safari 17+, Android Chrome 121+, Firefox 122+
- 7 new tests (serde roundtrip, default cache name, script identifier presence, invalid JSON rejection, SIMD compiled_with type, struct defaults, feature-gated SIMD assertion)

#### Server Prefix-KV Cache Wiring (`oxillama-server` + `oxillama-runtime`)
- **`InferenceEngine::prime_with_prefix`**: restores KV cache from a `CachedKvState` snapshot then forward-passes suffix tokens, returning initial logits for the decode loop — skips re-prefilling shared system prompts on cache hits
- **`InferenceEngine::generate_with_logits`**: decode-only loop starting from pre-computed initial logits; used after a prefix-cache hit to avoid a second prefill pass
- **`InferenceEngine::store_kv_in_prefix_cache`**: public helper that snapshots current KV state into a `PrefixKvCache` without exposing the mutable KV reference across crate boundaries
- **`CachedKvState::new`**: public constructor enabling reconstruction from cloned K/V buffers after the `Mutex` guard is released
- **Server-side `PrefixKvCache` wiring**: `AppState` now holds `prefix_cache: Arc<Mutex<PrefixKvCache>>`; the worker thread looks up the longest matching token prefix, restores KV state on hit, generates via `generate_with_logits`, and stores the post-generation KV state; full-prefill fallback on miss
- **Per-request `cache_prompt` flag**: new `cache_prompt: bool` field on `ChatCompletionRequest` (default `true`) and `BatchRequest::Generate` / `GenerateStream`; setting `false` disables caching for that request
- **Type aliases for complex worker types**: `PrefixHitData` and `WorkerHandles` suppress clippy "very complex type" lint without losing information

#### Server Multi-LoRA Registry + Per-Request Adapter Selection (`oxillama-server` + `oxillama-arch`)
- **`InferenceEngine::unapply_all_loras`**: clears `lora_stack` and calls `ForwardPass::unapply_all_loras()` — properly reverts all `QuantLinear.lora` fields set by `apply_lora_stack()`, restoring the base model weights for the next request
- **`ForwardPass::unapply_all_loras`**: new default no-op trait method; overridden in LLaMA, LLaVA, and Command-R model impls to iterate all linear layers and call `clear_lora()`
- **`AppState::loras` registry**: `loras: Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>` field on `AppState`; `spawn_inference_worker` accepts the registry Arc and resolves adapter names inside the worker
- **Per-request LoRA selection**: new `lora_selection: Vec<(String, f32)>` on `BatchRequest::Generate` / `GenerateStream`; chat route parses `"lora": "name"` (string form with scale 1.0) or `"lora": [{"name": "...", "scale": 0.8}]` (array form), resolves names → 400 on unknown
- **Admin LoRA CRUD endpoints**: `POST /admin/loras` (load GGUF + register), `DELETE /admin/loras/{name}` (unregister, 404 if not found), `GET /admin/loras` (list registered names); backed by `admin/loras.rs` (~220 LoC, 5 tests)
- **`LoraSelection` public type alias**: re-exported from `oxillama_server` for use in downstream crates

#### AVX-512 K-Quant Kernels (`oxillama-quant`)
- **`Q2_KAvx512`** (`simd/avx512/q2_k.rs`, ~700 LoC): full AVX-512F dequant + GEMV kernel for Q2_K super-block format (84 bytes, 256 weights, 2-bit packed with scale-of-scales); 2× wider lanes vs the AVX2 path via `_mm512_*` intrinsics; registered in `dispatch.rs`
- **`Q3_KAvx512`** (`simd/avx512/q3_k.rs`, ~830 LoC): full AVX-512F dequant + GEMV kernel for Q3_K (110 bytes, 3-bit packed from 32-byte `qs` + 8-byte `hmask`, 6-bit scale array); signed [-4..3] weight reconstruction; registered in `dispatch.rs`
- AVX-512 coverage table now includes Q2_K and Q3_K (previously scalar + AVX2 only)

#### GPU Legacy Quad Kernels (`oxillama-gpu`)
- **`Q4_1GpuKernel`** (`kernels/q4_1.rs`, ~370 LoC): WGSL GEMV kernel for Q4_1 blocks (20 bytes: 2-byte `d` + 2-byte `m` + 16 nibble bytes); registered in `GpuDispatcher`
- **`Q5_0GpuKernel`** (`kernels/q5_0.rs`, ~410 LoC): WGSL GEMV kernel for Q5_0 blocks (22 bytes: 2-byte `d` + 4-byte `qh` high bits + 16-byte `qs`); 5-bit unpacking with separate high-bit array
- **`Q5_1GpuKernel`** (`kernels/q5_1.rs`, ~430 LoC): WGSL GEMV kernel for Q5_1 blocks (24 bytes: 2-byte `d` + 2-byte `m` + 4-byte `qh` + 16-byte `qs`)
- **`Q8_1GpuKernel`** (`kernels/q8_1.rs`, ~360 LoC): WGSL GEMV kernel for Q8_1 blocks (36 bytes: 2-byte `d` + 2-byte `sum` + 32 signed-byte `qs`)
- GPU dispatcher now covers 18 quantization types (was 14); Q4_1/Q5_0/Q5_1/Q8_1 cover ~85% of community-quantized HuggingFace GGUF uploads

### Quality
- **2,235 tests passing**, up from 1,825 in v0.1.2 (+410 tests)
- **0 warnings** maintained (`cargo clippy --workspace -- -D warnings`)

## [0.1.2] - 2026-04-25

### Added

#### Session Persistence (`oxillama-runtime`)
- **Conversation save/resume** (`session.rs`): `Session::save()` and `Session::load()` serialize the full conversation history via `oxicode` (COOLJAPAN Pure Rust codec); SHA-256 KV sidecar validates integrity on load; schema version guard rejects incompatible session files
- **`/save` and `/load` slash commands**: interactive CLI commands to persist and restore conversation sessions across process restarts

#### HuggingFace Hub Integration (`oxillama-cli`)
- **`oxillama hub pull/list/rm` subcommands**: download models directly from HuggingFace Hub (`hf-hub 0.5`), list cached models, and remove cached entries; uses `ureq` with `rustls` for Pure Rust TLS and the `directories` crate for platform-appropriate cache paths

#### TUI Chat Mode (`oxillama-cli`)
- **Full-screen TUI chat** (`ratatui 0.30` + `crossterm 0.29`): scrollable chat history, input line, live streaming via `spawn_blocking` + `mpsc` channel for async token delivery without blocking the TUI event loop; 6 unit tests covering layout, input handling, and message rendering

#### New Architecture Loaders (`oxillama-arch`)
- **DBRX GGUF loader**: support for Databricks DBRX mixture-of-experts architecture
- **Grok-1 GGUF loader**: support for xAI Grok-1 architecture
- **Mamba-2 GGUF loader**: support for state-space model architecture with `embed()` override for Mamba-2-specific token embedding logic

#### KV Cache API Extensions (`oxillama-arch`)
- **`KvCacheAccess` trait extensions**: new `kv_dim()`, `for_each_key()`, and `for_each_value()` methods on the `KvCacheAccess` trait; `PagedKvCache` fully implements multi-page support for all three methods
- **`BatchedKvView` + `KvSlot`**: moved to `oxillama-arch/traits.rs` for cross-crate reuse
- **`ForwardPass::forward_batched`**: default implementation added to the trait; LLaMA provides a concrete optimized implementation

#### Quality
- **2,020 tests passing**, up from 1,979 in v0.1.1

## [0.1.1] - 2026-04-24

### Added

#### GGUF Loader Hardening (`oxillama-gguf`)
- **Partial-download resume** (`resume.rs`): `GgufModel::resume()` reads an adjacent `.oxiresume` sidecar checkpoint, validates the last-valid byte offset, and provides a `ResumeHandle::finish()` path once the download completes — survives interrupted HuggingFace pulls without re-downloading
- **Sharded multi-file loading** (`sharded.rs`): `ShardedGgufModel::load_sharded()` auto-discovers all HuggingFace-named sibling shards (`<base>-NNNNN-of-MMMMM.gguf`) from a single shard path and presents a unified logical model
- **Quantize-on-the-fly** (`quantize_on_load.rs`): optional pass that dequantizes and re-quantizes tensors to a target format during load, eliminating a separate conversion step for deployment

#### Runtime Snapshot/Resume (`oxillama-runtime`)
- **`EngineSnapshot`** (`snapshot.rs`): captures the full KV-cache and sampler RNG state into a byte blob via `InferenceEngine::snapshot()`; `InferenceEngine::resume()` validates the model fingerprint and restores from the blob, enabling session persistence across process restarts
- **Oxicode serialization**: `EngineSnapshot` is serialized with `oxicode` (COOLJAPAN Pure Rust codec) rather than `bincode`, in compliance with the workspace serialization policy

#### Facade Examples & Cookbook (`oxillama`)
- **`examples/load_and_generate.rs`**: end-to-end example: load a GGUF, configure the sampler, and stream tokens to stdout
- **`examples/lora_apply.rs`**: demonstrates hot-swapping two LoRA adapters on a running engine without model reload
- **`examples/speculative.rs`**: shows the `SpeculativeEngine` API with a 1B draft model and 70B target
- **`RECIPES.md`**: 8-recipe task-oriented cookbook covering generation, serving, LoRA, speculative decoding, snapshot/resume, WASM browser chat, partial-download resume, and sharded model loading

#### Quantization (`oxillama-quant`)
- **AVX2 kernels**: Q4_1, Q5_0, Q5_1, Q8_1 — 4 new legacy-quant AVX2 dot-product kernels, completing full AVX2 coverage for all legacy quantization types
- **NEON kernels**: Q4_1, Q5_0, Q5_1, Q8_1, Q2_K, Q3_K — 6 new Apple Silicon NEON kernels; combined with IQ/TQ additions gives near-complete NEON coverage across all quantization families
- **NEON AArch64 kernels**: IQ2_XXS, IQ2_XS, IQ3_S, IQ4_XS, IQ4_NL, TQ1_0, TQ2_0, IQ1_S, IQ1_M, IQ3_XXS, IQ2_S — all 11 IQ types now have Apple Silicon NEON acceleration
- **AVX-512 kernels**: TQ1_0, TQ2_0, Q5_0, Q8_K — AVX-512 coverage extended to 10 types

#### GPU Backend (`oxillama-gpu`)
- **Q2_K GEMV shader**: WGSL compute shader for Q2_K dequant + dot-product on GPU
- **Q3_K GEMV shader**: WGSL compute shader for Q3_K dequant + dot-product on GPU
- **Q8_K GEMV shader**: WGSL compute shader for Q8_K dequant + dot-product on GPU
- **IQ4_XS GEMV shader**: WGSL compute shader for IQ4_XS dequant + dot-product on GPU
- **Async WebGPU bridge** (`gpu_bridge.rs`): `initWebGpuDevice()`, `webgpuDequantQ4_0Async()`, `webgpuGemvAsync()` using `wasm_bindgen_futures::JsFuture` for real GPU dispatch in browsers with WebGPU support

#### Architectures (`oxillama-arch`)
- **Multi-head Latent Attention (MLA)**: Low-rank KV compression primitive (`MlaLayer`) with decoupled RoPE — reduces KV-cache memory footprint by up to 93% vs standard MHA
- **DeepSeek-V2 architecture**: Full `DeepSeekV2Model` with MLA attention, DeepSeekMoE sparse FFN routing (N shared experts + top-K routed experts), 3-bit/8-bit quantized expert dispatch

#### Developer Experience
- **oxillama (meta)**: `openai_server.rs` example showing programmatic server startup; `python_bridge.rs` example documenting Rust↔Python API parity
- **oxillama-cli**: Colorized output via `colored` (cyan/bold key labels, green banners); `indicatif` spinner progress bar during model loading; 5 integration smoke tests in `tests/cli_smoke.rs`

## [0.1.0] - 2026-04-15

### Added

#### Core
- GGUF v3 binary format parser (`oxillama-gguf`): magic, version, KV metadata, tensor info, mmap loading
- 25 quantization types (`oxillama-quant`): F32/F16/BF16 pass-through; legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1; K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K; I-quants IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS; Q1_0_G128
- SIMD-accelerated kernels: AVX-512 → AVX2 → NEON → scalar dispatch for Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q1_0_G128

#### Architectures (`oxillama-arch`)
- LLaMA 2/3/4 with GQA + RoPE
- Qwen3 with attention bias
- Mistral with sliding-window attention
- Gemma 2/3 with GeGLU, post-norm, logit soft-capping
- Phi-3/4 with merged QKV and partial RoPE
- StarCoder (GPT-BigCode) with MQA, LayerNorm, GELU, absolute position embeddings
- Command-R/R+ with logit scaling and optional Q/K norms
- Mixtral-MoE: LLaMA extended with sparse MoE FFN routing (Mixtral-7B/8x7B compatible)
- LLaVA-1.5 multimodal: full CLIP ViT-L/14 encoder (patch extraction, CLS token, position embeddings, N transformer layers, post-LN) + MmProjector 2-layer MLP

#### Runtime (`oxillama-runtime`)
- Paged KV cache with block-level memory management
- Sampling: greedy argmax, top-K, top-P (nucleus), temperature scaling, min-P, mirostat-v2, repetition penalty, seeded RNG
- GBNF grammar-constrained sampling
- Rayon row-parallel GEMV (feature-gated, scalar fallback for WASM)
- Continuous batching (`BatchRequest` queue + worker task; KV-cache reset between requests)
- Speculative decoding (`SpeculativeEngine`): draft/target model pair, token-level accept/reject, KV-cache resync
- LoRA adapter loading: `LoadedLora` from GGUF, `apply_lora()` runtime API, `QuantLinear` LoRA field

#### Server (`oxillama-server`)
- OpenAI-compatible HTTP API: `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`
- SSE streaming with `data:` lines and `[DONE]` sentinel
- llama.cpp CLI flag aliases: `-n/--n-predict`, `--temperature`, `-c/--n-ctx`, `--seed`, `--repeat-penalty`, `--min-p`
- `bench` subcommand for throughput measurement

#### Bindings
- Python bindings (`oxillama-py`): `Engine`, `SpeculativeEngine`, `LoadedLora` via PyO3 0.24; GIL-releasing inference; streaming callback; maturin wheel
- WebAssembly bindings (`oxillama-wasm`): `InferenceEngine::load_model_from_bytes()`, GGUF parsing exposed via wasm-bindgen

#### GPU Backend (`oxillama-gpu`)
- wgpu 29.0.1 compute backend (feature-gated `gpu = ["dep:wgpu"]`, off by default)
- Q4_0 + Q8_0 WGSL f32 GEMV shaders; `GpuDispatcher::try_init()` with graceful CPU fallback

#### Quality
- 1,205 tests, 0 warnings
- 87%+ test coverage (region/function/line)
- 3 cargo-fuzz targets for GGUF parser
- Criterion benchmarks: all 25 quant types + sampling pipeline

#### Project Structure
- `oxillama` meta crate: unified re-export of all subcrates (`oxillama::gguf`, `oxillama::quant`, etc.)
- `oxillama-cli` binary crate: CLI moved from workspace root to `crates/oxillama-cli/`

[0.1.4]: https://github.com/cool-japan/oxillama/releases/tag/v0.1.4
[0.1.1]: https://github.com/cool-japan/oxillama/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxillama/releases/tag/v0.1.0

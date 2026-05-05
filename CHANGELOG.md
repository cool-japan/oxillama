# Changelog

All notable changes to OxiLLaMa are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
OxiLLaMa uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[0.1.1]: https://github.com/cool-japan/oxillama/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxillama/releases/tag/v0.1.0

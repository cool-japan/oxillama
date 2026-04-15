# oxillama-bench — TODO

## 1. Overview

Benchmark suite for the OxiLLaMa inference engine. Provides a Criterion harness
plus reusable throughput, latency, and memory-measurement helpers that every
downstream crate can plug into. Feature-gated (not part of the default
workspace build) so that the heavy Criterion dependency tree stays opt-in.

Dependency role: terminal leaf — consumes `oxillama-runtime`, `oxillama-quant`,
and `oxillama-gguf`, but nothing depends on it. Runs as `cargo bench` or as a
standalone binary target; never linked into production builds.

## 2. Status Snapshot

| Field | Value |
|---|---|
| Version | 0.1.0 |
| Completion | ~78% |
| src files | 7 (`lib.rs`, `latency.rs`, `throughput.rs`, `memory.rs`, `e2e.rs`, `prefill_decode.rs`, `arch_config.rs`) |
| Bench targets | kernel-level (quant dequant/GEMV/GEMM, sampling) |
| Criterion version | workspace-pinned (latest) |
| Pure Rust | yes (no C/FFI in bench harness) |
| Default feature | off (opt-in `bench` flag at workspace root) |

Completion rationale (78%): kernel-level micro-bench coverage is thorough and
stable, macOS RSS, end-to-end harness, prefill/decode split, and per-architecture
configurations now ship, but cross-SIMD, KV-cache scaling, and batched-inference
breadth remain absent.

## 3. Module Map

| File | Responsibility |
|---|---|
| `src/lib.rs` | Crate root; re-exports the three public helper APIs (latency, memory, throughput) behind a flat surface. |
| `src/latency.rs` | Per-token and time-to-first-token timers; percentile aggregation (P50/P95/P99) via `LatencyTimer`, `LatencyConfig`, `LatencyResult`, `TokenLatencyResult`. |
| `src/throughput.rs` | Sustained tokens-per-second measurement with warm-up / measurement windowing; FLOP/s attachment; `ThroughputTracker`, `TrackerConfig`, `TokenThroughputResult`, `aggregate_throughput`. |
| `src/memory.rs` | Cross-platform RSS sampling (`/proc/self/status` on Linux; `ps` on macOS); model-weight and KV-cache byte estimators; `RssTracker`, `MemoryEstimate`. |
| `src/prefill_decode.rs` | Prefill/decode split benchmarking (`PrefillDecodeBench` trait, `run_prefill_decode_bench`, P95 calculation, formatted summary table). |
| `src/arch_config.rs` | Architecture-specific bench configurations (LLaMA-3, Qwen3, Mistral, Gemma, Phi — `from_name`, `known_architectures`, conversion to `PrefillDecodeConfig`/`E2eBenchConfig`). |
| `src/e2e.rs` | End-to-end benchmark harness (`InferenceBenchmark` trait, `run_e2e_bench()`). |

## 4. Shipped in v0.1.0

- Criterion harness wired into every quant kernel: dequant + GEMV + GEMM for
  all 25 supported quant types (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q1_0_G128, Q8_K,
  and the remaining K-/IQ-family variants exposed by `oxillama-quant`).
- Sampling-pipeline benchmarks (7 bench functions): greedy, top-k, top-p,
  temperature, repeat-penalty, typical, and combined-chain samplers.
- Throughput helper: warm-up + measurement windows, token-rate accumulation,
  optional FLOP/s attachment, aggregate over multiple runs.
- Latency helper: warm-up iterations, percentile extraction (P50/P95/P99),
  per-token and time-to-first-token variants.
- Memory helper: RSS snapshot (Linux via `/proc/self/status`), model-weight
  byte estimator for fractional bit-widths (e.g. 4.5 bpw for Q4_K), KV-cache
  byte estimator (`n_layers × n_heads × head_dim × ctx × 2 × 2`).
- Pure-Rust, zero-FFI implementation of all three trackers; compatible with
  every target `oxillama-runtime` supports.
- macOS RSS reading via `ps` (pure Rust, no FFI).
- End-to-end benchmark harness (`InferenceBenchmark` trait, `run_e2e_bench()`,
  configurable warmup/measure).
- Prefill/decode split benchmarking: `PrefillDecodeBench` trait,
  `run_prefill_decode_bench()`, P95 calculation, formatted summary table.
- Architecture-specific bench configurations: LLaMA-3, Qwen3, Mistral, Gemma,
  Phi — `from_name()`, `known_architectures()`, conversion to
  `PrefillDecodeConfig` / `E2eBenchConfig`.

## 5. Known Gaps / Incomplete

The 35%-gap is breadth, not depth. Kernel micro-benches are solid; the
missing portion is comparative and per-architecture measurement.

- ~~**No prefill / decode split.**~~ ✅ Shipped: `PrefillDecodeBench` trait,
  `run_prefill_decode_bench()`, P95 calculation, formatted summary table.
  Prefill (prompt-processing) and decode (autoregressive) throughput are now
  reported separately.
- ~~**No per-architecture benchmark.**~~ ✅ Shipped: architecture-specific bench
  configurations for LLaMA-3, Qwen3, Mistral, Gemma, and Phi via `from_name()`,
  `known_architectures()`, and conversion to `PrefillDecodeConfig` /
  `E2eBenchConfig`. Architecture-specific regressions are now visible.
- **No cross-SIMD dispatch comparison.** The harness never forces a specific
  backend, so there is no AVX-512 vs AVX2 vs NEON vs scalar comparison table.
- **No KV-cache scaling curve.** No benchmark sweeps context length
  (1K / 4K / 8K / 32K) to expose the O(N) decode cost of growing KV state.
- **No memory-fragmentation profile.** `MemoryResult` is a point-in-time
  snapshot; there is no time-series of allocator behaviour under load.
- **No `criterion --save-baseline` discipline** — regressions between commits
  have to be caught by hand; no baseline is checked into the repo and no CI
  job enforces a comparison threshold.
- **No batched-inference benchmark.** Single-request decode is covered;
  many-concurrent-request throughput (the metric that matters for the
  server crate) is not measured.
- **No tokenizer throughput bench.** BPE encode/decode rates are not
  profiled even though they set an upper bound on prefill latency for
  short prompts.

## 6. v1.1 Roadmap

1. **End-to-end tokens/sec benches.** Add `benches/end_to_end.rs` that runs
   `InferenceEngine::generate` for each (architecture × quant × context)
   combination in the matrix below:

   | Architecture | Sizes | Quant types |
   |---|---|---|
   | LLaMA-3 | 8B | Q4_K_M, Q8_0 |
   | Qwen3 | 7B | Q4_K_M, Q8_0 |
   | Mistral | 7B | Q4_K_M, Q8_0 |

2. **Prefill vs decode isolation.** Report prefill `tok/s` (batch GEMM) and
   decode `tok/s` (per-token GEMV) as separate Criterion groups.
3. **Cross-SIMD table.** Add a bench that uses a runtime-dispatch override
   (env var or `OXI_SIMD_FORCE=avx2|avx512|neon|scalar`) to produce a
   side-by-side comparison table for every SIMD-backed quant type.
4. **Long-context curves.** Sweep `ctx ∈ {4K, 8K, 32K}` and plot decode
   `tok/s` as the KV cache grows.
5. **Memory-profiling module.** Add `src/profile.rs` with a time-series
   sampler (`Vec<(Instant, usize)>`) and peak / P95 aggregation, plus a
   markdown-table report writer.
6. **CI hook.** Wire `criterion --save-baseline master` into a scheduled CI
   job so regressions get surfaced on PRs without manual baselining.

All code examples in this crate must remain `unwrap`-free — prefer
`ok_or_else(|| BenchError::...)` and `?`. No deviation from the No-Unwrap
policy in benchmark harnesses either, even though they are not strictly
"production code".

## 7. v2.0+ Vision

- **Power / watt benchmarks.** Report `tokens / joule` via RAPL on Linux
  (Intel `/sys/class/powercap/intel-rapl:*`) and IOKit on macOS (when a
  Pure-Rust binding is available). Critical metric for edge deployment.
- **Latency-vs-batch-size heatmap.** 2-D Criterion plot sweeping
  `batch_size × seq_len` to expose the point where continuous batching stops
  paying off; rendered as an SVG heatmap per architecture.
- **CI regression gate.** Hard fail (not warn) when `tok/s` regresses by
  more than a configurable threshold against the saved baseline; block the
  merge queue until the regression is explained or reverted.
- **Flame-graph integration.** Optional `flame` feature that emits
  per-benchmark `*.folded` + SVG, using a Pure-Rust `pprof` replacement (or
  gating this feature when no Pure-Rust sampler is available).
- **Comparative benches vs llama.cpp.** External subprocess harness that
  invokes a pinned `llama-cli` build on the same GGUF and quant and reports
  the relative `tok/s`; feature-gated off by default to keep the default
  bench run hermetic and C-free.
- **Multi-node / distributed bench.** Measure throughput when a single
  generation is tensor-parallel-sharded across two hosts (aligns with the
  v2.0 runtime roadmap).
- **Regression history dashboard.** Persist criterion JSON to a small
  on-repo time-series and render a rolling-window plot per metric.

*Last updated: 2026-04-15 (v0.1.0 release)*

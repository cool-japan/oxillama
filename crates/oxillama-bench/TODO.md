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
| Version | 0.1.1 |
| Tests | 79 passing |
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
- **No cross-SIMD dispatch comparison.** ~~The harness never forces a specific
  backend, so there is no AVX-512 vs AVX2 vs NEON vs scalar comparison table.~~
  ✅ Shipped: `run_dequant_comparison`, `run_gemv_comparison`, and
  `format_comparison_table` in `simd_comparison.rs`; Criterion bench binary at
  `benches/cross_simd.rs` with `bench_cross_simd` group.
- **No KV-cache scaling curve.** ~~No benchmark sweeps context length
  (1K / 4K / 8K / 32K) to expose the O(N) decode cost of growing KV state.~~
  ✅ Shipped: `run_kv_cache_scaling`, `KvCacheScalingConfig`, `KvCacheScalingResult`,
  `KvCacheScalingPoint` in `prefill_decode.rs`; Criterion bench at `benches/kv_cache.rs`.
  Also added `run_prefill_vs_decode_isolation` for prefill/decode ratio reporting.
- **No memory-fragmentation profile.** ~~`MemoryResult` is a point-in-time
  snapshot; there is no time-series of allocator behaviour under load.~~
  ✅ Shipped: `MemoryProfiler` with rolling-window RSS sampling and `MemoryReport`
  (baseline, peak, P50, P99, sample count) in `memory.rs`; Criterion bench at
  `benches/memory_profile.rs`.
- **No tokenizer throughput bench.** ~~BPE encode/decode rates are not
  profiled even though they set an upper bound on prefill latency for
  short prompts.~~
  ✅ Shipped: `TokenizerBench` trait, `StubBpeTokenizer`, `bench_tokenizer_encode`,
  `bench_tokenizer_decode`, `TokenizerThroughputResult`, `TOKENIZER_SAMPLE_TEXT`
  in `throughput.rs`; Criterion bench at `benches/tokenizer.rs`.
- ~~**No E2E tokens/sec bench binary.**~~ ✅ Shipped (stub harness): `benches/end_to_end.rs`
  with Criterion groups for LLaMA-3, Qwen3, and Mistral using a synthetic
  `StubEngine`; real model support gated on a future GGUF cache in CI.

## 6. v1.1 Roadmap

1. ~~**End-to-end tokens/sec benches.**~~ ✅ Shipped: `benches/end_to_end.rs` with
   Criterion groups for LLaMA-3, Qwen3, and Mistral (stub `StubEngine`; real GGUF
   support pending model cache in CI).

2. **Prefill vs decode isolation.** Report prefill `tok/s` (batch GEMM) and
   decode `tok/s` (per-token GEMV) as separate Criterion groups.
- [x] **E1 — Cross-SIMD dispatch matrix + memory profiling (planned 2026-04-20)**
  - **Goal:** Benchmarks measure all SIMD paths (scalar / AVX2 / AVX512 / NEON) of every shipped quant kernel in a single matrix-style report. Memory profiling module records peak RSS, KV-cache occupancy, and weight memory at fixed sample intervals during inference.
  - **Design:**
    - New module `crates/oxillama-bench/src/dispatch_matrix.rs`:
      - For each `(quant_type, simd_path)` combo, runs `matvec_q8` and `matvec_q8_fused` on a 4096×4096 GEMV; reports tokens/s and µs/iter.
      - Result table written to CSV at `target/bench-dispatch.csv`.
      - Selectable via `cargo bench --bench dispatch_matrix --features simd-avx2` etc.
    - New module `crates/oxillama-bench/src/memory_profiler.rs`:
      - `MemoryProfiler::start(interval_ms)` spawns a tokio task that samples process RSS via `sysinfo` (Pure Rust, cross-platform) at the interval.
      - Inference engine emits events (via existing tracing spans) on `kv_cache_alloc`, `kv_cache_free`, `state_alloc`, `state_free`; profiler captures and aggregates.
      - Output: JSON at `target/bench-memory-<timestamp>.json`; ASCII table summary via `tabled` crate (Pure Rust).
    - Bench harness: criterion bench targets in `benches/dispatch_matrix.rs` and `benches/memory.rs`.
  - **Files:** `crates/oxillama-bench/src/dispatch_matrix.rs` (new, ~400 LoC); `crates/oxillama-bench/src/memory_profiler.rs` (new, ~500 LoC); `benches/dispatch_matrix.rs`, `benches/memory.rs` (new); `crates/oxillama-bench/Cargo.toml` (add `sysinfo`, `tabled` from crates.io latest).
  - **Prerequisites:** none.
  - **Tests:** (a) `dispatch_matrix_runs_all_paths` — CSV contains expected number of rows. (b) `memory_profiler_captures_baseline` — RSS bump within 10% of expected after ~100MB allocation. (c) `memory_events_correlate_with_kv_alloc` — KV slot allocation appears in profiler output.
  - **Risk:** `sysinfo` RSS reporting can be coarse on macOS (~1 MB granularity). Document. Report bench results alongside cpuinfo for context.

4. **Long-context curves.** Sweep `ctx ∈ {4K, 8K, 32K}` and plot decode
   `tok/s` as the KV cache grows.
5. ~~**Memory-profiling module.**~~ Promoted to [x] E1 above (combined with Cross-SIMD dispatch matrix).
6. ~~**CI hook.**~~ ✅ Shipped: `.github/workflows/bench_ci.yml` — weekly Monday
   schedule + `workflow_dispatch`; runs `--test` for compile/sanity, then
   `--save-baseline master`, and uploads `target/criterion/` as an artifact.

All code examples in this crate must remain `unwrap`-free — prefer
`ok_or_else(|| BenchError::...)` and `?`. No deviation from the No-Unwrap
policy in benchmark harnesses either, even though they are not strictly
"production code".

## 7. v2.0+ Vision

- ~~**Power / watt benchmarks.**~~ ✅ Shipped (v0.1.6): `RaplReader` in
  `src/power.rs` reads `/sys/class/powercap/intel-rapl:*` energy counters on
  Linux; `measure_tokens_per_joule` wraps any closure; `compute_tokens_per_joule_from_delta`
  is exposed for unit-testable formula verification.  Criterion bench at
  `benches/power.rs` with `OXILLAMA_BENCH_PRINT_POWER=1` for optional output.
  Gracefully returns `NoRapl` on non-Linux or when permissions are insufficient.
- ~~**Latency-vs-batch-size heatmap.**~~ ✅ Shipped: `BatchHeatmap::run`
  sweeps `batch_size × seq_len`, records `toks_per_sec`, `p99_latency_ms`,
  and `memory_bytes` per cell; `summary_table` and `p99_table` render Markdown
  grids; Criterion bench at `benches/batch_heatmap.rs` with
  `OXILLAMA_BENCH_PRINT_HEATMAP=1` for optional table output.
- ~~**CI regression gate.**~~ ✅ Shipped (v0.1.6): `RegressionGate` in
  `src/regression_gate.rs` loads a JSON baseline (`from_file` / `save_baseline`),
  checks per-metric thresholds (throughput: higher is better; latency: lower is
  better), and returns structured `RegressionFailure` list.  `format_report`
  emits a Markdown table.  New benchmarks absent from the baseline are silently
  skipped.
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

*Last updated: 2026-04-20 (v0.1.1 — 79 tests; KV-cache scaling, cross-SIMD, memory profiler, tokenizer throughput, E2E stub bench all shipped)*

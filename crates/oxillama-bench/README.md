# oxillama-bench

Benchmark suite for OxiLLaMa quantization kernels and the inference pipeline.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.5 — **Tests:** 146 passing — **Status:** Alpha (opt-in `bench` feature)

## What It Benchmarks

### Benchmark Binaries

- **Quantization kernels** (`benches/dispatch_matrix.rs`): GEMV throughput for every shipped GGUF quant type across all available SIMD paths
- **KV-cache scaling** (`benches/kv_cache.rs`): context-length sweep (1K/4K/8K/32K)
- **Cross-SIMD comparison** (`benches/cross_simd.rs`): AVX2 vs AVX-512 vs NEON vs scalar
- **End-to-end, synthetic** (`benches/end_to_end.rs`): LLaMA-3, Qwen3, Mistral tokens/sec via a `StubEngine` that busy-loops instead of running a forward pass — exercises the Criterion harness and throughput reporting without a GGUF file, but every number is fabricated.
- **End-to-end, real** (`benches/real_e2e.rs`): drives the actual `oxillama-runtime::InferenceEngine` — model load wall-time, prefill tok/s, decode tok/s (fixed 64-token greedy, EOS ignored so the number is reproducible), and peak RSS, against a real GGUF file named by `OXILLAMA_BENCH_MODEL`. Not a Criterion benchmark (loading an 8B-parameter model dozens of times to satisfy a statistics engine is not useful); a plain one-shot report instead. Skips gracefully (prints a message, exits 0) when the env var is unset, so `cargo bench -p oxillama-bench` still passes in CI without a model on disk.

### Library Modules

| Module | Key Types / Functions | Description |
|---|---|---|
| `src/latency.rs` | — | P50/P95/P99 per-token latency and time-to-first-token |
| `src/throughput.rs` | — | Sustained tok/s with warm-up and measurement windows |
| `src/memory.rs` | — | RSS profiling, peak/P95, model-weight and KV-cache estimators |
| `src/dispatch_matrix.rs` | `DispatchMatrixRow`, `detect_available_simd_paths`, `run_dispatch_matrix` | Cross-SIMD comparison tables showing throughput per kernel across all available SIMD paths |
| `src/simd_comparison.rs` | `SimdComparisonConfig`, `run_dequant_comparison`, `run_gemv_comparison` | Per-kernel SIMD benchmark comparing scalar, AVX2, AVX-512, and NEON implementations |
| `src/memory_profiler.rs` | `AsyncMemoryProfiler`, `MemEvent` | Async background RSS sampler with configurable polling interval |
| `src/arch_config.rs` | — | Per-architecture benchmark configs for LLaMA-3, Qwen3, Mistral, Gemma, and Phi |
| `src/prefill_decode.rs` | `PrefillDecodeBench` | Trait and implementations for prefill vs. decode phase benchmarks with KV-cache scaling |
| `src/real_e2e.rs` | `run_real_e2e_bench`, `RealE2eConfig`, `RealE2eReport` | Real end-to-end benchmark against `oxillama-runtime::InferenceEngine` — the only module in this crate that loads an actual GGUF file and runs a real forward pass rather than simulating one |

All benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistical rigor, except `real_e2e` (see above — deliberately not Criterion-driven).

## Running

```bash
# Run all benchmarks
cargo bench -p oxillama-bench

# Run only quantization benchmarks
cargo bench -p oxillama-bench --bench dispatch_matrix

# Run only cross-SIMD comparison benchmarks
cargo bench -p oxillama-bench --bench cross_simd

# Filter to a specific kernel
cargo bench -p oxillama-bench -- q4_0

# Save a baseline for comparison
cargo bench -p oxillama-bench -- --save-baseline main

# Compare against saved baseline
cargo bench -p oxillama-bench -- --baseline main

# Real end-to-end: load time, prefill/decode tok/s, peak RSS against an
# actual GGUF file (skips with a message if the env var is unset)
OXILLAMA_BENCH_MODEL=/path/to/model.gguf cargo bench -p oxillama-bench --bench real_e2e
```

Criterion HTML reports are written to `target/criterion/`.

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

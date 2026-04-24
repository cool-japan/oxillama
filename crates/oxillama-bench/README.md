# oxillama-bench

Benchmark suite for OxiLLaMa quantization kernels and the inference sampling pipeline.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.1 — **Tests:** 88 passing — **Status:** Alpha (opt-in `bench` feature)

## What It Benchmarks

### Benchmark Binaries

- **Quantization kernels** (`benches/quant_kernels.rs`): dequantize throughput for every GGUF type (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ-series, Q1_0_G128, …)
- **Sampling pipeline** (`benches/sampling.rs`): greedy, top-K, top-P, mirostat v1/v2 at various vocabulary sizes
- **KV-cache scaling** (`benches/kv_cache.rs`): context-length sweep (4K/8K/32K)
- **Cross-SIMD comparison** (`benches/cross_simd.rs`): AVX2 vs AVX-512 vs NEON vs scalar
- **End-to-end** (`benches/end_to_end.rs`): LLaMA-3, Qwen3, Mistral via stub engine

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

All benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistical rigor.

## Running

```bash
# Run all benchmarks
cargo bench -p oxillama-bench

# Run only quantization benchmarks
cargo bench -p oxillama-bench --bench quant_kernels

# Run only sampling benchmarks
cargo bench -p oxillama-bench --bench sampling

# Filter to a specific kernel
cargo bench -p oxillama-bench -- q4_0

# Save a baseline for comparison
cargo bench -p oxillama-bench -- --save-baseline main

# Compare against saved baseline
cargo bench -p oxillama-bench -- --baseline main
```

Criterion HTML reports are written to `target/criterion/`.

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

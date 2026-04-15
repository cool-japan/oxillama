# oxillama-bench

Benchmark suite for OxiLLaMa quantization kernels and the inference sampling pipeline.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Benchmarks

- **Quantization kernels** (`benches/quant_kernels.rs`): dequantize throughput for every GGUF type (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ-series, Q1_0_G128, …)
- **Sampling pipeline** (`benches/sampling.rs`): greedy, top-K, top-P, mirostat v1/v2 at various vocabulary sizes

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

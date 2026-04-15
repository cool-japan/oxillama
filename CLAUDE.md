# CLAUDE.md

This file provides guidance to Claude Code when working with the OxiLLaMa codebase.

## Project Overview

OxiLLaMa is a Pure Rust reimplementation of llama.cpp — the de facto standard for local LLM inference. It provides GGUF model loading, multi-format quantized inference, and an OpenAI-compatible API server, all without any C, C++, or Fortran code.

## Key Commands

```bash
# Full workspace check
cargo check --workspace

# Clippy (no warnings policy)
cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check

# Run all tests
cargo nextest run --workspace --no-fail-fast

# Run specific crate tests
cargo nextest run -p oxillama-gguf

# Run the CLI
cargo run -- run --model <path.gguf> --prompt "Hello"
cargo run -- serve --model <path.gguf> --port 8080
cargo run -- info --model <path.gguf>

# Build release
cargo build --release
```

## Architecture

```
oxillama (binary)
├── oxillama-gguf      GGUF v3 parser + tensor loader
├── oxillama-quant     Quantization kernels (Q4_0, Q4_K, Q8_0, Q1_0_G128, ...)
├── oxillama-arch      Model architectures (LLaMA, Qwen3, Mistral, Gemma, Phi)
├── oxillama-runtime   Inference engine, KV cache, sampling, tokenizer
├── oxillama-server    OpenAI-compatible HTTP API (optional, feature: "server")
└── oxillama-bench     Benchmark suite (optional, feature: "bench")
```

Dependency order: gguf → quant → arch → runtime → server/bench

## Design Principles

1. **Pure Rust:** Zero C/C++/Fortran. Zero FFI. Zero system library dependencies.
2. **No file > 2000 lines** (splitrs policy). Split into submodules when approaching limit.
3. **Zero `unwrap()` in production code.** Use `?`, `ok_or_else`, or proper error handling.
4. **No warnings policy.** `cargo clippy -- -D warnings` must pass.
5. **Latest crates policy.** Always use the latest versions available on crates.io.
6. **All errors use `thiserror`.** Each crate has its own `XxxError` + `XxxResult<T>` type.

## COOLJAPAN Ecosystem Dependencies

- `scirs2-core` / `scirs2-linalg` / `scirs2-neural` — tensor primitives and neural ops
- `oxiblas` — Pure Rust BLAS (GEMM/GEMV)
- `oxifft` — Pure Rust FFT (RoPE acceleration)

These MUST be used instead of C-backed alternatives (no OpenBLAS, MKL, FFTW, etc.).

## Testing Strategy

- Use `cargo nextest` (not `cargo test`)
- Property-based tests with `proptest` for quantization kernels
- Reference comparison against llama.cpp output for correctness
- Each crate has its own unit tests in `#[cfg(test)]` modules

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `server` | Enable HTTP API server | Yes |
| `bench` | Enable benchmark crate | No |
| `simd-avx2` | AVX2 SIMD kernels | No |
| `simd-avx512` | AVX-512 SIMD kernels | No |
| `simd-neon` | ARM NEON SIMD kernels | No |

## OxiBonsai Absorption

OxiBonsai (../oxibonsai/) will be absorbed into OxiLLaMa:
- Q1_0_G128 kernels → `oxillama-quant/src/q1_0_g128.rs`
- Qwen3 architecture → `oxillama-arch/src/qwen3/`
- Bonsai runtime → absorbed by `oxillama-runtime`

## Error Hierarchy

```
GgufError    (oxillama-gguf)     ← standalone
QuantError   (oxillama-quant)    ← standalone
ArchError    (oxillama-arch)     ← wraps GgufError, QuantError
RuntimeError (oxillama-runtime)  ← wraps ArchError, GgufError, QuantError
ServerError  (oxillama-server)   ← wraps RuntimeError
```

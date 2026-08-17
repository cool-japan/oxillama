# oxillama-cli

Command-line interface for OxiLLaMa — Pure Rust LLM inference engine.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace.

## Status

**Version:** 0.1.5 — **Tests:** 175 passing (default features) / 179 passing (`--all-features`) — **Status:** Alpha

## Installation

```bash
cargo install oxillama-cli
```

The binary is named `oxillama`.

## Usage

```bash
# Run inference
oxillama run --model model.gguf --prompt "Hello" --max-tokens 256

# Halve KV-cache memory by storing it as f16 (accumulation stays f32).
# Default is f32; `run` and `serve` both accept the flag.
oxillama run --model model.gguf --prompt "Hello" --kv-dtype f16

# [debug] Dump the full final-position logit vector of every generation step
# (headerless little-endian f32, one value per token id) plus a manifest.json,
# for numerically comparing OxiLLaMa against another implementation.
# `--prompt-tokens` bypasses the tokenizer and `--force-tokens` teacher-forces
# the fed token, so both sides can be conditioned on identical ids.
oxillama run --model model.gguf --dump-logits ./dump \
  --prompt-tokens 785,6722,315,9625,374 --max-tokens 8 --temp 0

# Start OpenAI-compatible API server
oxillama serve --model model.gguf --port 8080

# Print model info
oxillama info --model model.gguf

# Run benchmarks
oxillama bench --model model.gguf

# Interactive chat REPL (supports /save <path> and /load <path> for session persistence)
oxillama chat --model model.gguf

# Full-screen TUI chat (requires `tui` feature)
oxillama chat --model model.gguf --tui

# Download GGUF from HuggingFace Hub (requires `hub` feature)
oxillama hub pull <repo> [--sha256 <hash>]

# List cached Hub models
oxillama hub list

# Remove a cached Hub repo
oxillama hub rm <repo>

# Shell completions
oxillama completions bash > ~/.bash_completion.d/oxillama

# Generate man page
oxillama generate-manpage --output-dir /usr/local/share/man/man1

# Version banner (build target, SIMD features, wired architectures — verbose by default)
oxillama version
```

## Configuration

OxiLLaMa CLI supports layered configuration:

- `--config <path>` flag or `OXILLAMA_CONFIG` env var
- Per-model profile files (`~/.config/oxillama/models/*.toml`)
- Global config defaults

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `server` | yes | Enable OpenAI-compatible HTTP server |
| `tui` | yes | Full-screen TUI chat mode (ratatui + crossterm) |
| `simd-avx2` | yes | AVX2 SIMD kernels |
| `simd-neon` | yes | ARM NEON SIMD kernels |
| `bench` | no | Enable benchmark subcommand |
| `simd-avx512` | no | AVX-512 SIMD kernels |
| `hub` | no | HuggingFace Hub subcommands (`hub pull`, `hub list`, `hub rm`) |
| `gpu` | no | Device-resident weights via wgpu (`oxillama-runtime/gpu`). No `--gpu` clap argument on `run`/`serve` yet — enabling the feature does not currently expose a way to turn it on from the CLI. Off by default: pulls in the whole wgpu stack. **New in v0.1.4.** |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

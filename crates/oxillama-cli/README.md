# oxillama-cli

Command-line interface for OxiLLaMa — Pure Rust LLM inference engine.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace.

## Installation

```bash
cargo install oxillama-cli
```

The binary is named `oxillama`.

## Usage

```bash
# Run inference
oxillama run --model model.gguf --prompt "Hello" --max-tokens 256

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
| `bench` | no | Enable benchmark subcommand |
| `simd-avx2` | no | AVX2 SIMD kernels |
| `simd-avx512` | no | AVX-512 SIMD kernels |
| `simd-neon` | no | ARM NEON SIMD kernels |
| `hub` | no | HuggingFace Hub subcommands (`hub pull`, `hub list`, `hub rm`) |
| `tui` | no | Full-screen TUI chat mode (ratatui + crossterm) |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

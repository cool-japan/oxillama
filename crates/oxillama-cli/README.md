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

# Interactive chat REPL
oxillama chat --model model.gguf

# Shell completions
oxillama completions bash > ~/.bash_completion.d/oxillama

# Generate man page
oxillama generate-manpage --output-dir /usr/local/share/man/man1

# Verbose version banner (build target, SIMD features, wired architectures)
oxillama version --verbose
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

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

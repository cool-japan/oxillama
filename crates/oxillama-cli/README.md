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
```

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

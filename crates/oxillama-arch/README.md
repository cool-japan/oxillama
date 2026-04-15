# oxillama-arch

Model architecture implementations for transformer-based LLMs.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Supported Architectures

| Architecture | Feature Flag | Notes |
|-------------|--------------|-------|
| LLaMA 2 / 3 / 4 | `llama` (default) | Meta's foundational decoder |
| Qwen3 | `qwen3` (default) | Alibaba Qwen3 decoder |
| Mistral | `mistral` (default) | Mistral 7B / Mixtral-MoE |
| Gemma | `gemma` (default) | Google Gemma 2 / 3 |
| Phi-3 / Phi-4 | `phi` (default) | Microsoft Phi small models |
| Command-R | `command-r` (default) | Cohere Command-R |
| StarCoder | `starcoder` (default) | BigCode StarCoder 2 |
| LLaVA | `llava` (default) | Multimodal vision+language (depends on `llama`) |
| Mixtral-MoE | included in `mistral` | Sparse mixture-of-experts |

## Key Types

| Type | Description |
|------|-------------|
| `ModelArchitecture` | Enum variant per supported architecture |
| `LlamaModel` | Weight tensors + config for LLaMA-family models |
| `ForwardPass` | Trait: `forward(&self, tokens, cache) -> logits` |
| `ArchError` | Unified error type wrapping `GgufError` and `QuantError` |

## Usage

```rust
use oxillama_arch::{load_model, ModelArchitecture, ArchResult};
use oxillama_gguf::GgufFile;

fn load(path: &str) -> ArchResult<()> {
    let gguf = GgufFile::open(path)?;
    let arch  = ModelArchitecture::detect(&gguf)?;
    let model = load_model(arch, &gguf)?;

    println!("Loaded {:?} with {} layers", arch, model.num_layers());
    Ok(())
}
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `llama` | yes | LLaMA 2/3/4 forward pass |
| `qwen3` | yes | Qwen3 forward pass |
| `mistral` | yes | Mistral / Mixtral forward pass |
| `gemma` | yes | Gemma 2/3 forward pass |
| `phi` | yes | Phi-3/4 forward pass |
| `command-r` | yes | Command-R forward pass |
| `starcoder` | yes | StarCoder 2 forward pass |
| `llava` | yes | LLaVA multimodal (requires `llama`) |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

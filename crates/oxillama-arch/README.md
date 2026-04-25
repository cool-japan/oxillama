# oxillama-arch

Model architecture implementations for transformer-based LLMs.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.2 — **Tests:** 397 passing — **Architectures:** 20 — **Status:** Alpha

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
| LLaVA | `llava` (default) | Multimodal vision+language (requires `llama`) |
| Falcon | `falcon` (default) | TII Falcon (old + new variants) |
| MiniCPM | `minicpm` (default) | MiniCPM scaled-embedding variant |
| OLMo2 | `olmo2` (default) | Allen AI OLMo2 (reordered post-norms) |
| Granite | `granite` (default) | IBM Granite 3.x dense decoder |
| DeepSeek-V2 / V3 | `deepseek` (default) | MLA + MoE; sigmoid-with-bias scoring for V3 |
| DBRX | `dbrx` (default) | Databricks DBRX (16-expert MoE, top-4) — new in v0.1.1; GGUF loaders completed in v0.1.2 |
| Grok-1 | `grok` (default) | xAI Grok-1 (8-expert MoE, top-2, RoPE θ=1e6) — new in v0.1.1; GGUF loaders completed in v0.1.2 |
| Mamba-2 | `mamba2` (default) | Selective-scan SSM — new in v0.1.1; GGUF loaders completed in v0.1.2 |
| Jamba | `jamba` (default) | Hybrid attention+SSM (AI21 Labs) — interleaves transformer attention with Mamba-2 SSM layers (requires `mamba2`) — **new in v0.1.1** |
| Yi | always-on | 01.AI Yi (LLaMA topology, compiled unconditionally) |
| InternLM3 | always-on | Shanghai AI Lab InternLM3 (compiled unconditionally) |
| Mixtral-MoE | included in `mistral` | Sparse mixture-of-experts |

## Key Types

| Type | Description |
|------|-------------|
| `ModelArchitecture` | Enum variant per supported architecture |
| `LlamaModel` | Weight tensors + config for LLaMA-family models |
| `ForwardPass` | Trait: `forward(&self, tokens, cache) -> logits`; `forward_batched` added in v0.1.2 (NotSupported default + LLaMA impl) |
| `SequenceState` | Trait: generalises `KvCacheAccess` for SSMs — **new in v0.1.1** |
| `BatchedKvView` | View over a batch slice of the KV cache; `KvSlot` per sequence — moved to `traits.rs` in v0.1.2 |
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

18 feature flags (one per feature-gated architecture family plus one debug path):

| Feature | Default | Description |
|---------|---------|-------------|
| `llama` | yes | LLaMA 2/3/4 forward pass |
| `qwen3` | yes | Qwen3 forward pass |
| `mistral` | yes | Mistral / Mixtral forward pass |
| `gemma` | yes | Gemma 2/3 forward pass |
| `phi` | yes | Phi-3/4 forward pass |
| `command-r` | yes | Command-R forward pass |
| `starcoder` | yes | StarCoder 2 forward pass |
| `llava` | yes | LLaVA multimodal (enables `llama`) |
| `falcon` | yes | Falcon old + new variants |
| `minicpm` | yes | MiniCPM scaled embedding |
| `olmo2` | yes | OLMo2 reordered post-norms |
| `granite` | yes | Granite 3.x dense decoder |
| `deepseek` | yes | DeepSeek-V2 / V3 (MLA + MoE) |
| `dbrx` | yes | DBRX 16-expert MoE |
| `grok` | yes | Grok-1 8-expert MoE |
| `mamba2` | yes | Mamba-2 selective-scan SSM |
| `jamba` | yes | Hybrid attention+SSM (enables `mamba2`) |
| `reference-f32` | no | Scalar F32 reference path for testing/debugging |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

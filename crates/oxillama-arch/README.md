# oxillama-arch

Model architecture implementations for transformer-based LLMs.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.5 — **Tests:** 1036 passing — **Architectures:** 26 — **Status:** Alpha

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
| LLaVA-1.6 / LLaVA-NeXT | `llava16` (default) | Anyres tiling multimodal vision+language |
| Qwen2-VL | `qwen2-vl` (default) | M-RoPE multimodal vision+language |
| Falcon | `falcon` (default) | TII Falcon (old + new variants) |
| MiniCPM | `minicpm` (default) | MiniCPM scaled-embedding variant |
| OLMo2 | `olmo2` (default) | Allen AI OLMo2 (reordered post-norms) |
| Granite | `granite` (default) | IBM Granite 3.x dense decoder |
| DeepSeek-V2 / V3 | `deepseek` (default) | MLA + MoE; sigmoid-with-bias scoring for V3 |
| DBRX | `dbrx` (default) | Databricks DBRX (16-expert MoE, top-4) — new in v0.1.1; GGUF loaders completed in v0.1.2 |
| Grok-1 | `grok` (default) | xAI Grok-1 (8-expert MoE, top-2, RoPE θ=1e6) — new in v0.1.1; GGUF loaders completed in v0.1.2 |
| Mamba-2 | `mamba2` (default) | Selective-scan SSM — new in v0.1.1; GGUF loaders completed in v0.1.2 |
| Jamba | `jamba` (**not** default — omitted from `all-architectures` in `Cargo.toml`) | Hybrid attention+SSM (AI21 Labs) — interleaves transformer attention with Mamba-2 SSM layers (requires `mamba2`) — **new in v0.1.1** |
| Yi | always-on | 01.AI Yi (LLaMA topology, compiled unconditionally) |
| InternLM3 | always-on | Shanghai AI Lab InternLM3 (compiled unconditionally) |
| Mixtral-MoE | `mixtral` (default) | Sparse MoE FFN over LLaMA topology — own module `src/mixtral/`, enables `llama` (not part of `mistral`) |
| StableLM | `stablelm` (default) | Stability AI StableLM |
| GPT-NeoX | `gptneox` (default) | EleutherAI GPT-NeoX / Pythia family |
| BLOOM | `bloom` (default) | BigScience BLOOM (ALiBi, no RoPE) |
| Phi-3.5-MoE | `phimoe` (default) | Microsoft Phi-3.5 sparse MoE (16 experts, top-2) |

## Key Types

| Type | Description |
|------|-------------|
| `ModelArchitecture` | Enum variant per supported architecture |
| `LlamaModel` | Weight tensors + config for LLaMA-family models |
| `ForwardPass` | Trait: `forward(&self, tokens, cache) -> logits`; `forward_batched` added in v0.1.2 (NotSupported default + LLaMA impl) |
| `SequenceState` | Trait: generalises `KvCacheAccess` for SSMs — **new in v0.1.1** |
| `BatchedKvView` | View over a batch slice of the KV cache; `KvSlot` per sequence — moved to `traits.rs` in v0.1.2 |
| `common::fetch_keys` / `fetch_values` | dtype-agnostic KV reads: borrow contiguous f32 storage with no copy, gather otherwise. Every attention kernel in this crate goes through them, so `KvCacheDtype::F16` works on every architecture — **new in v0.1.4**. The gather also spans a `PagedKvCache`'s pages correctly, but paged attention is still **not** wired up: `PagedKvCache` has no `stored_len`, so the row written at the current position is invisible until `advance()` |
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

25 feature flags (24 feature-gated architecture families plus one debug path):

| Feature | Default | Description |
|---------|---------|-------------|
| `llama` | yes | LLaMA 2/3/4 forward pass |
| `qwen3` | yes | Qwen3 forward pass |
| `mistral` | yes | Mistral forward pass |
| `gemma` | yes | Gemma 2/3 forward pass |
| `phi` | yes | Phi-3/4 forward pass |
| `command-r` | yes | Command-R forward pass |
| `starcoder` | yes | StarCoder 2 forward pass |
| `llava` | yes | LLaVA multimodal (enables `llama`) |
| `llava16` | yes | LLaVA-1.6/NeXT anyres tiling multimodal (enables `llava`) |
| `qwen2-vl` | yes | Qwen2-VL M-RoPE multimodal |
| `falcon` | yes | Falcon old + new variants |
| `minicpm` | yes | MiniCPM scaled embedding |
| `olmo2` | yes | OLMo2 reordered post-norms |
| `granite` | yes | Granite 3.x dense decoder |
| `deepseek` | yes | DeepSeek-V2 / V3 (MLA + MoE) |
| `dbrx` | yes | DBRX 16-expert MoE |
| `grok` | yes | Grok-1 8-expert MoE |
| `mamba2` | yes | Mamba-2 selective-scan SSM |
| `jamba` | **no** — omitted from `all-architectures` (see `Cargo.toml`) | Hybrid attention+SSM (enables `mamba2`) |
| `mixtral` | yes | Mixtral sparse MoE FFN over LLaMA topology (enables `llama`) |
| `stablelm` | yes | StableLM forward pass |
| `gptneox` | yes | GPT-NeoX / Pythia forward pass |
| `bloom` | yes | BLOOM ALiBi forward pass (no RoPE) |
| `phimoe` | yes | Phi-3.5-MoE 16-expert top-2 sparse MoE |
| `reference-f32` | no | Scalar F32 reference path for testing/debugging |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

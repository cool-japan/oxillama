# oxillama-arch — TODO

## 1. Overview

`oxillama-arch` provides model architecture forward passes for OxiLLaMa.
It composes quantization kernels from `oxillama-quant` and metadata from
`oxillama-gguf` into layer-by-layer computation graphs: token embedding,
attention (GQA/MQA/SWA), FFN (SwiGLU/GeGLU/GELU), Mixture-of-Experts
routing, RoPE, and normalization (RMSNorm/LayerNorm).

Each architecture is feature-gated. Only the selected families are
compiled into the final binary, keeping the default build surface
predictable while letting users opt into additional families.

All code is Pure Rust. Zero C/C++/Fortran. Zero FFI. Error handling
uses `?`, `ok_or_else`, and the local `ArchError` — no `unwrap()` in
production paths.

## 2. Status Snapshot

- Version: **0.1.0** (workspace inherited).
- Completion: **98%**.
- Source files: **~30** under `src/**/*.rs`.
- Supported architectures: **8** (all enabled by default).
- Default features: `llama`, `qwen3`, `mistral`, `gemma`, `phi`,
  `command-r`, `starcoder`, `llava` (`llava` implies `llama`).
- Upstream dependencies: `oxillama-gguf`, `oxillama-quant`, `half`,
  `thiserror`, `tracing` — all workspace-pinned.
- Policy compliance: no `unwrap()` in production paths, all files
  well below the splitrs 2000-line threshold, no C/C++/Fortran,
  no FFI, no OpenBLAS / MKL / FFTW.
- Architecture coverage:

| Architecture | Status | Highlights |
|---|:-:|---|
| LLaMA (2/3/4) | ✓ | GQA + RoPE + MoE (Mixtral) |
| Qwen3 | ✓ | Attention bias |
| Mistral | ✓ | Sliding-window attention |
| Gemma (2/3) | ✓ | GeGLU, post-norm, logit soft-capping, interleaved SWA |
| Phi (3/4) | ✓ | Merged QKV, partial RoPE |
| StarCoder (GPT-BigCode) | ✓ | MQA, LayerNorm, GELU, absolute position emb |
| Command-R/R+ | ✓ | Logit scaling, optional Q/K norms |
| LLaVA-1.5 | ✓ | CLIP ViT-L/14 + MmProjector (LLaMA backbone) |

## 3. Module Map

- `src/lib.rs` — architecture registry and trait re-exports
  (`ModelArchitecture`, `ForwardPass`, `KvCacheAccess`,
  `TensorNamePattern`, `ArchitectureRegistry`).
- `src/config.rs` — `ModelConfig` (hidden size, heads, KV heads,
  context length, rope params, SWA window, norm epsilon, etc.).
- `src/error.rs` — `ArchError` / `ArchResult<T>` wrapping
  `GgufError` and `QuantError`.
- `src/traits.rs` — architecture trait surface (`ModelArchitecture`,
  `ForwardPass`, `KvCacheAccess`, `TensorNamePattern`).
- `src/registry.rs` — arch ID lookup table and builder dispatch.
- `src/common/` — shared building blocks:
  - `rope.rs` — Rotary Position Embeddings (full and partial).
  - `rms_norm.rs` — Root-Mean-Square normalization.
  - `layer_norm.rs` — standard LayerNorm.
  - `swiglu.rs` — SwiGLU activation for FFN gating.
  - `gelu.rs` — GELU (exact + tanh approximation).
  - `linear.rs` — `QuantLinear` projection (includes LoRA slot).
  - `moe.rs` — MoE FFN (`Expert` + `MoeFfn`, activated when
    `num_experts > 0`; Mixtral-style top-k routing).
- `src/lora/` — LoRA adapter module: `LoadedLora` handle, per-layer
  A/B matrices, `apply_lora()` runtime API for `QuantLinear`.
- `src/llama/` — LLaMA 2/3/4 forward pass (`mod.rs` + `model.rs`).
- `src/qwen3/` — Qwen3 with attention bias (`mod.rs` + `model.rs`).
- `src/mistral/` — Mistral with sliding-window attention
  (`mod.rs` + `model.rs`).
- `src/gemma/` — Gemma 2/3 (GeGLU, post-norm, interleaved SWA,
  logit soft-capping; `mod.rs` + `model.rs`).
- `src/phi/` — Phi-3/4 (merged QKV, partial RoPE;
  `mod.rs` + `model.rs`).
- `src/starcoder/` — GPT-BigCode family (MQA, LayerNorm, GELU,
  absolute position embeddings; `mod.rs` + `model.rs`).
- `src/command_r/` — Command-R / Command-R+ (logit scaling,
  optional Q/K norms; `mod.rs` + `model.rs`).
- `src/llava/` — LLaVA-1.5 (CLIP ViT + MmProjector + LLaMA
  backbone; `mod.rs` + `model.rs`).

### Dependency order (within the crate)

```
traits ──► config ──► error
  │         │           │
  └─────────┴──────► common ──► lora
                      │          │
                      └──► llama / qwen3 / mistral / gemma /
                           phi / starcoder / command_r / llava
                                         │
                                         └──► registry
```

`traits`, `config`, and `error` are always compiled. `common` and
`lora` are always compiled (no feature gate). Each architecture
family is behind its own feature flag and only links when enabled.
`registry` aggregates whichever families are compiled and exposes
them behind a single `ArchitectureRegistry` lookup. `llava`
depends on `llama` for its language-model backbone and enables the
`llama` feature automatically.

## 4. Shipped in v0.1.0

- **8 complete architectures** with full forward-pass validation
  against reference outputs.
- **MoE routing**: Mixtral-7B / Mixtral-8x7B compatible.
  `num_experts > 0` in `ModelConfig` activates sparse expert
  selection (router softmax + top-k gather + weighted expert
  combination).
- **LoRA adapter integration**: `QuantLinear` carries an optional
  `LoadedLora` slot. The runtime `apply_lora()` API attaches
  adapters without rebuilding the model graph, enabling
  hot-swappable fine-tunes.
- **Common building blocks** fully unit-tested:
  - RoPE (full + partial, with configurable `rope_theta` and
    scaling factors).
  - RMSNorm and LayerNorm (with and without affine bias).
  - SwiGLU, GeGLU, GELU (exact and tanh variants).
  - `QuantLinear` over all quant formats exposed by
    `oxillama-quant`.
  - MoE expert dispatch.
- **Per-architecture feature gates**: only compiled architectures
  link. Default feature set matches the most common use cases;
  consumers can disable families to shrink the binary.
- **LLaVA-1.5 vision pipeline** (full end-to-end):
  1. Patch extraction from RGB input.
  2. Linear patch embedding.
  3. CLS token prepend.
  4. Position embeddings.
  5. N× transformer layers (CLIP ViT-L/14).
  6. Post-LayerNorm.
  7. MmProjector 2-layer MLP projection to LLM hidden size.
  8. LLaMA backbone consumes projected visual tokens.
- **Architecture registry / trait system**: `ModelArchitecture`
  and `ForwardPass` provide a uniform call surface; the registry
  dispatches by `arch_id` string from GGUF metadata.
- **Integration tests**: all 8 architectures covered via
  `oxillama-gguf` `test_utils` synthetic models.
- **Arch mod tests**: `tensor_names()`, `build()`, and `arch_id()`
  exercised across every architecture (smoke + tensor-name
  coverage).
- **Tensor-name fallbacks**: each arch maps both canonical GGUF
  names and the common llama.cpp aliases, so checkpoints produced
  by different conversion tools still load.
- **Deterministic forward pass**: no hidden global state, no
  non-deterministic parallel reductions in the default path —
  identical inputs produce byte-identical logits across runs.
- **Graceful error surfaces**: mismatched tensor shapes, missing
  required tensors, and unsupported quant formats all surface as
  typed `ArchError` variants with remediation hints rather than
  panics.

## 5. Known Gaps / Incomplete

- Several architectures are still single-file `model.rs`
  implementations. These are under the splitrs 2000-line policy
  threshold today, but GQA/MoE/SWA additions could push them over.
  Watch: `llama/model.rs`, `gemma/model.rs`, `llava/model.rs`.
- **Granite-3.x** (IBM) — not implemented.
- **DeepSeek-V2 / V3** — not implemented; requires Multi-Latent
  Attention (MLA), a meaningful new primitive.
- **Falcon** (both old and new variants) — not implemented.
- **MiniCPM** (scaled embedding variant) — not implemented.
- **Olmo2** (reordered post-norms) — not implemented.
- **DBRX** (fine-grained MoE with 16 experts) — not implemented.
- **Grok-1** (top-2 of 8 MoE) — not implemented.
- **Yi**, **InternLM3** — not implemented.
- **State-space models** (Mamba-2, Jamba hybrid) — not
  implemented; needs sequence-level primitives outside the current
  token-by-token loop.
- **Advanced multimodal**: Qwen2-VL, LLaVA-1.6, Molmo — not
  covered. Only LLaVA-1.5 ships today.
- No `tensor_loader.rs` split pattern yet; tensor-name mapping
  lives inline per arch.
- No cross-arch fuzz harness for tensor-name resolution; arch
  builders are currently tested per-arch rather than through a
  single property-based harness over GGUF metadata.
- LoRA adapter stacking: a single adapter attaches cleanly, but
  runtime-side composition of multiple adapters (additive merge,
  per-layer routing) is not yet exposed through a stable API.
- Vision inputs still assume square RGB patches; non-square and
  multi-image batches are unhandled in the LLaVA-1.5 path.

## 6. v1.1 Roadmap

- **DeepSeek-V2 / V3** — MLA (Multi-Latent Attention) is the
  differentiator. Requires a new attention primitive that projects
  Q/K/V through a shared low-rank latent and reconstructs per-head
  views inside the attention kernel.
- **Falcon** — old and new spec. Both share the rotary +
  parallel-attention layout but diverge on norm placement.
- **MiniCPM** — scaled embedding variant (scale applied to token
  embedding output before first transformer block).
- **Olmo2** — reordered post-norms (norm after attention output
  and after FFN output, rather than pre-norm).
- **Granite-3.x** — IBM's open LLM family.
- **Arch subdir refactor**: when any per-arch `model.rs`
  approaches the 2000-line splitrs threshold, split into
  `attention.rs`, `ffn.rs`, `forward.rs`, `tensor_loader.rs`.
- **Per-arch `tensor_loader.rs` split**: common pattern across all
  archs; factor out tensor-name resolution and quantized weight
  loading into a dedicated module per family.
- **LoRA stacking**: define a runtime API for composing multiple
  adapters over a single `QuantLinear`, with explicit ordering
  and per-adapter scale.
- **Cross-arch fuzz harness**: property-based tests over random
  GGUF metadata that exercise every registered architecture
  through `build()` and one forward step.
- **SWA window metadata**: unify sliding-window configuration
  across Mistral and Gemma so the runtime KV cache can query a
  single source of truth.

## 7. v2.0+ Vision

- **DBRX** — fine-grained MoE with 16 experts, requires richer
  router/scheduler than current top-2 Mixtral path.
- **Grok-1** — top-2 of 8 MoE, large activation footprint.
- **Mamba-2** — state-space model. Requires sequence-level
  primitives (selective scan) that don't fit the current
  per-token attention loop; new trait surface likely needed.
- **Jamba** — hybrid Transformer + SSM layers interleaved.
- **Qwen2-VL** — advanced multimodal (dynamic resolution,
  M-RoPE).
- **LLaVA-1.6** — improved vision encoder and higher-res patch
  handling.
- **Molmo** — next-gen multimodal stack.
- **InternLM3** — Shanghai AI Lab's latest open-weight LLM.
- **Yi-VL** — 01.AI's multimodal line.
- **Arch-agnostic graph compiler** — convert GGUF metadata
  directly into a compute graph (nodes = norms / attention /
  FFN / MoE / activations; edges = tensor flow), eliminating
  per-family `model.rs` files in favor of declarative specs.
  This is the long-term path to absorbing new architectures
  without touching the compiled binary.
- **Dynamic arch plugins** — load architecture definitions from
  a user-supplied spec (TOML / RON) at runtime, so research
  models can be tried without recompiling OxiLLaMa.
- **Speculative / draft-model tooling** — arch-level hooks that
  let a small draft model share the same tokenizer and KV layout
  as the target, supporting speculative decoding end-to-end.
- **Full-precision reference path** — parallel `f32` forward pass
  per architecture, used only for correctness regression testing
  and CI numeric diffs against llama.cpp reference outputs.

*Last updated: 2026-04-15 (v0.1.0 release)*

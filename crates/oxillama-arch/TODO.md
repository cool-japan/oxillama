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

- Version: **0.1.1** (workspace inherited).
- Completion: **98%**.
- Source files: **~45** under `src/**/*.rs`.
- Supported architectures: **18** (16 feature-gated + Yi and InternLM3 compiled unconditionally).
- Default features (16): `llama`, `qwen3`, `mistral`, `gemma`, `phi`,
  `command-r`, `starcoder`, `llava`, `falcon`, `minicpm`, `olmo2`,
  `granite`, `deepseek`, `dbrx`, `grok`, `mamba2` (`llava` implies `llama`).
- Tests: **334 passing**.
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
| Falcon | ✓ | Old + new variants; rotary + parallel-attention |
| MiniCPM | ✓ | Scaled embedding before first transformer block |
| OLMo2 | ✓ | Reordered post-norms (after attention + after FFN) |
| Granite 3.x | ✓ | Dense decoder-only; IBM open LLM family |
| DeepSeek-V2 / V3 | ✓ | MLA (compressed latent KV) + shared/routed MoE |
| DBRX | ✓ | Fine-grained 16-expert MoE, top-4 routing — **new in v0.1.1** |
| Grok-1 | ✓ | 8-expert MoE, top-2, RoPE θ=1e6 — **new in v0.1.1** |
| Mamba-2 | ✓ | Selective-scan SSM with learned Δ — **new in v0.1.1** |
| Yi | ✓ | 01.AI Yi; LLaMA topology, GGUF arch `"yi"` |
| InternLM3 | ✓ | Shanghai AI Lab InternLM3 |

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
    **[2026-04-16]** Added `RopeScalingType` (Standard / Linear / YaRN) +
    `ModelConfig::rope_scaling_type/factor` populated from GGUF metadata.
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
- **Granite-3.x** (IBM) — [x] implemented (`crates/oxillama-arch/src/granite/`); dense decoder-only, LLaMA topology, GGUF arch key `"granite"`.
- [x] **DeepSeek-V2 / V3** — implemented; Multi-Latent Attention (MLA)
  primitive in `common/mla.rs`, MoE FFN in `deepseek/moe.rs`,
  full transformer in `deepseek/model.rs`. (2026-04-19)
- **Falcon** (both old and new variants) — implemented (`crates/oxillama-arch/src/falcon/`).
- **MiniCPM** (scaled embedding variant) — [x] implemented (`crates/oxillama-arch/src/minicpm/`).
- **Olmo2** (reordered post-norms) — [x] implemented (`crates/oxillama-arch/src/olmo2/`).
- **DBRX** — [x] implemented (`crates/oxillama-arch/src/dbrx/`); fine-grained 16-expert MoE, top-4 routing. (v0.1.1)
- **Grok-1** — [x] implemented (`crates/oxillama-arch/src/grok/`); 8-expert MoE, top-2, RoPE θ=1e6. (v0.1.1)
- **Yi** — [x] implemented (`crates/oxillama-arch/src/yi/`).
- **InternLM3** — [x] implemented (`crates/oxillama-arch/src/internlm3/`).
- **State-space models** — [x] Mamba-2 implemented (`crates/oxillama-arch/src/mamba2/`);
  `SequenceState` trait in `common/sequence_state.rs` generalises KvCacheAccess for SSMs. (v0.1.1)
  Jamba hybrid: see B1 below.
- **Advanced multimodal**: Qwen2-VL, LLaVA-1.6, Molmo — not
  covered. Only LLaVA-1.5 ships today.
- [x] **B4 — `tensor_loader.rs` preventive split via splitrs (planned 2026-04-20)** — DROPPED: `tensor_loader.rs` does not exist in the tree. Largest arch file is `llava/model.rs` at 1,210 lines (well under the 2,000-line splitrs threshold). The split intent is subsumed by B3's lora submodule restructure.
  - **Goal:** `crates/oxillama-arch/src/tensor_loader.rs` is approaching the 2000-LoC limit. Split before B1's Jamba additions push it over.
  - **Design:** Run `rslines 50 crates/oxillama-arch/src/tensor_loader.rs` to confirm size and identify natural split boundaries. Run `splitrs crates/oxillama-arch/src/tensor_loader.rs` to refactor into `tensor_loader/{mod,llama,qwen,mamba2,...}.rs` along arch boundaries. Re-export the public API from `tensor_loader/mod.rs` so callers see no API change. Verify with `cargo check -p oxillama-arch --all-features` and `cargo nextest run -p oxillama-arch --all-features`.
  - **Files:** `crates/oxillama-arch/src/tensor_loader.rs` (split into directory module by splitrs).
  - **Prerequisites:** none. Sequenced before B1 so B1 writes into the post-split structure.
  - **Tests:** existing `oxillama-arch` test suite must pass unchanged.
  - **Risk:** splitrs sometimes produces sub-files whose imports need manual cleanup. Run `cargo clippy` to surface issues.
- Cross-arch fuzz harness shipped at `tests/arch_fuzz.rs` (proptest, 5 property tests, 32 cases each — covers tensor-name resolution across all registered architectures).
- [x] **B3 — LoRA stacking public API + multi-adapter composition (planned 2026-04-20, refined 2026-04-20)**
  - **Goal:** Stable public API to load N LoRA adapters and apply them in defined order.
  - **Design:**
    - New trait `LoraAdapter`:
      ```rust
      pub trait LoraAdapter: Send + Sync {
          fn rank(&self) -> usize;
          fn alpha(&self) -> f32;
          fn target_modules(&self) -> &[TargetModule];
          fn delta(&self, target: TargetModule, layer: usize) -> Option<&LoraDelta>;
      }
      pub struct LoraDelta { pub a: Vec<f32>, pub b: Vec<f32> }
      pub struct LoraStack { adapters: Vec<Arc<dyn LoraAdapter>> }
      ```
    - Application order is push order; final delta = `Σᵢ (αᵢ / rᵢ) · Aᵢ · Bᵢ`.
    - `ArchModel::with_lora_stack(stack: LoraStack)` extends the trait; archs apply via inline second matmul (memory-efficient — no merged weight).
    - `LoraAdapter::from_gguf(path)` — exposes structured public API over already-supported PEFT-style GGUFs.
    - Incompatible adapters bubble `ArchError::LoraIncompatible`.
  - **Files:** `crates/oxillama-arch/src/lora/{mod,adapter,stack,loader}.rs` (split of existing 711-line lora/mod.rs + new public API, ~700 LoC total); `crates/oxillama-arch/src/traits.rs` (extend `ForwardPass` with `with_lora_stack`); per-arch `model.rs` files (apply stack in forward — additive, default no-op impl acceptable).
  - **Prerequisites:** none.
  - **Tests:** (a) `single_lora_matches_baseline_no_lora`. (b) `two_loras_compose_additively`. (c) `incompatible_rank_errors_clearly`. (d) `lora_persistence_across_requests`.
  - **Risk:** Per-token LoRA matmul overhead is negligible (low rank, small). Avoid precomputed merged weights.
- Vision inputs still assume square RGB patches; non-square and
  multi-image batches are unhandled in the LLaVA-1.5 path.

## 6. v1.1 Roadmap

- [x] **DeepSeek-V2 / V3** — MLA (Multi-Latent Attention) implemented
  in `common/mla.rs`; MoE FFN with shared+routed experts in
  `deepseek/moe.rs`; full transformer model in `deepseek/model.rs`.
  Registered under arch id `"deepseek2"`. (2026-04-19)
- [x] **Falcon** — old and new spec. Both share the rotary +
  parallel-attention layout but diverge on norm placement.
- **MiniCPM** — scaled embedding variant (scale applied to token
  embedding output before first transformer block).
- **Olmo2** — reordered post-norms (norm after attention output
  and after FFN output, rather than pre-norm).
- ~~**Granite-3.x**~~ — [x] implemented (dense decoder-only; IBM's open LLM family).
- **Arch subdir refactor**: when any per-arch `model.rs`
  approaches the 2000-line splitrs threshold, split into
  `attention.rs`, `ffn.rs`, `forward.rs`, `tensor_loader.rs`.
- **Per-arch `tensor_loader.rs` split**: common pattern across all
  archs; factor out tensor-name resolution and quantized weight
  loading into a dedicated module per family.
- ~~**LoRA stacking**: define a runtime API for composing multiple
  adapters over a single `QuantLinear`, with explicit ordering
  and per-adapter scale.~~ ✅ Done — `LoraStack` in `src/lora/mod.rs` with
  `push`, `apply`, `entries`; `ForwardPass::apply_lora_stack` default impl in `traits.rs`.
- ~~**Cross-arch fuzz harness**: property-based tests over random
  GGUF metadata that exercise every registered architecture
  through `build()` and one forward step.~~ ✅ Shipped — `tests/arch_fuzz.rs`
  with proptest exercises all 11 builtins via `tensor_names()` (5 property tests, 32 cases each).
- **SWA window metadata**: ~~unify sliding-window configuration
  across Mistral and Gemma so the runtime KV cache can query a
  single source of truth.~~ **DONE** — `ModelConfig::swa_window`/`swa_interleaved`,
  `effective_attention_span()` in `common::attention`, `swa_config()` on
  `ModelArchitecture`/`ForwardPass`, implemented in Mistral and Gemma.

## 7. v2.0+ Vision

- ~~**DBRX**~~ ✅ Shipped in v0.1.1 (`src/dbrx/`).
- ~~**Grok-1**~~ ✅ Shipped in v0.1.1 (`src/grok/`).
- ~~**Mamba-2**~~ ✅ Shipped in v0.1.1 (`src/mamba2/`); `SequenceState` trait
  in `common/sequence_state.rs`.
- [x] **B1 — Jamba hybrid (LLaMA × Mamba-2) architecture + close A1↔B1 trait seam (planned 2026-04-20, refined 2026-04-20)**
  - **Goal:** New `oxillama_arch::jamba` module registered under `"jamba"`. Genuinely hybrid: alternating layers of Mamba-2 SSM blocks and LLaMA-style attention + FFN, per the published Jamba spec. Feature `jamba` default-on.
  - **Design:**
    - Layout: `src/jamba/{mod,model,config}.rs`.
    - Per-layer dispatch: `jamba.layer_pattern: Vec<LayerKind>` parsed from GGUF metadata `jamba.attention_layer_offsets` (every Nth layer is attention, rest are SSM; default N=8).
    - `LayerKind::Attention` reuses `oxillama_arch::llama::attention::self_attention_forward` (already public).
    - `LayerKind::Ssm` reuses `oxillama_arch::mamba2::ssm::selective_scan` and `mamba2::conv::depthwise_conv1d`.
    - `SequenceState` impl: Jamba state is a heterogeneous `Vec<LayerState>` where each entry is `LayerState::Kv(KvCache)` or `LayerState::Ssm(SsmLayerState)`. Implement `SequenceState` trait with branch-per-layer `reset` / `advance`.
    - Config: `jamba.attn_layer_period`, `jamba.attn_layer_offset`, `jamba.expert_count` (MoE FFN layers via existing `deepseek::moe` infrastructure).
    - Registry wiring + feature flag forward to `crates/oxillama/Cargo.toml`.
    - Fixture: `build_minimal_jamba_gguf()` in `crates/oxillama-gguf/src/test_utils.rs` (additive).
  - **Files:** `crates/oxillama-arch/src/traits.rs` (add `allocate_sequence_state` with default KV impl to `ModelArchitecture`); `crates/oxillama-arch/src/mamba2/model.rs` (override); `crates/oxillama-arch/src/jamba/{mod,model,config}.rs` (new, ~800–1100 LoC total); `crates/oxillama-arch/src/registry.rs` (add `"jamba"` arm); `crates/oxillama-arch/src/config.rs` (extend with Jamba fields); `crates/oxillama-arch/Cargo.toml` (`jamba` feature, default-on); `crates/oxillama/Cargo.toml` (forward); `crates/oxillama-gguf/src/test_utils.rs` (fixture, additive); `crates/oxillama-arch/tests/jamba.rs` (integration); `crates/oxillama-runtime/src/sequence_pool.rs` (route allocation through the new trait method).
  - **Prerequisites:** B3 first (keeps traits.rs churn linear); B2 parallel-safe. Integration sub-item: add `allocate_sequence_state(&self) -> Box<dyn SequenceState>` to `ModelArchitecture` with default KV wrapper impl; override in Mamba-2 and Jamba; update `sequence_pool.rs` to route through the new method.
  - **Tests:** tensor_names; per-layer dispatch correctness; integration forward via fixture; shape + finiteness; mixed-state isolation.
  - **Risk:** Jamba's exact MoE config varies by checkpoint. Default to top-2 of 16 (published config); overridable via metadata.
- **Qwen2-VL** — advanced multimodal (dynamic resolution,
  M-RoPE).
- **LLaVA-1.6** — improved vision encoder and higher-res patch
  handling.
- **Molmo** — next-gen multimodal stack.
- ~~**InternLM3**~~ ✅ Shipped (`src/internlm3/`).
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
- [x] **B2 — Full-precision f32 reference path (CI numeric diff) (planned 2026-04-20, refined 2026-04-20)**
  - **Goal:** Compile-time feature `reference-f32` on `oxillama-arch` that bypasses all quantization: weights dequantized eagerly at load, all GEMMs run on f32 oxiblas, all ops use f32 accumulators. Used for CI numeric-diff testing. Not for production; not enabled by default.
  - **Design:**
    - New module `crates/oxillama-arch/src/reference/{mod,loader,forward}.rs`.
    - `ReferenceLoader::dequantize_all(model: &GgufModel) -> ReferenceWeights` — calls `oxillama_quant::reference::dequantize(...)` (scalar oracle) for every tensor.
    - `ReferenceModel::forward(input)` — same arch forward but every `matvec_q8` / `matvec_q8_fused` routes to `reference_matmul_f32` (oxiblas f32).
    - Public API: feature-gated. `oxillama_arch::reference::load_as_reference(path)` returns `Box<dyn ArchModel>`.
    - CI: `crates/oxillama-arch/tests/reference_diff.rs` asserts max abs diff per arch (`Q4_K`: `1e-1`, `Q8_0`: `1e-3`, `F16`: `1e-3`).
  - **Files:** `crates/oxillama-arch/src/reference/{mod,loader,forward}.rs` (new, ~600 LoC); `crates/oxillama-arch/Cargo.toml` (`reference-f32 = []` feature); `tests/reference_diff.rs` (new).
  - **Prerequisites:** none.
  - **Tests:** (a) `reference_dequantizes_q4_k_to_f32` — bit-equal to scalar reference. (b) `reference_forward_q4_k_within_tolerance` — `1e-1`. (c) `reference_forward_f16_within_tolerance` — `1e-3`.
  - **Risk:** Eager dequant inflates memory ~4×. Acceptable for CI; document "reference-f32 is for diff testing only".

## 8. Planned Work (v0.1.1)

### C1 — MLA primitive `common/mla.rs` (planned 2026-04-19)

- [x] Standalone MLA primitive in `oxillama-arch::common::mla` with full paper-accurate forward pass — compressed Q latent, compressed KV latent, decoupled RoPE. Arch-internal `MlaLatentCache` (plain struct, NOT a KvCacheAccess trait extension — that is deferred to v0.1.2). (planned 2026-04-19)
  - Config: `MlaConfig { num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, rope_theta, softmax_scale }`.
  - Weights: `MlaWeights { w_q_a, q_a_norm, w_q_b, w_kv_a, kv_a_norm, w_kv_b, w_o }` (QuantLinear + RmsNorm).
  - Cache: `MlaLatentCache { kv_latent: Vec<f32>, k_rope: Vec<f32>, seq_len: usize }` with `new/append/clear`.
  - Forward: q_latent → q_nope/q_rope via w_q_b; kv_combined → kv_latent/k_rope via w_kv_a; RoPE on q_rope+k_rope; cache append; lazy reconstruct via w_kv_b → k_nope+v; attention with decoupled k=[k_nope|k_rope]; output projection.
  - Public API: `pub fn mla_forward(x, weights, cfg, cache, position) -> ArchResult<Vec<f32>>`.
  - Files: `src/common/mla.rs` (new, ~500–700 LoC), `src/common/mod.rs`, `src/config.rs` (MlaConfig field).
  - Out of scope: any edit to `src/traits.rs` (KvCacheAccess) or oxillama-runtime.
  - Tests: shape coherence, determinism, cached-vs-uncached equivalence (2-token), RoPE-decoupled isolation; all `#[cfg(test)]`.
  - Risk: Cached-vs-uncached equivalence catches lazy-reconstruct shape/ordering bugs.

### C2 — DeepSeek-V2 full forward pass + registry (planned 2026-04-19)

- [x] `oxillama_arch::deepseek` module with `DeepSeekModel` using C1's MLA, registered as `"deepseek2"`. Latent KV cache is `Vec<MlaLatentCache>` owned by `DeepSeekModel` (one per layer). Feature `deepseek` default-on. (planned 2026-04-19)
  - Files: `src/deepseek/mod.rs`, `src/deepseek/model.rs` (DeepSeekModel, per-layer Vec<MlaLatentCache>), `src/deepseek/moe.rs` (SwiGLU expert FFN, shared+routed, ScoringMode::Softmax|SigmoidWithBias enum).
  - Forward: RMSNorm → MLA → residual → RMSNorm → MoE(shared+topk-routed) → residual.
  - Tensor names: `blk.{L}.attn_q_a.weight`, `attn_q_a_norm.weight`, `attn_q_b.weight`, `attn_kv_a_mqa.weight`, `attn_kv_a_norm.weight`, `attn_kv_b.weight`, `attn_output.weight`, `ffn_gate_inp.weight`, `ffn_gate_exps.weight`, `ffn_down_exps.weight`, `ffn_up_exps.weight`, `ffn_gate_shexp.weight`.
  - Config: pull `deepseek2.attention.*`, `deepseek2.expert_*` from GGUF metadata.
  - Feature `deepseek = []` in arch Cargo.toml (default-on). Forward to `crates/oxillama/Cargo.toml` (additive only).
  - Fixture: `build_minimal_deepseek_gguf()` in `crates/oxillama-gguf/src/test_utils.rs` (additive only — do not touch existing fixtures).
  - Files: `src/deepseek/mod.rs`, `src/deepseek/model.rs`, `src/deepseek/moe.rs`, `src/config.rs` (extend), `src/registry.rs` (extend), `Cargo.toml` (feature), `crates/oxillama/Cargo.toml` (feature forward), `crates/oxillama-gguf/src/test_utils.rs` (additive fixture), `tests/deepseek.rs`.
  - Out of scope: any edit to `crates/oxillama-runtime/**` or `crates/oxillama-server/**`.
  - Tests: tensor_names unit, MoE routing property test, integration forward-pass shape+finiteness via test fixture.
  - Risk: No runtime KV cache changes — latent cache is DeepSeekModel private state.

### D1 — DBRX architecture (fine-grained 16-expert MoE) (planned 2026-04-19)

- [x] New `oxillama_arch::dbrx` registered under GGUF arch id `"dbrx"`. Fine-grained MoE with 16 experts, top-4 routing. Feature `dbrx` default-on.
  - Goal: New `oxillama_arch::dbrx` registered under GGUF arch id `"dbrx"`. Fine-grained MoE with 16 experts, top-4 routing. Feature `dbrx` default-on.
  - Design: Layout: `src/dbrx/mod.rs`, `model.rs`, `config.rs`. Forward: `RMSNorm → MHA → residual → RMSNorm → MoE(16 experts, top-4) → residual`. Reuse `deepseek::moe::MoeConfig` — just different config values. Attention: standard MHA, no MLA. Tensor names: `blk.{L}.attn_norm.weight`, `blk.{L}.attn_q/k/v/output.weight`, `blk.{L}.ffn_norm.weight`, `blk.{L}.ffn_gate_inp.weight` (router), `blk.{L}.ffn_gate/down/up_exps.weight`. Config: `dbrx.expert_count=16`, `dbrx.expert_used_count=4`. Registry arm `"dbrx"`.
  - Files: `src/dbrx/mod.rs`, `model.rs`, `config.rs` (new, ~600–800 LoC total); `src/config.rs` (extend); `src/registry.rs` (extend); `Cargo.toml` (feature `dbrx = []` in default); `crates/oxillama/Cargo.toml` (feature forward additive); `crates/oxillama-gguf/src/test_utils.rs` (additive `build_minimal_dbrx_gguf()`); `tests/dbrx.rs` (new integration).
  - Prerequisites: existing MoE infrastructure (deepseek/moe.rs from v0.1.1).
  - Tests: tensor_names unit; integration forward-pass via `build_minimal_dbrx_gguf()`; shape+finiteness.
  - Risk: Router config — default to plain softmax over top-k.

### D2 — Grok-1 architecture (top-2 of 8 MoE) (planned 2026-04-19)

- [x] New `oxillama_arch::grok` registered under `"grok"`. Top-2 of 8 MoE. Feature `grok = []` default-on.
  - Goal: New `oxillama_arch::grok` registered under `"grok"`. Top-2 of 8 MoE. Feature `grok = []` default-on.
  - Design: Forward: `RMSNorm → MHA → residual → RMSNorm → MoE(8 experts, top-2) → residual`. Grok-1 uses RoPE theta=1e6 — rope_theta field in config. Config: `grok.expert_count=8`, `grok.expert_used_count=2`. Test fixture uses small hidden (32) for speed.
  - Files: mirror D1 structure in `src/grok/`; `Cargo.toml`; `crates/oxillama/Cargo.toml`; `crates/oxillama-gguf/src/test_utils.rs`; `tests/grok.rs`.
  - Tests: mirror D1.
  - Risk: Large hidden size in reference (6144) — test fixture must use small dims.

### D3 — DeepSeek-V3 sigmoid-with-bias MoE scoring (planned 2026-04-19)

- [x] Wire `ScoringMode::SigmoidWithBias` fully: sigmoid over router logits, additive per-expert bias, top-k select, sum with routed_scaling_factor. Enabled via GGUF metadata `deepseek2.scoring_func = 1`.
  - Goal: Wire `ScoringMode::SigmoidWithBias` fully: sigmoid over router logits, additive per-expert bias, top-k select, sum with routed_scaling_factor. Enabled via GGUF metadata `deepseek2.scoring_func = 1`.
  - Design: Extend `deepseek/moe.rs::compute_routing_weights` — `SigmoidWithBias` branch: `scores = sigmoid(logits + expert_bias)`; top-k select; normalise selected scores by their sum; scale by `routed_scaling_factor`. Add `e_score_correction_bias: Option<Vec<f32>>` to `MoeWeights`; load from `blk.{L}.exp_probs_b.weight`. Auto-detect: arch `"deepseek2"` + `expert_weights_scale` present + layer has `exp_probs_b` → switch to SigmoidWithBias.
  - Files: `src/deepseek/moe.rs` (extend); `src/deepseek/config.rs` or `src/config.rs` (extend); `tests/deepseek_v3.rs` (new).
  - Prerequisites: v0.1.1's DeepSeek-V2.
  - Tests: (a) `sigmoid_bias_topk_sums_to_one_after_normalisation`; (b) `sigmoid_bias_routing_vs_softmax_differs`; (c) `deepseek_v3_forward_with_bias` integration.
  - Risk: Normalisation order: sigmoid → bias add → topk → normalise selected (NOT same as softmax). moe.rs already ~620 LoC — stay under 2000.

### E1 — `SequenceState` trait extension for state-space models (planned 2026-04-19)

- [x] New trait in `oxillama-arch::common::sequence_state` that generalises `KvCacheAccess` for non-attention sequence models. Arch-internal, NOT a runtime extension.
  - Goal: New trait in `oxillama-arch::common::sequence_state` that generalises `KvCacheAccess` for non-attention sequence models. Arch-internal, NOT a runtime extension.
  - Design:
    ```rust
    pub trait SequenceState {
        fn reset(&mut self);
        fn step_position(&self) -> usize;
        fn advance(&mut self);
        fn capacity(&self) -> usize;
    }
    ```
    Default impl: newtype wrapper around `KvCacheAccess`. Mamba impl: owns `Vec<SsmLayerState>` with `(d_state, d_inner)` state tensor per layer + current token position.
  - Files: `src/common/sequence_state.rs` (new, ~150 LoC); `src/common/mod.rs` (add `pub mod sequence_state;`).
  - Out of scope: any edit to `crates/oxillama-runtime/**`.
  - Tests: shape-coherence test for default-impl wrapper; reset + advance test.

### E2 — Mamba-2 architecture (selective-scan state-space model) (planned 2026-04-19)

- [x] New `oxillama_arch::mamba2` registered under `"mamba2"`. Selective-scan SSM with learned Δ. Feature `mamba2 = []` default-on. Arch-internal state via E1.
  - Goal: New `oxillama_arch::mamba2` registered under `"mamba2"`. Selective-scan SSM with learned Δ. Feature `mamba2 = []` default-on. Arch-internal state via E1.
  - Design: Layout: `src/mamba2/mod.rs`, `model.rs`, `ssm.rs`, `conv.rs`. Mamba-2 block: (1) `x = rms_norm(hidden)`; (2) `z = x @ w_z`, `y = x @ w_in`; (3) `y = silu(conv1d(y, w_conv))`; (4) `B = y@w_B`, `C = y@w_C`, `Δ_raw = y@w_Δ`; (5) `Δ = softplus(Δ_raw + b_Δ)`; (6) selective scan: `h_t = exp(-Δ_t * exp(log_A)) * h_{t-1} + Δ_t * B_t * u_t`, `output_t = C_t * h_t`; (7) `out = silu(z) * out`; (8) `hidden += out @ w_out`. Sequential scan for v0.1.2. State: `h[d_state × d_inner]` per layer in `Mamba2LayerState`. CRITICAL: A stored as `log(A)` in GGUF — apply `exp(-Δ * exp(log_A))`.
  - Tensor names: `blk.{L}.ssm_in.weight`, `blk.{L}.ssm_conv1d.weight`, `blk.{L}.ssm_conv1d.bias`, `blk.{L}.ssm_x.weight`, `blk.{L}.ssm_dt.weight`, `blk.{L}.ssm_dt.bias`, `blk.{L}.ssm_A`, `blk.{L}.ssm_D`, `blk.{L}.ssm_out.weight`.
  - Config: `mamba2.d_state=128`, `mamba2.d_conv=4`, `mamba2.expand=2`, `mamba2.n_layer`. Registry `"mamba2"` arm. Feature `mamba2 = []` in default.
  - Files: `src/mamba2/mod.rs`, `model.rs`, `ssm.rs`, `conv.rs` (new, ~1200–1600 LoC combined); `src/config.rs`; `src/registry.rs`; `Cargo.toml`; `crates/oxillama/Cargo.toml`; `crates/oxillama-gguf/src/test_utils.rs` (additive `build_minimal_mamba2_gguf()` — hidden=16, d_state=8, d_inner=16, d_conv=4, 1 layer); `tests/mamba2.rs` (new).
  - Out of scope: `crates/oxillama-runtime/**`, `crates/oxillama-server/**`.
  - Prerequisites: E1.
  - Tests: (a) `ssm_scan_matches_reference` — 32-token sequence, tol 1e-5; (b) `conv1d_depthwise_matches_reference`; (c) `mamba2_forward_shape_and_finite` via `build_minimal_mamba2_gguf()`; (d) `sequence_state_reset_roundtrip`.
  - Risk: A stored as log(A) — MUST use `exp(-Δ * exp(log_A))`. ssm.rs should stay under 1000 LoC; split into scan.rs + state.rs if needed.

*Last updated: 2026-04-24 (v0.1.1 shipped — D1 DBRX, D2 Grok-1, D3 DeepSeek-V3 sigmoid MoE, E1 SequenceState, E2 Mamba-2 all [x]; 18 architectures, 334 tests; cross-arch fuzz harness shipped at tests/arch_fuzz.rs)*

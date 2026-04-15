# oxillama — TODO

## 1. Overview

Meta crate — the unified re-export facade for the entire OxiLLaMa workspace.
Downstream users depend on this crate to access every subcrate (`gguf`, `quant`, `arch`, `runtime`, `server`, `bench`, `gpu`) through a single namespace, so it acts as the root of OxiLLaMa's public API and the canonical entry point for cargo-add.

## 2. Status Snapshot

| Field | Value |
|-------|-------|
| Version | `0.1.0` (workspace-inherited) |
| Completion | 100% (re-export shell; contents live in subcrates) |
| Source files | 1 (`src/lib.rs`, ~55 lines) |
| Direct deps | 4 required + 3 optional subcrates |
| Public items | 0 own items — pure re-export facade |
| License | Apache-2.0 |

Default features: `server`, `bench`.
Opt-in runtime features: `gpu`, `simd-avx2`, `simd-avx512`, `simd-neon`.
Opt-in architecture features: `llama`, `qwen3`, `mistral`, `gemma`, `phi`, `command-r`, `starcoder`, `llava`.

Architecture flags intentionally forward to both `oxillama-arch` and `oxillama-runtime` so enabling a model at the meta level pulls in the full kernel + inference path without additional per-crate plumbing.

## 3. Module Map

| Path | Role |
|------|------|
| `src/lib.rs` | Sole source file. Contains crate-level rustdoc, the module-to-crate mapping table, a `no_run` quickstart doctest, and seven `pub use` re-exports (`gguf`, `quant`, `arch`, `runtime`, `server`, `bench`, `gpu`). The `server`, `bench`, and `gpu` re-exports are `#[cfg(feature = ...)]`-gated. |
| `Cargo.toml` | Declares the feature graph and optional-dep wiring. The canonical place to expose or retire a workspace feature to end users. |
| `README.md` | Crate-level landing page on crates.io — feature table and one code snippet (expand under v1.1). |

No submodules, no helpers, no binaries — intentionally thin. Any public surface addition should land in the owning subcrate, not here. This minimalism is load-bearing: it keeps the meta crate immune to cascading API breakage when a subcrate refactors.

## 4. Shipped in v0.1.0

- Unified namespaces: `oxillama::gguf`, `::quant`, `::arch`, `::runtime`, `::server`, `::bench`, `::gpu`.
- Feature propagation to subcrates:
  - Per-architecture flags (`llama`/`qwen3`/`mistral`/`gemma`/`phi`/`command-r`/`starcoder`/`llava`) forwarded to both `oxillama-arch` and `oxillama-runtime`.
  - SIMD flags (`simd-avx2`/`simd-avx512`/`simd-neon`) forwarded to `oxillama-quant`.
  - `gpu` forwards to `oxillama-gpu/gpu` and the optional `oxillama-gpu` dep.
- `server`, `bench`, `gpu` wired as optional feature-gated subcrates — zero compile or link cost when disabled.
- Quickstart doctest in `lib.rs` covering load + generate, written with proper error handling (no `unwrap()` in any code example).
- Module-level rustdoc table mapping each re-export to its underlying crate and feature gate, rendered cleanly on docs.rs.
- Subcrate-specific defaults forwarded: `mmap` for `oxillama-gguf`, `parallel` for `oxillama-quant`, `tokenizer-wasm` for `oxillama-runtime`.
- Workspace version inheritance — a single `version.workspace = true` in `Cargo.toml` keeps the facade pinned to the release train.

## 5. Known Gaps / Incomplete

- No curated `examples/` directory at the meta-crate level — users currently infer usage from the single doctest.
- No user-facing guides. The rustdoc on `lib.rs` is terse; topics like "how to load a LoRA adapter", "how to use speculative decoding", "how to run in a browser", or "how to target WebGPU" have no narrative docs anchored here.
- Crate `README.md` is minimal (~60 lines): feature table plus one code snippet, no task-oriented walkthroughs.
- No cross-crate integration tests anchored at this crate — each subcrate tests itself, so combined-feature regressions (e.g. `gpu + qwen3`) would only surface downstream.
- No benchmarks or examples demonstrating realistic feature-flag combinations.
- No `[package.metadata.docs.rs]` stanza — docs.rs builds use defaults, so optional subcrates render with feature-stub notes rather than full docs.

## 6. v1.1 Roadmap

- `examples/` directory with runnable end-to-end samples:
  - `load_and_generate.rs` — minimal GGUF load + token stream.
  - `openai_server.rs` — spin up the OpenAI-compatible server.
  - `lora_apply.rs` — load a base model plus one LoRA adapter.
  - `speculative.rs` — draft + target speculative decoding.
  - `wasm_browser/` — browser demo using `oxillama-wasm` through the facade.
  - `python_bridge/` — parity sample mirroring `oxillama-py` usage.
  - `gpu_enabled.rs` — `gpu` feature on, Q4_0 GEMV dispatched via wgpu.
- mdBook user guide (`book/`): installation, model loading, sampling, server deployment, LoRA, speculative decoding, feature-flag cookbook.
- Cookbook / recipes document (`RECIPES.md`) with copy-paste task snippets.
- `[package.metadata.docs.rs]` entry that enables `server`, `bench`, `gpu`, and representative arch/SIMD flags so docs.rs renders the full facade.
- Integration tests exercising feature-combination matrices to catch interaction regressions before release.

## 7. v2.0+ Vision

- Interactive docs site (hosted mdBook + live playground) where users tweak prompts and sampling knobs in-browser and see generation in real time.
- Video walkthroughs covering the end-to-end journey: download GGUF → quantize → serve → call from Python/TypeScript/Swift.
- Tutorial series mapped to the capability ladder (quickstart → custom sampler → LoRA stack → speculative → GPU → WASM → multi-model router) with paired recipe code.
- Screencast-paired cookbook entries — one topic per recipe, each backed by a committed example so drift is caught by CI.
- Community-contributed examples gallery curated through the meta crate, keeping the facade the canonical discovery surface for the entire ecosystem.

*Last updated: 2026-04-15 (v0.1.0 release)*

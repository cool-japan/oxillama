# oxillama — TODO

## 1. Overview

Meta crate — the unified re-export facade for the entire OxiLLaMa workspace.
Downstream users depend on this crate to access every subcrate (`gguf`, `quant`, `arch`, `runtime`, `server`, `bench`, `gpu`) through a single namespace, so it acts as the root of OxiLLaMa's public API and the canonical entry point for cargo-add.

## 2. Status Snapshot

| Field | Value |
|-------|-------|
| Version | `0.1.1` (workspace-inherited) |
| Completion | 100% (re-export shell; contents live in subcrates) |
| Source files | 1 (`src/lib.rs`, ~55 lines) |
| Direct deps | 4 required + 3 optional subcrates |
| Public items | 0 own items — pure re-export facade |
| License | Apache-2.0 |

Default features: `server`, `bench`, `dbrx`, `grok`, `mamba2`.
Opt-in runtime features: `gpu`, `simd-avx2`, `simd-avx512`, `simd-neon`.
Opt-in architecture features: `llama`, `qwen3`, `mistral`, `gemma`, `phi`, `command-r`, `starcoder`, `deepseek`, `llava`.

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
  - Per-architecture flags (`llama`/`qwen3`/`mistral`/`gemma`/`phi`/`command-r`/`starcoder`/`deepseek`/`dbrx`/`grok`/`mamba2`/`llava`) forwarded to both `oxillama-arch` and `oxillama-runtime`.
  - SIMD flags (`simd-avx2`/`simd-avx512`/`simd-neon`) forwarded to `oxillama-quant`.
  - `gpu` forwards to `oxillama-gpu/gpu` and the optional `oxillama-gpu` dep.
- `server`, `bench`, `gpu` wired as optional feature-gated subcrates — zero compile or link cost when disabled.
- Quickstart doctest in `lib.rs` covering load + generate, written with proper error handling (no `unwrap()` in any code example).
- Module-level rustdoc table mapping each re-export to its underlying crate and feature gate, rendered cleanly on docs.rs.
- Subcrate-specific defaults forwarded: `mmap` for `oxillama-gguf`, `parallel` for `oxillama-quant`, `tokenizer-wasm` for `oxillama-runtime`.
- Workspace version inheritance — a single `version.workspace = true` in `Cargo.toml` keeps the facade pinned to the release train.

## 4.1 Shipped in v0.1.1

- `dbrx`, `grok`, and `mamba2` architecture flags promoted to default features — enabled without opt-in.
- `deepseek` architecture flag added as an opt-in feature, wired to both `oxillama-arch` and `oxillama-runtime`.
- `[package.metadata.docs.rs]` stanza added: `all-features = true`, `rustdoc-args = ["--cfg", "docsrs"]`.
- `examples/openai_server.rs`, `examples/python_bridge.rs`, `examples/gpu_enabled.rs` shipped.
- Cross-crate integration tests: `tests/feature_matrix.rs` + `tests/error_types.rs` (19 tests passing).
- Re-export count: 19 passing tests in the meta crate integration suite.

## 5. Known Gaps / Incomplete

- [x] `examples/` directory: 8 runnable samples (01_load_model, 02_inference, 03_streaming, 04_lora, 05_speculative, 06_metrics, openai_server, python_bridge).
- No user-facing guides. The rustdoc on `lib.rs` is terse; topics like "how to load a LoRA adapter", "how to use speculative decoding", "how to run in a browser", or "how to target WebGPU" have no narrative docs anchored here.
- Crate `README.md` is minimal (~60 lines): feature table plus one code snippet, no task-oriented walkthroughs.
- [x] Cross-crate integration tests: `tests/feature_matrix.rs` and `tests/error_types.rs` (19 tests, all passing).
- No benchmarks or examples demonstrating realistic feature-flag combinations.
- [x] `[package.metadata.docs.rs]` stanza added: `all-features = true`, `rustdoc-args = ["--cfg", "docsrs"]`, `targets = ["x86_64-unknown-linux-gnu"]`.

## 6. v1.1 Roadmap

- `examples/` directory with runnable end-to-end samples:
  - ~~`openai_server.rs` — spin up the OpenAI-compatible server.~~ ✅ Shipped in `crates/oxillama/examples/openai_server.rs`.
  - ~~`wasm_browser/` — browser demo using `oxillama-wasm` through the facade.~~ ✅ Shipped in `crates/oxillama/examples/wasm_browser/` (M1).
  - ~~`python_bridge/` — parity sample mirroring `oxillama-py` usage.~~ ✅ Shipped in `crates/oxillama/examples/python_bridge.rs`.
  - ~~`gpu_enabled.rs` — `gpu` feature on, Q4_0 GEMV dispatched via wgpu.~~ ✅ Shipped in `crates/oxillama/examples/gpu_enabled.rs`.
- [x] Facade examples (`load_and_generate`, `lora_apply`, `speculative`) + `RECIPES.md` cookbook — mdBook deferred to next `/ultra` round (planned 2026-04-24)
  - **Goal:** Three runnable examples completing the v1.1 examples grid + an 8-recipe RECIPES.md cookbook. The 8-chapter mdBook user guide is explicitly deferred to its own `/ultra` slice to protect infra-slice quality.
  - **Design:**
    - `examples/load_and_generate.rs` (~120 LoC): load GGUF, build InferenceEngine, stream 20 tokens; --help via clap.
    - `examples/lora_apply.rs` (~150 LoC): load base model, push_lora, stream, pop_lora hot-swap demo.
    - `examples/speculative.rs` (~180 LoC): SpeculativeEngine draft+target pair, 5-token accept window, reports acceptance rate.
    - `RECIPES.md`: 8 task-oriented recipes (30-80 LoC each): (1) load+generate, (2) serve OpenAI API, (3) LoRA adapter at runtime, (4) speculative decoding, (5) snapshot+resume session (R1), (6) browser chat with oxillama-wasm, (7) resume interrupted HF pull (G1), (8) load sharded 70B model (G1).
  - **Files:** `examples/load_and_generate.rs`, `examples/lora_apply.rs`, `examples/speculative.rs`, `RECIPES.md`, `tests/recipes_doctest.rs`, `Cargo.toml` ([[example]] blocks), `TODO.md`.
  - **Tests:** `cargo check --examples -p oxillama --all-features`, `cargo run --example load_and_generate -- --help`, RECIPES.md doctest extraction via `#[doc = include_str!("../RECIPES.md")]` with `rust,no_run` fences.
- mdBook user guide (`book/`): installation, model loading, sampling, server deployment, LoRA, speculative decoding, feature-flag cookbook. (deferred to next `/ultra` round — standalone slice)
- `[package.metadata.docs.rs]` entry that enables `server`, `bench`, `gpu`, and representative arch/SIMD flags so docs.rs renders the full facade.
- Integration tests exercising feature-combination matrices to catch interaction regressions before release.

## 7. v2.0+ Vision

- Interactive docs site (hosted mdBook + live playground) where users tweak prompts and sampling knobs in-browser and see generation in real time.
- Video walkthroughs covering the end-to-end journey: download GGUF → quantize → serve → call from Python/TypeScript/Swift.
- Tutorial series mapped to the capability ladder (quickstart → custom sampler → LoRA stack → speculative → GPU → WASM → multi-model router) with paired recipe code.
- Screencast-paired cookbook entries — one topic per recipe, each backed by a committed example so drift is caught by CI.
- Community-contributed examples gallery curated through the meta crate, keeping the facade the canonical discovery surface for the entire ecosystem.

*Last updated: 2026-04-24 (F1 plan — facade examples + RECIPES.md cookbook; mdBook deferred to next /ultra round)*

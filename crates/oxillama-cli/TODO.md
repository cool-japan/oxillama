# oxillama-cli — TODO

## 1. Overview

CLI binary crate for the OxiLLaMa workspace. Entry point `oxillama` — a llama.cpp-compatible drop-in command-line front end that wires `oxillama-gguf`, `oxillama-quant`, `oxillama-arch`, `oxillama-runtime`, and (optionally) `oxillama-server` / `oxillama-bench` into a single Pure Rust executable.

Terminal leaf in the workspace dependency chain: nothing depends on this crate, so API churn is cheap. The crate is deliberately thin — argument parsing, tracing setup, and engine/server wiring only; all substantive work lives in the libraries it composes.

## 2. Status Snapshot

| Field | Value |
|---|---|
| Version | 0.1.0 (workspace) |
| Completion | 100% |
| Source files | 1 (`src/main.rs`, ~420 lines) |
| Binary name | `oxillama` |
| Subcommands | `run`, `serve`, `info`, `bench` |
| llama.cpp flag aliases | 3 primary (`-n/--n-predict`, `-c/--n-ctx`, `--temperature`) plus llama.cpp-style `--repeat-penalty`, `--min-p`, `-s/--seed`, `-t/--threads` |
| Default features | `server` |
| Optional features | `bench`, `simd-avx2`, `simd-avx512`, `simd-neon` |
| Async runtime | `tokio` (multi-thread) |
| CLI parser | `clap` v4 derive |
| Config format | `toml` (loader wired, schema pending) |

## 3. Module Map

| Path | Role |
|---|---|
| `src/main.rs` | Single-file entry point — `Cli` / `Commands` derive, `#[tokio::main]` dispatcher. |
| `src/main.rs::Commands::Run` | Inference invocation; builds `SamplerConfig` + `EngineConfig`, streams tokens via `engine.generate()`. |
| `src/main.rs::Commands::Serve` | `#[cfg(feature = "server")]` — spawns `oxillama_server::inference_worker`, binds axum listener. |
| `src/main.rs::Commands::Info` | Opens GGUF via `oxillama_gguf::GgufModel::load`, prints summary + optional `--tensors` / `--metadata`. |
| `src/main.rs::Commands::Bench` | `#[cfg(feature = "bench")]` — warmup + iteration loop, reports tokens/s. |
| `benches/inference.rs` | Criterion harness (`harness = false`) for dev-time CLI benchmarks. |
| `Cargo.toml` | `[[bin]] name = "oxillama"`, feature flags fanning out to workspace crates. |
| `README.md` | End-user install / usage snippets for the four subcommands. |

Single-file at ~420 lines — well within the 2000-line splitrs budget; split is not yet warranted. When `chat`, `completions`, `hub`, and `convert` land, `src/main.rs` will be split into `src/cli/{run,serve,info,bench,chat,...}.rs` submodules.

## 4. Shipped in v0.1.0

- Single-file `main.rs` entry point with clap v4 derive + `#[tokio::main]` dispatcher.
- `run` subcommand — streaming generation with sampler knobs (temp, top-p, top-k, min-p, repeat-penalty, seed).
- `serve` subcommand (feature = `server`) — OpenAI-compatible axum server, queue + single inference worker, model-id derived from file stem.
- `info` subcommand — GGUF summary, optional `--tensors` listing (shape + dtype + MB), optional `--metadata` key/value dump sorted by key.
- `bench` subcommand (feature = `bench`) — warmup + timed iterations, aggregate tokens/s report.
- llama.cpp drop-in flag aliases via `conflicts_with`: `-n/--n-predict`, `-c/--n-ctx`, `--temperature`.
- llama.cpp-style flags natively: `--repeat-penalty`, `--min-p`, `-s/--seed`, `-t/--threads`.
- `toml` crate wired for configuration loading (schema-less today — accepts well-formed TOML).
- `tracing-subscriber` with `RUST_LOG` env-filter fallback to `info`.
- SIMD feature passthrough (`simd-avx2`, `simd-avx512`, `simd-neon`) to `oxillama-quant`.
- Criterion `inference` bench target for dev-time profiling.
- Seed handling: `--seed 0` maps to `None` (random), non-zero seeds are deterministic.
- Explicit `anyhow::bail!` on missing model path (no panic, no `unwrap()` in production branches).
- Tokenizer auto-detection via `--tokenizer <path>` optional override.
- `serve` reuses sampler config as default for incoming requests; per-request overrides handled inside `oxillama-server`.
- Streaming stdout via closure callback `|token| print!("{token}")` — zero-allocation printing loop.
- All subcommand branches return `anyhow::Result<()>`, so errors propagate to the process exit code.

## 5. Known Gaps / Incomplete

- No interactive chat REPL — multi-turn conversations require repeated `run` invocations with manual prompt assembly.
- No shell-completion generation (bash / zsh / fish / powershell / elvish).
- No config schema or validation — toml files are parsed but not structurally checked or documented.
- No per-model profile files (e.g. `~/.config/oxillama/models/qwen3-7b.toml` with baked-in sampler defaults).
- No readline-style line editing — no input history, no Ctrl-R search, no completion.
- No conversation save/resume — token streams and KV-cache state are not persisted between invocations.
- `serve`'s `model_id` extraction falls back to `"oxillama-model"` when file stem is non-UTF8; no override flag.
- No `--config <path>` flag or `OXILLAMA_CONFIG` env var wired into clap — toml loader is staged for v1.1.
- No man-page generation (clap_mangen not yet wired).
- No `--version` detail (uses clap default; no build-hash / feature-flag banner).
- No integration test harness — current coverage is exercised via downstream `oxillama-runtime` / `oxillama-server` tests.
- No pipe-input mode (`oxillama run -` for stdin) or `--file prompt.txt` loader.
- No colorized output or progress bar for long generations; stdout is plain text only.

## 6. v1.1 Roadmap

- `oxillama chat` — interactive REPL subcommand using `rustyline` for line editing, history file (`~/.local/state/oxillama/history`), optional readline-compatible keybindings; multi-turn KV-cache reuse inside one session.
- `oxillama completions <shell>` — emit completion script via `clap_complete` for bash, zsh, fish, powershell, elvish.
- Config schema — typed `OxillamaConfig` struct with `serde` + JSON-schema export for editor tooling; clear errors on unknown keys.
- Per-model profiles — `~/.config/oxillama/models/*.toml` with sampler defaults, context size, tokenizer hints, chat template; `oxillama run --profile qwen3-7b` resolves a profile name to its toml.
- `--config <path>` flag + `OXILLAMA_CONFIG` env var, layered: CLI flag > env > profile > global defaults.
- Man-page generation via `clap_mangen` (installed to `$OUT_DIR/man/`).
- `--version --verbose` banner listing build target, enabled SIMD features, wired architectures.
- Structured error surface: map `anyhow` context into exit codes (2 = invalid args, 3 = model load, 4 = runtime).
- `oxillama run --file prompt.txt` and `oxillama run -` (stdin) for piped prompts.

## 7. v2.0+ Vision

- TUI mode — `ratatui`-based dashboard with live token-stream pane, GPU utilization chart, KV-cache heatmap, per-layer attention summary, sampler histogram; keyboard shortcuts for pausing, reseeding, switching samplers mid-generation.
- Plugin hooks — custom sampler / logits-processor callbacks invoked as shell scripts (stdin: logits, stdout: modified logits) and as WASM modules via `oxillama-wasm` for sandboxed execution.
- `oxillama hub pull <model>` — HuggingFace Hub downloads (resumable, hash-verified, GGUF-manifest-aware) using Pure Rust HTTP via `oxillama-server` dependencies; sibling `oxillama hub ls` and `oxillama hub prune`.
- Multi-model orchestration — `oxillama run --model a.gguf --model b.gguf` with arbiter sampling (majority vote, logit-average).
- Session snapshotting — `oxillama chat --save session.bin` with `oxicode`-serialized KV-cache and conversation state.
- Interactive prompt composer — in-TUI template editor with live token-count + context-fit visualization.
- `oxillama serve --multi-model` — fleet mode with per-model warm slots, request routing by requested `model` field.
- `oxillama convert` — GGUF from safetensors / HF snapshot, reusing `oxillama-gguf` writer + `oxillama-quant` kernels.
- Self-contained `scirs2-core` / `oxiblas` / `oxifft` feature-flag banner surfaced via `oxillama --about` so end users can see their sovereignty posture at a glance.

*Last updated: 2026-04-15 (v0.1.0 release)*

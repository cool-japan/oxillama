# oxillama-cli — TODO

## 1. Overview

CLI binary crate for the OxiLLaMa workspace. Entry point `oxillama` — a llama.cpp-compatible drop-in command-line front end that wires `oxillama-gguf`, `oxillama-quant`, `oxillama-arch`, `oxillama-runtime`, and (optionally) `oxillama-server` / `oxillama-bench` into a single Pure Rust executable.

Terminal leaf in the workspace dependency chain: nothing depends on this crate, so API churn is cheap. The crate is deliberately thin — argument parsing, tracing setup, and engine/server wiring only; all substantive work lives in the libraries it composes.

## 2. Status Snapshot

| Field | Value |
|---|---|
| Version | 0.1.1 (workspace) |
| Completion | 100% |
| Source files | 1 (`src/main.rs`, ~420 lines) |
| Binary name | `oxillama` |
| Subcommands | `run`, `serve`, `info`, `bench`, `chat`, `completions`, `generate-manpage`, `version --verbose` |
| llama.cpp flag aliases | 3 primary (`-n/--n-predict`, `-c/--n-ctx`, `--temperature`) plus llama.cpp-style `--repeat-penalty`, `--min-p`, `-s/--seed`, `-t/--threads` |
| Default features | `server` |
| Optional features | `bench`, `simd-avx2`, `simd-avx512`, `simd-neon` |
| Async runtime | `tokio` (multi-thread) |
| CLI parser | `clap` v4 derive |
| Config format | `toml` typed `OxillamaConfig` with validation |
| Tests | 15 passing |
| Public API | 5 items (`config` module + `exit_codes` module) |

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

With `chat`, `completions`, `generate-manpage`, and `version --verbose` now shipping, `src/main.rs` has grown beyond the original ~420 lines; if it approaches 800 lines it will be split into `src/cli/{run,serve,info,bench,chat,...}.rs` submodules via splitrs.

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

## 4.1 Shipped in v0.1.1

- `chat` subcommand — interactive REPL via `rustyline` (history, Ctrl-R, arrow keys); multi-turn KV-cache reuse; `indicatif` spinner during model load.
- `completions <shell>` subcommand — shell completion scripts via `clap_complete` for bash, zsh, fish, powershell, elvish.
- `generate-manpage --output-dir <dir>` subcommand — writes `oxillama.1` via `clap_mangen`.
- `version --verbose` subcommand — prints build target, enabled SIMD feature flags, and wired architectures.
- Typed `OxillamaConfig` struct (`serde` + JSON-schema): validation with descriptive errors on unknown keys.
- Per-model profile files (`~/.config/oxillama/models/*.toml`); `--profile <name>` resolves to a profile.
- `--config <path>` CLI flag + `OXILLAMA_CONFIG` env var; layered resolution order: CLI > env > profile > global defaults.
- Structured exit codes: 2 = invalid args, 3 = model load failure, 4 = runtime error.
- `--file <path>` and `--stdin` flags on `run` for piped prompts.
- `colored` output: cyan/bold key labels, green banners; `indicatif` spinner in `run`, `chat`, and `serve`.
- 15 passing tests (up from 5 smoke tests in v0.1.0).

## 5. Known Gaps / Incomplete

- ~~No interactive chat REPL~~ ✅ `chat` REPL with rustyline (history, Ctrl-R, arrow keys).
- ~~No shell-completion generation~~ ✅ `completions` subcommand via `clap_complete` (bash/zsh/fish/powershell/elvish).
- ~~No readline-style line editing~~ ✅ `rustyline::DefaultEditor` with persistent history at `~/.local/state/oxillama/history`.
- ~~No config schema or validation — toml files are parsed but not structurally checked or documented.~~ ✅ Typed `OxillamaConfig` struct with `serde` + JSON-schema export; unknown keys produce clear errors.
- ~~No per-model profile files (e.g. `~/.config/oxillama/models/qwen3-7b.toml` with baked-in sampler defaults).~~ ✅ Per-model profiles in `~/.config/oxillama/models/*.toml`; resolved via `--profile <name>`.
- ~~No conversation save/resume — token streams and KV-cache state are not persisted between invocations.~~ ✅ `session.rs` — `/save <path>` and `/load <path>` slash-commands in chat REPL; atomic write via tempfile+rename; oxicode-serde serialisation; SHA-256 KV-sidecar verification; schema version guard.
- `serve`'s `model_id` extraction falls back to `"oxillama-model"` when file stem is non-UTF8; no override flag.
- ~~No `--config <path>` flag or `OXILLAMA_CONFIG` env var wired into clap — toml loader is staged for v1.1.~~ ✅ `--config` flag + `OXILLAMA_CONFIG` env var wired; layered resolution: CLI flag > env > profile > defaults.
- ~~No man-page generation (clap_mangen not yet wired).~~ ✅ `generate-manpage` subcommand writes `oxillama.1` to `--output-dir`.
- ~~No `--version` detail (uses clap default; no build-hash / feature-flag banner).~~ ✅ `oxillama version --verbose` prints build target, enabled SIMD features, and wired architectures.
- ~~No integration test harness~~ ✅ Shipped: `tests/cli_smoke.rs` with 5 smoke tests covering `--help`, `--version`, `completions bash`, and failure modes.
- ~~No pipe-input mode (`oxillama run -` for stdin) or `--file prompt.txt` loader.~~ ✅ `--file <path>` and `--stdin` flags on `run`.
- ~~No colorized output or progress bar~~ ✅ Shipped: `colored` for cyan/bold key labels and green banners; `indicatif` spinner during model loading in `run`, `chat`, and `serve` subcommands.

## 6. v1.1 Roadmap

- [x] `oxillama chat` — interactive REPL subcommand using `rustyline` for line editing, history file (`~/.local/state/oxillama/history`), optional readline-compatible keybindings; multi-turn KV-cache reuse inside one session.
- [x] `oxillama completions <shell>` — emit completion script via `clap_complete` for bash, zsh, fish, powershell, elvish.
- [x] Config schema — typed `OxillamaConfig` struct with `serde` + JSON-schema export for editor tooling; clear errors on unknown keys.
- [x] Per-model profiles — `~/.config/oxillama/models/*.toml` with sampler defaults, context size, tokenizer hints, chat template; `oxillama run --profile qwen3-7b` resolves a profile name to its toml.
- [x] `--config <path>` flag + `OXILLAMA_CONFIG` env var, layered: CLI flag > env > profile > global defaults.
- [x] Man-page generation via `clap_mangen` — `oxillama generate-manpage --output-dir <dir>` writes `oxillama.1`.
- [x] `--version --verbose` banner listing build target, enabled SIMD features, wired architectures.
- [x] Structured error surface: map `anyhow` context into exit codes (2 = invalid args, 3 = model load, 4 = runtime).
- [x] `oxillama run --file prompt.txt` and `oxillama run --stdin` for piped prompts.

## 7. Shipped in v0.1.2

- [x] Conversation save/resume — `session.rs` module with `/save <path>` and `/load <path>` slash-commands wired into the chat REPL. Snapshots are written atomically (tempfile+rename) using oxicode-serde binary serialisation. Schema version guard (rejects future formats), SHA-256 KV-cache sidecar integrity check, and model-ID mismatch detection. 5 unit tests.
- [x] `oxillama hub pull/list/rm` — HuggingFace Hub subcommand group (`feature = "hub"`). Pure Rust transport via `hf-hub 0.5.0 + ureq + rustls`. `hub list` enumerates `**/*.gguf` under the platform cache dir. `hub rm` removes a cached repo directory. `hub pull` downloads via `hf-hub` with optional SHA-256 verification and auto-selects the first `.gguf` from the repo manifest. Cache dir: `~/Library/Caches/oxillama/models` (macOS) / `~/.cache/oxillama/models` (Linux) via `directories` crate. 5 unit tests (no live network needed).
- [x] TUI chat mode — `feature = "tui"` gated `crates/oxillama-cli/src/tui/` module tree using `ratatui 0.30` + `crossterm 0.29`. Full-screen layout: conversation pane (scrollable), stats sidebar (tokens/s, KV usage), status bar, and multi-line input box. Slash commands: `/save <path>`, `/load <path>`, `/clear`, `/quit`, `/help`. Activated with `oxillama chat --tui`. 6 unit tests via `ratatui::TestBackend` (no real TTY). Full async engine hand-off implemented: `tokio::task::spawn_blocking` + `std::sync::mpsc` worker in `tui/app.rs`; `Token(String)` / `GenerationDone` / `GenerationError(String)` event variants in `events.rs`; partial-assistant accumulation; live tokens/sec stats; 6 new unit tests.

## 8. v2.0+ Vision

- TUI mode — `ratatui`-based dashboard with live token-stream pane, GPU utilization chart, KV-cache heatmap, per-layer attention summary, sampler histogram; keyboard shortcuts for pausing, reseeding, switching samplers mid-generation.
- Plugin hooks — custom sampler / logits-processor callbacks invoked as shell scripts (stdin: logits, stdout: modified logits) and as WASM modules via `oxillama-wasm` for sandboxed execution.
- Multi-model orchestration — `oxillama run --model a.gguf --model b.gguf` with arbiter sampling (majority vote, logit-average).
- Interactive prompt composer — in-TUI template editor with live token-count + context-fit visualization.
- `oxillama serve --multi-model` — fleet mode with per-model warm slots, request routing by requested `model` field.
- `oxillama convert` — GGUF from safetensors / HF snapshot, reusing `oxillama-gguf` writer + `oxillama-quant` kernels.
- Self-contained `scirs2-core` / `oxiblas` / `oxifft` feature-flag banner surfaced via `oxillama --about` so end users can see their sovereignty posture at a glance.

*Last updated: 2026-04-25 (v0.1.2 — conversation save/resume, hub pull/list/rm, TUI async generation)*

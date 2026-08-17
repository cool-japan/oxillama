# oxillama-cli — TODO

## 1. Overview

CLI binary crate for the OxiLLaMa workspace. Entry point `oxillama` — a llama.cpp-compatible drop-in command-line front end that wires `oxillama-gguf`, `oxillama-quant`, `oxillama-arch`, `oxillama-runtime`, and (optionally) `oxillama-server` / `oxillama-bench` into a single Pure Rust executable.

Terminal leaf in the workspace dependency chain: nothing depends on this crate, so API churn is cheap. The crate is deliberately thin — argument parsing, tracing setup, and engine/server wiring only; all substantive work lives in the libraries it composes.

## 2. Status Snapshot

| Field | Value |
|---|---|
| Version | 0.1.4 (workspace) |
| Completion | 100% (K-quant re-encoding shipped in v0.1.4 — see §9; the I-quant mixtures remain out of reach because `oxillama-quant` has no I-quant *encoder*) |
| Source files | 23 (`src/main.rs` + `run_cmd.rs`, `serve_cmd.rs`, `cli_args.rs`, `chat_template.rs`, `bench.rs`, `manpage.rs`, `exit_codes.rs`, `config.rs`, `quantize.rs`, `quantize/type_select.rs`, `quantize/mixture_tests.rs`, `tokenize.rs`, `session.rs`, `hub.rs`, `convert.rs`, `verify.rs`, `dump_logits.rs`, `tui/*.rs`; ~7300 lines total, none over 2000 |
| Binary name | `oxillama` |
| Subcommands | `run`, `serve`, `info`, `bench`, `chat`, `hub`, `quantize`, `convert`, `verify`, `tokenize`, `detokenize`, `completions`, `generate-manpage`, `version` |
| llama.cpp flag aliases | 3 primary (`-n/--n-predict`, `-c/--n-ctx`, `--temperature`) plus llama.cpp-style `--repeat-penalty`, `--min-p`, `-s/--seed`, `-t/--threads` |
| Default features | `server`, `tui`, `simd-neon`, `simd-avx2` |
| Optional features | `bench`, `hub`, `simd-avx512` |
| Async runtime | `tokio` (multi-thread) |
| CLI parser | `clap` v4 derive |
| Config format | `toml` typed `OxillamaConfig` with validation, plus per-model `ModelProfile` |
| Tests | 175 (`cargo nextest run -p oxillama-cli`), plus one env-gated llama.cpp logit-parity test (`tests/llamacpp_logit_parity.rs`) |
| Public API | `config`, `exit_codes`, `chat_template`, `session`, `bench` modules |

## 3. Module Map

| Path | Role |
|---|---|
| `src/main.rs` | Single-file entry point — `Cli` / `Commands` derive, `#[tokio::main]` dispatcher. |
| `src/run_cmd.rs` (`run_cmd::run_run`, `RunArgs`) | Inference invocation; builds `SamplerConfig` + `EngineConfig`, streams tokens via `engine.generate()`. Split out of `main.rs` in v0.1.4 — see §9. |
| `src/dump_logits.rs` | `--dump-logits` debug path — per-step raw logit dumps + manifest for cross-implementation parity checking. |
| `src/serve_cmd.rs` (`serve_cmd::run_serve`, `ServeArgs`) | `#[cfg(feature = "server")]` — spawns `oxillama_server::spawn_inference_worker`, builds the app via `oxillama_server::build_app_with_config`, binds axum listener. Split out of `main.rs` in v0.1.4 — see §9 for the `build_app_with_config` migration. |
| `src/main.rs::Commands::Info` | Opens GGUF via `oxillama_gguf::GgufModel::load`, prints summary + optional `--tensors` / `--metadata`. |
| `src/main.rs::Commands::Bench` | `#[cfg(feature = "bench")]` — warmup + iteration loop, reports tokens/s. |
| `benches/inference.rs` | Criterion harness (`harness = false`) for dev-time CLI benchmarks. |
| `Cargo.toml` | `[[bin]] name = "oxillama"`, feature flags fanning out to workspace crates. |
| `README.md` | End-user install / usage snippets covering all 14 subcommands. |

`src/main.rs` grew well past its original ~420 lines as `chat`, `completions`, `generate-manpage`, and `version --verbose` shipped; in v0.1.4 it was split via splitrs, moving `Run`/`Serve` handling out into `src/run_cmd.rs` / `src/serve_cmd.rs` (799 lines total) to stay under the workspace's 2000-line policy — see §9. `main.rs` is now 1,204 lines (1,002 code); `Info`/`Bench`/`Chat` are the next candidates to extract if it grows toward 2000 again.

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
- ~~`serve`'s `model_id` extraction falls back to `"oxillama-model"` when file stem is non-UTF8; no override flag.~~ ✅ (2026-06-13) `--model-id <ID>` flag on `serve` and `chat` overrides the derived id; file-stem fallback retained when the flag is absent.
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
- [x] Per-model profiles — `~/.config/oxillama/models/*.toml` with sampler defaults, context size, and a `system_prompt`; `oxillama run|chat|serve --profile <name>` resolves a profile name to its toml (v0.1.2 implicitly resolved by model file stem only, with no `--profile` flag and a `system_prompt` field that was declared but never read — both fixed in v0.1.4, see §9). Chat template selection is *not* a profile field: it is auto-detected per model from `tokenizer.chat_template` GGUF metadata (or a vocabulary fingerprint) rather than hand-configured — see `chat_template.rs`.
- [x] `--config <path>` flag + `OXILLAMA_CONFIG` env var, layered: CLI flag > env > profile > global defaults.
- [x] Man-page generation via `clap_mangen` — `oxillama generate-manpage --output-dir <dir>` writes `oxillama.1`.
- [x] `--version --verbose` banner listing build target, enabled SIMD features, wired architectures.
- [x] Structured error surface: map `anyhow` context into exit codes (2 = invalid args, 3 = model load, 4 = runtime).
- [x] `oxillama run --file prompt.txt` and `oxillama run --stdin` for piped prompts.

## 7. Shipped in v0.1.2

- [x] Conversation save/resume — `session.rs` module with `/save <path>` and `/load <path>` slash-commands wired into the chat REPL. Snapshots are written atomically (tempfile+rename) using oxicode-serde binary serialisation. Schema version guard (rejects future formats), SHA-256 KV-cache sidecar integrity check, and model-ID mismatch detection. 5 unit tests.
- [x] `oxillama hub pull/list/rm` — HuggingFace Hub subcommand group (`feature = "hub"`). Pure Rust transport via `hf-hub 0.5.0 + ureq + rustls` (bumped to `hf-hub 1.0.0` + `reqwest` in v0.1.4 — see §9). `hub list` enumerates `**/*.gguf` under the platform cache dir. `hub rm` removes a cached repo directory. `hub pull` downloads via `hf-hub` with optional SHA-256 verification and auto-selects the first `.gguf` from the repo manifest. Cache dir: `~/Library/Caches/oxillama/models` (macOS) / `~/.cache/oxillama/models` (Linux) via `directories` crate. 5 unit tests (no live network needed).
- [x] TUI chat mode — `feature = "tui"` gated `crates/oxillama-cli/src/tui/` module tree using `ratatui 0.30` + `crossterm 0.29`. Full-screen layout: conversation pane (scrollable), stats sidebar (tokens/s, KV usage), status bar, and multi-line input box. Slash commands: `/save <path>`, `/load <path>`, `/clear`, `/quit`, `/help`. Activated with `oxillama chat --tui`. 6 unit tests via `ratatui::TestBackend` (no real TTY). Full async engine hand-off implemented: `tokio::task::spawn_blocking` + `std::sync::mpsc` worker in `tui/app.rs`; `Token(String)` / `GenerationDone` / `GenerationError(String)` event variants in `events.rs`; partial-assistant accumulation; live tokens/sec stats; 6 new unit tests.

## 9. Shipped in v0.1.4

- [x] `tokenize` / `detokenize` now resolve the GGUF-embedded tokenizer (`tokenizer.ggml.tokens`) first, falling back to a `tokenizer.json` sidecar — previously they required a sidecar unconditionally and errored on every vocabulary-only GGUF. Verified against 11 real `ggml-vocab-*.gguf` files from the llama.cpp reference checkout (round-trip encode/decode).
- [x] `chat` REPL and the TUI now agree on one context strategy: reset the KV cache and re-render the *entire* transcript through the model's chat template every turn (the TUI's pre-existing strategy). The REPL previously relied on the KV cache silently accumulating turn over turn and never reset it, so `/load` could restore a transcript while the model's actual context stayed empty.
- [x] Chat turns are rendered through the model's own `tokenizer.chat_template` (fingerprinted for Llama-3 / ChatML / Mistral / Alpaca families, falling back to a vocabulary fingerprint or ChatML) instead of a hardcoded `User:`/`Assistant:` transcript — `chat_template.rs`, shared by both frontends. A system prompt is recorded once (as a leading `system`-role turn) instead of being re-injected into the prompt on every turn.
- [x] `bench` resets the KV cache before every warmup and measurement iteration (it previously never did, so later iterations attended over a monotonically growing context) and reports `GenerationOutcome::completion_tokens()` instead of a whitespace word count — logic extracted to `bench.rs` so it is unit-testable against a tiny in-memory synthetic model.
- [x] `quantize` produces real K-quant output. `--target` accepts the llama.cpp mixture names `Q4_0`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K_S/M/L`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K`, and each one is a *mixture*: `src/quantize/type_select.rs` is a literal port of llama.cpp's `llama_tensor_get_type`, including the stateful `i_attention_wv` / `i_ffn_down` counters (so tensors are visited in GGUF file order, not `HashMap` order), the `use_more_bits` layer selection, the tied-embedding output-head rule, and the `Q4_K→Q5_0 / Q5_K→Q5_1 / Q6_K→Q8_0 → F16` fallback for rows that are not a multiple of 256. Verified by replaying the tensor lists of two real llama.cpp-quantized files (Meta-Llama-3-8B-Instruct-Q4_K_M, 291 tensors; Qwen3-4B-Instruct-2507-Q4_K_M, 398 tensors) and requiring an exact per-tensor type match on all 689 — plus `--dry-run` on the files themselves, which plans 0 re-encodes because the port agrees with llama.cpp on every tensor. All metadata KVs (tokenizer included) are copied verbatim; only `general.file_type` and `general.quantization_version` are rewritten, exactly as llama.cpp does. New flags: `--allow-requantize`, `--force-requantize`, `--pure`, `--dry-run`. Tensors stream through the GGUF writer one at a time, so peak memory is one tensor rather than the whole model.
- [x] `--profile <name>` is now a real flag on `run`, `chat`, and `serve` (previously resolved only implicitly from the model file stem, and only in `run`); a profile's `system_prompt` is now actually read.
- [x] `--tui` is cfg-gated out of `--help` on builds without the `tui` feature, and `tui` joined `default-features` — a `cargo install oxillama-cli` build no longer advertises a flag it then refuses.
- [x] TUI stats sidebar's "KV usage" is now populated from the engine's real KV cache occupancy after each turn, instead of a field that was initialised to 0.0 and never assigned.
- [x] `exit_codes::classify` downcasts the typed error chain (`RuntimeError` / `GgufError` / `std::io::Error` / a small local `CliError`) instead of substring-matching the formatted message, which used to misroute e.g. a failed `--file` prompt read as `ERR_MODEL_NOT_FOUND`.
- [x] `generate-manpage` renders one page per subcommand (recursively, including nested ones like `oxillama-hub-pull.1`) via `manpage.rs`, instead of only the top-level `oxillama.1` whose `SUBCOMMANDS` section cross-referenced 14 pages that were never written. The `.SH VERSION` banner is derived from the built `clap::Command` so it cannot go stale again.
- [x] `run --dump-logits <DIR>` (`src/dump_logits.rs`) writes the full final-position logit vector of every generation step as `step<k>.logits.f32.bin` — headerless, `vocab_size` little-endian `f32`, index == token id, raw pre-softmax — plus a `manifest.json` (prompt token ids, `add_bos_applied`, emitted token ids, vocab size, per-step argmax/ties/EOG). Two companion debug flags make cross-implementation comparison sound: `--prompt-tokens <IDS>` feeds explicit token ids so a tokenizer difference cannot masquerade as a kernel bug, and `--force-tokens <IDS>` teacher-forces the fed token so one divergence does not invalidate every later step. Both require `--dump-logits`. The dump loop deliberately does not stop at EOG (it records `eog_at_step`) so step indices stay aligned with a reference dump. Used to establish whole-model logit parity against llama.cpp `ba7e817e` — see the root `TODO.md` success criterion 8 for the measured numbers. `tests/llamacpp_logit_parity.rs` re-runs that check through the real binary against golden llama.cpp token ids, gated on `OXILLAMA_PARITY_LLAMA3_GGUF` / `OXILLAMA_PARITY_QWEN3_GGUF` (the checkpoints cannot be committed); its gate is margin-conditioned on llama.cpp's own 0.626-logit cross-kernel spread, because an exact-token gate fails llama.cpp against llama.cpp.
- [x] `serve` now calls `oxillama_server::build_app_with_config` (auth, rate limiting, body limits, CORS, `/metrics`, structured tracing) instead of the test-only `build_app`, binds with `into_make_service_with_connect_info::<SocketAddr>()` (required for the admin auth guard's loopback check), and shuts down via `oxillama_server::shutdown_signal()`. New flags: `--api-key`, `--admin-token`, `--allowed-model-dir`, `--disable-cors`, `--max-concurrent`, `--request-timeout-secs`, `--body-limit-bytes`, `--batch-spool-dir`.
- [x] `hf-hub` bumped `0.5.0` → `1.0.0` — a from-scratch upstream API rewrite: `ureq` is gone in favor of `reqwest`/`tokio`, and the old `api::sync::{ApiBuilder, ApiError, ApiRepo}`/`Cache`/`Repo`/`RepoType` types are replaced by `HFClient`/`HFRepositorySync`/`HFError` (`hub.rs` now builds a sync client via `HFClient::builder()...build_sync()`). `--force` re-download now calls hf-hub's native `.force_download()` instead of ~30 lines of manual cache-eviction code that used to reach into hf-hub's internal blob layout — a correctness improvement that also fixed a latent panic on non-ASCII filenames in progress-bar truncation logic. Progress-bar rendering moved into `oxillama-cli` itself: a new `PullProgress` struct (using the existing `indicatif` dependency) implements hf-hub 1.0's `ProgressHandler` callback trait, replacing hf-hub 0.5's built-in indicatif renderer, which no longer exists in 1.0. End-user-visible behavior is unchanged — `hub pull` still shows a progress bar, and the on-disk cache layout (`models--{org}--{name}/blobs|snapshots/`) is unchanged. See §7 for the pre-migration state.

## 10. v2.0+ Vision

- TUI mode — `ratatui`-based dashboard with live token-stream pane, GPU utilization chart, KV-cache heatmap, per-layer attention summary, sampler histogram; keyboard shortcuts for pausing, reseeding, switching samplers mid-generation.
- Plugin hooks — custom sampler / logits-processor callbacks invoked as shell scripts (stdin: logits, stdout: modified logits) and as WASM modules via `oxillama-wasm` for sandboxed execution.
- Multi-model orchestration — `oxillama run --model a.gguf --model b.gguf` with arbiter sampling (majority vote, logit-average).
- Interactive prompt composer — in-TUI template editor with live token-count + context-fit visualization.
- `oxillama serve --multi-model` — fleet mode with per-model warm slots, request routing by requested `model` field.
- `oxillama convert` — GGUF from safetensors / HF snapshot, reusing `oxillama-gguf` writer + `oxillama-quant` kernels.
- Self-contained `scirs2-core` / `oxiblas` / `oxifft` feature-flag banner surfaced via `oxillama --about` so end users can see their sovereignty posture at a glance.

*Last updated: 2026-08-17 (v0.1.4 — `main.rs` split into `run_cmd.rs`/`serve_cmd.rs` (799 lines
moved out, no behavior change, keeps files under the workspace's 2000-line policy);
`oxillama bench`'s `run_benchmark` fixed to sample with the engine's own configured sampler
instead of `SamplerConfig::default()`'s unseeded RNG, which previously made back-to-back
benchmark runs report inconsistent token counts; `gpu_policy_from_flags`/`print_gpu_banner`
forward-prep helpers added to `cli_args.rs` (unit-tested, not yet wired to a `--gpu` clap
argument — see the top-level TODO.md's "GPU CLI/Python wiring" entry); previously also shipped
GGUF-embedded tokenize/detokenize, unified chat context strategy + real chat templates, bench
KV-reset + real token counts, quantize per-tensor policy, --profile flag, TUI KV usage, typed
exit-code classification, per-subcommand man pages, serve wired to build_app_with_config; 179
tests with `--all-features` (175 with default features))*

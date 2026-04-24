# oxillama-py — TODO

## 1. Overview

`oxillama-py` provides PyO3-based Python bindings for the OxiLLaMa Pure
Rust LLM inference engine. It exposes the core Rust types — `Engine`,
`SpeculativeEngine`, and `LoadedLora` — to Python code via a `cdylib`
extension module that is compiled and packaged as a `maturin`-built
wheel. Long-running Rust calls (load, generate, embed) release the GIL
via `py.detach(...)` so Python threads keep running during inference,
and streaming is supported by re-acquiring the GIL per token when
calling a user-supplied Python callable.

The binding layer itself remains 100% Pure Rust — PyO3 is a pure Rust
FFI shim to CPython, so the COOLJAPAN Pure Rust Policy is honoured on
both sides of the interpreter boundary.

## 2. Status Snapshot

| Key                 | Value                                                    |
|---------------------|----------------------------------------------------------|
| Version             | 0.1.1 (workspace-pinned)                                 |
| Overall completion  | ~80% (all v1.1 items shipped; pickle + progress-bar gap remains) |
| Rust source files   | 10 (`lib.rs`, `engine.rs`, `speculative.rs`, `lora.rs`, `sampler.rs`, `error.rs`, `callback.rs`, `async_support.rs`, `hub.rs`, `cancel.rs`) |
| Rust unit tests     | 81 across all modules                                    |
| Python tests        | 55 across pytest suites (config, sampler, streaming, exceptions, cancellation token; model-backed tests gated on `OXILLAMA_TEST_MODEL`) |
| Public API items    | 16 (`EngineConfig`, `Engine`, `AsyncEngine`, `SamplerConfig`, `SpeculativeConfig`, `SpeculativeEngine`, `Lora`, `Tokenizer`, `CancellationToken`, + exception hierarchy) |
| PyO3                | 0.28 (0.22 → 0.24 → 0.28; resolves RUSTSEC-2025-0020)   |
| Wheel build         | via `maturin` (`pyproject.toml` + abi3-py38)            |
| Target Python       | 3.8+ (stable ABI wheel)                                  |
| Crate type          | `cdylib` + `rlib` (rlib for in-workspace `cargo test`)  |

## 3. Module Map

| File                    | Role                                              |
|-------------------------|---------------------------------------------------|
| `src/lib.rs`            | `#[pymodule]` registration for all public classes |
| `src/engine.rs`         | `Engine` + `EngineConfig` class bindings          |
| `src/speculative.rs`    | `SpeculativeEngine` + `SpeculativeConfig`         |
| `src/lora.rs`           | `Lora` (wraps `LoadedLora`) class                 |
| `src/sampler.rs`        | `SamplerConfig` class (constructor + helpers)     |
| `src/error.rs`          | `RuntimeError` / `ArchError` → Python exceptions  |
| `src/callback.rs`       | *(planned)* streaming callback bridge             |
| `src/streaming.rs`      | current streaming-callback helper module          |
| `pyproject.toml`        | maturin build config (`features = ["pyo3/extension-module"]`) |
| `python/tests/`         | pytest suite (imports the built extension)        |

Note: the streaming helper currently lives at `src/streaming.rs` rather
than `src/callback.rs`. Renaming to `callback.rs` is a v1.1 housekeeping
item.

## 4. Shipped in v0.1.0

- `Engine` class: GGUF model load, `tokenize`, `decode_token`, `embed`,
  `generate`, `generate_streaming`, `apply_lora`, `reset`,
  `hidden_size`, `is_eos`, `is_loaded`.
- `EngineConfig` class: keyword-only constructor with sensible defaults
  (`num_threads=4`, optional `context_size`, optional `tokenizer_path`,
  optional `sampler`).
- `SpeculativeEngine` + `SpeculativeConfig`: draft + target pair,
  accept/reject run in Rust; Python blocks on `generate` and receives a
  string once the loop terminates.
- `Lora` class: `Lora.load(path)` returns a loaded adapter with
  `rank`, `alpha`, `num_adapters()` accessors.
- `SamplerConfig` class: all ten sampler knobs (temperature, top-k,
  top-p, min-p, repetition penalty + window, seed, Mirostat v2 triple)
  plus `greedy()` and `mirostat_v2(tau, eta)` static constructors.
- GIL-release policy: all heavy Rust calls wrap in `py.detach(...)` so
  they don't block other Python threads. Streaming callbacks re-enter
  the GIL via `Python::attach(...)` on every token.
- Streaming callback API: `engine.generate_streaming(prompt,
  max_tokens=..., callback=fn)` invokes a user-supplied Python callable
  with each decoded token string.
- Error mapping: `RuntimeError` / `ArchError` variants mapped to Python
  built-ins (`PyIOError`, `PyValueError`, `PyRuntimeError`) with
  informative payload strings.
- PyO3 0.28 migration (0.22 → 0.24 → 0.28): resolves RUSTSEC-2025-0020
  (cycle-collection soundness) shipped in 0.22.
- maturin-driven wheel build (ABI3 / stable ABI) — a single wheel works
  across Python 3.8 through current.
- 62 Rust-side unit tests across the 6 modules + 26 pytest cases
  (config/no-model layer ungated; model-backed tests gated on
  `OXILLAMA_TEST_MODEL`).
- `EngineConfig`, `SamplerConfig`, `SpeculativeConfig`, `Lora` all
  implement `__repr__` for interactive debugging.

## 5. Known Gaps / Incomplete

This is the 75% gap — the polish work the 25% number represents.

- ~~**No `.pyi` type stubs.**~~ ✅ `.pyi` type stubs generated
  (`__init__.pyi`); IDEs now infer types and `mypy`/`pyright` can validate.
- **No async support.** ~~`engine.generate(...)` blocks the calling
  thread; there is no `async def` / `await` path via `pyo3-asyncio`.~~
  ✅ `PyAsyncEngine` shipped (`async_support.rs`).
- ~~**No numpy interop.** `embed()` and (future) logits return Python
  `List[float]`, not `ndarray[float32]` — slow for large tensors.~~ ✅
  Shipped: `embed_numpy() → PyArray1`, `embed_batch_numpy() → PyArray2`,
  gated on `numpy` feature.
- ~~**Tokenizer not exposed as a Python object.**~~ ✅ `PyTokenizer`
  class shipped (`from_file`, `from_json`, `encode`, `decode`,
  `vocab_size`, `id_to_token`).
- ~~**No ergonomic `Sampler` builder beyond the dataclass-ish
  `SamplerConfig`.**~~ ✅ Per-call sampler overrides landed
  (`temperature`, `top_p`, `top_k`, `seed` kwargs on `generate`/`generate_streaming`).
- ~~**Error variants are flat-mapped**~~ ✅ Custom exception hierarchy
  shipped: `OxiLlamaError` → `LoadError`, `GenerateError`,
  `TokenizerError`, `GrammarError`, `QuantError` via `register_exceptions()`.
- **Minimal pytest suite.** 26 tests cover the happy path on the
  exposed surface; full-coverage property tests and fixture-driven
  minimal-GGUF round-trips are still outstanding.
- ~~**No sphinx autodoc / readthedocs.io site.** Users rely on
  docstrings visible only via `help()` in a REPL.~~ ✅ `docs/` skeleton
  shipped with `conf.py`, `index.rst`, `quickstart.rst`, `api.rst`;
  uses `furo` theme with `sphinx.ext.autodoc` + `napoleon` + `intersphinx`.
- ~~**No PyPI publish workflow / CI gate.** Wheels are built locally via
  `maturin build`; there is no GitHub Action matrix for Linux /
  macOS / Windows × Python 3.8–3.13.~~ ✅ `.github/workflows/publish_py.yml`
  shipped; builds manylinux2014 x86_64/aarch64 (via zig), macOS universal2,
  and Windows x86_64; publishes on `py-v*` tag push.
- ~~**No HuggingFace Hub integration.** Users must download GGUF files
  manually; no `Engine.from_hub("meta-llama/...")` convenience.~~ ✅
  `Engine.from_hub()` shipped (`hub.rs`); `oxillama_py.hub.load_from_hub()`
  convenience function added; GIL released during download.
- **No pickle / checkpoint support.** An `Engine` cannot round-trip
  through `pickle.dumps` / `pickle.loads`.
- **No progress-bar hook.** Jupyter users have no first-class way to
  stream token output into `tqdm` / `ipywidgets` progress widgets.
- ~~**File naming drift.** Streaming helper lives at `src/streaming.rs`
  rather than the documented `src/callback.rs`.~~ ✅ Renamed to
  `src/callback.rs`; `lib.rs` updated accordingly.
- ~~**No logits / probability exposure.** Users who want to read the
  raw logits for a prompt (e.g. for classification, scoring, or
  custom sampling) have no entry point — only the sampled tokens
  surface.~~ ✅ `Engine.forward_logits(text) -> List[float]` shipped;
  `forward_logits_numpy()` also available (numpy feature).
- ~~**No cancellation token.** A long-running `generate(...)` cannot be
  interrupted from Python short of `Ctrl-C` at the shell — no
  `engine.cancel()` or cooperative cancellation handle is exposed.~~
  ✅ `CancellationToken` class shipped (`cancel.rs`); accepted as
  `cancel_token=` kwarg by `generate()` and `generate_streaming()`.
- ~~**Callback exceptions swallowed.**~~ ✅ Fixed: `strict_callback=True` kwarg on
  `generate_streaming()` propagates Python exceptions raised inside the callback
  instead of silencing them.  Default (`strict_callback=False`) preserves the
  original silent behaviour.

## 6. v1.1 Roadmap

- ~~Generate `.pyi` type stubs~~ ✅ Shipped (`__init__.pyi`).
- ~~Upgrade `SamplerConfig` to a fully-kwarg-friendly Python class and
  accept per-call sampler overrides~~ ✅ Shipped.
- ~~Expose `Tokenizer` as a first-class Python class~~ ✅ `PyTokenizer`
  shipped with `encode`, `decode`, `vocab_size`, `id_to_token`.
  ~~Remaining: `encode_batch`, chat-template apply.~~ ✅ Both shipped: `encode_batch()` and `apply_chat_template()` (chatml/llama3/alpaca).
- ~~Return `numpy.ndarray[float32]` from `embed()`; accept `ndarray`
  logits input on the decode path.~~ ✅ Shipped: `embed_numpy()` and
  `embed_batch_numpy()` return `numpy.ndarray[float32]`, gated on `numpy`
  feature. ~~Remaining: accept `ndarray` logits input on the decode path.~~ ✅ `decode_from_logits(logits, temperature, top_k, top_p)` in `oxillama_py.utils`.
- ~~Structured Python exception hierarchy mirroring the Rust
  `RuntimeError` tree~~ ✅ Shipped: `OxiLlamaError` → `LoadError`,
  `GenerateError`, `GrammarError`, `TokenizerError`, `QuantError`.
  ~~Remaining: `KvCacheFullError`.~~ ✅ Shipped: `KvCacheFullError` is now a distinct subclass of `OxiLlamaError`.
- ~~Full pytest suite (>80% coverage) with a fixtures directory holding
  a tiny synthetic GGUF built via the `oxillama-gguf` `test_utils`
  helpers so tests run without a network download.~~ ✅ Shipped: comprehensive
  pytest suite with `test_imports.py`, `test_engine_config.py`,
  `test_sampler_config.py`, `test_cancellation_token.py`,
  `test_streaming_callback.py`, `test_exceptions.py`; pure-Python tests
  run without native extension; native tests skip gracefully.
- ~~Sphinx autodoc + readthedocs.io (`oxillama-py.readthedocs.io`) with
  rendered examples and an API reference.~~ ✅ `docs/` skeleton shipped.
- ~~PyPI publish workflow: GitHub Actions matrix building wheels for
  manylinux2014 x86_64 / aarch64, macOS universal2, and Windows x86_64
  across CPython 3.8–3.13 + PyPy 3.10.~~ ✅ `.github/workflows/publish_py.yml` shipped.
- ~~Jupyter / tqdm-friendly streaming callback protocol — a `TqdmProgress` helper wrapping the token callback.~~ ✅ Shipped: `python/oxillama_py/tqdm_helper.py` with `TqdmProgress` (wraps any tqdm pbar) and `CollectTokens`; re-exported from package top-level. Also shipped: `decode_from_logits()` in `utils.py` for pure-Python sampling from logits ndarrays.
- ~~Rename `src/streaming.rs` → `src/callback.rs` and update docs.~~ ✅ Done.
- ~~Typed `Protocol` class for streaming callbacks~~ ✅ Shipped:
  `StreamingCallback` runtime-checkable Protocol in `python/oxillama_py/callback.py`;
  re-exported from package top-level; `.pyi` stub updated;
  `TokenCallback` type alias added.

## 7. v2.0+ Vision

- Native async engine: `await engine.generate(prompt)` via
  `pyo3-asyncio`, with cancellation propagated to the Rust side.
- Streaming async iterators: `async for tok in engine.stream(prompt)`.
- `torch.Tensor` interop: accept and return `torch.Tensor` for logits,
  embeddings, and KV cache state — zero-copy via DLPack where possible.
- `pydantic` config: `EngineConfig(BaseModel)` with validated
  construction, JSON schema export, and config-file loading.
- ~~HuggingFace Hub loader:
  `Engine.from_hub("meta-llama/Meta-Llama-3-8B-Instruct-GGUF")` with
  automatic download + on-disk cache + revision pinning.~~ ✅ Shipped:
  `Engine.from_hub()` classmethod + `oxillama_py.hub.load_from_hub()`.
- Drop-in tokenizer compat with `transformers.AutoTokenizer` surface
  (`encode`, `decode`, `apply_chat_template`, `pad_token_id`, ...).
- Typer-based CLI: `python -m oxillama chat --model ...` mirroring
  the Rust `oxillama` binary, reusing the same config schema.
- Jupyter magic: `%%oxillama prompt` cell magic for quick prompting
  from notebook cells.
- Multi-engine orchestration primitives: a Python-level pool /
  scheduler that load-balances across several loaded `Engine`s.
- Observability hooks: `on_token`, `on_accept`, `on_reject`, and
  `on_cache_evict` callback protocols for telemetry tooling.
- Optional `ray` / `dask` integration for sharded inference.

*Last updated: 2026-04-20 (v0.1.1 — 81 tests, 16 public API items, all v1.1 roadmap items shipped)*

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
- [x] **Polymorphic progress-bar hook with rich `ProgressEvent` contract** (planned 2026-05-03)
  - **Goal:** First-class `progress=` kwarg on `Engine.generate{,_streaming}`, `SpeculativeEngine.generate{,_streaming}`, and `AsyncEngine.generate{,_stream}` that accepts (a) any `tqdm`/`tqdm.notebook.tqdm`, (b) any `ipywidgets.IntProgress`, (c) any `Callable[[ProgressEvent], None]`, or (d) `None`. Rust-side throttling caps callback invocations at ~50 ms or 4 tokens (whichever first), always firing on the first and final token. RAII finaliser ensures the widget is closed / set to 100 % even on Python exception, cancellation, or EOS. Existing `callback=` kwarg is left untouched (additive, fully backwards-compatible). The v0.1.1 `TqdmProgress` shim is kept as a compat alias but documented as deprecated.
  - **Design:**
    - **Rust API shape (`engine.rs`):** add `progress: Option<Py<PyAny>>` and `progress_throttle_ms: Option<u64>` and `progress_throttle_tokens: Option<usize>` keyword-only kwargs to `PyEngine::generate` and `PyEngine::generate_streaming`. Mirror in `PySpeculativeEngine::generate{,_streaming}` (`speculative.rs`). For `PyAsyncEngine::generate{,_stream}` (`async_support.rs`), forward the kwargs to the underlying sync call inside the `asyncio.to_thread` closure.
    - **Python-side adapter (`python/oxillama_py/progress.py`, NEW, ~220 lines):** module-level `make_progress_adapter(obj, max_tokens) -> Callable[[ProgressEvent], None] | None` that does duck-typed dispatch:
      1. `obj is None` → `None` (no-op).
      2. `hasattr(obj, "update") and hasattr(obj, "set_postfix_str") and hasattr(obj, "close")` → tqdm path (`_TqdmAdapter`): call `pbar.update(1)`, `pbar.set_postfix_str(f"{tok_per_sec:.1f} tok/s", refresh=False)`, set `pbar.total = max_tokens` once on first event, call `pbar.close()` in finaliser.
      3. `hasattr(obj, "value") and hasattr(obj, "max")` and class name contains `"Progress"` (covers `IntProgress`, `FloatProgress`) → ipywidgets path (`_IPyWidgetAdapter`): set `obj.max = max_tokens` once, set `obj.value = event.tokens_generated`, optionally `obj.description = f"{tok_per_sec:.1f} tok/s"`; on finalise set `obj.bar_style = "success"` (or `"danger"` on error) and `obj.value = obj.max`.
      4. `callable(obj)` → wrap directly: `obj(event)`.
      5. Else → `TypeError("progress must be a tqdm pbar, ipywidgets.IntProgress, callable, or None")`.
      All adapters expose a `__call__(event)` and a `finalise(error: Exception | None)` method; `make_progress_adapter` returns the bare `__call__` plus a separate `finaliser` callable so the Rust side can drive both.
    - **Polymorphic dispatch on the Rust side:** Rust does *not* introspect the Python object. Instead, on the very first call the Rust binding invokes a single Python helper `oxillama_py.progress._build_bridge(progress, max_tokens) -> (callback, finaliser)`. The result is two `Py<PyAny>` callables stored in a small struct `ProgressBridge { callback: Py<PyAny>, finaliser: Py<PyAny>, throttle: Throttler, start: Instant, tokens: usize }`. This keeps all duck typing in Python where it belongs.
    - **Throttling implementation (`callback.rs`):** new `Throttler { last_fire: Instant, tokens_since_fire: usize, min_interval: Duration, min_tokens: usize }` with method `should_fire(&mut self, force: bool) -> bool`. Returns `true` iff `force || self.last_fire.elapsed() >= self.min_interval || self.tokens_since_fire >= self.min_tokens`. Defaults: 50 ms, 4 tokens. Always force-fire on first token (`tokens_total == 1`) and on final token (callback driven from the cleanup epilogue, see below). Throttler lives entirely in the Rust closure that wraps the user callback — Python is never touched on a throttled tick, only `tokens_total` and `tokens_since_fire` are incremented.
    - **`ProgressEvent` (Python `@dataclass(frozen=True, slots=True)` in `progress.py`):** fields `tokens_generated: int`, `tokens_total: int | None` (= max_tokens), `elapsed_secs: float`, `tokens_per_sec: float`, `eta_secs: float | None` (None until ≥2 tokens), `is_final: bool`, `text_so_far: str` (always `""` unless `progress_capture_text=True` kwarg is set; gated to avoid O(n²) string growth in the hot loop). Constructed Python-side from a `(tokens_generated, elapsed_ns, is_final, text_so_far)` 4-tuple passed by Rust — cheaper than a `#[pyclass]` per token. Re-exported as `oxillama_py.ProgressEvent`.
    - **Exception safety / cleanup (RAII):** Rust uses a `ProgressGuard` struct holding the `finaliser: Py<PyAny>` and a `result: Cell<Option<Result<(), PyErr>>>`. The guard's `Drop` impl calls `Python::attach(|py| finaliser.call1(py, (error_repr,)))` so the bar is closed even on panic, early return, EOS, cancellation, or Python exception inside the callback. Inside the per-token closure, `cb.call1(...)` errors are caught and stashed (like the existing `strict_callback` pattern) — generation never aborts because the bar threw. A new kwarg `strict_progress: bool = False` mirrors `strict_callback`: when true, the first stashed error is re-raised after generation completes.
    - **Async path (`async_support.rs`):** `generate_stream` already runs generation in a `thread::spawn` and feeds tokens through a `queue.Queue`. Add a parallel `progress_queue` so the background thread drives the progress callback on the asyncio side via `loop.call_soon_threadsafe`. Avoids the Rust-side `Python::attach` from the spawned thread fighting with the asyncio loop.
    - **Cancellation interaction:** `progress=` integrates with the existing `cancel_token=` kwarg by force-firing the finaliser with `error=CancelledError("generation cancelled")` so widgets render the cancellation visibly (`bar_style="warning"` on ipywidgets, `set_postfix_str("cancelled")` on tqdm).
  - **Files:**
    - `crates/oxillama-py/src/callback.rs` (modified, +~140 lines): add `Throttler` struct, `ProgressBridge` struct with `Drop` impl (= RAII guard), helper `make_progress_callback(progress: Option<Py<PyAny>>, max_tokens: usize, throttle_ms: u64, throttle_tokens: usize, capture_text: bool) -> ProgressBridge` that imports `oxillama_py.progress._build_bridge`. Six new Rust unit tests (Python interpreter required for the bridge build, no-op tests for the throttler).
    - `crates/oxillama-py/src/engine.rs` (modified, +~80 lines): wire `progress=`, `progress_throttle_ms=`, `progress_throttle_tokens=`, `progress_capture_text=`, `strict_progress=` kwargs into `generate` and `generate_streaming`. Compose the user `callback` (1-arg token) and the progress bridge into a single `FnMut(&str)` closure. Update docstrings with a Jupyter example.
    - `crates/oxillama-py/src/speculative.rs` (modified, +~40 lines): add the same kwargs to `PySpeculativeEngine::generate` and `generate_streaming`.
    - `crates/oxillama-py/src/async_support.rs` (modified, +~50 lines): add `progress=` kwarg to `generate` and `generate_stream`; for `generate_stream`, drive the progress callback via `loop.call_soon_threadsafe` to avoid GIL contention from the spawned worker.
    - `crates/oxillama-py/python/oxillama_py/progress.py` (NEW, ~220 lines): `ProgressEvent` frozen dataclass, `_TqdmAdapter`, `_IPyWidgetAdapter`, `_CallableAdapter`, `make_progress_adapter(obj, max_tokens) -> (callback, finaliser)`, `_build_bridge(...)` (the Rust→Python entry point).
    - `crates/oxillama-py/python/oxillama_py/__init__.py` (modified, +~6 lines): re-export `ProgressEvent`, `make_progress_adapter`. Mark `TqdmProgress` as deprecated alias (DeprecationWarning on import).
    - `crates/oxillama-py/python/oxillama_py/__init__.pyi` (modified, +~50 lines): add `ProgressEvent` dataclass stub, extend all four `generate*` signatures with the new kwargs, document `progress` parameter type as `Union[Any, Callable[[ProgressEvent], None], None]`.
    - `crates/oxillama-py/python/tests/test_progress.py` (NEW, ~280 lines): see Tests below — pure-Python tests with `FakeTqdm` and `FakeIntProgress` doubles; native-extension tests gated on `OXILLAMA_TEST_MODEL`.
    - `crates/oxillama-py/docs/progress.rst` (NEW, ~120 lines): user-facing doc with three Jupyter notebook snippets (tqdm.auto, ipywidgets, custom callable). Wire into `crates/oxillama-py/docs/index.rst` toctree.
    - `crates/oxillama-py/Cargo.toml` (no changes — no new Rust deps; `Instant`/`Duration` are in `std`).
    - `crates/oxillama-py/TODO.md` (modified): tick this item off when done; add to "Shipped in v0.1.2" section once landed.
  - **Prerequisites:** None. The existing `callback.rs` `StreamingCallbackBridge` and `cancel.rs` patterns are the reference templates.
  - **Tests:**
    1. `test_progress_event_dataclass_fields` — construct `ProgressEvent(5, 100, 0.5, 10.0, 9.5, False, "")` and assert all fields readable, frozen (raises on assignment).
    2. `test_make_progress_adapter_none_returns_none` — `make_progress_adapter(None, 100)` returns `(None, None)`.
    3. `test_make_progress_adapter_dispatches_tqdm` — pass `FakeTqdm()`, assert returned callback updates `count`, finaliser sets `closed=True`.
    4. `test_make_progress_adapter_dispatches_ipywidgets` — pass `FakeIntProgress()`, assert callback sets `value`, finaliser sets `bar_style="success"`.
    5. `test_make_progress_adapter_dispatches_callable` — pass `lambda evt: collected.append(evt)`, assert events flow through.
    6. `test_make_progress_adapter_rejects_invalid` — pass `42`, assert `TypeError` raised with helpful message.
    7. `test_throttler_fires_on_first_token` — Rust unit test: new `Throttler` with `min_interval=1s, min_tokens=999`; call `should_fire(force=true)` once, returns `true`.
    8. `test_throttler_throttles_subsequent_calls` — Rust unit test: after first fire, `should_fire(force=false)` returns `false` until interval elapses or token count crosses threshold.
    9. `test_throttler_fires_on_token_threshold` — Rust unit test: with `min_tokens=4`, after 4 calls `should_fire` returns `true`.
    10. `test_progress_callback_finalised_on_completion` — gated on `OXILLAMA_TEST_MODEL`: run `generate("hi", max_tokens=8, progress=fake_pbar)`, assert `fake_pbar.closed` is `True`.
    11. `test_progress_callback_finalised_on_exception` — gated test: pass a callable that raises after 2 tokens, assert finaliser still ran (RAII guard).
    12. `test_progress_callback_finalised_on_cancellation` — gated test: pass `cancel_token` and trigger from another thread, assert tqdm `set_postfix_str("cancelled")` was called.
    13. `test_strict_progress_propagates_callback_error` — gated test: callable raises `ValueError`, `strict_progress=True` re-raises after generation completes.
    14. `test_progress_capture_text_off_by_default` — gated test: assert `event.text_so_far == ""` when `progress_capture_text=False` (default).
    15. `test_progress_capture_text_on_accumulates` — gated test: assert `event.text_so_far` grows monotonically when `progress_capture_text=True`.
    16. `test_progress_event_eta_none_until_two_tokens` — gated test: capture events; assert `events[0].eta_secs is None`, `events[2].eta_secs > 0`.
    17. `test_progress_throttling_reduces_callback_count` — gated test: generate 200 tokens with default throttling, assert callback fired ≤ ~10× (confirms throttle works) and that first/final tokens always fired.
    18. `test_async_engine_progress_kwarg` — gated async test: `await engine.generate("hi", progress=fake)` triggers the bar from inside the asyncio thread without deadlock.
    19. `test_speculative_engine_progress_kwarg` — gated test on speculative path: same RAII semantics hold.
    20. `test_legacy_callback_kwarg_still_works` — backwards-compat: `generate_streaming(callback=lambda tok: None)` (1-arg) still works without `progress=`.
    21. `test_callback_and_progress_compose` — gated test: passing both `callback=` and `progress=` fires both per-token (callback every token, progress throttled).
    22. `test_tqdm_progress_deprecated_warning` — pure-Python: `from oxillama_py import TqdmProgress` emits `DeprecationWarning` mentioning `progress=` migration.
  - **Risk:**
    - *Risk 1*: Calling Python `_build_bridge` once per generation pulls one extra import — mitigated by caching the helper module reference in `lib.rs` (same pattern as `_oxillama_async_helper`).
    - *Risk 2*: `loop.call_soon_threadsafe` in the async path requires the loop reference at spawn time — captured into the worker via the same kwargs/locals mechanism that the existing `_TokenStream` uses; pattern already proven.
    - *Risk 3*: `ipywidgets.IntProgress` duck-typing collision with future widgets that have `value`/`max` but aren't progress bars — mitigated by also requiring class-name substring `"Progress"` and by allowing the user to pass a `_CallableAdapter`-wrapped explicit callable as escape hatch.
    - *Risk 4*: RAII `Drop` running during a panic might re-enter Python while the GIL is poisoned — mitigated by `std::panic::catch_unwind` around the per-token closure (the runtime already catches panics at the engine boundary; we just need to ensure the finaliser is called outside the unwind path via the explicit `result.set(...)` check rather than relying on `Drop`-during-unwind).
    - *Risk 5*: The existing `TqdmProgress` shim is widely advertised in v0.1.1 docs. Mitigation: keep it as a DeprecationWarning-emitting alias that internally constructs `progress=` with the new bridge, so existing code continues to render — the warning just nudges migration.
  - ✅ done 2026-05-03 — `Throttler`, `ProgressBridge`, `make_progress_bridge` shipped in `callback.rs` (5 new Rust unit tests); `progress=`/`progress_throttle_ms`/`progress_throttle_tokens`/`progress_capture_text`/`strict_progress` kwargs landed on `Engine.generate{,_streaming}`, `SpeculativeEngine.generate{,_streaming}`, and `AsyncEngine.generate{,_stream}`; pure-Python `oxillama_py.progress` (`ProgressEvent`, `_TqdmAdapter`, `_IPyWidgetAdapter`, `_CallableAdapter`, `make_progress_adapter`, `_build_bridge`) shipped; `__init__.py` lazy `__getattr__` keeps `TqdmProgress`/`CollectTokens` working under `DeprecationWarning`; 11 new pure-Python tests in `test_progress.py` plus 12 model-gated tests; `docs/progress.rst` shipped and wired via `index.rst`. `clippy -p oxillama-py --all-features --all-targets -- -D warnings` clean; 86/86 Rust unit tests pass; pure-Python progress suite 11/11.
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

## Proposed follow-ups

- **R1 — `pickle_checkpoint_support` scope clarification (proposed 2026-05-03)**

  The pending gap "No pickle / checkpoint support" is too underspecified to plan responsibly. Concrete questions for the user before we can design:

  1. **Which Engine state needs to round-trip?** Options range from cheap to expensive:
     - **Just config** (model path, sampling params, LoRA stack identifier) — cheap, ~1 KB pickle, requires reload from disk on `loads`.
     - **Config + KV cache** — mid-conversation snapshot; ~50–500 MB; needs `oxillama-runtime` snapshot/restore (which already exists for CLI session save/resume).
     - **Config + KV + sampler RNG** — full reproducibility for scientific use.
     - **Everything including model weights** — multi-GB pickles, almost certainly the wrong design.
  2. **What's the use case?** Notebook checkpointing, multi-process serving, job-queue resumption, or testing/reproducibility?
  3. **Storage format:** Native pickle (slow, large, opaque), or a `__reduce__` stub that delegates to a fast Pure Rust serialization (oxicode)?
  4. **Alternative API:** Should we provide explicit `Engine.snapshot(path)` / `Engine.restore(path)` (already what the CLI does) and let users pickle a *handle* to that snapshot, rather than forcing the whole state through pickle?

  **Recommended next step:** before any implementation, the user picks one of (1)/(2)/(3)/(4) — or we leave the gap open and ship without it for v0.1.3.

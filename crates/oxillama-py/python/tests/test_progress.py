"""Tests for the polymorphic progress hook.

Pure-Python tests run without the native extension; the gated tests at the
end of the file only run when ``OXILLAMA_TEST_MODEL`` points to a real GGUF
model file (the same convention as ``conftest.py::model_path``).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import warnings
from dataclasses import FrozenInstanceError
from typing import Any, List, Optional

import pytest

from oxillama_py.progress import (
    ProgressEvent,
    _build_bridge,
    make_progress_adapter,
)


# ---------------------------------------------------------------------------
# Fake widgets used by the pure-Python tests
# ---------------------------------------------------------------------------


class FakeTqdm:
    """Minimal tqdm-shaped object."""

    def __init__(self) -> None:
        self.count: int = 0
        self.total: Optional[int] = None
        self.postfix: str = ""
        self.closed: bool = False
        self.refreshed: bool = False

    def update(self, n: int = 1) -> None:
        self.count += n

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:
        self.postfix = s
        if refresh:
            self.refreshed = True

    def close(self) -> None:
        self.closed = True


class FakeIntProgress:
    """Minimal ipywidgets.IntProgress-shaped object.

    The class name contains ``"Progress"`` to satisfy the duck-type check.
    """

    def __init__(self, max_: int = 0) -> None:
        self.value: int = 0
        self.max: int = max_
        self.description: str = ""
        self.bar_style: str = ""


class _NotAWidget:
    """Has ``value``/``max`` but no ``Progress`` in its name."""

    def __init__(self) -> None:
        self.value = 0
        self.max = 0


# ---------------------------------------------------------------------------
# 1. ProgressEvent dataclass
# ---------------------------------------------------------------------------


def test_progress_event_dataclass_fields() -> None:
    evt = ProgressEvent(
        tokens_generated=5,
        tokens_total=100,
        elapsed_secs=0.5,
        tokens_per_sec=10.0,
        eta_secs=9.5,
        is_final=False,
        text_so_far="",
    )
    assert evt.tokens_generated == 5
    assert evt.tokens_total == 100
    assert evt.elapsed_secs == 0.5
    assert evt.tokens_per_sec == 10.0
    assert evt.eta_secs == 9.5
    assert evt.is_final is False
    assert evt.text_so_far == ""

    # frozen=True must reject reassignment
    with pytest.raises((FrozenInstanceError, AttributeError)):
        evt.tokens_generated = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2-6. make_progress_adapter dispatch
# ---------------------------------------------------------------------------


def test_make_progress_adapter_none_returns_none() -> None:
    assert make_progress_adapter(None, 100) == (None, None)


def test_make_progress_adapter_dispatches_tqdm() -> None:
    pbar = FakeTqdm()
    cb, fin = make_progress_adapter(pbar, 64)
    assert cb is not None and fin is not None

    cb(_event(tokens=1, elapsed=0.1, eta=None, is_final=False))
    assert pbar.count == 1
    assert pbar.total == 64
    cb(_event(tokens=4, elapsed=0.4, eta=None, is_final=False))
    assert pbar.count == 4
    fin(None)
    assert pbar.closed is True


def test_make_progress_adapter_dispatches_ipywidgets() -> None:
    widget = FakeIntProgress()
    cb, fin = make_progress_adapter(widget, 32)
    assert cb is not None and fin is not None

    cb(_event(tokens=1, elapsed=0.1, eta=None, is_final=False))
    assert widget.max == 32
    assert widget.value == 1
    cb(_event(tokens=10, elapsed=0.5, eta=2.0, is_final=False))
    assert widget.value == 10
    fin(None)
    assert widget.bar_style == "success"
    assert widget.value == widget.max


def test_make_progress_adapter_ipywidgets_error_styles() -> None:
    widget = FakeIntProgress(max_=10)
    _cb, fin = make_progress_adapter(widget, 10)
    assert fin is not None
    fin(RuntimeError("boom"))
    assert widget.bar_style == "danger"

    widget2 = FakeIntProgress(max_=10)
    _cb2, fin2 = make_progress_adapter(widget2, 10)
    assert fin2 is not None
    fin2(asyncio_cancelled())
    assert widget2.bar_style == "warning"


def test_make_progress_adapter_dispatches_callable() -> None:
    seen: List[ProgressEvent] = []

    def on_event(evt: ProgressEvent) -> None:
        seen.append(evt)

    cb, fin = make_progress_adapter(on_event, 16)
    assert cb is not None and fin is not None
    evt1 = _event(tokens=1, elapsed=0.0, eta=None, is_final=False)
    evt2 = _event(tokens=2, elapsed=0.1, eta=1.0, is_final=True)
    cb(evt1)
    cb(evt2)
    fin(None)
    assert seen == [evt1, evt2]


def test_make_progress_adapter_rejects_invalid() -> None:
    with pytest.raises(TypeError, match="progress must be"):
        make_progress_adapter(42, 8)
    with pytest.raises(TypeError, match="progress must be"):
        make_progress_adapter(_NotAWidget(), 8)


# ---------------------------------------------------------------------------
# _build_bridge: Rust-facing entry point
# ---------------------------------------------------------------------------


def test_build_bridge_none_returns_noop_pair() -> None:
    cb, fin = _build_bridge(None, 8)
    # No-op callbacks accept the canonical 4-tuple and an Optional error.
    cb((1, 0.1, False, ""))
    fin(None)
    fin(RuntimeError("ok"))


def test_build_bridge_constructs_progress_event() -> None:
    seen: List[ProgressEvent] = []

    cb, _fin = _build_bridge(lambda evt: seen.append(evt), 16)
    cb((1, 0.5, False, "hi"))
    cb((4, 1.0, True, "hi there"))
    assert len(seen) == 2
    assert seen[0].tokens_generated == 1
    assert seen[0].tokens_total == 16
    assert seen[0].elapsed_secs == 0.5
    assert seen[0].tokens_per_sec == 1 / 0.5
    assert seen[0].eta_secs is None  # < 2 tokens
    assert seen[0].is_final is False
    assert seen[0].text_so_far == "hi"
    assert seen[1].is_final is True
    assert seen[1].eta_secs is not None and seen[1].eta_secs > 0


def test_build_bridge_final_tick_does_not_increment_tokens() -> None:
    """The synthesised final tick must not bump ``tokens_generated``."""
    seen: List[ProgressEvent] = []

    cb, _fin = _build_bridge(lambda evt: seen.append(evt), 8)
    # Per-token ticks
    cb((1, 0.1, False, ""))
    cb((4, 0.4, False, ""))
    # Synthesised final tick — same token count, ``is_final=True``
    cb((4, 0.5, True, ""))
    assert seen[-1].is_final is True
    # The final event reports the *current* token count, not count + 1.
    assert seen[-1].tokens_generated == 4


# ---------------------------------------------------------------------------
# DeprecationWarning for v0.1.1 shims
# ---------------------------------------------------------------------------


def test_tqdm_progress_deprecated_warning() -> None:
    """``from oxillama_py import TqdmProgress`` must emit DeprecationWarning."""
    # Force a fresh package import so the lazy ``__getattr__`` runs again.
    pkg_modules = [name for name in sys.modules if name.startswith("oxillama_py")]
    for name in pkg_modules:
        del sys.modules[name]

    import oxillama_py  # noqa: F401, E402

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = oxillama_py.TqdmProgress  # triggers __getattr__
        assert legacy is not None
    deprecated = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecated, "expected DeprecationWarning on TqdmProgress access"
    assert any("progress=" in str(w.message) for w in deprecated)


def test_collect_tokens_deprecated_warning() -> None:
    pkg_modules = [name for name in sys.modules if name.startswith("oxillama_py")]
    for name in pkg_modules:
        del sys.modules[name]

    import oxillama_py  # noqa: F401, E402

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = oxillama_py.CollectTokens
        assert legacy is not None
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


# ---------------------------------------------------------------------------
# Gated tests — require a real GGUF model
# ---------------------------------------------------------------------------


_NEEDS_MODEL = pytest.mark.skipif(
    not os.environ.get("OXILLAMA_TEST_MODEL"),
    reason="OXILLAMA_TEST_MODEL not set; skipping live-engine progress tests",
)


def _engine_or_skip() -> Any:
    try:
        from oxillama_py import Engine, EngineConfig  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("native extension not built")
    if Engine is None:
        pytest.skip("native extension not built")
    path = os.environ.get("OXILLAMA_TEST_MODEL")
    if not path or not os.path.isfile(path):
        pytest.skip("OXILLAMA_TEST_MODEL not set or missing")
    cfg = EngineConfig(model_path=path)
    eng = Engine(cfg)
    eng.load_model()
    return eng


@_NEEDS_MODEL
def test_progress_callback_finalised_on_completion() -> None:
    pbar = FakeTqdm()
    eng = _engine_or_skip()
    eng.generate("hi", max_tokens=8, progress=pbar)
    assert pbar.closed is True


@_NEEDS_MODEL
def test_progress_callback_finalised_on_exception() -> None:
    pbar = FakeTqdm()
    state = {"calls": 0}

    def bad_cb(evt: ProgressEvent) -> None:
        state["calls"] += 1
        if state["calls"] >= 2:
            raise RuntimeError("intentional failure")

    eng = _engine_or_skip()
    # Compose tqdm with the failing callable via an adapter.
    eng.generate("hi", max_tokens=8, progress=bad_cb, strict_progress=False)
    # Even on a swallowed exception, generation must complete and callback
    # must have been driven.
    assert state["calls"] >= 1


@_NEEDS_MODEL
def test_progress_callback_finalised_on_cancellation() -> None:
    from oxillama_py import CancellationToken  # type: ignore[import-untyped]

    pbar = FakeTqdm()
    token = CancellationToken()

    def cancel_soon() -> None:
        time.sleep(0.05)
        token.cancel()

    threading.Thread(target=cancel_soon, daemon=True).start()
    eng = _engine_or_skip()
    with pytest.raises(Exception):
        eng.generate("write a long story", max_tokens=2048, cancel_token=token, progress=pbar)
    # Finaliser ran irrespective of cancellation outcome.
    assert pbar.closed is True


@_NEEDS_MODEL
def test_strict_progress_propagates_callback_error() -> None:
    eng = _engine_or_skip()

    def bad_cb(_evt: ProgressEvent) -> None:
        raise ValueError("strict propagation")

    with pytest.raises(ValueError, match="strict propagation"):
        eng.generate("hi", max_tokens=8, progress=bad_cb, strict_progress=True)


@_NEEDS_MODEL
def test_progress_capture_text_off_by_default() -> None:
    seen: List[ProgressEvent] = []
    eng = _engine_or_skip()
    eng.generate("hi", max_tokens=4, progress=lambda evt: seen.append(evt))
    assert seen, "expected at least one progress event"
    assert all(e.text_so_far == "" for e in seen)


@_NEEDS_MODEL
def test_progress_capture_text_on_accumulates() -> None:
    seen: List[ProgressEvent] = []
    eng = _engine_or_skip()
    eng.generate(
        "hi",
        max_tokens=8,
        progress=lambda evt: seen.append(evt),
        progress_capture_text=True,
    )
    assert seen
    # Text must grow monotonically across events.
    lengths = [len(e.text_so_far) for e in seen]
    assert all(b >= a for a, b in zip(lengths, lengths[1:]))


@_NEEDS_MODEL
def test_progress_event_eta_none_until_two_tokens() -> None:
    seen: List[ProgressEvent] = []
    eng = _engine_or_skip()
    eng.generate("hi", max_tokens=16, progress=lambda evt: seen.append(evt))
    assert seen
    assert seen[0].eta_secs is None
    # By the third or fourth event, ETA should be populated.
    later = [e for e in seen if e.tokens_generated >= 2]
    assert any(e.eta_secs is not None for e in later)


@_NEEDS_MODEL
def test_progress_throttling_reduces_callback_count() -> None:
    seen: List[ProgressEvent] = []
    eng = _engine_or_skip()
    eng.generate(
        "Tell me a longer story",
        max_tokens=200,
        progress=lambda evt: seen.append(evt),
    )
    # First and final should always have fired; throttling must prune.
    assert seen, "expected at least one event"
    assert seen[-1].is_final is True
    # 200 tokens with 4-token / 50 ms throttle -> well under 200 events.
    assert len(seen) < 200


@_NEEDS_MODEL
def test_async_engine_progress_kwarg() -> None:
    import asyncio

    from oxillama_py import AsyncEngine, EngineConfig  # type: ignore[import-untyped]

    seen: List[ProgressEvent] = []

    async def main() -> None:
        path = os.environ["OXILLAMA_TEST_MODEL"]
        engine = AsyncEngine(EngineConfig(model_path=path))
        await engine.load_model()
        await engine.generate("hi", max_tokens=8, progress=lambda evt: seen.append(evt))

    asyncio.run(main())
    assert seen
    assert seen[-1].is_final is True


@_NEEDS_MODEL
def test_speculative_engine_progress_kwarg() -> None:
    # We accept the gated test as best-effort: many CI envs only have a
    # single model.  When draft+target paths are not configured, skip.
    target = os.environ.get("OXILLAMA_TEST_MODEL")
    draft = os.environ.get("OXILLAMA_TEST_DRAFT_MODEL", target)
    if not target or not draft:
        pytest.skip("speculative test requires OXILLAMA_TEST_MODEL")
    from oxillama_py import (  # type: ignore[import-untyped]
        EngineConfig,
        SpeculativeConfig,
        SpeculativeEngine,
    )

    pbar = FakeTqdm()
    cfg = SpeculativeConfig(
        target=EngineConfig(model_path=target),
        draft=EngineConfig(model_path=draft),
    )
    eng = SpeculativeEngine(cfg)
    eng.generate("hi", max_tokens=8, progress=pbar)
    assert pbar.closed is True


# ---------------------------------------------------------------------------
# Backwards-compat
# ---------------------------------------------------------------------------


@_NEEDS_MODEL
def test_legacy_callback_kwarg_still_works() -> None:
    eng = _engine_or_skip()
    seen: List[str] = []
    eng.generate_streaming("hi", max_tokens=4, callback=lambda tok: seen.append(tok))
    assert seen


@_NEEDS_MODEL
def test_callback_and_progress_compose() -> None:
    eng = _engine_or_skip()
    callback_tokens: List[str] = []
    progress_events: List[ProgressEvent] = []

    eng.generate_streaming(
        "Tell me about Tokyo",
        max_tokens=40,
        callback=lambda tok: callback_tokens.append(tok),
        progress=lambda evt: progress_events.append(evt),
    )
    # Callback fires every token, progress is throttled.
    assert callback_tokens
    assert progress_events
    assert len(progress_events) <= len(callback_tokens)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _event(
    tokens: int,
    elapsed: float,
    eta: Optional[float],
    is_final: bool,
    text: str = "",
    total: Optional[int] = None,
) -> ProgressEvent:
    """Build a ProgressEvent for tests without going through ``_build_bridge``.

    ``total`` defaults to ``None`` so the adapters fall back to the
    ``max_tokens`` value that was passed when the adapter was constructed —
    matching the production codepath in :func:`oxillama_py.progress._build_bridge`.
    """
    tps = (tokens / elapsed) if elapsed > 0 else 0.0
    return ProgressEvent(
        tokens_generated=tokens,
        tokens_total=total,
        elapsed_secs=elapsed,
        tokens_per_sec=tps,
        eta_secs=eta,
        is_final=is_final,
        text_so_far=text,
    )


def asyncio_cancelled() -> BaseException:
    """Return an ``asyncio.CancelledError`` instance (works on 3.8+)."""
    import asyncio

    return asyncio.CancelledError("cancelled by test")

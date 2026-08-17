"""Tests for CancellationToken — skipped when native extension is absent."""

from __future__ import annotations

import os
import threading
import time

import pytest


def _import_native():
    try:
        import oxillama_py.oxillama_py as _m  # type: ignore[import-untyped]

        return _m
    except ImportError:
        return None


_NATIVE = _import_native()
_SKIP = pytest.mark.skipif(
    _NATIVE is None, reason="Native extension not built (run `maturin develop`)"
)


@_SKIP
def test_cancellation_token_default_not_cancelled():
    ct = _NATIVE.CancellationToken()
    assert ct.is_cancelled() is False


@_SKIP
def test_cancellation_token_cancel():
    ct = _NATIVE.CancellationToken()
    ct.cancel()
    assert ct.is_cancelled() is True


@_SKIP
def test_cancellation_token_reset_after_cancel():
    ct = _NATIVE.CancellationToken()
    ct.cancel()
    assert ct.is_cancelled() is True
    ct.reset()
    assert ct.is_cancelled() is False


@_SKIP
def test_cancellation_token_multiple_cancels_idempotent():
    ct = _NATIVE.CancellationToken()
    ct.cancel()
    ct.cancel()
    assert ct.is_cancelled() is True


@_SKIP
def test_cancellation_token_thread_safety():
    """Cancelling from one thread should be visible from another."""
    ct = _NATIVE.CancellationToken()
    results = []

    def canceller():
        ct.cancel()

    def checker():
        # busy-wait up to 1 s
        import time

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if ct.is_cancelled():
                results.append(True)
                return
            time.sleep(0.001)
        results.append(False)

    t_cancel = threading.Thread(target=canceller)
    t_check = threading.Thread(target=checker)
    t_check.start()
    t_cancel.start()
    t_cancel.join()
    t_check.join()

    assert results == [True]


@_SKIP
def test_cancellation_token_repr():
    ct = _NATIVE.CancellationToken()
    r = repr(ct)
    assert "CancellationToken" in r or "cancelled" in r.lower()


# ---------------------------------------------------------------------------
# Live-engine cancellation
#
# `CancellationToken` used to be observationally correct but practically
# useless: the Rust decode loop had no cancellation hook at all, so
# `generate(..., max_tokens=2048, cancel_token=t)` cancelled at token 1 still
# burned the full 2048 forward passes before raising. The flag is now bound to
# `GenerationConfig::cancel_flag` and checked once per decode iteration, so
# cancellation ends generation at the next token boundary.
#
# These are gated on OXILLAMA_TEST_MODEL (same convention as conftest.py).
# ---------------------------------------------------------------------------

_NEEDS_MODEL = pytest.mark.skipif(
    not os.environ.get("OXILLAMA_TEST_MODEL"),
    reason="OXILLAMA_TEST_MODEL not set; skipping live-engine cancellation tests",
)


def _engine_or_skip():
    try:
        from oxillama_py import Engine, EngineConfig  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("native extension not built")
    if Engine is None:
        pytest.skip("native extension not built")
    path = os.environ.get("OXILLAMA_TEST_MODEL")
    if not path or not os.path.isfile(path):
        pytest.skip("OXILLAMA_TEST_MODEL not set or missing")
    cfg = EngineConfig(model_path=path, context_size=512)
    eng = Engine(cfg)
    eng.load_model()
    return eng


def _time_generate(engine, max_tokens, token=None):
    """Run one generation and return (elapsed_seconds, raised_exception)."""
    started = time.monotonic()
    raised = None
    try:
        engine.generate(
            "Write a very long story about the sea.",
            max_tokens=max_tokens,
            cancel_token=token,
        )
    except Exception as exc:  # noqa: BLE001 - cancellation surfaces as RuntimeError
        raised = exc
    return time.monotonic() - started, raised


@_SKIP
@_NEEDS_MODEL
def test_cancel_mid_generation_returns_promptly():
    """Cancelling mid-flight must shorten wall-clock, not just raise later.

    The comparison is against a *measured* uncancelled baseline rather than a
    fixed number of seconds, so the test says the same thing on a fast machine
    and a loaded one: a generation cancelled a fraction of a second in must
    finish in far less time than the full budget would take.
    """
    from oxillama_py import CancellationToken  # type: ignore[import-untyped]

    engine = _engine_or_skip()

    # Baseline: how long do 32 uncancelled tokens take on this machine?
    baseline, err = _time_generate(engine, 32)
    assert err is None, f"baseline generation must not raise: {err!r}"
    per_token = baseline / 32.0

    # Now ask for 40x that budget and cancel almost immediately.
    budget = 1280
    token = CancellationToken()

    def cancel_soon():
        time.sleep(0.2)
        token.cancel()

    threading.Thread(target=cancel_soon, daemon=True).start()
    elapsed, raised = _time_generate(engine, budget, token)

    assert raised is not None, "a cancelled generation must report cancellation"
    assert "cancel" in str(raised).lower(), f"unexpected error: {raised!r}"

    projected_full_run = per_token * budget
    assert elapsed < projected_full_run / 4, (
        "cancellation did not shorten the run: it took "
        f"{elapsed:.2f}s against a projected {projected_full_run:.2f}s for the "
        f"full {budget}-token budget ({per_token * 1000:.1f} ms/token baseline). "
        "The decode loop is ignoring GenerationConfig::cancel_flag."
    )


@_SKIP
@_NEEDS_MODEL
def test_cancel_before_generation_costs_almost_nothing():
    """An already-cancelled token must be seen before the first sample."""
    from oxillama_py import CancellationToken  # type: ignore[import-untyped]

    engine = _engine_or_skip()
    baseline, err = _time_generate(engine, 32)
    assert err is None

    token = CancellationToken()
    token.cancel()
    elapsed, raised = _time_generate(engine, 1280, token)

    assert raised is not None, "a pre-cancelled generation must report cancellation"
    assert elapsed < baseline, (
        f"a pre-cancelled generation took {elapsed:.2f}s, which is not less "
        f"than the {baseline:.2f}s a 32-token run costs — the flag is being "
        "read only after generation, not during it"
    )


@_SKIP
@_NEEDS_MODEL
def test_generate_detailed_reports_cancelled_finish_reason():
    """`generate_detailed` surfaces cancellation as a real finish reason."""
    from oxillama_py import (  # type: ignore[import-untyped]
        CancellationToken,
        GenerationConfig,
    )

    engine = _engine_or_skip()
    token = CancellationToken()

    def cancel_soon():
        time.sleep(0.2)
        token.cancel()

    threading.Thread(target=cancel_soon, daemon=True).start()
    with pytest.raises(Exception) as excinfo:
        engine.generate_detailed(
            "Write a very long story about the sea.",
            GenerationConfig(1280),
            cancel_token=token,
        )
    assert "cancel" in str(excinfo.value).lower()


@_SKIP
def test_finish_reason_exposes_cancelled_variant():
    """`FinishReason.Cancelled` must exist on the shipped surface.

    The `.pyi` stub declares it; nothing else would catch the stub drifting
    from the extension, since stubs are not compiled and pytest never imports
    them.
    """
    from oxillama_py import FinishReason  # type: ignore[import-untyped]

    assert hasattr(FinishReason, "Cancelled")
    assert FinishReason.Cancelled.is_cancelled() is True
    assert FinishReason.Cancelled.is_natural() is False
    assert FinishReason.Cancelled.as_openai_str() == "cancelled"

    # The pre-existing variants keep their meanings.
    for name, openai, natural in (
        ("Eos", "stop", True),
        ("MaxTokens", "length", False),
        ("ContextFull", "length", False),
        ("Stopped", "stop", True),
    ):
        variant = getattr(FinishReason, name)
        assert variant.as_openai_str() == openai
        assert variant.is_natural() is natural
        assert variant.is_cancelled() is False

"""Tests for CancellationToken — skipped when native extension is absent."""

from __future__ import annotations

import threading

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

"""Tests for the StreamingCallback Protocol.

These are pure-Python tests — they do NOT require the native extension.
"""

from __future__ import annotations

import pytest
from oxillama_py.callback import StreamingCallback, TokenCallback


# ---------------------------------------------------------------------------
# Helper conforming implementations
# ---------------------------------------------------------------------------


def plain_function_callback(token: str, token_id: int, is_final: bool) -> None:
    """A bare function that satisfies StreamingCallback."""


class ClassCallback:
    """A class with __call__ that satisfies StreamingCallback."""

    def __call__(self, token: str, token_id: int, is_final: bool) -> None:
        pass


class CallbackWithState:
    """Stateful callback — still satisfies the protocol."""

    def __init__(self) -> None:
        self.tokens: list[str] = []

    def __call__(self, token: str, token_id: int, is_final: bool) -> None:
        self.tokens.append(token)


class WrongSignatureCallback:
    """Only accepts one argument — does NOT satisfy the protocol."""

    def __call__(self, token: str) -> None:
        pass


class NoCallCallback:
    """No __call__ at all — does NOT satisfy the protocol."""

    def process(self, token: str) -> None:
        pass


# ---------------------------------------------------------------------------
# isinstance / Protocol checks
# ---------------------------------------------------------------------------


def test_plain_function_satisfies_protocol():
    assert isinstance(plain_function_callback, StreamingCallback)


def test_class_with_call_satisfies_protocol():
    cb = ClassCallback()
    assert isinstance(cb, StreamingCallback)


def test_stateful_callback_satisfies_protocol():
    cb = CallbackWithState()
    assert isinstance(cb, StreamingCallback)


def test_object_without_call_does_not_satisfy_protocol():
    obj = NoCallCallback()
    assert not isinstance(obj, StreamingCallback)


def test_plain_int_does_not_satisfy_protocol():
    assert not isinstance(42, StreamingCallback)


def test_none_does_not_satisfy_protocol():
    assert not isinstance(None, StreamingCallback)


# ---------------------------------------------------------------------------
# Functional invocation
# ---------------------------------------------------------------------------


def test_plain_function_callable():
    """Verify the protocol function can actually be called."""
    plain_function_callback("hello", 1024, False)


def test_class_callback_callable():
    cb = ClassCallback()
    cb("world", 2048, True)


def test_stateful_callback_accumulates_tokens():
    cb = CallbackWithState()
    tokens = [("Hello", 1, False), (" world", 2, False), ("!", 3, True)]
    for tok, tid, fin in tokens:
        cb(tok, tid, fin)
    assert cb.tokens == ["Hello", " world", "!"]


# ---------------------------------------------------------------------------
# TokenCallback type alias is exposed
# ---------------------------------------------------------------------------


def test_token_callback_alias_importable():
    assert TokenCallback is not None


# ---------------------------------------------------------------------------
# Lambda satisfies the protocol
# ---------------------------------------------------------------------------


def test_lambda_satisfies_protocol():
    cb = lambda token, token_id, is_final: None  # noqa: E731
    assert isinstance(cb, StreamingCallback)

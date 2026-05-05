"""Tests for the AsyncEngine Python class (Track D - OxiLLaMa v0.1.5).

These are pure-Python tests that do NOT require the native extension to be
built.  ``AsyncEngine`` is implemented in ``oxillama_py/__init__.py`` and
wraps any object that exposes ``generate`` / ``generate_streaming`` methods.

Tests cover:
  - Class attributes and coroutine introspection (no execution)
  - Functional tests using mock engine objects with ``asyncio.run``
  - Streaming via ``async for`` with mock generate_streaming callbacks
  - Error propagation from the underlying engine
  - Optional kwarg forwarding (temperature, top_p, top_k, seed)
  - Sentinel / completion signalling in the queue bridge
  - Thread-pool serialisation (single worker)
  - Presence of ``async_engine()`` on the native ``Engine`` class (skipped
    when the extension is not built)
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

import oxillama_py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockEngine:
    """Minimal synchronous mock that satisfies the AsyncEngine contract."""

    def __init__(self, response: str = "hello") -> None:
        self.response = response
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, max_tokens: int = 128, **kwargs: Any) -> str:
        self.generate_calls.append(
            {"prompt": prompt, "max_tokens": max_tokens, **kwargs}
        )
        return self.response

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
        callback: Any = None,
        **kwargs: Any,
    ) -> str:
        self.stream_calls.append(
            {"prompt": prompt, "max_tokens": max_tokens, **kwargs}
        )
        tokens = list(self.response)  # split into individual characters
        for tok in tokens:
            if callback is not None:
                callback(tok)
        return self.response


class ErrorEngine:
    """Mock that raises RuntimeError on every call."""

    def generate(self, prompt: str, max_tokens: int = 128, **kwargs: Any) -> str:
        raise RuntimeError("generate failed")

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
        callback: Any = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("generate_streaming failed")


class MultiTokenEngine:
    """Mock that yields a configurable list of tokens."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens

    def generate(self, prompt: str, max_tokens: int = 128, **kwargs: Any) -> str:
        return "".join(self.tokens)

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
        callback: Any = None,
        **kwargs: Any,
    ) -> str:
        for tok in self.tokens:
            if callback is not None:
                callback(tok)
        return "".join(self.tokens)


# ---------------------------------------------------------------------------
# Structural / introspection tests (no execution)
# ---------------------------------------------------------------------------


def test_async_engine_class_exists() -> None:
    """AsyncEngine must be exported from the top-level oxillama_py package."""
    assert hasattr(oxillama_py, "AsyncEngine"), (
        "AsyncEngine not found in oxillama_py"
    )


def test_async_engine_class_is_type() -> None:
    """AsyncEngine must be a class/type (not None, not a function)."""
    assert isinstance(oxillama_py.AsyncEngine, type), (
        f"Expected a type, got {type(oxillama_py.AsyncEngine)}"
    )


def test_async_engine_init_exists() -> None:
    """AsyncEngine must have an __init__ method."""
    assert hasattr(oxillama_py.AsyncEngine, "__init__")
    assert callable(oxillama_py.AsyncEngine.__init__)


def test_async_engine_has_generate() -> None:
    """AsyncEngine must expose a ``generate`` attribute."""
    assert hasattr(oxillama_py.AsyncEngine, "generate"), (
        "AsyncEngine.generate is missing"
    )


def test_async_engine_has_stream() -> None:
    """AsyncEngine must expose a ``stream`` attribute."""
    assert hasattr(oxillama_py.AsyncEngine, "stream"), (
        "AsyncEngine.stream is missing"
    )


def test_async_engine_generate_is_coroutine_function() -> None:
    """AsyncEngine.generate must be an async coroutine function."""
    assert asyncio.iscoroutinefunction(oxillama_py.AsyncEngine.generate), (
        "AsyncEngine.generate is not an async coroutine function"
    )


def test_async_engine_stream_is_async_generator_function() -> None:
    """AsyncEngine.stream must be an async generator function."""
    assert inspect.isasyncgenfunction(oxillama_py.AsyncEngine.stream), (
        "AsyncEngine.stream is not an async generator function"
    )


def test_async_engine_in_all() -> None:
    """AsyncEngine must be listed in oxillama_py.__all__."""
    assert "AsyncEngine" in oxillama_py.__all__, (
        "AsyncEngine is missing from oxillama_py.__all__"
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_async_engine_accepts_mock_engine() -> None:
    """AsyncEngine.__init__ must accept any object (including mocks)."""
    ae = oxillama_py.AsyncEngine(MockEngine())
    assert ae is not None


def test_async_engine_stores_engine_reference() -> None:
    """The wrapped engine must be accessible as _engine."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    assert ae._engine is mock


def test_async_engine_creates_thread_pool() -> None:
    """AsyncEngine must create a private ThreadPoolExecutor."""
    ae = oxillama_py.AsyncEngine(MockEngine())
    assert hasattr(ae, "_pool"), "_pool attribute missing"
    import concurrent.futures

    assert isinstance(ae._pool, concurrent.futures.ThreadPoolExecutor), (
        "_pool is not a ThreadPoolExecutor"
    )


# ---------------------------------------------------------------------------
# generate() functional tests
# ---------------------------------------------------------------------------


def test_async_engine_generate_returns_string() -> None:
    """await ae.generate(...) must return the engine's response string."""
    ae = oxillama_py.AsyncEngine(MockEngine("hello world"))
    result = asyncio.run(ae.generate("test prompt"))
    assert result == "hello world"


def test_async_engine_generate_passes_prompt() -> None:
    """generate() must forward the prompt to the underlying engine."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    asyncio.run(ae.generate("my prompt"))
    assert len(mock.generate_calls) == 1
    assert mock.generate_calls[0]["prompt"] == "my prompt"


def test_async_engine_generate_passes_max_tokens() -> None:
    """generate() must forward the max_tokens argument."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    asyncio.run(ae.generate("prompt", max_tokens=256))
    assert mock.generate_calls[0]["max_tokens"] == 256


def test_async_engine_generate_default_max_tokens() -> None:
    """generate() default max_tokens must be 512."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    asyncio.run(ae.generate("x"))
    assert mock.generate_calls[0]["max_tokens"] == 512


def test_async_engine_generate_passes_temperature() -> None:
    """generate() must forward the temperature kwarg when provided."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    asyncio.run(ae.generate("x", temperature=0.5))
    assert abs(mock.generate_calls[0]["temperature"] - 0.5) < 1e-6


def test_async_engine_generate_omits_none_temperature() -> None:
    """generate() must NOT forward temperature=None to the engine."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    asyncio.run(ae.generate("x", temperature=None))
    assert "temperature" not in mock.generate_calls[0]


def test_async_engine_generate_passes_top_p() -> None:
    """generate() must forward top_p when provided."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    asyncio.run(ae.generate("x", top_p=0.9))
    assert abs(mock.generate_calls[0]["top_p"] - 0.9) < 1e-6


def test_async_engine_generate_passes_top_k() -> None:
    """generate() must forward top_k when provided."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    asyncio.run(ae.generate("x", top_k=40))
    assert mock.generate_calls[0]["top_k"] == 40


def test_async_engine_generate_passes_seed() -> None:
    """generate() must forward seed when provided."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    asyncio.run(ae.generate("x", seed=42))
    assert mock.generate_calls[0]["seed"] == 42


def test_async_engine_generate_passes_kwargs() -> None:
    """generate() must forward arbitrary extra kwargs to the engine."""
    mock = MockEngine()
    ae = oxillama_py.AsyncEngine(mock)
    asyncio.run(ae.generate("x", custom_flag=True))
    assert mock.generate_calls[0]["custom_flag"] is True


def test_async_engine_generate_error_propagation() -> None:
    """generate() must propagate RuntimeError raised by the engine."""
    ae = oxillama_py.AsyncEngine(ErrorEngine())
    with pytest.raises(RuntimeError, match="generate failed"):
        asyncio.run(ae.generate("x"))


def test_async_engine_generate_multiple_calls() -> None:
    """Multiple sequential awaits on the same engine must all succeed."""
    mock = MockEngine("token")
    ae = oxillama_py.AsyncEngine(mock)

    async def _run() -> list[str]:
        r1 = await ae.generate("a")
        r2 = await ae.generate("b")
        r3 = await ae.generate("c")
        return [r1, r2, r3]

    results = asyncio.run(_run())
    assert results == ["token", "token", "token"]
    assert len(mock.generate_calls) == 3


# ---------------------------------------------------------------------------
# stream() functional tests
# ---------------------------------------------------------------------------


def test_async_engine_stream_yields_tokens() -> None:
    """stream() must yield each token produced by generate_streaming."""
    tokens = ["h", "e", "l", "l", "o"]
    ae = oxillama_py.AsyncEngine(MultiTokenEngine(tokens))

    async def _collect() -> list[str]:
        return [tok async for tok in ae.stream("hi")]

    result = asyncio.run(_collect())
    assert result == tokens


def test_async_engine_stream_concatenated_equals_full_text() -> None:
    """Concatenating all yielded tokens must equal the full response."""
    text = "hello world"
    ae = oxillama_py.AsyncEngine(MultiTokenEngine(list(text)))

    async def _collect() -> str:
        return "".join([tok async for tok in ae.stream("hi")])

    result = asyncio.run(_collect())
    assert result == text


def test_async_engine_stream_empty_response() -> None:
    """stream() on an engine that produces no tokens must yield nothing."""
    ae = oxillama_py.AsyncEngine(MultiTokenEngine([]))

    async def _collect() -> list[str]:
        return [tok async for tok in ae.stream("hi")]

    result = asyncio.run(_collect())
    assert result == []


def test_async_engine_stream_single_token() -> None:
    """stream() with a single-token response must yield exactly one item."""
    ae = oxillama_py.AsyncEngine(MultiTokenEngine(["only"]))

    async def _collect() -> list[str]:
        return [tok async for tok in ae.stream("hi")]

    result = asyncio.run(_collect())
    assert result == ["only"]


def test_async_engine_stream_error_propagation() -> None:
    """stream() must propagate RuntimeError raised by generate_streaming."""
    ae = oxillama_py.AsyncEngine(ErrorEngine())

    async def _drain() -> None:
        async for _ in ae.stream("x"):
            pass

    with pytest.raises(RuntimeError, match="generate_streaming failed"):
        asyncio.run(_drain())


def test_async_engine_stream_passes_max_tokens() -> None:
    """stream() must forward max_tokens to generate_streaming."""
    mock = MockEngine("ab")
    ae = oxillama_py.AsyncEngine(mock)

    async def _run() -> None:
        async for _ in ae.stream("x", max_tokens=64):
            pass

    asyncio.run(_run())
    assert mock.stream_calls[0]["max_tokens"] == 64


def test_async_engine_stream_default_max_tokens() -> None:
    """stream() default max_tokens must be 512."""
    mock = MockEngine("x")
    ae = oxillama_py.AsyncEngine(mock)

    async def _run() -> None:
        async for _ in ae.stream("y"):
            pass

    asyncio.run(_run())
    assert mock.stream_calls[0]["max_tokens"] == 512


def test_async_engine_stream_passes_temperature() -> None:
    """stream() must forward temperature kwarg when provided."""
    mock = MockEngine("x")
    ae = oxillama_py.AsyncEngine(mock)

    async def _run() -> None:
        async for _ in ae.stream("y", temperature=0.8):
            pass

    asyncio.run(_run())
    assert abs(mock.stream_calls[0]["temperature"] - 0.8) < 1e-6


def test_async_engine_stream_omits_none_temperature() -> None:
    """stream() must NOT forward temperature=None to the engine."""
    mock = MockEngine("x")
    ae = oxillama_py.AsyncEngine(mock)

    async def _run() -> None:
        async for _ in ae.stream("y", temperature=None):
            pass

    asyncio.run(_run())
    assert "temperature" not in mock.stream_calls[0]


# ---------------------------------------------------------------------------
# Native Engine.async_engine() method (requires native extension)
# ---------------------------------------------------------------------------


def _native_available() -> bool:
    """Return True if the native PyO3 extension is importable."""
    try:
        import oxillama_py.oxillama_py  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        return False


_REQUIRES_NATIVE = pytest.mark.skipif(
    not _native_available(), reason="Native extension not built (run `maturin develop`)"
)


@_REQUIRES_NATIVE
def test_engine_has_async_engine_method() -> None:
    """Engine must expose an async_engine() method (Rust-side)."""
    assert hasattr(oxillama_py.Engine, "async_engine"), (
        "Engine.async_engine method is missing"
    )


@_REQUIRES_NATIVE
def test_engine_async_engine_method_callable() -> None:
    """Engine.async_engine must be callable."""
    assert callable(oxillama_py.Engine.async_engine)


@_REQUIRES_NATIVE
def test_engine_async_engine_returns_async_engine_instance() -> None:
    """engine.async_engine() must return an AsyncEngine wrapping the engine."""
    cfg = oxillama_py.EngineConfig(model_path="dummy.gguf")
    engine = oxillama_py.Engine(cfg)
    ae = engine.async_engine()
    assert isinstance(ae, oxillama_py.AsyncEngine), (
        f"async_engine() returned {type(ae)}, expected AsyncEngine"
    )


@_REQUIRES_NATIVE
def test_engine_async_engine_wraps_same_instance() -> None:
    """The AsyncEngine returned by async_engine() must wrap the caller engine."""
    cfg = oxillama_py.EngineConfig(model_path="dummy.gguf")
    engine = oxillama_py.Engine(cfg)
    ae = engine.async_engine()
    # The _engine attribute should be the original Engine instance.
    assert ae._engine is engine, (
        "async_engine()._engine does not point back to the caller"
    )

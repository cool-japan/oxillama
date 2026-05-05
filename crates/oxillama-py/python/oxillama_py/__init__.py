"""OxiLLaMa - Pure Rust LLM inference engine, Python bindings.

>>> import oxillama_py
>>> config = oxillama_py.EngineConfig(model_path="model.gguf")
>>> engine = oxillama_py.Engine(config)
>>> engine.load_model()
>>> print(engine.generate("Hello", max_tokens=64))
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
from typing import TYPE_CHECKING, Any, AsyncIterator

from oxillama_py.callback import StreamingCallback, TokenCallback
from oxillama_py.progress import ProgressEvent, make_progress_adapter
from oxillama_py.utils import decode_from_logits

# ---------------------------------------------------------------------------
# AsyncEngine — pure-Python async wrapper around any engine object
# ---------------------------------------------------------------------------


class AsyncEngine:
    """Async wrapper around a synchronous inference engine.

    Accepts any object that exposes ``generate(prompt, **kwargs) -> str``
    and ``generate_streaming(prompt, callback=..., **kwargs) -> str``
    methods — including the native :class:`Engine` and plain mock objects.

    All blocking calls are offloaded to a private single-threaded
    :class:`~concurrent.futures.ThreadPoolExecutor` so that the asyncio
    event loop is never blocked.

    .. note::
        For the streaming variant, token delivery is mediated by a
        :class:`queue.Queue` bridge: the generation thread pushes tokens
        (or a sentinel ``None`` on completion), and :meth:`stream` pulls
        them one at a time via ``loop.run_in_executor``.

    Example::

        import asyncio
        import oxillama_py

        async def main():
            cfg = oxillama_py.EngineConfig("model.gguf")
            engine = oxillama_py.Engine(cfg)
            engine.load_model()

            ae = engine.async_engine()
            text = await ae.generate("Hello", max_tokens=128)

            async for token in ae.stream("Hello", max_tokens=64):
                print(token, end="", flush=True)

        asyncio.run(main())
    """

    __slots__ = ("_engine", "_pool")

    def __init__(self, engine: Any) -> None:
        """Wrap *engine* for async use.

        Args:
            engine: Any object with ``generate`` and ``generate_streaming``
                    methods.  Typically a native :class:`Engine` instance or
                    a test mock.
        """
        self._engine = engine
        # Single-threaded pool: the underlying engine is not thread-safe, so
        # we must serialise all calls to it.
        self._pool: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Return the full generated text for *prompt*.

        Runs :meth:`Engine.generate` in the thread-pool executor so that
        the asyncio event loop is not blocked during inference.

        Args:
            prompt:      Input text.
            max_tokens:  Maximum tokens to generate (default 512).
            temperature: Sampling temperature override.
            top_p:       Nucleus sampling threshold override.
            top_k:       Top-k limit override.
            seed:        Random seed override.
            **kwargs:    Additional keyword arguments forwarded to the
                         underlying ``generate`` method.

        Returns:
            The generated text (not including the prompt).
        """
        loop = asyncio.get_running_loop()

        # Build keyword arguments dict, omitting None overrides so that the
        # engine's own defaults apply when a value is not specified.
        call_kwargs: dict[str, Any] = {}
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if top_p is not None:
            call_kwargs["top_p"] = top_p
        if top_k is not None:
            call_kwargs["top_k"] = top_k
        if seed is not None:
            call_kwargs["seed"] = seed
        call_kwargs.update(kwargs)

        engine = self._engine
        return await loop.run_in_executor(
            self._pool,
            lambda: engine.generate(prompt, max_tokens, **call_kwargs),
        )

    async def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async generator that yields tokens as they are produced.

        Generation runs in the thread-pool executor; each token is handed
        off through a :class:`queue.Queue` so the event loop is never
        blocked.

        Args:
            prompt:      Input text.
            max_tokens:  Maximum tokens to generate (default 512).
            temperature: Sampling temperature override.
            top_p:       Nucleus sampling threshold override.
            top_k:       Top-k limit override.
            seed:        Random seed override.
            **kwargs:    Additional keyword arguments forwarded to the
                         underlying ``generate_streaming`` method.

        Yields:
            Individual decoded token strings.

        Raises:
            RuntimeError: if the underlying generation raises an exception.
        """
        loop = asyncio.get_running_loop()
        token_queue: queue.Queue[Any] = queue.Queue()
        _sentinel = object()

        call_kwargs: dict[str, Any] = {}
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if top_p is not None:
            call_kwargs["top_p"] = top_p
        if top_k is not None:
            call_kwargs["top_k"] = top_k
        if seed is not None:
            call_kwargs["seed"] = seed
        call_kwargs.update(kwargs)

        engine = self._engine

        def _run_generation() -> None:
            """Execute blocking generation and feed the queue."""
            try:
                engine.generate_streaming(
                    prompt,
                    max_tokens,
                    lambda tok: token_queue.put(tok),
                    **call_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                token_queue.put(exc)
            finally:
                token_queue.put(_sentinel)

        self._pool.submit(_run_generation)

        while True:
            item = await loop.run_in_executor(None, token_queue.get)
            if item is _sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]


# Re-export everything from the native extension module produced by PyO3.
# Wrapped in try/except so that the pure-Python parts of this package remain
# importable even when the compiled extension has not been built yet.
try:
    from oxillama_py.oxillama_py import (  # type: ignore[import-untyped]
        Engine,
        EngineConfig,
        GenerateError,
        GrammarError,
        LoadError,
        Lora,
        OxiLlamaError,
        QuantError,
        SamplerConfig,
        SnapshotInfo,
        SpeculativeConfig,
        SpeculativeEngine,
        Tokenizer,
        TokenizerError,
    )
except ImportError:
    # Native extension not yet built. Only the pure-Python symbols are available.
    Engine = None  # type: ignore[assignment,misc]
    EngineConfig = None  # type: ignore[assignment,misc]
    GenerateError = None  # type: ignore[assignment,misc]
    GrammarError = None  # type: ignore[assignment,misc]
    LoadError = None  # type: ignore[assignment,misc]
    Lora = None  # type: ignore[assignment,misc]
    OxiLlamaError = None  # type: ignore[assignment,misc]
    QuantError = None  # type: ignore[assignment,misc]
    SamplerConfig = None  # type: ignore[assignment,misc]
    SnapshotInfo = None  # type: ignore[assignment,misc]
    SpeculativeConfig = None  # type: ignore[assignment,misc]
    SpeculativeEngine = None  # type: ignore[assignment,misc]
    Tokenizer = None  # type: ignore[assignment,misc]
    TokenizerError = None  # type: ignore[assignment,misc]

from oxillama_py import snapshot
from oxillama_py.snapshot import SnapshotError
from oxillama_py import torch_helper as _torch_helper

__version__ = "0.1.0"

# Patch torch interop helpers onto Engine (no-op if torch is not installed).
import sys as _sys
_torch_helper.try_patch(_sys.modules[__name__])

# Names that should still be importable for back-compat but trigger a
# DeprecationWarning on first access.  The lazy ``__getattr__`` hook below
# keeps the original ``from oxillama_py import TqdmProgress`` form working.
_DEPRECATED_NAMES = ("TqdmProgress", "CollectTokens")


def __getattr__(name: str) -> Any:
    """Module-level attribute hook for deprecated symbols.

    Importing ``TqdmProgress``/``CollectTokens`` from this package emits a
    :class:`DeprecationWarning` and forwards to the legacy
    :mod:`oxillama_py.tqdm_helper` shim.  Both names are still listed in
    ``__all__`` so wildcard imports continue to work.
    """
    if name in _DEPRECATED_NAMES:
        import warnings

        warnings.warn(
            f"oxillama_py.{name} is deprecated; pass progress= to generate*() "
            "instead.  See oxillama_py.progress.make_progress_adapter for the "
            "new API.",
            DeprecationWarning,
            stacklevel=2,
        )
        from oxillama_py.tqdm_helper import CollectTokens, TqdmProgress

        return {"TqdmProgress": TqdmProgress, "CollectTokens": CollectTokens}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    # Static type-checkers (mypy/pyright) cannot follow the ``__getattr__``
    # hook, so re-export the legacy names under ``TYPE_CHECKING`` guards.
    from oxillama_py.tqdm_helper import CollectTokens, TqdmProgress  # noqa: F401


__all__ = [
    # Core classes
    "EngineConfig",
    "Engine",
    "SamplerConfig",
    "SpeculativeConfig",
    "SpeculativeEngine",
    "Lora",
    "Tokenizer",
    # Async engine (v0.1.5)
    "AsyncEngine",
    # Snapshot API (v0.1.3)
    "SnapshotInfo",
    "snapshot",
    # Exceptions
    "OxiLlamaError",
    "LoadError",
    "GenerateError",
    "TokenizerError",
    "GrammarError",
    "QuantError",
    "SnapshotError",
    # Callback protocol
    "StreamingCallback",
    "TokenCallback",
    # Progress hook (v0.1.3)
    "ProgressEvent",
    "make_progress_adapter",
    # Deprecated v0.1.1 shims (kept for back-compat)
    "TqdmProgress",
    "CollectTokens",
]

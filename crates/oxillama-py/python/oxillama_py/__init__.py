"""OxiLLaMa - Pure Rust LLM inference engine, Python bindings.

>>> import oxillama_py
>>> config = oxillama_py.EngineConfig(model_path="model.gguf")
>>> engine = oxillama_py.Engine(config)
>>> engine.load_model()
>>> print(engine.generate("Hello", max_tokens=64))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from oxillama_py.callback import StreamingCallback, TokenCallback
from oxillama_py.progress import ProgressEvent, make_progress_adapter
from oxillama_py.utils import decode_from_logits

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
    SpeculativeConfig = None  # type: ignore[assignment,misc]
    SpeculativeEngine = None  # type: ignore[assignment,misc]
    Tokenizer = None  # type: ignore[assignment,misc]
    TokenizerError = None  # type: ignore[assignment,misc]

__version__ = "0.1.0"

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
    # Exceptions
    "OxiLlamaError",
    "LoadError",
    "GenerateError",
    "TokenizerError",
    "GrammarError",
    "QuantError",
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

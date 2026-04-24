"""OxiLLaMa — Pure Rust LLM inference engine, Python bindings.

>>> import oxillama_py
>>> config = oxillama_py.EngineConfig(model_path="model.gguf")
>>> engine = oxillama_py.Engine(config)
>>> engine.load_model()
>>> print(engine.generate("Hello", max_tokens=64))
"""

from __future__ import annotations

from oxillama_py.callback import StreamingCallback, TokenCallback
from oxillama_py.tqdm_helper import CollectTokens, TqdmProgress
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
]

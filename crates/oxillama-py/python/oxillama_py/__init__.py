"""OxiLLaMa — Pure Rust LLM inference engine, Python bindings.

>>> import oxillama_py
>>> config = oxillama_py.EngineConfig(model_path="model.gguf")
>>> engine = oxillama_py.Engine(config)
>>> engine.load_model()
>>> print(engine.generate("Hello", max_tokens=64))
"""

from __future__ import annotations

# Re-export everything from the native extension module produced by PyO3.
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
]

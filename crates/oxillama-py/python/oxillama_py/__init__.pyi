"""Type stubs for the ``oxillama_py`` native extension module.

Generated for PyO3 bindings exposed by OxiLLaMa.
"""

from typing import Callable, Optional, Sequence

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]

try:
    import numpy as np
    import numpy.typing as npt
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

__version__: str

__all__: list[str]

# ---------------------------------------------------------------------------
# StreamingCallback Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StreamingCallback(Protocol):
    """Protocol for streaming token callbacks."""

    def __call__(self, token: str, token_id: int, is_final: bool) -> None: ...

#: Convenience alias for a bare callable matching the streaming callback signature.
TokenCallback = Callable[[str, int, bool], None]

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class OxiLlamaError(Exception):
    """Base exception for all OxiLLaMa errors."""
    ...

class LoadError(OxiLlamaError):
    """Model loading failures (file I/O, GGUF parse errors)."""
    ...

class GenerateError(OxiLlamaError):
    """Inference / generation failures."""
    ...

class TokenizerError(OxiLlamaError):
    """Tokenizer initialisation or encoding errors."""
    ...

class GrammarError(OxiLlamaError):
    """Grammar parse or constraint failures."""
    ...

class QuantError(OxiLlamaError):
    """Quantization kernel errors."""
    ...

class KvCacheFullError(OxiLlamaError):
    """KV cache capacity exceeded during generation."""
    ...

# ---------------------------------------------------------------------------
# SamplerConfig
# ---------------------------------------------------------------------------

class SamplerConfig:
    """Sampling configuration for text generation."""

    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float
    repetition_penalty_window: int
    seed: Optional[int]
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float

    def __init__(
        self,
        *,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
        repetition_penalty_window: int = 64,
        seed: Optional[int] = None,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ) -> None: ...
    @staticmethod
    def greedy() -> SamplerConfig: ...
    @staticmethod
    def mirostat_v2(tau: float = 5.0, eta: float = 0.1) -> SamplerConfig: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# EngineConfig
# ---------------------------------------------------------------------------

class EngineConfig:
    """Configuration for the inference engine."""

    model_path: str
    tokenizer_path: Optional[str]
    context_size: Optional[int]
    num_threads: int
    sampler: SamplerConfig

    def __init__(
        self,
        model_path: str,
        *,
        context_size: Optional[int] = None,
        num_threads: int = 4,
        tokenizer_path: Optional[str] = None,
        sampler: Optional[SamplerConfig] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    """Main inference engine.

    Manages model loading and provides methods for tokenization, embedding,
    and text generation.
    """

    def __init__(self, config: EngineConfig) -> None: ...
    def load_model(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def reset(self) -> None: ...
    def tokenize(self, text: str) -> list[int]: ...
    def decode_token(self, token: int) -> str: ...
    def is_eos(self, token: int) -> bool: ...
    def hidden_size(self) -> Optional[int]: ...
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str: ...
    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
        callback: Optional[Callable[[str], None]] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        strict_callback: bool = False,
    ) -> str: ...
    def embed(self, text: str) -> list[float]: ...
    def embed_numpy(self, text: str) -> "np.ndarray[tuple[int], np.dtype[np.float32]]": ...
    def embed_batch_numpy(self, texts: Sequence[str]) -> "np.ndarray[tuple[int, int], np.dtype[np.float32]]": ...
    def apply_lora(self, lora_path: str) -> None: ...
    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        config: Optional[EngineConfig] = None,
    ) -> Engine: ...

# ---------------------------------------------------------------------------
# SpeculativeConfig
# ---------------------------------------------------------------------------

class SpeculativeConfig:
    """Configuration for speculative decoding."""

    target: EngineConfig
    draft: EngineConfig
    num_speculative: int
    seed: Optional[int]

    def __init__(
        self,
        target: EngineConfig,
        draft: EngineConfig,
        *,
        num_speculative: int = 4,
        seed: Optional[int] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# SpeculativeEngine
# ---------------------------------------------------------------------------

class SpeculativeEngine:
    """Speculative decoding engine using a draft + target model pair."""

    def __init__(self, config: SpeculativeConfig) -> None: ...
    def generate(self, prompt: str, max_tokens: int = 128) -> str: ...
    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str: ...

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """A standalone tokenizer loaded from a HuggingFace tokenizer.json file."""

    @staticmethod
    def from_file(path: str) -> Tokenizer: ...
    @staticmethod
    def from_json(json: str) -> Tokenizer: ...
    def encode(self, text: str) -> list[int]: ...
    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    def id_to_token(self, id: int) -> Optional[str]: ...
    def apply_chat_template(
        self,
        messages: list[dict],
        template: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Lora
# ---------------------------------------------------------------------------

class Lora:
    """A loaded LoRA adapter."""

    @staticmethod
    def load(path: str) -> Lora: ...
    @property
    def rank(self) -> int: ...
    @property
    def alpha(self) -> float: ...
    def num_adapters(self) -> int: ...
    def __repr__(self) -> str: ...

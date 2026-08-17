"""Type stubs for the ``oxillama_py`` native extension module.

Generated for PyO3 bindings exposed by OxiLLaMa.
"""

import os
from typing import Any, AsyncIterator, Callable, Optional, Sequence, Union

try:
    from typing import Protocol, TypedDict, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, TypedDict, runtime_checkable  # type: ignore[assignment]

try:
    import numpy as np
    import numpy.typing as npt
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

__version__: str

__all__: list[str]

# ---------------------------------------------------------------------------
# ProgressEvent and progress hook
# ---------------------------------------------------------------------------

class ProgressEvent:
    """A single progress update emitted from the Rust generation loop."""

    tokens_generated: int
    tokens_total: Optional[int]
    elapsed_secs: float
    tokens_per_sec: float
    eta_secs: Optional[float]
    is_final: bool
    text_so_far: str

    def __init__(
        self,
        tokens_generated: int,
        tokens_total: Optional[int],
        elapsed_secs: float,
        tokens_per_sec: float,
        eta_secs: Optional[float],
        is_final: bool,
        text_so_far: str,
    ) -> None: ...

#: Convenience alias for the user-facing progress callable signature.
ProgressCallback = Callable[[ProgressEvent], None]

#: Anything ``progress=`` will accept on the ``generate*`` methods.
ProgressLike = Union[Any, ProgressCallback, None]

def make_progress_adapter(
    obj: ProgressLike, max_tokens: int
) -> tuple[
    Optional[Callable[[ProgressEvent], None]],
    Optional[Callable[[Optional[BaseException]], None]],
]: ...

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
# HubOrigin — hub-aware snapshot metadata (Track F, v0.1.6)
# ---------------------------------------------------------------------------

class HubOrigin(TypedDict):
    """HuggingFace Hub origin for a GGUF model.

    Supplied to ``Engine.snapshot(hub_origin=...)`` so that
    ``Engine.restore()`` / ``Engine.from_snapshot_with_hub()`` can
    re-download the model automatically when the local file is absent.
    """

    repo_id: str
    """HuggingFace repository identifier, e.g. ``"mistralai/Mixtral-8x7B-Instruct-v0.1"``."""
    filename: str
    """Filename within the repository, e.g. ``"mixtral-8x7b.Q4_K_M.gguf"``."""
    sha256: str
    """Lower-case hex SHA-256 digest of the GGUF file (verified after download)."""

# ---------------------------------------------------------------------------
# SnapshotInfo
# ---------------------------------------------------------------------------

class SnapshotInfo:
    """Metadata extracted from a snapshot file without loading the model."""

    arch_id: str
    model_path: str
    tokenizer_path: Optional[str]
    max_context_length: int
    num_threads: int
    version: int
    magic: bytes
    tokens_count: int

    def __repr__(self) -> str: ...

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

class GpuUnavailableError(OxiLlamaError):
    """GPU offload was requested but no usable device could be bound."""
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
    frequency_penalty: float
    """OpenAI-style frequency penalty: ``logit[t] -= count(t) * frequency_penalty``
    (0.0 = disabled)."""
    presence_penalty: float
    """OpenAI-style presence penalty: a flat ``logit[t] -= presence_penalty``
    for every distinct token seen at least once (0.0 = disabled)."""
    eog_token_ids: list[int]
    """End-of-generation token IDs consulted by grammar-constrained sampling.
    Only meaningful together with a grammar (grammars are not exposed via
    this class — use the string-grammar API on :class:`Engine` instead)."""

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
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        eog_token_ids: Optional[list[int]] = None,
    ) -> None: ...
    @staticmethod
    def greedy() -> SamplerConfig: ...
    @staticmethod
    def mirostat_v2(tau: float = 5.0, eta: float = 0.1) -> SamplerConfig: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# FinishReason / GenerationConfig / GenerationOutcome
# ---------------------------------------------------------------------------

class FinishReason:
    """Why generation stopped. A PyO3 "simple enum" — behaves like
    :class:`enum.Enum` with variant class attributes and supports
    ``==``/``int()`` (see :meth:`__eq__`/:meth:`__int__`)."""

    Eos: "FinishReason"
    """An end-of-generation token was sampled."""
    MaxTokens: "FinishReason"
    """The requested ``max_tokens`` budget was exhausted."""
    ContextFull: "FinishReason"
    """The KV cache reached the model's context length."""
    Stopped: "FinishReason"
    """A caller-supplied stop sequence was produced."""
    Cancelled: "FinishReason"
    """A ``CancellationToken`` passed as ``cancel_token=`` was cancelled.

    The decode loop stops at the next token boundary and the outcome's
    ``text`` holds everything produced up to that point."""

    def as_openai_str(self) -> str:
        """The OpenAI ``finish_reason`` string.

        ``"stop"`` for ``Eos``/``Stopped``, ``"length"`` for
        ``MaxTokens``/``ContextFull``, and ``"cancelled"`` for ``Cancelled``
        (which has no OpenAI counterpart, so it is reported honestly rather
        than laundered into ``"stop"``)."""
        ...
    def is_natural(self) -> bool:
        """``True`` when the model chose to stop rather than being cut off."""
        ...
    def is_cancelled(self) -> bool:
        """``True`` when generation ended because the caller cancelled it."""
        ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...

class GenerationConfig:
    """Per-request generation settings for :meth:`Engine.generate_detailed`."""

    max_tokens: int
    sampler: SamplerConfig
    stop: list[str]
    render_special: bool
    add_special: bool
    parse_special: bool

    def __init__(
        self,
        max_tokens: int = 128,
        *,
        sampler: Optional[SamplerConfig] = None,
        stop: Optional[list[str]] = None,
        render_special: bool = False,
        add_special: bool = True,
        parse_special: bool = True,
    ) -> None: ...
    def __repr__(self) -> str: ...

class GenerationOutcome:
    """Everything describing a completed :meth:`Engine.generate_detailed` call."""

    text: str
    finish_reason: FinishReason
    generated_tokens: list[int]
    prompt_tokens: int
    stop_sequence: Optional[str]

    def completion_tokens(self) -> int: ...
    def total_tokens(self) -> int: ...
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
        cancel_token: Optional["CancellationToken"] = None,
        progress: ProgressLike = None,
        progress_throttle_ms: Optional[int] = None,
        progress_throttle_tokens: Optional[int] = None,
        progress_capture_text: bool = False,
        strict_progress: bool = False,
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
        cancel_token: Optional["CancellationToken"] = None,
        strict_callback: bool = False,
        progress: ProgressLike = None,
        progress_throttle_ms: Optional[int] = None,
        progress_throttle_tokens: Optional[int] = None,
        progress_capture_text: bool = False,
        strict_progress: bool = False,
    ) -> str: ...
    def generate_detailed(
        self,
        prompt: str,
        config: GenerationConfig,
        callback: Optional[Callable[[str], None]] = None,
        *,
        cancel_token: Optional["CancellationToken"] = None,
    ) -> GenerationOutcome: ...
    def eog_token_ids(self) -> list[int]: ...
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
    def snapshot(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        hub_origin: Optional[HubOrigin] = None,
    ) -> None: ...
    def snapshot_bytes(self) -> bytes: ...
    @classmethod
    def snapshot_info(cls, path: Union[str, os.PathLike[str]]) -> SnapshotInfo: ...
    @classmethod
    def restore(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        model_path: Optional[Union[str, os.PathLike[str]]] = None,
    ) -> Engine: ...
    @classmethod
    def from_snapshot_with_hub(
        cls,
        snapshot_path: Union[str, os.PathLike[str]],
    ) -> Engine: ...
    def logits_dlpack(self, text: str) -> object: ...
    """Return the logits for ``text`` as a DLPack ``"dltensor"`` PyCapsule.

    Shape: ``[vocab_size]``, dtype float32, device CPU.
    Compatible with ``torch.from_dlpack``, ``jax.dlpack.from_dlpack``, etc.
    """
    def embeddings_dlpack(self, text: str) -> object: ...
    """Return the last hidden-state embedding for ``text`` as a DLPack capsule.

    Shape: ``[1, hidden_size]``, dtype float32, device CPU.
    """
    def logits_torch(self, text: str, **kwargs: Any) -> "torch.Tensor": ...
    """Return the logit vector for ``text`` as a ``torch.Tensor``.

    Zero-copy via DLPack.  Shape: ``[vocab_size]``, dtype float32, device CPU.
    Requires ``torch`` to be installed (``pip install torch``).
    """
    def embeddings_torch(self, text: str, **kwargs: Any) -> "torch.Tensor": ...
    """Return the embedding vector for ``text`` as a ``torch.Tensor``.

    Zero-copy via DLPack.  Shape: ``[1, hidden_size]``, dtype float32, device CPU.
    Requires ``torch`` to be installed (``pip install torch``).
    """
    def async_engine(self) -> "AsyncEngine": ...
    def __reduce__(self) -> None: ...
    def __reduce_ex__(self, protocol: int) -> None: ...

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
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        *,
        progress: ProgressLike = None,
        progress_throttle_ms: Optional[int] = None,
        progress_throttle_tokens: Optional[int] = None,
        progress_capture_text: bool = False,
        strict_progress: bool = False,
    ) -> str: ...
    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
        callback: Optional[Callable[[str], None]] = None,
        *,
        progress: ProgressLike = None,
        progress_throttle_ms: Optional[int] = None,
        progress_throttle_tokens: Optional[int] = None,
        progress_capture_text: bool = False,
        strict_progress: bool = False,
    ) -> str: ...
    def snapshot(self, path: str) -> None: ...
    def snapshot_bytes(self) -> bytes: ...
    @classmethod
    def restore(cls, path: str, target_model: str, draft_model: str) -> "SpeculativeEngine": ...
    def __reduce__(self) -> tuple: ...
    def __reduce_ex__(self, protocol: int) -> tuple: ...

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

# ---------------------------------------------------------------------------
# CancellationToken
# ---------------------------------------------------------------------------

class CancellationToken:
    """A thread-safe cancellation token."""

    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...
    def reset(self) -> None: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# AsyncEngine (pure-Python, v0.1.5)
# ---------------------------------------------------------------------------

class AsyncEngine:
    """Pure-Python async wrapper around any synchronous inference engine.

    Accepts any object that exposes ``generate(prompt, max_tokens, **kwargs)``
    and ``generate_streaming(prompt, max_tokens, callback, **kwargs)`` methods,
    including the native :class:`Engine` and plain mock objects.

    Blocking calls are offloaded to a private
    :class:`~concurrent.futures.ThreadPoolExecutor` so that the asyncio event
    loop is never blocked during inference.

    Create via :meth:`Engine.async_engine` or directly::

        ae = AsyncEngine(engine)
        text = await ae.generate("Hello", max_tokens=128)

        async for token in ae.stream("Hello", max_tokens=64):
            print(token, end="", flush=True)
    """

    def __init__(self, engine: Any) -> None:
        """Wrap *engine* for async use.

        Args:
            engine: Any object with ``generate`` and ``generate_streaming``
                    methods.  Typically a native :class:`Engine` instance or
                    a test mock.
        """
        ...

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Return the full generated text for *prompt*.

        Runs ``Engine.generate`` in the thread-pool executor so that the
        asyncio event loop is not blocked during inference.

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
        ...

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
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
        """
        ...

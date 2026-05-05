"""
Zero-copy torch.Tensor interop via DLPack capsules.

Usage (auto-patched onto Engine at import time)::

    from oxillama_py import Engine

    engine = Engine.from_file(...)
    t = engine.logits_torch("hello")       # → torch.Tensor, shape [vocab_size], dtype float32
    e = engine.embeddings_torch("hello")   # → torch.Tensor, shape [1, hidden_size], dtype float32

Both methods delegate to the already-shipped DLPack capsule methods
``logits_dlpack`` and ``embeddings_dlpack`` and convert the resulting
``PyCapsule`` objects to ``torch.Tensor`` via ``torch.from_dlpack``.

The ``torch`` import happens lazily inside each method so that importing
this module — and therefore importing ``oxillama_py`` — does not fail when
PyTorch is not installed.  An :class:`ImportError` with an actionable
installation message is raised only at call time.

This module is patched onto the ``Engine`` class automatically by
``__init__.py`` via :func:`try_patch`.  If the native extension is not yet
built (no ``Engine`` class) the patch is silently skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Per-method helpers
# ---------------------------------------------------------------------------


def _make_logits_torch(self: Any, text: str, **kwargs: Any) -> "torch.Tensor":
    """Return the logit vector for ``text`` as a ``torch.Tensor``.

    Equivalent to calling ``engine.logits_dlpack(text)`` and converting the
    resulting DLPack capsule to a ``torch.Tensor`` via ``torch.from_dlpack``.

    The tensor has shape ``[vocab_size]``, dtype ``float32``, and lives on
    CPU.  The data is zero-copy when PyTorch supports DLPack on the CPU
    backend (PyTorch ≥ 1.10).

    Args:
        text: Input text whose logits are computed.
        **kwargs: Additional keyword arguments forwarded to
                  ``logits_dlpack``.

    Returns:
        ``torch.Tensor`` of shape ``[vocab_size]``, dtype ``float32``.

    Raises:
        ImportError: if ``torch`` is not installed.
        RuntimeError: if no model is loaded.
        ValueError: if ``text`` tokenizes to the empty sequence.
    """
    try:
        import torch as _torch
    except ImportError as exc:
        raise ImportError(
            "torch is not installed.  Install it with:\n"
            "    pip install torch\n"
            "logits_torch() requires PyTorch to convert DLPack capsules."
        ) from exc

    capsule = self.logits_dlpack(text, **kwargs)
    return _torch.from_dlpack(capsule)


def _make_embeddings_torch(self: Any, text: str, **kwargs: Any) -> "torch.Tensor":
    """Return the embedding vector for ``text`` as a ``torch.Tensor``.

    Equivalent to calling ``engine.embeddings_dlpack(text)`` and converting
    the resulting DLPack capsule to a ``torch.Tensor`` via
    ``torch.from_dlpack``.

    The tensor has shape ``[1, hidden_size]``, dtype ``float32``, and lives
    on CPU.

    Args:
        text: Input text whose embedding is computed.
        **kwargs: Additional keyword arguments forwarded to
                  ``embeddings_dlpack``.

    Returns:
        ``torch.Tensor`` of shape ``[1, hidden_size]``, dtype ``float32``.

    Raises:
        ImportError: if ``torch`` is not installed.
        RuntimeError: if no model is loaded.
    """
    try:
        import torch as _torch
    except ImportError as exc:
        raise ImportError(
            "torch is not installed.  Install it with:\n"
            "    pip install torch\n"
            "embeddings_torch() requires PyTorch to convert DLPack capsules."
        ) from exc

    capsule = self.embeddings_dlpack(text, **kwargs)
    return _torch.from_dlpack(capsule)


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------


def patch_engine_class(engine_cls: type) -> None:
    """Monkey-patch ``logits_torch`` and ``embeddings_torch`` onto *engine_cls*.

    The two methods are added as regular instance methods so they are
    visible from Python as ``engine.logits_torch(text)`` and
    ``engine.embeddings_torch(text)``.

    This function is idempotent — calling it multiple times on the same
    class is safe.

    Args:
        engine_cls: The class to patch.  Typically the native ``Engine``
                    class but any class with ``logits_dlpack`` and
                    ``embeddings_dlpack`` methods will work.
    """
    engine_cls.logits_torch = _make_logits_torch  # type: ignore[attr-defined]
    engine_cls.embeddings_torch = _make_embeddings_torch  # type: ignore[attr-defined]


def try_patch(module: Any) -> None:
    """Attempt to patch the ``Engine`` class found in *module*.

    Called from ``__init__.py`` immediately after the native extension is
    imported.  If ``Engine`` is not present (extension not built, or set to
    ``None`` due to a failed import) the function returns silently without
    raising.

    Any unexpected exception is also swallowed so that a buggy torch
    installation or a future API change never prevents ``oxillama_py``
    from importing.

    Args:
        module: The module object to search for an ``Engine`` attribute.
                Typically ``sys.modules[__name__]`` of ``oxillama_py``.
    """
    try:
        engine_cls = getattr(module, "Engine", None)
        if engine_cls is not None:
            patch_engine_class(engine_cls)
    except Exception:  # noqa: BLE001
        # Never fail at import time — missing torch or any other issue must
        # not prevent the rest of the package from being usable.
        pass

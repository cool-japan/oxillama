"""snapshot — Pure-Rust engine persistence for OxiLLaMa.

This module provides a ``pickle``-shaped surface for saving and restoring
:class:`~oxillama_py.Engine` state.  Unlike ``pickle``, the format is
stable, Pure-Rust, and does not serialize model weights (it stores only KV
cache, sampler config, and a Blake3 fingerprint of the GGUF file).

File format
-----------
``OXISNAP1`` magic + version ``u32`` + oxicode-encoded payload, identical to
the format used by ``oxillama-runtime`` internally.

Known limitations
-----------------
* Only transformer (attention) architectures round-trip — Mamba-2 / Jamba
  KV state is tracked internally but not yet emitted by ``snapshot()``.
* Grammar state resets to initial after restore (grammar *source* is
  preserved, but parser state is not).
* Sampler RNG reflects the config seed, not in-flight generation state.
* Offload policy is reset to ``None`` on restore.
* ``from_hub``-loaded engines snapshot the local HF cache path; restoring on
  a different machine requires the same GGUF to be present at the stored
  path, or an explicit ``model_path=`` override.  Hub-aware snapshots are
  tracked as R3 in ``crates/oxillama-py/TODO.md``.
* Token history (``tokens_count``) is always 0 in v0.1.3 because token
  history is not tracked at engine level.
* ``AsyncEngine.restore`` is not available in v0.1.3 — restore a snapshot
  synchronously via ``Engine.restore(path)`` and wrap the result in
  ``AsyncEngine`` if async access is needed.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from oxillama_py import Engine, SnapshotInfo

# OxiLlamaError is None when the native extension has not been compiled yet.
# Fall back to Exception so that SnapshotError is always importable and
# the pure-Python layer remains usable without a native build.
try:
    from oxillama_py.oxillama_py import OxiLlamaError as _OxiLlamaError  # type: ignore[import-untyped]
    _BASE_ERROR: type = _OxiLlamaError
except ImportError:
    _BASE_ERROR = Exception

__all__ = [
    "SnapshotError",
    "dump",
    "dumps",
    "load",
    "loads",
    "snapshot_info",
]


class SnapshotError(_BASE_ERROR):  # type: ignore[misc]
    """Raised when a snapshot operation fails (convenience alias).

    The underlying Rust methods raise either :exc:`~oxillama_py.LoadError`
    (model fingerprint mismatch) or :exc:`~oxillama_py.GenerateError`
    (malformed / incompatible snapshot).  Catch ``SnapshotError`` to handle
    both at once::

        try:
            engine = oxillama_py.snapshot.load("snap.oxsn")
        except oxillama_py.snapshot.SnapshotError as exc:
            print(f"restore failed: {exc}")
    """


def dump(engine: "Engine", path: Union[str, "os.PathLike[str]"]) -> None:
    """Write *engine* state to *path* atomically.

    Equivalent to ``engine.snapshot(path)``.  Raises :exc:`SnapshotError`
    (or a subclass) if the engine is not loaded or the write fails.
    """
    engine.snapshot(path)


def dumps(engine: "Engine") -> bytes:
    """Return *engine* state as a :class:`bytes` object.

    Equivalent to ``engine.snapshot_bytes()``.  Suitable for in-memory
    transport (e.g. multiprocessing queues, network sockets).
    """
    return engine.snapshot_bytes()


def load(
    path: Union[str, "os.PathLike[str]"],
    *,
    model_path: Optional[Union[str, "os.PathLike[str]"]] = None,
) -> "Engine":
    """Reconstruct an :class:`~oxillama_py.Engine` from *path*.

    If *model_path* is ``None`` (default), the model path embedded in the
    snapshot is used.  Pass an explicit *model_path* to override (useful
    when moving a snapshot between machines where the GGUF lives at a
    different absolute path).

    The GGUF model is re-loaded from disk on every restore.

    Raises :exc:`SnapshotError` (specifically :exc:`~oxillama_py.LoadError`
    for fingerprint mismatches and :exc:`~oxillama_py.GenerateError` for
    corrupted / incompatible snapshots).
    """
    from oxillama_py import Engine
    return Engine.restore(path, model_path=model_path)


def loads(
    blob: bytes,
    *,
    model_path: Union[str, "os.PathLike[str]"],
) -> "Engine":
    """Reconstruct an :class:`~oxillama_py.Engine` from an in-memory *blob*.

    *model_path* is required — the embedded path in the snapshot is used as
    the loading target, but *model_path* must be provided explicitly when
    working with an in-memory blob (since there is no file to re-read from).

    Writes the blob to a temporary file and calls :func:`load`.  This is the
    mirror of :func:`dumps`.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".oxsn", delete=False) as fh:
        fh.write(blob)
        tmp_path = fh.name
    try:
        return load(tmp_path, model_path=model_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def snapshot_info(path: Union[str, "os.PathLike[str]"]) -> "SnapshotInfo":
    """Return metadata from the snapshot at *path* without loading the model.

    The returned :class:`~oxillama_py.SnapshotInfo` exposes ``arch_id``,
    ``model_path``, ``tokenizer_path``, ``max_context_length``,
    ``num_threads``, ``version``, ``magic``, and ``tokens_count``.
    """
    from oxillama_py import Engine
    return Engine.snapshot_info(path)

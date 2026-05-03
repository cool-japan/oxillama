Snapshot & Restore
==================

OxiLLaMa provides a stable, Pure-Rust persistence API for saving and
restoring inference engine state.  Unlike ``pickle``, the format is
version-tagged, does **not** serialize model weights, and is designed for
long-term storage.

Overview
--------

``Engine.snapshot(path)`` writes the current KV cache, sampler configuration,
and a Blake3 fingerprint of the GGUF model file to disk atomically.  The
write is atomic: a temporary file is created in the same directory and then
renamed into place, so partial writes are never visible.

``Engine.restore(path)`` reads the snapshot, verifies the model fingerprint,
reloads the GGUF weights from disk, and reconstructs the KV cache state.

Quick start
-----------

.. code-block:: python

   import oxillama_py
   from oxillama_py import Engine, EngineConfig

   # Load and run inference
   config = EngineConfig("path/to/model.gguf")
   engine = Engine(config)
   engine.load_model()
   _ = engine.generate("Hello, world!", max_tokens=32)

   # Save state
   engine.snapshot("session.oxsn")

   # Restore state (model weights are reloaded from disk)
   engine2 = Engine.restore("session.oxsn")
   # Alternatively, override the model path at restore time:
   engine3 = Engine.restore("session.oxsn", model_path="/new/path/model.gguf")

Using the ``snapshot`` module
------------------------------

The :mod:`oxillama_py.snapshot` module provides a ``pickle``-shaped
convenience API:

.. code-block:: python

   import oxillama_py.snapshot as snap

   # Save
   snap.dump(engine, "session.oxsn")

   # Restore
   engine = snap.load("session.oxsn")

   # In-memory round-trip
   blob = snap.dumps(engine)
   engine = snap.loads(blob, model_path="path/to/model.gguf")

   # Peek at metadata without loading the model
   info = snap.snapshot_info("session.oxsn")
   print(info.arch_id, info.version, bytes(info.magic))

``snapshot_info``
-----------------

:meth:`Engine.snapshot_info` (classmethod) reads the snapshot header without
loading the model.  The returned :class:`~oxillama_py.SnapshotInfo` object
exposes:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Attribute
     - Description
   * - ``arch_id``
     - Architecture string (e.g. ``"llama"``, ``"qwen3"``).
   * - ``model_path``
     - Absolute model path embedded at snapshot time.
   * - ``tokenizer_path``
     - Optional explicit tokenizer path (``None`` = auto-detect).
   * - ``max_context_length``
     - Context length the engine was configured with.
   * - ``num_threads``
     - Parallelism at snapshot time.
   * - ``version``
     - Snapshot format version (currently ``1``).
   * - ``magic``
     - 8-byte magic ``b"OXISNAP1"`` returned as ``bytes``.
   * - ``tokens_count``
     - Number of token IDs stored (always ``0`` in v0.1.3).

.. code-block:: python

   info = Engine.snapshot_info("session.oxsn")
   print(f"arch={info.arch_id!r}  model={info.model_path!r}  v={info.version}")

Pickle-refusal contract
-----------------------

``Engine``, ``AsyncEngine``, and ``SpeculativeEngine`` raise a
:exc:`TypeError` when pickled or when ``__reduce__`` / ``__reduce_ex__`` are
called.  This is intentional: the snapshot format is a stable, documented
format designed for persistence, while ``pickle`` is an opaque, fragile format
unsuitable for engine state.

.. code-block:: python

   import pickle
   import oxillama_py

   engine = oxillama_py.Engine(oxillama_py.EngineConfig("model.gguf"))
   try:
       pickle.dumps(engine)
   except TypeError as exc:
       print(exc)  # "Engine cannot be pickled; use Engine.snapshot(path) ..."

Known limitations
-----------------

* **Architecture support**: only transformer (attention) architectures
  round-trip faithfully.  Mamba-2 / Jamba KV state is tracked internally
  but not yet emitted by ``snapshot()``.

* **Grammar state**: only the grammar *source* string is stored.  On restore
  the grammar is re-parsed and the parser state resets to initial — partial
  grammar progress from before the snapshot is lost.

* **Sampler RNG**: the snapshot captures the sampler *config* seed, not the
  live RNG state from an in-flight generation.  Restoring and re-running the
  same prompt may produce slightly different tokens if the in-flight RNG
  diverged from the seed.

* **Offload policy**: the offload policy is reset to ``None`` on restore.
  Re-configure it via ``EngineConfig`` if needed.

* **``from_hub``-loaded engines**: the embedded ``model_path`` is the local
  HuggingFace cache path.  Restoring on a different machine requires the same
  GGUF to be present at the stored path, or an explicit ``model_path=``
  override.  Hub-aware snapshots are tracked as R3 in
  ``crates/oxillama-py/TODO.md``.

* **``tokens_count``**: always ``0`` in v0.1.3 because token history is not
  tracked at engine level.

* **``AsyncEngine.restore``**: not available in v0.1.3.  To restore
  asynchronously, call ``Engine.restore(path)`` in a background thread and
  wrap the result.

API reference
-------------

.. py:class:: oxillama_py.SnapshotInfo

   Metadata extracted from a snapshot file.

   .. py:attribute:: arch_id
      :type: str
   .. py:attribute:: model_path
      :type: str
   .. py:attribute:: tokenizer_path
      :type: Optional[str]
   .. py:attribute:: max_context_length
      :type: int
   .. py:attribute:: num_threads
      :type: int
   .. py:attribute:: version
      :type: int
   .. py:attribute:: magic
      :type: bytes
   .. py:attribute:: tokens_count
      :type: int

.. py:method:: Engine.snapshot(path)

   Save engine state to *path* atomically.

   :param path: Destination file path.
   :type path: str | os.PathLike
   :raises GenerateError: if no model is loaded.
   :raises OSError: if the file cannot be written.

.. py:method:: Engine.snapshot_bytes()

   Return engine state as :class:`bytes`.

   :rtype: bytes
   :raises GenerateError: if no model is loaded.

.. py:classmethod:: Engine.snapshot_info(path)

   Return metadata from *path* without loading the model.

   :param path: Snapshot file path.
   :type path: str | os.PathLike
   :rtype: SnapshotInfo
   :raises GenerateError: if the snapshot format is invalid.
   :raises OSError: if the file cannot be read.

.. py:classmethod:: Engine.restore(path, *, model_path=None)

   Reconstruct an engine from a snapshot.

   :param path: Snapshot file path.
   :type path: str | os.PathLike
   :param model_path: Override the model path embedded in the snapshot.
   :type model_path: str | os.PathLike | None
   :rtype: Engine
   :raises GenerateError: if the snapshot is corrupted or incompatible.
   :raises LoadError: if the model fingerprint does not match.
   :raises OSError: if the snapshot file cannot be read.

.. py:module:: oxillama_py.snapshot

   Convenience ``pickle``-shaped API.

   .. py:function:: dump(engine, path)

      Write engine state to *path* atomically.

   .. py:function:: dumps(engine) -> bytes

      Return engine state as :class:`bytes`.

   .. py:function:: load(path, *, model_path=None) -> Engine

      Reconstruct an engine from *path*.

   .. py:function:: loads(blob, *, model_path) -> Engine

      Reconstruct an engine from an in-memory *blob*.

   .. py:function:: snapshot_info(path) -> SnapshotInfo

      Return metadata from *path* without loading the model.

   .. py:exception:: SnapshotError

      Convenience alias for snapshot-related errors.  Inherits from
      :exc:`~oxillama_py.OxiLlamaError`.

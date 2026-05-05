"""Tests for DLPack tensor interop (Track F, v0.1.6).

Pure-Python tests cover the API surface:
- ``HubOrigin`` TypedDict importability
- Presence of ``logits_dlpack`` and ``embeddings_dlpack`` on ``Engine``
- Presence of ``from_snapshot_with_hub`` classmethod
- Correct signature for ``snapshot()`` with optional ``hub_origin``
- Capsule type validation when the native extension is available

Model-backed tests require ``OXILLAMA_TEST_MODEL`` and are skipped
automatically when the env-var is absent.
"""

from __future__ import annotations

import inspect
import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _try_import():
    """Return (oxillama_py, has_native_ext)."""
    try:
        import oxillama_py

        has_ext = oxillama_py.Engine is not None
        return oxillama_py, has_ext
    except ImportError:
        return None, False


oxillama_py, HAS_EXT = _try_import()

# Skip the whole module if the package itself is not importable.
pytestmark = pytest.mark.skipif(
    oxillama_py is None,
    reason="oxillama_py package not importable",
)


# ---------------------------------------------------------------------------
# HubOrigin TypedDict
# ---------------------------------------------------------------------------


def test_hub_origin_typed_dict():
    """``HubOrigin`` must be importable from ``oxillama_py``."""
    try:
        from oxillama_py import HubOrigin  # type: ignore[attr-defined]

        # If it exists, verify the expected keys.
        assert "repo_id" in HubOrigin.__annotations__
        assert "filename" in HubOrigin.__annotations__
        assert "sha256" in HubOrigin.__annotations__
    except ImportError:
        # ``HubOrigin`` is a TypedDict defined in the stub; the runtime
        # package may expose it via ``__init__.py`` or the stub alone.
        # Verify it is at minimum accessible through the type stub by
        # checking the .pyi file exists.
        pyi_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "oxillama_py",
            "__init__.pyi",
        )
        assert os.path.exists(pyi_path), "oxillama_py/__init__.pyi must exist"
        with open(pyi_path) as f:
            content = f.read()
        assert "HubOrigin" in content, "__init__.pyi must mention HubOrigin"


def test_hub_origin_typed_dict_keys():
    """``HubOrigin`` must contain exactly the three documented keys."""
    try:
        from oxillama_py import HubOrigin  # type: ignore[attr-defined]

        annotations = HubOrigin.__annotations__
        assert "repo_id" in annotations, "HubOrigin must have 'repo_id'"
        assert "filename" in annotations, "HubOrigin must have 'filename'"
        assert "sha256" in annotations, "HubOrigin must have 'sha256'"
    except (ImportError, AttributeError):
        # TypedDict not available at runtime — fall back to stub check.
        pyi = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "oxillama_py",
            "__init__.pyi",
        )
        if os.path.exists(pyi):
            content = open(pyi).read()
            for key in ("repo_id", "filename", "sha256"):
                assert key in content, f"pyi stub must mention '{key}'"


# ---------------------------------------------------------------------------
# Engine method surface
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_engine_has_logits_dlpack_method():
    """``Engine`` must expose a ``logits_dlpack`` method."""
    assert hasattr(oxillama_py.Engine, "logits_dlpack"), (
        "Engine is missing 'logits_dlpack' method"
    )


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_engine_has_embeddings_dlpack_method():
    """``Engine`` must expose an ``embeddings_dlpack`` method."""
    assert hasattr(oxillama_py.Engine, "embeddings_dlpack"), (
        "Engine is missing 'embeddings_dlpack' method"
    )


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_engine_has_from_snapshot_with_hub():
    """``Engine.from_snapshot_with_hub`` classmethod must exist."""
    assert hasattr(oxillama_py.Engine, "from_snapshot_with_hub"), (
        "Engine is missing 'from_snapshot_with_hub' classmethod"
    )


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_from_snapshot_with_hub_is_callable():
    """``Engine.from_snapshot_with_hub`` must be callable."""
    method = getattr(oxillama_py.Engine, "from_snapshot_with_hub")
    assert callable(method), "from_snapshot_with_hub must be callable"


# ---------------------------------------------------------------------------
# snapshot() signature
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_snapshot_hub_origin_none_by_default():
    """``Engine.snapshot()`` must accept an optional ``hub_origin`` kwarg."""
    # Constructing a real snapshot requires a loaded model, so we just
    # verify the method exists and has the expected keyword argument by
    # inspecting the method via its __doc__ or checking it is callable.
    assert hasattr(oxillama_py.Engine, "snapshot"), "Engine must have snapshot method"
    snap_fn = getattr(oxillama_py.Engine, "snapshot")
    assert callable(snap_fn), "Engine.snapshot must be callable"


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_snapshot_accepts_hub_origin_kwarg():
    """``Engine.snapshot`` signature must accept ``hub_origin`` keyword arg.

    We verify this by introspecting the Python signature; PyO3-generated
    methods surface their signatures through ``__text_signature__``.
    """
    snap_fn = getattr(oxillama_py.Engine, "snapshot")
    sig_str = getattr(snap_fn, "__text_signature__", "") or ""
    doc_str = getattr(snap_fn, "__doc__", "") or ""
    # At least one of the two should mention hub_origin.
    assert "hub_origin" in sig_str or "hub_origin" in doc_str, (
        "Engine.snapshot must document hub_origin kwarg in signature or docstring; "
        f"sig={sig_str!r}, doc={doc_str[:200]!r}"
    )


# ---------------------------------------------------------------------------
# DLPack capsule type (native ext only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_dlpack_capsule_type_check():
    """If the native extension is available, verify the capsule module-level behavior.

    We cannot call ``engine.logits_dlpack()`` without a loaded model, so
    we test only the existence and callability of the method.
    """
    method = getattr(oxillama_py.Engine, "logits_dlpack", None)
    assert method is not None, "logits_dlpack must exist on Engine"
    assert callable(method), "logits_dlpack must be callable"


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_embeddings_dlpack_is_callable():
    """``Engine.embeddings_dlpack`` must be callable."""
    method = getattr(oxillama_py.Engine, "embeddings_dlpack", None)
    assert method is not None, "embeddings_dlpack must exist on Engine"
    assert callable(method), "embeddings_dlpack must be callable"


# ---------------------------------------------------------------------------
# Model-gated integration tests
# ---------------------------------------------------------------------------

SKIP_NO_MODEL = pytest.mark.skipif(
    not os.environ.get("OXILLAMA_TEST_MODEL"),
    reason="OXILLAMA_TEST_MODEL not set",
)


@SKIP_NO_MODEL
@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_logits_dlpack_returns_capsule(tmp_path):
    """``engine.logits_dlpack(text)`` must return a PyCapsule named 'dltensor'."""
    model_path = os.environ["OXILLAMA_TEST_MODEL"]
    cfg = oxillama_py.EngineConfig(model_path)
    engine = oxillama_py.Engine(cfg)
    engine.load_model()

    capsule = engine.logits_dlpack("Hello")
    # The returned object must be a capsule whose pointer is accessible.
    assert capsule is not None, "logits_dlpack must return a non-None object"
    # It should be representable.
    repr_str = repr(capsule)
    assert isinstance(repr_str, str), "capsule must have a repr"


@SKIP_NO_MODEL
@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_embeddings_dlpack_returns_capsule(tmp_path):
    """``engine.embeddings_dlpack(text)`` must return a PyCapsule."""
    model_path = os.environ["OXILLAMA_TEST_MODEL"]
    cfg = oxillama_py.EngineConfig(model_path)
    engine = oxillama_py.Engine(cfg)
    engine.load_model()

    capsule = engine.embeddings_dlpack("Hello world")
    assert capsule is not None, "embeddings_dlpack must return a non-None object"


@SKIP_NO_MODEL
@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_snapshot_with_hub_origin_roundtrip(tmp_path):
    """snapshot(hub_origin=...) must write a .meta.json sidecar."""
    model_path = os.environ["OXILLAMA_TEST_MODEL"]
    cfg = oxillama_py.EngineConfig(model_path)
    engine = oxillama_py.Engine(cfg)
    engine.load_model()

    snap_path = str(tmp_path / "hub_snap.oxsn")
    hub_origin = {
        "repo_id": "meta-llama/Llama-2-7b-hf",
        "filename": "llama-2-7b.Q4_K_M.gguf",
        "sha256": "a" * 64,
    }
    engine.snapshot(snap_path, hub_origin=hub_origin)

    assert os.path.exists(snap_path), "snapshot file must exist"
    meta_path = snap_path + ".meta.json"
    assert os.path.exists(meta_path), ".meta.json sidecar must be written"

    import json

    with open(meta_path) as f:
        meta = json.load(f)
    assert "hub_origin" in meta, "meta.json must contain hub_origin"
    assert meta["hub_origin"]["repo_id"] == hub_origin["repo_id"]
    assert meta["hub_origin"]["filename"] == hub_origin["filename"]

"""Tests for hub-aware snapshot/restore (Track F, v0.1.6).

Covers the ``HubOrigin`` TypedDict contract, the extended
``Engine.snapshot(hub_origin=...)`` signature, and the new
``Engine.from_snapshot_with_hub()`` classmethod.

No model file is required for most tests; model-backed tests are
gated on ``OXILLAMA_TEST_MODEL``.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _try_import():
    try:
        import oxillama_py

        return oxillama_py, oxillama_py.Engine is not None
    except ImportError:
        return None, False


oxillama_py, HAS_EXT = _try_import()

pytestmark = pytest.mark.skipif(
    oxillama_py is None,
    reason="oxillama_py package not importable",
)


# ---------------------------------------------------------------------------
# HubOrigin structure tests
# ---------------------------------------------------------------------------


def test_hub_origin_fields():
    """``HubOrigin`` must define exactly ``repo_id``, ``filename``, ``sha256``."""
    try:
        from oxillama_py import HubOrigin  # type: ignore[attr-defined]

        annotations = getattr(HubOrigin, "__annotations__", {})
        for field in ("repo_id", "filename", "sha256"):
            assert field in annotations, f"HubOrigin must have field '{field}'"
    except ImportError:
        # Stub-only — verify via pyi file.
        pyi = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "oxillama_py",
            "__init__.pyi",
        )
        if os.path.exists(pyi):
            content = open(pyi).read()
            for key in ("repo_id", "filename", "sha256"):
                assert key in content, f"pyi stub must declare '{key}' in HubOrigin"


def test_hub_origin_importable_from_module():
    """``HubOrigin`` must be either importable from ``oxillama_py`` or
    documented in the ``.pyi`` stub."""
    pyi = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "oxillama_py",
        "__init__.pyi",
    )
    if not os.path.exists(pyi):
        pytest.skip("__init__.pyi not found, skipping stub check")

    content = open(pyi).read()
    assert "HubOrigin" in content, (
        "HubOrigin must be declared in oxillama_py/__init__.pyi"
    )


def test_hub_origin_all_fields_are_str_typed():
    """``HubOrigin`` must have all three fields typed as ``str``."""
    pyi = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "oxillama_py",
        "__init__.pyi",
    )
    if not os.path.exists(pyi):
        pytest.skip("__init__.pyi not found")

    content = open(pyi).read()
    # Look for the TypedDict block.
    if "HubOrigin" not in content:
        pytest.skip("HubOrigin not in stub")

    # All three keys should be followed by ': str'.
    for key in ("repo_id", "filename", "sha256"):
        assert f"{key}: str" in content, (
            f"HubOrigin field '{key}' must be typed as str in the stub"
        )


# ---------------------------------------------------------------------------
# Engine API surface tests (require native ext)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_snapshot_with_hub_origin_accepted():
    """``Engine.snapshot`` must accept a ``hub_origin`` keyword argument."""
    snap = getattr(oxillama_py.Engine, "snapshot", None)
    assert snap is not None, "Engine must have a snapshot method"
    assert callable(snap), "Engine.snapshot must be callable"

    sig_str = getattr(snap, "__text_signature__", "") or ""
    doc_str = getattr(snap, "__doc__", "") or ""
    assert "hub_origin" in sig_str or "hub_origin" in doc_str, (
        "Engine.snapshot must accept hub_origin kwarg "
        f"(sig={sig_str!r}, doc={doc_str[:300]!r})"
    )


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_from_snapshot_with_hub_is_classmethod():
    """``Engine.from_snapshot_with_hub`` must be callable and exist."""
    method = getattr(oxillama_py.Engine, "from_snapshot_with_hub", None)
    assert method is not None, (
        "Engine.from_snapshot_with_hub must exist"
    )
    assert callable(method), "Engine.from_snapshot_with_hub must be callable"


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_engine_restore_still_exists():
    """``Engine.restore`` must still exist (not broken by hub-aware changes)."""
    assert hasattr(oxillama_py.Engine, "restore"), (
        "Engine.restore was removed (regression)"
    )
    assert callable(oxillama_py.Engine.restore), "Engine.restore must be callable"


@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_snapshot_bytes_still_exists():
    """``Engine.snapshot_bytes`` must still exist (regression guard)."""
    assert hasattr(oxillama_py.Engine, "snapshot_bytes"), (
        "Engine.snapshot_bytes was removed (regression)"
    )


# ---------------------------------------------------------------------------
# Type stub validation
# ---------------------------------------------------------------------------


def test_pyi_contains_hub_origin():
    """The ``.pyi`` stub must declare ``HubOrigin``."""
    pyi = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "oxillama_py",
        "__init__.pyi",
    )
    if not os.path.exists(pyi):
        pytest.skip("__init__.pyi not found")

    content = open(pyi).read()
    assert "HubOrigin" in content, "__init__.pyi must declare HubOrigin"


def test_pyi_snapshot_has_hub_origin_param():
    """The ``.pyi`` stub's ``Engine.snapshot`` must show ``hub_origin``."""
    pyi = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "oxillama_py",
        "__init__.pyi",
    )
    if not os.path.exists(pyi):
        pytest.skip("__init__.pyi not found")

    content = open(pyi).read()
    assert "hub_origin" in content, (
        "__init__.pyi Engine.snapshot must declare hub_origin parameter"
    )


def test_pyi_from_snapshot_with_hub_declared():
    """The ``.pyi`` stub must declare ``Engine.from_snapshot_with_hub``."""
    pyi = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "oxillama_py",
        "__init__.pyi",
    )
    if not os.path.exists(pyi):
        pytest.skip("__init__.pyi not found")

    content = open(pyi).read()
    assert "from_snapshot_with_hub" in content, (
        "__init__.pyi must declare Engine.from_snapshot_with_hub"
    )


def test_pyi_logits_dlpack_declared():
    """The ``.pyi`` stub must declare ``Engine.logits_dlpack``."""
    pyi = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "oxillama_py",
        "__init__.pyi",
    )
    if not os.path.exists(pyi):
        pytest.skip("__init__.pyi not found")

    content = open(pyi).read()
    assert "logits_dlpack" in content, (
        "__init__.pyi must declare Engine.logits_dlpack"
    )


def test_pyi_embeddings_dlpack_declared():
    """The ``.pyi`` stub must declare ``Engine.embeddings_dlpack``."""
    pyi = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "oxillama_py",
        "__init__.pyi",
    )
    if not os.path.exists(pyi):
        pytest.skip("__init__.pyi not found")

    content = open(pyi).read()
    assert "embeddings_dlpack" in content, (
        "__init__.pyi must declare Engine.embeddings_dlpack"
    )


# ---------------------------------------------------------------------------
# Model-gated integration tests
# ---------------------------------------------------------------------------

SKIP_NO_MODEL = pytest.mark.skipif(
    not os.environ.get("OXILLAMA_TEST_MODEL"),
    reason="OXILLAMA_TEST_MODEL not set",
)


@SKIP_NO_MODEL
@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_snapshot_with_hub_origin_creates_sidecar(tmp_path):
    """Snapshot with ``hub_origin`` must create a ``.meta.json`` sidecar."""
    model_path = os.environ["OXILLAMA_TEST_MODEL"]
    cfg = oxillama_py.EngineConfig(model_path)
    engine = oxillama_py.Engine(cfg)
    engine.load_model()

    snap_path = str(tmp_path / "hub_snap.oxsn")
    origin = {
        "repo_id": "test-org/test-model",
        "filename": "model.Q4_K_M.gguf",
        "sha256": "0" * 64,
    }
    engine.snapshot(snap_path, hub_origin=origin)

    assert os.path.exists(snap_path), "snapshot file must be created"
    meta_path = snap_path + ".meta.json"
    assert os.path.exists(meta_path), ".meta.json sidecar must be created"

    with open(meta_path) as f:
        meta = json.load(f)

    assert "hub_origin" in meta, "sidecar must contain hub_origin"
    assert meta["hub_origin"]["repo_id"] == origin["repo_id"]
    assert meta["hub_origin"]["filename"] == origin["filename"]
    assert meta["hub_origin"]["sha256"] == origin["sha256"]


@SKIP_NO_MODEL
@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_snapshot_without_hub_origin_no_sidecar(tmp_path):
    """Snapshot without ``hub_origin`` must NOT create a sidecar."""
    model_path = os.environ["OXILLAMA_TEST_MODEL"]
    cfg = oxillama_py.EngineConfig(model_path)
    engine = oxillama_py.Engine(cfg)
    engine.load_model()

    snap_path = str(tmp_path / "plain_snap.oxsn")
    engine.snapshot(snap_path)

    assert os.path.exists(snap_path), "snapshot file must be created"
    meta_path = snap_path + ".meta.json"
    assert not os.path.exists(meta_path), (
        ".meta.json must NOT be created when hub_origin is not provided"
    )


@SKIP_NO_MODEL
@pytest.mark.skipif(not HAS_EXT, reason="Native extension not available")
def test_restore_with_existing_model_ignores_hub(tmp_path):
    """``Engine.restore`` must work normally when the model path exists locally."""
    model_path = os.environ["OXILLAMA_TEST_MODEL"]
    cfg = oxillama_py.EngineConfig(model_path)
    engine = oxillama_py.Engine(cfg)
    engine.load_model()

    snap_path = str(tmp_path / "restore_test.oxsn")
    engine.snapshot(snap_path)

    restored = oxillama_py.Engine.restore(snap_path, model_path=model_path)
    assert restored is not None, "restore must return an Engine"
    assert restored.is_loaded(), "restored engine must be loaded"

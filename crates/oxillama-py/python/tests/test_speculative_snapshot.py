"""Tests for SpeculativeEngine snapshot/restore API (Track E, v0.1.4).

Non-gated tests cover only the presence and signature of the public methods.
Model-backed tests are gated on the OXILLAMA_TEST_TARGET_MODEL environment
variable to avoid CI failures on machines without GGUF files.
"""

import os
import pathlib
import tempfile
import inspect

import pytest

try:
    import oxillama_py
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False

pytestmark = pytest.mark.skipif(
    not HAS_EXTENSION,
    reason="oxillama_py native extension not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SKIP_NO_MODEL = pytest.mark.skipif(
    not os.environ.get("OXILLAMA_TEST_TARGET_MODEL"),
    reason="OXILLAMA_TEST_TARGET_MODEL not set",
)


# ---------------------------------------------------------------------------
# Surface / signature tests (no model required)
# ---------------------------------------------------------------------------

def test_speculative_engine_has_snapshot_method():
    """SpeculativeEngine must expose a 'snapshot' method."""
    assert hasattr(oxillama_py.SpeculativeEngine, "snapshot"), (
        "SpeculativeEngine is missing the 'snapshot' method"
    )


def test_speculative_engine_has_snapshot_bytes_method():
    """SpeculativeEngine must expose a 'snapshot_bytes' method."""
    assert hasattr(oxillama_py.SpeculativeEngine, "snapshot_bytes"), (
        "SpeculativeEngine is missing the 'snapshot_bytes' method"
    )


def test_speculative_engine_has_restore_classmethod():
    """SpeculativeEngine must expose a 'restore' classmethod."""
    assert hasattr(oxillama_py.SpeculativeEngine, "restore"), (
        "SpeculativeEngine is missing the 'restore' classmethod"
    )


def test_speculative_engine_has_reduce():
    """SpeculativeEngine must expose '__reduce__' for pickle support."""
    assert hasattr(oxillama_py.SpeculativeEngine, "__reduce__"), (
        "SpeculativeEngine is missing '__reduce__'"
    )


def test_speculative_engine_has_reduce_ex():
    """SpeculativeEngine must expose '__reduce_ex__' for pickle support."""
    assert hasattr(oxillama_py.SpeculativeEngine, "__reduce_ex__"), (
        "SpeculativeEngine is missing '__reduce_ex__'"
    )


def test_speculative_config_attributes():
    """SpeculativeConfig must expose target, draft, num_speculative, seed."""
    for attr in ("target", "draft", "num_speculative", "seed"):
        assert hasattr(oxillama_py.SpeculativeConfig, attr), (
            f"SpeculativeConfig is missing attribute '{attr}'"
        )


def test_snapshot_info_class_exists():
    """SnapshotInfo class must be importable from the extension."""
    assert hasattr(oxillama_py, "SnapshotInfo"), (
        "oxillama_py.SnapshotInfo not found"
    )


def test_engine_has_snapshot_method():
    """Single-model Engine must still expose 'snapshot' (not broken by Track E)."""
    assert hasattr(oxillama_py.Engine, "snapshot"), (
        "Engine.snapshot was removed (regression)"
    )


def test_engine_has_restore_classmethod():
    """Single-model Engine must still expose 'restore'."""
    assert hasattr(oxillama_py.Engine, "restore"), (
        "Engine.restore was removed (regression)"
    )


def test_speculative_engine_class_is_in_module():
    """SpeculativeEngine must be importable from the extension top-level."""
    assert hasattr(oxillama_py, "SpeculativeEngine"), (
        "SpeculativeEngine not found in oxillama_py"
    )


def test_speculative_config_class_is_in_module():
    """SpeculativeConfig must be importable from the extension top-level."""
    assert hasattr(oxillama_py, "SpeculativeConfig"), (
        "SpeculativeConfig not found in oxillama_py"
    )


# ---------------------------------------------------------------------------
# Model-gated integration tests
# ---------------------------------------------------------------------------

@SKIP_NO_MODEL
def test_spec_snapshot_roundtrip_real():
    """Full snapshot → restore roundtrip with real model files."""
    target = os.environ["OXILLAMA_TEST_TARGET_MODEL"]
    draft = os.environ.get("OXILLAMA_TEST_DRAFT_MODEL", target)

    target_cfg = oxillama_py.EngineConfig(model_path=target)
    draft_cfg = oxillama_py.EngineConfig(model_path=draft)
    spec_cfg = oxillama_py.SpeculativeConfig(
        target=target_cfg,
        draft=draft_cfg,
        num_speculative=3,
    )
    eng = oxillama_py.SpeculativeEngine(spec_cfg)

    with tempfile.TemporaryDirectory() as d:
        snap_path = str(pathlib.Path(d) / "spec.snap")
        eng.snapshot(snap_path)
        assert os.path.exists(snap_path), "snapshot file must be created"
        assert os.path.getsize(snap_path) > 0, "snapshot file must not be empty"

        eng2 = oxillama_py.SpeculativeEngine.restore(snap_path, target, draft)
        assert eng2 is not None, "restore must return a SpeculativeEngine"


@SKIP_NO_MODEL
def test_spec_snapshot_bytes_roundtrip_real():
    """snapshot_bytes → write to file → restore roundtrip."""
    target = os.environ["OXILLAMA_TEST_TARGET_MODEL"]
    draft = os.environ.get("OXILLAMA_TEST_DRAFT_MODEL", target)

    target_cfg = oxillama_py.EngineConfig(model_path=target)
    draft_cfg = oxillama_py.EngineConfig(model_path=draft)
    spec_cfg = oxillama_py.SpeculativeConfig(
        target=target_cfg,
        draft=draft_cfg,
        num_speculative=2,
    )
    eng = oxillama_py.SpeculativeEngine(spec_cfg)

    with tempfile.TemporaryDirectory() as d:
        snap_path = pathlib.Path(d) / "spec_bytes.snap"
        raw = eng.snapshot_bytes()
        assert isinstance(raw, bytes), "snapshot_bytes must return bytes"
        assert len(raw) > 0, "snapshot bytes must not be empty"

        snap_path.write_bytes(raw)
        eng2 = oxillama_py.SpeculativeEngine.restore(
            str(snap_path), target, draft
        )
        assert eng2 is not None


@SKIP_NO_MODEL
def test_spec_restore_generates_text():
    """Restored engine must be able to generate text."""
    target = os.environ["OXILLAMA_TEST_TARGET_MODEL"]
    draft = os.environ.get("OXILLAMA_TEST_DRAFT_MODEL", target)

    target_cfg = oxillama_py.EngineConfig(model_path=target)
    draft_cfg = oxillama_py.EngineConfig(model_path=draft)
    spec_cfg = oxillama_py.SpeculativeConfig(
        target=target_cfg,
        draft=draft_cfg,
        num_speculative=2,
    )
    eng = oxillama_py.SpeculativeEngine(spec_cfg)

    with tempfile.TemporaryDirectory() as d:
        snap_path = str(pathlib.Path(d) / "gen_test.snap")
        eng.snapshot(snap_path)
        eng2 = oxillama_py.SpeculativeEngine.restore(snap_path, target, draft)
        result = eng2.generate("Hello", max_tokens=4)
        assert isinstance(result, str), "restored engine must produce a string"

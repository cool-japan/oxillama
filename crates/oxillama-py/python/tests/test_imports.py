"""Test that all public symbols are importable from oxillama_py.

These tests validate only the Python-layer imports (callback.py, __init__.py)
and the native-extension symbols when the shared library is available.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Pure-Python symbols (always available — no native extension needed)
# ---------------------------------------------------------------------------


def test_import_streaming_callback_from_callback_module():
    from oxillama_py.callback import StreamingCallback  # noqa: F401


def test_import_token_callback_from_callback_module():
    from oxillama_py.callback import TokenCallback  # noqa: F401


def test_import_streaming_callback_from_top_level():
    """StreamingCallback must be re-exported at the package top level."""
    import oxillama_py

    assert hasattr(oxillama_py, "StreamingCallback")


def test_import_token_callback_from_top_level():
    import oxillama_py

    assert hasattr(oxillama_py, "TokenCallback")


# ---------------------------------------------------------------------------
# Native-extension symbols (skipped when the .so/.dylib is absent)
# ---------------------------------------------------------------------------


def _try_import():
    """Attempt to import the native extension; return (module, available)."""
    try:
        import oxillama_py.oxillama_py as _m  # type: ignore[import-untyped]

        return _m, True
    except ImportError:
        return None, False


def test_native_engine_config_importable():
    mod, ok = _try_import()
    if not ok:
        import pytest

        pytest.skip("Native extension not built")
    assert hasattr(mod, "EngineConfig")


def test_native_engine_importable():
    mod, ok = _try_import()
    if not ok:
        import pytest

        pytest.skip("Native extension not built")
    assert hasattr(mod, "Engine")


def test_native_sampler_config_importable():
    mod, ok = _try_import()
    if not ok:
        import pytest

        pytest.skip("Native extension not built")
    assert hasattr(mod, "SamplerConfig")


def test_native_cancellation_token_importable():
    mod, ok = _try_import()
    if not ok:
        import pytest

        pytest.skip("Native extension not built")
    assert hasattr(mod, "CancellationToken")


def test_native_exceptions_importable():
    mod, ok = _try_import()
    if not ok:
        import pytest

        pytest.skip("Native extension not built")
    for name in ("OxiLlamaError", "LoadError", "GenerateError", "TokenizerError"):
        assert hasattr(mod, name), f"Missing exception class: {name}"


def test_snapshot_info_importable_from_oxillama_py():
    """SnapshotInfo must be importable from oxillama_py (when native ext is built)."""
    mod, ok = _try_import()
    if not ok:
        import pytest
        pytest.skip("Native extension not built")
    assert hasattr(mod, "SnapshotInfo"), "SnapshotInfo must be exported from native extension"


def test_snapshot_error_importable():
    """SnapshotError must be importable from oxillama_py or oxillama_py.snapshot."""
    import oxillama_py.snapshot as snap_mod
    assert hasattr(snap_mod, "SnapshotError"), "SnapshotError must be exported from oxillama_py.snapshot"
    import oxillama_py
    assert hasattr(oxillama_py, "SnapshotError"), "SnapshotError must be re-exported from oxillama_py"


def test_snapshot_dump_callable():
    """oxillama_py.snapshot.dump must be callable."""
    import oxillama_py.snapshot as snap_mod
    assert callable(snap_mod.dump), "oxillama_py.snapshot.dump must be callable"

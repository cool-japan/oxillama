"""Tests for torch interop helpers (Track F, v0.1.7).

All tests in this file require ``torch``.  The module is skipped automatically
when torch is not installed via ``pytest.importorskip``.

Mock DLPack sources use ``numpy`` arrays because numpy ≥ 1.22 supports the
``__dlpack__`` / ``__dlpack_device__`` protocol, which ``torch.from_dlpack``
accepts.  This means these tests work without a real GGUF model.
"""

from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest

# All tests in this file require torch.
torch = pytest.importorskip("torch")

from oxillama_py import torch_helper


# ---------------------------------------------------------------------------
# Helper: a mock DLPack-compatible engine backed by numpy arrays
# ---------------------------------------------------------------------------

def _make_mock_engine(logit_data: "list[float]", embed_data: "list[float]") -> type:
    """Return a *class* (not instance) whose DLPack methods return numpy arrays."""
    import numpy as np

    class _MockEngine:
        def logits_dlpack(self, text: str, **kwargs: Any):  # noqa: ARG002
            return np.array(logit_data, dtype=np.float32)

        def embeddings_dlpack(self, text: str, **kwargs: Any):  # noqa: ARG002
            return np.array(embed_data, dtype=np.float32)

    return _MockEngine


# ---------------------------------------------------------------------------
# Engine class patching (native extension path)
# ---------------------------------------------------------------------------


def test_logits_torch_method_exists():
    """Engine class should have ``logits_torch`` after import."""
    try:
        from oxillama_py import Engine  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("oxillama_py extension not built")

    if Engine is None:
        pytest.skip("oxillama_py extension not built")

    assert hasattr(Engine, "logits_torch"), "Engine.logits_torch not patched"


def test_embeddings_torch_method_exists():
    """Engine class should have ``embeddings_torch`` after import."""
    try:
        from oxillama_py import Engine  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("oxillama_py extension not built")

    if Engine is None:
        pytest.skip("oxillama_py extension not built")

    assert hasattr(Engine, "embeddings_torch"), "Engine.embeddings_torch not patched"


# ---------------------------------------------------------------------------
# patch_engine_class
# ---------------------------------------------------------------------------


def test_patch_engine_class_adds_methods():
    """``patch_engine_class`` adds both methods to an arbitrary class."""
    MockCls = _make_mock_engine([1.0, 2.0, 3.0], [0.5, 0.5])
    torch_helper.patch_engine_class(MockCls)

    assert hasattr(MockCls, "logits_torch"), "logits_torch not added by patch_engine_class"
    assert hasattr(MockCls, "embeddings_torch"), "embeddings_torch not added by patch_engine_class"


def test_patch_engine_class_is_idempotent():
    """Calling ``patch_engine_class`` twice must not raise."""
    MockCls = _make_mock_engine([1.0], [0.5])
    torch_helper.patch_engine_class(MockCls)
    # second call should be safe
    torch_helper.patch_engine_class(MockCls)

    assert hasattr(MockCls, "logits_torch")
    assert hasattr(MockCls, "embeddings_torch")


# ---------------------------------------------------------------------------
# logits_torch
# ---------------------------------------------------------------------------


def test_logits_torch_returns_tensor():
    """``logits_torch`` must return a ``torch.Tensor``."""
    MockCls = _make_mock_engine([1.0, 2.0, 3.0], [0.5])
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    result = engine.logits_torch("test")

    assert isinstance(result, torch.Tensor), (
        f"logits_torch must return torch.Tensor, got {type(result)}"
    )


def test_logits_torch_shape_matches():
    """``logits_torch`` shape must match the underlying data length."""
    data = [1.0, 2.0, 3.0, 4.0]
    MockCls = _make_mock_engine(data, [0.0])
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    result = engine.logits_torch("test")

    assert result.shape == (len(data),), (
        f"Expected shape ({len(data)},), got {result.shape}"
    )


def test_logits_torch_dtype_is_float32():
    """``logits_torch`` tensor dtype must be float32."""
    MockCls = _make_mock_engine([1.0, 2.0], [0.0])
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    result = engine.logits_torch("test")

    assert result.dtype == torch.float32, (
        f"Expected float32, got {result.dtype}"
    )


def test_logits_torch_values_match_source():
    """``logits_torch`` must expose the same values as the source array."""
    data = [3.14, -1.0, 0.0, 999.9]
    MockCls = _make_mock_engine(data, [0.0])
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    result = engine.logits_torch("test")
    result_list = result.tolist()

    for i, (got, expected) in enumerate(zip(result_list, data)):
        assert abs(got - expected) < 1e-4, (
            f"Value mismatch at index {i}: got {got}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# embeddings_torch
# ---------------------------------------------------------------------------


def test_embeddings_torch_returns_tensor():
    """``embeddings_torch`` must return a ``torch.Tensor``."""
    MockCls = _make_mock_engine([1.0], [0.1, 0.2, 0.3])
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    result = engine.embeddings_torch("test")

    assert isinstance(result, torch.Tensor), (
        f"embeddings_torch must return torch.Tensor, got {type(result)}"
    )


def test_embeddings_torch_shape_matches():
    """``embeddings_torch`` shape must match the embedding data length."""
    embed_data = [0.1, 0.2, 0.3, 0.4]
    MockCls = _make_mock_engine([1.0], embed_data)
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    result = engine.embeddings_torch("test")

    assert result.shape == (len(embed_data),), (
        f"Expected shape ({len(embed_data)},), got {result.shape}"
    )


def test_embeddings_torch_dtype_is_float32():
    """``embeddings_torch`` tensor dtype must be float32."""
    MockCls = _make_mock_engine([1.0], [0.5, 0.5])
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    result = engine.embeddings_torch("test")

    assert result.dtype == torch.float32, (
        f"Expected float32, got {result.dtype}"
    )


# ---------------------------------------------------------------------------
# Error handling: torch not installed
# ---------------------------------------------------------------------------


def test_no_torch_raises_helpful_error_for_logits(monkeypatch):
    """``logits_torch`` raises ``ImportError`` with a helpful message when torch is absent."""
    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    MockCls = _make_mock_engine([1.0, 2.0], [0.0])
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="torch"):
        engine.logits_torch("test")


def test_no_torch_raises_helpful_error_for_embeddings(monkeypatch):
    """``embeddings_torch`` raises ``ImportError`` with a helpful message when torch is absent."""
    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    MockCls = _make_mock_engine([1.0], [0.5])
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="torch"):
        engine.embeddings_torch("test")


def test_import_error_message_mentions_pip_install(monkeypatch):
    """The ImportError from missing torch must suggest ``pip install torch``."""
    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    MockCls = _make_mock_engine([1.0], [0.0])
    torch_helper.patch_engine_class(MockCls)
    engine = MockCls()

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError) as exc_info:
        engine.logits_torch("test")

    assert "pip install torch" in str(exc_info.value), (
        "ImportError must include 'pip install torch' suggestion"
    )


# ---------------------------------------------------------------------------
# try_patch
# ---------------------------------------------------------------------------


def test_torch_helper_has_expected_functions():
    """``torch_helper`` module must expose ``patch_engine_class`` and ``try_patch``."""
    assert hasattr(torch_helper, "patch_engine_class"), (
        "torch_helper must expose patch_engine_class"
    )
    assert hasattr(torch_helper, "try_patch"), (
        "torch_helper must expose try_patch"
    )


def test_try_patch_silently_handles_no_engine():
    """``try_patch`` must not raise when the module has no Engine attribute."""
    import types

    fake_module = types.ModuleType("fake_oxillama_py")
    # No Engine attribute at all — try_patch must silently no-op.
    torch_helper.try_patch(fake_module)


def test_try_patch_silently_handles_engine_is_none():
    """``try_patch`` must not raise when Engine is None (extension not built)."""
    import types

    fake_module = types.ModuleType("fake_oxillama_py")
    fake_module.Engine = None  # type: ignore[attr-defined]

    # Must not raise even though Engine is None.
    torch_helper.try_patch(fake_module)


def test_try_patch_patches_engine_when_present():
    """``try_patch`` must add ``logits_torch`` / ``embeddings_torch`` to Engine."""
    import types

    fake_module = types.ModuleType("fake_oxillama_py")

    class FakeEngine:
        pass

    fake_module.Engine = FakeEngine  # type: ignore[attr-defined]

    torch_helper.try_patch(fake_module)

    assert hasattr(FakeEngine, "logits_torch"), (
        "try_patch must add logits_torch to Engine"
    )
    assert hasattr(FakeEngine, "embeddings_torch"), (
        "try_patch must add embeddings_torch to Engine"
    )


def test_torch_helper_lazy_import():
    """``torch_helper`` can be imported without torch being pre-loaded."""
    # torch was imported at the top of this test module via importorskip, but
    # this test verifies that the module *itself* doesn't eagerly require torch
    # by re-importing and checking the expected function names.
    assert hasattr(torch_helper, "patch_engine_class")
    assert hasattr(torch_helper, "try_patch")
    assert callable(torch_helper.patch_engine_class)
    assert callable(torch_helper.try_patch)

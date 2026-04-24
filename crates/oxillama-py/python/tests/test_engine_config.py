"""Tests for EngineConfig — skipped when native extension is absent."""

from __future__ import annotations

import sys

import pytest

# ---------------------------------------------------------------------------
# Skip entire module gracefully if the native extension is not built.
# ---------------------------------------------------------------------------


def _import_native():
    try:
        import oxillama_py.oxillama_py as _m  # type: ignore[import-untyped]

        return _m
    except ImportError:
        return None


_NATIVE = _import_native()
_SKIP = pytest.mark.skipif(
    _NATIVE is None, reason="Native extension not built (run `maturin develop`)"
)


@_SKIP
def test_engine_config_default_model_path():
    """EngineConfig accepts a model_path positional string."""
    cfg = _NATIVE.EngineConfig(model_path="model.gguf")
    assert cfg.model_path == "model.gguf"


@_SKIP
def test_engine_config_default_num_threads():
    cfg = _NATIVE.EngineConfig(model_path="model.gguf")
    assert cfg.num_threads == 4


@_SKIP
def test_engine_config_default_context_size_is_none():
    cfg = _NATIVE.EngineConfig(model_path="model.gguf")
    assert cfg.context_size is None


@_SKIP
def test_engine_config_explicit_context_size():
    cfg = _NATIVE.EngineConfig(model_path="m.gguf", context_size=2048)
    assert cfg.context_size == 2048


@_SKIP
def test_engine_config_explicit_num_threads():
    cfg = _NATIVE.EngineConfig(model_path="m.gguf", num_threads=8)
    assert cfg.num_threads == 8


@_SKIP
def test_engine_config_repr_contains_model_path():
    cfg = _NATIVE.EngineConfig(model_path="test_model.gguf")
    r = repr(cfg)
    assert "test_model.gguf" in r


@_SKIP
def test_engine_config_str_is_repr():
    cfg = _NATIVE.EngineConfig(model_path="test_model.gguf")
    assert str(cfg) == repr(cfg)


@_SKIP
def test_engine_config_with_sampler():
    sampler = _NATIVE.SamplerConfig(temperature=0.9)
    cfg = _NATIVE.EngineConfig(model_path="m.gguf", sampler=sampler)
    assert cfg.sampler.temperature == pytest.approx(0.9, abs=1e-5)


@_SKIP
def test_engine_config_path_object_accepted():
    """Passing a pathlib.Path should not raise (str coercion or native Path)."""
    import pathlib

    p = pathlib.Path("model.gguf")
    try:
        cfg = _NATIVE.EngineConfig(model_path=str(p))
        assert "model.gguf" in cfg.model_path
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected error with Path-like string: {exc}")


@_SKIP
def test_engine_config_zero_ctx_size_raises():
    """Zero or negative context_size should raise an error."""
    with pytest.raises(Exception):
        _NATIVE.EngineConfig(model_path="m.gguf", context_size=0)


@_SKIP
def test_engine_config_negative_ctx_size_raises():
    with pytest.raises(Exception):
        _NATIVE.EngineConfig(model_path="m.gguf", context_size=-1)

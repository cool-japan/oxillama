"""Tests for SamplerConfig — skipped when native extension is absent."""

from __future__ import annotations

import pytest


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
def test_sampler_config_default_temperature():
    sc = _NATIVE.SamplerConfig()
    assert sc.temperature == pytest.approx(0.7, abs=1e-5)


@_SKIP
def test_sampler_config_default_top_k():
    sc = _NATIVE.SamplerConfig()
    assert sc.top_k == 40


@_SKIP
def test_sampler_config_default_top_p():
    sc = _NATIVE.SamplerConfig()
    assert sc.top_p == pytest.approx(0.9, abs=1e-5)


@_SKIP
def test_sampler_config_default_seed_is_none():
    sc = _NATIVE.SamplerConfig()
    assert sc.seed is None


@_SKIP
def test_sampler_config_explicit_temperature():
    sc = _NATIVE.SamplerConfig(temperature=1.5)
    assert sc.temperature == pytest.approx(1.5, abs=1e-5)


@_SKIP
def test_sampler_config_explicit_top_k():
    sc = _NATIVE.SamplerConfig(top_k=10)
    assert sc.top_k == 10


@_SKIP
def test_sampler_config_explicit_seed():
    sc = _NATIVE.SamplerConfig(seed=42)
    assert sc.seed == 42


@_SKIP
def test_sampler_config_greedy():
    gsc = _NATIVE.SamplerConfig.greedy()
    assert gsc.temperature == pytest.approx(0.0, abs=1e-5)


@_SKIP
def test_sampler_config_mirostat_v2():
    msc = _NATIVE.SamplerConfig.mirostat_v2(tau=3.0, eta=0.05)
    assert msc.mirostat == 2
    assert msc.mirostat_tau == pytest.approx(3.0, abs=1e-5)
    assert msc.mirostat_eta == pytest.approx(0.05, abs=1e-5)


@_SKIP
def test_sampler_config_negative_temperature_raises():
    with pytest.raises(Exception):
        _NATIVE.SamplerConfig(temperature=-0.1)


@_SKIP
def test_sampler_config_repr():
    sc = _NATIVE.SamplerConfig(temperature=0.5)
    r = repr(sc)
    assert "SamplerConfig" in r or "0.5" in r

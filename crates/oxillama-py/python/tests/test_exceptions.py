"""Tests for the exception hierarchy — skipped when native extension is absent."""

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
def test_oxi_llama_error_is_exception():
    assert issubclass(_NATIVE.OxiLlamaError, Exception)


@_SKIP
def test_load_error_inherits_oxi_llama_error():
    assert issubclass(_NATIVE.LoadError, _NATIVE.OxiLlamaError)


@_SKIP
def test_generate_error_inherits_oxi_llama_error():
    assert issubclass(_NATIVE.GenerateError, _NATIVE.OxiLlamaError)


@_SKIP
def test_tokenizer_error_inherits_oxi_llama_error():
    assert issubclass(_NATIVE.TokenizerError, _NATIVE.OxiLlamaError)


@_SKIP
def test_grammar_error_inherits_oxi_llama_error():
    assert issubclass(_NATIVE.GrammarError, _NATIVE.OxiLlamaError)


@_SKIP
def test_quant_error_inherits_oxi_llama_error():
    assert issubclass(_NATIVE.QuantError, _NATIVE.OxiLlamaError)


@_SKIP
def test_load_error_caught_as_oxi_llama_error():
    """An instance of LoadError can be caught by the parent class."""
    try:
        raise _NATIVE.LoadError("test error")
    except _NATIVE.OxiLlamaError as exc:
        assert "test error" in str(exc)
    else:
        pytest.fail("LoadError not caught by OxiLlamaError")


@_SKIP
def test_generate_error_caught_as_oxi_llama_error():
    try:
        raise _NATIVE.GenerateError("gen error")
    except _NATIVE.OxiLlamaError:
        pass
    else:
        pytest.fail("GenerateError not caught by OxiLlamaError")


@_SKIP
def test_generate_error_caught_as_base_exception():
    try:
        raise _NATIVE.GenerateError("gen error")
    except Exception:
        pass
    else:
        pytest.fail("GenerateError not caught by Exception")

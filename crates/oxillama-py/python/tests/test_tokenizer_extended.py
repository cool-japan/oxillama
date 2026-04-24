"""Tests for new PyTokenizer features and KvCacheFullError."""

import pytest


def test_encode_batch_method_exists():
    from oxillama_py import Tokenizer

    assert hasattr(Tokenizer, "encode_batch"), "Tokenizer must have encode_batch"


def test_apply_chat_template_method_exists():
    from oxillama_py import Tokenizer

    assert hasattr(
        Tokenizer, "apply_chat_template"
    ), "Tokenizer must have apply_chat_template"


def test_kv_cache_full_error_import():
    from oxillama_py import KvCacheFullError, OxiLlamaError

    assert issubclass(
        KvCacheFullError, OxiLlamaError
    ), "KvCacheFullError must subclass OxiLlamaError"


def test_kv_cache_full_error_is_exception():
    from oxillama_py import KvCacheFullError

    assert issubclass(
        KvCacheFullError, Exception
    ), "KvCacheFullError must subclass Exception"


def test_generate_streaming_strict_callback_kwarg():
    """Engine.generate_streaming must accept strict_callback kwarg (type check only)."""
    from oxillama_py import Engine

    import inspect

    sig = inspect.signature(Engine.generate_streaming)
    assert (
        "strict_callback" in sig.parameters
    ), "generate_streaming must accept strict_callback parameter"

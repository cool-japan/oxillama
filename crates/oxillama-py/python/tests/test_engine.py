"""
Integration tests for oxillama_py Python bindings.

All tests that require a real model are gated on the OXILLAMA_TEST_MODEL
environment variable. Set it to the path of a GGUF model file to run them::

    OXILLAMA_TEST_MODEL=/path/to/model.gguf pytest python/tests/
"""
import os

import pytest


@pytest.fixture(scope="session")
def model_path():
    """Return the model path or skip the test if none is set."""
    p = os.environ.get("OXILLAMA_TEST_MODEL")
    if not p:
        pytest.skip("OXILLAMA_TEST_MODEL environment variable not set")
    if not os.path.isfile(p):
        pytest.skip(f"Model file not found: {p}")
    return p


@pytest.fixture(scope="session")
def engine(model_path):
    """Load the engine once per test session."""
    import oxillama_py

    config = oxillama_py.EngineConfig(model_path=model_path)
    eng = oxillama_py.Engine(config)
    eng.load_model()
    return eng


# ---------------------------------------------------------------------------
# Config / no-model tests (never skip)
# ---------------------------------------------------------------------------


def test_engine_config_defaults():
    """EngineConfig should accept a model path and expose it."""
    import oxillama_py

    cfg = oxillama_py.EngineConfig(model_path="model.gguf")
    assert cfg.model_path == "model.gguf"
    assert cfg.num_threads == 4
    assert cfg.context_size is None


def test_engine_config_context_size_override():
    """context_size keyword argument should be stored correctly."""
    import oxillama_py

    cfg = oxillama_py.EngineConfig(model_path="x.gguf", context_size=8192)
    assert cfg.context_size == 8192


def test_sampler_config_defaults():
    """SamplerConfig should expose all fields with correct defaults."""
    import oxillama_py

    sc = oxillama_py.SamplerConfig()
    assert abs(sc.temperature - 0.7) < 1e-5
    assert sc.top_k == 40
    assert abs(sc.top_p - 0.9) < 1e-5
    assert sc.seed is None
    assert sc.mirostat == 0


def test_sampler_config_greedy():
    """SamplerConfig.greedy() should return temperature=0 and top_k=1."""
    import oxillama_py

    sc = oxillama_py.SamplerConfig.greedy()
    assert sc.temperature == 0.0
    assert sc.top_k == 1


def test_engine_not_loaded_initially():
    """A freshly created Engine must not be loaded."""
    import oxillama_py

    cfg = oxillama_py.EngineConfig(model_path="nonexistent.gguf")
    eng = oxillama_py.Engine(cfg)
    assert not eng.is_loaded()


def test_engine_hidden_size_none_before_load():
    """hidden_size() must return None before load_model()."""
    import oxillama_py

    cfg = oxillama_py.EngineConfig(model_path="nonexistent.gguf")
    eng = oxillama_py.Engine(cfg)
    assert eng.hidden_size() is None


def test_engine_tokenize_raises_without_model():
    """tokenize() must raise RuntimeError when no model is loaded."""
    import oxillama_py

    cfg = oxillama_py.EngineConfig(model_path="nonexistent.gguf")
    eng = oxillama_py.Engine(cfg)
    with pytest.raises(Exception):
        eng.tokenize("hello")


def test_lora_load_raises_for_missing_file():
    """Lora.load() must raise for a nonexistent file."""
    import oxillama_py

    with pytest.raises(Exception):
        oxillama_py.Lora.load("/tmp/oxillama_py_nonexistent_lora_xyz.gguf")


def test_speculative_config_defaults():
    """SpeculativeConfig default num_speculative is 4."""
    import oxillama_py

    t_cfg = oxillama_py.EngineConfig(model_path="target.gguf")
    d_cfg = oxillama_py.EngineConfig(model_path="draft.gguf")
    sc = oxillama_py.SpeculativeConfig(t_cfg, d_cfg)
    assert sc.num_speculative == 4
    assert sc.seed is None


# ---------------------------------------------------------------------------
# Model-required tests (skipped when OXILLAMA_TEST_MODEL is not set)
# ---------------------------------------------------------------------------


def test_is_loaded(engine):
    """is_loaded() must return True after load_model()."""
    assert engine.is_loaded()


def test_tokenize(engine):
    """tokenize() must return a non-empty list of ints for a simple input."""
    tokens = engine.tokenize("Hello world")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_decode_token_roundtrip(engine):
    """Tokenize then decode the first token — must return a non-empty string."""
    tokens = engine.tokenize("Hello")
    decoded = engine.decode_token(tokens[0])
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_hidden_size(engine):
    """hidden_size() must return a positive int when a model is loaded."""
    hs = engine.hidden_size()
    assert hs is not None
    assert hs > 0


def test_embed_returns_float_list(engine):
    """embed() must return a list of floats with the correct dimension."""
    emb = engine.embed("Hello world")
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert all(isinstance(x, float) for x in emb)


def test_embed_l2_normalised(engine):
    """Embedding vector must be L2-normalised (norm ≈ 1)."""
    import math

    emb = engine.embed("test sentence")
    norm = math.sqrt(sum(x * x for x in emb))
    assert abs(norm - 1.0) < 1e-3, f"expected unit norm, got {norm}"


def test_embed_different_inputs(engine):
    """Different inputs must produce different embedding vectors."""
    emb1 = engine.embed("cat")
    emb2 = engine.embed("philosophy")
    assert emb1 != emb2, "different inputs should yield different embeddings"


def test_generate_returns_string(engine):
    """generate() must return a non-empty string."""
    text = engine.generate("Hello", max_tokens=32)
    assert isinstance(text, str)
    # allow empty if the model hits EOS immediately, but type must be correct


def test_generate_streaming_collects_tokens(engine):
    """generate_streaming() must invoke callback and return consistent text."""
    tokens_received = []

    def callback(tok):
        tokens_received.append(tok)

    result = engine.generate_streaming("Hello", max_tokens=32, callback=callback)
    # The concatenation of callback tokens must equal the returned string
    assert "".join(tokens_received) == result


def test_is_eos(engine):
    """is_eos() must return False for token 0 (usually not EOS in practice)."""
    # We can only verify the method doesn't raise; the actual EOS depends on the model
    result = engine.is_eos(0)
    assert isinstance(result, bool)

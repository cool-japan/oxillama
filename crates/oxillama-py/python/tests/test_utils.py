"""Tests for decode_from_logits utility."""
import pytest

numpy = pytest.importorskip("numpy")
from oxillama_py.utils import decode_from_logits


def test_greedy_argmax():
    import numpy as np

    logits = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)
    assert decode_from_logits(logits) == 1


def test_temperature_zero_like():
    import numpy as np

    # With very low temperature, the peak dominates even more
    logits = np.array([1.0, 100.0, 2.0], dtype=np.float32)
    assert decode_from_logits(logits, temperature=0.01) == 1


def test_top_k():
    import numpy as np

    # top_k=1 should always return the argmax
    logits = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)
    assert decode_from_logits(logits, top_k=1) == 1


def test_wrong_shape_raises():
    import numpy as np

    with pytest.raises(ValueError):
        decode_from_logits(np.array([[1.0, 2.0]]))

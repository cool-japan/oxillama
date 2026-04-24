"""Utility helpers for the OxiLLaMa Python bindings."""
from __future__ import annotations

from typing import List

__all__ = ["decode_from_logits"]


def decode_from_logits(
    logits: "numpy.ndarray",  # noqa: F821
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Run greedy/temperature sampling over a logits array and return the argmax token id.

    This is a pure-Python utility — it does NOT call the Rust engine.  It mirrors
    the sampler behaviour for use in notebooks and custom decoding loops.

    Args:
        logits: 1-D float32 ndarray of shape ``(vocab_size,)``.
        temperature: Softmax temperature. 1.0 = unscaled, < 1.0 = sharper.
        top_k: If > 0, restrict sampling to the top-k highest-probability tokens.
        top_p: Nucleus probability mass threshold. 1.0 = no restriction.

    Returns:
        The sampled token id (integer index into the vocabulary).
    """
    import numpy as np

    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D logits array, got shape {arr.shape}")

    if temperature != 1.0 and temperature > 0.0:
        arr = arr / temperature

    if top_k > 0:
        kth = np.partition(arr, -top_k)[-top_k]
        arr = np.where(arr >= kth, arr, -np.inf)

    # Softmax
    arr = arr - arr.max()
    exp_arr = np.exp(arr)
    probs = exp_arr / exp_arr.sum()

    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumprobs = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cumprobs, top_p))
        allowed = sorted_idx[: cutoff + 1]
        mask = np.zeros_like(probs)
        mask[allowed] = 1.0
        probs = probs * mask
        total = float(probs.sum())
        if total > 0:
            probs = probs / total
        else:
            probs = mask / mask.sum()

    return int(np.argmax(probs))

"""Jupyter / tqdm-compatible streaming callback helpers for OxiLLaMa."""
from __future__ import annotations

from typing import Any, List

__all__ = ["TqdmProgress", "CollectTokens"]


class TqdmProgress:
    """Stream tokens into a tqdm progress bar or any tqdm-like widget.

    Works in standard terminals (tqdm) and Jupyter notebooks
    (tqdm.notebook.tqdm / ipywidgets).  The progress bar shows the count
    of decoded tokens; the postfix shows the most-recently decoded piece.

    Example::

        from tqdm.auto import tqdm
        from oxillama_py.tqdm_helper import TqdmProgress

        bar = tqdm(desc="Generating", unit="tok")
        cb = TqdmProgress(bar)
        engine.generate_streaming(prompt, callback=cb)
        bar.close()
    """

    def __init__(self, pbar: Any, flush_on_newline: bool = True) -> None:
        """Initialise with an already-open tqdm instance.

        Args:
            pbar: An open ``tqdm`` instance (any variant: auto, notebook, …).
            flush_on_newline: If *True*, call ``pbar.refresh()`` whenever a
                newline character is encountered in the token stream so that
                Jupyter output repagination is triggered promptly.
        """
        self._pbar = pbar
        self._flush_on_newline = flush_on_newline
        self._tokens: List[str] = []

    def __call__(self, token: str) -> None:
        """Token-callback compatible with ``StreamingCallback`` protocol."""
        self._tokens.append(token)
        self._pbar.update(1)
        self._pbar.set_postfix_str(repr(token)[:30], refresh=False)
        if self._flush_on_newline and "\n" in token:
            self._pbar.refresh()

    @property
    def text(self) -> str:
        """The full decoded text seen so far."""
        return "".join(self._tokens)

    def reset(self) -> None:
        """Clear the accumulated token list and reset the progress bar counter."""
        self._tokens.clear()
        self._pbar.reset()


class CollectTokens:
    """Simple token collector — no progress display, just accumulate text.

    Useful when you want the full text after generation but still want a
    callback-compatible object.

    Example::

        col = CollectTokens()
        engine.generate_streaming(prompt, callback=col)
        print(col.text)
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []

    def __call__(self, token: str) -> None:
        self._tokens.append(token)

    @property
    def text(self) -> str:
        return "".join(self._tokens)

    def reset(self) -> None:
        self._tokens.clear()

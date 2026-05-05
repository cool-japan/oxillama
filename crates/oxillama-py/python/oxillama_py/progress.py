"""Progress hooks for OxiLLaMa generation.

Polymorphic dispatch for ``tqdm``, ``ipywidgets``, and bare callables.

The Rust binding accepts a ``progress=`` kwarg on every ``generate*`` method.
That kwarg may be:

* ``None`` — no progress reporting (the default).
* Any ``tqdm``/``tqdm.notebook.tqdm`` instance — duck-typed by
  ``update``/``set_postfix_str``/``close``.
* Any ``ipywidgets.IntProgress`` (or ``FloatProgress``) instance — duck-typed
  by ``value``/``max`` plus ``"Progress"`` in the class name.
* A bare callable ``Callable[[ProgressEvent], None]``.

Internally, every adapter exposes a ``__call__(event)`` that updates the
display and a ``finalise(error)`` that closes the widget.  The Rust side
calls ``_build_bridge(progress, max_tokens)`` once per generation to obtain
a ``(callback, finaliser)`` pair; the callback is a tiny lambda that turns
a 4-tuple ``(tokens, elapsed_secs, is_final, text_so_far)`` from Rust into a
fully-fleshed :class:`ProgressEvent` before dispatching to the adapter.

Example
-------

.. code-block:: python

   from tqdm.auto import tqdm
   from oxillama_py import Engine, EngineConfig

   engine = Engine(EngineConfig("model.gguf"))
   engine.load_model()

   with tqdm(desc="Generating", unit="tok") as bar:
       text = engine.generate("Hello", max_tokens=128, progress=bar)

ipywidgets variant:

.. code-block:: python

   import ipywidgets as widgets
   from IPython.display import display

   bar = widgets.IntProgress(min=0, max=128, description="Generating")
   display(bar)
   text = engine.generate("Hello", max_tokens=128, progress=bar)

Custom callable variant:

.. code-block:: python

   def on_progress(event):
       print(f"{event.tokens_generated}/{event.tokens_total}: {event.tokens_per_sec:.1f} tok/s")

   text = engine.generate("Hello", max_tokens=128, progress=on_progress)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

__all__ = [
    "ProgressEvent",
    "make_progress_adapter",
]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A single progress update emitted from the Rust generation loop.

    Construction is cheap (frozen ``dataclass`` with ``__slots__``) so even
    when the throttler permits one event per token there is negligible
    overhead.  Built once per fired tick by :func:`_build_bridge`.
    """

    #: Number of tokens decoded *so far*, including the token that triggered
    #: this event.
    tokens_generated: int
    #: Maximum number of tokens the engine intends to generate, or ``None`` if
    #: ``max_tokens`` was zero (unbounded).
    tokens_total: Optional[int]
    #: Wall-clock seconds elapsed since the bridge was built (start of decode).
    elapsed_secs: float
    #: Decoded-token throughput averaged across the entire generation so far.
    tokens_per_sec: float
    #: Estimated remaining seconds.  ``None`` until at least two tokens have
    #: been observed (so the throughput estimate is meaningful).
    eta_secs: Optional[float]
    #: ``True`` for the synthetic "generation finished" event that always
    #: fires after the last decoded token.
    is_final: bool
    #: Concatenated text decoded so far.  Only populated when the caller
    #: passed ``progress_capture_text=True``; otherwise the empty string.
    text_so_far: str


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _TqdmAdapter:
    """Drive any ``tqdm``/``tqdm.notebook.tqdm`` instance from progress events."""

    __slots__ = ("_pbar", "_max_tokens", "_first", "_last_tokens")

    def __init__(self, pbar: Any, max_tokens: int) -> None:
        self._pbar = pbar
        self._max_tokens = max_tokens
        self._first = True
        self._last_tokens = 0

    def __call__(self, event: ProgressEvent) -> None:
        if self._first:
            try:
                self._pbar.total = event.tokens_total or self._max_tokens
            except Exception:
                # Some tqdm variants disallow setting `total` after creation;
                # silently degrade to count-only display.
                pass
            self._first = False
        delta = event.tokens_generated - self._last_tokens
        if delta > 0:
            try:
                self._pbar.update(delta)
            except Exception:
                pass
            self._last_tokens = event.tokens_generated
        try:
            self._pbar.set_postfix_str(
                f"{event.tokens_per_sec:.1f} tok/s", refresh=False
            )
        except Exception:
            pass

    def finalise(self, error: Optional[BaseException]) -> None:
        if error is not None:
            label = type(error).__name__
            try:
                self._pbar.set_postfix_str(f"error: {label}", refresh=False)
            except Exception:
                pass
        try:
            self._pbar.close()
        except Exception:
            pass


class _IPyWidgetAdapter:
    """Drive any ``ipywidgets.IntProgress``-shaped widget from progress events."""

    __slots__ = ("_w", "_max_tokens", "_first")

    def __init__(self, widget: Any, max_tokens: int) -> None:
        self._w = widget
        self._max_tokens = max_tokens
        self._first = True

    def __call__(self, event: ProgressEvent) -> None:
        if self._first:
            try:
                self._w.max = event.tokens_total or self._max_tokens
            except Exception:
                pass
            self._first = False
        try:
            self._w.value = event.tokens_generated
        except Exception:
            pass
        try:
            self._w.description = f"{event.tokens_per_sec:.1f} tok/s"
        except Exception:
            pass

    def finalise(self, error: Optional[BaseException]) -> None:
        if error is None:
            style = "success"
        elif "Cancel" in type(error).__name__:
            style = "warning"
        else:
            style = "danger"
        try:
            self._w.bar_style = style
        except Exception:
            pass
        try:
            # Snap the bar to 100 % so the widget reads as fully consumed.
            self._w.value = self._w.max
        except Exception:
            pass


class _CallableAdapter:
    """Wrap a bare ``Callable[[ProgressEvent], None]`` so it has a finaliser."""

    __slots__ = ("_fn",)

    def __init__(self, fn: Callable[[ProgressEvent], None]) -> None:
        self._fn = fn

    def __call__(self, event: ProgressEvent) -> None:
        self._fn(event)

    def finalise(self, error: Optional[BaseException]) -> None:
        # Bare callables get one final event with `is_final=True`; nothing
        # extra to clean up here.
        return None


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def _is_tqdm(obj: Any) -> bool:
    """Heuristic: does *obj* quack like a tqdm progress bar?"""
    return all(hasattr(obj, attr) for attr in ("update", "set_postfix_str", "close"))


def _is_ipywidget(obj: Any) -> bool:
    """Heuristic: does *obj* quack like an ipywidgets progress widget?

    We require both ``value``/``max`` attributes *and* a class name that
    contains ``"Progress"`` so an unrelated ``namedtuple``-style object with
    those two fields does not get auto-coerced.
    """
    if not (hasattr(obj, "value") and hasattr(obj, "max")):
        return False
    return "Progress" in type(obj).__name__


def make_progress_adapter(
    obj: Any, max_tokens: int
) -> Tuple[Optional[Callable[[ProgressEvent], None]], Optional[Callable[[Optional[BaseException]], None]]]:
    """Return ``(callback, finaliser)`` for *obj* or ``(None, None)``.

    Polymorphically dispatches to :class:`_TqdmAdapter`, :class:`_IPyWidgetAdapter`,
    or :class:`_CallableAdapter`.

    Raises:
        TypeError: if *obj* is not ``None``, a tqdm-shaped pbar, an ipywidgets
            progress widget, or a callable.
    """
    if obj is None:
        return (None, None)
    if _is_tqdm(obj):
        adapter: Any = _TqdmAdapter(obj, max_tokens)
    elif _is_ipywidget(obj):
        adapter = _IPyWidgetAdapter(obj, max_tokens)
    elif callable(obj):
        adapter = _CallableAdapter(obj)
    else:
        raise TypeError(
            "progress must be a tqdm pbar, ipywidgets.IntProgress, callable, "
            f"or None; got {type(obj).__name__}"
        )
    return (adapter.__call__, adapter.finalise)


# ---------------------------------------------------------------------------
# Rust-facing entry point
# ---------------------------------------------------------------------------


def _build_bridge(
    obj: Any, max_tokens: int
) -> Tuple[Callable[[Tuple[int, float, bool, str]], None], Callable[[Optional[BaseException]], None]]:
    """Rust-facing entry point.

    Always returns a ``(callback, finaliser)`` tuple — even when *obj* is
    ``None`` — so the Rust side always has something to invoke.  The Rust
    binding hands the callback a 4-tuple
    ``(tokens, elapsed_secs, is_final, text_so_far)`` which this wrapper
    converts to a :class:`ProgressEvent` before dispatching.
    """
    cb, fin = make_progress_adapter(obj, max_tokens)
    if cb is None or fin is None:
        # Construct a no-op pair so Rust always has something to call.
        def _noop_callback(_payload: Tuple[int, float, bool, str]) -> None:
            return None

        def _noop_finaliser(_error: Optional[BaseException]) -> None:
            return None

        return (_noop_callback, _noop_finaliser)

    cb_resolved: Callable[[ProgressEvent], None] = cb
    fin_resolved: Callable[[Optional[BaseException]], None] = fin

    def _wrapped_callback(payload: Tuple[int, float, bool, str]) -> None:
        tokens, elapsed_secs, is_final, text_so_far = payload
        if tokens >= 2 and elapsed_secs > 0:
            tps = tokens / elapsed_secs
            remaining = max(max_tokens - tokens, 0)
            eta: Optional[float] = remaining / tps if tps > 0 else None
        else:
            eta = None
        tps = (tokens / elapsed_secs) if elapsed_secs > 0 else 0.0
        event = ProgressEvent(
            tokens_generated=tokens,
            tokens_total=max_tokens if max_tokens > 0 else None,
            elapsed_secs=elapsed_secs,
            tokens_per_sec=tps,
            eta_secs=eta,
            is_final=is_final,
            text_so_far=text_so_far,
        )
        cb_resolved(event)

    return (_wrapped_callback, fin_resolved)

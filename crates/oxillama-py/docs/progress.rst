Progress hooks
==============

Every ``generate*`` method on :class:`oxillama_py.Engine`,
:class:`oxillama_py.SpeculativeEngine`, and :class:`oxillama_py.AsyncEngine`
accepts a ``progress=`` keyword argument that drives a polymorphic progress
display.  The hook is throttled on the Rust side (default: 50 ms or 4 tokens,
whichever first) and is finalised exactly once even on Python exception,
cancellation, or end-of-sequence.

Accepted argument types
-----------------------

``progress=`` accepts any of the following:

* ``None`` — no progress reporting (the default).
* Any ``tqdm``/``tqdm.notebook.tqdm`` instance — duck-typed via
  ``update``/``set_postfix_str``/``close``.
* Any ``ipywidgets.IntProgress`` (or ``FloatProgress``) instance — duck-typed
  via ``value``/``max`` plus ``"Progress"`` in the class name.
* Any ``Callable[[ProgressEvent], None]`` — invoked once per throttled tick
  (and always for the first and last tokens).

The Rust side dispatches to the appropriate adapter via
:func:`oxillama_py.progress.make_progress_adapter` so all duck-typing logic
lives in pure Python.

ProgressEvent contract
----------------------

.. autoclass:: oxillama_py.progress.ProgressEvent
   :members:

Examples
--------

tqdm in a notebook
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tqdm.auto import tqdm
   from oxillama_py import Engine, EngineConfig

   engine = Engine(EngineConfig("model.gguf"))
   engine.load_model()

   with tqdm(desc="Generating", unit="tok") as bar:
       text = engine.generate("Hello", max_tokens=128, progress=bar)

ipywidgets progress widget
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ipywidgets as widgets
   from IPython.display import display

   bar = widgets.IntProgress(min=0, max=128, description="Generating")
   display(bar)
   text = engine.generate("Hello", max_tokens=128, progress=bar)
   # When generation completes the bar snaps to 100 % and ``bar_style``
   # becomes "success".  On cancellation it becomes "warning"; on error,
   # "danger".

Custom callable
~~~~~~~~~~~~~~~

.. code-block:: python

   def on_progress(event):
       print(
           f"{event.tokens_generated}/{event.tokens_total}: "
           f"{event.tokens_per_sec:.1f} tok/s"
       )

   text = engine.generate("Hello", max_tokens=128, progress=on_progress)

Tuning the throttle
~~~~~~~~~~~~~~~~~~~

Two keyword arguments tune the throttle gates (both default to ``None``,
which falls back to 50 ms / 4 tokens):

* ``progress_throttle_ms`` — minimum milliseconds between consecutive
  callback fires.
* ``progress_throttle_tokens`` — minimum number of decoded tokens between
  consecutive callback fires.

The first decoded token always fires, and a synthetic final event always
fires after generation completes (with ``is_final=True``).

Capturing the decoded text
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default :class:`~oxillama_py.progress.ProgressEvent` ``text_so_far`` is
the empty string — populating it would force an O(n) string copy on every
fired tick.  Pass ``progress_capture_text=True`` to opt in:

.. code-block:: python

   text = engine.generate(
       "Hello",
       max_tokens=128,
       progress=lambda evt: print(evt.text_so_far[-32:]),
       progress_capture_text=True,
   )

Strict error handling
~~~~~~~~~~~~~~~~~~~~~

Exceptions raised inside the progress callback are silently swallowed by
default so that a misbehaving widget cannot abort generation.  Pass
``strict_progress=True`` to re-raise the first stashed exception once
generation completes.

Migrating from ``TqdmProgress``
-------------------------------

The v0.1.1 :class:`oxillama_py.tqdm_helper.TqdmProgress` shim still works
and is re-exported from the package top-level under a
:class:`DeprecationWarning`.  To migrate, drop the wrapper and pass the bar
directly:

.. code-block:: python

   # Before (v0.1.1):
   from tqdm.auto import tqdm
   from oxillama_py import TqdmProgress
   bar = tqdm(desc="Generating", unit="tok")
   engine.generate_streaming(prompt, callback=TqdmProgress(bar))
   bar.close()

   # After (v0.1.3+):
   from tqdm.auto import tqdm
   with tqdm(desc="Generating", unit="tok") as bar:
       engine.generate(prompt, progress=bar)

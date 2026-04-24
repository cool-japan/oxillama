OxiLLaMa Python Bindings
=========================

A Pure Rust LLM inference engine with Python bindings via PyO3.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api

Quickstart
----------

.. code-block:: python

   from oxillama_py import Engine, EngineConfig

   config = EngineConfig("path/to/model.gguf")
   engine = Engine(config)
   print(engine.generate("Hello, world!", max_tokens=50))

API Reference
-------------

See :doc:`api`.

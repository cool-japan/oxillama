Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install oxillama-py

Or build from source with `maturin <https://github.com/PyO3/maturin>`_:

.. code-block:: bash

   cd crates/oxillama-py
   maturin develop --features pyo3/extension-module

Basic Usage
-----------

.. code-block:: python

   from oxillama_py import Engine, EngineConfig, SamplerConfig

   config = EngineConfig("model.gguf", num_threads=4)
   engine = Engine(config)

   # Generate text
   result = engine.generate("What is Rust?", max_tokens=100)
   print(result)

   # Streaming generation
   def on_token(tok: str, token_id: int, is_final: bool) -> None:
       print(tok, end="", flush=True)

   engine.generate_streaming("Tell me about COOLJAPAN.", callback=on_token)

Loading from HuggingFace Hub
-----------------------------

.. code-block:: python

   from oxillama_py import Engine

   engine = Engine.from_hub(
       "TheBloke/Llama-2-7B-GGUF",
       filename="llama-2-7b.Q4_K_M.gguf",
   )
   print(engine.generate("Hello!", max_tokens=50))

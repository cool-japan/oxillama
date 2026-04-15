# oxillama-py

Python bindings for OxiLLaMa — high-performance LLM inference from Python.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- `Engine` — load a GGUF model and generate text; releases the GIL during inference
- `SpeculativeEngine` — draft + target model pair for faster generation
- `LoadedLora` — load a LoRA adapter and hot-swap it onto an `Engine`
- Full Python type annotations and docstrings
- Wheels built with [maturin](https://www.maturin.rs/)

## Installation

```bash
pip install maturin
maturin develop --release          # in-place development install
# or
maturin build --release            # build a wheel
pip install target/wheels/oxillama_py-*.whl
```

## Usage

```python
import oxillama_py as ox

# Load model
engine = ox.Engine("llama-3.2-3b.Q4_K_M.gguf")

# Basic generation (GIL is released during the Rust inference call)
output = engine.generate(
    prompt="Tell me about the Rust programming language.",
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.95,
)
print(output)

# Speculative decoding: 3-8x faster on large models
draft   = ox.Engine("llama-3.2-1b.Q4_K_M.gguf")
target  = ox.Engine("llama-3.2-8b.Q4_K_M.gguf")
spec    = ox.SpeculativeEngine(draft=draft, target=target, gamma=4)
output  = spec.generate("Once upon a time", max_new_tokens=512)
print(output)

# LoRA adapter
lora   = ox.LoadedLora("my-adapter.gguf")
engine.apply_lora(lora)
output = engine.generate("Write a haiku.", max_new_tokens=64)
engine.remove_lora()
```

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

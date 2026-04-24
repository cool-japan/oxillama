# oxillama-py

Python bindings for OxiLLaMa — high-performance LLM inference from Python.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- `EngineConfig` — configuration dataclass for thread count, context size, tokenizer path, and sampler defaults
- `Engine` — load a GGUF model and generate text; releases the GIL during inference
- `AsyncEngine` — async/await interface; streams tokens to Python coroutines without blocking the event loop
- `SamplerConfig` — all ten sampler knobs with `greedy()` and `mirostat_v2()` static constructors
- `SpeculativeConfig` / `SpeculativeEngine` — draft + target model pair for faster generation
- `Lora` — load a LoRA adapter and hot-swap it onto an `Engine`
- `Tokenizer` — first-class tokenizer object with `encode`, `decode`, `encode_batch`, `apply_chat_template`
- `CancellationToken` — cooperative cancellation handle accepted by `generate()` and `generate_streaming()`
- Structured exception hierarchy: `OxiLlamaError` → `LoadError`, `GenerateError`, `TokenizerError`, `GrammarError`, `QuantError`, `KvCacheFullError`
- Full Python type annotations (`.pyi` stubs) and docstrings
- Wheels built with [maturin](https://www.maturin.rs/) (ABI3, Python 3.8+)
- Optional numpy interop (`embed_numpy()`, `embed_batch_numpy()`, `forward_logits_numpy()`) via `numpy` feature

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

# Streaming generation with a callback
engine.generate_streaming(
    "Explain quantum computing.",
    max_tokens=256,
    callback=lambda tok: print(tok, end="", flush=True),
)

# Async engine (non-blocking, event-loop friendly)
import asyncio

async def run():
    aengine = ox.AsyncEngine("llama-3.2-3b.Q4_K_M.gguf")
    result = await aengine.generate("Hello async world", max_new_tokens=64)
    print(result)

asyncio.run(run())

# Cooperative cancellation
token = ox.CancellationToken()
engine.generate_streaming("Tell me a story", max_tokens=1024,
                          callback=print, cancel_token=token)
token.cancel()  # stop from another thread

# Speculative decoding: 3-8x faster on large models
draft  = ox.Engine("llama-3.2-1b.Q4_K_M.gguf")
target = ox.Engine("llama-3.2-8b.Q4_K_M.gguf")
spec   = ox.SpeculativeEngine(draft=draft, target=target, gamma=4)
output = spec.generate("Once upon a time", max_new_tokens=512)
print(output)

# LoRA adapter
lora   = ox.Lora.load("my-adapter.gguf")
engine.apply_lora(lora)
output = engine.generate("Write a haiku.", max_new_tokens=64)
engine.remove_lora()

# Tokenizer
tokenizer = ox.Tokenizer.from_file("tokenizer.json")
ids = tokenizer.encode("Hello, world!")
text = tokenizer.decode(ids)

# HuggingFace Hub loader
engine = ox.Engine.from_hub("meta-llama/Llama-3.2-3B-GGUF")
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `numpy` | no | numpy interop for `embed_numpy()`, `embed_batch_numpy()`, `forward_logits_numpy()` |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

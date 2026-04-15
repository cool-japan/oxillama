# oxillama-wasm

WebAssembly bindings for OxiLLaMa — GGUF parsing and LLM inference in the browser.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- GGUF metadata and tensor catalogue parsing from a `Uint8Array` in the browser
- Full text generation via `oxillama-runtime` (behind the `inference` feature)
- Pure-Rust tokenizer backend (`fancy-regex`, no Oniguruma C library) — safe for `wasm32-unknown-unknown`
- No SIMD rayon threads — single-threaded, browser-compatible

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `inference` | yes | Include `oxillama-runtime` for full generation |
| `console_error_panic_hook` | yes | Pretty panic messages in the browser console |

## Build

```bash
# Install wasm-pack once
cargo install wasm-pack

# Build for the browser (ES module)
wasm-pack build --release --target web -p oxillama-wasm

# Output lands in pkg/
# oxillama_wasm.js    — JS glue
# oxillama_wasm_bg.wasm — compiled WebAssembly
```

## Usage (JavaScript)

```js
import init, { GgufWasm, InferenceWasm } from "./pkg/oxillama_wasm.js";

await init();

// Parse GGUF metadata from a fetched ArrayBuffer
const resp   = await fetch("/models/llama-3.2-1b.Q4_K_M.gguf");
const buf    = await resp.arrayBuffer();
const gguf   = new GgufWasm(new Uint8Array(buf));
console.log("arch:", gguf.get_metadata("general.architecture"));

// Run inference (requires `inference` feature)
const engine = new InferenceWasm(new Uint8Array(buf));
const output = engine.generate("Hello, world!", 128, 0.8);
console.log(output);
```

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

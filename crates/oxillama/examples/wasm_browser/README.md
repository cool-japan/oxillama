# OxiLLaMa Browser Demo

Run LLaMA-family models directly in your browser via WebAssembly.

## Prerequisites
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/): `cargo install wasm-pack`

## Build
```bash
cd crates/oxillama/examples/wasm_browser
./build.sh
```

## Run
```bash
python3 -m http.server 8080
```
Then open http://localhost:8080 in Chrome/Firefox/Safari (desktop).

## Model
Download a GGUF model. Recommended for browser use (fits in 2 GB RAM):
- TinyLlama-1.1B-Chat Q4_K_M (~670 MB): https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
- Qwen2-0.5B Q8_0 (~500 MB): smaller and very fast

You also need the model's `tokenizer.json` from HuggingFace (available on the model's Files tab).

Load both files using the file pickers in the demo.

## How it works
- `init()` sets up panic hooks and WASM initialization (called automatically on module load)
- `loadModelFromBytesWithProgress(modelBytes, tokenizerJson, onProgress)` loads model weights from
  a `Uint8Array` and tokenizer from a JSON string, returning a `WasmEngine` handle
- `WasmEngine.generate(prompt, maxTokens, onToken)` streams tokens to the `onToken` callback,
  reusing the already-loaded model without re-parsing the GGUF on each call

## API exported by oxillama-wasm
```js
// Parse just the GGUF header (useful for model inspection)
const header = parseGgufHeader(bytes);          // { tensorCount, metadataCount, version }

// Parse typed metadata (arch, context_length, embedding_length, ...)
const meta = parseGgufMetadata(bytes);

// List all tensor names
const names = listTensorNames(bytes);           // string[]

// Load a model, get back a reusable engine handle
const engine = await loadModelFromBytesWithProgress(
    modelBytes,       // Uint8Array  — raw .gguf bytes
    tokenizerJson,    // string      — contents of tokenizer.json
    (pct) => { ... } // optional progress callback (0, 25, 100)
);

// Generate text (reuses the loaded model)
const fullText = engine.generate(prompt, maxTokens, (tok) => { ... });
```

## Binary size
Raw: ~3.5 MB
Brotli-compressed: ~1.2 MB (CDN-friendly)

For smaller binaries, run `wasm-opt -Oz pkg/oxillama_wasm_bg.wasm -o pkg/oxillama_wasm_bg.wasm` after building.

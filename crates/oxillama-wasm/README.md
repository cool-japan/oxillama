# oxillama-wasm

WebAssembly bindings for OxiLLaMa — GGUF parsing and LLM inference in the browser.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- GGUF metadata and tensor catalogue parsing from a `Uint8Array` in the browser
- Typed metadata export via `parseGgufMetadata()` returning a `GgufMetadataJs` object (arch, context_length, embedding_length, etc.)
- Full text generation via `oxillama-runtime` (behind the `inference` feature) with optional per-token `onToken` callback
- K-quant dequantization bindings: `dequantQ4_0`, `dequantQ4K`, `dequantQ5K`, `dequantQ6K`
- Model loading with progress callbacks: `loadModelFromBytesWithProgress(bytes, onProgress)`
- WebGPU async bridge: `initWebGpuDevice()`, `webgpuDequantQ4_0Async()`, `webgpuGemvAsync()`
- IndexedDB model cache: `cacheModel()`, `loadCachedModel()`, `listCachedModels()`, `deleteCachedModel()`
- Streaming GGUF load via `GgufChunkLoader` for incremental byte feeds
- Web-worker message-passing API: `parseWorkerMessage()` / `workerTokenEvent()`
- Pure-Rust tokenizer backend (`fancy-regex`, no Oniguruma C library) — safe for `wasm32-unknown-unknown`
- No SIMD rayon threads — single-threaded, browser-compatible; SIMD128 proposal enabled at compile time

## Status

**Version:** 0.1.2 — **Tests:** 51 passing

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
import init, {
  parseGgufHeader, parseGgufMetadata, listTensorNames,
  dequantQ4_0, dequantQ4K, dequantQ5K, dequantQ6K,
  loadModelFromBytesWithProgress, WasmEngine,
} from "./pkg/oxillama_wasm.js";

await init();

// Fetch and parse GGUF metadata (no weights loaded)
const resp   = await fetch("/models/llama-3.2-1b.Q4_K_M.gguf");
const bytes  = new Uint8Array(await resp.arrayBuffer());
const header = parseGgufHeader(bytes);
const meta   = parseGgufMetadata(bytes);
console.log("arch:", meta.arch, "context:", meta.context_length);
console.log("tensors:", listTensorNames(bytes));

// Load model with progress callback (requires `inference` feature)
const engine = await loadModelFromBytesWithProgress(bytes, (pct) => {
  console.log(`loading: ${pct}%`);
});

// Generate text with per-token streaming
const output = engine.generate("Hello, world!", 128, 0.8, (token) => {
  process.stdout.write(token);
});
console.log(output);

// IndexedDB cache — persist across reloads
await cacheModel("llama-3b", bytes);
const cached = await loadCachedModel("llama-3b");

// WebGPU acceleration (where supported)
await initWebGpuDevice();
const result = await webgpuDequantQ4_0Async(quantizedBlock);
```

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

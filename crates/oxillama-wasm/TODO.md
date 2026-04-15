# oxillama-wasm — TODO

## 1. Overview

`oxillama-wasm` is the WebAssembly binding layer for OxiLLaMa. It exposes
GGUF model loading and the full `generate()` inference pipeline to browser
JavaScript via `wasm-bindgen`, keeping the entire stack Pure Rust and free
of C/C++/Fortran dependencies (no Oniguruma, no OpenBLAS, no FFTW).

The heavyweight `inference` feature is gated behind a Cargo flag, enabled by
default but turnable off for GGUF-metadata-only builds. Consumers who only
need header introspection can therefore ship a minimal wasm binary. The full
release build weighs in at 3.5 MB raw, which compresses to roughly 1.2 MB
over HTTP with standard brotli, making it practical to serve directly from
any static CDN without a custom runtime.

The crate relies on `tokenizers/unstable_wasm` (fancy-regex backend) instead
of the default Oniguruma backend, preserving the COOLJAPAN Pure Rust Policy
on the `wasm32-unknown-unknown` target. Rayon parallelism is intentionally
omitted from the wasm build; the scalar fallback path is used instead, since
neither the browser nor `wasm32-unknown-unknown` provides the threading
primitives rayon depends on. All error propagation crosses the JS boundary
as typed `JsValue` rejections — no panics leak through.

## 2. Status Snapshot

| Item                    | Value                                                       |
|-------------------------|-------------------------------------------------------------|
| Version                 | 0.1.0                                                       |
| Completion              | 92%                                                         |
| Source files            | 1 (`src/lib.rs`)                                            |
| Release wasm size       | 3.5 MB (raw) / ~1.2 MB (brotli over HTTP)                   |
| `wasm-bindgen` version  | workspace pinned                                            |
| Tokenizer backend       | `tokenizers/unstable_wasm` (fancy-regex, no C deps)         |
| Feature flags           | `inference` (default), `console_error_panic_hook` (default) |
| Rayon                   | disabled (scalar path on wasm32)                            |
| Tested browsers         | Chrome, Firefox, Safari (desktop)                           |
| Reference model         | Bonsai-8B Q1_0_G128 (works fully in browser)                |
| Crate type              | `cdylib` + `rlib` (dual output)                             |

## 3. Module Map

| File          | Role                                                            |
|---------------|-----------------------------------------------------------------|
| `src/lib.rs`  | Sole source file. Hosts the `#[wasm_bindgen]` surface: `init()` panic hook, `parseGgufHeader`, `listTensorNames`, `dequantQ4_0`, and `generate()` (feature-gated). Keeps the public surface intentionally small for a tight wasm binary. |

The crate is deliberately single-file: the wasm boundary is narrow, all
heavy lifting lives in `oxillama-gguf`, `oxillama-quant`, and
`oxillama-runtime`, so introducing submodules here would add cost without
scan-readability gains. If the surface grows past ~800 lines we will split
into `src/gguf.rs` / `src/dequant.rs` / `src/generate.rs` via splitrs.

The build exports both `cdylib` (for `wasm-pack` / the browser) and `rlib`
(so host unit tests can exercise the underlying library logic without
standing up a wasm runtime). All dependencies on `oxillama-*` crates are
declared with `default-features = false` and re-enabled selectively, so no
rayon, no filesystem I/O, and no Oniguruma ever reach the wasm32 target.

## 4. Shipped in v0.1.0

- `InferenceEngine::load_model_from_bytes(Uint8Array)` — filesystem-free
  model load for sandboxed browser environments (no `fs::read` required).
- `InferenceEngine::generate(prompt, max_tokens)` — full autoregressive
  token generation running entirely inside the browser wasm sandbox.
- GGUF parsing (`parseGgufHeader`, `listTensorNames`) exposed via
  `wasm-bindgen` for metadata introspection without loading weights.
- Q4_0 dequantization (`dequantQ4_0`) callable directly from JS with a
  strict block-length check that returns a descriptive `JsValue` error on
  malformed input rather than panicking.
- `tokenizers/unstable_wasm` backend wired up (replaces the C-backed
  Oniguruma path — Pure Rust Policy compliant on `wasm32-unknown-unknown`).
- `wasm-bindgen` JS interop glue with typed return values (`JsValue`
  objects, `Vec<f32>`, `String`) and explicit error propagation through
  `Result<T, JsValue>` — no implicit panics leak to the JS host.
- Feature-gated `inference` compile switch: disabling it strips
  `oxillama-runtime` for metadata-only wasm bundles.
- 3.5 MB release wasm binary, ~1.2 MB after brotli — suitable for CDN.
- Rayon parallel features disabled for the wasm32 target (scalar fallback).
- Confirmed working in Chrome, Firefox, and Safari on desktop.
- Verified end-to-end against Bonsai-8B (Q1_0_G128) running fully client-side.
- `console_error_panic_hook` wired into `#[wasm_bindgen(start)]` so Rust
  panics surface as readable stack traces in the browser devtools console.
- Unit tests at the underlying-library level (no `JsValue` machinery in
  host tests) plus `wasm-bindgen-test` dev-dependency for on-target coverage.
- LLaMA, Qwen3, Mistral, Gemma, and Phi architecture features forwarded
  to `oxillama-runtime`, so the browser build covers the same model set as
  the native runtime.

## 5. Known Gaps / Incomplete

Accounting for the outstanding 13% toward 100% completion:

- **No WebGPU path.** `wgpu` supports WebGPU, but `oxillama-gpu` is not yet
  wired into the wasm build — all matmul runs on the CPU wasm path today.
- **No streaming GGUF load.** `load_model_from_bytes` requires the full
  `Uint8Array` resident in memory before parsing. Multi-GB models cannot
  be loaded incrementally via `ReadableStream`.
- **Mobile browsers untested.** iOS Safari and Android Chrome are not in
  the validation matrix yet; memory limits and SIMD quirks unverified.
- **No service-worker / IndexedDB cache.** Every page refresh re-downloads
  the GGUF bytes; no persistent client-side model cache.
- **No web-worker offload helper.** Inference blocks the main thread unless
  the calling page already runs inside a worker context.
- **No `onProgress` callback for load.** JS cannot render a progress bar
  during header parse and tensor indexing.
- **No load-time quantization.** F16 weights cannot be reduced to Q4_0 at
  load time to shrink the resident RAM footprint.
- **SIMD128 wasm-opt not default.** The SIMD proposal is available in all
  major browsers but is not enabled by the default build pipeline here.
- ~~**Streaming token callback not plumbed to JS.**~~ ✅ `generate()`
  now accepts an `on_token: Option<js_sys::Function>` callback that
  forwards each decoded token to a JS function for real-time UI updates.
- ~~**Only Q4_0 is exposed for standalone dequant.**~~ ✅ K-quant
  dequantization bindings shipped: `dequantQ4K`, `dequantQ5K`,
  `dequantQ6K` alongside the existing `dequantQ4_0`.

## 6. v1.1 Roadmap

- **Streaming / chunked GGUF load** via `fetch` + `ReadableStream`, piped
  directly into the parser without a full-buffer `Uint8Array` copy.
- **WebGPU backend bridge** from `oxillama-gpu` for Q4_0 and Q8_0 matmul
  kernels; fall back to CPU wasm when WebGPU is absent.
- **IndexedDB model cache** so GGUF bytes persist across reloads and across
  tabs of the same origin.
- **Headless-browser CI** using Playwright and/or `wasm-pack test --headless`
  to catch regressions on every PR.
- **Mobile browser validation matrix** — iOS Safari, Android Chrome — with
  memory-pressure tests for Bonsai-8B Q1_0_G128.
- **Web-worker offload helper** plus a thin message-passing API so inference
  never blocks the main thread by default.
- **`onProgress` callback** during model load, reporting bytes processed
  and tensor-index completion.
- ~~**Streaming token callback**~~ ✅ Shipped via `on_token:
  Option<js_sys::Function>` on `generate()`.
- ~~**Individual K-quant dequant bindings**~~ ✅ Shipped: `dequantQ4K`,
  `dequantQ5K`, `dequantQ6K` exposed to JS alongside `dequantQ4_0`.
- **`wasm-opt -O4` with SIMD128** baked into the default release pipeline
  to shave another ~500 KB off the raw binary and enable vector dequant.
- **Typed GGUF metadata export** via `serde-wasm-bindgen` (already listed
  as an optional dep) so JS sees a structured object rather than numeric
  field reads through `js_sys::Reflect`.

## 7. v2.0+ Vision

- **Mobile SIMD128 optimization** — full wasm SIMD proposal coverage across
  Q4_0, Q4_K, Q8_0, and Q1_0_G128 kernels on mobile Safari / mobile Chrome.
- **In-browser quantization** — F16 → Q4_0 at load time to shrink RAM and
  enable larger models on resource-constrained devices.
- **OPFS (Origin Private File System) storage** for multi-GB GGUF files
  without hitting IndexedDB blob-size ceilings.
- **Multi-tab shared engine** via `SharedArrayBuffer` and a broadcast
  channel, so one loaded model serves every tab of the origin.
- **WebGPU K-quant shaders** covering all Q\*_K types (Q2_K through Q6_K),
  bridging `oxillama-gpu` into the browser GPU device.
- **WebCodecs integration** for streaming multimodal input/output pipelines.
- **TypeScript SDK** — `@cooljapan/oxillama-js` npm package with fully typed
  bindings, auto-generated from the `wasm-bindgen` surface.
- **Framework hooks** — `useOxillama()` for React, a Vue composable, and a
  Svelte store adapter, all built on the same TypeScript SDK core.
- **Offline-first demo app** packaged as a PWA, shipping a quantized
  Bonsai-8B under 2 GB of OPFS storage for air-gapped inference.

*Last updated: 2026-04-15 (v0.1.0 release)*

# Mobile Browser Test Matrix

## Tested Configurations

| Browser | Version | Platform    | SIMD128     | Service Worker | IndexedDB | Status              |
|---------|---------|-------------|-------------|----------------|-----------|---------------------|
| Safari  | 17.0+   | iOS 17+     | compiled    | yes            | yes       | Manual test needed  |
| Chrome  | 121+    | Android 12+ | compiled    | yes            | yes       | Manual test needed  |
| Firefox | 122+    | Android     | compiled    | yes            | yes       | Manual test needed  |
| Chrome  | 121+    | Desktop     | compiled    | yes            | yes       | CI tested           |
| Safari  | 17+     | macOS       | compiled    | yes            | yes       | Manual test needed  |
| Edge    | 121+    | Desktop     | compiled    | yes            | yes       | Manual test needed  |

**SIMD128 "compiled"** — The binary was built with `RUSTFLAGS="-C target-feature=+simd128"`.
Runtime detection currently mirrors the compile-time flag; JS-side detection via
`wasm-feature-detect` is recommended for production.

## How to Test Manually

### 1. Build with SIMD128 enabled

```bash
# From the workspace root
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build crates/oxillama-wasm \
    --target web \
    --out-dir crates/oxillama-wasm/examples/pkg
```

### 2. Serve the demo page

```bash
# Python 3 built-in server (serves from the examples/ directory)
cd crates/oxillama-wasm/examples
python3 -m http.server 8080
```

Then open `http://localhost:8080/service_worker_demo.html`.

### 3. Verify SIMD status

The **SIMD128 Status** card should show:

```
compiled_with=true, runtime_detected=true
```

If the binary was built without `+simd128`, both values will be `false`.

### 4. Verify service-worker registration

- The **Service Worker** card should show `Registered. Scope: <url>`.
- Open **DevTools → Application → Service Workers** and confirm the worker
  is listed as *activated and running*.

### 5. Verify GGUF model caching

1. Copy a small `.gguf` file to `examples/models/test.gguf`.
2. Load it with `fetch('/models/test.gguf')` from the browser console.
3. **Refresh the page.**
4. Reload `fetch('/models/test.gguf')` — it should be served from the SW
   cache with no network request (verify in DevTools → Network → Size shows
   *(ServiceWorker)*).

## Known Limitations

| Limitation | Details |
|------------|---------|
| HTTPS required for SW (non-localhost) | Safari and Chrome enforce HTTPS for service-worker registration on all origins except `localhost`. Use a self-signed certificate or a tunneling tool (e.g. `ngrok`) for device testing on the local network. |
| Blob URL SW scope restriction | A service worker registered from a Blob URL is scoped to the creating tab in some browsers. For full cross-tab caching, serve `oxillama-sw.js` as a static file from the origin root. |
| SIMD128 runtime_detected mirrors compiled_with | True runtime detection requires trying a SIMD instruction from JS and catching the resulting `WebAssembly.RuntimeError`. Use the `wasm-feature-detect` npm package for authoritative per-device detection. |
| iOS IndexedDB blob limits | Storing multi-GB GGUF files in IndexedDB on iOS may hit per-origin storage limits. For models > 512 MB, prefer the Cache Storage API (already used by the generated service worker) or the Origin Private File System (OPFS). |
| Rayon / threading disabled | The WASM build disables Rayon; inference runs single-threaded. `SharedArrayBuffer`-backed threading requires COOP/COEP headers and is not yet wired up. |

## Headless CI Integration (future work)

Run `wasm-pack test --headless --chrome` to exercise the wasm-bindgen tests in a
real browser without a graphical display:

```bash
wasm-pack test crates/oxillama-wasm --headless --chrome
```

This requires `chromedriver` to be available on `PATH`.  Add this step to
`.github/workflows/ci.yml` to catch regressions on every PR.

---

*Last updated: 2026-05-05 (Track F — Mobile SIMD128 sanity-check and service-worker model cache)*

//! Service-worker registration helper for the OxiLLaMa WASM browser build.
//!
//! This module provides two public API surface points:
//!
//! 1. [`get_service_worker_script`] — returns a self-contained JavaScript
//!    service-worker source string that intercepts GGUF fetch requests and
//!    serves them from the Cache Storage API, avoiding re-download on refresh.
//!
//! 2. [`register_service_worker`] — registers a service-worker script URL via
//!    `navigator.serviceWorker.register(url)` and returns the resulting
//!    `Promise<ServiceWorkerRegistration>` to the caller.
//!
//! ## JavaScript usage
//!
//! ```js
//! import { getServiceWorkerScript, registerServiceWorker } from './oxillama_wasm.js';
//!
//! // Build options
//! const opts = JSON.stringify({
//!   gguf_path_prefix: '/models/',
//!   cache_name: 'oxillama-model-cache-v1',
//! });
//!
//! // Generate SW script and host it as a Blob URL
//! const script = getServiceWorkerScript(opts);
//! const blob = new Blob([script], { type: 'application/javascript' });
//! const url  = URL.createObjectURL(blob);
//!
//! // Register
//! const reg = await registerServiceWorker(url);
//! console.log('SW scope:', reg.scope);
//! ```
//!
//! ## Caching strategy
//!
//! The generated service-worker uses the **Cache Storage API** (not IndexedDB)
//! so that the browser can serve cached responses directly from the fetch event
//! handler without an async IDB round-trip.  The flow is:
//!
//! ```text
//! fetch event for /models/*.gguf
//!   └── caches.open(CACHE_NAME)
//!         ├── cache hit  → return cached Response immediately
//!         └── cache miss → network fetch → cache.put() → return Response
//! ```
//!
//! This approach is fully offline-capable after the first successful download.

use js_sys::Reflect;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ── Options ───────────────────────────────────────────────────────────────────

/// Configuration for the generated service-worker script.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceWorkerOptions {
    /// URL pathname prefix for GGUF fetch interception, e.g. `"/models/"`.
    ///
    /// Any request whose `pathname` starts with this string is considered a
    /// GGUF model request and will be served from the cache when available.
    pub gguf_path_prefix: String,

    /// Name of the Cache Storage bucket used to persist GGUF responses.
    ///
    /// Increment the version suffix (e.g. `"v1"` → `"v2"`) to bust the cache
    /// and force a clean re-download on the next page load.
    pub cache_name: String,
}

impl Default for ServiceWorkerOptions {
    fn default() -> Self {
        Self {
            gguf_path_prefix: "/models/".to_string(),
            cache_name: "oxillama-model-cache-v1".to_string(),
        }
    }
}

// ── Script generation ─────────────────────────────────────────────────────────

/// Return the service-worker JavaScript source as a `String`.
///
/// Deserializes `options_json` as [`ServiceWorkerOptions`], then calls
/// [`generate_service_worker_script`] to produce a self-contained script.
///
/// The caller is responsible for hosting the script: either write it to a
/// `.js` file served from the same origin, or encode it as a Blob URL:
///
/// ```js
/// const src  = getServiceWorkerScript(JSON.stringify({ gguf_path_prefix: '/models/', cache_name: 'v1' }));
/// const blob = new Blob([src], { type: 'application/javascript' });
/// const url  = URL.createObjectURL(blob);
/// ```
///
/// Note: some browsers restrict Blob URL service workers to the tab that
/// created them.  For production deployments, serving the script as a static
/// file from the origin root is strongly preferred.
///
/// # Errors
///
/// Returns a `JsValue` error string if `options_json` is not valid JSON or
/// does not match the [`ServiceWorkerOptions`] schema.
#[wasm_bindgen(js_name = "getServiceWorkerScript")]
pub fn get_service_worker_script(options_json: &str) -> Result<String, JsValue> {
    let opts: ServiceWorkerOptions = serde_json::from_str(options_json)
        .map_err(|e| JsValue::from_str(&format!("ServiceWorkerOptions parse error: {e}")))?;
    Ok(generate_service_worker_script(&opts))
}

/// Build a self-contained service-worker JavaScript script string.
///
/// The generated script:
/// 1. Listens for the `install` event and calls `skipWaiting()` to activate
///    immediately without waiting for existing tabs to close.
/// 2. Listens for the `activate` event and calls `clients.claim()` so that
///    already-open tabs are controlled by the new service worker.
/// 3. Intercepts `fetch` events whose URL pathname starts with
///    `opts.gguf_path_prefix`.  For matching requests:
///    - On **cache hit**: returns the cached `Response` directly.
///    - On **cache miss**: fetches from the network, caches a clone of the
///      response (only when `response.ok` is true), and returns the response.
///
/// Non-matching fetch events are passed through untouched.
fn generate_service_worker_script(opts: &ServiceWorkerOptions) -> String {
    // The format! placeholders use named arguments to keep the template
    // readable.  Double-braces {{ }} are literal { } in the output.
    format!(
        r#"// OxiLLaMa GGUF model cache service worker
// Auto-generated by oxillama-wasm getServiceWorkerScript().
// DO NOT EDIT — regenerate via the Rust WASM bindings.

const CACHE_NAME = '{cache_name}';
const GGUF_PREFIX = '{prefix}';

// ── Install ───────────────────────────────────────────────────────────────────
// Skip the waiting phase so the new worker activates immediately,
// without waiting for all existing tabs to close.
self.addEventListener('install', (event) => {{
  self.skipWaiting();
}});

// ── Activate ─────────────────────────────────────────────────────────────────
// Claim all existing clients immediately so that the current page is
// controlled by this service worker on first activation.
self.addEventListener('activate', (event) => {{
  event.waitUntil(clients.claim());
}});

// ── Fetch ─────────────────────────────────────────────────────────────────────
// Intercept requests whose pathname starts with GGUF_PREFIX and serve
// them from the Cache Storage API when available.
self.addEventListener('fetch', (event) => {{
  const url = new URL(event.request.url);
  if (url.pathname.startsWith(GGUF_PREFIX)) {{
    event.respondWith(handleGgufFetch(event.request));
  }}
  // Non-GGUF requests fall through to the browser default handler.
}});

// ── GGUF fetch handler ────────────────────────────────────────────────────────
// Cache-first strategy: return cached response if available; otherwise
// fetch from network, cache the successful response, then return it.
async function handleGgufFetch(request) {{
  const cache = await caches.open(CACHE_NAME);

  // 1. Try cache first.
  const cached = await cache.match(request);
  if (cached) {{
    return cached;
  }}

  // 2. Cache miss — fetch from network.
  let response;
  try {{
    response = await fetch(request);
  }} catch (networkError) {{
    // Network failure with no cached fallback: propagate the error.
    throw networkError;
  }}

  // 3. Cache the response only when the server returned a successful status
  //    (2xx).  Do not cache error responses (4xx / 5xx) or opaque redirects.
  if (response.ok) {{
    // Clone before consuming: the original is returned to the browser,
    // the clone is stored in the cache.
    cache.put(request, response.clone());
  }}

  return response;
}}
"#,
        cache_name = opts.cache_name,
        prefix = opts.gguf_path_prefix,
    )
}

// ── Service-worker registration ───────────────────────────────────────────────

/// Register a service-worker script and return the resulting
/// `Promise<ServiceWorkerRegistration>`.
///
/// Accesses `navigator.serviceWorker.register(script_url)` via
/// `js_sys::Reflect` so that no additional `web-sys` features are required.
/// The returned `Promise` resolves to the `ServiceWorkerRegistration` object
/// on success, or rejects with an error if:
/// - The browser does not support service workers (`navigator.serviceWorker`
///   is `undefined`).
/// - The registration call throws (e.g. HTTPS not enforced on non-localhost).
///
/// # JavaScript usage
///
/// ```js
/// const reg = await registerServiceWorker('/oxillama-sw.js');
/// console.log('registered, scope =', reg.scope);
/// ```
#[wasm_bindgen(js_name = "registerServiceWorker")]
pub fn register_service_worker(script_url: &str) -> js_sys::Promise {
    // Resolve the global object.  Inside a browser window context this is
    // the `Window` object; inside a SharedWorker / ServiceWorker context it
    // would be their respective globals.  We use js_sys::global() which works
    // in all three environments without requiring web-sys Window bindings.
    let global = js_sys::global();

    // Retrieve navigator from the global scope.
    let navigator = match Reflect::get(&global, &JsValue::from_str("navigator")) {
        Ok(val) if !val.is_undefined() && !val.is_null() => val,
        _ => {
            return js_sys::Promise::reject(&JsValue::from_str(
                "navigator is not available in this context",
            ));
        }
    };

    // Retrieve navigator.serviceWorker (ServiceWorkerContainer).
    let sw_container = match Reflect::get(&navigator, &JsValue::from_str("serviceWorker")) {
        Ok(val) if !val.is_undefined() && !val.is_null() => val,
        _ => {
            return js_sys::Promise::reject(&JsValue::from_str(
                "navigator.serviceWorker is not available — \
                 service workers require HTTPS (or localhost)",
            ));
        }
    };

    // Retrieve the `register` method from the ServiceWorkerContainer.
    let register_fn_val = match Reflect::get(&sw_container, &JsValue::from_str("register")) {
        Ok(val) if val.is_function() => val,
        _ => {
            return js_sys::Promise::reject(&JsValue::from_str(
                "navigator.serviceWorker.register is not a function",
            ));
        }
    };

    let register_fn = js_sys::Function::from(register_fn_val);

    // Call register(script_url) — the return value is a Promise.
    let args = js_sys::Array::new();
    args.push(&JsValue::from_str(script_url));

    match register_fn.apply(&sw_container, &args) {
        Ok(promise_val) => {
            // The spec guarantees register() returns a Promise.
            js_sys::Promise::from(promise_val)
        }
        Err(e) => js_sys::Promise::reject(&JsValue::from_str(&format!(
            "serviceWorker.register() threw: {e:?}"
        ))),
    }
}

// ── Non-WASM unit tests ───────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;

    /// Serialize `ServiceWorkerOptions` to JSON and deserialize back;
    /// verify all fields survive the round-trip.
    #[test]
    fn service_worker_options_serde_roundtrip() {
        let original = ServiceWorkerOptions {
            gguf_path_prefix: "/assets/models/".to_string(),
            cache_name: "my-model-cache-v3".to_string(),
        };

        let json =
            serde_json::to_string(&original).expect("ServiceWorkerOptions must serialize to JSON");
        let restored: ServiceWorkerOptions = serde_json::from_str(&json)
            .expect("JSON must deserialize back to ServiceWorkerOptions");

        assert_eq!(
            restored.gguf_path_prefix, original.gguf_path_prefix,
            "gguf_path_prefix must survive JSON round-trip"
        );
        assert_eq!(
            restored.cache_name, original.cache_name,
            "cache_name must survive JSON round-trip"
        );
    }

    /// The default `ServiceWorkerOptions::cache_name` must be the
    /// documented well-known string `"oxillama-model-cache-v1"`.
    #[test]
    fn service_worker_default_cache_name() {
        let opts = ServiceWorkerOptions::default();
        assert_eq!(
            opts.cache_name, "oxillama-model-cache-v1",
            "default cache_name must be 'oxillama-model-cache-v1'"
        );
    }

    /// Verify that the generated script contains the expected identifiers.
    #[test]
    fn generated_script_contains_expected_identifiers() {
        let opts = ServiceWorkerOptions {
            gguf_path_prefix: "/models/".to_string(),
            cache_name: "test-cache-v1".to_string(),
        };
        let script = generate_service_worker_script(&opts);

        assert!(
            script.contains("test-cache-v1"),
            "cache name must appear in generated script"
        );
        assert!(
            script.contains("/models/"),
            "GGUF prefix must appear in generated script"
        );
        assert!(
            script.contains("handleGgufFetch"),
            "fetch handler must be present"
        );
        assert!(
            script.contains("skipWaiting"),
            "install handler must call skipWaiting"
        );
        assert!(
            script.contains("clients.claim"),
            "activate handler must call clients.claim"
        );
    }

    /// Parsing invalid JSON via `serde_json` must return an error.
    /// We test the underlying deserialization directly rather than calling
    /// `get_service_worker_script` (which wraps the error in a JsValue that
    /// cannot be constructed outside a WASM runtime).
    #[test]
    fn service_worker_options_rejects_invalid_json() {
        let result: Result<ServiceWorkerOptions, _> = serde_json::from_str("{not valid json}");
        assert!(
            result.is_err(),
            "invalid JSON must fail to deserialize into ServiceWorkerOptions"
        );
    }

    /// Parsing valid JSON via `serde_json` must succeed and carry through to
    /// the script generator.
    #[test]
    fn service_worker_options_accepts_valid_json() {
        let json = r#"{"gguf_path_prefix": "/llm/", "cache_name": "llm-v2"}"#;
        let opts: ServiceWorkerOptions =
            serde_json::from_str(json).expect("valid JSON must deserialize");
        let script = generate_service_worker_script(&opts);
        assert!(script.contains("/llm/"));
        assert!(script.contains("llm-v2"));
    }
}

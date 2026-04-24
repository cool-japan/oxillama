//! IndexedDB model cache for persisting GGUF model bytes across page reloads.
//!
//! Exposes four async functions to JavaScript:
//! - `cacheModel(name, data)` — store bytes
//! - `loadCachedModel(name)` — retrieve bytes (or null)
//! - `listCachedModels()` — list stored names
//! - `deleteCachedModel(name)` — remove a stored model

use js_sys::Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen(inline_js = r#"
const DB_NAME = "oxillama-model-cache";
const STORE_NAME = "models";

function openDb() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = (e) => {
            e.target.result.createObjectStore(STORE_NAME);
        };
        req.onsuccess = (e) => resolve(e.target.result);
        req.onerror = (e) => reject(e.target.error);
    });
}

export async function idb_put_model(name, data) {
    const db = await openDb();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).put(data, name);
        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
    });
}

export async function idb_get_model(name) {
    const db = await openDb();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly");
        const req = tx.objectStore(STORE_NAME).get(name);
        req.onsuccess = (e) => resolve(e.target.result ?? null);
        req.onerror = (e) => reject(e.target.error);
    });
}

export async function idb_list_models() {
    const db = await openDb();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly");
        const req = tx.objectStore(STORE_NAME).getAllKeys();
        req.onsuccess = (e) => resolve(e.target.result);
        req.onerror = (e) => reject(e.target.error);
    });
}

export async function idb_delete_model(name) {
    const db = await openDb();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).delete(name);
        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
    });
}
"#)]
extern "C" {
    async fn idb_put_model(name: &str, data: &[u8]) -> JsValue;
    async fn idb_get_model(name: &str) -> JsValue;
    async fn idb_list_models() -> JsValue;
    async fn idb_delete_model(name: &str) -> JsValue;
}

/// Store a GGUF model in the browser's IndexedDB under the given name.
#[wasm_bindgen(js_name = cacheModel)]
pub async fn cache_model(name: &str, data: &[u8]) -> Result<(), JsValue> {
    idb_put_model(name, data).await;
    Ok(())
}

/// Retrieve a previously-cached GGUF model from IndexedDB.
///
/// Returns `null` in JS if the model is not cached, or a `Uint8Array` if found.
#[wasm_bindgen(js_name = loadCachedModel)]
pub async fn load_cached_model(name: &str) -> Result<JsValue, JsValue> {
    let result = idb_get_model(name).await;
    Ok(result)
}

/// List all model names currently stored in the IndexedDB cache.
#[wasm_bindgen(js_name = listCachedModels)]
pub async fn list_cached_models() -> Result<Array, JsValue> {
    let result = idb_list_models().await;
    result
        .dyn_into::<Array>()
        .map_err(|_| JsValue::from_str("Expected Array from idb_list_models"))
}

/// Remove a model from the IndexedDB cache.
#[wasm_bindgen(js_name = deleteCachedModel)]
pub async fn delete_cached_model(name: &str) -> Result<(), JsValue> {
    idb_delete_model(name).await;
    Ok(())
}

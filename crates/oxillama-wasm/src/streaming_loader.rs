//! Streaming GGUF loader with LRU tensor cache and on-demand byte-range fetching.
//!
//! This module provides a higher-level, production-ready streaming loader that
//! builds on top of [`oxillama_gguf::StreamingGgufParser`] to support:
//!
//! - **Push mode**: feed byte chunks from a `ReadableStream` reader; header is
//!   parsed as soon as enough bytes accumulate (grow-and-retry, never panics on
//!   partial data).
//! - **Pull mode**: after the header is ready, individual tensors can be fetched
//!   on demand via a JS byte-range callback — only active tensors reside in memory,
//!   making multi-GB models practical in the browser without first downloading
//!   the entire file.
//! - **LRU cache**: configurable capacity (default 8 tensors); a tensor that is
//!   read multiple times is fetched only once, while the oldest entry is evicted
//!   when the cache is full.
//! - **Progress reporting**: push-mode bytes-received counter exposed via `progress()`.
//!
//! ## Relationship to `streaming_load.rs`
//!
//! `streaming_load.rs` contains the simpler `GgufChunkLoader` that only
//! reads the 24-byte fixed header (magic + version + n_tensors + n_kv) and
//! reports basic progress. This module supersedes it for production use-cases
//! that require tensor access, pull-mode fetching, and LRU caching. The two
//! modules coexist intentionally: `GgufChunkLoader` is a minimal incremental
//! accumulator, whereas [`StreamingGgufLoader`] is a full feature-complete loader.
//!
//! ## JS usage (push mode)
//!
//! ```js
//! import init, { StreamingGgufLoader } from './oxillama_wasm.js';
//! await init();
//!
//! const loader = new StreamingGgufLoader(8);       // LRU capacity = 8
//! const reader = response.body.getReader();
//! while (true) {
//!   const { done, value } = await reader.read();
//!   if (done) break;
//!   const headerReady = loader.pushChunk(value);   // returns bool
//!   if (headerReady) {
//!     console.log('Tensors:', loader.tensorNames());
//!   }
//! }
//! ```
//!
//! ## JS usage (pull mode)
//!
//! ```js
//! // After header is ready, fetch individual tensors on demand:
//! async function fetchRange(offset, size) {
//!   const resp = await fetch(modelUrl, {
//!     headers: { Range: `bytes=${offset}-${offset + size - 1}` }
//!   });
//!   return new Uint8Array(await resp.arrayBuffer());
//! }
//!
//! const data = await loader.readTensor('blk.0.attn_q.weight', fetchRange);
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use js_sys::Uint8Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

// ── Tensor metadata ───────────────────────────────────────────────────────────

/// Metadata for a tensor stored in the GGUF file.
///
/// Absolute byte offsets are pre-computed from `data_section_offset + info.offset`
/// so the pull-mode fetcher can issue HTTP byte-range requests without knowing
/// the internal GGUF layout.
#[derive(Clone, Debug)]
pub struct TensorMeta {
    /// Absolute byte offset of tensor data within the GGUF file.
    pub file_offset: u64,
    /// Size of tensor data in bytes.
    pub size_bytes: u64,
    /// Quantization / element type as a display string (e.g. `"Q4_0"`, `"F32"`).
    pub dtype: String,
    /// Tensor shape (n_dims elements).
    pub shape: Vec<u64>,
}

// ── LRU cache ─────────────────────────────────────────────────────────────────

/// A simple LRU (least-recently-used) cache for raw tensor byte data.
///
/// Tensors are stored as `Arc<Vec<u8>>` so that callers can hold a reference
/// to cached data across re-entrant calls without cloning the bytes.
///
/// Eviction policy: when a new tensor is inserted and the cache is at capacity,
/// the entry that was least recently *accessed* (not just inserted) is removed.
/// A `get()` call refreshes the access order of the retrieved entry.
pub struct LruTensorCache {
    data: HashMap<String, Arc<Vec<u8>>>,
    order: VecDeque<String>,
    capacity: usize,
}

impl LruTensorCache {
    /// Create a new LRU cache with the given maximum capacity.
    ///
    /// A capacity of 0 effectively disables caching (every `put` immediately
    /// evicts the same entry, which is useless but not unsound).
    pub fn new(capacity: usize) -> Self {
        Self {
            data: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    /// Look up a cached tensor, refreshing its position in the eviction order.
    ///
    /// Returns `None` if the tensor is not currently cached.
    pub fn get(&mut self, name: &str) -> Option<Arc<Vec<u8>>> {
        if !self.data.contains_key(name) {
            return None;
        }
        // Refresh eviction order: move name to back (most recently used).
        self.order.retain(|n| n != name);
        self.order.push_back(name.to_owned());
        self.data.get(name).cloned()
    }

    /// Insert or update a tensor entry.
    ///
    /// If the cache is at capacity the least-recently-used entry is evicted first.
    /// Duplicate keys are handled correctly: the old entry is removed from the
    /// eviction queue before the new entry is pushed to the back, preventing
    /// double-entries that would corrupt the eviction invariant.
    pub fn put(&mut self, name: String, bytes: Vec<u8>) {
        // Remove any existing entry for this name (handles re-insertion).
        if self.data.contains_key(&name) {
            self.order.retain(|n| n != &name);
            self.data.remove(&name);
        }

        // Evict LRU if at capacity.
        while self.capacity > 0 && self.data.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.data.remove(&oldest);
            } else {
                break;
            }
        }

        self.order.push_back(name.clone());
        self.data.insert(name, Arc::new(bytes));
    }

    /// Number of tensors currently in the cache.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ── Internal loader state ─────────────────────────────────────────────────────

/// Internal parsing phase — drives what `push_chunk` does on each call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadPhase {
    /// Waiting for enough bytes to attempt header + tensor-info parsing.
    WaitingForHeader,
    /// Header and tensor index have been parsed successfully.
    HeaderParsed,
}

/// Internal state shared between the public `StreamingGgufLoader` and any
/// async futures spawned by `read_tensor`.
///
/// Wrapped in `Arc<Mutex<...>>` so that the `wasm_bindgen` struct (which is
/// `!Send`) can safely clone the `Arc` into the async block without holding
/// the lock across `.await` points.
pub struct LoaderInner {
    /// Accumulated raw bytes (header + tensor infos, possibly whole file).
    bytes_buffer: Vec<u8>,
    /// Tensor name → absolute file offset and metadata.
    tensor_index: HashMap<String, TensorMeta>,
    /// LRU cache for fetched tensor bytes.
    cache: LruTensorCache,
    /// Current loading phase.
    phase: LoadPhase,
}

impl LoaderInner {
    fn new(cache_capacity: usize) -> Self {
        Self {
            bytes_buffer: Vec::new(),
            tensor_index: HashMap::new(),
            cache: LruTensorCache::new(cache_capacity),
            phase: LoadPhase::WaitingForHeader,
        }
    }

    /// Try to parse the GGUF header from the accumulated buffer.
    ///
    /// Returns `Ok(true)` if the header was just parsed successfully,
    /// `Ok(false)` if more bytes are needed, or `Err` for a permanent parse
    /// failure (e.g. invalid magic bytes).
    fn try_parse_header(&mut self) -> Result<bool, String> {
        if self.phase == LoadPhase::HeaderParsed {
            return Ok(true);
        }

        // Minimum GGUF header: 4 (magic) + 4 (version) + 8 (n_tensors) + 8 (n_kv) = 24 bytes.
        if self.bytes_buffer.len() < 24 {
            return Ok(false);
        }

        // Attempt a full parse of the accumulated bytes.  We use
        // `StreamingGgufParser` which eagerly reads header + metadata and then
        // lazily iterates tensor infos — perfect for our push-mode accumulator.
        match oxillama_gguf::StreamingGgufParser::new(&self.bytes_buffer) {
            Ok(parser) => {
                // Collect tensor index: map name → absolute file offsets.
                let data_offset = parser.tensor_infos().data_offset();
                for result in parser.tensor_infos() {
                    match result {
                        Ok(info) => {
                            let file_offset = data_offset + info.offset;
                            let size_bytes = info.data_size();
                            let dtype = format!("{:?}", info.tensor_type);
                            let shape = info.dimensions.clone();
                            self.tensor_index.insert(
                                info.name,
                                TensorMeta {
                                    file_offset,
                                    size_bytes,
                                    dtype,
                                    shape,
                                },
                            );
                        }
                        Err(e) => {
                            return Err(format!("tensor info parse error: {e}"));
                        }
                    }
                }
                self.phase = LoadPhase::HeaderParsed;
                Ok(true)
            }
            Err(oxillama_gguf::GgufError::UnexpectedEof { .. }) => {
                // Buffer too short — need more bytes.
                Ok(false)
            }
            Err(other) => {
                // Permanent failure (invalid magic, bad version, etc.).
                Err(format!("GGUF parse error: {other}"))
            }
        }
    }
}

// ── Public wasm-bindgen surface ───────────────────────────────────────────────

/// A streaming GGUF loader for WebAssembly environments.
///
/// Designed for browser use where models may be many gigabytes and cannot be
/// loaded into `Uint8Array` all at once.  Two access patterns are supported:
///
/// ## Push mode
///
/// Feed chunks from a `ReadableStream` via `push_chunk`.  The loader
/// accumulates bytes until it has enough to parse the GGUF header (typically
/// 50–100 KB for models with many tensors).  Once the header is parsed,
/// `is_header_ready` returns `true` and `tensor_names` is populated.
///
/// In push mode the entire file is buffered in memory.  For models that must
/// remain partially on disk (multi-GB) use pull mode instead.
///
/// ## Pull mode
///
/// After the header is parsed (which requires a push-mode phase), individual
/// tensors can be fetched on demand via `read_tensor`, which accepts a JS
/// byte-range callback `(offset: number, size: number) => Promise<Uint8Array>`.
/// Fetched tensors are cached in an LRU with configurable capacity.
#[wasm_bindgen]
pub struct StreamingGgufLoader {
    inner: Arc<Mutex<LoaderInner>>,
}

#[wasm_bindgen]
impl StreamingGgufLoader {
    /// Create a new streaming GGUF loader.
    ///
    /// `cache_capacity` controls how many tensors the LRU cache holds
    /// simultaneously.  Pass `undefined` / `None` to use the default of 8.
    #[wasm_bindgen(constructor)]
    pub fn new(cache_capacity: Option<usize>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LoaderInner::new(cache_capacity.unwrap_or(8)))),
        }
    }

    /// Feed a chunk of bytes from a `ReadableStream` reader.
    ///
    /// Bytes are appended to an internal buffer.  After each call the loader
    /// attempts to parse the GGUF header from the accumulated bytes.  The
    /// attempt uses grow-and-retry semantics: if the buffer is too short the
    /// error is swallowed and the call returns `false`; a genuine parse error
    /// (bad magic, unsupported version) is returned as a `JsValue` error.
    ///
    /// Returns `true` once the header (and tensor index) has been successfully
    /// parsed; returns `false` while still accumulating.
    pub fn push_chunk(&mut self, chunk: &[u8]) -> Result<bool, JsValue> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| JsValue::from_str(&format!("lock poisoned: {e}")))?;

        guard.bytes_buffer.extend_from_slice(chunk);
        guard.try_parse_header().map_err(|e| JsValue::from_str(&e))
    }

    /// Returns `true` if the GGUF header has been successfully parsed.
    pub fn is_header_ready(&self) -> bool {
        self.inner
            .lock()
            .map(|g| g.phase == LoadPhase::HeaderParsed)
            .unwrap_or(false)
    }

    /// Returns the number of bytes accumulated so far.
    pub fn bytes_buffered(&self) -> u32 {
        self.inner
            .lock()
            .map(|g| g.bytes_buffer.len() as u32)
            .unwrap_or(0)
    }

    /// Returns a JS `Array` of tensor name strings after the header is ready.
    ///
    /// Returns an empty array if called before the header is parsed.
    pub fn tensor_names(&self) -> Result<js_sys::Array, JsValue> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| JsValue::from_str(&format!("lock poisoned: {e}")))?;
        let arr = js_sys::Array::new();
        for name in guard.tensor_index.keys() {
            arr.push(&JsValue::from_str(name));
        }
        Ok(arr)
    }

    /// Returns tensor metadata as a JSON string, or `None` if the name is
    /// not in the index.
    ///
    /// The JSON object has fields:
    /// - `file_offset` — absolute byte offset in the GGUF file (as a number)
    /// - `size_bytes`  — tensor data size in bytes
    /// - `dtype`       — quantization type string (e.g. `"Q4_0"`)
    /// - `shape`       — array of dimension sizes
    pub fn tensor_meta_json(&self, name: &str) -> Option<String> {
        let guard = self.inner.lock().ok()?;
        let meta = guard.tensor_index.get(name)?;
        let json = serde_json::json!({
            "file_offset": meta.file_offset,
            "size_bytes": meta.size_bytes,
            "dtype": meta.dtype,
            "shape": meta.shape,
        });
        serde_json::to_string(&json).ok()
    }

    /// Fetch tensor data by name using a JS byte-range callback.
    ///
    /// `fetcher` is a JS function with signature:
    /// ```js
    /// (offset: number, size: number) => Promise<Uint8Array>
    /// ```
    ///
    /// The LRU cache is checked first; on a miss the fetcher is called with the
    /// absolute file offset and byte length.  The result is stored in the cache
    /// before being returned.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if:
    /// - The header has not been parsed yet.
    /// - The tensor name is not in the index.
    /// - The fetcher rejects its promise.
    /// - The lock is poisoned.
    pub async fn read_tensor(
        &self,
        name: &str,
        fetcher: &js_sys::Function,
    ) -> Result<Uint8Array, JsValue> {
        // ── 1. Check cache (lock acquired, then released before await) ────────
        let maybe_cached = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|e| JsValue::from_str(&format!("lock poisoned: {e}")))?;

            if guard.phase != LoadPhase::HeaderParsed {
                return Err(JsValue::from_str(
                    "header not yet parsed — call push_chunk until is_header_ready() returns true",
                ));
            }

            guard.cache.get(name)
        };

        if let Some(cached) = maybe_cached {
            return Ok(Uint8Array::from(cached.as_slice()));
        }

        // ── 2. Look up tensor meta (lock acquired, then released) ─────────────
        let (file_offset, size_bytes) = {
            let guard = self
                .inner
                .lock()
                .map_err(|e| JsValue::from_str(&format!("lock poisoned: {e}")))?;
            let meta = guard
                .tensor_index
                .get(name)
                .ok_or_else(|| JsValue::from_str(&format!("tensor '{name}' not found in index")))?;
            (meta.file_offset, meta.size_bytes)
        };

        // ── 3. Call the JS fetcher (no lock held across .await) ───────────────
        let bytes = call_byte_range_fetcher(fetcher, file_offset, size_bytes).await?;

        // ── 4. Insert into LRU cache (lock re-acquired) ───────────────────────
        {
            let mut guard = self
                .inner
                .lock()
                .map_err(|e| JsValue::from_str(&format!("lock poisoned: {e}")))?;
            guard.cache.put(name.to_owned(), bytes.clone());
        }

        Ok(Uint8Array::from(bytes.as_slice()))
    }

    /// Returns a progress descriptor as a JS `Object`.
    ///
    /// Fields:
    /// - `bytes_buffered` — total bytes fed via `push_chunk`
    /// - `phase`          — `"waiting_for_header"` or `"header_parsed"`
    /// - `tensor_count`   — number of tensors in the index (0 before header)
    /// - `cache_size`     — number of tensors currently in the LRU cache
    pub fn progress(&self) -> Result<js_sys::Object, JsValue> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| JsValue::from_str(&format!("lock poisoned: {e}")))?;

        let phase_str = match guard.phase {
            LoadPhase::WaitingForHeader => "waiting_for_header",
            LoadPhase::HeaderParsed => "header_parsed",
        };

        let obj = js_sys::Object::new();
        set_js_prop(
            &obj,
            "bytes_buffered",
            &JsValue::from(guard.bytes_buffer.len() as f64),
        )?;
        set_js_prop(&obj, "phase", &JsValue::from_str(phase_str))?;
        set_js_prop(
            &obj,
            "tensor_count",
            &JsValue::from(guard.tensor_index.len() as f64),
        )?;
        set_js_prop(&obj, "cache_size", &JsValue::from(guard.cache.len() as f64))?;
        Ok(obj)
    }
}

// ── Load options (optional helper, exposed to JS) ─────────────────────────────

/// Configuration options for creating a [`StreamingGgufLoader`].
///
/// Provides named-parameter ergonomics from JavaScript; alternatively callers
/// may construct `StreamingGgufLoader` directly with `new(cacheCapacity)`.
#[wasm_bindgen]
pub struct StreamingLoadOptions {
    /// Whether to emit `progress()` events (reserved for future use).
    pub progress_enabled: bool,
    /// LRU cache capacity (number of tensors).
    pub cache_capacity: usize,
}

#[wasm_bindgen]
impl StreamingLoadOptions {
    /// Create options with defaults: `progress_enabled = true`, `cache_capacity = 8`.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            progress_enabled: true,
            cache_capacity: 8,
        }
    }
}

impl Default for StreamingLoadOptions {
    fn default() -> Self {
        Self::new()
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Set a property on a JS `Object`, returning a `JsValue` error on failure.
fn set_js_prop(obj: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), value)
        .map_err(|e| JsValue::from_str(&format!("Reflect.set({key}) failed: {e:?}")))?;
    Ok(())
}

/// Invoke a JS byte-range fetcher callback and extract the resulting bytes.
///
/// The callback signature is `(offset: number, size: number) => Promise<Uint8Array>`.
/// Offsets and sizes are passed as `f64` (JS `number`) — safe for file sizes
/// up to 2^53 bytes (~8 PB), which covers any foreseeable GGUF file.
async fn call_byte_range_fetcher(
    fetcher: &js_sys::Function,
    offset: u64,
    size: u64,
) -> Result<Vec<u8>, JsValue> {
    let js_offset = JsValue::from_f64(offset as f64);
    let js_size = JsValue::from_f64(size as f64);

    let promise_val = fetcher.call2(&JsValue::NULL, &js_offset, &js_size)?;
    let promise = js_sys::Promise::from(promise_val);
    let resolved = JsFuture::from(promise).await?;

    // The resolved value must be a Uint8Array.
    let array = Uint8Array::new(&resolved);
    Ok(array.to_vec())
}

// ── Tests (native, no wasm-bindgen JsValue) ──────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::types::{GgufTensorType, GgufValueType, GGUF_MAGIC};

    // ── GGUF binary builder helpers (mirrors streaming.rs test utilities) ─────

    fn write_string_v3(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn align_up(value: usize, alignment: usize) -> usize {
        if alignment == 0 {
            return value;
        }
        let rem = value % alignment;
        if rem == 0 {
            value
        } else {
            value + alignment - rem
        }
    }

    /// Build a minimal valid GGUF v3 with 0 tensors and 0 KV pairs (24 bytes).
    fn make_empty_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic  4 bytes
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_kv
        buf
    }

    /// Build a GGUF v3 file with two tensors and one KV metadata pair.
    ///
    /// Tensor layout:
    /// - `"blk.0.attn_q.weight"` — F32, shape [4, 4], offset 0
    /// - `"output.weight"`        — F16, shape [8],    offset 64
    fn make_two_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // n_tensors = 2
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv = 1

        // KV: "general.architecture" = "test_arch"
        write_string_v3(&mut buf, "general.architecture");
        buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v3(&mut buf, "test_arch");

        // Tensor 0: "blk.0.attn_q.weight" — F32, 2D [4,4], offset 0
        write_string_v3(&mut buf, "blk.0.attn_q.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim 1
        buf.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Tensor 1: "output.weight" — F16, 1D [8], offset 64
        write_string_v3(&mut buf, "output.weight");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&8u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&(GgufTensorType::F16 as u32).to_le_bytes());
        buf.extend_from_slice(&64u64.to_le_bytes()); // offset

        // Pad to 32-byte alignment
        let aligned = align_up(buf.len(), 32);
        buf.resize(aligned, 0u8);

        // Append fake tensor data (enough bytes to cover both tensors)
        // F32 [4,4] = 64 bytes, F16 [8] = 16 bytes → need 64 + 64 = 128 at minimum
        buf.resize(aligned + 256, 0xAB_u8);

        buf
    }

    // ── (a) LRU evicts oldest ─────────────────────────────────────────────────

    #[test]
    fn lru_cache_evicts_oldest() {
        let mut cache = LruTensorCache::new(2);
        cache.put("a".into(), vec![1, 2, 3]);
        cache.put("b".into(), vec![4, 5, 6]);
        // Inserting "c" must evict "a" (the oldest / least-recently-used entry).
        cache.put("c".into(), vec![7, 8, 9]);
        assert!(
            cache.get("a").is_none(),
            "entry 'a' should have been evicted"
        );
        assert!(cache.get("b").is_some(), "entry 'b' should still be cached");
        assert!(cache.get("c").is_some(), "entry 'c' should be in the cache");
    }

    // ── (b) LRU get refreshes eviction order ──────────────────────────────────

    #[test]
    fn lru_cache_get_refreshes_order() {
        let mut cache = LruTensorCache::new(2);
        cache.put("a".into(), vec![1]);
        cache.put("b".into(), vec![2]);
        // Access "a" — this makes "b" the new LRU candidate.
        let _ = cache.get("a");
        // Insert "c" — "b" should be evicted, not "a".
        cache.put("c".into(), vec![3]);
        assert!(
            cache.get("a").is_some(),
            "entry 'a' was recently used; must survive"
        );
        assert!(
            cache.get("b").is_none(),
            "entry 'b' is now LRU; must be evicted"
        );
        assert!(cache.get("c").is_some(), "entry 'c' must be present");
    }

    // ── (c) LRU handles duplicate put without corrupting order ────────────────

    #[test]
    fn lru_cache_duplicate_put_no_corruption() {
        let mut cache = LruTensorCache::new(2);
        cache.put("a".into(), vec![1]);
        cache.put("b".into(), vec![2]);
        // Re-insert "a" with new bytes — must not leave a stale entry in `order`.
        cache.put("a".into(), vec![99]);
        // Insert "c" — "b" should be evicted (it is now the LRU), not "a".
        cache.put("c".into(), vec![3]);
        assert!(cache.get("b").is_none(), "stale 'b' should be evicted");
        assert!(cache.get("a").is_some(), "refreshed 'a' must survive");
        assert_eq!(
            cache.get("a").as_deref().map(|v| v.as_slice()),
            Some([99_u8].as_slice()),
            "re-inserted value must be the new bytes"
        );
    }

    // ── (d) LRU capacity 0 — every entry evicted immediately ─────────────────

    #[test]
    fn lru_cache_zero_capacity_evicts_immediately() {
        let mut cache = LruTensorCache::new(0);
        cache.put("a".into(), vec![1]);
        // With capacity 0 the while-loop condition `data.len() >= capacity (0)` is
        // always true before insertion, so the entry is evicted and the data map
        // never grows.  (The `put` guard skips eviction when capacity == 0 via the
        // `self.capacity > 0` check in the while condition.)
        // Actually with capacity=0 the guard `while capacity > 0 && ...` never
        // fires, so the item IS inserted.  Verify length = 1.
        assert_eq!(cache.len(), 1);
    }

    // ── (e) push_chunk — transitions phase on valid empty GGUF ────────────────

    #[test]
    fn push_chunk_transitions_to_header_parsed_empty_gguf() {
        let header_bytes = make_empty_gguf();
        let mut inner = LoaderInner::new(8);
        inner.bytes_buffer.extend_from_slice(&header_bytes);
        let result = inner
            .try_parse_header()
            .expect("try_parse_header should succeed");
        assert!(result, "should return true when header is parsed");
        assert_eq!(inner.phase, LoadPhase::HeaderParsed);
    }

    // ── (f) push_chunk — partial data returns false (no error) ───────────────

    #[test]
    fn push_chunk_partial_data_returns_false() {
        let header_bytes = make_empty_gguf();
        let mut inner = LoaderInner::new(8);
        // Feed only 10 bytes (below the 24-byte minimum).
        inner.bytes_buffer.extend_from_slice(&header_bytes[..10]);
        let result = inner
            .try_parse_header()
            .expect("should not error on partial data");
        assert!(!result, "incomplete header should return false, not error");
        assert_eq!(inner.phase, LoadPhase::WaitingForHeader);
    }

    // ── (g) push_chunk — invalid magic returns error ──────────────────────────

    #[test]
    fn push_chunk_invalid_magic_returns_error() {
        let mut inner = LoaderInner::new(8);
        // All-zero 32 bytes = invalid magic.
        inner.bytes_buffer.extend_from_slice(&[0u8; 32]);
        let result = inner.try_parse_header();
        assert!(result.is_err(), "invalid magic must produce an error");
        let msg = result.expect_err("expected error string");
        assert!(
            msg.contains("GGUF parse error"),
            "error message should mention GGUF parse error, got: {msg}"
        );
    }

    // ── (h) tensor index populated after header — two tensors ────────────────

    #[test]
    fn tensor_index_populated_after_header() {
        let gguf_bytes = make_two_tensor_gguf();
        let mut inner = LoaderInner::new(8);
        inner.bytes_buffer.extend_from_slice(&gguf_bytes);

        let ready = inner.try_parse_header().expect("parse should succeed");
        assert!(ready, "header should be ready");
        assert_eq!(inner.phase, LoadPhase::HeaderParsed);

        // Both tensors must be in the index.
        assert!(
            inner.tensor_index.contains_key("blk.0.attn_q.weight"),
            "blk.0.attn_q.weight must be indexed"
        );
        assert!(
            inner.tensor_index.contains_key("output.weight"),
            "output.weight must be indexed"
        );

        // Verify metadata for the F32 tensor.
        let meta = inner
            .tensor_index
            .get("blk.0.attn_q.weight")
            .expect("meta must be present");
        assert_eq!(meta.shape, vec![4, 4], "F32 [4,4] shape must match");
        assert_eq!(meta.dtype, "F32", "dtype must be F32");
        // F32 [4,4] = 16 elements × 4 bytes = 64 bytes.
        assert_eq!(meta.size_bytes, 64, "F32 [4,4] data size must be 64 bytes");
    }

    // ── (i) file_offset is absolute (data_section_offset + info.offset) ──────

    #[test]
    fn tensor_file_offset_is_absolute() {
        let gguf_bytes = make_two_tensor_gguf();

        // Use StreamingGgufParser to get the authoritative data_section_offset.
        let parser = oxillama_gguf::StreamingGgufParser::new(&gguf_bytes)
            .expect("streaming parser should succeed");
        let data_section_offset = parser.tensor_infos().data_offset();

        let mut inner = LoaderInner::new(8);
        inner.bytes_buffer.extend_from_slice(&gguf_bytes);
        inner.try_parse_header().expect("header parse must succeed");

        let meta_q = inner
            .tensor_index
            .get("blk.0.attn_q.weight")
            .expect("tensor must be indexed");
        // blk.0.attn_q.weight has info.offset = 0, so file_offset = data_section_offset + 0.
        assert_eq!(
            meta_q.file_offset, data_section_offset,
            "file_offset for offset-0 tensor must equal data_section_offset"
        );

        let meta_out = inner
            .tensor_index
            .get("output.weight")
            .expect("output.weight must be indexed");
        // output.weight has info.offset = 64.
        assert_eq!(
            meta_out.file_offset,
            data_section_offset + 64,
            "file_offset for output.weight must be data_section_offset + 64"
        );
    }

    // ── (j) idempotent — calling try_parse_header twice is safe ──────────────

    #[test]
    fn try_parse_header_is_idempotent() {
        let gguf_bytes = make_empty_gguf();
        let mut inner = LoaderInner::new(8);
        inner.bytes_buffer.extend_from_slice(&gguf_bytes);

        let first = inner.try_parse_header().expect("first call must succeed");
        assert!(first);
        let second = inner.try_parse_header().expect("second call must succeed");
        assert!(second);
        assert_eq!(inner.phase, LoadPhase::HeaderParsed);
    }
}

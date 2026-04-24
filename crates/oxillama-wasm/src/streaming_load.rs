//! Streaming / chunked GGUF loader.
//!
//! Provides a stateful accumulator that JS can feed byte chunks into as they
//! arrive from `fetch()` + `ReadableStream`. Once enough data is accumulated,
//! the GGUF header (and eventually tensors) can be parsed incrementally.
//!
//! JavaScript usage:
//! ```js
//! const loader = new GgufChunkLoader();
//! const reader = response.body.getReader();
//! while (true) {
//!   const {done, value} = await reader.read();
//!   if (done) break;
//!   const result = loader.feed(value);
//!   const status = JSON.parse(result);
//!   if (status.header_ready) console.log('header parsed!', status.n_tensors);
//! }
//! ```

use wasm_bindgen::prelude::*;

/// GGUF file magic bytes (little-endian `GGUF`).
const GGUF_MAGIC: u32 = 0x4655_4747;

/// Minimum bytes required to extract magic + version + tensor count + kv count.
const GGUF_MIN_HEADER_BYTES: usize = 24;

/// Internal state for incremental GGUF parsing.
///
/// Separated from the wasm-bindgen wrapper so that it can be tested
/// with `cargo nextest` on native targets without `Send` constraints.
pub struct GgufChunkLoaderInner {
    buffer: Vec<u8>,
    header_parsed: bool,
    n_tensors: u64,
    total_bytes_fed: usize,
}

impl GgufChunkLoaderInner {
    /// Create a new inner loader.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            header_parsed: false,
            n_tensors: 0,
            total_bytes_fed: 0,
        }
    }

    /// Feed a slice of bytes, attempting to parse the header when sufficient
    /// data has accumulated.
    ///
    /// Returns a tuple `(bytes_received, header_ready, n_tensors)`.
    ///
    /// # Errors
    /// Returns a string error if the magic bytes are invalid.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<(usize, bool, u64), String> {
        self.buffer.extend_from_slice(chunk);
        self.total_bytes_fed += chunk.len();

        if !self.header_parsed && self.buffer.len() >= GGUF_MIN_HEADER_BYTES {
            let magic_bytes: [u8; 4] = self.buffer[0..4]
                .try_into()
                .map_err(|_| "buffer too short for magic".to_string())?;
            let magic = u32::from_le_bytes(magic_bytes);
            if magic != GGUF_MAGIC {
                return Err("invalid GGUF magic bytes".to_string());
            }

            let n_tensors_bytes: [u8; 8] = self.buffer[8..16]
                .try_into()
                .map_err(|_| "buffer too short for tensor count".to_string())?;
            self.n_tensors = u64::from_le_bytes(n_tensors_bytes);
            self.header_parsed = true;
        }

        Ok((self.total_bytes_fed, self.header_parsed, self.n_tensors))
    }

    /// Total bytes received so far.
    pub fn bytes_received(&self) -> usize {
        self.total_bytes_fed
    }

    /// Whether the GGUF header has been fully parsed.
    pub fn header_ready(&self) -> bool {
        self.header_parsed
    }

    /// Number of tensors declared in the header (0 if header not yet parsed).
    pub fn n_tensors(&self) -> u64 {
        self.n_tensors
    }
}

impl Default for GgufChunkLoaderInner {
    fn default() -> Self {
        Self::new()
    }
}

// ── wasm-bindgen wrapper ──────────────────────────────────────────────────────

/// A stateful GGUF chunk accumulator exposed to JavaScript.
///
/// JavaScript usage:
/// ```js
/// const loader = new GgufChunkLoader();
/// // Feed chunks as they arrive
/// const reader = response.body.getReader();
/// while (true) {
///   const {done, value} = await reader.read();
///   if (done) break;
///   const result = loader.feed(value);
///   if (result.header_ready) console.log('header parsed!', result.n_tensors);
/// }
/// ```
#[wasm_bindgen]
pub struct GgufChunkLoader {
    inner: GgufChunkLoaderInner,
}

#[wasm_bindgen]
impl GgufChunkLoader {
    /// Create a new chunk loader.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: GgufChunkLoaderInner::new(),
        }
    }

    /// Feed a new chunk of bytes. Returns a JSON status object.
    ///
    /// # Errors
    /// Returns a JS error string on parse failure.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<JsValue, JsValue> {
        let (bytes_received, header_ready, n_tensors) =
            self.inner.feed(chunk).map_err(|e| JsValue::from_str(&e))?;

        let status = serde_json::json!({
            "bytes_received": bytes_received,
            "header_ready": header_ready,
            "n_tensors": n_tensors,
        });
        serde_json::to_string(&status)
            .map(|s| JsValue::from_str(&s))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Total bytes received so far.
    pub fn bytes_received(&self) -> usize {
        self.inner.bytes_received()
    }

    /// Whether the GGUF header has been fully parsed.
    pub fn header_ready(&self) -> bool {
        self.inner.header_ready()
    }

    /// Number of tensors declared in the header (0 if header not yet parsed).
    pub fn n_tensors(&self) -> u64 {
        self.inner.n_tensors()
    }
}

impl Default for GgufChunkLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gguf_magic_header() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic
        v.extend_from_slice(&3u32.to_le_bytes()); // version
        v.extend_from_slice(&42u64.to_le_bytes()); // n_tensors
        v.extend_from_slice(&0u64.to_le_bytes()); // n_kv
        v
    }

    #[test]
    fn feed_valid_header_sets_header_ready() {
        let mut loader = GgufChunkLoaderInner::new();
        let header = make_gguf_magic_header();
        let result = loader.feed(&header);
        assert!(result.is_ok());
        assert!(loader.header_ready());
        assert_eq!(loader.n_tensors(), 42);
    }

    #[test]
    fn feed_invalid_magic_returns_error() {
        let mut loader = GgufChunkLoaderInner::new();
        let bad: Vec<u8> = vec![0u8; 32]; // all zeros = bad magic
        let result = loader.feed(&bad);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("invalid GGUF magic"), "unexpected: {msg}");
    }

    #[test]
    fn chunked_feed_accumulates_bytes() {
        let mut loader = GgufChunkLoaderInner::new();
        let header = make_gguf_magic_header();
        // Feed in two halves
        let _ = loader.feed(&header[..8]);
        assert!(
            !loader.header_ready(),
            "header should not be ready with only 8 bytes"
        );
        let _ = loader.feed(&header[8..]);
        assert!(loader.header_ready());
        assert_eq!(loader.bytes_received(), header.len());
    }

    #[test]
    fn bytes_received_tracks_total() {
        let mut loader = GgufChunkLoaderInner::new();
        let header = make_gguf_magic_header();
        let _ = loader.feed(&header[..4]);
        let _ = loader.feed(&header[4..12]);
        // Only 12 bytes fed so far, not enough for header
        assert_eq!(loader.bytes_received(), 12);
        assert!(!loader.header_ready());
    }

    #[test]
    fn default_trait_works() {
        let loader = GgufChunkLoaderInner::default();
        assert!(!loader.header_ready());
        assert_eq!(loader.n_tensors(), 0);
        assert_eq!(loader.bytes_received(), 0);
    }

    #[test]
    fn large_tensor_count() {
        let mut v = Vec::new();
        v.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&999_999u64.to_le_bytes()); // n_tensors
        v.extend_from_slice(&0u64.to_le_bytes());

        let mut loader = GgufChunkLoaderInner::new();
        let result = loader.feed(&v);
        assert!(result.is_ok());
        assert_eq!(loader.n_tensors(), 999_999);
    }
}

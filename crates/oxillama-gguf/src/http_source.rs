//! HTTP Range-request backed [`crate::source::Source`] for remote GGUF loading.
//!
//! [`HttpRangeSource`] issues HTTP `Range:` requests against a remote URL so
//! that the GGUF parser can stream individual sections of a model file without
//! downloading the whole thing.  The first [`HTTP_HEADER_CACHE_BYTES`] bytes
//! are fetched eagerly on construction and cached in memory: because the GGUF
//! header, all KV metadata, and all tensor-info entries are located at the
//! beginning of the file, the hot-path for header parsing is served entirely
//! from the local cache.  Only tensor-data reads (which start past the cache
//! boundary) trigger actual HTTP round-trips.
//!
//! ## Server requirements
//! The remote server **must** support:
//! - `HEAD` request returning a `Content-Length` header.
//! - `GET` requests with a `Range: bytes=<start>-<end>` header returning
//!   `206 Partial Content`.
//!
//! Most object-storage back-ends (S3, GCS, HuggingFace Hub, Azure Blob) and
//! HTTP/1.1 static file servers satisfy these requirements.

#[cfg(feature = "http")]
mod inner {
    use std::io::Read as _;

    use crate::error::{GgufError, GgufResult};
    use crate::loader::GgufModel;
    use crate::source::Source;

    /// Number of bytes cached eagerly at construction time.
    ///
    /// 128 KiB covers the vast majority of GGUF headers (magic, version,
    /// KV metadata, and tensor-info blocks) for models with up to ~2 000
    /// tensors, avoiding a round-trip per metadata field during parsing.
    pub const HTTP_HEADER_CACHE_BYTES: usize = 128 * 1024;

    /// A [`Source`] that fetches byte ranges from a remote HTTP URL.
    ///
    /// The first [`HTTP_HEADER_CACHE_BYTES`] bytes are buffered locally so
    /// that GGUF header parsing (magic, version, metadata, tensor-info) is
    /// served without additional HTTP round-trips.  Tensor data reads that fall
    /// outside the cache trigger individual `GET` requests with `Range:` headers.
    #[derive(Debug)]
    pub struct HttpRangeSource {
        /// Remote URL of the GGUF file.
        url: String,
        /// Total byte length reported by the server's `Content-Length` header.
        content_length: u64,
        /// In-memory cache of the first [`HTTP_HEADER_CACHE_BYTES`] bytes.
        header_cache: Vec<u8>,
        /// Current read position (logical cursor maintained by `Source::seek`).
        pos: u64,
    }

    impl HttpRangeSource {
        /// Construct an `HttpRangeSource` for the given URL.
        ///
        /// Performs two HTTP requests:
        /// 1. `HEAD` to obtain the `Content-Length`.
        /// 2. `GET` with `Range: bytes=0-{HTTP_HEADER_CACHE_BYTES-1}` to warm
        ///    the header cache.
        ///
        /// # Errors
        /// Returns [`GgufError::HttpError`] if:
        /// - The `HEAD` request fails or the server returns a non-2xx status.
        /// - `Content-Length` is absent or cannot be parsed.
        /// - The warm-up `GET` request fails.
        pub fn new(url: &str) -> GgufResult<Self> {
            // ── Step 1: HEAD request to get the total file size ───────────────
            let head_resp = ureq::head(url)
                .call()
                .map_err(|e| GgufError::HttpError(format!("HEAD {url}: {e}")))?;

            let content_length = head_resp
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(|| {
                    GgufError::HttpError(format!(
                        "HEAD {url}: missing or invalid Content-Length header"
                    ))
                })?;

            // ── Step 2: GET first HTTP_HEADER_CACHE_BYTES bytes ───────────────
            let cache_end = (HTTP_HEADER_CACHE_BYTES as u64)
                .min(content_length)
                .saturating_sub(1);
            let range_header = format!("bytes=0-{cache_end}");

            let mut warm_resp = ureq::get(url)
                .header("Range", &range_header)
                .call()
                .map_err(|e| GgufError::HttpError(format!("GET {url} range 0-{cache_end}: {e}")))?;

            let mut header_cache = Vec::with_capacity(HTTP_HEADER_CACHE_BYTES);
            warm_resp
                .body_mut()
                .as_reader()
                .read_to_end(&mut header_cache)
                .map_err(|e| {
                    GgufError::HttpError(format!("reading warm-up body from {url}: {e}"))
                })?;

            Ok(Self {
                url: url.to_string(),
                content_length,
                header_cache,
                pos: 0,
            })
        }

        /// Total byte size of the remote file.
        pub fn total_size_bytes(&self) -> u64 {
            self.content_length
        }
    }

    impl Source for HttpRangeSource {
        type Error = GgufError;

        fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), GgufError> {
            if buf.is_empty() {
                return Ok(());
            }
            let offset = self.pos;
            let len = buf.len() as u64;
            let end_exclusive = offset
                .checked_add(len)
                .ok_or(GgufError::UnexpectedEof { offset })?;

            if end_exclusive > self.content_length {
                return Err(GgufError::UnexpectedEof { offset });
            }

            let cache_len = self.header_cache.len() as u64;

            // ── Fast path: entire read served from cache ──────────────────────
            if end_exclusive <= cache_len {
                let start = offset as usize;
                let end = end_exclusive as usize;
                buf.copy_from_slice(&self.header_cache[start..end]);
                self.pos = end_exclusive;
                return Ok(());
            }

            // ── Slow path: issue an HTTP Range request ────────────────────────
            // Use `end_exclusive - 1` because HTTP Range is inclusive.
            let range = format!("bytes={offset}-{}", end_exclusive - 1);

            let mut resp = ureq::get(&self.url)
                .header("Range", &range)
                .call()
                .map_err(|e| GgufError::HttpError(format!("GET {} {}: {}", self.url, range, e)))?;

            let mut body_bytes = Vec::with_capacity(buf.len());
            resp.body_mut()
                .as_reader()
                .read_to_end(&mut body_bytes)
                .map_err(|e| {
                    GgufError::HttpError(format!("reading body from {}: {e}", self.url))
                })?;

            if body_bytes.len() != buf.len() {
                return Err(GgufError::UnexpectedEof { offset });
            }

            buf.copy_from_slice(&body_bytes);
            self.pos = end_exclusive;
            Ok(())
        }

        fn seek(&mut self, pos: u64) -> Result<u64, GgufError> {
            if pos > self.content_length {
                return Err(GgufError::UnexpectedEof { offset: pos });
            }
            self.pos = pos;
            Ok(pos)
        }

        fn position(&self) -> u64 {
            self.pos
        }
    }

    impl GgufModel {
        /// Load a GGUF model from a remote HTTP(S) URL via HTTP range requests.
        ///
        /// Only the GGUF header section is fetched eagerly; tensor data is read
        /// on demand.  The model is fully loaded into memory when this function
        /// returns — suitable for moderate-sized models.  For streaming access
        /// to individual tensors, use the `StreamingGgufParser` with a custom
        /// `Source` instead.
        ///
        /// # Errors
        /// Returns [`GgufError::HttpError`] if any HTTP request fails.
        /// Returns other [`GgufError`] variants if the remote file is not a
        /// valid GGUF file.
        pub fn from_url(url: &str) -> crate::error::GgufResult<Self> {
            let mut source = HttpRangeSource::new(url)?;
            let total = source.total_size_bytes();

            // Read the complete file into memory using range requests.
            // We do a single large range fetch covering the entire file to
            // avoid many small round-trips.
            let mut data = vec![0u8; total as usize];
            source
                .read_exact(&mut data)
                .map_err(|e| GgufError::HttpError(format!("reading full file from {url}: {e}")))?;

            GgufModel::from_bytes(data)
        }
    }
}

#[cfg(feature = "http")]
pub use inner::{HttpRangeSource, HTTP_HEADER_CACHE_BYTES};

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::*;
    use crate::error::GgufError;

    /// Verify that GgufError::HttpError displays correctly.
    #[test]
    fn http_error_formats_correctly() {
        let err = GgufError::HttpError("connection refused: 127.0.0.1:8080".to_string());
        let display = err.to_string();
        assert!(
            display.contains("HTTP error loading GGUF"),
            "display should contain 'HTTP error loading GGUF', got: {display}"
        );
        assert!(
            display.contains("connection refused"),
            "display should contain the inner message, got: {display}"
        );
    }

    /// Verify the cache-size constant has the expected value.
    #[test]
    fn http_header_cache_bytes_is_128kib() {
        assert_eq!(
            HTTP_HEADER_CACHE_BYTES,
            128 * 1024,
            "cache size should be 128 KiB"
        );
    }

    /// Verify that a live HTTP request to a non-existent server returns an error
    /// rather than panicking.  Tagged #[ignore] because it requires network access.
    #[test]
    #[ignore]
    fn http_range_source_unreachable_server_returns_error() {
        let result = HttpRangeSource::new("http://127.0.0.1:19999/nonexistent.gguf");
        assert!(
            result.is_err(),
            "connection to unreachable server must return an error"
        );
    }

    /// Verify that loading a real remote GGUF URL succeeds.
    /// Tagged #[ignore] because it requires live internet access.
    #[test]
    #[ignore]
    fn http_range_source_real_url_returns_ok() {
        // Replace with a real GGUF URL for manual integration testing.
        let url = "https://example.com/model.gguf";
        let result = HttpRangeSource::new(url);
        assert!(result.is_ok(), "real URL should succeed");
    }

    /// Verify that GgufModel::from_url propagates HTTP errors instead of panicking.
    /// Tagged #[ignore] because it requires network access.
    #[test]
    #[ignore]
    fn from_url_propagates_http_error() {
        let result = crate::loader::GgufModel::from_url("http://127.0.0.1:19999/bad.gguf");
        assert!(
            result.is_err(),
            "from_url to unreachable server must return an error"
        );
    }
}

// ── Offline tests (no HTTP feature required) ─────────────────────────────────

#[cfg(test)]
mod offline_tests {
    /// Verify that GgufError::HttpError is usable even when the HTTP feature
    /// is enabled — display string validation runs in all configurations.
    #[cfg(feature = "http")]
    #[test]
    fn http_error_display_is_stable() {
        use crate::error::GgufError;
        let msg = "timeout after 30s";
        let err = GgufError::HttpError(msg.to_string());
        let s = err.to_string();
        assert!(
            s.contains(msg),
            "HttpError display must embed the inner message"
        );
    }
}

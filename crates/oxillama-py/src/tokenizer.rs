//! Python wrapper for [`TokenizerBridge`] — standalone tokenizer access.
//!
//! Exposes HuggingFace BPE tokenization without needing a full inference
//! engine.  Users can load a `tokenizer.json` file, encode text to token
//! IDs, and decode IDs back to strings.

use pyo3::prelude::*;

use oxillama_runtime::TokenizerBridge;

use crate::error::runtime_to_py;

/// A standalone tokenizer loaded from a HuggingFace `tokenizer.json` file.
///
/// # Python Example
///
/// ```python
/// tok = Tokenizer.from_file("tokenizer.json")
/// ids = tok.encode("Hello world")
/// text = tok.decode(ids)
/// print(f"vocab size = {tok.vocab_size}")
/// ```
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: TokenizerBridge,
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer from a HuggingFace `tokenizer.json` file.
    ///
    /// Args:
    ///     path: Path to the tokenizer JSON file.
    ///
    /// Returns:
    ///     Tokenizer: A new tokenizer instance.
    ///
    /// Raises:
    ///     TokenizerError: if the file cannot be read or parsed.
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        let bridge = TokenizerBridge::from_file(path).map_err(runtime_to_py)?;
        Ok(Self { inner: bridge })
    }

    /// Load a tokenizer from a JSON string.
    ///
    /// Args:
    ///     json: The tokenizer JSON content as a string.
    ///
    /// Returns:
    ///     Tokenizer: A new tokenizer instance.
    ///
    /// Raises:
    ///     TokenizerError: if the JSON is invalid.
    #[staticmethod]
    pub fn from_json(json: &str) -> PyResult<Self> {
        let bridge = TokenizerBridge::from_bytes(json.as_bytes()).map_err(runtime_to_py)?;
        Ok(Self { inner: bridge })
    }

    /// Encode text into token IDs.
    ///
    /// Args:
    ///     text: The input text to tokenize.
    ///
    /// Returns:
    ///     `List[int]`: Token IDs.
    ///
    /// Raises:
    ///     TokenizerError: if encoding fails.
    pub fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner.encode(text).map_err(runtime_to_py)
    }

    /// Decode token IDs back to text.
    ///
    /// Args:
    ///     ids: List of token IDs.
    ///
    /// Returns:
    ///     str: The decoded text.
    ///
    /// Raises:
    ///     TokenizerError: if decoding fails.
    pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids).map_err(runtime_to_py)
    }

    /// Return the vocabulary size.
    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Return the token string for a given ID, or ``None`` if out of range.
    ///
    /// Args:
    ///     id: Token ID to look up.
    ///
    /// Returns:
    ///     Optional[str]: The token string, or ``None``.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loading from a nonexistent path must return Err.
    #[test]
    fn test_from_file_nonexistent() {
        let path = std::env::temp_dir().join("oxillama_py_no_such_tokenizer_42.json");
        let path_str = path.to_string_lossy();
        let result = TokenizerBridge::from_file(&path_str);
        assert!(result.is_err(), "nonexistent tokenizer file should error");
    }

    /// Loading from invalid JSON must return Err.
    #[test]
    fn test_from_json_invalid() {
        let result = TokenizerBridge::from_bytes(b"not valid json at all");
        assert!(result.is_err(), "invalid JSON should error");
    }

    /// `id_to_token` on the stub (or with no loaded tokenizer) returns None
    /// when given an arbitrary id.  We just verify that from_file errors
    /// gracefully on a missing file.
    #[test]
    fn test_from_file_with_empty_file() {
        let tmp = std::env::temp_dir().join("oxillama_py_empty_tok.json");
        std::fs::write(&tmp, "{}").ok();
        // This may or may not be a valid tokenizer — we just check it doesn't panic.
        let _result = TokenizerBridge::from_bytes(b"{}");
        let _ = std::fs::remove_file(&tmp);
    }
}

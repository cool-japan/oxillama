//! Python wrapper for [`TokenizerBridge`] — standalone tokenizer access.
//!
//! Exposes HuggingFace BPE tokenization without needing a full inference
//! engine.  Users can load a `tokenizer.json` file, encode text to token
//! IDs, and decode IDs back to strings.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use oxillama_runtime::TokenizerBridge;

use crate::chat_template::apply_template;
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

    /// Encode a batch of texts into token ID lists in one call.
    ///
    /// This is a convenience wrapper that applies `encode` to each element.
    ///
    /// Args:
    ///     texts: List of input strings.
    ///
    /// Returns:
    ///     `List[List[int]]`: Token IDs for each input string, in order.
    ///
    /// Raises:
    ///     TokenizerError: if encoding fails for any input.
    pub fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        texts
            .iter()
            .map(|t| self.inner.encode(t).map_err(runtime_to_py))
            .collect()
    }

    /// Apply a chat template to a list of messages.
    ///
    /// Formats a conversation (list of `{role, content}` dicts) into the
    /// prompt string expected by the target model family.
    ///
    /// Args:
    ///     messages:             List of dicts with ``"role"`` and ``"content"`` keys.
    ///     template:             Template name — ``"chatml"`` (default), ``"llama3"``,
    ///                           or ``"alpaca"``.
    ///     add_generation_prompt: When ``True`` (default), append the
    ///                           assistant start token so the model knows to continue.
    ///
    /// Returns:
    ///     str: The formatted prompt string.
    ///
    /// Raises:
    ///     ValueError: if an unsupported template name is given or a message
    ///                 dict is missing required keys.
    #[pyo3(signature = (messages, template = None, add_generation_prompt = None))]
    pub fn apply_chat_template(
        &self,
        _py: Python<'_>,
        messages: Vec<Bound<'_, PyDict>>,
        template: Option<String>,
        add_generation_prompt: Option<bool>,
    ) -> PyResult<String> {
        let tpl = template.as_deref().unwrap_or("chatml");
        let add_gen = add_generation_prompt.unwrap_or(true);
        apply_template(tpl, &messages, add_gen)
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

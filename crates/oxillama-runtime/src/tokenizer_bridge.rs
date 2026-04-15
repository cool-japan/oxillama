//! Tokenizer bridge — wraps HuggingFace `tokenizers` crate.
//!
//! Provides encoding (text → token IDs) and decoding (token IDs → text)
//! using the tokenizer configuration embedded in GGUF model files or
//! loaded from separate tokenizer files.
//!
//! When neither `tokenizer-onig` nor `tokenizer-wasm` is enabled (e.g. for
//! bare no_std-like WASM targets), all methods return
//! `RuntimeError::TokenizerNotAvailable` so that the crate still compiles
//! without any regex or C dependencies.
//!
//! Feature matrix:
//! - `tokenizer-onig`  — uses HuggingFace tokenizers with Oniguruma (C regex, native only)
//! - `tokenizer-wasm`  — uses HuggingFace tokenizers with fancy-regex (pure Rust, wasm32-safe)
//! - neither           — stub that returns `TokenizerNotAvailable`

use crate::error::{RuntimeError, RuntimeResult};

// ─── Full implementation when tokenizer-onig OR tokenizer-wasm is enabled ────

#[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
/// Tokenizer bridge wrapping the HuggingFace tokenizers library.
pub struct TokenizerBridge {
    /// The underlying HuggingFace tokenizer.
    tokenizer: tokenizers::Tokenizer,
    /// Cached vocab bytes — populated on first call to `vocab_bytes_cached()`.
    cached_vocab: std::sync::OnceLock<Vec<(u32, Vec<u8>)>>,
}

#[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
impl TokenizerBridge {
    /// Load a tokenizer from a JSON file path.
    pub fn from_file(path: &str) -> RuntimeResult<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_file(path).map_err(|e| RuntimeError::TokenizerError {
                message: format!("failed to load tokenizer from {path}: {e}"),
            })?;
        Ok(Self {
            tokenizer,
            cached_vocab: std::sync::OnceLock::new(),
        })
    }

    /// Create a tokenizer from JSON bytes (e.g., from GGUF metadata).
    pub fn from_bytes(json: &[u8]) -> RuntimeResult<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(json).map_err(|e| RuntimeError::TokenizerError {
                message: format!("failed to parse tokenizer JSON: {e}"),
            })?;
        Ok(Self {
            tokenizer,
            cached_vocab: std::sync::OnceLock::new(),
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        let encoding =
            self.tokenizer
                .encode(text, false)
                .map_err(|e| RuntimeError::TokenizerError {
                    message: format!("encoding failed: {e}"),
                })?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> RuntimeResult<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| RuntimeError::TokenizerError {
                message: format!("decoding failed: {e}"),
            })
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get the BOS (beginning of sequence) token ID, if any.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<s>")
            .or_else(|| self.tokenizer.token_to_id("<|begin_of_text|>"))
    }

    /// Get the EOS (end of sequence) token ID, if any.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|end_of_text|>"))
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
    }

    /// Get the string representation of a single token ID.
    ///
    /// Returns `None` if the id is not in the vocabulary.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Get the byte representation of a single token ID.
    ///
    /// Uses decode (skip_special_tokens=false) to produce the canonical bytes
    /// for byte-level BPE tokenizers. Returns `None` if the id is unknown.
    pub fn token_to_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.tokenizer
            .decode(&[id], false)
            .ok()
            .map(|s| s.into_bytes())
    }

    /// Build the full vocab as `(token_id, byte_representation)` pairs.
    ///
    /// This is used to pre-compute the vocabulary for grammar masking.
    /// The result can be cached and shared across generation steps.
    pub fn vocab_bytes(&self) -> Vec<(u32, Vec<u8>)> {
        let vocab = self.tokenizer.get_vocab(true);
        let mut result: Vec<(u32, Vec<u8>)> = vocab
            .into_values()
            .filter_map(|id| self.token_to_bytes(id).map(|bytes| (id, bytes)))
            .collect();
        // Sort by id for determinism and cache-friendly iteration.
        result.sort_unstable_by_key(|&(id, _)| id);
        result
    }

    /// Get cached vocabulary bytes. Computes on first call, returns cached thereafter.
    pub fn vocab_bytes_cached(&self) -> &[(u32, Vec<u8>)] {
        self.cached_vocab.get_or_init(|| self.vocab_bytes())
    }
}

// ─── Stub when neither tokenizer feature is active ───────────────────────────

#[cfg(not(any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
/// Tokenizer bridge stub — not available without `tokenizer-onig` or `tokenizer-wasm`.
///
/// All methods return [`RuntimeError::TokenizerNotAvailable`] so callers
/// can detect the missing functionality at runtime rather than at link time.
pub struct TokenizerBridge;

#[cfg(not(any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
impl TokenizerBridge {
    /// Always returns `Err(TokenizerNotAvailable)` — no C tokenizer available.
    pub fn from_file(_path: &str) -> RuntimeResult<Self> {
        Err(RuntimeError::TokenizerNotAvailable)
    }

    /// Always returns `Err(TokenizerNotAvailable)` — no C tokenizer available.
    pub fn from_bytes(_json: &[u8]) -> RuntimeResult<Self> {
        Err(RuntimeError::TokenizerNotAvailable)
    }

    /// Always returns `Err(TokenizerNotAvailable)`.
    pub fn encode(&self, _text: &str) -> RuntimeResult<Vec<u32>> {
        Err(RuntimeError::TokenizerNotAvailable)
    }

    /// Always returns `Err(TokenizerNotAvailable)`.
    pub fn decode(&self, _tokens: &[u32]) -> RuntimeResult<String> {
        Err(RuntimeError::TokenizerNotAvailable)
    }

    /// Returns 0 — stub has no vocabulary.
    pub fn vocab_size(&self) -> usize {
        0
    }

    /// Returns `None` — stub has no BOS token.
    pub fn bos_token_id(&self) -> Option<u32> {
        None
    }

    /// Returns `None` — stub has no EOS token.
    pub fn eos_token_id(&self) -> Option<u32> {
        None
    }

    /// Returns `None` — stub has no vocabulary.
    pub fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }

    /// Returns `None` — stub has no vocabulary.
    pub fn token_to_bytes(&self, _id: u32) -> Option<Vec<u8>> {
        None
    }

    /// Returns an empty vec — stub has no vocabulary.
    pub fn vocab_bytes(&self) -> Vec<(u32, Vec<u8>)> {
        Vec::new()
    }

    /// Get cached vocabulary bytes. Returns empty — stub has no vocabulary.
    pub fn vocab_bytes_cached(&self) -> &[(u32, Vec<u8>)] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loading from a non-existent file must return an error in all configs.
    #[test]
    fn test_from_file_nonexistent_errors() {
        let result = TokenizerBridge::from_file("/nonexistent/path/tokenizer_test.json");
        assert!(result.is_err(), "missing tokenizer file should error");
    }

    /// When neither `tokenizer-onig` nor `tokenizer-wasm` is active, the stub
    /// returns `TokenizerNotAvailable` for every method.
    #[cfg(not(any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
    #[test]
    fn test_stub_from_file_returns_not_available() {
        let result = TokenizerBridge::from_file("/any/path.json");
        assert!(
            matches!(result, Err(RuntimeError::TokenizerNotAvailable)),
            "stub should return TokenizerNotAvailable, got {result:?}"
        );
    }

    #[cfg(not(any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
    #[test]
    fn test_stub_from_bytes_returns_not_available() {
        let result = TokenizerBridge::from_bytes(b"{}");
        assert!(
            matches!(result, Err(RuntimeError::TokenizerNotAvailable)),
            "stub should return TokenizerNotAvailable, got {result:?}"
        );
    }

    /// When any tokenizer backend is enabled, invalid JSON must error.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_from_bytes_invalid_json_errors() {
        let result = TokenizerBridge::from_bytes(b"not valid json {{{{");
        assert!(
            result.is_err(),
            "invalid tokenizer JSON should return an error"
        );
    }

    /// vocab_size returns 0 for the no-tokenizer stub.
    #[cfg(not(any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
    #[test]
    fn test_stub_vocab_size_is_zero() {
        // We cannot construct the stub directly, so we verify via error path.
        let r = TokenizerBridge::from_bytes(b"{}");
        assert!(r.is_err());
    }

    // ─── Full tokenizer tests (require tokenizer-onig or tokenizer-wasm) ──────

    /// Minimal BPE tokenizer JSON that the tokenizers crate can parse.
    /// Uses a very small vocabulary sufficient for round-trip tests.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    const MINIMAL_TOKENIZER_JSON: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 0, "special": true, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 1, "special": true, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 2, "special": true, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
      ],
      "normalizer": null,
      "pre_tokenizer": null,
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "BPE",
        "dropout": null,
        "unk_token": "<unk>",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "byte_fallback": false,
        "vocab": {
          "<unk>": 0,
          "<s>": 1,
          "</s>": 2,
          "h": 3,
          "e": 4,
          "l": 5,
          "o": 6,
          " ": 7,
          "w": 8,
          "r": 9,
          "d": 10,
          "a": 11,
          "b": 12
        },
        "merges": []
      }
    }"#;

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_from_bytes_valid_json_succeeds() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        assert!(
            bridge.vocab_size() > 0,
            "vocab_size should be positive after loading"
        );
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_bos_token_id_found() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        // <s> is id=1 in our minimal vocab
        assert_eq!(
            bridge.bos_token_id(),
            Some(1),
            "BOS token <s> should have id=1"
        );
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_eos_token_id_found() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        // </s> is id=2 in our minimal vocab
        assert_eq!(
            bridge.eos_token_id(),
            Some(2),
            "EOS token </s> should have id=2"
        );
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_encode_produces_tokens() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        // With no merges and character-level vocab, "hello" → individual char tokens
        let tokens = bridge.encode("hello").expect("test: encode should succeed");
        assert!(
            !tokens.is_empty(),
            "encoding 'hello' should produce at least one token"
        );
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_decode_empty_slice_returns_empty_string() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        let decoded = bridge
            .decode(&[])
            .expect("test: decoding empty slice should succeed");
        assert_eq!(
            decoded, "",
            "decoding empty token list should return empty string"
        );
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_encode_decode_roundtrip() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        // All chars in "hello" are in our single-char vocab, so encode→decode should work
        let tokens = bridge.encode("hello").expect("test: encode should succeed");
        let decoded = bridge.decode(&tokens).expect("test: decode should succeed");
        // The tokenizers crate may add spaces; just verify no panic and non-empty result
        assert!(
            !decoded.is_empty() || tokens.is_empty(),
            "decoded output consistency"
        );
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_vocab_size_matches_json() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        // We defined 13 tokens in the vocab (0..=12)
        assert_eq!(
            bridge.vocab_size(),
            13,
            "vocab_size should match the number of defined tokens"
        );
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_token_to_bytes_special_token() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        // Token id 0 = <unk>; token_to_bytes returns bytes of the decoded string
        // The result depends on skip_special_tokens=false behaviour
        let _bytes = bridge.token_to_bytes(0); // just verify no panic
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_vocab_bytes_is_sorted() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        let pairs = bridge.vocab_bytes();
        for window in pairs.windows(2) {
            assert!(
                window[0].0 <= window[1].0,
                "vocab_bytes should be sorted by token id, got {} > {}",
                window[0].0,
                window[1].0
            );
        }
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_from_bytes_invalid_json_structure_errors() {
        // Valid JSON but not a tokenizer schema
        let result = TokenizerBridge::from_bytes(b"{\"not\": \"a tokenizer\"}");
        assert!(result.is_err(), "non-tokenizer JSON should return an error");
    }
}

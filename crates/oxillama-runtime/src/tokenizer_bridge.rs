// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer bridge — one interface over two vocabulary sources.
//!
//! | backend | source | availability |
//! |---|---|---|
//! | [`GgufVocab`] | the vocabulary embedded in the GGUF file itself | always |
//! | HuggingFace `tokenizers` | a `tokenizer.json` file or byte buffer | `tokenizer-onig` / `tokenizer-wasm` |
//!
//! The GGUF backend is the primary one: every GGUF converted by
//! `convert_hf_to_gguf.py` carries `tokenizer.ggml.*` metadata, so a stock
//! download runs without any sidecar file.  The HuggingFace backend remains for
//! callers that hand over a `tokenizer.json` explicitly (and for
//! `load_model_from_bytes`, where there is no GGUF metadata to read on WASM).
//!
//! # Special tokens
//!
//! Whichever backend is in use, BOS/EOS/EOT and the end-of-generation set are
//! taken from GGUF metadata when it is available ([`SpecialTokens`]).  Probing
//! the vocabulary by name — the previous approach — is actively unsafe for
//! byte-level BPE: Qwen3's vocabulary contains the literal string `</s>` as an
//! ordinary merged token at id 128247, so a name probe silently elects a normal
//! word as EOS and generation never terminates.
//!
//! Feature matrix:
//! - `tokenizer-onig`  — HuggingFace tokenizers with Oniguruma (C regex, native only)
//! - `tokenizer-wasm`  — HuggingFace tokenizers with fancy-regex (pure Rust, wasm32-safe)
//! - neither           — GGUF-embedded vocabularies only; `from_file`/`from_bytes` error

use std::collections::HashSet;
use std::sync::OnceLock;

use oxillama_gguf::{MetadataStore, MetadataValue};

use crate::error::{RuntimeError, RuntimeResult};
use crate::gguf_vocab::{self, GgufVocab};

/// End-of-generation token names, kept in sync with llama.cpp's list.
///
/// Used only as a fallback for backends without GGUF metadata; both backends
/// require a name match to also be a *special* token, so a lookalike ordinary
/// token can never be selected.
const EOG_NAMES: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
];

/// Candidate BOS names, tried only when metadata is silent.
const BOS_NAMES: &[&str] = &["<s>", "<|begin_of_text|>", "<|startoftext|>"];

/// Candidate end-of-turn names, tried only when metadata is silent.
const EOT_NAMES: &[&str] = &["<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>"];

/// Authoritative special-token information read from GGUF metadata.
///
/// This overlay is applied regardless of which backend supplies the vocabulary,
/// so a model whose GGUF names `<|im_end|>` as EOS resolves correctly even when
/// a `tokenizer.json` sidecar happens to be loaded.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token.
    pub bos: Option<u32>,
    /// End-of-sequence token.
    pub eos: Option<u32>,
    /// End-of-turn token (`<|eot_id|>`, `<|im_end|>`, …).
    pub eot: Option<u32>,
    /// Unknown token.
    pub unk: Option<u32>,
    /// Padding token.
    pub pad: Option<u32>,
    /// Separator token.
    pub sep: Option<u32>,
    /// Whether the model wants BOS prepended when encoding a prompt.
    pub add_bos: bool,
    /// Whether the model wants EOS appended when encoding a prompt.
    pub add_eos: bool,
    /// Every token that ends generation.
    ///
    /// llama.cpp tracks a *set* rather than a single id because instruct models
    /// routinely stop on a token that is not `eos_token_id` — Llama-3-Instruct
    /// emits `<|eot_id|>` while its EOS is `<|end_of_text|>`.
    pub eog: HashSet<u32>,
}

impl SpecialTokens {
    /// Read the `tokenizer.ggml.*` special-token keys.
    ///
    /// Returns an all-`None` value when the metadata carries none of them,
    /// which callers should treat as "fall back to backend probing".
    pub fn from_metadata(metadata: &MetadataStore) -> Self {
        let read = |key: &str| -> Option<u32> {
            metadata.get(key).and_then(|v| {
                v.as_u32()
                    .or_else(|| v.as_i32().and_then(|i| u32::try_from(i).ok()))
            })
        };
        let read_bool = |key: &str| metadata.get(key).and_then(MetadataValue::as_bool);
        let bos = read("tokenizer.ggml.bos_token_id");
        let eos = read("tokenizer.ggml.eos_token_id");
        let eot = read("tokenizer.ggml.eot_token_id");
        let eom = read("tokenizer.ggml.eom_token_id");
        let mut eog: HashSet<u32> = HashSet::new();
        eog.extend(eos);
        eog.extend(eot);
        eog.extend(eom);
        Self {
            bos,
            eos,
            eot,
            unk: read("tokenizer.ggml.unknown_token_id"),
            pad: read("tokenizer.ggml.padding_token_id"),
            sep: read("tokenizer.ggml.separator_token_id"),
            add_bos: read_bool("tokenizer.ggml.add_bos_token").unwrap_or(false),
            add_eos: read_bool("tokenizer.ggml.add_eos_token").unwrap_or(false),
            eog,
        }
    }

    /// `true` when nothing useful was found.
    pub fn is_empty(&self) -> bool {
        self.bos.is_none() && self.eos.is_none() && self.eot.is_none() && self.eog.is_empty()
    }
}

/// The vocabulary source backing a [`TokenizerBridge`].
enum Backend {
    /// Vocabulary read straight out of GGUF metadata.
    Gguf(Box<GgufVocab>),
    /// A HuggingFace `tokenizer.json`.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    Hf(Box<tokenizers::Tokenizer>),
}

/// Encoding / decoding front-end used by the inference engine.
pub struct TokenizerBridge {
    backend: Backend,
    special: SpecialTokens,
    cached_vocab: OnceLock<Vec<(u32, Vec<u8>)>>,
}

impl core::fmt::Debug for TokenizerBridge {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = match &self.backend {
            Backend::Gguf(_) => "gguf",
            #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
            Backend::Hf(_) => "huggingface",
        };
        f.debug_struct("TokenizerBridge")
            .field("backend", &kind)
            .field("vocab_size", &self.vocab_size())
            .field("bos", &self.bos_token_id())
            .field("eos", &self.eos_token_id())
            .field("n_eog", &self.eog_token_ids().len())
            .finish()
    }
}

impl TokenizerBridge {
    // ── constructors ────────────────────────────────────────────────────────

    /// Build a tokenizer from the vocabulary embedded in GGUF metadata.
    ///
    /// This is the path that lets a stock HuggingFace GGUF run with no sidecar
    /// file at all.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::TokenizerError`] when the metadata carries no
    /// `tokenizer.ggml.tokens` array or names an unsupported tokenizer model.
    pub fn from_gguf_metadata(metadata: &MetadataStore) -> RuntimeResult<Self> {
        let vocab = GgufVocab::from_metadata(metadata)?;
        let special = SpecialTokens {
            bos: vocab.bos_id(),
            eos: vocab.eos_id(),
            eot: vocab.eot_id(),
            unk: vocab.unk_id(),
            pad: vocab.pad_id(),
            sep: vocab.sep_id(),
            add_bos: vocab.add_bos(),
            add_eos: vocab.add_eos(),
            eog: vocab.eog_ids().clone(),
        };
        Ok(Self {
            backend: Backend::Gguf(Box::new(vocab)),
            special,
            cached_vocab: OnceLock::new(),
        })
    }

    /// `true` when `metadata` carries a vocabulary [`Self::from_gguf_metadata`]
    /// can use.
    pub fn metadata_has_vocab(metadata: &MetadataStore) -> bool {
        gguf_vocab::has_embedded_vocab(metadata)
    }

    /// Overlay authoritative special-token ids from GGUF metadata.
    ///
    /// Applied on top of *any* backend so that a `tokenizer.json` sidecar can
    /// never override what the model file itself declares.  Fields absent from
    /// the metadata are left untouched.
    pub fn apply_gguf_specials(&mut self, metadata: &MetadataStore) {
        let from_meta = SpecialTokens::from_metadata(metadata);
        if from_meta.bos.is_some() {
            self.special.bos = from_meta.bos;
        }
        if from_meta.eos.is_some() {
            self.special.eos = from_meta.eos;
        }
        if from_meta.eot.is_some() {
            self.special.eot = from_meta.eot;
        }
        if from_meta.unk.is_some() {
            self.special.unk = from_meta.unk;
        }
        if from_meta.pad.is_some() {
            self.special.pad = from_meta.pad;
        }
        if from_meta.sep.is_some() {
            self.special.sep = from_meta.sep;
        }
        if metadata.get("tokenizer.ggml.add_bos_token").is_some() {
            self.special.add_bos = from_meta.add_bos;
        }
        if metadata.get("tokenizer.ggml.add_eos_token").is_some() {
            self.special.add_eos = from_meta.add_eos;
        }
        self.special.eog.extend(from_meta.eog.iter().copied());
        // Round out the EOG set from the backend's own vocabulary.
        for id in self.probe_eog_ids() {
            self.special.eog.insert(id);
        }
        self.special.eog.extend(self.special.eos);
        self.special.eot = self.special.eot.or_else(|| self.probe_eot_id());
        self.special.eog.extend(self.special.eot);
        if self.special.bos.is_none() {
            self.special.bos = self.probe_bos_id();
        }
        if self.special.eos.is_none() {
            self.special.eos = self
                .special
                .eot
                .or_else(|| self.special.eog.iter().min().copied());
        }
    }

    /// Load a tokenizer from a JSON file path.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::TokenizerNotAvailable`] when neither tokenizer
    /// feature is compiled in, or [`RuntimeError::TokenizerError`] when the file
    /// cannot be parsed.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    pub fn from_file(path: &str) -> RuntimeResult<Self> {
        Self::from_path(std::path::Path::new(path))
    }

    /// Always returns `Err(TokenizerNotAvailable)` — no HuggingFace backend.
    #[cfg(not(any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
    pub fn from_file(_path: &str) -> RuntimeResult<Self> {
        Err(RuntimeError::TokenizerNotAvailable)
    }

    /// Load a tokenizer from a filesystem path.
    ///
    /// Prefer this over [`Self::from_file`]: it accepts any `Path`, so a model
    /// living under a directory whose name is not valid UTF-8 still loads
    /// instead of silently falling back to a relative path.
    ///
    /// # Errors
    ///
    /// As [`Self::from_file`].
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    pub fn from_path(path: &std::path::Path) -> RuntimeResult<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_file(path).map_err(|e| RuntimeError::TokenizerError {
                message: format!("failed to load tokenizer from {}: {e}", path.display()),
            })?;
        Ok(Self::from_hf(tokenizer))
    }

    /// Always returns `Err(TokenizerNotAvailable)` — no HuggingFace backend.
    #[cfg(not(any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
    pub fn from_path(_path: &std::path::Path) -> RuntimeResult<Self> {
        Err(RuntimeError::TokenizerNotAvailable)
    }

    /// Create a tokenizer from `tokenizer.json` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::TokenizerNotAvailable`] when neither tokenizer
    /// feature is compiled in, or [`RuntimeError::TokenizerError`] on a parse
    /// failure.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    pub fn from_bytes(json: &[u8]) -> RuntimeResult<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(json).map_err(|e| RuntimeError::TokenizerError {
                message: format!("failed to parse tokenizer JSON: {e}"),
            })?;
        Ok(Self::from_hf(tokenizer))
    }

    /// Always returns `Err(TokenizerNotAvailable)` — no HuggingFace backend.
    #[cfg(not(any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
    pub fn from_bytes(_json: &[u8]) -> RuntimeResult<Self> {
        Err(RuntimeError::TokenizerNotAvailable)
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    fn from_hf(tokenizer: tokenizers::Tokenizer) -> Self {
        let mut bridge = Self {
            backend: Backend::Hf(Box::new(tokenizer)),
            special: SpecialTokens::default(),
            cached_vocab: OnceLock::new(),
        };
        bridge.special.bos = bridge.probe_bos_id();
        bridge.special.eot = bridge.probe_eot_id();
        bridge.special.eog = bridge.probe_eog_ids();
        bridge.special.eos = bridge
            .special
            .eot
            .or_else(|| bridge.special.eog.iter().min().copied());
        bridge
    }

    // ── special token resolution ────────────────────────────────────────────

    /// The BOS token id, if the model has one.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.special.bos
    }

    /// The EOS token id, if the model has one.
    ///
    /// Sourced from `tokenizer.ggml.eos_token_id` when available.  A `None`
    /// here means generation cannot stop early, so callers should treat it as a
    /// configuration problem rather than a normal state.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.special.eos
    }

    /// The end-of-turn token id (`<|eot_id|>`, `<|im_end|>`, …), if present.
    pub fn eot_token_id(&self) -> Option<u32> {
        self.special.eot
    }

    /// The unknown token id, if present.
    pub fn unk_token_id(&self) -> Option<u32> {
        self.special.unk
    }

    /// The padding token id, if present.
    pub fn pad_token_id(&self) -> Option<u32> {
        self.special.pad
    }

    /// Every token that terminates generation.
    pub fn eog_token_ids(&self) -> &HashSet<u32> {
        &self.special.eog
    }

    /// `true` when `token` terminates generation.
    ///
    /// This is a set membership test rather than an equality check against a
    /// single EOS id, because instruct-tuned models routinely stop on a token
    /// that is not `eos_token_id`.
    pub fn is_eog(&self, token: u32) -> bool {
        self.special.eog.contains(&token)
    }

    /// Whether `encode` prepends BOS for this model.
    pub fn add_bos(&self) -> bool {
        self.special.add_bos
    }

    /// The resolved special tokens.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    /// Find a special token by name, refusing ordinary vocabulary entries.
    fn special_id_by_name(&self, name: &str) -> Option<u32> {
        match &self.backend {
            Backend::Gguf(vocab) => vocab.token_to_id(name).filter(|&id| {
                matches!(
                    vocab.token_type(id),
                    Some(gguf_vocab::TokenType::Control | gguf_vocab::TokenType::UserDefined)
                )
            }),
            #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
            Backend::Hf(tokenizer) => tokenizer
                .get_added_tokens_decoder()
                .into_iter()
                .find(|(_, added)| added.special && added.content == name)
                .map(|(id, _)| id),
        }
    }

    fn probe_bos_id(&self) -> Option<u32> {
        BOS_NAMES
            .iter()
            .find_map(|name| self.special_id_by_name(name))
    }

    fn probe_eot_id(&self) -> Option<u32> {
        EOT_NAMES
            .iter()
            .find_map(|name| self.special_id_by_name(name))
    }

    fn probe_eog_ids(&self) -> HashSet<u32> {
        EOG_NAMES
            .iter()
            .filter_map(|name| self.special_id_by_name(name))
            .collect()
    }

    // ── encoding ────────────────────────────────────────────────────────────

    /// Encode text to token IDs, applying the model's own BOS/EOS policy.
    ///
    /// Special tokens present in `text` (`<|im_start|>`, `<|eot_id|>`, …) are
    /// recognised rather than split, which is what chat templates require.  Use
    /// [`Self::encode_with`] for explicit control.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::TokenizerNotAvailable`] when no backend is
    /// compiled in, or [`RuntimeError::TokenizerError`] on an encoding failure.
    pub fn encode(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        self.encode_with(text, true, true)
    }

    /// Encode text without adding BOS/EOS and without parsing special tokens.
    ///
    /// This reproduces the pre-0.1.4 behaviour of `encode` and is what
    /// llama.cpp's tokenizer conformance tests use.
    ///
    /// # Errors
    ///
    /// As [`Self::encode`].
    pub fn encode_raw(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        self.encode_with(text, false, false)
    }

    /// Encode text with explicit control over special-token handling.
    ///
    /// * `add_special` — honour `tokenizer.ggml.add_bos_token` /
    ///   `add_eos_token`.
    /// * `parse_special` — recognise control tokens appearing in `text`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::TokenizerError`] when the backend fails.
    pub fn encode_with(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> RuntimeResult<Vec<u32>> {
        match &self.backend {
            Backend::Gguf(vocab) => Ok(vocab.tokenize(text, add_special, parse_special)),
            #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
            Backend::Hf(tokenizer) => {
                let encoding = tokenizer.encode(text, add_special).map_err(|e| {
                    RuntimeError::TokenizerError {
                        message: format!("encoding failed: {e}"),
                    }
                })?;
                let mut ids = encoding.get_ids().to_vec();
                // A `tokenizer.json` post-processor may not know about the GGUF
                // `add_bos_token` flag; enforce it here so both backends agree.
                if add_special && self.special.add_bos {
                    if let Some(bos) = self.special.bos {
                        if ids.first() != Some(&bos) {
                            ids.insert(0, bos);
                        }
                    }
                }
                Ok(ids)
            }
        }
    }

    // ── decoding ────────────────────────────────────────────────────────────

    /// Decode token IDs back to text, dropping special tokens.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::TokenizerError`] when the backend fails.
    pub fn decode(&self, tokens: &[u32]) -> RuntimeResult<String> {
        self.decode_with(tokens, true)
    }

    /// Decode token IDs, choosing whether special tokens are rendered.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::TokenizerError`] when the backend fails.
    pub fn decode_with(&self, tokens: &[u32], skip_special: bool) -> RuntimeResult<String> {
        match &self.backend {
            Backend::Gguf(vocab) => Ok(vocab.detokenize(tokens, skip_special)),
            #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
            Backend::Hf(tokenizer) => {
                tokenizer
                    .decode(tokens, skip_special)
                    .map_err(|e| RuntimeError::TokenizerError {
                        message: format!("decoding failed: {e}"),
                    })
            }
        }
    }

    /// Decode token IDs to their exact bytes.
    ///
    /// Unlike [`Self::decode`] this never substitutes `U+FFFD`, so a token that
    /// carries only part of a multi-byte character can be stitched together with
    /// the next one — see [`crate::stream_decode::Utf8StreamDecoder`].
    pub fn decode_bytes(&self, tokens: &[u32], skip_special: bool) -> Vec<u8> {
        match &self.backend {
            Backend::Gguf(vocab) => vocab.detokenize_bytes(tokens, skip_special),
            #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
            Backend::Hf(tokenizer) => {
                // `token_to_bytes` renders special tokens, so `skip_special`
                // has to be honoured here rather than inside the decoder.
                let specials: HashSet<u32> = if skip_special {
                    tokenizer
                        .get_added_tokens_decoder()
                        .into_iter()
                        .filter(|(_, added)| added.special)
                        .map(|(id, _)| id)
                        .collect()
                } else {
                    HashSet::new()
                };
                let mut out = Vec::new();
                for &id in tokens {
                    if specials.contains(&id) {
                        continue;
                    }
                    if let Some(bytes) = self.token_to_bytes(id) {
                        out.extend_from_slice(&bytes);
                    }
                }
                out
            }
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match &self.backend {
            Backend::Gguf(vocab) => vocab.len(),
            #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
            Backend::Hf(tokenizer) => tokenizer.get_vocab_size(true),
        }
    }

    /// Get the string representation of a single token ID.
    ///
    /// Returns `None` if the id is not in the vocabulary.  For byte-level BPE
    /// this is the *raw* vocabulary entry (`Ġthe`), not the decoded text.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        match &self.backend {
            Backend::Gguf(vocab) => vocab.id_to_token(id).map(str::to_string),
            #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
            Backend::Hf(tokenizer) => tokenizer.id_to_token(id),
        }
    }

    /// Get the byte representation of a single token ID.
    ///
    /// The GGUF backend returns the exact bytes.  The HuggingFace backend has to
    /// route through its `String`-returning decoder, so a token holding a
    /// partial UTF-8 sequence comes back as `U+FFFD` — one more reason the GGUF
    /// backend is preferred.
    pub fn token_to_bytes(&self, id: u32) -> Option<Vec<u8>> {
        match &self.backend {
            Backend::Gguf(vocab) => vocab.token_bytes(id).map(<[u8]>::to_vec),
            #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
            Backend::Hf(tokenizer) => tokenizer
                .decode(&[id], false)
                .ok()
                .map(std::string::String::into_bytes),
        }
    }

    /// Build the full vocab as `(token_id, byte_representation)` pairs.
    ///
    /// Used to pre-compute the vocabulary for grammar masking.
    pub fn vocab_bytes(&self) -> Vec<(u32, Vec<u8>)> {
        match &self.backend {
            Backend::Gguf(vocab) => vocab.vocab_bytes(),
            #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
            Backend::Hf(tokenizer) => {
                let vocab = tokenizer.get_vocab(true);
                let mut result: Vec<(u32, Vec<u8>)> = vocab
                    .into_values()
                    .filter_map(|id| self.token_to_bytes(id).map(|bytes| (id, bytes)))
                    .collect();
                result.sort_unstable_by_key(|&(id, _)| id);
                result
            }
        }
    }

    /// Get cached vocabulary bytes. Computes on first call, returns cached thereafter.
    pub fn vocab_bytes_cached(&self) -> &[(u32, Vec<u8>)] {
        self.cached_vocab.get_or_init(|| self.vocab_bytes())
    }

    /// `true` when this bridge is backed by the GGUF-embedded vocabulary.
    pub fn is_gguf_backed(&self) -> bool {
        matches!(self.backend, Backend::Gguf(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny SentencePiece vocabulary expressed as GGUF metadata.
    fn spm_metadata() -> MetadataStore {
        let mut md = MetadataStore::new();
        md.insert(
            "tokenizer.ggml.model".to_string(),
            MetadataValue::String("llama".to_string()),
        );
        let tokens = [
            "<unk>", "<s>", "</s>", "<0x0A>", "<0x20>", "▁", "▁h", "e", "l", "o", "▁he", "llo",
            "▁hello", "▁world", "w", "r", "d",
        ];
        md.insert(
            "tokenizer.ggml.tokens".to_string(),
            MetadataValue::Array(
                tokens
                    .iter()
                    .map(|t| MetadataValue::String((*t).to_string()))
                    .collect(),
            ),
        );
        let scores = [
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -1.5, -1.5, -8.0,
            -8.0, -8.0,
        ];
        md.insert(
            "tokenizer.ggml.scores".to_string(),
            MetadataValue::Array(scores.iter().map(|s| MetadataValue::Float32(*s)).collect()),
        );
        let types = [2, 3, 3, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        md.insert(
            "tokenizer.ggml.token_type".to_string(),
            MetadataValue::Array(types.iter().map(|t| MetadataValue::Int32(*t)).collect()),
        );
        md.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            MetadataValue::Uint32(1),
        );
        md.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            MetadataValue::Uint32(2),
        );
        md.insert(
            "tokenizer.ggml.add_bos_token".to_string(),
            MetadataValue::Bool(true),
        );
        md
    }

    #[test]
    fn gguf_backend_loads_without_any_sidecar() {
        let bridge = TokenizerBridge::from_gguf_metadata(&spm_metadata())
            .expect("test: GGUF vocabulary must load");
        assert!(bridge.is_gguf_backed());
        assert_eq!(bridge.vocab_size(), 17);
    }

    #[test]
    fn gguf_backend_reads_eos_from_metadata() {
        let bridge = TokenizerBridge::from_gguf_metadata(&spm_metadata())
            .expect("test: GGUF vocabulary must load");
        assert_eq!(bridge.eos_token_id(), Some(2));
        assert_eq!(bridge.bos_token_id(), Some(1));
        assert!(bridge.is_eog(2), "EOS must be in the end-of-generation set");
    }

    #[test]
    fn gguf_encode_prepends_bos_when_model_asks() {
        let bridge = TokenizerBridge::from_gguf_metadata(&spm_metadata())
            .expect("test: GGUF vocabulary must load");
        let ids = bridge.encode("hello").expect("test: encode must succeed");
        assert_eq!(ids.first(), Some(&1), "BOS must be prepended, got {ids:?}");
        let raw = bridge
            .encode_raw("hello")
            .expect("test: encode_raw must succeed");
        assert_ne!(raw.first(), Some(&1), "encode_raw must not add BOS");
    }

    #[test]
    fn gguf_roundtrip_decodes_exact_text() {
        let bridge = TokenizerBridge::from_gguf_metadata(&spm_metadata())
            .expect("test: GGUF vocabulary must load");
        let ids = bridge
            .encode_raw("hello world")
            .expect("test: encode must succeed");
        let text = bridge.decode(&ids).expect("test: decode must succeed");
        assert_eq!(text.trim_start(), "hello world");
    }

    #[test]
    fn gguf_byte_fallback_covers_unknown_characters() {
        let bridge = TokenizerBridge::from_gguf_metadata(&spm_metadata())
            .expect("test: GGUF vocabulary must load");
        // '\n' is only reachable through the <0x0A> byte-fallback token.
        let ids = bridge.encode_raw("\n").expect("test: encode must succeed");
        assert!(ids.contains(&3), "byte fallback must be used, got {ids:?}");
        let text = bridge.decode(&ids).expect("test: decode must succeed");
        assert!(text.contains('\n'));
    }

    #[test]
    fn token_to_bytes_is_exact_for_gguf_backend() {
        let bridge = TokenizerBridge::from_gguf_metadata(&spm_metadata())
            .expect("test: GGUF vocabulary must load");
        assert_eq!(bridge.token_to_bytes(3), Some(vec![b'\n']));
        assert_eq!(bridge.token_to_bytes(5), Some(vec![b' ']));
    }

    #[test]
    fn missing_vocabulary_is_an_error() {
        let md = MetadataStore::new();
        assert!(!TokenizerBridge::metadata_has_vocab(&md));
        assert!(TokenizerBridge::from_gguf_metadata(&md).is_err());
    }

    #[test]
    fn unsupported_tokenizer_model_is_an_error() {
        let mut md = spm_metadata();
        md.insert(
            "tokenizer.ggml.model".to_string(),
            MetadataValue::String("rwkv".to_string()),
        );
        let err = TokenizerBridge::from_gguf_metadata(&md)
            .expect_err("test: unsupported model must error");
        assert!(format!("{err}").contains("rwkv"), "got {err}");
    }

    /// Loading from a non-existent file must return an error in all configs.
    #[test]
    fn from_file_nonexistent_errors() {
        let result = TokenizerBridge::from_file("/nonexistent/path/tokenizer_test.json");
        assert!(result.is_err(), "missing tokenizer file should error");
    }

    #[cfg(not(any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
    #[test]
    fn stub_from_bytes_returns_not_available() {
        let result = TokenizerBridge::from_bytes(b"{}");
        assert!(
            matches!(result, Err(RuntimeError::TokenizerNotAvailable)),
            "stub should return TokenizerNotAvailable"
        );
    }

    // ─── HuggingFace backend tests ───────────────────────────────────────────

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
          "<unk>": 0, "<s>": 1, "</s>": 2,
          "h": 3, "e": 4, "l": 5, "o": 6, " ": 7,
          "w": 8, "r": 9, "d": 10, "a": 11, "b": 12
        },
        "merges": []
      }
    }"#;

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn hf_from_bytes_valid_json_succeeds() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        assert!(!bridge.is_gguf_backed());
        assert_eq!(bridge.vocab_size(), 13);
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn hf_bos_and_eos_come_from_added_tokens() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        assert_eq!(bridge.bos_token_id(), Some(1));
        assert_eq!(bridge.eos_token_id(), Some(2));
    }

    /// A byte-level BPE vocabulary that contains the literal text `</s>` as an
    /// *ordinary* token must not have it elected as EOS.  This is precisely the
    /// Qwen3 failure mode (id 128247).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn hf_normal_token_named_like_eos_is_not_elected() {
        let json = r#"{
          "version": "1.0", "truncation": null, "padding": null,
          "added_tokens": [
            {"id": 0, "special": true, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
          ],
          "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
          "model": {
            "type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": false, "byte_fallback": false,
            "vocab": {"<|im_end|>": 0, "a": 1, "</s>": 2},
            "merges": []
          }
        }"#;
        let bridge = TokenizerBridge::from_bytes(json.as_bytes()).expect("test: JSON should parse");
        assert_eq!(
            bridge.eos_token_id(),
            Some(0),
            "the control token <|im_end|> must win over the ordinary token </s>"
        );
        assert!(
            !bridge.is_eog(2),
            "ordinary token 2 must not end generation"
        );
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn hf_from_bytes_invalid_json_errors() {
        assert!(TokenizerBridge::from_bytes(b"not valid json {{{{").is_err());
        assert!(TokenizerBridge::from_bytes(b"{\"not\": \"a tokenizer\"}").is_err());
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn hf_decode_empty_slice_returns_empty_string() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        assert_eq!(bridge.decode(&[]).expect("test: decode should succeed"), "");
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn hf_vocab_bytes_is_sorted() {
        let bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: valid tokenizer JSON should parse");
        let pairs = bridge.vocab_bytes();
        for window in pairs.windows(2) {
            assert!(window[0].0 <= window[1].0, "vocab_bytes must be id-sorted");
        }
    }

    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn gguf_specials_override_sidecar_resolution() {
        let mut bridge = TokenizerBridge::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
            .expect("test: JSON should parse");
        let mut md = MetadataStore::new();
        md.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            MetadataValue::Uint32(11),
        );
        bridge.apply_gguf_specials(&md);
        assert_eq!(bridge.eos_token_id(), Some(11));
        assert!(bridge.is_eog(11));
    }
}

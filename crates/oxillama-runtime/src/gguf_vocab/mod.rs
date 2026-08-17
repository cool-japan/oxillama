// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer built from the vocabulary embedded in a GGUF file.
//!
//! Every GGUF produced by `convert_hf_to_gguf.py` carries its complete
//! vocabulary in metadata:
//!
//! | key | meaning |
//! |---|---|
//! | `tokenizer.ggml.model` | `llama` (SentencePiece), `gpt2` (byte-level BPE), `bert` (WordPiece), … |
//! | `tokenizer.ggml.pre` | pre-tokenizer family for BPE vocabularies |
//! | `tokenizer.ggml.tokens` | the vocabulary itself |
//! | `tokenizer.ggml.scores` | SentencePiece merge scores |
//! | `tokenizer.ggml.token_type` | 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte |
//! | `tokenizer.ggml.merges` | BPE merge rules, highest priority first |
//! | `tokenizer.ggml.*_token_id` | BOS / EOS / EOT / UNK / SEP / PAD / MASK |
//! | `tokenizer.ggml.add_bos_token` | whether BOS is prepended by `encode` |
//!
//! Before this module existed nothing in the workspace read any of these keys,
//! so a stock HuggingFace GGUF could not be run at all — inference required a
//! `tokenizer.json` sidecar that most GGUF downloads do not ship.
//!
//! # Why a native implementation
//!
//! The algorithms are ported from llama.cpp's `src/llama-vocab.cpp` rather than
//! rebuilt on top of the `tokenizers` crate, for three reasons:
//!
//! 1. **SentencePiece.** llama.cpp does *not* run Unigram/Viterbi decoding over
//!    the scores.  It runs a greedy highest-score bigram merge
//!    (`llm_tokenizer_spm_session`) with a `resegment` fallback, which is what
//!    LLaMA's SentencePiece-BPE model was trained with.  Feeding the same
//!    scores to a Unigram implementation yields different token IDs.
//! 2. **Byte-exact detokenisation.** `tokenizers` decodes through
//!    `String::from_utf8_lossy`, so a token holding the first two bytes of a
//!    three-byte CJK character comes back as `U+FFFD` and the information
//!    needed to reassemble it is gone.  Streaming generation needs the raw
//!    bytes ([`crate::stream_decode`]), and so does grammar masking.
//! 3. **One copy of the vocabulary.** A 128 k-entry vocabulary plus a 280 k-entry
//!    merge table is not something to hold twice in an inference process.
//!
//! [`PreTokenizer`] uses `fancy-regex` (pure Rust) for the pre-tokenizer split
//! patterns, which are transcribed verbatim from llama.cpp.

mod bpe;
pub mod byte_level;
pub mod pretok;
mod spm;
mod symbols;
mod wpm;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use oxillama_gguf::{MetadataStore, MetadataValue};

use crate::error::{RuntimeError, RuntimeResult};
pub use pretok::{PreTokenizer, PreType};

/// GGUF metadata key holding the vocabulary.
pub const KEY_TOKENS: &str = "tokenizer.ggml.tokens";
/// GGUF metadata key naming the tokenizer algorithm.
pub const KEY_MODEL: &str = "tokenizer.ggml.model";
/// GGUF metadata key naming the BPE pre-tokenizer family.
pub const KEY_PRE: &str = "tokenizer.ggml.pre";
/// GGUF metadata key holding SentencePiece scores.
pub const KEY_SCORES: &str = "tokenizer.ggml.scores";
/// GGUF metadata key holding per-token type codes.
pub const KEY_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
/// GGUF metadata key holding BPE merge rules.
pub const KEY_MERGES: &str = "tokenizer.ggml.merges";

/// The SentencePiece space marker (`U+2581 LOWER ONE EIGHTH BLOCK`).
pub const SPM_SPACE: char = '\u{2581}';

/// Which tokenization algorithm a GGUF vocabulary uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabType {
    /// SentencePiece (`tokenizer.ggml.model` = `llama`).
    Spm,
    /// Byte-level BPE (`gpt2`, `qwen`, `qwen2`, `deepseek`, `bloom`, …).
    Bpe,
    /// WordPiece (`bert`).
    Wpm,
}

impl VocabType {
    /// Resolve the `tokenizer.ggml.model` string.
    ///
    /// The accepted names cover everything the conversion scripts emit today,
    /// including the values this workspace's own GGUF test fixtures fabricate
    /// (`qwen`, `phi`, `deepseek`, `bloom`, `grok`, `dbrx`, `mamba2`).
    pub fn from_model_name(name: &str) -> Option<Self> {
        match name {
            "llama" | "spm" | "sentencepiece" | "phi" | "baichuan" | "aquila" | "mamba2" => {
                Some(Self::Spm)
            }
            "gpt2" | "bpe" | "qwen" | "qwen2" | "deepseek" | "bloom" | "grok" | "dbrx"
            | "starcoder" | "falcon" | "mpt" | "refact" | "command-r" | "gpt-neox" => {
                Some(Self::Bpe)
            }
            "bert" | "wpm" | "wordpiece" => Some(Self::Wpm),
            _ => None,
        }
    }
}

/// Per-token classification from `tokenizer.ggml.token_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TokenType {
    /// No information supplied.
    Undefined = 0,
    /// An ordinary vocabulary entry.
    Normal = 1,
    /// The `<unk>` token.
    Unknown = 2,
    /// A control token such as `<s>`, `<|im_end|>` — never produced by merging.
    Control = 3,
    /// An explicitly added token that must never be split.
    UserDefined = 4,
    /// A reserved slot that the model never emits.
    Unused = 5,
    /// A SentencePiece byte-fallback token in `<0xNN>` form.
    Byte = 6,
}

impl TokenType {
    /// Map a raw GGUF `token_type` code.
    fn from_code(code: i32) -> Self {
        match code {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Undefined,
        }
    }

    /// `true` for tokens llama.cpp suppresses when `special == false`.
    fn is_special(self) -> bool {
        matches!(self, Self::Control | Self::Unknown)
    }

    /// `true` for tokens that must be pre-split out of the raw text.
    fn is_pre_split(self) -> bool {
        matches!(self, Self::Control | Self::UserDefined | Self::Unknown)
    }
}

/// Token names that llama.cpp treats as end-of-generation regardless of what
/// `tokenizer.ggml.eos_token_id` says.
///
/// Keeping the full set matters: Llama-3-Instruct stops on `<|eot_id|>` while
/// its `eos_token_id` is `<|end_of_text|>`, and Qwen stops on `<|im_end|>`.
const EOG_TOKEN_NAMES: &[&str] = &[
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
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
];

/// Token names that identify the end-of-turn token.
const EOT_TOKEN_NAMES: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "<end_of_utterance>",
];

/// Candidate BOS token names, used only when metadata does not say.
const BOS_TOKEN_NAMES: &[&str] = &["<s>", "<|begin_of_text|>", "<|startoftext|>", "[CLS]"];

/// A tokenizer built entirely from GGUF metadata.
pub struct GgufVocab {
    vocab_type: VocabType,
    pre_type: PreType,
    tokens: Vec<String>,
    scores: Vec<f32>,
    types: Vec<TokenType>,
    /// Exact bytes contributed by each token, specials rendered as their text.
    token_bytes: Vec<Vec<u8>>,
    token_to_id: HashMap<String, u32>,
    /// Merge rank keyed by the raw `"left right"` merge string.
    bpe_ranks: HashMap<String, u32>,
    pre_tokenizer: Option<PreTokenizer>,
    /// Control / user-defined / unknown ids, longest text first.
    pre_split_ids: Vec<u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    eot_id: Option<u32>,
    eom_id: Option<u32>,
    unk_id: Option<u32>,
    sep_id: Option<u32>,
    pad_id: Option<u32>,
    eog_ids: HashSet<u32>,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    ignore_merges: bool,
    /// Longest token text in bytes, used to bound WordPiece's greedy scan.
    max_token_len: usize,
}

impl core::fmt::Debug for GgufVocab {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GgufVocab")
            .field("vocab_type", &self.vocab_type)
            .field("pre_type", &self.pre_type)
            .field("n_tokens", &self.tokens.len())
            .field("n_merges", &self.bpe_ranks.len())
            .field("bos_id", &self.bos_id)
            .field("eos_id", &self.eos_id)
            .field("eot_id", &self.eot_id)
            .field("n_eog", &self.eog_ids.len())
            .field("add_bos", &self.add_bos)
            .field("add_space_prefix", &self.add_space_prefix)
            .finish()
    }
}

/// `true` when `metadata` carries an embedded vocabulary we can build from.
pub fn has_embedded_vocab(metadata: &MetadataStore) -> bool {
    metadata
        .get(KEY_TOKENS)
        .and_then(MetadataValue::as_array)
        .is_some_and(|a| !a.is_empty())
}

impl GgufVocab {
    /// Build a tokenizer from GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::TokenizerError`] when the vocabulary is missing,
    /// when `tokenizer.ggml.model` names an algorithm this build does not
    /// implement, or when a pre-tokenizer pattern fails to compile.
    pub fn from_metadata(metadata: &MetadataStore) -> RuntimeResult<Self> {
        let tokens = read_string_array(metadata, KEY_TOKENS).ok_or_else(|| {
            RuntimeError::TokenizerError {
                message: format!("GGUF metadata has no '{KEY_TOKENS}' array"),
            }
        })?;
        if tokens.is_empty() {
            return Err(RuntimeError::TokenizerError {
                message: format!("GGUF metadata '{KEY_TOKENS}' array is empty"),
            });
        }

        let model_name = metadata
            .get(KEY_MODEL)
            .and_then(MetadataValue::as_str)
            .unwrap_or("llama");
        let vocab_type =
            VocabType::from_model_name(model_name).ok_or_else(|| RuntimeError::TokenizerError {
                message: format!(
                    "unsupported GGUF tokenizer model '{model_name}' \
                     (supported: llama/spm, gpt2/bpe, bert/wpm)"
                ),
            })?;

        let n = tokens.len();
        let scores = read_f32_array(metadata, KEY_SCORES).unwrap_or_default();
        let scores = normalize_len(scores, n, 0.0);
        let type_codes = read_i32_array(metadata, KEY_TOKEN_TYPE).unwrap_or_default();
        let types: Vec<TokenType> = (0..n)
            .map(|i| match type_codes.get(i) {
                Some(&c) => TokenType::from_code(c),
                // No token_type array: llama.cpp treats everything as normal.
                None => TokenType::Normal,
            })
            .collect();

        let mut token_to_id = HashMap::with_capacity(n);
        for (id, text) in tokens.iter().enumerate() {
            // First writer wins, matching llama.cpp's `emplace`.
            let id_u32 = u32::try_from(id).map_err(|_| RuntimeError::TokenizerError {
                message: "vocabulary larger than u32::MAX".to_string(),
            })?;
            token_to_id.entry(text.clone()).or_insert(id_u32);
        }

        let bpe_ranks = if vocab_type == VocabType::Bpe {
            read_merges(metadata)
        } else {
            HashMap::new()
        };

        // ── pre-tokenizer resolution ────────────────────────────────────────
        let pre_name = metadata.get(KEY_PRE).and_then(MetadataValue::as_str);
        let mut pre_type = PreType::Default;
        if vocab_type == VocabType::Bpe {
            pre_type = match pre_name {
                Some(name) => match pretok::pre_type_from_name(name) {
                    Some(t) => t,
                    None => {
                        tracing::warn!(
                            pre = name,
                            "unknown tokenizer.ggml.pre — falling back to the GPT-2 pattern; \
                             tokenization may differ from llama.cpp"
                        );
                        PreType::Default
                    }
                },
                None => {
                    // llama.cpp warns and uses `default` here.  A GGUF converted
                    // before `tokenizer.ggml.pre` existed (Meta-Llama-3-8B is the
                    // common case) still has the LLaMA-3 special tokens, so probe
                    // for them rather than silently degrading quality.
                    let looks_llama3 = token_to_id.contains_key("<|begin_of_text|>")
                        && token_to_id.contains_key("<|eot_id|>");
                    if looks_llama3 {
                        tracing::warn!(
                            "tokenizer.ggml.pre is missing; detected LLaMA-3 special tokens \
                             and selected the llama3 pre-tokenizer"
                        );
                        PreType::Llama3
                    } else {
                        tracing::warn!(
                            "tokenizer.ggml.pre is missing; using the default (GPT-2) \
                             pre-tokenizer — consider regenerating the GGUF"
                        );
                        PreType::Default
                    }
                }
            };
        }

        // ── llama.cpp per-type defaults ─────────────────────────────────────
        let (mut add_bos, mut add_eos, mut add_space_prefix) = match vocab_type {
            VocabType::Spm => (true, false, true),
            VocabType::Bpe => (pretok::implies_add_bos(pre_type), false, false),
            VocabType::Wpm => (true, false, false),
        };
        let ignore_merges = vocab_type == VocabType::Bpe && pretok::ignore_merges(pre_type);

        let pre_tokenizer = if vocab_type == VocabType::Bpe {
            Some(PreTokenizer::new(pre_type)?)
        } else {
            None
        };

        // ── special token ids ───────────────────────────────────────────────
        let lookup = |key: &str| -> Option<u32> {
            metadata
                .get(key)
                .and_then(|v| {
                    v.as_u32()
                        .or_else(|| v.as_i32().and_then(|i| u32::try_from(i).ok()))
                })
                .filter(|&id| (id as usize) < n)
        };
        let mut bos_id = lookup("tokenizer.ggml.bos_token_id");
        let mut eos_id = lookup("tokenizer.ggml.eos_token_id");
        let mut eot_id = lookup("tokenizer.ggml.eot_token_id");
        let eom_id = lookup("tokenizer.ggml.eom_token_id");
        let mut unk_id = lookup("tokenizer.ggml.unknown_token_id");
        let sep_id = lookup("tokenizer.ggml.separator_token_id");
        let pad_id = lookup("tokenizer.ggml.padding_token_id");

        if vocab_type == VocabType::Spm {
            // llama.cpp's SPM defaults, applied only when metadata is silent.
            bos_id = bos_id.or_else(|| find_id(&token_to_id, "<s>"));
            eos_id = eos_id.or_else(|| find_id(&token_to_id, "</s>"));
            unk_id = unk_id.or_else(|| find_id(&token_to_id, "<unk>"));
        }
        if bos_id.is_none() {
            bos_id = find_control_id(&token_to_id, &types, BOS_TOKEN_NAMES);
        }

        if let Some(v) = metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(MetadataValue::as_bool)
        {
            add_bos = v;
        }
        if let Some(v) = metadata
            .get("tokenizer.ggml.add_eos_token")
            .and_then(MetadataValue::as_bool)
        {
            add_eos = v;
        }
        if let Some(v) = metadata
            .get("tokenizer.ggml.add_space_prefix")
            .and_then(MetadataValue::as_bool)
        {
            add_space_prefix = v;
        }
        if add_bos && bos_id.is_none() {
            // Nothing to prepend — do not emit a bogus id.
            add_bos = false;
        }
        if add_eos && eos_id.is_none() {
            add_eos = false;
        }

        // Auto-detect EOT when metadata omits it (llama.cpp does the same).
        if eot_id.is_none() {
            eot_id = find_control_id(&token_to_id, &types, EOT_TOKEN_NAMES);
        }

        // ── end-of-generation set ───────────────────────────────────────────
        //
        // Name matches must additionally be typed CONTROL or USER_DEFINED.
        // llama.cpp accepts a name match of any type and overrides the type
        // with a warning; that is what makes Qwen3's *ordinary* token 128247
        // (whose text happens to be "</s>") an end-of-generation token and
        // truncates output on legitimate text.  Explicit `*_token_id` metadata
        // is still honoured whatever the type says.
        let mut eog_ids: HashSet<u32> = HashSet::new();
        for name in EOG_TOKEN_NAMES {
            match find_control_id(&token_to_id, &types, &[name]) {
                Some(id) => {
                    eog_ids.insert(id);
                }
                None => {
                    if let Some(&id) = token_to_id.get(*name) {
                        tracing::debug!(
                            token = *name,
                            id,
                            ty = ?types.get(id as usize),
                            "token spells an end marker but is not a control token — ignored"
                        );
                    }
                }
            }
        }
        eog_ids.extend(eos_id);
        eog_ids.extend(eot_id);
        eog_ids.extend(eom_id);

        // ── pre-split (special) token ids, longest text first ───────────────
        let mut pre_split_ids: Vec<u32> = (0..n)
            .filter(|&i| types[i].is_pre_split() && !tokens[i].is_empty())
            .map(|i| i as u32)
            .collect();
        pre_split_ids.sort_by(|&a, &b| {
            let (ta, tb) = (&tokens[a as usize], &tokens[b as usize]);
            tb.len().cmp(&ta.len()).then(a.cmp(&b))
        });

        let max_token_len = tokens.iter().map(String::len).max().unwrap_or(0);
        let mut vocab = Self {
            vocab_type,
            pre_type,
            tokens,
            scores,
            types,
            token_bytes: Vec::new(),
            token_to_id,
            bpe_ranks,
            pre_tokenizer,
            pre_split_ids,
            bos_id,
            eos_id,
            eot_id,
            eom_id,
            unk_id,
            sep_id,
            pad_id,
            eog_ids,
            add_bos,
            add_eos,
            add_space_prefix,
            ignore_merges,
            max_token_len,
        };
        vocab.token_bytes = vocab.build_token_bytes();

        tracing::info!(
            model = model_name,
            pre = ?vocab.pre_type,
            n_tokens = vocab.tokens.len(),
            n_merges = vocab.bpe_ranks.len(),
            bos = ?vocab.bos_id,
            eos = ?vocab.eos_id,
            eot = ?vocab.eot_id,
            n_eog = vocab.eog_ids.len(),
            add_bos = vocab.add_bos,
            "GGUF-embedded vocabulary loaded"
        );
        Ok(vocab)
    }

    /// Precompute the exact byte representation of every token.
    fn build_token_bytes(&self) -> Vec<Vec<u8>> {
        (0..self.tokens.len())
            .map(|i| {
                let text = &self.tokens[i];
                let ty = self.types[i];
                match self.vocab_type {
                    VocabType::Spm => match ty {
                        TokenType::Control | TokenType::Unknown | TokenType::UserDefined => {
                            text.as_bytes().to_vec()
                        }
                        TokenType::Normal => unescape_whitespace(text).into_bytes(),
                        TokenType::Byte => {
                            parse_byte_token(text).map(|b| vec![b]).unwrap_or_default()
                        }
                        TokenType::Unused | TokenType::Undefined => Vec::new(),
                    },
                    VocabType::Bpe | VocabType::Wpm => match ty {
                        TokenType::Control | TokenType::Unknown | TokenType::UserDefined => {
                            text.as_bytes().to_vec()
                        }
                        TokenType::Normal => byte_level::decode_bytes(text),
                        TokenType::Byte | TokenType::Unused | TokenType::Undefined => Vec::new(),
                    },
                }
            })
            .collect()
    }

    // ── accessors ───────────────────────────────────────────────────────────

    /// Number of entries in the vocabulary.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// `true` when the vocabulary is empty (never the case after loading).
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Which tokenization algorithm this vocabulary uses.
    pub fn vocab_type(&self) -> VocabType {
        self.vocab_type
    }

    /// Which pre-tokenizer family was resolved (BPE only).
    pub fn pre_type(&self) -> PreType {
        self.pre_type
    }

    /// The BOS token id, if the model declares one.
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    /// The EOS token id, if the model declares one.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// The end-of-turn token id (`<|eot_id|>`, `<|im_end|>`, …), if present.
    pub fn eot_id(&self) -> Option<u32> {
        self.eot_id
    }

    /// The end-of-message token id (`<|eom_id|>`), if present.
    pub fn eom_id(&self) -> Option<u32> {
        self.eom_id
    }

    /// The unknown token id, if present.
    pub fn unk_id(&self) -> Option<u32> {
        self.unk_id
    }

    /// The separator token id, if present.
    pub fn sep_id(&self) -> Option<u32> {
        self.sep_id
    }

    /// The padding token id, if present.
    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }

    /// Every token that ends generation.
    pub fn eog_ids(&self) -> &HashSet<u32> {
        &self.eog_ids
    }

    /// `true` when `id` ends generation.
    pub fn is_eog(&self, id: u32) -> bool {
        self.eog_ids.contains(&id)
    }

    /// Whether the model wants BOS prepended by `encode`.
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Whether the model wants EOS appended by `encode`.
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    /// The raw vocabulary text of `id` (still byte-level / `▁`-escaped).
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Look up a token by its exact vocabulary text.
    pub fn token_to_id(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    /// The classification of `id`.
    pub fn token_type(&self, id: u32) -> Option<TokenType> {
        self.types.get(id as usize).copied()
    }

    /// The score of `id` (SentencePiece only; `0.0` elsewhere).
    pub fn token_score(&self, id: u32) -> f32 {
        self.scores.get(id as usize).copied().unwrap_or(0.0)
    }

    /// The exact bytes `id` contributes to the output stream.
    ///
    /// Special / control tokens render as their literal text (`<|im_end|>`);
    /// use [`Self::is_special`] to filter them out when that is not wanted.
    /// Unlike a `String`-returning decode this never substitutes `U+FFFD`, so a
    /// token carrying a partial UTF-8 sequence survives intact.
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.token_bytes.get(id as usize).map(Vec::as_slice)
    }

    /// `true` when `id` is a control or unknown token.
    pub fn is_special(&self, id: u32) -> bool {
        self.types
            .get(id as usize)
            .copied()
            .is_some_and(TokenType::is_special)
    }

    /// The full `(id, bytes)` table, used for grammar-constrained sampling.
    pub fn vocab_bytes(&self) -> Vec<(u32, Vec<u8>)> {
        self.token_bytes
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.clone()))
            .collect()
    }

    // ── internal helpers used by the algorithm modules ──────────────────────

    pub(crate) fn merge_rank(&self, left: &str, right: &str) -> Option<u32> {
        let mut key = String::with_capacity(left.len() + right.len() + 1);
        key.push_str(left);
        key.push(' ');
        key.push_str(right);
        self.bpe_ranks.get(&key).copied()
    }

    pub(crate) fn ignore_merges_enabled(&self) -> bool {
        self.ignore_merges
    }

    pub(crate) fn pre_tokenizer(&self) -> Option<&PreTokenizer> {
        self.pre_tokenizer.as_ref()
    }

    pub(crate) fn max_token_len(&self) -> usize {
        self.max_token_len
    }

    /// Map a raw byte to its byte-fallback token, mirroring
    /// `llama_vocab::impl::byte_to_token`.
    pub(crate) fn byte_to_token(&self, b: u8) -> Option<u32> {
        match self.vocab_type {
            VocabType::Spm => {
                let hex = format!("<0x{b:02X}>");
                if let Some(&id) = self.token_to_id.get(&hex) {
                    return Some(id);
                }
                // Fall back to the bare byte as a one-character string.
                let s = char::from(b).to_string();
                self.token_to_id.get(&s).copied()
            }
            VocabType::Bpe | VocabType::Wpm => self
                .token_to_id
                .get(&byte_level::byte_to_string(b))
                .copied(),
        }
    }

    // ── tokenization ────────────────────────────────────────────────────────

    /// Encode `text` to token ids.
    ///
    /// * `add_special` — prepend BOS / append EOS when the model asks for it
    ///   (`tokenizer.ggml.add_bos_token`).
    /// * `parse_special` — recognise control tokens such as `<|im_start|>` in
    ///   the input instead of tokenizing them as ordinary text.  User-defined
    ///   tokens are always recognised, matching llama.cpp.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let fragments = self.partition_specials(text, parse_special);
        match self.vocab_type {
            VocabType::Spm => self.tokenize_spm(&fragments, add_special, &mut out),
            VocabType::Bpe => self.tokenize_bpe(&fragments, add_special, &mut out),
            VocabType::Wpm => self.tokenize_wpm(&fragments, add_special, &mut out),
        }
        out
    }

    fn tokenize_spm(&self, fragments: &[Fragment<'_>], add_special: bool, out: &mut Vec<u32>) {
        // OG SentencePiece behaviour: the first raw fragment is prefixed with a
        // space, and so is any fragment that follows a special token.
        let mut is_prev_special = true;
        if add_special && self.add_bos {
            out.extend(self.bos_id);
        }
        for fragment in fragments {
            match *fragment {
                Fragment::Token(id) => {
                    out.push(id);
                    is_prev_special = true;
                }
                Fragment::Text(raw) => {
                    let mut text = String::with_capacity(raw.len() + 1);
                    if self.add_space_prefix && is_prev_special {
                        text.push(' ');
                    }
                    text.push_str(raw);
                    spm::tokenize(self, &escape_whitespace(&text), out);
                    is_prev_special = false;
                }
            }
        }
        if add_special && self.add_eos {
            out.extend(self.eos_id);
        }
    }

    fn tokenize_bpe(&self, fragments: &[Fragment<'_>], add_special: bool, out: &mut Vec<u32>) {
        if add_special && self.add_bos {
            out.extend(self.bos_id);
        }
        for fragment in fragments {
            match *fragment {
                Fragment::Token(id) => out.push(id),
                Fragment::Text(raw) => bpe::tokenize(self, raw, out),
            }
        }
        if add_special && self.add_eos {
            out.extend(self.eos_id);
        }
    }

    fn tokenize_wpm(&self, fragments: &[Fragment<'_>], add_special: bool, out: &mut Vec<u32>) {
        if add_special {
            out.extend(self.bos_id);
        }
        for fragment in fragments {
            match *fragment {
                Fragment::Token(id) => out.push(id),
                Fragment::Text(raw) => wpm::tokenize(self, raw, out),
            }
        }
        if add_special {
            out.extend(self.sep_id);
        }
    }

    /// Split `text` around special tokens, mirroring `tokenizer_st_partition`.
    fn partition_specials<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Fragment<'a>> {
        let mut fragments: Vec<Fragment<'a>> = if text.is_empty() {
            Vec::new()
        } else {
            vec![Fragment::Text(text)]
        };
        for &id in &self.pre_split_ids {
            let idx = id as usize;
            let ty = self.types[idx];
            if !parse_special && ty.is_special() {
                // Control / unknown tokens stay literal; user-defined tokens are
                // always split out.
                continue;
            }
            let needle = self.tokens[idx].as_str();
            let mut next = Vec::with_capacity(fragments.len());
            for fragment in fragments.drain(..) {
                match fragment {
                    Fragment::Token(t) => next.push(Fragment::Token(t)),
                    Fragment::Text(mut rest) => {
                        while let Some(pos) = rest.find(needle) {
                            if pos > 0 {
                                next.push(Fragment::Text(&rest[..pos]));
                            }
                            next.push(Fragment::Token(id));
                            rest = &rest[pos + needle.len()..];
                        }
                        if !rest.is_empty() {
                            next.push(Fragment::Text(rest));
                        }
                    }
                }
            }
            fragments = next;
        }
        fragments
    }

    // ── detokenization ──────────────────────────────────────────────────────

    /// Concatenate the exact bytes of `tokens`.
    ///
    /// When `skip_special` is set, control and unknown tokens contribute
    /// nothing.  The result may end mid-UTF-8-sequence when `tokens` is a
    /// prefix of a longer stream — that is the point; see
    /// `crate::stream_decode::IncrementalDetokenizer`.
    pub fn detokenize_bytes(&self, tokens: &[u32], skip_special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in tokens {
            if skip_special && self.is_special(id) {
                continue;
            }
            if let Some(bytes) = self.token_bytes(id) {
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    /// Decode `tokens` to text.
    ///
    /// Invalid trailing bytes are replaced with `U+FFFD`; call
    /// [`Self::detokenize_bytes`] instead when the caller needs to stitch
    /// partial sequences together across calls.
    pub fn detokenize(&self, tokens: &[u32], skip_special: bool) -> String {
        String::from_utf8_lossy(&self.detokenize_bytes(tokens, skip_special)).into_owned()
    }
}

/// One piece of the input during special-token partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fragment<'a> {
    /// Raw text still to be tokenized.
    Text(&'a str),
    /// A special token recognised verbatim.
    Token(u32),
}

// ── metadata helpers ────────────────────────────────────────────────────────

fn read_string_array(metadata: &MetadataStore, key: &str) -> Option<Vec<String>> {
    let arr = metadata.get(key)?.as_array()?;
    Some(
        arr.iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect(),
    )
}

fn read_f32_array(metadata: &MetadataStore, key: &str) -> Option<Vec<f32>> {
    let arr = metadata.get(key)?.as_array()?;
    Some(arr.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect())
}

fn read_i32_array(metadata: &MetadataStore, key: &str) -> Option<Vec<i32>> {
    let arr = metadata.get(key)?.as_array()?;
    Some(
        arr.iter()
            .map(|v| {
                v.as_i32()
                    .or_else(|| v.as_u32().and_then(|u| i32::try_from(u).ok()))
                    .unwrap_or(0)
            })
            .collect(),
    )
}

/// Parse `tokenizer.ggml.merges` into `"left right" → rank`.
///
/// Entries may be flat strings (`"Ġ t"`) or two-element arrays; both forms are
/// normalised to the flat representation because byte-level tokens can never
/// contain a literal space, which makes `"left right"` an unambiguous key.
fn read_merges(metadata: &MetadataStore) -> HashMap<String, u32> {
    let Some(arr) = metadata.get(KEY_MERGES).and_then(MetadataValue::as_array) else {
        return HashMap::new();
    };
    let mut ranks = HashMap::with_capacity(arr.len());
    for (rank, entry) in arr.iter().enumerate() {
        let Ok(rank) = u32::try_from(rank) else {
            break;
        };
        let key = match entry {
            MetadataValue::String(s) => s.clone(),
            MetadataValue::Array(pair) if pair.len() == 2 => {
                let left = pair[0].as_str().unwrap_or_default();
                let right = pair[1].as_str().unwrap_or_default();
                format!("{left} {right}")
            }
            _ => continue,
        };
        // llama.cpp searches for the separator from index 1, so a merge whose
        // left side is a bare space still parses.  Skip entries with none.
        if key.len() < 2 || !key.as_bytes()[1..].contains(&b' ') {
            continue;
        }
        ranks.entry(key).or_insert(rank);
    }
    ranks
}

fn normalize_len<T: Clone>(mut v: Vec<T>, n: usize, fill: T) -> Vec<T> {
    if v.len() < n {
        v.resize(n, fill);
    } else {
        v.truncate(n);
    }
    v
}

fn find_id(map: &HashMap<String, u32>, name: &str) -> Option<u32> {
    map.get(name).copied()
}

/// Find the first of `names` present in the vocabulary *and* typed as a control
/// or user-defined token.
///
/// The type check is what stops the classic failure mode: byte-level BPE
/// vocabularies frequently contain the literal text `</s>` as an ordinary
/// merged token (id 128247 in Qwen3's), so a plain name lookup silently picks
/// a normal word as the EOS token.
fn find_control_id(map: &HashMap<String, u32>, types: &[TokenType], names: &[&str]) -> Option<u32> {
    names.iter().find_map(|name| {
        map.get(*name).copied().filter(|&id| {
            types
                .get(id as usize)
                .copied()
                .is_some_and(|t| matches!(t, TokenType::Control | TokenType::UserDefined))
        })
    })
}

/// Replace every space with the SentencePiece marker `▁`.
pub fn escape_whitespace(text: &str) -> String {
    text.replace(' ', "\u{2581}")
}

/// Replace every SentencePiece marker `▁` with a space.
pub fn unescape_whitespace(text: &str) -> String {
    text.replace('\u{2581}', " ")
}

/// Parse a SentencePiece byte-fallback token of the form `<0xNN>`.
fn parse_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

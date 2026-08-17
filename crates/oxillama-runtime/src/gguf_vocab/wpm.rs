// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! WordPiece tokenization for BERT-style GGUF vocabularies.
//!
//! Ported from llama.cpp's `llm_tokenizer_wpm_session`: the text is normalised
//! and split into words, each word is prefixed with `▁`, and the longest
//! vocabulary entry starting at each position is taken greedily.  A word that
//! cannot be covered completely falls back to the unknown token.
//!
//! # Known deviation
//!
//! llama.cpp strips combining accents via its generated Unicode decomposition
//! table.  Doing the same here would require a Unicode normalisation dependency,
//! so this implementation lower-cases and drops control characters but leaves
//! accents in place.  BERT is not one of the architectures
//! `oxillama-arch` can run, so this path exists for completeness (and for
//! `oxillama tokenize`) rather than for inference.

use super::GgufVocab;

/// Tokenize raw `text`, appending ids to `out`.
pub(crate) fn tokenize(vocab: &GgufVocab, text: &str, out: &mut Vec<u32>) {
    for word in preprocess(text) {
        let escaped = format!("\u{2581}{word}");
        let before = out.len();
        let bytes = escaped.len();
        let mut i = 0usize;
        while i < bytes {
            let mut matched = false;
            let upper = (i + vocab.max_token_len()).min(bytes);
            let mut j = upper;
            while j > i {
                if let Some(candidate) = escaped.get(i..j) {
                    if let Some(id) = vocab.token_to_id(candidate) {
                        out.push(id);
                        i = j;
                        matched = true;
                        break;
                    }
                }
                j -= 1;
            }
            if !matched {
                out.truncate(before);
                break;
            }
        }
        if out.len() == before {
            out.extend(vocab.unk_id());
        }
    }
}

/// Normalise and split `text` into WordPiece words.
///
/// Whitespace and control characters separate words; punctuation, symbols and
/// CJK ideographs each become a word of their own.
fn preprocess(text: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_control() || ch == '\u{fffd}' {
            if !current.is_empty() {
                words.push(core::mem::take(&mut current));
            }
            continue;
        }
        if is_standalone(ch) {
            if !current.is_empty() {
                words.push(core::mem::take(&mut current));
            }
            words.push(ch.to_lowercase().collect());
            continue;
        }
        current.extend(ch.to_lowercase());
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// `true` for characters that always form a word of their own.
fn is_standalone(ch: char) -> bool {
    if ch.is_ascii_punctuation() {
        return true;
    }
    matches!(ch as u32,
        0x3000..=0x303F      // CJK symbols and punctuation
        | 0x3400..=0x4DBF    // CJK extension A
        | 0x4E00..=0x9FFF    // CJK unified ideographs
        | 0xF900..=0xFAFF    // CJK compatibility ideographs
        | 0xFF00..=0xFFEF    // halfwidth / fullwidth forms
        | 0x20000..=0x2A6DF  // CJK extension B
        | 0x2F800..=0x2FA1F) // CJK compatibility supplement
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_lowercases_and_splits_on_whitespace() {
        assert_eq!(preprocess("Hello World"), ["hello", "world"]);
    }

    #[test]
    fn preprocess_isolates_punctuation() {
        assert_eq!(preprocess("hi, there!"), ["hi", ",", "there", "!"]);
    }

    #[test]
    fn preprocess_isolates_cjk() {
        assert_eq!(preprocess("a漢字b"), ["a", "漢", "字", "b"]);
    }

    #[test]
    fn preprocess_of_blank_is_empty() {
        assert!(preprocess("   \n\t ").is_empty());
    }
}

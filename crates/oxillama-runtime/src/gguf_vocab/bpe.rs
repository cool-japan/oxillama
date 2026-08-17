// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Byte-level BPE tokenization, ported from llama.cpp's
//! `llm_tokenizer_bpe_session`.
//!
//! The pipeline is:
//!
//! 1. split the raw text with the model's pre-tokenizer regexes
//!    ([`super::pretok`]),
//! 2. map each piece through the GPT-2 byte-level alphabet so that every byte
//!    becomes a printable code point ([`super::byte_level`]),
//! 3. run the merge loop inside each piece, popping the lowest-ranked merge
//!    first,
//! 4. emit the surviving symbols, falling back to single-byte tokens for
//!    anything that is somehow not in the vocabulary.
//!
//! Step 3 is skipped entirely for pre-tokenizers that set `ignore_merges`
//! (LLaMA-3 and friends): if the whole pre-token is already a vocabulary entry
//! it is emitted directly.  Omitting that check is the classic source of
//! "almost right" LLaMA-3 token ids.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::byte_level;
use super::symbols::{absorb, split_chars, Symbol};
use super::GgufVocab;

/// A candidate merge of two adjacent symbols.
#[derive(Debug, Clone, Copy)]
struct Bigram {
    left: i32,
    right: i32,
    rank: u32,
    /// Lengths at queue time, used to detect stale entries.
    left_len: usize,
    right_len: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bigram {
    /// Lowest rank first; ties broken towards the left-most position.
    ///
    /// Mirrors llama.cpp's `llm_bigram_bpe::comparator`.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

/// Tokenize raw `text`, appending ids to `out`.
pub(crate) fn tokenize(vocab: &GgufVocab, text: &str, out: &mut Vec<u32>) {
    if text.is_empty() {
        return;
    }
    let pieces: Vec<&str> = match vocab.pre_tokenizer() {
        Some(pre) => pre.split(text),
        None => vec![text],
    };
    for piece in pieces {
        let word = byte_level::encode_bytes(piece);
        tokenize_word(vocab, &word, out);
    }
}

fn tokenize_word(vocab: &GgufVocab, word: &str, out: &mut Vec<u32>) {
    if word.is_empty() {
        return;
    }
    // `ignore_merges`: a pre-token that is already a vocabulary entry is taken
    // whole rather than re-derived through the merge table.
    if vocab.ignore_merges_enabled() {
        if let Some(id) = vocab.token_to_id(word) {
            out.push(id);
            return;
        }
    }

    let mut symbols = split_chars(word);
    let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();
    for i in 1..symbols.len() {
        add_bigram(vocab, word, &symbols, i - 1, i, &mut queue);
    }

    while let Some(bigram) = queue.pop() {
        let (left, right) = (bigram.left as usize, bigram.right as usize);
        if symbols[left].len != bigram.left_len || symbols[right].len != bigram.right_len {
            // Stale: one of the two sides has already grown or been absorbed.
            continue;
        }
        if symbols[left].len == 0 || symbols[right].len == 0 {
            continue;
        }
        absorb(&mut symbols, left, right);

        let prev = symbols[left].prev;
        let next = symbols[left].next;
        if prev >= 0 {
            add_bigram(vocab, word, &symbols, prev as usize, left, &mut queue);
        }
        if next >= 0 {
            add_bigram(vocab, word, &symbols, left, next as usize, &mut queue);
        }
    }

    emit(vocab, word, &symbols, out);
}

fn emit(vocab: &GgufVocab, word: &str, symbols: &[Symbol], out: &mut Vec<u32>) {
    let mut index = 0i32;
    while index != -1 {
        let symbol = symbols[index as usize];
        index = symbol.next;
        if symbol.len == 0 {
            continue;
        }
        let piece = symbol.text(word);
        match vocab.token_to_id(piece) {
            Some(id) => out.push(id),
            None => {
                // llama.cpp retries one raw byte at a time; only single-byte
                // (ASCII) code points can ever match a vocabulary entry, so
                // anything else is dropped exactly as upstream drops it.
                for ch in piece.chars() {
                    if !ch.is_ascii() {
                        continue;
                    }
                    let mut buf = [0u8; 4];
                    if let Some(id) = vocab.token_to_id(ch.encode_utf8(&mut buf)) {
                        out.push(id);
                    }
                }
            }
        }
    }
}

fn add_bigram(
    vocab: &GgufVocab,
    word: &str,
    symbols: &[Symbol],
    left: usize,
    right: usize,
    queue: &mut BinaryHeap<Bigram>,
) {
    let left_text = symbols[left].text(word);
    let right_text = symbols[right].text(word);
    let Some(rank) = vocab.merge_rank(left_text, right_text) else {
        return;
    };
    queue.push(Bigram {
        left: left as i32,
        right: right as i32,
        rank,
        left_len: left_text.len(),
        right_len: right_text.len(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigram_order_prefers_lower_rank() {
        let mut heap = BinaryHeap::new();
        heap.push(Bigram {
            left: 0,
            right: 1,
            rank: 10,
            left_len: 1,
            right_len: 1,
        });
        heap.push(Bigram {
            left: 4,
            right: 5,
            rank: 2,
            left_len: 1,
            right_len: 1,
        });
        let top = heap.pop().expect("test: heap is non-empty");
        assert_eq!(top.rank, 2, "lowest rank must merge first");
    }

    #[test]
    fn bigram_order_breaks_ties_leftmost_first() {
        let mut heap = BinaryHeap::new();
        heap.push(Bigram {
            left: 9,
            right: 10,
            rank: 3,
            left_len: 1,
            right_len: 1,
        });
        heap.push(Bigram {
            left: 1,
            right: 2,
            rank: 3,
            left_len: 1,
            right_len: 1,
        });
        let top = heap.pop().expect("test: heap is non-empty");
        assert_eq!(top.left, 1, "equal ranks must resolve left-to-right");
    }
}

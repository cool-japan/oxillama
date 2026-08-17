// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! SentencePiece tokenization, ported from llama.cpp's
//! `llm_tokenizer_spm_session`.
//!
//! # Why not Unigram/Viterbi
//!
//! GGUF SentencePiece vocabularies ship `tokenizer.ggml.scores`, which invites
//! the assumption that decoding is Unigram Viterbi.  It is not: LLaMA's
//! SentencePiece model was trained in **BPE mode**, and llama.cpp reproduces it
//! with a greedy highest-score bigram merge.  Running Viterbi over the same
//! scores yields a different — and wrong — segmentation for LLaMA-family
//! models, so this module follows llama.cpp exactly:
//!
//! 1. cut the (already `▁`-escaped) text into UTF-8 characters,
//! 2. seed a max-heap with every adjacent pair that exists in the vocabulary,
//!    keyed by the merged token's score,
//! 3. pop the best pair, merge it, and offer the two new neighbours,
//! 4. finally walk the surviving symbols; anything that is *not* itself a
//!    vocabulary entry is re-split along the merge that produced it
//!    (`resegment`), and if even that fails the raw bytes are emitted through
//!    the `<0xNN>` byte-fallback tokens.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::symbols::{absorb, split_chars, Symbol};
use super::GgufVocab;

/// A candidate merge of two adjacent symbols.
#[derive(Debug, Clone, Copy)]
struct Bigram {
    left: i32,
    right: i32,
    score: f32,
    /// Combined byte length at the time the candidate was queued.
    size: usize,
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
    /// Highest score first; ties broken towards the left-most position.
    ///
    /// This reproduces llama.cpp's `llm_bigram_spm::comparator`, which orders a
    /// `std::priority_queue` so that the top element has the greatest score and,
    /// among equal scores, the smallest `left` index.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

/// Tokenize `text` — which must already be `▁`-escaped — appending to `out`.
pub(crate) fn tokenize(vocab: &GgufVocab, text: &str, out: &mut Vec<u32>) {
    if text.is_empty() {
        return;
    }
    let mut symbols = split_chars(text);
    let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();
    // The merge that produced a given piece of text, so `resegment` can undo it.
    let mut rev_merge: HashMap<String, (usize, usize)> = HashMap::new();

    for i in 1..symbols.len() {
        try_add_bigram(vocab, text, &symbols, i - 1, i, &mut queue, &mut rev_merge);
    }

    while let Some(bigram) = queue.pop() {
        let (left, right) = (bigram.left as usize, bigram.right as usize);
        let (left_len, right_len) = (symbols[left].len, symbols[right].len);
        if left_len == 0 || right_len == 0 || left_len + right_len != bigram.size {
            // One side was already merged into something else.
            continue;
        }
        absorb(&mut symbols, left, right);

        let prev = symbols[left].prev;
        let next = symbols[left].next;
        if prev >= 0 {
            try_add_bigram(
                vocab,
                text,
                &symbols,
                prev as usize,
                left,
                &mut queue,
                &mut rev_merge,
            );
        }
        if next >= 0 {
            try_add_bigram(
                vocab,
                text,
                &symbols,
                left,
                next as usize,
                &mut queue,
                &mut rev_merge,
            );
        }
    }

    let mut index = 0i32;
    while index != -1 {
        let symbol = symbols[index as usize];
        resegment(vocab, text, &symbols, &rev_merge, symbol, out);
        index = symbol.next;
    }
}

/// Emit `symbol`, splitting it back apart when it is not a vocabulary entry.
fn resegment(
    vocab: &GgufVocab,
    text: &str,
    symbols: &[Symbol],
    rev_merge: &HashMap<String, (usize, usize)>,
    symbol: Symbol,
    out: &mut Vec<u32>,
) {
    // Explicit stack rather than recursion: a pathological input could nest as
    // deeply as the text is long.
    let mut stack = vec![symbol];
    while let Some(sym) = stack.pop() {
        if sym.len == 0 {
            continue;
        }
        let piece = sym.text(text);
        if let Some(id) = vocab.token_to_id(piece) {
            out.push(id);
            continue;
        }
        // Split back along the merge that produced this text.  The two halves
        // must be strictly shorter than the piece, otherwise a stale entry
        // (a symbol that grew after the bigram was queued) would loop forever.
        let split = rev_merge
            .get(piece)
            .filter(|&&(left, right)| symbols[left].len < sym.len && symbols[right].len < sym.len);
        match split {
            Some(&(left, right)) => {
                // Push right first so the left half is processed first.
                stack.push(symbols[right]);
                stack.push(symbols[left]);
            }
            None => {
                // No merge produced this text — fall back to raw bytes.
                for &byte in piece.as_bytes() {
                    match vocab.byte_to_token(byte) {
                        Some(id) => out.push(id),
                        None => out.extend(vocab.unk_id()),
                    }
                }
            }
        }
    }
}

fn try_add_bigram(
    vocab: &GgufVocab,
    text: &str,
    symbols: &[Symbol],
    left: usize,
    right: usize,
    queue: &mut BinaryHeap<Bigram>,
    rev_merge: &mut HashMap<String, (usize, usize)>,
) {
    let start = symbols[left].start;
    let size = symbols[left].len + symbols[right].len;
    let Some(merged) = text.get(start..start + size) else {
        return;
    };
    let Some(id) = vocab.token_to_id(merged) else {
        return;
    };
    queue.push(Bigram {
        left: left as i32,
        right: right as i32,
        score: vocab.token_score(id),
        size,
    });
    rev_merge.insert(merged.to_string(), (left, right));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigram_order_prefers_higher_score() {
        let mut heap = BinaryHeap::new();
        heap.push(Bigram {
            left: 5,
            right: 6,
            score: -1.0,
            size: 2,
        });
        heap.push(Bigram {
            left: 0,
            right: 1,
            score: -3.0,
            size: 2,
        });
        let top = heap.pop().expect("test: heap is non-empty");
        assert_eq!(top.left, 5, "higher score must come first");
    }

    #[test]
    fn bigram_order_breaks_ties_leftmost_first() {
        let mut heap = BinaryHeap::new();
        heap.push(Bigram {
            left: 7,
            right: 8,
            score: -2.0,
            size: 2,
        });
        heap.push(Bigram {
            left: 2,
            right: 3,
            score: -2.0,
            size: 2,
        });
        let top = heap.pop().expect("test: heap is non-empty");
        assert_eq!(top.left, 2, "equal scores must resolve left-to-right");
    }
}

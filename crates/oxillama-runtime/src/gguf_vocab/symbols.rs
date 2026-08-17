// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The intrusive symbol list shared by the SPM and BPE merge loops.
//!
//! Both algorithms start by cutting the input into single UTF-8 characters and
//! then repeatedly merge adjacent pairs.  Rather than shifting a vector on every
//! merge, llama.cpp keeps a doubly-linked list of `(start, len)` spans over the
//! original buffer and marks merged-away symbols with `len == 0`.  The same
//! layout is reproduced here so the merge order — and therefore the resulting
//! token ids — match exactly.

/// A span of the input text participating in the merge loop.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Symbol {
    /// Index of the previous live symbol, or `-1`.
    pub prev: i32,
    /// Index of the next live symbol, or `-1`.
    pub next: i32,
    /// Byte offset of this span within the working buffer.
    pub start: usize,
    /// Byte length of this span; `0` marks a symbol that was merged away.
    pub len: usize,
}

impl Symbol {
    /// The text of this symbol within `buf`.
    pub fn text<'a>(&self, buf: &'a str) -> &'a str {
        buf.get(self.start..self.start + self.len).unwrap_or("")
    }
}

/// Cut `text` into one symbol per UTF-8 character.
pub(crate) fn split_chars(text: &str) -> Vec<Symbol> {
    let mut symbols: Vec<Symbol> = Vec::with_capacity(text.len());
    for (start, ch) in text.char_indices() {
        let index = symbols.len();
        let len = ch.len_utf8();
        symbols.push(Symbol {
            prev: index as i32 - 1,
            next: -1,
            start,
            len,
        });
        if index > 0 {
            symbols[index - 1].next = index as i32;
        }
    }
    symbols
}

/// Unlink `right` from the chain after its bytes were absorbed into `left`.
pub(crate) fn absorb(symbols: &mut [Symbol], left: usize, right: usize) {
    let right_len = symbols[right].len;
    let right_next = symbols[right].next;
    symbols[left].len += right_len;
    symbols[right].len = 0;
    symbols[left].next = right_next;
    if right_next >= 0 {
        symbols[right_next as usize].prev = left as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_chars_links_are_consistent() {
        let text = "aé漢";
        let symbols = split_chars(text);
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0].prev, -1);
        assert_eq!(symbols[0].next, 1);
        assert_eq!(symbols[2].next, -1);
        assert_eq!(symbols[0].text(text), "a");
        assert_eq!(symbols[1].text(text), "é");
        assert_eq!(symbols[2].text(text), "漢");
    }

    #[test]
    fn absorb_merges_and_relinks() {
        let text = "abc";
        let mut symbols = split_chars(text);
        absorb(&mut symbols, 0, 1);
        assert_eq!(symbols[0].text(text), "ab");
        assert_eq!(symbols[1].len, 0);
        assert_eq!(symbols[0].next, 2);
        assert_eq!(symbols[2].prev, 0);
    }

    #[test]
    fn split_chars_of_empty_is_empty() {
        assert!(split_chars("").is_empty());
    }
}

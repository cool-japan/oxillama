// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Streaming detokenization helpers.
//!
//! Two problems appear the moment generated tokens are streamed to a caller one
//! at a time, and both are solved here:
//!
//! * **Split UTF-8.** Byte-level BPE happily splits a single CJK character or
//!   emoji across two or three tokens.  Decoding each token on its own turns
//!   those partial byte sequences into `U+FFFD`, permanently destroying the
//!   character — `"こんにちは"` arrives as a row of replacement glyphs.
//!   [`Utf8StreamDecoder`] buffers the incomplete tail and releases text only
//!   once it is a complete UTF-8 sequence, which is what llama.cpp's server
//!   does.
//! * **Stop sequences that straddle a token boundary.** A caller asking to stop
//!   at `"\nUser:"` cannot un-send text it has already been streamed, so the
//!   generator must withhold any trailing text that could still grow into a
//!   stop sequence.  [`StopSequenceBuffer`] holds back exactly the longest
//!   suffix that is a prefix of some stop string and no more.

/// Reassembles UTF-8 text from a stream of raw token bytes.
///
/// Bytes are appended as they arrive; each call returns only the text that is
/// now complete.  An incomplete trailing sequence is retained until the
/// following call supplies the rest, so no `U+FFFD` is ever emitted for text
/// that is merely still arriving.  Genuinely invalid bytes are replaced once,
/// exactly as `String::from_utf8_lossy` would.
#[derive(Debug, Clone, Default)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    /// Create an empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `bytes` and return every character that is now complete.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        if bytes.is_empty() && self.pending.is_empty() {
            return String::new();
        }
        self.pending.extend_from_slice(bytes);
        let mut out = String::with_capacity(self.pending.len());
        let mut consumed = 0usize;
        loop {
            let rest = &self.pending[consumed..];
            if rest.is_empty() {
                break;
            }
            match core::str::from_utf8(rest) {
                Ok(text) => {
                    out.push_str(text);
                    consumed = self.pending.len();
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        // Safe by construction: `valid_up_to` is the length of a
                        // verified-valid prefix.
                        match core::str::from_utf8(&rest[..valid_up_to]) {
                            Ok(text) => out.push_str(text),
                            Err(_) => break,
                        }
                    }
                    match err.error_len() {
                        // Truly invalid bytes: replace and continue.
                        Some(bad) => {
                            out.push('\u{fffd}');
                            consumed += valid_up_to + bad;
                        }
                        // Incomplete tail: keep it for the next push.
                        None => {
                            consumed += valid_up_to;
                            break;
                        }
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        out
    }

    /// `true` when an incomplete sequence is still buffered.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Release any buffered bytes, replacing an incomplete tail with `U+FFFD`.
    ///
    /// Call this once generation has finished; a well-formed stream leaves
    /// nothing behind and this returns an empty string.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }
}

/// What [`StopSequenceBuffer::push`] decided about the text it was given.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StopFeed {
    /// Text that is safe to emit to the caller right now.
    pub emit: String,
    /// The stop sequence that was matched, if any.
    ///
    /// When this is `Some`, generation must stop and the matched text — plus
    /// anything after it — has already been discarded from the output.
    pub matched: Option<String>,
}

/// Withholds trailing text that could still complete a stop sequence.
///
/// Feed decoded text in as it is produced.  Text is released as soon as it can
/// no longer become part of a stop sequence, so a consumer never sees a partial
/// stop string it would have to retract.
#[derive(Debug, Clone, Default)]
pub struct StopSequenceBuffer {
    stops: Vec<String>,
    held: String,
    /// Longest stop sequence, in bytes — the most that ever needs holding.
    max_stop_len: usize,
}

impl StopSequenceBuffer {
    /// Create a buffer for `stops`.
    ///
    /// Empty stop strings are ignored; an empty list makes every `push` a
    /// pass-through.
    pub fn new(stops: Vec<String>) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let max_stop_len = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            held: String::new(),
            max_stop_len,
        }
    }

    /// `true` when no stop sequences are configured.
    pub fn is_disabled(&self) -> bool {
        self.stops.is_empty()
    }

    /// Feed newly decoded `text`.
    pub fn push(&mut self, text: &str) -> StopFeed {
        if self.stops.is_empty() {
            return StopFeed {
                emit: text.to_string(),
                matched: None,
            };
        }
        self.held.push_str(text);

        // A complete stop sequence anywhere in the buffer ends generation.  The
        // earliest match wins; among matches at the same position the longest
        // does, so `["ab", "abc"]` on "abc" reports "abc".
        let mut best: Option<(usize, &String)> = None;
        for stop in &self.stops {
            if let Some(pos) = self.held.find(stop.as_str()) {
                let better = match best {
                    None => true,
                    Some((best_pos, best_stop)) => {
                        pos < best_pos || (pos == best_pos && stop.len() > best_stop.len())
                    }
                };
                if better {
                    best = Some((pos, stop));
                }
            }
        }
        if let Some((pos, stop)) = best {
            let matched = stop.clone();
            let emit = self.held[..pos].to_string();
            self.held.clear();
            return StopFeed {
                emit,
                matched: Some(matched),
            };
        }

        // No full match: hold back the longest suffix that is a prefix of some
        // stop sequence, release everything before it.
        let keep = self.partial_suffix_len();
        let split = self.held.len() - keep;
        let emit = self.held[..split].to_string();
        self.held.drain(..split);
        StopFeed {
            emit,
            matched: None,
        }
    }

    /// Length in bytes of the longest suffix of the buffer that is a proper
    /// prefix of some stop sequence.
    fn partial_suffix_len(&self) -> usize {
        let max = self.max_stop_len.min(self.held.len());
        for len in (1..=max).rev() {
            let start = self.held.len() - len;
            if !self.held.is_char_boundary(start) {
                continue;
            }
            let suffix = &self.held[start..];
            if self
                .stops
                .iter()
                .any(|stop| stop.len() > suffix.len() && stop.starts_with(suffix))
            {
                return len;
            }
        }
        0
    }

    /// Release everything still held (generation ended without a stop match).
    pub fn flush(&mut self) -> String {
        core::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_decoder_reassembles_split_japanese() {
        // "こんにちは世界" split at arbitrary byte boundaries, as byte-level BPE
        // would hand it over one token at a time.
        let text = "こんにちは世界";
        let bytes = text.as_bytes();
        let mut decoder = Utf8StreamDecoder::new();
        let mut out = String::new();
        for chunk in bytes.chunks(2) {
            out.push_str(&decoder.push(chunk));
        }
        out.push_str(&decoder.finish());
        assert_eq!(out, text);
        assert!(
            !out.contains('\u{fffd}'),
            "streamed output must not contain U+FFFD, got {out:?}"
        );
    }

    #[test]
    fn utf8_decoder_reassembles_emoji_byte_by_byte() {
        let text = "🚀🎌";
        let mut decoder = Utf8StreamDecoder::new();
        let mut out = String::new();
        for b in text.as_bytes() {
            out.push_str(&decoder.push(&[*b]));
        }
        out.push_str(&decoder.finish());
        assert_eq!(out, text);
        assert!(!out.contains('\u{fffd}'));
    }

    #[test]
    fn utf8_decoder_emits_ascii_immediately() {
        let mut decoder = Utf8StreamDecoder::new();
        assert_eq!(decoder.push(b"hello"), "hello");
        assert!(!decoder.has_pending());
    }

    #[test]
    fn utf8_decoder_holds_incomplete_tail() {
        let mut decoder = Utf8StreamDecoder::new();
        // First two bytes of "こ" (E3 81 93).
        assert_eq!(decoder.push(&[0xE3, 0x81]), "");
        assert!(decoder.has_pending());
        assert_eq!(decoder.push(&[0x93]), "こ");
        assert!(!decoder.has_pending());
    }

    #[test]
    fn utf8_decoder_replaces_genuinely_invalid_bytes() {
        let mut decoder = Utf8StreamDecoder::new();
        let out = decoder.push(&[b'a', 0xFF, b'b']);
        assert_eq!(out, "a\u{fffd}b");
    }

    #[test]
    fn utf8_decoder_finish_flushes_truncated_tail() {
        let mut decoder = Utf8StreamDecoder::new();
        assert_eq!(decoder.push(&[0xE3, 0x81]), "");
        assert_eq!(decoder.finish(), "\u{fffd}");
        assert!(!decoder.has_pending());
    }

    #[test]
    fn stop_buffer_without_stops_is_pass_through() {
        let mut buf = StopSequenceBuffer::new(Vec::new());
        let fed = buf.push("anything at all");
        assert_eq!(fed.emit, "anything at all");
        assert!(fed.matched.is_none());
    }

    #[test]
    fn stop_buffer_withholds_partial_match_across_tokens() {
        let mut buf = StopSequenceBuffer::new(vec!["\nUser:".to_string()]);
        assert_eq!(buf.push("hello").emit, "hello");
        // "\nUse" could still become "\nUser:" — nothing may be emitted yet.
        assert_eq!(buf.push("\nUse").emit, "");
        let fed = buf.push("r:");
        assert_eq!(fed.emit, "");
        assert_eq!(fed.matched.as_deref(), Some("\nUser:"));
    }

    #[test]
    fn stop_buffer_releases_when_partial_match_fails() {
        let mut buf = StopSequenceBuffer::new(vec!["STOP".to_string()]);
        assert_eq!(buf.push("ST").emit, "");
        // "STA" cannot become "STOP" any more.
        assert_eq!(buf.push("A").emit, "STA");
    }

    #[test]
    fn stop_buffer_emits_text_before_the_stop() {
        let mut buf = StopSequenceBuffer::new(vec!["END".to_string()]);
        let fed = buf.push("some text END trailing");
        assert_eq!(fed.emit, "some text ");
        assert_eq!(fed.matched.as_deref(), Some("END"));
    }

    #[test]
    fn stop_buffer_prefers_the_earliest_then_longest_match() {
        let mut buf = StopSequenceBuffer::new(vec!["ab".to_string(), "abc".to_string()]);
        let fed = buf.push("xxabc");
        assert_eq!(fed.emit, "xx");
        assert_eq!(fed.matched.as_deref(), Some("abc"));
    }

    #[test]
    fn stop_buffer_handles_multibyte_boundaries() {
        let mut buf = StopSequenceBuffer::new(vec!["世界".to_string()]);
        assert_eq!(buf.push("こんにちは").emit, "こんにちは");
        assert_eq!(buf.push("世").emit, "");
        let fed = buf.push("界");
        assert_eq!(fed.emit, "");
        assert_eq!(fed.matched.as_deref(), Some("世界"));
    }

    #[test]
    fn stop_buffer_flush_returns_held_text() {
        let mut buf = StopSequenceBuffer::new(vec!["STOP".to_string()]);
        assert_eq!(buf.push("ST").emit, "");
        assert_eq!(buf.flush(), "ST");
        assert_eq!(buf.flush(), "");
    }
}

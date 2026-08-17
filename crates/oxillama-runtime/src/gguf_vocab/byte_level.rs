// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPT-2 byte-level alphabet mapping.
//!
//! Byte-level BPE vocabularies (GGUF `tokenizer.ggml.model == "gpt2"`) never
//! store raw bytes.  Instead every one of the 256 possible bytes is mapped to a
//! *printable* Unicode code point so that the whole vocabulary is valid UTF-8
//! and can round-trip through JSON / GGUF string arrays.
//!
//! The mapping is the one introduced by OpenAI's GPT-2 `bytes_to_unicode()`:
//!
//! * bytes that are already printable ASCII/Latin-1 (`!`..`~`, `¡`..`¬`,
//!   `®`..`ÿ`) map to themselves,
//! * every remaining byte `b` maps to `U+0100 + n` where `n` counts the
//!   remaining bytes in ascending order.
//!
//! The most visible consequence is that a space becomes `Ġ` (U+0120) and a
//! newline becomes `Ċ` (U+010A).
//!
//! This module provides the forward table (byte → char) and the reverse table
//! (char → byte) so that a GGUF token string can be turned back into the exact
//! byte sequence it represents — which is what makes byte-exact streaming
//! detokenisation possible (see [`crate::stream_decode`]).

use std::collections::HashMap;
use std::sync::OnceLock;

/// Number of distinct byte values.
const N_BYTES: usize = 256;

/// Returns `true` when byte `b` is kept as itself by GPT-2's `bytes_to_unicode`.
fn is_printable_byte(b: u8) -> bool {
    // '!'..='~' | '¡'..='¬' | '®'..=0xFF
    (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE
}

/// Build the forward byte → code-point table.
fn build_forward() -> [char; N_BYTES] {
    let mut table = ['\0'; N_BYTES];
    let mut next_extra = 0u32;
    for (b, slot) in table.iter_mut().enumerate() {
        // `b` is bounded by 256 so the cast is lossless.
        let byte = b as u8;
        if is_printable_byte(byte) {
            // Latin-1 code points are always valid `char`s.
            *slot = char::from_u32(u32::from(byte)).unwrap_or('\u{fffd}');
        } else {
            *slot = char::from_u32(0x100 + next_extra).unwrap_or('\u{fffd}');
            next_extra += 1;
        }
    }
    table
}

/// The byte → code-point table used by GPT-2 style byte-level BPE.
pub fn byte_to_char_table() -> &'static [char; N_BYTES] {
    static TABLE: OnceLock<[char; N_BYTES]> = OnceLock::new();
    TABLE.get_or_init(build_forward)
}

/// The code-point → byte table used by GPT-2 style byte-level BPE.
pub fn char_to_byte_table() -> &'static HashMap<char, u8> {
    static TABLE: OnceLock<HashMap<char, u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let forward = byte_to_char_table();
        let mut map = HashMap::with_capacity(N_BYTES);
        for (b, &c) in forward.iter().enumerate() {
            // `b < 256` by construction.
            map.insert(c, b as u8);
        }
        map
    })
}

/// Encode a single byte to its byte-level code point.
pub fn byte_to_char(b: u8) -> char {
    byte_to_char_table()[usize::from(b)]
}

/// Encode a single byte to its byte-level representation as a `String`.
pub fn byte_to_string(b: u8) -> String {
    byte_to_char(b).to_string()
}

/// Encode raw text into the byte-level alphabet.
///
/// Every *byte* of `text` (not every char) is mapped through the table.
pub fn encode_bytes(text: &str) -> String {
    let table = byte_to_char_table();
    let mut out = String::with_capacity(text.len());
    for &b in text.as_bytes() {
        out.push(table[usize::from(b)]);
    }
    out
}

/// Decode a byte-level string back into the exact bytes it represents.
///
/// Code points that are not part of the byte-level alphabet are passed through
/// as their own UTF-8 encoding.  This mirrors llama.cpp's `llama_decode_text`,
/// which leaves unmapped characters untouched rather than failing — some
/// converted vocabularies contain user-defined tokens written in plain text.
pub fn decode_bytes(text: &str) -> Vec<u8> {
    let table = char_to_byte_table();
    let mut out = Vec::with_capacity(text.len());
    let mut buf = [0u8; 4];
    for ch in text.chars() {
        match table.get(&ch) {
            Some(&b) => out.push(b),
            None => out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_maps_to_g_dot() {
        assert_eq!(byte_to_char(b' '), 'Ġ');
    }

    #[test]
    fn newline_maps_to_c_dot() {
        assert_eq!(byte_to_char(b'\n'), 'Ċ');
    }

    #[test]
    fn ascii_printable_is_identity() {
        for b in b'!'..=b'~' {
            assert_eq!(byte_to_char(b), char::from(b));
        }
    }

    #[test]
    fn table_is_a_bijection() {
        let fwd = byte_to_char_table();
        let rev = char_to_byte_table();
        assert_eq!(rev.len(), 256, "reverse table must contain 256 entries");
        for (b, &c) in fwd.iter().enumerate() {
            assert_eq!(rev.get(&c).copied().map(usize::from), Some(b));
        }
    }

    #[test]
    fn roundtrip_utf8_text() {
        let samples = ["hello world", "こんにちは世界", "🚀 emoji", "\n\t mixed "];
        for s in samples {
            let encoded = encode_bytes(s);
            let decoded = decode_bytes(&encoded);
            assert_eq!(decoded, s.as_bytes(), "roundtrip failed for {s:?}");
        }
    }

    #[test]
    fn decode_partial_multibyte_yields_partial_bytes() {
        // "こ" is E3 81 93; the first two bytes alone are not valid UTF-8, yet
        // their byte-level spelling must decode back to exactly those bytes.
        let partial: String = [0xE3u8, 0x81].iter().map(|&b| byte_to_char(b)).collect();
        assert_eq!(partial, "ãģ");
        assert_eq!(decode_bytes(&partial), vec![0xE3, 0x81]);
        // Adding the third byte completes the character.
        let full: String = [0xE3u8, 0x81, 0x93]
            .iter()
            .map(|&b| byte_to_char(b))
            .collect();
        assert_eq!(String::from_utf8_lossy(&decode_bytes(&full)), "こ");
    }
}

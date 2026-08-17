// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the GGUF-embedded vocabulary loader.
//!
//! These build a [`MetadataStore`] directly rather than a GGUF file, so they
//! exercise exactly the code path `load_tokenizer` takes without needing any
//! model on disk.  Conformance against llama.cpp's own reference vocabularies
//! lives in `tests/gguf_tokenizer_fixtures.rs`.

use super::*;
use oxillama_gguf::{MetadataStore, MetadataValue};

fn str_array(values: &[&str]) -> MetadataValue {
    MetadataValue::Array(
        values
            .iter()
            .map(|v| MetadataValue::String((*v).to_string()))
            .collect(),
    )
}

fn i32_array(values: &[i32]) -> MetadataValue {
    MetadataValue::Array(values.iter().map(|v| MetadataValue::Int32(*v)).collect())
}

fn f32_array(values: &[f32]) -> MetadataValue {
    MetadataValue::Array(values.iter().map(|v| MetadataValue::Float32(*v)).collect())
}

// ── SentencePiece fixture ───────────────────────────────────────────────────

/// A tiny SentencePiece vocabulary with full byte fallback.
fn spm_metadata() -> MetadataStore {
    let mut tokens: Vec<String> = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    let mut types: Vec<i32> = vec![2, 3, 3];
    let mut scores: Vec<f32> = vec![0.0, 0.0, 0.0];
    // Full <0x00>..<0xFF> byte-fallback block, as real LLaMA vocabularies have.
    for b in 0u16..256 {
        tokens.push(format!("<0x{b:02X}>"));
        types.push(6);
        scores.push(0.0);
    }
    // Ordinary pieces.
    for (piece, score) in [
        ("\u{2581}", -1.0f32),
        ("\u{2581}h", -8.0),
        ("e", -3.0),
        ("l", -4.0),
        ("o", -5.0),
        ("\u{2581}he", -7.0),
        ("ll", -6.5),
        ("llo", -6.0),
        ("\u{2581}hello", -2.0),
        ("\u{2581}world", -2.5),
        ("\u{2581}w", -9.0),
    ] {
        tokens.push(piece.to_string());
        types.push(1);
        scores.push(score);
    }

    let mut md = MetadataStore::new();
    md.insert(
        KEY_MODEL.to_string(),
        MetadataValue::String("llama".to_string()),
    );
    md.insert(
        KEY_TOKENS.to_string(),
        MetadataValue::Array(
            tokens
                .iter()
                .map(|t| MetadataValue::String(t.clone()))
                .collect(),
        ),
    );
    md.insert(KEY_TOKEN_TYPE.to_string(), i32_array(&types));
    md.insert(KEY_SCORES.to_string(), f32_array(&scores));
    md.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        MetadataValue::Uint32(1),
    );
    md.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        MetadataValue::Uint32(2),
    );
    md.insert(
        "tokenizer.ggml.unknown_token_id".to_string(),
        MetadataValue::Uint32(0),
    );
    md.insert(
        "tokenizer.ggml.add_bos_token".to_string(),
        MetadataValue::Bool(true),
    );
    md
}

// ── byte-level BPE fixture ──────────────────────────────────────────────────

/// A byte-level BPE vocabulary containing the whole 256-byte alphabet, a few
/// merges, one control token and — deliberately — an ordinary token whose text
/// is `</s>`.
fn bpe_metadata(pre: Option<&str>) -> MetadataStore {
    let mut tokens: Vec<String> = Vec::new();
    let mut types: Vec<i32> = Vec::new();
    for b in 0u16..256 {
        // `b < 256`, so the cast is lossless.
        tokens.push(byte_level::byte_to_string(b as u8));
        types.push(1);
    }
    // Merge order *is* priority: rank 0 is applied first.  The `Ġthe` chain has
    // to outrank `h e`, otherwise " the" merges as "Ġt" + "he".
    let merges = [
        "Ġ t", "Ġt h", "Ġth e", // Ġthe
        "h e", "he l", "hel l", "hell o", // hello
        "ã ģ",    // first two bytes of "こ" (E3 81)
        "ãģ ĵ",   // complete "こ"
        "1 2", "12 3",
    ];
    for merged in [
        "he", "hel", "hell", "hello", "Ġt", "Ġth", "Ġthe", "ãģ", "ãģĵ", "12", "123",
    ] {
        tokens.push(merged.to_string());
        types.push(1);
    }
    // An ordinary token that *looks* like an EOS marker — the Qwen3 trap.
    tokens.push("</s>".to_string());
    types.push(1);
    // Real control tokens.
    tokens.push("<|im_start|>".to_string());
    types.push(3);
    tokens.push("<|im_end|>".to_string());
    types.push(3);

    let mut md = MetadataStore::new();
    md.insert(
        KEY_MODEL.to_string(),
        MetadataValue::String("gpt2".to_string()),
    );
    if let Some(pre) = pre {
        md.insert(KEY_PRE.to_string(), MetadataValue::String(pre.to_string()));
    }
    md.insert(
        KEY_TOKENS.to_string(),
        MetadataValue::Array(
            tokens
                .iter()
                .map(|t| MetadataValue::String(t.clone()))
                .collect(),
        ),
    );
    md.insert(KEY_TOKEN_TYPE.to_string(), i32_array(&types));
    md.insert(KEY_MERGES.to_string(), str_array(&merges));
    md
}

fn bpe_id(vocab: &GgufVocab, text: &str) -> u32 {
    vocab
        .token_to_id(text)
        .unwrap_or_else(|| panic!("test: token {text:?} must exist"))
}

// ── loading ─────────────────────────────────────────────────────────────────

#[test]
fn has_embedded_vocab_detects_the_token_array() {
    assert!(has_embedded_vocab(&spm_metadata()));
    assert!(!has_embedded_vocab(&MetadataStore::new()));
}

#[test]
fn spm_metadata_loads_as_sentencepiece() {
    let vocab = GgufVocab::from_metadata(&spm_metadata()).expect("test: must load");
    assert_eq!(vocab.vocab_type(), VocabType::Spm);
    assert_eq!(vocab.bos_id(), Some(1));
    assert_eq!(vocab.eos_id(), Some(2));
    assert!(vocab.add_bos());
}

#[test]
fn bpe_metadata_loads_as_byte_level_bpe() {
    let vocab =
        GgufVocab::from_metadata(&bpe_metadata(Some("llama-bpe"))).expect("test: must load");
    assert_eq!(vocab.vocab_type(), VocabType::Bpe);
    assert_eq!(vocab.pre_type(), PreType::Llama3);
}

#[test]
fn unknown_pre_falls_back_to_default() {
    let vocab =
        GgufVocab::from_metadata(&bpe_metadata(Some("not-a-real-one"))).expect("test: must load");
    assert_eq!(vocab.pre_type(), PreType::Default);
}

#[test]
fn missing_pre_without_llama3_markers_uses_default() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(None)).expect("test: must load");
    assert_eq!(vocab.pre_type(), PreType::Default);
}

#[test]
fn missing_pre_with_llama3_markers_selects_llama3() {
    let mut md = bpe_metadata(None);
    let mut tokens: Vec<String> = read_string_array(&md, KEY_TOKENS).expect("test: tokens present");
    let mut types: Vec<i32> = read_i32_array(&md, KEY_TOKEN_TYPE).expect("test: types present");
    tokens.push("<|begin_of_text|>".to_string());
    types.push(3);
    tokens.push("<|eot_id|>".to_string());
    types.push(3);
    md.insert(
        KEY_TOKENS.to_string(),
        MetadataValue::Array(
            tokens
                .iter()
                .map(|t| MetadataValue::String(t.clone()))
                .collect(),
        ),
    );
    md.insert(KEY_TOKEN_TYPE.to_string(), i32_array(&types));
    let vocab = GgufVocab::from_metadata(&md).expect("test: must load");
    assert_eq!(
        vocab.pre_type(),
        PreType::Llama3,
        "a GGUF converted before tokenizer.ggml.pre existed must still be detected"
    );
    assert!(vocab.add_bos(), "LLaMA-3 prepends BOS");
    assert_eq!(vocab.bos_id(), vocab.token_to_id("<|begin_of_text|>"));
}

#[test]
fn empty_vocabulary_is_rejected() {
    let mut md = MetadataStore::new();
    md.insert(KEY_TOKENS.to_string(), MetadataValue::Array(Vec::new()));
    assert!(GgufVocab::from_metadata(&md).is_err());
}

// ── EOS / EOG resolution ────────────────────────────────────────────────────

#[test]
fn ordinary_token_named_eos_is_not_elected() {
    // No eos_token_id in the metadata; the vocabulary contains a *normal* token
    // whose text is "</s>" plus a real control token "<|im_end|>".
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("qwen2"))).expect("test: must load");
    let fake_eos = bpe_id(&vocab, "</s>");
    assert!(
        !vocab.is_eog(fake_eos),
        "an ordinary token that merely spells </s> must not end generation"
    );
    let im_end = bpe_id(&vocab, "<|im_end|>");
    assert!(
        vocab.is_eog(im_end),
        "<|im_end|> is a control token and must end generation"
    );
    assert_eq!(vocab.eot_id(), Some(im_end));
}

#[test]
fn eos_token_id_from_metadata_is_authoritative() {
    let mut md = bpe_metadata(Some("qwen2"));
    let vocab = GgufVocab::from_metadata(&md).expect("test: must load");
    let im_end = bpe_id(&vocab, "<|im_end|>");
    md.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        MetadataValue::Uint32(im_end),
    );
    let vocab = GgufVocab::from_metadata(&md).expect("test: must load");
    assert_eq!(vocab.eos_id(), Some(im_end));
    assert!(vocab.is_eog(im_end));
}

#[test]
fn eog_set_contains_every_end_marker() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("qwen2"))).expect("test: must load");
    let im_end = bpe_id(&vocab, "<|im_end|>");
    assert!(vocab.eog_ids().contains(&im_end));
    assert!(!vocab.eog_ids().contains(&bpe_id(&vocab, "</s>")));
}

// ── SentencePiece behaviour ─────────────────────────────────────────────────

#[test]
fn spm_prefixes_a_space_and_merges_by_score() {
    let vocab = GgufVocab::from_metadata(&spm_metadata()).expect("test: must load");
    let ids = vocab.tokenize("hello", false, false);
    let hello = vocab
        .token_to_id("\u{2581}hello")
        .expect("test: ▁hello exists");
    assert_eq!(
        ids,
        vec![hello],
        "the highest-scoring merge covers the whole word"
    );
}

#[test]
fn spm_adds_bos_when_requested() {
    let vocab = GgufVocab::from_metadata(&spm_metadata()).expect("test: must load");
    let ids = vocab.tokenize("hello", true, false);
    assert_eq!(ids.first(), Some(&1));
}

#[test]
fn spm_byte_fallback_covers_unknown_characters() {
    let vocab = GgufVocab::from_metadata(&spm_metadata()).expect("test: must load");
    // 'こ' is not in the vocabulary; it must come out as three <0xNN> tokens.
    let ids = vocab.tokenize("こ", false, false);
    let bytes = vocab.detokenize_bytes(&ids, false);
    assert_eq!(
        String::from_utf8_lossy(&bytes).trim_start(),
        "こ",
        "byte fallback must reconstruct the character exactly"
    );
    for &id in &ids {
        if id == vocab.token_to_id("\u{2581}").unwrap_or(u32::MAX) {
            continue;
        }
        assert_eq!(
            vocab.token_type(id),
            Some(TokenType::Byte),
            "unknown characters must use <0xNN> byte tokens"
        );
    }
}

#[test]
fn spm_detokenize_unescapes_the_space_marker() {
    let vocab = GgufVocab::from_metadata(&spm_metadata()).expect("test: must load");
    let ids = vocab.tokenize("hello world", false, false);
    assert_eq!(vocab.detokenize(&ids, false), " hello world");
}

#[test]
fn spm_control_tokens_are_split_out_when_parsing_specials() {
    let vocab = GgufVocab::from_metadata(&spm_metadata()).expect("test: must load");
    let with = vocab.tokenize("</s>", false, true);
    assert_eq!(with, vec![2], "control token must be recognised verbatim");
    let without = vocab.tokenize("</s>", false, false);
    assert_ne!(
        without,
        vec![2],
        "with parse_special=false the text must be tokenized normally"
    );
}

// ── byte-level BPE behaviour ────────────────────────────────────────────────

#[test]
fn bpe_merges_by_rank() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("gpt-2"))).expect("test: must load");
    let ids = vocab.tokenize("hello", false, false);
    assert_eq!(ids, vec![bpe_id(&vocab, "hello")]);
}

#[test]
fn bpe_encodes_leading_space_as_g_dot() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("gpt-2"))).expect("test: must load");
    let ids = vocab.tokenize(" the", false, false);
    assert_eq!(ids, vec![bpe_id(&vocab, "Ġthe")]);
}

#[test]
fn bpe_roundtrips_exactly() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("gpt-2"))).expect("test: must load");
    for text in ["hello the", " the hello", "hello\nworld", "abc 123 xyz"] {
        let ids = vocab.tokenize(text, false, false);
        assert_eq!(
            vocab.detokenize(&ids, false),
            text,
            "round trip must be exact for {text:?}"
        );
    }
}

#[test]
fn bpe_roundtrips_multibyte_text() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("gpt-2"))).expect("test: must load");
    let text = "こんにちは世界 🚀";
    let ids = vocab.tokenize(text, false, false);
    assert_eq!(vocab.detokenize(&ids, false), text);
}

#[test]
fn bpe_control_tokens_are_never_split() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("qwen2"))).expect("test: must load");
    let ids = vocab.tokenize("hello<|im_end|>", false, true);
    assert_eq!(ids.last(), Some(&bpe_id(&vocab, "<|im_end|>")));
    assert_eq!(ids.len(), 2, "got {ids:?}");
}

#[test]
fn llama3_pre_splits_digits_into_groups_of_three() {
    let vocab =
        GgufVocab::from_metadata(&bpe_metadata(Some("llama-bpe"))).expect("test: must load");
    // "123" is a vocabulary entry; the LLaMA-3 pre-tokenizer keeps runs of at
    // most three digits together, so "1234" is "123" + "4".
    let ids = vocab.tokenize("1234", false, false);
    assert_eq!(ids, vec![bpe_id(&vocab, "123"), bpe_id(&vocab, "4")]);
}

#[test]
fn qwen2_pre_splits_digits_one_at_a_time() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("qwen2"))).expect("test: must load");
    let ids = vocab.tokenize("123", false, false);
    assert_eq!(
        ids,
        vec![
            bpe_id(&vocab, "1"),
            bpe_id(&vocab, "2"),
            bpe_id(&vocab, "3")
        ]
    );
}

#[test]
fn ignore_merges_takes_whole_pretoken_when_present() {
    let llama3 =
        GgufVocab::from_metadata(&bpe_metadata(Some("llama-bpe"))).expect("test: must load");
    assert!(llama3.ignore_merges_enabled());
    let gpt2 = GgufVocab::from_metadata(&bpe_metadata(Some("gpt-2"))).expect("test: must load");
    assert!(!gpt2.ignore_merges_enabled());
}

// ── byte-exact detokenization ───────────────────────────────────────────────

#[test]
fn token_bytes_preserve_partial_utf8_sequences() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("gpt-2"))).expect("test: must load");
    // "ãģ" is the byte-level spelling of E3 81 — the first two bytes of "こ".
    let partial = bpe_id(&vocab, "ãģ");
    assert_eq!(
        vocab.token_bytes(partial),
        Some([0xE3u8, 0x81].as_slice()),
        "a token holding half a character must report those exact bytes"
    );
    // Decoding it on its own is necessarily lossy — this is precisely why
    // streaming must go through the byte API.
    assert!(vocab.detokenize(&[partial], false).contains('\u{fffd}'));
}

#[test]
fn detokenize_bytes_of_a_split_character_reassembles() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("gpt-2"))).expect("test: must load");
    let first = bpe_id(&vocab, "ãģ");
    let second = bpe_id(&vocab, "ĵ");
    let bytes = vocab.detokenize_bytes(&[first, second], false);
    assert_eq!(String::from_utf8_lossy(&bytes), "こ");
}

#[test]
fn skip_special_drops_control_tokens_only() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("qwen2"))).expect("test: must load");
    let ids = vocab.tokenize("hello<|im_end|>", false, true);
    assert_eq!(vocab.detokenize(&ids, true), "hello");
    assert_eq!(vocab.detokenize(&ids, false), "hello<|im_end|>");
}

#[test]
fn vocab_bytes_is_id_indexed_and_complete() {
    let vocab = GgufVocab::from_metadata(&bpe_metadata(Some("gpt-2"))).expect("test: must load");
    let pairs = vocab.vocab_bytes();
    assert_eq!(pairs.len(), vocab.len());
    for (i, (id, _)) in pairs.iter().enumerate() {
        assert_eq!(*id as usize, i, "vocab_bytes must be id-ordered");
    }
}

// ── merge table parsing ─────────────────────────────────────────────────────

#[test]
fn merges_may_be_stored_as_pairs() {
    let mut md = bpe_metadata(Some("gpt-2"));
    md.insert(
        KEY_MERGES.to_string(),
        MetadataValue::Array(vec![MetadataValue::Array(vec![
            MetadataValue::String("h".to_string()),
            MetadataValue::String("e".to_string()),
        ])]),
    );
    let vocab = GgufVocab::from_metadata(&md).expect("test: must load");
    assert_eq!(vocab.merge_rank("h", "e"), Some(0));
}

#[test]
fn escape_and_unescape_are_inverse() {
    let text = "a b  c";
    assert_eq!(unescape_whitespace(&escape_whitespace(text)), text);
}

#[test]
fn byte_token_text_parses() {
    assert_eq!(parse_byte_token("<0x0A>"), Some(b'\n'));
    assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
    assert_eq!(parse_byte_token("not a byte token"), None);
}

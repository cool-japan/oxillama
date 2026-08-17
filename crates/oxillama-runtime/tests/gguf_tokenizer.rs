// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the GGUF-embedded tokenizer.
//!
//! Two groups:
//!
//! * **Self-contained** tests that build a vocabulary in memory and exercise
//!   the whole encode → stream-decode path, including the multi-byte UTF-8
//!   streaming case.  These always run.
//! * **Conformance** tests against llama.cpp's own vocabulary fixtures
//!   (`models/ggml-vocab-*.gguf` with their `.inp`/`.out` pairs).  Those files
//!   are not part of this repository, so the tests skip unless they are found.
//!   Point `OXILLAMA_VOCAB_FIXTURES` at a llama.cpp `models/` directory to run
//!   them; otherwise `$HOME/work/refs/llama.cpp/models` is tried.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use oxillama_gguf::{GgufModel, MetadataStore, MetadataValue};
use oxillama_runtime::gguf_vocab::{byte_level, GgufVocab};
use oxillama_runtime::stream_decode::Utf8StreamDecoder;
use oxillama_runtime::TokenizerBridge;

// ── in-memory fixtures ──────────────────────────────────────────────────────

/// A byte-level BPE vocabulary that deliberately splits `こ` across two tokens.
///
/// Real byte-level vocabularies do exactly this for rare CJK characters and
/// emoji, which is what makes per-token `decode()` produce `U+FFFD`.
fn split_cjk_metadata() -> MetadataStore {
    let mut tokens: Vec<String> = (0u16..256)
        // `b < 256`, so the cast is lossless.
        .map(|b| byte_level::byte_to_string(b as u8))
        .collect();
    let mut types: Vec<i32> = vec![1; tokens.len()];

    // Merged tokens that each hold *part* of a multi-byte character.
    // "こ" is E3 81 93, "ん" is E3 82 93, "🚀" is F0 9F 9A 80.
    for merged in ["ãģ", "ãģĵ", "ãĤ", "ãĤĵ", "ðŁ", "ðŁļ", "ðŁļĢ"] {
        tokens.push(merged.to_string());
        types.push(1);
    }
    let merges = [
        "ã ģ",
        "ãģ ĵ", // こ
        "ã Ĥ",
        "ãĤ ĵ", // ん
        "ð Ł",
        "ðŁ ļ",
        "ðŁļ Ģ", // 🚀
    ];

    let mut md = MetadataStore::new();
    md.insert(
        "tokenizer.ggml.model".to_string(),
        MetadataValue::String("gpt2".to_string()),
    );
    md.insert(
        "tokenizer.ggml.pre".to_string(),
        MetadataValue::String("gpt-2".to_string()),
    );
    md.insert(
        "tokenizer.ggml.tokens".to_string(),
        MetadataValue::Array(
            tokens
                .into_iter()
                .map(MetadataValue::String)
                .collect::<Vec<_>>(),
        ),
    );
    md.insert(
        "tokenizer.ggml.token_type".to_string(),
        MetadataValue::Array(types.into_iter().map(MetadataValue::Int32).collect()),
    );
    md.insert(
        "tokenizer.ggml.merges".to_string(),
        MetadataValue::Array(
            merges
                .iter()
                .map(|m| MetadataValue::String((*m).to_string()))
                .collect(),
        ),
    );
    md
}

/// Simulate the engine's streaming path: decode one token at a time through the
/// incremental detokenizer and collect everything the callback would see.
fn stream_tokens(vocab: &GgufVocab, ids: &[u32]) -> String {
    let mut decoder = Utf8StreamDecoder::new();
    let mut out = String::new();
    for &id in ids {
        let bytes = vocab.detokenize_bytes(&[id], false);
        out.push_str(&decoder.push(&bytes));
    }
    out.push_str(&decoder.finish());
    out
}

#[test]
fn streaming_japanese_text_never_yields_replacement_characters() {
    let vocab = GgufVocab::from_metadata(&split_cjk_metadata()).expect("vocabulary must load");
    let text = "こんにちは世界";
    let ids = vocab.tokenize(text, false, false);
    assert!(ids.len() > 1, "the text must span multiple tokens");

    // Per-token decode — the old behaviour — is provably lossy here.
    let naive: String = ids
        .iter()
        .map(|&id| vocab.detokenize(&[id], false))
        .collect();
    assert!(
        naive.contains('\u{fffd}'),
        "per-token decoding is expected to corrupt this text; got {naive:?}"
    );

    // Streaming through the incremental detokenizer is exact.
    let streamed = stream_tokens(&vocab, &ids);
    assert!(
        !streamed.contains('\u{fffd}'),
        "streamed output must not contain U+FFFD, got {streamed:?}"
    );
    assert_eq!(streamed, text);
}

#[test]
fn streaming_emoji_never_yields_replacement_characters() {
    let vocab = GgufVocab::from_metadata(&split_cjk_metadata()).expect("vocabulary must load");
    let text = "🚀 launch 🚀";
    let ids = vocab.tokenize(text, false, false);
    let streamed = stream_tokens(&vocab, &ids);
    assert!(
        !streamed.contains('\u{fffd}'),
        "streamed emoji must not contain U+FFFD, got {streamed:?}"
    );
    assert_eq!(streamed, text);
}

#[test]
fn streaming_mixed_scripts_roundtrip() {
    let vocab = GgufVocab::from_metadata(&split_cjk_metadata()).expect("vocabulary must load");
    for text in [
        "こんにちは、世界！ Hello 🚀 42",
        "ん",
        "あ",
        "The capital of France is Paris.",
    ] {
        let ids = vocab.tokenize(text, false, false);
        assert_eq!(stream_tokens(&vocab, &ids), text, "roundtrip for {text:?}");
    }
}

// ── llama.cpp conformance fixtures ──────────────────────────────────────────

/// Locate llama.cpp's vocabulary fixture directory, if the host has one.
fn fixture_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("OXILLAMA_VOCAB_FIXTURES") {
        let path = PathBuf::from(dir);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let path = Path::new(&home).join("work/refs/llama.cpp/models");
    path.is_dir().then_some(path)
}

/// Parse a `.inp`/`.out` fixture pair into `(text, expected_ids)`.
fn read_fixture(gguf: &Path) -> Option<Vec<(String, Vec<u32>)>> {
    let inp = std::fs::read_to_string(gguf.with_extension("gguf.inp")).ok()?;
    let out = std::fs::read_to_string(gguf.with_extension("gguf.out")).ok()?;

    const SEP: &str = "\n__ggml_vocab_test__\n";
    // Mirrors llama.cpp's `read_tests`: a file ending in the separator yields no
    // trailing empty case.  Getting this wrong makes every fixture "unparseable"
    // and the conformance test silently vacuous.
    let mut inputs: Vec<String> = Vec::new();
    let mut rest = inp.as_str();
    while !rest.is_empty() {
        match rest.find(SEP) {
            Some(pos) => {
                inputs.push(rest[..pos].to_string());
                rest = &rest[pos + SEP.len()..];
            }
            None => {
                inputs.push(rest.to_string());
                break;
            }
        }
    }
    let expected: Vec<Vec<u32>> = out
        .lines()
        .map(|line| {
            line.split_whitespace()
                .filter_map(|t| t.parse::<u32>().ok())
                .collect()
        })
        .collect();
    assert_eq!(
        inputs.len(),
        expected.len(),
        "fixture {} has {} inputs but {} expected outputs",
        gguf.display(),
        inputs.len(),
        expected.len()
    );
    Some(inputs.into_iter().zip(expected).collect())
}

/// Run one fixture, returning `(n_pass, n_total, first_failure)`.
fn run_fixture(path: &Path) -> Option<(usize, usize, Option<String>)> {
    let cases = read_fixture(path)?;
    let model = GgufModel::load(path.to_string_lossy().as_ref()).ok()?;
    let vocab = GgufVocab::from_metadata(&model.file.metadata).ok()?;

    let mut pass = 0usize;
    let mut first_failure = None;
    for (text, expected) in &cases {
        // llama.cpp's own harness uses add_special = parse_special = false.
        let got = vocab.tokenize(text, false, false);
        if &got == expected {
            pass += 1;
        } else if first_failure.is_none() {
            first_failure = Some(format!(
                "input {text:?}\n  expected {expected:?}\n  got      {got:?}"
            ));
        }
    }
    Some((pass, cases.len(), first_failure))
}

#[test]
fn llama_cpp_vocab_fixtures_match_exactly() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipping: no llama.cpp vocabulary fixtures found");
        return;
    };
    // Vocabularies whose tokenizer family this build implements.
    let names = [
        "ggml-vocab-llama-spm",
        "ggml-vocab-llama-bpe",
        "ggml-vocab-qwen2",
        "ggml-vocab-phi-3",
        "ggml-vocab-gpt-2",
        "ggml-vocab-mpt",
        "ggml-vocab-starcoder",
        "ggml-vocab-refact",
        "ggml-vocab-command-r",
        "ggml-vocab-falcon",
        "ggml-vocab-deepseek-coder",
        "ggml-vocab-deepseek-llm",
    ];
    let mut summary: HashMap<&str, (usize, usize)> = HashMap::new();
    let mut failures: Vec<String> = Vec::new();
    let mut ran = 0usize;

    for name in names {
        let path = dir.join(format!("{name}.gguf"));
        if !path.is_file() {
            continue;
        }
        let Some((pass, total, first_failure)) = run_fixture(&path) else {
            continue;
        };
        ran += 1;
        summary.insert(name, (pass, total));
        if let Some(detail) = first_failure {
            failures.push(format!("{name}: {pass}/{total} passed\n  {detail}"));
        }
    }

    if ran == 0 {
        eprintln!(
            "skipping: no usable vocabulary fixtures in {}",
            dir.display()
        );
        return;
    }
    let mut report: Vec<String> = summary
        .iter()
        .map(|(name, (pass, total))| format!("{name}: {pass}/{total}"))
        .collect();
    report.sort();
    eprintln!("llama.cpp vocab conformance:\n  {}", report.join("\n  "));

    assert!(
        failures.is_empty(),
        "tokenizer output diverges from llama.cpp:\n{}",
        failures.join("\n")
    );
}

#[test]
fn llama_cpp_vocab_fixtures_roundtrip_through_detokenizer() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipping: no llama.cpp vocabulary fixtures found");
        return;
    };
    for name in [
        "ggml-vocab-llama-bpe",
        "ggml-vocab-qwen2",
        "ggml-vocab-gpt-2",
    ] {
        let path = dir.join(format!("{name}.gguf"));
        if !path.is_file() {
            continue;
        }
        let Ok(model) = GgufModel::load(path.to_string_lossy().as_ref()) else {
            continue;
        };
        let vocab = GgufVocab::from_metadata(&model.file.metadata).expect("vocabulary must load");
        for text in [
            "The capital of France is Paris.",
            "こんにちは世界",
            "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated)",
            "нещо на Български",
            "Hello, y'all! How are you 😁 ?",
        ] {
            let ids = vocab.tokenize(text, false, false);
            assert_eq!(
                stream_tokens(&vocab, &ids),
                text,
                "{name} must round-trip {text:?}"
            );
        }
    }
}

#[test]
fn llama_cpp_vocab_fixtures_resolve_special_tokens() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipping: no llama.cpp vocabulary fixtures found");
        return;
    };
    let path = dir.join("ggml-vocab-llama-bpe.gguf");
    if !path.is_file() {
        return;
    }
    let model = GgufModel::load(path.to_string_lossy().as_ref()).expect("fixture must parse");
    let bridge =
        TokenizerBridge::from_gguf_metadata(&model.file.metadata).expect("vocabulary must load");
    assert_eq!(bridge.bos_token_id(), Some(128000), "<|begin_of_text|>");
    assert_eq!(bridge.eos_token_id(), Some(128001), "<|end_of_text|>");
    assert!(
        bridge.is_eog(128009),
        "<|eot_id|> (128009) must end generation for Llama-3"
    );
    assert!(
        bridge.add_bos(),
        "the llama3 pre-tokenizer implies add_bos = true"
    );
}

// ── real model checks (opt-in) ──────────────────────────────────────────────

/// Load a real model's tokenizer straight from its GGUF, with no sidecar.
fn bridge_for(env_key: &str) -> Option<TokenizerBridge> {
    let path = std::env::var(env_key).ok()?;
    if !Path::new(&path).is_file() {
        return None;
    }
    let model = GgufModel::load(&path).ok()?;
    TokenizerBridge::from_gguf_metadata(&model.file.metadata).ok()
}

#[test]
fn real_llama3_model_tokenizes_without_a_sidecar() {
    let Some(bridge) = bridge_for("OXILLAMA_TEST_LLAMA3_GGUF") else {
        eprintln!("skipping: set OXILLAMA_TEST_LLAMA3_GGUF to a Llama-3 GGUF to run");
        return;
    };
    assert!(bridge.is_gguf_backed());
    let ids = bridge
        .encode("The capital of France is")
        .expect("encode must succeed");
    assert_eq!(ids.first(), Some(&128000), "BOS must be prepended");
    let text = bridge.decode(&ids).expect("decode must succeed");
    assert_eq!(text, "The capital of France is");
    assert!(bridge.is_eog(128009), "<|eot_id|> must end generation");
}

#[test]
fn real_qwen3_model_resolves_im_end_as_eos() {
    let Some(bridge) = bridge_for("OXILLAMA_TEST_QWEN3_GGUF") else {
        eprintln!("skipping: set OXILLAMA_TEST_QWEN3_GGUF to a Qwen3 GGUF to run");
        return;
    };
    assert_eq!(
        bridge.eos_token_id(),
        Some(151645),
        "Qwen3's EOS is <|im_end|> = 151645, not the ordinary token 128247 that \
         merely spells </s>"
    );
    let text = "こんにちは世界";
    let ids = bridge.encode_raw(text).expect("encode must succeed");
    let decoded = bridge.decode(&ids).expect("decode must succeed");
    assert_eq!(decoded, text);
}

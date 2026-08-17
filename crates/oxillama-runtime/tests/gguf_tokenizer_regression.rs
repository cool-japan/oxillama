// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression test for "a stock HuggingFace GGUF cannot be loaded".
//!
//! Before 0.1.4, `InferenceEngine::load_model` failed on any GGUF that did not
//! happen to have a `tokenizer.json` sitting next to it:
//!
//! ```text
//! error: tokenizer error: no tokenizer found: provide --tokenizer path
//!        or place tokenizer.json next to the model file
//! ```
//!
//! …because the tokenizer loader looked for a metadata key
//! (`tokenizer.huggingface.json`) that no real GGUF contains, and never read the
//! `tokenizer.ggml.*` vocabulary that every real GGUF *does* contain.
//!
//! This file is deliberately written against the API that existed **before** the
//! fix, so it compiles against both revisions and the fail-before / pass-after
//! ordering can be demonstrated by reverting the source and re-running it.
//!
//! It needs a real model, so it is opt-in:
//!
//! ```sh
//! OXILLAMA_TEST_LLAMA3_GGUF=/path/to/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
//!   cargo nextest run -p oxillama-runtime --test gguf_tokenizer_regression
//! ```

use std::path::Path;

use oxillama_runtime::{EngineConfig, InferenceEngine};

/// Resolve an opt-in model path from the environment.
fn model_path(env_key: &str) -> Option<String> {
    let path = std::env::var(env_key).ok()?;
    Path::new(&path).is_file().then_some(path)
}

/// A model directory that has no `tokenizer.json` sidecar must still load.
#[test]
fn model_without_sidecar_loads_and_generates() {
    let Some(path) = model_path("OXILLAMA_TEST_LLAMA3_GGUF") else {
        eprintln!("skipping: set OXILLAMA_TEST_LLAMA3_GGUF to a Llama-3 GGUF to run");
        return;
    };
    let sidecar = Path::new(&path)
        .parent()
        .unwrap_or(Path::new("."))
        .join("tokenizer.json");
    assert!(
        !sidecar.exists(),
        "this regression test is only meaningful without a sidecar; found {}",
        sidecar.display()
    );

    let mut engine = InferenceEngine::new(EngineConfig {
        model_path: path,
        context_size: Some(256),
        ..EngineConfig::default()
    });
    engine
        .load_model()
        .expect("a stock GGUF must load using its embedded tokenizer.ggml.* vocabulary");

    let tokens = engine
        .tokenize("The capital of France is")
        .expect("tokenize must succeed");
    assert!(!tokens.is_empty(), "the prompt must produce tokens");
    assert_eq!(
        tokens.first(),
        Some(&128_000),
        "Llama-3 prepends <|begin_of_text|>, got {tokens:?}"
    );
    assert!(
        engine.is_eos(128_009),
        "<|eot_id|> must be recognised as end-of-generation"
    );
    assert!(
        !engine.is_eos(tokens[1]),
        "an ordinary prompt token must not be end-of-generation"
    );
}

/// A model directory that *does* have a sidecar must prefer the GGUF vocabulary
/// and therefore resolve the model's own EOS.
#[test]
fn model_with_sidecar_prefers_the_embedded_vocabulary() {
    let Some(path) = model_path("OXILLAMA_TEST_QWEN3_GGUF") else {
        eprintln!("skipping: set OXILLAMA_TEST_QWEN3_GGUF to a Qwen3 GGUF to run");
        return;
    };
    let mut engine = InferenceEngine::new(EngineConfig {
        model_path: path,
        context_size: Some(256),
        ..EngineConfig::default()
    });
    engine.load_model().expect("Qwen3 must load");

    assert!(
        engine.is_eos(151_645),
        "Qwen3 must stop on <|im_end|> (151645)"
    );
    assert!(
        !engine.is_eos(128_247),
        "128247 is an ordinary token that merely spells </s>; it must not stop generation"
    );
}

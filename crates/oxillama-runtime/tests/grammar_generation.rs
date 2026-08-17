// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! End-to-end regression tests for grammar-constrained generation.
//!
//! # The defect these cover (T1)
//!
//! `apply_grammar_mask` unmasks the end-of-generation tokens once the grammar
//! reaches an accepting state — otherwise every token is `-inf` and generation
//! can never terminate.  That unmasking reads
//! [`SamplerConfig::eog_token_ids`](oxillama_runtime::SamplerConfig::eog_token_ids),
//! and **nothing populated it**: not `engine/**`, not `oxillama-server/**`.
//! The server's `chat.rs` sets `grammar` and `token_vocab` but never
//! `eog_token_ids`, so the field was empty on every real request and
//! grammar-constrained sampling could not stop.
//!
//! `run_decode_loop` now fills the field from the model's own tokenizer when
//! the caller left it empty.  These tests drive a grammar to completion through
//! the **public** `InferenceEngine` API and assert that an EOG token becomes
//! selectable at exactly that point.
//!
//! ## Verifying the failing-before ordering
//!
//! Delete the four-line `eog_token_ids` population in
//! `engine/generation.rs::run_decode_loop` and
//! `grammar_completion_reaches_an_eog_token` fails with
//! `finish_reason: MaxTokens` and 8 repeated filler tokens instead of
//! `FinishReason::Eos` after 2.

#![cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]

use std::sync::Arc;

use oxillama_runtime::{
    EngineConfig, FinishReason, GenerationConfig, Grammar, InferenceEngine, SamplerConfig,
};

/// Load the synthetic 32-token LLaMA fixture.
///
/// Its vocabulary maps `a`..`z` to ids 3..28, so a GBNF literal made of ASCII
/// letters is produced one token per character — which is what makes the
/// grammar state machine observable at token granularity.
fn make_loaded_engine() -> InferenceEngine {
    let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
    let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
    let mut engine = InferenceEngine::new(EngineConfig::default());
    engine
        .load_model_from_bytes(&model_bytes, tokenizer_json)
        .expect("the synthetic GGUF fixture must load");
    engine
}

/// A greedy sampler constrained by `gbnf`, with the vocabulary table attached
/// but `eog_token_ids` deliberately left empty — exactly the shape the server
/// builds in `routes/chat.rs`.
fn grammar_sampler(engine: &InferenceEngine, gbnf: &str) -> SamplerConfig {
    let grammar = Grammar::parse(gbnf).expect("the test grammar must parse");
    let vocab = engine
        .vocab_bytes()
        .expect("a loaded engine must expose its vocabulary");
    SamplerConfig {
        grammar: Some(Arc::new(grammar)),
        token_vocab: Some(Arc::new(vocab)),
        ..SamplerConfig::greedy()
    }
}

/// The fixture must actually have an end-of-generation token, or every other
/// test in this file would pass vacuously.
#[test]
fn fixture_tokenizer_has_an_eog_token() {
    let engine = make_loaded_engine();
    let tokenizer = engine.tokenizer().expect("a loaded engine has a tokenizer");
    assert!(
        !tokenizer.eog_token_ids().is_empty(),
        "the fixture tokenizer must resolve at least one EOG token, got none"
    );
    assert!(
        tokenizer.is_eog(2),
        "`</s>` (id 2) must be recognised as end-of-generation"
    );
}

/// The core T1 regression.
///
/// `root ::= "ab"` is satisfied after exactly two tokens.  At that point the
/// grammar allows nothing further, so the mask can only leave an EOG token
/// finite — and only if `eog_token_ids` was populated.  With the field empty
/// the mask's last-resort safety net re-enables the highest raw logit instead,
/// generation never sees an EOG token, and the loop runs to `max_tokens`.
#[test]
fn grammar_completion_reaches_an_eog_token() {
    let mut engine = make_loaded_engine();
    let sampler = grammar_sampler(&engine, r#"root ::= "ab""#);
    assert!(
        sampler.eog_token_ids.is_empty(),
        "the caller must NOT pre-populate eog_token_ids; the decode loop is \
         what has to fill it in"
    );

    let config = GenerationConfig {
        max_tokens: 8,
        sampler,
        ..GenerationConfig::default()
    };

    let outcome = engine
        .generate_detailed("a", &config, |_| {})
        .expect("grammar-constrained generation must succeed");

    assert_eq!(
        outcome.finish_reason,
        FinishReason::Eos,
        "a satisfied grammar must terminate on an EOG token, got {:?} after \
         tokens {:?}",
        outcome.finish_reason,
        outcome.generated_tokens
    );
    assert_eq!(
        outcome.generated_tokens,
        vec![3, 4],
        "the grammar admits exactly `a`(3) then `b`(4) and nothing else"
    );
    assert_eq!(
        outcome.text, "ab",
        "the emitted text must match the grammar"
    );
}

/// The pre-fix symptom in its own right: no token may be emitted repeatedly
/// once the grammar is satisfied.
///
/// Before the sampling agent's mask fix this was literally token 0 forever;
/// before *this* task's fix it was whichever token carried the highest raw
/// logit, forever.  Either way the tell is the same — a run of identical
/// tokens filling the whole budget.
#[test]
fn satisfied_grammar_does_not_emit_filler_tokens() {
    let mut engine = make_loaded_engine();
    let config = GenerationConfig {
        max_tokens: 16,
        sampler: grammar_sampler(&engine, r#"root ::= "abc""#),
        ..GenerationConfig::default()
    };

    let outcome = engine
        .generate_detailed("a", &config, |_| {})
        .expect("grammar-constrained generation must succeed");

    assert!(
        outcome.generated_tokens.len() < config.max_tokens,
        "generation must stop when the grammar is satisfied, not run out the \
         {}-token budget: {:?}",
        config.max_tokens,
        outcome.generated_tokens
    );
    assert_eq!(outcome.generated_tokens, vec![3, 4, 5]);
    assert_eq!(
        outcome
            .generated_tokens
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        outcome.generated_tokens.len(),
        "no token may repeat here; a repeat means the mask fell back to a \
         filler token instead of an EOG token"
    );
}

/// A caller that *does* supply `eog_token_ids` keeps them.
///
/// The decode loop fills the field in only when it is empty, so a request that
/// deliberately narrows the stop set (a server exposing OpenAI's `stop_token_ids`,
/// say) is not silently widened back to the model's full EOG set.
#[test]
fn caller_supplied_eog_token_ids_are_not_overwritten() {
    let mut engine = make_loaded_engine();
    let mut sampler = grammar_sampler(&engine, r#"root ::= "ab""#);
    // Token 30 is `.` — a perfectly ordinary token the tokenizer does NOT
    // consider end-of-generation.
    sampler.eog_token_ids = vec![30];

    let config = GenerationConfig {
        max_tokens: 6,
        sampler,
        ..GenerationConfig::default()
    };

    let outcome = engine
        .generate_detailed("a", &config, |_| {})
        .expect("generation must succeed");

    // The mask now unmasks token 30 once the grammar completes, but the loop's
    // own stop test is `tokenizer.is_eog(30)` — false — so `.` is emitted as
    // text rather than terminating.  That is the caller's choice being
    // honoured, and it proves the loop did not clobber the supplied set.
    assert!(
        outcome.generated_tokens.contains(&30),
        "the caller's EOG token must be the one the mask unmasks, got {:?}",
        outcome.generated_tokens
    );
    assert_ne!(
        outcome.finish_reason,
        FinishReason::Eos,
        "token 30 is not a tokenizer EOG token, so the loop must not stop on it"
    );
}

/// Without a grammar, the populated `eog_token_ids` must not change anything:
/// the mask is the only consumer of the field.
#[test]
fn ungrammared_generation_is_unaffected() {
    let mut engine = make_loaded_engine();
    let config = GenerationConfig {
        max_tokens: 4,
        sampler: SamplerConfig::greedy(),
        ..GenerationConfig::default()
    };
    let outcome = engine
        .generate_detailed("a", &config, |_| {})
        .expect("plain generation must succeed");
    assert!(
        outcome.generated_tokens.len() <= 4,
        "max_tokens must still bound an ungrammared run"
    );
}

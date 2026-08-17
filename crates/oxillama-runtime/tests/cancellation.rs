// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! End-to-end regression tests for cooperative cancellation of the decode loop.
//!
//! # The defect these cover
//!
//! `oxillama-py` shipped a `CancellationToken` whose `cancel()` did not
//! shorten wall-clock generation time by a single forward pass. The decode
//! loop (`run_decode_loop`) had no cancellation check at all and its
//! per-token callback is infallible (`impl FnMut(&str)`, no way to signal
//! "stop"), so `generate(..., max_tokens=2048, cancel_token=t)` cancelled at
//! token 1 still ran all 2048 forward passes; only *afterwards* did the
//! Python layer convert the already-finished generation into a raised
//! `RuntimeError("generation cancelled")`.
//!
//! [`GenerationConfig::cancel_flag`] is the fix: an optional
//! `Arc<AtomicBool>` checked once per decode iteration, ahead of the
//! `max_tokens` and context-length checks, that ends generation with
//! [`FinishReason::Cancelled`] and returns the text produced so far.
//!
//! ## Verifying the failing-before ordering
//!
//! Delete the `cancel_flag` check at the top of `run_decode_loop`'s loop in
//! `engine/generation.rs` and `cancelling_mid_generation_stops_early` fails
//! with `finish_reason: MaxTokens` and the full `MAX_TOKENS` budget of
//! generated tokens instead of stopping at `CANCEL_AFTER`.

#![cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use oxillama_runtime::{
    EngineConfig, FinishReason, GenerationConfig, InferenceEngine, SamplerConfig,
};

/// Generous budget: every assertion below is "stopped well before this".
const MAX_TOKENS: usize = 64;
/// Cancel once this many tokens have reached the caller's callback.
const CANCEL_AFTER: usize = 3;

/// Load the synthetic 32-token LLaMA fixture (no external model file needed).
fn make_loaded_engine() -> InferenceEngine {
    let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
    let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
    let mut engine = InferenceEngine::new(EngineConfig::default());
    engine
        .load_model_from_bytes(&model_bytes, tokenizer_json)
        .expect("the synthetic GGUF fixture must load");
    engine
}

/// Greedy, and with `render_special` on.
///
/// The fixture's greedy trajectory is token 0 (`<unk>`) forever, which the
/// default `render_special = false` swallows — the stream callback would then
/// never fire and a test that raises the cancel flag *from* that callback
/// would silently degrade into "never cancelled at all". Rendering control
/// tokens makes the per-token callback observable, which is exactly what
/// these tests need to hook.
fn greedy_config(max_tokens: usize) -> GenerationConfig {
    GenerationConfig::new(max_tokens)
        .with_render_special(true)
        .with_sampler(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        })
}

/// Baseline: with no cancellation the fixture runs the *whole* `max_tokens`
/// budget. Without this, the cancellation assertions below could pass
/// vacuously on a model that stops early on its own.
#[test]
fn uncancelled_generation_runs_the_full_budget() {
    let mut engine = make_loaded_engine();
    engine.reset();
    let outcome = engine
        .generate_detailed("hello", &greedy_config(MAX_TOKENS), |_| {})
        .expect("generation must succeed");

    assert_eq!(
        outcome.finish_reason,
        FinishReason::MaxTokens,
        "the fixture must not terminate on its own, or the cancellation tests \
         below prove nothing (got {} after {} tokens)",
        outcome.finish_reason,
        outcome.completion_tokens()
    );
    assert_eq!(outcome.completion_tokens(), MAX_TOKENS);
}

/// The core regression: cancelling from inside the stream callback (i.e. from
/// "another thread's point of view", mid-decode) must stop the loop at the
/// next token boundary — not after `max_tokens` more forward passes.
#[test]
fn cancelling_mid_generation_stops_early() {
    let mut engine = make_loaded_engine();
    engine.reset();

    let flag = Arc::new(AtomicBool::new(false));
    let flag_cb = Arc::clone(&flag);
    let mut seen = 0usize;

    let config = greedy_config(MAX_TOKENS).with_cancel_flag(Arc::clone(&flag));
    let outcome = engine
        .generate_detailed("hello", &config, move |_chunk| {
            seen += 1;
            if seen >= CANCEL_AFTER {
                flag_cb.store(true, Ordering::Relaxed);
            }
        })
        .expect("a cancelled generation returns Ok with what it produced, not an error");

    assert_eq!(
        outcome.finish_reason,
        FinishReason::Cancelled,
        "cancellation must be reported as such, not laundered into {}",
        outcome.finish_reason
    );
    assert!(
        outcome.completion_tokens() <= CANCEL_AFTER + 1,
        "generation must stop at the next token boundary after the flag is \
         raised; produced {} tokens (budget was {MAX_TOKENS})",
        outcome.completion_tokens()
    );
    assert!(
        outcome.completion_tokens() < MAX_TOKENS,
        "the whole point is not burning the remaining budget"
    );
}

/// A token cancelled before the call even starts must cost zero forward
/// passes, not one full generation.
#[test]
fn pre_cancelled_generation_emits_nothing() {
    let mut engine = make_loaded_engine();
    engine.reset();

    let flag = Arc::new(AtomicBool::new(true));
    let mut emitted = String::new();
    let config = greedy_config(MAX_TOKENS).with_cancel_flag(flag);
    let outcome = engine
        .generate_detailed("hello", &config, |chunk| emitted.push_str(chunk))
        .expect("a pre-cancelled generation still returns Ok");

    assert_eq!(outcome.finish_reason, FinishReason::Cancelled);
    assert_eq!(
        outcome.completion_tokens(),
        0,
        "an already-raised flag must be observed before the first sample"
    );
    assert!(emitted.is_empty(), "nothing should have been streamed");
}

/// An attached but never-raised flag must not change behaviour at all — the
/// hook has to be inert unless actually used.
#[test]
fn unraised_cancel_flag_does_not_alter_the_outcome() {
    let mut engine = make_loaded_engine();

    engine.reset();
    let plain = engine
        .generate_detailed("hello", &greedy_config(8), |_| {})
        .expect("baseline generation");

    engine.reset();
    let flag = Arc::new(AtomicBool::new(false));
    let with_flag = engine
        .generate_detailed("hello", &greedy_config(8).with_cancel_flag(flag), |_| {})
        .expect("generation with an unraised flag");

    assert_eq!(plain.finish_reason, with_flag.finish_reason);
    assert_eq!(plain.generated_tokens, with_flag.generated_tokens);
    assert_eq!(plain.text, with_flag.text);
}

/// Cancellation must reach the token-array entry point too — the Python
/// bindings and the CLI both go through `generate_detailed_with_tokens` for
/// prompts they have already tokenized themselves.
#[test]
fn cancellation_applies_to_the_pretokenized_entry_point() {
    let mut engine = make_loaded_engine();
    engine.reset();
    let prompt_tokens = engine.tokenize("hello").expect("tokenize must succeed");

    let flag = Arc::new(AtomicBool::new(false));
    let flag_cb = Arc::clone(&flag);
    let mut seen = 0usize;
    let config = greedy_config(MAX_TOKENS).with_cancel_flag(flag);

    let outcome = engine
        .generate_detailed_with_tokens(prompt_tokens, &config, move |_| {
            seen += 1;
            if seen >= CANCEL_AFTER {
                flag_cb.store(true, Ordering::Relaxed);
            }
        })
        .expect("generation must succeed");

    assert_eq!(outcome.finish_reason, FinishReason::Cancelled);
    assert!(outcome.completion_tokens() <= CANCEL_AFTER + 1);
}

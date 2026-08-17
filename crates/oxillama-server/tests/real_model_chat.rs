// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Real-model regression tests for `POST /v1/chat/completions`.
//!
//! # The defect these cover
//!
//! `routes/chat.rs::format_chat_prompt` used to render every request through
//! a fabricated, model-agnostic skeleton:
//!
//! ```text
//! <|system|>\n{content}\n<|end|>\n
//! <|user|>\n{content}\n<|end|>\n
//! <|assistant|>\n
//! ```
//!
//! No model was trained on those markers. Reproduced on Qwen3 (whose real
//! markers are ChatML's `<|im_start|>`/`<|im_end|>`), a request for a one
//! sentence greeting came back as `"content": "Hello! <|end|#>"` — the
//! fabricated `<|end|>` tokenized as ordinary text, and the model, never
//! having seen that framing, imitated it back as literal output.
//!
//! Unit tests over the render function cannot catch this class of bug on
//! their own: the failure is *the model's reaction* to a prompt shape, which
//! only a real checkpoint exhibits. So these load a real GGUF, serve it over
//! a real ephemeral socket, and assert on what the model actually says.
//!
//! # Running
//!
//! Skipped (with a printed notice) unless the GGUF env vars are set:
//!
//! ```text
//! OXILLAMA_SERVER_LLAMA3_GGUF=/path/to/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
//! OXILLAMA_SERVER_QWEN3_GGUF=/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
//!   cargo nextest run -p oxillama-server --test real_model_chat --no-capture
//! ```
//!
//! Each case loads a multi-GB checkpoint and decodes tokens on CPU, so expect
//! minutes rather than seconds.

mod common;

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

use common::{post_json, spawn_server, unique_temp_dir};

use oxillama_runtime::{EngineConfig, InferenceEngine, PrefixCacheConfig};
use oxillama_server::{
    build_app, spawn_inference_worker, AppState, PrefixCacheRegistry, DEFAULT_MAX_NAMESPACES,
};

/// One served checkpoint.
struct Case {
    /// Human-readable name used in assertion messages.
    id: &'static str,
    /// Environment variable naming the GGUF file.
    env: &'static str,
    /// The chat template family this checkpoint must be detected as. Asserted
    /// rather than assumed: if detection regresses, every other assertion in
    /// the case would still pass while the model silently got the wrong
    /// prompt shape.
    expected_template: &'static str,
}

const CASES: &[Case] = &[
    Case {
        id: "Qwen3-4B-Instruct",
        env: "OXILLAMA_SERVER_QWEN3_GGUF",
        expected_template: "chatml",
    },
    Case {
        id: "Meta-Llama-3-8B-Instruct",
        env: "OXILLAMA_SERVER_LLAMA3_GGUF",
        expected_template: "llama3",
    },
];

/// Load `model_path`, serve it on an ephemeral port, POST one chat request,
/// and return `(detected_template, response_json)`.
async fn serve_and_chat(model_path: &str, body: &str) -> (String, serde_json::Value) {
    let config = EngineConfig {
        model_path: model_path.to_string(),
        context_size: Some(512),
        ..EngineConfig::default()
    };
    let mut engine = InferenceEngine::new(config);
    engine.load_model().expect("the checkpoint must load");

    let cached_sampler = engine.config().sampler.clone();
    let hidden_size = engine.hidden_size().unwrap_or(0);
    let vocab_bytes = engine.vocab_bytes().map(Arc::new);
    let chat_template = engine
        .chat_template()
        .expect("a loaded engine must resolve a chat template");

    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let registry = Arc::new(PrefixCacheRegistry::new(
        PrefixCacheConfig::default(),
        DEFAULT_MAX_NAMESPACES,
    ));
    let worker_alive = Arc::new(AtomicBool::new(false));
    let loras = Arc::new(RwLock::new(HashMap::new()));
    spawn_inference_worker(
        engine,
        rx,
        "test-model".to_string(),
        Arc::clone(&registry),
        loras,
        Arc::clone(&worker_alive),
    );

    let state = AppState::new(
        tx,
        "test-model".to_string(),
        cached_sampler,
        vocab_bytes,
        hidden_size,
        chat_template,
        registry,
        worker_alive,
        Some(unique_temp_dir("real_model_spool")),
    )
    .expect("AppState::new should succeed");

    let addr = spawn_server(build_app(Arc::new(state))).await;
    let response = post_json(addr, "/v1/chat/completions", body).await;
    assert_eq!(
        response.status,
        200,
        "chat request failed: {}",
        String::from_utf8_lossy(&response.body)
    );
    let json: serde_json::Value =
        serde_json::from_slice(&response.body).expect("response must be JSON");
    (chat_template.to_string(), json)
}

/// The headline regression: a real model's reply must contain no control-token
/// text at all. Before the fix, Qwen3 answered `"Hello! <|end|#>"`.
#[tokio::test(flavor = "multi_thread")]
async fn chat_reply_contains_no_control_token_text() {
    let mut ran = 0usize;
    for case in CASES {
        let Ok(model) = std::env::var(case.env) else {
            continue;
        };
        if !std::path::Path::new(&model).exists() {
            eprintln!("{} points at a missing file ({model}); skipping", case.env);
            continue;
        }

        let body = r#"{"model":"test-model",
                       "messages":[{"role":"user",
                                    "content":"Say hello in one short sentence."}],
                       "max_tokens":64,"temperature":0}"#;
        let (template, json) = serve_and_chat(&model, body).await;

        assert_eq!(
            template, case.expected_template,
            "{}: chat template detection regressed",
            case.id
        );

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default();
        eprintln!("[{}] template={template} content={content:?}", case.id);

        assert!(
            !content.is_empty(),
            "{}: the model produced no content: {json}",
            case.id
        );
        assert!(
            !content.contains("<|"),
            "{}: reply leaked control-token text — the model is being prompted \
             in a format it was not trained on: {content:?}",
            case.id
        );
        // The specific fabricated markers, named so a failure is unambiguous.
        for fake in ["<|end|", "<|system|", "<|user|", "<|assistant|", "<|im_"] {
            assert!(
                !content.contains(fake),
                "{}: reply contains {fake}: {content:?}",
                case.id
            );
        }
        assert_eq!(
            json["choices"][0]["finish_reason"].as_str(),
            Some("stop"),
            "{}: a 64-token budget should be ample for one sentence, so the \
             model should have stopped on its own EOG token: {json}",
            case.id
        );

        ran += 1;
    }

    if ran == 0 {
        eprintln!(
            "skipped: set OXILLAMA_SERVER_QWEN3_GGUF / OXILLAMA_SERVER_LLAMA3_GGUF \
             to real checkpoints to run this test"
        );
    }
}

/// `finish_reason` must be `"length"` — not `"stop"` — when a real generation
/// is actually truncated by `max_tokens`.
#[tokio::test(flavor = "multi_thread")]
async fn chat_reports_length_when_truncated_by_max_tokens() {
    let mut ran = 0usize;
    for case in CASES {
        let Ok(model) = std::env::var(case.env) else {
            continue;
        };
        if !std::path::Path::new(&model).exists() {
            continue;
        }

        // Four tokens cannot complete a paragraph, so this must truncate.
        let body = r#"{"model":"test-model",
                       "messages":[{"role":"user",
                                    "content":"Write a long paragraph about the sea."}],
                       "max_tokens":4,"temperature":0}"#;
        let (_template, json) = serve_and_chat(&model, body).await;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default();
        eprintln!(
            "[{}] truncated content={content:?} finish_reason={:?}",
            case.id, json["choices"][0]["finish_reason"]
        );
        assert_eq!(
            json["choices"][0]["finish_reason"].as_str(),
            Some("length"),
            "{}: a generation cut off at max_tokens must not claim the model \
             chose to stop: {json}",
            case.id
        );
        ran += 1;
    }

    if ran == 0 {
        eprintln!(
            "skipped: set OXILLAMA_SERVER_QWEN3_GGUF / OXILLAMA_SERVER_LLAMA3_GGUF \
             to real checkpoints to run this test"
        );
    }
}

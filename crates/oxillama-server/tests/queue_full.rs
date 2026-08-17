//! D5 regression (end-to-end): when the inference request queue is at
//! capacity, a new request must be rejected with `429 Too Many Requests`
//! (load shedding via `try_send`), not accepted and left to park the
//! caller indefinitely behind a `.send(...).await` that can never make
//! progress.
//!
//! The queue is filled *directly* (bypassing HTTP) before the server ever
//! starts, so there is no timing race to win: the single HTTP request this
//! test makes is guaranteed to observe a full channel.

mod common;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use oxillama_runtime::sampling::SamplerConfig;
use oxillama_runtime::ChatTemplate;
use oxillama_server::{AppState, BatchRequest, PrefixCacheRegistry, DEFAULT_MAX_NAMESPACES};

use common::{post_json, spawn_server};

#[tokio::test]
async fn chat_completions_returns_429_when_queue_is_full() {
    // Capacity-1 channel, and we deliberately never spawn a consumer —
    // `_rx` is kept alive (so the channel stays "open", i.e. sends fail
    // with `Full` rather than `Closed`) but nothing ever calls `.recv()`.
    let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(1);

    // Occupy the one slot directly, before starting the server, so the
    // single real HTTP request below deterministically hits a full queue.
    let (dummy_reply_tx, _dummy_reply_rx) =
        tokio::sync::oneshot::channel::<oxillama_server::queue::GenerateReply>();
    tx.try_send(BatchRequest::Generate {
        prompt: "occupy the only queue slot".to_string(),
        max_tokens: 1,
        config: SamplerConfig::default(),
        cache_prompt: false,
        lora_selection: vec![],
        add_special: true,
        reply: dummy_reply_tx,
    })
    .expect("filling the capacity-1 queue directly must succeed");

    let registry = Arc::new(PrefixCacheRegistry::new(
        Default::default(),
        DEFAULT_MAX_NAMESPACES,
    ));
    let worker_alive = Arc::new(AtomicBool::new(true));
    let spool_dir = common::unique_temp_dir("queue_full_spool");

    let state = AppState::new(
        tx,
        "test-model".to_string(),
        SamplerConfig::default(),
        None,
        0,
        ChatTemplate::default(),
        registry,
        worker_alive,
        Some(spool_dir),
    )
    .expect("AppState::new should succeed");

    let app = oxillama_server::build_app(Arc::new(state));
    let addr = spawn_server(app).await;

    let body = r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}]}"#;
    let resp = post_json(addr, "/v1/chat/completions", body).await;

    assert_eq!(
        resp.status,
        429,
        "a request against an already-full queue must be shed with 429, got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
    assert!(
        resp.headers.to_lowercase().contains("retry-after"),
        "429 response should carry a retry-after header: {}",
        resp.headers
    );
}

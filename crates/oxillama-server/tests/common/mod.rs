//! Shared test infrastructure for oxillama-server integration tests.
//!
//! These tests boot a *real* axum server on an ephemeral loopback port
//! (via `axum::serve`) rather than exercising the router in-process with
//! `tower::ServiceExt::oneshot`. This matters specifically for D1: admin
//! auth's `is_loopback` check reads the genuine TCP peer address from the
//! `ConnectInfo<SocketAddr>` extension, which only axum's
//! `into_make_service_with_connect_info` populates for a real accepted
//! connection — a `oneshot`-based unit test cannot exercise that path
//! end-to-end the way a real socket can.
//!
//! No HTTP client dependency is added to the workspace for this — a
//! minimal raw HTTP/1.1 client (`http_request`) is implemented directly
//! over `tokio::net::TcpStream`, sending `Connection: close` so the
//! response can simply be read to EOF without a proper chunked/
//! content-length body parser.
//!
//! `#![allow(dead_code)]`: each `tests/*.rs` file compiles this module
//! into its own separate binary crate and only uses a subset of the
//! helpers below — the unused ones in any given binary are not actually
//! dead across the test suite as a whole.
#![allow(dead_code)]

use std::net::SocketAddr;

use axum::Router;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Bind `app` to an ephemeral loopback port and serve it in the
/// background, with real `ConnectInfo<SocketAddr>` propagation (required
/// for the admin-auth loopback check to see genuine peer addresses).
///
/// Returns the bound address. The server task is intentionally not
/// awaited or explicitly shut down — it is aborted when the `#[tokio::test]`
/// runtime is torn down at the end of the test function.
pub async fn spawn_server(app: Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("read local_addr");

    tokio::spawn(async move {
        let make_service = app.into_make_service_with_connect_info::<SocketAddr>();
        let _ = axum::serve(listener, make_service).await;
    });

    addr
}

/// Result of a raw HTTP request: status code and raw response body bytes.
pub struct RawResponse {
    pub status: u16,
    pub headers: String,
    pub body: Vec<u8>,
}

/// Perform a minimal raw HTTP/1.1 request against `addr` and return the
/// parsed status code, header block (raw, unparsed), and body.
///
/// Always sends `Connection: close` so the server closes the connection
/// once the response is fully written, letting us just `read_to_end`
/// rather than implement content-length/chunked framing.
pub async fn http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    extra_headers: &[(&str, &str)],
    body: &[u8],
) -> RawResponse {
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to test server");

    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));

    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request headers");
    if !body.is_empty() {
        stream.write_all(body).await.expect("write request body");
    }
    stream.flush().await.expect("flush request");

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .expect("read response to EOF");

    let split_at = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
        .unwrap_or(raw.len());
    let (head, body) = raw.split_at(split_at);
    let head_text = String::from_utf8_lossy(head).to_string();

    let status = head_text
        .lines()
        .next()
        .and_then(|status_line| status_line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .unwrap_or(0);

    RawResponse {
        status,
        headers: head_text,
        body: body.to_vec(),
    }
}

/// Convenience wrapper: GET request, no extra headers, no body.
pub async fn get(addr: SocketAddr, path: &str) -> RawResponse {
    http_request(addr, "GET", path, &[], &[]).await
}

/// Convenience wrapper: GET request with extra headers, no body.
pub async fn get_with_headers(
    addr: SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
) -> RawResponse {
    http_request(addr, "GET", path, headers, &[]).await
}

/// Convenience wrapper: DELETE request, no extra headers, no body.
pub async fn delete(addr: SocketAddr, path: &str) -> RawResponse {
    http_request(addr, "DELETE", path, &[], &[]).await
}

/// Convenience wrapper: POST a JSON body.
pub async fn post_json(addr: SocketAddr, path: &str, json_body: &str) -> RawResponse {
    http_request(
        addr,
        "POST",
        path,
        &[("Content-Type", "application/json")],
        json_body.as_bytes(),
    )
    .await
}

// ── AppState / config builders ──────────────────────────────────────────────

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use oxillama_runtime::sampling::SamplerConfig;
use oxillama_runtime::ChatTemplate;
use oxillama_server::files_store::FilesStore;
use oxillama_server::{
    new_run_queue, AppState, BatchRequest, PrefixCacheRegistry, ServerConfig, ThreadStore,
    DEFAULT_MAX_NAMESPACES,
};

/// A unique temp-dir path for a test-scoped store, prefixed for easy
/// identification/cleanup and namespaced by a random UUID so parallel test
/// runs never collide.
pub fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "oxillama_it_{tag}_{}",
        uuid::Uuid::new_v4().as_simple()
    ))
}

/// Build a minimal `Arc<AppState>` with a live (but never-consumed by
/// default) inference queue, and files/threads stores configured so their
/// routes are reachable rather than returning 503.
///
/// `queue_capacity` lets callers control how easily the inference queue
/// fills up (used by the D5 queue-full integration test); the returned
/// `Sender` is what ends up in `AppState::queue`.
pub async fn build_test_state(queue_capacity: usize) -> Arc<AppState> {
    let (tx, _rx) = tokio::sync::mpsc::channel::<BatchRequest>(queue_capacity.max(1));
    let registry = Arc::new(PrefixCacheRegistry::new(
        Default::default(),
        DEFAULT_MAX_NAMESPACES,
    ));
    let worker_alive = Arc::new(AtomicBool::new(true));
    let spool_dir = unique_temp_dir("spool");

    let mut state = AppState::new(
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
    .expect("AppState::new should succeed in test environment");

    let files_store = Arc::new(FilesStore::new(unique_temp_dir("files")).expect("FilesStore::new"));
    state = state.with_files(files_store);

    let threads_store =
        Arc::new(ThreadStore::new(unique_temp_dir("threads")).expect("ThreadStore::new"));
    let (run_tx, _run_rx) = new_run_queue();
    state = state.with_threads(threads_store, run_tx);

    Arc::new(state)
}

/// A `ServerConfig` with every optional layer disabled except what a
/// specific test opts into by mutating the returned value — starts from
/// `ServerConfig::default()` so admin/CORS/tracing/etc. defaults are
/// realistic, not a stripped-down fake.
pub fn base_config() -> ServerConfig {
    ServerConfig::default()
}

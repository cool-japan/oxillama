//! D2 regression (end-to-end): request-supplied resource IDs containing
//! path-traversal sequences must never let a caller read or delete files
//! outside a store's root directory.
//!
//! These are integration tests (not unit tests) specifically because the
//! vulnerability depends on axum's real routing/percent-decoding
//! behavior: a URL path segment like `..%2F..%2Fetc%2Fpasswd` is matched
//! as a *single* route segment (routing splits on literal `/` bytes
//! before decoding), and only decoded into `../../etc/passwd` afterwards
//! when the handler extracts it via `Path<String>`. A unit test that
//! constructs a `Request` with an already-decoded ID in hand cannot
//! exercise that decode-after-match ordering; only a real HTTP request
//! line can.

mod common;

use common::{build_test_state, delete, get, spawn_server};

/// Files API: a `file_id` containing a percent-encoded `../` sequence must
/// be rejected (400), not treated as a literal path.
#[tokio::test]
async fn files_get_content_rejects_traversal_over_http() {
    let state = build_test_state(4).await;
    let app = oxillama_server::build_app(state);
    let addr = spawn_server(app).await;

    // Raw request line contains `..%2F..%2F..%2Fetc%2Fpasswd`, which axum
    // matches as a single `{file_id}` segment and percent-decodes to
    // `../../../etc/passwd` before it ever reaches `get_file_content_handler`.
    let resp = get(addr, "/v1/files/..%2F..%2F..%2Fetc%2Fpasswd/content").await;

    assert_eq!(
        resp.status,
        400,
        "path traversal file_id must be rejected with 400, got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

/// Files API: the destructive `DELETE` path must reject traversal ids too
/// — this is the highest-severity sink (`fs::remove_dir_all`).
#[tokio::test]
async fn files_delete_rejects_traversal_over_http() {
    let state = build_test_state(4).await;
    let app = oxillama_server::build_app(state);
    let addr = spawn_server(app).await;

    let resp = delete(addr, "/v1/files/..%2F..%2F..%2Ftmp/content").await;

    // The DELETE route itself is `/v1/files/{file_id}` (no `/content`
    // suffix); route it correctly but keep the traversal payload.
    let resp2 = delete(addr, "/v1/files/..%2F..%2F..%2Ftmp").await;

    for r in [&resp, &resp2] {
        assert_ne!(
            r.status, 200,
            "a path-traversal id must never report a successful delete"
        );
    }
    // The correctly-routed DELETE must be a clean rejection (400 from our
    // validation), not a 404/500 that might indicate the traversal was
    // attempted and merely didn't find anything at that particular path.
    assert_eq!(
        resp2.status,
        400,
        "DELETE with traversal id must be rejected with 400, got {}: {}",
        resp2.status,
        String::from_utf8_lossy(&resp2.body)
    );
}

/// Threads API: a `thread_id` containing a percent-encoded `../` sequence
/// must be rejected (mapped to `ThreadNotFound` → 404), not resolved to a
/// directory outside the thread store root.
#[tokio::test]
async fn threads_get_rejects_traversal_over_http() {
    let state = build_test_state(4).await;
    let app = oxillama_server::build_app(state);
    let addr = spawn_server(app).await;

    let resp = get(addr, "/v1/threads/..%2F..%2F..%2Fetc").await;

    assert_eq!(
        resp.status,
        404,
        "path traversal thread_id must be rejected (as not-found), got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

/// Disk-spooled Batch API: a `job_id` containing a percent-encoded `../`
/// sequence must be rejected, not resolved outside the spool root.
#[tokio::test]
async fn batch_jobs_get_rejects_traversal_over_http() {
    let state = build_test_state(4).await;
    let app = oxillama_server::build_app(state);
    let addr = spawn_server(app).await;

    let resp = get(addr, "/v1/batch_jobs/..%2F..%2F..%2Fetc").await;

    assert_eq!(
        resp.status,
        404,
        "path traversal job_id must be rejected (as not-found), got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

/// Disk-spooled Batch API: the output-streaming endpoint must also reject
/// traversal ids rather than attempting to open a file outside the spool
/// root.
#[tokio::test]
async fn batch_jobs_output_rejects_traversal_over_http() {
    let state = build_test_state(4).await;
    let app = oxillama_server::build_app(state);
    let addr = spawn_server(app).await;

    let resp = get(addr, "/v1/batch_jobs/..%2F..%2F..%2Fetc/output").await;

    assert_eq!(
        resp.status,
        404,
        "path traversal job_id on /output must be rejected (as not-found), got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

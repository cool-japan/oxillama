//! D1 regression (end-to-end): `/admin/*` must reject unauthenticated
//! requests on the router that `build_app_with_config` produces — the
//! router the CLI's `serve` command actually builds and exposes to the
//! network. Before the fix, `build_app_with_config` injected an
//! `Extension(AdminAuth)` but never applied the corresponding
//! `admin_auth_middleware` layer at all, so every `/admin/*` route was
//! completely unauthenticated regardless of configuration.

mod common;

use common::{build_test_state, get_with_headers, spawn_server};

/// With a bearer token configured, a request to `/admin/models` with no
/// `Authorization` header must be rejected (401) on the real
/// `build_app_with_config` router — not merely on a hand-assembled test
/// router that separately re-wires the middleware (which is what unit
/// tests in `admin/routes.rs` exercise).
#[tokio::test]
async fn admin_route_rejects_missing_bearer_token_on_production_router() {
    let state = build_test_state(4).await;
    let mut config = common::base_config();
    config.admin_bearer_token = Some("integration-test-admin-secret".to_string());

    let app = oxillama_server::build_app_with_config(state, &config)
        .expect("build_app_with_config should succeed with a valid config");
    let addr = spawn_server(app).await;

    let resp = get_with_headers(addr, "/admin/models", &[]).await;

    assert_eq!(
        resp.status,
        401,
        "admin route without a bearer token must be rejected on the production router, got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

/// The same route, with the *correct* bearer token supplied, must succeed
/// — proving the rejection above is really about authentication and not
/// some unrelated failure (e.g. a route that 401s unconditionally).
#[tokio::test]
async fn admin_route_accepts_correct_bearer_token_on_production_router() {
    let state = build_test_state(4).await;
    let mut config = common::base_config();
    config.admin_bearer_token = Some("integration-test-admin-secret".to_string());

    let app = oxillama_server::build_app_with_config(state, &config)
        .expect("build_app_with_config should succeed with a valid config");
    let addr = spawn_server(app).await;

    let resp = get_with_headers(
        addr,
        "/admin/models",
        &[("Authorization", "Bearer integration-test-admin-secret")],
    )
    .await;

    assert_eq!(
        resp.status,
        200,
        "admin route with the correct bearer token must succeed, got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

/// With no admin bearer token configured, admin auth falls back to
/// loopback-only. A real loopback TCP connection (this test always
/// connects to `127.0.0.1`) must be allowed through.
#[tokio::test]
async fn admin_route_allows_real_loopback_peer_when_no_token_configured() {
    let state = build_test_state(4).await;
    let config = common::base_config(); // admin_bearer_token: None by default
    let app = oxillama_server::build_app_with_config(state, &config)
        .expect("build_app_with_config should succeed with a valid config");
    let addr = spawn_server(app).await;

    let resp = get_with_headers(addr, "/admin/models", &[]).await;

    assert_eq!(
        resp.status,
        200,
        "a genuine loopback peer must be allowed through in token-less mode, got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

/// D1 regression: even in token-less/loopback-only mode, a genuinely
/// loopback TCP peer cannot use a spoofed `X-Forwarded-For` header to make
/// itself look like it's forwarding on behalf of a non-loopback client and
/// somehow escalate — this specifically proves the narrowing-only
/// direction of the forwarded-header logic holds over a real connection.
#[tokio::test]
async fn admin_route_denies_when_loopback_peer_forwards_remote_client() {
    let state = build_test_state(4).await;
    let config = common::base_config();
    let app = oxillama_server::build_app_with_config(state, &config)
        .expect("build_app_with_config should succeed with a valid config");
    let addr = spawn_server(app).await;

    let resp = get_with_headers(addr, "/admin/models", &[("X-Forwarded-For", "203.0.113.7")]).await;

    assert_eq!(
        resp.status,
        401,
        "a loopback peer forwarding on behalf of a remote client must be denied, got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

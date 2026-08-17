//! Regression (end-to-end): when `ServerConfig::api_keys` is non-empty,
//! an inference request with no `Authorization` header must be rejected
//! (401) on the real `build_app_with_config` router — proving the auth
//! layer is actually mounted around the inference routes in production,
//! not just exercised via a hand-built test router.

mod common;

use common::{build_test_state, get_with_headers, spawn_server};

#[tokio::test]
async fn models_list_rejects_missing_key_when_api_keys_configured() {
    let state = build_test_state(4).await;
    let mut config = common::base_config();
    config.api_keys = vec!["sk-integration-test-key".to_string()];

    let app = oxillama_server::build_app_with_config(state, &config)
        .expect("build_app_with_config should succeed with a valid config");
    let addr = spawn_server(app).await;

    let resp = get_with_headers(addr, "/v1/models", &[]).await;

    assert_eq!(
        resp.status,
        401,
        "a keyless request must be rejected when api_keys is configured, got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

/// The same route, with the correct bearer key, must succeed — proving
/// the rejection above is really about missing credentials.
#[tokio::test]
async fn models_list_accepts_correct_key_when_api_keys_configured() {
    let state = build_test_state(4).await;
    let mut config = common::base_config();
    config.api_keys = vec!["sk-integration-test-key".to_string()];

    let app = oxillama_server::build_app_with_config(state, &config)
        .expect("build_app_with_config should succeed with a valid config");
    let addr = spawn_server(app).await;

    let resp = get_with_headers(
        addr,
        "/v1/models",
        &[("Authorization", "Bearer sk-integration-test-key")],
    )
    .await;

    assert_eq!(
        resp.status,
        200,
        "a request with the correct bearer key must succeed, got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

/// A request with a *wrong* key must also be rejected (401), not merely
/// requests with no key at all.
#[tokio::test]
async fn models_list_rejects_wrong_key_when_api_keys_configured() {
    let state = build_test_state(4).await;
    let mut config = common::base_config();
    config.api_keys = vec!["sk-integration-test-key".to_string()];

    let app = oxillama_server::build_app_with_config(state, &config)
        .expect("build_app_with_config should succeed with a valid config");
    let addr = spawn_server(app).await;

    let resp = get_with_headers(
        addr,
        "/v1/models",
        &[("Authorization", "Bearer sk-totally-wrong-key")],
    )
    .await;

    assert_eq!(
        resp.status,
        401,
        "a request with an incorrect bearer key must be rejected, got {}: {}",
        resp.status,
        String::from_utf8_lossy(&resp.body)
    );
}

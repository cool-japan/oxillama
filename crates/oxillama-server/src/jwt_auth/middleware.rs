//! Axum middleware that verifies JWT tokens and enforces scope requirements.

use std::sync::Arc;

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};

use super::verifier::JwtVerifier;

/// Axum middleware function that:
/// 1. Extracts `Authorization: Bearer <token>` from the request.
/// 2. Verifies the JWT using the shared [`JwtVerifier`].
/// 3. Checks that the claims carry all scopes required for the request path.
///
/// On success, the request is forwarded to the next handler unchanged.
/// On failure, a `401 Unauthorized` or `403 Forbidden` response is returned
/// immediately with a human-readable message (no OpenAI error envelope here —
/// auth errors should be explicit).
pub async fn jwt_auth_middleware(
    axum::extract::State(verifier): axum::extract::State<Arc<JwtVerifier>>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();

    // Extract Bearer token from the Authorization header.
    let token = match req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    {
        Some(t) => t.to_string(),
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                "Missing or malformed Authorization header (expected: Bearer <token>)",
            )
                .into_response();
        }
    };

    // Verify the JWT.
    let claims = match verifier.verify(&token) {
        Ok(c) => c,
        Err(e) => {
            return (StatusCode::UNAUTHORIZED, e.to_string()).into_response();
        }
    };

    // Check scope requirements for this path.
    let required = verifier.required_scopes_for_path(&path);
    if !required.is_empty() {
        let user_scopes = verifier.scopes_from_claims(&claims);
        for req_scope in required {
            if !user_scopes.contains(req_scope) {
                return (
                    StatusCode::FORBIDDEN,
                    format!("Insufficient scope: missing {}", req_scope.as_str()),
                )
                    .into_response();
            }
        }
    }

    next.run(req).await
}

//! Admin API bearer-token authentication middleware.
//!
//! The admin API has a separate authentication policy from the inference API:
//!
//! - **Token configured** → all `/admin/*` routes require
//!   `Authorization: Bearer <token>` regardless of origin.
//! - **No token configured** → requests are only forwarded if they originate
//!   from the loopback interface (`127.0.0.1` / `::1`).
//!   Non-loopback requests receive `401 Unauthorized`.
//!
//! The startup check (`ensure_admin_security`) must be called before the
//! server begins accepting connections; it returns `Err` (rather than
//! terminating the process itself) if the admin listen address is
//! non-loopback AND no token is configured, so the caller can decide how to
//! report the failure.

use std::net::SocketAddr;

use axum::{
    body::Body,
    extract::{ConnectInfo, Request},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::error::{ServerError, ServerResult};

/// Admin authentication configuration, shared with the middleware via
/// `axum::Extension`.
#[derive(Debug, Clone)]
pub struct AdminAuth {
    /// The expected bearer token value, or `None` if auth is token-less.
    pub token: Option<String>,
}

/// Axum middleware that enforces admin auth policy.
pub async fn admin_auth_middleware(
    axum::extract::Extension(auth): axum::extract::Extension<AdminAuth>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if let Some(expected) = &auth.token {
        // Token-based auth: check the Authorization header.
        if bearer_token_matches(req.headers(), expected) {
            next.run(req).await
        } else {
            unauthorized_response("Missing or invalid admin bearer token")
        }
    } else {
        // No token — only allow loopback.
        if is_loopback(&req) {
            next.run(req).await
        } else {
            unauthorized_response(
                "Admin API requires a bearer token when not accessed from loopback",
            )
        }
    }
}

/// Check that the `Authorization: Bearer <token>` header matches.
fn bearer_token_matches(headers: &header::HeaderMap, expected: &str) -> bool {
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return false;
    };
    let Ok(str_val) = value.to_str() else {
        return false;
    };
    let Some(token) = str_val.strip_prefix("Bearer ") else {
        return false;
    };
    token == expected
}

/// Determine if the request is from a loopback address.
///
/// The **authoritative** source of truth is the actual TCP peer address,
/// delivered by axum via the [`ConnectInfo<SocketAddr>`] request extension
/// (populated when the server is bound with
/// `into_make_service_with_connect_info::<SocketAddr>()`). If that extension
/// is absent — e.g. the service was bound without connect-info propagation —
/// we **fail closed** and deny loopback status rather than trusting
/// client-controlled headers. This was previously the reverse: no
/// `ConnectInfo` and no forwarding headers defaulted to `true` (allow),
/// which meant every request on any deployment that forgot
/// `into_make_service_with_connect_info` was treated as trusted loopback
/// traffic regardless of its real origin (D1).
///
/// `X-Forwarded-For` / `X-Real-IP` are honored only to *narrow* the
/// decision (a real loopback peer forwarding on behalf of a non-loopback
/// client is correctly denied); they can never *widen* it — a spoofed
/// header claiming `127.0.0.1` from a genuinely remote peer is denied,
/// because the real peer address is checked first and is authoritative
/// whenever present.
fn is_loopback(req: &Request<Body>) -> bool {
    let Some(ConnectInfo(peer)) = req.extensions().get::<ConnectInfo<SocketAddr>>() else {
        // No verified peer address available: fail closed.
        return false;
    };

    if !peer.ip().is_loopback() {
        // The real socket peer is not loopback. A forwarding header cannot
        // override this — it would let a remote attacker simply claim to be
        // loopback via a spoofed `X-Forwarded-For: 127.0.0.1`.
        return false;
    }

    // The real peer *is* loopback (e.g. a local reverse proxy). If it
    // identifies a further upstream client via forwarding headers, that
    // client's address is what actually matters, so check it — this can
    // only turn a `true` into `false`, never the reverse.
    if let Some(xff) = req.headers().get("x-forwarded-for") {
        if let Ok(val) = xff.to_str() {
            let first = val.split(',').next().unwrap_or("").trim();
            if let Ok(ip) = first.parse::<std::net::IpAddr>() {
                return ip.is_loopback();
            }
        }
    }
    if let Some(xri) = req.headers().get("x-real-ip") {
        if let Ok(val) = xri.to_str() {
            if let Ok(ip) = val.trim().parse::<std::net::IpAddr>() {
                return ip.is_loopback();
            }
        }
    }

    true
}

fn unauthorized_response(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "authentication_error",
        }
    });
    (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response()
}

/// Safety check called at startup.
///
/// If `admin_host` is a non-loopback address AND no `token` is configured,
/// returns `Err(ServerError::InsecureAdminBinding)` instead of starting the
/// server. This prevents accidentally exposing the admin API to the network
/// without any authentication. Callers (typically `main`/CLI) are
/// responsible for reporting the error and exiting — library code must not
/// call `std::process::exit` itself (D11).
pub fn ensure_admin_security(admin_host: &str, token: &Option<String>) -> ServerResult<()> {
    if token.is_some() {
        return Ok(()); // Token present — safe regardless of address.
    }

    // Parse the host part (strip port if present).
    let host = admin_host.split(':').next().unwrap_or(admin_host).trim();

    let is_loopback_addr = matches!(
        host.parse::<std::net::IpAddr>(),
        Ok(ip) if ip.is_loopback()
    ) || matches!(host, "localhost");

    if !is_loopback_addr {
        tracing::error!(
            admin_host,
            "admin listen address is non-loopback but no admin bearer_token is configured; \
             set [admin] bearer_token = \"...\" in server.toml or bind the admin interface \
             to 127.0.0.1"
        );
        return Err(ServerError::InsecureAdminBinding(admin_host.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;

    fn make_req_with_auth(token: &str) -> Request<Body> {
        Request::builder()
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .expect("build request")
    }

    fn make_req_without_auth() -> Request<Body> {
        Request::builder()
            .body(Body::empty())
            .expect("build request")
    }

    /// Build a request carrying a `ConnectInfo<SocketAddr>` extension for
    /// the given peer IP, as axum would when the server is bound with
    /// `into_make_service_with_connect_info::<SocketAddr>()`.
    fn req_from_peer(ip: &str) -> Request<Body> {
        let addr: SocketAddr = format!("{ip}:54321").parse().expect("valid socket addr");
        Request::builder()
            .extension(ConnectInfo(addr))
            .body(Body::empty())
            .expect("build request")
    }

    #[test]
    fn bearer_matches_correct_token() {
        let req = make_req_with_auth("secret");
        assert!(bearer_token_matches(req.headers(), "secret"));
    }

    #[test]
    fn bearer_rejects_wrong_token() {
        let req = make_req_with_auth("wrong");
        assert!(!bearer_token_matches(req.headers(), "secret"));
    }

    #[test]
    fn bearer_rejects_missing_header() {
        let req = make_req_without_auth();
        assert!(!bearer_token_matches(req.headers(), "secret"));
    }

    #[test]
    fn bearer_rejects_basic_scheme() {
        let req = Request::builder()
            .header("authorization", "Basic dXNlcjpwYXNz")
            .body(Body::empty())
            .expect("build");
        assert!(!bearer_token_matches(req.headers(), "secret"));
    }

    #[test]
    fn ensure_admin_security_passes_with_token() {
        assert!(ensure_admin_security("0.0.0.0:8888", &Some("tok".to_string())).is_ok());
    }

    #[test]
    fn ensure_admin_security_passes_loopback_no_token() {
        assert!(ensure_admin_security("127.0.0.1:8888", &None).is_ok());
        assert!(ensure_admin_security("localhost:8888", &None).is_ok());
    }

    /// D11 regression: a non-loopback bind with no token must return a
    /// typed error instead of calling `std::process::exit` (which would
    /// abort the whole test binary, not just fail this test).
    #[test]
    fn ensure_admin_security_rejects_non_loopback_no_token_without_exiting() {
        let result = ensure_admin_security("0.0.0.0:8888", &None);
        assert!(matches!(
            result,
            Err(ServerError::InsecureAdminBinding(ref addr)) if addr == "0.0.0.0:8888"
        ));
    }

    /// D1 regression: with no `ConnectInfo` extension present (e.g. the
    /// service wasn't bound with `into_make_service_with_connect_info`),
    /// the old behavior defaulted to "loopback" (fail open). It must now
    /// fail closed.
    #[test]
    fn is_loopback_denies_when_connect_info_missing() {
        let req = make_req_without_auth();
        assert!(!is_loopback(&req));
    }

    /// D1 regression: a genuinely remote peer cannot claim loopback status
    /// via a spoofed `X-Forwarded-For` header.
    #[test]
    fn is_loopback_denies_spoofed_forwarded_for_header() {
        let mut req = req_from_peer("203.0.113.7");
        req.headers_mut().insert(
            "x-forwarded-for",
            "127.0.0.1".parse().expect("valid header value"),
        );
        assert!(!is_loopback(&req));
    }

    /// A real loopback peer with no forwarding headers is allowed.
    #[test]
    fn is_loopback_allows_real_loopback_peer() {
        let req = req_from_peer("127.0.0.1");
        assert!(is_loopback(&req));
    }

    /// A real remote peer with no forwarding headers is denied.
    #[test]
    fn is_loopback_denies_real_remote_peer() {
        let req = req_from_peer("203.0.113.7");
        assert!(!is_loopback(&req));
    }

    /// A real loopback peer (e.g. a local reverse proxy) forwarding on
    /// behalf of a genuinely remote client is denied — the forwarding
    /// header may only narrow, not widen, trust.
    #[test]
    fn is_loopback_denies_loopback_peer_forwarding_remote_client() {
        let mut req = req_from_peer("127.0.0.1");
        req.headers_mut().insert(
            "x-forwarded-for",
            "203.0.113.7".parse().expect("valid header value"),
        );
        assert!(!is_loopback(&req));
    }
}

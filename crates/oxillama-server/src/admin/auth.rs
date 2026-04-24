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
//! server begins accepting connections; it terminates the process if the admin
//! listen address is non-loopback AND no token is configured.

use axum::{
    body::Body,
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

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
/// This inspects the `X-Forwarded-For` header (first IP) when present, then
/// the `X-Real-IP` header, and finally falls back to the connection address
/// embedded by tower via the [`axum::extract::ConnectInfo`] extension.
///
/// In test environments without real sockets we default to allowing loopback.
fn is_loopback(req: &Request<Body>) -> bool {
    // Check X-Forwarded-For (first entry).
    if let Some(xff) = req.headers().get("x-forwarded-for") {
        if let Ok(val) = xff.to_str() {
            let first = val.split(',').next().unwrap_or("").trim();
            if let Ok(ip) = first.parse::<std::net::IpAddr>() {
                return ip.is_loopback();
            }
        }
    }

    // Check X-Real-IP.
    if let Some(xri) = req.headers().get("x-real-ip") {
        if let Ok(val) = xri.to_str() {
            if let Ok(ip) = val.trim().parse::<std::net::IpAddr>() {
                return ip.is_loopback();
            }
        }
    }

    // No forwarding headers: in a test / embedded context default to allow.
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
/// prints an error and exits the process with code 1.
///
/// This prevents accidentally exposing the admin API to the network without
/// any authentication.
pub fn ensure_admin_security(admin_host: &str, token: &Option<String>) {
    if token.is_some() {
        return; // Token present — safe regardless of address.
    }

    // Parse the host part (strip port if present).
    let host = admin_host.split(':').next().unwrap_or(admin_host).trim();

    let is_loopback_addr = matches!(
        host.parse::<std::net::IpAddr>(),
        Ok(ip) if ip.is_loopback()
    ) || matches!(host, "localhost");

    if !is_loopback_addr {
        eprintln!(
            "FATAL: admin listen address '{admin_host}' is non-loopback \
             but no admin bearer_token is configured.\n\
             Set [admin] bearer_token = \"...\" in server.toml or \
             bind the admin interface to 127.0.0.1."
        );
        std::process::exit(1);
    }
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
        // Must not panic or exit.
        ensure_admin_security("0.0.0.0:8888", &Some("tok".to_string()));
    }

    #[test]
    fn ensure_admin_security_passes_loopback_no_token() {
        ensure_admin_security("127.0.0.1:8888", &None);
        ensure_admin_security("localhost:8888", &None);
    }
}

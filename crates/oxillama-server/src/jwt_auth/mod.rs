//! JWT authentication with scope-based authorization.
//!
//! This module provides a 100% Pure Rust JWT implementation (no C/Fortran
//! dependencies) using `hmac`, `sha2`, `rsa`, and `base64` from the
//! RustCrypto project.
//!
//! ## Supported algorithms
//!
//! | Algorithm | Implementation |
//! |-----------|----------------|
//! | HS256 | `hmac` + `sha2` (constant-time HMAC-SHA256) |
//! | RS256 | `rsa` + `sha2` (PKCS1v15, DER-encoded public key) |
//!
//! ## Security guarantees
//!
//! - `alg: "none"` is **always** rejected regardless of key configuration.
//! - Only explicitly configured algorithms are accepted (no downgrade).
//! - HS256 comparison is constant-time via `hmac::Mac::verify_slice`.
//!
//! ## Usage (wiring into `build_app_with_config`)
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use oxillama_server::jwt_auth::{JwtConfig, JwtAlgorithm, JwtVerifier};
//!
//! let config = JwtConfig {
//!     algorithm: JwtAlgorithm::Hs256 { secret: b"my-secret".to_vec() },
//!     ..Default::default()
//! };
//! let verifier = Arc::new(JwtVerifier::new(config));
//! // Pass `verifier` to `build_app_with_config` via `ServerConfig::jwt`.
//! ```

pub mod middleware;
pub mod scopes;
pub mod verifier;

pub use middleware::jwt_auth_middleware;
pub use scopes::Scope;
pub use verifier::{JwtAlgorithm, JwtClaims, JwtConfig, JwtError, JwtResult, JwtVerifier};

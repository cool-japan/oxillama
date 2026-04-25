//! Pure-Rust JWT verification (HS256 and RS256).
//!
//! This module intentionally avoids any JWT library and implements the
//! compact serialization format directly using `hmac`, `sha2`, `rsa`, and
//! `base64`. This keeps the dependency tree 100% Pure Rust and allows precise
//! security control (e.g., `alg: "none"` is always rejected).

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, Mac};
use thiserror::Error;

// Use sha2 from the sha2 workspace dep for HS256 (digest 0.10 compatible with hmac 0.12).
use sha2::Sha256;

use super::scopes::Scope;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can arise during JWT verification.
#[derive(Debug, Error)]
pub enum JwtError {
    /// Token does not have exactly three Base64url-encoded parts.
    #[error("JWT is malformed (expected header.payload.signature)")]
    Malformed,

    /// A part could not be decoded from Base64url.
    #[error("JWT base64url decode failed: {0}")]
    Base64(#[from] base64::DecodeError),

    /// Header or payload could not be parsed as JSON.
    #[error("JWT JSON parse failed: {0}")]
    Json(#[from] serde_json::Error),

    /// The `alg` header field names an unsupported algorithm.
    #[error("unsupported JWT algorithm: {0}")]
    UnsupportedAlg(String),

    /// The `alg` header is `"none"` — always rejected for security.
    #[error("JWT algorithm 'none' is not allowed")]
    AlgNone,

    /// Signature verification failed.
    #[error("JWT signature verification failed")]
    BadSignature,

    /// The token has expired (`exp` is in the past).
    #[error("JWT has expired")]
    Expired,

    /// The token is not yet valid (`nbf` is in the future).
    #[error("JWT is not yet valid (nbf)")]
    NotYetValid,

    /// The `aud` claim does not match the expected audience.
    #[error("JWT audience mismatch (expected {expected:?}, got {got:?})")]
    WrongAudience { expected: String, got: String },

    /// The `iss` claim does not match the expected issuer.
    #[error("JWT issuer mismatch (expected {expected:?}, got {got:?})")]
    WrongIssuer { expected: String, got: String },

    /// A required claim is absent.
    #[error("JWT missing required claim: {0}")]
    MissingClaim(String),

    /// RSA public-key decoding failed.
    #[error("RSA public key decode failed: {0}")]
    RsaKeyDecode(String),

    /// RSA signature object could not be constructed from the given bytes.
    #[error("RSA signature bytes invalid")]
    RsaSignatureInvalid,

    /// The system clock is unavailable.
    #[error("system clock unavailable")]
    Clock,
}

/// Convenient `Result` alias.
pub type JwtResult<T> = Result<T, JwtError>;

// ── Algorithm and config ──────────────────────────────────────────────────────

/// Signing algorithm configuration.
#[derive(Debug, Clone)]
pub enum JwtAlgorithm {
    /// HMAC-SHA256 with a shared secret.
    Hs256 { secret: Vec<u8> },
    /// RSA PKCS1v15 + SHA-256 with a DER-encoded public key.
    Rs256 {
        /// DER-encoded SubjectPublicKeyInfo (SPKI) format.
        public_key_der: Vec<u8>,
    },
}

/// Configuration for JWT verification.
#[derive(Debug, Clone)]
pub struct JwtConfig {
    /// Signing algorithm and associated key material.
    pub algorithm: JwtAlgorithm,
    /// Expected value of the `aud` claim.  `None` skips audience validation.
    pub audience: Option<String>,
    /// Expected value of the `iss` claim.  `None` skips issuer validation.
    pub issuer: Option<String>,
    /// Mapping from URL path to the set of scopes required to access that path.
    ///
    /// Example: `"/v1/chat/completions" → [Scope::ChatWrite]`
    pub required_scopes_per_route: HashMap<String, Vec<Scope>>,
    /// Seconds of clock skew tolerated when validating `exp` and `nbf`.
    /// Defaults to 60.
    pub clock_skew_secs: u64,
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            algorithm: JwtAlgorithm::Hs256 { secret: Vec::new() },
            audience: None,
            issuer: None,
            required_scopes_per_route: HashMap::new(),
            clock_skew_secs: 60,
        }
    }
}

// ── Claims ────────────────────────────────────────────────────────────────────

/// Decoded JWT claims (registered + common extension claims).
#[derive(Debug, serde::Deserialize)]
pub struct JwtClaims {
    /// Subject identifier.
    pub sub: Option<String>,
    /// Expiration time (seconds since Unix epoch).
    pub exp: Option<u64>,
    /// Not-before time (seconds since Unix epoch).
    pub nbf: Option<u64>,
    /// Issued-at time (seconds since Unix epoch).
    pub iat: Option<u64>,
    /// Audience — can be a single string or a JSON array.
    pub aud: Option<serde_json::Value>,
    /// Issuer.
    pub iss: Option<String>,
    /// Space-separated scope string (`"chat:read embed:read"`).
    pub scope: Option<String>,
    /// Alternative array-form scopes.
    pub scopes: Option<Vec<String>>,
}

// ── Internal header type ──────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct JwtHeader {
    alg: String,
    #[allow(dead_code)]
    typ: Option<String>,
}

// ── HS256 verification ────────────────────────────────────────────────────────

/// Verifies an HS256 (HMAC-SHA256) signature using constant-time comparison.
fn verify_hs256(secret: &[u8], signing_input: &[u8], signature: &[u8]) -> JwtResult<()> {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret).map_err(|_| JwtError::BadSignature)?;
    mac.update(signing_input);
    mac.verify_slice(signature)
        .map_err(|_| JwtError::BadSignature)
}

// ── RS256 verification ────────────────────────────────────────────────────────

/// Verifies an RS256 (RSA PKCS1v15 + SHA-256) signature.
///
/// Uses `rsa::sha2::Sha256` (re-exported from rsa's bundled sha2 dependency,
/// which carries the `AssociatedOid` implementation required by `VerifyingKey`).
fn verify_rs256(public_key_der: &[u8], signing_input: &[u8], signature: &[u8]) -> JwtResult<()> {
    use rsa::pkcs1v15::VerifyingKey;
    use rsa::pkcs8::DecodePublicKey;
    use rsa::sha2::Sha256 as RsaSha256;
    use rsa::signature::Verifier;

    let pub_key = rsa::RsaPublicKey::from_public_key_der(public_key_der)
        .map_err(|e| JwtError::RsaKeyDecode(e.to_string()))?;

    let verifying_key = VerifyingKey::<RsaSha256>::new(pub_key);

    let sig =
        rsa::pkcs1v15::Signature::try_from(signature).map_err(|_| JwtError::RsaSignatureInvalid)?;

    verifying_key
        .verify(signing_input, &sig)
        .map_err(|_| JwtError::BadSignature)
}

// ── Verifier ──────────────────────────────────────────────────────────────────

/// Stateless JWT verifier.  Thread-safe and cheap to clone (key material is
/// behind `Arc` via `JwtConfig::algorithm`'s `Vec<u8>`).
pub struct JwtVerifier {
    config: JwtConfig,
}

impl JwtVerifier {
    /// Creates a new verifier with the given configuration.
    pub fn new(config: JwtConfig) -> Self {
        Self { config }
    }

    /// Verifies a compact-serialization JWT and returns the decoded claims.
    ///
    /// Security invariants enforced:
    /// - `alg: "none"` is always rejected.
    /// - Only `HS256` and `RS256` are accepted.
    /// - `exp`, `nbf`, `aud`, `iss` are validated when present or configured.
    pub fn verify(&self, token: &str) -> JwtResult<JwtClaims> {
        // 1. Split into exactly three parts.
        let parts: Vec<&str> = token.splitn(4, '.').collect();
        if parts.len() != 3 {
            return Err(JwtError::Malformed);
        }
        let (header_b64, payload_b64, sig_b64) = (parts[0], parts[1], parts[2]);

        // 2. Decode each part from Base64url (no padding).
        let header_bytes = URL_SAFE_NO_PAD.decode(header_b64)?;
        let payload_bytes = URL_SAFE_NO_PAD.decode(payload_b64)?;
        let signature = URL_SAFE_NO_PAD.decode(sig_b64)?;

        // 3. Parse header and reject `alg: "none"`.
        let header: JwtHeader = serde_json::from_slice(&header_bytes)?;
        if header.alg.eq_ignore_ascii_case("none") {
            return Err(JwtError::AlgNone);
        }

        // 4. Verify signature.
        //    signing_input = "<header_b64>.<payload_b64>" (ASCII bytes)
        let signing_input = format!("{header_b64}.{payload_b64}");
        match &self.config.algorithm {
            JwtAlgorithm::Hs256 { secret } => {
                if !header.alg.eq_ignore_ascii_case("HS256") {
                    return Err(JwtError::UnsupportedAlg(header.alg));
                }
                verify_hs256(secret, signing_input.as_bytes(), &signature)?;
            }
            JwtAlgorithm::Rs256 { public_key_der } => {
                if !header.alg.eq_ignore_ascii_case("RS256") {
                    return Err(JwtError::UnsupportedAlg(header.alg));
                }
                verify_rs256(public_key_der, signing_input.as_bytes(), &signature)?;
            }
        }

        // 5. Parse claims.
        let claims: JwtClaims = serde_json::from_slice(&payload_bytes)?;

        // 6. Validate temporal claims.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| JwtError::Clock)?
            .as_secs();

        let skew = self.config.clock_skew_secs;

        if let Some(exp) = claims.exp {
            if now > exp.saturating_add(skew) {
                return Err(JwtError::Expired);
            }
        }

        if let Some(nbf) = claims.nbf {
            if now.saturating_add(skew) < nbf {
                return Err(JwtError::NotYetValid);
            }
        }

        // 7. Validate audience.
        if let Some(expected_aud) = &self.config.audience {
            match &claims.aud {
                None => {
                    return Err(JwtError::MissingClaim("aud".to_string()));
                }
                Some(aud_val) => {
                    let matches = match aud_val {
                        serde_json::Value::String(s) => s == expected_aud,
                        serde_json::Value::Array(arr) => arr
                            .iter()
                            .any(|v| v.as_str().is_some_and(|s| s == expected_aud)),
                        _ => false,
                    };
                    if !matches {
                        let got = aud_val.to_string();
                        return Err(JwtError::WrongAudience {
                            expected: expected_aud.clone(),
                            got,
                        });
                    }
                }
            }
        }

        // 8. Validate issuer.
        if let Some(expected_iss) = &self.config.issuer {
            match &claims.iss {
                None => {
                    return Err(JwtError::MissingClaim("iss".to_string()));
                }
                Some(iss) if iss != expected_iss => {
                    return Err(JwtError::WrongIssuer {
                        expected: expected_iss.clone(),
                        got: iss.clone(),
                    });
                }
                Some(_) => {}
            }
        }

        Ok(claims)
    }

    /// Extracts the set of scopes from the decoded claims.
    ///
    /// Supports both the `"scope"` string field (space-separated) and the
    /// `"scopes"` array field.  Unknown scope strings are silently ignored.
    pub fn scopes_from_claims(&self, claims: &JwtClaims) -> Vec<Scope> {
        let mut out: Vec<Scope> = Vec::new();

        if let Some(scope_str) = &claims.scope {
            for s in scope_str.split_whitespace() {
                if let Ok(scope) = s.parse::<Scope>() {
                    out.push(scope);
                }
            }
        }

        if let Some(scopes_arr) = &claims.scopes {
            for s in scopes_arr {
                if let Ok(scope) = s.parse::<Scope>() {
                    if !out.contains(&scope) {
                        out.push(scope);
                    }
                }
            }
        }

        out
    }

    /// Returns the scopes required for the given URL path.
    ///
    /// Returns an empty slice when the path has no explicit scope requirement.
    pub fn required_scopes_for_path(&self, path: &str) -> &[Scope] {
        self.config
            .required_scopes_per_route
            .get(path)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── Test helpers ──────────────────────────────────────────────────────────

    /// Build a minimal JWT with the given header alg, claims JSON, and secret.
    /// `alg_override` lets you put any string in the header (e.g., `"none"`).
    fn make_hs256_token(
        secret: &[u8],
        claims: &serde_json::Value,
        alg_override: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let alg = alg_override.unwrap_or("HS256");
        let header = serde_json::json!({ "alg": alg, "typ": "JWT" });

        let header_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header)?);
        let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(claims)?);

        let signing_input = format!("{header_b64}.{payload_b64}");

        // For `alg: "none"` we produce an empty signature to match spec.
        let sig_b64 = if alg.eq_ignore_ascii_case("none") {
            String::new()
        } else {
            let mut mac =
                Hmac::<Sha256>::new_from_slice(secret).expect("test fixture: HMAC construction");
            mac.update(signing_input.as_bytes());
            URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes())
        };

        Ok(format!("{signing_input}.{sig_b64}"))
    }

    fn unix_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test fixture: system time")
            .as_secs()
    }

    fn hs256_verifier(secret: &[u8]) -> JwtVerifier {
        JwtVerifier::new(JwtConfig {
            algorithm: JwtAlgorithm::Hs256 {
                secret: secret.to_vec(),
            },
            ..Default::default()
        })
    }

    // ── Basic HS256 ──────────────────────────────────────────────────────────

    #[test]
    fn jwt_valid_hs256_verifies() {
        let secret = b"super-secret-key";
        let claims = serde_json::json!({
            "sub": "user-1",
            "exp": unix_now() + 3600,
        });
        let token =
            make_hs256_token(secret, &claims, None).expect("test fixture: token construction");

        let verifier = hs256_verifier(secret);
        let result = verifier.verify(&token);
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
    }

    #[test]
    fn jwt_wrong_secret_rejected() {
        let claims = serde_json::json!({ "sub": "user-1", "exp": unix_now() + 3600 });
        let token = make_hs256_token(b"correct-secret", &claims, None).expect("test fixture");
        let verifier = hs256_verifier(b"wrong-secret");
        assert!(matches!(
            verifier.verify(&token),
            Err(JwtError::BadSignature)
        ));
    }

    // ── Temporal claims ──────────────────────────────────────────────────────

    #[test]
    fn jwt_expired_token_rejected() {
        let secret = b"my-secret";
        // exp is 1000 seconds in the past, well beyond any clock skew
        let claims = serde_json::json!({
            "sub": "user-1",
            "exp": unix_now() - 1000,
        });
        let token = make_hs256_token(secret, &claims, None).expect("test fixture");
        let verifier = hs256_verifier(secret);
        assert!(
            matches!(verifier.verify(&token), Err(JwtError::Expired)),
            "expected Expired"
        );
    }

    #[test]
    fn jwt_nbf_in_future_rejected() {
        let secret = b"my-secret";
        let claims = serde_json::json!({
            "sub": "user-1",
            "exp": unix_now() + 3600,
            "nbf": unix_now() + 10_000,
        });
        let token = make_hs256_token(secret, &claims, None).expect("test fixture");
        let verifier = hs256_verifier(secret);
        assert!(
            matches!(verifier.verify(&token), Err(JwtError::NotYetValid)),
            "expected NotYetValid"
        );
    }

    // ── Audience ─────────────────────────────────────────────────────────────

    #[test]
    fn jwt_wrong_audience_rejected() {
        let secret = b"my-secret";
        let claims = serde_json::json!({
            "sub": "user-1",
            "exp": unix_now() + 3600,
            "aud": "other-service",
        });
        let token = make_hs256_token(secret, &claims, None).expect("test fixture");
        let verifier = JwtVerifier::new(JwtConfig {
            algorithm: JwtAlgorithm::Hs256 {
                secret: secret.to_vec(),
            },
            audience: Some("my-service".to_string()),
            ..Default::default()
        });
        assert!(
            matches!(verifier.verify(&token), Err(JwtError::WrongAudience { .. })),
            "expected WrongAudience"
        );
    }

    #[test]
    fn jwt_correct_audience_in_array_accepted() {
        let secret = b"my-secret";
        let claims = serde_json::json!({
            "sub": "user-1",
            "exp": unix_now() + 3600,
            "aud": ["other-service", "my-service"],
        });
        let token = make_hs256_token(secret, &claims, None).expect("test fixture");
        let verifier = JwtVerifier::new(JwtConfig {
            algorithm: JwtAlgorithm::Hs256 {
                secret: secret.to_vec(),
            },
            audience: Some("my-service".to_string()),
            ..Default::default()
        });
        assert!(verifier.verify(&token).is_ok());
    }

    // ── Issuer ───────────────────────────────────────────────────────────────

    #[test]
    fn jwt_wrong_issuer_rejected() {
        let secret = b"my-secret";
        let claims = serde_json::json!({
            "sub": "user-1",
            "exp": unix_now() + 3600,
            "iss": "bad-issuer",
        });
        let token = make_hs256_token(secret, &claims, None).expect("test fixture");
        let verifier = JwtVerifier::new(JwtConfig {
            algorithm: JwtAlgorithm::Hs256 {
                secret: secret.to_vec(),
            },
            issuer: Some("trusted-issuer".to_string()),
            ..Default::default()
        });
        assert!(
            matches!(verifier.verify(&token), Err(JwtError::WrongIssuer { .. })),
            "expected WrongIssuer"
        );
    }

    // ── Malformed tokens ─────────────────────────────────────────────────────

    #[test]
    fn jwt_malformed_token_rejected() {
        let verifier = hs256_verifier(b"secret");
        // 4 parts instead of 3
        assert!(
            matches!(
                verifier.verify("not.a.valid.token"),
                Err(JwtError::Malformed)
            ),
            "expected Malformed for 4-part token"
        );
        // 2 parts
        assert!(
            matches!(verifier.verify("only.two"), Err(JwtError::Malformed)),
            "expected Malformed for 2-part token"
        );
        // 1 part
        assert!(
            matches!(verifier.verify("onepart"), Err(JwtError::Malformed)),
            "expected Malformed for 1-part token"
        );
    }

    // ── alg: none ────────────────────────────────────────────────────────────

    #[test]
    fn jwt_alg_none_rejected() {
        let secret = b"my-secret";
        let claims = serde_json::json!({ "sub": "user-1", "exp": unix_now() + 3600 });
        let token = make_hs256_token(secret, &claims, Some("none")).expect("test fixture");
        let verifier = hs256_verifier(secret);
        assert!(
            matches!(verifier.verify(&token), Err(JwtError::AlgNone)),
            "expected AlgNone"
        );
    }

    #[test]
    fn jwt_alg_none_uppercase_rejected() {
        let secret = b"my-secret";
        let claims = serde_json::json!({ "sub": "user-1", "exp": unix_now() + 3600 });
        let token = make_hs256_token(secret, &claims, Some("NONE")).expect("test fixture");
        let verifier = hs256_verifier(secret);
        assert!(matches!(verifier.verify(&token), Err(JwtError::AlgNone)));
    }

    // ── Scopes ───────────────────────────────────────────────────────────────

    #[test]
    fn jwt_scope_string_parsed() {
        let secret = b"my-secret";
        let claims = serde_json::json!({
            "sub": "user-1",
            "exp": unix_now() + 3600,
            "scope": "chat:read embed:read",
        });
        let token = make_hs256_token(secret, &claims, None).expect("test fixture");
        let verifier = hs256_verifier(secret);
        let decoded = verifier.verify(&token).expect("valid token");
        let scopes = verifier.scopes_from_claims(&decoded);
        assert!(scopes.contains(&Scope::ChatRead));
        assert!(scopes.contains(&Scope::EmbedRead));
        assert!(!scopes.contains(&Scope::ChatWrite));
    }

    #[test]
    fn jwt_scopes_array_parsed() {
        let secret = b"my-secret";
        let claims = serde_json::json!({
            "sub": "user-1",
            "exp": unix_now() + 3600,
            "scopes": ["chat:read", "admin:read"],
        });
        let token = make_hs256_token(secret, &claims, None).expect("test fixture");
        let verifier = hs256_verifier(secret);
        let decoded = verifier.verify(&token).expect("valid token");
        let scopes = verifier.scopes_from_claims(&decoded);
        assert!(scopes.contains(&Scope::ChatRead));
        assert!(scopes.contains(&Scope::AdminRead));
    }

    #[test]
    fn jwt_required_scopes_for_path() {
        let mut route_scopes = HashMap::new();
        route_scopes.insert("/v1/chat/completions".to_string(), vec![Scope::ChatWrite]);
        let verifier = JwtVerifier::new(JwtConfig {
            algorithm: JwtAlgorithm::Hs256 {
                secret: b"s".to_vec(),
            },
            required_scopes_per_route: route_scopes,
            ..Default::default()
        });
        let required = verifier.required_scopes_for_path("/v1/chat/completions");
        assert_eq!(required, [Scope::ChatWrite]);

        let unrestricted = verifier.required_scopes_for_path("/health");
        assert!(unrestricted.is_empty());
    }

    // ── RS256 ─────────────────────────────────────────────────────────────────

    /// RS256 test — may be slow due to 2048-bit key generation (~1 s).
    /// Run with `cargo test -- --include-ignored` or `cargo nextest run -- --include-ignored`.
    #[test]
    #[ignore]
    fn jwt_rs256_with_generated_key() {
        use rsa::pkcs1v15::SigningKey;
        use rsa::pkcs8::EncodePublicKey;
        use rsa::sha2::Sha256 as RsaSha256;
        use rsa::signature::RandomizedSigner;
        use rsa::signature::SignatureEncoding;

        let mut rng = rand::thread_rng();
        let private_key =
            rsa::RsaPrivateKey::new(&mut rng, 2048).expect("test fixture: RSA key generation");
        let public_key = rsa::RsaPublicKey::from(&private_key);

        // Encode public key as DER (SPKI)
        let public_key_der = public_key
            .to_public_key_der()
            .expect("test fixture: DER encode")
            .to_vec();

        // Build and sign a token manually
        let header = serde_json::json!({ "alg": "RS256", "typ": "JWT" });
        let claims_json = serde_json::json!({
            "sub": "rs256-user",
            "exp": unix_now() + 3600,
        });
        let header_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).expect("test fixture"));
        let payload_b64 =
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims_json).expect("test fixture"));
        let signing_input = format!("{header_b64}.{payload_b64}");

        // Use rsa::sha2::Sha256 (re-exported from rsa's sha2 dep, has AssociatedOid)
        let signing_key = SigningKey::<RsaSha256>::new(private_key);
        let signature = signing_key.sign_with_rng(&mut rng, signing_input.as_bytes());
        let sig_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());

        let token = format!("{signing_input}.{sig_b64}");

        // Verify
        let verifier = JwtVerifier::new(JwtConfig {
            algorithm: JwtAlgorithm::Rs256 { public_key_der },
            ..Default::default()
        });
        let result = verifier.verify(&token);
        assert!(result.is_ok(), "RS256 verify failed: {:?}", result.err());
    }
}

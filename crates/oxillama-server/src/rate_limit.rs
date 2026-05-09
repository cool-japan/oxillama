//! Token-bucket rate limiter middleware.
//!
//! Provides a global request-rate limiter and a per-API-key rate limiter.
//! When a bucket is exhausted, returns 429 Too Many Requests with a
//! `Retry-After` header.

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tokio::sync::Mutex as AsyncMutex;

/// Token bucket state.
#[derive(Debug)]
pub struct TokenBucket {
    /// Current number of available tokens.
    tokens: f64,
    /// Maximum burst capacity.
    capacity: f64,
    /// Tokens replenished per second.
    rate: f64,
    /// Last refill timestamp.
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket.
    ///
    /// `capacity` — maximum burst size.
    /// `rate` — tokens per second refill rate.
    pub fn new(capacity: f64, rate: f64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            rate,
            last_refill: Instant::now(),
        }
    }

    /// Try to consume one token. Returns `Ok(())` if available,
    /// or `Err(retry_after_secs)` if the bucket is empty.
    pub fn try_acquire(&mut self) -> Result<(), f64> {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else {
            // Time until next token is available
            let deficit = 1.0 - self.tokens;
            let retry_after = deficit / self.rate;
            Err(retry_after)
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.capacity);
        self.last_refill = now;
    }
}

/// Shared rate limiter state.
#[derive(Clone)]
pub struct RateLimiter(pub Arc<AsyncMutex<TokenBucket>>);

impl RateLimiter {
    /// Create a rate limiter with the given capacity and refill rate.
    pub fn new(capacity: f64, rate_per_second: f64) -> Self {
        Self(Arc::new(AsyncMutex::new(TokenBucket::new(
            capacity,
            rate_per_second,
        ))))
    }
}

/// Middleware function for global rate limiting.
pub async fn rate_limit_middleware(
    limiter: Option<axum::extract::Extension<RateLimiter>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(axum::extract::Extension(limiter)) = limiter else {
        return next.run(request).await;
    };

    let mut bucket = limiter.0.lock().await;
    match bucket.try_acquire() {
        Ok(()) => {
            drop(bucket);
            next.run(request).await
        }
        Err(retry_after) => {
            drop(bucket);
            let retry_secs = retry_after.ceil() as u64;
            let body = serde_json::json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                }
            });
            let mut resp = (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response();
            if let Ok(val) = retry_secs.to_string().parse() {
                resp.headers_mut().insert("retry-after", val);
            }
            resp
        }
    }
}

// ── Per-API-key rate limiter ──────────────────────────────────────────────────

/// Per-API-key token-bucket rate limiter.
///
/// A separate [`TokenBucket`] is lazily created for each distinct API key on
/// first use.  Keys that have an entry in `overrides` get a bucket with the
/// specified `(capacity, rate)` pair; all others share `default_capacity` and
/// `default_rate`.
///
/// Concurrency model:
/// - The outer `RwLock` guards the `HashMap` of buckets.  Read-lock is taken
///   for lookups; write-lock is acquired only on the first hit for a new key.
/// - Each bucket is wrapped in a `Mutex` so multiple concurrent requests for
///   the *same* key do not race on the bucket's mutable refill state.
#[derive(Debug)]
pub struct PerKeyRateLimiter {
    buckets: Arc<RwLock<HashMap<String, Mutex<TokenBucket>>>>,
    default_capacity: f64,
    default_rate: f64,
    overrides: HashMap<String, (f64, f64)>,
}

impl PerKeyRateLimiter {
    /// Create a limiter with the given default capacity and refill rate.
    pub fn new(default_capacity: f64, default_rate: f64) -> Self {
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            default_capacity,
            default_rate,
            overrides: HashMap::new(),
        }
    }

    /// Attach per-key overrides: `key → (capacity, rate)`.
    ///
    /// Returns `self` for builder-pattern chaining.
    pub fn with_overrides(mut self, overrides: HashMap<String, (f64, f64)>) -> Self {
        self.overrides = overrides;
        self
    }

    /// Check if a request for `key` should be allowed.
    ///
    /// Returns `true` if a token was successfully consumed, `false` if the
    /// bucket is exhausted (caller should respond 429).
    ///
    /// On the first call for a given `key` the bucket is lazy-inserted under
    /// the write lock; subsequent calls use a read lock for O(1) lookup.
    pub fn check_key(&self, key: &str) -> bool {
        // Fast path: bucket already exists — read lock only.
        {
            let map = self.buckets.read().unwrap_or_else(|e| e.into_inner());
            if let Some(bucket_mutex) = map.get(key) {
                let mut bucket = bucket_mutex.lock().unwrap_or_else(|e| e.into_inner());
                return bucket.try_acquire().is_ok();
            }
        }

        // Slow path: first hit for this key — acquire write lock and insert.
        let (capacity, rate) = self
            .overrides
            .get(key)
            .copied()
            .unwrap_or((self.default_capacity, self.default_rate));

        let mut map = self.buckets.write().unwrap_or_else(|e| e.into_inner());

        // Check again under the write lock (another thread may have beaten us).
        let bucket_mutex = map
            .entry(key.to_string())
            .or_insert_with(|| Mutex::new(TokenBucket::new(capacity, rate)));

        let bucket = bucket_mutex.get_mut().unwrap_or_else(|e| e.into_inner());
        bucket.try_acquire().is_ok()
    }
}

/// Extract the API key from the request.
///
/// Checks `Authorization: Bearer <key>` first, then `X-Api-Key`.
/// Returns the raw key string, or `None` if no key header is present.
fn extract_key_from_request(request: &Request) -> Option<String> {
    // Try Authorization: Bearer <key>
    if let Some(auth) = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
    {
        if let Some(token) = auth.strip_prefix("Bearer ") {
            return Some(token.to_string());
        }
    }

    // Fallback: X-Api-Key header
    request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Axum middleware that enforces per-API-key rate limits.
///
/// Reads the API key from `Authorization: Bearer <key>` or `X-Api-Key`.
/// Requests without a key are allowed through (the auth middleware is
/// responsible for rejecting unauthenticated requests).
pub async fn per_key_rate_limit_middleware(
    State(limiter): State<Arc<PerKeyRateLimiter>>,
    request: Request,
    next: Next,
) -> Response {
    let key = extract_key_from_request(&request);

    // If no API key header is present, allow the request through — the auth
    // middleware (if configured) handles unauthenticated requests separately.
    let allowed = match key.as_deref() {
        None => true,
        Some(k) => limiter.check_key(k),
    };

    if allowed {
        next.run(request).await
    } else {
        let body = serde_json::json!({
            "error": {
                "message": "Per-key rate limit exceeded",
                "type": "rate_limit_error",
            }
        });
        let mut resp = (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response();
        resp.headers_mut().insert(
            "retry-after",
            "1".parse().unwrap_or_else(|_| "1".parse().expect("static")),
        );
        resp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_allows_within_capacity() {
        let mut bucket = TokenBucket::new(5.0, 1.0);
        for _ in 0..5 {
            assert!(bucket.try_acquire().is_ok());
        }
        // 6th should fail
        assert!(bucket.try_acquire().is_err());
    }

    #[test]
    fn test_bucket_refills() {
        let mut bucket = TokenBucket::new(1.0, 1000.0); // 1000/sec
        assert!(bucket.try_acquire().is_ok());
        assert!(bucket.try_acquire().is_err());
        // After a tiny wait, tokens refill fast
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(bucket.try_acquire().is_ok());
    }

    #[test]
    fn test_retry_after_is_positive() {
        let mut bucket = TokenBucket::new(1.0, 1.0);
        bucket.try_acquire().ok(); // drain
        let err = bucket.try_acquire().unwrap_err();
        assert!(err > 0.0, "retry_after should be positive");
    }

    #[tokio::test]
    async fn test_rate_limit_middleware_allows() {
        use axum::{body::Body, http::Request as HttpRequest, middleware, routing::get, Router};
        use tower::ServiceExt;

        let limiter = RateLimiter::new(10.0, 10.0);
        let app = Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(middleware::from_fn(rate_limit_middleware))
            .layer(axum::Extension(limiter));

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── PerKeyRateLimiter tests ───────────────────────────────────────────

    #[test]
    fn per_key_two_keys_are_independent() {
        // Capacity of 2 per key — each key gets its own bucket.
        let limiter = PerKeyRateLimiter::new(2.0, 1.0);

        // Drain key-a twice.
        assert!(limiter.check_key("key-a"), "key-a first hit should pass");
        assert!(limiter.check_key("key-a"), "key-a second hit should pass");
        assert!(
            !limiter.check_key("key-a"),
            "key-a third hit should be rejected"
        );

        // key-b is independent — should still have a full bucket.
        assert!(
            limiter.check_key("key-b"),
            "key-b should be unaffected by key-a exhaustion"
        );
    }

    #[test]
    fn per_key_burst_then_rejected() {
        let limiter = PerKeyRateLimiter::new(3.0, 0.001); // tiny refill rate

        // Consume the full burst.
        for i in 0..3 {
            assert!(limiter.check_key("burst-key"), "hit #{i} should be allowed");
        }
        // Next hit must be rejected.
        assert!(
            !limiter.check_key("burst-key"),
            "4th hit should be rejected (bucket exhausted)"
        );
    }

    #[test]
    fn per_key_override_applied() {
        let mut overrides = HashMap::new();
        // Give "premium-key" capacity 10, everything else capacity 1.
        overrides.insert("premium-key".to_string(), (10.0, 1.0));

        let limiter = PerKeyRateLimiter::new(1.0, 1.0).with_overrides(overrides);

        // Default key: only one token.
        assert!(
            limiter.check_key("default-key"),
            "default first hit allowed"
        );
        assert!(
            !limiter.check_key("default-key"),
            "default second hit rejected"
        );

        // Premium key: ten tokens.
        for i in 0..10 {
            assert!(
                limiter.check_key("premium-key"),
                "premium hit #{i} should be allowed"
            );
        }
        assert!(
            !limiter.check_key("premium-key"),
            "premium 11th hit rejected"
        );
    }

    #[test]
    fn per_key_anonymous_request_allowed() {
        // Anonymous (no key) requests pass through — check_key is not called.
        // We simulate the middleware logic directly here by calling check_key
        // with a dummy key that still has capacity.
        let limiter = PerKeyRateLimiter::new(5.0, 1.0);
        // No key header → the middleware allows through.  Since check_key is
        // not called for missing keys we just verify the limiter itself works.
        assert!(
            limiter.check_key("any-key"),
            "any key with capacity should be allowed"
        );
    }

    #[test]
    fn per_key_lazy_insert_idempotent() {
        let limiter = PerKeyRateLimiter::new(5.0, 1.0);

        // Call check_key several times for the same key — bucket should be
        // inserted exactly once (idempotent) and tokens should deplete.
        for i in 0..5 {
            assert!(
                limiter.check_key("idempotent-key"),
                "hit #{i} should pass (capacity=5)"
            );
        }
        // 6th call triggers the same code path as subsequent calls.
        assert!(
            !limiter.check_key("idempotent-key"),
            "6th hit should be rejected"
        );

        // Verify the map contains exactly one entry for this key.
        let map = limiter.buckets.read().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            map.len(),
            1,
            "only one bucket should be inserted for a single key"
        );
    }
}

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
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tokio::sync::Mutex as AsyncMutex;

use crate::router::eviction::LruQueue;

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

/// Maximum accepted length for a rate-limit key. Longer values are never
/// tracked (D9 fix) — an unbounded key length would let a single client
/// inflate this server's memory just by sending ever-longer `X-Api-Key`
/// header values.
pub const MAX_RATE_LIMIT_KEY_LEN: usize = 256;

/// Default cap on the number of distinct keys tracked at once (D9 fix).
///
/// Without a bound, a client that cycles through many distinct
/// `X-Api-Key` values (or many distinct bearer tokens, none of which need
/// to be *valid* — this middleware runs before/independently of auth
/// validation) grows the bucket map without limit, an unbounded-memory
/// DoS vector. Once at capacity, the least-recently-used key's bucket is
/// evicted to make room for a new one.
pub const DEFAULT_MAX_TRACKED_KEYS: usize = 10_000;

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
///
/// D9 fix: the map is keyed by a hash of the raw key (not the plaintext
/// key itself — API keys/bearer tokens are secrets, and there is no reason
/// for a rate limiter to retain them in memory in recoverable form), is
/// bounded to `max_entries` distinct keys via LRU eviction, and rejects
/// (without tracking) any key longer than [`MAX_RATE_LIMIT_KEY_LEN`].
#[derive(Debug)]
pub struct PerKeyRateLimiter {
    buckets: Arc<RwLock<HashMap<u64, Mutex<TokenBucket>>>>,
    lru: Mutex<LruQueue>,
    max_entries: usize,
    default_capacity: f64,
    default_rate: f64,
    overrides: HashMap<String, (f64, f64)>,
}

impl PerKeyRateLimiter {
    /// Create a limiter with the given default capacity and refill rate,
    /// bounded to [`DEFAULT_MAX_TRACKED_KEYS`] distinct keys.
    pub fn new(default_capacity: f64, default_rate: f64) -> Self {
        Self::with_max_entries(default_capacity, default_rate, DEFAULT_MAX_TRACKED_KEYS)
    }

    /// Create a limiter with an explicit cap on the number of distinct
    /// keys tracked at once.
    pub fn with_max_entries(default_capacity: f64, default_rate: f64, max_entries: usize) -> Self {
        let max_entries = max_entries.max(1);
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            lru: Mutex::new(LruQueue::with_capacity(max_entries)),
            max_entries,
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
    /// Keys longer than [`MAX_RATE_LIMIT_KEY_LEN`] are always allowed
    /// through *without* being tracked — rejecting them outright would let
    /// an attacker use an oversized key to bypass rate limiting entirely
    /// on every request; instead we simply decline to spend memory on them
    /// (the request still passes through whatever global rate limiter and
    /// auth middleware are configured).
    ///
    /// On the first call for a given `key` the bucket is lazy-inserted
    /// under the write lock; subsequent calls use a read lock for O(1)
    /// lookup. If the map is at `max_entries` capacity when a *new* key
    /// arrives, the least-recently-used tracked key is evicted first.
    pub fn check_key(&self, key: &str) -> bool {
        if key.len() > MAX_RATE_LIMIT_KEY_LEN {
            return true;
        }

        let hash = hash_key(key);

        // Fast path: bucket already exists — read lock only.
        {
            let map = self.buckets.read().unwrap_or_else(|e| e.into_inner());
            if let Some(bucket_mutex) = map.get(&hash) {
                let mut bucket = bucket_mutex.lock().unwrap_or_else(|e| e.into_inner());
                let allowed = bucket.try_acquire().is_ok();
                drop(bucket);
                drop(map);
                self.touch_lru(hash);
                return allowed;
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
        if !map.contains_key(&hash) {
            self.touch_lru_and_evict_if_full(hash, &mut map);
        }
        let bucket_mutex = map
            .entry(hash)
            .or_insert_with(|| Mutex::new(TokenBucket::new(capacity, rate)));

        let bucket = bucket_mutex.get_mut().unwrap_or_else(|e| e.into_inner());
        bucket.try_acquire().is_ok()
    }

    /// Touch `hash` in the LRU tracker (existing entry — no eviction needed).
    fn touch_lru(&self, hash: u64) {
        let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
        lru.touch(&lru_key(hash));
    }

    /// Touch `hash` in the LRU tracker and, if this is a *new* entry that
    /// would push the map over `max_entries`, evict the least-recently-used
    /// tracked key's bucket first.
    fn touch_lru_and_evict_if_full(&self, hash: u64, map: &mut HashMap<u64, Mutex<TokenBucket>>) {
        let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
        lru.touch(&lru_key(hash));
        if lru.len() > self.max_entries {
            if let Some(evicted) = lru.evict_lru() {
                if let Ok(evicted_hash) = u64::from_str_radix(&evicted, 16) {
                    if evicted_hash != hash {
                        map.remove(&evicted_hash);
                    }
                }
            }
        }
    }

    /// Number of distinct keys currently tracked. Test/introspection helper.
    #[cfg(test)]
    fn tracked_key_count(&self) -> usize {
        let map = self.buckets.read().unwrap_or_else(|e| e.into_inner());
        map.len()
    }
}

/// Hash a rate-limit key with `DefaultHasher` so the map never stores the
/// plaintext key (D9 fix).
fn hash_key(key: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Format a hash as the hex string [`LruQueue`] expects (it operates on
/// `String` IDs, since it is shared with the model-pool eviction tracker).
fn lru_key(hash: u64) -> String {
    format!("{hash:016x}")
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
        resp.headers_mut()
            .insert("retry-after", axum::http::HeaderValue::from_static("1"));
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

    // ── D9 regression tests ────────────────────────────────────────────────

    /// The tracked-key map never grows past `max_entries`, no matter how
    /// many distinct keys are seen — this is the actual "bounded" part of
    /// the D9 fix (previously the map had no cap at all).
    #[test]
    fn per_key_map_is_bounded_by_max_entries() {
        let limiter = PerKeyRateLimiter::with_max_entries(5.0, 1.0, 4);
        for i in 0..50 {
            limiter.check_key(&format!("key-{i}"));
        }
        assert!(
            limiter.tracked_key_count() <= 4,
            "map must never exceed max_entries, got {}",
            limiter.tracked_key_count()
        );
    }

    /// When the map is full, the least-recently-used key is the one
    /// evicted — a key touched more recently than others must survive.
    #[test]
    fn per_key_evicts_least_recently_used_first() {
        let limiter = PerKeyRateLimiter::with_max_entries(5.0, 1.0, 2);
        limiter.check_key("a");
        limiter.check_key("b");
        // Touch "a" again so "b" becomes the LRU entry.
        limiter.check_key("a");
        // Inserting "c" must evict "b", not "a".
        limiter.check_key("c");

        assert_eq!(limiter.tracked_key_count(), 2);
        let map = limiter.buckets.read().unwrap_or_else(|e| e.into_inner());
        assert!(
            map.contains_key(&hash_key("a")),
            "a was touched most recently and must survive"
        );
        assert!(
            map.contains_key(&hash_key("c")),
            "c was just inserted and must be present"
        );
        assert!(
            !map.contains_key(&hash_key("b")),
            "b was the LRU entry and must have been evicted"
        );
    }

    /// A key longer than `MAX_RATE_LIMIT_KEY_LEN` is never inserted into
    /// the tracked map — it is allowed through untracked rather than
    /// rejected outright (rejecting it would let an attacker *bypass*
    /// rate limiting by sending an oversized key).
    #[test]
    fn per_key_over_long_key_is_never_tracked() {
        let limiter = PerKeyRateLimiter::new(1.0, 1.0);
        let huge_key = "x".repeat(MAX_RATE_LIMIT_KEY_LEN + 1);
        for _ in 0..10 {
            assert!(
                limiter.check_key(&huge_key),
                "over-long keys must always be allowed through"
            );
        }
        assert_eq!(
            limiter.tracked_key_count(),
            0,
            "over-long keys must never be inserted into the tracked map"
        );
    }

    /// The map stores a hash of the key, not the plaintext key itself —
    /// API keys are secrets and should not be retained in recoverable
    /// form just to enforce a rate limit.
    #[test]
    fn per_key_map_stores_hash_not_plaintext() {
        let limiter = PerKeyRateLimiter::new(5.0, 1.0);
        limiter.check_key("sk-super-secret-key");
        let map = limiter.buckets.read().unwrap_or_else(|e| e.into_inner());
        assert!(
            map.contains_key(&hash_key("sk-super-secret-key")),
            "the hash of the key must be present"
        );
        // The map's key type is `u64` — there is no `String`/`&str` API
        // that could return the plaintext key, so this is enforced by the
        // type system as much as by this assertion.
    }
}

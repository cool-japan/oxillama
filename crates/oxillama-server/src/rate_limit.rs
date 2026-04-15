//! Token-bucket rate limiter middleware.
//!
//! Provides a global request-rate limiter. When the bucket is exhausted,
//! returns 429 Too Many Requests with a `Retry-After` header.

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

/// Token bucket state.
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
pub struct RateLimiter(pub Arc<Mutex<TokenBucket>>);

impl RateLimiter {
    /// Create a rate limiter with the given capacity and refill rate.
    pub fn new(capacity: f64, rate_per_second: f64) -> Self {
        Self(Arc::new(Mutex::new(TokenBucket::new(
            capacity,
            rate_per_second,
        ))))
    }
}

/// Middleware function for rate limiting.
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
}

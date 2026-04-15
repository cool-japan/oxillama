//! Error types for the HTTP API server.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

/// Result type alias for server operations.
pub type ServerResult<T> = Result<T, ServerError>;

/// Errors that can occur in the API server.
#[derive(Error, Debug)]
pub enum ServerError {
    /// Failed to bind to the specified address.
    #[error("failed to bind to {addr}: {source}")]
    BindError {
        /// The address that failed to bind.
        addr: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// Error from the inference runtime.
    #[error("runtime error: {0}")]
    Runtime(#[from] oxillama_runtime::RuntimeError),

    /// JSON serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid request parameters.
    #[error("invalid request: {message}")]
    InvalidRequest {
        /// Description of what was wrong.
        message: String,
    },

    /// Model is not ready for inference.
    #[error("model not ready")]
    ModelNotReady,

    /// The inference request queue is full; the server is overloaded.
    #[error("inference queue is full — server overloaded")]
    QueueFull,

    /// The inference worker has exited; no new requests can be processed.
    #[error("inference worker is no longer running")]
    WorkerDead,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = match &self {
            ServerError::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            ServerError::ModelNotReady => StatusCode::SERVICE_UNAVAILABLE,
            ServerError::QueueFull => StatusCode::TOO_MANY_REQUESTS,
            ServerError::WorkerDead => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": match &self {
                    ServerError::InvalidRequest { .. } => "invalid_request_error",
                    ServerError::ModelNotReady => "service_unavailable",
                    ServerError::QueueFull => "rate_limit_error",
                    ServerError::WorkerDead => "service_unavailable",
                    _ => "internal_error",
                },
            }
        });

        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    fn status_of(err: ServerError) -> StatusCode {
        let resp = err.into_response();
        resp.status()
    }

    #[test]
    fn test_invalid_request_returns_400() {
        let err = ServerError::InvalidRequest {
            message: "bad param".to_string(),
        };
        assert_eq!(status_of(err), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_model_not_ready_returns_503() {
        assert_eq!(
            status_of(ServerError::ModelNotReady),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn test_queue_full_returns_429() {
        assert_eq!(
            status_of(ServerError::QueueFull),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[test]
    fn test_worker_dead_returns_503() {
        assert_eq!(
            status_of(ServerError::WorkerDead),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn test_serialization_error_returns_500() {
        // Construct a serde_json::Error via invalid JSON parsing.
        let json_err = serde_json::from_str::<serde_json::Value>("not json")
            .expect_err("parsing invalid JSON should fail");
        let err = ServerError::Serialization(json_err);
        assert_eq!(status_of(err), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_error_display_invalid_request() {
        let err = ServerError::InvalidRequest {
            message: "missing field".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("missing field"),
            "display should contain message: {msg}"
        );
    }

    #[test]
    fn test_error_display_model_not_ready() {
        let msg = ServerError::ModelNotReady.to_string();
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_error_display_queue_full() {
        let msg = ServerError::QueueFull.to_string();
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_error_display_worker_dead() {
        let msg = ServerError::WorkerDead.to_string();
        assert!(!msg.is_empty());
    }
}

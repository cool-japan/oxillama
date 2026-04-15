//! Server-Sent Events (SSE) streaming support.
//!
//! Used for streaming chat completion responses in the OpenAI-compatible API.

use serde::Serialize;

/// A single SSE event for streaming responses.
#[derive(Debug, Clone, Serialize)]
pub struct SseEvent {
    /// The event data payload (JSON-encoded).
    pub data: String,
}

impl SseEvent {
    /// Create a new SSE event from a serializable payload.
    pub fn new<T: Serialize>(payload: &T) -> Result<Self, serde_json::Error> {
        Ok(Self {
            data: serde_json::to_string(payload)?,
        })
    }

    /// Create a `"[DONE]"` termination event.
    pub fn done() -> Self {
        Self {
            data: "[DONE]".to_string(),
        }
    }

    /// Format as an SSE text line.
    pub fn to_sse_string(&self) -> String {
        format!("data: {}\n\n", self.data)
    }
}

//! WebSocket streaming endpoint for `/v1/chat/ws`.
//!
//! Clients that prefer full-duplex communication over SSE can use this
//! endpoint.  The message protocol mirrors the token-by-token structure of
//! the SSE streaming response but sent as JSON-framed WebSocket text
//! messages.
//!
//! ## Protocol
//!
//! 1. Client upgrades to WebSocket (`GET /v1/chat/ws`).
//! 2. Client sends a single JSON text frame containing a [`WsRequest`].
//! 3. Server streams back [`WsEvent`] JSON text frames until generation is
//!    complete.
//! 4. Server sends a final `{"type":"done", …}` frame and closes the
//!    connection.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

// ── Request / response types ─────────────────────────────────────────────

/// Incoming WebSocket request payload (subset of OpenAI chat completion
/// parameters).
#[derive(Debug, Deserialize)]
pub struct WsRequest {
    /// Model identifier (currently ignored; the loaded model is always used).
    pub model: Option<String>,
    /// Conversation messages.
    pub messages: Vec<WsMessage>,
    /// Maximum tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

/// A single conversation message.
#[derive(Debug, Deserialize, Serialize)]
pub struct WsMessage {
    /// Role: `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Text content of the message.
    pub content: String,
}

fn default_max_tokens() -> u32 {
    512
}

fn default_temperature() -> f32 {
    0.7
}

// ── Outgoing events ──────────────────────────────────────────────────────

/// Outgoing WebSocket event (one JSON text frame per variant).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsEvent {
    /// A single generated token delta.
    Token {
        /// Decoded token text.
        delta: String,
    },
    /// Final frame sent after all tokens have been streamed.
    Done {
        /// Reason generation stopped (e.g., `"stop"`, `"length"`).
        finish_reason: String,
        /// Token usage summary.
        usage: UsageSummary,
    },
    /// Error frame sent when request parsing or generation fails.
    Error {
        /// Human-readable error description.
        message: String,
    },
}

/// Token usage counters attached to the `done` event.
#[derive(Debug, Serialize)]
pub struct UsageSummary {
    /// Number of prompt tokens consumed.
    pub prompt_tokens: u32,
    /// Number of completion tokens generated.
    pub completion_tokens: u32,
}

// ── Handler ──────────────────────────────────────────────────────────────

/// Axum handler: upgrade the connection to WebSocket and delegate to
/// `handle_socket`.
pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Drive a single WebSocket session end-to-end.
///
/// 1. Receive one text frame containing a [`WsRequest`].
/// 2. Stream placeholder token events (real inference is wired in via the
///    inference worker queue in a future integration step).
/// 3. Send a `done` event and close.
async fn handle_socket(mut socket: WebSocket, _state: Arc<AppState>) {
    // ── Step 1: receive the request ──────────────────────────────────────
    let text = match receive_text(&mut socket).await {
        Some(t) => t,
        None => return,
    };

    let req: WsRequest = match serde_json::from_str(&text) {
        Ok(r) => r,
        Err(e) => {
            send_error(&mut socket, &format!("Invalid JSON request: {e}")).await;
            return;
        }
    };

    // ── Step 2: stream tokens ────────────────────────────────────────────
    // Placeholder: echo a fixed token stream.
    // In the full integration this will submit `req` to `_state.queue` and
    // forward streamed tokens from the response channel.
    let _ = req.model; // suppress unused field warning until full integration
    let stub_tokens: &[&str] = &["Hello", " from", " OxiLLaMa", " via", " WebSocket"];
    let mut sent = 0u32;
    for token in stub_tokens {
        let event = WsEvent::Token {
            delta: (*token).to_string(),
        };
        if !send_event(&mut socket, &event).await {
            return;
        }
        sent += 1;
    }

    // ── Step 3: send done and close ──────────────────────────────────────
    let done = WsEvent::Done {
        finish_reason: "stop".to_string(),
        usage: UsageSummary {
            prompt_tokens: 0,
            completion_tokens: sent,
        },
    };
    send_event(&mut socket, &done).await;
    // Close is implicit when socket is dropped.
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Receive the next text frame from the socket.
///
/// Returns `None` if the client sent a close frame or the connection was
/// dropped.  Sends an error event and returns `None` for non-text frames.
async fn receive_text(socket: &mut WebSocket) -> Option<String> {
    match socket.recv().await {
        Some(Ok(Message::Text(t))) => Some(t.to_string()),
        Some(Ok(Message::Close(_))) | None => None,
        Some(Ok(_)) => {
            send_error(socket, "Expected a JSON text frame as the first message").await;
            None
        }
        Some(Err(e)) => {
            send_error(socket, &format!("WebSocket receive error: {e}")).await;
            None
        }
    }
}

/// Serialize `event` to JSON and send it as a text frame.
///
/// Returns `true` on success, `false` if the connection is broken.
async fn send_event(socket: &mut WebSocket, event: &WsEvent) -> bool {
    match serde_json::to_string(event) {
        Ok(json) => socket.send(Message::Text(json.into())).await.is_ok(),
        Err(_) => false,
    }
}

/// Send an error event best-effort (errors during sending are ignored).
async fn send_error(socket: &mut WebSocket, message: &str) {
    let event = WsEvent::Error {
        message: message.to_string(),
    };
    if let Ok(json) = serde_json::to_string(&event) {
        let _ = socket.send(Message::Text(json.into())).await;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_event_token_serializes_correctly() {
        let event = WsEvent::Token {
            delta: "hello".into(),
        };
        let json = serde_json::to_string(&event).expect("serialize failed");
        assert!(json.contains("\"type\":\"token\""));
        assert!(json.contains("\"delta\":\"hello\""));
    }

    #[test]
    fn ws_event_done_serializes_correctly() {
        let event = WsEvent::Done {
            finish_reason: "stop".into(),
            usage: UsageSummary {
                prompt_tokens: 5,
                completion_tokens: 10,
            },
        };
        let json = serde_json::to_string(&event).expect("serialize failed");
        assert!(json.contains("\"type\":\"done\""));
        assert!(json.contains("\"finish_reason\":\"stop\""));
        assert!(json.contains("\"prompt_tokens\":5"));
        assert!(json.contains("\"completion_tokens\":10"));
    }

    #[test]
    fn ws_event_error_serializes_correctly() {
        let event = WsEvent::Error {
            message: "oops".into(),
        };
        let json = serde_json::to_string(&event).expect("serialize failed");
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"message\":\"oops\""));
    }

    #[test]
    fn ws_request_deserializes_with_defaults() {
        let json = r#"{"messages": [{"role": "user", "content": "hello"}]}"#;
        let req: WsRequest = serde_json::from_str(json).expect("deserialize failed");
        assert_eq!(req.max_tokens, 512);
        assert!((req.temperature - 0.7).abs() < 0.001);
        assert!(req.model.is_none());
    }

    #[test]
    fn ws_request_deserializes_explicit_fields() {
        let json = r#"{
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 128,
            "temperature": 0.5
        }"#;
        let req: WsRequest = serde_json::from_str(json).expect("deserialize failed");
        assert_eq!(req.model.as_deref(), Some("local"));
        assert_eq!(req.max_tokens, 128);
        assert!((req.temperature - 0.5).abs() < 0.001);
    }
}

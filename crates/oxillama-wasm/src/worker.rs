//! Web-worker message-passing helpers for offloaded inference.
//!
//! JavaScript usage:
//! ```js
//! const worker = new Worker('oxillama_worker.js');
//! worker.postMessage({type: 'generate', prompt: 'Hello', maxTokens: 128});
//! worker.onmessage = (e) => {
//!   if (e.data.type === 'token') console.log(e.data.delta);
//!   if (e.data.type === 'done')  console.log('done:', e.data.text);
//! };
//! ```
//!
//! This module provides the Rust-side message types and a dispatcher
//! that JS can call from inside a Worker context.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// An incoming message from the JavaScript host.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerInMessage {
    /// Generate text from a prompt.
    Generate {
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: u32,
        #[serde(default = "default_temperature")]
        temperature: f32,
    },
    /// Reset the model context.
    Reset,
    /// Ping for health-check.
    Ping,
}

fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f32 {
    0.7
}

/// An outgoing message to the JavaScript host.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerOutMessage {
    /// A single generated token delta.
    Token { delta: String },
    /// Generation complete.
    Done {
        text: String,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    /// Worker is healthy.
    Pong,
    /// An error occurred.
    Error { message: String },
}

/// Parse an incoming JSON message from JavaScript.
///
/// Returns a JSON string representing the response message.
///
/// # Errors
/// Returns a description of the parse error if the JSON is invalid.
#[wasm_bindgen(js_name = parseWorkerMessage)]
pub fn parse_worker_message(json: &str) -> Result<JsValue, JsValue> {
    let msg: WorkerInMessage =
        serde_json::from_str(json).map_err(|e| JsValue::from_str(&format!("parse error: {e}")))?;

    // Return the canonical re-serialized form so JS can inspect the type
    let response = match msg {
        WorkerInMessage::Ping => WorkerOutMessage::Pong,
        WorkerInMessage::Reset => WorkerOutMessage::Done {
            text: "[context reset]".into(),
            prompt_tokens: 0,
            completion_tokens: 0,
        },
        WorkerInMessage::Generate {
            prompt,
            max_tokens,
            temperature: _,
        } => {
            // Stub — real implementation would call the engine
            WorkerOutMessage::Done {
                text: format!(
                    "[stub: received prompt of {} chars, max_tokens={}]",
                    prompt.len(),
                    max_tokens
                ),
                prompt_tokens: 0,
                completion_tokens: 0,
            }
        }
    };

    let json = serde_json::to_string(&response)
        .map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))?;
    Ok(JsValue::from_str(&json))
}

/// Serialize a token delta event to JSON for posting back to the host.
///
/// # Errors
/// Returns an error if serialization fails.
#[wasm_bindgen(js_name = workerTokenEvent)]
pub fn worker_token_event(delta: &str) -> Result<JsValue, JsValue> {
    let msg = WorkerOutMessage::Token {
        delta: delta.to_string(),
    };
    serde_json::to_string(&msg)
        .map(|s| JsValue::from_str(&s))
        .map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ping_message() {
        let json = r#"{"type":"ping"}"#;
        let msg: WorkerInMessage = serde_json::from_str(json).expect("parse");
        assert!(matches!(msg, WorkerInMessage::Ping));
    }

    #[test]
    fn parse_generate_message_with_defaults() {
        let json = r#"{"type":"generate","prompt":"hello"}"#;
        let msg: WorkerInMessage = serde_json::from_str(json).expect("parse");
        if let WorkerInMessage::Generate {
            prompt, max_tokens, ..
        } = msg
        {
            assert_eq!(prompt, "hello");
            assert_eq!(max_tokens, 512);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn serialize_token_event() {
        let msg = WorkerOutMessage::Token {
            delta: "hello".into(),
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"token\""));
        assert!(json.contains("\"delta\":\"hello\""));
    }

    #[test]
    fn serialize_done_event() {
        let msg = WorkerOutMessage::Done {
            text: "hi".into(),
            prompt_tokens: 3,
            completion_tokens: 1,
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"done\""));
    }

    #[test]
    fn pong_serializes_correctly() {
        let msg = WorkerOutMessage::Pong;
        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"pong\""));
    }

    #[test]
    fn parse_reset_message() {
        let json = r#"{"type":"reset"}"#;
        let msg: WorkerInMessage = serde_json::from_str(json).expect("parse");
        assert!(matches!(msg, WorkerInMessage::Reset));
    }

    #[test]
    fn parse_generate_with_explicit_tokens() {
        let json = r#"{"type":"generate","prompt":"hi","max_tokens":64,"temperature":0.5}"#;
        let msg: WorkerInMessage = serde_json::from_str(json).expect("parse");
        if let WorkerInMessage::Generate {
            max_tokens,
            temperature,
            ..
        } = msg
        {
            assert_eq!(max_tokens, 64);
            assert!((temperature - 0.5).abs() < 1e-6);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn error_message_serializes() {
        let msg = WorkerOutMessage::Error {
            message: "oops".into(),
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"message\":\"oops\""));
    }
}

//! Tool-invocation runtime callbacks.
//!
//! Provides infrastructure for detecting, parsing, and dispatching tool calls
//! produced by language models during generation.
//!
//! ## Overview
//!
//! Different model families emit tool calls using different delimiter syntax:
//! - LLaMA 3: `<|tool_call|>{ ... }<|/tool_call|>`
//! - Qwen: `<tool_call>{ ... }</tool_call>`
//! - Mistral: `[TOOL_CALLS][ ... ]`
//! - Custom: user-supplied open/close delimiters
//!
//! The [`ToolCallDetector`] accumulates token text, detects open/close
//! delimiters, validates the JSON payload, and fires whenever a complete tool
//! call is parsed.
//!
//! Tool results can be queued for injection back into the generation stream
//! via the engine's injection queue mechanism.

use serde_json::Value;
use std::sync::Arc;

// ─── Core trait ───────────────────────────────────────────────────────────────

/// Dispatches tool calls to registered handler implementations.
///
/// Implement this trait to handle tool invocations produced by the model
/// during generation.  The implementation must be `Send + Sync` so it can
/// be shared across threads and stored in `Arc`.
///
/// # Example
///
/// ```
/// use oxillama_runtime::tool_dispatch::{ToolDispatcher, ToolResult};
/// use serde_json::Value;
///
/// struct WeatherTool;
///
/// impl ToolDispatcher for WeatherTool {
///     fn invoke(&self, name: &str, args: &Value) -> ToolResult {
///         if name == "get_weather" {
///             ToolResult::Ok(Value::String("sunny, 22°C".to_string()))
///         } else {
///             ToolResult::Err(format!("unknown tool: {name}"))
///         }
///     }
/// }
/// ```
pub trait ToolDispatcher: Send + Sync {
    /// Invoke the named tool with the given JSON arguments.
    ///
    /// Returns [`ToolResult::Ok`] with the tool's output value on success, or
    /// [`ToolResult::Err`] with an error message on failure.
    fn invoke(&self, name: &str, args: &Value) -> ToolResult;
}

/// Result of a tool invocation.
#[derive(Debug, Clone)]
pub enum ToolResult {
    /// Successful invocation with a JSON result value.
    Ok(Value),
    /// Failed invocation with a human-readable error message.
    Err(String),
}

impl ToolResult {
    /// Format the result as a string suitable for injection into the token stream.
    pub fn as_injection_string(&self) -> String {
        match self {
            ToolResult::Ok(v) => {
                format!("<tool_result>{}</tool_result>", v)
            }
            ToolResult::Err(e) => {
                format!(
                    "<tool_result>{{\"error\":{}}}</tool_result>",
                    serde_json::json!(e)
                )
            }
        }
    }
}

// ─── Grammar ─────────────────────────────────────────────────────────────────

/// Specifies the delimiter syntax used by a given model for tool calls.
///
/// Choose the variant that matches your deployed model:
/// - [`Llama3`](ToolCallGrammar::Llama3) for LLaMA 3 models.
/// - [`Qwen`](ToolCallGrammar::Qwen) for Qwen / Qwen2 models.
/// - [`Mistral`](ToolCallGrammar::Mistral) for Mistral / Mixtral function-calling models.
/// - [`Custom`](ToolCallGrammar::Custom) for any other format.
#[derive(Debug, Clone)]
pub enum ToolCallGrammar {
    /// LLaMA 3 tool-call format: `<|tool_call|>...</|tool_call|>`.
    Llama3,
    /// Qwen / Qwen2 format: `<tool_call>...</tool_call>`.
    Qwen,
    /// Mistral function-calling format: `[TOOL_CALLS][...]`.
    Mistral,
    /// User-supplied open/close delimiter pair.
    Custom {
        /// Opening delimiter (e.g. `"<tool_call>"`).
        open: String,
        /// Closing delimiter (e.g. `"</tool_call>"`).
        close: String,
    },
}

impl ToolCallGrammar {
    /// Return the opening delimiter for this grammar variant.
    pub fn open_delimiter(&self) -> &str {
        match self {
            ToolCallGrammar::Llama3 => "<|tool_call|>",
            ToolCallGrammar::Qwen => "<tool_call>",
            ToolCallGrammar::Mistral => "[TOOL_CALLS][",
            ToolCallGrammar::Custom { open, .. } => open.as_str(),
        }
    }

    /// Return the closing delimiter for this grammar variant.
    pub fn close_delimiter(&self) -> &str {
        match self {
            ToolCallGrammar::Llama3 => "<|/tool_call|>",
            ToolCallGrammar::Qwen => "</tool_call>",
            ToolCallGrammar::Mistral => "]",
            ToolCallGrammar::Custom { close, .. } => close.as_str(),
        }
    }
}

// ─── Parsed tool call ─────────────────────────────────────────────────────────

/// A fully-parsed tool call extracted from the model's output stream.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments to pass to the tool as a JSON value.
    pub args: Value,
}

// ─── Detection state machine ─────────────────────────────────────────────────

/// Internal state of the tool-call detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolDetectionState {
    /// No tool call in progress; scanning for the open delimiter.
    Idle,
    /// Open delimiter has been seen; capturing token text until the close.
    Capturing,
}

/// Incremental tool-call detector.
///
/// Feed token text via [`feed`](ToolCallDetector::feed) as tokens are generated.
/// When a complete tool call (open delimiter + valid JSON + close delimiter)
/// is recognised, `feed` returns `Some(ToolCall)`.
///
/// The detector maintains a rolling scan buffer to handle delimiters that
/// span multiple tokens.
///
/// # Example
///
/// ```
/// use oxillama_runtime::tool_dispatch::{ToolCallDetector, ToolCallGrammar};
///
/// let mut detector = ToolCallDetector::new(ToolCallGrammar::Llama3);
/// let call = detector.feed("<|tool_call|>{\"name\":\"ping\",\"args\":{}}<|/tool_call|>");
/// assert!(call.is_some());
/// ```
pub struct ToolCallDetector {
    grammar: ToolCallGrammar,
    state: ToolDetectionState,
    /// Text accumulated since the start of the current token or since the last
    /// full-delimiter match candidate.
    buffer: String,
}

impl ToolCallDetector {
    /// Construct a new detector for the given grammar.
    pub fn new(grammar: ToolCallGrammar) -> Self {
        Self {
            grammar,
            state: ToolDetectionState::Idle,
            buffer: String::new(),
        }
    }

    /// Feed one token's decoded text into the detector.
    ///
    /// Returns `Some(ToolCall)` when a complete, valid tool call is detected.
    /// Returns `None` while the tool call is still accumulating or if the text
    /// is not a tool call.
    ///
    /// After returning `Some`, the detector automatically resets to `Idle`.
    /// This means it can detect multiple sequential tool calls: feed text
    /// from the second call and it will be detected in a subsequent call.
    pub fn feed(&mut self, token_text: &str) -> Option<ToolCall> {
        self.buffer.push_str(token_text);
        self.try_parse()
    }

    /// Reset the detector to the idle state, discarding any buffered content.
    pub fn reset(&mut self) {
        self.state = ToolDetectionState::Idle;
        self.buffer.clear();
    }

    // ─── Internal parsing ─────────────────────────────────────────────────────

    /// Try to parse a complete tool call from the current buffer.
    ///
    /// This is the core state machine: it searches for open and close
    /// delimiters in the buffer and attempts JSON parsing on the content
    /// between them.
    ///
    /// Multiple calls are detected by scanning for repeated open/close pairs.
    fn try_parse(&mut self) -> Option<ToolCall> {
        let open = self.grammar.open_delimiter().to_string();
        let close = self.grammar.close_delimiter().to_string();

        loop {
            match self.state {
                ToolDetectionState::Idle => {
                    // Look for the opening delimiter.
                    if let Some(start) = self.buffer.find(open.as_str()) {
                        // Discard everything before the open delimiter.
                        let after_open = start + open.len();
                        self.buffer = self.buffer[after_open..].to_string();
                        self.state = ToolDetectionState::Capturing;
                        // Fall through and look for the close delimiter.
                    } else {
                        // No open delimiter yet; keep only the trailing portion
                        // that could be a partial delimiter prefix.
                        self.trim_idle_buffer(&open);
                        return None;
                    }
                }

                ToolDetectionState::Capturing => {
                    if let Some(end) = self.buffer.find(close.as_str()) {
                        // Extract the JSON payload.
                        let payload = self.buffer[..end].trim().to_string();
                        // Consume past the close delimiter.
                        let after_close = end + close.len();
                        let remainder = self.buffer[after_close..].to_string();
                        self.buffer = remainder;
                        self.state = ToolDetectionState::Idle;

                        // Parse and validate the JSON.
                        if let Some(call) = parse_tool_call_json(&payload) {
                            return Some(call);
                        }
                        // Bad JSON — continue scanning the remainder.
                        // (fall back to Idle and try again)
                    } else {
                        // Close delimiter not yet seen; keep capturing.
                        return None;
                    }
                }
            }
        }
    }

    /// Trim the idle buffer to at most `max_suffix` chars that could be a
    /// prefix of the open delimiter.  Prevents unbounded growth of the buffer
    /// when no tool call is ever emitted.
    fn trim_idle_buffer(&mut self, open: &str) {
        let max_keep = open.len().saturating_sub(1);
        if self.buffer.len() > max_keep {
            let trim_to = self.buffer.len() - max_keep;
            self.buffer = self.buffer[trim_to..].to_string();
        }
    }
}

// ─── JSON parsing ─────────────────────────────────────────────────────────────

/// Parse a JSON string as a tool call object with `name` and `args` fields.
///
/// Accepts two JSON shapes:
/// 1. `{"name": "...", "args": { ... }}` — preferred
/// 2. `{"name": "...", "arguments": { ... }}` — OpenAI-compat alias
///
/// Returns `None` if the string is not valid JSON or the expected fields
/// are missing.
fn parse_tool_call_json(payload: &str) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(payload).ok()?;
    let obj = v.as_object()?;

    let name = obj.get("name")?.as_str()?.to_string();

    // Accept either "args" or "arguments" for OpenAI compatibility.
    let args = obj
        .get("args")
        .or_else(|| obj.get("arguments"))
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    Some(ToolCall { name, args })
}

// ─── Tool-dispatcher no-op helper ────────────────────────────────────────────

/// A no-op dispatcher that returns a stub `Ok(null)` for every tool call.
///
/// Useful for testing or when you want to detect tool calls but not execute
/// them yet.
pub struct NoOpDispatcher;

impl ToolDispatcher for NoOpDispatcher {
    fn invoke(&self, _name: &str, _args: &Value) -> ToolResult {
        ToolResult::Ok(Value::Null)
    }
}

/// Create a no-op dispatcher wrapped in an `Arc`.
pub fn no_op_dispatcher() -> Arc<dyn ToolDispatcher> {
    Arc::new(NoOpDispatcher)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── A: Basic detection ────────────────────────────────────────────────────

    /// (a) Complete LLaMA-3 tool call in a single feed() call.
    #[test]
    fn tool_call_detection_llama3() {
        let mut det = ToolCallDetector::new(ToolCallGrammar::Llama3);
        let result = det
            .feed(r#"<|tool_call|>{"name":"get_weather","args":{"city":"Tokyo"}}<|/tool_call|>"#);
        assert!(result.is_some(), "must detect a complete Llama3 tool call");
        let call = result.expect("detection should succeed");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.args["city"], Value::String("Tokyo".to_string()));
    }

    /// (b) Tool call where open delimiter, JSON body, and close delimiter arrive
    ///     in separate feed() calls (simulates streaming tokenizer output).
    #[test]
    fn tool_call_streamed_across_chunks() {
        let mut det = ToolCallDetector::new(ToolCallGrammar::Llama3);

        // Chunk 1: opening delimiter
        let r1 = det.feed("<|tool_call|>");
        assert!(r1.is_none(), "open delimiter alone must not fire");

        // Chunk 2: JSON body (no close yet)
        let r2 = det.feed(r#"{"name":"add","args":{"a":1,"b":2}}"#);
        assert!(r2.is_none(), "body without close must not fire");

        // Chunk 3: closing delimiter
        let r3 = det.feed("<|/tool_call|>");
        assert!(
            r3.is_some(),
            "close delimiter should complete the detection"
        );
        let call = r3.expect("detection should succeed");
        assert_eq!(call.name, "add");
        assert_eq!(call.args["a"], 1);
        assert_eq!(call.args["b"], 2);
    }

    /// (c) Malformed / unclosed JSON must not produce a ToolCall.
    #[test]
    fn malformed_json_does_not_return_call() {
        let mut det = ToolCallDetector::new(ToolCallGrammar::Llama3);

        // Send an open delimiter followed by unclosed JSON.
        let r1 = det.feed("<|tool_call|>{\"name\":\"broken\"");
        assert!(r1.is_none(), "partial JSON must not fire");

        // Never close — detector should stay in Capturing without panic.
        for _ in 0..5 {
            let r = det.feed("more garbage");
            assert!(r.is_none(), "unfinished tool call must not fire");
        }
    }

    /// (d) Two complete tool calls back-to-back in the same buffer.
    ///     The detector must fire twice (once per call).
    #[test]
    fn multiple_calls_sequentially() {
        let mut det = ToolCallDetector::new(ToolCallGrammar::Qwen);

        let r1 = det.feed(
            r#"<tool_call>{"name":"tool1","args":{"x":1}}</tool_call><tool_call>{"name":"tool2","args":{"y":2}}</tool_call>"#,
        );
        assert!(r1.is_some(), "first call must be detected");
        let c1 = r1.expect("first call");
        assert_eq!(c1.name, "tool1");

        // Second call should be detected on an empty feed (it's still in buffer).
        let r2 = det.feed("");
        assert!(r2.is_some(), "second call must be detected from remainder");
        let c2 = r2.expect("second call");
        assert_eq!(c2.name, "tool2");
    }

    // ── B: Grammar variant tests ──────────────────────────────────────────────

    /// Qwen format detection.
    #[test]
    fn tool_call_detection_qwen() {
        let mut det = ToolCallDetector::new(ToolCallGrammar::Qwen);
        let result = det.feed(r#"<tool_call>{"name":"calc","args":{"expr":"1+1"}}</tool_call>"#);
        assert!(result.is_some());
        let call = result.expect("qwen call");
        assert_eq!(call.name, "calc");
    }

    /// Mistral format detection.
    #[test]
    fn tool_call_detection_mistral() {
        let mut det = ToolCallDetector::new(ToolCallGrammar::Mistral);
        let result = det.feed(r#"[TOOL_CALLS][{"name":"search","args":{"q":"rust"}}]"#);
        assert!(result.is_some());
        let call = result.expect("mistral call");
        assert_eq!(call.name, "search");
        assert_eq!(call.args["q"], "rust");
    }

    /// Custom grammar detection.
    #[test]
    fn tool_call_detection_custom() {
        let mut det = ToolCallDetector::new(ToolCallGrammar::Custom {
            open: "<<TOOL>>".to_string(),
            close: "<</TOOL>>".to_string(),
        });
        let result = det.feed(r#"<<TOOL>>{"name":"echo","args":{"msg":"hi"}}<</TOOL>>"#);
        assert!(result.is_some());
        let call = result.expect("custom call");
        assert_eq!(call.name, "echo");
    }

    // ── C: Grammar delimiter accessors ───────────────────────────────────────

    #[test]
    fn grammar_delimiters_llama3() {
        let g = ToolCallGrammar::Llama3;
        assert_eq!(g.open_delimiter(), "<|tool_call|>");
        assert_eq!(g.close_delimiter(), "<|/tool_call|>");
    }

    #[test]
    fn grammar_delimiters_qwen() {
        let g = ToolCallGrammar::Qwen;
        assert_eq!(g.open_delimiter(), "<tool_call>");
        assert_eq!(g.close_delimiter(), "</tool_call>");
    }

    #[test]
    fn grammar_delimiters_mistral() {
        let g = ToolCallGrammar::Mistral;
        assert_eq!(g.open_delimiter(), "[TOOL_CALLS][");
        assert_eq!(g.close_delimiter(), "]");
    }

    #[test]
    fn grammar_delimiters_custom() {
        let g = ToolCallGrammar::Custom {
            open: "START".to_string(),
            close: "END".to_string(),
        };
        assert_eq!(g.open_delimiter(), "START");
        assert_eq!(g.close_delimiter(), "END");
    }

    // ── D: Reset test ─────────────────────────────────────────────────────────

    /// After reset(), the detector treats new input as if it had never
    /// seen the previous stream (can detect a new call from scratch).
    #[test]
    fn reset_clears_state() {
        let mut det = ToolCallDetector::new(ToolCallGrammar::Llama3);

        // Start a call but don't finish it.
        det.feed("<|tool_call|>{\"name\":\"half");
        assert_eq!(det.state, ToolDetectionState::Capturing);

        // Reset.
        det.reset();
        assert_eq!(det.state, ToolDetectionState::Idle);
        assert!(det.buffer.is_empty());

        // A fresh call after reset should still work.
        let r = det.feed(r#"<|tool_call|>{"name":"fresh","args":{}}<|/tool_call|>"#);
        assert!(r.is_some(), "should detect call after reset");
    }

    // ── E: ToolResult injection string ────────────────────────────────────────

    #[test]
    fn tool_result_ok_injection_string() {
        let result = ToolResult::Ok(Value::String("42°C".to_string()));
        let s = result.as_injection_string();
        assert!(s.contains("<tool_result>"), "must contain opening tag");
        assert!(s.contains("</tool_result>"), "must contain closing tag");
        assert!(s.contains("42°C"), "must contain result value");
    }

    #[test]
    fn tool_result_err_injection_string() {
        let result = ToolResult::Err("not found".to_string());
        let s = result.as_injection_string();
        assert!(s.contains("<tool_result>"), "must contain opening tag");
        assert!(s.contains("error"), "must contain error key");
    }

    // ── F: OpenAI-compat "arguments" field ────────────────────────────────────

    /// parse_tool_call_json must accept "arguments" as alias for "args".
    #[test]
    fn tool_call_arguments_alias() {
        let mut det = ToolCallDetector::new(ToolCallGrammar::Llama3);
        let r = det.feed(r#"<|tool_call|>{"name":"fn","arguments":{"k":"v"}}<|/tool_call|>"#);
        assert!(r.is_some(), "arguments alias should be accepted");
        let call = r.expect("call with arguments");
        assert_eq!(call.args["k"], "v");
    }

    // ── G: NoOpDispatcher ─────────────────────────────────────────────────────

    #[test]
    fn no_op_dispatcher_returns_ok_null() {
        let d = no_op_dispatcher();
        let result = d.invoke("anything", &Value::Null);
        assert!(matches!(result, ToolResult::Ok(Value::Null)));
    }
}

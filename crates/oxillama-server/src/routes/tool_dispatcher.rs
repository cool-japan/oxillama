//! Server-side `ToolDispatcher` implementation.
//!
//! When an incoming OpenAI chat request includes `tools: [...]`, the server
//! constructs a `ServerToolDispatcher` that holds the JSON Schema map for
//! all declared tools.  The dispatcher logs the tool call and returns a
//! placeholder result (`{ "result": "tool_call_logged" }`).
//!
//! # v0.1.3 scope
//!
//! Actual external tool invocation (HTTP callbacks, subprocess execution, etc.)
//! is out of scope for v0.1.3.  This implementation provides the plumbing so
//! that the engine can emit structured tool-call output and future versions can
//! swap in a real dispatcher.

use std::collections::HashMap;
use std::sync::Arc;

use oxillama_runtime::tool_dispatch::{ToolDispatcher, ToolResult};
use serde_json::Value;

/// A server-side dispatcher that logs tool invocations and returns a
/// placeholder result.
///
/// In v0.1.3 the dispatcher:
/// 1. Validates that the called tool name is in the registered schema map.
/// 2. Logs the call at `info` level.
/// 3. Returns `{ "result": "tool_call_logged" }` as a placeholder.
pub struct ServerToolDispatcher {
    /// Maps tool name → its JSON Schema (`parameters` object).
    pub schema_map: HashMap<String, Value>,
}

impl ServerToolDispatcher {
    /// Create a new dispatcher from a set of `(name, schema)` pairs.
    pub fn new(schema_map: HashMap<String, Value>) -> Self {
        Self { schema_map }
    }
}

impl ToolDispatcher for ServerToolDispatcher {
    fn invoke(&self, name: &str, args: &Value) -> ToolResult {
        if self.schema_map.contains_key(name) {
            tracing::info!(
                tool_name = name,
                args = %args,
                "tool call received (placeholder dispatcher — no external invocation in v0.1.3)"
            );
            ToolResult::Ok(serde_json::json!({ "result": "tool_call_logged" }))
        } else {
            tracing::warn!(tool_name = name, "unknown tool called");
            ToolResult::Err(format!("tool '{name}' is not registered on this server"))
        }
    }
}

/// Build an optional `Arc<dyn ToolDispatcher>` from a slice of OpenAI tool
/// definitions.
///
/// Returns `None` when the `tools` slice is empty.
pub fn build_tool_dispatcher(
    tools: &[crate::routes::tools::Tool],
) -> Option<Arc<dyn ToolDispatcher>> {
    if tools.is_empty() {
        return None;
    }
    let schema_map: HashMap<String, Value> = tools
        .iter()
        .map(|t| (t.function.name.clone(), t.function.parameters.clone()))
        .collect();

    Some(Arc::new(ServerToolDispatcher::new(schema_map)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_dispatcher() -> ServerToolDispatcher {
        let mut map = HashMap::new();
        map.insert(
            "get_weather".to_string(),
            json!({ "type": "object", "properties": { "location": { "type": "string" } } }),
        );
        ServerToolDispatcher::new(map)
    }

    #[test]
    fn dispatcher_returns_placeholder_for_known_tool() {
        let d = make_dispatcher();
        let result = d.invoke("get_weather", &json!({"location": "Tokyo"}));
        match result {
            ToolResult::Ok(v) => {
                assert_eq!(
                    v["result"].as_str(),
                    Some("tool_call_logged"),
                    "placeholder result mismatch: {v}"
                );
            }
            ToolResult::Err(e) => panic!("expected Ok, got Err: {e}"),
        }
    }

    #[test]
    fn dispatcher_returns_error_for_unknown_tool() {
        let d = make_dispatcher();
        let result = d.invoke("unknown_tool", &json!({}));
        match result {
            ToolResult::Ok(v) => panic!("expected Err, got Ok: {v}"),
            ToolResult::Err(msg) => assert!(
                msg.contains("not registered"),
                "error message should mention 'not registered': {msg}"
            ),
        }
    }

    #[test]
    fn build_tool_dispatcher_returns_none_for_empty_tools() {
        let result = build_tool_dispatcher(&[]);
        assert!(result.is_none(), "empty tools should yield None dispatcher");
    }

    #[test]
    fn build_tool_dispatcher_returns_some_for_tools() {
        use crate::routes::tools::{FunctionDef, Tool};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: None,
                parameters: json!({"type": "object"}),
            },
        }];
        let result = build_tool_dispatcher(&tools);
        assert!(
            result.is_some(),
            "non-empty tools should yield a dispatcher"
        );
    }
}

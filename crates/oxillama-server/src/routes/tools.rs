//! OpenAI-compatible tool/function calling types and GBNF grammar generation.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// A tool definition (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Type of tool — currently only "function" is supported.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition.
    pub function: FunctionDef,
}

/// Describes a callable function exposed as a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Function name.
    pub name: String,
    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the function parameters.
    pub parameters: serde_json::Value,
}

/// Tool choice policy.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"none"` — never call a tool.
    /// `"auto"` — model decides.
    /// `"required"` — must call a tool.
    Mode(String),
    /// Force a specific function:
    /// `{"type": "function", "function": {"name": "..."}}`
    Specific {
        #[serde(rename = "type")]
        tool_type: String,
        function: ToolChoiceFunction,
    },
}

/// Names a single function in a `ToolChoice::Specific`.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// A tool call emitted by the assistant in a non-streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function invocation details.
    pub function: FunctionCall,
}

/// Concrete function invocation inside a `ToolCall`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// Arguments serialized as a JSON string.
    pub arguments: String,
}

/// Incremental tool call data for SSE streaming.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    /// Index of the tool call in the array.
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Incremental function call data for SSE streaming.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// GBNF grammar generation
// ---------------------------------------------------------------------------

/// Generate a GBNF grammar that constrains the model to produce valid
/// JSON tool-call output matching the supplied tool definitions.
///
/// Returns an empty string when tool calling is disabled (`"none"` mode),
/// which signals the caller to skip grammar-constrained sampling.
pub fn tools_to_gbnf(tools: &[Tool], tool_choice: &Option<ToolChoice>) -> String {
    if tools.is_empty() {
        return String::new();
    }

    // Determine which functions are eligible.
    let (mode, forced_name) = match tool_choice {
        Some(ToolChoice::Mode(m)) => (m.as_str(), None),
        Some(ToolChoice::Specific { function, .. }) => ("specific", Some(function.name.as_str())),
        None => ("auto", None),
    };

    if mode == "none" {
        return String::new();
    }

    let eligible: Vec<&Tool> = match forced_name {
        Some(name) => tools.iter().filter(|t| t.function.name == name).collect(),
        None => tools.iter().collect(),
    };

    if eligible.is_empty() {
        return String::new();
    }

    let mut grammar = String::with_capacity(2048);

    // Root rule depends on mode.
    match mode {
        "auto" => {
            // Model may produce either plain text OR a tool call JSON.
            grammar.push_str("root ::= tool-call | free-text\n");
            grammar.push_str("free-text ::= [^{] [^\\x00]*\n");
        }
        _ => {
            // "required" or "specific" — must produce a tool call.
            grammar.push_str("root ::= tool-call\n");
        }
    }

    // tool-call envelope
    grammar.push_str(
        "tool-call ::= \"{\" ws \
         \"\\\"name\\\"\" ws \":\" ws function-name ws \
         \",\" ws \
         \"\\\"arguments\\\"\" ws \":\" ws arguments ws \
         \"}\"\n",
    );

    // function-name alternatives
    grammar.push_str("function-name ::= ");
    for (i, tool) in eligible.iter().enumerate() {
        if i > 0 {
            grammar.push_str(" | ");
        }
        grammar.push_str(&format!("\"\\\"{}\\\"\"", tool.function.name));
    }
    grammar.push('\n');

    // Per-function argument rules.
    // We generate a dedicated rule for each function's parameter schema so
    // the model is constrained to the correct argument shape.
    for tool in &eligible {
        let rule_name = format!("args-{}", sanitize_rule_name(&tool.function.name));
        generate_object_rule(&mut grammar, &rule_name, &tool.function.parameters);
    }

    // The top-level `arguments` rule dispatches to the correct per-function
    // args rule based on the preceding function-name.  GBNF does not support
    // context-dependent dispatch, so for multiple tools we fall back to a
    // generic JSON object rule.  For a single tool we can use its specific
    // rule directly.
    if eligible.len() == 1 {
        let rule_name = format!("args-{}", sanitize_rule_name(&eligible[0].function.name));
        grammar.push_str(&format!("arguments ::= {rule_name}\n"));
    } else {
        grammar.push_str("arguments ::= json-object\n");
        append_generic_json_rules(&mut grammar);
    }

    // Whitespace
    grammar.push_str("ws ::= [ \\t\\n]*\n");

    grammar
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sanitize a function name into a valid GBNF rule name (letters, digits,
/// hyphens only).
fn sanitize_rule_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Generate a GBNF object rule from a JSON Schema `parameters` value.
///
/// Handles the common case of `{"type": "object", "properties": {...}}`.
/// Falls back to a generic JSON object rule for anything exotic.
fn generate_object_rule(grammar: &mut String, rule_name: &str, schema: &serde_json::Value) {
    let props = match schema.get("properties").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => {
            // Fallback: accept any JSON object.
            grammar.push_str(&format!("{rule_name} ::= json-object\n"));
            append_generic_json_rules(grammar);
            return;
        }
    };

    if props.is_empty() {
        grammar.push_str(&format!("{rule_name} ::= \"{{}}\" | \"{{\" ws \"}}\"\n"));
        return;
    }

    let required: Vec<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    // Build a fixed-order key sequence.
    // Required properties are mandatory; optional properties are wrapped in
    // `( ... )?` groups. For simplicity we emit them in iteration order with
    // commas between required props and optional trailing groups.
    let mut parts: Vec<String> = Vec::new();
    let mut prop_rules: Vec<String> = Vec::new();

    for (key, prop_schema) in props {
        let val_rule = format!("{rule_name}-{}", sanitize_rule_name(key));
        generate_value_rule(grammar, &val_rule, prop_schema);

        let kv = format!("\"\\\"{}\\\"\" ws \":\" ws {}", key, val_rule);

        let is_required = required.contains(&key.as_str());

        if is_required {
            parts.push(kv);
        } else {
            // Optional: wrap in a group that may not appear.
            parts.push(format!("({kv})?"));
        }

        prop_rules.push(key.clone());
    }

    // Emit the object rule. We join parts with comma-separated whitespace.
    // For a fully correct grammar we would need to handle optional comma
    // suppression, but the simple approach works for most schemas: we
    // always emit commas between parts and rely on the model to produce
    // well-formed JSON. This is the same strategy used by llama.cpp.
    grammar.push_str(&format!("{rule_name} ::= \"{{\" ws "));
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            grammar.push_str(" \",\" ws ");
        }
        grammar.push_str(part);
    }
    grammar.push_str(" ws \"}\"\n");
}

/// Generate a GBNF rule for a single JSON Schema value.
fn generate_value_rule(grammar: &mut String, rule_name: &str, schema: &serde_json::Value) {
    let ty = schema
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("string");

    match ty {
        "string" => {
            if let Some(enum_vals) = schema.get("enum").and_then(|e| e.as_array()) {
                // Enum of string literals.
                let alts: Vec<String> = enum_vals
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| format!("\"\\\"{}\\\"\"", s))
                    .collect();
                if alts.is_empty() {
                    grammar.push_str(&format!("{rule_name} ::= json-string\n"));
                } else {
                    grammar.push_str(&format!("{rule_name} ::= {}\n", alts.join(" | ")));
                }
            } else {
                grammar.push_str(&format!("{rule_name} ::= json-string\n"));
            }
        }
        "integer" => {
            grammar.push_str(&format!("{rule_name} ::= \"-\"? [0-9]+\n"));
        }
        "number" => {
            grammar.push_str(&format!("{rule_name} ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n"));
        }
        "boolean" => {
            grammar.push_str(&format!("{rule_name} ::= \"true\" | \"false\"\n"));
        }
        "array" => {
            let item_rule = format!("{rule_name}-item");
            if let Some(items_schema) = schema.get("items") {
                generate_value_rule(grammar, &item_rule, items_schema);
            } else {
                grammar.push_str(&format!("{item_rule} ::= json-value\n"));
                append_generic_json_rules(grammar);
            }
            grammar.push_str(&format!(
                "{rule_name} ::= \"[\" ws ({item_rule} (\",\" ws {item_rule})*)? ws \"]\"\n"
            ));
        }
        "object" => {
            generate_object_rule(grammar, rule_name, schema);
        }
        _ => {
            // Unknown type — accept any JSON value.
            grammar.push_str(&format!("{rule_name} ::= json-value\n"));
            append_generic_json_rules(grammar);
        }
    }
}

/// Append the generic JSON value / object / array / string / number primitive
/// rules if they have not been appended yet.
///
/// We use a simple marker comment to avoid duplicating the block.
fn append_generic_json_rules(grammar: &mut String) {
    const MARKER: &str = "json-value ::=";
    if grammar.contains(MARKER) {
        return;
    }
    grammar.push_str(concat!(
        "json-value ::= json-string | json-number | json-object | json-array | \"true\" | \"false\" | \"null\"\n",
        "json-string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" [\"\\\\/bfnrt] | \"\\\\u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* \"\\\"\"\n",
        "json-number ::= \"-\"? [0-9]+ (\".\" [0-9]+)? ([eE] [\"+\\-\"]? [0-9]+)?\n",
        "json-object ::= \"{\" ws (json-string ws \":\" ws json-value (\",\" ws json-string ws \":\" ws json-value)*)? ws \"}\"\n",
        "json-array ::= \"[\" ws (json-value (\",\" ws json-value)*)? ws \"]\"\n",
    ));
}

/// Parse raw model output that was grammar-constrained to a tool call and
/// wrap it in a `ToolCall` structure.
///
/// Expects JSON of the form `{"name": "...", "arguments": {...}}`.
/// Returns `None` if parsing fails so the caller can fall back to a plain
/// text response.
pub fn parse_tool_call_output(raw: &str, call_id: &str) -> Option<ToolCall> {
    let parsed: serde_json::Value = serde_json::from_str(raw.trim()).ok()?;
    let name = parsed.get("name")?.as_str()?;
    let arguments = parsed.get("arguments")?;

    Some(ToolCall {
        id: call_id.to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    })
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ----- Serialization / Deserialization -----

    #[test]
    fn test_tool_deserializes() {
        let json_str = r#"{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }"#;
        let tool: Tool = serde_json::from_str(json_str).expect("deserialize Tool");
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "get_weather");
        assert_eq!(
            tool.function.description.as_deref(),
            Some("Get the weather")
        );
    }

    #[test]
    fn test_tool_choice_mode_none() {
        let tc: ToolChoice = serde_json::from_str(r#""none""#).expect("deserialize");
        match tc {
            ToolChoice::Mode(m) => assert_eq!(m, "none"),
            _ => panic!("expected Mode"),
        }
    }

    #[test]
    fn test_tool_choice_mode_auto() {
        let tc: ToolChoice = serde_json::from_str(r#""auto""#).expect("deserialize");
        match tc {
            ToolChoice::Mode(m) => assert_eq!(m, "auto"),
            _ => panic!("expected Mode"),
        }
    }

    #[test]
    fn test_tool_choice_mode_required() {
        let tc: ToolChoice = serde_json::from_str(r#""required""#).expect("deserialize");
        match tc {
            ToolChoice::Mode(m) => assert_eq!(m, "required"),
            _ => panic!("expected Mode"),
        }
    }

    #[test]
    fn test_tool_choice_specific() {
        let json_str = r#"{"type": "function", "function": {"name": "do_thing"}}"#;
        let tc: ToolChoice = serde_json::from_str(json_str).expect("deserialize");
        match tc {
            ToolChoice::Specific {
                tool_type,
                function,
            } => {
                assert_eq!(tool_type, "function");
                assert_eq!(function.name, "do_thing");
            }
            _ => panic!("expected Specific"),
        }
    }

    #[test]
    fn test_tool_call_serializes() {
        let tc = ToolCall {
            id: "call_abc123".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"Tokyo"}"#.to_string(),
            },
        };
        let json = serde_json::to_value(&tc).expect("serialize");
        assert_eq!(json["type"], "function");
        assert_eq!(json["id"], "call_abc123");
        assert_eq!(json["function"]["name"], "get_weather");
    }

    // ----- GBNF generation -----

    #[test]
    fn test_tools_to_gbnf_empty_tools() {
        let gbnf = tools_to_gbnf(&[], &None);
        assert!(gbnf.is_empty(), "empty tools → empty grammar");
    }

    #[test]
    fn test_tools_to_gbnf_none_mode() {
        let tools = vec![make_weather_tool()];
        let choice = Some(ToolChoice::Mode("none".to_string()));
        let gbnf = tools_to_gbnf(&tools, &choice);
        assert!(gbnf.is_empty(), "none mode → empty grammar");
    }

    #[test]
    fn test_tools_to_gbnf_required_mode() {
        let tools = vec![make_weather_tool()];
        let choice = Some(ToolChoice::Mode("required".to_string()));
        let gbnf = tools_to_gbnf(&tools, &choice);
        assert!(
            gbnf.contains("root ::= tool-call"),
            "required → root is tool-call"
        );
        assert!(!gbnf.contains("free-text"), "required → no free-text");
        assert!(
            gbnf.contains("get_weather"),
            "should reference function name"
        );
    }

    #[test]
    fn test_tools_to_gbnf_auto_mode() {
        let tools = vec![make_weather_tool()];
        let gbnf = tools_to_gbnf(&tools, &None);
        assert!(
            gbnf.contains("root ::= tool-call | free-text"),
            "auto → allows free-text: {gbnf}"
        );
    }

    #[test]
    fn test_tools_to_gbnf_specific_function() {
        let tools = vec![make_weather_tool(), make_search_tool()];
        let choice = Some(ToolChoice::Specific {
            tool_type: "function".to_string(),
            function: ToolChoiceFunction {
                name: "search".to_string(),
            },
        });
        let gbnf = tools_to_gbnf(&tools, &choice);
        assert!(gbnf.contains("search"), "should reference forced function");
        // Should NOT reference the non-forced function name in function-name
        // alternatives (it may still appear in other comments/rules).
        assert!(
            gbnf.contains("function-name ::= \"\\\"search\\\"\""),
            "only the forced function should appear: {gbnf}"
        );
    }

    #[test]
    fn test_tools_to_gbnf_multiple_functions() {
        let tools = vec![make_weather_tool(), make_search_tool()];
        let choice = Some(ToolChoice::Mode("required".to_string()));
        let gbnf = tools_to_gbnf(&tools, &choice);
        assert!(gbnf.contains("get_weather"), "should list get_weather");
        assert!(gbnf.contains("search"), "should list search");
        assert!(
            gbnf.contains("json-object"),
            "multi-func uses generic json-object"
        );
    }

    #[test]
    fn test_tools_to_gbnf_has_ws_rule() {
        let tools = vec![make_weather_tool()];
        let gbnf = tools_to_gbnf(&tools, &None);
        assert!(gbnf.contains("ws ::="), "grammar must define ws rule");
    }

    #[test]
    fn test_tools_to_gbnf_single_function_uses_specific_args() {
        let tools = vec![make_weather_tool()];
        let gbnf = tools_to_gbnf(&tools, &Some(ToolChoice::Mode("required".to_string())));
        assert!(
            gbnf.contains("args-get-weather"),
            "single function should have specific args rule: {gbnf}"
        );
        assert!(
            gbnf.contains("arguments ::= args-get-weather"),
            "arguments should dispatch to specific rule: {gbnf}"
        );
    }

    // ----- parse_tool_call_output -----

    #[test]
    fn test_parse_tool_call_output_valid() {
        let raw = r#"{"name": "get_weather", "arguments": {"location": "Tokyo"}}"#;
        let tc = parse_tool_call_output(raw, "call_001").expect("should parse");
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.id, "call_001");
        assert!(tc.function.arguments.contains("Tokyo"));
    }

    #[test]
    fn test_parse_tool_call_output_with_whitespace() {
        let raw = r#"  {"name": "search", "arguments": {"q": "rust"}}  "#;
        let tc = parse_tool_call_output(raw, "call_002").expect("should parse");
        assert_eq!(tc.function.name, "search");
    }

    #[test]
    fn test_parse_tool_call_output_invalid_json() {
        let raw = "not json at all";
        assert!(parse_tool_call_output(raw, "x").is_none());
    }

    #[test]
    fn test_parse_tool_call_output_missing_name() {
        let raw = r#"{"arguments": {"a": 1}}"#;
        assert!(parse_tool_call_output(raw, "x").is_none());
    }

    #[test]
    fn test_parse_tool_call_output_missing_arguments() {
        let raw = r#"{"name": "foo"}"#;
        assert!(parse_tool_call_output(raw, "x").is_none());
    }

    // ----- Value rule generation -----

    #[test]
    fn test_enum_string_generates_alternatives() {
        let schema = json!({
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
        });
        let mut grammar = String::new();
        generate_value_rule(&mut grammar, "unit", &schema);
        assert!(
            grammar.contains("celsius"),
            "enum values in grammar: {grammar}"
        );
        assert!(
            grammar.contains("fahrenheit"),
            "enum values in grammar: {grammar}"
        );
    }

    #[test]
    fn test_boolean_value_rule() {
        let schema = json!({"type": "boolean"});
        let mut grammar = String::new();
        generate_value_rule(&mut grammar, "flag", &schema);
        assert!(grammar.contains("\"true\""), "boolean rule: {grammar}");
        assert!(grammar.contains("\"false\""), "boolean rule: {grammar}");
    }

    #[test]
    fn test_integer_value_rule() {
        let schema = json!({"type": "integer"});
        let mut grammar = String::new();
        generate_value_rule(&mut grammar, "count", &schema);
        assert!(grammar.contains("[0-9]+"), "integer rule: {grammar}");
    }

    #[test]
    fn test_array_value_rule() {
        let schema = json!({
            "type": "array",
            "items": {"type": "string"}
        });
        let mut grammar = String::new();
        generate_value_rule(&mut grammar, "tags", &schema);
        assert!(grammar.contains("tags-item"), "array items rule: {grammar}");
        assert!(grammar.contains("["), "array brackets: {grammar}");
    }

    // ----- ToolCallDelta serialization -----

    #[test]
    fn test_tool_call_delta_serializes_with_skip() {
        let delta = ToolCallDelta {
            index: 0,
            id: Some("call_123".to_string()),
            tool_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some("get_weather".to_string()),
                arguments: None,
            }),
        };
        let json = serde_json::to_value(&delta).expect("serialize");
        assert_eq!(json["index"], 0);
        assert_eq!(json["id"], "call_123");
        assert_eq!(json["type"], "function");
        assert!(json.get("function").is_some());
        // arguments is None → should be absent due to skip_serializing_if
        assert!(
            json["function"].get("arguments").is_none(),
            "None arguments should be skipped: {json}"
        );
    }

    #[test]
    fn test_tool_call_delta_minimal() {
        let delta = ToolCallDelta {
            index: 1,
            id: None,
            tool_type: None,
            function: Some(FunctionCallDelta {
                name: None,
                arguments: Some("{\"loc".to_string()),
            }),
        };
        let json = serde_json::to_value(&delta).expect("serialize");
        assert_eq!(json["index"], 1);
        assert!(json.get("id").is_none(), "None id should be skipped");
        assert!(json.get("type").is_none(), "None type should be skipped");
        assert_eq!(json["function"]["arguments"], "{\"loc");
    }

    // ----- Helper factories -----

    fn make_weather_tool() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: Some("Get the weather for a location".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }),
            },
        }
    }

    fn make_search_tool() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "search".to_string(),
                description: Some("Search the web".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }),
            },
        }
    }
}

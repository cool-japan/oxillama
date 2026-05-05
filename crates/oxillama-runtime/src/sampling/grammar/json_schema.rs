//! JSON Schema (subset) → GBNF compiler.
//!
//! Converts a JSON Schema document into a GBNF grammar string that can be
//! parsed by [`Grammar::parse`] and used for constrained generation.
//!
//! # Supported schema keywords
//!
//! | Keyword | Notes |
//! |---------|-------|
//! | `type` | `"string"`, `"number"`, `"integer"`, `"boolean"`, `"null"`, `"object"`, `"array"` |
//! | `properties` + `required` | For `object` type; unknown properties are ignored |
//! | `enum` | String values only |
//! | `items` | Single sub-schema for `array` type |
//! | `minimum`, `maximum` | Numeric range (informational — produces digit pattern) |
//! | `minLength`, `maxLength` | String length constraints (informational) |
//! | `pattern` | Only literal strings (no regex metacharacters) |
//! | Nested objects / arrays | Fully supported via recursive rule generation |
//!
//! # Generated GBNF dialect
//!
//! Rules are emitted in the same format the existing GBNF parser understands:
//! - `rule-name ::= body`
//! - Sequences: `item1 item2`
//! - Alternations: `body1 | body2`
//! - Repetitions: `body*`, `body+`, `body?`
//! - Quoted literals: `"text"`
//! - Character classes: `[a-z]`, `[0-9]`, etc.

use std::collections::HashSet;

use serde_json::Value;

use super::error::{GrammarError, GrammarResult};
use super::parser::Grammar;

/// Compiles a JSON Schema (subset) to a GBNF [`Grammar`].
pub struct JsonSchemaCompiler;

impl JsonSchemaCompiler {
    /// Compile a JSON Schema JSON string into a GBNF [`Grammar`].
    ///
    /// The top-level schema becomes the `root` rule. Nested schemas are
    /// assigned generated rule names of the form `rule-N`.
    ///
    /// # Errors
    ///
    /// Returns [`GrammarError::ParseError`] when:
    /// - The input is not valid JSON.
    /// - The schema contains an unsupported `type` value.
    /// - The schema is structurally invalid (e.g. `properties` is not an object).
    ///
    /// Returns [`GrammarError::UnknownRule`] when a `$ref` is encountered
    /// (not yet supported — use inline schemas instead).
    pub fn compile(schema_json: &str) -> GrammarResult<Grammar> {
        let schema: Value =
            serde_json::from_str(schema_json).map_err(|e| GrammarError::ParseError {
                pos: 0,
                msg: format!("invalid JSON in schema: {e}"),
            })?;

        let mut compiler = SchemaCompiler::new();
        compiler.compile_root(&schema)?;
        let gbnf = compiler.build_gbnf();
        Grammar::parse(&gbnf)
    }
}

// ─── Internal compiler state ──────────────────────────────────────────────────

/// Counter used to generate unique rule names.
struct SchemaCompiler {
    /// All generated rules: rule_name → GBNF body string.
    rules: Vec<(String, String)>,
    /// Rule-name counter for auto-generated names.
    counter: usize,
}

impl SchemaCompiler {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            counter: 0,
        }
    }

    /// Reserve a new unique rule name.
    fn next_rule_name(&mut self) -> String {
        let name = format!("rule-{}", self.counter);
        self.counter += 1;
        name
    }

    /// Add a rule and return its name.
    fn add_rule(&mut self, name: String, body: String) -> String {
        self.rules.push((name.clone(), body));
        name
    }

    /// Compile the root schema node, adding a "root" rule.
    fn compile_root(&mut self, schema: &Value) -> GrammarResult<()> {
        let body = self.compile_schema(schema)?;
        self.rules.insert(0, ("root".to_string(), body));
        Ok(())
    }

    /// Build the final GBNF string from all collected rules.
    fn build_gbnf(&self) -> String {
        let mut out = String::new();
        for (name, body) in &self.rules {
            out.push_str(&format!("{name} ::= {body}\n"));
        }
        out
    }

    /// Recursively compile a schema node, returning a GBNF body expression.
    ///
    /// Leaf types (string, number, etc.) return inline expressions.
    /// Complex types (object, array) generate helper rules and return a
    /// reference to those rules.
    fn compile_schema(&mut self, schema: &Value) -> GrammarResult<String> {
        let obj = match schema {
            Value::Object(o) => o,
            Value::Bool(true) => {
                // A bare `true` schema allows any JSON value.
                return Ok(self.any_json_expr());
            }
            Value::Bool(false) => {
                // A bare `false` schema allows nothing — produce an unmatchable rule.
                return Ok("\"__never__\"".to_string());
            }
            other => {
                return Err(GrammarError::ParseError {
                    pos: 0,
                    msg: format!("schema must be a JSON object, got {other}"),
                });
            }
        };

        // Reject unsupported $ref to avoid silent mis-compilation.
        if obj.contains_key("$ref") {
            return Err(GrammarError::UnknownRule {
                rule: "$ref (JSON Schema $ref is not supported — use inline schemas)".to_string(),
            });
        }

        // Handle `enum` first — it overrides `type`.
        if let Some(enum_val) = obj.get("enum") {
            return self.compile_enum(enum_val);
        }

        // Determine the type.
        let type_str = match obj.get("type") {
            Some(Value::String(t)) => t.as_str(),
            Some(Value::Array(_)) => {
                // Multi-type: not deeply supported; fall back to any-JSON.
                return Ok(self.any_json_expr());
            }
            Some(other) => {
                return Err(GrammarError::ParseError {
                    pos: 0,
                    msg: format!("`type` must be a string, got {other}"),
                });
            }
            None => {
                // No type specified — check for object structure hints.
                if obj.contains_key("properties") {
                    "object"
                } else if obj.contains_key("items") {
                    "array"
                } else {
                    // Truly unconstrained: allow any JSON value.
                    return Ok(self.any_json_expr());
                }
            }
        };

        match type_str {
            "string" => self.compile_string_type(obj),
            "number" => self.compile_number_type(obj),
            "integer" => self.compile_integer_type(obj),
            "boolean" => Ok(self.boolean_expr()),
            "null" => Ok(self.null_expr()),
            "object" => self.compile_object_type(obj),
            "array" => self.compile_array_type(obj),
            unknown => Err(GrammarError::ParseError {
                pos: 0,
                msg: format!("unsupported JSON Schema type: `{unknown}`"),
            }),
        }
    }

    // ── Primitive type generators ─────────────────────────────────────────────

    /// Inline GBNF expression that matches any JSON boolean.
    fn boolean_expr(&self) -> String {
        r#""true" | "false""#.to_string()
    }

    /// Inline GBNF expression that matches JSON null.
    fn null_expr(&self) -> String {
        r#""null""#.to_string()
    }

    /// Inline GBNF expression for a JSON string (full Unicode safe subset).
    /// Produces: `"\"" string-char* "\""`
    ///
    /// We emit a helper rule so the body stays on one line.
    fn compile_string_type(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> GrammarResult<String> {
        // Handle `pattern` — only literal strings allowed (no metacharacters).
        if let Some(Value::String(pattern)) = obj.get("pattern") {
            // Reject patterns that look like they contain regex metacharacters.
            let metacharacters = [
                '.', '*', '+', '?', '(', ')', '[', ']', '{', '}', '^', '$', '|', '\\',
            ];
            if pattern.chars().any(|c| metacharacters.contains(&c)) {
                return Err(GrammarError::ParseError {
                    pos: 0,
                    msg: "schema `pattern` with regex metacharacters is not supported; only literal strings are allowed".to_string(),
                });
            }
            // For a literal pattern, the string must equal the pattern exactly.
            let escaped = escape_gbnf_literal(pattern);
            return Ok(format!(r#""{escaped}""#));
        }

        // Ensure the string-char helper rule exists (add once).
        self.ensure_string_char_rule();
        Ok(r#""\"" string-char* "\"" "#.trim().to_string())
    }

    /// Inline GBNF expression for a JSON number (integer or float).
    fn compile_number_type(
        &mut self,
        _obj: &serde_json::Map<String, Value>,
    ) -> GrammarResult<String> {
        // Produce a rule for JSON numbers: optional minus, digits, optional fraction.
        self.ensure_number_rule();
        Ok("json-number".to_string())
    }

    /// Inline GBNF expression for a JSON integer.
    fn compile_integer_type(
        &mut self,
        _obj: &serde_json::Map<String, Value>,
    ) -> GrammarResult<String> {
        self.ensure_integer_rule();
        Ok("json-integer".to_string())
    }

    // ── enum ──────────────────────────────────────────────────────────────────

    /// Compile an `enum` keyword.  Only string values are supported.
    fn compile_enum(&mut self, enum_val: &Value) -> GrammarResult<String> {
        let variants = match enum_val {
            Value::Array(arr) => arr,
            other => {
                return Err(GrammarError::ParseError {
                    pos: 0,
                    msg: format!("`enum` must be an array, got {other}"),
                });
            }
        };

        if variants.is_empty() {
            return Err(GrammarError::ParseError {
                pos: 0,
                msg: "`enum` array must not be empty".to_string(),
            });
        }

        let mut alternatives: Vec<String> = Vec::with_capacity(variants.len());
        for v in variants {
            match v {
                Value::String(s) => {
                    let escaped = escape_gbnf_literal(s);
                    // Generate: `"\"" "<escaped>" "\""` — a JSON-quoted string literal.
                    let alt = format!("\"\\\"\" \"{escaped}\" \"\\\"\"");
                    alternatives.push(alt);
                }
                Value::Null => alternatives.push("\"null\"".to_string()),
                Value::Bool(b) => alternatives.push(format!("\"{b}\"")),
                Value::Number(n) => alternatives.push(format!("\"{n}\"")),
                other => {
                    return Err(GrammarError::ParseError {
                        pos: 0,
                        msg: format!("unsupported enum value type: {other}"),
                    });
                }
            }
        }

        Ok(alternatives.join(" | "))
    }

    // ── object ────────────────────────────────────────────────────────────────

    /// Compile `{"type": "object", "properties": {...}, "required": [...]}`.
    fn compile_object_type(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> GrammarResult<String> {
        let ws = self.ensure_ws_rule();

        let props = match obj.get("properties") {
            Some(Value::Object(p)) => p,
            Some(other) => {
                return Err(GrammarError::ParseError {
                    pos: 0,
                    msg: format!("`properties` must be an object, got {other}"),
                });
            }
            None => {
                // No properties defined — match any JSON object.
                // Build the object body via concatenation to avoid format! brace-escape issues.
                // Produces: "{" ws json-pair (ws "," ws json-pair)* ws "}" | "{" ws "}"
                self.ensure_string_char_rule();
                let any_val = self.any_json_expr();
                // json-pair = "\"" string-char* "\"" ws ":" ws json-value
                if !self.has_rule("json-pair") {
                    let pair_body = [
                        "\"\\\"\"",
                        " string-char* ",
                        "\"\\\"\"",
                        " ",
                        &ws,
                        " \":\" ",
                        &ws,
                        " ",
                        &any_val,
                    ]
                    .concat();
                    self.rules.push(("json-pair".to_string(), pair_body));
                }
                // "{" ws "}" | "{" ws json-pair (ws "," ws json-pair)* ws "}"
                let empty_obj = ["\"{\"", " ", &ws, " ", "\"}\""].concat();
                let with_members = [
                    "\"{\"",
                    " ",
                    &ws,
                    " json-pair (",
                    &ws,
                    " \",\" ",
                    &ws,
                    " json-pair)* ",
                    &ws,
                    " \"}\"",
                ]
                .concat();
                return Ok([empty_obj, " | ".to_string(), with_members].concat());
            }
        };

        let required_set: HashSet<String> = match obj.get("required") {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect(),
            Some(other) => {
                return Err(GrammarError::ParseError {
                    pos: 0,
                    msg: format!("`required` must be an array, got {other}"),
                });
            }
            None => HashSet::new(),
        };

        // Separate required and optional properties.
        let mut required_props: Vec<(&String, &Value)> = Vec::new();
        let mut optional_props: Vec<(&String, &Value)> = Vec::new();

        for (key, val_schema) in props {
            if required_set.contains(key) {
                required_props.push((key, val_schema));
            } else {
                optional_props.push((key, val_schema));
            }
        }

        // For deterministic ordering, sort each group by key name.
        required_props.sort_by_key(|(k, _)| k.as_str());
        optional_props.sort_by_key(|(k, _)| k.as_str());

        // Compile each property's value schema.
        let mut all_parts: Vec<String> = Vec::new();

        for (key, val_schema) in &required_props {
            let val_expr = self.compile_schema(val_schema)?;
            let val_rule = if is_inline_expr(&val_expr) {
                val_expr.clone()
            } else {
                // Wrap complex expression in a helper rule.
                let rule_name = self.next_rule_name();
                self.add_rule(rule_name.clone(), val_expr);
                rule_name
            };
            let key_escaped = escape_gbnf_literal(key);
            // Produce: `"\"" "keyname" "\"" ws ":" ws val_rule`
            let prop_expr =
                format!("\"\\\"\" \"{key_escaped}\" \"\\\"\" {ws} \":\" {ws} {val_rule}");
            all_parts.push(prop_expr);
        }

        for (key, val_schema) in &optional_props {
            let val_expr = self.compile_schema(val_schema)?;
            let val_rule = if is_inline_expr(&val_expr) {
                val_expr.clone()
            } else {
                let rule_name = self.next_rule_name();
                self.add_rule(rule_name.clone(), val_expr);
                rule_name
            };
            let key_escaped = escape_gbnf_literal(key);
            let prop_body =
                format!("\"\\\"\" \"{key_escaped}\" \"\\\"\" {ws} \":\" {ws} {val_rule}");
            // Optional: wrap in (...)?
            let rule_name = self.next_rule_name();
            self.add_rule(rule_name.clone(), prop_body);
            all_parts.push(format!("({rule_name})?"));
        }

        // Build object body: `{` ws members ws `}`
        // Members are joined by `,` ws.
        // We use string concatenation rather than format! to avoid the Rust
        // formatter treating `{` as a format-string interpolation.
        let open_brace = "\"{\"";
        let close_brace = "\"}\"";
        let comma = "\",\"";

        let body = if all_parts.is_empty() {
            // Empty object: `"{"` ws `"}"`
            [open_brace, " ", &ws, " ", close_brace].concat()
        } else {
            let sep = [" ", &ws, " ", comma, " ", &ws, " "].concat();
            let joined = all_parts.join(&sep);
            [
                open_brace,
                " ",
                &ws,
                " ",
                &joined,
                " ",
                &ws,
                " ",
                close_brace,
            ]
            .concat()
        };

        Ok(body)
    }

    // ── array ─────────────────────────────────────────────────────────────────

    /// Compile `{"type": "array", "items": {...}}`.
    fn compile_array_type(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> GrammarResult<String> {
        let ws = self.ensure_ws_rule();

        let items_expr = match obj.get("items") {
            Some(item_schema) => self.compile_schema(item_schema)?,
            None => self.any_json_expr(),
        };

        // If the items expression is complex, give it its own rule.
        let items_rule = if is_inline_expr(&items_expr) {
            items_expr
        } else {
            let rule_name = self.next_rule_name();
            self.add_rule(rule_name.clone(), items_expr);
            rule_name
        };

        // Array: `[` ws (item (ws `,` ws item)*)? ws `]`
        Ok(format!(
            r#""[" {ws} ({items_rule} ({ws} "," {ws} {items_rule})*)? {ws} "]""#
        ))
    }

    // ── Helper rule management ────────────────────────────────────────────────

    /// Ensure the `ws` (whitespace) helper rule exists, returning its name.
    fn ensure_ws_rule(&mut self) -> String {
        if !self.has_rule("ws") {
            self.rules
                .push(("ws".to_string(), r#"[ \t\n\r]*"#.to_string()));
        }
        "ws".to_string()
    }

    /// Ensure the `string-char` helper rule exists.
    fn ensure_string_char_rule(&mut self) {
        if !self.has_rule("string-char") {
            // Allow any byte except the control bytes and the double-quote/backslash.
            // This is a conservative safe subset: printable ASCII minus `"` and `\`.
            self.rules
                .push(("string-char".to_string(), r#"[^\x00-\x1f"\\]"#.to_string()));
        }
    }

    /// Ensure the `json-number` helper rule exists.
    fn ensure_number_rule(&mut self) {
        if !self.has_rule("json-number") {
            // Optional minus, one or more digits, optional decimal fraction.
            self.rules.push((
                "json-number".to_string(),
                r#""-"? [0-9]+ ("." [0-9]+)?"#.to_string(),
            ));
        }
    }

    /// Ensure the `json-integer` helper rule exists.
    fn ensure_integer_rule(&mut self) {
        if !self.has_rule("json-integer") {
            self.rules
                .push(("json-integer".to_string(), r#""-"? [0-9]+"#.to_string()));
        }
    }

    /// Returns true when a rule with the given name has already been added.
    fn has_rule(&self, name: &str) -> bool {
        self.rules.iter().any(|(n, _)| n == name)
    }

    /// Return an inline GBNF expression that matches any simple JSON value.
    ///
    /// This is a simplified "any value" pattern used when the schema does not
    /// constrain the type.  It covers the common JSON primitives.
    fn any_json_expr(&mut self) -> String {
        self.ensure_string_char_rule();
        self.ensure_number_rule();
        // Return a reference to a helper rule that covers all primitives.
        if !self.has_rule("json-value") {
            let num = "json-number";
            self.rules.push((
                "json-value".to_string(),
                format!(r#""\"" string-char* "\"" | {num} | "true" | "false" | "null""#),
            ));
        }
        "json-value".to_string()
    }
}

// ─── Utility functions ────────────────────────────────────────────────────────

/// Escape a plain string so it can appear safely inside GBNF double-quoted literals.
///
/// Only characters that have special meaning inside GBNF string literals need
/// escaping: `"` → `\"`, `\` → `\\`, and the common ASCII control codes.
fn escape_gbnf_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Heuristic: decide whether a compiled expression should be inlined directly
/// in a parent rule body, or whether it needs its own named rule.
///
/// Returns `true` for short, single-token expressions like `"true"`, rule
/// references without spaces, or character classes.  Returns `false` for
/// anything containing `::=` (already a rule reference placeholder) or
/// multi-part expressions that would make the parent unreadable.
fn is_inline_expr(expr: &str) -> bool {
    // Rule references and simple literals are short.
    expr.len() < 64 && !expr.contains('\n')
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compile and check that it succeeds, then return the grammar.
    fn compile_ok(schema: &str) -> Grammar {
        JsonSchemaCompiler::compile(schema)
            .unwrap_or_else(|e| panic!("compile_ok: compilation failed: {e}"))
    }

    // ── Basic type schemas ────────────────────────────────────────────────────

    #[test]
    fn compile_simple_object() {
        let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
        let g = compile_ok(schema);
        assert!(!g.rules.is_empty(), "grammar must have at least one rule");
        assert!(g.rules.contains_key("root"), "root rule must be present");
    }

    #[test]
    fn compile_enum_string() {
        let schema = r#"{"type": "string", "enum": ["yes", "no", "maybe"]}"#;
        let g = compile_ok(schema);
        // The root rule should encode three alternatives.
        let root_body = g.rules.get("root").expect("root rule");
        // Verify each value appears in the GBNF source somewhere.
        assert!(g.source.contains("yes"), "root body should reference 'yes'");
        assert!(g.source.contains("no"), "root body should reference 'no'");
        assert!(
            g.source.contains("maybe"),
            "root body should reference 'maybe'"
        );
        // root rule must exist and not be trivially empty.
        assert!(!format!("{root_body:?}").is_empty());
    }

    #[test]
    fn compile_required_fields() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"}
            },
            "required": ["id", "name"]
        }"#;
        let g = compile_ok(schema);
        // Both required fields should appear in the generated GBNF.
        assert!(
            g.source.contains("id"),
            "required field 'id' must appear in grammar"
        );
        assert!(
            g.source.contains("name"),
            "required field 'name' must appear in grammar"
        );
    }

    #[test]
    fn compile_array_with_items() {
        let schema = r#"{"type": "array", "items": {"type": "number"}}"#;
        let g = compile_ok(schema);
        assert!(g.rules.contains_key("root"));
        // The generated grammar should reference a number rule.
        assert!(
            g.source.contains("json-number"),
            "array-of-numbers grammar should reference json-number rule"
        );
    }

    #[test]
    fn compile_nested_object() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "integer"}
                    }
                }
            }
        }"#;
        // Should not return an error for nested objects.
        let g = compile_ok(schema);
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn compile_unknown_keyword_errors() {
        // $ref is explicitly unsupported — must return GrammarError.
        let schema = r##"{"$ref": "#/definitions/Foo"}"##;
        let result = JsonSchemaCompiler::compile(schema);
        assert!(result.is_err(), "unsupported $ref should return an error");
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn compile_boolean_type() {
        let schema = r#"{"type": "boolean"}"#;
        let g = compile_ok(schema);
        assert!(
            g.source.contains("true"),
            "boolean grammar should include 'true'"
        );
        assert!(
            g.source.contains("false"),
            "boolean grammar should include 'false'"
        );
    }

    #[test]
    fn compile_null_type() {
        let schema = r#"{"type": "null"}"#;
        let g = compile_ok(schema);
        assert!(
            g.source.contains("null"),
            "null grammar should include 'null'"
        );
    }

    #[test]
    fn compile_integer_type() {
        let schema = r#"{"type": "integer"}"#;
        let g = compile_ok(schema);
        assert!(
            g.source.contains("json-integer"),
            "integer grammar should reference json-integer rule"
        );
    }

    #[test]
    fn compile_number_type() {
        let schema = r#"{"type": "number"}"#;
        let g = compile_ok(schema);
        assert!(
            g.source.contains("json-number"),
            "number grammar should reference json-number rule"
        );
    }

    #[test]
    fn compile_string_type() {
        let schema = r#"{"type": "string"}"#;
        let g = compile_ok(schema);
        assert!(
            g.source.contains("string-char"),
            "string grammar should reference string-char rule"
        );
    }

    #[test]
    fn compile_invalid_json_errors() {
        let result = JsonSchemaCompiler::compile("this is not json {{");
        assert!(result.is_err(), "invalid JSON should return an error");
    }

    #[test]
    fn compile_unsupported_type_errors() {
        let schema = r#"{"type": "binary"}"#;
        let result = JsonSchemaCompiler::compile(schema);
        assert!(
            result.is_err(),
            "unsupported type 'binary' should return an error"
        );
    }

    #[test]
    fn compile_object_no_properties() {
        // An object with no `properties` should still compile.
        let schema = r#"{"type": "object"}"#;
        let g = compile_ok(schema);
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn compile_array_no_items() {
        // An array with no `items` should compile using any-json fallback.
        let schema = r#"{"type": "array"}"#;
        let g = compile_ok(schema);
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn compile_string_with_literal_pattern() {
        let schema = r#"{"type": "string", "pattern": "hello"}"#;
        let g = compile_ok(schema);
        assert!(
            g.source.contains("hello"),
            "literal pattern should appear in grammar"
        );
    }

    #[test]
    fn compile_string_with_regex_pattern_errors() {
        let schema = r#"{"type": "string", "pattern": "^[a-z]+"}"#;
        let result = JsonSchemaCompiler::compile(schema);
        assert!(
            result.is_err(),
            "regex metacharacters in pattern should return an error"
        );
    }

    #[test]
    fn compile_deeply_nested_array_of_objects() {
        let schema = r#"{
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "val": {"type": "number"}
                },
                "required": ["val"]
            }
        }"#;
        let g = compile_ok(schema);
        assert!(g.rules.contains_key("root"));
    }
}

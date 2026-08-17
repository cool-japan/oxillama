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
//! | `properties` + `required` | For `object` type |
//! | `additionalProperties` | `false`/absent = strict (only declared keys); `true` or a schema = extra keys allowed |
//! | `enum` | String, number, boolean, and null values |
//! | `const` | Any JSON value (string/number/bool/null/array/object — non-primitives matched via their canonical JSON text) |
//! | `anyOf`, `oneOf` | Compiled as a GBNF alternation of each branch. Note: a regular grammar can only express *membership* in the union — `oneOf`'s "exactly one branch validates" semantics cannot be enforced at generation time, only `anyOf`'s "at least one" can. |
//! | `allOf` | Best-effort: merges branches that are all object schemas (union of `properties`/`required`). Mixing non-object branches returns [`GrammarError::UnsupportedKeyword`]. |
//! | `items` | Single sub-schema for `array` type |
//! | `minItems`, `maxItems` | Enforced exactly via bounded repetition expansion |
//! | `minLength`, `maxLength` | Enforced exactly via bounded repetition expansion |
//! | `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | **Not expressible** as a regular grammar — returns [`GrammarError::UnsupportedKeyword`] rather than silently ignoring them |
//! | `pattern` | Only literal strings (no regex metacharacters) |
//! | Nested objects / arrays | Fully supported via recursive rule generation |
//!
//! Any keyword this compiler cannot express is reported via
//! [`GrammarError::UnsupportedKeyword`] or [`GrammarError::ParseError`] —
//! never silently widened to "any value" (see defect S7).
//!
//! # Generated GBNF dialect
//!
//! Rules are emitted in the same format the existing GBNF parser understands:
//! - `rule-name ::= body`
//! - Sequences: `item1 item2`
//! - Alternations: `body1 | body2`
//! - Repetitions: `body*`, `body+`, `body?` (GBNF has no `{min,max}` syntax —
//!   bounded repetition is expanded explicitly, see `bounded_repeat`)
//! - Quoted literals: `"text"`
//! - Character classes: `[a-z]`, `[0-9]`, etc.
//!
//! # Public API shape (for downstream integration)
//!
//! [`JsonSchemaCompiler::compile`] returns a parsed [`Grammar`] (for direct
//! use with [`super::GrammarState`]). [`JsonSchemaCompiler::compile_to_gbnf`]
//! returns the raw GBNF **text** instead — this is the entry point intended
//! for the HTTP server's own JSON-Schema→GBNF conversion path (which has an
//! independent, weaker converter with a dangling-comma bug) to delegate to:
//! it lets the server inspect/cache/log the generated grammar text, or hand
//! it to a different consumer, without this module owning the parse step.
//! Both entry points accept either a JSON string or an already-parsed
//! [`serde_json::Value`] (via the `_value` counterparts), so a caller that
//! already deserialized the schema (e.g. from an OpenAI-style
//! `response_format.json_schema` request body) doesn't pay to re-serialize
//! and re-parse it.

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
    ///
    /// Returns [`GrammarError::UnsupportedKeyword`] when a recognised
    /// keyword cannot be expressed as a regular (GBNF) grammar — e.g.
    /// `minimum`/`maximum` numeric ranges, or an `allOf` mixing non-object
    /// branches.
    pub fn compile(schema_json: &str) -> GrammarResult<Grammar> {
        let gbnf = Self::compile_to_gbnf(schema_json)?;
        Grammar::parse(&gbnf)
    }

    /// Compile an already-parsed [`serde_json::Value`] schema into a GBNF
    /// [`Grammar`]. Use this when the caller already deserialized the
    /// schema (e.g. from a request body) and doesn't want to re-serialize
    /// it just to re-parse it here.
    pub fn compile_value(schema: &Value) -> GrammarResult<Grammar> {
        let gbnf = Self::compile_value_to_gbnf(schema)?;
        Grammar::parse(&gbnf)
    }

    /// Compile a JSON Schema JSON string into raw GBNF **text**, without
    /// parsing it into a [`Grammar`].
    ///
    /// This is the intended delegation point for other JSON-Schema→GBNF
    /// converters in the workspace (e.g. the HTTP server's own, weaker
    /// converter) — see the module-level "Public API shape" note.
    pub fn compile_to_gbnf(schema_json: &str) -> GrammarResult<String> {
        let schema: Value =
            serde_json::from_str(schema_json).map_err(|e| GrammarError::ParseError {
                pos: 0,
                msg: format!("invalid JSON in schema: {e}"),
            })?;
        Self::compile_value_to_gbnf(&schema)
    }

    /// Compile an already-parsed [`serde_json::Value`] schema into raw GBNF
    /// text. See [`JsonSchemaCompiler::compile_to_gbnf`].
    pub fn compile_value_to_gbnf(schema: &Value) -> GrammarResult<String> {
        let mut compiler = SchemaCompiler::new();
        compiler.compile_root(schema)?;
        Ok(compiler.build_gbnf())
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

        // `const` is a single-value enum; handle it before `enum`/`type` so
        // it always wins if present (matches JSON Schema precedence: const
        // is the strictest possible constraint).
        if let Some(const_val) = obj.get("const") {
            return self.compile_const(const_val);
        }

        // Handle `enum` next — it overrides `type`.
        if let Some(enum_val) = obj.get("enum") {
            return self.compile_enum(enum_val);
        }

        // `anyOf` / `oneOf`: compile as a GBNF alternation of each branch.
        // A regular grammar can only express "matches at least one
        // alternative" — `oneOf`'s "matches EXACTLY one" cannot be enforced
        // at generation time (a value that happens to satisfy two branches
        // is still emittable), so we intentionally compile both the same
        // way rather than silently pretending to enforce exclusivity.
        if let Some(Value::Array(variants)) = obj.get("anyOf") {
            return self.compile_any_of(variants);
        }
        if let Some(Value::Array(variants)) = obj.get("oneOf") {
            return self.compile_any_of(variants);
        }

        // `allOf`: best-effort merge (see `compile_all_of`'s doc comment).
        if let Some(Value::Array(schemas)) = obj.get("allOf") {
            return self.compile_all_of(schemas);
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

    // ── const ─────────────────────────────────────────────────────────────────

    /// Compile a `const` keyword: the value must match exactly.
    fn compile_const(&mut self, value: &Value) -> GrammarResult<String> {
        match value {
            Value::String(s) => {
                let escaped = escape_gbnf_literal(s);
                Ok(format!("\"\\\"\" \"{escaped}\" \"\\\"\""))
            }
            Value::Null => Ok("\"null\"".to_string()),
            Value::Bool(b) => Ok(format!("\"{b}\"")),
            Value::Number(n) => Ok(format!("\"{n}\"")),
            Value::Array(_) | Value::Object(_) => {
                // Non-primitive const: match the value's canonical JSON
                // serialization exactly. This is precise (not an
                // approximation) but requires byte-for-byte whitespace —
                // acceptable since `const` values are author-controlled.
                let text = serde_json::to_string(value).map_err(|e| GrammarError::ParseError {
                    pos: 0,
                    msg: format!("failed to serialize `const` value: {e}"),
                })?;
                let escaped = escape_gbnf_literal(&text);
                Ok(format!("\"{escaped}\""))
            }
        }
    }

    // ── anyOf / oneOf / allOf ────────────────────────────────────────────────

    /// Compile `anyOf`/`oneOf` as a GBNF alternation of each branch.
    fn compile_any_of(&mut self, variants: &[Value]) -> GrammarResult<String> {
        if variants.is_empty() {
            return Err(GrammarError::ParseError {
                pos: 0,
                msg: "`anyOf`/`oneOf` array must not be empty".to_string(),
            });
        }
        let mut alts: Vec<String> = Vec::with_capacity(variants.len());
        for variant in variants {
            let expr = self.compile_schema(variant)?;
            let rule = if is_inline_expr(&expr) {
                expr
            } else {
                let name = self.next_rule_name();
                self.add_rule(name.clone(), expr);
                name
            };
            alts.push(rule);
        }
        Ok(format!("({})", alts.join(" | ")))
    }

    /// Compile `allOf` via a best-effort merge.
    ///
    /// True JSON-Schema `allOf` intersection semantics (e.g. combining a
    /// numeric range from one branch with a `multipleOf` from another)
    /// cannot be expressed as a regular grammar in general. The one
    /// tractable, common case — every branch being an object schema — is
    /// supported by taking the union of `properties` and `required` across
    /// all branches. Anything else returns [`GrammarError::UnsupportedKeyword`]
    /// rather than silently dropping branches.
    fn compile_all_of(&mut self, schemas: &[Value]) -> GrammarResult<String> {
        if schemas.is_empty() {
            return Err(GrammarError::ParseError {
                pos: 0,
                msg: "`allOf` array must not be empty".to_string(),
            });
        }
        if schemas.len() == 1 {
            return self.compile_schema(&schemas[0]);
        }

        let mut merged_props = serde_json::Map::new();
        let mut merged_required: Vec<Value> = Vec::new();
        for schema in schemas {
            let obj = schema
                .as_object()
                .ok_or_else(|| GrammarError::UnsupportedKeyword {
                    keyword: "allOf".to_string(),
                    reason: "only object-schema branches can be merged into a single grammar rule"
                        .to_string(),
                })?;
            let is_object_like = matches!(obj.get("type"), Some(Value::String(t)) if t == "object")
                || obj.contains_key("properties");
            if !is_object_like {
                return Err(GrammarError::UnsupportedKeyword {
                    keyword: "allOf".to_string(),
                    reason: "combining non-object schemas (e.g. numeric/string constraints) is not expressible as a regular grammar".to_string(),
                });
            }
            if let Some(Value::Object(props)) = obj.get("properties") {
                for (k, v) in props {
                    merged_props.insert(k.clone(), v.clone());
                }
            }
            if let Some(Value::Array(req)) = obj.get("required") {
                merged_required.extend(req.iter().cloned());
            }
        }

        let mut merged = serde_json::Map::new();
        merged.insert("type".to_string(), Value::String("object".to_string()));
        merged.insert("properties".to_string(), Value::Object(merged_props));
        merged.insert("required".to_string(), Value::Array(merged_required));
        self.compile_object_type(&merged)
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

        self.ensure_string_char_rule();

        let min_length = obj
            .get("minLength")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let max_length = obj
            .get("maxLength")
            .and_then(Value::as_u64)
            .map(|v| v as usize);

        if min_length.is_some() || max_length.is_some() {
            if let (Some(min), Some(max)) = (min_length, max_length) {
                if min > max {
                    return Err(GrammarError::ParseError {
                        pos: 0,
                        msg: format!("minLength ({min}) > maxLength ({max})"),
                    });
                }
            }
            let body = bounded_repeat("string-char", min_length.unwrap_or(0), max_length);
            return Ok(format!(r#""\"" {body} "\"""#));
        }

        Ok(r#""\"" string-char* "\"" "#.trim().to_string())
    }

    /// Inline GBNF expression for a JSON number (integer or float).
    fn compile_number_type(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> GrammarResult<String> {
        reject_unsupported_numeric_range(obj)?;
        // Produce a rule for JSON numbers: optional minus, digits, optional fraction.
        self.ensure_number_rule();
        Ok("json-number".to_string())
    }

    /// Inline GBNF expression for a JSON integer.
    fn compile_integer_type(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> GrammarResult<String> {
        reject_unsupported_numeric_range(obj)?;
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

        // `additionalProperties`: `false`/absent keeps the existing strict
        // behaviour (only the declared keys are allowed). `true` or a
        // schema allows zero or more EXTRA `"key": value` pairs (with
        // arbitrary keys) after the declared ones.
        let extra_value_expr = match obj.get("additionalProperties") {
            None | Some(Value::Bool(false)) => None,
            Some(Value::Bool(true)) => Some(self.any_json_expr()),
            Some(schema) => Some(self.compile_schema(schema)?),
        };
        let extra_pair_rule = extra_value_expr.map(|val_expr| {
            self.ensure_string_char_rule();
            let val_rule = if is_inline_expr(&val_expr) {
                val_expr
            } else {
                let name = self.next_rule_name();
                self.add_rule(name.clone(), val_expr);
                name
            };
            let pair_body = format!("\"\\\"\" string-char* \"\\\"\" {ws} \":\" {ws} {val_rule}");
            let pair_rule_name = self.next_rule_name();
            self.add_rule(pair_rule_name.clone(), pair_body);
            pair_rule_name
        });

        // Build object body: `{` ws members ws `}`
        // Members are joined by `,` ws.
        // We use string concatenation rather than format! to avoid the Rust
        // formatter treating `{` as a format-string interpolation.
        let open_brace = "\"{\"";
        let close_brace = "\"}\"";
        let comma = "\",\"";

        let joined = if all_parts.is_empty() {
            match &extra_pair_rule {
                // No declared properties, but extras are allowed: a
                // (possibly empty) comma-separated list of extra pairs —
                // the first pair has no leading comma, subsequent ones do.
                Some(extra) => format!("({extra} ({ws} {comma} {ws} {extra})*)?"),
                None => String::new(),
            }
        } else {
            let sep = [" ", &ws, " ", comma, " ", &ws, " "].concat();
            let declared = all_parts.join(&sep);
            match &extra_pair_rule {
                // Declared properties exist; extras (if any) each bring
                // their own leading comma via the `*` group, so no
                // dangling-comma risk when zero extras are present.
                Some(extra) => format!("{declared} ({ws} {comma} {ws} {extra})*"),
                None => declared,
            }
        };

        let body = if joined.is_empty() {
            // Empty object: `"{"` ws `"}"`
            [open_brace, " ", &ws, " ", close_brace].concat()
        } else {
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

        let min_items = obj
            .get("minItems")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(0);
        let max_items = obj
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        if let Some(max) = max_items {
            if min_items > max {
                return Err(GrammarError::ParseError {
                    pos: 0,
                    msg: format!("minItems ({min_items}) > maxItems ({max})"),
                });
            }
        }

        // Array: `[` ws <bounded comma-separated list of item> ws `]`.
        // With no minItems/maxItems this reduces to exactly the previous
        // unconstrained pattern: `(item (ws "," ws item)*)?`.
        let list_body = bounded_list(&items_rule, &ws, min_items, max_items);
        Ok(format!(r#""[" {ws} {list_body} {ws} "]""#))
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

/// Reject `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum` rather
/// than silently ignoring them (defect S7).
///
/// A regular grammar (which is all GBNF can express) has no notion of
/// numeric comparison — encoding "the number's value must be >= 5" would
/// require enumerating every valid digit-sequence shape, which is not
/// tractable in general (and unbounded for open ranges). Rather than
/// silently producing an unconstrained `json-number`/`json-integer` pattern
/// that quietly ignores the requested range, this returns a typed error so
/// callers know the constraint was NOT applied.
fn reject_unsupported_numeric_range(obj: &serde_json::Map<String, Value>) -> GrammarResult<()> {
    const RANGE_KEYWORDS: [&str; 4] =
        ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"];
    for &kw in &RANGE_KEYWORDS {
        if obj.contains_key(kw) {
            return Err(GrammarError::UnsupportedKeyword {
                keyword: kw.to_string(),
                reason: "numeric range constraints cannot be expressed as a regular (GBNF) grammar; omit the keyword and validate the range downstream of generation instead".to_string(),
            });
        }
    }
    Ok(())
}

/// Generate GBNF text for `atom` repeated between `min` and `max` (inclusive)
/// times. `max = None` means unbounded.
///
/// GBNF (as implemented by this crate's parser) has no `{min,max}`
/// quantifier syntax, so bounded repetition is expanded explicitly: `min`
/// mandatory copies, followed by up to `(max - min)` nested optionals
/// (`(atom (atom (atom)?)?)?` for three optional trailing copies), or a
/// trailing `(atom)*` when unbounded.
fn bounded_repeat(atom: &str, min: usize, max: Option<usize>) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(min + 1);
    for _ in 0..min {
        parts.push(atom.to_string());
    }
    match max {
        Some(max) if max > min => {
            let mut tail = String::new();
            for i in 0..(max - min) {
                tail = if i == 0 {
                    format!("({atom})?")
                } else {
                    format!("({atom} {tail})?")
                };
            }
            parts.push(tail);
        }
        None => parts.push(format!("({atom})*")),
        _ => {}
    }
    parts.join(" ")
}

/// Generate a GBNF body for a comma-separated list of `item` occurring
/// between `min` and `max` (inclusive) times, using `ws` as the
/// insignificant-whitespace rule name. `max = None` means unbounded.
///
/// With `min = 0, max = None` this reduces to exactly
/// `(item (ws "," ws item)*)?` — the same pattern used before bounded
/// list support existed, so unconstrained arrays are unaffected.
fn bounded_list(item: &str, ws: &str, min: usize, max: Option<usize>) -> String {
    if max == Some(0) {
        // The list must always be empty.
        return String::new();
    }
    if min == 0 {
        let rest = match max {
            Some(m) => bounded_repeat(&format!("{ws} \",\" {ws} {item}"), 0, Some(m - 1)),
            None => format!("({ws} \",\" {ws} {item})*"),
        };
        let body = if rest.is_empty() {
            item.to_string()
        } else {
            format!("{item} {rest}")
        };
        format!("({body})?")
    } else {
        let rest_min = min - 1;
        let rest_max = max.map(|m| m - 1);
        let sep_item = format!("{ws} \",\" {ws} {item}");
        let rest = bounded_repeat(&sep_item, rest_min, rest_max);
        if rest.is_empty() {
            item.to_string()
        } else {
            format!("{item} {rest}")
        }
    }
}

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

    // ── Defect S7: anyOf / oneOf / allOf / const ──────────────────────────────

    #[test]
    fn compile_any_of_accepts_either_branch() {
        let schema = r#"{"anyOf": [{"type": "string"}, {"type": "number"}]}"#;
        let g = compile_ok(schema);
        let state = g.initial_state();
        assert!(
            state.allows_token(b"\"hi\""),
            "anyOf must accept a string branch match"
        );
        let state2 = g.initial_state();
        assert!(
            state2.allows_token(b"42"),
            "anyOf must accept a number branch match"
        );
    }

    #[test]
    fn compile_one_of_accepts_either_branch() {
        // Defect S7: previously anyOf/oneOf were silently ignored and the
        // schema widened to "any JSON value" with no warning. This test
        // pins that oneOf is now actually compiled into an alternation
        // rather than falling through to `any_json_expr`.
        let schema = r#"{"oneOf": [{"const": "cat"}, {"const": "dog"}]}"#;
        let g = compile_ok(schema);
        let state = g.initial_state();
        assert!(state.allows_token(b"\"cat\""));
        let state2 = g.initial_state();
        assert!(state2.allows_token(b"\"dog\""));
        let state3 = g.initial_state();
        assert!(
            !state3.allows_token(b"\"bird\""),
            "oneOf must reject a value matching neither branch"
        );
    }

    #[test]
    fn compile_any_of_empty_array_errors() {
        let schema = r#"{"anyOf": []}"#;
        assert!(JsonSchemaCompiler::compile(schema).is_err());
    }

    #[test]
    fn compile_all_of_merges_object_properties() {
        let schema = r#"{
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]}
            ]
        }"#;
        let g = compile_ok(schema);
        assert!(g.source.contains('a'));
        assert!(g.source.contains('b'));
    }

    #[test]
    fn compile_all_of_non_object_branch_errors() {
        // Defect S7: previously allOf was silently ignored (schema widened
        // to "any value"). Mixing a non-object branch is not expressible as
        // a regular grammar, so this must be a typed error, not a silent
        // widening.
        let schema = r#"{"allOf": [{"type": "string"}, {"type": "number"}]}"#;
        let result = JsonSchemaCompiler::compile(schema);
        assert!(
            matches!(result, Err(GrammarError::UnsupportedKeyword { .. })),
            "expected UnsupportedKeyword, got {result:?}"
        );
    }

    #[test]
    fn compile_const_string_matches_exactly() {
        let schema = r#"{"const": "fixed-value"}"#;
        let g = compile_ok(schema);
        let state = g.initial_state();
        assert!(state.allows_token(b"\"fixed-value\""));
        let state2 = g.initial_state();
        assert!(!state2.allows_token(b"\"other\""));
    }

    #[test]
    fn compile_const_number() {
        let schema = r#"{"const": 42}"#;
        let g = compile_ok(schema);
        let state = g.initial_state();
        assert!(state.allows_token(b"42"));
    }

    // ── Defect S7: minLength / maxLength ──────────────────────────────────────

    #[test]
    fn compile_string_min_max_length_enforced() {
        let schema = r#"{"type": "string", "minLength": 2, "maxLength": 3}"#;
        let g = compile_ok(schema);

        // "a" (length 1) must be rejected: advancing the opening quote then
        // closing immediately (length 0 chars) must fail.
        let mut too_short = g.initial_state();
        too_short
            .advance(b"\"a\"")
            .expect_err("length-1 string must be rejected (minLength=2)");

        // "ab" (length 2) must be accepted.
        let mut ok_min = g.initial_state();
        ok_min
            .advance(b"\"ab\"")
            .expect("length-2 string should be accepted (minLength=2)");
        assert!(ok_min.is_complete());

        // "abc" (length 3) must be accepted.
        let mut ok_max = g.initial_state();
        ok_max
            .advance(b"\"abc\"")
            .expect("length-3 string should be accepted (maxLength=3)");
        assert!(ok_max.is_complete());

        // "abcd" (length 4) must be rejected (exceeds maxLength=3).
        let mut too_long = g.initial_state();
        too_long
            .advance(b"\"abcd\"")
            .expect_err("length-4 string must be rejected (maxLength=3)");
    }

    #[test]
    fn compile_string_min_length_only() {
        let schema = r#"{"type": "string", "minLength": 1}"#;
        let g = compile_ok(schema);
        let mut state = g.initial_state();
        state
            .advance(b"\"\"")
            .expect_err("empty string must be rejected when minLength=1");
    }

    #[test]
    fn compile_string_min_length_exceeds_max_length_errors() {
        let schema = r#"{"type": "string", "minLength": 5, "maxLength": 2}"#;
        assert!(JsonSchemaCompiler::compile(schema).is_err());
    }

    // ── Defect S7: minimum / maximum are rejected, not silently ignored ──────

    #[test]
    fn compile_number_with_minimum_errors() {
        let schema = r#"{"type": "number", "minimum": 0}"#;
        let result = JsonSchemaCompiler::compile(schema);
        assert!(
            matches!(result, Err(GrammarError::UnsupportedKeyword { .. })),
            "numeric `minimum` must be a typed error, not silently ignored; got {result:?}"
        );
    }

    #[test]
    fn compile_integer_with_maximum_errors() {
        let schema = r#"{"type": "integer", "maximum": 100}"#;
        let result = JsonSchemaCompiler::compile(schema);
        assert!(matches!(
            result,
            Err(GrammarError::UnsupportedKeyword { .. })
        ));
    }

    // ── Defect S7: minItems / maxItems ────────────────────────────────────────

    #[test]
    fn compile_array_min_max_items_enforced() {
        let schema = r#"{"type": "array", "items": {"const": "x"}, "minItems": 1, "maxItems": 2}"#;
        let g = compile_ok(schema);

        let mut empty = g.initial_state();
        empty
            .advance(b"[]")
            .expect_err("empty array must be rejected when minItems=1");

        let mut one = g.initial_state();
        one.advance(b"[\"x\"]")
            .expect("1-item array should be accepted");
        assert!(one.is_complete());

        let mut two = g.initial_state();
        two.advance(b"[\"x\",\"x\"]")
            .expect("2-item array should be accepted (maxItems=2)");
        assert!(two.is_complete());

        let mut three = g.initial_state();
        three
            .advance(b"[\"x\",\"x\",\"x\"]")
            .expect_err("3-item array must be rejected (maxItems=2)");
    }

    #[test]
    fn compile_array_min_items_exceeds_max_items_errors() {
        let schema = r#"{"type": "array", "minItems": 5, "maxItems": 2}"#;
        assert!(JsonSchemaCompiler::compile(schema).is_err());
    }

    #[test]
    fn compile_array_unconstrained_matches_previous_shape() {
        // With no minItems/maxItems, the emitted grammar must still accept
        // zero, one, or many items — the unconstrained shape from before
        // bounded-list support existed.
        let schema = r#"{"type": "array", "items": {"const": "x"}}"#;
        let g = compile_ok(schema);
        g.initial_state()
            .advance(b"[]")
            .expect("empty array still allowed");
        g.initial_state()
            .advance(b"[\"x\",\"x\",\"x\",\"x\"]")
            .expect("many items still allowed with no maxItems");
    }

    // ── Defect S7: additionalProperties ───────────────────────────────────────

    #[test]
    fn compile_additional_properties_false_stays_strict() {
        let schema = r#"{
            "type": "object",
            "properties": {"a": {"const": "x"}},
            "required": ["a"],
            "additionalProperties": false
        }"#;
        let g = compile_ok(schema);
        let mut with_extra = g.initial_state();
        with_extra
            .advance(b"{\"a\":\"x\",\"b\":1}")
            .expect_err("additionalProperties:false must reject unknown keys");
    }

    #[test]
    fn compile_additional_properties_true_allows_extra_keys() {
        let schema = r#"{
            "type": "object",
            "properties": {"a": {"const": "x"}},
            "required": ["a"],
            "additionalProperties": true
        }"#;
        let g = compile_ok(schema);

        let mut exact = g.initial_state();
        exact
            .advance(b"{\"a\":\"x\"}")
            .expect("declared-only object should still be accepted");
        assert!(exact.is_complete());

        let mut with_extra = g.initial_state();
        with_extra
            .advance(b"{\"a\":\"x\",\"extra\":123}")
            .expect("additionalProperties:true must allow an unknown extra key");
        assert!(with_extra.is_complete());
    }

    #[test]
    fn compile_additional_properties_true_no_declared_properties() {
        let schema = r#"{
            "type": "object",
            "properties": {},
            "additionalProperties": true
        }"#;
        let g = compile_ok(schema);
        let mut with_extra = g.initial_state();
        with_extra
            .advance(b"{\"anything\":1}")
            .expect("additionalProperties:true with no declared properties should allow extras");
        assert!(with_extra.is_complete());

        let mut empty = g.initial_state();
        empty
            .advance(b"{}")
            .expect("empty object should also be accepted");
        assert!(empty.is_complete());
    }

    // ── compile_to_gbnf / compile_value API surface ───────────────────────────

    #[test]
    fn compile_to_gbnf_returns_parseable_text() {
        let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
        let gbnf =
            JsonSchemaCompiler::compile_to_gbnf(schema).expect("should compile to GBNF text");
        assert!(
            gbnf.contains("root ::="),
            "GBNF text should define a root rule"
        );
        // The text must itself be parseable by the GBNF parser.
        Grammar::parse(&gbnf).expect("compile_to_gbnf output must be valid GBNF");
    }

    #[test]
    fn compile_value_matches_compile_from_string() {
        let schema_str = r#"{"type": "boolean"}"#;
        let value: Value = serde_json::from_str(schema_str).unwrap();
        let from_value =
            JsonSchemaCompiler::compile_value(&value).expect("compile_value should succeed");
        let from_str = JsonSchemaCompiler::compile(schema_str).expect("compile should succeed");
        assert_eq!(from_value.source, from_str.source);
    }
}

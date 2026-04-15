//! GBNF grammar parser.
//!
//! Parses a subset of GBNF (Grammar-Based Natural Form) syntax, which is the
//! grammar format used by llama.cpp for constrained generation.
//!
//! Supported syntax:
//! - Rules: `rule_name ::= body`
//! - Sequences: `item1 item2`
//! - Alternations: `body1 | body2`
//! - Character classes: `[abc]`, `[a-z]`, `[0-9]`, `[^...]` (negated)
//! - Repetitions: `body*`, `body+`, `body?`
//! - Quoted strings: `"text"` (ASCII + UTF-8)
//! - Rule references: bare identifiers
//! - Grouping: `(body)`

use std::collections::HashMap;

use super::error::{GrammarError, GrammarResult};

/// An inclusive byte-range in a character class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharRange {
    /// Inclusive lower bound.
    pub lo: u8,
    /// Inclusive upper bound.
    pub hi: u8,
}

impl CharRange {
    /// Construct a single-byte range.
    pub fn single(b: u8) -> Self {
        Self { lo: b, hi: b }
    }

    /// Construct a range from `lo` to `hi` inclusive.
    pub fn range(lo: u8, hi: u8) -> Self {
        Self { lo, hi }
    }

    /// Returns true if `b` is within this range.
    pub fn contains(&self, b: u8) -> bool {
        b >= self.lo && b <= self.hi
    }
}

/// A single node in the grammar tree.
#[derive(Debug, Clone)]
pub enum GrammarNode {
    /// Matches an exact sequence of bytes.
    Literal(Vec<u8>),
    /// Matches a single byte that falls within any of the listed ranges.
    /// If `negated` is true, matches bytes that fall outside all ranges.
    CharClass {
        ranges: Vec<CharRange>,
        negated: bool,
    },
    /// Reference to another named rule.
    RuleRef(String),
    /// Matches all items in the sequence, in order.
    Sequence(Vec<GrammarNode>),
    /// Matches one of the alternatives.
    Alternation(Vec<GrammarNode>),
    /// Repeat a node `min..=max` times (max=None means unbounded).
    Repeat {
        node: Box<GrammarNode>,
        min: usize,
        max: Option<usize>,
    },
}

/// A fully parsed GBNF grammar.
#[derive(Debug, Clone)]
pub struct Grammar {
    /// Map from rule name to its grammar node.
    pub rules: HashMap<String, GrammarNode>,
    /// The root rule name (first rule defined, or "root" if present).
    pub root: String,
}

impl Grammar {
    /// Parse a GBNF grammar string into a `Grammar` structure.
    pub fn parse(input: &str) -> GrammarResult<Self> {
        let mut parser = GbnfParser::new(input);
        parser.parse_grammar()
    }

    /// Get the grammar node for the root rule.
    pub fn root_node(&self) -> GrammarResult<&GrammarNode> {
        self.rules
            .get(&self.root)
            .ok_or_else(|| GrammarError::UnknownRule {
                rule: self.root.clone(),
            })
    }
}

// ─── Parser internals ────────────────────────────────────────────────────────

/// Recursive descent parser for GBNF grammars.
struct GbnfParser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> GbnfParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    // ── Error helpers ────────────────────────────────────────────────────────

    fn parse_error(&self, msg: impl Into<String>) -> GrammarError {
        GrammarError::ParseError {
            pos: self.pos,
            msg: msg.into(),
        }
    }

    // ── Character tests ──────────────────────────────────────────────────────

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.input.get(self.pos + 1).copied()
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn consume(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.advance();
        Some(b)
    }

    fn expect(&mut self, expected: u8) -> GrammarResult<()> {
        match self.consume() {
            Some(b) if b == expected => Ok(()),
            Some(b) => Err(self.parse_error(format!(
                "expected '{}' but got '{}'",
                expected as char, b as char
            ))),
            None => Err(self.parse_error(format!(
                "expected '{}' but reached end of input",
                expected as char
            ))),
        }
    }

    fn is_ident_start(b: u8) -> bool {
        b.is_ascii_alphabetic() || b == b'_'
    }

    fn is_ident_cont(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
    }

    // ── Whitespace / comment skipping ────────────────────────────────────────

    /// Skip spaces and tabs (but NOT newlines — those separate rules).
    fn skip_inline_ws(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t')) {
            self.advance();
        }
    }

    /// Skip all whitespace including newlines and `#`-comments.
    fn skip_ws_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => {
                    self.advance();
                }
                Some(b'#') => {
                    while !matches!(self.peek(), None | Some(b'\n')) {
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    // ── Top-level grammar parsing ────────────────────────────────────────────

    fn parse_grammar(&mut self) -> GrammarResult<Grammar> {
        let mut rules: HashMap<String, GrammarNode> = HashMap::new();
        let mut first_rule: Option<String> = None;

        self.skip_ws_and_comments();

        while self.pos < self.input.len() {
            let name = self.parse_ident()?;
            self.skip_inline_ws();

            // Expect ::=
            if self.pos + 3 > self.input.len() || &self.input[self.pos..self.pos + 3] != b"::=" {
                return Err(self.parse_error("expected '::=' after rule name"));
            }
            self.pos += 3;
            self.skip_inline_ws();

            let node = self.parse_alternation()?;

            if first_rule.is_none() {
                first_rule = Some(name.clone());
            }
            rules.insert(name, node);

            self.skip_ws_and_comments();
        }

        if rules.is_empty() {
            return Err(GrammarError::ParseError {
                pos: 0,
                msg: "grammar has no rules".to_string(),
            });
        }

        // The root rule is "root" if it exists, otherwise the first rule.
        let root = if rules.contains_key("root") {
            "root".to_string()
        } else {
            first_rule.unwrap_or_default()
        };

        Ok(Grammar { rules, root })
    }

    // ── Identifier parsing ───────────────────────────────────────────────────

    fn parse_ident(&mut self) -> GrammarResult<String> {
        let start = self.pos;
        match self.peek() {
            Some(b) if Self::is_ident_start(b) => {
                self.advance();
            }
            _ => return Err(self.parse_error("expected identifier")),
        }
        while matches!(self.peek(), Some(b) if Self::is_ident_cont(b)) {
            self.advance();
        }
        let slice = &self.input[start..self.pos];
        Ok(String::from_utf8_lossy(slice).into_owned())
    }

    // ── Alternation (lowest precedence) ─────────────────────────────────────

    fn parse_alternation(&mut self) -> GrammarResult<GrammarNode> {
        let first = self.parse_sequence()?;
        self.skip_inline_ws();

        if !matches!(self.peek(), Some(b'|')) {
            return Ok(first);
        }

        let mut alternatives = vec![first];
        while matches!(self.peek(), Some(b'|')) {
            self.advance(); // consume '|'
            self.skip_inline_ws();
            alternatives.push(self.parse_sequence()?);
            self.skip_inline_ws();
        }

        Ok(GrammarNode::Alternation(alternatives))
    }

    // ── Sequence ─────────────────────────────────────────────────────────────

    fn parse_sequence(&mut self) -> GrammarResult<GrammarNode> {
        let mut items: Vec<GrammarNode> = Vec::new();

        loop {
            self.skip_inline_ws();
            // Stop at end of input, newline (rule boundary), '|', ')', or ']'
            match self.peek() {
                None | Some(b'\n') | Some(b'\r') | Some(b'|') | Some(b')') => break,
                // '#' starts a comment — stop the sequence
                Some(b'#') => break,
                _ => {}
            }
            let item = self.parse_item()?;
            items.push(item);
        }

        match items.len() {
            0 => Ok(GrammarNode::Literal(vec![])),
            1 => Ok(items.remove(0)),
            _ => Ok(GrammarNode::Sequence(items)),
        }
    }

    // ── Single item (possibly with suffix quantifier) ────────────────────────

    fn parse_item(&mut self) -> GrammarResult<GrammarNode> {
        let base = self.parse_atom()?;
        self.parse_quantifier(base)
    }

    fn parse_quantifier(&mut self, base: GrammarNode) -> GrammarResult<GrammarNode> {
        match self.peek() {
            Some(b'*') => {
                self.advance();
                Ok(GrammarNode::Repeat {
                    node: Box::new(base),
                    min: 0,
                    max: None,
                })
            }
            Some(b'+') => {
                self.advance();
                Ok(GrammarNode::Repeat {
                    node: Box::new(base),
                    min: 1,
                    max: None,
                })
            }
            Some(b'?') => {
                self.advance();
                Ok(GrammarNode::Repeat {
                    node: Box::new(base),
                    min: 0,
                    max: Some(1),
                })
            }
            _ => Ok(base),
        }
    }

    // ── Atoms ────────────────────────────────────────────────────────────────

    fn parse_atom(&mut self) -> GrammarResult<GrammarNode> {
        match self.peek() {
            Some(b'"') => self.parse_string_literal(),
            Some(b'[') => self.parse_char_class(),
            Some(b'(') => {
                self.advance(); // consume '('
                self.skip_inline_ws();
                let inner = self.parse_alternation()?;
                self.skip_inline_ws();
                self.expect(b')')?;
                Ok(inner)
            }
            Some(b) if Self::is_ident_start(b) => {
                let name = self.parse_ident()?;
                Ok(GrammarNode::RuleRef(name))
            }
            Some(b) => {
                Err(self.parse_error(format!("unexpected character '{}' in grammar", b as char)))
            }
            None => Err(self.parse_error("unexpected end of input")),
        }
    }

    // ── String literal parsing ───────────────────────────────────────────────

    fn parse_string_literal(&mut self) -> GrammarResult<GrammarNode> {
        self.expect(b'"')?;
        let mut bytes: Vec<u8> = Vec::new();

        loop {
            match self.consume() {
                None => return Err(self.parse_error("unterminated string literal")),
                Some(b'"') => break,
                Some(b'\\') => {
                    let escaped = self.parse_escape_sequence()?;
                    bytes.extend_from_slice(&escaped);
                }
                Some(b) => bytes.push(b),
            }
        }

        Ok(GrammarNode::Literal(bytes))
    }

    /// Parse a backslash escape sequence, returning the byte(s) it represents.
    fn parse_escape_sequence(&mut self) -> GrammarResult<Vec<u8>> {
        match self.consume() {
            Some(b'n') => Ok(vec![b'\n']),
            Some(b'r') => Ok(vec![b'\r']),
            Some(b't') => Ok(vec![b'\t']),
            Some(b'\\') => Ok(vec![b'\\']),
            Some(b'"') => Ok(vec![b'"']),
            Some(b'\'') => Ok(vec![b'\'']),
            Some(b'0') => Ok(vec![0u8]),
            Some(b'x') => {
                // \xHH — two hex digits
                let hi = self.consume_hex_digit()?;
                let lo = self.consume_hex_digit()?;
                Ok(vec![(hi << 4) | lo])
            }
            Some(b'u') => {
                // \uHHHH — four hex digits, encode as UTF-8
                self.expect(b'{')?;
                let mut codepoint: u32 = 0;
                let mut digits = 0usize;
                while !matches!(self.peek(), Some(b'}')) {
                    let d = self.consume_hex_digit()?;
                    codepoint = (codepoint << 4) | (d as u32);
                    digits += 1;
                    if digits > 6 {
                        return Err(self.parse_error("\\u{...} codepoint too large"));
                    }
                }
                self.expect(b'}')?;
                let ch = char::from_u32(codepoint)
                    .ok_or_else(|| self.parse_error("invalid Unicode codepoint"))?;
                let mut buf = [0u8; 4];
                Ok(ch.encode_utf8(&mut buf).as_bytes().to_vec())
            }
            Some(b) => Err(self.parse_error(format!("unknown escape sequence '\\{}'", b as char))),
            None => Err(self.parse_error("truncated escape sequence")),
        }
    }

    fn consume_hex_digit(&mut self) -> GrammarResult<u8> {
        match self.consume() {
            Some(b @ b'0'..=b'9') => Ok(b - b'0'),
            Some(b @ b'a'..=b'f') => Ok(b - b'a' + 10),
            Some(b @ b'A'..=b'F') => Ok(b - b'A' + 10),
            Some(b) => Err(self.parse_error(format!("expected hex digit, got '{}'", b as char))),
            None => Err(self.parse_error("expected hex digit, reached end of input")),
        }
    }

    // ── Character class parsing ──────────────────────────────────────────────

    fn parse_char_class(&mut self) -> GrammarResult<GrammarNode> {
        self.expect(b'[')?;
        let negated = if matches!(self.peek(), Some(b'^')) {
            self.advance();
            true
        } else {
            false
        };

        let mut ranges: Vec<CharRange> = Vec::new();

        loop {
            match self.peek() {
                None => return Err(self.parse_error("unterminated character class")),
                Some(b']') => {
                    self.advance();
                    break;
                }
                Some(b'\\') => {
                    self.advance(); // consume '\'
                    let b = self.parse_class_escape()?;
                    // Check for range: \x-y
                    if matches!(self.peek(), Some(b'-')) && !matches!(self.peek2(), Some(b']')) {
                        self.advance(); // consume '-'
                        let hi = self.parse_class_byte()?;
                        ranges.push(CharRange::range(b, hi));
                    } else {
                        ranges.push(CharRange::single(b));
                    }
                }
                Some(_) => {
                    let lo = self.parse_class_byte()?;
                    if matches!(self.peek(), Some(b'-')) && !matches!(self.peek2(), Some(b']')) {
                        self.advance(); // consume '-'
                        let hi = self.parse_class_byte()?;
                        ranges.push(CharRange::range(lo, hi));
                    } else {
                        ranges.push(CharRange::single(lo));
                    }
                }
            }
        }

        Ok(GrammarNode::CharClass { ranges, negated })
    }

    /// Parse a single literal byte inside a character class.
    fn parse_class_byte(&mut self) -> GrammarResult<u8> {
        match self.peek() {
            Some(b'\\') => {
                self.advance();
                self.parse_class_escape()
            }
            Some(b) => {
                self.advance();
                Ok(b)
            }
            None => Err(self.parse_error("unexpected end inside character class")),
        }
    }

    /// Parse a backslash escape inside `[...]`.
    fn parse_class_escape(&mut self) -> GrammarResult<u8> {
        match self.consume() {
            Some(b'n') => Ok(b'\n'),
            Some(b'r') => Ok(b'\r'),
            Some(b't') => Ok(b'\t'),
            Some(b'\\') => Ok(b'\\'),
            Some(b']') => Ok(b']'),
            Some(b'-') => Ok(b'-'),
            Some(b'^') => Ok(b'^'),
            Some(b'x') => {
                let hi = self.consume_hex_digit()?;
                let lo = self.consume_hex_digit()?;
                Ok((hi << 4) | lo)
            }
            Some(b) => Err(self.parse_error(format!("unknown class escape '\\{}'", b as char))),
            None => Err(self.parse_error("truncated class escape")),
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_literal_alternation() {
        let g = Grammar::parse(r#"root ::= "yes" | "no""#).unwrap();
        assert_eq!(g.root, "root");
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_char_class_range() {
        let g = Grammar::parse("root ::= [a-z]+").unwrap();
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_sequence() {
        let g = Grammar::parse(r#"root ::= [a-z]+ ":" [0-9]+"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_parse_multiple_rules() {
        let grammar_str = "root ::= greeting\ngreeting ::= \"hello\"";
        let g = Grammar::parse(grammar_str).unwrap();
        assert_eq!(g.root, "root");
        assert!(g.rules.contains_key("greeting"));
    }

    #[test]
    fn test_parse_empty_grammar_fails() {
        let result = Grammar::parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_negated_char_class() {
        let g = Grammar::parse("root ::= [^0-9]+").unwrap();
        assert!(g.rules.contains_key("root"));
        match g.rules.get("root").unwrap() {
            GrammarNode::Repeat { node, .. } => match node.as_ref() {
                GrammarNode::CharClass { negated, .. } => assert!(*negated),
                other => panic!("expected CharClass, got {:?}", other),
            },
            other => panic!("expected Repeat, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_grouping() {
        let g = Grammar::parse(r#"root ::= ("a" | "b")+"#).unwrap();
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_json_simple() {
        // A small JSON-like grammar
        let grammar_str = r#"root ::= "{" ws "}"
ws ::= [ \t\n]*"#;
        let g = Grammar::parse(grammar_str).unwrap();
        assert_eq!(g.root, "root");
    }

    // ── String literal escape sequences ──────────────────────────────────────

    #[test]
    fn test_parse_escape_newline() {
        // \n in a string literal
        let g = Grammar::parse(r#"root ::= "\n""#).expect("test: \\n escape should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Literal(bytes) => assert_eq!(bytes, b"\n"),
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_escape_tab() {
        let g = Grammar::parse(r#"root ::= "\t""#).expect("test: \\t escape should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Literal(bytes) => assert_eq!(bytes, b"\t"),
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_escape_carriage_return() {
        let g = Grammar::parse(r#"root ::= "\r""#).expect("test: \\r escape should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Literal(bytes) => assert_eq!(bytes, b"\r"),
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_escape_backslash() {
        let g = Grammar::parse(r#"root ::= "\\""#).expect("test: \\\\ escape should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Literal(bytes) => assert_eq!(bytes, b"\\"),
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_escape_quote() {
        let g = Grammar::parse(r#"root ::= "\"hi\"""#).expect("test: escaped quote should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Literal(bytes) => {
                assert_eq!(bytes[0], b'"');
                assert_eq!(*bytes.last().expect("test: last byte"), b'"');
            }
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_escape_null() {
        let g = Grammar::parse(r#"root ::= "\0""#).expect("test: \\0 escape should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Literal(bytes) => assert_eq!(bytes, &[0u8]),
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_escape_hex() {
        // \x41 = 'A'
        let g = Grammar::parse(r#"root ::= "\x41""#).expect("test: \\x41 should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Literal(bytes) => assert_eq!(bytes, &[0x41u8]),
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_escape_unicode() {
        // \u{0041} = 'A' in UTF-8
        let g = Grammar::parse(r#"root ::= "\u{0041}""#).expect("test: \\u{0041} should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Literal(bytes) => assert_eq!(bytes, b"A"),
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_escape_unicode_multibyte() {
        // \u{263A} = ☺, U+263A → UTF-8: 0xE2 0x98 0xBA
        let g = Grammar::parse(r#"root ::= "\u{263A}""#).expect("test: \\u{263A} should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Literal(bytes) => {
                assert_eq!(bytes, "☺".as_bytes());
            }
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    // ── String literal error paths ────────────────────────────────────────────

    #[test]
    fn test_parse_unterminated_string_errors() {
        let result = Grammar::parse(r#"root ::= "abc"#);
        assert!(result.is_err(), "unterminated string should be an error");
    }

    #[test]
    fn test_parse_unknown_escape_errors() {
        let result = Grammar::parse(r#"root ::= "\q""#);
        assert!(result.is_err(), "unknown escape \\q should be an error");
    }

    #[test]
    fn test_parse_truncated_escape_errors() {
        // String ends with bare backslash
        let result = Grammar::parse("root ::= \"\\\"");
        // The parser will either see a \"  (escaped quote, then unterminated)
        // or fail on truncated. Either error is acceptable.
        assert!(
            result.is_err(),
            "truncated escape at end of input should error"
        );
    }

    #[test]
    fn test_parse_hex_escape_invalid_digit_errors() {
        // \xGG — 'G' is not a valid hex digit
        let result = Grammar::parse(r#"root ::= "\xGG""#);
        assert!(result.is_err(), "invalid hex digit in \\x should error");
    }

    #[test]
    fn test_parse_unicode_escape_invalid_codepoint_errors() {
        // \u{D800} — surrogate, not a valid Unicode scalar
        // Build the grammar string without a raw string literal to avoid Rust
        // interpreting the { } as a format argument.
        let grammar_str = "root ::= \"\\u{D800}\"";
        let result = Grammar::parse(grammar_str);
        assert!(
            result.is_err(),
            "surrogate codepoint \\u{{D800}} should error"
        );
    }

    #[test]
    fn test_parse_unicode_escape_too_many_digits_errors() {
        // \u{1234567} — 7 hex digits is too many
        let grammar_str = "root ::= \"\\u{1234567}\"";
        let result = Grammar::parse(grammar_str);
        assert!(result.is_err(), "\\u{{...}} with >6 digits should error");
    }

    // ── Character class escape sequences ─────────────────────────────────────

    #[test]
    fn test_parse_char_class_escape_newline() {
        let g = Grammar::parse(r"root ::= [\n]").expect("test: [\\n] should parse");
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_char_class_escape_tab() {
        let g = Grammar::parse(r"root ::= [\t]").expect("test: [\\t] should parse");
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_char_class_escape_dash() {
        // \- inside a char class should represent a literal '-'
        let g = Grammar::parse(r"root ::= [\-]").expect("test: [\\-] should parse");
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_char_class_escape_caret() {
        // \^ inside a char class is a literal '^'
        let g = Grammar::parse(r"root ::= [\^]").expect("test: [\\^] should parse");
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_char_class_escape_bracket() {
        // \] inside a char class is a literal ']'
        let g = Grammar::parse(r"root ::= [\]]").expect("test: [\\]] should parse");
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_char_class_hex_escape() {
        // [\x41-\x5A] = [A-Z]
        let g = Grammar::parse(r"root ::= [\x41-\x5A]+").expect("test: [\\x41-\\x5A] should parse");
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_char_class_unknown_escape_errors() {
        let result = Grammar::parse(r"root ::= [\q]");
        assert!(
            result.is_err(),
            "unknown char class escape \\q should error"
        );
    }

    #[test]
    fn test_parse_unterminated_char_class_errors() {
        let result = Grammar::parse("root ::= [a-z");
        assert!(result.is_err(), "unterminated char class should error");
    }

    // ── Grammar structure error paths ─────────────────────────────────────────

    #[test]
    fn test_parse_missing_assign_errors() {
        let result = Grammar::parse("root = \"hello\"");
        assert!(result.is_err(), "missing ::= should error");
    }

    #[test]
    fn test_parse_unexpected_char_errors() {
        // '?' at start of an atom (not after something) is unexpected
        let result = Grammar::parse("root ::= @");
        assert!(
            result.is_err(),
            "unexpected char '@' in grammar atom should error"
        );
    }

    #[test]
    fn test_parse_unclosed_group_errors() {
        let result = Grammar::parse(r#"root ::= ("abc""#);
        assert!(result.is_err(), "unclosed group should error");
    }

    #[test]
    fn test_parse_first_rule_becomes_root_when_no_root() {
        // When there's no rule named "root", the first defined rule is the root.
        let g = Grammar::parse("entry ::= \"hello\"").expect("test: single rule without root");
        assert_eq!(g.root, "entry");
    }

    #[test]
    fn test_parse_optional_quantifier() {
        // body? → Repeat{min:0, max:Some(1)}
        let g = Grammar::parse(r#"root ::= "a"?"#).expect("test: ? quantifier should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Repeat { min, max, .. } => {
                assert_eq!(*min, 0);
                assert_eq!(*max, Some(1));
            }
            other => panic!("expected Repeat, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_star_quantifier() {
        let g = Grammar::parse(r#"root ::= "a"*"#).expect("test: * quantifier should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Repeat { min, max, .. } => {
                assert_eq!(*min, 0);
                assert_eq!(*max, None);
            }
            other => panic!("expected Repeat, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_plus_quantifier() {
        let g = Grammar::parse(r#"root ::= "a"+"#).expect("test: + quantifier should parse");
        match g.rules.get("root").expect("test: root rule") {
            GrammarNode::Repeat { min, max, .. } => {
                assert_eq!(*min, 1);
                assert_eq!(*max, None);
            }
            other => panic!("expected Repeat, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_comment_skipped() {
        // Lines starting with '#' are comments and should be ignored
        let grammar_str = "# This is a comment\nroot ::= \"hello\"";
        let g = Grammar::parse(grammar_str).expect("test: comment should be ignored");
        assert_eq!(g.root, "root");
    }

    #[test]
    fn test_root_node_accessor_ok() {
        let g = Grammar::parse(r#"root ::= "hi""#).expect("test: should parse");
        g.root_node()
            .expect("test: root_node() should succeed when root rule exists");
    }

    #[test]
    fn test_root_node_accessor_missing_rule_errors() {
        // Manually construct a Grammar with a root pointing to a non-existent rule.
        let g = Grammar {
            rules: std::collections::HashMap::new(),
            root: "missing".to_string(),
        };
        let result = g.root_node();
        assert!(result.is_err(), "root_node() on missing rule should error");
        match result {
            Err(super::super::error::GrammarError::UnknownRule { rule }) => {
                assert_eq!(rule, "missing");
            }
            other => panic!("expected UnknownRule, got {:?}", other),
        }
    }

    #[test]
    fn test_char_range_contains() {
        let r = CharRange::range(b'a', b'z');
        assert!(r.contains(b'a'));
        assert!(r.contains(b'z'));
        assert!(r.contains(b'm'));
        assert!(!r.contains(b'A'));
        assert!(!r.contains(b'0'));
    }

    #[test]
    fn test_char_range_single() {
        let r = CharRange::single(b'X');
        assert!(r.contains(b'X'));
        assert!(!r.contains(b'Y'));
        assert!(!r.contains(b'W'));
    }
}

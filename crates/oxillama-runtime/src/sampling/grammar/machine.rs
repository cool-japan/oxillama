//! Grammar state machine for constrained sampling.
//!
//! Implements an NFA-simulation approach: given the current parse state
//! (represented as a stack of grammar frames), we can determine which
//! byte sequences are valid continuations. For each candidate token
//! we run a simulation to check whether accepting those bytes can advance
//! the parse without error.

use super::error::{GrammarError, GrammarResult};
use super::parser::{Grammar, GrammarNode};

/// Maximum recursion depth for grammar simulation.
const MAX_DEPTH: usize = 128;
/// Maximum number of bytes to simulate (long tokens are rare; avoids hangs).
const MAX_SIM_BYTES: usize = 64;

// ─── Public state type ────────────────────────────────────────────────────────

/// The live parse state for constrained generation.
///
/// This is a continuation-based representation: at each step we hold the
/// remaining grammar "obligations" — the list of nodes that still need to be
/// matched, in order, before the parse is complete.
///
/// An empty continuation means we have matched the entire grammar (accepting
/// state). A non-empty continuation means more input is expected.
#[derive(Debug, Clone)]
pub struct GrammarState {
    /// Stack of remaining grammar nodes to match (front = soonest to match).
    /// Each element is a `(rule_context, node)` so we can detect accept states.
    continuation: Vec<ContNode>,
    /// The grammar this state is for (needed to dereference rule refs).
    grammar: Grammar,
}

/// A continuation node: a grammar node together with a rule-name hint
/// (used only for error messages and depth-tracking).
#[derive(Debug, Clone)]
struct ContNode {
    node: GrammarNode,
}

impl ContNode {
    fn new(node: GrammarNode) -> Self {
        Self { node }
    }
}

impl GrammarState {
    /// Create the initial grammar state (beginning of the root rule).
    pub(super) fn new(grammar: Grammar) -> Self {
        let root = grammar.root.clone();
        let mut state = Self {
            continuation: Vec::new(),
            grammar,
        };
        state
            .continuation
            .push(ContNode::new(GrammarNode::RuleRef(root)));
        state
    }

    /// Returns true if the current state is a valid accepting state —
    /// i.e., no more tokens are required.
    pub fn is_complete(&self) -> bool {
        // We're complete when the continuation is empty or all remaining
        // nodes can match empty strings.
        self.can_match_empty_continuation(&self.continuation, 0)
    }

    /// Returns true if the given token's byte sequence is a valid continuation
    /// from the current parse state.
    pub fn allows_token(&self, token_bytes: &[u8]) -> bool {
        if token_bytes.is_empty() {
            // An empty token is always allowed (it doesn't advance the parse).
            return true;
        }
        if token_bytes.len() > MAX_SIM_BYTES {
            // Very long tokens: conservatively allow them.
            return true;
        }
        // Simulate consuming the token bytes from the current continuation.
        let mut sim = SimState {
            grammar: &self.grammar,
            depth: 0,
        };
        sim.simulate_bytes(&self.continuation, token_bytes)
    }

    /// Advance the grammar state by consuming a token's bytes.
    pub fn advance(&mut self, token_bytes: &[u8]) -> GrammarResult<()> {
        if token_bytes.is_empty() {
            return Ok(());
        }
        let mut sim = SimState {
            grammar: &self.grammar,
            depth: 0,
        };
        let new_cont = sim.advance_bytes(&self.continuation, token_bytes)?;
        self.continuation = new_cont;
        Ok(())
    }

    /// Check whether a continuation list can match the empty string.
    fn can_match_empty_continuation(&self, cont: &[ContNode], depth: usize) -> bool {
        if depth > MAX_DEPTH {
            return false;
        }
        if cont.is_empty() {
            return true;
        }
        let Some((first, rest)) = cont.split_first() else {
            return false;
        };
        self.node_can_match_empty(&first.node, depth + 1)
            && self.can_match_empty_continuation(rest, depth + 1)
    }

    fn node_can_match_empty(&self, node: &GrammarNode, depth: usize) -> bool {
        if depth > MAX_DEPTH {
            return false;
        }
        match node {
            GrammarNode::Literal(bytes) => bytes.is_empty(),
            GrammarNode::CharClass { .. } => false,
            GrammarNode::RuleRef(name) => {
                if let Some(rule_node) = self.grammar.rules.get(name) {
                    self.node_can_match_empty(rule_node, depth + 1)
                } else {
                    false
                }
            }
            GrammarNode::Sequence(items) => items
                .iter()
                .all(|n| self.node_can_match_empty(n, depth + 1)),
            GrammarNode::Alternation(alts) => {
                alts.iter().any(|n| self.node_can_match_empty(n, depth + 1))
            }
            GrammarNode::Repeat { min, .. } => *min == 0,
        }
    }
}

// ─── Simulation engine ────────────────────────────────────────────────────────

/// Stateless byte-simulation context.
struct SimState<'g> {
    grammar: &'g Grammar,
    depth: usize,
}

impl<'g> SimState<'g> {
    /// Returns true if `bytes` can be consumed starting from `cont`.
    /// A successful simulation means all bytes were consumed (possibly with
    /// continuation left over).
    fn simulate_bytes(&mut self, cont: &[ContNode], bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return true;
        }
        // Expand the first node in the continuation to get all possible
        // one-byte transitions, try each that matches bytes[0], then recurse.
        self.try_consume_byte(cont, bytes[0], &bytes[1..])
    }

    /// Attempt to consume one byte `b` from `cont`, then continue with `rest`.
    fn try_consume_byte(&mut self, cont: &[ContNode], b: u8, rest: &[u8]) -> bool {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            return true; // conservative: allow when we hit the limit
        }

        let result = self.try_consume_byte_inner(cont, b, rest);
        self.depth -= 1;
        result
    }

    fn try_consume_byte_inner(&mut self, cont: &[ContNode], b: u8, rest: &[u8]) -> bool {
        if cont.is_empty() {
            return false; // more bytes but nothing left to match
        }

        let Some((first, tail)) = cont.split_first() else {
            return false;
        };

        match &first.node {
            GrammarNode::Literal(bytes) => {
                if bytes.is_empty() {
                    // Empty literal — skip it and consume from tail
                    self.try_consume_byte(tail, b, rest)
                } else if bytes[0] == b {
                    // First byte matches; produce a new continuation with the remainder
                    let remainder = &bytes[1..];
                    if remainder.is_empty() {
                        // Fully consumed this literal; continue with tail
                        self.simulate_bytes(tail, rest)
                    } else {
                        let mut new_cont: Vec<ContNode> = Vec::with_capacity(tail.len() + 1);
                        new_cont.push(ContNode::new(GrammarNode::Literal(remainder.to_vec())));
                        new_cont.extend_from_slice(tail);
                        self.simulate_bytes(&new_cont, rest)
                    }
                } else {
                    false
                }
            }

            GrammarNode::CharClass { ranges, negated } => {
                let in_class = ranges.iter().any(|r| r.contains(b));
                let matches = if *negated { !in_class } else { in_class };
                if matches {
                    self.simulate_bytes(tail, rest)
                } else {
                    false
                }
            }

            GrammarNode::RuleRef(name) => {
                let rule_node = match self.grammar.rules.get(name) {
                    Some(n) => n.clone(),
                    None => return false,
                };
                let mut new_cont: Vec<ContNode> = Vec::with_capacity(tail.len() + 1);
                new_cont.push(ContNode::new(rule_node));
                new_cont.extend_from_slice(tail);
                self.try_consume_byte(&new_cont, b, rest)
            }

            GrammarNode::Sequence(items) => {
                if items.is_empty() {
                    self.try_consume_byte(tail, b, rest)
                } else {
                    // Push all items onto the continuation (in order) before tail
                    let mut new_cont: Vec<ContNode> = Vec::with_capacity(items.len() + tail.len());
                    for item in items {
                        new_cont.push(ContNode::new(item.clone()));
                    }
                    new_cont.extend_from_slice(tail);
                    self.try_consume_byte(&new_cont, b, rest)
                }
            }

            GrammarNode::Alternation(alts) => {
                // Try each alternative; succeed if any succeeds
                for alt in alts {
                    let mut new_cont: Vec<ContNode> = Vec::with_capacity(tail.len() + 1);
                    new_cont.push(ContNode::new(alt.clone()));
                    new_cont.extend_from_slice(tail);
                    if self.try_consume_byte(&new_cont, b, rest) {
                        return true;
                    }
                }
                false
            }

            GrammarNode::Repeat { node, min, max } => {
                // Generate the set of possible unrollings. We try:
                // 1. Skip this repetition entirely (valid if min==0)
                // 2. Take one occurrence (produce node ++ Repeat{min-1..} ++ tail)
                let min = *min;
                let max = *max;

                // Option A: zero occurrences (valid when min==0)
                if min == 0 && self.try_consume_byte(tail, b, rest) {
                    return true;
                }

                // Option B: take at least one occurrence
                let can_take_more = max.is_none_or(|m| m > 0);
                if can_take_more {
                    let new_min = min.saturating_sub(1);
                    let new_max = max.map(|m| m.saturating_sub(1));
                    let inner = node.as_ref().clone();
                    let repeat_rest = GrammarNode::Repeat {
                        node: Box::new(inner.clone()),
                        min: new_min,
                        max: new_max,
                    };
                    let mut new_cont: Vec<ContNode> = Vec::with_capacity(tail.len() + 2);
                    new_cont.push(ContNode::new(inner));
                    new_cont.push(ContNode::new(repeat_rest));
                    new_cont.extend_from_slice(tail);
                    if self.try_consume_byte(&new_cont, b, rest) {
                        return true;
                    }
                }

                false
            }
        }
    }

    // ── Advance (commit) ─────────────────────────────────────────────────────

    /// Returns the new continuation after consuming `bytes` from `cont`.
    /// Returns `Err(GrammarError::Stuck)` if no valid continuation exists.
    fn advance_bytes(&mut self, cont: &[ContNode], bytes: &[u8]) -> GrammarResult<Vec<ContNode>> {
        if bytes.is_empty() {
            return Ok(cont.to_vec());
        }
        self.advance_one_byte(cont, bytes[0], &bytes[1..])
    }

    fn advance_one_byte(
        &mut self,
        cont: &[ContNode],
        b: u8,
        rest: &[u8],
    ) -> GrammarResult<Vec<ContNode>> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            return Err(GrammarError::RecursionLimit {
                rule: "(advance)".to_string(),
            });
        }
        let result = self.advance_one_byte_inner(cont, b, rest);
        self.depth -= 1;
        result
    }

    fn advance_one_byte_inner(
        &mut self,
        cont: &[ContNode],
        b: u8,
        rest: &[u8],
    ) -> GrammarResult<Vec<ContNode>> {
        if cont.is_empty() {
            return Err(GrammarError::Stuck);
        }
        let (first, tail) = cont.split_first().ok_or(GrammarError::Stuck)?;

        match &first.node {
            GrammarNode::Literal(bytes) => {
                if bytes.is_empty() {
                    self.advance_one_byte(tail, b, rest)
                } else if bytes[0] == b {
                    let remainder = &bytes[1..];
                    let mut new_cont: Vec<ContNode> = Vec::new();
                    if !remainder.is_empty() {
                        new_cont.push(ContNode::new(GrammarNode::Literal(remainder.to_vec())));
                    }
                    new_cont.extend_from_slice(tail);
                    self.advance_bytes(&new_cont, rest)
                } else {
                    Err(GrammarError::Stuck)
                }
            }

            GrammarNode::CharClass { ranges, negated } => {
                let in_class = ranges.iter().any(|r| r.contains(b));
                let matches = if *negated { !in_class } else { in_class };
                if matches {
                    self.advance_bytes(tail, rest)
                } else {
                    Err(GrammarError::Stuck)
                }
            }

            GrammarNode::RuleRef(name) => {
                let rule_node = self
                    .grammar
                    .rules
                    .get(name)
                    .ok_or_else(|| GrammarError::UnknownRule { rule: name.clone() })?
                    .clone();
                let mut new_cont: Vec<ContNode> = Vec::with_capacity(tail.len() + 1);
                new_cont.push(ContNode::new(rule_node));
                new_cont.extend_from_slice(tail);
                self.advance_one_byte(&new_cont, b, rest)
            }

            GrammarNode::Sequence(items) => {
                if items.is_empty() {
                    self.advance_one_byte(tail, b, rest)
                } else {
                    let mut new_cont: Vec<ContNode> = Vec::with_capacity(items.len() + tail.len());
                    for item in items {
                        new_cont.push(ContNode::new(item.clone()));
                    }
                    new_cont.extend_from_slice(tail);
                    self.advance_one_byte(&new_cont, b, rest)
                }
            }

            GrammarNode::Alternation(alts) => {
                // Try each alternative; return the first successful one
                for alt in alts {
                    let mut new_cont: Vec<ContNode> = Vec::with_capacity(tail.len() + 1);
                    new_cont.push(ContNode::new(alt.clone()));
                    new_cont.extend_from_slice(tail);
                    match self.advance_one_byte(&new_cont, b, rest) {
                        Ok(c) => return Ok(c),
                        Err(_) => continue,
                    }
                }
                Err(GrammarError::Stuck)
            }

            GrammarNode::Repeat { node, min, max } => {
                let min = *min;
                let max = *max;

                // Option A: zero occurrences (valid when min==0)
                if min == 0 {
                    if let Ok(c) = self.advance_one_byte(tail, b, rest) {
                        return Ok(c);
                    }
                }

                // Option B: take one more occurrence
                let can_take_more = max.is_none_or(|m| m > 0);
                if can_take_more {
                    let new_min = min.saturating_sub(1);
                    let new_max = max.map(|m| m.saturating_sub(1));
                    let inner = node.as_ref().clone();
                    let repeat_rest = GrammarNode::Repeat {
                        node: Box::new(inner.clone()),
                        min: new_min,
                        max: new_max,
                    };
                    let mut new_cont: Vec<ContNode> = Vec::with_capacity(tail.len() + 2);
                    new_cont.push(ContNode::new(inner));
                    new_cont.push(ContNode::new(repeat_rest));
                    new_cont.extend_from_slice(tail);
                    if let Ok(c) = self.advance_one_byte(&new_cont, b, rest) {
                        return Ok(c);
                    }
                }

                Err(GrammarError::Stuck)
            }
        }
    }
}

// ─── Logit masking ────────────────────────────────────────────────────────────

/// Zero out (set to `f32::NEG_INFINITY`) logits for tokens that are not allowed
/// by the current grammar state.
///
/// # Arguments
/// * `logits` - Raw logits vector; modified in-place.
/// * `state`  - Current grammar parse state.
/// * `token_vocab` - Mapping `(token_id, utf-8 bytes)` for every vocabulary entry.
pub fn apply_grammar_mask(
    logits: &mut [f32],
    state: &GrammarState,
    token_vocab: &[(u32, Vec<u8>)],
) {
    for (token_id, token_bytes) in token_vocab {
        let id = *token_id as usize;
        if id < logits.len() && !state.allows_token(token_bytes) {
            logits[id] = f32::NEG_INFINITY;
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::grammar::parser::Grammar;

    fn make_state(grammar_str: &str) -> (Grammar, GrammarState) {
        let g = Grammar::parse(grammar_str).unwrap();
        let state = GrammarState::new(g.clone());
        (g, state)
    }

    #[test]
    fn test_allows_yes_no() {
        let (_g, state) = make_state(r#"root ::= "yes" | "no""#);
        assert!(state.allows_token(b"yes"));
        assert!(state.allows_token(b"no"));
        assert!(!state.allows_token(b"maybe"));
        assert!(!state.allows_token(b"yes!"));
    }

    #[test]
    fn test_initial_state_not_complete() {
        let (_g, state) = make_state(r#"root ::= "hello""#);
        assert!(!state.is_complete());
    }

    #[test]
    fn test_complete_after_full_match() {
        let (_, mut state) = make_state(r#"root ::= "hi""#);
        state.advance(b"hi").unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn test_partial_literal() {
        // "h" should be allowed as a prefix of "hi"
        let (_g, state) = make_state(r#"root ::= "hi""#);
        assert!(state.allows_token(b"h"));
        assert!(!state.allows_token(b"x"));
    }

    #[test]
    fn test_advance_stuck_returns_error() {
        let (_, mut state) = make_state(r#"root ::= "yes""#);
        let result = state.advance(b"no");
        assert!(result.is_err());
    }

    #[test]
    fn test_char_class() {
        let (_g, state) = make_state("root ::= [a-z]+");
        assert!(state.allows_token(b"hello"));
        assert!(!state.allows_token(b"Hello")); // 'H' not in [a-z]
        assert!(!state.allows_token(b"123"));
    }

    #[test]
    fn test_optional() {
        let (_g, state) = make_state(r#"root ::= "a"? "b""#);
        assert!(state.allows_token(b"ab"));
        assert!(state.allows_token(b"b"));
        assert!(!state.allows_token(b"c"));
    }

    #[test]
    fn test_apply_grammar_mask() {
        let (_, state) = make_state(r#"root ::= "yes" | "no""#);
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let vocab: Vec<(u32, Vec<u8>)> = vec![
            (0, b"maybe".to_vec()),
            (1, b"yes".to_vec()),
            (2, b"no".to_vec()),
            (3, b"nope".to_vec()),
        ];
        apply_grammar_mask(&mut logits, &state, &vocab);
        assert_eq!(logits[0], f32::NEG_INFINITY); // "maybe" not allowed
        assert!(logits[1].is_finite()); // "yes" allowed
        assert!(logits[2].is_finite()); // "no" allowed
        assert_eq!(logits[3], f32::NEG_INFINITY); // "nope" not allowed
    }

    #[test]
    fn test_empty_token_always_allowed() {
        let (_g, state) = make_state(r#"root ::= "hello""#);
        assert!(state.allows_token(b""));
    }

    #[test]
    fn test_sequence_advance() {
        let (_, mut state) = make_state(r#"root ::= "a" "b""#);
        assert!(state.allows_token(b"a"));
        state.advance(b"a").unwrap();
        assert!(state.allows_token(b"b"));
        assert!(!state.allows_token(b"a"));
    }

    // ── Rule reference advance ────────────────────────────────────────────────

    #[test]
    fn test_advance_through_rule_ref() {
        // Grammar with a rule reference: root → greeting → "hi"
        let (_, mut state) = make_state("root ::= greeting\ngreeting ::= \"hi\"");
        assert!(
            state.allows_token(b"hi"),
            "initial state should allow 'hi' via rule ref"
        );
        state
            .advance(b"hi")
            .expect("test: advancing 'hi' through rule ref should succeed");
        assert!(
            state.is_complete(),
            "state should be complete after consuming all expected bytes"
        );
    }

    #[test]
    fn test_rule_ref_allows_correct_bytes() {
        let (_g, state) = make_state("root ::= num\nnum ::= [0-9]+");
        assert!(
            state.allows_token(b"42"),
            "rule ref should allow valid bytes"
        );
        assert!(
            !state.allows_token(b"abc"),
            "rule ref should reject invalid bytes"
        );
    }

    // ── Negated char class ────────────────────────────────────────────────────

    #[test]
    fn test_advance_negated_char_class() {
        // [^0-9] matches anything except digits
        let (_, mut state) = make_state("root ::= [^0-9]");
        assert!(
            state.allows_token(b"a"),
            "non-digit should be allowed by [^0-9]"
        );
        assert!(
            !state.allows_token(b"5"),
            "digit should not be allowed by [^0-9]"
        );
        state
            .advance(b"a")
            .expect("test: advancing a non-digit should succeed");
        assert!(
            state.is_complete(),
            "should be complete after consuming one [^0-9] char"
        );
    }

    #[test]
    fn test_advance_negated_char_class_rejects_digit() {
        let (_, mut state) = make_state("root ::= [^0-9]");
        let result = state.advance(b"3");
        assert!(
            result.is_err(),
            "advancing a digit into [^0-9] should return Stuck error"
        );
    }

    // ── is_complete with optional (min=0) repeat ─────────────────────────────

    #[test]
    fn test_is_complete_on_optional_grammar() {
        // root ::= "a"? → min=0, so initial state can already be complete
        let (_g, state) = make_state(r#"root ::= "a"?"#);
        assert!(
            state.is_complete(),
            "optional grammar should be complete in initial state"
        );
    }

    #[test]
    fn test_is_complete_on_star_grammar() {
        // root ::= "a"* → min=0, complete from the start
        let (_g, state) = make_state(r#"root ::= "a"*"#);
        assert!(
            state.is_complete(),
            "star grammar should be complete in initial state"
        );
    }

    #[test]
    fn test_is_not_complete_on_plus_grammar() {
        // root ::= "a"+ → min=1, NOT complete at start
        let (_g, state) = make_state(r#"root ::= "a"+"#);
        assert!(
            !state.is_complete(),
            "plus grammar should NOT be complete in initial state"
        );
    }

    // ── Very long token conservative allow ───────────────────────────────────

    #[test]
    fn test_allows_very_long_token_conservatively() {
        // Tokens longer than MAX_SIM_BYTES (64) are always allowed conservatively
        let (_g, state) = make_state(r#"root ::= "x""#);
        let long_token: Vec<u8> = vec![b'z'; 65]; // 65 bytes, clearly doesn't match "x"
        assert!(
            state.allows_token(&long_token),
            "tokens >64 bytes should be conservatively allowed"
        );
    }

    // ── Advance on empty bytes ────────────────────────────────────────────────

    #[test]
    fn test_advance_empty_bytes_is_noop() {
        let (_, mut state) = make_state(r#"root ::= "hello""#);
        state
            .advance(b"")
            .expect("test: advancing empty bytes should succeed");
        assert!(
            !state.is_complete(),
            "state should not be complete after empty advance"
        );
        assert!(
            state.allows_token(b"hello"),
            "should still allow 'hello' after empty advance"
        );
    }

    // ── apply_grammar_mask with no vocab ────────────────────────────────────

    #[test]
    fn test_apply_grammar_mask_empty_vocab() {
        let (_, state) = make_state(r#"root ::= "abc""#);
        let mut logits = vec![1.0f32, 2.0, 3.0];
        // Empty vocab — should not change logits
        apply_grammar_mask(&mut logits, &state, &[]);
        assert_eq!(logits, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_grammar_mask_token_id_beyond_logit_len() {
        // Token IDs beyond logit length should be silently skipped
        let (_, state) = make_state(r#"root ::= "yes""#);
        let mut logits = vec![1.0f32, 2.0]; // only 2 entries
        let vocab: Vec<(u32, Vec<u8>)> = vec![
            (0, b"yes".to_vec()),
            (5, b"no".to_vec()), // id 5 is beyond logits len=2, should be skipped
        ];
        apply_grammar_mask(&mut logits, &state, &vocab);
        // logits[0] = "yes" which IS allowed → should stay finite
        assert!(logits[0].is_finite(), "allowed token should not be masked");
        assert!(logits[1].is_finite(), "untouched logit should stay finite");
    }

    // ── initial_state via Grammar::initial_state() ───────────────────────────

    #[test]
    fn test_initial_state_via_grammar_method() {
        let g = Grammar::parse(r#"root ::= "ok""#).expect("test: should parse");
        let state = g.initial_state();
        assert!(
            state.allows_token(b"ok"),
            "initial state should allow matching token"
        );
        assert!(
            !state.allows_token(b"no"),
            "initial state should reject non-matching token"
        );
    }

    // ── Multi-rule grammar with advance ──────────────────────────────────────

    #[test]
    fn test_advance_with_alternation() {
        let (_, mut state) = make_state(r#"root ::= "yes" | "no""#);
        // Advance with "yes"
        state
            .advance(b"yes")
            .expect("test: advancing 'yes' should succeed");
        assert!(
            state.is_complete(),
            "should be complete after consuming full 'yes' literal"
        );
    }

    #[test]
    fn test_advance_alternation_second_branch() {
        let (_, mut state) = make_state(r#"root ::= "yes" | "no""#);
        state
            .advance(b"no")
            .expect("test: advancing 'no' should succeed");
        assert!(
            state.is_complete(),
            "should be complete after consuming full 'no' literal"
        );
    }

    #[test]
    fn test_advance_stuck_on_char_class_mismatch() {
        // [a-z] won't accept a digit
        let (_, mut state) = make_state("root ::= [a-z]");
        let result = state.advance(b"3");
        assert!(
            result.is_err(),
            "advancing a digit into [a-z] should return Stuck error"
        );
    }
}

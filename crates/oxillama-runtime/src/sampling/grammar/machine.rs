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
    ///
    /// This is a best-effort, infallible view over [`GrammarState::allows_token_checked`]:
    /// any simulator error (recursion limit, oversized token) is treated as
    /// "not allowed" (fail-closed). Callers that need to distinguish a clean
    /// rejection from a simulator limit should use the checked variant.
    pub fn allows_token(&self, token_bytes: &[u8]) -> bool {
        self.allows_token_checked(token_bytes).unwrap_or(false)
    }

    /// Returns whether `token_bytes` is a valid continuation from the
    /// current parse state, or an error when the simulator's limits are
    /// exceeded.
    ///
    /// # Soundness (defect S8)
    ///
    /// Two limits previously **failed open** (conservatively returned
    /// `true`/"allowed") when exceeded:
    /// - Tokens longer than `MAX_SIM_BYTES` were unconditionally allowed.
    /// - Hitting `MAX_DEPTH` recursion during simulation was treated as
    ///   "allow".
    ///
    /// Both cases now return a typed error instead. [`apply_grammar_mask`]
    /// treats any such error as "not allowed" (fail-**closed**) — the
    /// opposite, safe direction — rather than silently admitting a token the
    /// simulator could not actually verify.
    pub fn allows_token_checked(&self, token_bytes: &[u8]) -> GrammarResult<bool> {
        if token_bytes.is_empty() {
            // An empty token is always allowed (it doesn't advance the parse).
            return Ok(true);
        }
        if token_bytes.len() > MAX_SIM_BYTES {
            return Err(GrammarError::TokenTooLong {
                len: token_bytes.len(),
                max: MAX_SIM_BYTES,
            });
        }
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
    fn simulate_bytes(&mut self, cont: &[ContNode], bytes: &[u8]) -> GrammarResult<bool> {
        if bytes.is_empty() {
            return Ok(true);
        }
        // Expand the first node in the continuation to get all possible
        // one-byte transitions, try each that matches bytes[0], then recurse.
        self.try_consume_byte(cont, bytes[0], &bytes[1..])
    }

    /// Attempt to consume one byte `b` from `cont`, then continue with `rest`.
    ///
    /// Depth-limit exceeded is a hard error (fail-closed — see defect S8),
    /// not the previous "conservatively allow".
    fn try_consume_byte(&mut self, cont: &[ContNode], b: u8, rest: &[u8]) -> GrammarResult<bool> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            return Err(GrammarError::RecursionLimit {
                rule: "(allows_token)".to_string(),
            });
        }

        let result = self.try_consume_byte_inner(cont, b, rest);
        self.depth -= 1;
        result
    }

    fn try_consume_byte_inner(
        &mut self,
        cont: &[ContNode],
        b: u8,
        rest: &[u8],
    ) -> GrammarResult<bool> {
        if cont.is_empty() {
            return Ok(false); // more bytes but nothing left to match
        }

        let Some((first, tail)) = cont.split_first() else {
            return Ok(false);
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
                    Ok(false)
                }
            }

            GrammarNode::CharClass { ranges, negated } => {
                let in_class = ranges.iter().any(|r| r.contains(b));
                let matches = if *negated { !in_class } else { in_class };
                if matches {
                    self.simulate_bytes(tail, rest)
                } else {
                    Ok(false)
                }
            }

            GrammarNode::RuleRef(name) => {
                let rule_node = match self.grammar.rules.get(name) {
                    Some(n) => n.clone(),
                    None => return Ok(false),
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
                // Try each alternative; succeed if any succeeds. A
                // recursion-limit error from one alternative aborts the
                // whole check immediately (fail-closed) rather than trying
                // to salvage a verdict from the remaining alternatives —
                // see the module-level soundness note on defect S8.
                for alt in alts {
                    let mut new_cont: Vec<ContNode> = Vec::with_capacity(tail.len() + 1);
                    new_cont.push(ContNode::new(alt.clone()));
                    new_cont.extend_from_slice(tail);
                    if self.try_consume_byte(&new_cont, b, rest)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            GrammarNode::Repeat { node, min, max } => {
                // Generate the set of possible unrollings. We try:
                // 1. Skip this repetition entirely (valid if min==0)
                // 2. Take one occurrence (produce node ++ Repeat{min-1..} ++ tail)
                let min = *min;
                let max = *max;

                // Option A: zero occurrences (valid when min==0)
                if min == 0 && self.try_consume_byte(tail, b, rest)? {
                    return Ok(true);
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
                    if self.try_consume_byte(&new_cont, b, rest)? {
                        return Ok(true);
                    }
                }

                Ok(false)
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
/// # Defect S1 fix
///
/// Once the grammar is **complete** (`state.is_complete()`), a non-empty
/// continuation-based grammar can never accept *any* further non-empty
/// token — `allows_token` correctly returns `false` for every one, since
/// there is nothing left in the continuation to match. Previously that
/// meant EVERY logit was masked to `-inf`, including the caller's
/// end-of-generation token(s), so `super::super::argmax`-style selection
/// degenerated forever (see the S1 write-up for the exact mechanism).
///
/// Two changes fix this:
/// 1. `eog_token_ids` are exempted from masking once the grammar is
///    complete, so the model can actually stop.
/// 2. As a last-resort safety net, if masking would still eliminate every
///    candidate (e.g. the grammar is complete but the caller didn't
///    populate `eog_token_ids`, or a pathological grammar), the single
///    highest pre-mask logit is restored rather than handing the caller a
///    fully degenerate `-inf` vector.
///
/// # Arguments
/// * `logits` - Raw logits vector; modified in-place.
/// * `state`  - Current grammar parse state.
/// * `token_vocab` - Mapping `(token_id, utf-8 bytes)` for every vocabulary entry.
/// * `eog_token_ids` - End-of-generation token IDs (e.g. `</s>`,
///   `<|im_end|>`) that should remain selectable once the grammar is
///   satisfied. Populate from `SamplerConfig::eog_token_ids`.
pub fn apply_grammar_mask(
    logits: &mut [f32],
    state: &GrammarState,
    token_vocab: &[(u32, Vec<u8>)],
    eog_token_ids: &[u32],
) {
    let complete = state.is_complete();

    // Track the single best pre-mask candidate for the last-resort fallback
    // below, without a second vocab-sized pass or allocation.
    let mut best_id: Option<usize> = None;
    let mut best_val = f32::NEG_INFINITY;

    for (token_id, token_bytes) in token_vocab {
        let id = *token_id as usize;
        if id >= logits.len() {
            continue;
        }

        if logits[id].is_finite() && (best_id.is_none() || logits[id] > best_val) {
            best_val = logits[id];
            best_id = Some(id);
        }

        let allowed = if complete && eog_token_ids.contains(token_id) {
            true
        } else {
            // Fail closed on a simulator error (see defect S8): a token we
            // could not verify is treated as disallowed, not allowed.
            state.allows_token_checked(token_bytes).unwrap_or(false)
        };

        if !allowed {
            logits[id] = f32::NEG_INFINITY;
        }
    }

    // Safety net: never return a fully-`-inf` distribution when a
    // pre-mask candidate existed (defect S1's second fix — see doc above).
    if let Some(id) = best_id {
        if logits.iter().all(|v| !v.is_finite()) {
            logits[id] = best_val;
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
        apply_grammar_mask(&mut logits, &state, &vocab, &[]);
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

    // ── Very long token: defect S8 fail-closed (was fail-open) ──────────────

    #[test]
    fn test_allows_very_long_token_fails_closed() {
        // Tokens longer than MAX_SIM_BYTES (64) used to be *conservatively
        // allowed*. Defect S8: this must now fail CLOSED (disallowed),
        // since the simulator never actually verified the token.
        let (_g, state) = make_state(r#"root ::= "x""#);
        let long_token: Vec<u8> = vec![b'z'; 65]; // 65 bytes, clearly doesn't match "x"
        assert!(
            !state.allows_token(&long_token),
            "tokens > MAX_SIM_BYTES must now be disallowed (fail-closed), not conservatively allowed"
        );
        let checked = state.allows_token_checked(&long_token);
        assert!(
            matches!(checked, Err(GrammarError::TokenTooLong { .. })),
            "expected TokenTooLong error, got {checked:?}"
        );
    }

    #[test]
    fn test_allows_token_fails_closed_on_recursion_limit() {
        // Build a grammar with far more than MAX_DEPTH (128) layers of pure
        // rule-reference indirection, so verifying a single byte requires
        // more nested calls than the simulator allows.
        let mut src = String::new();
        for i in 0..200u32 {
            src.push_str(&format!("r{i} ::= r{}\n", i + 1));
        }
        src.push_str("r200 ::= \"x\"\n");
        let g = Grammar::parse(&src).expect("deeply nested indirection grammar should parse");
        let state = g.initial_state();

        let checked = state.allows_token_checked(b"x");
        assert!(
            matches!(checked, Err(GrammarError::RecursionLimit { .. })),
            "expected RecursionLimit error, got {checked:?}"
        );
        assert!(
            !state.allows_token(b"x"),
            "defect S8: hitting the recursion limit must fail CLOSED (disallow), not fail open"
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
        apply_grammar_mask(&mut logits, &state, &[], &[]);
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
        apply_grammar_mask(&mut logits, &state, &vocab, &[]);
        // logits[0] = "yes" which IS allowed → should stay finite
        assert!(logits[0].is_finite(), "allowed token should not be masked");
        assert!(logits[1].is_finite(), "untouched logit should stay finite");
    }

    // ── Defect S1: grammar-complete EOG handling ─────────────────────────────

    #[test]
    fn test_apply_grammar_mask_allows_eog_when_complete() {
        // Once the grammar is complete, a configured EOG token must remain
        // selectable even though its literal bytes don't match the grammar.
        let (_, mut state) = make_state(r#"root ::= "hi""#);
        state.advance(b"hi").unwrap();
        assert!(state.is_complete());

        let mut logits = vec![1.0f32, 1.0, 1.0];
        let vocab: Vec<(u32, Vec<u8>)> = vec![
            (0, b"more".to_vec()),  // not EOG, doesn't match grammar -> masked
            (1, b"<eos>".to_vec()), // EOG -> must survive
            (2, b"other".to_vec()), // not EOG -> masked
        ];
        apply_grammar_mask(&mut logits, &state, &vocab, &[1]);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert!(
            logits[1].is_finite(),
            "EOG token must remain selectable once the grammar is complete"
        );
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_apply_grammar_mask_never_fully_masks_when_survivor_available() {
        // Grammar complete, but NO eog_token_ids configured. Previously this
        // masked every logit to -inf, which fed straight into argmax's
        // silent "return index 0" bug (defect S1).
        let (_, mut state) = make_state(r#"root ::= "hi""#);
        state.advance(b"hi").unwrap();
        assert!(state.is_complete());

        let mut logits = vec![3.0f32, 1.0, 2.0];
        let vocab: Vec<(u32, Vec<u8>)> =
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())];
        apply_grammar_mask(&mut logits, &state, &vocab, &[]); // no EOG configured
        let finite = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(
            finite, 1,
            "the last-resort safety net must leave exactly one survivor, got {logits:?}"
        );
        assert_eq!(
            logits[0], 3.0,
            "the highest pre-mask logit must be the surviving candidate"
        );
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

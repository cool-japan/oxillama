//! Error types for GBNF grammar-constrained sampling.

use thiserror::Error;

/// Errors that can occur during grammar parsing or state machine execution.
#[derive(Debug, Error, Clone)]
pub enum GrammarError {
    /// Syntax error during GBNF grammar parsing.
    #[error("grammar parse error at position {pos}: {msg}")]
    ParseError {
        /// Byte offset in the input where the error occurred.
        pos: usize,
        /// Human-readable description.
        msg: String,
    },

    /// Grammar state machine reached a dead state — no valid next tokens exist.
    #[error("grammar reached a stuck state — no valid next tokens")]
    Stuck,

    /// A rule reference in the grammar points to a rule that was never defined.
    #[error("unknown rule reference: '{rule}'")]
    UnknownRule {
        /// The missing rule name.
        rule: String,
    },

    /// Recursion depth limit exceeded during grammar simulation.
    #[error(
        "grammar recursion depth limit exceeded (possible infinite recursion in rule '{rule}')"
    )]
    RecursionLimit {
        /// Rule that was being evaluated.
        rule: String,
    },

    /// A candidate token exceeded the maximum number of bytes the grammar
    /// simulator will verify. Previously such tokens were *conservatively
    /// allowed* (a soundness hole — see defect S8); callers now receive this
    /// error so they can decide how to treat the token (the default,
    /// fail-closed policy used by [`super::machine::apply_grammar_mask`] is
    /// to mask it out).
    #[error("token exceeds the maximum {max} bytes verified by the grammar simulator (got {len})")]
    TokenTooLong {
        /// The token's actual byte length.
        len: usize,
        /// The simulator's configured maximum.
        max: usize,
    },

    /// A JSON Schema keyword was recognised but cannot be expressed as a
    /// GBNF (regular) grammar, so it was rejected rather than silently
    /// ignored (see defect S7).
    #[error("JSON Schema keyword `{keyword}` cannot be expressed as a GBNF grammar: {reason}")]
    UnsupportedKeyword {
        /// The offending keyword (e.g. `"minimum"`, `"anyOf"`).
        keyword: String,
        /// Human-readable explanation of why it can't be compiled.
        reason: String,
    },
}

/// Convenience alias.
pub type GrammarResult<T> = Result<T, GrammarError>;

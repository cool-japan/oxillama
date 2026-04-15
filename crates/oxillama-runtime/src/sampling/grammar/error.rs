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
}

/// Convenience alias.
pub type GrammarResult<T> = Result<T, GrammarError>;

//! GBNF grammar-constrained sampling.
//!
//! This module provides:
//! - [`Grammar`] — a parsed GBNF grammar.
//! - [`GrammarState`] — stateful parse position, advanced token-by-token.
//! - [`GrammarError`] — error type for parse and state-machine failures.
//! - [`apply_grammar_mask`] — zero out logits for tokens disallowed by the
//!   current grammar state.

pub mod error;
pub mod machine;
pub mod parser;

pub use error::{GrammarError, GrammarResult};
pub use machine::{apply_grammar_mask, GrammarState};
pub use parser::{CharRange, Grammar, GrammarNode};

impl Grammar {
    /// Create the initial parse state for this grammar.
    pub fn initial_state(&self) -> GrammarState {
        GrammarState::new(self.clone())
    }
}

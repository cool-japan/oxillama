//! Token embedding table — re-exported from [`crate::common::embedding`].
//!
//! This module used to hold a byte-for-byte duplicate of
//! `crate::llama::embedding` (see that crate's history): both copied the same
//! `TokenEmbedding` enum because the `llama` and `qwen3` feature-gated
//! modules cannot reference each other directly.  The shared implementation
//! now lives in `crate::common::embedding`; this module re-exports it so
//! `crate::qwen3::embedding::TokenEmbedding` and `crate::qwen3::TokenEmbedding`
//! keep resolving exactly as before for every existing caller.

pub use crate::common::embedding::TokenEmbedding;

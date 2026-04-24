// Compile-check every `rust,no_run` fenced block in RECIPES.md.
//
// rustdoc extracts `rust` fenced blocks and compiles them; because every
// block in RECIPES.md is marked `rust,no_run`, they are compiled but never
// executed.  This gives us "does this snippet at least parse and type-check"
// coverage without requiring real GGUF files.
//
// The `js,no_run` block for Recipe 6 (browser/JavaScript) is transparently
// ignored by rustdoc since it is not a `rust` fence.

/// OxiLLaMa recipe compilation checks.
///
#[doc = include_str!("../RECIPES.md")]
pub struct RecipesDoc;

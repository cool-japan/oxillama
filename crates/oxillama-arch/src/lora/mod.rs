//! LoRA adapter loading, stacking, and trait-based composition.
//!
//! ## Submodule layout
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`adapter`] | [`TargetModule`], [`LoraDelta`], [`LoraAdapterTrait`] — stable public API |
//! | [`stack`]   | [`LoraStack`] — ordered composition of multiple LoRA adapters |
//! | [`loader`]  | [`LoadedLora`] — GGUF-backed LoRA adapter loading |

pub mod adapter;
pub mod loader;
pub mod stack;

pub use adapter::{LoraAdapterTrait, LoraDelta, TargetModule};
pub use loader::LoadedLora;
pub use stack::LoraStack;

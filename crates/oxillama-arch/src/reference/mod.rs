//! Full-precision f32 reference path for CI numeric diff testing.
//!
//! Enabled with the `reference-f32` feature. All tensors are dequantized to
//! f32 at load time, trading memory (~4× vs Q4_0) for a simple reference
//! implementation that produces deterministic, finite outputs.
//!
//! **Not for production use.** Use the quantized forward paths instead.
//!
//! # Usage
//!
//! ```rust,ignore
//! #[cfg(feature = "reference-f32")]
//! {
//!     use oxillama_arch::reference;
//!     use oxillama_gguf::GgufModel;
//!
//!     let bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
//!     let model = GgufModel::from_bytes(bytes).unwrap();
//!     let fwd = reference::load_as_reference(&model, 32, 512, 32).unwrap();
//!     // fwd implements ForwardPass
//! }
//! ```

#[cfg(feature = "reference-f32")]
pub mod forward;
#[cfg(feature = "reference-f32")]
pub mod loader;

#[cfg(feature = "reference-f32")]
pub use forward::ReferenceModel;
#[cfg(feature = "reference-f32")]
pub use loader::{ReferenceLoader, ReferenceWeights};

/// Load a GGUF model in full-precision f32 mode for CI numeric diff testing.
///
/// Eagerly dequantizes all tensors. **WARNING:** ~4× memory vs quantized.
/// For CI use only.
///
/// # Errors
///
/// Returns [`ArchError`](crate::error::ArchError) if any tensor cannot be dequantized.
#[cfg(feature = "reference-f32")]
pub fn load_as_reference(
    model: &oxillama_gguf::GgufModel,
    vocab_size: usize,
    max_context_length: usize,
    hidden_size: usize,
) -> crate::error::ArchResult<Box<dyn crate::traits::ForwardPass>> {
    let weights = loader::ReferenceLoader::dequantize_all(model)?;
    Ok(Box::new(forward::ReferenceModel::new(
        weights,
        vocab_size,
        max_context_length,
        hidden_size,
    )))
}

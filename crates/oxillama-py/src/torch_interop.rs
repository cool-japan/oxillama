//! Torch interop registration hook.
//!
//! The actual bridging logic lives in the Python layer at
//! `python/oxillama_py/torch_helper.py`.  This file exists only to keep
//! the module structure explicit and to provide a registration entry-point
//! that can be extended if Rust-level helpers are ever needed.
//!
//! The bridge is purely Python-side:
//! 1. `torch_helper.py` monkey-patches `Engine.logits_torch` and
//!    `Engine.embeddings_torch` onto the class at import time.
//! 2. Both methods call the already-shipped `logits_dlpack()` /
//!    `embeddings_dlpack()` Rust methods and convert the `PyCapsule` to a
//!    `torch.Tensor` via `torch.from_dlpack(capsule)`.
//! 3. Torch is imported lazily inside each method so that the absence of
//!    PyTorch does not prevent the rest of the package from loading.

use pyo3::prelude::*;

/// Register torch interop helpers in the Python module.
///
/// Currently a no-op at the Rust level — the monkey-patching is performed
/// in `__init__.py` via `torch_helper.try_patch(...)`.
pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    /// `register` must not panic or return an error when called without a
    /// real Python interpreter in unit-test context.  Because PyO3 requires
    /// a live interpreter we just confirm the function exists and is callable
    /// at the Rust type level via the function pointer.
    #[test]
    fn register_fn_exists() {
        // Verify the symbol is reachable; calling it requires a live Python
        // interpreter which is not available in pure-Rust unit tests.
        let _ =
            super::register as fn(&pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()>;
    }
}

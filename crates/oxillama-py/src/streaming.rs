//! Python-callable streaming bridge utilities.
//!
//! This module provides the `StreamingCallback` helper that wraps a Python
//! callable and safely invokes it from within a `py.detach(...)` scope
//! by re-acquiring the GIL on each token.
//!
//! The bridge is generic: any Rust `FnMut(&str)` closure that needs to call
//! back into Python can wrap a `Py<PyAny>` with this helper.

use pyo3::prelude::*;
use pyo3::types::PyAny;

/// A bridge that wraps a Python callable and implements `FnMut(&str)`.
///
/// Use this to pass a Python callback into a Rust inference call that operates
/// with the GIL released:
///
/// ```rust,ignore
/// let mut bridge = StreamingCallbackBridge::new(py_callback);
/// py.detach(|| {
///     engine.generate(prompt, max_tokens, |tok| bridge.call(tok))
/// })
/// ```
pub struct StreamingCallbackBridge {
    callback: Py<PyAny>,
}

impl StreamingCallbackBridge {
    /// Wrap a Python callable.
    pub fn new(callback: Py<PyAny>) -> Self {
        Self { callback }
    }

    /// Re-acquire the GIL and invoke the Python callable with a token string.
    ///
    /// Errors from the Python callable are silently swallowed so that a
    /// Python-side exception does not abort the entire generation loop.
    /// The caller should check for Python exceptions after generation
    /// completes if strict error propagation is required.
    pub fn call(&self, token: &str) {
        Python::attach(|py| {
            let _ = self.callback.call1(py, (token,));
        });
    }
}

/// Create a `FnMut(&str)` closure that re-acquires the GIL on each call.
///
/// Returns `None` if `callback` is `None`, yielding a no-op closure in that
/// case.  This avoids boilerplate `if let Some(cb) = …` in callers.
pub fn make_callback(callback: Option<Py<PyAny>>) -> impl FnMut(&str) {
    move |tok: &str| {
        if let Some(ref cb) = callback {
            Python::attach(|py| {
                let _ = cb.call1(py, (tok,));
            });
        }
    }
}

#[cfg(test)]
mod tests {
    // Unit tests for the streaming module are limited because creating real
    // Python callables requires an embedded interpreter.  The compile-time
    // tests below ensure the API is sound.

    use super::*;

    /// `make_callback` with `None` must compile and be callable without panic.
    #[test]
    fn test_make_callback_none_is_noop() {
        let mut cb = make_callback(None);
        // Calling with None must not panic
        cb("hello");
        cb("world");
    }
}

//! Python-accessible `CancellationToken` for cooperative cancellation of generation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pyo3::prelude::*;

/// A thread-safe cancellation token that can be shared across Python threads.
///
/// Pass an instance to `Engine.generate()` or `Engine.generate_streaming()` to
/// allow cooperative cancellation from another thread.
///
/// # Python example
///
/// ```python
/// import threading, oxillama_py
///
/// token = oxillama_py.CancellationToken()
///
/// def cancel_after_1s():
///     import time; time.sleep(1); token.cancel()
///
/// threading.Thread(target=cancel_after_1s, daemon=True).start()
/// engine.generate("Hello", max_tokens=2048, cancel_token=token)
/// ```
#[pyclass(name = "CancellationToken", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCancellationToken {
    pub(crate) cancelled: Arc<AtomicBool>,
}

#[pymethods]
impl PyCancellationToken {
    /// Create a new, non-cancelled token.
    #[new]
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal cancellation.  Calling this from any thread will cause any ongoing
    /// generation that holds this token to stop at the next token boundary.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Return `True` if `cancel()` has been called.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Reset the token so it can be reused for a subsequent generation call.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Relaxed);
    }

    fn __repr__(&self) -> String {
        format!(
            "CancellationToken(cancelled={})",
            self.cancelled.load(Ordering::Relaxed)
        )
    }
}

impl Default for PyCancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state_not_cancelled() {
        let t = PyCancellationToken::new();
        assert!(!t.is_cancelled());
    }

    #[test]
    fn test_cancel_sets_flag() {
        let t = PyCancellationToken::new();
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn test_reset_clears_flag() {
        let t = PyCancellationToken::new();
        t.cancel();
        t.reset();
        assert!(!t.is_cancelled());
    }

    #[test]
    fn test_clone_shares_state() {
        let t = PyCancellationToken::new();
        let t2 = t.clone();
        t.cancel();
        assert!(t2.is_cancelled());
    }
}

//! Python-callable streaming bridge utilities.
//!
//! This module provides:
//!
//! * [`StreamingCallbackBridge`] — wraps a Python callable for use as a
//!   token-by-token streaming sink (legacy v0.1.x API surface).
//! * [`make_callback`] — convenience helper that returns a `FnMut(&str)` no-op
//!   when the supplied Python callable is `None`.
//! * [`Throttler`] — pure-Rust throttle helper used by the rich progress hook
//!   to cap callback invocations at ~50 ms or 4 tokens (whichever first), with
//!   first/final tokens always firing.
//! * [`ProgressBridge`] — RAII guard that owns a `(callback, finaliser)`
//!   pair built by `oxillama_py.progress._build_bridge`.  The bridge throttles
//!   per-token invocations on the Rust side and guarantees the finaliser runs
//!   exactly once even in the face of Python exceptions, cancellation, EOS,
//!   or unwinding from a panic in user code.
//! * [`make_progress_bridge`] — factory that translates an `Option<Py<PyAny>>`
//!   from the Python kwarg into an `Option<ProgressBridge>` by delegating
//!   duck-type dispatch to `oxillama_py.progress._build_bridge`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyTuple};

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

// ---------------------------------------------------------------------------
// Throttler — pure Rust, used by the rich progress hook.
// ---------------------------------------------------------------------------

/// Default minimum interval between throttled callback invocations (50 ms).
pub const DEFAULT_THROTTLE_MS: u64 = 50;
/// Default minimum token count between throttled callback invocations (4).
pub const DEFAULT_THROTTLE_TOKENS: usize = 4;

/// Cap callback firing frequency by elapsed wall time *or* token count,
/// whichever crosses its threshold first.
///
/// `should_fire(force=true)` always fires (used for the very first token
/// and for the synthesised final-token tick).  Otherwise, the call returns
/// `true` iff at least `min_interval` has elapsed since the last fire OR at
/// least `min_tokens` have been seen since the last fire.
///
/// On a positive return, the internal counters are reset.
pub struct Throttler {
    last_fire: Instant,
    tokens_since_fire: usize,
    min_interval: Duration,
    min_tokens: usize,
}

impl Throttler {
    /// Build a new throttle gate with the given interval (in milliseconds)
    /// and per-fire token budget.
    pub fn new(min_interval_ms: u64, min_tokens: usize) -> Self {
        Self {
            // Seed `last_fire` to `now` so the very first non-forced
            // `should_fire` call does not trip on `min_interval` simply because
            // the wall clock has been running before the bridge was built.
            last_fire: Instant::now(),
            tokens_since_fire: 0,
            min_interval: Duration::from_millis(min_interval_ms),
            min_tokens,
        }
    }

    /// Record that one token has been observed (independent of whether the
    /// resulting `should_fire(false)` will return `true`).
    pub fn note_token(&mut self) {
        self.tokens_since_fire = self.tokens_since_fire.saturating_add(1);
    }

    /// Return whether the caller should fire the throttled callback now.
    ///
    /// Resets internal counters on a `true` return.
    pub fn should_fire(&mut self, force: bool) -> bool {
        let elapsed_ok = self.last_fire.elapsed() >= self.min_interval;
        let tokens_ok = self.tokens_since_fire >= self.min_tokens;
        let fire = force || elapsed_ok || tokens_ok;
        if fire {
            self.last_fire = Instant::now();
            self.tokens_since_fire = 0;
        }
        fire
    }
}

// ---------------------------------------------------------------------------
// ProgressBridge — RAII wrapper around the (callback, finaliser) pair.
// ---------------------------------------------------------------------------

/// Owns the throttle state and the `(callback, finaliser)` pair returned by
/// `oxillama_py.progress._build_bridge`.
///
/// On every token the engine emits, [`note_token`](Self::note_token) is
/// invoked.  The first token always fires the callback, intermediate tokens
/// fire only when the [`Throttler`] permits, and the final token (signalled by
/// `is_final=true`) always fires.
///
/// [`finalise`](Self::finalise) is the explicit cleanup entry point.  The
/// `Drop` impl is a safety net that runs `finalise(py, None)` if the explicit
/// path was somehow skipped — for example when a panic unwinds out of a
/// callback.  All errors raised inside the finaliser are swallowed so that
/// `Drop` cannot itself abort the process.
pub struct ProgressBridge {
    callback: Py<PyAny>,
    finaliser: Py<PyAny>,
    throttle: Throttler,
    start: Instant,
    tokens_total: AtomicUsize,
    capture_text: bool,
    accumulated_text: String,
    finalised: AtomicBool,
    stashed_error: Option<PyErr>,
}

impl ProgressBridge {
    /// Number of tokens passed to [`note_token`] so far (across all calls).
    pub fn tokens_total(&self) -> usize {
        self.tokens_total.load(Ordering::Relaxed)
    }

    /// Take the most recent stashed Python error (used by `strict_progress`).
    pub fn take_stashed_error(&mut self) -> Option<PyErr> {
        self.stashed_error.take()
    }

    /// Note a token and, if the throttle permits, fire the callback.
    ///
    /// `is_final` forces the callback to fire (used for the synthesised
    /// "generation finished" tick that runs after the last decoded token).
    /// `strict` controls whether errors raised inside the Python callback are
    /// propagated immediately (`true`) or stashed for later reporting (`false`).
    pub fn note_token(
        &mut self,
        py: Python<'_>,
        token: &str,
        is_final: bool,
        strict: bool,
    ) -> PyResult<()> {
        let prev = self.tokens_total.fetch_add(1, Ordering::Relaxed);
        let tokens_now = prev + 1;

        if self.capture_text {
            // Amortised O(1) growth — never a temporary `format!` allocation.
            self.accumulated_text.push_str(token);
        }

        // Always fire on the very first token and on the synthesised final
        // tick.  Intermediate ticks defer to the throttler.
        let force = is_final || tokens_now == 1;
        self.throttle.note_token();
        if !self.throttle.should_fire(force) {
            return Ok(());
        }

        let elapsed = self.start.elapsed().as_secs_f64();
        // Use a `&str` (or `""`) for `text_so_far` to avoid an extra clone if
        // the user did not request text capture.
        let text_view: &str = if self.capture_text {
            self.accumulated_text.as_str()
        } else {
            ""
        };

        // Build the (tokens, elapsed_secs, is_final, text_so_far) 4-tuple
        // that the Python wrapper turns into a `ProgressEvent`.
        let payload = PyTuple::new(py, [tokens_now.into_pyobject(py)?.into_any(),
            elapsed.into_pyobject(py)?.into_any(),
            is_final.into_pyobject(py)?.to_owned().into_any(),
            text_view.into_pyobject(py)?.into_any()])?;

        match self.callback.call1(py, (payload,)) {
            Ok(_) => Ok(()),
            Err(err) => {
                if strict {
                    Err(err)
                } else {
                    // Stash only the first error to avoid clobbering the
                    // earliest cause when the callback misbehaves repeatedly.
                    if self.stashed_error.is_none() {
                        self.stashed_error = Some(err);
                    }
                    Ok(())
                }
            }
        }
    }

    /// Force-fire the callback one last time with `is_final=true` and the
    /// current token count (without incrementing it).
    ///
    /// This is the synthesised "generation finished" event the docs promise
    /// will always fire after the last decoded token.  Errors raised by the
    /// callback are silently dropped here — the explicit `finalise` epilogue
    /// is the cleanup path, and `strict_progress` will already have stashed
    /// any earlier per-token errors.
    pub fn fire_final(&mut self, py: Python<'_>) {
        let tokens_now = self.tokens_total.load(Ordering::Relaxed);
        let elapsed = self.start.elapsed().as_secs_f64();
        let text_view: &str = if self.capture_text {
            self.accumulated_text.as_str()
        } else {
            ""
        };
        let payload = match (
            tokens_now.into_pyobject(py),
            elapsed.into_pyobject(py),
            true.into_pyobject(py),
            text_view.into_pyobject(py),
        ) {
            (Ok(t), Ok(e), Ok(f), Ok(s)) => {
                let owned_f = f.to_owned();
                PyTuple::new(
                    py,
                    [t.into_any(), e.into_any(), owned_f.into_any(), s.into_any()],
                )
            }
            _ => return,
        };
        let payload = match payload {
            Ok(p) => p,
            Err(_) => return,
        };
        let _ = self.callback.call1(py, (payload,));
    }

    /// Run the finaliser exactly once.
    ///
    /// All errors raised inside the finaliser are swallowed — the finaliser is
    /// the cleanup path and must be safe to call from `Drop` as well as from
    /// the explicit success/error epilogue.
    pub fn finalise(&mut self, py: Python<'_>, error: Option<&PyErr>) {
        if self.finalised.swap(true, Ordering::Relaxed) {
            return;
        }
        let arg: Py<PyAny> = match error {
            Some(err) => err.clone_ref(py).into_value(py).into_any(),
            None => py.None(),
        };
        // Swallow finaliser errors — never abort the process from cleanup.
        let _ = self.finaliser.call1(py, (arg,));
    }
}

impl Drop for ProgressBridge {
    fn drop(&mut self) {
        if self.finalised.load(Ordering::Relaxed) {
            return;
        }
        // `Python::attach` is reentrant within the same OS thread, so this is
        // safe whether we are unwinding from a panic in user code or simply
        // dropping the bridge after the explicit `finalise` call.
        Python::attach(|py| {
            self.finalise(py, None);
        });
    }
}

/// Build a [`ProgressBridge`] from a Python `progress=` kwarg value.
///
/// Returns `Ok(None)` when `progress` is `None` (no progress hook requested).
/// Otherwise calls `oxillama_py.progress._build_bridge(progress, max_tokens)`
/// to obtain the `(callback, finaliser)` pair and constructs the bridge.
pub fn make_progress_bridge(
    py: Python<'_>,
    progress: Option<&Py<PyAny>>,
    max_tokens: usize,
    throttle_ms: u64,
    throttle_tokens: usize,
    capture_text: bool,
) -> PyResult<Option<ProgressBridge>> {
    let progress = match progress {
        Some(obj) => obj,
        None => return Ok(None),
    };
    let module = PyModule::import(py, "oxillama_py.progress")?;
    let builder = module.getattr("_build_bridge")?;
    let pair = builder.call1((progress.bind(py), max_tokens))?;
    let tuple: Bound<'_, PyTuple> = pair.cast_into::<PyTuple>().map_err(|e| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "_build_bridge must return a (callback, finaliser) tuple: {e}"
        ))
    })?;
    if tuple.len() != 2 {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "_build_bridge must return a 2-tuple (callback, finaliser)",
        ));
    }
    let callback: Py<PyAny> = tuple.get_item(0)?.unbind();
    let finaliser: Py<PyAny> = tuple.get_item(1)?.unbind();

    Ok(Some(ProgressBridge {
        callback,
        finaliser,
        throttle: Throttler::new(throttle_ms, throttle_tokens),
        start: Instant::now(),
        tokens_total: AtomicUsize::new(0),
        capture_text,
        accumulated_text: String::new(),
        finalised: AtomicBool::new(false),
        stashed_error: None,
    }))
}

#[cfg(test)]
mod tests {
    // Unit tests for the streaming module are limited because creating real
    // Python callables requires an embedded interpreter.  The compile-time
    // tests below ensure the API is sound.

    use super::*;
    use std::thread::sleep;

    /// `make_callback` with `None` must compile and be callable without panic.
    #[test]
    fn test_make_callback_none_is_noop() {
        let mut cb = make_callback(None);
        // Calling with None must not panic
        cb("hello");
        cb("world");
    }

    /// Throttler must always fire when `force=true` is passed.
    #[test]
    fn test_throttler_fires_on_first_token() {
        // Long interval and high token threshold so neither condition trips.
        let mut t = Throttler::new(60_000, 999);
        assert!(t.should_fire(true), "force=true must always fire");
    }

    /// After firing, subsequent non-forced calls below both thresholds are
    /// throttled.
    #[test]
    fn test_throttler_throttles_subsequent_calls() {
        let mut t = Throttler::new(60_000, 999);
        assert!(t.should_fire(true));
        // Add a couple of tokens — well under the 999 threshold.
        t.note_token();
        t.note_token();
        assert!(
            !t.should_fire(false),
            "throttler should not fire while both gates are closed"
        );
    }

    /// Crossing the per-fire token threshold opens the gate.
    #[test]
    fn test_throttler_fires_on_token_threshold() {
        // Long interval (so only the token gate can trip) and a 4-token budget.
        let mut t = Throttler::new(60_000, 4);
        // First fire to reset the throttler to a known state.
        assert!(t.should_fire(true));
        for _ in 0..3 {
            t.note_token();
            assert!(
                !t.should_fire(false),
                "should not fire before crossing the 4-token threshold"
            );
        }
        t.note_token();
        assert!(
            t.should_fire(false),
            "should fire once the 4-token threshold is reached"
        );
    }

    /// Crossing the elapsed-time threshold opens the gate even with zero
    /// tokens accumulated.
    #[test]
    fn test_throttler_fires_on_interval() {
        let mut t = Throttler::new(20, 999);
        assert!(t.should_fire(true));
        sleep(Duration::from_millis(35));
        assert!(
            t.should_fire(false),
            "should fire once the 20 ms interval has elapsed"
        );
    }

    /// `should_fire(true)` resets both counters.
    #[test]
    fn test_throttler_force_resets_counters() {
        let mut t = Throttler::new(60_000, 4);
        for _ in 0..3 {
            t.note_token();
        }
        assert!(t.should_fire(true), "force fire");
        // After reset, three more tokens are still below the 4-token threshold.
        for _ in 0..3 {
            t.note_token();
            assert!(
                !t.should_fire(false),
                "counters were not reset by the force fire"
            );
        }
    }
}

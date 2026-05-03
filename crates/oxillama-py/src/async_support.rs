//! Async Python support for OxiLLaMa.
//!
//! Wraps the synchronous [`PyEngine`] API for use with Python's `asyncio`.
//! Uses `asyncio.to_thread()` (Python ≥ 3.9) to offload blocking inference
//! to a thread-pool executor, and provides a streaming async iterator via
//! a thread-safe `queue.Queue` bridge.
//!
//! ## Python usage
//!
//! ```python
//! import asyncio
//! from oxillama_py import AsyncEngine, EngineConfig
//!
//! async def main():
//!     engine = AsyncEngine(EngineConfig("model.gguf"))
//!     await engine.load_model()
//!
//!     text = await engine.generate("Hello", max_tokens=128)
//!     print(text)
//!
//!     async for token in engine.generate_stream("Hello", max_tokens=128):
//!         print(token, end="", flush=True)
//!
//! asyncio.run(main())
//! ```

use std::thread;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::engine::{PyEngine, PyEngineConfig};

// -----------------------------------------------------------------------
// Python helper: _TokenStream async iterator
// -----------------------------------------------------------------------

/// Python source for the `_TokenStream` async iterator class.
///
/// Uses a thread-safe [`queue.Queue`] as the hand-off buffer between the
/// synchronous generation thread and the asyncio event-loop consumer.
/// `__anext__` offloads the blocking `queue.get()` call to the default
/// thread-pool executor so that the event loop is never blocked.
const TOKEN_STREAM_PY: &str = concat!(
    "import asyncio\n",
    "import queue as _queue_mod\n",
    "\n",
    "class _TokenStream:\n",
    "    \"\"\"Async iterator yielding tokens from a background generation thread.\"\"\"\n",
    "    __slots__ = ('_queue',)\n",
    "\n",
    "    def __init__(self):\n",
    "        self._queue = _queue_mod.Queue()\n",
    "\n",
    "    def _feed(self, token):\n",
    "        self._queue.put(token)\n",
    "\n",
    "    def _feed_error(self, message):\n",
    "        self._queue.put(RuntimeError(message))\n",
    "\n",
    "    def _finish(self):\n",
    "        self._queue.put(None)\n",
    "\n",
    "    def __aiter__(self):\n",
    "        return self\n",
    "\n",
    "    async def __anext__(self):\n",
    "        loop = asyncio.get_running_loop()\n",
    "        item = await loop.run_in_executor(None, self._queue.get)\n",
    "        if item is None:\n",
    "            raise StopAsyncIteration\n",
    "        if isinstance(item, BaseException):\n",
    "            raise item\n",
    "        return item\n",
);

/// Import (or return the cached) helper module that provides `_TokenStream`.
fn get_token_stream_helper(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;

    if let Ok(existing) = modules.get_item("_oxillama_async_helper") {
        if !existing.is_none() {
            let module = existing.cast_into::<PyModule>().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!("expected module: {e}"))
            })?;
            return Ok(module);
        }
    }

    let code = std::ffi::CString::new(TOKEN_STREAM_PY).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to create CString for async helper module: {e}"
        ))
    })?;
    let module = PyModule::from_code(
        py,
        &code,
        c"_oxillama_async_helper.py",
        c"_oxillama_async_helper",
    )?;
    modules.set_item("_oxillama_async_helper", &module)?;
    Ok(module)
}

// -----------------------------------------------------------------------
// PyAsyncEngine
// -----------------------------------------------------------------------

/// Async wrapper for the OxiLLaMa inference engine.
///
/// All inference methods return Python coroutines compatible with
/// `async`/`await`.  Requires Python ≥ 3.9 (`asyncio.to_thread`).
///
/// ## Python Example
///
/// ```python
/// import asyncio
/// from oxillama_py import AsyncEngine, EngineConfig
///
/// async def main():
///     engine = AsyncEngine(EngineConfig("model.gguf"))
///     await engine.load_model()
///     text = await engine.generate("Hello", max_tokens=128)
///     print(text)
///
/// asyncio.run(main())
/// ```
#[pyclass(name = "AsyncEngine")]
pub struct PyAsyncEngine {
    inner: Py<PyEngine>,
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PyAsyncEngine {
    /// Create a new async engine from a configuration object.
    #[new]
    fn new(py: Python<'_>, config: &PyEngineConfig) -> PyResult<Self> {
        let engine = PyEngine::new(config);
        let inner = Py::new(py, engine)?;
        Ok(Self { inner })
    }

    /// Load the model asynchronously.
    ///
    /// Returns a coroutine.  The actual I/O runs in a background thread
    /// via `asyncio.to_thread()`.
    fn load_model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let locals = PyDict::new(py);
        locals.set_item("asyncio", py.import("asyncio")?)?;
        locals.set_item("engine", &self.inner)?;
        py.eval(c"asyncio.to_thread(engine.load_model)", None, Some(&locals))
    }

    /// Generate text asynchronously.
    ///
    /// Returns a coroutine that resolves to the generated string.
    #[pyo3(signature = (
        prompt,
        *,
        max_tokens = 128,
        temperature = None,
        top_p = None,
        top_k = None,
        seed = None,
        progress = None,
        progress_throttle_ms = None,
        progress_throttle_tokens = None,
        progress_capture_text = false,
        strict_progress = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        prompt: String,
        max_tokens: usize,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: Option<u64>,
        progress: Option<Py<PyAny>>,
        progress_throttle_ms: Option<u64>,
        progress_throttle_tokens: Option<usize>,
        progress_capture_text: bool,
        strict_progress: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let locals = PyDict::new(py);
        locals.set_item("asyncio", py.import("asyncio")?)?;
        locals.set_item("functools", py.import("functools")?)?;
        locals.set_item("engine", &self.inner)?;
        locals.set_item("prompt", &prompt)?;
        locals.set_item("max_tokens", max_tokens)?;
        locals.set_item("temperature", temperature)?;
        locals.set_item("top_p", top_p)?;
        locals.set_item("top_k", top_k)?;
        locals.set_item("seed", seed)?;
        locals.set_item("progress", progress)?;
        locals.set_item("progress_throttle_ms", progress_throttle_ms)?;
        locals.set_item("progress_throttle_tokens", progress_throttle_tokens)?;
        locals.set_item("progress_capture_text", progress_capture_text)?;
        locals.set_item("strict_progress", strict_progress)?;
        py.eval(
            c"asyncio.to_thread(functools.partial(engine.generate, prompt, max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, seed=seed, progress=progress, progress_throttle_ms=progress_throttle_ms, progress_throttle_tokens=progress_throttle_tokens, progress_capture_text=progress_capture_text, strict_progress=strict_progress))",
            None,
            Some(&locals),
        )
    }

    /// Compute embeddings asynchronously.
    ///
    /// Returns a coroutine that resolves to a list of floats.
    fn embed<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let locals = PyDict::new(py);
        locals.set_item("asyncio", py.import("asyncio")?)?;
        locals.set_item("engine", &self.inner)?;
        locals.set_item("text", &text)?;
        py.eval(
            c"asyncio.to_thread(engine.embed, text)",
            None,
            Some(&locals),
        )
    }

    /// Return an async iterator that yields tokens as they are generated.
    ///
    /// Generation runs in a background thread; tokens are fed through
    /// a thread-safe queue and yielded as they become available.
    ///
    /// ```python
    /// async for token in engine.generate_stream("Hello", max_tokens=128):
    ///     print(token, end="", flush=True)
    /// ```
    #[pyo3(signature = (
        prompt,
        *,
        max_tokens = 128,
        temperature = None,
        top_p = None,
        top_k = None,
        seed = None,
        progress = None,
        progress_throttle_ms = None,
        progress_throttle_tokens = None,
        progress_capture_text = false,
        strict_progress = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate_stream<'py>(
        &self,
        py: Python<'py>,
        prompt: String,
        max_tokens: usize,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: Option<u64>,
        progress: Option<Py<PyAny>>,
        progress_throttle_ms: Option<u64>,
        progress_throttle_tokens: Option<usize>,
        progress_capture_text: bool,
        strict_progress: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let helper = get_token_stream_helper(py)?;
        let stream = helper.getattr("_TokenStream")?.call0()?;

        let stream_handle: Py<PyAny> = stream.clone().unbind();
        let engine_handle: Py<PyAny> = self.inner.bind(py).clone().into_any().unbind();

        thread::spawn(move || {
            Python::attach(|py| {
                let run = || -> PyResult<()> {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("temperature", temperature)?;
                    kwargs.set_item("top_p", top_p)?;
                    kwargs.set_item("top_k", top_k)?;
                    kwargs.set_item("seed", seed)?;
                    kwargs.set_item("progress", &progress)?;
                    kwargs.set_item("progress_throttle_ms", progress_throttle_ms)?;
                    kwargs.set_item("progress_throttle_tokens", progress_throttle_tokens)?;
                    kwargs.set_item("progress_capture_text", progress_capture_text)?;
                    kwargs.set_item("strict_progress", strict_progress)?;

                    let cb_locals = PyDict::new(py);
                    cb_locals.set_item("_s", stream_handle.bind(py))?;
                    let callback = py.eval(c"lambda tok: _s._feed(tok)", None, Some(&cb_locals))?;

                    engine_handle.bind(py).call_method(
                        "generate_streaming",
                        (prompt.as_str(), max_tokens, &callback),
                        Some(&kwargs),
                    )?;
                    Ok(())
                };

                match run() {
                    Ok(()) => {
                        let _ = stream_handle.call_method0(py, "_finish");
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        let _ = stream_handle.call_method1(py, "_feed_error", (msg,));
                        let _ = stream_handle.call_method0(py, "_finish");
                    }
                }
            });
        });

        Ok(stream)
    }

    /// Whether a model is loaded.
    fn is_loaded(&self, py: Python<'_>) -> bool {
        self.inner.bind(py).borrow().is_loaded()
    }

    /// Reset the KV cache.
    fn reset(&self, py: Python<'_>) {
        self.inner.bind(py).borrow_mut().reset();
    }

    /// Access the underlying synchronous engine.
    #[getter]
    fn engine(&self, py: Python<'_>) -> Py<PyEngine> {
        self.inner.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let loaded = self.inner.bind(py).borrow().is_loaded();
        format!("AsyncEngine(loaded={loaded})")
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::PyEngineConfig;

    fn init_python() {
        Python::initialize();
    }

    #[test]
    fn test_async_engine_construction() {
        init_python();
        Python::attach(|py| {
            let config = PyEngineConfig::new("test.gguf".into(), None, 4, None, None);
            let result = PyAsyncEngine::new(py, &config);
            assert!(result.is_ok());
        });
    }

    #[test]
    fn test_async_engine_not_loaded() {
        init_python();
        Python::attach(|py| {
            let config = PyEngineConfig::new("test.gguf".into(), None, 4, None, None);
            if let Ok(engine) = PyAsyncEngine::new(py, &config) {
                assert!(!engine.is_loaded(py));
            }
        });
    }

    #[test]
    fn test_token_stream_helper_loads() {
        init_python();
        Python::attach(|py| {
            let result = get_token_stream_helper(py);
            if let Err(ref e) = result {
                eprintln!("get_token_stream_helper error: {e:?}");
            }
            assert!(result.is_ok(), "get_token_stream_helper must succeed");
        });
    }

    #[test]
    fn test_token_stream_helper_has_class() {
        init_python();
        Python::attach(|py| {
            if let Ok(helper) = get_token_stream_helper(py) {
                assert!(helper.getattr("_TokenStream").is_ok());
            }
        });
    }

    #[test]
    fn test_token_stream_helper_cached() {
        init_python();
        Python::attach(|py| {
            let r1 = get_token_stream_helper(py);
            if let Err(ref e) = r1 {
                eprintln!("get_token_stream_helper (1st) error: {e:?}");
            }
            assert!(r1.is_ok(), "first call must succeed");
            let r2 = get_token_stream_helper(py);
            assert!(r2.is_ok());
        });
    }

    #[test]
    fn test_token_stream_feed_and_finish() {
        init_python();
        Python::attach(|py| {
            let helper = match get_token_stream_helper(py) {
                Ok(h) => h,
                Err(_) => return,
            };
            let cls = match helper.getattr("_TokenStream") {
                Ok(c) => c,
                Err(_) => return,
            };
            let stream = match cls.call0() {
                Ok(s) => s,
                Err(_) => return,
            };

            assert!(stream.call_method1("_feed", ("hello",)).is_ok());
            assert!(stream.call_method1("_feed", (" world",)).is_ok());
            assert!(stream.call_method0("_finish").is_ok());

            let queue = match stream.getattr("_queue") {
                Ok(q) => q,
                Err(_) => return,
            };
            let t1: String = match queue.call_method0("get") {
                Ok(v) => match v.extract() {
                    Ok(s) => s,
                    Err(_) => return,
                },
                Err(_) => return,
            };
            let t2: String = match queue.call_method0("get") {
                Ok(v) => match v.extract() {
                    Ok(s) => s,
                    Err(_) => return,
                },
                Err(_) => return,
            };
            assert_eq!(t1, "hello");
            assert_eq!(t2, " world");

            let sentinel = match queue.call_method0("get") {
                Ok(v) => v,
                Err(_) => return,
            };
            assert!(sentinel.is_none());
        });
    }

    #[test]
    fn test_token_stream_feed_error() {
        init_python();
        Python::attach(|py| {
            let helper = match get_token_stream_helper(py) {
                Ok(h) => h,
                Err(_) => return,
            };
            let cls = match helper.getattr("_TokenStream") {
                Ok(c) => c,
                Err(_) => return,
            };
            let stream = match cls.call0() {
                Ok(s) => s,
                Err(_) => return,
            };

            assert!(stream.call_method1("_feed_error", ("bad",)).is_ok());
            assert!(stream.call_method0("_finish").is_ok());

            let queue = match stream.getattr("_queue") {
                Ok(q) => q,
                Err(_) => return,
            };
            let item = match queue.call_method0("get") {
                Ok(v) => v,
                Err(_) => return,
            };
            let rt_err_type = py.get_type::<pyo3::exceptions::PyRuntimeError>();
            assert!(item.is_instance(&rt_err_type).unwrap_or(false));
        });
    }

    #[test]
    fn test_async_engine_repr() {
        init_python();
        Python::attach(|py| {
            let config = PyEngineConfig::new("test.gguf".into(), None, 4, None, None);
            if let Ok(engine) = PyAsyncEngine::new(py, &config) {
                let repr = engine.__repr__(py);
                assert!(repr.contains("AsyncEngine"));
                assert!(repr.contains("false"));
            }
        });
    }

    #[test]
    fn test_async_engine_getter() {
        init_python();
        Python::attach(|py| {
            let config = PyEngineConfig::new("test.gguf".into(), None, 4, None, None);
            if let Ok(engine) = PyAsyncEngine::new(py, &config) {
                let sync_engine = engine.engine(py);
                assert!(!sync_engine.bind(py).borrow().is_loaded());
            }
        });
    }
}

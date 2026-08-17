//! Python wrappers for [`InferenceEngine`] and `EngineConfig`.
//!
//! ## GIL policy
//!
//! All long-running Rust calls (generation, embedding, model loading)
//! release the GIL via `py.detach(...)` so Python threads are not
//! blocked during inference.  Streaming callbacks re-acquire the GIL
//! for each call via `Python::attach(...)`.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCapsule};

use oxillama_runtime::{
    EngineConfig as RustEngineConfig, GenerationConfig, InferenceEngine, SamplerConfig,
};

use crate::callback::{
    make_progress_bridge, ProgressBridge, DEFAULT_THROTTLE_MS, DEFAULT_THROTTLE_TOKENS,
};
use crate::cancel::PyCancellationToken;

use crate::error::runtime_to_py;
use crate::generation::{PyGenerationConfig, PyGenerationOutcome};
use crate::sampler::PySamplerConfig;

/// Configuration for the inference engine.
///
/// All fields have Python-friendly defaults.
///
/// # Python Example
///
/// ```python
/// config = EngineConfig(
///     model_path="model.gguf",
///     context_size=4096,
///     num_threads=4,
/// )
/// ```
#[pyclass(name = "EngineConfig", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyEngineConfig {
    /// Path to the GGUF model file.
    #[pyo3(get, set)]
    pub model_path: String,
    /// Path to the tokenizer JSON (None = try to find automatically).
    #[pyo3(get, set)]
    pub tokenizer_path: Option<String>,
    /// Maximum context length.  None = use the model's built-in default.
    #[pyo3(get, set)]
    pub context_size: Option<usize>,
    /// Number of CPU threads for parallel computation.
    #[pyo3(get, set)]
    pub num_threads: usize,
    /// Sampling configuration (uses Python `SamplerConfig` defaults).
    #[pyo3(get, set)]
    pub sampler: PySamplerConfig,
}

#[pymethods]
impl PyEngineConfig {
    /// Create a new `EngineConfig`.
    ///
    /// `model_path` is the only positional argument; all others are keyword-only.
    ///
    /// Raises:
    ///     ValueError: if `model_path` is empty (or all whitespace), or if
    ///         `context_size` is `0` — pass `None` (the default) to use the
    ///         model's own default context length instead of explicitly
    ///         requesting zero.
    #[new]
    #[pyo3(signature = (
        model_path,
        *,
        context_size = None,
        num_threads = 4,
        tokenizer_path = None,
        sampler = None,
    ))]
    pub fn py_new(
        model_path: String,
        context_size: Option<usize>,
        num_threads: usize,
        tokenizer_path: Option<String>,
        sampler: Option<PySamplerConfig>,
    ) -> PyResult<Self> {
        if model_path.trim().is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model_path must not be empty",
            ));
        }
        if context_size == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "context_size must be > 0 (got 0); pass None to use the model's default \
                 context length instead of explicitly requesting zero",
            ));
        }
        Ok(Self::new(
            model_path,
            context_size,
            num_threads,
            tokenizer_path,
            sampler,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "EngineConfig(model_path={:?}, context_size={:?}, num_threads={})",
            self.model_path, self.context_size, self.num_threads,
        )
    }
}

impl PyEngineConfig {
    /// Construct a config without validation.
    ///
    /// Internal, infallible constructor used by Rust callers that already
    /// know their arguments are valid (this module's own tests,
    /// [`PyEngine::from_hub`]'s default-config fallback). Python code always
    /// goes through [`PyEngineConfig::py_new`] (the `#[new]` / `__new__`
    /// entry point), which validates `model_path` and `context_size` first
    /// and raises `ValueError` on failure — this function does not repeat
    /// those checks, so it must never be reachable directly from Python.
    pub fn new(
        model_path: String,
        context_size: Option<usize>,
        num_threads: usize,
        tokenizer_path: Option<String>,
        sampler: Option<PySamplerConfig>,
    ) -> Self {
        Self {
            model_path,
            tokenizer_path,
            context_size,
            num_threads,
            sampler: sampler.unwrap_or_else(PySamplerConfig::default_config),
        }
    }

    /// Convert to the Rust `EngineConfig`.
    pub fn to_rust(&self) -> RustEngineConfig {
        RustEngineConfig {
            model_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
            context_size: self.context_size,
            num_threads: self.num_threads,
            sampler: self.sampler.to_rust(),
            prefill_chunk_size: 512,
            offload_policy: oxillama_runtime::OffloadPolicy::None,
            // The Python surface does not expose a KV dtype knob yet; keep the
            // lossless default so bindings never change output silently.
            kv_dtype: oxillama_runtime::KvCacheDtype::F32,
            // The Python surface does not expose GPU offload knobs yet; keep
            // offload off so bindings stay on the CPU path. `GpuPolicy::On` is
            // a hard requirement upstream (load fails outright if no device
            // binds), so defaulting to anything else here would risk turning
            // an opt-in feature into a surprise hard failure for callers who
            // never asked for GPU offload.
            gpu: oxillama_runtime::GpuPolicy::Off,
        }
    }
}

/// The main inference engine.
///
/// Manages model loading and provides methods for tokenization, embedding,
/// and text generation.
///
/// # Python Example
///
/// ```python
/// config = EngineConfig(model_path="model.gguf")
/// engine = Engine(config)
/// engine.load_model()
/// text = engine.generate("Hello", max_tokens=128)
/// ```
#[pyclass(name = "Engine")]
pub struct PyEngine {
    pub(crate) inner: InferenceEngine,
}

#[pymethods]
#[allow(clippy::useless_conversion, clippy::too_many_arguments)]
impl PyEngine {
    /// Create a new `Engine` with the given configuration.
    ///
    /// Note: this does **not** load the model.  Call `load_model()` before
    /// using the engine for inference.
    #[new]
    pub fn new(config: &PyEngineConfig) -> Self {
        Self {
            inner: InferenceEngine::new(config.to_rust()),
        }
    }

    /// Load the model from the configured GGUF file.
    ///
    /// This performs the full loading pipeline: GGUF parsing → model
    /// configuration → architecture construction → KV cache → tokenizer.
    ///
    /// Releases the GIL while loading so Python can remain responsive.
    ///
    /// Raises:
    ///     IOError: if the model file cannot be read.
    ///     RuntimeError: if the architecture is unsupported.
    pub fn load_model(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| inner.load_model()).map_err(runtime_to_py)
    }

    /// Return `True` if a model has been loaded.
    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    /// Reset the KV cache (start a fresh conversation context).
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Tokenize `text` and return a list of token IDs.
    ///
    /// Requires a loaded model.
    ///
    /// Returns:
    ///     `List[int]`: token IDs.
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    ///     ValueError: if the tokenizer encounters an error.
    pub fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner.tokenize(text).map_err(runtime_to_py)
    }

    /// Decode a single token ID to its string representation.
    ///
    /// Requires a loaded model.
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    pub fn decode_token(&self, token: u32) -> PyResult<String> {
        self.inner.decode_token(token).map_err(runtime_to_py)
    }

    /// Return `True` if `token` is the end-of-sequence token.
    pub fn is_eos(&self, token: u32) -> bool {
        self.inner.is_eos(token)
    }

    /// Return the model's hidden state dimension, or `None` if not loaded.
    pub fn hidden_size(&self) -> Option<usize> {
        self.inner.hidden_size()
    }

    /// Generate text from `prompt`.
    ///
    /// Releases the GIL during inference.
    ///
    /// Args:
    ///     prompt:     Input text.
    ///     max_tokens: Maximum number of new tokens to generate (default 128).
    ///     temperature: Override sampling temperature (keyword-only).
    ///     top_p:       Override nucleus sampling threshold (keyword-only).
    ///     top_k:       Override top-k limit (keyword-only).
    ///     seed:        Override random seed (keyword-only).
    ///     cancel_token: Cooperative cancellation handle (keyword-only).
    ///     progress:    A ``tqdm`` pbar, ``ipywidgets.IntProgress`` widget, a
    ///                  ``Callable[[ProgressEvent], None]``, or ``None``.
    ///                  See :mod:`oxillama_py.progress` for the contract.
    ///     progress_throttle_ms: Minimum milliseconds between throttled
    ///                  progress callbacks (default 50).
    ///     progress_throttle_tokens: Minimum tokens between throttled
    ///                  progress callbacks (default 4).
    ///     progress_capture_text: When ``True``, populate
    ///                  ``ProgressEvent.text_so_far``.  Off by default to
    ///                  avoid the O(n) string copy on every fired tick.
    ///     strict_progress: When ``True``, re-raise the first Python exception
    ///                  raised from the progress callback after generation
    ///                  completes; otherwise the exception is silently
    ///                  swallowed.
    ///
    /// Returns:
    ///     str: The generated text (not including the prompt).
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    ///
    /// # Jupyter example
    ///
    /// ```python
    /// from tqdm.auto import tqdm
    /// with tqdm(desc="Generating", unit="tok") as bar:
    ///     text = engine.generate("Hello", max_tokens=128, progress=bar)
    /// ```
    #[pyo3(signature = (
        prompt,
        max_tokens = 128,
        *,
        temperature = None,
        top_p = None,
        top_k = None,
        seed = None,
        cancel_token = None,
        progress = None,
        progress_throttle_ms = None,
        progress_throttle_tokens = None,
        progress_capture_text = false,
        strict_progress = false,
    ))]
    pub fn generate(
        &mut self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: Option<u64>,
        cancel_token: Option<Py<PyCancellationToken>>,
        progress: Option<Py<PyAny>>,
        progress_throttle_ms: Option<u64>,
        progress_throttle_tokens: Option<usize>,
        progress_capture_text: bool,
        strict_progress: bool,
    ) -> PyResult<String> {
        let inner = &mut self.inner;
        let cancelled = cancel_token
            .as_ref()
            .map(|ct| Python::attach(|py| ct.borrow(py).cancelled.clone()));
        let bridge = make_progress_bridge(
            py,
            progress.as_ref(),
            max_tokens,
            progress_throttle_ms.unwrap_or(DEFAULT_THROTTLE_MS),
            progress_throttle_tokens.unwrap_or(DEFAULT_THROTTLE_TOKENS),
            progress_capture_text,
        )?;
        let bridge_arc: Option<Arc<Mutex<ProgressBridge>>> =
            bridge.map(|b| Arc::new(Mutex::new(b)));
        let make_cb = || {
            // The cancellation flag is *not* read here any more. It used to
            // be — `flag.load(Ordering::Relaxed);` with no `if`, a dead read
            // whose result was discarded — which looked like a cancellation
            // check but could not stop anything, because this callback is
            // infallible and has no way to signal the decode loop. The flag
            // is now handed to the decode loop itself via
            // `GenerationConfig::cancel_flag`, which breaks out at the next
            // token boundary.
            let bridge_inner = bridge_arc.clone();
            move |tok: &str| {
                if let Some(ref bridge) = bridge_inner {
                    Python::attach(|py| {
                        if let Ok(mut b) = bridge.lock() {
                            // strict=false here — the post-call epilogue
                            // raises the stashed error if `strict_progress`
                            // was set.
                            let _ = b.note_token(py, tok, false, false);
                        }
                    });
                }
            }
        };
        let gen_config = build_generation_config(
            inner,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            cancelled.clone(),
        );
        let result = py
            .detach(|| inner.generate_detailed(prompt, &gen_config, make_cb()))
            .map(|outcome| outcome.text)
            .map_err(runtime_to_py);
        let was_cancelled = cancelled
            .as_ref()
            .map(|f| f.load(Ordering::Relaxed))
            .unwrap_or(false);
        let final_result = if was_cancelled {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "generation cancelled",
            ))
        } else {
            result
        };
        // Synthesise a final-tick callback (only on success) before the
        // cleanup finaliser runs.  Then run the explicit finaliser (and
        // propagate any stashed Python error if strict_progress was
        // requested) before the bridge is dropped so the finaliser sees the
        // actual error context.
        if let Some(bridge) = bridge_arc.as_ref() {
            if let Ok(mut b) = bridge.lock() {
                if final_result.is_ok() {
                    b.fire_final(py);
                }
                b.finalise(py, final_result.as_ref().err());
                if strict_progress {
                    if let Some(err) = b.take_stashed_error() {
                        return Err(err);
                    }
                }
            }
        }
        final_result
    }

    /// Generate text from `prompt`, invoking `callback` with each token as it
    /// is produced.
    ///
    /// The callback must accept a single `str` argument.
    ///
    /// The GIL is released during the Rust forward passes and re-acquired for
    /// each callback invocation.
    ///
    /// Args:
    ///     prompt:     Input text.
    ///     max_tokens: Maximum number of new tokens.
    ///     callback:   Python callable invoked with each decoded token string.
    ///     temperature: Override sampling temperature (keyword-only).
    ///     top_p:       Override nucleus sampling threshold (keyword-only).
    ///     top_k:       Override top-k limit (keyword-only).
    ///     seed:        Override random seed (keyword-only).
    ///     cancel_token: Cooperative cancellation handle (keyword-only).
    ///     strict_callback: When ``True`` re-raise the first Python exception
    ///                  raised inside ``callback`` after generation completes.
    ///     progress:    See :meth:`generate` (same contract as the non-streaming
    ///                  variant).  Compose freely with ``callback``.
    ///     progress_throttle_ms: see :meth:`generate`.
    ///     progress_throttle_tokens: see :meth:`generate`.
    ///     progress_capture_text: see :meth:`generate`.
    ///     strict_progress: see :meth:`generate`.
    ///
    /// Returns:
    ///     str: The full generated text (concatenation of all callback inputs).
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    #[pyo3(signature = (
        prompt,
        max_tokens = 128,
        callback = None,
        *,
        temperature = None,
        top_p = None,
        top_k = None,
        seed = None,
        cancel_token = None,
        strict_callback = false,
        progress = None,
        progress_throttle_ms = None,
        progress_throttle_tokens = None,
        progress_capture_text = false,
        strict_progress = false,
    ))]
    pub fn generate_streaming(
        &mut self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        callback: Option<Py<PyAny>>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: Option<u64>,
        cancel_token: Option<Py<PyCancellationToken>>,
        strict_callback: bool,
        progress: Option<Py<PyAny>>,
        progress_throttle_ms: Option<u64>,
        progress_throttle_tokens: Option<usize>,
        progress_capture_text: bool,
        strict_progress: bool,
    ) -> PyResult<String> {
        let inner = &mut self.inner;
        let cancelled = cancel_token
            .as_ref()
            .map(|ct| Python::attach(|py| ct.borrow(py).cancelled.clone()));
        // Shared slot for propagating Python callback errors when strict_callback=true.
        let error_slot: Arc<Mutex<Option<pyo3::PyErr>>> = Arc::new(Mutex::new(None));
        let error_slot_inner = error_slot.clone();

        // Build the optional progress bridge before releasing the GIL so the
        // helper-module import happens with the GIL we already hold.
        let bridge = make_progress_bridge(
            py,
            progress.as_ref(),
            max_tokens,
            progress_throttle_ms.unwrap_or(DEFAULT_THROTTLE_MS),
            progress_throttle_tokens.unwrap_or(DEFAULT_THROTTLE_TOKENS),
            progress_capture_text,
        )?;
        let bridge_arc: Option<Arc<Mutex<ProgressBridge>>> =
            bridge.map(|b| Arc::new(Mutex::new(b)));

        let gen_config = build_generation_config(
            inner,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            cancelled.clone(),
        );

        let result = py
            .detach(|| {
                let cancelled_inner = cancelled.clone();
                let bridge_inner = bridge_arc.clone();
                let cb = move |tok: &str| {
                    // Belt-and-braces: the decode loop already stops at the
                    // next token boundary via `GenerationConfig::cancel_flag`,
                    // so at most one more chunk can arrive after `cancel()`.
                    // Suppressing it keeps the user's callback from being
                    // handed output produced after they asked to stop.
                    if let Some(ref flag) = cancelled_inner {
                        if flag.load(Ordering::Relaxed) {
                            return;
                        }
                    }
                    if let Some(ref cb) = callback {
                        let call_result = Python::attach(|py| cb.call1(py, (tok,)));
                        if let Err(err) = call_result {
                            if strict_callback {
                                if let Ok(mut slot) = error_slot_inner.lock() {
                                    // Only store the first error.
                                    if slot.is_none() {
                                        *slot = Some(err);
                                    }
                                }
                            }
                            // else: swallow the error (legacy behaviour)
                        }
                    }
                    if let Some(ref bridge) = bridge_inner {
                        Python::attach(|py| {
                            if let Ok(mut b) = bridge.lock() {
                                let _ = b.note_token(py, tok, false, false);
                            }
                        });
                    }
                };
                inner.generate_detailed(prompt, &gen_config, cb)
            })
            .map(|outcome| outcome.text)
            .map_err(runtime_to_py);

        // If strict_callback captured a Python error, propagate it now.
        if strict_callback {
            if let Ok(mut slot) = error_slot.lock() {
                if let Some(py_err) = slot.take() {
                    // Drive the finaliser with the error so the widget renders
                    // the failure state before we propagate.
                    if let Some(bridge) = bridge_arc.as_ref() {
                        if let Ok(mut b) = bridge.lock() {
                            b.finalise(py, Some(&py_err));
                        }
                    }
                    return Err(py_err);
                }
            }
        }

        let was_cancelled = cancelled
            .as_ref()
            .map(|f| f.load(Ordering::Relaxed))
            .unwrap_or(false);
        let final_result = if was_cancelled {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "generation cancelled",
            ))
        } else {
            result
        };

        if let Some(bridge) = bridge_arc.as_ref() {
            if let Ok(mut b) = bridge.lock() {
                if final_result.is_ok() {
                    b.fire_final(py);
                }
                b.finalise(py, final_result.as_ref().err());
                if strict_progress {
                    if let Some(err) = b.take_stashed_error() {
                        return Err(err);
                    }
                }
            }
        }
        final_result
    }

    /// Generate from `prompt`, reporting why generation stopped.
    ///
    /// Unlike `generate()`/`generate_streaming()` (bare `str` return), this
    /// is the entry point for building an OpenAI-compatible response: the
    /// returned `GenerationOutcome` distinguishes natural completion from
    /// truncation (`finish_reason`), names the stop sequence that matched
    /// (when any), and reports prompt/completion token counts.  Stop
    /// sequences (`GenerationConfig.stop`) are only reachable through this
    /// method.
    ///
    /// Releases the GIL during inference; `callback` (if given) re-acquires
    /// it for each decoded chunk of text.
    ///
    /// Args:
    ///     prompt:       Input text.
    ///     config:       A `GenerationConfig` describing `max_tokens`,
    ///                   `sampler`, `stop` sequences, and special-token
    ///                   handling.
    ///     callback:     Optional callable invoked with each decoded token
    ///                   chunk as it is produced.
    ///     cancel_token: Cooperative cancellation handle (keyword-only).
    ///
    /// Returns:
    ///     GenerationOutcome: `text`, `finish_reason`, `generated_tokens`,
    ///     `prompt_tokens`, `stop_sequence`, plus the `completion_tokens()`
    ///     / `total_tokens()` helper methods.
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    #[pyo3(signature = (
        prompt,
        config,
        callback = None,
        *,
        cancel_token = None,
    ))]
    pub fn generate_detailed(
        &mut self,
        py: Python<'_>,
        prompt: &str,
        config: &PyGenerationConfig,
        callback: Option<Py<PyAny>>,
        cancel_token: Option<Py<PyCancellationToken>>,
    ) -> PyResult<PyGenerationOutcome> {
        let inner = &mut self.inner;
        let cancelled = cancel_token
            .as_ref()
            .map(|ct| Python::attach(|py| ct.borrow(py).cancelled.clone()));
        let rust_config = GenerationConfig {
            // `PyGenerationConfig` has no cancellation field of its own; the
            // token arrives as a separate kwarg and is bound to the decode
            // loop here.
            cancel_flag: cancelled.clone(),
            ..config.to_rust()
        };

        // Shared slot for propagating a Python callback error raised while
        // the GIL is released (mirrors `generate_streaming`'s strict-error
        // plumbing, minus the opt-out — a callback error here always
        // propagates since there is no `strict_callback` flag to suppress it).
        let error_slot: Arc<Mutex<Option<pyo3::PyErr>>> = Arc::new(Mutex::new(None));
        let error_slot_inner = error_slot.clone();

        let result = py
            .detach(|| {
                let cancelled_inner = cancelled.clone();
                let cb = move |tok: &str| {
                    if let Some(ref flag) = cancelled_inner {
                        if flag.load(Ordering::Relaxed) {
                            return;
                        }
                    }
                    if let Some(ref cb) = callback {
                        let call_result = Python::attach(|py| cb.call1(py, (tok,)));
                        if let Err(err) = call_result {
                            if let Ok(mut slot) = error_slot_inner.lock() {
                                if slot.is_none() {
                                    *slot = Some(err);
                                }
                            }
                        }
                    }
                };
                inner.generate_detailed(prompt, &rust_config, cb)
            })
            .map_err(runtime_to_py);

        if let Ok(mut slot) = error_slot.lock() {
            if let Some(py_err) = slot.take() {
                return Err(py_err);
            }
        }

        let was_cancelled = cancelled
            .as_ref()
            .map(|f| f.load(Ordering::Relaxed))
            .unwrap_or(false);
        if was_cancelled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "generation cancelled",
            ));
        }

        result.map(Into::into)
    }

    /// Return the end-of-generation token IDs known to the tokenizer (e.g.
    /// `</s>`, `<|im_end|>`, `<|endoftext|>`).
    ///
    /// Returns an empty list when no model is loaded.
    pub fn eog_token_ids(&self) -> Vec<u32> {
        self.inner.eog_token_ids()
    }

    /// Compute a semantic embedding vector for `text`.
    ///
    /// Returns an L2-normalised float vector of dimension `hidden_size`.
    /// The KV cache is reset before each call so that embeddings for
    /// different inputs are independent.
    ///
    /// Releases the GIL during the forward passes.
    ///
    /// Returns:
    ///     List\[float\]: L2-normalised embedding vector.
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    pub fn embed(&mut self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        let inner = &mut self.inner;
        py.detach(|| inner.embed(text)).map_err(runtime_to_py)
    }

    /// Compute a semantic embedding vector and return as a 1-D numpy array.
    ///
    /// Identical to `embed()` but returns a ``numpy.ndarray`` of dtype
    /// ``float32`` instead of ``List[float]``.
    ///
    /// Requires the ``numpy`` feature to be enabled at build time.
    ///
    /// Returns:
    ///     numpy.ndarray: L2-normalised embedding vector (shape ``(hidden_size,)``).
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    #[cfg(feature = "numpy")]
    pub fn embed_numpy<'py>(
        &mut self,
        py: Python<'py>,
        text: &str,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
        let inner = &mut self.inner;
        let vec = py.detach(|| inner.embed(text)).map_err(runtime_to_py)?;
        Ok(numpy::PyArray1::from_vec(py, vec))
    }

    /// Compute embeddings for multiple texts and return as a 2-D numpy array.
    ///
    /// Each row is an L2-normalised embedding vector of dimension
    /// ``hidden_size``.
    ///
    /// Returns:
    ///     numpy.ndarray: shape ``(len(texts), hidden_size)``, dtype ``float32``.
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    ///     ValueError:   if the resulting array cannot be constructed.
    #[cfg(feature = "numpy")]
    #[pyo3(signature = (texts))]
    pub fn embed_batch_numpy<'py>(
        &mut self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let inner = &mut self.inner;
        let results: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| py.detach(|| inner.embed(t)).map_err(runtime_to_py))
            .collect::<Result<_, _>>()?;

        if results.is_empty() {
            return numpy::PyArray2::from_vec2(py, &results).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("numpy array creation failed: {e}"))
            });
        }

        numpy::PyArray2::from_vec2(py, &results).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("numpy array creation failed: {e}"))
        })
    }

    /// Apply a LoRA adapter from `lora_path` to the loaded model.
    ///
    /// The engine must have a model loaded before calling this.
    ///
    /// Args:
    ///     lora_path: Path to the LoRA GGUF adapter file.
    ///
    /// Raises:
    ///     IOError:      if the adapter file cannot be parsed.
    ///     RuntimeError: if no model is loaded.
    pub fn apply_lora(&mut self, lora_path: &str) -> PyResult<()> {
        oxillama_runtime::lora_loader::apply_lora(&mut self.inner, lora_path).map_err(runtime_to_py)
    }

    /// Return the raw logit vector for the final token position of `text`.
    ///
    /// The text is tokenized; all tokens except the last are prefilled into the
    /// KV cache, then a single forward pass is executed for the last token and
    /// the resulting logit vector (one float per vocab entry) is returned.
    ///
    /// The KV cache is **not** reset before this call — the caller should call
    /// `reset()` first if an independent pass is desired.
    ///
    /// Returns:
    ///     List\[float\]: Raw (un-softmaxed) logits, length = vocab\_size.
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    ///     ValueError:   if `text` tokenizes to the empty sequence.
    #[pyo3(signature = (text))]
    pub fn forward_logits(&mut self, py: Python<'_>, text: String) -> PyResult<Vec<f32>> {
        let inner = &mut self.inner;
        py.detach(|| {
            let tokens = inner.tokenize(&text)?;
            if tokens.is_empty() {
                return Err(oxillama_runtime::RuntimeError::TokenizerError {
                    message: "text tokenizes to empty sequence".into(),
                });
            }
            let (prefill_tokens, last_slice) = tokens.split_at(tokens.len() - 1);
            let last_token = last_slice[0];
            inner.prefill(prefill_tokens)?;
            inner.forward_one(last_token)
        })
        .map_err(runtime_to_py)
    }

    /// Same as `forward_logits` but returns a 1-D numpy array of dtype float32.
    ///
    /// Requires the ``numpy`` feature to be enabled at build time.
    ///
    /// Returns:
    ///     numpy.ndarray: Raw logits, shape ``(vocab_size,)``, dtype ``float32``.
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    ///     ValueError:   if `text` tokenizes to the empty sequence.
    #[cfg(feature = "numpy")]
    #[pyo3(signature = (text))]
    pub fn forward_logits_numpy<'py>(
        &mut self,
        py: Python<'py>,
        text: String,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
        let vec = self.forward_logits(py, text)?;
        Ok(numpy::PyArray1::from_vec(py, vec))
    }

    /// Return the logits from the last forward pass as a DLPack capsule.
    ///
    /// The logits vector is produced by calling `forward_logits(text)` and
    /// then wrapping the result as a 1-D `float32` DLPack tensor with shape
    /// `[vocab_size]`.
    ///
    /// The returned object is a `PyCapsule` with name `"dltensor"` that any
    /// DLPack-aware framework (PyTorch, JAX, NumPy ≥ 1.22, etc.) can consume
    /// zero-copy via `__dlpack__` protocol.
    ///
    /// Args:
    ///     text: Input text whose logits to compute.
    ///
    /// Returns:
    ///     A `PyCapsule` wrapping the `[vocab_size]` float32 logit vector.
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    ///     ValueError: if `text` tokenizes to the empty sequence.
    #[pyo3(name = "logits_dlpack")]
    #[pyo3(signature = (text))]
    pub fn logits_dlpack(&mut self, py: Python<'_>, text: String) -> PyResult<Py<PyCapsule>> {
        let logits = self.forward_logits(py, text)?;
        let vocab_size = logits.len() as i64;
        crate::dlpack::vec_to_dlpack(py, logits, vec![vocab_size])
    }

    /// Return the last hidden-state embeddings as a DLPack capsule.
    ///
    /// Computes a semantic embedding for `text` (same as `embed()`) and
    /// returns the result as a 2-D `float32` DLPack tensor with shape
    /// `[1, hidden_size]`.  The extra leading dimension makes the tensor
    /// compatible with batched embedding APIs.
    ///
    /// Returns:
    ///     A `PyCapsule` wrapping the `[1, hidden_size]` float32 embedding.
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    #[pyo3(name = "embeddings_dlpack")]
    #[pyo3(signature = (text))]
    pub fn embeddings_dlpack(&mut self, py: Python<'_>, text: String) -> PyResult<Py<PyCapsule>> {
        let embedding = self.embed(py, &text)?;
        let hidden_size = embedding.len() as i64;
        crate::dlpack::vec_to_dlpack(py, embedding, vec![1_i64, hidden_size])
    }

    /// Download a GGUF model from HuggingFace Hub and construct a loaded
    /// `Engine` in one step.
    ///
    /// The GIL is released while the file is being downloaded so other Python
    /// threads remain responsive.
    ///
    /// Args:
    ///     repo_id:  HuggingFace repository ID, e.g. ``"TheBloke/Llama-2-7B-GGUF"``.
    ///     filename: Specific GGUF file.  If *None*, the first ``*.gguf`` file
    ///               found in the repository is used.
    ///     revision: Git revision / branch / tag.  Defaults to ``"main"``.
    ///     token:    HF access token.  Falls back to ``$HF_TOKEN`` /
    ///               ``$HUGGINGFACE_HUB_TOKEN``.
    ///     config:   Optional engine configuration.  Uses defaults if *None*.
    ///
    /// Returns:
    ///     Engine: A fully loaded engine ready for inference.
    ///
    /// Raises:
    ///     IOError:      if the download or GGUF file fails.
    ///     RuntimeError: if no ``.gguf`` file is found in the repository.
    #[cfg(feature = "hub")]
    #[classmethod]
    #[pyo3(signature = (
        repo_id,
        *,
        filename = None,
        revision = None,
        token = None,
        config = None,
    ))]
    pub fn from_hub(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        repo_id: String,
        filename: Option<String>,
        revision: Option<String>,
        token: Option<String>,
        config: Option<PyEngineConfig>,
    ) -> PyResult<Self> {
        let local_path = py.detach(|| {
            crate::hub::download_model_from_hub(
                &repo_id,
                filename.as_deref(),
                revision.as_deref(),
                token.as_deref(),
            )
        })?;

        let mut engine_config =
            config.unwrap_or_else(|| PyEngineConfig::new(local_path.clone(), None, 4, None, None));
        engine_config.model_path = local_path;

        let mut engine = Self {
            inner: InferenceEngine::new(engine_config.to_rust()),
        };
        let inner = &mut engine.inner;
        py.detach(|| inner.load_model()).map_err(runtime_to_py)?;
        Ok(engine)
    }

    /// Save the engine state to `path` atomically.
    ///
    /// The snapshot format is `OXISNAP1` — a portable byte blob containing
    /// the KV cache, sampler config, and a Blake3 fingerprint of the model
    /// file.  Model weights are **not** stored.
    ///
    /// An optional `hub_origin` keyword argument accepts a Python `dict` with
    /// keys `repo_id`, `filename`, and `sha256`.  When provided, the origin
    /// metadata is written to a JSON sidecar file (`path + ".meta.json"`) so
    /// that `Engine.restore()` / `Engine.from_snapshot_with_hub()` can
    /// re-download the GGUF if it is absent on the restoring machine.
    ///
    /// Args:
    ///     path:       Destination path for the snapshot file.
    ///     hub_origin: Optional dict with `repo_id`, `filename`, `sha256` keys.
    ///
    /// Raises:
    ///     GenerateError: if no model is loaded.
    ///     OSError: if the file cannot be written.
    ///     ValueError: if `hub_origin` is provided but malformed.
    #[pyo3(signature = (path, *, hub_origin = None))]
    pub fn snapshot(
        &self,
        py: Python<'_>,
        path: PathBuf,
        hub_origin: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        let bytes = py.detach(|| self.inner.snapshot()).map_err(runtime_to_py)?;
        py.detach(|| crate::snapshot::write_snapshot_atomic(&path, &bytes))
            .map_err(crate::snapshot::io_to_py)?;

        // Write the JSON meta sidecar when a hub_origin is supplied.
        if let Some(dict) = hub_origin {
            let origin = crate::snapshot::HubOrigin::from_py_dict(&dict)?;
            let snap_bytes = std::fs::read(&path).map_err(crate::snapshot::io_to_py)?;
            let raw_snap = oxillama_runtime::snapshot::EngineSnapshot::deserialize(&snap_bytes)
                .map_err(runtime_to_py)?;
            let meta =
                crate::snapshot::EngineSnapshotMeta::from_engine_snapshot(&raw_snap, Some(origin));
            crate::snapshot::write_meta(&path, &meta)?;
        }
        Ok(())
    }

    /// Return the engine state as a `bytes` object.
    ///
    /// Equivalent to `snapshot(path)` but returns the raw bytes in-memory
    /// instead of writing them to disk.  Useful for streaming over a network
    /// or storing in a database.
    ///
    /// Raises:
    ///     GenerateError: if no model is loaded.
    pub fn snapshot_bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        py.detach(|| self.inner.snapshot()).map_err(runtime_to_py)
    }

    /// Return metadata from the snapshot at `path` without loading the model.
    ///
    /// Returns a :class:`SnapshotInfo` with fields `arch_id`, `model_path`,
    /// `tokenizer_path`, `max_context_length`, `num_threads`, `version`,
    /// `magic`, and `tokens_count`.
    ///
    /// Raises:
    ///     OSError: if the file cannot be read.
    ///     GenerateError: if the snapshot format is invalid.
    #[classmethod]
    pub fn snapshot_info(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        path: PathBuf,
    ) -> PyResult<crate::snapshot::PySnapshotInfo> {
        py.detach(|| crate::snapshot::snapshot_info_from_path(&path))
    }

    /// Reconstruct an `Engine` from a snapshot at `path`.
    ///
    /// If `model_path` is `None` (default), the model path embedded in the
    /// snapshot is used.  When a JSON metadata sidecar exists (`path +
    /// ".meta.json"`) and contains `hub_origin`, the GGUF is re-downloaded
    /// automatically from HuggingFace Hub if the local path is missing.
    ///
    /// Pass an explicit `model_path` to override all automatic resolution,
    /// which is useful when moving a snapshot between machines where the GGUF
    /// lives at a different absolute path.
    ///
    /// NOTE: When `model_path=None`, the snapshot bytes are deserialized twice
    /// (once to peek the embedded path, once in `resume`) — acceptable overhead
    /// for v0.1.3.
    ///
    /// The GGUF model is re-loaded from disk on every restore.
    ///
    /// Raises:
    ///     GenerateError: if the snapshot is corrupted or incompatible.
    ///     LoadError: if the model fingerprint does not match.
    ///     OSError: if the snapshot file cannot be read.
    ///     ValueError: if the SHA-256 of a re-downloaded file does not match.
    #[classmethod]
    #[pyo3(signature = (path, *, model_path = None))]
    pub fn restore(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        path: PathBuf,
        model_path: Option<PathBuf>,
    ) -> PyResult<Self> {
        // Read snapshot bytes with GIL released.
        let bytes = py
            .detach(|| std::fs::read(&path))
            .map_err(crate::snapshot::io_to_py)?;

        // Resolve model path: explicit override takes priority.
        let resolved_model_path: PathBuf = match model_path {
            Some(p) => p,
            None => {
                // Peek the embedded model path from the snapshot.
                let snap = oxillama_runtime::snapshot::EngineSnapshot::deserialize(&bytes)
                    .map_err(runtime_to_py)?;
                let embedded_path = PathBuf::from(&snap.model_path);

                // Check for a hub-aware metadata sidecar.
                let meta_opt = crate::snapshot::read_meta(&path)?;
                if let Some(meta) = meta_opt {
                    if let Some(hub_origin) = meta.hub_origin {
                        // Hub-aware path: may trigger a re-download.
                        py.detach(|| {
                            crate::snapshot::resolve_model_path_with_hub(
                                &meta.model_path,
                                &hub_origin,
                            )
                        })?
                    } else {
                        embedded_path
                    }
                } else {
                    embedded_path
                }
            }
        };

        // Load model and restore state with GIL released (may take seconds).
        py.detach(|| oxillama_runtime::InferenceEngine::resume(&bytes, &resolved_model_path))
            .map_err(runtime_to_py)
            .map(|inner| Self { inner })
    }

    /// Reconstruct an `Engine` from a snapshot, using hub-aware model
    /// resolution.
    ///
    /// This is a convenience classmethod that calls `restore(path)` and relies
    /// entirely on the hub metadata written by `snapshot(path,
    /// hub_origin=...)`.  If the GGUF is absent locally it is re-downloaded
    /// from HuggingFace Hub and its SHA-256 is verified before loading.
    ///
    /// Equivalent to ``Engine.restore(path)`` when a ``.meta.json`` sidecar
    /// exists alongside the snapshot.  Provided as a distinct method to make
    /// the intent explicit at the call site.
    ///
    /// Args:
    ///     snapshot_path: Path to the snapshot file.
    ///
    /// Returns:
    ///     Engine: A fully loaded engine restored from the snapshot.
    ///
    /// Raises:
    ///     GenerateError: if the snapshot is corrupted or incompatible.
    ///     LoadError: if the model fingerprint does not match.
    ///     OSError: if the snapshot file or sidecar cannot be read.
    ///     ValueError: if the SHA-256 of the re-downloaded file does not match.
    #[classmethod]
    #[pyo3(name = "from_snapshot_with_hub")]
    pub fn from_snapshot_with_hub(
        cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        snapshot_path: PathBuf,
    ) -> PyResult<Self> {
        Self::restore(cls, py, snapshot_path, None)
    }

    /// Create an async-friendly wrapper around this engine.
    ///
    /// Returns a Python-level :class:`AsyncEngine` instance that wraps
    /// ``self`` and exposes ``generate`` and ``stream`` coroutines, both
    /// of which offload blocking inference to a thread-pool executor via
    /// ``asyncio.get_running_loop().run_in_executor``.
    ///
    /// The returned :class:`AsyncEngine` holds a reference to this
    /// :class:`Engine` object; the caller is responsible for ensuring the
    /// engine remains alive for the duration of async use.
    ///
    /// Returns:
    ///     AsyncEngine: an async-capable wrapper around ``self``.
    ///
    /// Raises:
    ///     ImportError: if the ``oxillama_py`` package cannot be imported.
    ///     RuntimeError: if :class:`AsyncEngine` cannot be instantiated.
    ///
    /// # Python example
    ///
    /// Fenced as `text`, not left as a bare reST `Example::` indented block:
    /// rustdoc compiles an *indented* block inside a doc comment as Rust, so
    /// the Python below was a failing doctest under `cargo test --doc` (which
    /// `cargo nextest` does not run, so nothing reported it).
    ///
    /// ```text
    /// import asyncio
    /// from oxillama_py import EngineConfig, Engine
    ///
    /// cfg = EngineConfig("model.gguf")
    /// engine = Engine(cfg)
    /// engine.load_model()
    ///
    /// ae = engine.async_engine()
    ///
    /// async def main():
    ///     text = await ae.generate("Hello", max_tokens=64)
    ///     print(text)
    ///
    /// asyncio.run(main())
    /// ```
    #[pyo3(signature = ())]
    pub fn async_engine<'py>(
        slf: &Bound<'py, PyEngine>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = py.import("oxillama_py")?;
        let async_engine_cls = module.getattr("AsyncEngine")?;
        async_engine_cls.call1((slf,))
    }

    /// Pickle refusal — Engine state must be persisted via `Engine.snapshot(path)`.
    fn __reduce__(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Engine cannot be pickled; use Engine.snapshot(path) and Engine.restore(path) \
             instead — see oxillama_py.snapshot docs.",
        ))
    }

    /// Pickle refusal (protocol-aware variant).
    fn __reduce_ex__(&self, _protocol: i32) -> PyResult<()> {
        self.__reduce__()
    }
}

/// Build a [`SamplerConfig`] by cloning the engine's current config and
/// applying any per-call overrides supplied by the Python caller.
fn build_override_config(
    engine: &InferenceEngine,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
) -> SamplerConfig {
    let mut cfg = engine.config().sampler.clone();
    if let Some(t) = temperature {
        cfg.temperature = t;
    }
    if let Some(p) = top_p {
        cfg.top_p = p;
    }
    if let Some(k) = top_k {
        cfg.top_k = k;
    }
    if let Some(s) = seed {
        cfg.seed = Some(s);
    }
    cfg
}

/// Build the [`GenerationConfig`] for a `generate()` / `generate_streaming()`
/// call, including the cooperative cancellation hook.
///
/// This replaces the old split between `InferenceEngine::generate` and
/// `generate_with_config`: both of those build a `GenerationConfig` internally
/// and give the caller no way to attach a `cancel_flag`, which is why
/// `cancel_token=` used to be unable to shorten a generation by a single
/// forward pass. Everything else matches those wrappers exactly — the engine's
/// own sampler when no override is given, `GenerationConfig::default()` for
/// the remaining fields.
fn build_generation_config(
    engine: &InferenceEngine,
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> GenerationConfig {
    let sampler = if temperature.is_some() || top_p.is_some() || top_k.is_some() || seed.is_some() {
        build_override_config(engine, temperature, top_p, top_k, seed)
    } else {
        engine.config().sampler.clone()
    };
    GenerationConfig {
        max_tokens,
        sampler,
        cancel_flag,
        ..GenerationConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `PyEngineConfig` must carry the model_path through to the Rust config.
    #[test]
    fn test_engine_config_model_path() {
        let cfg = PyEngineConfig::new("test.gguf".to_string(), None, 4, None, None);
        assert_eq!(cfg.model_path, "test.gguf");
        let rust_cfg = cfg.to_rust();
        assert_eq!(rust_cfg.model_path, "test.gguf");
    }

    /// `PyEngineConfig` default num_threads is 4.
    #[test]
    fn test_engine_config_default_threads() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        assert_eq!(cfg.num_threads, 4);
    }

    /// `PyEngineConfig` default context_size is None.
    #[test]
    fn test_engine_config_default_context_size() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        assert!(cfg.context_size.is_none());
    }

    /// `PyEngineConfig` context_size override is forwarded.
    #[test]
    fn test_engine_config_context_size_override() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), Some(8192), 4, None, None);
        let rust_cfg = cfg.to_rust();
        assert_eq!(rust_cfg.context_size, Some(8192));
    }

    /// A freshly constructed `PyEngine` must not be loaded.
    #[test]
    fn test_engine_not_loaded_initially() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let engine = PyEngine::new(&cfg);
        assert!(!engine.is_loaded());
    }

    /// `tokenize` on an unloaded engine must return Err (ModelNotLoaded).
    #[test]
    fn test_tokenize_err_when_not_loaded() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let engine = PyEngine::new(&cfg);
        // We can only test the Rust side (no Python runtime needed)
        let result = engine.inner.tokenize("hello");
        assert!(
            result.is_err(),
            "tokenize should return Err when no model is loaded"
        );
    }

    /// `hidden_size` returns None before a model is loaded.
    #[test]
    fn test_hidden_size_none_when_not_loaded() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let engine = PyEngine::new(&cfg);
        assert!(engine.hidden_size().is_none());
    }

    /// `__repr__` contains the model path and context_size.
    #[test]
    fn test_engine_config_repr_contains_path_and_context() {
        let cfg = PyEngineConfig::new("my_model.gguf".to_string(), Some(2048), 4, None, None);
        let repr = cfg.__repr__();
        assert!(
            repr.contains("my_model.gguf"),
            "repr missing model path: {repr}"
        );
        assert!(repr.contains("2048"), "repr missing context_size: {repr}");
    }

    /// `__repr__` with no context_size contains "None".
    #[test]
    fn test_engine_config_repr_no_context_size() {
        let cfg = PyEngineConfig::new("path.gguf".to_string(), None, 4, None, None);
        let repr = cfg.__repr__();
        assert!(repr.contains("path.gguf"), "repr missing path: {repr}");
    }

    /// `to_rust()` propagates `tokenizer_path` correctly.
    #[test]
    fn test_engine_config_to_rust_tokenizer_path() {
        let cfg = PyEngineConfig::new(
            "m.gguf".to_string(),
            None,
            4,
            Some("tok.json".to_string()),
            None,
        );
        let rust_cfg = cfg.to_rust();
        assert_eq!(rust_cfg.tokenizer_path, Some("tok.json".to_string()));
    }

    /// `to_rust()` propagates `tokenizer_path = None` correctly.
    #[test]
    fn test_engine_config_to_rust_no_tokenizer_path() {
        let cfg = PyEngineConfig::new("m.gguf".to_string(), None, 4, None, None);
        let rust_cfg = cfg.to_rust();
        assert!(rust_cfg.tokenizer_path.is_none());
    }

    /// `to_rust()` propagates non-default num_threads.
    #[test]
    fn test_engine_config_to_rust_custom_threads() {
        let cfg = PyEngineConfig::new("m.gguf".to_string(), None, 16, None, None);
        let rust_cfg = cfg.to_rust();
        assert_eq!(rust_cfg.num_threads, 16);
    }

    /// `to_rust()` with a custom sampler propagates temperature.
    #[test]
    fn test_engine_config_to_rust_custom_sampler_temperature() {
        let mut sampler = PySamplerConfig::default_config();
        sampler.temperature = 1.5;
        let cfg = PyEngineConfig::new("m.gguf".to_string(), None, 4, None, Some(sampler));
        let rust_cfg = cfg.to_rust();
        assert!(
            (rust_cfg.sampler.temperature - 1.5).abs() < 1e-6,
            "sampler temperature not propagated: {}",
            rust_cfg.sampler.temperature
        );
    }

    /// `reset()` on an unloaded engine must not panic.
    #[test]
    fn test_reset_on_unloaded_engine_does_not_panic() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let mut engine = PyEngine::new(&cfg);
        engine.reset();
    }

    /// `is_eos()` on an unloaded engine must not panic.
    #[test]
    fn test_is_eos_on_unloaded_engine_does_not_panic() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let engine = PyEngine::new(&cfg);
        // Just verify it does not panic; the return value is false when no
        // model is loaded (no EOS token is known).
        let _ = engine.is_eos(2);
    }

    /// `decode_token` on an unloaded engine returns Err.
    #[test]
    fn test_decode_token_err_when_not_loaded() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let engine = PyEngine::new(&cfg);
        let result = engine.inner.decode_token(0);
        assert!(
            result.is_err(),
            "decode_token should return Err when not loaded"
        );
    }

    /// `apply_lora` on an unloaded engine with a nonexistent path returns Err.
    #[test]
    fn test_apply_lora_err_nonexistent_path() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let mut engine = PyEngine::new(&cfg);
        let path = std::env::temp_dir().join("oxillama_py_nosuchfile_abc123.gguf");
        let path_str = path.to_string_lossy();
        let result = oxillama_runtime::lora_loader::apply_lora(&mut engine.inner, &path_str);
        assert!(
            result.is_err(),
            "apply_lora should return Err for missing file"
        );
    }

    /// `build_override_config` with all `None` returns unchanged defaults.
    #[test]
    fn test_build_override_config_no_overrides() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let engine = PyEngine::new(&cfg);
        let sampler = build_override_config(&engine.inner, None, None, None, None);
        let default = SamplerConfig::default();
        assert!(
            (sampler.temperature - default.temperature).abs() < 1e-6,
            "temperature should be unchanged"
        );
        assert_eq!(sampler.top_k, default.top_k);
        assert!(
            (sampler.top_p - default.top_p).abs() < 1e-6,
            "top_p should be unchanged"
        );
        assert!(sampler.seed.is_none(), "seed should remain None");
    }

    /// `build_override_config` applies partial overrides.
    #[test]
    fn test_build_override_config_partial() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let engine = PyEngine::new(&cfg);
        let sampler = build_override_config(&engine.inner, Some(0.5), None, Some(10), Some(42));
        assert!(
            (sampler.temperature - 0.5).abs() < 1e-6,
            "temperature override failed"
        );
        assert_eq!(sampler.top_k, 10, "top_k override failed");
        assert_eq!(sampler.seed, Some(42), "seed override failed");
        // top_p should remain at default since not overridden
        let default = SamplerConfig::default();
        assert!(
            (sampler.top_p - default.top_p).abs() < 1e-6,
            "top_p should be unchanged"
        );
    }

    /// `build_override_config` with all overrides set.
    #[test]
    fn test_build_override_config_all() {
        let cfg = PyEngineConfig::new("x.gguf".to_string(), None, 4, None, None);
        let engine = PyEngine::new(&cfg);
        let sampler =
            build_override_config(&engine.inner, Some(1.0), Some(0.95), Some(50), Some(99));
        assert!((sampler.temperature - 1.0).abs() < 1e-6);
        assert!((sampler.top_p - 0.95).abs() < 1e-6);
        assert_eq!(sampler.top_k, 50);
        assert_eq!(sampler.seed, Some(99));
    }

    // ── `py_new` validation (the pyo3-facing constructor) ─────────────────

    /// `py_new` with a valid config must succeed and match `new`'s output.
    #[test]
    fn test_py_new_accepts_valid_config() {
        let cfg = PyEngineConfig::py_new("model.gguf".to_string(), Some(2048), 4, None, None)
            .expect("valid config must not raise");
        assert_eq!(cfg.model_path, "model.gguf");
        assert_eq!(cfg.context_size, Some(2048));
    }

    /// `py_new` with `context_size = None` (the "use the model's default"
    /// spelling) must succeed — only `Some(0)` is rejected.
    #[test]
    fn test_py_new_accepts_none_context_size() {
        PyEngineConfig::py_new("model.gguf".to_string(), None, 4, None, None)
            .expect("None context_size must not raise");
    }

    /// `context_size = 0` must raise `ValueError` — this is the exact
    /// pre-existing pytest gap Mission B closes (`context_size=0` used to
    /// silently mean "auto" with no validation).
    #[test]
    fn test_py_new_rejects_zero_context_size() {
        let result = PyEngineConfig::py_new("model.gguf".to_string(), Some(0), 4, None, None);
        assert!(result.is_err(), "context_size=0 must raise, not succeed");
    }

    /// An empty `model_path` must raise `ValueError`.
    #[test]
    fn test_py_new_rejects_empty_model_path() {
        let result = PyEngineConfig::py_new(String::new(), None, 4, None, None);
        assert!(result.is_err(), "empty model_path must raise");
    }

    /// A whitespace-only `model_path` must also raise (`.trim().is_empty()`,
    /// not a bare `.is_empty()` check).
    #[test]
    fn test_py_new_rejects_whitespace_only_model_path() {
        let result = PyEngineConfig::py_new("   ".to_string(), None, 4, None, None);
        assert!(result.is_err(), "whitespace-only model_path must raise");
    }

    /// `num_threads = 0` is documented as "auto" on the Rust `EngineConfig`
    /// and must remain accepted — this is not one of the validated fields.
    #[test]
    fn test_py_new_accepts_zero_num_threads_as_auto() {
        let cfg = PyEngineConfig::py_new("model.gguf".to_string(), None, 0, None, None)
            .expect("num_threads=0 (auto) must not raise");
        assert_eq!(cfg.num_threads, 0);
    }
}

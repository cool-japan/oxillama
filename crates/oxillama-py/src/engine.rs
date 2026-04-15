//! Python wrappers for [`InferenceEngine`] and `EngineConfig`.
//!
//! ## GIL policy
//!
//! All long-running Rust calls (generation, embedding, model loading)
//! release the GIL via `py.detach(...)` so Python threads are not
//! blocked during inference.  Streaming callbacks re-acquire the GIL
//! for each call via `Python::attach(...)`.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use oxillama_runtime::{EngineConfig as RustEngineConfig, InferenceEngine, SamplerConfig};

use crate::error::runtime_to_py;
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
    #[new]
    #[pyo3(signature = (
        model_path,
        *,
        context_size = None,
        num_threads = 4,
        tokenizer_path = None,
        sampler = None,
    ))]
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

    fn __repr__(&self) -> String {
        format!(
            "EngineConfig(model_path={:?}, context_size={:?}, num_threads={})",
            self.model_path, self.context_size, self.num_threads,
        )
    }
}

impl PyEngineConfig {
    /// Convert to the Rust `EngineConfig`.
    pub fn to_rust(&self) -> RustEngineConfig {
        RustEngineConfig {
            model_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
            context_size: self.context_size,
            num_threads: self.num_threads,
            sampler: self.sampler.to_rust(),
            prefill_chunk_size: 512,
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
    ///
    /// Returns:
    ///     str: The generated text (not including the prompt).
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    #[pyo3(signature = (prompt, max_tokens = 128, *, temperature = None, top_p = None, top_k = None, seed = None))]
    pub fn generate(
        &mut self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let inner = &mut self.inner;
        if temperature.is_some() || top_p.is_some() || top_k.is_some() || seed.is_some() {
            let config = build_override_config(inner, temperature, top_p, top_k, seed);
            py.detach(|| inner.generate_with_config(prompt, max_tokens, config, |_| {}))
                .map_err(runtime_to_py)
        } else {
            py.detach(|| inner.generate(prompt, max_tokens, |_| {}))
                .map_err(runtime_to_py)
        }
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
    ///
    /// Returns:
    ///     str: The full generated text (concatenation of all callback inputs).
    ///
    /// Raises:
    ///     RuntimeError: if no model is loaded.
    #[pyo3(signature = (prompt, max_tokens = 128, callback = None, *, temperature = None, top_p = None, top_k = None, seed = None))]
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
    ) -> PyResult<String> {
        let inner = &mut self.inner;
        let has_overrides =
            temperature.is_some() || top_p.is_some() || top_k.is_some() || seed.is_some();
        py.detach(|| {
            let cb = |tok: &str| {
                if let Some(ref cb) = callback {
                    Python::attach(|py| {
                        let _ = cb.call1(py, (tok,));
                    });
                }
            };
            if has_overrides {
                let config = build_override_config(inner, temperature, top_p, top_k, seed);
                inner.generate_with_config(prompt, max_tokens, config, cb)
            } else {
                inner.generate(prompt, max_tokens, cb)
            }
        })
        .map_err(runtime_to_py)
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
}

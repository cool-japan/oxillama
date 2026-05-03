//! Python wrappers for [`SpeculativeEngine`] and `SpeculativeConfig`.
//!
//! Speculative decoding uses a small draft model to propose candidate tokens
//! that are then verified by a large target model, achieving equivalent output
//! quality at higher throughput.
//!
//! ## GIL policy
//!
//! The `generate` and `generate_streaming` methods release the GIL via
//! `py.detach(...)` so Python threads remain unblocked during the
//! Rust inference loop.  Streaming callbacks re-acquire the GIL per-token.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::PyAny;

use oxillama_runtime::{SpeculativeConfig as RustSpeculativeConfig, SpeculativeEngine};

use crate::callback::{
    make_progress_bridge, ProgressBridge, DEFAULT_THROTTLE_MS, DEFAULT_THROTTLE_TOKENS,
};
use crate::engine::PyEngineConfig;
use crate::error::runtime_to_py;

/// Configuration for speculative decoding.
///
/// # Python Example
///
/// ```python
/// target_cfg = EngineConfig(model_path="llama-7b.gguf")
/// draft_cfg  = EngineConfig(model_path="llama-1b.gguf")
/// spec_cfg   = SpeculativeConfig(target=target_cfg, draft=draft_cfg, num_speculative=4)
/// engine     = SpeculativeEngine(spec_cfg)
/// text       = engine.generate("Hello", max_tokens=256)
/// ```
#[pyclass(name = "SpeculativeConfig", from_py_object)]
#[derive(Clone)]
pub struct PySpeculativeConfig {
    /// Configuration for the target (large) model.
    #[pyo3(get, set)]
    pub target: PyEngineConfig,
    /// Configuration for the draft (small) model.
    #[pyo3(get, set)]
    pub draft: PyEngineConfig,
    /// Number of tokens the draft model generates per speculation round.
    #[pyo3(get, set)]
    pub num_speculative: usize,
    /// Random seed for accept/reject sampling.  None = deterministic default.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl PySpeculativeConfig {
    /// Create a `SpeculativeConfig`.
    ///
    /// `target` and `draft` are the only positional arguments.
    #[new]
    #[pyo3(signature = (target, draft, *, num_speculative = 4, seed = None))]
    pub fn new(
        target: PyEngineConfig,
        draft: PyEngineConfig,
        num_speculative: usize,
        seed: Option<u64>,
    ) -> Self {
        Self {
            target,
            draft,
            num_speculative,
            seed,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SpeculativeConfig(target={:?}, draft={:?}, num_speculative={}, seed={:?})",
            self.target.model_path, self.draft.model_path, self.num_speculative, self.seed,
        )
    }
}

impl PySpeculativeConfig {
    /// Convert to the Rust `SpeculativeConfig`.
    pub fn to_rust(&self) -> RustSpeculativeConfig {
        RustSpeculativeConfig {
            target: self.target.to_rust(),
            draft: self.draft.to_rust(),
            num_speculative: self.num_speculative,
            seed: self.seed,
        }
    }
}

/// Speculative decoding engine.
///
/// Loads both a target (large) and draft (small) model, then uses the draft
/// model's token proposals to accelerate the target model's generation.
///
/// # Python Example
///
/// ```python
/// spec_cfg = SpeculativeConfig(target=target_cfg, draft=draft_cfg)
/// engine   = SpeculativeEngine(spec_cfg)
/// text     = engine.generate("Hello", max_tokens=256)
/// ```
#[pyclass(name = "SpeculativeEngine")]
pub struct PySpeculativeEngine {
    inner: SpeculativeEngine,
}

#[pymethods]
#[allow(clippy::useless_conversion)]
impl PySpeculativeEngine {
    /// Create and load both models.
    ///
    /// This constructor calls `load_model()` on both the target and draft
    /// engines; it may take a significant amount of time for large models.
    ///
    /// Releases the GIL while loading.
    ///
    /// Raises:
    ///     IOError:      if either model file cannot be read.
    ///     RuntimeError: if an unsupported architecture is encountered.
    #[new]
    pub fn new(py: Python<'_>, config: &PySpeculativeConfig) -> PyResult<Self> {
        let rust_cfg = config.to_rust();
        let inner = py
            .detach(|| SpeculativeEngine::new(rust_cfg))
            .map_err(runtime_to_py)?;
        Ok(Self { inner })
    }

    /// Generate text from `prompt` using speculative decoding.
    ///
    /// Releases the GIL during inference.
    ///
    /// Args:
    ///     prompt:     Input text.
    ///     max_tokens: Maximum number of new tokens (default 128).
    ///     progress:   See :class:`oxillama_py.Engine.generate` (same contract).
    ///     progress_throttle_ms: see :class:`oxillama_py.Engine.generate`.
    ///     progress_throttle_tokens: see :class:`oxillama_py.Engine.generate`.
    ///     progress_capture_text: see :class:`oxillama_py.Engine.generate`.
    ///     strict_progress: see :class:`oxillama_py.Engine.generate`.
    ///
    /// Returns:
    ///     str: Generated text.
    #[pyo3(signature = (
        prompt,
        max_tokens = 128,
        *,
        progress = None,
        progress_throttle_ms = None,
        progress_throttle_tokens = None,
        progress_capture_text = false,
        strict_progress = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        progress: Option<Py<PyAny>>,
        progress_throttle_ms: Option<u64>,
        progress_throttle_tokens: Option<usize>,
        progress_capture_text: bool,
        strict_progress: bool,
    ) -> PyResult<String> {
        let inner = &mut self.inner;
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
        let result = py
            .detach(|| {
                let bridge_inner = bridge_arc.clone();
                inner.generate(prompt, max_tokens, move |tok| {
                    if let Some(ref bridge) = bridge_inner {
                        Python::attach(|py| {
                            if let Ok(mut b) = bridge.lock() {
                                let _ = b.note_token(py, tok, false, false);
                            }
                        });
                    }
                })
            })
            .map_err(runtime_to_py);
        if let Some(bridge) = bridge_arc.as_ref() {
            if let Ok(mut b) = bridge.lock() {
                if result.is_ok() {
                    b.fire_final(py);
                }
                b.finalise(py, result.as_ref().err());
                if strict_progress {
                    if let Some(err) = b.take_stashed_error() {
                        return Err(err);
                    }
                }
            }
        }
        result
    }

    /// Generate text from `prompt`, invoking `callback` with each accepted
    /// token as it is produced.
    ///
    /// The GIL is released during Rust forward passes and re-acquired per
    /// callback invocation.
    ///
    /// Args:
    ///     prompt:     Input text.
    ///     max_tokens: Maximum number of new tokens.
    ///     callback:   Python callable invoked with each decoded token string.
    ///     progress:   See :class:`oxillama_py.Engine.generate` (same contract).
    ///     progress_throttle_ms: see :class:`oxillama_py.Engine.generate`.
    ///     progress_throttle_tokens: see :class:`oxillama_py.Engine.generate`.
    ///     progress_capture_text: see :class:`oxillama_py.Engine.generate`.
    ///     strict_progress: see :class:`oxillama_py.Engine.generate`.
    ///
    /// Returns:
    ///     str: The full generated text.
    #[pyo3(signature = (
        prompt,
        max_tokens = 128,
        callback = None,
        *,
        progress = None,
        progress_throttle_ms = None,
        progress_throttle_tokens = None,
        progress_capture_text = false,
        strict_progress = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming(
        &mut self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        callback: Option<Py<PyAny>>,
        progress: Option<Py<PyAny>>,
        progress_throttle_ms: Option<u64>,
        progress_throttle_tokens: Option<usize>,
        progress_capture_text: bool,
        strict_progress: bool,
    ) -> PyResult<String> {
        let inner = &mut self.inner;
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
        let result = py
            .detach(|| {
                let bridge_inner = bridge_arc.clone();
                inner.generate(prompt, max_tokens, |tok| {
                    if let Some(ref cb) = callback {
                        Python::attach(|py| {
                            let _ = cb.call1(py, (tok,));
                        });
                    }
                    if let Some(ref bridge) = bridge_inner {
                        Python::attach(|py| {
                            if let Ok(mut b) = bridge.lock() {
                                let _ = b.note_token(py, tok, false, false);
                            }
                        });
                    }
                })
            })
            .map_err(runtime_to_py);
        if let Some(bridge) = bridge_arc.as_ref() {
            if let Ok(mut b) = bridge.lock() {
                if result.is_ok() {
                    b.fire_final(py);
                }
                b.finalise(py, result.as_ref().err());
                if strict_progress {
                    if let Some(err) = b.take_stashed_error() {
                        return Err(err);
                    }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::PySamplerConfig;

    fn make_engine_config(path: &str) -> PyEngineConfig {
        PyEngineConfig::new(path.to_string(), None, 4, None, None)
    }

    /// `PySpeculativeConfig` default `num_speculative` is 4.
    #[test]
    fn test_speculative_config_default_k() {
        let cfg = PySpeculativeConfig::new(
            make_engine_config("target.gguf"),
            make_engine_config("draft.gguf"),
            4,
            None,
        );
        assert_eq!(cfg.num_speculative, 4);
    }

    /// Overriding `num_speculative` is preserved.
    #[test]
    fn test_speculative_config_override_k() {
        let cfg = PySpeculativeConfig::new(
            make_engine_config("target.gguf"),
            make_engine_config("draft.gguf"),
            8,
            None,
        );
        assert_eq!(cfg.num_speculative, 8);
    }

    /// `to_rust()` propagates model paths correctly.
    #[test]
    fn test_speculative_config_to_rust() {
        let cfg = PySpeculativeConfig::new(
            make_engine_config("target.gguf"),
            make_engine_config("draft.gguf"),
            4,
            Some(42),
        );
        let rust = cfg.to_rust();
        assert_eq!(rust.target.model_path, "target.gguf");
        assert_eq!(rust.draft.model_path, "draft.gguf");
        assert_eq!(rust.num_speculative, 4);
        assert_eq!(rust.seed, Some(42));
    }

    /// `PySamplerConfig` default is wired into the engine config correctly.
    #[test]
    fn test_engine_config_default_sampler() {
        let cfg = make_engine_config("x.gguf");
        let rust = cfg.to_rust();
        let default_sampler = PySamplerConfig::default_config().to_rust();
        assert!(
            (rust.sampler.temperature - default_sampler.temperature).abs() < 1e-6,
            "sampler temperature should match default"
        );
    }

    /// `__repr__` contains both model paths, k, and seed.
    #[test]
    fn test_speculative_config_repr() {
        let cfg = PySpeculativeConfig::new(
            make_engine_config("big.gguf"),
            make_engine_config("tiny.gguf"),
            6,
            Some(99),
        );
        let repr = cfg.__repr__();
        assert!(
            repr.contains("big.gguf"),
            "repr missing target path: {repr}"
        );
        assert!(
            repr.contains("tiny.gguf"),
            "repr missing draft path: {repr}"
        );
        assert!(repr.contains('6'), "repr missing num_speculative: {repr}");
        assert!(repr.contains("99"), "repr missing seed: {repr}");
    }

    /// `to_rust()` with `seed = None` propagates correctly.
    #[test]
    fn test_speculative_config_to_rust_no_seed() {
        let cfg = PySpeculativeConfig::new(
            make_engine_config("t.gguf"),
            make_engine_config("d.gguf"),
            4,
            None,
        );
        let rust = cfg.to_rust();
        assert!(rust.seed.is_none(), "seed should be None");
    }

    /// Non-default `num_threads` on target and draft propagate through `to_rust()`.
    #[test]
    fn test_speculative_config_custom_threads_propagate() {
        let target = PyEngineConfig::new("t.gguf".to_string(), None, 12, None, None);
        let draft = PyEngineConfig::new("d.gguf".to_string(), None, 3, None, None);
        let cfg = PySpeculativeConfig::new(target, draft, 4, None);
        let rust = cfg.to_rust();
        assert_eq!(rust.target.num_threads, 12);
        assert_eq!(rust.draft.num_threads, 3);
    }

    /// Context sizes on target and draft propagate through `to_rust()`.
    #[test]
    fn test_speculative_config_context_sizes_propagate() {
        let target = PyEngineConfig::new("t.gguf".to_string(), Some(8192), 4, None, None);
        let draft = PyEngineConfig::new("d.gguf".to_string(), Some(2048), 4, None, None);
        let cfg = PySpeculativeConfig::new(target, draft, 4, None);
        let rust = cfg.to_rust();
        assert_eq!(rust.target.context_size, Some(8192));
        assert_eq!(rust.draft.context_size, Some(2048));
    }

    /// `to_rust()` roundtrip preserves all fields of a fully specified config.
    #[test]
    fn test_speculative_config_full_roundtrip() {
        let cfg = PySpeculativeConfig::new(
            make_engine_config("llama-7b.gguf"),
            make_engine_config("llama-1b.gguf"),
            3,
            Some(777),
        );
        let rust = cfg.to_rust();
        assert_eq!(rust.target.model_path, "llama-7b.gguf");
        assert_eq!(rust.draft.model_path, "llama-1b.gguf");
        assert_eq!(rust.num_speculative, 3);
        assert_eq!(rust.seed, Some(777));
    }
}

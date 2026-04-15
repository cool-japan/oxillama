//! Python wrapper for [`SamplerConfig`].
//!
//! Exposes the sampling knobs (temperature, top-k, top-p, min-p, repetition
//! penalty, seed, Mirostat v2) to Python.  Grammar-constrained sampling is not
//! exposed here — use the string-grammar API on `Engine` instead.

use pyo3::prelude::*;

use oxillama_runtime::SamplerConfig;

/// Sampling configuration for text generation.
///
/// All parameters have sensible defaults:
/// - `temperature = 0.7`
/// - `top_k = 40`
/// - `top_p = 0.9`
/// - `min_p = 0.0`
/// - `repetition_penalty = 1.1`
/// - `repetition_penalty_window = 64`
/// - `seed = None` (random)
/// - `mirostat = 0` (disabled)
/// - `mirostat_tau = 5.0`
/// - `mirostat_eta = 0.1`
#[pyclass(name = "SamplerConfig", from_py_object)]
#[derive(Debug, Clone)]
pub struct PySamplerConfig {
    /// Temperature for logit scaling (1.0 = unchanged, 0.0 = greedy).
    #[pyo3(get, set)]
    pub temperature: f32,
    /// Top-K: restrict to the K most likely tokens (0 = disabled).
    #[pyo3(get, set)]
    pub top_k: usize,
    /// Top-P (nucleus): cumulative probability threshold.
    #[pyo3(get, set)]
    pub top_p: f32,
    /// Min-P: minimum probability as a fraction of the top token's probability.
    #[pyo3(get, set)]
    pub min_p: f32,
    /// Repetition penalty factor (1.0 = no penalty).
    #[pyo3(get, set)]
    pub repetition_penalty: f32,
    /// Token history window for repetition penalty.
    #[pyo3(get, set)]
    pub repetition_penalty_window: usize,
    /// Optional random seed for reproducible sampling.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    /// Mirostat mode: 0 = disabled, 2 = Mirostat v2.
    #[pyo3(get, set)]
    pub mirostat: u8,
    /// Mirostat target surprise (tau).
    #[pyo3(get, set)]
    pub mirostat_tau: f32,
    /// Mirostat learning rate (eta).
    #[pyo3(get, set)]
    pub mirostat_eta: f32,
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PySamplerConfig {
    /// Create a new `SamplerConfig` with the given parameters.
    ///
    /// All parameters are keyword-only and have defaults matching the Rust defaults.
    #[new]
    #[pyo3(signature = (
        *,
        temperature = 0.7,
        top_k = 40,
        top_p = 0.9,
        min_p = 0.0,
        repetition_penalty = 1.1,
        repetition_penalty_window = 64,
        seed = None,
        mirostat = 0,
        mirostat_tau = 5.0,
        mirostat_eta = 0.1,
    ))]
    pub fn new(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        repetition_penalty: f32,
        repetition_penalty_window: usize,
        seed: Option<u64>,
        mirostat: u8,
        mirostat_tau: f32,
        mirostat_eta: f32,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            repetition_penalty_window,
            seed,
            mirostat,
            mirostat_tau,
            mirostat_eta,
        }
    }

    /// Return a greedy config (temperature=0, top_k=1).
    #[staticmethod]
    pub fn greedy() -> Self {
        let cfg = SamplerConfig::greedy();
        Self::from_rust(cfg)
    }

    /// Return a Mirostat v2 config.
    #[staticmethod]
    #[pyo3(signature = (tau = 5.0, eta = 0.1))]
    pub fn mirostat_v2(tau: f32, eta: f32) -> Self {
        let cfg = SamplerConfig::mirostat_v2(tau, eta);
        Self::from_rust(cfg)
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplerConfig(temperature={}, top_k={}, top_p={}, min_p={}, \
             repetition_penalty={}, seed={:?}, mirostat={})",
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.seed,
            self.mirostat,
        )
    }
}

impl PySamplerConfig {
    /// Create a default config with the Rust defaults.
    pub fn default_config() -> Self {
        Self::from_rust(SamplerConfig::default())
    }

    /// Convert to the Rust [`SamplerConfig`].
    pub fn to_rust(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            repetition_penalty_window: self.repetition_penalty_window,
            seed: self.seed,
            mirostat: self.mirostat,
            mirostat_tau: self.mirostat_tau,
            mirostat_eta: self.mirostat_eta,
            grammar: None,
            token_vocab: None,
        }
    }

    /// Construct from a Rust [`SamplerConfig`] (no grammar/vocab fields).
    pub fn from_rust(cfg: SamplerConfig) -> Self {
        Self {
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            min_p: cfg.min_p,
            repetition_penalty: cfg.repetition_penalty,
            repetition_penalty_window: cfg.repetition_penalty_window,
            seed: cfg.seed,
            mirostat: cfg.mirostat,
            mirostat_tau: cfg.mirostat_tau,
            mirostat_eta: cfg.mirostat_eta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults_match_rust() {
        let py_cfg = PySamplerConfig::new(
            0.7,  // temperature
            40,   // top_k
            0.9,  // top_p
            0.0,  // min_p
            1.1,  // repetition_penalty
            64,   // repetition_penalty_window
            None, // seed
            0,    // mirostat
            5.0,  // mirostat_tau
            0.1,  // mirostat_eta
        );
        let rust_cfg = SamplerConfig::default();
        assert!(
            (py_cfg.temperature - rust_cfg.temperature).abs() < 1e-6,
            "temperature default mismatch"
        );
        assert_eq!(py_cfg.top_k, rust_cfg.top_k, "top_k default mismatch");
        assert!(
            (py_cfg.top_p - rust_cfg.top_p).abs() < 1e-6,
            "top_p default mismatch"
        );
        assert_eq!(
            py_cfg.mirostat, rust_cfg.mirostat,
            "mirostat default mismatch"
        );
    }

    #[test]
    fn test_greedy_static_method() {
        let cfg = PySamplerConfig::greedy();
        assert_eq!(cfg.temperature, 0.0, "greedy temperature must be 0");
        assert_eq!(cfg.top_k, 1, "greedy top_k must be 1");
    }

    #[test]
    fn test_mirostat_v2_static_method() {
        let cfg = PySamplerConfig::mirostat_v2(3.0, 0.05);
        assert_eq!(cfg.mirostat, 2, "mirostat mode must be 2");
        assert!((cfg.mirostat_tau - 3.0).abs() < 1e-6, "tau mismatch");
        assert!((cfg.mirostat_eta - 0.05).abs() < 1e-6, "eta mismatch");
    }

    #[test]
    fn test_to_rust_roundtrip() {
        let py_cfg = PySamplerConfig::new(1.2, 20, 0.85, 0.05, 1.3, 32, Some(42), 0, 5.0, 0.1);
        let rust_cfg = py_cfg.to_rust();
        assert!((rust_cfg.temperature - 1.2).abs() < 1e-6);
        assert_eq!(rust_cfg.top_k, 20);
        assert_eq!(rust_cfg.seed, Some(42));
        assert!(rust_cfg.grammar.is_none(), "grammar should be None");
        assert!(rust_cfg.token_vocab.is_none(), "token_vocab should be None");
    }

    /// `default_config()` matches `SamplerConfig::default()` on all scalar fields.
    #[test]
    fn test_default_config_matches_rust_default() {
        let py_cfg = PySamplerConfig::default_config();
        let rust_default = SamplerConfig::default();
        assert!(
            (py_cfg.temperature - rust_default.temperature).abs() < 1e-6,
            "temperature mismatch"
        );
        assert_eq!(py_cfg.top_k, rust_default.top_k, "top_k mismatch");
        assert!(
            (py_cfg.top_p - rust_default.top_p).abs() < 1e-6,
            "top_p mismatch"
        );
        assert!(
            (py_cfg.min_p - rust_default.min_p).abs() < 1e-6,
            "min_p mismatch"
        );
        assert!(
            (py_cfg.repetition_penalty - rust_default.repetition_penalty).abs() < 1e-6,
            "repetition_penalty mismatch"
        );
        assert_eq!(
            py_cfg.repetition_penalty_window, rust_default.repetition_penalty_window,
            "repetition_penalty_window mismatch"
        );
        assert_eq!(py_cfg.mirostat, rust_default.mirostat, "mirostat mismatch");
    }

    /// `from_rust` → `to_rust` roundtrip preserves every field.
    #[test]
    fn test_from_rust_to_rust_roundtrip() {
        let original = SamplerConfig {
            temperature: 0.42,
            top_k: 15,
            top_p: 0.77,
            min_p: 0.02,
            repetition_penalty: 1.05,
            repetition_penalty_window: 32,
            seed: Some(1234),
            mirostat: 2,
            mirostat_tau: 4.0,
            mirostat_eta: 0.08,
            grammar: None,
            token_vocab: None,
        };
        let py_cfg = PySamplerConfig::from_rust(original.clone());
        let back = py_cfg.to_rust();
        assert!((back.temperature - original.temperature).abs() < 1e-6);
        assert_eq!(back.top_k, original.top_k);
        assert!((back.top_p - original.top_p).abs() < 1e-6);
        assert!((back.min_p - original.min_p).abs() < 1e-6);
        assert!((back.repetition_penalty - original.repetition_penalty).abs() < 1e-6);
        assert_eq!(
            back.repetition_penalty_window,
            original.repetition_penalty_window
        );
        assert_eq!(back.seed, original.seed);
        assert_eq!(back.mirostat, original.mirostat);
        assert!((back.mirostat_tau - original.mirostat_tau).abs() < 1e-6);
        assert!((back.mirostat_eta - original.mirostat_eta).abs() < 1e-6);
        assert!(back.grammar.is_none());
        assert!(back.token_vocab.is_none());
    }

    /// `__repr__` contains the most important field names and values.
    #[test]
    fn test_repr_contains_key_fields() {
        let cfg = PySamplerConfig::new(0.9, 50, 0.95, 0.0, 1.0, 64, Some(7), 0, 5.0, 0.1);
        let repr = cfg.__repr__();
        assert!(
            repr.contains("temperature"),
            "repr missing 'temperature': {repr}"
        );
        assert!(repr.contains("top_k"), "repr missing 'top_k': {repr}");
        assert!(
            repr.contains("0.9"),
            "repr missing temperature value: {repr}"
        );
        assert!(repr.contains("50"), "repr missing top_k value: {repr}");
    }

    /// `to_rust()` always produces `grammar = None` and `token_vocab = None`.
    #[test]
    fn test_to_rust_grammar_and_vocab_always_none() {
        let cfg = PySamplerConfig::default_config();
        let rust = cfg.to_rust();
        assert!(
            rust.grammar.is_none(),
            "grammar must be None after to_rust()"
        );
        assert!(
            rust.token_vocab.is_none(),
            "token_vocab must be None after to_rust()"
        );
    }

    /// Mutating `temperature` on a `PySamplerConfig` is reflected in `to_rust()`.
    #[test]
    fn test_temperature_mutation_propagates_to_rust() {
        let mut cfg = PySamplerConfig::default_config();
        cfg.temperature = 0.0;
        let rust = cfg.to_rust();
        assert!(
            rust.temperature.abs() < 1e-6,
            "temperature mutation not propagated"
        );
    }
}

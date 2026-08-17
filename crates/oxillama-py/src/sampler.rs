//! Python wrapper for [`SamplerConfig`].
//!
//! Exposes the sampling knobs (temperature, top-k, top-p, min-p, repetition
//! penalty, frequency/presence penalty, seed, Mirostat v2, EOG token ids) to
//! Python.  Grammar-constrained sampling is not exposed here — use the
//! string-grammar API on `Engine` instead.

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
/// - `frequency_penalty = 0.0` (disabled)
/// - `presence_penalty = 0.0` (disabled)
/// - `eog_token_ids = []`
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
    /// OpenAI-style frequency penalty: `logit[t] -= count(t) * frequency_penalty`
    /// (0.0 = disabled). Additive, scaled by how many times the token occurred
    /// in the repetition-penalty window.
    #[pyo3(get, set)]
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty: a flat `logit[t] -= presence_penalty`
    /// for every distinct token that appeared at least once in the window
    /// (0.0 = disabled).
    #[pyo3(get, set)]
    pub presence_penalty: f32,
    /// End-of-generation token IDs consulted by grammar-constrained sampling
    /// so the model can still terminate once the grammar is satisfied.
    /// Only meaningful together with a grammar (not exposed via this class —
    /// use the string-grammar API on `Engine` instead), but kept here so the
    /// field round-trips through `to_rust()`/`from_rust()` instead of being
    /// silently dropped.
    #[pyo3(get, set)]
    pub eog_token_ids: Vec<u32>,
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PySamplerConfig {
    /// Create a new `SamplerConfig` with the given parameters.
    ///
    /// All parameters are keyword-only and have defaults matching the Rust defaults.
    ///
    /// Raises:
    ///     ValueError: if `temperature` is negative or NaN; if `top_p` or
    ///         `min_p` fall outside the closed interval `[0.0, 1.0]`; if
    ///         `repetition_penalty` is not a positive, finite number; or if
    ///         `mirostat` is not `0` (disabled), `1` (Mirostat v1), or `2`
    ///         (Mirostat v2).
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
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        eog_token_ids = None,
    ))]
    pub fn py_new(
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
        frequency_penalty: f32,
        presence_penalty: f32,
        eog_token_ids: Option<Vec<u32>>,
    ) -> PyResult<Self> {
        // `temperature < 0.0` and `repetition_penalty <= 0.0` are plain
        // comparisons, which are always `false` for NaN — checked explicitly
        // so a NaN does not silently reach the sampler chain, where it would
        // corrupt every downstream logit (`1.0 / NaN`, `x *= NaN`, ...).
        // `top_p`/`min_p` use `RangeInclusive::contains`, which already
        // rejects NaN on its own (every `PartialOrd` comparison against NaN
        // is `false`), so no separate `is_nan()` check is needed there.
        if temperature.is_nan() || temperature < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "temperature must be >= 0.0 (0.0 = greedy); got {temperature}"
            )));
        }
        if !(0.0..=1.0).contains(&top_p) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "top_p must be within [0.0, 1.0]; got {top_p}"
            )));
        }
        if !(0.0..=1.0).contains(&min_p) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "min_p must be within [0.0, 1.0]; got {min_p}"
            )));
        }
        if repetition_penalty.is_nan() || repetition_penalty <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "repetition_penalty must be > 0.0 (1.0 = no penalty); got {repetition_penalty}"
            )));
        }
        if mirostat > 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mirostat must be 0 (disabled), 1 (Mirostat v1), or 2 (Mirostat v2); got {mirostat}"
            )));
        }
        Ok(Self::new(
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
            frequency_penalty,
            presence_penalty,
            eog_token_ids,
        ))
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
             repetition_penalty={}, seed={:?}, mirostat={}, \
             frequency_penalty={}, presence_penalty={})",
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.seed,
            self.mirostat,
            self.frequency_penalty,
            self.presence_penalty,
        )
    }
}

impl PySamplerConfig {
    /// Construct a config without validation.
    ///
    /// Internal, infallible constructor used by Rust callers that already
    /// know their arguments are valid (this module's own tests). Python code
    /// always goes through [`PySamplerConfig::py_new`] (the `#[new]` /
    /// `__new__` entry point), which validates `temperature`, `top_p`,
    /// `min_p`, `repetition_penalty`, and `mirostat` first and raises
    /// `ValueError` on failure — this function does not repeat those checks,
    /// so it must never be reachable directly from Python.
    #[allow(clippy::too_many_arguments)]
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
        frequency_penalty: f32,
        presence_penalty: f32,
        eog_token_ids: Option<Vec<u32>>,
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
            frequency_penalty,
            presence_penalty,
            eog_token_ids: eog_token_ids.unwrap_or_default(),
        }
    }

    /// Create a default config with the Rust defaults.
    pub fn default_config() -> Self {
        Self::from_rust(SamplerConfig::default())
    }

    /// Convert to the Rust [`SamplerConfig`].
    pub fn to_rust(&self) -> SamplerConfig {
        let defaults = SamplerConfig::default();
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
            eog_token_ids: self.eog_token_ids.clone(),
            logit_bias: std::collections::HashMap::new(),
            banned_tokens: Vec::new(),
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            // Advanced sampler stages — not exposed in the Python API; use defaults.
            dry_multiplier: defaults.dry_multiplier,
            dry_base: defaults.dry_base,
            dry_allowed_length: defaults.dry_allowed_length,
            xtc_threshold: defaults.xtc_threshold,
            xtc_probability: defaults.xtc_probability,
            typical_p: defaults.typical_p,
            top_a: defaults.top_a,
            eta_cutoff: defaults.eta_cutoff,
            epsilon_cutoff: defaults.epsilon_cutoff,
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
            frequency_penalty: cfg.frequency_penalty,
            presence_penalty: cfg.presence_penalty,
            eog_token_ids: cfg.eog_token_ids,
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
            0.0,  // frequency_penalty
            0.0,  // presence_penalty
            None, // eog_token_ids
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
        let py_cfg = PySamplerConfig::new(
            1.2,
            20,
            0.85,
            0.05,
            1.3,
            32,
            Some(42),
            0,
            5.0,
            0.1,
            0.0,
            0.0,
            None,
        );
        let rust_cfg = py_cfg.to_rust();
        assert!((rust_cfg.temperature - 1.2).abs() < 1e-6);
        assert_eq!(rust_cfg.top_k, 20);
        assert_eq!(rust_cfg.seed, Some(42));
        assert!(rust_cfg.grammar.is_none(), "grammar should be None");
        assert!(rust_cfg.token_vocab.is_none(), "token_vocab should be None");
    }

    /// `frequency_penalty` / `presence_penalty` default to 0.0 and round-trip
    /// through `to_rust()`.
    ///
    /// Regression test for the P8 gap: before this fix `SamplerConfig` (the
    /// Rust struct) gained `frequency_penalty`/`presence_penalty` fields that
    /// `PySamplerConfig` neither exposed nor forwarded — `to_rust()` silently
    /// hardcoded them to the Rust defaults, so a Python caller had no way to
    /// set an OpenAI-style frequency/presence penalty.
    #[test]
    fn test_frequency_presence_penalty_default_and_roundtrip() {
        let cfg = PySamplerConfig::default_config();
        assert_eq!(cfg.frequency_penalty, 0.0, "frequency_penalty default");
        assert_eq!(cfg.presence_penalty, 0.0, "presence_penalty default");

        let py_cfg = PySamplerConfig::new(
            0.7, 40, 0.9, 0.0, 1.1, 64, None, 0, 5.0, 0.1, 0.4, // frequency_penalty
            0.6, // presence_penalty
            None,
        );
        assert!((py_cfg.frequency_penalty - 0.4).abs() < 1e-6);
        assert!((py_cfg.presence_penalty - 0.6).abs() < 1e-6);
        let rust_cfg = py_cfg.to_rust();
        assert!((rust_cfg.frequency_penalty - 0.4).abs() < 1e-6);
        assert!((rust_cfg.presence_penalty - 0.6).abs() < 1e-6);
    }

    /// `eog_token_ids` defaults to empty and round-trips through
    /// `to_rust()`/`from_rust()` instead of being dropped.
    #[test]
    fn test_eog_token_ids_default_and_roundtrip() {
        let cfg = PySamplerConfig::default_config();
        assert!(cfg.eog_token_ids.is_empty(), "eog_token_ids default");

        let py_cfg = PySamplerConfig::new(
            0.7,
            40,
            0.9,
            0.0,
            1.1,
            64,
            None,
            0,
            5.0,
            0.1,
            0.0,
            0.0,
            Some(vec![2, 32000]),
        );
        assert_eq!(py_cfg.eog_token_ids, vec![2, 32000]);
        let rust_cfg = py_cfg.to_rust();
        assert_eq!(rust_cfg.eog_token_ids, vec![2, 32000]);

        let back = PySamplerConfig::from_rust(rust_cfg);
        assert_eq!(back.eog_token_ids, vec![2, 32000]);
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
            logit_bias: std::collections::HashMap::new(),
            banned_tokens: Vec::new(),
            // Advanced sampler stages — keep defaults for this round-trip test.
            ..SamplerConfig::default()
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
        let cfg = PySamplerConfig::new(
            0.9,
            50,
            0.95,
            0.0,
            1.0,
            64,
            Some(7),
            0,
            5.0,
            0.1,
            0.0,
            0.0,
            None,
        );
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

    // ── `py_new` validation (the pyo3-facing constructor) ─────────────────

    /// Call `py_new` with every parameter at its documented default except
    /// the five under test — keeps the validation tests below focused on one
    /// field each instead of repeating all 13 positional arguments.
    #[allow(clippy::too_many_arguments)]
    fn call_py_new(
        temperature: f32,
        top_p: f32,
        min_p: f32,
        repetition_penalty: f32,
        mirostat: u8,
    ) -> PyResult<PySamplerConfig> {
        PySamplerConfig::py_new(
            temperature,
            40,
            top_p,
            min_p,
            repetition_penalty,
            64,
            None,
            mirostat,
            5.0,
            0.1,
            0.0,
            0.0,
            None,
        )
    }

    /// Negative temperature must raise — the exact pre-existing pytest gap
    /// Mission B closes (`temperature=-0.1` used to construct successfully).
    #[test]
    fn test_py_new_rejects_negative_temperature() {
        assert!(
            call_py_new(-0.1, 0.9, 0.0, 1.1, 0).is_err(),
            "negative temperature must raise"
        );
    }

    /// `temperature = 0.0` (greedy) must remain accepted.
    #[test]
    fn test_py_new_accepts_zero_temperature() {
        assert!(
            call_py_new(0.0, 0.9, 0.0, 1.1, 0).is_ok(),
            "temperature=0.0 (greedy) must not raise"
        );
    }

    /// NaN temperature must raise. `temperature < 0.0` alone is `false` for
    /// NaN, so this specifically exercises the `is_nan()` guard — without
    /// it, NaN would reach the sampler chain and corrupt every logit.
    #[test]
    fn test_py_new_rejects_nan_temperature() {
        assert!(
            call_py_new(f32::NAN, 0.9, 0.0, 1.1, 0).is_err(),
            "NaN temperature must raise"
        );
    }

    /// `top_p` above 1.0 must raise.
    #[test]
    fn test_py_new_rejects_top_p_above_one() {
        assert!(call_py_new(0.7, 1.5, 0.0, 1.1, 0).is_err());
    }

    /// Negative `top_p` must raise — without this, the sampler chain
    /// silently degrades to "keep only the top-1 token" instead of raising
    /// (see `TopP::apply`'s cumulative-probability loop in `chain.rs`).
    #[test]
    fn test_py_new_rejects_negative_top_p() {
        assert!(call_py_new(0.7, -0.1, 0.0, 1.1, 0).is_err());
    }

    /// `top_p` at the closed-interval boundaries (0.0 and 1.0) is valid.
    #[test]
    fn test_py_new_accepts_top_p_boundaries() {
        assert!(call_py_new(0.7, 0.0, 0.0, 1.1, 0).is_ok());
        assert!(call_py_new(0.7, 1.0, 0.0, 1.1, 0).is_ok());
    }

    /// `min_p` above 1.0 must raise — without this, `MinP::apply` computes a
    /// threshold above the top token's own probability and masks every
    /// logit to `-inf` (see `chain.rs`).
    #[test]
    fn test_py_new_rejects_min_p_above_one() {
        assert!(call_py_new(0.7, 0.9, 1.5, 1.1, 0).is_err());
    }

    /// Negative `min_p` must raise.
    #[test]
    fn test_py_new_rejects_negative_min_p() {
        assert!(call_py_new(0.7, 0.9, -0.1, 1.1, 0).is_err());
    }

    /// `min_p` at the closed-interval boundaries (0.0 and 1.0) is valid.
    #[test]
    fn test_py_new_accepts_min_p_boundaries() {
        assert!(call_py_new(0.7, 0.9, 0.0, 1.1, 0).is_ok());
        assert!(call_py_new(0.7, 0.9, 1.0, 1.1, 0).is_ok());
    }

    /// `repetition_penalty = 0.0` must raise — without this,
    /// `RepetitionPenalty::apply` divides a positive logit by `0.0`,
    /// producing `f32::INFINITY` (see `chain.rs`).
    #[test]
    fn test_py_new_rejects_zero_repetition_penalty() {
        assert!(call_py_new(0.7, 0.9, 0.0, 0.0, 0).is_err());
    }

    /// Negative `repetition_penalty` must raise (flips the sign of any
    /// positive logit it touches instead of penalising it).
    #[test]
    fn test_py_new_rejects_negative_repetition_penalty() {
        assert!(call_py_new(0.7, 0.9, 0.0, -1.0, 0).is_err());
    }

    /// `repetition_penalty = 1.0` (the documented "no penalty" value) must
    /// remain accepted.
    #[test]
    fn test_py_new_accepts_repetition_penalty_one() {
        assert!(call_py_new(0.7, 0.9, 0.0, 1.0, 0).is_ok());
    }

    /// `mirostat` outside `{0, 1, 2}` must raise — without this it silently
    /// falls through to the standard (non-mirostat) pipeline instead of
    /// erroring, discarding the caller's intent (see `sampling/mod.rs`'s
    /// `mirostat == 1 || mirostat == 2` dispatch).
    #[test]
    fn test_py_new_rejects_invalid_mirostat() {
        assert!(call_py_new(0.7, 0.9, 0.0, 1.1, 3).is_err());
        assert!(call_py_new(0.7, 0.9, 0.0, 1.1, 255).is_err());
    }

    /// Every documented mirostat mode (0, 1, 2) is valid.
    #[test]
    fn test_py_new_accepts_valid_mirostat_modes() {
        assert!(call_py_new(0.7, 0.9, 0.0, 1.1, 0).is_ok());
        assert!(call_py_new(0.7, 0.9, 0.0, 1.1, 1).is_ok());
        assert!(call_py_new(0.7, 0.9, 0.0, 1.1, 2).is_ok());
    }

    /// The all-defaults configuration (mirroring `SamplerConfig::default()`)
    /// must construct successfully through `py_new`, not just through the
    /// unvalidated internal `new`.
    #[test]
    fn test_py_new_accepts_all_defaults() {
        assert!(call_py_new(0.7, 0.9, 0.0, 1.1, 0).is_ok());
    }
}

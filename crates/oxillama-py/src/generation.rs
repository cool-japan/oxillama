//! Python wrappers for [`FinishReason`], [`GenerationConfig`], and
//! [`GenerationOutcome`].
//!
//! These are the types behind [`crate::engine::PyEngine::generate_detailed`],
//! the entry point that reports *why* generation stopped instead of only
//! returning a bare string.

use pyo3::prelude::*;

use oxillama_runtime::{FinishReason, GenerationConfig, GenerationOutcome};

use crate::sampler::PySamplerConfig;

/// Why generation stopped.
///
/// - `EOS`: an end-of-generation token was sampled (covers the whole EOG
///   set, not just a single `eos_token_id`).
/// - `MAX_TOKENS`: the requested `max_tokens` budget was exhausted.
/// - `CONTEXT_FULL`: the KV cache reached the model's context length.
/// - `STOPPED`: a caller-supplied stop sequence was produced.
/// - `CANCELLED`: a `CancellationToken` passed as `cancel_token=` was
///   cancelled; the decode loop stopped at the next token boundary and
///   `text` holds everything produced up to that point.
#[pyclass(name = "FinishReason", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyFinishReason {
    /// An end-of-generation token was sampled.
    Eos,
    /// The requested `max_tokens` budget was exhausted.
    MaxTokens,
    /// The KV cache reached the model's context length.
    ContextFull,
    /// A caller-supplied stop sequence was produced.
    Stopped,
    /// The caller's `CancellationToken` was cancelled mid-generation.
    Cancelled,
}

#[pymethods]
impl PyFinishReason {
    /// The OpenAI `finish_reason` string for this outcome
    /// (`"stop"`, `"length"`, or `"cancelled"`).
    pub fn as_openai_str(&self) -> &'static str {
        FinishReason::from(*self).as_openai_str()
    }

    /// `True` when the model chose to stop rather than being cut off.
    pub fn is_natural(&self) -> bool {
        FinishReason::from(*self).is_natural()
    }

    /// `True` when generation ended because the caller cancelled it.
    pub fn is_cancelled(&self) -> bool {
        FinishReason::from(*self).is_cancelled()
    }

    fn __repr__(&self) -> String {
        format!("FinishReason.{}", FinishReason::from(*self))
    }

    fn __str__(&self) -> String {
        FinishReason::from(*self).to_string()
    }
}

impl From<FinishReason> for PyFinishReason {
    fn from(reason: FinishReason) -> Self {
        match reason {
            FinishReason::Eos => Self::Eos,
            FinishReason::MaxTokens => Self::MaxTokens,
            FinishReason::ContextFull => Self::ContextFull,
            FinishReason::Stopped => Self::Stopped,
            FinishReason::Cancelled => Self::Cancelled,
        }
    }
}

impl From<PyFinishReason> for FinishReason {
    fn from(reason: PyFinishReason) -> Self {
        match reason {
            PyFinishReason::Eos => Self::Eos,
            PyFinishReason::MaxTokens => Self::MaxTokens,
            PyFinishReason::ContextFull => Self::ContextFull,
            PyFinishReason::Stopped => Self::Stopped,
            PyFinishReason::Cancelled => Self::Cancelled,
        }
    }
}

/// Per-request generation settings.
///
/// Passed to [`crate::engine::PyEngine::generate_detailed`]. Unlike
/// `Engine.generate()`/`generate_streaming()` (which only accept a handful
/// of sampler overrides), `GenerationConfig` exposes the full request shape:
/// stop sequences, and the special-token rendering/parsing policy.
#[pyclass(name = "GenerationConfig", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyGenerationConfig {
    /// Maximum number of tokens to generate.
    #[pyo3(get, set)]
    pub max_tokens: usize,
    /// Sampler settings for this request.
    #[pyo3(get, set)]
    pub sampler: PySamplerConfig,
    /// Stop sequences. Generation ends as soon as one is produced; the stop
    /// text itself is excluded from the returned text.
    #[pyo3(get, set)]
    pub stop: Vec<String>,
    /// Whether control tokens appear in the generated text (default `False`,
    /// matching llama.cpp's `--special` default).
    #[pyo3(get, set)]
    pub render_special: bool,
    /// Whether prompt encoding applies the model's BOS/EOS policy.
    #[pyo3(get, set)]
    pub add_special: bool,
    /// Whether control tokens written in the prompt are recognised.
    #[pyo3(get, set)]
    pub parse_special: bool,
}

#[pymethods]
impl PyGenerationConfig {
    /// Create a new `GenerationConfig`.
    ///
    /// `max_tokens` is the only positional argument; all others are
    /// keyword-only.
    #[new]
    #[pyo3(signature = (
        max_tokens = 128,
        *,
        sampler = None,
        stop = None,
        render_special = false,
        add_special = true,
        parse_special = true,
    ))]
    pub fn new(
        max_tokens: usize,
        sampler: Option<PySamplerConfig>,
        stop: Option<Vec<String>>,
        render_special: bool,
        add_special: bool,
        parse_special: bool,
    ) -> Self {
        Self {
            max_tokens,
            sampler: sampler.unwrap_or_else(PySamplerConfig::default_config),
            stop: stop.unwrap_or_default(),
            render_special,
            add_special,
            parse_special,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GenerationConfig(max_tokens={}, stop={:?}, render_special={}, \
             add_special={}, parse_special={})",
            self.max_tokens, self.stop, self.render_special, self.add_special, self.parse_special,
        )
    }
}

impl PyGenerationConfig {
    /// Convert to the Rust [`GenerationConfig`].
    pub fn to_rust(&self) -> GenerationConfig {
        GenerationConfig {
            max_tokens: self.max_tokens,
            sampler: self.sampler.to_rust(),
            stop: self.stop.clone(),
            render_special: self.render_special,
            add_special: self.add_special,
            parse_special: self.parse_special,
            // The Python `GenerationConfig` has no `cancel_flag` field:
            // cancellation is expressed through the `cancel_token=` kwarg,
            // and `PyEngine` attaches the token's flag to the config it
            // builds.  See `crate::engine`.
            cancel_flag: None,
        }
    }
}

/// Everything a caller needs to describe a completed generation.
///
/// Returned by [`crate::engine::PyEngine::generate_detailed`]. Distinguishes
/// natural completion (`FinishReason.EOS` / `STOPPED`) from truncation
/// (`MAX_TOKENS` / `CONTEXT_FULL`), which a bare `str` return value cannot.
#[pyclass(name = "GenerationOutcome", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyGenerationOutcome {
    /// The generated text, excluding any matched stop sequence.
    #[pyo3(get)]
    pub text: String,
    /// Why generation stopped.
    #[pyo3(get)]
    pub finish_reason: PyFinishReason,
    /// The generated token ids, in order. A terminating EOG token is *not*
    /// included — it was never emitted.
    #[pyo3(get)]
    pub generated_tokens: Vec<u32>,
    /// How many tokens the prompt occupied.
    #[pyo3(get)]
    pub prompt_tokens: usize,
    /// The stop sequence that matched, when `finish_reason` is `STOPPED`.
    #[pyo3(get)]
    pub stop_sequence: Option<String>,
}

#[pymethods]
impl PyGenerationOutcome {
    /// The number of generated tokens.
    pub fn completion_tokens(&self) -> usize {
        self.generated_tokens.len()
    }

    /// Total tokens billed for this request (`prompt_tokens + completion_tokens()`).
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "GenerationOutcome(text={:?}, finish_reason={}, prompt_tokens={}, \
             completion_tokens={})",
            self.text,
            FinishReason::from(self.finish_reason),
            self.prompt_tokens,
            self.generated_tokens.len(),
        )
    }
}

impl From<GenerationOutcome> for PyGenerationOutcome {
    fn from(outcome: GenerationOutcome) -> Self {
        Self {
            text: outcome.text,
            finish_reason: outcome.finish_reason.into(),
            generated_tokens: outcome.generated_tokens,
            prompt_tokens: outcome.prompt_tokens,
            stop_sequence: outcome.stop_sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `FinishReason` round-trips through the Rust <-> Python conversion.
    #[test]
    fn test_finish_reason_roundtrip() {
        for reason in [
            FinishReason::Eos,
            FinishReason::MaxTokens,
            FinishReason::ContextFull,
            FinishReason::Stopped,
        ] {
            let py: PyFinishReason = reason.into();
            let back: FinishReason = py.into();
            assert_eq!(reason, back);
        }
    }

    /// `as_openai_str` matches the Rust mapping (`Eos`/`Stopped` -> "stop",
    /// `MaxTokens`/`ContextFull` -> "length").
    #[test]
    fn test_finish_reason_as_openai_str() {
        assert_eq!(PyFinishReason::Eos.as_openai_str(), "stop");
        assert_eq!(PyFinishReason::Stopped.as_openai_str(), "stop");
        assert_eq!(PyFinishReason::MaxTokens.as_openai_str(), "length");
        assert_eq!(PyFinishReason::ContextFull.as_openai_str(), "length");
    }

    /// `is_natural` matches the Rust semantics.
    #[test]
    fn test_finish_reason_is_natural() {
        assert!(PyFinishReason::Eos.is_natural());
        assert!(PyFinishReason::Stopped.is_natural());
        assert!(!PyFinishReason::MaxTokens.is_natural());
        assert!(!PyFinishReason::ContextFull.is_natural());
    }

    /// `GenerationConfig::new` defaults match the Rust `GenerationConfig::default()`.
    #[test]
    fn test_generation_config_defaults_match_rust() {
        let py_cfg = PyGenerationConfig::new(128, None, None, false, true, true);
        let rust_cfg = py_cfg.to_rust();
        let rust_default = GenerationConfig::default();
        assert_eq!(rust_cfg.max_tokens, rust_default.max_tokens);
        assert_eq!(rust_cfg.stop, rust_default.stop);
        assert_eq!(rust_cfg.render_special, rust_default.render_special);
        assert_eq!(rust_cfg.add_special, rust_default.add_special);
        assert_eq!(rust_cfg.parse_special, rust_default.parse_special);
    }

    /// `stop` sequences and `render_special` are forwarded through `to_rust()`.
    #[test]
    fn test_generation_config_to_rust_forwards_stop_and_render_special() {
        let py_cfg = PyGenerationConfig::new(
            64,
            None,
            Some(vec!["</s>".to_string(), "\n\n".to_string()]),
            true,
            false,
            false,
        );
        let rust_cfg = py_cfg.to_rust();
        assert_eq!(rust_cfg.max_tokens, 64);
        assert_eq!(rust_cfg.stop, vec!["</s>".to_string(), "\n\n".to_string()]);
        assert!(rust_cfg.render_special);
        assert!(!rust_cfg.add_special);
        assert!(!rust_cfg.parse_special);
    }

    /// `GenerationOutcome` conversion preserves every field and derives the
    /// convenience accessors correctly.
    #[test]
    fn test_generation_outcome_from_rust_and_accessors() {
        let rust_outcome = GenerationOutcome {
            text: "hello".to_string(),
            finish_reason: FinishReason::Stopped,
            generated_tokens: vec![1, 2, 3],
            prompt_tokens: 5,
            stop_sequence: Some("</s>".to_string()),
        };
        let py_outcome: PyGenerationOutcome = rust_outcome.into();
        assert_eq!(py_outcome.text, "hello");
        assert_eq!(py_outcome.finish_reason, PyFinishReason::Stopped);
        assert_eq!(py_outcome.completion_tokens(), 3);
        assert_eq!(py_outcome.total_tokens(), 8);
        assert_eq!(py_outcome.stop_sequence.as_deref(), Some("</s>"));
    }
}

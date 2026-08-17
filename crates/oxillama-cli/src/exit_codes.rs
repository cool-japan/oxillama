//! Process exit codes for OxiLLaMa CLI, and the typed errors this crate
//! raises directly so [`classify`] can route them without substring matching.

#![allow(dead_code)]

use std::io;

use oxillama_gguf::GgufError;
use oxillama_runtime::RuntimeError;

/// Successful execution.
pub const SUCCESS: i32 = 0;

/// Model file not found or path is invalid.
pub const ERR_MODEL_NOT_FOUND: i32 = 2;

/// Invalid or malformed configuration / arguments.
pub const ERR_INVALID_CONFIG: i32 = 3;

/// Inference engine failure (load, generate, etc.).
pub const ERR_INFERENCE_FAILED: i32 = 4;

/// HTTP server startup or runtime failure.
pub const ERR_SERVER_FAILED: i32 = 5;

/// Generic I/O error (file read, stdin, etc.).
pub const ERR_IO: i32 = 6;

/// Errors this crate raises directly (as opposed to propagating another
/// crate's typed error), carrying enough structure for [`classify`] to route
/// them without inspecting a formatted message.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// A `--model` path does not exist on disk.
    #[error("model file not found: {0}")]
    ModelNotFound(String),
}

impl CliError {
    fn exit_code(&self) -> i32 {
        match self {
            CliError::ModelNotFound(_) => ERR_MODEL_NOT_FOUND,
        }
    }
}

/// Map an `anyhow::Error` to the most appropriate exit code by walking its
/// **typed** error chain and downcasting, rather than substring-matching the
/// formatted message.
///
/// `err.chain()` yields the top-level error followed by every source
/// attached via `?` / `.context()` / `.with_context()`, in order — so a
/// [`RuntimeError`] or [`GgufError`] that was wrapped in extra context text
/// (e.g. `.with_context(|| format!("cannot load GGUF file '{path}'"))`) is
/// still found and classified correctly, even though the top-level message
/// no longer looks like the original error.
///
/// The pre-fix implementation matched fixed substrings over the formatted
/// error chain in a fixed order, which misrouted in ways that were easy to
/// trigger and hard to predict:
/// - A failed `--file` prompt read produced "reading prompt file '…': No
///   such file or directory", which matched the `"model file not found"` /
///   `"No such file"` branch and exited 2 (`ERR_MODEL_NOT_FOUND`) even
///   though no model was involved — reading the prompt file happens before
///   the model is ever touched.
/// - The `inference` branch's substrings (`"inference"`, `"generate"`,
///   `"load_model"`, `"engine"`) are broad enough to swallow almost any
///   message that mentions those words anywhere in its path or text, so the
///   `ERR_IO` branch that follows it was nearly unreachable regardless of
///   branch order — and any I/O error whose *path* happened to contain the
///   substring `"engine"` (e.g. a directory literally named `engine`) was
///   classified as an inference failure.
/// - A missing tokenizer produced a message matching neither branch and
///   fell through to `ERR_INVALID_CONFIG` — the "right" answer, but by
///   accident of not matching anything else, not by design.
pub fn classify(err: &anyhow::Error) -> i32 {
    for cause in err.chain() {
        if let Some(cli_err) = cause.downcast_ref::<CliError>() {
            return cli_err.exit_code();
        }
        if let Some(runtime_err) = cause.downcast_ref::<RuntimeError>() {
            return classify_runtime_error(runtime_err);
        }
        if cause.downcast_ref::<GgufError>().is_some() {
            return ERR_MODEL_NOT_FOUND;
        }
        #[cfg(feature = "server")]
        if cause
            .downcast_ref::<oxillama_server::ServerError>()
            .is_some()
        {
            return ERR_SERVER_FAILED;
        }
        if let Some(io_err) = cause.downcast_ref::<io::Error>() {
            return classify_io_error(io_err);
        }
    }

    // No typed cause matched: an ad-hoc `anyhow!`/`bail!` site with no
    // preserved source (there are a few left — e.g. TOML config parsing,
    // which propagates through `toml::de::Error` chained via `anyhow`'s
    // `Context`, or a `hub` subcommand failure), or a genuinely generic
    // usage error. These are almost always usage/config problems, so this
    // is a deliberate, documented default — not a silent catch-all reached
    // by accident, unlike the pre-fix final `else` branch.
    ERR_INVALID_CONFIG
}

fn classify_runtime_error(err: &RuntimeError) -> i32 {
    match err {
        RuntimeError::Gguf(_) | RuntimeError::ModelLoadError { .. } => ERR_MODEL_NOT_FOUND,
        RuntimeError::TokenizerNotAvailable | RuntimeError::TokenizerError { .. } => {
            ERR_INVALID_CONFIG
        }
        RuntimeError::Io(io_err) => classify_io_error(io_err),
        _ => ERR_INFERENCE_FAILED,
    }
}

fn classify_io_error(_err: &io::Error) -> i32 {
    ERR_IO
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_not_found_cli_error_classifies_as_model_not_found() {
        let err: anyhow::Error = CliError::ModelNotFound("m.gguf".to_string()).into();
        assert_eq!(classify(&err), ERR_MODEL_NOT_FOUND);
    }

    #[test]
    fn bare_prompt_file_io_error_does_not_classify_as_model_not_found() {
        // Regression test for the pre-fix substring-matching bug: a failed
        // `--file` prompt read used to produce "reading prompt file '…': No
        // such file or directory" and get classified as ERR_MODEL_NOT_FOUND
        // via a substring match on "No such file" — even though no model
        // was involved. `main.rs`'s prompt-file read now uses
        // `.with_context(...)`, which preserves the io::Error as a typed
        // source rather than only interpolating its text, so this must
        // classify as ERR_IO, never ERR_MODEL_NOT_FOUND.
        let io_err = io::Error::new(io::ErrorKind::NotFound, "No such file or directory");
        let err: anyhow::Error =
            anyhow::Error::new(io_err).context("reading prompt file '/tmp/does-not-exist.txt'");
        assert_eq!(
            classify(&err),
            ERR_IO,
            "an io::Error reading a prompt file must classify as ERR_IO, not ERR_MODEL_NOT_FOUND"
        );
    }

    #[test]
    fn runtime_gguf_error_classifies_as_model_not_found() {
        let gguf_err = GgufError::InvalidMagic { magic: 0xDEAD_BEEF };
        let runtime_err: RuntimeError = gguf_err.into();
        let err: anyhow::Error = runtime_err.into();
        assert_eq!(classify(&err), ERR_MODEL_NOT_FOUND);
    }

    #[test]
    fn bare_gguf_error_classifies_as_model_not_found() {
        let gguf_err = GgufError::TensorNotFound {
            name: "token_embd.weight".to_string(),
        };
        let err: anyhow::Error = gguf_err.into();
        assert_eq!(classify(&err), ERR_MODEL_NOT_FOUND);
    }

    #[test]
    fn tokenizer_not_available_classifies_as_invalid_config() {
        // A missing tokenizer must classify as ERR_INVALID_CONFIG by design
        // (an explicit match arm), not by falling through every other
        // branch and landing there by accident.
        let err: anyhow::Error = RuntimeError::TokenizerNotAvailable.into();
        assert_eq!(classify(&err), ERR_INVALID_CONFIG);
    }

    #[test]
    fn runtime_error_wrapped_in_context_still_classifies_correctly() {
        // `.with_context()` prepends an opaque string error ahead of the
        // typed cause in the chain; classify() must still find it.
        let err: anyhow::Error =
            anyhow::Error::new(RuntimeError::ModelNotLoaded).context("starting inference");
        assert_eq!(classify(&err), ERR_INFERENCE_FAILED);
    }

    #[test]
    fn path_containing_engine_does_not_force_inference_classification() {
        // Regression test for the pre-fix bug: any I/O error whose path
        // happened to contain the substring "engine" was classified as an
        // inference failure purely because the *formatted message*
        // contained that word. A plain io::Error about a path named
        // "engine" must classify as ERR_IO.
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "permission denied");
        let err: anyhow::Error =
            anyhow::Error::new(io_err).context("reading /home/user/engine/notes.txt");
        assert_eq!(classify(&err), ERR_IO);
    }

    #[test]
    fn ad_hoc_anyhow_message_with_no_typed_source_defaults_to_invalid_config() {
        let err = anyhow::anyhow!("something went generically wrong");
        assert_eq!(classify(&err), ERR_INVALID_CONFIG);
    }
}

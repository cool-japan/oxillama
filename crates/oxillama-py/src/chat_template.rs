//! Pure-Rust chat template engine for common HuggingFace prompt formats.
//!
//! Supports the three most widely deployed formats:
//!
//! | name      | Used by                              |
//! |-----------|--------------------------------------|
//! | `chatml`  | Qwen, Mistral-Instruct (new), etc.   |
//! | `llama3`  | Meta LLaMA-3 family                  |
//! | `alpaca`  | WizardLM, Alpaca-style fine-tunes    |

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Apply a named chat template to a list of messages.
///
/// # Arguments
/// * `template`      – one of `"chatml"`, `"llama3"`, `"alpaca"` (case-insensitive)
/// * `messages`      – list of Python dicts with `"role"` and `"content"` keys
/// * `add_gen_prompt` – when `true`, append the beginning-of-assistant-turn marker
///
/// # Errors
/// Returns `Err` if an unknown template name is given, or if a message dict
/// is missing `"role"` or `"content"`.
pub fn apply_template(
    template: &str,
    messages: &[Bound<'_, PyDict>],
    add_gen_prompt: bool,
) -> PyResult<String> {
    // Extract (role, content) pairs from Python dicts.
    let pairs: Vec<(String, String)> = messages
        .iter()
        .map(|msg| {
            let role = get_role(msg)?;
            let content = get_content(msg)?;
            Ok((role, content))
        })
        .collect::<PyResult<_>>()?;

    let pair_refs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(r, c)| (r.as_str(), c.as_str()))
        .collect();

    match template.to_ascii_lowercase().as_str() {
        "chatml" => Ok(format_chatml(&pair_refs, add_gen_prompt)),
        "llama3" | "llama-3" => Ok(format_llama3(&pair_refs, add_gen_prompt)),
        "alpaca" => Ok(format_alpaca(&pair_refs, add_gen_prompt)),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown chat template '{other}'. Supported: chatml, llama3, alpaca"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the `"role"` string from a message dict.
fn get_role(msg: &Bound<'_, PyDict>) -> PyResult<String> {
    msg.get_item("role")?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Each message dict must have a 'role' key")
        })?
        .extract::<String>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("'role' must be a string"))
}

/// Extract the `"content"` string from a message dict.
fn get_content(msg: &Bound<'_, PyDict>) -> PyResult<String> {
    msg.get_item("content")?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Each message dict must have a 'content' key")
        })?
        .extract::<String>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("'content' must be a string"))
}

// ---------------------------------------------------------------------------
// Pure-Rust formatting core (testable without Python interpreter)
// ---------------------------------------------------------------------------

/// Format a sequence of `(role, content)` pairs using the ChatML template.
pub(crate) fn format_chatml(messages: &[(&str, &str)], add_gen_prompt: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    if add_gen_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// Format a sequence of `(role, content)` pairs using the LLaMA-3 template.
pub(crate) fn format_llama3(messages: &[(&str, &str)], add_gen_prompt: bool) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for (role, content) in messages {
        out.push_str(&format!(
            "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        ));
    }
    if add_gen_prompt {
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    out
}

/// Format a sequence of `(role, content)` pairs using the Alpaca template.
pub(crate) fn format_alpaca(messages: &[(&str, &str)], add_gen_prompt: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        let header = match role.to_ascii_lowercase().as_str() {
            "system" => "System",
            "user" => "User",
            "assistant" => "Assistant",
            _ => "Unknown",
        };
        out.push_str(&format!("### {header}:\n{content}\n\n"));
    }
    if add_gen_prompt {
        out.push_str("### Assistant:\n");
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{format_alpaca, format_chatml, format_llama3};

    #[test]
    fn test_chatml_empty_no_gen_prompt() {
        assert_eq!(format_chatml(&[], false), "");
    }

    #[test]
    fn test_chatml_empty_with_gen_prompt() {
        assert_eq!(format_chatml(&[], true), "<|im_start|>assistant\n");
    }

    #[test]
    fn test_chatml_single_user_message() {
        let result = format_chatml(&[("user", "Hello!")], false);
        assert_eq!(result, "<|im_start|>user\nHello!<|im_end|>\n");
    }

    #[test]
    fn test_llama3_begins_with_bos() {
        assert!(format_llama3(&[], false).starts_with("<|begin_of_text|>"));
    }

    #[test]
    fn test_llama3_gen_prompt_ends_with_assistant_header() {
        let result = format_llama3(&[], true);
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_llama3_single_message_contains_role_content_eot() {
        let result = format_llama3(&[("user", "hi")], false);
        assert!(result.contains("user"));
        assert!(result.contains("hi"));
        assert!(result.contains("<|eot_id|>"));
    }

    #[test]
    fn test_alpaca_gen_prompt_ends_with_assistant() {
        assert!(format_alpaca(&[], true).ends_with("### Assistant:\n"));
    }

    #[test]
    fn test_alpaca_system_role() {
        let result = format_alpaca(&[("system", "Be helpful.")], false);
        assert!(result.contains("### System:"));
        assert!(result.contains("Be helpful."));
    }

    #[test]
    fn test_alpaca_user_role() {
        let result = format_alpaca(&[("user", "What is 2+2?")], false);
        assert!(result.contains("### User:"));
    }

    #[test]
    fn test_alpaca_unknown_role_defaults_to_unknown() {
        let result = format_alpaca(&[("moderator", "Note.")], false);
        assert!(result.contains("### Unknown:"));
    }
}

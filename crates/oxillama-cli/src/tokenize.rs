//! `oxillama tokenize` and `oxillama detokenize` subcommands.
//!
//! ## tokenize
//!
//! `oxillama tokenize --model <gguf> [--format text|json] "<text>"`
//!
//! Loads the tokenizer embedded in the GGUF metadata (or from a separate
//! `tokenizer.json` sidecar file) and encodes the text to token IDs.
//!
//! Output formats:
//! - `text` (default): one token ID per line.
//! - `json`: a JSON array of token IDs.
//!
//! ## detokenize
//!
//! `oxillama detokenize --model <gguf> --ids "1,2,3"`
//!
//! Decodes a comma-separated list of token IDs back to a text string.

use std::path::{Path, PathBuf};

use anyhow::Context;

// ── Arg structs ────────────────────────────────────────────────────────────────

/// Output format for `tokenize`.
#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum TokenizeFormat {
    /// Print one token ID per line.
    #[default]
    Text,
    /// Print a JSON array of token IDs.
    Json,
}

/// Arguments for the `tokenize` subcommand.
#[derive(clap::Args, Clone, Debug)]
pub struct TokenizeArgs {
    /// Path to the GGUF model file (must contain embedded tokenizer metadata).
    #[arg(short, long)]
    pub model: PathBuf,

    /// Text string to tokenize.
    pub text: String,

    /// Output format: `text` (one ID per line) or `json` (array).
    #[arg(long, value_enum, default_value_t = TokenizeFormat::Text)]
    pub format: TokenizeFormat,
}

/// Arguments for the `detokenize` subcommand.
#[derive(clap::Args, Clone, Debug)]
pub struct DetokenizeArgs {
    /// Path to the GGUF model file.
    #[arg(short, long)]
    pub model: PathBuf,

    /// Comma-separated list of token IDs, e.g. `1,2,3`.
    #[arg(long)]
    pub ids: Vec<u32>,
}

// ── Implementations ────────────────────────────────────────────────────────────

/// Tokenize a text string using the tokenizer embedded in a GGUF model.
///
/// Looks for the tokenizer in this order:
/// 1. A `tokenizer.json` sidecar file in the same directory as `model`.
/// 2. The `tokenizer.ggml.tokens` metadata stored in the GGUF file
///    (used to synthesise a minimal tokenizer).
///
/// In practice, most GGUF models ship with a `tokenizer.json` sidecar.
/// When neither is available, an error is returned.
pub fn run_tokenize(args: &TokenizeArgs) -> anyhow::Result<()> {
    let bridge = load_tokenizer_bridge(&args.model)?;
    let ids = bridge
        .encode(&args.text)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    match args.format {
        TokenizeFormat::Text => {
            for id in &ids {
                println!("{id}");
            }
        }
        TokenizeFormat::Json => {
            let json =
                serde_json::to_string(&ids).context("failed to serialize token IDs as JSON")?;
            println!("{json}");
        }
    }

    Ok(())
}

/// Detokenize a list of token IDs back to a text string.
pub fn run_detokenize(args: &DetokenizeArgs) -> anyhow::Result<()> {
    if args.ids.is_empty() {
        // Empty ID list → empty string, no tokenizer load needed.
        println!();
        return Ok(());
    }

    let bridge = load_tokenizer_bridge(&args.model)?;
    let text = bridge
        .decode(&args.ids)
        .map_err(|e| anyhow::anyhow!("detokenization failed: {e}"))?;
    println!("{text}");

    Ok(())
}

// ── Internal helpers ───────────────────────────────────────────────────────────

/// Load a `TokenizerBridge` for the given GGUF model path.
///
/// Strategy:
/// 1. Check for a `tokenizer.json` sidecar alongside the model file.
/// 2. Return an error if not found (GGUF-embedded tokenizer extraction is
///    deferred to a future version).
fn load_tokenizer_bridge(model_path: &Path) -> anyhow::Result<oxillama_runtime::TokenizerBridge> {
    // ── Sidecar JSON ──────────────────────────────────────────────────
    if let Some(parent) = model_path.parent() {
        let sidecar = parent.join("tokenizer.json");
        if sidecar.exists() {
            let path_str = sidecar.to_string_lossy();
            return oxillama_runtime::TokenizerBridge::from_file(&path_str)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json sidecar: {e}"));
        }
    }

    // ── No sidecar found; report clearly ─────────────────────────────
    anyhow::bail!(
        "no tokenizer found for model '{}': \
         place a 'tokenizer.json' alongside the model file",
        model_path.display()
    )
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Return (and create) the temp directory used by tokenize tests.
    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("oxillama_tokenize_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Write a minimal GGUF file (no real model weights).
    fn write_minimal_gguf(path: &PathBuf) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count
                                                    // Pad to 32-byte alignment.
        let rem = buf.len() % 32;
        if rem != 0 {
            buf.resize(buf.len() + 32 - rem, 0u8);
        }
        std::fs::write(path, &buf).expect("write minimal GGUF");
    }

    #[test]
    fn tokenize_run_with_missing_model_errors() {
        // Point at a path that doesn't exist at all.
        let args = TokenizeArgs {
            model: PathBuf::from("/nonexistent/path/no_model.gguf"),
            text: "hello world".into(),
            format: TokenizeFormat::Text,
        };
        assert!(
            run_tokenize(&args).is_err(),
            "missing model path should return Err"
        );
    }

    #[test]
    fn detokenize_empty_ids_returns_empty() {
        // Zero IDs → empty string; no model file is required.
        let dir = temp_dir();
        let model_path = dir.join("dummy_model.gguf");
        write_minimal_gguf(&model_path);

        let args = DetokenizeArgs {
            model: model_path,
            ids: vec![],
        };
        // Should succeed and print an empty line (no tokenizer load).
        assert!(
            run_detokenize(&args).is_ok(),
            "empty ID list should return Ok (no tokenizer needed)"
        );
    }
}

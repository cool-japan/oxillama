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
/// Strategy (mirrors `oxillama_runtime::engine::load_tokenizer`'s priority
/// order, since a stock HuggingFace GGUF must tokenize identically whether it
/// runs through `oxillama run` or `oxillama tokenize`):
///
/// 1. The vocabulary embedded in the GGUF file itself
///    (`tokenizer.ggml.tokens` and friends) — this is the vocabulary the
///    weights were quantized against, so it always agrees with the model's
///    special-token ids and decodes byte-exactly. It deliberately outranks a
///    sidecar.
/// 2. A `tokenizer.json` sidecar alongside the model file, if the GGUF has no
///    embedded vocabulary or it could not be used.
///
/// Returns an error only when neither source is available.
fn load_tokenizer_bridge(model_path: &Path) -> anyhow::Result<oxillama_runtime::TokenizerBridge> {
    let model = oxillama_gguf::GgufModel::load(model_path)
        .with_context(|| format!("failed to load GGUF model '{}'", model_path.display()))?;
    let metadata = &model.file.metadata;

    // ── 1. GGUF-embedded vocabulary ──────────────────────────────────
    if oxillama_runtime::TokenizerBridge::metadata_has_vocab(metadata) {
        match oxillama_runtime::TokenizerBridge::from_gguf_metadata(metadata) {
            Ok(bridge) => return Ok(bridge),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "GGUF-embedded vocabulary could not be used; falling back to a tokenizer.json sidecar"
                );
            }
        }
    }

    // ── 2. Sidecar JSON ───────────────────────────────────────────────
    if let Some(parent) = model_path.parent() {
        let sidecar = parent.join("tokenizer.json");
        if sidecar.exists() {
            let mut bridge = oxillama_runtime::TokenizerBridge::from_path(&sidecar)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json sidecar: {e}"))?;
            bridge.apply_gguf_specials(metadata);
            return Ok(bridge);
        }
    }

    // ── Neither source available; report clearly ─────────────────────
    anyhow::bail!(
        "no tokenizer found for model '{}': the GGUF has no embedded vocabulary \
         (no 'tokenizer.ggml.tokens' metadata) and no 'tokenizer.json' sidecar \
         exists alongside the model file",
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

    // ── GGUF-embedded tokenizer regression tests (C1) ───────────────────────
    //
    // Before the fix, `load_tokenizer_bridge` only ever checked for a
    // `tokenizer.json` sidecar and `anyhow::bail!`'d otherwise — so every one
    // of these vocabulary-only GGUFs (which carry a full embedded vocabulary
    // and no sidecar) made `tokenize`/`detokenize` fail unconditionally, even
    // though `oxillama_runtime::TokenizerBridge::from_gguf_metadata` has been
    // available since the GGUF-embedded tokenizer landed. This is the
    // regression test for that: it fails against the pre-fix
    // `load_tokenizer_bridge` (sidecar-only) and passes now that the
    // GGUF-embedded vocabulary is consulted first.
    //
    // These reference files live outside the repo
    // (`~/work/refs/llama.cpp/models/ggml-vocab-*.gguf`) and are not present
    // on every machine, so each test skips (rather than fails) when the file
    // is absent — this keeps the suite portable while still exercising the
    // real fix end-to-end wherever the llama.cpp reference checkout exists.

    /// Resolve a `ggml-vocab-<name>.gguf` path under the llama.cpp reference
    /// checkout, or `None` if that checkout isn't present on this machine.
    fn vocab_gguf_path(name: &str) -> Option<PathBuf> {
        let home = dirs::home_dir()?;
        let path = home
            .join("work")
            .join("refs")
            .join("llama.cpp")
            .join("models")
            .join(format!("ggml-vocab-{name}.gguf"));
        path.exists().then_some(path)
    }

    /// Round-trip `tokenize` → `detokenize` against a real vocabulary-only
    /// GGUF, with no `tokenizer.json` sidecar anywhere nearby — the exact
    /// scenario the pre-fix code could never handle.
    fn assert_gguf_embedded_round_trip(vocab_name: &str, text: &str) {
        let Some(model) = vocab_gguf_path(vocab_name) else {
            eprintln!(
                "skipping: ~/work/refs/llama.cpp/models/ggml-vocab-{vocab_name}.gguf not present"
            );
            return;
        };

        let bridge = load_tokenizer_bridge(&model)
            .unwrap_or_else(|e| panic!("load_tokenizer_bridge({vocab_name}) should succeed from the GGUF-embedded vocabulary alone: {e}"));

        let ids = bridge
            .encode(text)
            .unwrap_or_else(|e| panic!("encode({vocab_name}) should succeed: {e}"));
        assert!(
            !ids.is_empty(),
            "{vocab_name}: encoding should produce tokens"
        );

        let decoded = bridge
            .decode(&ids)
            .unwrap_or_else(|e| panic!("decode({vocab_name}) should succeed: {e}"));
        assert!(
            decoded.contains(text.trim()) || decoded.trim() == text.trim(),
            "{vocab_name}: round trip should recover the original text; got {decoded:?} from {text:?}"
        );

        // Also exercise the actual `run_tokenize` / `run_detokenize` entry
        // points (format::Json), matching how the CLI is really invoked.
        let tok_args = TokenizeArgs {
            model: model.clone(),
            text: text.to_string(),
            format: TokenizeFormat::Json,
        };
        assert!(
            run_tokenize(&tok_args).is_ok(),
            "{vocab_name}: run_tokenize should succeed with no sidecar present"
        );

        let detok_args = DetokenizeArgs { model, ids };
        assert!(
            run_detokenize(&detok_args).is_ok(),
            "{vocab_name}: run_detokenize should succeed with no sidecar present"
        );
    }

    #[test]
    fn gguf_embedded_round_trip_llama_bpe() {
        assert_gguf_embedded_round_trip("llama-bpe", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_llama_spm() {
        assert_gguf_embedded_round_trip("llama-spm", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_qwen2() {
        assert_gguf_embedded_round_trip("qwen2", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_phi_3() {
        assert_gguf_embedded_round_trip("phi-3", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_gpt_2() {
        assert_gguf_embedded_round_trip("gpt-2", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_starcoder() {
        assert_gguf_embedded_round_trip("starcoder", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_falcon() {
        assert_gguf_embedded_round_trip("falcon", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_deepseek_coder() {
        assert_gguf_embedded_round_trip("deepseek-coder", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_command_r() {
        assert_gguf_embedded_round_trip("command-r", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_mpt() {
        assert_gguf_embedded_round_trip("mpt", "Hello, world");
    }

    #[test]
    fn gguf_embedded_round_trip_refact() {
        assert_gguf_embedded_round_trip("refact", "Hello, world");
    }
}

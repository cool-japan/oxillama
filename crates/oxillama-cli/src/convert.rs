//! `oxillama convert <input.safetensors> <output.gguf>` — safetensors → GGUF.
//!
//! Uses [`oxillama_gguf::SafetensorsConverter`] to parse a `.safetensors`
//! binary file and write a fully conformant GGUF v3 file.

use std::path::PathBuf;

use anyhow::Context;

/// Arguments for the `convert` subcommand.
#[derive(clap::Args, Clone, Debug)]
pub struct ConvertArgs {
    /// Path to the input `.safetensors` file.
    pub input: PathBuf,

    /// Path to write the output `.gguf` file.
    pub output: PathBuf,
}

/// Convert a safetensors file to GGUF format.
///
/// 1. Reads the input file as raw bytes.
/// 2. Calls `SafetensorsConverter::from_bytes` to parse and produce an
///    in-memory `GgufModel`.
/// 3. Uses `GgufWriter` to serialise the model back to a valid GGUF v3 file.
///    Metadata KV pairs and tensor data are preserved verbatim.
pub fn run_convert(args: &ConvertArgs) -> anyhow::Result<()> {
    // ── 1. Read input ─────────────────────────────────────────────────
    let input_bytes = std::fs::read(&args.input)
        .with_context(|| format!("cannot read safetensors file '{}'", args.input.display()))?;

    // ── 2. Parse safetensors → GgufModel ─────────────────────────────
    let model =
        oxillama_gguf::SafetensorsConverter::from_bytes(&input_bytes).with_context(|| {
            format!(
                "failed to parse safetensors file '{}'",
                args.input.display()
            )
        })?;

    // ── 3. Re-serialise as GGUF v3 using GgufWriter ───────────────────
    let mut writer = oxillama_gguf::GgufWriter::new();

    // Copy all metadata KV pairs.
    for (key, value) in model.file.metadata.iter() {
        writer.add_metadata(key, value.clone());
    }

    // Copy all tensors (data + metadata).
    for (name, info) in model.file.tensors.iter() {
        let raw = model
            .tensor_data(name)
            .with_context(|| format!("failed to read tensor data for '{name}'"))?;
        writer.add_tensor(name, &info.dimensions, info.tensor_type, raw);
    }

    // ── 4. Write output file ──────────────────────────────────────────
    writer
        .write_to_file(&args.output)
        .with_context(|| format!("failed to write GGUF file '{}'", args.output.display()))?;

    println!(
        "converted: '{}' → '{}' ({} tensors)",
        args.input.display(),
        args.output.display(),
        model.file.header.tensor_count,
    );

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal safetensors byte buffer.
    fn build_safetensors(json: &str, data: &[u8]) -> Vec<u8> {
        let json_bytes = json.as_bytes();
        let header_size = json_bytes.len() as u64;
        let mut buf = Vec::new();
        buf.extend_from_slice(&header_size.to_le_bytes());
        buf.extend_from_slice(json_bytes);
        buf.extend_from_slice(data);
        buf
    }

    /// Return the temp directory path for these tests, creating it if needed.
    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("oxillama_convert_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn convert_invalid_input_errors() {
        // Write a file that is definitely not safetensors format.
        let dir = temp_dir();
        let input = dir.join("not_safetensors.bin");
        let output = dir.join("not_safetensors_out.gguf");
        std::fs::write(&input, b"this is not a safetensors file!!!").expect("write");

        let args = ConvertArgs { input, output };
        assert!(
            run_convert(&args).is_err(),
            "non-safetensors input should return Err"
        );
    }

    #[test]
    fn convert_empty_safetensors_produces_gguf() {
        // An empty safetensors (just __metadata__, no tensors) should produce
        // a valid GGUF file on disk.
        let json = r#"{"__metadata__": {"format": "pt"}}"#;
        let st_bytes = build_safetensors(json, &[]);

        let dir = temp_dir();
        let input = dir.join("empty.safetensors");
        let output = dir.join("empty_out.gguf");
        std::fs::write(&input, &st_bytes).expect("write safetensors");

        let args = ConvertArgs {
            input,
            output: output.clone(),
        };
        run_convert(&args).expect("convert empty safetensors should succeed");
        assert!(output.exists(), "output GGUF file should be created");

        // Verify the output is a valid GGUF.
        let out_bytes = std::fs::read(&output).expect("read output");
        assert!(
            out_bytes.starts_with(b"GGUF"),
            "output must start with GGUF magic"
        );
    }
}

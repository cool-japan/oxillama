//! `oxillama quantize <input.gguf> <output.gguf> --target <TYPE>` — re-quantize.
//!
//! Loads an existing GGUF file, dequantizes every tensor to FP32, then
//! re-quantizes to the requested target format and writes a new GGUF v3 file.
//!
//! ## Supported encoding targets
//!
//! | CLI flag  | GGUF type   | Status |
//! |-----------|-------------|--------|
//! | `Q4_0`    | Q4_0        | supported — `quantize_f32_to_q4_0` |
//! | `Q8_0`    | Q8_0        | supported — `quantize_f32_to_q8_0` |
//! | `Q4_K_M`  | Q4_K (Q4K)  | not encodable — returns typed `Err` |
//! | `Q5_K_M`  | Q5_K (Q5K)  | not encodable — returns typed `Err` |
//! | `Q6_K`    | Q6_K (Q6K)  | not encodable — returns typed `Err` |
//!
//! K-quant formats (Q4_K, Q5_K, Q6_K) use a complex super-block layout with
//! nested 6-bit scale quantization.  Reference *dequantization* kernels for
//! all three exist in `oxillama-quant`, but the inverse *quantization* (FP32 →
//! K-quant) is not yet implemented.  Passing one of these targets is not a
//! programming error — `run_quantize` returns a descriptive `anyhow::Error`
//! rather than panicking or using `unimplemented!()`.

use std::path::PathBuf;

use anyhow::Context;
use oxillama_gguf::GgufTensorType;
use oxillama_quant::dequantize_to_f32;

/// CLI-level quantization target enum.
#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum QuantTarget {
    /// Legacy 4-bit quantization (18 bytes / 32 weights).
    #[value(name = "Q4_0")]
    Q4_0,
    /// K-quant 4-bit medium (not yet encodable — returns error).
    #[value(name = "Q4_K_M")]
    Q4Km,
    /// K-quant 5-bit medium (not yet encodable — returns error).
    #[value(name = "Q5_K_M")]
    Q5Km,
    /// K-quant 6-bit (not yet encodable — returns error).
    #[value(name = "Q6_K")]
    Q6K,
    /// 8-bit quantization (34 bytes / 32 weights).
    #[value(name = "Q8_0")]
    Q8_0,
}

impl QuantTarget {
    /// Returns the corresponding `GgufTensorType` when an encoder is available,
    /// or `None` for targets whose *encoding* (FP32 → quantized bytes) is not
    /// yet implemented.
    ///
    /// Callers convert `None` into a descriptive `anyhow::Error` via `ok_or_else`;
    /// this must never panic or use `unimplemented!()`.
    fn as_tensor_type(&self) -> Option<GgufTensorType> {
        match self {
            QuantTarget::Q4_0 => Some(GgufTensorType::Q4_0),
            QuantTarget::Q8_0 => Some(GgufTensorType::Q8_0),
            // K-quant formats have reference dequantization kernels in oxillama-quant
            // but no matching encoder (FP32 → K-quant block bytes).  Returns None so
            // `run_quantize` produces a descriptive error rather than panicking.
            QuantTarget::Q4Km | QuantTarget::Q5Km | QuantTarget::Q6K => None,
        }
    }

    /// Human-readable target name for error messages.
    fn display_name(&self) -> &'static str {
        match self {
            QuantTarget::Q4_0 => "Q4_0",
            QuantTarget::Q4Km => "Q4_K_M",
            QuantTarget::Q5Km => "Q5_K_M",
            QuantTarget::Q6K => "Q6_K",
            QuantTarget::Q8_0 => "Q8_0",
        }
    }
}

/// Arguments for the `quantize` subcommand.
#[derive(clap::Args, Clone, Debug)]
pub struct QuantizeArgs {
    /// Path to the source GGUF file.
    pub input: PathBuf,

    /// Path to write the re-quantized GGUF file.
    pub output: PathBuf,

    /// Target quantization format.
    #[arg(long, value_enum)]
    pub target: QuantTarget,
}

/// Re-quantize an existing GGUF file to a different format.
///
/// Algorithm per tensor:
/// 1. Dequantize existing bytes → `Vec<f32>` using `dequantize_to_f32`.
/// 2. Pad the f32 slice to a multiple of 32 (the block size for Q4_0 / Q8_0).
/// 3. Re-quantize using the appropriate `quantize_f32_to_*` function.
/// 4. Write a new GGUF v3 file with the same metadata, re-quantized tensor data,
///    and updated `tensor_type` fields.
pub fn run_quantize(args: &QuantizeArgs) -> anyhow::Result<()> {
    // ── 1. Validate target has an encoder ─────────────────────────────
    let target_type = args.target.as_tensor_type().ok_or_else(|| {
        anyhow::anyhow!(
            "quantization encoding for '{}' is not yet implemented; \
             only Q4_0 and Q8_0 are currently supported as output targets",
            args.target.display_name()
        )
    })?;

    // ── 2. Load the input GGUF ─────────────────────────────────────────
    let model = oxillama_gguf::GgufModel::load(&args.input)
        .with_context(|| format!("cannot load GGUF file '{}'", args.input.display()))?;

    // ── 3. Build the output writer, copying metadata ───────────────────
    let mut writer = oxillama_gguf::GgufWriter::new();
    for (key, value) in model.file.metadata.iter() {
        writer.add_metadata(key, value.clone());
    }

    // ── 4. Re-quantize each tensor ────────────────────────────────────
    for (name, info) in model.file.tensors.iter() {
        let raw = model
            .tensor_data(name)
            .with_context(|| format!("failed to read tensor '{name}'"))?;

        let n_elements = info.n_elements() as usize;

        // Dequantize to f32.
        let f32_values = dequantize_to_f32(raw, info.tensor_type, n_elements)
            .with_context(|| format!("failed to dequantize tensor '{name}'"))?;

        // Pad to block-size multiple (32 for Q4_0 / Q8_0).
        let block_size = target_type.block_size();
        let padded = pad_to_multiple(&f32_values, block_size);

        // Encode to target format.
        let quantized = encode_to_target(&padded, target_type)
            .with_context(|| format!("failed to quantize tensor '{name}' to {target_type}"))?;

        writer.add_tensor(name, &info.dimensions, target_type, &quantized);
    }

    // ── 5. Write output ────────────────────────────────────────────────
    writer
        .write_to_file(&args.output)
        .with_context(|| format!("failed to write output '{}'", args.output.display()))?;

    println!(
        "quantized: '{}' → '{}' (target {}, {} tensors)",
        args.input.display(),
        args.output.display(),
        args.target.display_name(),
        model.file.header.tensor_count,
    );

    Ok(())
}

/// Pad an f32 slice to the next multiple of `block_size` by appending zeros.
fn pad_to_multiple(values: &[f32], block_size: usize) -> Vec<f32> {
    if block_size == 0 || values.len().is_multiple_of(block_size) {
        return values.to_vec();
    }
    let needed = block_size - (values.len() % block_size);
    let mut padded = values.to_vec();
    padded.resize(values.len() + needed, 0.0f32);
    padded
}

/// Encode an f32 slice into the bytes for `target_type`.
///
/// Only Q4_0 and Q8_0 are supported; all other types return an error.
fn encode_to_target(values: &[f32], target_type: GgufTensorType) -> anyhow::Result<Vec<u8>> {
    match target_type {
        GgufTensorType::Q4_0 => oxillama_quant::quantize_f32_to_q4_0(values)
            .map_err(|e| anyhow::anyhow!("Q4_0 encoding error: {e}")),
        GgufTensorType::Q8_0 => oxillama_quant::quantize_f32_to_q8_0(values)
            .map_err(|e| anyhow::anyhow!("Q8_0 encoding error: {e}")),
        other => {
            anyhow::bail!("encode_to_target: unsupported type {other} — this should not be reached")
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::{GgufWriter, MetadataValue};

    /// Build and write a minimal GGUF with a single F32 tensor.
    ///
    /// The tensor contains `n_elements` F32 values set to `fill_value`.
    /// Returns the path of the written temp file.
    fn write_minimal_f32_gguf(filename: &str, n_elements: usize, fill_value: f32) -> PathBuf {
        let dir = std::env::temp_dir().join("oxillama_quantize_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let data: Vec<f32> = vec![fill_value; n_elements];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut writer = GgufWriter::new();
        writer.add_metadata("general.architecture", MetadataValue::String("test".into()));
        writer.add_tensor(
            "test.weight",
            &[n_elements as u64],
            GgufTensorType::F32,
            &bytes,
        );

        let path = dir.join(filename);
        writer.write_to_file(&path).expect("write test GGUF");
        path
    }

    /// Return the temp directory used by quantize tests.
    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("oxillama_quantize_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn quantize_unsupported_target_errors() {
        let input = write_minimal_f32_gguf("unsupported_in.gguf", 32, 0.5);
        let output = temp_dir().join("unsupported_out.gguf");

        // Q4_K_M has no encoder yet; must return Err.
        let args = QuantizeArgs {
            input,
            output,
            target: QuantTarget::Q4Km,
        };
        assert!(
            run_quantize(&args).is_err(),
            "Q4_K_M target should return Err (no encoder)"
        );
    }

    #[test]
    fn quantize_q8_0_output_path_written() {
        // 32 elements → exactly 1 Q8_0 block, no padding needed.
        let input = write_minimal_f32_gguf("q8_0_in.gguf", 32, 1.0);
        let output = temp_dir().join("q8_0_out.gguf");

        let args = QuantizeArgs {
            input,
            output: output.clone(),
            target: QuantTarget::Q8_0,
        };
        run_quantize(&args).expect("Q8_0 quantization should succeed");
        assert!(output.exists(), "Q8_0 output file should be created");
    }
}

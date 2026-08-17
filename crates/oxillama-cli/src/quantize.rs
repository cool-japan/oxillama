//! `oxillama quantize <input.gguf> <output.gguf> --target <TYPE>` — re-quantize.
//!
//! Reproduces llama.cpp's `llama_model_quantize` for the mixtures OxiLLaMa can
//! encode. A target such as `Q4_K_M` is a *mixture*, not a single type: the
//! output head, some value projections and some FFN down projections are
//! promoted to higher precision by [`type_select::TypeSelector`], a literal
//! port of `llama_tensor_get_type`.
//!
//! ## Supported targets
//!
//! | `--target` | bulk type | notes                                          |
//! |------------|-----------|------------------------------------------------|
//! | `Q4_0`     | Q4_0      | output head promoted to Q6_K                    |
//! | `Q5_0`     | Q5_0      | output head promoted to Q6_K                    |
//! | `Q5_1`     | Q5_1      | output head promoted to Q6_K                    |
//! | `Q8_0`     | Q8_0      | output head stays Q8_0                          |
//! | `Q2_K`     | Q2_K      | `attn_v`/`ffn_down`/`attn_output` promoted      |
//! | `Q3_K_S/M/L` | Q3_K    | increasingly generous promotions                |
//! | `Q4_K_S`   | Q4_K      | first four `attn_v` and first eighth of `ffn_down` at Q5_K |
//! | `Q4_K_M`   | Q4_K      | `use_more_bits` layers' `attn_v`/`ffn_down` at Q6_K |
//! | `Q5_K_S`   | Q5_K      | output head at Q6_K                             |
//! | `Q5_K_M`   | Q5_K      | `use_more_bits` layers' `ffn_down` at Q6_K      |
//! | `Q6_K`     | Q6_K      | uniform                                         |
//!
//! ## Metadata
//!
//! Every metadata key/value in the input is copied to the output verbatim —
//! including the whole `tokenizer.ggml.*` block, so a quantized model stays
//! self-contained and needs no sidecar tokenizer. Three keys change:
//! `general.file_type` (to the new `llama_ftype`) and
//! `general.quantization_version`, both of which llama.cpp also rewrites, and
//! `general.alignment`, which is set to the alignment `GgufWriter` actually
//! pads to so the file cannot describe a layout it does not have.
//!
//! ## Memory
//!
//! Tensors are streamed: the header and tensor-info section are written from
//! the declared shapes first, then each tensor is dequantized, re-encoded and
//! flushed one at a time. Peak memory is one tensor's FP32 expansion plus its
//! encoded bytes, not the whole model.

#[cfg(test)]
mod mixture_tests;
mod type_select;

use std::path::PathBuf;

use anyhow::Context;
use oxillama_gguf::{GgufTensorType, MetadataValue};
use oxillama_quant::{dequantize_to_f32, quantize_f32_rows};

use type_select::{Fallback, Ftype, ModelInfo, TypeSelector};

/// `GGML_QNT_VERSION` — the value llama.cpp stamps into every file it
/// quantizes.
const QUANTIZATION_VERSION: u32 = 2;

/// The tensor-data alignment `GgufWriter` actually emits.
///
/// The writer pads to `oxillama_gguf`'s `GGUF_DEFAULT_ALIGNMENT` and does not
/// take an override, so `general.alignment` in the output must say the same.
const WRITER_ALIGNMENT: u32 = 32;

/// CLI-level quantization target enum.
///
/// The names match llama.cpp's `--type` values so a command line can be moved
/// between the two tools unchanged.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantTarget {
    /// Legacy 4-bit quantization (18 bytes / 32 weights).
    #[value(name = "Q4_0")]
    Q4_0,
    /// Legacy 5-bit quantization (22 bytes / 32 weights).
    #[value(name = "Q5_0")]
    Q5_0,
    /// Legacy 5-bit quantization with a per-block minimum (24 bytes / 32).
    #[value(name = "Q5_1")]
    Q5_1,
    /// 8-bit quantization (34 bytes / 32 weights).
    #[value(name = "Q8_0")]
    Q8_0,
    /// K-quant 2-bit.
    #[value(name = "Q2_K")]
    Q2K,
    /// K-quant 3-bit, small.
    #[value(name = "Q3_K_S")]
    Q3Ks,
    /// K-quant 3-bit, medium.
    #[value(name = "Q3_K_M")]
    Q3Km,
    /// K-quant 3-bit, large.
    #[value(name = "Q3_K_L")]
    Q3Kl,
    /// K-quant 4-bit, small.
    #[value(name = "Q4_K_S")]
    Q4Ks,
    /// K-quant 4-bit, medium — the most common local-inference mixture.
    #[value(name = "Q4_K_M")]
    Q4Km,
    /// K-quant 5-bit, small.
    #[value(name = "Q5_K_S")]
    Q5Ks,
    /// K-quant 5-bit, medium.
    #[value(name = "Q5_K_M")]
    Q5Km,
    /// K-quant 6-bit.
    #[value(name = "Q6_K")]
    Q6K,
}

impl QuantTarget {
    /// The `llama_ftype` this target corresponds to.
    fn ftype(self) -> Ftype {
        match self {
            QuantTarget::Q4_0 => Ftype::Q4_0,
            QuantTarget::Q5_0 => Ftype::Q5_0,
            QuantTarget::Q5_1 => Ftype::Q5_1,
            QuantTarget::Q8_0 => Ftype::Q8_0,
            QuantTarget::Q2K => Ftype::Q2K,
            QuantTarget::Q3Ks => Ftype::Q3KS,
            QuantTarget::Q3Km => Ftype::Q3KM,
            QuantTarget::Q3Kl => Ftype::Q3KL,
            QuantTarget::Q4Ks => Ftype::Q4KS,
            QuantTarget::Q4Km => Ftype::Q4KM,
            QuantTarget::Q5Ks => Ftype::Q5KS,
            QuantTarget::Q5Km => Ftype::Q5KM,
            QuantTarget::Q6K => Ftype::Q6K,
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

    /// Allow reading weights that are already quantized.
    ///
    /// Quantizing an already-quantized model compounds the error of both
    /// passes; llama.cpp refuses by default and so does this. Quantize from
    /// the F16/BF16/F32 checkpoint when you have it.
    #[arg(long)]
    pub allow_requantize: bool,

    /// Re-encode even tensors whose type already matches the target.
    ///
    /// llama.cpp (and this tool by default) copies such tensors through
    /// untouched, since decoding and re-encoding them can only lose
    /// precision. This flag forces the round trip anyway — it exists to
    /// exercise the encoders end to end. Implies `--allow-requantize`.
    #[arg(long)]
    pub force_requantize: bool,

    /// Quantize every tensor to the bulk type, with no per-tensor promotions.
    ///
    /// Mirrors llama.cpp's `--pure`. Also disables the incompatible-shape
    /// fallback, exactly as llama.cpp does — both live behind the same guard.
    #[arg(long)]
    pub pure: bool,

    /// Print the per-tensor plan and the resulting size without writing.
    #[arg(long)]
    pub dry_run: bool,
}

/// What the driver decided to do with one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Copy the source bytes verbatim, keeping the source type.
    Copy,
    /// Dequantize to FP32 and re-encode as this type.
    Encode(GgufTensorType),
}

/// One planned output tensor.
struct Planned {
    name: String,
    dimensions: Vec<u64>,
    src_type: GgufTensorType,
    dst_type: GgufTensorType,
    action: Action,
    src_bytes: u64,
    dst_bytes: u64,
}

/// Re-quantize an existing GGUF file to a different format.
pub fn run_quantize(args: &QuantizeArgs) -> anyhow::Result<()> {
    let ftype = args.target.ftype();

    let model = oxillama_gguf::GgufModel::load(&args.input)
        .with_context(|| format!("cannot load GGUF file '{}'", args.input.display()))?;

    let info = read_model_info(&model)?;
    let plan = build_plan(&model, &info, args, ftype)?;

    let total_src: u64 = plan.iter().map(|p| p.src_bytes).sum();
    let total_dst: u64 = plan.iter().map(|p| p.dst_bytes).sum();

    for (idx, p) in plan.iter().enumerate() {
        match p.action {
            Action::Copy => eprintln!(
                "[{:4}/{:4}] {:40} {:?} {} -> copy ({:.2} MiB)",
                idx + 1,
                plan.len(),
                p.name,
                p.dimensions,
                p.src_type,
                mib(p.src_bytes),
            ),
            Action::Encode(_) => eprintln!(
                "[{:4}/{:4}] {:40} {:?} {} -> {} ({:.2} -> {:.2} MiB)",
                idx + 1,
                plan.len(),
                p.name,
                p.dimensions,
                p.src_type,
                p.dst_type,
                mib(p.src_bytes),
                mib(p.dst_bytes),
            ),
        }
    }

    if args.dry_run {
        println!(
            "dry run: '{}' -> {} would be {:.2} MiB (from {:.2} MiB), {} tensors",
            args.input.display(),
            ftype.name(),
            mib(total_dst),
            mib(total_src),
            plan.len(),
        );
        return Ok(());
    }

    // ── Header: every metadata key verbatim, two keys rewritten ───────────
    let mut writer = oxillama_gguf::GgufWriter::new();
    // The reader stores metadata in a `HashMap`, whose iteration order is
    // arbitrary; sorting makes the output byte-reproducible across runs.
    let mut kvs: Vec<(&String, &MetadataValue)> = model.file.metadata.iter().collect();
    kvs.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in kvs {
        let value = match key.as_str() {
            "general.file_type" => MetadataValue::Uint32(ftype.file_type_value()),
            "general.quantization_version" => MetadataValue::Uint32(QUANTIZATION_VERSION),
            // `GgufWriter` pads tensor data to `GGUF_DEFAULT_ALIGNMENT`
            // unconditionally. Copying a different `general.alignment` through
            // would leave the output claiming a layout it does not have, and a
            // reader that trusted the key would compute every tensor offset
            // wrong. Rewrite it to the alignment actually used.
            "general.alignment" => {
                if value.as_u32() != Some(WRITER_ALIGNMENT) {
                    eprintln!(
                        "warning: input declares general.alignment = {value}; the output is \
                         written with {WRITER_ALIGNMENT}-byte alignment and its metadata is \
                         updated to match"
                    );
                }
                MetadataValue::Uint32(WRITER_ALIGNMENT)
            }
            _ => value.clone(),
        };
        writer.add_metadata(key, value);
    }
    if model.file.metadata.get("general.file_type").is_none() {
        writer.add_metadata(
            "general.file_type",
            MetadataValue::Uint32(ftype.file_type_value()),
        );
    }
    if model
        .file
        .metadata
        .get("general.quantization_version")
        .is_none()
    {
        writer.add_metadata(
            "general.quantization_version",
            MetadataValue::Uint32(QUANTIZATION_VERSION),
        );
    }

    for p in &plan {
        writer
            .declare_tensor(&p.name, &p.dimensions, p.dst_type)
            .with_context(|| format!("cannot declare tensor '{}'", p.name))?;
    }

    let mut data_writer = writer
        .into_file_data_writer(&args.output)
        .with_context(|| format!("failed to create output '{}'", args.output.display()))?;

    let mut copied = 0usize;
    let mut encoded = 0usize;
    for p in &plan {
        let raw = model
            .tensor_data(&p.name)
            .with_context(|| format!("failed to read tensor '{}'", p.name))?;
        match p.action {
            Action::Copy => {
                data_writer
                    .write_tensor(&p.name, raw)
                    .with_context(|| format!("failed to write tensor '{}'", p.name))?;
                copied += 1;
            }
            Action::Encode(dst) => {
                let n_elements: usize = p.dimensions.iter().product::<u64>() as usize;
                let f32_values = dequantize_to_f32(raw, p.src_type, n_elements)
                    .with_context(|| format!("failed to dequantize tensor '{}'", p.name))?;
                let n_per_row = p.dimensions.first().copied().unwrap_or(0) as usize;
                let bytes = quantize_f32_rows(&f32_values, n_per_row, dst)
                    .with_context(|| format!("failed to encode tensor '{}' as {dst}", p.name))?;
                data_writer
                    .write_tensor(&p.name, &bytes)
                    .with_context(|| format!("failed to write tensor '{}'", p.name))?;
                encoded += 1;
            }
        }
    }

    data_writer
        .finish()
        .with_context(|| format!("failed to finalize output '{}'", args.output.display()))?;

    println!(
        "quantized: '{}' -> '{}' (target {}, {} tensors: {} re-encoded, {} copied, \
         {:.2} MiB -> {:.2} MiB)",
        args.input.display(),
        args.output.display(),
        ftype.name(),
        plan.len(),
        encoded,
        copied,
        mib(total_src),
        mib(total_dst),
    );

    Ok(())
}

/// Bytes → MiB, for log lines.
fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Read the hyper-parameters `llama_tensor_get_type` consults.
fn read_model_info(model: &oxillama_gguf::GgufModel) -> anyhow::Result<ModelInfo> {
    let meta = &model.file.metadata;
    let arch = meta.get_string_or("general.architecture", "unknown");

    let u32_or = |key: String, default: u32| -> u32 { meta.get_u32(&key).unwrap_or(default) };

    let n_layer = u32_or(format!("{arch}.block_count"), 0) as i32;
    let n_head = u32_or(format!("{arch}.attention.head_count"), 0);
    let n_head_kv = u32_or(format!("{arch}.attention.head_count_kv"), n_head);
    let n_expert = u32_or(format!("{arch}.expert_count"), 0);

    let mut has_output = false;
    let mut n_attention_wv = 0i32;
    for (name, _) in model.file.tensors.iter() {
        if name == "output.weight" {
            has_output = true;
        }
        if name.contains("attn_v.weight")
            || name.contains("attn_qkv.weight")
            || name.contains("attn_kv_b.weight")
        {
            n_attention_wv += 1;
        }
    }

    Ok(ModelInfo {
        arch,
        n_layer,
        n_head,
        n_head_kv,
        n_expert,
        has_output,
        n_attention_wv,
    })
}

/// Decide what happens to every tensor, in GGUF file order.
fn build_plan(
    model: &oxillama_gguf::GgufModel,
    info: &ModelInfo,
    args: &QuantizeArgs,
    ftype: Ftype,
) -> anyhow::Result<Vec<Planned>> {
    // `TensorStore` is keyed by name and iterates in `HashMap` order, but the
    // type selector's counters make the visit order part of the algorithm.
    // Sort by data offset — which is file order for every GGUF ever written,
    // since offsets are assigned in tensor-info order — with the name as a
    // deterministic tie-break for zero-sized tensors.
    let mut ordered: Vec<_> = model.file.tensors.iter().map(|(_, i)| i).collect();
    ordered.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.name.cmp(&b.name)));

    let mut selector = TypeSelector::new(info, ftype);
    let mut plan = Vec::with_capacity(ordered.len());

    for tensor in ordered {
        let name = &tensor.name;
        let src_type = tensor.tensor_type;
        let ne0 = tensor.dimensions.first().copied().unwrap_or(0) as i64;

        let mut dst_type = src_type;
        let mut quantize = type_select::should_quantize(name, tensor.dimensions.len());

        if quantize {
            dst_type = ftype.default_type();
            if !args.pure {
                dst_type = selector
                    .select(name, ne0)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

                match type_select::fallback_for_incompatible_row(dst_type, ne0) {
                    None => {}
                    Some(Fallback::Use(fallback)) => {
                        eprintln!(
                            "warning: tensor '{name}' has {ne0} columns, not divisible by \
                             {}'s block size; falling back to {fallback}",
                            dst_type
                        );
                        dst_type = fallback;
                    }
                    Some(Fallback::NeedsIq4Nl) => {
                        anyhow::bail!(
                            "tensor '{name}' has {ne0} columns, which {dst_type} cannot \
                             represent; llama.cpp falls back to IQ4_NL here, and OxiLLaMa \
                             has no IQ4_NL encoder. Pick a different --target for this model."
                        );
                    }
                    Some(Fallback::Unsupported) => {
                        anyhow::bail!(
                            "tensor '{name}' has {ne0} columns, not a multiple of {dst_type}'s \
                             block size, and llama.cpp defines no fallback for {dst_type} \
                             (it aborts with \"Unsupported tensor size encountered\"). \
                             Pick a different --target for this model."
                        );
                    }
                }
            }
            // If the tensor is already the target type there is nothing to do
            // — decoding and re-encoding it could only lose precision.
            quantize = src_type != dst_type || args.force_requantize;
        }

        if !quantize {
            dst_type = src_type;
        }

        if quantize && !oxillama_quant::can_encode(dst_type) {
            anyhow::bail!(
                "tensor '{name}': no encoder for {dst_type} (OxiLLaMa can read this type but \
                 not write it)"
            );
        }

        // llama.cpp checks `allow_requantize` only on the path that actually
        // dequantizes, so `--dry-run` can plan a re-quantization it would
        // refuse to perform. Mirror that: a plan is not a conversion, and
        // being able to inspect the mixture without committing to the round
        // trip is the point of the flag.
        let source_is_quantized = !matches!(
            src_type,
            GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::Bf16
        );
        if quantize
            && !args.dry_run
            && source_is_quantized
            && !(args.allow_requantize || args.force_requantize)
        {
            anyhow::bail!(
                "tensor '{name}': refusing to requantize from {src_type} — quantizing an \
                 already-quantized model compounds the error of both passes. Pass \
                 --allow-requantize if that is what you want, or quantize from the \
                 F16/BF16/F32 checkpoint."
            );
        }

        let action = if quantize {
            Action::Encode(dst_type)
        } else {
            Action::Copy
        };
        let src_bytes = tensor.try_data_size().with_context(|| {
            format!("tensor '{name}': cannot compute source size — corrupt header?")
        })?;
        let dst_bytes = if quantize {
            row_bytes(&tensor.dimensions, dst_type)?
        } else {
            src_bytes
        };

        plan.push(Planned {
            name: name.clone(),
            dimensions: tensor.dimensions.clone(),
            src_type,
            dst_type,
            action,
            src_bytes,
            dst_bytes,
        });
    }

    Ok(plan)
}

/// Encoded size of a tensor: `nrows * row_size`, exactly as ggml computes it.
fn row_bytes(dimensions: &[u64], ty: GgufTensorType) -> anyhow::Result<u64> {
    let n_per_row = dimensions.first().copied().unwrap_or(0);
    if n_per_row == 0 {
        anyhow::bail!("tensor has a zero first dimension");
    }
    let n_elements: u64 = dimensions.iter().product();
    let block = ty.block_size() as u64;
    if !n_per_row.is_multiple_of(block) {
        anyhow::bail!("row length {n_per_row} is not a multiple of {ty}'s block size {block}");
    }
    let rows = n_elements / n_per_row;
    Ok(rows * (n_per_row / block) * ty.block_bytes() as u64)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::GgufWriter;

    /// Build and write a minimal GGUF with a single named F32 tensor of the
    /// given shape. Returns the path of the written temp file.
    fn write_named_f32_gguf(
        filename: &str,
        tensor_name: &str,
        dims: &[u64],
        fill_value: f32,
    ) -> PathBuf {
        write_gguf(filename, &[(tensor_name, dims, fill_value)])
    }

    /// Build and write a minimal GGUF with several named F32 tensors.
    fn write_gguf(filename: &str, tensors: &[(&str, &[u64], f32)]) -> PathBuf {
        let dir = temp_dir();
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        );
        writer.add_metadata("llama.block_count", MetadataValue::Uint32(1));
        writer.add_metadata("llama.attention.head_count", MetadataValue::Uint32(4));
        writer.add_metadata("llama.attention.head_count_kv", MetadataValue::Uint32(4));
        writer.add_metadata("tokenizer.ggml.model", MetadataValue::String("gpt2".into()));
        for (name, dims, fill) in tensors {
            let n_elements: usize = dims.iter().map(|&d| d as usize).product();
            let data: Vec<u8> = vec![*fill; n_elements]
                .iter()
                .flat_map(|v: &f32| v.to_le_bytes())
                .collect();
            writer.add_tensor(name, dims, GgufTensorType::F32, &data);
        }
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

    fn args(input: PathBuf, output: PathBuf, target: QuantTarget) -> QuantizeArgs {
        QuantizeArgs {
            input,
            output,
            target,
            allow_requantize: false,
            force_requantize: false,
            pure: false,
            dry_run: false,
        }
    }

    #[test]
    fn quantize_q4_k_m_encodes_a_2d_weight() {
        // 256 columns = exactly one Q4_K super-block per row.
        let input = write_named_f32_gguf("q4km_in.gguf", "blk.0.attn_q.weight", &[256, 4], 0.125);
        let output = temp_dir().join("q4km_out.gguf");
        run_quantize(&args(input, output.clone(), QuantTarget::Q4Km)).expect("Q4_K_M quantize");

        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output GGUF");
        let info = out_model
            .file
            .tensors
            .get("blk.0.attn_q.weight")
            .expect("tensor present");
        assert_eq!(info.tensor_type, GgufTensorType::Q4K);
        assert_eq!(info.data_size(), 4 * 144);
    }

    #[test]
    fn quantize_q6_k_encodes_a_2d_weight() {
        let input = write_named_f32_gguf("q6k_in.gguf", "blk.0.attn_q.weight", &[256, 4], -0.75);
        let output = temp_dir().join("q6k_out.gguf");
        run_quantize(&args(input, output.clone(), QuantTarget::Q6K)).expect("Q6_K quantize");

        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output GGUF");
        let info = out_model
            .file
            .tensors
            .get("blk.0.attn_q.weight")
            .expect("tensor present");
        assert_eq!(info.tensor_type, GgufTensorType::Q6K);
        assert_eq!(info.data_size(), 4 * 210);
    }

    #[test]
    fn quantize_q8_0_output_path_written() {
        // 2-D [32, 4] = 4 rows of one Q8_0 block each.
        let input = write_named_f32_gguf("q8_0_in.gguf", "blk.0.attn_q.weight", &[32, 4], 1.0);
        let output = temp_dir().join("q8_0_out.gguf");
        run_quantize(&args(input, output.clone(), QuantTarget::Q8_0)).expect("Q8_0 quantize");
        assert!(output.exists(), "Q8_0 output file should be created");

        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output GGUF");
        let info = out_model
            .file
            .tensors
            .get("blk.0.attn_q.weight")
            .expect("tensor present");
        assert_eq!(info.tensor_type, GgufTensorType::Q8_0);
    }

    /// The whole metadata block, tokenizer included, must survive.
    #[test]
    fn quantize_preserves_every_metadata_key() {
        let input = write_named_f32_gguf("meta_in.gguf", "blk.0.attn_q.weight", &[256, 4], 0.5);
        let output = temp_dir().join("meta_out.gguf");
        run_quantize(&args(input.clone(), output.clone(), QuantTarget::Q4Km))
            .expect("quantize should succeed");

        let in_model = oxillama_gguf::GgufModel::load(&input).expect("load input");
        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output");
        for (key, value) in in_model.file.metadata.iter() {
            let got = out_model
                .file
                .metadata
                .get(key)
                .unwrap_or_else(|| panic!("output lost metadata key '{key}'"));
            if key == "general.file_type" || key == "general.quantization_version" {
                continue;
            }
            assert_eq!(
                format!("{got}"),
                format!("{value}"),
                "metadata key '{key}' changed"
            );
        }
        assert_eq!(
            out_model
                .file
                .metadata
                .get_string("tokenizer.ggml.model")
                .ok(),
            Some("gpt2"),
            "the embedded tokenizer must survive re-quantization"
        );
        assert_eq!(
            out_model.file.metadata.get_u32("general.file_type").ok(),
            Some(15),
            "general.file_type must be rewritten to LLAMA_FTYPE_MOSTLY_Q4_K_M"
        );
        assert_eq!(
            out_model
                .file
                .metadata
                .get_u32("general.quantization_version")
                .ok(),
            Some(2)
        );
    }

    /// 1-D norm gains are never quantized: llama.cpp gates on `ndims >= 2` and
    /// separately excludes `*_norm.weight`.
    #[test]
    fn quantize_preserves_norm_tensor_bytes_exactly() {
        let input = write_gguf(
            "norm_in.gguf",
            &[
                ("output_norm.weight", &[16][..], 3.5),
                ("blk.0.attn_q.weight", &[256, 2][..], 0.25),
            ],
        );
        let output = temp_dir().join("norm_out.gguf");
        run_quantize(&args(input.clone(), output.clone(), QuantTarget::Q4Km))
            .expect("quantization should succeed");

        let in_model = oxillama_gguf::GgufModel::load(&input).expect("load input GGUF");
        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output GGUF");

        let out_info = out_model
            .file
            .tensors
            .get("output_norm.weight")
            .expect("tensor present in output");
        assert_eq!(
            out_info.tensor_type,
            GgufTensorType::F32,
            "a 1-D norm tensor must be kept at its original type"
        );
        assert_eq!(
            in_model.tensor_data("output_norm.weight").expect("in"),
            out_model.tensor_data("output_norm.weight").expect("out"),
            "a passed-through norm tensor's bytes must be byte-identical"
        );
    }

    /// With no `output.weight`, `token_embd.weight` is the output head and
    /// takes the output branch — Q6_K, or Q8_0 when its rows are too short.
    #[test]
    fn tied_token_embd_takes_the_output_branch() {
        let input = write_named_f32_gguf("embd_in.gguf", "token_embd.weight", &[256, 4], 0.25);
        let output = temp_dir().join("embd_out.gguf");
        run_quantize(&args(input, output.clone(), QuantTarget::Q4Km)).expect("quantize");

        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output GGUF");
        assert_eq!(
            out_model
                .file
                .tensors
                .get("token_embd.weight")
                .expect("tensor present")
                .tensor_type,
            GgufTensorType::Q6K,
            "a tied token_embd is quantized as the output head"
        );
    }

    #[test]
    fn short_rows_fall_back_to_a_compatible_type() {
        // 32 columns: Q4_K (256) does not fit, Q5_0 (32) does.
        let input = write_named_f32_gguf("short_in.gguf", "blk.0.attn_q.weight", &[32, 4], 0.5);
        let output = temp_dir().join("short_out.gguf");
        run_quantize(&args(input, output.clone(), QuantTarget::Q4Km)).expect("quantize");

        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output GGUF");
        assert_eq!(
            out_model
                .file
                .tensors
                .get("blk.0.attn_q.weight")
                .expect("tensor present")
                .tensor_type,
            GgufTensorType::Q5_0,
            "llama.cpp's Q4_K -> Q5_0 fallback"
        );
        // ...and the fallback path actually encoded the weights, rather than
        // just relabelling them.
        let bytes = out_model
            .tensor_data("blk.0.attn_q.weight")
            .expect("read tensor");
        let restored =
            dequantize_to_f32(bytes, GgufTensorType::Q5_0, 32 * 4).expect("dequantize Q5_0");
        assert_eq!(restored.len(), 128);
        for (i, v) in restored.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-3,
                "element {i}: expected ~0.5, got {v}"
            );
        }
    }

    #[test]
    fn rows_divisible_by_nothing_fall_back_to_f16() {
        // 100 columns: divisible by neither 256 (Q4_K) nor 32 (Q5_0).
        let input = write_named_f32_gguf("f16_in.gguf", "blk.0.attn_q.weight", &[100, 3], 0.5);
        let output = temp_dir().join("f16_out.gguf");
        run_quantize(&args(input, output.clone(), QuantTarget::Q4Km)).expect("quantize");

        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output GGUF");
        let info = out_model
            .file
            .tensors
            .get("blk.0.attn_q.weight")
            .expect("tensor present");
        assert_eq!(info.tensor_type, GgufTensorType::F16);
        assert_eq!(info.data_size(), 100 * 3 * 2);
    }

    #[test]
    fn q2_k_on_an_indivisible_row_reports_the_missing_iq4_nl_encoder() {
        let input = write_named_f32_gguf("iq_in.gguf", "blk.0.attn_q.weight", &[96, 4], 0.5);
        let output = temp_dir().join("iq_out.gguf");
        let err = run_quantize(&args(input, output, QuantTarget::Q2K))
            .expect_err("Q2_K with a 96-column row must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("IQ4_NL"),
            "the error must name the missing fallback encoder, got: {msg}"
        );
    }

    /// Re-quantizing an already-quantized model is refused unless asked for.
    #[test]
    fn requantizing_requires_an_explicit_flag() {
        let f32_input =
            write_named_f32_gguf("rq_src.gguf", "blk.0.attn_q.weight", &[256, 4], 0.375);
        let q4 = temp_dir().join("rq_q4.gguf");
        run_quantize(&args(f32_input, q4.clone(), QuantTarget::Q4Km)).expect("first pass");

        let q6 = temp_dir().join("rq_q6.gguf");
        let err = run_quantize(&args(q4.clone(), q6.clone(), QuantTarget::Q6K))
            .expect_err("requantization must be refused by default");
        assert!(format!("{err:#}").contains("--allow-requantize"));

        let mut permitted = args(q4, q6.clone(), QuantTarget::Q6K);
        permitted.allow_requantize = true;
        run_quantize(&permitted).expect("requantization with the flag");
        let out_model = oxillama_gguf::GgufModel::load(&q6).expect("load output");
        assert_eq!(
            out_model
                .file
                .tensors
                .get("blk.0.attn_q.weight")
                .expect("tensor")
                .tensor_type,
            GgufTensorType::Q6K
        );
    }

    /// A tensor already stored as the target type is copied, not re-encoded —
    /// llama.cpp's `quantize = tensor->type != new_type`.
    #[test]
    fn same_type_tensors_are_copied_not_re_encoded() {
        let f32_input =
            write_named_f32_gguf("idem_src.gguf", "blk.0.attn_q.weight", &[256, 4], 0.375);
        let first = temp_dir().join("idem_1.gguf");
        run_quantize(&args(f32_input, first.clone(), QuantTarget::Q4Km)).expect("first pass");

        let second = temp_dir().join("idem_2.gguf");
        let mut again = args(first.clone(), second.clone(), QuantTarget::Q4Km);
        again.allow_requantize = true;
        run_quantize(&again).expect("second pass");

        let a = oxillama_gguf::GgufModel::load(&first).expect("load first");
        let b = oxillama_gguf::GgufModel::load(&second).expect("load second");
        assert_eq!(
            a.tensor_data("blk.0.attn_q.weight").expect("a"),
            b.tensor_data("blk.0.attn_q.weight").expect("b"),
            "an already-Q4_K tensor must be copied through byte-for-byte"
        );

        // ...and --force-requantize makes it go through the encoder again.
        let third = temp_dir().join("idem_3.gguf");
        let mut forced = args(first, third.clone(), QuantTarget::Q4Km);
        forced.force_requantize = true;
        run_quantize(&forced).expect("forced pass");
        let c = oxillama_gguf::GgufModel::load(&third).expect("load third");
        assert_eq!(
            c.file
                .tensors
                .get("blk.0.attn_q.weight")
                .expect("tensor")
                .tensor_type,
            GgufTensorType::Q4K
        );
    }

    /// `general.alignment` must describe the file that was actually written,
    /// not the input's declaration — `GgufWriter` always pads to 32.
    #[test]
    fn alignment_metadata_is_rewritten_to_the_writers_alignment() {
        let dir = temp_dir();
        let mut w = GgufWriter::new();
        w.add_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        );
        w.add_metadata("general.alignment", MetadataValue::Uint32(64));
        let data: Vec<u8> = vec![0.5f32; 256 * 2]
            .iter()
            .flat_map(|v: &f32| v.to_le_bytes())
            .collect();
        w.add_tensor("blk.0.attn_q.weight", &[256, 2], GgufTensorType::F32, &data);
        let input = dir.join("align_in.gguf");
        w.write_to_file(&input).expect("write input");

        let output = dir.join("align_out.gguf");
        run_quantize(&args(input, output.clone(), QuantTarget::Q4Km)).expect("quantize");

        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output");
        assert_eq!(
            out_model.file.metadata.get_u32("general.alignment").ok(),
            Some(WRITER_ALIGNMENT),
            "the output must not claim an alignment it was not written with"
        );
        // The reader honours the key, so a wrong value would mis-locate the
        // tensor; reading it back proves the file is self-consistent.
        let bytes = out_model
            .tensor_data("blk.0.attn_q.weight")
            .expect("read tensor");
        let restored =
            dequantize_to_f32(bytes, GgufTensorType::Q4K, 512).expect("dequantize output");
        for v in &restored {
            assert!((v - 0.5).abs() < 1e-2, "expected ~0.5, got {v}");
        }
    }

    #[test]
    fn dry_run_writes_nothing() {
        let input = write_named_f32_gguf("dry_in.gguf", "blk.0.attn_q.weight", &[256, 4], 0.5);
        let output = temp_dir().join("dry_out.gguf");
        let _ = std::fs::remove_file(&output);
        let mut a = args(input, output.clone(), QuantTarget::Q4Km);
        a.dry_run = true;
        run_quantize(&a).expect("dry run");
        assert!(
            !output.exists(),
            "--dry-run must not create the output file"
        );
    }

    /// Planning is not converting: `--dry-run` may inspect the mixture of an
    /// already-quantized model without `--allow-requantize`, exactly as
    /// llama.cpp's dry run does (its guard sits on the conversion path).
    #[test]
    fn dry_run_does_not_require_allow_requantize() {
        let f32_input =
            write_named_f32_gguf("dryrq_src.gguf", "blk.0.attn_q.weight", &[256, 4], 0.375);
        let q4 = temp_dir().join("dryrq_q4.gguf");
        run_quantize(&args(f32_input, q4.clone(), QuantTarget::Q4Km)).expect("first pass");

        let out = temp_dir().join("dryrq_q6.gguf");
        let _ = std::fs::remove_file(&out);
        let mut plan_only = args(q4.clone(), out.clone(), QuantTarget::Q6K);
        plan_only.dry_run = true;
        run_quantize(&plan_only).expect("dry run over a quantized source");
        assert!(!out.exists());

        // ...but actually performing the conversion is still refused.
        assert!(run_quantize(&args(q4, out, QuantTarget::Q6K)).is_err());
    }

    #[test]
    fn pure_skips_the_per_tensor_promotions() {
        let input = write_named_f32_gguf("pure_in.gguf", "token_embd.weight", &[256, 4], 0.25);
        let output = temp_dir().join("pure_out.gguf");
        let mut a = args(input, output.clone(), QuantTarget::Q4Km);
        a.pure = true;
        run_quantize(&a).expect("pure quantize");

        let out_model = oxillama_gguf::GgufModel::load(&output).expect("load output");
        assert_eq!(
            out_model
                .file
                .tensors
                .get("token_embd.weight")
                .expect("tensor")
                .tensor_type,
            GgufTensorType::Q4K,
            "--pure keeps the bulk type even for the output head"
        );
    }

    #[test]
    fn row_bytes_matches_ggml_row_size() {
        assert_eq!(
            row_bytes(&[256, 4], GgufTensorType::Q4K).expect("q4k"),
            4 * 144
        );
        assert_eq!(
            row_bytes(&[512, 3], GgufTensorType::Q6K).expect("q6k"),
            3 * 2 * 210
        );
        assert_eq!(
            row_bytes(&[32, 8], GgufTensorType::Q8_0).expect("q8_0"),
            8 * 34
        );
        assert_eq!(row_bytes(&[64, 2], GgufTensorType::F16).expect("f16"), 256);
        assert!(row_bytes(&[100, 2], GgufTensorType::Q4K).is_err());
    }
}

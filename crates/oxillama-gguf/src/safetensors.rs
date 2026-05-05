//! Safetensors binary-format import bridge.
//!
//! The [safetensors specification](https://github.com/huggingface/safetensors)
//! is a simple binary format developed by Hugging Face:
//!
//! ```text
//! [8 bytes LE u64 header_size]
//! [header_size bytes UTF-8 JSON]
//! [raw tensor data ...]
//! ```
//!
//! The JSON header is a flat object where each key is a tensor name mapping to
//! a descriptor:
//!
//! ```json
//! {
//!   "tensor_name": {
//!     "dtype":        "F32",
//!     "shape":        [1, 2, 3],
//!     "data_offsets": [0, 24]
//!   },
//!   "__metadata__": { "format": "pt" }
//! }
//! ```
//!
//! `data_offsets` are relative to the start of the data section (i.e. byte
//! `8 + header_size`).
//!
//! ## Dtype mapping
//! | safetensors dtype | [`GgufTensorType`] |
//! |-------------------|--------------------|
//! | `"F32"`           | `F32`              |
//! | `"F16"`           | `F16`              |
//! | `"BF16"`          | `Bf16`             |
//! | `"I8"`            | `Q8_0` *(approximate; see note)* |
//!
//! **Note on `I8` → `Q8_0`:** Safetensors `I8` stores raw signed 8-bit
//! integers with no per-block scale factor.  `Q8_0` is a _quantized_ format
//! with a 2-byte FP16 scale per 32-weight block.  The mapping is therefore
//! approximate and is provided only to allow the data to be represented in the
//! GGUF type system; callers that need numerically accurate inference should
//! re-quantize properly.
//!
//! All other dtypes cause [`GgufError::UnsupportedDtype`] to be returned.

use std::path::Path;

use serde_json::Value;

use crate::error::{GgufError, GgufResult};
use crate::loader::GgufModel;
use crate::types::{GgufTensorType, GGUF_DEFAULT_ALIGNMENT};

/// The magic string used for the sentinel metadata key that safetensors embeds
/// to carry arbitrary file-level metadata.
const SAFETENSORS_META_KEY: &str = "__metadata__";

/// Converter that reads a `.safetensors` file and presents its contents as a
/// [`GgufModel`] with owned tensor data.
///
/// The resulting [`GgufModel`] carries:
/// - A single metadata key: `general.architecture = "safetensors_import"`.
/// - One [`crate::tensor_info::TensorInfo`] per tensor found in the safetensors header.
/// - All raw tensor bytes stored in-memory as `GgufData::Owned`.
///
/// Because safetensors does not use GGUF's alignment-padded data section, the
/// tensor data is stored contiguously in the order it appears in the file;
/// tensor offsets in [`crate::tensor_info::TensorInfo`] are absolute byte positions within that
/// owned buffer.
pub struct SafetensorsConverter;

impl SafetensorsConverter {
    /// Load a `.safetensors` file from disk and return it as a [`GgufModel`].
    ///
    /// # Errors
    /// - [`GgufError::MmapError`] if the file cannot be read.
    /// - All errors from [`SafetensorsConverter::from_bytes`].
    pub fn load(path: &Path) -> GgufResult<GgufModel> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a safetensors byte buffer and return it as a [`GgufModel`].
    ///
    /// This is the core conversion routine. It is also used in tests so that
    /// the format can be exercised without touching the filesystem.
    ///
    /// # Errors
    /// - [`GgufError::UnexpectedEof`] if the buffer is too short to contain
    ///   the 8-byte length prefix or the declared JSON header.
    /// - [`GgufError::SafetensorsParseError`] if the JSON header is invalid.
    /// - [`GgufError::UnsupportedDtype`] if a tensor uses an unrecognised dtype.
    /// - [`GgufError::UnexpectedEof`] if a tensor's data range extends beyond
    ///   the buffer.
    pub fn from_bytes(bytes: &[u8]) -> GgufResult<GgufModel> {
        // ── 1. Parse the 8-byte little-endian header_size prefix ─────────────
        if bytes.len() < 8 {
            return Err(GgufError::UnexpectedEof { offset: 0 });
        }
        let header_size = u64::from_le_bytes(
            bytes[..8]
                .try_into()
                .map_err(|_| GgufError::UnexpectedEof { offset: 0 })?,
        ) as usize;

        // ── 2. Slice out the JSON header bytes ────────────────────────────────
        let json_end = 8_usize
            .checked_add(header_size)
            .ok_or(GgufError::UnexpectedEof { offset: 8 })?;
        let json_bytes = bytes
            .get(8..json_end)
            .ok_or(GgufError::UnexpectedEof { offset: 8 })?;

        // The byte offset of the first tensor datum.
        let data_section_start = json_end;

        // ── 3. Deserialize JSON ───────────────────────────────────────────────
        let root: serde_json::Map<String, Value> = serde_json::from_slice(json_bytes)
            .map_err(|e| GgufError::SafetensorsParseError(format!("JSON parse error: {e}")))?;

        // ── 5. Parse tensor entries ───────────────────────────────────────────
        // Compute the range of raw tensor bytes we need to retain.  We collect
        // all [start, end) byte ranges in the data section, then copy the
        // minimal contiguous slice that covers them all.
        let mut tensor_entries: Vec<ParsedTensor> = Vec::new();
        let mut max_end: usize = 0;

        for (name, desc) in &root {
            // Skip the __metadata__ sentinel.
            if name == SAFETENSORS_META_KEY {
                continue;
            }

            let tensor = parse_tensor_entry(name, desc)?;

            // Resolve data_offsets relative to data_section_start.
            let abs_start = data_section_start
                .checked_add(tensor.data_start)
                .ok_or_else(|| {
                    GgufError::SafetensorsParseError(format!(
                        "tensor '{name}': data_start overflows usize"
                    ))
                })?;
            let abs_end = data_section_start
                .checked_add(tensor.data_end)
                .ok_or_else(|| {
                    GgufError::SafetensorsParseError(format!(
                        "tensor '{name}': data_end overflows usize"
                    ))
                })?;

            // Bounds-check against the input buffer.
            if abs_end > bytes.len() {
                return Err(GgufError::UnexpectedEof {
                    offset: abs_start as u64,
                });
            }

            if abs_end > max_end {
                max_end = abs_end;
            }

            tensor_entries.push(ParsedTensor {
                name: name.clone(),
                dtype: tensor.dtype,
                dimensions: tensor.dimensions,
                abs_start,
            });
        }

        // ── 6. Build the owned data buffer ────────────────────────────────────
        // We copy the raw tensor-data section into a single owned Vec<u8> that
        // becomes the `GgufData::Owned` payload of the returned `GgufModel`.
        //
        // To avoid storing the entire safetensors file (header + data), we only
        // retain the data-section bytes, with a preamble that is a minimal
        // synthetic GGUF header so that `GgufFile::parse` can locate tensors.
        //
        // Strategy: build a full in-memory GGUF blob so we can call
        // `GgufModel::from_bytes`.  This is heavier than strictly necessary but
        // avoids duplicating the GGUF parsing machinery.

        let raw_data_slice = if max_end > data_section_start {
            &bytes[data_section_start..max_end]
        } else {
            &[][..]
        };

        // Build synthetic GGUF v3 binary in memory.
        let gguf_bytes = build_synthetic_gguf(&tensor_entries, raw_data_slice, data_section_start)?;

        GgufModel::from_bytes(gguf_bytes)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Intermediate parsed representation of one safetensors tensor entry.
struct RawTensorEntry {
    dtype: GgufTensorType,
    dimensions: Vec<u64>,
    data_start: usize,
    data_end: usize,
}

/// Parsed tensor entry with absolute byte position of its data start.
struct ParsedTensor {
    name: String,
    dtype: GgufTensorType,
    dimensions: Vec<u64>,
    /// Absolute start offset in the original `bytes` buffer.
    abs_start: usize,
}

/// Parse one tensor descriptor from the safetensors JSON.
///
/// Expected JSON shape (all fields required):
/// ```json
/// { "dtype": "F32", "shape": [4, 8], "data_offsets": [0, 128] }
/// ```
fn parse_tensor_entry(name: &str, desc: &Value) -> GgufResult<RawTensorEntry> {
    let obj = desc.as_object().ok_or_else(|| {
        GgufError::SafetensorsParseError(format!(
            "tensor '{name}': descriptor must be a JSON object"
        ))
    })?;

    // dtype
    let dtype_str = obj.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
        GgufError::SafetensorsParseError(format!(
            "tensor '{name}': missing or invalid 'dtype' field"
        ))
    })?;
    let dtype = map_dtype(dtype_str)?;

    // shape
    let shape_arr = obj.get("shape").and_then(|v| v.as_array()).ok_or_else(|| {
        GgufError::SafetensorsParseError(format!(
            "tensor '{name}': missing or invalid 'shape' field"
        ))
    })?;
    let dimensions: Vec<u64> = shape_arr
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_u64().ok_or_else(|| {
                GgufError::SafetensorsParseError(format!(
                    "tensor '{name}': shape[{i}] is not a non-negative integer"
                ))
            })
        })
        .collect::<GgufResult<Vec<_>>>()?;

    // data_offsets
    let offsets_arr = obj
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            GgufError::SafetensorsParseError(format!(
                "tensor '{name}': missing or invalid 'data_offsets' field"
            ))
        })?;
    if offsets_arr.len() != 2 {
        return Err(GgufError::SafetensorsParseError(format!(
            "tensor '{name}': 'data_offsets' must be an array of exactly 2 elements, got {}",
            offsets_arr.len()
        )));
    }
    let data_start = offsets_arr[0].as_u64().ok_or_else(|| {
        GgufError::SafetensorsParseError(format!(
            "tensor '{name}': data_offsets[0] is not a non-negative integer"
        ))
    })? as usize;
    let data_end = offsets_arr[1].as_u64().ok_or_else(|| {
        GgufError::SafetensorsParseError(format!(
            "tensor '{name}': data_offsets[1] is not a non-negative integer"
        ))
    })? as usize;

    if data_end < data_start {
        return Err(GgufError::SafetensorsParseError(format!(
            "tensor '{name}': data_offsets[1] ({data_end}) < data_offsets[0] ({data_start})"
        )));
    }

    Ok(RawTensorEntry {
        dtype,
        dimensions,
        data_start,
        data_end,
    })
}

/// Map a safetensors dtype string to a [`GgufTensorType`].
fn map_dtype(dtype_str: &str) -> GgufResult<GgufTensorType> {
    match dtype_str {
        "F32" => Ok(GgufTensorType::F32),
        "F16" => Ok(GgufTensorType::F16),
        "BF16" => Ok(GgufTensorType::Bf16),
        // I8 → Q8_0 is approximate (no per-block scale); documented in module
        // docs.
        "I8" => Ok(GgufTensorType::Q8_0),
        other => Err(GgufError::UnsupportedDtype(other.to_string())),
    }
}

/// Build a minimal valid GGUF v3 byte buffer from synthetic tensor metadata
/// and the raw tensor data copied verbatim from the safetensors file.
///
/// Layout of the generated buffer:
/// ```text
/// GGUF header  (magic, version, tensor_count, kv_count)
/// KV pairs     (general.architecture = "safetensors_import")
/// Tensor infos (one per tensor, with offset relative to data section)
/// Alignment padding (to 32-byte boundary)
/// Raw tensor data  (concatenated, offsets recomputed)
/// ```
fn build_synthetic_gguf(
    tensors: &[ParsedTensor],
    raw_data_slice: &[u8],
    data_section_start: usize,
) -> GgufResult<Vec<u8>> {
    use crate::types::GGUF_MAGIC;

    let mut buf = Vec::<u8>::new();

    // ── GGUF header ───────────────────────────────────────────────────────────
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
    buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count = 1

    // ── KV: general.architecture = "safetensors_import" ─────────────────────
    write_gguf_string(&mut buf, "general.architecture");
    // GgufValueType::String = 8
    buf.extend_from_slice(&8u32.to_le_bytes());
    write_gguf_string(&mut buf, "safetensors_import");

    // ── Tensor infos ──────────────────────────────────────────────────────────
    // We must recompute tensor offsets relative to the data section that will
    // follow.  Since we copy the entire raw_data_slice verbatim, each tensor's
    // offset is its absolute start in the original data section minus
    // `data_section_start`.
    for t in tensors {
        write_gguf_string(&mut buf, &t.name);
        let n_dims = t.dimensions.len() as u32;
        buf.extend_from_slice(&n_dims.to_le_bytes());
        for &dim in &t.dimensions {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&(t.dtype as u32).to_le_bytes());
        // Offset relative to the data section start.
        let rel_offset = (t.abs_start - data_section_start) as u64;
        buf.extend_from_slice(&rel_offset.to_le_bytes());
    }

    // ── Alignment padding ─────────────────────────────────────────────────────
    let align = GGUF_DEFAULT_ALIGNMENT as usize;
    let rem = buf.len() % align;
    if rem != 0 {
        let pad = align - rem;
        buf.resize(buf.len() + pad, 0u8);
    }

    // ── Raw tensor data ───────────────────────────────────────────────────────
    buf.extend_from_slice(raw_data_slice);

    Ok(buf)
}

/// Write a GGUF v3 string (8-byte LE length prefix followed by UTF-8 bytes).
fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: build a minimal safetensors byte buffer ───────────────────────

    /// Write a safetensors u64 LE header_size prefix + JSON body + data.
    fn build_safetensors(json: &str, data: &[u8]) -> Vec<u8> {
        let header_bytes = json.as_bytes();
        let header_size = header_bytes.len() as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_size.to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(data);
        buf
    }

    // ── Test 1: empty (no tensors), just metadata ─────────────────────────────

    /// A safetensors file with only `__metadata__` and no tensors must parse
    /// successfully and produce a `GgufModel` with zero tensors and the correct
    /// architecture string.
    #[test]
    fn safetensors_empty_data_section_parses() {
        let json = r#"{"__metadata__": {"format": "pt"}}"#;
        let buf = build_safetensors(json, &[]);

        let model = SafetensorsConverter::from_bytes(&buf)
            .expect("empty safetensors should parse without error");

        assert_eq!(
            model.file.header.tensor_count, 0,
            "no tensors should be registered"
        );
        assert_eq!(
            model.architecture().expect("architecture must be present"),
            "safetensors_import",
            "architecture must be 'safetensors_import'"
        );
    }

    // ── Test 2: single F32 tensor roundtrip ───────────────────────────────────

    /// Build a safetensors buffer with one F32 scalar (4 bytes = 1 element),
    /// load it, and verify that the tensor shape and dtype survive the round-trip.
    #[test]
    fn safetensors_single_f32_tensor_roundtrip() {
        // One F32 element: bytes [0x00, 0x00, 0x80, 0x3F] = 1.0f32 in LE.
        let tensor_data: &[u8] = &[0x00, 0x00, 0x80, 0x3F];

        let json = r#"{
            "weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4]
            },
            "__metadata__": {"format": "pt"}
        }"#;

        let buf = build_safetensors(json, tensor_data);
        let model = SafetensorsConverter::from_bytes(&buf).expect("single F32 tensor should parse");

        assert_eq!(
            model.file.header.tensor_count, 1,
            "exactly one tensor expected"
        );
        assert_eq!(
            model.architecture().expect("architecture"),
            "safetensors_import"
        );

        // Verify the tensor info: name, shape, dtype.
        let info = model
            .file
            .tensors
            .get("weight")
            .expect("tensor 'weight' should be present");
        assert_eq!(info.n_dims, 1, "tensor should be 1-dimensional");
        assert_eq!(info.dimensions, vec![1u64], "shape should be [1]");
        assert_eq!(info.tensor_type, GgufTensorType::F32, "dtype should be F32");

        // Verify the data bytes are accessible.
        let data = model
            .tensor_data("weight")
            .expect("tensor data should be accessible");
        assert_eq!(data.len(), 4, "F32 tensor data should be 4 bytes");
        assert_eq!(data, tensor_data, "tensor data bytes must match original");
    }

    // ── Test 3: truncated buffer ───────────────────────────────────────────────

    /// Bytes that are too short to contain even the 8-byte length prefix should
    /// produce an `UnexpectedEof` error rather than panicking or producing a
    /// corrupt model.
    #[test]
    fn safetensors_rejects_truncated() {
        // Only 5 bytes — cannot even read the 8-byte header_size field.
        let truncated: &[u8] = &[0x10, 0x00, 0x00, 0x00, 0x00];
        let result = SafetensorsConverter::from_bytes(truncated);
        assert!(result.is_err(), "truncated buffer must produce an error");

        // The error must be UnexpectedEof.
        match result {
            Err(GgufError::UnexpectedEof { .. }) => {}
            Err(other) => panic!("expected UnexpectedEof, got: {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    // ── Test 4: unsupported dtype ─────────────────────────────────────────────

    /// A tensor with an unrecognised dtype string must cause
    /// `GgufError::UnsupportedDtype` rather than a panic or corrupt parse.
    #[test]
    fn safetensors_unsupported_dtype_errors() {
        let json = r#"{
            "t": {
                "dtype": "BFLOAT9",
                "shape": [2],
                "data_offsets": [0, 4]
            }
        }"#;
        let buf = build_safetensors(json, &[0u8; 4]);
        let result = SafetensorsConverter::from_bytes(&buf);
        assert!(result.is_err(), "unsupported dtype should produce an error");

        match result {
            Err(GgufError::UnsupportedDtype(_)) => {}
            Err(other) => panic!("expected UnsupportedDtype, got: {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    // ── Test 5: multiple tensors with correct offsets ─────────────────────────

    /// Two F16 tensors with different shapes must both appear in the store with
    /// the correct shapes and non-overlapping data.
    #[test]
    fn safetensors_two_f16_tensors_round_trip() {
        // a: shape [2], dtype F16 → 2 * 2 = 4 bytes, offsets [0, 4]
        // b: shape [3], dtype F16 → 3 * 2 = 6 bytes, offsets [4, 10]
        let data: &[u8] = &[0x00, 0x3C, 0x00, 0x40, 0x00, 0x44, 0x00, 0x48, 0x00, 0x4C];
        //                   ─────── a ───────  ─────────────── b ────────────────────

        let json = r#"{
            "a": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
            "b": {"dtype": "F16", "shape": [3], "data_offsets": [4, 10]}
        }"#;

        let buf = build_safetensors(json, data);
        let model = SafetensorsConverter::from_bytes(&buf).expect("two F16 tensors should parse");

        assert_eq!(model.file.header.tensor_count, 2);

        let a = model.file.tensors.get("a").expect("tensor 'a'");
        assert_eq!(a.dimensions, vec![2u64]);
        assert_eq!(a.tensor_type, GgufTensorType::F16);

        let b = model.file.tensors.get("b").expect("tensor 'b'");
        assert_eq!(b.dimensions, vec![3u64]);
        assert_eq!(b.tensor_type, GgufTensorType::F16);

        // Data for tensor a should be the first 4 bytes.
        let a_data = model.tensor_data("a").expect("tensor data 'a'");
        assert_eq!(a_data.len(), 4, "tensor 'a' data should be 4 bytes");
        assert_eq!(a_data, &data[..4], "tensor 'a' data mismatch");
    }

    // ── Test 6: malformed JSON ────────────────────────────────────────────────

    /// Invalid JSON in the header section must produce `SafetensorsParseError`.
    #[test]
    fn safetensors_invalid_json_errors() {
        // Header size = 5, but the 5 JSON bytes are not valid JSON.
        let mut buf = Vec::new();
        let bad_json = b"{nope";
        buf.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        buf.extend_from_slice(bad_json);

        let result = SafetensorsConverter::from_bytes(&buf);
        assert!(result.is_err(), "invalid JSON should produce an error");

        match result {
            Err(GgufError::SafetensorsParseError(_)) => {}
            Err(other) => panic!("expected SafetensorsParseError, got: {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    // ── Test 7: safetensors file load from disk (temp file) ───────────────────

    /// Write a safetensors buffer to a temp file and verify `load()` returns
    /// the same result as `from_bytes()`.
    #[test]
    fn safetensors_load_from_disk_matches_from_bytes() {
        let json = r#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}"#;
        let data: &[u8] = &[0x00, 0x00, 0x80, 0x3F]; // 1.0f32 LE
        let buf = build_safetensors(json, data);

        let dir = std::env::temp_dir();
        let path = dir.join("oxillama_safetensors_test_load.safetensors");
        std::fs::write(&path, &buf).expect("write temp safetensors file");

        let from_file = SafetensorsConverter::load(&path)
            .expect("SafetensorsConverter::load from temp file should succeed");
        let _ = std::fs::remove_file(&path);

        assert_eq!(from_file.file.header.tensor_count, 1);
        assert_eq!(
            from_file.architecture().expect("arch"),
            "safetensors_import"
        );

        let info = from_file.file.tensors.get("w").expect("tensor 'w'");
        assert_eq!(info.tensor_type, GgufTensorType::F32);
        assert_eq!(info.dimensions, vec![1u64]);
    }
}

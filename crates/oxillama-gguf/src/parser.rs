//! Complete GGUF file parser.
//!
//! Parses the entire GGUF binary format: header, metadata KV pairs,
//! tensor info entries, and computes the data section offset.

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec::Vec};

use crate::error::{GgufError, GgufResult};
use crate::header::GgufHeader;
use crate::metadata::{MetadataStore, MetadataValue};
use crate::reader::BinaryReader;
use crate::tensor_info::{TensorInfo, TensorStore};
use crate::types::{GgufTensorType, GgufValueType, GGUF_DEFAULT_ALIGNMENT};

/// Maximum nesting depth allowed for `Array`-of-`Array` metadata values.
///
/// A GGUF array whose element type is itself `Array` costs only 12 bytes of
/// input per nesting level (a u32 element-type id + a u64 element count),
/// but recurses one Rust stack frame per level with no bound — a file with
/// tens of thousands of levels stack-overflows and aborts the process
/// (uncatchable, unlike a panic). Eight levels is far beyond anything a
/// real GGUF file needs (arrays of arrays do not appear in practice; this
/// exists purely as a safety valve).
const MAX_METADATA_ARRAY_DEPTH: u32 = 8;

/// Maximum number of tensor dimensions accepted.
///
/// GGML tensors never exceed 4 dimensions; this also bounds the
/// `Vec::with_capacity(n_dims as usize)` allocation below so an
/// attacker-controlled `n_dims` (read from just 4 bytes of input) cannot
/// request an outsized allocation.
const MAX_TENSOR_DIMS: u32 = 4;

/// Minimum number of encoded bytes a single metadata value of `value_type`
/// can possibly occupy, used to sanity-bound an array's declared `count`
/// against the bytes actually remaining in the reader before trusting it
/// enough to preallocate a `Vec`.
fn min_encoded_value_size(value_type: GgufValueType, version: u32) -> u64 {
    match value_type {
        GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => 1,
        GgufValueType::Uint16 | GgufValueType::Int16 => 2,
        GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => 4,
        GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => 8,
        // A string's minimum encoding is just its (possibly zero) length
        // prefix: 8 bytes for v3, 4 bytes for v2/v1.
        GgufValueType::String => {
            if version >= 3 {
                8
            } else {
                4
            }
        }
        // A nested array's minimum encoding is an empty array: 4 bytes for
        // the element-type id plus the count field (8 bytes v3, 4 v2/v1).
        GgufValueType::Array => {
            if version >= 3 {
                12
            } else {
                8
            }
        }
    }
}

/// Maximum alignment value accepted from `general.alignment` metadata.
///
/// Real GGUF files use small power-of-two alignments (32 is the default;
/// some writers use larger values for mmap-friendliness). 1 GiB is
/// enormous headroom over any legitimate value while remaining nowhere
/// near large enough to make `value + alignment` overflow `u64` for any
/// realistic `value` (a parsed header/metadata/tensor-info byte offset).
const MAX_GGUF_ALIGNMENT: u64 = 1 << 30;

/// Validate a `general.alignment` metadata value before it is used to
/// compute the tensor data-section offset.
///
/// `alignment` is fully attacker-controlled. The alignment arithmetic
/// (`value + alignment - rem`, evaluated left-to-right) can overflow `u64`
/// when `alignment` is astronomically large (e.g. `u64::MAX`), panicking
/// in a debug build and wrapping `data_section_offset` to an
/// attacker-chosen small value in a release build — silently steering
/// every tensor's absolute file offset, the same class of bug as V1's
/// `check()` bypass. Rejecting non-power-of-two or oversized values here
/// closes that off before any arithmetic runs; `align_up` below also
/// switches to checked arithmetic as defense in depth.
fn validate_alignment(alignment: u64) -> GgufResult<()> {
    // `align_up` treats 0 as "no alignment" (identity); that is not itself
    // a safety issue, so it passes through unchanged.
    if alignment == 0 {
        return Ok(());
    }
    if !alignment.is_power_of_two() || alignment > MAX_GGUF_ALIGNMENT {
        return Err(GgufError::InvalidMetadata {
            key: "general.alignment".to_string(),
            reason: format!(
                "alignment must be a power of two no greater than {MAX_GGUF_ALIGNMENT}, got {alignment}"
            ),
        });
    }
    Ok(())
}

/// A fully parsed GGUF file (metadata + tensor registry).
#[derive(Debug)]
pub struct GgufFile {
    /// Parsed file header.
    pub header: GgufHeader,
    /// Metadata key-value store.
    pub metadata: MetadataStore,
    /// Tensor info registry.
    pub tensors: TensorStore,
    /// Alignment value used for tensor data (from metadata or default 32).
    pub alignment: u64,
}

impl GgufFile {
    /// Parse a complete GGUF file from a byte slice.
    ///
    /// This parses the header, all metadata KV pairs, and all tensor info entries.
    /// It does NOT load tensor data — use `tensor_data()` to access raw data.
    pub fn parse(data: &[u8]) -> GgufResult<Self> {
        let (header, offset) = GgufHeader::parse(data, 0)?;
        let mut reader = BinaryReader::new(data, offset as usize);

        // Parse metadata KV pairs
        let metadata = parse_metadata(&mut reader, &header)?;

        // Read alignment from metadata (or use default)
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        validate_alignment(alignment)?;

        // Parse tensor info entries
        let mut tensors = parse_tensor_infos(&mut reader, &header)?;

        // Compute data section offset: align current position to alignment boundary
        let data_offset = align_up(reader.position() as u64, alignment);
        tensors.set_data_offset(data_offset);

        Ok(Self {
            header,
            metadata,
            tensors,
            alignment,
        })
    }

    /// Get raw tensor data bytes for a named tensor.
    ///
    /// Returns a slice into the original data buffer.
    pub fn tensor_data<'a>(&self, data: &'a [u8], name: &str) -> GgufResult<&'a [u8]> {
        let (start, len) = self.tensor_data_range(data.len(), name)?;
        Ok(&data[start..start + len])
    }

    /// Locate a named tensor's payload as a `(start, len)` byte range inside a
    /// backing buffer of `data_len` bytes.
    ///
    /// This is the borrow-free half of [`Self::tensor_data`]: it lets a caller
    /// build a reference-counted [`SharedBytes`][crate::SharedBytes] view over
    /// the tensor without first materialising a `&[u8]` tied to the buffer's
    /// lifetime — which is what keeps mmap-backed weights copy-free.
    pub fn tensor_data_range(&self, data_len: usize, name: &str) -> GgufResult<(usize, usize)> {
        let info = self.tensors.get(name)?;

        // `data_offset() + info.offset` used to be a plain `+`: with
        // `info.offset` near `u64::MAX` (fully attacker-controlled — it is
        // read straight off the file) this wraps to a small value, which
        // then passes the `end > data_len` check below and hands back
        // bytes from *inside* the GGUF header instead of erroring.
        let base_offset = self.tensors.data_offset();
        let abs_offset = base_offset
            .checked_add(info.offset)
            .ok_or(GgufError::UnexpectedEof {
                offset: base_offset,
            })?;

        // `try_data_size()` rejects the dimension-product / block-count
        // overflow described in V3 instead of silently wrapping to a
        // small size (the infallible `data_size()` would).
        let size = info.try_data_size()?;

        // Validate in `u64` and convert with `try_from` rather than `as`:
        // an `as usize` cast silently truncates on 32-bit/wasm32 targets,
        // which can make an out-of-range offset reappear in-bounds.
        let start = usize::try_from(abs_offset)
            .map_err(|_| GgufError::UnexpectedEof { offset: abs_offset })?;
        let size_usize =
            usize::try_from(size).map_err(|_| GgufError::UnexpectedEof { offset: abs_offset })?;
        let end = start
            .checked_add(size_usize)
            .ok_or(GgufError::UnexpectedEof { offset: abs_offset })?;

        if end > data_len {
            return Err(GgufError::UnexpectedEof { offset: abs_offset });
        }

        Ok((start, size_usize))
    }

    /// Get the model architecture string from metadata.
    pub fn architecture(&self) -> GgufResult<&str> {
        self.metadata.get_string("general.architecture")
    }

    /// Get the model name from metadata.
    pub fn model_name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }
}

/// Parse all metadata KV pairs from the reader.
fn parse_metadata(reader: &mut BinaryReader<'_>, header: &GgufHeader) -> GgufResult<MetadataStore> {
    let mut store = MetadataStore::new();

    for _ in 0..header.metadata_kv_count {
        let key = if header.version >= 3 {
            reader.read_string()?
        } else {
            reader.read_string_v2()?
        };

        let value_type_id = reader.read_u32()?;
        let value_type =
            GgufValueType::from_u32(value_type_id).ok_or_else(|| GgufError::InvalidMetadata {
                key: key.clone(),
                reason: format!("unknown value type: {value_type_id}"),
            })?;

        let value = read_metadata_value(reader, value_type, header.version, 0)?;
        store.insert(key, value);
    }

    Ok(store)
}

/// Read a single metadata value based on its type.
///
/// `depth` counts `Array`-of-`Array` nesting levels and is enforced against
/// [`MAX_METADATA_ARRAY_DEPTH`] to bound recursion — see that constant's
/// docs for why an unbounded version is a process-aborting DoS.
fn read_metadata_value(
    reader: &mut BinaryReader<'_>,
    value_type: GgufValueType,
    version: u32,
    depth: u32,
) -> GgufResult<MetadataValue> {
    if depth > MAX_METADATA_ARRAY_DEPTH {
        return Err(GgufError::InvalidMetadata {
            key: "<array>".to_string(),
            reason: format!("array nesting depth exceeds the limit of {MAX_METADATA_ARRAY_DEPTH}"),
        });
    }
    match value_type {
        GgufValueType::Uint8 => Ok(MetadataValue::Uint8(reader.read_u8()?)),
        GgufValueType::Int8 => Ok(MetadataValue::Int8(reader.read_u8()? as i8)),
        GgufValueType::Uint16 => Ok(MetadataValue::Uint16(reader.read_u16()?)),
        GgufValueType::Int16 => Ok(MetadataValue::Int16(reader.read_i16()?)),
        GgufValueType::Uint32 => Ok(MetadataValue::Uint32(reader.read_u32()?)),
        GgufValueType::Int32 => Ok(MetadataValue::Int32(reader.read_i32()?)),
        GgufValueType::Float32 => Ok(MetadataValue::Float32(reader.read_f32()?)),
        GgufValueType::Float64 => Ok(MetadataValue::Float64(reader.read_f64()?)),
        GgufValueType::Bool => Ok(MetadataValue::Bool(reader.read_bool()?)),
        GgufValueType::String => {
            let s = if version >= 3 {
                reader.read_string()?
            } else {
                reader.read_string_v2()?
            };
            Ok(MetadataValue::String(s))
        }
        GgufValueType::Uint64 => Ok(MetadataValue::Uint64(reader.read_u64()?)),
        GgufValueType::Int64 => Ok(MetadataValue::Int64(reader.read_i64()?)),
        GgufValueType::Array => {
            let elem_type_id = reader.read_u32()?;
            let elem_type = GgufValueType::from_u32(elem_type_id).ok_or_else(|| {
                GgufError::InvalidMetadata {
                    key: "<array>".to_string(),
                    reason: format!("unknown array element type: {elem_type_id}"),
                }
            })?;

            let count_u64 = if version >= 3 {
                reader.read_u64()?
            } else {
                u64::from(reader.read_u32()?)
            };

            // Bound `count` by the bytes actually remaining before trusting
            // it enough to preallocate: a 12-byte input can otherwise claim
            // a billion-element array and reserve tens of MB of `Vec`
            // capacity per nesting level from a handful of input bytes.
            // `remaining()` is exact here (BinaryReader wraps the whole
            // in-memory file), so this can never reject a legitimate array.
            let min_elem_size = min_encoded_value_size(elem_type, version);
            let max_count = reader.remaining() as u64 / min_elem_size;
            if count_u64 > max_count {
                return Err(GgufError::InvalidMetadata {
                    key: "<array>".to_string(),
                    reason: format!(
                        "array count {count_u64} cannot fit in the {} bytes remaining (min {min_elem_size} bytes/element)",
                        reader.remaining()
                    ),
                });
            }
            let count = count_u64 as usize; // safe: count_u64 <= max_count <= remaining() (a usize)

            let mut elements = Vec::with_capacity(count.min(4096));
            for _ in 0..count {
                elements.push(read_metadata_value(reader, elem_type, version, depth + 1)?);
            }
            Ok(MetadataValue::Array(elements))
        }
    }
}

/// Parse all tensor info entries from the reader.
fn parse_tensor_infos(
    reader: &mut BinaryReader<'_>,
    header: &GgufHeader,
) -> GgufResult<TensorStore> {
    let mut store = TensorStore::new();

    for _ in 0..header.tensor_count {
        let name = if header.version >= 3 {
            reader.read_string()?
        } else {
            reader.read_string_v2()?
        };

        let n_dims = reader.read_u32()?;
        if n_dims == 0 || n_dims > MAX_TENSOR_DIMS {
            return Err(GgufError::IntegrityError {
                tensor_name: name,
                reason: format!(
                    "tensor declares {n_dims} dimensions, expected 1..={MAX_TENSOR_DIMS}"
                ),
            });
        }

        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = if header.version >= 3 {
                reader.read_u64()?
            } else {
                reader.read_u32()? as u64
            };
            dimensions.push(dim);
        }

        let type_id = reader.read_u32()?;
        let tensor_type =
            GgufTensorType::from_u32(type_id).ok_or(GgufError::UnsupportedQuantType { type_id })?;

        let offset = if header.version >= 2 {
            reader.read_u64()?
        } else {
            reader.read_u32()? as u64
        };

        store.try_insert(TensorInfo {
            name,
            n_dims,
            dimensions,
            tensor_type,
            offset,
        })?;
    }

    if store.len() as u64 != header.tensor_count {
        return Err(GgufError::IntegrityError {
            tensor_name: "<tensor_infos>".to_string(),
            reason: format!(
                "expected {} tensors, parsed {}",
                header.tensor_count,
                store.len()
            ),
        });
    }

    Ok(store)
}

/// Align a value up to the given alignment boundary.
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        return value;
    }
    // `alignment - rem` cannot overflow: `rem < alignment` always holds
    // when `alignment != 0` (a property of `%`). The original
    // `value + alignment - rem` instead computed `value + alignment`
    // first, which overflows `u64` when `alignment` is large — panicking
    // in debug and wrapping in release. `validate_alignment` (called by
    // callers before this function ever sees an attacker-controlled
    // alignment) already rejects anything large enough to matter here;
    // this `checked_add`/saturate is defense in depth for direct callers
    // of this public-ish utility.
    let pad = alignment - rem;
    value.saturating_add(pad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GGUF_MAGIC;

    /// Build a minimal valid GGUF v3 file in memory for testing.
    fn build_test_gguf() -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&2u64.to_le_bytes()); // 2 KV pairs

        // KV pair 1: "general.architecture" = "llama"
        write_string_v3(&mut data, "general.architecture");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v3(&mut data, "llama");

        // KV pair 2: "llama.block_count" = 32u32
        write_string_v3(&mut data, "llama.block_count");
        data.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        data.extend_from_slice(&32u32.to_le_bytes());

        // Tensor info: "output.weight", 2D [32, 32], Q4_0, offset 0
        write_string_v3(&mut data, "output.weight");
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&32u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&32u64.to_le_bytes()); // dim 1
        data.extend_from_slice(&(GgufTensorType::Q4_0 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Pad to alignment (32 bytes)
        let current = data.len();
        let aligned = align_up(current as u64, 32) as usize;
        data.resize(aligned, 0);

        // Add some fake tensor data
        data.resize(aligned + 1024, 0xAB);

        data
    }

    fn write_string_v3(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn test_parse_full_gguf() {
        let data = build_test_gguf();
        let gguf = GgufFile::parse(&data).expect("should parse");

        assert_eq!(gguf.header.version, 3);
        assert_eq!(gguf.header.tensor_count, 1);
        assert_eq!(gguf.header.metadata_kv_count, 2);
        assert_eq!(gguf.architecture().unwrap(), "llama");
        assert_eq!(gguf.metadata.get_u32("llama.block_count").unwrap(), 32);
        assert_eq!(gguf.tensors.len(), 1);

        let tensor = gguf.tensors.get("output.weight").unwrap();
        assert_eq!(tensor.n_dims, 2);
        assert_eq!(tensor.dimensions, vec![32, 32]);
        assert_eq!(tensor.tensor_type, GgufTensorType::Q4_0);
    }

    #[test]
    fn test_tensor_data_access() {
        let data = build_test_gguf();
        let gguf = GgufFile::parse(&data).expect("should parse");
        let tensor_bytes = gguf.tensor_data(&data, "output.weight").unwrap();
        assert!(!tensor_bytes.is_empty());
    }

    #[test]
    fn test_missing_tensor() {
        let data = build_test_gguf();
        let gguf = GgufFile::parse(&data).expect("should parse");
        assert!(gguf.tensor_data(&data, "nonexistent").is_err());
    }

    #[test]
    fn test_model_name_present() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&2u64.to_le_bytes()); // 2 KV pairs

        // KV pair 1: "general.architecture" = "llama"
        write_string_v3(&mut data, "general.architecture");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v3(&mut data, "llama");

        // KV pair 2: "general.name" = "TestModel"
        write_string_v3(&mut data, "general.name");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v3(&mut data, "TestModel");

        let gguf = GgufFile::parse(&data).expect("should parse v3 with name");
        assert_eq!(gguf.model_name(), Some("TestModel"));
        assert_eq!(gguf.architecture().expect("arch"), "llama");
    }

    #[test]
    fn test_model_name_absent() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV pair

        write_string_v3(&mut data, "general.architecture");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v3(&mut data, "mistral");

        let gguf = GgufFile::parse(&data).expect("should parse");
        assert_eq!(gguf.model_name(), None);
    }

    /// Build a minimal valid GGUF v2 byte stream.
    ///
    /// GGUF v2 uses u32 for tensor_count / metadata_kv_count in the header
    /// and u32 string-length prefixes, and u32 array-count.
    fn build_gguf_v2() -> Vec<u8> {
        let mut data = Vec::new();

        // Header (v2 uses u32 for both count fields)
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // version = 2
        data.extend_from_slice(&0u32.to_le_bytes()); // tensor_count = 0  (u32 in v2)
        data.extend_from_slice(&1u32.to_le_bytes()); // metadata_kv_count = 1 (u32 in v2)

        // KV pair: "general.architecture" = "qwen" using v2 string encoding (u32 length)
        write_string_v2(&mut data, "general.architecture");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v2(&mut data, "qwen");

        data
    }

    fn write_string_v2(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes()); // 4-byte length
        buf.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn test_parse_v2_gguf() {
        let data = build_gguf_v2();
        let gguf = GgufFile::parse(&data).expect("v2 GGUF should parse");
        assert_eq!(gguf.header.version, 2);
        assert_eq!(gguf.header.tensor_count, 0);
        assert_eq!(gguf.header.metadata_kv_count, 1);
        assert_eq!(gguf.architecture().expect("arch"), "qwen");
        assert_eq!(gguf.tensors.len(), 0);
    }

    #[test]
    fn test_align_up_already_aligned() {
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(64, 32), 64);
        assert_eq!(align_up(0, 32), 0);
    }

    #[test]
    fn test_align_up_needs_padding() {
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn test_align_up_zero_alignment() {
        // Zero alignment is a no-op
        assert_eq!(align_up(17, 0), 17);
    }

    #[test]
    fn test_parse_all_scalar_value_types() {
        // Build a GGUF v3 with every scalar metadata type to exercise the full match arm coverage.
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&12u64.to_le_bytes()); // 12 KV pairs

        let write_kv = |buf: &mut Vec<u8>, key: &str, vtype: GgufValueType, val_bytes: &[u8]| {
            write_string_v3(buf, key);
            buf.extend_from_slice(&(vtype as u32).to_le_bytes());
            buf.extend_from_slice(val_bytes);
        };

        write_kv(&mut data, "k_u8", GgufValueType::Uint8, &[7u8]);
        write_kv(&mut data, "k_i8", GgufValueType::Int8, &[(-3i8) as u8]);
        write_kv(
            &mut data,
            "k_u16",
            GgufValueType::Uint16,
            &42u16.to_le_bytes(),
        );
        write_kv(
            &mut data,
            "k_i16",
            GgufValueType::Int16,
            &(-10i16).to_le_bytes(),
        );
        write_kv(
            &mut data,
            "k_u32",
            GgufValueType::Uint32,
            &99u32.to_le_bytes(),
        );
        write_kv(
            &mut data,
            "k_i32",
            GgufValueType::Int32,
            &(-5i32).to_le_bytes(),
        );
        write_kv(
            &mut data,
            "k_f32",
            GgufValueType::Float32,
            &1.5f32.to_le_bytes(),
        );
        write_kv(
            &mut data,
            "k_f64",
            GgufValueType::Float64,
            &2.5f64.to_le_bytes(),
        );
        write_kv(&mut data, "k_bool", GgufValueType::Bool, &[1u8]);
        write_kv(
            &mut data,
            "k_u64",
            GgufValueType::Uint64,
            &123u64.to_le_bytes(),
        );
        write_kv(
            &mut data,
            "k_i64",
            GgufValueType::Int64,
            &(-7i64).to_le_bytes(),
        );

        // Array of 2 uint32 values
        write_string_v3(&mut data, "k_arr");
        data.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        data.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes()); // count (v3 = u64)
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&20u32.to_le_bytes());

        let gguf = GgufFile::parse(&data).expect("all scalar types should parse");
        assert_eq!(gguf.metadata.get("k_u8").and_then(|v| v.as_u64()), Some(7));
        assert_eq!(
            gguf.metadata.get("k_u32").and_then(|v| v.as_u64()),
            Some(99)
        );
        assert_eq!(
            gguf.metadata.get("k_bool").and_then(|v| {
                if let MetadataValue::Bool(b) = v {
                    Some(*b)
                } else {
                    None
                }
            }),
            Some(true)
        );
    }

    #[test]
    fn test_tensor_data_too_short_errors() {
        // Build a GGUF where tensor data section is undersized
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 KV pairs

        // Tensor info: "w", 2D [128, 128], F32, offset 0
        write_string_v3(&mut data, "w");
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&128u64.to_le_bytes());
        data.extend_from_slice(&128u64.to_le_bytes());
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Only 4 bytes of tensor data (need 128*128*4 = 65536)
        let current = data.len();
        let aligned = align_up(current as u64, 32) as usize;
        data.resize(aligned, 0);
        data.extend_from_slice(&[0u8; 4]); // far too small

        let gguf = GgufFile::parse(&data).expect("should parse header");
        assert!(
            gguf.tensor_data(&data, "w").is_err(),
            "truncated tensor data should return an error"
        );
    }

    /// Build a GGUF v2 file with tensor info (v2: u32 dims, u64 offset).
    fn build_gguf_v2_with_tensor() -> Vec<u8> {
        let mut data = Vec::new();

        // Header (v2: u32 counts)
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // version = 2
        data.extend_from_slice(&1u32.to_le_bytes()); // tensor_count = 1 (u32)
        data.extend_from_slice(&1u32.to_le_bytes()); // metadata_kv_count = 1 (u32)

        // KV pair: "general.architecture" = "llama" (v2 string = u32 length)
        write_string_v2(&mut data, "general.architecture");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v2(&mut data, "llama");

        // Tensor info: "embed.weight", 2D [64, 32], Q8_0, offset 0
        // v2: u32 string length, u32 dims, u64 offset
        write_string_v2(&mut data, "embed.weight");
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&64u32.to_le_bytes()); // dim 0 (u32 in v2)
        data.extend_from_slice(&32u32.to_le_bytes()); // dim 1 (u32 in v2)
        data.extend_from_slice(&(GgufTensorType::Q8_0 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset (u64 in v2)

        // Pad to alignment
        let current = data.len();
        let aligned = align_up(current as u64, 32) as usize;
        data.resize(aligned, 0);

        // Fake tensor data (Q8_0: 34 bytes per block of 32 weights => 64*32/32*34 = 2176 bytes)
        data.resize(aligned + 4096, 0xCD);

        data
    }

    #[test]
    fn test_parse_v2_with_tensor() {
        let data = build_gguf_v2_with_tensor();
        let gguf = GgufFile::parse(&data).expect("v2 with tensor should parse");

        assert_eq!(gguf.header.version, 2);
        assert_eq!(gguf.header.tensor_count, 1);
        assert_eq!(gguf.header.metadata_kv_count, 1);
        assert_eq!(gguf.architecture().expect("arch"), "llama");

        let tensor = gguf.tensors.get("embed.weight").expect("tensor lookup");
        assert_eq!(tensor.n_dims, 2);
        assert_eq!(tensor.dimensions, vec![64, 32]);
        assert_eq!(tensor.tensor_type, GgufTensorType::Q8_0);
    }

    /// Build a GGUF v1 file (v1: u32 counts, u32 string lengths, u32 dims, u32 offset).
    fn build_gguf_v1() -> Vec<u8> {
        let mut data = Vec::new();

        // Header (v1: u32 counts, same layout as v2)
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // version = 1
        data.extend_from_slice(&1u32.to_le_bytes()); // tensor_count = 1 (u32)
        data.extend_from_slice(&1u32.to_le_bytes()); // metadata_kv_count = 1 (u32)

        // KV pair: "general.architecture" = "gpt2" (v1 string = u32 length)
        write_string_v2(&mut data, "general.architecture");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v2(&mut data, "gpt2");

        // Tensor info: "token_embd.weight", 2D [16, 16], F16, offset 0
        // v1: u32 string length, u32 dims, u32 offset
        write_string_v2(&mut data, "token_embd.weight");
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&16u32.to_le_bytes()); // dim 0 (u32 in v1)
        data.extend_from_slice(&16u32.to_le_bytes()); // dim 1 (u32 in v1)
        data.extend_from_slice(&(GgufTensorType::F16 as u32).to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // offset (u32 in v1!)

        // Pad to alignment
        let current = data.len();
        let aligned = align_up(current as u64, 32) as usize;
        data.resize(aligned, 0);

        // Fake tensor data (F16: 2 bytes per weight => 16*16*2 = 512 bytes)
        data.resize(aligned + 1024, 0xAA);

        data
    }

    #[test]
    fn test_parse_v1_gguf() {
        let data = build_gguf_v1();
        let gguf = GgufFile::parse(&data).expect("v1 GGUF should parse");

        assert_eq!(gguf.header.version, 1);
        assert_eq!(gguf.header.tensor_count, 1);
        assert_eq!(gguf.header.metadata_kv_count, 1);
        assert_eq!(gguf.architecture().expect("arch"), "gpt2");

        let tensor = gguf
            .tensors
            .get("token_embd.weight")
            .expect("tensor lookup");
        assert_eq!(tensor.n_dims, 2);
        assert_eq!(tensor.dimensions, vec![16, 16]);
        assert_eq!(tensor.tensor_type, GgufTensorType::F16);
        assert_eq!(tensor.offset, 0);
    }

    #[test]
    fn test_parse_v1_no_tensors() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // version = 1
        data.extend_from_slice(&0u32.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1u32.to_le_bytes()); // 1 KV

        write_string_v2(&mut data, "general.architecture");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v2(&mut data, "phi");

        let gguf = GgufFile::parse(&data).expect("v1 no-tensor should parse");
        assert_eq!(gguf.header.version, 1);
        assert_eq!(gguf.architecture().expect("arch"), "phi");
        assert_eq!(gguf.tensors.len(), 0);
    }

    #[test]
    fn test_reject_version_0() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // version = 0
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let err = GgufFile::parse(&data).unwrap_err();
        assert!(
            matches!(err, GgufError::UnsupportedVersion { version: 0 }),
            "version 0 should be rejected"
        );
    }

    #[test]
    fn test_reject_version_4() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&4u32.to_le_bytes()); // version = 4
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let err = GgufFile::parse(&data).unwrap_err();
        assert!(
            matches!(err, GgufError::UnsupportedVersion { version: 4 }),
            "version 4 should be rejected"
        );
    }

    #[test]
    fn test_v2_array_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // version = 2
        data.extend_from_slice(&0u32.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1u32.to_le_bytes()); // 1 KV

        // Array of 3 uint32 values with v2 encoding (u32 array count)
        write_string_v2(&mut data, "test.values");
        data.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        data.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // count (u32 in v2)
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&20u32.to_le_bytes());
        data.extend_from_slice(&30u32.to_le_bytes());

        let gguf = GgufFile::parse(&data).expect("v2 array should parse");
        let arr = gguf.metadata.get("test.values").expect("array key");
        if let MetadataValue::Array(elems) = arr {
            assert_eq!(elems.len(), 3);
        } else {
            panic!("expected array metadata value");
        }
    }

    #[test]
    fn test_v3_still_works_after_version_dispatch() {
        // Re-verify the existing v3 builder still works
        let data = build_test_gguf();
        let gguf = GgufFile::parse(&data).expect("v3 should still parse");
        assert_eq!(gguf.header.version, 3);
        assert_eq!(gguf.header.tensor_count, 1);

        let tensor = gguf.tensors.get("output.weight").expect("tensor");
        assert_eq!(tensor.dimensions, vec![32, 32]);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Vulnerability regression tests (V1-V6, see task report)
    // ═══════════════════════════════════════════════════════════════════

    // ── V2: unbounded metadata-array recursion ──────────────────────────

    /// Builds a minimal v3 GGUF with one metadata key whose value is
    /// `nesting` levels of `Array`-of-`Array` (each level costs exactly 12
    /// bytes: a u32 element-type id + a u64 element count), terminated by
    /// an empty `Array<Uint8>` leaf. Built iteratively (not recursively) so
    /// constructing the fixture itself never risks a stack overflow, even
    /// for very large `nesting`.
    fn build_nested_array_kv_gguf(nesting: usize) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV pair

        write_string_v3(&mut data, "deep");
        data.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes()); // KV value type

        for _ in 0..nesting {
            data.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes()); // elem type = Array
            data.extend_from_slice(&1u64.to_le_bytes()); // 1 element
        }
        // Terminal leaf: an empty array of Uint8.
        data.extend_from_slice(&(GgufValueType::Uint8 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        data
    }

    #[test]
    fn test_v2_moderate_nested_array_depth_rejected() {
        // 20 levels is trivial for the old unbounded recursion (would
        // parse fine) but far exceeds MAX_METADATA_ARRAY_DEPTH.
        let data = build_nested_array_kv_gguf(20);
        let err =
            GgufFile::parse(&data).expect_err("array nesting beyond the depth limit must error");
        assert!(
            matches!(err, GgufError::InvalidMetadata { .. }),
            "expected InvalidMetadata, got {err:?}"
        );
    }

    #[test]
    fn test_v2_shallow_nested_array_still_parses() {
        // 3 levels is well within the limit and must keep working
        // identically to before the fix.
        let data = build_nested_array_kv_gguf(3);
        let gguf = GgufFile::parse(&data).expect("shallow nesting must still parse");
        assert!(gguf.metadata.get("deep").is_some());
    }

    /// Exact-repro regression: 200,000 levels of `Array`-of-`Array` nesting
    /// (~2.4 MB file, matching the auditor's repro almost byte-for-byte)
    /// used to recurse one native stack frame per level with no bound,
    /// causing an uncatchable stack-overflow abort in a release build. The
    /// depth limit now rejects the file after only 9 levels of *actual*
    /// recursion regardless of how deep it claims to go, so this returns a
    /// typed error near-instantly instead of ever attempting to recurse
    /// 200,000 levels deep.
    #[test]
    fn test_v2_deeply_nested_array_200k_levels_does_not_abort() {
        let data = build_nested_array_kv_gguf(200_000);
        assert!(
            data.len() > 2_000_000,
            "sanity: fixture should be a couple MB, matching the reported repro"
        );
        let err = GgufFile::parse(&data)
            .expect_err("200,000 levels of nesting must be rejected, not recursed into");
        assert!(matches!(err, GgufError::InvalidMetadata { .. }));
    }

    // ── V3: unchecked arithmetic on file-derived tensor offsets/sizes ──────

    /// The most dangerous variant from the report: with the old unchecked
    /// `data_offset() + info.offset` addition, `32 + (u64::MAX - 16)` wraps
    /// to `15`, which passes the `end > data_len` bounds check and hands
    /// back bytes from *inside* the GGUF header instead of erroring. Built
    /// directly against `GgufFile`'s public fields (bypassing byte-level
    /// parsing) to pin down the exact arithmetic in isolation.
    #[test]
    fn test_v3_tensor_offset_overflow_rejected_not_wrong_data() {
        let mut tensors = TensorStore::new();
        tensors.set_data_offset(32);
        tensors
            .try_insert(TensorInfo {
                name: "evil".to_string(),
                n_dims: 1,
                dimensions: vec![1],
                tensor_type: GgufTensorType::F32,
                offset: u64::MAX - 16,
            })
            .expect("test: insert");

        let gguf = GgufFile {
            header: GgufHeader {
                version: 3,
                tensor_count: 1,
                metadata_kv_count: 0,
            },
            metadata: MetadataStore::new(),
            tensors,
            alignment: 32,
        };

        let data = vec![0xABu8; 64];
        let err = gguf
            .tensor_data(&data, "evil")
            .expect_err("offset overflow must be rejected, never wrap to an in-bounds start");
        assert!(matches!(err, GgufError::UnexpectedEof { .. }));
    }

    /// Dimensions overflowing `u64` when multiplied together (`n_elements`)
    /// must be rejected at tensor-data-access time via
    /// `TensorInfo::try_data_size`, not silently wrapped to a small size
    /// (release) or allowed to panic (debug) inside the old infallible
    /// `data_size()` path.
    #[test]
    fn test_v3_tensor_dims_overflow_rejected_at_data_access() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 KV

        write_string_v3(&mut data, "huge");
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&(1u64 << 32).to_le_bytes()); // dim 0 = 2^32
        data.extend_from_slice(&(1u64 << 32).to_le_bytes()); // dim 1 = 2^32 (product overflows u64)
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        data.resize(128, 0);

        // Parsing succeeds — dimension values themselves are not validated
        // eagerly (matches the existing `test_tensor_data_too_short_errors`
        // contract: bounds/size problems surface at data-access time).
        let gguf = GgufFile::parse(&data).expect("header-level parse must still succeed");
        let err = gguf
            .tensor_data(&data, "huge")
            .expect_err("overflowing dims must be rejected, not wrapped to a small size");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
    }

    // ── V4: n_dims must be validated before use ─────────────────────────

    #[test]
    fn test_v4_n_dims_exceeds_max_rejected() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 KV

        write_string_v3(&mut data, "w");
        data.extend_from_slice(&5u32.to_le_bytes()); // n_dims = 5, exceeds the cap of 4
        for _ in 0..5 {
            data.extend_from_slice(&2u64.to_le_bytes()); // 5 fully valid dim values
        }
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        // The file is otherwise well-formed — before the fix this parsed
        // successfully and silently accepted an invalid tensor shape.
        let err = GgufFile::parse(&data).expect_err("n_dims > 4 must be rejected");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
    }

    #[test]
    fn test_v4_n_dims_zero_rejected() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 KV

        write_string_v3(&mut data, "w");
        data.extend_from_slice(&0u32.to_le_bytes()); // n_dims = 0
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let err = GgufFile::parse(&data).expect_err("n_dims == 0 must be rejected");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
    }

    // ── V6: duplicate tensor names must not silently collapse ──────────

    #[test]
    fn test_v6_duplicate_tensor_names_rejected() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes()); // declares 2 tensors
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 KV

        for offset in [0u64, 4u64] {
            write_string_v3(&mut data, "x"); // same name both times
            data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            data.extend_from_slice(&1u64.to_le_bytes()); // dim
            data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
            data.extend_from_slice(&offset.to_le_bytes());
        }

        // Before the fix this parsed successfully and silently collapsed
        // to a single-entry store despite declaring 2 tensors.
        let err = GgufFile::parse(&data).expect_err("duplicate tensor name must be rejected");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
    }

    // ── align_up overflow (general.alignment is attacker-controlled) ───

    #[test]
    fn test_alignment_overflow_value_rejected_not_panicking() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV

        write_string_v3(&mut data, "general.alignment");
        data.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
        data.extend_from_slice(&u64::MAX.to_le_bytes());

        // Before the fix, `align_up(reader.position(), u64::MAX)` computed
        // `value + alignment` first, overflowing `u64` — a debug-mode
        // panic (and a release-mode wrap to an attacker-chosen small
        // `data_section_offset`).
        let err = GgufFile::parse(&data)
            .expect_err("astronomically large alignment must be rejected, not panic");
        assert!(matches!(err, GgufError::InvalidMetadata { .. }));
    }

    #[test]
    fn test_alignment_non_power_of_two_rejected() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        write_string_v3(&mut data, "general.alignment");
        data.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
        data.extend_from_slice(&3u64.to_le_bytes()); // not a power of two

        let err = GgufFile::parse(&data).expect_err("non-power-of-two alignment must be rejected");
        assert!(matches!(err, GgufError::InvalidMetadata { .. }));
    }

    #[test]
    fn test_alignment_legitimate_value_still_works() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        write_string_v3(&mut data, "general.alignment");
        data.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
        data.extend_from_slice(&64u64.to_le_bytes()); // legitimate power-of-two alignment

        let gguf = GgufFile::parse(&data).expect("legitimate alignment must still parse");
        assert_eq!(gguf.alignment, 64);
        assert_eq!(gguf.tensors.data_offset() % 64, 0);
    }
}

//! Core GGUF parse logic generic over any [`Source`].
//!
//! This module provides the `parse_gguf` function family which can be driven
//! from any byte source implementing the [`Source`] trait — both in `std`
//! environments (via [`ReadSource`][crate::source::ReadSource] or
//! [`SliceSource`][crate::source::SliceSource]) and in `no_std + alloc` environments (via
//! [`SliceSource`][crate::source::SliceSource]).
//!
//! All parse helpers in this module use `alloc` types (`String`, `Vec`) and
//! `core` primitives, making them fully `no_std`-compatible.

#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use core::fmt;

use crate::error::{GgufError, GgufResult};
use crate::metadata::{MetadataStore, MetadataValue};
use crate::source::Source;
use crate::tensor_info::{TensorInfo, TensorStore};
use crate::types::{GgufTensorType, GgufValueType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};

// ── Source-error bridging ───────────────────────────────────────────────────

/// Helper: convert a source read error into a `GgufError::UnexpectedEof`.
#[inline]
fn eof_at(offset: u64) -> GgufError {
    GgufError::UnexpectedEof { offset }
}

// ── Low-level primitive readers ─────────────────────────────────────────────

fn read_u8<S: Source>(src: &mut S) -> GgufResult<u8>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let offset = src.position();
    let mut buf = [0u8; 1];
    src.read_exact(&mut buf).map_err(|_| eof_at(offset))?;
    Ok(buf[0])
}

fn read_u16_le<S: Source>(src: &mut S) -> GgufResult<u16>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let offset = src.position();
    let mut buf = [0u8; 2];
    src.read_exact(&mut buf).map_err(|_| eof_at(offset))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16_le<S: Source>(src: &mut S) -> GgufResult<i16>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let offset = src.position();
    let mut buf = [0u8; 2];
    src.read_exact(&mut buf).map_err(|_| eof_at(offset))?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32_le<S: Source>(src: &mut S) -> GgufResult<u32>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let offset = src.position();
    let mut buf = [0u8; 4];
    src.read_exact(&mut buf).map_err(|_| eof_at(offset))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_le<S: Source>(src: &mut S) -> GgufResult<i32>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let offset = src.position();
    let mut buf = [0u8; 4];
    src.read_exact(&mut buf).map_err(|_| eof_at(offset))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64_le<S: Source>(src: &mut S) -> GgufResult<u64>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let offset = src.position();
    let mut buf = [0u8; 8];
    src.read_exact(&mut buf).map_err(|_| eof_at(offset))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64_le<S: Source>(src: &mut S) -> GgufResult<i64>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let offset = src.position();
    let mut buf = [0u8; 8];
    src.read_exact(&mut buf).map_err(|_| eof_at(offset))?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32_le<S: Source>(src: &mut S) -> GgufResult<f32>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let offset = src.position();
    let mut buf = [0u8; 4];
    src.read_exact(&mut buf).map_err(|_| eof_at(offset))?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64_le<S: Source>(src: &mut S) -> GgufResult<f64>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let offset = src.position();
    let mut buf = [0u8; 8];
    src.read_exact(&mut buf).map_err(|_| eof_at(offset))?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bool<S: Source>(src: &mut S) -> GgufResult<bool>
where
    S::Error: fmt::Debug + fmt::Display,
{
    Ok(read_u8(src)? != 0)
}

/// Read a GGUF v3 string: u64 length prefix + UTF-8 bytes (no null terminator).
fn read_string_v3<S: Source>(src: &mut S) -> GgufResult<String>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let str_offset = src.position();
    let len = read_u64_le(src)? as usize;
    let mut bytes = Vec::with_capacity(len.min(1_024 * 1_024));
    bytes.resize(len, 0u8);
    src.read_exact(&mut bytes).map_err(|_| eof_at(str_offset))?;
    String::from_utf8(bytes).map_err(|e| GgufError::InvalidString {
        offset: str_offset,
        source: e,
    })
}

/// Read a GGUF v2 string: u32 length prefix + UTF-8 bytes.
fn read_string_v2<S: Source>(src: &mut S) -> GgufResult<String>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let str_offset = src.position();
    let len = read_u32_le(src)? as usize;
    let mut bytes = Vec::with_capacity(len.min(1_024 * 1_024));
    bytes.resize(len, 0u8);
    src.read_exact(&mut bytes).map_err(|_| eof_at(str_offset))?;
    String::from_utf8(bytes).map_err(|e| GgufError::InvalidString {
        offset: str_offset,
        source: e,
    })
}

/// Read a string dispatched by version.
#[inline]
fn read_string_versioned<S: Source>(src: &mut S, version: u32) -> GgufResult<String>
where
    S::Error: fmt::Debug + fmt::Display,
{
    if version >= 3 {
        read_string_v3(src)
    } else {
        read_string_v2(src)
    }
}

// ── Header ──────────────────────────────────────────────────────────────────

/// The parsed GGUF file header produced by [`read_header`].
pub struct RawHeader {
    /// GGUF format version (1, 2, or 3).
    pub version: u32,
    /// Number of tensors.
    pub tensor_count: u64,
    /// Number of metadata KV pairs.
    pub metadata_kv_count: u64,
}

/// Read and validate the GGUF header from `src`.
pub fn read_header<S: Source>(src: &mut S) -> GgufResult<RawHeader>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let magic_offset = src.position();
    let magic = read_u32_le(src).map_err(|_| eof_at(magic_offset))?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic { magic });
    }

    let version_offset = src.position();
    let version = read_u32_le(src).map_err(|_| eof_at(version_offset))?;
    if !(1..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion { version });
    }

    let tensor_count = read_count(src, version)?;
    let metadata_kv_count = read_count(src, version)?;

    Ok(RawHeader {
        version,
        tensor_count,
        metadata_kv_count,
    })
}

/// Read a version-dispatched count field (u64 for v3, u32 for v2/v1).
fn read_count<S: Source>(src: &mut S, version: u32) -> GgufResult<u64>
where
    S::Error: fmt::Debug + fmt::Display,
{
    if version >= 3 {
        read_u64_le(src)
    } else {
        read_u32_le(src).map(u64::from)
    }
}

// ── Metadata ────────────────────────────────────────────────────────────────

/// Read all metadata KV pairs from `src`.
pub fn read_metadata_kv<S: Source>(
    src: &mut S,
    count: u64,
    version: u32,
) -> GgufResult<MetadataStore>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let mut store = MetadataStore::new();

    for _ in 0..count {
        let key = read_string_versioned(src, version)?;

        let value_type_id = read_u32_le(src)?;
        let value_type =
            GgufValueType::from_u32(value_type_id).ok_or_else(|| GgufError::InvalidMetadata {
                key: key.clone(),
                reason: format!("unknown value type: {value_type_id}"),
            })?;

        let value = read_metadata_value(src, value_type, version)?;
        store.insert(key, value);
    }

    Ok(store)
}

/// Read a single typed metadata value.
fn read_metadata_value<S: Source>(
    src: &mut S,
    value_type: GgufValueType,
    version: u32,
) -> GgufResult<MetadataValue>
where
    S::Error: fmt::Debug + fmt::Display,
{
    match value_type {
        GgufValueType::Uint8 => Ok(MetadataValue::Uint8(read_u8(src)?)),
        GgufValueType::Int8 => Ok(MetadataValue::Int8(read_u8(src)? as i8)),
        GgufValueType::Uint16 => Ok(MetadataValue::Uint16(read_u16_le(src)?)),
        GgufValueType::Int16 => Ok(MetadataValue::Int16(read_i16_le(src)?)),
        GgufValueType::Uint32 => Ok(MetadataValue::Uint32(read_u32_le(src)?)),
        GgufValueType::Int32 => Ok(MetadataValue::Int32(read_i32_le(src)?)),
        GgufValueType::Float32 => Ok(MetadataValue::Float32(read_f32_le(src)?)),
        GgufValueType::Float64 => Ok(MetadataValue::Float64(read_f64_le(src)?)),
        GgufValueType::Bool => Ok(MetadataValue::Bool(read_bool(src)?)),
        GgufValueType::String => {
            let s = read_string_versioned(src, version)?;
            Ok(MetadataValue::String(s))
        }
        GgufValueType::Uint64 => Ok(MetadataValue::Uint64(read_u64_le(src)?)),
        GgufValueType::Int64 => Ok(MetadataValue::Int64(read_i64_le(src)?)),
        GgufValueType::Array => {
            let elem_type_id = read_u32_le(src)?;
            let elem_type = GgufValueType::from_u32(elem_type_id).ok_or_else(|| {
                GgufError::InvalidMetadata {
                    key: "<array>".to_string(),
                    reason: format!("unknown array element type: {elem_type_id}"),
                }
            })?;

            let count = if version >= 3 {
                read_u64_le(src)? as usize
            } else {
                read_u32_le(src)? as usize
            };

            let mut elements = Vec::with_capacity(count.min(1_000_000));
            for _ in 0..count {
                elements.push(read_metadata_value(src, elem_type, version)?);
            }
            Ok(MetadataValue::Array(elements))
        }
    }
}

// ── Tensor infos ────────────────────────────────────────────────────────────

/// Read all tensor info entries from `src`.
pub fn read_tensor_infos<S: Source>(
    src: &mut S,
    count: u64,
    version: u32,
) -> GgufResult<TensorStore>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let mut store = TensorStore::new();

    for _ in 0..count {
        let name = read_string_versioned(src, version)?;

        let n_dims = read_u32_le(src)?;

        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = if version >= 3 {
                read_u64_le(src)?
            } else {
                read_u32_le(src)? as u64
            };
            dimensions.push(dim);
        }

        let type_id = read_u32_le(src)?;
        let tensor_type =
            GgufTensorType::from_u32(type_id).ok_or(GgufError::UnsupportedQuantType { type_id })?;

        let offset = if version >= 2 {
            read_u64_le(src)?
        } else {
            read_u32_le(src)? as u64
        };

        store.insert(TensorInfo {
            name,
            n_dims,
            dimensions,
            tensor_type,
            offset,
        });
    }

    Ok(store)
}

// ── Top-level entry point ────────────────────────────────────────────────────

/// Parsed GGUF file result from [`parse_gguf`].
#[derive(Debug)]
pub struct ParsedGguf {
    /// GGUF format version.
    pub version: u32,
    /// Total tensor count.
    pub tensor_count: u64,
    /// Total metadata KV count.
    pub metadata_kv_count: u64,
    /// Parsed metadata store.
    pub metadata: MetadataStore,
    /// Parsed tensor store.
    pub tensors: TensorStore,
    /// Alignment value (from metadata or default 32).
    pub alignment: u64,
    /// Absolute byte offset of the tensor data section.
    pub data_offset: u64,
}

/// Align a value up to the given alignment boundary.
pub fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + alignment - rem
    }
}

/// Parse a complete GGUF file from any [`Source`].
///
/// This is the `no_std`-compatible entry point for the GGUF parser.
/// It reads the header, all metadata KV pairs, and all tensor info entries.
/// It does **not** load tensor data — use the returned `data_offset` and
/// individual `TensorInfo` offsets to locate data in the source.
pub fn parse_gguf<S: Source>(src: &mut S) -> GgufResult<ParsedGguf>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let header = read_header(src)?;

    let metadata = read_metadata_kv(src, header.metadata_kv_count, header.version)?;

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

    let mut tensors = read_tensor_infos(src, header.tensor_count, header.version)?;

    let data_offset = align_up(src.position(), alignment);
    tensors.set_data_offset(data_offset);

    Ok(ParsedGguf {
        version: header.version,
        tensor_count: header.tensor_count,
        metadata_kv_count: header.metadata_kv_count,
        metadata,
        tensors,
        alignment,
        data_offset,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SliceSource;

    // ── Test helpers ────────────────────────────────────────────────────────

    fn write_string_v3(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_string_v2(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn build_minimal_v3_header(tensor_count: u64, kv_count: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf
    }

    // ── slice_source_round_trip ─────────────────────────────────────────────

    #[test]
    fn slice_source_round_trip() {
        let original: Vec<u8> = (0u8..=255u8).collect();
        let mut src = SliceSource::new(&original);

        let mut first_half = vec![0u8; 128];
        src.read_exact(&mut first_half)
            .expect("test: read first half");
        assert_eq!(first_half, &original[..128]);
        assert_eq!(src.position(), 128);

        // Seek back to position 64
        src.seek(64).expect("test: seek");
        assert_eq!(src.position(), 64);

        let mut chunk = vec![0u8; 16];
        src.read_exact(&mut chunk).expect("test: read chunk");
        assert_eq!(chunk, &original[64..80]);
        assert_eq!(src.position(), 80);
    }

    // ── reader_core_parses_minimal_header ───────────────────────────────────

    #[test]
    fn reader_core_parses_minimal_header() {
        // magic + version 3 + 0 tensors + 0 KV
        let buf = build_minimal_v3_header(0, 0);
        let mut src = SliceSource::new(&buf);
        let result = parse_gguf(&mut src).expect("test: parse minimal");
        assert_eq!(result.version, 3);
        assert_eq!(result.tensor_count, 0);
        assert_eq!(result.metadata_kv_count, 0);
        assert_eq!(result.tensors.len(), 0);
        assert!(result.metadata.get("any").is_none());
    }

    // ── reader_core_parses_metadata_kv_strings ──────────────────────────────

    #[test]
    fn reader_core_parses_metadata_kv_strings() {
        let mut buf = build_minimal_v3_header(0, 2);

        // KV 1: "general.architecture" = String "llama"
        write_string_v3(&mut buf, "general.architecture");
        buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v3(&mut buf, "llama");

        // KV 2: "llama.block_count" = Uint32(32)
        write_string_v3(&mut buf, "llama.block_count");
        buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());

        let mut src = SliceSource::new(&buf);
        let result = parse_gguf(&mut src).expect("test: parse kv strings");

        let arch = result
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .expect("test: architecture");
        assert_eq!(arch, "llama");

        let blocks = result
            .metadata
            .get("llama.block_count")
            .and_then(|v| v.as_u32())
            .expect("test: block_count");
        assert_eq!(blocks, 32);
    }

    // ── reader_core_rejects_bad_magic ────────────────────────────────────────

    #[test]
    fn reader_core_rejects_bad_magic() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x_DEAD_BEEFu32.to_le_bytes()); // wrong magic
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let mut src = SliceSource::new(&buf);
        let err = parse_gguf(&mut src).expect_err("test: bad magic must error");
        assert!(
            matches!(err, GgufError::InvalidMagic { .. }),
            "expected InvalidMagic, got {err:?}"
        );
    }

    // ── reader_core_reads_tensor_info ────────────────────────────────────────

    #[test]
    fn reader_core_reads_tensor_info() {
        let mut buf = build_minimal_v3_header(1, 0);

        // Tensor "embed.weight", 2D [64, 32], Q8_0, offset 0
        write_string_v3(&mut buf, "embed.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&64u64.to_le_bytes()); // dim0
        buf.extend_from_slice(&32u64.to_le_bytes()); // dim1
        buf.extend_from_slice(&(GgufTensorType::Q8_0 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut src = SliceSource::new(&buf);
        let result = parse_gguf(&mut src).expect("test: parse tensor info");

        let tensor = result
            .tensors
            .get("embed.weight")
            .expect("test: tensor lookup");
        assert_eq!(tensor.n_dims, 2);
        assert_eq!(tensor.dimensions, vec![64, 32]);
        assert_eq!(tensor.tensor_type, GgufTensorType::Q8_0);
        assert_eq!(tensor.offset, 0);
    }

    // ── reader_core_rejects_unsupported_version ──────────────────────────────

    #[test]
    fn reader_core_rejects_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&99u32.to_le_bytes()); // unsupported version
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let mut src = SliceSource::new(&buf);
        let err = parse_gguf(&mut src).expect_err("test: unsupported version must error");
        assert!(
            matches!(err, GgufError::UnsupportedVersion { .. }),
            "expected UnsupportedVersion, got {err:?}"
        );
    }

    // ── reader_core_parses_v2 ────────────────────────────────────────────────

    #[test]
    fn reader_core_parses_v2() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // version = 2
        buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_count (u32 in v2)
        buf.extend_from_slice(&1u32.to_le_bytes()); // kv_count (u32 in v2)

        // KV: "general.architecture" = "qwen" using v2 strings
        write_string_v2(&mut buf, "general.architecture");
        buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v2(&mut buf, "qwen");

        let mut src = SliceSource::new(&buf);
        let result = parse_gguf(&mut src).expect("test: parse v2");

        assert_eq!(result.version, 2);
        let arch = result
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .expect("test: arch");
        assert_eq!(arch, "qwen");
    }

    // ── align_up tests ───────────────────────────────────────────────────────

    #[test]
    fn align_up_already_aligned() {
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(64, 32), 64);
        assert_eq!(align_up(0, 32), 0);
    }

    #[test]
    fn align_up_needs_padding() {
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn align_up_zero_alignment() {
        assert_eq!(align_up(17, 0), 17);
    }
}

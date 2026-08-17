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

/// Maximum nesting depth allowed for `Array`-of-`Array` metadata values.
/// See the identical constant in `parser.rs` for the full rationale — this
/// module mirrors that parser over a generic [`Source`] instead of an
/// in-memory slice.
const MAX_METADATA_ARRAY_DEPTH: u32 = 8;

/// Maximum number of tensor dimensions accepted (mirrors `parser.rs`).
const MAX_TENSOR_DIMS: u32 = 4;

/// Hard fallback cap (in bytes) for a single string's declared length when
/// the source cannot report how many bytes remain (`remaining_hint()`
/// returns `None`, e.g. a network stream that has not reached EOF). No
/// legitimate GGUF string (a tensor name or a metadata string value) is
/// anywhere near this size; it exists purely to stop `bytes.resize(len, 0)`
/// from being handed an attacker-controlled length in the gigabytes,
/// which can abort the process via `handle_alloc_error` from just 8 bytes
/// of input.
const MAX_STRING_LEN_FALLBACK: u64 = 64 * 1024 * 1024;

/// Fallback cap on a metadata array's element count when the source cannot
/// report remaining bytes. Deliberately generous (legitimate GGUF arrays —
/// e.g. a 250k-entry tokenizer vocabulary — can be large) since rejecting
/// count alone is not the defense in this branch; not preallocating that
/// capacity is (see `read_metadata_value` below).
const ARRAY_COUNT_HARD_CEILING: u64 = 64 * 1024 * 1024;

/// Minimum number of encoded bytes a single metadata value of `value_type`
/// can possibly occupy (mirrors `parser.rs::min_encoded_value_size`).
fn min_encoded_value_size(value_type: GgufValueType, version: u32) -> u64 {
    match value_type {
        GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => 1,
        GgufValueType::Uint16 | GgufValueType::Int16 => 2,
        GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => 4,
        GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => 8,
        GgufValueType::String => {
            if version >= 3 {
                8
            } else {
                4
            }
        }
        GgufValueType::Array => {
            if version >= 3 {
                12
            } else {
                8
            }
        }
    }
}

/// Maximum alignment value accepted from `general.alignment` metadata
/// (mirrors `parser.rs::MAX_GGUF_ALIGNMENT`).
const MAX_GGUF_ALIGNMENT: u64 = 1 << 30;

/// Validate a `general.alignment` metadata value before it is used to
/// compute the tensor data-section offset. See `parser.rs::validate_alignment`
/// for the full rationale.
fn validate_alignment(alignment: u64) -> GgufResult<()> {
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

/// Validate a file-declared byte length before it drives an allocation.
///
/// When `src.remaining_hint()` is known (e.g. a `SliceSource` over an
/// in-memory buffer), `len` is rejected outright if it cannot possibly fit
/// — this can never false-positive on a legitimate file. When the source's
/// remaining length is unknown (e.g. a streaming network source),
/// `len` is instead checked against a generous hard cap: no legitimate
/// single GGUF string needs anywhere near that much space, so this closes
/// the allocation-bomb / `handle_alloc_error` abort without needing to
/// know the stream's true length.
fn validate_len_before_alloc<S: Source>(src: &S, len: u64, offset: u64) -> GgufResult<()>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let max_len = src.remaining_hint().unwrap_or(MAX_STRING_LEN_FALLBACK);
    if len > max_len {
        return Err(eof_at(offset));
    }
    Ok(())
}

/// Read a GGUF v3 string: u64 length prefix + UTF-8 bytes (no null terminator).
fn read_string_v3<S: Source>(src: &mut S) -> GgufResult<String>
where
    S::Error: fmt::Debug + fmt::Display,
{
    let str_offset = src.position();
    let len_u64 = read_u64_le(src)?;
    validate_len_before_alloc(src, len_u64, str_offset)?;
    // Safe: `validate_len_before_alloc` bounded `len_u64` to at most
    // `MAX_STRING_LEN_FALLBACK` (comfortably inside `usize` on every
    // supported target) or to the source's known remaining byte count
    // (itself derived from a `usize`), so the cast below cannot truncate.
    let len = len_u64 as usize;
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
    let len_u64 = u64::from(read_u32_le(src)?);
    validate_len_before_alloc(src, len_u64, str_offset)?;
    let len = len_u64 as usize;
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

        let value = read_metadata_value(src, value_type, version, 0)?;
        store.insert(key, value);
    }

    Ok(store)
}

/// Read a single typed metadata value.
///
/// `depth` counts `Array`-of-`Array` nesting levels; see
/// `MAX_METADATA_ARRAY_DEPTH` for why this is bounded.
fn read_metadata_value<S: Source>(
    src: &mut S,
    value_type: GgufValueType,
    version: u32,
    depth: u32,
) -> GgufResult<MetadataValue>
where
    S::Error: fmt::Debug + fmt::Display,
{
    if depth > MAX_METADATA_ARRAY_DEPTH {
        return Err(GgufError::InvalidMetadata {
            key: "<array>".to_string(),
            reason: format!("array nesting depth exceeds the limit of {MAX_METADATA_ARRAY_DEPTH}"),
        });
    }
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

            let count_u64 = if version >= 3 {
                read_u64_le(src)?
            } else {
                u64::from(read_u32_le(src)?)
            };

            let min_elem_size = min_encoded_value_size(elem_type, version);
            let (count, prealloc_cap) = match src.remaining_hint() {
                // Remaining length is known (e.g. `SliceSource`): reject a
                // count that could not possibly fit — this can never
                // false-positive on a legitimate array.
                Some(remaining) => {
                    let max_count = remaining / min_elem_size;
                    if count_u64 > max_count {
                        return Err(GgufError::InvalidMetadata {
                            key: "<array>".to_string(),
                            reason: format!(
                                "array count {count_u64} cannot fit in the {remaining} bytes remaining (min {min_elem_size} bytes/element)"
                            ),
                        });
                    }
                    let count = count_u64 as usize; // safe: bounded by `remaining`, itself a usize
                    (count, count.min(4096))
                }
                // Remaining length is unknown (e.g. a network stream that
                // has not hit EOF): don't reject on count alone — large
                // legitimate arrays exist (a 150k-250k entry tokenizer
                // vocabulary). Instead cap it against a generous hard
                // ceiling and, crucially, don't preallocate that much
                // capacity; a truncated/malicious stream fails fast on the
                // next `read_exact` regardless.
                None => {
                    if count_u64 > ARRAY_COUNT_HARD_CEILING {
                        return Err(GgufError::InvalidMetadata {
                            key: "<array>".to_string(),
                            reason: format!(
                                "array count {count_u64} exceeds the hard ceiling of {ARRAY_COUNT_HARD_CEILING} (stream length unknown)"
                            ),
                        });
                    }
                    let count = count_u64 as usize; // safe: bounded by the u64 ceiling above
                    (count, count.min(64))
                }
            };

            let mut elements = Vec::with_capacity(prealloc_cap);
            for _ in 0..count {
                elements.push(read_metadata_value(src, elem_type, version, depth + 1)?);
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

        store.try_insert(TensorInfo {
            name,
            n_dims,
            dimensions,
            tensor_type,
            offset,
        })?;
    }

    if store.len() as u64 != count {
        return Err(GgufError::IntegrityError {
            tensor_name: "<tensor_infos>".to_string(),
            reason: format!("expected {count} tensors, parsed {}", store.len()),
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
        return value;
    }
    // See `parser.rs::align_up` for why this is `checked_add` rather than
    // the original `value + alignment - rem` (which overflows `u64` for
    // large, attacker-controlled `alignment`). This function is publicly
    // exported, so it stays panic-free regardless of what callers outside
    // this crate's own validated parse path pass in.
    let pad = alignment - rem;
    value.saturating_add(pad)
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
    validate_alignment(alignment)?;

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

    // ═══════════════════════════════════════════════════════════════════
    // Vulnerability regression tests (V1-V6) — Source-generic parse path.
    // Mirrors parser.rs's regression tests; see that module for the full
    // rationale behind each one.
    // ═══════════════════════════════════════════════════════════════════

    fn build_nested_array_kv_gguf(nesting: usize) -> Vec<u8> {
        let mut data = build_minimal_v3_header(0, 1);
        write_string_v3(&mut data, "deep");
        data.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        for _ in 0..nesting {
            data.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            data.extend_from_slice(&1u64.to_le_bytes());
        }
        data.extend_from_slice(&(GgufValueType::Uint8 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data
    }

    #[test]
    fn test_v2_moderate_nested_array_depth_rejected() {
        let data = build_nested_array_kv_gguf(20);
        let mut src = SliceSource::new(&data);
        let err = parse_gguf(&mut src).expect_err("nesting beyond the depth limit must error");
        assert!(matches!(err, GgufError::InvalidMetadata { .. }));
    }

    #[test]
    fn test_v2_shallow_nested_array_still_parses() {
        let data = build_nested_array_kv_gguf(3);
        let mut src = SliceSource::new(&data);
        let result = parse_gguf(&mut src).expect("shallow nesting must still parse");
        assert!(result.metadata.get("deep").is_some());
    }

    /// Exact-repro regression matching the auditor's ~2.4 MB / 200,000-level
    /// file; see `parser.rs`'s identical test for the full rationale.
    #[test]
    fn test_v2_deeply_nested_array_200k_levels_does_not_abort() {
        let data = build_nested_array_kv_gguf(200_000);
        assert!(data.len() > 2_000_000);
        let mut src = SliceSource::new(&data);
        let err =
            parse_gguf(&mut src).expect_err("200,000 levels of nesting must be rejected fast");
        assert!(matches!(err, GgufError::InvalidMetadata { .. }));
    }

    #[test]
    fn test_v4_n_dims_exceeds_max_rejected() {
        let mut data = build_minimal_v3_header(1, 0);
        write_string_v3(&mut data, "w");
        data.extend_from_slice(&5u32.to_le_bytes()); // n_dims = 5, exceeds the cap of 4
        for _ in 0..5 {
            data.extend_from_slice(&2u64.to_le_bytes());
        }
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut src = SliceSource::new(&data);
        let err = parse_gguf(&mut src).expect_err("n_dims > 4 must be rejected");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
    }

    #[test]
    fn test_v4_n_dims_zero_rejected() {
        let mut data = build_minimal_v3_header(1, 0);
        write_string_v3(&mut data, "w");
        data.extend_from_slice(&0u32.to_le_bytes()); // n_dims = 0
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut src = SliceSource::new(&data);
        let err = parse_gguf(&mut src).expect_err("n_dims == 0 must be rejected");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
    }

    /// V4 allocation-bomb defused by construction: the dimension count
    /// check runs *before* `Vec::with_capacity(n_dims as usize)`, so an
    /// absurd `n_dims` (which would otherwise request billions of `u64`
    /// slots from 4 bytes of input) never reaches the allocator at all.
    /// Deliberately not exercised with the true `u32::MAX` value here — see
    /// the task report for why that specific case was reasoned about
    /// rather than empirically reproduced in this sandbox.
    #[test]
    fn test_v4_n_dims_huge_rejected_before_allocating() {
        let mut data = build_minimal_v3_header(1, 0);
        write_string_v3(&mut data, "w");
        data.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // n_dims = u32::MAX
                                                               // No further bytes — if the fix didn't run before allocating, the
                                                               // huge `Vec::with_capacity` attempt would come first anyway.

        let mut src = SliceSource::new(&data);
        let err = parse_gguf(&mut src).expect_err("must reject before attempting to allocate");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
    }

    /// V4 string-length allocation-bomb: a length wildly exceeding the
    /// source's known remaining bytes must be rejected immediately via
    /// `remaining_hint()`, never handed to `bytes.resize(len, 0)`.
    #[test]
    fn test_v4_string_length_exceeding_remaining_rejected() {
        let mut data = build_minimal_v3_header(0, 1);
        data.extend_from_slice(&10_000_000u64.to_le_bytes()); // declared key length: 10 MB
        data.extend_from_slice(b"tiny"); // actual remaining bytes: nowhere close

        let mut src = SliceSource::new(&data);
        let err =
            parse_gguf(&mut src).expect_err("oversized length vs. known remaining must error");
        assert!(matches!(err, GgufError::UnexpectedEof { .. }));
    }

    #[test]
    fn test_v6_duplicate_tensor_names_rejected() {
        let mut data = build_minimal_v3_header(2, 0);
        for offset in [0u64, 4u64] {
            write_string_v3(&mut data, "x");
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&1u64.to_le_bytes());
            data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
            data.extend_from_slice(&offset.to_le_bytes());
        }

        let mut src = SliceSource::new(&data);
        let err = parse_gguf(&mut src).expect_err("duplicate tensor name must be rejected");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
    }

    #[test]
    fn test_alignment_overflow_value_rejected_not_panicking() {
        let mut data = build_minimal_v3_header(0, 1);
        write_string_v3(&mut data, "general.alignment");
        data.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
        data.extend_from_slice(&u64::MAX.to_le_bytes());

        let mut src = SliceSource::new(&data);
        let err = parse_gguf(&mut src)
            .expect_err("astronomically large alignment must be rejected, not panic");
        assert!(matches!(err, GgufError::InvalidMetadata { .. }));
    }

    #[test]
    fn test_alignment_non_power_of_two_rejected() {
        let mut data = build_minimal_v3_header(0, 1);
        write_string_v3(&mut data, "general.alignment");
        data.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
        data.extend_from_slice(&3u64.to_le_bytes());

        let mut src = SliceSource::new(&data);
        let err = parse_gguf(&mut src).expect_err("non-power-of-two alignment must be rejected");
        assert!(matches!(err, GgufError::InvalidMetadata { .. }));
    }
}

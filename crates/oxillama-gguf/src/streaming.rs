//! Streaming / lazy GGUF parser.
//!
//! Unlike [`GgufFile::parse()`] which reads everything at once, this parser:
//! 1. Eagerly parses the fixed-size header
//! 2. Eagerly parses metadata (needed for alignment, architecture, etc.)
//! 3. Lazily iterates over tensor info entries (no `Vec<TensorInfo>` allocation)
//! 4. Provides zero-copy access to tensor data via byte offset + length
//!
//! This is useful for:
//! - Large models where loading all tensor info eagerly wastes memory
//! - Streaming from network/disk where you want to process tensors one at a time
//! - Selective loading where you only need specific tensors

use crate::error::{GgufError, GgufResult};
use crate::header::GgufHeader;
use crate::metadata::MetadataStore;
use crate::parser::GgufFile;
use crate::reader::BinaryReader;
use crate::tensor_info::{TensorInfo, TensorStore};
use crate::types::{GgufTensorType, GgufValueType, GGUF_DEFAULT_ALIGNMENT};

/// A streaming GGUF parser that lazily reads tensor data.
///
/// Unlike `GgufFile::parse()` which reads everything at once, this parser:
/// 1. Eagerly parses the fixed-size header
/// 2. Eagerly parses metadata (needed for alignment, architecture, etc.)
/// 3. Lazily iterates over tensor info entries (no `Vec<TensorInfo>` allocation)
/// 4. Provides zero-copy access to tensor data via byte offset + length
///
/// # Example
///
/// ```no_run
/// use oxillama_gguf::streaming::StreamingGgufParser;
///
/// # fn example(data: &[u8]) -> oxillama_gguf::GgufResult<()> {
/// let parser = StreamingGgufParser::new(data)?;
/// println!("Architecture: {}", parser.architecture()?);
///
/// // Iterate over tensors lazily
/// for result in parser.tensor_infos() {
///     let info = result?;
///     println!("{}: {:?} {:?}", info.name, info.dimensions, info.tensor_type);
/// }
///
/// // Or look up a specific tensor
/// let info = parser.find_tensor("output.weight")?;
/// let bytes = parser.tensor_data(&info)?;
/// # Ok(())
/// # }
/// ```
pub struct StreamingGgufParser<'a> {
    data: &'a [u8],
    header: GgufHeader,
    metadata: MetadataStore,
    alignment: u64,
    /// Byte offset where tensor info entries start.
    tensor_info_offset: usize,
    /// Byte offset where tensor data section starts (after all tensor infos, aligned).
    data_section_offset: u64,
}

impl<'a> StreamingGgufParser<'a> {
    /// Parse header and metadata, then position for lazy tensor iteration.
    ///
    /// This eagerly parses the header and all metadata KV pairs. Tensor info
    /// entries are NOT read — they are available via `tensor_infos()` for
    /// lazy iteration.
    ///
    /// To compute the data section offset, we perform a lightweight scan over
    /// tensor info entries (just skipping bytes, not building structs).
    pub fn new(data: &'a [u8]) -> GgufResult<Self> {
        let (header, offset) = GgufHeader::parse(data, 0)?;
        let mut reader = BinaryReader::new(data, offset as usize);

        // Eagerly parse metadata (needed for alignment, architecture, etc.)
        let metadata = parse_metadata_streaming(&mut reader, &header)?;

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        let tensor_info_offset = reader.position();

        // Lightweight scan: skip through all tensor infos to find the end position.
        // This does NOT build TensorInfo structs — just advances the reader past them.
        skip_tensor_infos(&mut reader, &header)?;

        let data_section_offset = align_up(reader.position() as u64, alignment);

        Ok(Self {
            data,
            header,
            metadata,
            alignment,
            tensor_info_offset,
            data_section_offset,
        })
    }

    /// Access the parsed header.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Access the parsed metadata.
    pub fn metadata(&self) -> &MetadataStore {
        &self.metadata
    }

    /// Alignment value (from metadata or default 32).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Get the model architecture string from metadata.
    pub fn architecture(&self) -> GgufResult<&str> {
        self.metadata.get_string("general.architecture")
    }

    /// Create an iterator over tensor info entries.
    ///
    /// Each call creates a fresh iterator from the beginning of the tensor
    /// info section. The iterator reads one `TensorInfo` at a time without
    /// allocating a vector of all entries upfront.
    pub fn tensor_infos(&self) -> TensorInfoIter<'a> {
        TensorInfoIter {
            reader: BinaryReader::new(self.data, self.tensor_info_offset),
            remaining: self.header.tensor_count,
            version: self.header.version,
            data_offset: self.data_section_offset,
        }
    }

    /// Look up a specific tensor by name (scans all tensor infos).
    ///
    /// This is O(n) but avoids building the full `TensorStore`.
    pub fn find_tensor(&self, name: &str) -> GgufResult<TensorInfo> {
        for result in self.tensor_infos() {
            let info = result?;
            if info.name == name {
                return Ok(info);
            }
        }
        Err(GgufError::TensorNotFound {
            name: name.to_string(),
        })
    }

    /// Get raw tensor data bytes for a `TensorInfo` entry.
    ///
    /// Returns a zero-copy slice into the original data buffer.
    pub fn tensor_data(&self, info: &TensorInfo) -> GgufResult<&'a [u8]> {
        let abs_offset = self.data_section_offset + info.offset;
        let size = info.data_size();

        let start = abs_offset as usize;
        let end = start
            .checked_add(size as usize)
            .ok_or(GgufError::UnexpectedEof { offset: abs_offset })?;

        if end > self.data.len() {
            return Err(GgufError::UnexpectedEof { offset: abs_offset });
        }

        Ok(&self.data[start..end])
    }

    /// Selectively load only the named tensors into a `TensorStore`.
    ///
    /// More efficient than loading all tensors when you only need a subset.
    /// Performs a single scan over all tensor infos, only keeping those whose
    /// names match the requested set.
    pub fn load_tensors(&self, names: &[&str]) -> GgufResult<TensorStore> {
        let name_set: std::collections::HashSet<&str> = names.iter().copied().collect();
        let mut store = TensorStore::new();
        store.set_data_offset(self.data_section_offset);

        for result in self.tensor_infos() {
            let info = result?;
            if name_set.contains(info.name.as_str()) {
                store.insert(info);
            }
        }

        Ok(store)
    }

    /// Convert to a full `GgufFile` by eagerly loading all tensor infos.
    pub fn into_full(self) -> GgufResult<GgufFile> {
        let mut tensors = TensorStore::new();
        tensors.set_data_offset(self.data_section_offset);

        for result in self.tensor_infos() {
            tensors.insert(result?);
        }

        Ok(GgufFile {
            header: self.header,
            metadata: self.metadata,
            tensors,
            alignment: self.alignment,
        })
    }
}

/// Iterator over tensor info entries in a GGUF file.
///
/// Reads `TensorInfo` entries one at a time from the binary data without
/// allocating a vector of all entries upfront.
pub struct TensorInfoIter<'a> {
    reader: BinaryReader<'a>,
    remaining: u64,
    version: u32,
    data_offset: u64,
}

impl<'a> TensorInfoIter<'a> {
    /// Parse one tensor info entry from the binary reader.
    fn parse_one(&mut self) -> GgufResult<TensorInfo> {
        let name = if self.version >= 3 {
            self.reader.read_string()?
        } else {
            self.reader.read_string_v2()?
        };

        let n_dims = self.reader.read_u32()?;

        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = if self.version >= 3 {
                self.reader.read_u64()?
            } else {
                self.reader.read_u32()? as u64
            };
            dimensions.push(dim);
        }

        let type_id = self.reader.read_u32()?;
        let tensor_type =
            GgufTensorType::from_u32(type_id).ok_or(GgufError::UnsupportedQuantType { type_id })?;

        let offset = if self.version >= 2 {
            self.reader.read_u64()?
        } else {
            self.reader.read_u32()? as u64
        };

        Ok(TensorInfo {
            name,
            n_dims,
            dimensions,
            tensor_type,
            offset,
        })
    }

    /// Returns the data section offset used by this iterator.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Returns the number of remaining tensor info entries.
    pub fn remaining(&self) -> u64 {
        self.remaining
    }
}

impl<'a> Iterator for TensorInfoIter<'a> {
    type Item = GgufResult<TensorInfo>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(self.parse_one())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse all metadata KV pairs from the reader (mirrors `parser::parse_metadata`).
fn parse_metadata_streaming(
    reader: &mut BinaryReader<'_>,
    header: &GgufHeader,
) -> GgufResult<MetadataStore> {
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

        let value = read_metadata_value(reader, value_type, header.version)?;
        store.insert(key, value);
    }

    Ok(store)
}

/// Read a single metadata value based on its type.
fn read_metadata_value(
    reader: &mut BinaryReader<'_>,
    value_type: GgufValueType,
    version: u32,
) -> GgufResult<crate::metadata::MetadataValue> {
    use crate::metadata::MetadataValue;

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

            let count = if version >= 3 {
                reader.read_u64()? as usize
            } else {
                reader.read_u32()? as usize
            };

            let mut elements = Vec::with_capacity(count.min(1_000_000));
            for _ in 0..count {
                elements.push(read_metadata_value(reader, elem_type, version)?);
            }
            Ok(MetadataValue::Array(elements))
        }
    }
}

/// Skip through all tensor info entries without building structs.
///
/// Advances the reader past each tensor info entry by reading (and discarding)
/// the name, dimensions, type, and offset fields. This is much cheaper than
/// constructing `TensorInfo` structs and allocating name strings.
fn skip_tensor_infos(reader: &mut BinaryReader<'_>, header: &GgufHeader) -> GgufResult<()> {
    for _ in 0..header.tensor_count {
        // Skip name string
        if header.version >= 3 {
            let len = reader.read_u64()? as usize;
            reader.skip(len)?;
        } else {
            let len = reader.read_u32()? as usize;
            reader.skip(len)?;
        }

        // Skip n_dims + dimension values
        let n_dims = reader.read_u32()?;
        let dim_bytes = if header.version >= 3 {
            n_dims as usize * 8 // u64 per dim
        } else {
            n_dims as usize * 4 // u32 per dim
        };
        reader.skip(dim_bytes)?;

        // Skip tensor type (u32)
        reader.skip(4)?;

        // Skip offset (v1: u32, v2+: u64)
        if header.version >= 2 {
            reader.skip(8)?;
        } else {
            reader.skip(4)?;
        }
    }
    Ok(())
}

/// Align a value up to the given alignment boundary.
fn align_up(value: u64, alignment: u64) -> u64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GGUF_MAGIC;

    /// Build a minimal valid GGUF v3 file with 1 tensor and 2 metadata KV pairs.
    fn build_test_gguf_v3() -> Vec<u8> {
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

        // Add some fake tensor data (Q4_0: 32x32 = 1024 elements,
        // block_size=32, 18 bytes/block => 1024/32*18 = 576 bytes)
        data.resize(aligned + 1024, 0xAB);

        data
    }

    /// Build a GGUF v3 file with multiple tensors.
    fn build_multi_tensor_gguf_v3() -> Vec<u8> {
        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&3u64.to_le_bytes()); // 3 tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV pair

        // KV: "general.architecture" = "llama"
        write_string_v3(&mut data, "general.architecture");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v3(&mut data, "llama");

        // Tensor 0: "blk.0.attn_q.weight", 2D [8,8], F32, offset 0
        write_string_v3(&mut data, "blk.0.attn_q.weight");
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&8u64.to_le_bytes());
        data.extend_from_slice(&8u64.to_le_bytes());
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        // Tensor 1: "blk.0.attn_k.weight", 2D [8,8], F32, offset 256
        write_string_v3(&mut data, "blk.0.attn_k.weight");
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&8u64.to_le_bytes());
        data.extend_from_slice(&8u64.to_le_bytes());
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&256u64.to_le_bytes());

        // Tensor 2: "output.weight", 1D [16], F16, offset 512
        write_string_v3(&mut data, "output.weight");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&16u64.to_le_bytes());
        data.extend_from_slice(&(GgufTensorType::F16 as u32).to_le_bytes());
        data.extend_from_slice(&512u64.to_le_bytes());

        // Pad to alignment
        let current = data.len();
        let aligned = align_up(current as u64, 32) as usize;
        data.resize(aligned, 0);

        // Add enough fake tensor data
        data.resize(aligned + 1024, 0xCD);

        data
    }

    /// Build a minimal GGUF v2 file.
    fn build_test_gguf_v2() -> Vec<u8> {
        let mut data = Vec::new();

        // Header (v2: u32 counts)
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // version
        data.extend_from_slice(&1u32.to_le_bytes()); // 1 tensor (u32 in v2)
        data.extend_from_slice(&1u32.to_le_bytes()); // 1 KV pair (u32 in v2)

        // KV: "general.architecture" = "qwen" (v2 string: u32 length)
        write_string_v2(&mut data, "general.architecture");
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string_v2(&mut data, "qwen");

        // Tensor: "embed.weight", 1D [16], F32, offset 0
        // v2: u32 string length, u32 dims, u32 per dim, u64 offset
        write_string_v2(&mut data, "embed.weight");
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&16u32.to_le_bytes()); // dim (u32 in v2)
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset (u64 in v2)

        let current = data.len();
        let aligned = align_up(current as u64, 32) as usize;
        data.resize(aligned, 0);

        // 16 F32 = 64 bytes
        data.resize(aligned + 128, 0xEE);
        data
    }

    /// Build an empty GGUF v3 (0 tensors, 0 metadata).
    fn build_empty_gguf() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 KV pairs
        data
    }

    fn write_string_v3(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_string_v2(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn test_streaming_header_and_metadata_match_full_parse() {
        let data = build_test_gguf_v3();

        let full = GgufFile::parse(&data).expect("full parse");
        let streaming = StreamingGgufParser::new(&data).expect("streaming parse");

        assert_eq!(streaming.header().version, full.header.version);
        assert_eq!(streaming.header().tensor_count, full.header.tensor_count);
        assert_eq!(
            streaming.header().metadata_kv_count,
            full.header.metadata_kv_count
        );
        assert_eq!(streaming.alignment(), full.alignment);
        assert_eq!(
            streaming.architecture().expect("arch"),
            full.architecture().expect("arch")
        );
        assert_eq!(
            streaming
                .metadata()
                .get_u32("llama.block_count")
                .expect("block_count"),
            full.metadata
                .get_u32("llama.block_count")
                .expect("block_count"),
        );
    }

    #[test]
    fn test_tensor_info_iter_count_matches_header() {
        let data = build_test_gguf_v3();
        let parser = StreamingGgufParser::new(&data).expect("parse");

        let count = parser
            .tensor_infos()
            .collect::<Result<Vec<_>, _>>()
            .expect("iter")
            .len();
        assert_eq!(count as u64, parser.header().tensor_count);
    }

    #[test]
    fn test_tensor_info_iter_multi_tensor() {
        let data = build_multi_tensor_gguf_v3();
        let parser = StreamingGgufParser::new(&data).expect("parse");

        let infos: Vec<TensorInfo> = parser
            .tensor_infos()
            .collect::<Result<Vec<_>, _>>()
            .expect("iter");

        assert_eq!(infos.len(), 3);
        assert_eq!(infos[0].name, "blk.0.attn_q.weight");
        assert_eq!(infos[0].dimensions, vec![8, 8]);
        assert_eq!(infos[0].tensor_type, GgufTensorType::F32);

        assert_eq!(infos[1].name, "blk.0.attn_k.weight");
        assert_eq!(infos[1].offset, 256);

        assert_eq!(infos[2].name, "output.weight");
        assert_eq!(infos[2].n_dims, 1);
        assert_eq!(infos[2].dimensions, vec![16]);
        assert_eq!(infos[2].tensor_type, GgufTensorType::F16);
    }

    #[test]
    fn test_find_tensor_existing() {
        let data = build_multi_tensor_gguf_v3();
        let parser = StreamingGgufParser::new(&data).expect("parse");

        let info = parser.find_tensor("blk.0.attn_k.weight").expect("found");
        assert_eq!(info.name, "blk.0.attn_k.weight");
        assert_eq!(info.offset, 256);
        assert_eq!(info.dimensions, vec![8, 8]);
    }

    #[test]
    fn test_find_tensor_missing() {
        let data = build_test_gguf_v3();
        let parser = StreamingGgufParser::new(&data).expect("parse");

        let err = parser.find_tensor("nonexistent");
        assert!(err.is_err());
        let e = err.unwrap_err();
        assert!(
            matches!(e, GgufError::TensorNotFound { ref name } if name == "nonexistent"),
            "expected TensorNotFound, got: {e:?}"
        );
    }

    #[test]
    fn test_tensor_data_returns_correct_slice() {
        let data = build_test_gguf_v3();
        let parser = StreamingGgufParser::new(&data).expect("parse");

        let info = parser.find_tensor("output.weight").expect("found");
        let bytes = parser.tensor_data(&info).expect("data");
        assert!(!bytes.is_empty());
        // All tensor data bytes were set to 0xAB
        assert!(bytes.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn test_load_tensors_subset() {
        let data = build_multi_tensor_gguf_v3();
        let parser = StreamingGgufParser::new(&data).expect("parse");

        let store = parser
            .load_tensors(&["blk.0.attn_q.weight", "output.weight"])
            .expect("load subset");

        assert_eq!(store.len(), 2);
        assert!(store.contains("blk.0.attn_q.weight"));
        assert!(store.contains("output.weight"));
        assert!(!store.contains("blk.0.attn_k.weight"));
    }

    #[test]
    fn test_load_tensors_empty_names() {
        let data = build_test_gguf_v3();
        let parser = StreamingGgufParser::new(&data).expect("parse");

        let store = parser.load_tensors(&[]).expect("load empty");
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_into_full_matches_gguf_file_parse() {
        let data = build_test_gguf_v3();

        let full_direct = GgufFile::parse(&data).expect("direct parse");
        let streaming = StreamingGgufParser::new(&data).expect("streaming");
        let full_from_streaming = streaming.into_full().expect("into_full");

        assert_eq!(
            full_from_streaming.header.version,
            full_direct.header.version
        );
        assert_eq!(
            full_from_streaming.header.tensor_count,
            full_direct.header.tensor_count
        );
        assert_eq!(full_from_streaming.alignment, full_direct.alignment);
        assert_eq!(full_from_streaming.tensors.len(), full_direct.tensors.len());

        // Compare individual tensor info
        let t1 = full_from_streaming
            .tensors
            .get("output.weight")
            .expect("t1");
        let t2 = full_direct.tensors.get("output.weight").expect("t2");
        assert_eq!(t1.name, t2.name);
        assert_eq!(t1.dimensions, t2.dimensions);
        assert_eq!(t1.tensor_type, t2.tensor_type);
        assert_eq!(t1.offset, t2.offset);
    }

    #[test]
    fn test_into_full_multi_tensor() {
        let data = build_multi_tensor_gguf_v3();

        let full_direct = GgufFile::parse(&data).expect("direct parse");
        let streaming = StreamingGgufParser::new(&data).expect("streaming");
        let full_from_streaming = streaming.into_full().expect("into_full");

        assert_eq!(full_from_streaming.tensors.len(), full_direct.tensors.len());

        for name in [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "output.weight",
        ] {
            let t1 = full_from_streaming.tensors.get(name).expect("t1");
            let t2 = full_direct.tensors.get(name).expect("t2");
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.dimensions, t2.dimensions);
            assert_eq!(t1.tensor_type, t2.tensor_type);
            assert_eq!(t1.offset, t2.offset);
        }
    }

    #[test]
    fn test_tensor_info_iter_can_be_created_multiple_times() {
        let data = build_multi_tensor_gguf_v3();
        let parser = StreamingGgufParser::new(&data).expect("parse");

        // First iteration
        let names1: Vec<String> = parser
            .tensor_infos()
            .map(|r| r.map(|i| i.name))
            .collect::<Result<Vec<_>, _>>()
            .expect("iter 1");

        // Second iteration — should produce identical results
        let names2: Vec<String> = parser
            .tensor_infos()
            .map(|r| r.map(|i| i.name))
            .collect::<Result<Vec<_>, _>>()
            .expect("iter 2");

        assert_eq!(names1, names2);
        assert_eq!(names1.len(), 3);
    }

    #[test]
    fn test_empty_gguf_parses_correctly() {
        let data = build_empty_gguf();
        let parser = StreamingGgufParser::new(&data).expect("parse empty");

        assert_eq!(parser.header().tensor_count, 0);
        assert_eq!(parser.header().metadata_kv_count, 0);
        assert_eq!(parser.alignment(), GGUF_DEFAULT_ALIGNMENT);

        let infos: Vec<TensorInfo> = parser
            .tensor_infos()
            .collect::<Result<Vec<_>, _>>()
            .expect("iter");
        assert!(infos.is_empty());
    }

    #[test]
    fn test_v2_streaming_parser() {
        let data = build_test_gguf_v2();
        let parser = StreamingGgufParser::new(&data).expect("parse v2");

        assert_eq!(parser.header().version, 2);
        assert_eq!(parser.header().tensor_count, 1);
        assert_eq!(parser.architecture().expect("arch"), "qwen");

        let info = parser.find_tensor("embed.weight").expect("found");
        assert_eq!(info.n_dims, 1);
        assert_eq!(info.dimensions, vec![16]);
        assert_eq!(info.tensor_type, GgufTensorType::F32);
        assert_eq!(info.offset, 0);

        // Verify tensor data access
        let bytes = parser.tensor_data(&info).expect("data");
        // 16 F32 = 64 bytes
        assert_eq!(bytes.len(), 64);
        assert!(bytes.iter().all(|&b| b == 0xEE));
    }

    #[test]
    fn test_v2_into_full_matches_full_parse() {
        let data = build_test_gguf_v2();

        let full_direct = GgufFile::parse(&data).expect("full parse");
        let streaming = StreamingGgufParser::new(&data).expect("streaming");
        let full_from_streaming = streaming.into_full().expect("into_full");

        assert_eq!(
            full_from_streaming.header.version,
            full_direct.header.version
        );
        assert_eq!(full_from_streaming.tensors.len(), full_direct.tensors.len());

        let t1 = full_from_streaming.tensors.get("embed.weight").expect("t1");
        let t2 = full_direct.tensors.get("embed.weight").expect("t2");
        assert_eq!(t1.name, t2.name);
        assert_eq!(t1.dimensions, t2.dimensions);
        assert_eq!(t1.tensor_type, t2.tensor_type);
        assert_eq!(t1.offset, t2.offset);
    }

    #[test]
    fn test_size_hint() {
        let data = build_multi_tensor_gguf_v3();
        let parser = StreamingGgufParser::new(&data).expect("parse");

        let mut iter = parser.tensor_infos();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        assert_eq!(iter.remaining(), 3);

        let _ = iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.remaining(), 2);
    }

    #[test]
    fn test_data_section_offset_matches_full_parse() {
        let data = build_test_gguf_v3();

        let full = GgufFile::parse(&data).expect("full");
        let streaming = StreamingGgufParser::new(&data).expect("streaming");

        assert_eq!(
            streaming.tensor_infos().data_offset(),
            full.tensors.data_offset()
        );
    }

    #[test]
    fn test_data_section_offset_multi_tensor() {
        let data = build_multi_tensor_gguf_v3();

        let full = GgufFile::parse(&data).expect("full");
        let streaming = StreamingGgufParser::new(&data).expect("streaming");

        assert_eq!(
            streaming.tensor_infos().data_offset(),
            full.tensors.data_offset()
        );
    }
}

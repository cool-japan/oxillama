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
        let info = self.tensors.get(name)?;
        let abs_offset = self.tensors.data_offset() + info.offset;
        let size = info.data_size();

        let start = abs_offset as usize;
        let end = start + size as usize;

        if end > data.len() {
            return Err(GgufError::UnexpectedEof { offset: abs_offset });
        }

        Ok(&data[start..end])
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
) -> GgufResult<MetadataValue> {
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

            let mut elements = Vec::with_capacity(count.min(1_000_000)); // cap prealloc
            for _ in 0..count {
                elements.push(read_metadata_value(reader, elem_type, version)?);
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
}

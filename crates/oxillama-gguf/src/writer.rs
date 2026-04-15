//! GGUF v3 binary writer — serialize models to the GGUF format.
//!
//! Provides a builder-pattern API for constructing GGUF files from scratch,
//! writing metadata key-value pairs and tensor data with proper alignment.
//!
//! # Example
//!
//! ```no_run
//! use oxillama_gguf::{GgufWriter, MetadataValue, GgufTensorType};
//!
//! let mut writer = GgufWriter::new();
//! writer.add_metadata("general.architecture", MetadataValue::String("llama".into()));
//! writer.add_tensor("output.weight", &[32, 32], GgufTensorType::F32, &vec![0u8; 4096]);
//! writer.write_to_file(std::path::Path::new("model.gguf")).unwrap();
//! ```

use std::io::Write;

use crate::error::{GgufError, GgufResult};
use crate::metadata::MetadataValue;
use crate::types::{GgufTensorType, GgufValueType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};

/// A pending tensor entry holding metadata and raw data.
struct PendingTensor {
    name: String,
    dimensions: Vec<u64>,
    tensor_type: GgufTensorType,
    data: Vec<u8>,
}

/// GGUF v3 file writer with builder-pattern API.
///
/// Accumulates metadata and tensor data, then serializes everything
/// to a conformant GGUF v3 binary stream.
pub struct GgufWriter {
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<PendingTensor>,
}

impl GgufWriter {
    /// Create a new empty GGUF writer.
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Add a metadata key-value pair.
    pub fn add_metadata(&mut self, key: &str, value: MetadataValue) {
        self.metadata.push((key.to_string(), value));
    }

    /// Add a tensor with its dimensions, type, and raw data bytes.
    pub fn add_tensor(
        &mut self,
        name: &str,
        dims: &[u64],
        tensor_type: GgufTensorType,
        data: &[u8],
    ) {
        self.tensors.push(PendingTensor {
            name: name.to_string(),
            dimensions: dims.to_vec(),
            tensor_type,
            data: data.to_vec(),
        });
    }

    /// Serialize the complete GGUF v3 file to a writer.
    pub fn write_to<W: Write>(self, writer: &mut W) -> GgufResult<()> {
        let mut offset: u64 = 0;

        // 1. Header
        offset += write_header(
            writer,
            self.tensors.len() as u64,
            self.metadata.len() as u64,
        )?;

        // 2. Metadata KV pairs
        for (key, value) in &self.metadata {
            offset += write_string(writer, key)?;
            offset += write_metadata_value(writer, value)?;
        }

        // 3. Tensor info entries — compute offsets within the data section
        let mut tensor_data_offset: u64 = 0;
        let mut tensor_offsets = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            tensor_offsets.push(tensor_data_offset);
            let data_len = tensor.data.len() as u64;
            tensor_data_offset += data_len;
            // Align to next tensor (except possibly last)
            let padding = alignment_padding(tensor_data_offset, GGUF_DEFAULT_ALIGNMENT);
            tensor_data_offset += padding;
        }

        for (i, tensor) in self.tensors.iter().enumerate() {
            offset += write_string(writer, &tensor.name)?;
            let n_dims = tensor.dimensions.len() as u32;
            writer
                .write_all(&n_dims.to_le_bytes())
                .map_err(io_write_err)?;
            offset += 4;
            for &dim in &tensor.dimensions {
                writer.write_all(&dim.to_le_bytes()).map_err(io_write_err)?;
                offset += 8;
            }
            writer
                .write_all(&(tensor.tensor_type as u32).to_le_bytes())
                .map_err(io_write_err)?;
            offset += 4;
            writer
                .write_all(&tensor_offsets[i].to_le_bytes())
                .map_err(io_write_err)?;
            offset += 8;
        }

        // 4. Alignment padding before data section
        let header_pad = alignment_padding(offset, GGUF_DEFAULT_ALIGNMENT);
        if header_pad > 0 {
            let zeros = vec![0u8; header_pad as usize];
            writer.write_all(&zeros).map_err(io_write_err)?;
        }

        // 5. Tensor data — each tensor aligned to 32 bytes
        for (i, tensor) in self.tensors.iter().enumerate() {
            writer.write_all(&tensor.data).map_err(io_write_err)?;
            // Pad between tensors (not after the last one)
            if i + 1 < self.tensors.len() {
                let data_len = tensor.data.len() as u64;
                let pad = alignment_padding(tensor_offsets[i] + data_len, GGUF_DEFAULT_ALIGNMENT);
                if pad > 0 {
                    let zeros = vec![0u8; pad as usize];
                    writer.write_all(&zeros).map_err(io_write_err)?;
                }
            }
        }

        writer.flush().map_err(io_write_err)?;
        Ok(())
    }

    /// Convenience method to write a GGUF file to disk.
    pub fn write_to_file(self, path: &std::path::Path) -> GgufResult<()> {
        let file = std::fs::File::create(path).map_err(io_write_err)?;
        let mut buf_writer = std::io::BufWriter::new(file);
        self.write_to(&mut buf_writer)
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Write the GGUF v3 header (magic + version + tensor_count + kv_count).
/// Returns number of bytes written.
fn write_header<W: Write>(
    writer: &mut W,
    tensor_count: u64,
    metadata_kv_count: u64,
) -> GgufResult<u64> {
    writer
        .write_all(&GGUF_MAGIC.to_le_bytes())
        .map_err(io_write_err)?;
    writer
        .write_all(&3u32.to_le_bytes())
        .map_err(io_write_err)?;
    writer
        .write_all(&tensor_count.to_le_bytes())
        .map_err(io_write_err)?;
    writer
        .write_all(&metadata_kv_count.to_le_bytes())
        .map_err(io_write_err)?;
    // 4 + 4 + 8 + 8 = 24
    Ok(24)
}

/// Write a length-prefixed UTF-8 string (u64 length + bytes).
/// Returns number of bytes written.
fn write_string<W: Write>(writer: &mut W, s: &str) -> GgufResult<u64> {
    let len = s.len() as u64;
    writer.write_all(&len.to_le_bytes()).map_err(io_write_err)?;
    writer.write_all(s.as_bytes()).map_err(io_write_err)?;
    Ok(8 + s.len() as u64)
}

/// Write a typed metadata value (type tag + value data).
/// Returns number of bytes written.
fn write_metadata_value<W: Write>(writer: &mut W, value: &MetadataValue) -> GgufResult<u64> {
    let type_id = metadata_value_type_id(value);
    writer
        .write_all(&type_id.to_le_bytes())
        .map_err(io_write_err)?;
    let mut written: u64 = 4; // type tag

    match value {
        MetadataValue::Uint8(v) => {
            writer.write_all(&[*v]).map_err(io_write_err)?;
            written += 1;
        }
        MetadataValue::Int8(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            written += 1;
        }
        MetadataValue::Uint16(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            written += 2;
        }
        MetadataValue::Int16(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            written += 2;
        }
        MetadataValue::Uint32(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            written += 4;
        }
        MetadataValue::Int32(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            written += 4;
        }
        MetadataValue::Float32(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            written += 4;
        }
        MetadataValue::Bool(v) => {
            writer
                .write_all(&[if *v { 1u8 } else { 0u8 }])
                .map_err(io_write_err)?;
            written += 1;
        }
        MetadataValue::String(s) => {
            written += write_string(writer, s)?;
        }
        MetadataValue::Uint64(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            written += 8;
        }
        MetadataValue::Int64(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            written += 8;
        }
        MetadataValue::Float64(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            written += 8;
        }
        MetadataValue::Array(elements) => {
            written += write_array(writer, elements)?;
        }
    }

    Ok(written)
}

/// Write array header (element type + count) and each element value (no type tag per element).
/// Returns number of bytes written (excluding the outer type tag already written).
fn write_array<W: Write>(writer: &mut W, elements: &[MetadataValue]) -> GgufResult<u64> {
    let elem_type = if elements.is_empty() {
        GgufValueType::Uint8 as u32
    } else {
        metadata_value_type_id(&elements[0])
    };
    writer
        .write_all(&elem_type.to_le_bytes())
        .map_err(io_write_err)?;
    let count = elements.len() as u64;
    writer
        .write_all(&count.to_le_bytes())
        .map_err(io_write_err)?;
    let mut written: u64 = 4 + 8; // element type + count

    for elem in elements {
        written += write_array_element(writer, elem)?;
    }

    Ok(written)
}

/// Write a single array element value (no type tag — the type is defined by the array header).
fn write_array_element<W: Write>(writer: &mut W, value: &MetadataValue) -> GgufResult<u64> {
    match value {
        MetadataValue::Uint8(v) => {
            writer.write_all(&[*v]).map_err(io_write_err)?;
            Ok(1)
        }
        MetadataValue::Int8(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            Ok(1)
        }
        MetadataValue::Uint16(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            Ok(2)
        }
        MetadataValue::Int16(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            Ok(2)
        }
        MetadataValue::Uint32(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            Ok(4)
        }
        MetadataValue::Int32(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            Ok(4)
        }
        MetadataValue::Float32(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            Ok(4)
        }
        MetadataValue::Bool(v) => {
            writer
                .write_all(&[if *v { 1u8 } else { 0u8 }])
                .map_err(io_write_err)?;
            Ok(1)
        }
        MetadataValue::String(s) => write_string(writer, s),
        MetadataValue::Uint64(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            Ok(8)
        }
        MetadataValue::Int64(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            Ok(8)
        }
        MetadataValue::Float64(v) => {
            writer.write_all(&v.to_le_bytes()).map_err(io_write_err)?;
            Ok(8)
        }
        MetadataValue::Array(elements) => {
            // Nested arrays: write element type + count + values
            write_array(writer, elements)
        }
    }
}

/// Map a `MetadataValue` variant to its GGUF value type ID.
fn metadata_value_type_id(value: &MetadataValue) -> u32 {
    match value {
        MetadataValue::Uint8(_) => GgufValueType::Uint8 as u32,
        MetadataValue::Int8(_) => GgufValueType::Int8 as u32,
        MetadataValue::Uint16(_) => GgufValueType::Uint16 as u32,
        MetadataValue::Int16(_) => GgufValueType::Int16 as u32,
        MetadataValue::Uint32(_) => GgufValueType::Uint32 as u32,
        MetadataValue::Int32(_) => GgufValueType::Int32 as u32,
        MetadataValue::Float32(_) => GgufValueType::Float32 as u32,
        MetadataValue::Bool(_) => GgufValueType::Bool as u32,
        MetadataValue::String(_) => GgufValueType::String as u32,
        MetadataValue::Array(_) => GgufValueType::Array as u32,
        MetadataValue::Uint64(_) => GgufValueType::Uint64 as u32,
        MetadataValue::Int64(_) => GgufValueType::Int64 as u32,
        MetadataValue::Float64(_) => GgufValueType::Float64 as u32,
    }
}

/// Compute the number of zero-padding bytes needed to reach `alignment`.
fn alignment_padding(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return 0;
    }
    let rem = offset % alignment;
    if rem == 0 {
        0
    } else {
        alignment - rem
    }
}

/// Convert an `io::Error` into a `GgufError::WriteError`.
fn io_write_err(e: std::io::Error) -> GgufError {
    GgufError::WriteError {
        reason: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::GgufFile;
    use crate::types::GgufTensorType;

    #[test]
    fn test_write_empty_file() {
        let writer = GgufWriter::new();
        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write empty file");

        let parsed = GgufFile::parse(&buf).expect("parse empty file");
        assert_eq!(parsed.header.version, 3);
        assert_eq!(parsed.header.tensor_count, 0);
        assert_eq!(parsed.header.metadata_kv_count, 0);
        assert_eq!(parsed.tensors.len(), 0);
    }

    #[test]
    fn test_write_string_metadata() {
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        );

        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write string metadata");

        let parsed = GgufFile::parse(&buf).expect("parse string metadata");
        assert_eq!(parsed.header.metadata_kv_count, 1);
        assert_eq!(
            parsed
                .metadata
                .get_string("general.architecture")
                .expect("get arch"),
            "llama"
        );
    }

    #[test]
    fn test_write_numeric_metadata() {
        let mut writer = GgufWriter::new();
        writer.add_metadata("u8val", MetadataValue::Uint8(42));
        writer.add_metadata("i8val", MetadataValue::Int8(-7));
        writer.add_metadata("u16val", MetadataValue::Uint16(1000));
        writer.add_metadata("i16val", MetadataValue::Int16(-500));
        writer.add_metadata("u32val", MetadataValue::Uint32(100_000));
        writer.add_metadata("i32val", MetadataValue::Int32(-100_000));
        writer.add_metadata("f32val", MetadataValue::Float32(1.234));
        writer.add_metadata("u64val", MetadataValue::Uint64(1_000_000_000_000));
        writer.add_metadata("i64val", MetadataValue::Int64(-1_000_000_000_000));
        writer.add_metadata("f64val", MetadataValue::Float64(9.876543210));

        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write numeric metadata");

        let parsed = GgufFile::parse(&buf).expect("parse numeric metadata");
        assert_eq!(parsed.header.metadata_kv_count, 10);

        let get = |k: &str| parsed.metadata.get(k).expect("missing key").clone();
        assert!(matches!(get("u8val"), MetadataValue::Uint8(42)));
        assert!(matches!(get("i8val"), MetadataValue::Int8(-7)));
        assert!(matches!(get("u16val"), MetadataValue::Uint16(1000)));
        assert!(matches!(get("i16val"), MetadataValue::Int16(-500)));
        assert!(matches!(get("u32val"), MetadataValue::Uint32(100_000)));
        assert!(matches!(get("i32val"), MetadataValue::Int32(-100_000)));
        assert!(matches!(
            get("u64val"),
            MetadataValue::Uint64(1_000_000_000_000)
        ));
        assert!(matches!(
            get("i64val"),
            MetadataValue::Int64(-1_000_000_000_000)
        ));

        if let MetadataValue::Float32(v) = get("f32val") {
            assert!((v - 1.234).abs() < 1e-5);
        } else {
            panic!("expected Float32");
        }

        if let MetadataValue::Float64(v) = get("f64val") {
            assert!((v - 9.876543210).abs() < 1e-9);
        } else {
            panic!("expected Float64");
        }
    }

    #[test]
    fn test_write_bool_metadata() {
        let mut writer = GgufWriter::new();
        writer.add_metadata("flag_true", MetadataValue::Bool(true));
        writer.add_metadata("flag_false", MetadataValue::Bool(false));

        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write bool metadata");

        let parsed = GgufFile::parse(&buf).expect("parse bool metadata");
        assert_eq!(
            parsed.metadata.get("flag_true").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            parsed.metadata.get("flag_false").and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn test_write_array_metadata() {
        let mut writer = GgufWriter::new();
        let arr = MetadataValue::Array(vec![
            MetadataValue::String("hello".into()),
            MetadataValue::String("world".into()),
            MetadataValue::String("test".into()),
        ]);
        writer.add_metadata("tokens", arr);

        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write array metadata");

        let parsed = GgufFile::parse(&buf).expect("parse array metadata");
        let arr = parsed
            .metadata
            .get("tokens")
            .and_then(|v| v.as_array())
            .expect("get array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_str(), Some("hello"));
        assert_eq!(arr[1].as_str(), Some("world"));
        assert_eq!(arr[2].as_str(), Some("test"));
    }

    #[test]
    fn test_write_single_tensor_roundtrip() {
        // Q4_0: block_size=32, block_bytes=18
        // For 64 elements: 2 blocks => 36 bytes
        let tensor_data = vec![0xABu8; 36];

        let mut writer = GgufWriter::new();
        writer.add_metadata("general.architecture", MetadataValue::String("test".into()));
        writer.add_tensor(
            "output.weight",
            &[32, 2],
            GgufTensorType::Q4_0,
            &tensor_data,
        );

        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write single tensor");

        let parsed = GgufFile::parse(&buf).expect("parse single tensor");
        assert_eq!(parsed.header.tensor_count, 1);

        let info = parsed
            .tensors
            .get("output.weight")
            .expect("get tensor info");
        assert_eq!(info.dimensions, vec![32, 2]);
        assert_eq!(info.tensor_type, GgufTensorType::Q4_0);

        let data = parsed
            .tensor_data(&buf, "output.weight")
            .expect("get tensor data");
        assert_eq!(data, &tensor_data[..]);
    }

    #[test]
    fn test_write_multiple_tensors() {
        let data0 = vec![0x11u8; 128]; // F32 tensor: 32 floats = 128 bytes
        let data1 = vec![0x22u8; 64]; // F32 tensor: 16 floats = 64 bytes
        let data2 = vec![0x33u8; 256]; // F32 tensor: 64 floats = 256 bytes

        let mut writer = GgufWriter::new();
        writer.add_tensor("t0", &[32], GgufTensorType::F32, &data0);
        writer.add_tensor("t1", &[16], GgufTensorType::F32, &data1);
        writer.add_tensor("t2", &[64], GgufTensorType::F32, &data2);

        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write multiple tensors");

        let parsed = GgufFile::parse(&buf).expect("parse multiple tensors");
        assert_eq!(parsed.header.tensor_count, 3);

        // Verify each tensor's offset is aligned to 32 bytes
        for (_, info) in parsed.tensors.iter() {
            assert_eq!(
                info.offset % GGUF_DEFAULT_ALIGNMENT,
                0,
                "tensor '{}' offset {} is not 32-byte aligned",
                info.name,
                info.offset
            );
        }

        // Verify roundtrip data
        assert_eq!(parsed.tensor_data(&buf, "t0").expect("get t0"), &data0[..]);
        assert_eq!(parsed.tensor_data(&buf, "t1").expect("get t1"), &data1[..]);
        assert_eq!(parsed.tensor_data(&buf, "t2").expect("get t2"), &data2[..]);
    }

    #[test]
    fn test_write_to_file() {
        let mut writer = GgufWriter::new();
        writer.add_metadata("general.architecture", MetadataValue::String("phi".into()));
        writer.add_tensor(
            "embed.weight",
            &[16, 8],
            GgufTensorType::F32,
            &vec![0u8; 512],
        );

        let dir = std::env::temp_dir().join("oxillama_gguf_writer_test");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("test_write.gguf");

        writer.write_to_file(&path).expect("write to file");

        let data = std::fs::read(&path).expect("read file back");
        let parsed = GgufFile::parse(&data).expect("parse file");
        assert_eq!(parsed.header.version, 3);
        assert_eq!(parsed.header.tensor_count, 1);
        assert_eq!(
            parsed
                .metadata
                .get_string("general.architecture")
                .expect("get arch"),
            "phi"
        );

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_alignment_padding() {
        assert_eq!(alignment_padding(0, 32), 0);
        assert_eq!(alignment_padding(1, 32), 31);
        assert_eq!(alignment_padding(31, 32), 1);
        assert_eq!(alignment_padding(32, 32), 0);
        assert_eq!(alignment_padding(33, 32), 31);
        assert_eq!(alignment_padding(64, 32), 0);
        assert_eq!(alignment_padding(100, 32), 28);
        // Edge: alignment = 0
        assert_eq!(alignment_padding(42, 0), 0);
        // Edge: alignment = 1
        assert_eq!(alignment_padding(42, 1), 0);
    }

    #[test]
    fn test_write_large_metadata() {
        let mut writer = GgufWriter::new();
        for i in 0..50 {
            let key = format!("meta.key_{i:03}");
            let value = MetadataValue::Uint32(i as u32);
            writer.add_metadata(&key, value);
        }

        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write large metadata");

        let parsed = GgufFile::parse(&buf).expect("parse large metadata");
        assert_eq!(parsed.header.metadata_kv_count, 50);

        for i in 0..50 {
            let key = format!("meta.key_{i:03}");
            let val = parsed.metadata.get_u32(&key).expect("get u32");
            assert_eq!(val, i as u32);
        }
    }

    #[test]
    fn test_write_array_of_ints() {
        let mut writer = GgufWriter::new();
        let arr = MetadataValue::Array(vec![
            MetadataValue::Int32(10),
            MetadataValue::Int32(20),
            MetadataValue::Int32(30),
        ]);
        writer.add_metadata("dims", arr);

        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write int array");

        let parsed = GgufFile::parse(&buf).expect("parse int array");
        let arr = parsed
            .metadata
            .get("dims")
            .and_then(|v| v.as_array())
            .expect("get array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_i32(), Some(10));
        assert_eq!(arr[1].as_i32(), Some(20));
        assert_eq!(arr[2].as_i32(), Some(30));
    }

    #[test]
    fn test_write_mixed_metadata_and_tensors() {
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        );
        writer.add_metadata("general.name", MetadataValue::String("TestModel".into()));
        writer.add_metadata("llama.block_count", MetadataValue::Uint32(32));
        writer.add_metadata("training.learning_rate", MetadataValue::Float32(0.001));

        // Two tensors with different types
        let f32_data = vec![0u8; 128]; // 32 F32 elements
        let q8_data = vec![0u8; 34]; // Q8_0: 1 block of 32 weights = 34 bytes
        writer.add_tensor("embed.weight", &[32], GgufTensorType::F32, &f32_data);
        writer.add_tensor("attn.weight", &[32], GgufTensorType::Q8_0, &q8_data);

        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write mixed");

        let parsed = GgufFile::parse(&buf).expect("parse mixed");
        assert_eq!(parsed.header.metadata_kv_count, 4);
        assert_eq!(parsed.header.tensor_count, 2);
        assert_eq!(
            parsed
                .metadata
                .get_string("general.architecture")
                .expect("arch"),
            "llama"
        );
        assert_eq!(
            parsed.metadata.get_string("general.name").expect("name"),
            "TestModel"
        );

        let embed = parsed
            .tensor_data(&buf, "embed.weight")
            .expect("embed data");
        assert_eq!(embed, &f32_data[..]);

        let attn = parsed.tensor_data(&buf, "attn.weight").expect("attn data");
        assert_eq!(attn, &q8_data[..]);
    }
}

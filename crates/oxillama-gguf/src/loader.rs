//! GGUF file loader with memory mapping support.
//!
//! Provides high-level loading of GGUF files via either memory mapping
//! (zero-copy, recommended for large models) or full file reads.

use std::path::Path;

use crate::error::{GgufError, GgufResult};
use crate::parser::GgufFile;

/// A loaded GGUF model file with its backing data.
///
/// The data is either memory-mapped (zero-copy) or fully loaded into memory.
/// The `GgufFile` metadata is parsed eagerly; tensor data is accessed lazily
/// via slicing into the backing data.
pub struct GgufModel {
    /// Parsed GGUF metadata and tensor registry.
    pub file: GgufFile,
    /// Backing data (either mmap or `Vec<u8>`).
    data: GgufData,
}

enum GgufData {
    /// Memory-mapped file data.
    #[cfg(feature = "mmap")]
    Mmap(memmap2::Mmap),
    /// Fully loaded file data.
    Owned(Vec<u8>),
}

impl GgufData {
    fn as_bytes(&self) -> &[u8] {
        match self {
            #[cfg(feature = "mmap")]
            GgufData::Mmap(m) => m,
            GgufData::Owned(v) => v,
        }
    }
}

impl GgufModel {
    /// Load a GGUF file using memory mapping (zero-copy).
    ///
    /// This is the recommended method for large model files.
    /// The file stays mapped for the lifetime of this struct.
    #[cfg(feature = "mmap")]
    pub fn load_mmap(path: impl AsRef<Path>) -> GgufResult<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(GgufError::MmapError)?;

        // SAFETY: We treat the file as read-only and don't share the mmap.
        // The file must not be modified while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(GgufError::MmapError)? };

        let parsed = GgufFile::parse(&mmap)?;

        Ok(Self {
            file: parsed,
            data: GgufData::Mmap(mmap),
        })
    }

    /// Load a GGUF file by reading the entire file into memory.
    ///
    /// Suitable for smaller models or when mmap is not available.
    pub fn load_read(path: impl AsRef<Path>) -> GgufResult<Self> {
        let path = path.as_ref();
        let data = std::fs::read(path).map_err(GgufError::MmapError)?;
        let parsed = GgufFile::parse(&data)?;

        Ok(Self {
            file: parsed,
            data: GgufData::Owned(data),
        })
    }

    /// Load a GGUF file using the best available method.
    ///
    /// Uses mmap when available, falls back to full read.
    pub fn load(path: impl AsRef<Path>) -> GgufResult<Self> {
        #[cfg(feature = "mmap")]
        {
            Self::load_mmap(path)
        }
        #[cfg(not(feature = "mmap"))]
        {
            Self::load_read(path)
        }
    }

    /// Parse a GGUF model from an in-memory byte slice.
    pub fn from_bytes(data: Vec<u8>) -> GgufResult<Self> {
        let parsed = GgufFile::parse(&data)?;
        Ok(Self {
            file: parsed,
            data: GgufData::Owned(data),
        })
    }

    /// Get raw tensor data bytes for a named tensor.
    pub fn tensor_data(&self, name: &str) -> GgufResult<&[u8]> {
        self.file.tensor_data(self.data.as_bytes(), name)
    }

    /// Get the full backing data buffer.
    pub fn raw_data(&self) -> &[u8] {
        self.data.as_bytes()
    }

    /// Get the architecture string.
    pub fn architecture(&self) -> GgufResult<&str> {
        self.file.architecture()
    }

    /// Get the model name.
    pub fn model_name(&self) -> Option<&str> {
        self.file.model_name()
    }

    /// Get the file size in bytes.
    pub fn file_size(&self) -> usize {
        self.data.as_bytes().len()
    }

    /// Print a summary of the model to the given writer.
    pub fn print_summary(&self, w: &mut impl std::fmt::Write) -> std::fmt::Result {
        writeln!(w, "GGUF Model Summary")?;
        writeln!(w, "===================")?;
        writeln!(w, "Version:       {}", self.file.header.version)?;
        writeln!(
            w,
            "Architecture:  {}",
            self.architecture().unwrap_or("unknown")
        )?;
        writeln!(
            w,
            "Name:          {}",
            self.model_name().unwrap_or("unknown")
        )?;
        writeln!(w, "Tensors:       {}", self.file.header.tensor_count)?;
        writeln!(w, "KV Pairs:      {}", self.file.header.metadata_kv_count)?;
        writeln!(w, "Alignment:     {} bytes", self.file.alignment)?;
        writeln!(
            w,
            "File Size:     {:.2} MB",
            self.file_size() as f64 / 1_048_576.0
        )?;
        writeln!(w, "Data Offset:   0x{:X}", self.file.tensors.data_offset())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::GgufFile;
    use crate::types::{GgufTensorType, GgufValueType, GGUF_MAGIC};

    fn build_minimal_gguf() -> Vec<u8> {
        // Build a minimal v3 GGUF with 0 tensors and 1 KV pair
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV pair

        // KV: "general.architecture" = "llama"
        let key = b"general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        let val = b"llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val);

        // Pad to alignment boundary (32 bytes)
        let align = 32usize;
        let rem = data.len() % align;
        if rem != 0 {
            data.resize(data.len() + align - rem, 0u8);
        }
        data
    }

    fn build_gguf_with_tensor() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV pair

        // KV: "general.architecture" = "llama"
        let key = b"general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        let val = b"llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val);

        // Tensor info: "output.weight", 1D [32], Q8_0, offset 0
        let tname = b"output.weight";
        data.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        data.extend_from_slice(tname);
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        data.extend_from_slice(&32u64.to_le_bytes()); // dim[0] = 32
        data.extend_from_slice(&(GgufTensorType::Q8_0 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        // Pad to alignment (32 bytes)
        let align = 32usize;
        let rem = data.len() % align;
        if rem != 0 {
            data.resize(data.len() + align - rem, 0u8);
        }
        // Fake tensor data
        data.resize(data.len() + 512, 0u8);
        data
    }

    #[test]
    fn test_load_read_nonexistent_returns_error() {
        let result = GgufModel::load_read("/nonexistent/path/does_not_exist.gguf");
        assert!(result.is_err(), "load_read of missing file should error");
    }

    #[test]
    fn test_from_bytes_truncated_magic_errors() {
        // Only 4 magic bytes — should fail parsing
        let data = b"GGUF".to_vec();
        let result = GgufModel::from_bytes(data);
        assert!(result.is_err(), "truncated data should fail");
    }

    #[test]
    fn test_from_bytes_valid_minimal() {
        let data = build_minimal_gguf();
        let model = GgufModel::from_bytes(data).expect("test: from_bytes should succeed");
        assert_eq!(model.file.header.version, 3);
        assert_eq!(model.file.header.tensor_count, 0);
        assert_eq!(model.architecture().expect("test: arch"), "llama");
    }

    #[test]
    fn test_from_bytes_file_size() {
        let data = build_minimal_gguf();
        let expected_len = data.len();
        let model = GgufModel::from_bytes(data).expect("test: from_bytes");
        assert_eq!(model.file_size(), expected_len);
    }

    #[test]
    fn test_raw_data_matches_input() {
        let data = build_minimal_gguf();
        let original = data.clone();
        let model = GgufModel::from_bytes(data).expect("test: from_bytes");
        assert_eq!(model.raw_data(), original.as_slice());
    }

    #[test]
    fn test_model_name_none_when_missing() {
        let data = build_minimal_gguf();
        let model = GgufModel::from_bytes(data).expect("test: from_bytes");
        // No "general.name" KV in minimal GGUF
        assert!(model.model_name().is_none());
    }

    #[test]
    fn test_tensor_data_missing_tensor_errors() {
        let data = build_gguf_with_tensor();
        let model = GgufModel::from_bytes(data).expect("test: from_bytes");
        let result = model.tensor_data("nonexistent.tensor");
        assert!(result.is_err(), "missing tensor should return error");
    }

    #[test]
    fn test_tensor_data_present_ok() {
        let data = build_gguf_with_tensor();
        let model = GgufModel::from_bytes(data).expect("test: from_bytes");
        let result = model.tensor_data("output.weight");
        assert!(result.is_ok(), "present tensor should return Ok");
    }

    #[test]
    fn test_print_summary_does_not_panic() {
        let data = build_minimal_gguf();
        let model = GgufModel::from_bytes(data).expect("test: from_bytes");
        let mut out = String::new();
        model
            .print_summary(&mut out)
            .expect("test: print_summary should succeed");
        assert!(out.contains("GGUF Model Summary"), "summary missing header");
        assert!(out.contains("llama"), "summary missing arch");
    }

    #[test]
    fn test_load_uses_fallback_read() {
        // Write a minimal GGUF to a temp file and load it
        let dir = std::env::temp_dir();
        let path = dir.join("oxillama_test_loader_load.gguf");
        let data = build_minimal_gguf();
        std::fs::write(&path, &data).expect("test: write temp file");
        let result = GgufModel::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_ok(), "load from temp file should succeed");
    }

    #[test]
    fn test_load_read_from_temp_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxillama_test_loader_read.gguf");
        let data = build_minimal_gguf();
        std::fs::write(&path, &data).expect("test: write temp file");
        let result = GgufModel::load_read(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_ok(), "load_read from temp file should succeed");
        let model = result.expect("test: load_read");
        assert_eq!(model.architecture().expect("test: arch"), "llama");
    }

    #[test]
    fn test_from_bytes_invalid_magic_errors() {
        let mut data = build_minimal_gguf();
        // Corrupt the magic number
        data[0] = 0xFF;
        data[1] = 0xFF;
        data[2] = 0xFF;
        data[3] = 0xFF;
        let result = GgufModel::from_bytes(data);
        assert!(result.is_err(), "invalid magic should error");
    }

    #[test]
    fn test_file_size_matches_raw_data_len() {
        let data = build_gguf_with_tensor();
        let len = data.len();
        let model = GgufModel::from_bytes(data).expect("test: from_bytes");
        assert_eq!(model.file_size(), len);
        assert_eq!(model.raw_data().len(), len);
    }

    /// GgufFile::parse is the underlying parser; verify it succeeds independently.
    #[test]
    fn test_gguf_file_parse_returns_correct_header() {
        let data = build_minimal_gguf();
        let file = GgufFile::parse(&data).expect("test: GgufFile::parse");
        assert_eq!(file.header.version, 3);
        assert_eq!(file.header.tensor_count, 0);
        assert_eq!(file.header.metadata_kv_count, 1);
    }
}

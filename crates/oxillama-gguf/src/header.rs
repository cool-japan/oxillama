//! GGUF file header parsing.

use crate::error::{GgufError, GgufResult};
use crate::types::GGUF_MAGIC;

/// The parsed GGUF file header.
///
/// Contains the format version, tensor count, and metadata KV pair count.
/// This is the first structure read from any GGUF file.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// GGUF format version (1, 2, or 3).
    pub version: u32,
    /// Number of tensors stored in the file.
    pub tensor_count: u64,
    /// Number of key-value metadata pairs.
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    /// Parse a GGUF header from a byte slice starting at the given offset.
    ///
    /// Returns the parsed header and the new offset after the header.
    pub fn parse(data: &[u8], offset: u64) -> GgufResult<(Self, u64)> {
        let mut pos = offset as usize;

        // Read and validate magic number (4 bytes, little-endian)
        if data.len() < pos + 4 {
            return Err(GgufError::UnexpectedEof { offset });
        }
        let magic = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic { magic });
        }

        // Read version (4 bytes)
        if data.len() < pos + 4 {
            return Err(GgufError::UnexpectedEof { offset: pos as u64 });
        }
        let version = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        if !(1..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion { version });
        }

        // Read tensor count (8 bytes for v3, 4 bytes for v2)
        let tensor_count = Self::read_count(data, &mut pos, version)?;
        let metadata_kv_count = Self::read_count(data, &mut pos, version)?;

        Ok((
            Self {
                version,
                tensor_count,
                metadata_kv_count,
            },
            pos as u64,
        ))
    }

    /// Read a count field (u64 for v3, u32 for v2).
    fn read_count(data: &[u8], pos: &mut usize, version: u32) -> GgufResult<u64> {
        if version >= 3 {
            if data.len() < *pos + 8 {
                return Err(GgufError::UnexpectedEof {
                    offset: *pos as u64,
                });
            }
            let val = u64::from_le_bytes([
                data[*pos],
                data[*pos + 1],
                data[*pos + 2],
                data[*pos + 3],
                data[*pos + 4],
                data[*pos + 5],
                data[*pos + 6],
                data[*pos + 7],
            ]);
            *pos += 8;
            Ok(val)
        } else {
            if data.len() < *pos + 4 {
                return Err(GgufError::UnexpectedEof {
                    offset: *pos as u64,
                });
            }
            let val =
                u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
            *pos += 4;
            Ok(u64::from(val))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_v3_header() {
        let mut data = Vec::new();
        // Magic: "GGUF" in LE
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version: 3
        data.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count: 10
        data.extend_from_slice(&10u64.to_le_bytes());
        // KV count: 5
        data.extend_from_slice(&5u64.to_le_bytes());

        let (header, offset) = GgufHeader::parse(&data, 0).expect("should parse");
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 10);
        assert_eq!(header.metadata_kv_count, 5);
        assert_eq!(offset, 24); // 4 (magic) + 4 (version) + 8 (tensor_count) + 8 (kv_count)
    }

    #[test]
    fn test_invalid_magic() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00];
        let err = GgufHeader::parse(&data, 0).unwrap_err();
        assert!(matches!(err, GgufError::InvalidMagic { magic: 0 }));
    }

    #[test]
    fn test_valid_v1_header() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // version = 1
        data.extend_from_slice(&5u32.to_le_bytes()); // tensor count (u32 in v1)
        data.extend_from_slice(&2u32.to_le_bytes()); // kv count (u32 in v1)

        let (header, offset) = GgufHeader::parse(&data, 0).expect("should parse v1");
        assert_eq!(header.version, 1);
        assert_eq!(header.tensor_count, 5);
        assert_eq!(header.metadata_kv_count, 2);
        assert_eq!(offset, 16); // 4 (magic) + 4 (version) + 4 (tensor) + 4 (kv)
    }

    #[test]
    fn test_valid_v2_header() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // tensor count (u32 in v2)
        data.extend_from_slice(&1u32.to_le_bytes()); // kv count (u32 in v2)

        let (header, offset) = GgufHeader::parse(&data, 0).expect("should parse v2");
        assert_eq!(header.version, 2);
        assert_eq!(header.tensor_count, 3);
        assert_eq!(header.metadata_kv_count, 1);
        assert_eq!(offset, 16);
    }

    #[test]
    fn test_reject_version_0() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let err = GgufHeader::parse(&data, 0).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion { version: 0 }));
    }

    /// Regression test for issue #1: a header whose magic bytes are the
    /// literal ASCII string `"GGUF"` (i.e. exactly what every real GGUF file
    /// starts with) must be accepted. Previously the constant carried a
    /// transposed-nibble typo and rejected valid files such as
    /// `Qwen3-1.7B-Q8_0.gguf` from lmstudio-community.
    #[test]
    fn test_issue_1_accepts_real_gguf_magic_bytes() {
        let mut data = Vec::new();
        // Magic: the literal ASCII bytes of "GGUF" — same as any real file.
        data.extend_from_slice(b"GGUF");
        // Version: 3
        data.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count: 0
        data.extend_from_slice(&0u64.to_le_bytes());
        // KV count: 0
        data.extend_from_slice(&0u64.to_le_bytes());

        let (header, _) = GgufHeader::parse(&data, 0).expect("real b\"GGUF\" magic must parse");
        assert_eq!(header.version, 3);
    }

    #[test]
    fn test_reject_version_99() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&99u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 16]);

        let err = GgufHeader::parse(&data, 0).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion { version: 99 }));
    }
}

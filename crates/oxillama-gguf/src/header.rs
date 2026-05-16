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

    /// Regression test for issue #1 (file-on-disk variant).
    ///
    /// Writes minimal GGUF files to `std::env::temp_dir()` and validates:
    /// 1. A file starting with correct magic bytes `b"GGUF"` parses successfully.
    /// 2. A file starting with the old wrong constant `0x46475547` (reversed nibbles)
    ///    is rejected with `GgufError::InvalidMagic`.
    /// 3. A file with completely wrong magic is rejected.
    ///
    /// The correct GGUF magic on-disk is the 4 ASCII bytes `G G U F`
    /// (`0x47 0x47 0x55 0x46`).  Interpreted as a little-endian `u32` this is
    /// `0x46554747` — **not** `0x46475547` which is the big-endian reading.
    #[cfg(feature = "std")]
    #[test]
    fn test_issue_1_gguf_magic_validation() {
        use std::io::Write as _;

        // Helper: build a minimal GGUF v3 byte payload with arbitrary magic.
        let make_header = |magic_bytes: &[u8; 4]| -> Vec<u8> {
            let mut v = Vec::new();
            v.extend_from_slice(magic_bytes);
            v.extend_from_slice(&3u32.to_le_bytes()); // version
            v.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            v.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
            v
        };

        let tmp = std::env::temp_dir();

        // ── 1. Correct magic (b"GGUF") must be accepted ──────────────────────
        let correct_path = tmp.join("oxillama_issue1_correct_magic.gguf");
        {
            let mut f = std::fs::File::create(&correct_path)
                .expect("create temp file for correct magic");
            f.write_all(&make_header(b"GGUF"))
                .expect("write correct header");
        }
        let correct_bytes =
            std::fs::read(&correct_path).expect("read correct magic file");
        let _ = std::fs::remove_file(&correct_path);
        let (hdr, _) = GgufHeader::parse(&correct_bytes, 0)
            .expect("correct GGUF magic b\"GGUF\" must parse without error");
        assert_eq!(hdr.version, 3, "version should be 3");

        // ── 2. Wrong magic (old transposed constant 0x46475547) ──────────────
        //      The bytes on disk would be [0x47, 0x46, 0x47, 0x55] — NOT "GGUF".
        let wrong_old_magic: u32 = 0x4647_5547;
        let wrong_old_path = tmp.join("oxillama_issue1_wrong_old_magic.gguf");
        {
            let mut f = std::fs::File::create(&wrong_old_path)
                .expect("create temp file for wrong old magic");
            f.write_all(&wrong_old_magic.to_le_bytes())
                .expect("write wrong old magic");
            f.write_all(&3u32.to_le_bytes())
                .expect("write version");
            f.write_all(&0u64.to_le_bytes())
                .expect("write tensor_count");
            f.write_all(&0u64.to_le_bytes())
                .expect("write kv_count");
        }
        let wrong_old_bytes =
            std::fs::read(&wrong_old_path).expect("read wrong old magic file");
        let _ = std::fs::remove_file(&wrong_old_path);
        let err = GgufHeader::parse(&wrong_old_bytes, 0)
            .expect_err("transposed-nibble magic must be rejected");
        assert!(
            matches!(err, GgufError::InvalidMagic { .. }),
            "expected InvalidMagic, got: {err:?}"
        );

        // ── 3. All-zero magic must also be rejected ───────────────────────────
        let zero_path = tmp.join("oxillama_issue1_zero_magic.gguf");
        {
            let mut f = std::fs::File::create(&zero_path)
                .expect("create temp file for zero magic");
            f.write_all(&make_header(&[0, 0, 0, 0]))
                .expect("write zero header");
        }
        let zero_bytes = std::fs::read(&zero_path).expect("read zero magic file");
        let _ = std::fs::remove_file(&zero_path);
        let err2 = GgufHeader::parse(&zero_bytes, 0)
            .expect_err("zero magic must be rejected");
        assert!(
            matches!(err2, GgufError::InvalidMagic { .. }),
            "expected InvalidMagic for zeros, got: {err2:?}"
        );
    }
}

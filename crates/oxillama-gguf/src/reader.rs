//! Binary reader utility for safe parsing of GGUF byte streams.
//!
//! Provides bounds-checked reading of primitive types from a byte slice,
//! tracking the current read position automatically.

use crate::error::{GgufError, GgufResult};

/// A cursor-based reader over a byte slice with bounds checking.
///
/// All reads advance the internal position and return structured errors
/// on out-of-bounds access.
pub struct BinaryReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinaryReader<'a> {
    /// Create a new reader starting at the given offset.
    pub fn new(data: &'a [u8], offset: usize) -> Self {
        Self { data, pos: offset }
    }

    /// Current byte position in the data.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Total length of the underlying data.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the underlying data is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Remaining bytes from current position.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Check that at least `n` bytes are available.
    fn check(&self, n: usize) -> GgufResult<()> {
        if self.pos + n > self.data.len() {
            return Err(GgufError::UnexpectedEof {
                offset: self.pos as u64,
            });
        }
        Ok(())
    }

    /// Read a single byte.
    pub fn read_u8(&mut self) -> GgufResult<u8> {
        self.check(1)?;
        let val = self.data[self.pos];
        self.pos += 1;
        Ok(val)
    }

    /// Read a little-endian u16.
    pub fn read_u16(&mut self) -> GgufResult<u16> {
        self.check(2)?;
        let val = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(val)
    }

    /// Read a little-endian i16.
    pub fn read_i16(&mut self) -> GgufResult<i16> {
        self.check(2)?;
        let val = i16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(val)
    }

    /// Read a little-endian u32.
    pub fn read_u32(&mut self) -> GgufResult<u32> {
        self.check(4)?;
        let val = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(val)
    }

    /// Read a little-endian i32.
    pub fn read_i32(&mut self) -> GgufResult<i32> {
        self.check(4)?;
        let val = i32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(val)
    }

    /// Read a little-endian u64.
    pub fn read_u64(&mut self) -> GgufResult<u64> {
        self.check(8)?;
        let val = u64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(val)
    }

    /// Read a little-endian i64.
    pub fn read_i64(&mut self) -> GgufResult<i64> {
        self.check(8)?;
        let val = i64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(val)
    }

    /// Read a little-endian f32.
    pub fn read_f32(&mut self) -> GgufResult<f32> {
        self.check(4)?;
        let val = f32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(val)
    }

    /// Read a little-endian f64.
    pub fn read_f64(&mut self) -> GgufResult<f64> {
        self.check(8)?;
        let val = f64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(val)
    }

    /// Read a bool (1 byte, 0 = false, non-zero = true).
    pub fn read_bool(&mut self) -> GgufResult<bool> {
        let val = self.read_u8()?;
        Ok(val != 0)
    }

    /// Read a GGUF string: u64 length prefix followed by UTF-8 bytes (no null terminator).
    pub fn read_string(&mut self) -> GgufResult<String> {
        let len = self.read_u64()? as usize;
        self.check(len)?;
        let bytes = self.data[self.pos..self.pos + len].to_vec();
        self.pos += len;
        String::from_utf8(bytes).map_err(|e| GgufError::InvalidString {
            offset: (self.pos - len) as u64,
            source: e,
        })
    }

    /// Read a GGUF v2 string: u32 length prefix followed by UTF-8 bytes.
    pub fn read_string_v2(&mut self) -> GgufResult<String> {
        let len = self.read_u32()? as usize;
        self.check(len)?;
        let bytes = self.data[self.pos..self.pos + len].to_vec();
        self.pos += len;
        String::from_utf8(bytes).map_err(|e| GgufError::InvalidString {
            offset: (self.pos - len) as u64,
            source: e,
        })
    }

    /// Read `n` bytes and return a slice.
    pub fn read_bytes(&mut self, n: usize) -> GgufResult<&'a [u8]> {
        self.check(n)?;
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Get a reference to the underlying data from current position.
    pub fn remaining_data(&self) -> &'a [u8] {
        if self.pos >= self.data.len() {
            &[]
        } else {
            &self.data[self.pos..]
        }
    }

    /// Align the position to the given boundary.
    pub fn align_to(&mut self, alignment: usize) {
        if alignment > 0 {
            let rem = self.pos % alignment;
            if rem != 0 {
                self.pos += alignment - rem;
            }
        }
    }

    /// Skip `n` bytes.
    pub fn skip(&mut self, n: usize) -> GgufResult<()> {
        self.check(n)?;
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_primitives() {
        let data = [
            0x42, // u8 = 66
            0x34, 0x12, // u16 = 0x1234
            0x78, 0x56, 0x34, 0x12, // u32 = 0x12345678
        ];
        let mut r = BinaryReader::new(&data, 0);
        assert_eq!(r.read_u8().unwrap(), 0x42);
        assert_eq!(r.read_u16().unwrap(), 0x1234);
        assert_eq!(r.read_u32().unwrap(), 0x1234_5678);
        assert_eq!(r.position(), 7);
    }

    #[test]
    fn test_read_string() {
        let mut data = Vec::new();
        data.extend_from_slice(&5u64.to_le_bytes()); // length = 5
        data.extend_from_slice(b"hello");
        let mut r = BinaryReader::new(&data, 0);
        assert_eq!(r.read_string().unwrap(), "hello");
    }

    #[test]
    fn test_alignment() {
        let data = [0u8; 64];
        let mut r = BinaryReader::new(&data, 5);
        r.align_to(32);
        assert_eq!(r.position(), 32);
    }

    #[test]
    fn test_eof_error() {
        let data = [0u8; 2];
        let mut r = BinaryReader::new(&data, 0);
        assert!(r.read_u32().is_err());
    }

    #[test]
    fn test_read_i16() {
        let val: i16 = -1000;
        let bytes = val.to_le_bytes();
        let mut r = BinaryReader::new(&bytes, 0);
        assert_eq!(r.read_i16().expect("test: read_i16"), val);
        assert_eq!(r.position(), 2);
    }

    #[test]
    fn test_read_i32() {
        let val: i32 = -123_456;
        let bytes = val.to_le_bytes();
        let mut r = BinaryReader::new(&bytes, 0);
        assert_eq!(r.read_i32().expect("test: read_i32"), val);
        assert_eq!(r.position(), 4);
    }

    #[test]
    fn test_read_u64() {
        let val: u64 = u64::MAX - 1;
        let bytes = val.to_le_bytes();
        let mut r = BinaryReader::new(&bytes, 0);
        assert_eq!(r.read_u64().expect("test: read_u64"), val);
        assert_eq!(r.position(), 8);
    }

    #[test]
    fn test_read_i64() {
        let val: i64 = -9_000_000_000i64;
        let bytes = val.to_le_bytes();
        let mut r = BinaryReader::new(&bytes, 0);
        assert_eq!(r.read_i64().expect("test: read_i64"), val);
        assert_eq!(r.position(), 8);
    }

    #[test]
    fn test_read_f32() {
        let val = std::f32::consts::PI;
        let bytes = val.to_le_bytes();
        let mut r = BinaryReader::new(&bytes, 0);
        let got = r.read_f32().expect("test: read_f32");
        assert!((got - val).abs() < 1e-7, "f32 mismatch: {got} vs {val}");
    }

    #[test]
    fn test_read_f64() {
        let val = std::f64::consts::E;
        let bytes = val.to_le_bytes();
        let mut r = BinaryReader::new(&bytes, 0);
        let got = r.read_f64().expect("test: read_f64");
        assert!((got - val).abs() < 1e-14, "f64 mismatch: {got} vs {val}");
    }

    #[test]
    fn test_read_bool_true_and_false() {
        let data = [0x01u8, 0x00u8];
        let mut r = BinaryReader::new(&data, 0);
        assert!(r.read_bool().expect("test: read_bool true"));
        assert!(!r.read_bool().expect("test: read_bool false"));
    }

    #[test]
    fn test_read_bool_nonzero_is_true() {
        let data = [0xFFu8];
        let mut r = BinaryReader::new(&data, 0);
        assert!(r.read_bool().expect("test: nonzero bool"));
    }

    #[test]
    fn test_read_bytes_slice() {
        let data = b"hello world";
        let mut r = BinaryReader::new(data, 0);
        let slice = r.read_bytes(5).expect("test: read_bytes");
        assert_eq!(slice, b"hello");
        assert_eq!(r.position(), 5);
    }

    #[test]
    fn test_read_bytes_eof() {
        let data = [0u8; 3];
        let mut r = BinaryReader::new(&data, 0);
        assert!(r.read_bytes(4).is_err(), "reading past end must error");
    }

    #[test]
    fn test_skip_advances_position() {
        let data = [0u8; 16];
        let mut r = BinaryReader::new(&data, 0);
        r.skip(7).expect("test: skip");
        assert_eq!(r.position(), 7);
    }

    #[test]
    fn test_skip_past_end_errors() {
        let data = [0u8; 4];
        let mut r = BinaryReader::new(&data, 0);
        assert!(r.skip(10).is_err(), "skip past end must error");
    }

    #[test]
    fn test_len_and_remaining() {
        let data = [0u8; 10];
        let mut r = BinaryReader::new(&data, 0);
        assert_eq!(r.len(), 10);
        assert_eq!(r.remaining(), 10);
        r.skip(3).expect("test: skip");
        assert_eq!(r.remaining(), 7);
    }

    #[test]
    fn test_is_empty_empty_slice() {
        let data: &[u8] = &[];
        let r = BinaryReader::new(data, 0);
        assert!(r.is_empty());
    }

    #[test]
    fn test_is_empty_nonempty_slice() {
        let data = [0u8; 1];
        let r = BinaryReader::new(&data, 0);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_remaining_data_at_offset() {
        let data = b"abcde";
        let mut r = BinaryReader::new(data, 0);
        r.skip(2).expect("test: skip");
        assert_eq!(r.remaining_data(), b"cde");
    }

    #[test]
    fn test_remaining_data_at_end() {
        let data = b"ab";
        let mut r = BinaryReader::new(data, 0);
        r.skip(2).expect("test: skip to end");
        assert_eq!(r.remaining_data(), b"");
    }

    #[test]
    fn test_align_to_already_aligned() {
        let data = [0u8; 32];
        let mut r = BinaryReader::new(&data, 8); // already aligned to 8
        r.align_to(8);
        assert_eq!(
            r.position(),
            8,
            "already-aligned position should not change"
        );
    }

    #[test]
    fn test_align_to_zero_alignment_is_noop() {
        let data = [0u8; 32];
        let mut r = BinaryReader::new(&data, 5);
        r.align_to(0);
        assert_eq!(r.position(), 5, "align_to(0) must be a no-op");
    }

    #[test]
    fn test_read_string_v2() {
        let mut data = Vec::new();
        data.extend_from_slice(&5u32.to_le_bytes()); // length = 5
        data.extend_from_slice(b"world");
        let mut r = BinaryReader::new(&data, 0);
        assert_eq!(r.read_string_v2().expect("test: read_string_v2"), "world");
    }

    #[test]
    fn test_read_string_invalid_utf8_errors() {
        let mut data = Vec::new();
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(&[0xFF, 0xFE]); // invalid UTF-8
        let mut r = BinaryReader::new(&data, 0);
        assert!(
            r.read_string().is_err(),
            "invalid UTF-8 should produce an error"
        );
    }

    #[test]
    fn test_new_with_nonzero_offset() {
        let data = [0u8, 0u8, 0x01u8, 0x00u8]; // u16 = 1 at offset 2
        let mut r = BinaryReader::new(&data, 2);
        assert_eq!(r.position(), 2);
        assert_eq!(r.read_u16().expect("test: read_u16 at offset"), 1u16);
    }

    #[test]
    fn test_u64_eof_error() {
        let data = [0u8; 4]; // only 4 bytes, need 8
        let mut r = BinaryReader::new(&data, 0);
        assert!(r.read_u64().is_err(), "u64 read on 4 bytes should error");
    }

    #[test]
    fn test_i64_eof_error() {
        let data = [0u8; 4];
        let mut r = BinaryReader::new(&data, 0);
        assert!(r.read_i64().is_err(), "i64 read on 4 bytes should error");
    }

    #[test]
    fn test_f64_eof_error() {
        let data = [0u8; 4];
        let mut r = BinaryReader::new(&data, 0);
        assert!(r.read_f64().is_err(), "f64 read on 4 bytes should error");
    }
}

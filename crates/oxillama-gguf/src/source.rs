//! Abstract byte-stream source for no_std-compatible GGUF parsing.
//!
//! The [`Source`] trait abstracts over any byte source that supports
//! `read_exact` and `seek`, enabling the core parser to work in both
//! `std` and `no_std + alloc` environments.
//!
//! Two implementations are provided:
//! - [`SliceSource`]: wraps a `&[u8]` — always available.
//! - [`ReadSource`]: wraps any `std::io::Read + std::io::Seek` — only with `std` feature.
//! - [`FileSource`]: type alias for `ReadSource<std::fs::File>` — only with `std` feature.

use core::fmt;

/// A byte-stream source that the GGUF parser can read from.
///
/// Implementors must support exact-length reads and absolute-position seeks.
pub trait Source {
    /// The error type returned by this source.
    type Error: fmt::Debug + fmt::Display;

    /// Read exactly `buf.len()` bytes into `buf`, advancing the position.
    ///
    /// Returns an error if fewer bytes are available.
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), Self::Error>;

    /// Seek to the given absolute byte position, returning the new position.
    ///
    /// The new position must equal `pos` on success.
    fn seek(&mut self, pos: u64) -> Result<u64, Self::Error>;

    /// Return the current byte position in the stream.
    fn position(&self) -> u64;
}

/// A [`Source`] backed by a byte slice.
///
/// Provides zero-copy parsing from an in-memory buffer. Always available
/// (no `std` feature required).
pub struct SliceSource<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SliceSource<'a> {
    /// Create a new `SliceSource` starting at position 0.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

/// Errors that can occur when reading from a [`SliceSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceError {
    /// The read would go past the end of the slice.
    UnexpectedEof,
    /// The seek position exceeds the slice length.
    InvalidSeek,
}

impl fmt::Display for SliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => f.write_str("unexpected end of slice"),
            Self::InvalidSeek => f.write_str("seek position exceeds slice length"),
        }
    }
}

impl Source for SliceSource<'_> {
    type Error = SliceError;

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), SliceError> {
        let end = self
            .pos
            .checked_add(buf.len())
            .ok_or(SliceError::UnexpectedEof)?;
        if end > self.data.len() {
            return Err(SliceError::UnexpectedEof);
        }
        buf.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        Ok(())
    }

    fn seek(&mut self, pos: u64) -> Result<u64, SliceError> {
        let pos_usize = pos as usize;
        if pos_usize > self.data.len() {
            return Err(SliceError::InvalidSeek);
        }
        self.pos = pos_usize;
        Ok(pos)
    }

    fn position(&self) -> u64 {
        self.pos as u64
    }
}

/// A [`Source`] backed by any `std::io::Read + std::io::Seek` implementor.
#[cfg(feature = "std")]
pub struct ReadSource<R: std::io::Read + std::io::Seek> {
    inner: R,
    pos: u64,
}

#[cfg(feature = "std")]
impl<R: std::io::Read + std::io::Seek> ReadSource<R> {
    /// Create a new `ReadSource` wrapping `inner`.
    ///
    /// The initial position is assumed to be 0. If `inner` has been seeked
    /// already, call [`seek`][Source::seek] to synchronise the position.
    pub fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }
}

#[cfg(feature = "std")]
impl<R: std::io::Read + std::io::Seek> Source for ReadSource<R> {
    type Error = std::io::Error;

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), std::io::Error> {
        std::io::Read::read_exact(&mut self.inner, buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn seek(&mut self, pos: u64) -> Result<u64, std::io::Error> {
        let new_pos =
            std::io::Seek::seek(&mut self.inner, std::io::SeekFrom::Start(pos))?;
        self.pos = new_pos;
        Ok(new_pos)
    }

    fn position(&self) -> u64 {
        self.pos
    }
}

/// A [`Source`] backed by a `std::fs::File`.
///
/// Convenience alias for the most common file-backed source.
#[cfg(feature = "std")]
pub type FileSource = ReadSource<std::fs::File>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_source_read_exact_basic() {
        let data = b"hello world";
        let mut src = SliceSource::new(data);
        let mut buf = [0u8; 5];
        src.read_exact(&mut buf).expect("test: read_exact");
        assert_eq!(&buf, b"hello");
        assert_eq!(src.position(), 5);
    }

    #[test]
    fn slice_source_seek_and_read() {
        let data = b"abcdefghij";
        let mut src = SliceSource::new(data);
        src.seek(3).expect("test: seek");
        assert_eq!(src.position(), 3);
        let mut buf = [0u8; 3];
        src.read_exact(&mut buf).expect("test: read_exact after seek");
        assert_eq!(&buf, b"def");
        assert_eq!(src.position(), 6);
    }

    #[test]
    fn slice_source_read_eof_error() {
        let data = [0u8; 2];
        let mut src = SliceSource::new(&data);
        let mut buf = [0u8; 4];
        assert!(
            src.read_exact(&mut buf).is_err(),
            "reading past end must error"
        );
    }

    #[test]
    fn slice_source_invalid_seek_error() {
        let data = [0u8; 4];
        let mut src = SliceSource::new(&data);
        assert!(
            src.seek(100).is_err(),
            "seeking past end must error"
        );
    }

    #[test]
    fn slice_source_seek_to_exact_end_ok() {
        let data = [0u8; 4];
        let mut src = SliceSource::new(&data);
        // Seeking to data.len() is allowed (position at end, no data left)
        let pos = src.seek(4).expect("test: seek to end");
        assert_eq!(pos, 4);
        assert_eq!(src.position(), 4);
    }

    #[test]
    fn slice_source_empty_read_ok() {
        let data = b"hi";
        let mut src = SliceSource::new(data);
        // Reading 0 bytes must succeed
        src.read_exact(&mut []).expect("test: empty read");
        assert_eq!(src.position(), 0);
    }

    #[test]
    fn slice_error_display() {
        assert_eq!(
            SliceError::UnexpectedEof.to_string(),
            "unexpected end of slice"
        );
        assert_eq!(
            SliceError::InvalidSeek.to_string(),
            "seek position exceeds slice length"
        );
    }
}

//! GGUF v3 binary writer — serialize models to the GGUF format.
//!
//! Provides a builder-pattern API for constructing GGUF files from scratch,
//! writing metadata key-value pairs and tensor data with proper alignment.
//!
//! Two write paths are available:
//!
//! - **Buffered** (`add_tensor` + `write_to`/`write_to_file`): every
//!   tensor's bytes are held in memory until the whole file is serialized.
//!   Simple, and fine for tests/small models, but peak memory is O(model
//!   size) — every tensor's bytes are alive simultaneously.
//! - **Streaming** (`declare_tensor` + `into_data_writer`/
//!   `into_file_data_writer`): tensor shapes/types are declared up front (so
//!   every tensor's file offset can be computed before any data is
//!   written — GGUF's tensor-info section precedes the data section and
//!   must state each tensor's final offset), then each tensor's bytes are
//!   streamed straight to the destination one at a time (or one chunk at a
//!   time, for a single tensor too large to buffer). Peak memory is bounded
//!   by whatever the caller holds for a single tensor/chunk, never by the
//!   whole model.
//!
//! The two paths are not interchangeable on one `GgufWriter`: tensors added
//! via `add_tensor` must be finished with `write_to`/`write_to_file`;
//! tensors declared via `declare_tensor` must be finished with
//! `into_data_writer`/`into_file_data_writer`. Mixing them on the same
//! writer returns an error.
//!
//! # Example (buffered)
//!
//! ```no_run
//! use oxillama_gguf::{GgufWriter, MetadataValue, GgufTensorType};
//!
//! let mut writer = GgufWriter::new();
//! writer.add_metadata("general.architecture", MetadataValue::String("llama".into()));
//! writer.add_tensor("output.weight", &[32, 32], GgufTensorType::F32, &vec![0u8; 4096]);
//! writer.write_to_file(std::path::Path::new("model.gguf")).unwrap();
//! ```
//!
//! # Example (streaming, bounded memory)
//!
//! ```no_run
//! use oxillama_gguf::{GgufWriter, MetadataValue, GgufTensorType};
//!
//! let mut writer = GgufWriter::new();
//! writer.add_metadata("general.architecture", MetadataValue::String("llama".into()));
//! writer.declare_tensor("output.weight", &[32, 32], GgufTensorType::F32).unwrap();
//!
//! let mut data_writer = writer
//!     .into_file_data_writer(std::path::Path::new("model.gguf"))
//!     .unwrap();
//! data_writer.write_tensor("output.weight", &vec![0u8; 4096]).unwrap();
//! data_writer.finish().unwrap();
//! ```

use std::io::Write;

use crate::error::{GgufError, GgufResult};
use crate::metadata::MetadataValue;
use crate::tensor_info::TensorInfo;
use crate::types::{GgufTensorType, GgufValueType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};

/// A tensor added via [`GgufWriter::add_tensor`]: shape/type plus its
/// fully-buffered bytes. Consumed by [`GgufWriter::write_to`] /
/// [`GgufWriter::write_to_file`].
struct PendingTensor {
    meta: TensorMeta,
    data: Vec<u8>,
}

/// A tensor's name, shape, type, and precomputed byte size.
///
/// Shared between the legacy buffered path (built from the actual data
/// length in [`GgufWriter::add_tensor`]) and the streaming path (built from
/// the dims/type formula in [`GgufWriter::declare_tensor`]) — both need
/// exactly this to compute every tensor's file offset up front, before any
/// tensor data is written.
#[derive(Clone)]
struct TensorMeta {
    name: String,
    dimensions: Vec<u64>,
    tensor_type: GgufTensorType,
    data_size: u64,
}

/// GGUF v3 file writer with builder-pattern API.
///
/// Accumulates metadata and tensor data, then serializes everything
/// to a conformant GGUF v3 binary stream. See the module docs for the
/// buffered vs. streaming write paths.
pub struct GgufWriter {
    metadata: Vec<(String, MetadataValue)>,
    /// Tensors added via [`Self::add_tensor`] — fully buffered in memory,
    /// consumed by [`Self::write_to`] / [`Self::write_to_file`].
    tensors: Vec<PendingTensor>,
    /// Tensor shapes declared via [`Self::declare_tensor`] — no data yet,
    /// consumed by [`Self::into_data_writer`] / [`Self::into_file_data_writer`].
    tensor_metas: Vec<TensorMeta>,
}

impl GgufWriter {
    /// Create a new empty GGUF writer.
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            tensor_metas: Vec::new(),
        }
    }

    /// Add a metadata key-value pair.
    pub fn add_metadata(&mut self, key: &str, value: MetadataValue) {
        self.metadata.push((key.to_string(), value));
    }

    /// Add a tensor with its dimensions, type, and raw data bytes.
    ///
    /// Buffers `data` in memory until [`Self::write_to`] /
    /// [`Self::write_to_file`] is called — peak memory is O(sum of every
    /// tensor added this way). For large models, prefer
    /// [`Self::declare_tensor`] + [`Self::into_data_writer`] (or
    /// [`Self::into_file_data_writer`]), which streams each tensor straight
    /// to the destination and never holds more than one tensor's (or one
    /// chunk's) bytes at a time. Do not mix the two APIs on one writer — see
    /// the module docs.
    pub fn add_tensor(
        &mut self,
        name: &str,
        dims: &[u64],
        tensor_type: GgufTensorType,
        data: &[u8],
    ) {
        self.tensors.push(PendingTensor {
            meta: TensorMeta {
                name: name.to_string(),
                dimensions: dims.to_vec(),
                tensor_type,
                data_size: data.len() as u64,
            },
            data: data.to_vec(),
        });
    }

    /// Declare a tensor's shape and type without providing data yet.
    ///
    /// This is the first half of the streaming write path: call
    /// `declare_tensor` for every tensor the output file will contain, in
    /// the exact order they should appear on disk. Then call
    /// [`Self::into_data_writer`] (or [`Self::into_file_data_writer`]) to
    /// finalize the header — at that point every tensor's file offset is
    /// already fixed, computed purely from the declared shapes/types, since
    /// GGUF's tensor-info section (which states each tensor's offset)
    /// precedes the tensor-data section. Finally, stream each tensor's
    /// bytes through the returned [`GgufTensorDataWriter`] in the same
    /// order, one tensor (or even one chunk of one tensor) at a time. Peak
    /// memory is bounded by whatever the caller holds for a single
    /// tensor/chunk, never by the whole model.
    ///
    /// Returns an error if `dims` overflows `u64` while computing the
    /// element count, or if the resulting block-count-times-bytes-per-block
    /// overflows `u64` (see [`TensorInfo::try_data_size`]).
    ///
    /// Do not mix this with [`Self::add_tensor`] on the same `GgufWriter`
    /// instance — see the module docs.
    pub fn declare_tensor(
        &mut self,
        name: &str,
        dims: &[u64],
        tensor_type: GgufTensorType,
    ) -> GgufResult<()> {
        let data_size = compute_tensor_data_size(name, dims, tensor_type)?;
        self.tensor_metas.push(TensorMeta {
            name: name.to_string(),
            dimensions: dims.to_vec(),
            tensor_type,
            data_size,
        });
        Ok(())
    }

    /// Serialize the complete GGUF v3 file to a writer.
    ///
    /// Requires every tensor to have been added via [`Self::add_tensor`]
    /// (not [`Self::declare_tensor`]) — this buffers every tensor's bytes
    /// in memory until the whole file is written, exactly as before. For
    /// large models, prefer [`Self::declare_tensor`] +
    /// [`Self::into_data_writer`], which streams each tensor straight to
    /// `writer` and never holds more than one tensor's bytes at a time.
    pub fn write_to<W: Write>(self, writer: &mut W) -> GgufResult<()> {
        if !self.tensor_metas.is_empty() {
            return Err(GgufError::WriteError {
                reason: "write_to()/write_to_file() requires every tensor to be added via \
                         add_tensor(); this GgufWriter has tensor(s) declared via \
                         declare_tensor() instead — use into_data_writer()/into_file_data_writer() \
                         for those, or use add_tensor() consistently"
                    .to_string(),
            });
        }

        let metas: Vec<TensorMeta> = self.tensors.iter().map(|p| p.meta.clone()).collect();
        write_gguf_prelude(writer, &self.metadata, &metas)?;

        let mut data_writer = GgufTensorDataWriter {
            writer,
            metas,
            next_index: 0,
            cumulative_offset: 0,
        };
        for tensor in &self.tensors {
            data_writer.write_tensor(&tensor.meta.name, &tensor.data)?;
        }
        data_writer.finish()?;
        Ok(())
    }

    /// Convenience method to write a GGUF file to disk.
    ///
    /// See [`Self::write_to`] for the memory characteristics — this is the
    /// buffered path. For large models, prefer [`Self::declare_tensor`] +
    /// [`Self::into_file_data_writer`].
    pub fn write_to_file(self, path: &std::path::Path) -> GgufResult<()> {
        let file = std::fs::File::create(path).map_err(io_write_err)?;
        let mut buf_writer = std::io::BufWriter::new(file);
        self.write_to(&mut buf_writer)
    }

    /// Finalize the header (magic/version/counts, metadata KV section,
    /// tensor-info section, and alignment padding) from tensors declared
    /// via [`Self::declare_tensor`], write it to `writer`, and return a
    /// [`GgufTensorDataWriter`] that streams each declared tensor's bytes
    /// into the same `writer`, in declared order.
    ///
    /// `writer` is taken by value: pass an owned sink (a `File`,
    /// `BufWriter<File>`, `Vec<u8>`, `io::sink()`, …) or a `&mut` borrow of
    /// one — `&mut W` itself implements `Write`, so it satisfies the bound.
    ///
    /// Returns an error if any tensor was instead added via
    /// [`Self::add_tensor`] (the two APIs are not interchangeable on one
    /// `GgufWriter`), or if writing the header section fails.
    pub fn into_data_writer<W: Write>(self, mut writer: W) -> GgufResult<GgufTensorDataWriter<W>> {
        if !self.tensors.is_empty() {
            return Err(GgufError::WriteError {
                reason: "into_data_writer()/into_file_data_writer() requires every tensor to be \
                         declared via declare_tensor(); this GgufWriter has tensor(s) added via \
                         add_tensor() instead — use write_to()/write_to_file() for those, or use \
                         declare_tensor() consistently"
                    .to_string(),
            });
        }
        write_gguf_prelude(&mut writer, &self.metadata, &self.tensor_metas)?;
        Ok(GgufTensorDataWriter {
            writer,
            metas: self.tensor_metas,
            next_index: 0,
            cumulative_offset: 0,
        })
    }

    /// Convenience wrapper around [`Self::into_data_writer`] that creates
    /// (truncating) `path` and wraps it in a [`std::io::BufWriter`].
    pub fn into_file_data_writer(
        self,
        path: &std::path::Path,
    ) -> GgufResult<GgufTensorDataWriter<std::io::BufWriter<std::fs::File>>> {
        let file = std::fs::File::create(path).map_err(io_write_err)?;
        self.into_data_writer(std::io::BufWriter::new(file))
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Streams tensor data into a GGUF v3 file after [`GgufWriter::into_data_writer`]
/// (or [`GgufWriter::into_file_data_writer`]) has already written the
/// header, metadata, and tensor-info sections.
///
/// Call [`Self::write_tensor`] (or [`Self::write_tensor_from_chunks`] for a
/// tensor too large to hold in one buffer) once per declared tensor, in the
/// exact order they were declared, then [`Self::finish`]. Peak memory is
/// bounded by whatever the caller passes to a single call — this type never
/// buffers more than one tensor's (or one chunk's) bytes at a time; bytes
/// are written straight through to the inner writer.
pub struct GgufTensorDataWriter<W: Write> {
    writer: W,
    metas: Vec<TensorMeta>,
    next_index: usize,
    cumulative_offset: u64,
}

impl<W: Write> GgufTensorDataWriter<W> {
    /// Validate that `name` is the next undelivered tensor, in declared
    /// order, and return its declared byte size.
    fn validate_next(&self, name: &str) -> GgufResult<u64> {
        let meta = self
            .metas
            .get(self.next_index)
            .ok_or_else(|| GgufError::WriteError {
                reason: format!(
                    "write_tensor('{name}') called but all {} declared tensor(s) were already written",
                    self.metas.len()
                ),
            })?;
        if meta.name != name {
            return Err(GgufError::WriteError {
                reason: format!(
                    "tensor write-order violation: expected '{}' next (position {} of {}), got '{name}'",
                    meta.name,
                    self.next_index,
                    self.metas.len()
                ),
            });
        }
        Ok(meta.data_size)
    }

    /// Record that `written` bytes were just written for the tensor at
    /// `next_index`, advance to the next tensor, and — unless this was the
    /// last declared tensor — write the zero padding needed before the next
    /// tensor starts. Mirrors the legacy buffered writer's behavior
    /// exactly: padding is computed for every tensor (including the last)
    /// when the tensor-info offsets are written up front, but only ever
    /// *emitted* between tensors, never trailing the final one.
    fn advance_after_tensor(&mut self, written: u64) -> GgufResult<()> {
        self.cumulative_offset =
            self.cumulative_offset
                .checked_add(written)
                .ok_or_else(|| GgufError::WriteError {
                    reason: "cumulative tensor data offset overflows u64".to_string(),
                })?;
        self.next_index += 1;
        if self.next_index < self.metas.len() {
            let pad = alignment_padding(self.cumulative_offset, GGUF_DEFAULT_ALIGNMENT);
            if pad > 0 {
                self.writer
                    .write_all(&ZERO_PAD[..pad as usize])
                    .map_err(io_write_err)?;
            }
            self.cumulative_offset = self.cumulative_offset.saturating_add(pad);
        }
        Ok(())
    }

    /// Write one tensor's complete data in a single call.
    ///
    /// `name` must be the next undelivered tensor in declared order, and
    /// `data.len()` must exactly equal the size implied by the dims/type
    /// passed to [`GgufWriter::declare_tensor`].
    pub fn write_tensor(&mut self, name: &str, data: &[u8]) -> GgufResult<()> {
        let expected_size = self.validate_next(name)?;
        let len = data.len() as u64;
        if len != expected_size {
            return Err(GgufError::WriteError {
                reason: format!(
                    "tensor '{name}': data length {len} does not match declared size {expected_size}"
                ),
            });
        }
        self.writer.write_all(data).map_err(io_write_err)?;
        self.advance_after_tensor(len)
    }

    /// Write one tensor's data from a sequence of chunks, without ever
    /// materializing the whole tensor in memory — e.g. a multi-gigabyte
    /// tensor produced by an on-the-fly generator. The sum of chunk lengths
    /// must exactly equal the declared size; `name` must be the next
    /// undelivered tensor in declared order.
    pub fn write_tensor_from_chunks<I>(&mut self, name: &str, chunks: I) -> GgufResult<()>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        let expected_size = self.validate_next(name)?;
        let mut written = 0u64;
        for chunk in chunks {
            let bytes = chunk.as_ref();
            let new_written =
                written
                    .checked_add(bytes.len() as u64)
                    .ok_or_else(|| GgufError::WriteError {
                        reason: format!("tensor '{name}': streamed byte count overflows u64"),
                    })?;
            if new_written > expected_size {
                return Err(GgufError::WriteError {
                    reason: format!(
                        "tensor '{name}': streamed {new_written} bytes so far, exceeding declared size {expected_size}"
                    ),
                });
            }
            self.writer.write_all(bytes).map_err(io_write_err)?;
            written = new_written;
        }
        if written != expected_size {
            return Err(GgufError::WriteError {
                reason: format!(
                    "tensor '{name}': streamed {written} bytes total, declared size is {expected_size}"
                ),
            });
        }
        self.advance_after_tensor(written)
    }

    /// Validate that every declared tensor was written, flush, and return
    /// the inner writer.
    pub fn finish(mut self) -> GgufResult<W> {
        if self.next_index != self.metas.len() {
            return Err(GgufError::WriteError {
                reason: format!(
                    "finish() called with {} of {} declared tensor(s) not yet written",
                    self.metas.len() - self.next_index,
                    self.metas.len()
                ),
            });
        }
        self.writer.flush().map_err(io_write_err)?;
        Ok(self.writer)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Reusable zero buffer for inter-section/inter-tensor padding. Large
/// enough for any padding this writer emits, since every value returned by
/// [`alignment_padding`] against [`GGUF_DEFAULT_ALIGNMENT`] is strictly less
/// than 32 bytes.
const ZERO_PAD: [u8; GGUF_DEFAULT_ALIGNMENT as usize] = [0u8; GGUF_DEFAULT_ALIGNMENT as usize];

/// Compute a declared tensor's exact byte size from its dims and type,
/// using the same checked (overflow-safe) formula the parser uses to
/// interpret tensor-info entries on read-back
/// ([`TensorInfo::try_data_size`]) — so a tensor declared here can never
/// silently disagree with how a conformant reader will size it.
fn compute_tensor_data_size(
    name: &str,
    dims: &[u64],
    tensor_type: GgufTensorType,
) -> GgufResult<u64> {
    let info = TensorInfo {
        name: name.to_string(),
        n_dims: dims.len() as u32,
        dimensions: dims.to_vec(),
        tensor_type,
        offset: 0,
    };
    info.try_data_size()
}

/// Write the GGUF v3 header, metadata KV section, tensor-info section (with
/// every tensor's file offset computed up front from `tensor_metas`), and
/// the alignment padding before the tensor-data section.
///
/// Does not write any tensor data — callers stream that immediately
/// afterward into the same `writer` (see [`GgufTensorDataWriter`]). Both the
/// offsets computed here and the padding [`GgufTensorDataWriter`] emits
/// while streaming data are pure functions of the same `data_size` sequence
/// via [`alignment_padding`], so they always agree by construction.
fn write_gguf_prelude<W: Write>(
    writer: &mut W,
    metadata: &[(String, MetadataValue)],
    tensor_metas: &[TensorMeta],
) -> GgufResult<()> {
    let mut offset: u64 = 0;

    // 1. Header
    offset += write_header(writer, tensor_metas.len() as u64, metadata.len() as u64)?;

    // 2. Metadata KV pairs
    for (key, value) in metadata {
        offset += write_string(writer, key)?;
        offset += write_metadata_value(writer, value)?;
    }

    // 3. Tensor info entries — compute offsets within the data section up front.
    let mut tensor_data_offset: u64 = 0;
    let mut tensor_offsets = Vec::with_capacity(tensor_metas.len());
    for meta in tensor_metas {
        tensor_offsets.push(tensor_data_offset);
        tensor_data_offset = tensor_data_offset
            .checked_add(meta.data_size)
            .ok_or_else(|| GgufError::IntegrityError {
                tensor_name: meta.name.clone(),
                reason: "cumulative tensor data offset overflows u64".to_string(),
            })?;
        // Align to next tensor (except possibly last).
        let padding = alignment_padding(tensor_data_offset, GGUF_DEFAULT_ALIGNMENT);
        tensor_data_offset = tensor_data_offset.saturating_add(padding);
    }

    for (i, meta) in tensor_metas.iter().enumerate() {
        offset += write_string(writer, &meta.name)?;
        let n_dims = meta.dimensions.len() as u32;
        writer
            .write_all(&n_dims.to_le_bytes())
            .map_err(io_write_err)?;
        offset += 4;
        for &dim in &meta.dimensions {
            writer.write_all(&dim.to_le_bytes()).map_err(io_write_err)?;
            offset += 8;
        }
        writer
            .write_all(&(meta.tensor_type as u32).to_le_bytes())
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
        writer
            .write_all(&ZERO_PAD[..header_pad as usize])
            .map_err(io_write_err)?;
    }

    Ok(())
}

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

    // ═══════════════════════════════════════════════════════════════════
    // Streaming writer: golden-byte regression
    // ═══════════════════════════════════════════════════════════════════
    //
    // `GOLDEN` was captured from the pre-refactor, fully-buffered
    // `GgufWriter` (the version that only had `add_tensor` +
    // `write_to`/`write_to_file`, before `declare_tensor` /
    // `into_data_writer` existed) writing the fixture built by
    // `golden_metadata`/`golden_tensors` below, and round-tripped through
    // `GgufFile::parse` to confirm it was valid before freezing it here.
    // The fixture covers every `MetadataValue` variant (including a flat
    // array and a 3-element nested array-of-arrays with an empty leaf) and
    // five tensors across four dtypes (F32, F16, Q4_0, Q8_0) with
    // deliberately non-32-aligned sizes, so real inter-tensor padding is
    // exercised, not just the trivial zero-padding case.

    #[rustfmt::skip]
    const GOLDEN: &[u8] = &[
        0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x67, 0x65, 0x6E, 0x65,
        0x72, 0x61, 0x6C, 0x2E, 0x61, 0x72, 0x63, 0x68, 0x69, 0x74, 0x65, 0x63,
        0x74, 0x75, 0x72, 0x65, 0x08, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x67, 0x6F, 0x6C, 0x64, 0x65, 0x6E, 0x2D, 0x66,
        0x69, 0x78, 0x74, 0x75, 0x72, 0x65, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x75, 0x38, 0x00, 0x00, 0x00,
        0x00, 0xC8, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6D, 0x65,
        0x74, 0x61, 0x2E, 0x69, 0x38, 0x01, 0x00, 0x00, 0x00, 0x9C, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x75,
        0x31, 0x36, 0x02, 0x00, 0x00, 0x00, 0x50, 0xC3, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x69, 0x31, 0x36,
        0x03, 0x00, 0x00, 0x00, 0xD0, 0x8A, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x75, 0x33, 0x32, 0x04, 0x00,
        0x00, 0x00, 0x00, 0x28, 0x6B, 0xEE, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x69, 0x33, 0x32, 0x05, 0x00,
        0x00, 0x00, 0x00, 0x6C, 0xCA, 0x88, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x66, 0x33, 0x32, 0x06, 0x00,
        0x00, 0x00, 0xB7, 0xE6, 0x40, 0x46, 0x0E, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x62, 0x6F, 0x6F, 0x6C, 0x5F,
        0x74, 0x72, 0x75, 0x65, 0x07, 0x00, 0x00, 0x00, 0x01, 0x0F, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x62, 0x6F,
        0x6F, 0x6C, 0x5F, 0x66, 0x61, 0x6C, 0x73, 0x65, 0x07, 0x00, 0x00, 0x00,
        0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6D, 0x65, 0x74,
        0x61, 0x2E, 0x75, 0x36, 0x34, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
        0xC5, 0xA1, 0xD8, 0xCC, 0xF9, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x69, 0x36, 0x34, 0x0B, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x7C, 0x1D, 0xAF, 0x93, 0x19, 0x83, 0x08, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x66, 0x36,
        0x34, 0x0C, 0x00, 0x00, 0x00, 0x9B, 0x0B, 0xEC, 0xE9, 0xD6, 0x1C, 0xF8,
        0x40, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6D, 0x65, 0x74,
        0x61, 0x2E, 0x61, 0x72, 0x72, 0x61, 0x79, 0x5F, 0x66, 0x6C, 0x61, 0x74,
        0x09, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, 0xFC, 0xFF, 0xFF, 0xFF, 0x11, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x6D, 0x65, 0x74, 0x61, 0x2E, 0x61, 0x72, 0x72,
        0x61, 0x79, 0x5F, 0x6E, 0x65, 0x73, 0x74, 0x65, 0x64, 0x09, 0x00, 0x00,
        0x00, 0x09, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x74, 0x69, 0x6E, 0x79, 0x2E, 0x66, 0x33, 0x32, 0x01, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x73, 0x6D, 0x61, 0x6C, 0x6C, 0x2E, 0x71, 0x34,
        0x5F, 0x30, 0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x0B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6D, 0x65,
        0x64, 0x69, 0x75, 0x6D, 0x2E, 0x71, 0x38, 0x5F, 0x30, 0x01, 0x00, 0x00,
        0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00,
        0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x66, 0x31, 0x36, 0x2E, 0x76, 0x65, 0x63,
        0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6C, 0x61, 0x73, 0x74,
        0x2E, 0x66, 0x33, 0x32, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F,
        0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB,
        0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xCD, 0xCD, 0xCD, 0xCD,
        0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD,
        0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD,
        0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD,
        0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD,
        0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD,
        0xCD, 0xCD, 0xCD, 0xCD, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x11, 0x11, 0x11,
        0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
        0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xC0, 0x40, 0x00, 0x00, 0x00, 0x41,
        0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x60, 0x41,
    ];

    /// Metadata for the golden fixture: one of every `MetadataValue`
    /// variant, plus a flat array and a nested array-of-arrays (with an
    /// empty leaf, to exercise `write_array`'s empty-array element-type
    /// fallback).
    fn golden_metadata(writer: &mut GgufWriter) {
        writer.add_metadata(
            "general.architecture",
            MetadataValue::String("golden-fixture".to_string()),
        );
        writer.add_metadata("meta.u8", MetadataValue::Uint8(200));
        writer.add_metadata("meta.i8", MetadataValue::Int8(-100));
        writer.add_metadata("meta.u16", MetadataValue::Uint16(50_000));
        writer.add_metadata("meta.i16", MetadataValue::Int16(-30_000));
        writer.add_metadata("meta.u32", MetadataValue::Uint32(4_000_000_000));
        writer.add_metadata("meta.i32", MetadataValue::Int32(-2_000_000_000));
        writer.add_metadata("meta.f32", MetadataValue::Float32(12_345.679));
        writer.add_metadata("meta.bool_true", MetadataValue::Bool(true));
        writer.add_metadata("meta.bool_false", MetadataValue::Bool(false));
        writer.add_metadata(
            "meta.u64",
            MetadataValue::Uint64(18_000_000_000_000_000_000),
        );
        writer.add_metadata("meta.i64", MetadataValue::Int64(-9_000_000_000_000_000_000));
        writer.add_metadata("meta.f64", MetadataValue::Float64(98765.432109876));
        writer.add_metadata(
            "meta.array_flat",
            MetadataValue::Array(vec![
                MetadataValue::Int32(1),
                MetadataValue::Int32(2),
                MetadataValue::Int32(3),
                MetadataValue::Int32(-4),
            ]),
        );
        writer.add_metadata(
            "meta.array_nested",
            MetadataValue::Array(vec![
                MetadataValue::Array(vec![MetadataValue::Uint8(1), MetadataValue::Uint8(2)]),
                MetadataValue::Array(vec![MetadataValue::Uint8(3)]),
                MetadataValue::Array(vec![]),
            ]),
        );
    }

    /// Five tensors across four dtypes with deliberately non-32-aligned
    /// sizes (12, 18, 68, 10 bytes) plus one already-aligned, last tensor
    /// (32 bytes) — exercises real inter-tensor padding computation, not
    /// just the trivial zero-padding case.
    #[allow(clippy::type_complexity)]
    fn golden_tensors() -> Vec<(&'static str, Vec<u64>, GgufTensorType, Vec<u8>)> {
        let f32_tiny: Vec<u8> = (0..3u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let f32_last: Vec<u8> = (0..8u32)
            .flat_map(|i| (i as f32 * 2.0).to_le_bytes())
            .collect();
        vec![
            ("tiny.f32", vec![3], GgufTensorType::F32, f32_tiny), // 12 bytes
            (
                "small.q4_0",
                vec![32],
                GgufTensorType::Q4_0,
                vec![0xABu8; 18],
            ), // 18 bytes
            (
                "medium.q8_0",
                vec![64],
                GgufTensorType::Q8_0,
                vec![0xCDu8; 68],
            ), // 68 bytes
            ("f16.vec", vec![5], GgufTensorType::F16, vec![0x11u8; 10]), // 10 bytes
            ("last.f32", vec![8], GgufTensorType::F32, f32_last), // 32 bytes, already aligned
        ]
    }

    #[test]
    fn test_golden_bytes_buffered_writer_matches_captured_fixture() {
        let mut writer = GgufWriter::new();
        golden_metadata(&mut writer);
        for (name, dims, ty, data) in golden_tensors() {
            writer.add_tensor(name, &dims, ty, &data);
        }
        let mut buf = Vec::new();
        writer.write_to(&mut buf).expect("write golden fixture");
        assert_eq!(
            buf, GOLDEN,
            "buffered add_tensor()/write_to() output drifted from the pre-refactor golden bytes"
        );
    }

    #[test]
    fn test_golden_bytes_streaming_writer_matches_buffered_and_captured_fixture() {
        let mut writer = GgufWriter::new();
        golden_metadata(&mut writer);
        for (name, dims, ty, _data) in golden_tensors() {
            writer
                .declare_tensor(name, &dims, ty)
                .expect("declare_tensor");
        }

        let mut buf = Vec::new();
        let mut data_writer = writer.into_data_writer(&mut buf).expect("into_data_writer");
        for (name, _dims, _ty, data) in golden_tensors() {
            data_writer.write_tensor(name, &data).expect("write_tensor");
        }
        data_writer.finish().expect("finish");

        assert_eq!(
            buf, GOLDEN,
            "streaming declare_tensor()/into_data_writer() output diverged from the golden bytes \
             the pre-refactor buffered writer produced for the identical fixture"
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Streaming writer: order/size validation and API-mixing guards
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_streaming_write_tensor_wrong_order_errors() {
        let mut writer = GgufWriter::new();
        writer
            .declare_tensor("a", &[4], GgufTensorType::F32)
            .expect("declare a");
        writer
            .declare_tensor("b", &[4], GgufTensorType::F32)
            .expect("declare b");

        let mut buf = Vec::new();
        let mut data_writer = writer.into_data_writer(&mut buf).expect("into_data_writer");
        let err = data_writer
            .write_tensor("b", &[0u8; 16])
            .expect_err("writing 'b' before 'a' must error");
        assert!(matches!(err, GgufError::WriteError { .. }));
    }

    #[test]
    fn test_streaming_write_tensor_size_mismatch_errors() {
        let mut writer = GgufWriter::new();
        writer
            .declare_tensor("a", &[4], GgufTensorType::F32) // 16 bytes
            .expect("declare a");

        let mut buf = Vec::new();
        let mut data_writer = writer.into_data_writer(&mut buf).expect("into_data_writer");
        let err = data_writer
            .write_tensor("a", &[0u8; 8])
            .expect_err("size mismatch must error");
        assert!(matches!(err, GgufError::WriteError { .. }));
    }

    #[test]
    fn test_streaming_write_tensor_past_end_errors() {
        let mut writer = GgufWriter::new();
        writer
            .declare_tensor("a", &[4], GgufTensorType::F32) // 16 bytes
            .expect("declare a");

        let mut buf = Vec::new();
        let mut data_writer = writer.into_data_writer(&mut buf).expect("into_data_writer");
        data_writer.write_tensor("a", &[0u8; 16]).expect("write a");
        let err = data_writer
            .write_tensor("a", &[0u8; 16])
            .expect_err("writing past the last declared tensor must error");
        assert!(matches!(err, GgufError::WriteError { .. }));
    }

    #[test]
    fn test_streaming_finish_with_undelivered_tensor_errors() {
        let mut writer = GgufWriter::new();
        writer
            .declare_tensor("a", &[4], GgufTensorType::F32)
            .expect("declare a");
        writer
            .declare_tensor("b", &[4], GgufTensorType::F32)
            .expect("declare b");

        let mut buf = Vec::new();
        let mut data_writer = writer.into_data_writer(&mut buf).expect("into_data_writer");
        data_writer.write_tensor("a", &[0u8; 16]).expect("write a");
        let err = data_writer
            .finish()
            .expect_err("finish() before all declared tensors are written must error");
        assert!(matches!(err, GgufError::WriteError { .. }));
    }

    #[test]
    fn test_mixing_add_tensor_and_declare_tensor_errors_on_write_to() {
        let mut writer = GgufWriter::new();
        writer.add_tensor("a", &[4], GgufTensorType::F32, &[0u8; 16]);
        writer
            .declare_tensor("b", &[4], GgufTensorType::F32)
            .expect("declare b");

        let mut buf = Vec::new();
        let err = writer
            .write_to(&mut buf)
            .expect_err("mixing add_tensor() with declare_tensor() must error on write_to()");
        assert!(matches!(err, GgufError::WriteError { .. }));
    }

    #[test]
    fn test_mixing_add_tensor_and_declare_tensor_errors_on_into_data_writer() {
        let mut writer = GgufWriter::new();
        writer.add_tensor("a", &[4], GgufTensorType::F32, &[0u8; 16]);
        writer
            .declare_tensor("b", &[4], GgufTensorType::F32)
            .expect("declare b");

        let mut buf = Vec::new();
        // `expect_err` would require `GgufTensorDataWriter<W>: Debug`, which
        // this type deliberately doesn't derive — match instead.
        match writer.into_data_writer(&mut buf) {
            Ok(_) => {
                panic!("mixing add_tensor() with declare_tensor() must error on into_data_writer()")
            }
            Err(e) => assert!(matches!(e, GgufError::WriteError { .. })),
        }
    }

    #[test]
    fn test_declare_tensor_dimension_overflow_errors() {
        let mut writer = GgufWriter::new();
        let err = writer
            .declare_tensor("huge", &[1u64 << 32, 1u64 << 32], GgufTensorType::F32)
            .expect_err("dims overflowing u64 while computing element count must error");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Streaming writer: chunked tensor data
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_write_tensor_from_chunks_streams_without_one_big_buffer() {
        let mut writer = GgufWriter::new();
        // 256 F32 elements = 1024 bytes, streamed as 16 chunks of 64 bytes.
        writer
            .declare_tensor("chunked", &[256], GgufTensorType::F32)
            .expect("declare");

        let mut buf = Vec::new();
        let mut data_writer = writer.into_data_writer(&mut buf).expect("into_data_writer");
        let chunk = vec![0x42u8; 64];
        data_writer
            .write_tensor_from_chunks("chunked", std::iter::repeat_n(chunk.as_slice(), 16))
            .expect("stream chunks");
        data_writer.finish().expect("finish");

        let parsed = GgufFile::parse(&buf).expect("parse chunked output");
        let data = parsed
            .tensor_data(&buf, "chunked")
            .expect("get chunked tensor data");
        assert_eq!(data.len(), 1024);
        assert!(data.iter().all(|&b| b == 0x42));
    }

    #[test]
    fn test_write_tensor_from_chunks_overshoot_errors() {
        let mut writer = GgufWriter::new();
        writer
            .declare_tensor("a", &[4], GgufTensorType::F32) // 16 bytes
            .expect("declare a");

        let mut buf = Vec::new();
        let mut data_writer = writer.into_data_writer(&mut buf).expect("into_data_writer");
        let err = data_writer
            .write_tensor_from_chunks("a", vec![vec![0u8; 20]])
            .expect_err("a chunk exceeding the declared size must error");
        assert!(matches!(err, GgufError::WriteError { .. }));
    }

    #[test]
    fn test_write_tensor_from_chunks_undershoot_errors() {
        let mut writer = GgufWriter::new();
        writer
            .declare_tensor("a", &[4], GgufTensorType::F32) // 16 bytes
            .expect("declare a");

        let mut buf = Vec::new();
        let mut data_writer = writer.into_data_writer(&mut buf).expect("into_data_writer");
        let err = data_writer
            .write_tensor_from_chunks("a", vec![vec![0u8; 8]])
            .expect_err("fewer bytes than declared must error");
        assert!(matches!(err, GgufError::WriteError { .. }));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Streaming writer: real-file check, 100+ tensors
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_write_100_plus_tensors_offsets_verified_against_manual_header_walk() {
        let dir = std::env::temp_dir().join("oxillama_gguf_writer_streaming_test");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("hundred_tensors.gguf");

        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "general.architecture",
            MetadataValue::String("stress".to_string()),
        );

        const N_TENSORS: usize = 137;
        let mut expected: Vec<(String, u64)> = Vec::new();
        for i in 0..N_TENSORS {
            let name = format!("blk.{i}.weight");
            let (dims, ty, size): (Vec<u64>, GgufTensorType, u64) = match i % 3 {
                0 => (
                    vec![(i as u64) + 1],
                    GgufTensorType::F32,
                    (i as u64 + 1) * 4,
                ),
                1 => (vec![32], GgufTensorType::Q4_0, 18),
                _ => (vec![64], GgufTensorType::Q8_0, 68),
            };
            writer
                .declare_tensor(&name, &dims, ty)
                .unwrap_or_else(|e| panic!("declare '{name}': {e}"));
            expected.push((name, size));
        }

        let mut data_writer = writer
            .into_file_data_writer(&path)
            .expect("into_file_data_writer");
        for (name, size) in &expected {
            let data = vec![0x5Au8; *size as usize];
            data_writer
                .write_tensor(name, &data)
                .unwrap_or_else(|e| panic!("write_tensor '{name}': {e}"));
        }
        data_writer.finish().expect("finish");

        // Re-parse from disk and independently recompute every tensor's
        // expected offset (32-byte alignment, cumulative over declared
        // sizes) without reusing any of the writer's own bookkeeping.
        let file_bytes = std::fs::read(&path).expect("read back");
        let parsed = GgufFile::parse(&file_bytes).expect("parse 100+ tensor file");
        assert_eq!(parsed.header.tensor_count, N_TENSORS as u64);

        let mut expected_offset = 0u64;
        for (name, size) in &expected {
            let info = parsed.tensors.get(name).expect("tensor present");
            assert_eq!(
                info.offset, expected_offset,
                "tensor '{name}' offset mismatch"
            );
            assert_eq!(
                info.offset % GGUF_DEFAULT_ALIGNMENT,
                0,
                "tensor '{name}' offset {} is not 32-byte aligned",
                info.offset
            );
            expected_offset += size;
            let rem = expected_offset % GGUF_DEFAULT_ALIGNMENT;
            if rem != 0 {
                expected_offset += GGUF_DEFAULT_ALIGNMENT - rem;
            }
        }

        // Hexdump-style manual walk of the raw header region: decode the
        // magic/version/counts, the single metadata KV pair, and the FIRST
        // tensor-info entry directly from the byte buffer (independent of
        // `GgufFile::parse`) and cross-check against what the parser saw.
        assert_eq!(&file_bytes[0..4], b"GGUF");
        assert_eq!(u32::from_le_bytes(file_bytes[4..8].try_into().unwrap()), 3);
        assert_eq!(
            u64::from_le_bytes(file_bytes[8..16].try_into().unwrap()),
            N_TENSORS as u64
        );
        assert_eq!(
            u64::from_le_bytes(file_bytes[16..24].try_into().unwrap()),
            1
        );

        let mut cursor = 24usize;
        let key_len = u64::from_le_bytes(file_bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8 + key_len as usize;
        let val_type = u32::from_le_bytes(file_bytes[cursor..cursor + 4].try_into().unwrap());
        assert_eq!(val_type, GgufValueType::String as u32);
        cursor += 4;
        let val_len = u64::from_le_bytes(file_bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8 + val_len as usize;

        // First tensor-info entry: "blk.0.weight", 1-D [1], F32, offset 0.
        let name_len = u64::from_le_bytes(file_bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let name_bytes = &file_bytes[cursor..cursor + name_len as usize];
        assert_eq!(name_bytes, b"blk.0.weight");
        cursor += name_len as usize;
        let n_dims = u32::from_le_bytes(file_bytes[cursor..cursor + 4].try_into().unwrap());
        assert_eq!(n_dims, 1);
        cursor += 4;
        let dim0 = u64::from_le_bytes(file_bytes[cursor..cursor + 8].try_into().unwrap());
        assert_eq!(dim0, 1);
        cursor += 8;
        let ttype = u32::from_le_bytes(file_bytes[cursor..cursor + 4].try_into().unwrap());
        assert_eq!(ttype, GgufTensorType::F32 as u32);
        cursor += 4;
        let toffset = u64::from_le_bytes(file_bytes[cursor..cursor + 8].try_into().unwrap());
        assert_eq!(toffset, 0, "first tensor's offset must be 0");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Streaming writer: memory bound (run manually with `/usr/bin/time -l`
    // against the compiled test binary — wrapping `cargo test`/`cargo
    // nextest` instead measures the build-tool process, not this one).
    // Streams into `io::sink()` so the multi-GB logical size never touches
    // disk or RAM as a single buffer: only one 4 MiB chunk is ever
    // materialized, reused (not reallocated) for every write.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    #[ignore = "slow / memory-sensitive; run manually with /usr/bin/time -l"]
    fn test_streaming_multi_gb_logical_tensor_bounded_memory() {
        const CHUNK_LEN: usize = 4 * 1024 * 1024; // 4 MiB
        const CHUNK_COUNT: usize = 1024; // 4 GiB logical tensor
        const TOTAL_BYTES: u64 = CHUNK_LEN as u64 * CHUNK_COUNT as u64;
        let n_elements = TOTAL_BYTES / 4; // F32: 4 bytes/element

        let mut writer = GgufWriter::new();
        writer
            .declare_tensor("huge", &[n_elements], GgufTensorType::F32)
            .expect("declare huge tensor");

        let mut data_writer = writer
            .into_data_writer(std::io::sink())
            .expect("into_data_writer(sink)");

        let pattern = vec![0xEFu8; CHUNK_LEN];
        data_writer
            .write_tensor_from_chunks("huge", std::iter::repeat_n(pattern.as_slice(), CHUNK_COUNT))
            .expect("stream a 4 GiB logical tensor without materializing it");
        data_writer.finish().expect("finish");
    }
}

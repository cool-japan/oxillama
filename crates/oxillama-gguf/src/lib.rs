//! # oxillama-gguf
//!
//! GGUF v3 binary format parser and tensor loader for OxiLLaMa.
//!
//! This crate provides complete parsing of the GGUF file format, including:
//! - Binary header validation (magic, version)
//! - Typed key-value metadata extraction
//! - Tensor info parsing (name, shape, quantization type, offset)
//! - Memory-mapped tensor data access (via `mmap` feature)
//! - Full file loading with mmap or read-to-memory
//!
//! ## Supported GGUF Versions
//! - Version 2 (legacy)
//! - Version 3 (current standard)
//!
//! ## Quick Start
//!
//! ```no_run
//! use oxillama_gguf::GgufModel;
//!
//! let model = GgufModel::load("model.gguf").unwrap();
//! println!("Architecture: {}", model.architecture().unwrap());
//! println!("Tensors: {}", model.file.header.tensor_count);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod error;
pub mod header;
pub mod metadata;
pub mod parser;
pub mod reader;
pub mod reader_core;
pub mod source;
pub mod tensor_info;
pub mod types;

#[cfg(feature = "std")]
pub mod loader;
#[cfg(feature = "std")]
pub mod quantize_on_load;
#[cfg(feature = "std")]
pub mod resume;
#[cfg(feature = "std")]
pub mod schema;
#[cfg(feature = "std")]
pub mod sharded;
#[cfg(feature = "std")]
pub mod streaming;
#[cfg(feature = "std")]
pub mod writer;

#[cfg(all(feature = "std", feature = "integrity"))]
pub mod integrity;

#[cfg(all(feature = "std", any(test, feature = "test-utils")))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;

pub use error::{GgufError, GgufResult};
pub use header::GgufHeader;
pub use metadata::{MetadataStore, MetadataValue};
pub use parser::GgufFile;
pub use reader::BinaryReader;
pub use reader_core::{align_up, parse_gguf, ParsedGguf};
pub use source::{SliceSource, Source};
pub use tensor_info::{TensorInfo, TensorStore};
pub use types::{GgufTensorType, GgufValueType};

#[cfg(feature = "std")]
pub use source::{FileSource, ReadSource};

#[cfg(feature = "std")]
pub use loader::GgufModel;
#[cfg(feature = "std")]
pub use quantize_on_load::{QuantPlan, QuantTarget};
#[cfg(feature = "std")]
pub use resume::{
    checkpoint_path_for, compute_fingerprint, compute_fingerprint_with_probe, load_checkpoint,
    save_checkpoint, validate_checkpoint, PrefixFingerprint, ResumeCheckpoint, ResumeHandle,
};
#[cfg(feature = "std")]
pub use schema::{validate_schema, SchemaValidator, SchemaViolation};
#[cfg(feature = "std")]
pub use sharded::ShardedGgufModel;
#[cfg(feature = "std")]
pub use streaming::{StreamingGgufParser, TensorInfoIter};
#[cfg(feature = "std")]
pub use writer::GgufWriter;

#[cfg(all(feature = "std", feature = "integrity"))]
pub use integrity::{
    compute_model_manifest, format_manifest, hash_tensor, hash_to_hex, verify_model, verify_tensor,
    IntegrityFailure, ModelHashManifest, TensorHash, TensorHashEntry, TensorHashValidator,
};

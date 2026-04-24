#![cfg_attr(docsrs, feature(doc_cfg))]
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

pub mod error;
pub mod header;
pub mod loader;
pub mod metadata;
pub mod parser;
pub mod quantize_on_load;
pub mod reader;
pub mod resume;
pub mod schema;
pub mod sharded;
pub mod streaming;
pub mod tensor_info;
pub mod types;
pub mod writer;

#[cfg(feature = "integrity")]
pub mod integrity;

#[cfg(any(test, feature = "test-utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;

pub use error::{GgufError, GgufResult};
pub use header::GgufHeader;
pub use loader::GgufModel;
pub use metadata::{MetadataStore, MetadataValue};
pub use parser::GgufFile;
pub use quantize_on_load::{QuantPlan, QuantTarget};
pub use reader::BinaryReader;
pub use resume::{
    checkpoint_path_for, compute_fingerprint, compute_fingerprint_with_probe, load_checkpoint,
    save_checkpoint, validate_checkpoint, PrefixFingerprint, ResumeCheckpoint, ResumeHandle,
};
pub use schema::{validate_schema, SchemaValidator, SchemaViolation};
pub use sharded::ShardedGgufModel;
pub use streaming::{StreamingGgufParser, TensorInfoIter};
pub use tensor_info::{TensorInfo, TensorStore};
pub use types::{GgufTensorType, GgufValueType};
pub use writer::GgufWriter;

#[cfg(feature = "integrity")]
pub use integrity::{
    compute_model_manifest, format_manifest, hash_tensor, hash_to_hex, verify_model, verify_tensor,
    IntegrityFailure, ModelHashManifest, TensorHash, TensorHashEntry, TensorHashValidator,
};

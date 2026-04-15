//! Fuzz target: GgufModel::from_bytes
//!
//! Exercises the higher-level GgufModel wrapper which owns its backing buffer.
//! On a successful parse this target also drives:
//!   - `tensor_data(name)` for every registered tensor (offset arithmetic,
//!     OOB detection, mmap/owned dispatch)
//!   - `architecture()` and `model_name()` metadata lookups
//!   - `print_summary()` format-string generation
//!
//! GgufModel::from_bytes takes Vec<u8> and keeps ownership of the buffer for
//! the lifetime of the model — so the tensor-data slices returned by
//! `tensor_data` borrow from within the model itself, giving a unified
//! code-path distinct from the GgufFile::parse + external-slice approach.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxillama_gguf::GgufModel;

fuzz_target!(|data: &[u8]| {
    // from_bytes takes Vec<u8>; clone the fuzz input to satisfy the signature.
    if let Ok(model) = GgufModel::from_bytes(data.to_vec()) {
        // Exercise tensor-data access for every tensor in the registry.
        let names: Vec<String> = model.file.tensors.names().cloned().collect();
        for name in &names {
            let _ = model.tensor_data(name);
        }

        // Exercise metadata accessor paths.
        let _ = model.architecture();
        let _ = model.model_name();

        // Exercise the summary formatter (exercises all header fields).
        let mut summary = String::new();
        let _ = model.print_summary(&mut summary);
    }
});

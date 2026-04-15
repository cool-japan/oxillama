//! Fuzz target: GgufFile::parse
//!
//! Exercises the full GGUF in-memory parser: magic validation, header
//! version negotiation (v2/v3), metadata KV pairs (including nested arrays
//! of every supported type), tensor info structs, alignment arithmetic, and
//! the tensor-data slice indexing logic via `tensor_data`.
//!
//! This is the highest-coverage target — any panic in the parser or
//! subsequent tensor-data access will be caught here.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Drive the full parse path. We keep the original slice so we can
    // exercise tensor_data access on any successfully parsed file.
    if let Ok(gguf) = oxillama_gguf::GgufFile::parse(data) {
        // Iterate every registered tensor and attempt a data-slice lookup.
        // This exercises the offset / length arithmetic and OOB detection.
        let names: Vec<String> = gguf.tensors.names().cloned().collect();
        for name in &names {
            let _ = gguf.tensor_data(data, name);
        }
    }
});

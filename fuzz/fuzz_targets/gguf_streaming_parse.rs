//! Fuzz target: `StreamingGgufParser`
//!
//! The lazy/streaming parser (`crates/oxillama-gguf/src/streaming.rs`) has
//! its own independent copies of metadata-value recursion, tensor-info
//! parsing, and offset arithmetic — none of which the `gguf_parse` /
//! `gguf_from_bytes` targets exercise, since those only drive
//! `GgufFile::parse`. This target exists specifically to reach:
//!   - `read_metadata_value`'s depth-limited array recursion (V2)
//!   - `TensorInfoIter::parse_one` / `skip_tensor_infos`'s `n_dims`
//!     validation (V4) and the `n_dims as usize * 8` multiply it guards
//!   - `tensor_data`'s `data_section_offset + info.offset` arithmetic (V3)
//!   - `into_full` / `load_tensors`'s duplicate-name rejection (V6)
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxillama_gguf::streaming::StreamingGgufParser;

fuzz_target!(|data: &[u8]| {
    if let Ok(parser) = StreamingGgufParser::new(data) {
        // Exercise lazy iteration and per-tensor data access.
        for result in parser.tensor_infos() {
            if let Ok(info) = result {
                let _ = parser.tensor_data(&info);
            }
        }

        // Exercise eager materialization paths (duplicate-name rejection).
        let _ = parser.into_full();
    }
});

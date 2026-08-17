//! Fuzz target: `reader_core::parse_gguf` over `SliceSource`
//!
//! `reader_core.rs` is the `no_std`-compatible parser generic over any
//! `Source`, with its own independent copies of every parse routine in
//! `parser.rs` (metadata recursion, string reading, tensor-info parsing).
//! None of the other fuzz targets touch this module — they all drive
//! `GgufFile::parse` (`parser.rs`) or `GgufModel::from_bytes` (`loader.rs`)
//! instead. This target exists specifically to reach:
//!   - the depth-limited array recursion in `read_metadata_value` (V2)
//!   - `read_string_v3`/`read_string_v2`'s `remaining_hint`-bounded length
//!     validation (V4) via `SliceSource`, which always reports a hint
//!   - `read_tensor_infos`'s `n_dims` validation and duplicate-name
//!     rejection via `TensorStore::try_insert` (V4, V6)
//!   - `align_up`'s overflow-safe arithmetic and `validate_alignment` (the
//!     `general.alignment` overflow finding)
//!
//! Also probes `SliceSource::seek` directly (V5) with fuzzer-supplied
//! offsets, independent of the parse path.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxillama_gguf::{parse_gguf, SliceSource, Source};

fuzz_target!(|data: &[u8]| {
    let mut src = SliceSource::new(data);
    let _ = parse_gguf(&mut src);

    // Independently probe seek() with a handful of fuzzer-derived offsets,
    // including ones far past the end of the buffer.
    if data.len() >= 8 {
        let mut probe = SliceSource::new(data);
        let raw = u64::from_le_bytes(data[..8].try_into().unwrap_or([0u8; 8]));
        let _ = probe.seek(raw);
        let _ = probe.seek(u64::MAX);
        let _ = probe.remaining_hint();
    }
});

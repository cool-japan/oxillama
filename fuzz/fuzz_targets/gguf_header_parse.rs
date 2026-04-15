//! Fuzz target: GgufHeader::parse
//!
//! Narrow target covering only the header parsing layer: magic validation,
//! version negotiation (v2/v3), and count-field width selection (u32 vs u64).
//!
//! This target is intentionally lightweight — it exits immediately after the
//! header parse rather than continuing to metadata or tensor info. That makes
//! it very fast to execute, ideal for bootstrapping a seed corpus and for
//! finding early-exit panics without the overhead of a full parse.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // GgufHeader::parse(data: &[u8], offset: u64) -> GgufResult<(GgufHeader, u64)>
    // Offset 0 is the normal starting position for a well-formed GGUF file.
    let _ = oxillama_gguf::GgufHeader::parse(data, 0);

    // Also probe from a non-zero offset to exercise the bounds-check paths
    // that guard pos + N < data.len() within read_count.
    if data.len() > 4 {
        let _ = oxillama_gguf::GgufHeader::parse(data, 4);
    }
});

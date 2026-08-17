# oxillama-gguf — TODO

## 1. Overview

`oxillama-gguf` is the Pure Rust GGUF v3 binary format parser and tensor
loader. It is the **first link** in the OxiLLaMa dependency chain —
consumed by `oxillama-quant`, `oxillama-arch`, and `oxillama-runtime` to
materialise model weights. No C, no C++, no Fortran, no FFI: a pure
hand-rolled bounds-checked binary reader + `memmap2` on stable Rust. Every downstream crate in the
workspace assumes this layer is correct and zero-copy where possible.

## 2. Status Snapshot

| Field | Value |
|---|---|
| Version | `0.1.4` (workspace-pinned) |
| Completion | ~93% (GGUF v1/v2/v3 complete + writer API + streaming parser) |
| Source files | 25 `.rs` under `src/` (11,228 LoC, tokei) |
| Format support | GGUF v1, v2, v3 (all supported) — version-dispatched layouts |
| Default feature | `mmap` (memmap2-backed zero-copy loader) |
| Optional feature | `test-utils` (stable since v0.1.1 — synthetic GGUF builders) |
| Fuzz targets | 6 under `fuzz/fuzz_targets/` (5 raw-byte + 1 arbitrary-structured) |
| Core deps | `thiserror`, `half`, `blake3`, `oxicode`, `serde_json`, `memmap2` (mmap), `ureq` (http) |
| Upstream consumers | `oxillama-quant`, `oxillama-arch`, `oxillama-runtime` |

## 3. Module Map

| File | Role |
|---|---|
| `src/lib.rs` | Crate root; re-exports the public API (`GgufFile`, `GgufModel`, `GgufHeader`, `MetadataStore`, `TensorStore`, `GgufWriter`, ...) |
| `src/error.rs` | `GgufError` + `GgufResult<T>` via `thiserror` — invalid magic, unsupported version, unexpected EOF, invalid metadata, mmap errors |
| `src/types.rs` | `GGUF_MAGIC` constant, `GgufValueType` (13 variants), `GgufTensorType` (all GGML dtype IDs incl. `Q1_0_G128`), `GGUF_DEFAULT_ALIGNMENT` |
| `src/header.rs` | `GgufHeader::parse()` — magic/version/tensor-count/KV-count validation |
| `src/reader.rs` | `BinaryReader` — bounds-checked cursor over `&[u8]` for all primitive reads |
| `src/reader_core.rs` | `no_std`-compatible parse core generic over any `Source` (`parse_gguf`, `read_header`, `read_metadata_kv`, `read_tensor_infos`, `align_up`) |
| `src/source.rs` | `Source` trait + `SliceSource` / `ReadSource` / `FileSource` — parse over a slice, a `Read + Seek`, or a file |
| `src/metadata.rs` | `MetadataStore` + `MetadataValue` — typed KV access (scalars, strings, nested arrays) |
| `src/tensor_info.rs` | `TensorInfo` + `TensorStore` — per-tensor descriptor (name, dimensions, tensor_type, offset) + registry |
| `src/parser.rs` | `GgufFile::parse()` — full-file parse (header + KV + tensor-info + alignment) without loading data |
| `src/loader.rs` | `GgufModel` — high-level handle: `load_mmap()` / `load_read()` / `load()` / `from_bytes()` entry points |
| `src/bytes.rs` | `SharedBytes` / `ByteOwner` — `Arc`-backed zero-copy byte views into mmap'd or owned tensor data |
| `src/streaming.rs` | `StreamingGgufParser` — lazy/streaming tensor parser (`TensorInfoIter`, `find_tensor`, `load_tensors`, `into_full`) |
| `src/writer.rs` | `GgufWriter` (buffered `add_tensor`/`write_to` + streaming `declare_tensor`/`into_data_writer`) and `GgufTensorDataWriter` |
| `src/resume.rs` | `ResumeCheckpoint`, `ResumeHandle`, `PrefixFingerprint` — partial-download resume via `.oxiresume` sidecar |
| `src/sharded.rs` | `ShardedGgufModel` — multi-file `model-0000N-of-0000M.gguf` sharding |
| `src/quantize_on_load.rs` | `QuantTarget`, `QuantPlan`, `GgufModel::load_with_quant_plan()` — on-the-fly quantization at load time |
| `src/schema.rs` | `SchemaValidator` trait + built-in per-architecture validators; `validate_schema()` |
| `src/integrity.rs` | `TensorHashValidator` + Blake3 manifest helpers, gated behind the `integrity`/`validate` feature |
| `src/safetensors.rs` | `SafetensorsConverter` — safetensors → `GgufModel` import bridge |
| `src/http_source.rs` | `HttpRangeSource`, `GgufModel::from_url()` — remote GGUF loading via HTTP range requests, gated behind the `http` feature |
| `src/test_utils/` | `build_minimal_llama_gguf()` + 22 multi-arch builders across `mod.rs`/`arch.rs`/`arch_ext.rs`/`arch_deepseek.rs`, `minimal_tokenizer_json()` |

## 4. Shipped in v0.1.0

### Format parsing
- Full GGUF v3 header: magic (`0x46554747`) check, version, tensor count,
  KV-pair count.
- All 13 GGUF metadata value types: `UINT8`/`INT8`/`UINT16`/`INT16`/
  `UINT32`/`INT32`/`UINT64`/`INT64`/`FLOAT32`/`FLOAT64`/`BOOL`/`STRING`
  and `ARRAY` (with nested-array support).
- Every GGML tensor dtype: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`,
  `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`..`Q8_K`, `IQ2_XXS`..`IQ4_XS`,
  `Q1_0_G128` (OxiBonsai ternary).
- Alignment-aware data-section offset computation (`general.alignment`
  override with `32`-byte default).

### Loader
- `GgufModel::load_mmap()` — zero-copy `memmap2` backend (recommended
  path, gated behind default `mmap` feature).
- `GgufModel::load_read()` / `GgufModel::from_bytes()` — read-to-memory
  fallback for WASM, sandboxes, and in-memory fixtures.
- Clean `reader` / `loader` module split, enabling future streaming
  extension without breaking the public API.

### Safety and error surface
- Zero `unwrap()` in production code — every parse path returns
  `GgufResult`.
- `BinaryReader` bounds-checks every primitive read; no
  `debug_assert!` shortcuts on the hot path.
- Structured errors: `InvalidMagic`, `UnsupportedVersion`,
  `UnexpectedEof { offset }`, `InvalidMetadata`, `MmapError`.
- All error variants implement `std::error::Error` via `thiserror`,
  making them composable with the upstream `ArchError` and
  `RuntimeError` wrappers.

### Fuzzing
- `fuzz/fuzz_targets/gguf_header_parse.rs` — header-level fuzzer
  (magic, version, count fields).
- `fuzz/fuzz_targets/gguf_parse.rs` — full `GgufFile::parse()` fuzzer
  covering KV and tensor-info stages.
- `fuzz/fuzz_targets/gguf_from_bytes.rs` — end-to-end
  `GgufModel::from_bytes()` loader fuzzer.

### GGUF writer
- GGUF v3 writer API (`GgufWriter` builder — metadata + tensor
  serialization with 32-byte alignment).
- Streaming write path (added post-v0.1.0, 2026-08):
  `GgufWriter::declare_tensor` (shape/type
  only, no data) + `into_data_writer`/`into_file_data_writer` finalize the
  header once every tensor is declared (offsets are computable up front from
  declared sizes + alignment), then `GgufTensorDataWriter::write_tensor` /
  `write_tensor_from_chunks` stream each tensor's bytes straight to the
  destination in declared order. Peak memory is O(largest single tensor)
  instead of O(model) — quantizing an 8B model no longer needs ~4.5 GB of
  RAM just to hold the output writer's buffered tensors. The original
  `add_tensor` + `write_to`/`write_to_file` (fully-buffered) API is
  unchanged and still the right choice for small fixtures/tests; the two
  APIs are not interchangeable on one `GgufWriter` instance (mixing them
  returns a typed error). `oxillama convert` migrated to the streaming path;
  `oxillama quantize` still uses the buffered path.

### Test fixtures (`test-utils` feature)
- `build_minimal_llama_gguf()` — synthetic 1-layer LLaMA GGUF binary.
- Multi-architecture builders: Qwen3, Mistral, Gemma, Phi3, Command-R,
  StarCoder (used across downstream crate tests).
- `minimal_tokenizer_json()` — matching 32-vocab BPE tokenizer JSON.

## 5. Known Gaps / Incomplete

- ~~**No true GGUF v1/v2 fallback path.**~~ ✅ Shipped. GGUF v1 header
  parsing landed with version-dispatched tensor-info layouts: `u32`
  dimensions for v1/v2 (vs `u64` for v3), `u32` tensor offset for v1
  (vs `u64` for v2/v3). Legacy HuggingFace archives now load correctly.
- ~~**No streaming parser.** Consumers must either `mmap` the full file
  or read the entire buffer into memory. No cursor-driven incremental
  API yet — blocks remote fetch and WASM chunked-load designs.~~ ✅
  Shipped: `StreamingGgufParser` with `TensorInfoIter`, `find_tensor()`,
  `load_tensors()`, and `into_full()` for lazy/streaming tensor parsing.
- [x] Loader hardening: partial-download resume + sharded multi-file + quantize-on-the-fly (shipped 2026-04-24)
  - **Goal:** Three loader blockers resolved in one slice:
    (1) partial-download resume via `.oxiresume` side-car checkpoint (bounded O(constant) probe: head/tail Blake3 + size + mtime);
    (2) sharded multi-file via `load_sharded(first_shard)` accepting the `model-00001-of-00004.gguf` HF naming convention;
    (3) quantize-on-the-fly via `load_with_quant_plan(path, plan)` streaming F16→Q4_0/Q8_0/Q4_K at load time.
  - **Design:**
    - `src/resume.rs`: `ResumeCheckpoint { file_size_expected, last_valid_offset, prefix_fingerprint: PrefixFingerprint, tensors_fully_loaded, version }` serialised via oxicode; `PrefixFingerprint { head_hash, tail_hash, probe_size=8MB, file_mtime_secs }`. API: `GgufModel::resume(path) -> GgufResult<ResumeHandle>`.
    - `src/sharded.rs`: `ShardedGgufModel` — shard-naming regex, loads each via `GgufModel::load_mmap`, validates header consistency, builds tensor→shard map. Errors: `ShardMismatch`, `ShardDuplicateTensor`.
    - `src/quantize_on_load.rs`: `QuantPlan { default: QuantTarget, overrides: HashMap<String, QuantTarget> }`. Per-tensor: read F16 → scalar-oracle quantise → owned `Vec<u8>` buffer → update `TensorStore`. Rejects re-quantisation.
  - **Files:** `src/resume.rs`, `src/sharded.rs`, `src/quantize_on_load.rs`, `src/lib.rs` (re-exports), `src/error.rs` (3 new variants), `src/loader.rs`, `Cargo.toml` (oxicode dep), `tests/resume.rs`, `tests/sharded.rs`, `tests/quantize_on_load.rs`.
  - **Tests:** `resume_roundtrip_valid_checkpoint`, `resume_rejects_hash_mismatch`, `resume_rejects_future_file_size`, `sharded_loads_two_shards_roundtrip`, `sharded_rejects_mismatched_architecture`, `sharded_rejects_duplicate_tensor`, `quantize_on_load_f16_to_q4_0`, `quantize_on_load_rejects_requantize`, `quantize_on_load_override_per_tensor`.
- ~~**`test-utils` feature stability.**~~ ✅ Resolved in v0.1.1 — builder
  signatures are now semver-stable within the 0.1.x series.
- Tensor-data hash validation shipped via `TensorHashValidator` in `src/integrity.rs` (v1.1); use `GgufModel::validate_tensor_hashes()` to verify integrity.
- Deep metadata schema validation shipped via pluggable `SchemaValidator` in `src/schema.rs` (v1.1); use `GgufModel::validate_schema()` with a custom validator.
- ~~**No `no_std` story.**~~ ✅ Resolved: `Source` trait + `SliceSource` in `src/source.rs`, `reader_core.rs` parse logic generic over any `Source`, `cfg_attr(not(feature = "std"), no_std)` in `lib.rs`, cfg-gated `HashMap`→`BTreeMap` in `metadata.rs`/`tensor_info.rs`, cfg-gated `Io` variant in `error.rs`. `cargo check --no-default-features --features alloc` green.

## 6. v1.1 Roadmap

Priority order (highest first):

1. ~~**GGUF v1/v2 version-dispatch parser.**~~ ✅ Shipped. v1 header
   parsing, `u32` dimensions for v1/v2, `u32` tensor offset for v1.
2. ~~**Lazy / streaming tensor parser.**~~ ✅ Shipped: `StreamingGgufParser`
   yields `TensorInfo` entries on demand via `TensorInfoIter`, with
   `find_tensor()`, `load_tensors()`, and `into_full()` for downstream
   crates to stream tensors during `mmap`-less loads (WASM, remote fetch).
3. ~~**GGUF writer API.**~~ ✅ Shipped — `GgufWriter` builder with
   metadata + tensor serialization and 32-byte alignment.
4. ~~**Stabilise `test-utils` feature.**~~ ✅ Done — builder signatures frozen,
   `#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]` on every public fn,
   module docs updated to v0.1.1 stability guarantee, alpha language removed.
5. ~~**Blake3 tensor-blob hash validation.**~~ ✅ Shipped. `TensorHashValidator`
   reads `general.tensor_hashes` from GGUF metadata (array of
   `"name:hexhash"` strings) and validates each tensor blob via Blake3.
   Gated behind `validate` feature (aliases `integrity`). `GgufModel::validate_tensor_hashes()`
   returns `Err(GgufError::HashMismatch)` on first mismatch.
6. ~~**Strict metadata schema check.**~~ ✅ Shipped. Pluggable `SchemaValidator`
   trait with built-in validators for LLaMA, Qwen3, Mistral, Gemma, Phi,
   Command-R, StarCoder. `validate_schema()` dispatches on `general.architecture`
   and returns a `Vec<SchemaViolation>` for callers to decide fail/warn.
7. [x] **`no_std` reader core.** — `Source` trait, `SliceSource`, `reader_core.rs` split, cfg-gated std/io in error.rs/types.rs, `cargo check --no-default-features --features alloc` green (2026-04-24)
8. ~~**Richer fuzzing.**~~ ✅ Done — `fuzz/fuzz_targets/gguf_metadata_arbitrary.rs`
   added using `arbitrary`-derived `ArbScalar`/`ArbMetadata` types that map
   directly to all `MetadataValue` variants including nested `Array` trees.
   Two more raw-byte targets landed alongside the `no_std` reader core and
   streaming parser: `gguf_reader_core_parse.rs` (fuzzes `reader_core::parse_gguf`
   over `SliceSource`, plus direct `SliceSource::seek` probing) and
   `gguf_streaming_parse.rs` (fuzzes `StreamingGgufParser` lazy tensor
   iteration, `tensor_data()`, and `into_full()`). 6 fuzz targets total.

## 7. v2.0+ Vision

- ~~**Remote GGUF streaming via HTTP range requests.**~~ ✅ Shipped (v0.1.3
  Track B). `HttpRangeSource` issues HTTP `Range:` requests, caches the
  first 128 KiB eagerly, and exposes `GgufModel::from_url(url)`. Gated
  behind `http` feature (`dep:ureq`, `native-tls`). All network-dependent
  tests carry `#[ignore]`.
- ~~**safetensors import bridge.**~~ ✅ Shipped (v0.1.3 Track B).
  `SafetensorsConverter::load(path)` / `from_bytes(bytes)` parses the
  8-byte LE header_size prefix + JSON tensor descriptors and produces an
  in-memory `GgufModel` with `general.architecture = "safetensors_import"`.
  Dtype mapping: `F32`→`F32`, `F16`→`F16`, `BF16`→`Bf16`, `I8`→`Q8_0`
  (approximate). Unsupported dtypes → `GgufError::UnsupportedDtype`.
- ~~**Quantize-on-the-fly.**~~ ✅ Shipped (2026-04-24, see §5). `QuantPlan` /
  `QuantTarget` in `src/quantize_on_load.rs`; `GgufModel::load_with_quant_plan(path, plan)`
  downconverts `F16`/`F32` weights to `Q4_0`/`Q8_0` during load.
- **Async / `tokio`-based parser.** Non-blocking loaders for the
  server crate, with back-pressure across HTTP range reads and a
  tokenised progress stream exposed to `oxillama-py` and
  `oxillama-wasm`.
- **GGUF v4 spec participation.** Track upstream spec evolution
  (`gguf-py`), land a clean dispatch layer so v4 features (e.g. richer
  sharded weights, tensor-group hints) can be adopted without breaking
  v3 consumers.
- **Tensor-level encryption / signing.** Optional AEAD per tensor blob
  for enterprise model-distribution use cases (paired with the v1.1
  hash-validation path); public-key signing for provenance.
- **Live-patched weights.** Mmap with `MAP_PRIVATE` + COW so
  `oxillama-runtime` can apply LoRA deltas in-place without evicting
  the base model from the page cache.
- ~~**Sharded / multi-file GGUF support.**~~ ✅ Shipped (2026-04-24, see §5).
  `ShardedGgufModel::load_sharded(first_shard)` in `src/sharded.rs` reads
  `model-00001-of-00004.gguf`-style HF-convention shards as one logical view.
- ~~**`no_std + alloc` profile.**~~ ✅ Shipped (2026-04-24, see §5/§6). The
  `alloc` feature builds without `std`; every `std::fs`/`std::path`-touching
  module (`loader`, `resume`, `sharded`, `writer`, `streaming`, ...) is
  `#[cfg(feature = "std")]`-gated out. `cargo check --no-default-features
  --features alloc` green (2026-08-17).

*Last updated: 2026-08-17 (v0.1.4 — zero-copy mmap weight loading via `SharedBytes`/`ByteOwner`
in `src/bytes.rs`, GGUF magic-byte regression test in `src/header.rs`; 354 tests passing)*

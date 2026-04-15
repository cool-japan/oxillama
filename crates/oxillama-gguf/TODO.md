# oxillama-gguf — TODO

## 1. Overview

`oxillama-gguf` is the Pure Rust GGUF v3 binary format parser and tensor
loader. It is the **first link** in the OxiLLaMa dependency chain —
consumed by `oxillama-quant`, `oxillama-arch`, and `oxillama-runtime` to
materialise model weights. No C, no C++, no Fortran, no FFI: pure
`byteorder` + `memmap2` on stable Rust. Every downstream crate in the
workspace assumes this layer is correct and zero-copy where possible.

## 2. Status Snapshot

| Field | Value |
|---|---|
| Version | `0.1.0` (workspace-pinned) |
| Completion | ~93% (GGUF v1/v2/v3 complete + writer API + streaming parser) |
| Source files | 11 `.rs` under `src/` (~4,500 LoC) |
| Format support | GGUF v1, v2, v3 (all supported) — version-dispatched layouts |
| Default feature | `mmap` (memmap2-backed zero-copy loader) |
| Optional feature | `test-utils` (alpha — synthetic GGUF builders) |
| Fuzz targets | 3 under `fuzz/fuzz_targets/` |
| Core deps | `thiserror`, `byteorder`, `half`, `memmap2`, `tracing` |
| Upstream consumers | `oxillama-quant`, `oxillama-arch`, `oxillama-runtime` |

## 3. Module Map

| File | Role |
|---|---|
| `src/lib.rs` | Crate root; re-exports `GgufFile`, `GgufModel`, `GgufHeader`, `MetadataStore`, `TensorStore` |
| `src/error.rs` | `GgufError` + `GgufResult<T>` via `thiserror` — invalid magic, unsupported version, unexpected EOF, mmap errors |
| `src/types.rs` | `GGUF_MAGIC` constant, `GgufValueType` (13 variants), `GgufTensorType` (all GGML dtype IDs), `GGUF_DEFAULT_ALIGNMENT` |
| `src/header.rs` | `GgufHeader::parse()` — magic/version/tensor-count/KV-count validation |
| `src/reader.rs` | `BinaryReader` — bounds-checked cursor over `&[u8]` for all primitive reads |
| `src/metadata.rs` | `MetadataStore` + `MetadataValue` — typed KV access (scalars, strings, nested arrays) |
| `src/tensor_info.rs` | `TensorInfo` + `TensorStore` — per-tensor descriptor (name, shape, dtype, offset) + registry |
| `src/parser.rs` | `GgufFile::parse()` — full-file parse (header + KV + tensor-info + alignment) without loading data |
| `src/loader.rs` | `GgufModel` — high-level handle with `load_mmap()` / `load_owned()` / `from_bytes()` entry points |
| `src/streaming.rs` | `StreamingGgufParser` — lazy/streaming tensor parser (`TensorInfoIter`, `find_tensor`, `load_tensors`, `into_full`) |
| `src/test_utils.rs` | `build_minimal_llama_gguf()` + multi-arch builders (Qwen3, Mistral, Gemma, Phi3, Command-R, StarCoder), `minimal_tokenizer_json()` |

## 4. Shipped in v0.1.0

### Format parsing
- Full GGUF v3 header: magic (`0x46475547`) check, version, tensor count,
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
- `GgufModel::load_owned()` / `GgufModel::from_bytes()` — read-to-memory
  fallback for WASM, sandboxes, and in-memory fixtures.
- Clean `reader` / `loader` module split, enabling future streaming
  extension without breaking the public API.

### Safety and error surface
- Zero `unwrap()` in production code — every parse path returns
  `GgufResult`.
- `BinaryReader` bounds-checks every primitive read; no
  `debug_assert!` shortcuts on the hot path.
- Structured errors: `InvalidMagic`, `UnsupportedVersion`,
  `UnexpectedEof { offset }`, `InvalidValueType`, `MmapError`.
- All error variants implement `std::error::Error` via `thiserror`,
  making them composable with the upstream `ArchError` and
  `RuntimeError` wrappers.
- Tracing spans via the `tracing` crate at every top-level
  parse / load entry point for structured diagnostics.

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

### Test fixtures (`test-utils` feature, alpha)
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
- **No partial-download resume.** A truncated GGUF raises
  `UnexpectedEof`; there is no checkpoint/restart facility for flaky
  connections or interrupted pulls from HuggingFace Hub mirrors.
- **`test-utils` feature stability.** API is marked alpha — builder
  signatures may change as new architectures are added. Downstream
  crates pin to workspace-local paths rather than semver.
- **No tensor-data hash validation.** GGUF v3 does not mandate it, but
  tensor-blob integrity checks (Blake3 or similar) are absent, leaving
  silent on-disk corruption undetected.
- **No deep metadata schema validation.** Malformed-but-parseable KV
  entries (missing `*.block_count`, out-of-range `*.head_count_kv`)
  propagate unchecked to `oxillama-arch`.
- **No `no_std` story.** `std::fs` and `std::path::Path` are used in
  `loader.rs`; embedded targets cannot yet reuse the reader core.

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
4. **Stabilise `test-utils` feature.** Freeze builder signatures, move
   to `#[doc(cfg(feature = "test-utils"))]`, promote from alpha to a
   semver-covered surface, document every synthetic builder.
5. **Blake3 tensor-blob hash validation.** Optional verification pass
   gated behind a `validate` feature; read expected hash from a new
   `general.tensor_hashes` KV if present. Hard-fail on mismatch,
   warn-log if absent.
6. **Strict metadata schema check.** Pluggable validator (`llama.*`,
   `qwen3.*`, `command_r.*`, ...) producing a single `ArchError`-ready
   diagnostic before reaching `oxillama-arch`.
7. **`no_std` reader core.** Split reader/types into a `no_std + alloc`
   sub-module to unlock embedded-target plans in v2.0.
8. **Richer fuzzing.** Add a structured-input fuzzer (arbitrary-derived
   `MetadataValue::Array` trees) on top of the three raw-byte fuzzers.

## 7. v2.0+ Vision

- **Remote GGUF streaming via HTTP range requests.** Fetch only the
  tensors a given architecture needs — header + KV first, then tensor
  offsets on demand. Major latency win for 30B+ models pulled from
  HuggingFace mirrors. Pairs with the v1.1 lazy parser.
- **safetensors import bridge.** Accept `.safetensors` on load, convert
  to an in-memory GGUF view; covers the remainder of the HuggingFace
  hub that has not been re-published in GGUF form.
- **Quantize-on-the-fly.** During parse, downcast `F16`/`BF16` weights
  to `Q4_0` / `Q8_0` via `oxillama-quant` kernels so that unquantised
  reference weights fit in consumer RAM without an explicit
  pre-conversion step.
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
- **Sharded / multi-file GGUF support.** Read `model.gguf` +
  `model-00002-of-00004.gguf` shards as a single logical view,
  mirroring HuggingFace `*.safetensors` sharding conventions.
- **`no_std + alloc` profile.** Fully embedded-target-capable reader
  and parser (no `std::fs`, no `std::path`) for on-device LLM shells.

*Last updated: 2026-04-15 (v0.1.0 release)*

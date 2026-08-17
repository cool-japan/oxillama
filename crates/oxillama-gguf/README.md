# oxillama-gguf

GGUF v3 file format parser and tensor loader for Rust.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

**354 tests passing** (unit + integration, as of v0.1.4; 4 additional `#[ignore]`-marked slow/memory tests run manually)

## What It Provides

- Full GGUF v3 (and v1/v2 backward-compatible) file parsing
- Metadata key-value store reader (`MetadataStore`)
- Tensor descriptor catalogue (`TensorStore`) with offset/shape/dtype info
- Optional memory-mapped loading via the `mmap` feature (enabled by default)
- Zero-copy tensor data access backed by `memmap2`
- **`resume`** — partial-download resume with O(constant) head+tail Blake3 hash probe (`ResumeCheckpoint`, `ResumeHandle`, `PrefixFingerprint`)
- **`sharded`** — multi-file `model-00001-of-00004.gguf` sharding (`ShardedGgufModel`)
- **`quantize_on_load`** — on-the-fly F16/F32→Q4_0/Q8_0 quantization at load time (`QuantTarget`, `QuantPlan`, `GgufModel::load_with_quant_plan`)
- **Streaming writer** — `GgufWriter::declare_tensor` + `into_data_writer`/`into_file_data_writer` stream tensor data straight to disk in declared order, so peak memory is bounded by the largest single tensor rather than the whole model; large tensors can be streamed in chunks via `GgufTensorDataWriter::write_tensor_from_chunks` without ever materializing the whole tensor. The original fully-buffered `add_tensor` + `write_to`/`write_to_file` API is unchanged and still available for small fixtures/tests.

## Integration Tests

The `tests/` directory contains integration tests covering the three new v0.1.2 modules:

| Test file | Coverage |
|-----------|----------|
| `tests/resume.rs` | `ResumeCheckpoint` round-trip, fingerprint mismatch detection |
| `tests/sharded.rs` | Multi-shard load, architecture consistency checks, duplicate-tensor rejection |
| `tests/quantize_on_load.rs` | F16→Q4_0, F16→Q8_0, F32→Q4_0, re-quantization rejection |

## What's New in v0.1.4 (2026-08-17)

- **Zero-copy mmap weight loading** — `SharedBytes`/`ByteOwner` (`src/bytes.rs`) let `QuantTensor` hold an `Arc`-backed view straight into the mmap instead of a private `Vec<u8>` copy; `GgufModel::tensor_bytes` replaces `.to_vec()` on the hot load path. Measured 4.2 GB → 148 MB peak footprint and 0.6 s → 14 ms weight-load time on a Qwen3-4B Q4_K_M checkpoint (downstream, in `oxillama-arch`/`oxillama-quant`).
- **GGUF magic-byte regression test** (`src/header.rs`) — explicit three-case test validating `GGUF_MAGIC = 0x4655_4747` byte-order correctness, including the transposed-nibble constant that a real GGUF parser bug once produced.
- Test count unchanged at 354 (4 additional `#[ignore]`-marked tests unaffected).

## What's New in v0.1.3 (2026-05-05)

- **BLOOM GGUF test fixture** — `build_minimal_bloom_gguf()` helper added to `test_utils.rs`; 1-layer, hidden=64, 8 heads; used by `oxillama-arch` BLOOM integration tests.
- **Phi-3.5-MoE GGUF test fixture** — `build_minimal_phi_moe_gguf()` helper; 1-layer, 4 experts, top-2; used by Phi-3.5-MoE integration tests.
- Test count: 278 → **290 tests passing**.

## What's New in v0.1.2

- Broken intra-doc links fixed for `SliceSource` reference in `reader_core.rs`

## Key Types

| Type | Description |
|------|-------------|
| `GgufFile` | Top-level handle; owns header, metadata, and tensor catalogue |
| `GgufHeader` | Version, tensor count, and KV pair count (magic is validated by `parse()` but not retained) |
| `MetadataStore` | Typed KV access: strings, integers, floats, arrays |
| `TensorStore` | Iterate or look up tensors by name; returns `TensorInfo` |
| `TensorInfo` | Name, shape, element type, and byte offset within the file |
| `GgufWriter` | GGUF v3 writer: metadata + tensor declaration with 32-byte alignment; buffered (`add_tensor`/`write_to`) or streaming (`declare_tensor`/`into_data_writer`) |
| `GgufTensorDataWriter` | Streams declared tensors' bytes to disk in order, O(largest tensor) peak memory; `write_tensor` (one call) or `write_tensor_from_chunks` (multi-GB tensors without full materialization) |
| `SchemaValidator` | Pluggable schema validator dispatching on `general.architecture` |
| `StreamingGgufParser` | Lazy/streaming tensor parser with `find_tensor`, `load_tensors`, `into_full` |
| `ResumeCheckpoint` | Persistent sidecar recording expected file size, mtime, and Blake3 fingerprint for resume |
| `ResumeHandle` | Active resume session returned by `GgufModel::resume`; call `.finish()` to complete |
| `PrefixFingerprint` | O(constant) head+tail Blake3 hash probe (default 8 MiB each end) |
| `ShardedGgufModel` | Unified view over `model-00001-of-00004.gguf` style multi-file shards |
| `QuantTarget` | Target quantization format for on-load conversion (`Q4_0` or `Q8_0`) |
| `QuantPlan` | Per-tensor mapping from tensor name patterns to `QuantTarget` |

## Usage

```rust
use oxillama_gguf::{GgufModel, GgufResult};

fn main() -> GgufResult<()> {
    let model = GgufModel::load("model.gguf")?;

    // Read metadata
    let arch: &str = model.file.metadata.get_string("general.architecture")?;
    println!("Architecture: {arch}");

    // Iterate tensors
    for (name, info) in model.file.tensors.iter() {
        println!(
            "tensor: {name}  shape: {:?}  dtype: {:?}",
            info.dimensions,
            info.tensor_type,
        );
    }

    Ok(())
}
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Standard-library support (file I/O, `HashMap`-backed stores) |
| `mmap` | yes | Memory-map tensor data via `memmap2` |
| `alloc` | no | `no_std` + `alloc` profile for embedded/WASM targets without `std` |
| `test-utils` | no | Expose helpers for downstream crate tests |
| `integrity` | no | Blake3 tensor-blob hash validation (`TensorHashValidator`) |
| `validate` | no | Alias for `integrity`; enables schema + hash validation paths |
| `http` | no | Remote GGUF loading via HTTP range requests (`HttpRangeSource`, `GgufModel::from_url`) |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

# oxillama-gguf

GGUF v3 file format parser and tensor loader for Rust.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- Full GGUF v3 (and v1/v2 backward-compatible) file parsing
- Metadata key-value store reader (`MetadataStore`)
- Tensor descriptor catalogue (`TensorStore`) with offset/shape/dtype info
- Optional memory-mapped loading via the `mmap` feature (enabled by default)
- Zero-copy tensor data access backed by `memmap2`

## Key Types

| Type | Description |
|------|-------------|
| `GgufFile` | Top-level handle; owns header, metadata, and tensor catalogue |
| `GgufHeader` | File magic, version, tensor count, and KV pair count |
| `MetadataStore` | Typed KV access: strings, integers, floats, arrays |
| `TensorStore` | Iterate or look up tensors by name; returns `TensorInfo` |
| `TensorInfo` | Name, shape, element type, and byte offset within the file |

## Usage

```rust
use oxillama_gguf::{GgufFile, GgufResult};

fn main() -> GgufResult<()> {
    let gguf = GgufFile::open("model.gguf")?;

    // Read metadata
    let arch: &str = gguf.metadata().get_str("general.architecture")?;
    println!("Architecture: {arch}");

    // Iterate tensors
    for info in gguf.tensors().iter() {
        println!(
            "tensor: {:?}  shape: {:?}  dtype: {:?}",
            info.name(),
            info.shape(),
            info.ggml_type(),
        );
    }

    Ok(())
}
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `mmap` | yes | Memory-map tensor data via `memmap2` |
| `test-utils` | no | Expose helpers for downstream crate tests |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

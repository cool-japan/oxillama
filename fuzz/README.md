# OxiLLaMa GGUF Fuzz Targets

This directory contains [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) harnesses
for the `oxillama-gguf` parser. Three targets are provided:

| Target | Entry point | Focus |
|---|---|---|
| `gguf_parse` | `GgufFile::parse` | Full parse: header, KV metadata, tensor info, data slicing |
| `gguf_header_parse` | `GgufHeader::parse` | Header only: magic, version, count fields |
| `gguf_from_bytes` | `GgufModel::from_bytes` | High-level model wrapper: parse + tensor/metadata access |

## Prerequisites

Nightly Rust is required to run the fuzzer (libfuzzer-sys links against LLVM libFuzzer).
`cargo check` on the fuzz crate itself also requires nightly because `libfuzzer-sys`
uses unstable features.

```bash
rustup install nightly
cargo install cargo-fuzz
```

## Running

```bash
# From the workspace root, run the primary full-parser target:
cargo +nightly fuzz run gguf_parse

# Run the lightweight header-only target (good for bootstrapping a seed corpus):
cargo +nightly fuzz run gguf_header_parse

# Run the high-level model wrapper target:
cargo +nightly fuzz run gguf_from_bytes
```

## Running with a Time Limit

```bash
# Run gguf_parse for one hour:
cargo +nightly fuzz run gguf_parse -- -max_total_time=3600

# Run with more parallel jobs:
cargo +nightly fuzz run gguf_parse -- -max_total_time=3600 -jobs=4
```

## Providing a Seed Corpus

Put representative `.gguf` files in a `corpus/gguf_parse/` directory:

```bash
mkdir -p fuzz/corpus/gguf_parse
cp /path/to/some-model.gguf fuzz/corpus/gguf_parse/
cargo +nightly fuzz run gguf_parse fuzz/corpus/gguf_parse
```

cargo-fuzz will minimise and deduplicate the corpus automatically.

## Triaging Crashes

Crash inputs are written to `artifacts/<target>/`. To reproduce and triage:

```bash
# Reproduce a specific crash:
cargo +nightly fuzz run gguf_parse fuzz/artifacts/gguf_parse/crash-<hash>

# Triage all artifacts for a target:
cargo +nightly fuzz triage fuzz/artifacts/gguf_parse/
```

## Workspace Exclusion

This directory is deliberately NOT a member of the parent Cargo workspace.
cargo-fuzz invokes it independently via `cargo +nightly fuzz run` from the
workspace root. Do not add `fuzz/` to the workspace `members` list.

//! Integration tests for the sharded GGUF loader.

use oxillama_gguf::{GgufError, GgufTensorType, GgufValueType, ShardedGgufModel};
use std::path::{Path, PathBuf};

fn build_shard_gguf(arch: &str, tensors: &[(&str, u64, GgufTensorType)]) -> Vec<u8> {
    use oxillama_gguf::types::GGUF_MAGIC;

    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV

    let key = b"general.architecture";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    let arch_b = arch.as_bytes();
    data.extend_from_slice(&(arch_b.len() as u64).to_le_bytes());
    data.extend_from_slice(arch_b);

    let mut offset: u64 = 0;
    let mut sizes: Vec<u64> = Vec::new();
    for &(name, dim0, ttype) in tensors {
        let nb = name.as_bytes();
        data.extend_from_slice(&(nb.len() as u64).to_le_bytes());
        data.extend_from_slice(nb);
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&dim0.to_le_bytes());
        data.extend_from_slice(&(ttype as u32).to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());

        let bs = ttype.block_size() as u64;
        let bb = ttype.block_bytes() as u64;
        let size = dim0.div_ceil(bs) * bb;
        sizes.push(size);
        offset += size;
    }

    // Pad to 32-byte alignment
    let rem = data.len() % 32;
    if rem != 0 {
        data.resize(data.len() + 32 - rem, 0u8);
    }

    for sz in sizes {
        data.resize(data.len() + sz as usize, 0xBEu8);
    }

    data
}

type ShardConfig<'a> = (&'a str, Vec<(&'a str, u64, GgufTensorType)>);

fn write_shards(dir: &Path, base: &str, configs: &[ShardConfig<'_>]) -> Vec<PathBuf> {
    let total = configs.len() as u32;
    configs
        .iter()
        .enumerate()
        .map(|(i, (arch, tensors))| {
            let path = dir.join(format!("{base}-{:05}-of-{:05}.gguf", i + 1, total));
            let bytes = build_shard_gguf(arch, tensors);
            std::fs::write(&path, &bytes).expect("test: write shard");
            path
        })
        .collect()
}

#[test]
fn sharded_loads_two_shards_roundtrip() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");

    let paths = write_shards(
        dir.path(),
        "model",
        &[
            (
                "llama",
                vec![("blk.0.attn_q.weight", 32, GgufTensorType::F32)],
            ),
            (
                "llama",
                vec![("blk.1.attn_q.weight", 32, GgufTensorType::F32)],
            ),
        ],
    );

    let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");

    assert_eq!(model.architecture(), "llama");
    assert_eq!(model.shard_count(), 2);
    assert_eq!(model.tensor_count(), 2);
    assert!(model.contains_tensor("blk.0.attn_q.weight"));
    assert!(model.contains_tensor("blk.1.attn_q.weight"));
    assert!(!model.contains_tensor("nonexistent.weight"));
}

#[test]
fn sharded_rejects_mismatched_architecture() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");

    let paths = write_shards(
        dir.path(),
        "badarch",
        &[
            ("llama", vec![("w.0", 32, GgufTensorType::F32)]),
            ("mistral", vec![("w.1", 32, GgufTensorType::F32)]),
        ],
    );

    let result = ShardedGgufModel::load_sharded(&paths[0]);
    assert!(
        matches!(result, Err(GgufError::ShardMismatch { .. })),
        "mismatched architectures should return ShardMismatch, got: {result:?}"
    );
}

#[test]
fn sharded_rejects_duplicate_tensor() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");

    let tensors_both = vec![("shared.weight", 32u64, GgufTensorType::F32)];

    let paths = write_shards(
        dir.path(),
        "dup",
        &[("llama", tensors_both.clone()), ("llama", tensors_both)],
    );

    let result = ShardedGgufModel::load_sharded(&paths[0]);
    assert!(
        matches!(result, Err(GgufError::ShardDuplicateTensor { .. })),
        "duplicate tensor should return ShardDuplicateTensor, got: {result:?}"
    );
}

#[test]
fn sharded_rejects_non_shard_filename() {
    let result = ShardedGgufModel::load_sharded("/tmp/model.gguf");
    assert!(
        matches!(result, Err(GgufError::ShardMismatch { .. })),
        "non-shard filename should return ShardMismatch, got: {result:?}"
    );
}

#[test]
fn sharded_tensor_data_from_correct_shard() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");

    let paths = write_shards(
        dir.path(),
        "datatest",
        &[
            ("llama", vec![("embed.weight", 64, GgufTensorType::F32)]),
            ("llama", vec![("output.weight", 64, GgufTensorType::F32)]),
        ],
    );

    let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");

    let embed = model.tensor_data("embed.weight").expect("test: embed data");
    assert!(!embed.is_empty());

    let output = model
        .tensor_data("output.weight")
        .expect("test: output data");
    assert!(!output.is_empty());
}

#[test]
fn sharded_tensor_data_missing_returns_error() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");

    let paths = write_shards(
        dir.path(),
        "errtest",
        &[("llama", vec![("embed.weight", 32, GgufTensorType::F32)])],
    );

    let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");
    let result = model.tensor_data("does.not.exist");
    assert!(
        matches!(result, Err(GgufError::TensorNotFound { .. })),
        "missing tensor should return TensorNotFound"
    );
}

#[test]
fn sharded_tensor_info_returns_correct_dims() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");

    let paths = write_shards(
        dir.path(),
        "infotest",
        &[
            ("llama", vec![("blk.0.w", 64, GgufTensorType::F32)]),
            ("llama", vec![("blk.1.w", 128, GgufTensorType::Q8_0)]),
        ],
    );

    let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");

    let info0 = model.tensor_info("blk.0.w").expect("test: info0");
    assert_eq!(info0.dimensions[0], 64);
    assert_eq!(info0.tensor_type, GgufTensorType::F32);

    let info1 = model.tensor_info("blk.1.w").expect("test: info1");
    assert_eq!(info1.dimensions[0], 128);
    assert_eq!(info1.tensor_type, GgufTensorType::Q8_0);
}

#[test]
fn sharded_tensor_names_yields_all() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");

    let paths = write_shards(
        dir.path(),
        "nametest",
        &[
            ("llama", vec![("alpha", 32, GgufTensorType::F32)]),
            ("llama", vec![("beta", 32, GgufTensorType::F32)]),
            ("llama", vec![("gamma", 32, GgufTensorType::F32)]),
        ],
    );

    let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");

    let names: std::collections::HashSet<&str> = model.tensor_names().collect();
    assert!(names.contains("alpha"));
    assert!(names.contains("beta"));
    assert!(names.contains("gamma"));
    assert_eq!(names.len(), 3);
}

#[test]
fn sharded_four_shards_all_tensors_visible() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");

    let paths = write_shards(
        dir.path(),
        "four",
        &[
            ("llama", vec![("blk.0.w", 32, GgufTensorType::F32)]),
            ("llama", vec![("blk.1.w", 32, GgufTensorType::F32)]),
            ("llama", vec![("blk.2.w", 32, GgufTensorType::F32)]),
            ("llama", vec![("blk.3.w", 32, GgufTensorType::F32)]),
        ],
    );

    let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");
    assert_eq!(model.shard_count(), 4);
    assert_eq!(model.tensor_count(), 4);

    for i in 0..4 {
        let name = format!("blk.{i}.w");
        assert!(
            model.contains_tensor(&name),
            "tensor {name} should be visible"
        );
    }
}

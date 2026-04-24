//! Sharded GGUF model support.
//!
//! Loads multiple GGUF shard files (e.g., from HuggingFace multi-part
//! uploads) as a single unified logical model.
//!
//! ## Naming convention
//!
//! Shards must follow the HuggingFace naming convention:
//!
//! ```text
//! <base>-00001-of-00004.gguf
//! <base>-00002-of-00004.gguf
//! <base>-00003-of-00004.gguf
//! <base>-00004-of-00004.gguf
//! ```
//!
//! The first shard path is passed to [`ShardedGgufModel::load_sharded`].
//! All remaining shards are auto-discovered in the same directory by
//! parsing the shard index/count from the filename.
//!
//! ## Architecture consistency
//!
//! All shards must share the same `general.architecture` value; otherwise
//! [`GgufError::ShardMismatch`] is returned.  Duplicate tensor names across
//! shards trigger [`GgufError::ShardDuplicateTensor`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{GgufError, GgufResult};
use crate::loader::GgufModel;
use crate::tensor_info::TensorInfo;

/// Parsed shard filename components.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ShardId {
    /// Directory containing the shard.
    dir: PathBuf,
    /// Filename prefix before the shard index portion.
    base: String,
    /// 1-based shard index (e.g., 1 for `00001`).
    index: u32,
    /// Total number of shards (e.g., 4 for `of-00004`).
    total: u32,
}

impl ShardId {
    /// Construct the full path for a shard with the given 1-based index.
    fn path_for_index(&self, index: u32) -> PathBuf {
        self.dir.join(format!(
            "{}-{:05}-of-{:05}.gguf",
            self.base, index, self.total
        ))
    }
}

/// Parse the shard name components from a GGUF filename.
///
/// Expected suffix pattern: `-NNNNN-of-MMMMM.gguf`
/// Returns `None` if the filename does not match the pattern.
fn parse_shard_id(path: &Path) -> Option<ShardId> {
    let dir = path.parent()?.to_path_buf();
    let stem = path.file_stem()?.to_str()?;

    // Expect the stem to end with "-NNNNN-of-MMMMM"
    // Strategy: scan backwards through '-' separated parts.
    // The last three tokens when split by '-' should be: total_digits, "of", index_digits.
    // Everything before is the base.
    let parts: Vec<&str> = stem.split('-').collect();
    if parts.len() < 4 {
        return None;
    }

    // Last part: total (e.g., "00004")
    let total_str = *parts.last()?;
    // Second to last: "of"
    let of_str = parts.get(parts.len() - 2)?;
    // Third to last: index (e.g., "00001")
    let index_str = parts.get(parts.len() - 3)?;

    if *of_str != "of" {
        return None;
    }

    let total: u32 = total_str.parse().ok()?;
    let index: u32 = index_str.parse().ok()?;

    if total == 0 || index == 0 || index > total {
        return None;
    }

    // Base is everything except the last three components joined by '-'
    let base_parts = &parts[..parts.len() - 3];
    let base = base_parts.join("-");

    Some(ShardId {
        dir,
        base,
        index,
        total,
    })
}

/// A descriptor mapping a tensor to the shard it lives in.
#[derive(Debug)]
struct ShardTensorRef {
    /// 0-based shard index into [`ShardedGgufModel::shards`].
    shard_index: usize,
    /// Tensor info from the shard's parsed GGUF.
    info: TensorInfo,
}

/// A loaded multi-shard GGUF model presenting a unified tensor view.
///
/// Construct via [`ShardedGgufModel::load_sharded`].
pub struct ShardedGgufModel {
    /// Individual shard models, in order from shard 1 to shard N.
    shards: Vec<GgufModel>,
    /// Combined tensor registry: tensor name → (shard index, TensorInfo).
    tensor_map: HashMap<String, ShardTensorRef>,
    /// Architecture string (shared across all shards).
    architecture: String,
    /// Total number of tensors across all shards.
    tensor_count: usize,
}

impl std::fmt::Debug for ShardedGgufModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedGgufModel")
            .field("architecture", &self.architecture)
            .field("shard_count", &self.shards.len())
            .field("tensor_count", &self.tensor_count)
            .finish()
    }
}

impl ShardedGgufModel {
    /// Load a sharded GGUF model by providing the path to any one shard.
    ///
    /// All sibling shards are discovered automatically in the same directory
    /// using the HuggingFace naming convention.  All shards must exist and be
    /// readable.
    ///
    /// # Errors
    ///
    /// - [`GgufError::ShardMismatch`] if the first shard's filename does not
    ///   match the expected pattern or if architectures differ across shards.
    /// - [`GgufError::ShardDuplicateTensor`] if any tensor name appears in
    ///   more than one shard.
    /// - Any IO / parse errors from the underlying [`GgufModel`] loader.
    pub fn load_sharded(first_shard: impl AsRef<Path>) -> GgufResult<Self> {
        let first_path = first_shard.as_ref();

        let shard_id = parse_shard_id(first_path).ok_or_else(|| GgufError::ShardMismatch {
            detail: format!(
                "path '{}' does not match shard naming convention '<base>-NNNNN-of-MMMMM.gguf'",
                first_path.display()
            ),
        })?;

        // Build all shard paths in order
        let paths: Vec<PathBuf> = (1..=shard_id.total)
            .map(|i| shard_id.path_for_index(i))
            .collect();

        // Load all shards
        let mut shards: Vec<GgufModel> = Vec::with_capacity(shard_id.total as usize);
        for path in &paths {
            let model = GgufModel::load(path)?;
            shards.push(model);
        }

        // Validate architecture consistency and build tensor map
        let reference_arch = shards[0]
            .architecture()
            .map_err(|_| GgufError::ShardMismatch {
                detail: "shard 1 missing general.architecture metadata".to_string(),
            })?
            .to_string();

        let mut tensor_map: HashMap<String, ShardTensorRef> = HashMap::new();

        for (shard_index, model) in shards.iter().enumerate() {
            // Check architecture consistency (skip first shard — that's the reference)
            if shard_index > 0 {
                let arch = model.architecture().map_err(|_| GgufError::ShardMismatch {
                    detail: format!(
                        "shard {} missing general.architecture metadata",
                        shard_index + 1
                    ),
                })?;

                if arch != reference_arch {
                    return Err(GgufError::ShardMismatch {
                        detail: format!(
                            "shard {} architecture '{}' != shard 1 architecture '{}'",
                            shard_index + 1,
                            arch,
                            reference_arch
                        ),
                    });
                }
            }

            // Register all tensors from this shard
            for (name, info) in model.file.tensors.iter() {
                if tensor_map.contains_key(name) {
                    return Err(GgufError::ShardDuplicateTensor { name: name.clone() });
                }
                tensor_map.insert(
                    name.clone(),
                    ShardTensorRef {
                        shard_index,
                        info: info.clone(),
                    },
                );
            }
        }

        let tensor_count = tensor_map.len();

        Ok(Self {
            shards,
            tensor_map,
            architecture: reference_arch,
            tensor_count,
        })
    }

    /// Get the model architecture string.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Get the total number of tensors across all shards.
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Get the number of shard files.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Get raw tensor data bytes for a named tensor.
    ///
    /// Looks up which shard contains the tensor and delegates to that shard's
    /// [`GgufModel::tensor_data`].
    pub fn tensor_data(&self, name: &str) -> GgufResult<&[u8]> {
        let tensor_ref = self
            .tensor_map
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound {
                name: name.to_string(),
            })?;

        self.shards[tensor_ref.shard_index].tensor_data(name)
    }

    /// Check whether a tensor is present in any shard.
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensor_map.contains_key(name)
    }

    /// List all tensor names across all shards, in arbitrary order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensor_map.keys().map(|s| s.as_str())
    }

    /// Get the [`TensorInfo`] for a named tensor.
    pub fn tensor_info(&self, name: &str) -> GgufResult<&TensorInfo> {
        let tensor_ref = self
            .tensor_map
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound {
                name: name.to_string(),
            })?;
        Ok(&tensor_ref.info)
    }

    /// Get a reference to the individual shard models.
    pub fn shards(&self) -> &[GgufModel] {
        &self.shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GgufTensorType, GgufValueType, GGUF_MAGIC};

    /// Build a GGUF shard binary with the given architecture and tensors.
    ///
    /// `tensors`: list of `(name, dim0, tensor_type)` tuples.
    fn build_shard_gguf(arch: &str, tensors: &[(&str, u64, GgufTensorType)]) -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV

        // KV: general.architecture
        let key = b"general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        let arch_bytes = arch.as_bytes();
        data.extend_from_slice(&(arch_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(arch_bytes);

        // Tensor infos
        let mut current_offset: u64 = 0;
        let mut tensor_sizes: Vec<u64> = Vec::new();
        for &(name, dim0, ttype) in tensors {
            let name_bytes = name.as_bytes();
            data.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            data.extend_from_slice(name_bytes);
            data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
            data.extend_from_slice(&dim0.to_le_bytes());
            data.extend_from_slice(&(ttype as u32).to_le_bytes());
            data.extend_from_slice(&current_offset.to_le_bytes());

            let block_size = ttype.block_size() as u64;
            let block_bytes = ttype.block_bytes() as u64;
            let n_blocks = dim0.div_ceil(block_size);
            let size = n_blocks * block_bytes;
            tensor_sizes.push(size);
            current_offset += size;
        }

        // Pad to 32-byte alignment
        let align = 32usize;
        let rem = data.len() % align;
        if rem != 0 {
            data.resize(data.len() + align - rem, 0u8);
        }

        // Fake tensor data
        for size in tensor_sizes {
            data.resize(data.len() + size as usize, 0xAB);
        }

        data
    }

    type ShardSpec<'a> = (&'a str, &'a [(&'a str, u64, GgufTensorType)]);

    fn write_shard_files<'a>(
        dir: &Path,
        base: &str,
        total: u32,
        shards: &[ShardSpec<'a>],
    ) -> Vec<PathBuf> {
        shards
            .iter()
            .enumerate()
            .map(|(i, (arch, tensors))| {
                let path = dir.join(format!("{base}-{:05}-of-{:05}.gguf", i + 1, total));
                let data = build_shard_gguf(arch, tensors);
                std::fs::write(&path, &data).expect("test: write shard");
                path
            })
            .collect()
    }

    #[test]
    fn parse_shard_id_valid() {
        let path = Path::new("/models/llama3-00001-of-00004.gguf");
        let id = parse_shard_id(path).expect("should parse");
        assert_eq!(id.index, 1);
        assert_eq!(id.total, 4);
        assert_eq!(id.base, "llama3");
    }

    #[test]
    fn parse_shard_id_hyphenated_base() {
        let path = Path::new("/models/llama-3-8b-00002-of-00003.gguf");
        let id = parse_shard_id(path).expect("should parse hyphenated base");
        assert_eq!(id.index, 2);
        assert_eq!(id.total, 3);
        assert_eq!(id.base, "llama-3-8b");
    }

    #[test]
    fn parse_shard_id_invalid_no_pattern() {
        let path = Path::new("/models/model.gguf");
        assert!(
            parse_shard_id(path).is_none(),
            "non-shard path should return None"
        );
    }

    #[test]
    fn parse_shard_id_path_for_index() {
        let path = Path::new("/models/llama3-00001-of-00004.gguf");
        let id = parse_shard_id(path).expect("should parse");
        let p2 = id.path_for_index(2);
        assert_eq!(p2, PathBuf::from("/models/llama3-00002-of-00004.gguf"));
    }

    #[test]
    fn sharded_loads_two_shards_roundtrip() {
        let dir = tempfile::TempDir::new().expect("test: tempdir");
        let tensors_s1: Vec<(&str, u64, GgufTensorType)> =
            vec![("blk.0.attn_q.weight", 32, GgufTensorType::F32)];
        let tensors_s2: Vec<(&str, u64, GgufTensorType)> =
            vec![("blk.1.attn_q.weight", 32, GgufTensorType::F32)];

        let paths = write_shard_files(
            dir.path(),
            "llama3",
            2,
            &[("llama", &tensors_s1), ("llama", &tensors_s2)],
        );

        let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");

        assert_eq!(model.architecture(), "llama");
        assert_eq!(model.shard_count(), 2);
        assert_eq!(model.tensor_count(), 2);
        assert!(model.contains_tensor("blk.0.attn_q.weight"));
        assert!(model.contains_tensor("blk.1.attn_q.weight"));
    }

    #[test]
    fn sharded_rejects_mismatched_architecture() {
        let dir = tempfile::TempDir::new().expect("test: tempdir");
        let tensors_s1: Vec<(&str, u64, GgufTensorType)> =
            vec![("blk.0.attn_q.weight", 32, GgufTensorType::F32)];
        let tensors_s2: Vec<(&str, u64, GgufTensorType)> =
            vec![("blk.1.attn_q.weight", 32, GgufTensorType::F32)];

        let paths = write_shard_files(
            dir.path(),
            "mixed",
            2,
            &[
                ("llama", &tensors_s1),
                ("mistral", &tensors_s2), // different arch!
            ],
        );

        let result = ShardedGgufModel::load_sharded(&paths[0]);
        assert!(
            matches!(result, Err(GgufError::ShardMismatch { .. })),
            "mismatched architectures should return ShardMismatch"
        );
    }

    #[test]
    fn sharded_rejects_duplicate_tensor() {
        let dir = tempfile::TempDir::new().expect("test: tempdir");

        // Both shards contain the same tensor name
        let tensors_both: Vec<(&str, u64, GgufTensorType)> =
            vec![("shared.weight", 32, GgufTensorType::F32)];

        let paths = write_shard_files(
            dir.path(),
            "duptest",
            2,
            &[("llama", &tensors_both), ("llama", &tensors_both)],
        );

        let result = ShardedGgufModel::load_sharded(&paths[0]);
        assert!(
            matches!(result, Err(GgufError::ShardDuplicateTensor { .. })),
            "duplicate tensor should return ShardDuplicateTensor"
        );
    }

    #[test]
    fn sharded_tensor_data_returns_bytes() {
        let dir = tempfile::TempDir::new().expect("test: tempdir");
        let tensors_s1: Vec<(&str, u64, GgufTensorType)> =
            vec![("embed.weight", 32, GgufTensorType::F32)];
        let tensors_s2: Vec<(&str, u64, GgufTensorType)> =
            vec![("output.weight", 32, GgufTensorType::F32)];

        let paths = write_shard_files(
            dir.path(),
            "datatest",
            2,
            &[("llama", &tensors_s1), ("llama", &tensors_s2)],
        );

        let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");

        let data1 = model
            .tensor_data("embed.weight")
            .expect("test: embed tensor data");
        assert!(!data1.is_empty());

        let data2 = model
            .tensor_data("output.weight")
            .expect("test: output tensor data");
        assert!(!data2.is_empty());
    }

    #[test]
    fn sharded_missing_tensor_errors() {
        let dir = tempfile::TempDir::new().expect("test: tempdir");
        let tensors_s1: Vec<(&str, u64, GgufTensorType)> =
            vec![("embed.weight", 32, GgufTensorType::F32)];

        let paths = write_shard_files(dir.path(), "errtest", 1, &[("llama", &tensors_s1)]);

        let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");
        let result = model.tensor_data("nonexistent.tensor");
        assert!(result.is_err(), "missing tensor should return error");
    }

    #[test]
    fn sharded_rejects_non_shard_path() {
        let result = ShardedGgufModel::load_sharded("/tmp/model.gguf");
        assert!(
            matches!(result, Err(GgufError::ShardMismatch { .. })),
            "non-shard path should return ShardMismatch"
        );
    }

    #[test]
    fn sharded_tensor_names_yields_all() {
        let dir = tempfile::TempDir::new().expect("test: tempdir");
        let tensors_s1: Vec<(&str, u64, GgufTensorType)> =
            vec![("blk.0.w", 32, GgufTensorType::F32)];
        let tensors_s2: Vec<(&str, u64, GgufTensorType)> =
            vec![("blk.1.w", 32, GgufTensorType::F32)];

        let paths = write_shard_files(
            dir.path(),
            "nametest",
            2,
            &[("llama", &tensors_s1), ("llama", &tensors_s2)],
        );

        let model = ShardedGgufModel::load_sharded(&paths[0]).expect("test: load_sharded");

        let names: std::collections::HashSet<&str> = model.tensor_names().collect();
        assert!(names.contains("blk.0.w"));
        assert!(names.contains("blk.1.w"));
        assert_eq!(names.len(), 2);
    }
}

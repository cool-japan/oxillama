//! GGUF tensor information and storage.

#[cfg(not(feature = "std"))]
use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
type TensorMap = HashMap<String, TensorInfo>;
#[cfg(not(feature = "std"))]
type TensorMap = BTreeMap<String, TensorInfo>;

use crate::error::{GgufError, GgufResult};
use crate::types::GgufTensorType;

/// Information about a single tensor in the GGUF file.
///
/// Parsed from the tensor info section of the GGUF header.
/// Does not contain the actual tensor data — that is loaded separately
/// via memory mapping or direct reads.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g., "blk.0.attn_q.weight").
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Shape of the tensor (dimensions).
    pub dimensions: Vec<u64>,
    /// Quantization / data type.
    pub tensor_type: GgufTensorType,
    /// Byte offset of tensor data relative to the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Returns the total number of elements in the tensor.
    pub fn n_elements(&self) -> u64 {
        if self.dimensions.is_empty() {
            return 0;
        }
        self.dimensions.iter().product()
    }

    /// Returns the total size of the tensor data in bytes.
    pub fn data_size(&self) -> u64 {
        let n_elements = self.n_elements();
        let block_size = self.tensor_type.block_size() as u64;
        let block_bytes = self.tensor_type.block_bytes() as u64;

        // Number of blocks = ceil(n_elements / block_size)
        let n_blocks = n_elements.div_ceil(block_size);
        n_blocks * block_bytes
    }
}

/// A collection of tensor infos and optional tensor data references.
///
/// Acts as the tensor registry for a loaded GGUF model, providing
/// name-based lookup of tensor metadata and data pointers.
#[derive(Debug, Default)]
pub struct TensorStore {
    /// Tensor info entries keyed by tensor name.
    infos: TensorMap,
    /// Base offset of the tensor data section in the file.
    data_section_offset: u64,
}

impl TensorStore {
    /// Create an empty tensor store.
    pub fn new() -> Self {
        Self {
            infos: TensorMap::new(),
            data_section_offset: 0,
        }
    }

    /// Set the base offset of the tensor data section.
    pub fn set_data_offset(&mut self, offset: u64) {
        self.data_section_offset = offset;
    }

    /// Returns the base offset of the tensor data section.
    pub fn data_offset(&self) -> u64 {
        self.data_section_offset
    }

    /// Insert a tensor info entry.
    pub fn insert(&mut self, info: TensorInfo) {
        self.infos.insert(info.name.clone(), info);
    }

    /// Look up a tensor by name.
    pub fn get(&self, name: &str) -> GgufResult<&TensorInfo> {
        self.infos
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound {
                name: name.to_string(),
            })
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.infos.contains_key(name)
    }

    /// Returns the number of tensors.
    pub fn len(&self) -> usize {
        self.infos.len()
    }

    /// Returns true if no tensors are stored.
    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }

    /// Returns an iterator over all tensor infos.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorInfo)> {
        self.infos.iter()
    }

    /// Returns all tensor names.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.infos.keys()
    }

    /// Update the tensor type for a named tensor in-place.
    ///
    /// Used after on-load quantization to keep the stored [`TensorInfo`] in
    /// sync with the override map so that downstream consumers reading
    /// `tensor_type` see the correct quantization format.
    ///
    /// Does nothing if `name` is not found (caller is responsible for
    /// ensuring the tensor exists before inserting into the override map).
    pub fn set_type(&mut self, name: &str, new_type: GgufTensorType) {
        if let Some(info) = self.infos.get_mut(name) {
            info.tensor_type = new_type;
        }
    }

    /// Returns the absolute byte offset of a tensor's data in the file.
    pub fn absolute_offset(&self, name: &str) -> GgufResult<u64> {
        let info = self.get(name)?;
        Ok(self.data_section_offset + info.offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GgufTensorType;

    fn make_info(name: &str, dims: Vec<u64>, offset: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            n_dims: dims.len() as u32,
            dimensions: dims,
            tensor_type: GgufTensorType::F32,
            offset,
        }
    }

    #[test]
    fn test_tensor_info_n_elements_2d() {
        let info = make_info("w", vec![4, 8], 0);
        assert_eq!(info.n_elements(), 32);
    }

    #[test]
    fn test_tensor_info_n_elements_empty_dims() {
        let info = make_info("w", vec![], 0);
        assert_eq!(info.n_elements(), 0);
    }

    #[test]
    fn test_tensor_info_data_size_f32() {
        // F32: block_size=1, block_bytes=4, so data_size = n_elements * 4
        let info = make_info("w", vec![8], 0);
        assert_eq!(info.data_size(), 32); // 8 * 4
    }

    #[test]
    fn test_tensor_store_insert_and_get() {
        let mut store = TensorStore::new();
        store.insert(make_info("layer0.weight", vec![10, 20], 0));
        let info = store
            .get("layer0.weight")
            .expect("test: get existing tensor");
        assert_eq!(info.n_dims, 2);
        assert_eq!(info.dimensions, vec![10, 20]);
    }

    #[test]
    fn test_tensor_store_get_missing_errors() {
        let store = TensorStore::new();
        assert!(store.get("missing").is_err(), "missing tensor should error");
    }

    #[test]
    fn test_tensor_store_contains() {
        let mut store = TensorStore::new();
        store.insert(make_info("a.weight", vec![2], 0));
        assert!(store.contains("a.weight"));
        assert!(!store.contains("b.weight"));
    }

    #[test]
    fn test_tensor_store_len_and_is_empty() {
        let mut store = TensorStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        store.insert(make_info("x", vec![1], 0));
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_tensor_store_iter_yields_all() {
        let mut store = TensorStore::new();
        store.insert(make_info("a", vec![2], 0));
        store.insert(make_info("b", vec![4], 0));
        let names: std::collections::HashSet<&str> =
            store.iter().map(|(k, _)| k.as_str()).collect();
        assert!(names.contains("a"));
        assert!(names.contains("b"));
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_tensor_store_names() {
        let mut store = TensorStore::new();
        store.insert(make_info("foo", vec![1], 0));
        store.insert(make_info("bar", vec![2], 0));
        let names: std::collections::HashSet<&str> = store.names().map(|s| s.as_str()).collect();
        assert!(names.contains("foo"));
        assert!(names.contains("bar"));
    }

    #[test]
    fn test_tensor_store_absolute_offset() {
        let mut store = TensorStore::new();
        store.set_data_offset(1024);
        store.insert(make_info("w", vec![1], 256));
        let abs = store.absolute_offset("w").expect("test: absolute_offset");
        assert_eq!(abs, 1024 + 256);
    }

    #[test]
    fn test_tensor_store_absolute_offset_missing_errors() {
        let store = TensorStore::new();
        assert!(
            store.absolute_offset("nonexistent").is_err(),
            "missing tensor offset should error"
        );
    }

    #[test]
    fn test_tensor_store_data_offset_default_zero() {
        let store = TensorStore::new();
        assert_eq!(store.data_offset(), 0);
    }

    #[test]
    fn test_tensor_store_set_data_offset() {
        let mut store = TensorStore::new();
        store.set_data_offset(4096);
        assert_eq!(store.data_offset(), 4096);
    }
}

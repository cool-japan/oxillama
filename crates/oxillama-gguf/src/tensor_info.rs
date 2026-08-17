//! GGUF tensor information and storage.

#[cfg(not(feature = "std"))]
use alloc::{
    collections::BTreeMap,
    format,
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

    /// Fallible variant of [`Self::n_elements`].
    ///
    /// `dimensions` is parsed directly from an untrusted GGUF file, so a
    /// naive `product()` over attacker-controlled `u64`s can overflow —
    /// e.g. dims `[2^32, 2^32]` wrap to `0` in a release build (and panic
    /// in debug). This returns a typed error instead of silently wrapping.
    pub fn try_n_elements(&self) -> GgufResult<u64> {
        if self.dimensions.is_empty() {
            return Ok(0);
        }
        self.dimensions.iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d).ok_or_else(|| GgufError::IntegrityError {
                tensor_name: self.name.clone(),
                reason: format!(
                    "tensor dimensions {:?} overflow u64 while computing element count",
                    self.dimensions
                ),
            })
        })
    }

    /// Fallible variant of [`Self::data_size`].
    ///
    /// Rejects overflow in the blocks-times-bytes-per-block multiplication
    /// (and any overflow surfaced by [`Self::try_n_elements`]) instead of
    /// silently wrapping to a small, attacker-influenced size — which is
    /// what let a crafted file steer downstream offset arithmetic.
    pub fn try_data_size(&self) -> GgufResult<u64> {
        let n_elements = self.try_n_elements()?;
        let block_size = self.tensor_type.block_size() as u64;
        if block_size == 0 {
            return Err(GgufError::IntegrityError {
                tensor_name: self.name.clone(),
                reason: "tensor type has a zero block size".to_string(),
            });
        }
        let block_bytes = self.tensor_type.block_bytes() as u64;
        let n_blocks = n_elements.div_ceil(block_size);
        n_blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| GgufError::IntegrityError {
                tensor_name: self.name.clone(),
                reason: format!(
                    "tensor data size overflow: {n_blocks} blocks * {block_bytes} bytes/block"
                ),
            })
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

    /// Insert a tensor info entry, silently overwriting any existing entry
    /// with the same name.
    ///
    /// Kept for callers (e.g. shard merging, on-load quantization bookkeeping)
    /// that intentionally build a store from data that has already been
    /// validated elsewhere. Parsing an *untrusted* GGUF file should use
    /// [`Self::try_insert`] instead — see its docs for why.
    pub fn insert(&mut self, info: TensorInfo) {
        self.infos.insert(info.name.clone(), info);
    }

    /// Insert a tensor info entry, rejecting a name collision instead of
    /// silently overwriting the previous entry.
    ///
    /// A GGUF file's tensor-info section is untrusted input: parsing it
    /// with the plain [`Self::insert`] lets a file that declares, say,
    /// 10,000 tensors all named `"x"` collapse to a single-entry store with
    /// no error — silently dropping 9,999 tensors a loader would otherwise
    /// expect to find. Use this method wherever tensor infos are read
    /// directly off an untrusted byte source.
    pub fn try_insert(&mut self, info: TensorInfo) -> GgufResult<()> {
        if self.infos.contains_key(&info.name) {
            return Err(GgufError::IntegrityError {
                tensor_name: info.name,
                reason: "duplicate tensor name in GGUF tensor-info section".to_string(),
            });
        }
        self.infos.insert(info.name.clone(), info);
        Ok(())
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

    // ── V3 regression: checked element-count / data-size arithmetic ────────

    #[test]
    fn test_try_n_elements_overflow_errors() {
        // 2^32 * 2^32 = 2^64, overflows u64.
        let info = make_info("huge", vec![1u64 << 32, 1u64 << 32], 0);
        let err = info
            .try_n_elements()
            .expect_err("dims overflowing u64 must error, not wrap");
        assert!(matches!(err, GgufError::IntegrityError { .. }));

        // Contrast with the old infallible path — this is profile-dependent
        // by design, matching the report's exact "panic in dev and wrap to
        // 0 in release" description: debug builds have overflow checks on
        // (`*.product()` panics), release builds do not (it silently wraps
        // to 0). Assert whichever behavior this build profile actually
        // exhibits, via `catch_unwind` so a debug-mode panic doesn't abort
        // this test itself.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| info.n_elements()));
        if cfg!(debug_assertions) {
            assert!(
                panicked.is_err(),
                "sanity: infallible n_elements() should panic on overflow in a debug build"
            );
        } else {
            assert_eq!(
                panicked.expect("release build must not panic on overflow"),
                0,
                "sanity: infallible n_elements() should silently wrap to 0 in a release build"
            );
        }
    }

    #[test]
    fn test_try_n_elements_normal_case_matches_infallible() {
        let info = make_info("w", vec![4, 8], 0);
        assert_eq!(info.try_n_elements().expect("no overflow"), 32);
        assert_eq!(
            info.try_n_elements().expect("no overflow"),
            info.n_elements()
        );
    }

    #[test]
    fn test_try_data_size_overflow_errors() {
        let info = make_info("huge", vec![1u64 << 40, 1u64 << 40], 0);
        assert!(info.try_data_size().is_err());
    }

    #[test]
    fn test_try_data_size_normal_case_matches_infallible() {
        let info = make_info("w", vec![8], 0); // F32: 8 * 4 = 32 bytes
        assert_eq!(info.try_data_size().expect("no overflow"), 32);
        assert_eq!(info.try_data_size().expect("no overflow"), info.data_size());
    }

    // ── V6 regression: duplicate tensor names must be rejected ─────────────

    #[test]
    fn test_try_insert_rejects_duplicate_name() {
        let mut store = TensorStore::new();
        store
            .try_insert(make_info("x", vec![1], 0))
            .expect("first insert must succeed");
        let err = store
            .try_insert(make_info("x", vec![1], 32))
            .expect_err("duplicate name must be rejected");
        assert!(matches!(err, GgufError::IntegrityError { .. }));
        // The original entry must be left untouched (not silently
        // overwritten by the rejected second insert).
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("x").expect("still present").offset, 0);
    }

    #[test]
    fn test_try_insert_distinct_names_all_succeed() {
        let mut store = TensorStore::new();
        for i in 0..5 {
            store
                .try_insert(make_info(&format!("t{i}"), vec![1], 0))
                .expect("distinct names must all succeed");
        }
        assert_eq!(store.len(), 5);
    }

    #[test]
    fn test_plain_insert_still_silently_overwrites() {
        // `insert()` keeps its historical overwrite behavior for callers
        // that intentionally rebuild entries (e.g. quantize-on-load type
        // updates); only the untrusted-parse paths were switched over to
        // `try_insert`.
        let mut store = TensorStore::new();
        store.insert(make_info("x", vec![1], 0));
        store.insert(make_info("x", vec![1], 99));
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("x").expect("present").offset, 99);
    }
}

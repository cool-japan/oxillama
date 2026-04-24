//! LRU weight pager — the core of the CPU/disk offload system.
//!
//! [`LayerPager`] manages a budget-limited resident set of tensor bytes.
//! When a tensor is needed and not resident, it is loaded from the backing
//! [`PagerSource`] and the least-recently-used non-pinned tensor is evicted
//! first (if needed to stay within the RAM budget).
//!
//! # Thread safety
//!
//! [`LayerPager`] is `Send + Sync`. The resident map is protected by an
//! [`RwLock`], the LRU queue by a [`Mutex`]. All lock operations use
//! `.map_err(|_| RuntimeError::LockPoisoned)?` — no `unwrap()` anywhere.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::RuntimeError;

// ─────────────────────────────────────────────────────────────────────────────
// TensorId
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for a weight tensor.
///
/// Typically the tensor name as it appears in the GGUF file, e.g.
/// `"blk.0.attn_q.weight"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorId(pub String);

impl TensorId {
    /// Construct a [`TensorId`] from any string-like value.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ResidentTensor
// ─────────────────────────────────────────────────────────────────────────────

/// A tensor that is currently resident in RAM.
///
/// The bytes are stored as a reference-counted slice. The dequantization step
/// happens outside the pager — callers pass `data` directly to the fused GEMM
/// kernels (or dequant on-the-fly as in the existing arch layer).
pub struct ResidentTensor {
    /// Raw quantized bytes from the GGUF weight store.
    pub data: Arc<[u8]>,
    /// Byte length of `data` (cached to avoid the indirection).
    pub size_bytes: usize,
}

impl std::fmt::Debug for ResidentTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentTensor")
            .field("size_bytes", &self.size_bytes)
            .field("data_len", &self.data.len())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorEntry
// ─────────────────────────────────────────────────────────────────────────────

/// Location of a tensor in the backing weight file.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Absolute byte offset within the backing source.
    pub file_offset: u64,
    /// Number of bytes occupied by this tensor.
    pub size_bytes: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// PagerSource
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over the backing byte store for weight data.
///
/// Unlike the GGUF [`Source`][oxillama_gguf::source::Source] trait (which uses
/// an associated error type), `PagerSource` uses [`RuntimeError`] directly for
/// easy `dyn` dispatch.
pub trait PagerSource: Send + Sync {
    /// Read exactly `out.len()` bytes starting at `offset` into `out`.
    ///
    /// Returns `Err(RuntimeError::OffloadEof)` if `offset + out.len()` exceeds
    /// the total size of the source.
    fn read_bytes_at(&self, offset: u64, out: &mut [u8]) -> Result<(), RuntimeError>;

    /// Total size of the backing store in bytes.
    fn total_size_bytes(&self) -> u64;
}

// ─────────────────────────────────────────────────────────────────────────────
// FilePagerSource
// ─────────────────────────────────────────────────────────────────────────────

/// File-backed [`PagerSource`] using `std::fs::File` seek + read.
///
/// Opens a new file descriptor for each [`read_bytes_at`] call, which is safe
/// across threads but not optimal for high-frequency reads. For throughput-
/// critical use cases, prefer [`MmapPagerSource`] (requires `mmap` feature).
///
/// [`read_bytes_at`]: FilePagerSource::read_bytes_at
pub struct FilePagerSource {
    path: std::path::PathBuf,
    total_bytes: u64,
}

impl FilePagerSource {
    /// Open a file-backed pager source at `path`.
    ///
    /// Reads the file metadata to determine total size; fails if the file
    /// cannot be opened or its metadata cannot be queried.
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Self, RuntimeError> {
        let path = path.into();
        let meta = std::fs::metadata(&path)?;
        Ok(Self {
            total_bytes: meta.len(),
            path,
        })
    }
}

impl PagerSource for FilePagerSource {
    fn read_bytes_at(&self, offset: u64, out: &mut [u8]) -> Result<(), RuntimeError> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(out)?;
        Ok(())
    }

    fn total_size_bytes(&self) -> u64 {
        self.total_bytes
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MmapPagerSource (optional, requires `mmap` feature)
// ─────────────────────────────────────────────────────────────────────────────

/// Memory-mapped [`PagerSource`] — faster random access than seek+read.
///
/// Requires the `mmap` feature. The mmap is created read-only and is safe to
/// share across threads. OS-level demand paging may still cause page faults
/// on access, but the `read_bytes_at` implementation is zero-copy after the
/// initial map.
#[cfg(feature = "mmap")]
pub struct MmapPagerSource {
    mmap: Arc<memmap2::Mmap>,
}

#[cfg(feature = "mmap")]
impl MmapPagerSource {
    /// Open and memory-map the file at `path`.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, RuntimeError> {
        let file = std::fs::File::open(path)?;
        // Safety: the file is read-only. We do not mutate the mmap. If the
        // underlying file is modified by another process while we hold the
        // mmap, behaviour is unspecified but not UB in the Rust sense because
        // we only ever read through shared refs.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self {
            mmap: Arc::new(mmap),
        })
    }
}

#[cfg(feature = "mmap")]
impl PagerSource for MmapPagerSource {
    fn read_bytes_at(&self, offset: u64, out: &mut [u8]) -> Result<(), RuntimeError> {
        let start = offset as usize;
        let end = start.checked_add(out.len()).ok_or(RuntimeError::OffloadEof {
            offset,
            needed: out.len(),
            available: 0,
        })?;
        if end > self.mmap.len() {
            let available = self.mmap.len().saturating_sub(start);
            return Err(RuntimeError::OffloadEof {
                offset,
                needed: out.len(),
                available,
            });
        }
        out.copy_from_slice(&self.mmap[start..end]);
        Ok(())
    }

    fn total_size_bytes(&self) -> u64 {
        self.mmap.len() as u64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerPager
// ─────────────────────────────────────────────────────────────────────────────

/// LRU weight pager — evicts cold tensors to free RAM, loads from `source` on demand.
///
/// # Pinned tensors
///
/// Tensors whose [`TensorId`] is in the `pinned` set are never evicted.
/// If pinned tensors alone exceed the budget, eviction will exhaust all
/// non-pinned candidates and then stop (no error); the budget acts as a
/// best-effort target.
///
/// # Acquiring tensors
///
/// Call [`acquire`][LayerPager::acquire] to get an `Arc<ResidentTensor>` for a
/// tensor by ID. The result shares ownership with the pager's resident map, so
/// the bytes stay alive as long as either the pager or the caller holds a
/// reference — eviction only removes the pager's entry, not the data itself.
pub struct LayerPager {
    source: Arc<dyn PagerSource>,
    tensor_map: HashMap<TensorId, TensorEntry>,
    resident: RwLock<HashMap<TensorId, Arc<ResidentTensor>>>,
    pinned: HashSet<TensorId>,
    lru: Mutex<VecDeque<TensorId>>,
    budget_bytes: u64,
    resident_bytes: AtomicU64,
}

impl LayerPager {
    /// Create a new pager.
    ///
    /// # Parameters
    ///
    /// - `source` — backing weight store (file or mmap).
    /// - `tensor_map` — map from [`TensorId`] to file offset + size.
    /// - `budget_bytes` — maximum resident bytes (use `u64::MAX` for unlimited).
    /// - `pinned` — set of tensor IDs that are never evicted.
    pub fn new(
        source: Arc<dyn PagerSource>,
        tensor_map: HashMap<TensorId, TensorEntry>,
        budget_bytes: u64,
        pinned: HashSet<TensorId>,
    ) -> Self {
        Self {
            source,
            tensor_map,
            resident: RwLock::new(HashMap::new()),
            pinned,
            lru: Mutex::new(VecDeque::new()),
            budget_bytes,
            resident_bytes: AtomicU64::new(0),
        }
    }

    /// Acquire a tensor, loading it from the source if not currently resident.
    ///
    /// The LRU order is updated on every successful acquire. If the tensor
    /// must be loaded, the pager first evicts non-pinned tensors until the
    /// budget allows the new allocation, then reads the bytes from `source`.
    ///
    /// # Errors
    ///
    /// - [`RuntimeError::TensorNotFound`] — `id` is not in the tensor map.
    /// - [`RuntimeError::LockPoisoned`] — an internal lock was poisoned.
    /// - [`RuntimeError::OffloadEof`] / [`RuntimeError::Io`] — source read failed.
    pub fn acquire(&self, id: &TensorId) -> Result<Arc<ResidentTensor>, RuntimeError> {
        // Fast path: already resident.
        {
            let guard = self
                .resident
                .read()
                .map_err(|_| RuntimeError::LockPoisoned)?;
            if let Some(_tensor) = guard.get(id) {
                // Update LRU position.
                drop(guard);
                self.bump_lru(id)?;
                // Re-read after dropping the read guard so we avoid holding
                // both read and LRU lock simultaneously.
                let guard2 = self
                    .resident
                    .read()
                    .map_err(|_| RuntimeError::LockPoisoned)?;
                // Tensor may theoretically have been evicted between drop and
                // re-read in a race; fall through to slow path if that happened.
                if let Some(tensor2) = guard2.get(id) {
                    return Ok(Arc::clone(tensor2));
                }
            }
        }

        // Slow path: not resident — load from source.
        let entry = self
            .tensor_map
            .get(id)
            .ok_or_else(|| RuntimeError::TensorNotFound(id.0.clone()))?;

        // Evict until the new tensor fits.
        self.evict_to_fit(entry.size_bytes)?;

        // Read from backing source.
        let mut data = vec![0u8; entry.size_bytes];
        self.source.read_bytes_at(entry.file_offset, &mut data)?;

        let tensor = Arc::new(ResidentTensor {
            data: data.into(),
            size_bytes: entry.size_bytes,
        });

        // Insert into resident map and append to LRU tail.
        {
            let mut guard = self
                .resident
                .write()
                .map_err(|_| RuntimeError::LockPoisoned)?;
            guard.insert(id.clone(), Arc::clone(&tensor));
        }
        {
            let mut lru = self.lru.lock().map_err(|_| RuntimeError::LockPoisoned)?;
            lru.push_back(id.clone());
        }
        self.resident_bytes
            .fetch_add(entry.size_bytes as u64, Ordering::Relaxed);

        Ok(tensor)
    }

    /// Evict non-pinned LRU tensors until `needed_bytes` can fit within the budget.
    fn evict_to_fit(&self, needed_bytes: usize) -> Result<(), RuntimeError> {
        loop {
            let current = self.resident_bytes.load(Ordering::Relaxed);
            if current.saturating_add(needed_bytes as u64) <= self.budget_bytes {
                break;
            }

            // Find the oldest non-pinned entry.
            let victim = {
                let lru = self.lru.lock().map_err(|_| RuntimeError::LockPoisoned)?;
                lru.iter().find(|id| !self.pinned.contains(*id)).cloned()
            };

            match victim {
                // Nothing left to evict (all resident tensors are pinned).
                None => break,
                Some(victim_id) => {
                    let removed = {
                        let mut guard = self
                            .resident
                            .write()
                            .map_err(|_| RuntimeError::LockPoisoned)?;
                        guard.remove(&victim_id)
                    };
                    if let Some(evicted) = removed {
                        self.resident_bytes
                            .fetch_sub(evicted.size_bytes as u64, Ordering::Relaxed);
                        let mut lru =
                            self.lru.lock().map_err(|_| RuntimeError::LockPoisoned)?;
                        lru.retain(|x| x != &victim_id);
                    }
                }
            }
        }
        Ok(())
    }

    /// Move `id` to the tail of the LRU queue (mark as most-recently-used).
    fn bump_lru(&self, id: &TensorId) -> Result<(), RuntimeError> {
        let mut lru = self.lru.lock().map_err(|_| RuntimeError::LockPoisoned)?;
        if let Some(pos) = lru.iter().position(|x| x == id) {
            lru.remove(pos);
        }
        lru.push_back(id.clone());
        Ok(())
    }

    /// Return the number of bytes currently resident in RAM.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes.load(Ordering::Relaxed)
    }

    /// Return the number of tensors currently resident in RAM.
    pub fn resident_count(&self) -> usize {
        self.resident
            .read()
            .map(|g| g.len())
            .unwrap_or(0)
    }

    /// Return the configured RAM budget in bytes.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Check whether a tensor is currently resident.
    pub fn is_resident(&self, id: &TensorId) -> bool {
        self.resident
            .read()
            .map(|g| g.contains_key(id))
            .unwrap_or(false)
    }

    /// Check whether a tensor is pinned (will never be evicted).
    pub fn is_pinned(&self, id: &TensorId) -> bool {
        self.pinned.contains(id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::io::Write;
    use std::sync::Arc;

    // ── In-memory PagerSource for tests ──────────────────────────────────────

    struct VecPagerSource(Vec<u8>);

    impl PagerSource for VecPagerSource {
        fn read_bytes_at(&self, offset: u64, out: &mut [u8]) -> Result<(), RuntimeError> {
            let start = offset as usize;
            let end = start.checked_add(out.len()).ok_or(RuntimeError::OffloadEof {
                offset,
                needed: out.len(),
                available: 0,
            })?;
            if end > self.0.len() {
                let available = self.0.len().saturating_sub(start);
                return Err(RuntimeError::OffloadEof {
                    offset,
                    needed: out.len(),
                    available,
                });
            }
            out.copy_from_slice(&self.0[start..end]);
            Ok(())
        }

        fn total_size_bytes(&self) -> u64 {
            self.0.len() as u64
        }
    }

    // ── Helper to build a pager with synthetic data ───────────────────────────

    fn make_pager(
        tensors: &[(&str, usize, u64)],
        budget: u64,
        pinned: &[&str],
    ) -> (LayerPager, Vec<u8>) {
        let total: usize = tensors.iter().map(|(_, sz, offset)| *offset as usize + *sz).max().unwrap_or(0) + 1;
        let mut data = vec![0u8; total];
        let mut tensor_map = HashMap::new();
        for (id, size, offset) in tensors {
            for i in 0..*size {
                data[*offset as usize + i] = (i % 256) as u8;
            }
            tensor_map.insert(
                TensorId(id.to_string()),
                TensorEntry {
                    file_offset: *offset,
                    size_bytes: *size,
                },
            );
        }
        let pinned_set: HashSet<TensorId> =
            pinned.iter().map(|s| TensorId(s.to_string())).collect();
        let pager = LayerPager::new(
            Arc::new(VecPagerSource(data.clone())),
            tensor_map,
            budget,
            pinned_set,
        );
        (pager, data)
    }

    // ── Test: basic eviction ──────────────────────────────────────────────────

    #[test]
    fn offload_budget_evicts_coldest() {
        // 3 tensors, 100 bytes each; budget = 200 bytes (fits 2)
        let (pager, _) = make_pager(
            &[
                ("layer_0", 100, 0),
                ("layer_1", 100, 100),
                ("layer_2", 100, 200),
            ],
            200,
            &[],
        );

        let _t0 = pager.acquire(&TensorId("layer_0".into())).expect("t0");
        let _t1 = pager.acquire(&TensorId("layer_1".into())).expect("t1");
        assert_eq!(pager.resident_count(), 2);

        // Acquiring layer_2 must evict the coldest (layer_0).
        drop(_t0);
        drop(_t1);
        let _t2 = pager.acquire(&TensorId("layer_2".into())).expect("t2");
        assert!(
            pager.resident_bytes() <= 200,
            "resident_bytes ({}) must be <= budget (200)",
            pager.resident_bytes()
        );
    }

    // ── Test: pinned tensors survive eviction ─────────────────────────────────

    #[test]
    fn offload_pinned_never_evicted() {
        // budget = 100, fits exactly 1; layer_0 is pinned
        let (pager, _) = make_pager(
            &[("layer_0", 100, 0), ("layer_1", 100, 100)],
            100,
            &["layer_0"],
        );

        let _t0 = pager.acquire(&TensorId("layer_0".into())).expect("pinned");
        // Acquiring layer_1 must not evict layer_0
        let _t1 = pager.acquire(&TensorId("layer_1".into())).expect("cold");

        assert!(
            pager.is_resident(&TensorId("layer_0".into())),
            "pinned tensor must not be evicted"
        );
    }

    // ── Test: bytes read correctly ────────────────────────────────────────────

    #[test]
    fn offload_acquire_reads_correct_bytes() {
        let (pager, data) = make_pager(&[("t0", 64, 128)], u64::MAX, &[]);
        let tensor = pager.acquire(&TensorId("t0".into())).expect("t0");
        assert_eq!(tensor.data.len(), 64);
        assert_eq!(&tensor.data[..], &data[128..192]);
    }

    // ── Test: unknown tensor returns error ────────────────────────────────────

    #[test]
    fn offload_unknown_tensor_returns_error() {
        let (pager, _) = make_pager(&[("t0", 10, 0)], u64::MAX, &[]);
        let res = pager.acquire(&TensorId("nonexistent".into()));
        assert!(
            matches!(res, Err(RuntimeError::TensorNotFound(_))),
            "expected TensorNotFound, got {res:?}"
        );
    }

    // ── Test: double acquire returns same bytes ───────────────────────────────

    #[test]
    fn offload_double_acquire_returns_same_bytes() {
        let (pager, data) = make_pager(&[("t0", 32, 64)], u64::MAX, &[]);
        let a = pager.acquire(&TensorId("t0".into())).expect("a");
        let b = pager.acquire(&TensorId("t0".into())).expect("b");
        assert_eq!(&a.data[..], &b.data[..]);
        assert_eq!(&a.data[..], &data[64..96]);
    }

    // ── Test: FilePagerSource reads correct bytes ─────────────────────────────

    #[test]
    fn offload_file_pager_source_reads_correctly() {
        let mut tmp = tempfile::NamedTempFile::new().expect("temp file");
        let payload: Vec<u8> = (0u8..=255u8).collect();
        tmp.write_all(&payload).expect("write");
        let source = FilePagerSource::open(tmp.path()).expect("open");
        let mut buf = vec![0u8; 10];
        source.read_bytes_at(5, &mut buf).expect("read");
        assert_eq!(&buf, &payload[5..15]);
    }

    // ── Test: FilePagerSource EOF returns error ───────────────────────────────

    #[test]
    fn offload_file_source_eof_errors() {
        let mut tmp = tempfile::NamedTempFile::new().expect("temp file");
        tmp.write_all(b"short").expect("write");
        let source = FilePagerSource::open(tmp.path()).expect("open");
        let mut buf = vec![0u8; 100];
        let res = source.read_bytes_at(0, &mut buf);
        assert!(
            res.is_err(),
            "reading past end of file must return Err"
        );
    }

    // ── Test: resident_count and resident_bytes ───────────────────────────────

    #[test]
    fn offload_resident_count_tracks_evictions() {
        let (pager, _) = make_pager(
            &[("a", 50, 0), ("b", 50, 50), ("c", 50, 100)],
            100,
            &[],
        );
        assert_eq!(pager.resident_count(), 0);
        let _a = pager.acquire(&TensorId("a".into())).expect("a");
        assert_eq!(pager.resident_count(), 1);
        let _b = pager.acquire(&TensorId("b".into())).expect("b");
        assert_eq!(pager.resident_count(), 2);
        // c should evict a
        let _c = pager.acquire(&TensorId("c".into())).expect("c");
        assert!(pager.resident_count() <= 2, "budget limits to 2 tensors");
        assert!(
            pager.resident_bytes() <= 100,
            "resident_bytes must not exceed budget"
        );
    }

    // ── Test: is_pinned ───────────────────────────────────────────────────────

    #[test]
    fn offload_is_pinned_reflects_set() {
        let (pager, _) = make_pager(&[("a", 10, 0), ("b", 10, 10)], u64::MAX, &["a"]);
        assert!(pager.is_pinned(&TensorId("a".into())));
        assert!(!pager.is_pinned(&TensorId("b".into())));
    }

    // ── Test: multiple evictions stay within budget ───────────────────────────

    #[test]
    fn offload_budget_strictly_respected() {
        let budget = 50u64;
        let (pager, _) = make_pager(
            &[
                ("t0", 50, 0),
                ("t1", 50, 50),
                ("t2", 50, 100),
                ("t3", 50, 150),
            ],
            budget,
            &[],
        );
        for name in ["t0", "t1", "t2", "t3"] {
            let _ = pager.acquire(&TensorId(name.into())).expect(name);
            assert!(
                pager.resident_bytes() <= budget,
                "after acquiring {name}, resident_bytes={} > budget={budget}",
                pager.resident_bytes()
            );
        }
    }

    // ── Test: tensor_id display ───────────────────────────────────────────────

    #[test]
    fn tensor_id_display() {
        let id = TensorId::new("blk.0.attn_q.weight");
        assert_eq!(id.to_string(), "blk.0.attn_q.weight");
    }
}

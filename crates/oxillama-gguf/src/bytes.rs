//! Reference-counted, immutable views into a GGUF byte payload.
//!
//! Tensor weights are the bulk of a GGUF file — 2.38 GB for a 4B `Q4_K_M`
//! checkpoint.  Handing every consumer its own `Vec<u8>` copy of a tensor
//! defeats the whole point of [`GgufModel::load_mmap`][crate::GgufModel] :
//! the kernels only ever *read* the block bytes, so a copy buys nothing and
//! costs one extra resident byte per weight byte.
//!
//! [`SharedBytes`] is the alternative: a `(owner, offset, len)` triple whose
//! owner is an `Arc` over whatever actually holds the memory — the `Mmap` for
//! a memory-mapped file, a `Vec<u8>` for an in-memory or quantized-on-load
//! payload.  Cloning one is an atomic increment; dereferencing one yields a
//! plain `&[u8]` sub-slice with no copy.
//!
//! ## Alignment
//!
//! Consumers get exactly the alignment the GGUF file provides.  Tensor data
//! offsets are multiples of `general.alignment` (32 by default) measured from
//! a page-aligned mapping base, so mmap-backed views are *at least* 32-byte
//! aligned — strictly better than the 1-byte guarantee a `Vec<u8>` carries.
//! No kernel may therefore regress: every one of them consumes `&[u8]` and
//! reads through unaligned-safe loads (`from_le_bytes`, `vld1q_u8`, `loadu`).
//!
//! ## Measured (Qwen3-4B `Q4_K_M`, Apple M3, 2026-08-04)
//!
//! Switching the Qwen3 loader from `tensor_data().to_vec()` to
//! `tensor_bytes()` — together with keeping `token_embd` quantized, see
//! `oxillama-arch`'s `qwen3::embedding` — moved the CLI from
//!
//! | | before | after |
//! |---|---|---|
//! | maximum resident set size | 6.736 / 6.727 GB | 2.684 / 2.684 GB |
//! | peak memory footprint | 4.204 / 4.201 GB | 0.148 / 0.148 GB |
//! | weight-load wall time | 0.604 / 0.449 s | 0.015 / 0.014 s |
//! | page reclaims | 413 k | 167 k |
//!
//! Decode throughput was unaffected (median-of-5, differencing harness:
//! 13.22 → 14.15 tok/s; a lower-noise 8→88-token harness over 15 interleaved
//! pairs put both at 13.0 vs 13.2 tok/s, i.e. parity inside run-to-run
//! spread), and generation stayed character-identical at a fixed seed.

#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::sync::Arc;

use core::fmt;
use core::ops::Deref;
use core::ptr::NonNull;

/// A container that owns a contiguous byte payload for as long as it is alive.
///
/// Implemented for `Vec<u8>` out of the box; [`crate::GgufModel`] implements
/// it for its own mmap-or-owned backing store.
///
/// # Safety
///
/// [`owned_bytes`][Self::owned_bytes] must return the **same address and the
/// same length** on every call, for as long as the owner is alive, and the
/// bytes it points at must not be mutated through any other handle.
/// [`SharedBytes`] resolves the pointer once at construction and dereferences
/// it without re-consulting the owner, so an implementation that moves or
/// resizes its payload would create a dangling slice.
pub unsafe trait ByteOwner: Send + Sync {
    /// The bytes this owner keeps alive.
    fn owned_bytes(&self) -> &[u8];
}

// SAFETY: the `Vec` lives inside the `Arc` and is never mutated through
// `SharedBytes` (which exposes read-only access only), so its heap buffer
// neither moves nor changes length.
unsafe impl ByteOwner for Vec<u8> {
    fn owned_bytes(&self) -> &[u8] {
        self
    }
}

/// An immutable byte range that keeps its backing store alive.
///
/// Behaves like a `&[u8]` (via [`Deref`]) but owns a share of the allocation,
/// so it can be stored in long-lived structures without borrowing.
///
/// The view's start pointer is resolved once, at construction, so [`Deref`] is
/// two loads and inlines away.  That matters: GEMV kernels re-slice a weight
/// payload once per output row — 151 936 times for a 4B model's LM head — and
/// a virtual call through the owner on each of those would be a measurable
/// tax on decode.  The obvious safe alternative, resolving
/// `owner.owned_bytes()` inside `deref`, was rejected for exactly that reason:
/// `Arc<dyn ByteOwner>` cannot inline, so every `&tensor.data[row..]` in a row
/// loop would pay an indirect call.  Qwen3 decode mostly takes the fused
/// entry point (one deref per GEMV, not per row) and did not measurably care,
/// but the unfused `gemv`/`gemm` kernels re-slice per row and would.
#[derive(Clone)]
pub struct SharedBytes {
    owner: Arc<dyn ByteOwner>,
    /// Start of the view: `owner.owned_bytes()[offset..].as_ptr()`, cached.
    ptr: NonNull<u8>,
    offset: usize,
    len: usize,
}

// SAFETY: `SharedBytes` hands out shared, read-only access to bytes kept alive
// by `owner: Arc<dyn ByteOwner>`, which is itself `Send + Sync`.  `ptr` is a
// derived pointer into that payload and is only ever read.  The raw pointer
// field is what suppresses the automatic impls; the invariants they need still
// hold.
unsafe impl Send for SharedBytes {}
// SAFETY: see the `Send` impl above — the view is immutable and its owner is
// `Sync`, so `&SharedBytes` is safe to share across threads.
unsafe impl Sync for SharedBytes {}

impl SharedBytes {
    /// Wrap an owned byte vector, taking the whole of it.
    pub fn from_vec(data: Vec<u8>) -> Self {
        let len = data.len();
        let owner: Arc<dyn ByteOwner> = Arc::new(data);
        // `from_owner` can only fail on an out-of-range request; 0..len always
        // fits, so this branch is unreachable — but stay total anyway.
        Self::from_owner(owner, 0, len).unwrap_or_else(Self::empty)
    }

    /// Take an empty view.  Useful as a default / placeholder.
    pub fn empty() -> Self {
        let owner: Arc<dyn ByteOwner> = Arc::new(Vec::new());
        let ptr = NonNull::from(&[] as &[u8]).cast::<u8>();
        Self {
            owner,
            ptr,
            offset: 0,
            len: 0,
        }
    }

    /// Borrow `offset..offset + len` out of `owner`.
    ///
    /// Returns `None` when the range does not fit inside the owner's payload,
    /// which keeps the constructor total (no panic on a malformed file).
    pub fn from_owner(owner: Arc<dyn ByteOwner>, offset: usize, len: usize) -> Option<Self> {
        let end = offset.checked_add(len)?;
        let payload = owner.owned_bytes();
        if end > payload.len() {
            return None;
        }
        let ptr = NonNull::from(payload.get(offset..end)?).cast::<u8>();
        Some(Self {
            owner,
            ptr,
            offset,
            len,
        })
    }

    /// The bytes this view covers.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr`/`len` were derived from `owner`'s payload in
        // `from_owner`, `owner` is alive for as long as `self` is, and the
        // `ByteOwner` contract forbids the payload from moving, resizing or
        // being mutated.  The lifetime is tied to `&self`, so the slice cannot
        // outlive the owning `Arc`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Byte offset of this view inside its owner's payload.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Narrow this view to `start..start + len`, relative to its own start.
    ///
    /// Returns `None` when the sub-range escapes the current view.  The result
    /// shares the same owner, so no bytes are copied.
    pub fn slice(&self, start: usize, len: usize) -> Option<Self> {
        let end = start.checked_add(len)?;
        if end > self.len {
            return None;
        }
        let ptr = NonNull::from(self.as_slice().get(start..end)?).cast::<u8>();
        Some(Self {
            owner: Arc::clone(&self.owner),
            ptr,
            offset: self.offset + start,
            len,
        })
    }
}

impl Deref for SharedBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for SharedBytes {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl From<Vec<u8>> for SharedBytes {
    fn from(data: Vec<u8>) -> Self {
        Self::from_vec(data)
    }
}

impl From<&[u8]> for SharedBytes {
    fn from(data: &[u8]) -> Self {
        Self::from_vec(data.to_vec())
    }
}

impl Default for SharedBytes {
    fn default() -> Self {
        Self::empty()
    }
}

/// Prints the length only — a tensor payload is gigabytes wide and dumping it
/// would be useless in any log or assertion message.
impl fmt::Debug for SharedBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedBytes")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

impl PartialEq for SharedBytes {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for SharedBytes {}

impl PartialEq<[u8]> for SharedBytes {
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}

impl PartialEq<&[u8]> for SharedBytes {
    fn eq(&self, other: &&[u8]) -> bool {
        self.as_slice() == *other
    }
}

impl PartialEq<Vec<u8>> for SharedBytes {
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<SharedBytes> for Vec<u8> {
    fn eq(&self, other: &SharedBytes) -> bool {
        self.as_slice() == other.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_covers_whole_payload() {
        let b = SharedBytes::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(b.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(b.len(), 4);
        assert_eq!(b.offset(), 0);
    }

    #[test]
    fn from_owner_validates_range() {
        let owner: Arc<dyn ByteOwner> = Arc::new(vec![0u8; 16]);
        assert!(SharedBytes::from_owner(Arc::clone(&owner), 8, 8).is_some());
        assert!(SharedBytes::from_owner(Arc::clone(&owner), 8, 9).is_none());
        assert!(SharedBytes::from_owner(owner, usize::MAX, 1).is_none());
    }

    #[test]
    fn from_owner_views_a_subrange_without_copying() {
        let owner: Arc<dyn ByteOwner> = Arc::new((0u8..32).collect::<Vec<u8>>());
        let view = SharedBytes::from_owner(Arc::clone(&owner), 8, 4).expect("test: in-range view");
        assert_eq!(view.as_slice(), &[8, 9, 10, 11]);
        assert_eq!(view.offset(), 8);
        // Same backing allocation, no copy.
        let base = owner.owned_bytes().as_ptr() as usize;
        assert_eq!(view.as_slice().as_ptr() as usize, base + 8);
    }

    #[test]
    fn clone_shares_the_allocation() {
        let b = SharedBytes::from_vec((0u8..64).collect());
        let c = b.clone();
        assert_eq!(b.as_slice().as_ptr(), c.as_slice().as_ptr());
    }

    #[test]
    fn slice_narrows_and_rejects_overflow() {
        let b = SharedBytes::from_vec((0u8..16).collect());
        let s = b.slice(4, 4).expect("test: in-range sub-slice");
        assert_eq!(s.as_slice(), &[4, 5, 6, 7]);
        assert!(b.slice(4, 13).is_none());
        assert!(b.slice(usize::MAX, 1).is_none());
    }

    #[test]
    fn equality_against_slices_and_vecs() {
        let b = SharedBytes::from_vec(vec![7, 8, 9]);
        assert_eq!(b, vec![7u8, 8, 9]);
        assert_eq!(vec![7u8, 8, 9], b);
        assert!(b == *[7u8, 8, 9].as_slice());
        assert_eq!(b, SharedBytes::from_vec(vec![7, 8, 9]));
    }

    #[test]
    fn debug_does_not_dump_the_payload() {
        let b = SharedBytes::from_vec(vec![0xAB; 1024]);
        let rendered = format!("{b:?}");
        assert!(rendered.contains("len: 1024"), "got {rendered}");
        assert!(!rendered.contains("171"), "payload must not be dumped");
    }

    #[test]
    fn deref_gives_a_plain_slice() {
        let b = SharedBytes::from_vec(vec![1, 2, 3]);
        let s: &[u8] = &b;
        assert_eq!(s, &[1, 2, 3]);
        assert!(!b.is_empty());
        assert!(SharedBytes::empty().is_empty());
        assert!(SharedBytes::default().is_empty());
    }
}

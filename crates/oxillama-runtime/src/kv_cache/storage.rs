// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Backing storage for [`KvCache`](super::KvCache).
//!
//! Two things live here that the cache itself no longer has to know about:
//!
//! * **the element type** — FP32 (llama.cpp's `f32` KV) or FP16 (llama.cpp's
//!   *default*), selected by [`KvCacheDtype`];
//! * **block growth** — buffers start empty and grow a block at a time instead
//!   of being allocated at the full trained context up front.
//!
//! ## Why growth matters
//!
//! Before 0.1.4 [`KvCache::new`](super::KvCache::new) ran
//! `vec![0.0f32; max_seq_len * kv_dim]` per layer for both K and V,
//! unconditionally.  For Llama-3-8B at a 4096-token context that is
//! `32 layers × 4096 × 1024 × 4 bytes × 2 = 1 GiB` paid before the first token,
//! whether the conversation is 20 tokens long or 4096.  With
//! [`DEFAULT_BLOCK_TOKENS`] a 20-token exchange touches one block per layer —
//! 64 MiB — and grows only as the conversation actually grows.
//!
//! ## Why FP16 still cannot be borrowed — and why that no longer matters
//!
//! [`KvCacheAccess::get_keys`](oxillama_arch::traits::KvCacheAccess::get_keys)
//! returns `ArchResult<&[f32]>`: a **borrowed contiguous FP32 slice**.  An FP16
//! buffer cannot produce one without materialising a conversion somewhere, and
//! the trait hands out that borrow from `&self` on a `Send + Sync` type, which
//! rules out an internal scratch buffer.  So [`KvCacheDtype::F16`] answers those
//! two calls with a typed error pointing at the per-token iterators — exactly
//! the shape [`PagedKvCache`](super::PagedKvCache) already uses for multi-page
//! reads.
//!
//! Until 0.1.4 that made FP16 unreachable in practice, because every
//! architecture read the cache through `get_keys`/`get_values` directly.  They
//! now go through `oxillama_arch::common::fetch_keys` / `fetch_values`, which
//! borrow when the storage allows it and gather through `for_each_key` /
//! `for_each_value` when it does not.  FP32 therefore keeps its zero-copy fast
//! path unchanged, and FP16 works on every architecture at the cost of a
//! per-layer conversion on read.
//!
//! `for_each_key` / `for_each_value` / `copy_keys_into` work in **both** dtypes.
//! See this module's `dtype_f16_halves_memory` test for the measured ratio, and
//! `oxillama-runtime/tests/kv_dtype_reachable.rs` for the reachability contract.

use half::f16;

/// Element type used for KV storage.
///
/// Mirrors llama.cpp's `--cache-type-k` / `--cache-type-v`: `f32` is the
/// lossless default here, `f16` halves the cache at the cost of the read-path
/// restriction documented on this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheDtype {
    /// 32-bit float storage: zero conversion cost, borrowable as `&[f32]`.
    #[default]
    F32,
    /// 16-bit float storage: half the memory, per-token conversion on read.
    F16,
}

impl KvCacheDtype {
    /// Bytes occupied by one stored element.
    pub fn size_of(self) -> usize {
        match self {
            Self::F32 => core::mem::size_of::<f32>(),
            Self::F16 => core::mem::size_of::<f16>(),
        }
    }

    /// The name used in diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

/// Number of token positions added by a single growth step.
///
/// 256 positions of a 32-layer, 1024-wide KV cache is 2 MiB of K plus 2 MiB of
/// V per growth step, so the allocator is touched at most 16 times on the way
/// to a 4096-token context while a short conversation never pays for more than
/// one step.
pub const DEFAULT_BLOCK_TOKENS: usize = 256;

/// One layer's key (or value) buffer.
///
/// The `f32` / `f16` split is an enum rather than a generic so that
/// [`KvCache`](super::KvCache) stays a concrete type — it is stored in
/// `Option<KvCache>` inside the engine and handed to `&mut dyn KvCacheAccess`,
/// neither of which tolerates a type parameter.
#[derive(Debug, Clone)]
pub(crate) enum LayerBuf {
    /// FP32 elements.
    F32(Vec<f32>),
    /// FP16 elements.
    F16(Vec<f16>),
}

impl LayerBuf {
    /// Create an empty buffer of the given dtype.
    pub(crate) fn new(dtype: KvCacheDtype) -> Self {
        match dtype {
            KvCacheDtype::F32 => Self::F32(Vec::new()),
            KvCacheDtype::F16 => Self::F16(Vec::new()),
        }
    }

    /// Bytes currently allocated (capacity, not length — this is what the
    /// process actually holds).
    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            Self::F32(v) => v.capacity() * core::mem::size_of::<f32>(),
            Self::F16(v) => v.capacity() * core::mem::size_of::<f16>(),
        }
    }

    /// Grow to at least `n` elements, zero-filling the new tail.
    ///
    /// Never shrinks: a cache that has already reached a length keeps it so a
    /// long conversation does not re-allocate on every `clear()`.
    pub(crate) fn grow_to(&mut self, n: usize) {
        match self {
            Self::F32(v) => {
                if v.len() < n {
                    v.resize(n, 0.0f32);
                }
            }
            Self::F16(v) => {
                if v.len() < n {
                    v.resize(n, f16::ZERO);
                }
            }
        }
    }

    /// Write `src` at element offset `offset`.
    ///
    /// The caller guarantees `offset + src.len() <= self.len()`.
    pub(crate) fn write_at(&mut self, offset: usize, src: &[f32]) {
        match self {
            Self::F32(v) => v[offset..offset + src.len()].copy_from_slice(src),
            Self::F16(v) => {
                for (dst, &s) in v[offset..offset + src.len()].iter_mut().zip(src.iter()) {
                    *dst = f16::from_f32(s);
                }
            }
        }
    }

    /// Borrow the first `end` elements as `&[f32]`, if the dtype allows it.
    ///
    /// Returns `None` for [`KvCacheDtype::F16`] — see the module docs.
    pub(crate) fn as_f32_prefix(&self, end: usize) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(&v[..end.min(v.len())]),
            Self::F16(_) => None,
        }
    }

    /// Copy the first `end` elements into `dst` as `f32`, replacing its
    /// contents.
    pub(crate) fn copy_f32_prefix_into(&self, end: usize, dst: &mut Vec<f32>) {
        dst.clear();
        match self {
            Self::F32(v) => dst.extend_from_slice(&v[..end.min(v.len())]),
            Self::F16(v) => dst.extend(v[..end.min(v.len())].iter().map(|h| h.to_f32())),
        }
    }

    /// Materialise the first `end` elements as an owned `Vec<f32>`.
    pub(crate) fn to_f32_prefix_vec(&self, end: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.copy_f32_prefix_into(end, &mut out);
        out
    }

    /// Read one `kv_dim`-sized row starting at `offset` into `row`.
    ///
    /// `row` must already be `kv_dim` long; this avoids an allocation per
    /// position inside `for_each_key`.
    pub(crate) fn read_row_into(&self, offset: usize, row: &mut [f32]) {
        let n = row.len();
        match self {
            Self::F32(v) => row.copy_from_slice(&v[offset..offset + n]),
            Self::F16(v) => {
                for (dst, src) in row.iter_mut().zip(v[offset..offset + n].iter()) {
                    *dst = src.to_f32();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(KvCacheDtype::F32.size_of(), 4);
        assert_eq!(KvCacheDtype::F16.size_of(), 2);
        assert_eq!(KvCacheDtype::default(), KvCacheDtype::F32);
    }

    #[test]
    fn grow_to_never_shrinks() {
        let mut buf = LayerBuf::new(KvCacheDtype::F32);
        buf.grow_to(16);
        assert_eq!(buf.to_f32_prefix_vec(usize::MAX).len(), 16);
        buf.grow_to(4);
        assert_eq!(
            buf.to_f32_prefix_vec(usize::MAX).len(),
            16,
            "grow_to must never shrink"
        );
    }

    #[test]
    fn f32_round_trip_is_exact() {
        let mut buf = LayerBuf::new(KvCacheDtype::F32);
        buf.grow_to(4);
        buf.write_at(0, &[1.5, -2.25, 3.75, 0.0]);
        let got = buf.to_f32_prefix_vec(4);
        assert_eq!(got, vec![1.5, -2.25, 3.75, 0.0]);
    }

    #[test]
    fn f16_round_trip_is_within_half_precision() {
        let mut buf = LayerBuf::new(KvCacheDtype::F16);
        buf.grow_to(4);
        let src = [1.5f32, -2.25, 3.75, 0.1];
        buf.write_at(0, &src);
        let got = buf.to_f32_prefix_vec(4);
        for (i, (&g, &s)) in got.iter().zip(src.iter()).enumerate() {
            assert!(
                (g - s).abs() <= 1e-3 * s.abs().max(1.0),
                "element {i}: f16 round trip {s} -> {g}"
            );
        }
    }

    #[test]
    fn f16_cannot_be_borrowed_as_f32() {
        let mut buf = LayerBuf::new(KvCacheDtype::F16);
        buf.grow_to(4);
        assert!(buf.as_f32_prefix(4).is_none());
        let mut f32buf = LayerBuf::new(KvCacheDtype::F32);
        f32buf.grow_to(4);
        assert!(f32buf.as_f32_prefix(4).is_some());
    }

    #[test]
    fn dtype_f16_halves_memory() {
        let n = 4096usize;
        let mut a = LayerBuf::new(KvCacheDtype::F32);
        let mut b = LayerBuf::new(KvCacheDtype::F16);
        a.grow_to(n);
        b.grow_to(n);
        assert_eq!(
            b.memory_bytes() * 2,
            a.memory_bytes(),
            "f16 storage must be exactly half of f32"
        );
    }

    #[test]
    fn read_row_into_matches_prefix() {
        for dtype in [KvCacheDtype::F32, KvCacheDtype::F16] {
            let mut buf = LayerBuf::new(dtype);
            buf.grow_to(6);
            buf.write_at(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let mut row = vec![0.0f32; 3];
            buf.read_row_into(3, &mut row);
            assert_eq!(row, vec![4.0, 5.0, 6.0], "dtype {}", dtype.as_str());
        }
    }
}

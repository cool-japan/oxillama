// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! dtype-agnostic reads of a [`KvCacheAccess`] layer.
//!
//! Every attention kernel in this crate wants the same thing: one contiguous
//! `[seq_len × kv_dim]` slice of `f32` keys and one of values.  Until 0.1.4 they
//! all obtained it by calling
//! [`get_keys`](KvCacheAccess::get_keys) / [`get_values`](KvCacheAccess::get_values),
//! which hand out a **borrow** — and a borrow is only possible when the cache
//! already stores `f32` in one contiguous run.  That excluded two real cache
//! layouts:
//!
//! * `KvCache` with `KvCacheDtype::F16` — half the memory, but the elements are
//!   `f16`, so there is no `&[f32]` to lend;
//! * `PagedKvCache` with a sequence spanning more than one page — the data
//!   exists as `f32` but not contiguously.
//!
//! Both of those answer `get_keys()` with a typed error, so simply *enabling*
//! them broke every architecture. [`fetch_keys`] / [`fetch_values`] close that
//! gap without costing the common case anything:
//!
//! * **f32, contiguous** → [`Cow::Borrowed`] straight from `get_keys()`.  Same
//!   pointer, same length, no copy, no allocation — the attention loop that
//!   consumes it is byte-for-byte the same code it was before.
//! * **anything else** → [`Cow::Owned`], gathered one token row at a time
//!   through [`for_each_key`](KvCacheAccess::for_each_key), which every cache
//!   implements (converting `f16 → f32` or walking pages as needed).
//!
//! Accumulation is unaffected: attention still runs in `f32`.  Only *storage*
//! narrows, which is exactly llama.cpp's arrangement — its KV cache defaults to
//! `f16` while the dot products stay `f32`.

use std::borrow::Cow;

use crate::error::ArchResult;
use crate::traits::KvCacheAccess;

/// Number of elements to pre-reserve when a gather is required.
///
/// `seq_len()` is the position of the token currently being written, so the
/// cache holds `seq_len() + 1` rows by the time attention reads it back
/// (`store_kv` publishes the new row before `advance()` bumps the position).
/// Over-reserving by one row is far cheaper than a reallocation mid-gather.
fn gather_capacity(cache: &dyn KvCacheAccess) -> usize {
    cache
        .seq_len()
        .saturating_add(1)
        .saturating_mul(cache.kv_dim())
}

/// All cached keys for `layer` as a contiguous `f32` slice.
///
/// Borrows when the cache can lend one; otherwise gathers into an owned buffer.
///
/// # Errors
///
/// When neither access path works, the error from
/// [`get_keys`](KvCacheAccess::get_keys) is returned unchanged, so callers see
/// exactly the diagnostic they saw before this helper existed.
pub fn fetch_keys(cache: &dyn KvCacheAccess, layer: usize) -> ArchResult<Cow<'_, [f32]>> {
    // Fast path: FP32 contiguous storage lends its buffer directly.
    let borrow_err = match cache.get_keys(layer) {
        Ok(keys) => return Ok(Cow::Borrowed(keys)),
        Err(e) => e,
    };
    let mut out: Vec<f32> = Vec::with_capacity(gather_capacity(cache));
    match cache.for_each_key(layer, &mut |_pos, row| out.extend_from_slice(row)) {
        Ok(()) => Ok(Cow::Owned(out)),
        // Neither path works — report the original borrow failure, which names
        // the dtype/layout, rather than the generic "kv_dim() not implemented".
        Err(_) => Err(borrow_err),
    }
}

/// All cached values for `layer` as a contiguous `f32` slice.
///
/// The value-side twin of [`fetch_keys`]; see it for the borrow/gather rules.
///
/// # Errors
///
/// Propagates the [`get_values`](KvCacheAccess::get_values) error when the
/// per-token iterator cannot serve the layer either.
pub fn fetch_values(cache: &dyn KvCacheAccess, layer: usize) -> ArchResult<Cow<'_, [f32]>> {
    let borrow_err = match cache.get_values(layer) {
        Ok(values) => return Ok(Cow::Borrowed(values)),
        Err(e) => e,
    };
    let mut out: Vec<f32> = Vec::with_capacity(gather_capacity(cache));
    match cache.for_each_value(layer, &mut |_pos, row| out.extend_from_slice(row)) {
        Ok(()) => Ok(Cow::Owned(out)),
        Err(_) => Err(borrow_err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ArchError;

    /// An f32 cache that lends its buffers — the fast path.
    struct BorrowCache {
        keys: Vec<f32>,
        values: Vec<f32>,
        kv_dim: usize,
    }

    impl KvCacheAccess for BorrowCache {
        fn seq_len(&self) -> usize {
            self.keys.len() / self.kv_dim
        }
        fn store_kv(&mut self, _layer: usize, _key: &[f32], _value: &[f32]) -> ArchResult<()> {
            Ok(())
        }
        fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.keys)
        }
        fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.values)
        }
        fn advance(&mut self) {}
        fn kv_dim(&self) -> usize {
            self.kv_dim
        }
    }

    /// A cache that refuses to lend (stands in for FP16 / paged storage) but
    /// serves the per-token iterators.
    struct IterOnlyCache {
        keys: Vec<f32>,
        values: Vec<f32>,
        kv_dim: usize,
    }

    impl KvCacheAccess for IterOnlyCache {
        fn seq_len(&self) -> usize {
            self.keys.len() / self.kv_dim
        }
        fn store_kv(&mut self, _layer: usize, _key: &[f32], _value: &[f32]) -> ArchResult<()> {
            Ok(())
        }
        fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
            Err(ArchError::ForwardPassError {
                layer,
                message: "cannot borrow keys".to_string(),
            })
        }
        fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
            Err(ArchError::ForwardPassError {
                layer,
                message: "cannot borrow values".to_string(),
            })
        }
        fn advance(&mut self) {}
        fn kv_dim(&self) -> usize {
            self.kv_dim
        }
        fn for_each_key(&self, _layer: usize, f: &mut dyn FnMut(usize, &[f32])) -> ArchResult<()> {
            for (pos, row) in self.keys.chunks_exact(self.kv_dim).enumerate() {
                f(pos, row);
            }
            Ok(())
        }
        fn for_each_value(
            &self,
            _layer: usize,
            f: &mut dyn FnMut(usize, &[f32]),
        ) -> ArchResult<()> {
            for (pos, row) in self.values.chunks_exact(self.kv_dim).enumerate() {
                f(pos, row);
            }
            Ok(())
        }
    }

    /// A cache that supports neither path.
    struct DeadCache;

    impl KvCacheAccess for DeadCache {
        fn seq_len(&self) -> usize {
            0
        }
        fn store_kv(&mut self, _layer: usize, _key: &[f32], _value: &[f32]) -> ArchResult<()> {
            Ok(())
        }
        fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
            Err(ArchError::ForwardPassError {
                layer,
                message: "borrow refused: distinctive marker".to_string(),
            })
        }
        fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
            Err(ArchError::ForwardPassError {
                layer,
                message: "borrow refused: distinctive marker".to_string(),
            })
        }
        fn advance(&mut self) {}
        // kv_dim() stays 0, so the default for_each_* fails too.
    }

    #[test]
    fn f32_cache_is_borrowed_not_copied() {
        let cache = BorrowCache {
            keys: vec![1.0, 2.0, 3.0, 4.0],
            values: vec![5.0, 6.0, 7.0, 8.0],
            kv_dim: 2,
        };
        let keys = fetch_keys(&cache, 0).expect("keys");
        assert!(matches!(keys, Cow::Borrowed(_)), "f32 path must not copy");
        assert_eq!(keys.as_ptr(), cache.keys.as_ptr(), "same buffer, no copy");
        let values = fetch_values(&cache, 0).expect("values");
        assert!(matches!(values, Cow::Borrowed(_)));
        assert_eq!(values.as_ptr(), cache.values.as_ptr());
    }

    #[test]
    fn non_borrowable_cache_is_gathered() {
        let cache = IterOnlyCache {
            keys: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            values: vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            kv_dim: 3,
        };
        let keys = fetch_keys(&cache, 0).expect("keys");
        assert!(matches!(keys, Cow::Owned(_)), "must gather");
        assert_eq!(&*keys, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let values = fetch_values(&cache, 0).expect("values");
        assert_eq!(&*values, &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
    }

    #[test]
    fn unreadable_cache_reports_the_borrow_error() {
        let cache = DeadCache;
        let err = fetch_keys(&cache, 3).expect_err("must fail");
        assert!(
            err.to_string().contains("distinctive marker"),
            "the original borrow error must survive, got: {err}"
        );
        let err = fetch_values(&cache, 3).expect_err("must fail");
        assert!(err.to_string().contains("distinctive marker"), "{err}");
    }

    #[test]
    fn empty_cache_gathers_nothing() {
        let cache = IterOnlyCache {
            keys: Vec::new(),
            values: Vec::new(),
            kv_dim: 4,
        };
        assert!(fetch_keys(&cache, 0).expect("keys").is_empty());
        assert!(fetch_values(&cache, 0).expect("values").is_empty());
    }
}

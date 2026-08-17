//! Key-Value cache for transformer attention.
//!
//! Stores the key and value tensors from previous tokens so they don't
//! need to be recomputed during autoregressive generation.
//!
//! Three implementations are provided:
//! - [`KvCache`]: Simple contiguous pre-allocated buffers (fast, simple)
//! - [`PagedKvCache`]: Page-based allocation (memory-efficient, supports variable lengths)
//! - [`PrefixKvCache`]: Radix-tree prefix sharing (reuse cached prefixes across requests)

pub mod paged;
pub mod prefix;
pub mod storage;

use oxicode::{Decode, Encode};
use oxillama_arch::traits::KvCacheAccess;
use oxillama_arch::{ArchError, ArchResult};

use crate::error::{RuntimeError, RuntimeResult};
use storage::LayerBuf;

pub use oxillama_arch::traits::{BatchedKvView, KvSlot};
pub use paged::PagedKvCache;
pub use prefix::{PrefixCacheConfig, PrefixKvCache};
pub use storage::{KvCacheDtype, DEFAULT_BLOCK_TOKENS};

/// A point-in-time snapshot of a [`KvCache`] state.
///
/// Created by [`KvCache::snapshot`] and restored via
/// [`KvCache::restore_from_snapshot`].  Used to checkpoint and restore KV state,
/// for example by the engine's `kv_snapshot` / `kv_restore` helpers.
///
/// Speculative decoding instead rolls the draft cache back with the cheaper
/// `O(1)` [`KvCache::truncate`] primitive (see
/// [`crate::speculative::SpeculativeDeltaSync`]).
#[derive(Debug, Clone, Encode, Decode)]
pub struct KvCacheSnapshot {
    /// Per-layer key vectors, each of length `seq_len * kv_dim`.
    pub keys: Vec<Vec<f32>>,
    /// Per-layer value vectors, each of length `seq_len * kv_dim`.
    pub values: Vec<Vec<f32>>,
    /// The sequence length at snapshot time.
    pub seq_len: usize,
}

/// Simple contiguous `BatchedKvView` backed by a `Vec<KvSlot>` paired with
/// a pool of flat key/value buffers.
///
/// Each entry `i` refers to slot `slots[i]`.  The key buffer for slot `i`
/// has length `position * kv_dim` and the value buffer likewise.
pub struct VecBatchedKvView {
    slots: Vec<KvSlot>,
    /// Flat key buffers, one per slot, length `position * kv_dim`.
    keys: Vec<Vec<f32>>,
    /// Flat value buffers, one per slot, length `position * kv_dim`.
    values: Vec<Vec<f32>>,
}

impl VecBatchedKvView {
    /// Construct a new `VecBatchedKvView` from parallel vecs.
    ///
    /// # Panics
    ///
    /// Panics if `slots.len() != keys.len()` or `slots.len() != values.len()`.
    pub fn new(slots: Vec<KvSlot>, keys: Vec<Vec<f32>>, values: Vec<Vec<f32>>) -> Self {
        assert_eq!(
            slots.len(),
            keys.len(),
            "slots and keys vecs must have equal length"
        );
        assert_eq!(
            slots.len(),
            values.len(),
            "slots and values vecs must have equal length"
        );
        Self {
            slots,
            keys,
            values,
        }
    }
}

impl BatchedKvView for VecBatchedKvView {
    fn slot_count(&self) -> usize {
        self.slots.len()
    }

    fn kv_for_slot(&self, slot: usize) -> (&[f32], &[f32]) {
        (&self.keys[slot], &self.values[slot])
    }

    fn position(&self, slot: usize) -> usize {
        self.slots[slot].position
    }
}

/// Simple contiguous KV cache implementation.
///
/// Stores key and value tensors for all layers in contiguous per-layer buffers.
///
/// # Growth
///
/// Buffers start **empty** and grow one [`DEFAULT_BLOCK_TOKENS`]-sized block at
/// a time, up to `max_seq_len`.  Before 0.1.4 the constructor allocated the
/// full trained context immediately — 1 GiB for Llama-3-8B at a 4096-token
/// context, paid whether the conversation ran to 20 tokens or 4096.  A cache
/// never shrinks, so a long conversation allocates once and every subsequent
/// [`clear`](Self::clear) reuses the same memory.
///
/// # Element type
///
/// FP32 by default.  [`KvCacheDtype::F16`] halves the footprint.  It cannot
/// serve [`KvCacheAccess::get_keys`], which hands out a borrowed `&[f32]`, but
/// every architecture reads through `oxillama_arch::common::fetch_keys` /
/// `fetch_values`, which fall back to [`for_each_key`](KvCacheAccess::for_each_key)
/// when a borrow is impossible — so FP16 is selectable end-to-end via
/// [`EngineConfig::kv_dtype`](crate::engine::EngineConfig::kv_dtype) or
/// `oxillama run --kv-dtype f16`.  See [`storage`] for the full rationale.
pub struct KvCache {
    /// Key buffers, one per layer.  Each holds `capacity_tokens * kv_dim`
    /// elements, which grows towards `max_seq_len * kv_dim`.
    keys: Vec<LayerBuf>,
    /// Value buffers, one per layer, sized exactly like `keys`.
    values: Vec<LayerBuf>,
    /// Current sequence length (number of fully-committed tokens).
    seq_len: usize,
    /// Number of token positions that have had K/V data written.
    ///
    /// Invariant: `stored_len >= seq_len`.  Between a `store_kv` call at
    /// position `seq_len` and the subsequent `advance()`, `stored_len ==
    /// seq_len + 1` so that attention can immediately read the just-written
    /// entry without requiring `advance()` to have been called first.
    stored_len: usize,
    /// Token positions currently backed by allocated buffer space.
    ///
    /// Invariant: `capacity_tokens >= stored_len` and
    /// `capacity_tokens <= max_seq_len`.
    capacity_tokens: usize,
    /// Token positions added per growth step.
    block_tokens: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// KV dimension per token (num_kv_heads * head_dim).
    kv_dim: usize,
    /// Number of layers.
    num_layers: usize,
    /// Element type of the backing buffers.
    dtype: KvCacheDtype,
}

impl KvCache {
    /// Allocate a new FP32 KV cache.
    ///
    /// No token storage is allocated up front — buffers grow on demand, see
    /// the type-level docs.
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers.
    /// * `max_seq_len` - Maximum context length.
    /// * `kv_dim` - KV dimension per token (num_kv_heads * head_dim).
    pub fn new(num_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        Self::with_dtype(num_layers, max_seq_len, kv_dim, KvCacheDtype::F32)
    }

    /// Allocate a new KV cache with an explicit element type.
    ///
    /// See [`KvCacheDtype`] for the trade-off; [`KvCacheDtype::F16`] halves the
    /// footprint but makes [`KvCacheAccess::get_keys`] return an error.
    pub fn with_dtype(
        num_layers: usize,
        max_seq_len: usize,
        kv_dim: usize,
        dtype: KvCacheDtype,
    ) -> Self {
        let keys = (0..num_layers).map(|_| LayerBuf::new(dtype)).collect();
        let values = (0..num_layers).map(|_| LayerBuf::new(dtype)).collect();
        let block_tokens = DEFAULT_BLOCK_TOKENS.clamp(1, max_seq_len.max(1));

        Self {
            keys,
            values,
            seq_len: 0,
            stored_len: 0,
            capacity_tokens: 0,
            block_tokens,
            max_seq_len,
            kv_dim,
            num_layers,
            dtype,
        }
    }

    /// The element type of the backing buffers.
    pub fn dtype(&self) -> KvCacheDtype {
        self.dtype
    }

    /// Token positions currently backed by allocated memory.
    ///
    /// Grows in [`DEFAULT_BLOCK_TOKENS`] steps as the sequence lengthens and
    /// never shrinks, so this is the high-water mark of the cache.
    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    /// Bytes currently held by the key and value buffers across all layers.
    ///
    /// This is what the process actually owns, not what `max_seq_len` would
    /// eventually require.
    pub fn memory_bytes(&self) -> usize {
        self.keys
            .iter()
            .chain(self.values.iter())
            .map(|b| b.memory_bytes())
            .sum()
    }

    /// Bytes this cache would hold once the full context is reached.
    pub fn max_memory_bytes(&self) -> usize {
        self.num_layers * self.max_seq_len * self.kv_dim * self.dtype.size_of() * 2
    }

    /// Ensure every layer buffer covers at least `tokens` positions.
    ///
    /// Rounds up to the next block so growth happens `max_seq_len /
    /// block_tokens` times at most over the life of the cache.
    fn ensure_capacity(&mut self, tokens: usize) {
        if tokens <= self.capacity_tokens {
            return;
        }
        let blocks = tokens.div_ceil(self.block_tokens);
        let target = (blocks * self.block_tokens)
            .min(self.max_seq_len)
            .max(tokens);
        let elems = target * self.kv_dim;
        for buf in self.keys.iter_mut().chain(self.values.iter_mut()) {
            buf.grow_to(elems);
        }
        self.capacity_tokens = target;
        tracing::trace!(
            tokens,
            capacity_tokens = target,
            bytes = self.memory_bytes(),
            "KV cache grown"
        );
    }

    /// Reset the cache, invalidating all stored KV pairs.
    ///
    /// This is an `O(1)` bookkeeping reset: the buffers are **not** zeroed.
    /// Zeroing them was pure waste — [`get_keys`](KvCacheAccess::get_keys) and
    /// [`get_values`](KvCacheAccess::get_values) are bounded by `stored_len`,
    /// so nothing past the reset point is reachable, yet `clear()` was
    /// memsetting the whole pre-allocated region (~1 GiB for Llama-3-8B at a
    /// 4096-token context) on every request and on every beam-search step.
    ///
    /// The allocation itself is retained so the next sequence reuses it.
    pub fn clear(&mut self) {
        self.seq_len = 0;
        self.stored_len = 0;
    }

    /// Reset the cache **and** release its buffers back to the allocator.
    ///
    /// [`clear`](Self::clear) deliberately keeps the allocation for reuse; this
    /// is the variant for a caller that is done with a model and wants the
    /// memory back without dropping the cache itself.
    pub fn clear_and_release(&mut self) {
        self.seq_len = 0;
        self.stored_len = 0;
        self.capacity_tokens = 0;
        for buf in self.keys.iter_mut().chain(self.values.iter_mut()) {
            *buf = LayerBuf::new(self.dtype);
        }
    }

    /// Returns the maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Returns the KV dimension per token.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Advance the sequence position by one token.
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
            if self.stored_len < self.seq_len {
                self.stored_len = self.seq_len;
            }
        }
    }

    /// Restore from a prefix cache snapshot.
    ///
    /// Copies the provided per-layer key/value data into internal buffers and
    /// sets `seq_len` to the snapshot's length.
    ///
    /// # Errors
    ///
    /// * [`RuntimeError::KvCacheFull`] when `seq_len` exceeds `max_seq_len`.
    /// * [`RuntimeError::SnapshotIncompatible`] when the snapshot has fewer
    ///   layers than the cache, or when any layer carries fewer than
    ///   `seq_len * kv_dim` elements.
    ///
    /// That second check is the whole point of this returning a `Result`.  The
    /// old signature copied `min(seq_len * kv_dim, src.len())` elements and then
    /// set `self.seq_len = seq_len` regardless, so a short snapshot marked
    /// positions **valid** that had never been written.  While `clear()` still
    /// memset the buffers those positions read as zeros; now that `clear()` is
    /// an `O(1)` reset they would read the *previous sequence's* keys, which is
    /// cross-request contamination rather than merely wrong logits.
    pub fn restore_from_snapshot(
        &mut self,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        seq_len: usize,
    ) -> RuntimeResult<()> {
        if seq_len > self.max_seq_len {
            return Err(RuntimeError::KvCacheFull {
                max_ctx: self.max_seq_len,
            });
        }
        if keys.len() < self.num_layers || values.len() < self.num_layers {
            return Err(RuntimeError::SnapshotIncompatible {
                detail: format!(
                    "snapshot has {} key / {} value layers, cache has {}",
                    keys.len(),
                    values.len(),
                    self.num_layers
                ),
            });
        }

        let copy_len = seq_len * self.kv_dim;
        for layer in 0..self.num_layers {
            if keys[layer].len() < copy_len || values[layer].len() < copy_len {
                return Err(RuntimeError::SnapshotIncompatible {
                    detail: format!(
                        "layer {layer} of the snapshot holds {} key / {} value floats but \
                         restoring {seq_len} tokens at kv_dim {} needs {copy_len}",
                        keys[layer].len(),
                        values[layer].len(),
                        self.kv_dim
                    ),
                });
            }
        }

        self.ensure_capacity(seq_len);
        for layer in 0..self.num_layers {
            self.keys[layer].write_at(0, &keys[layer][..copy_len]);
            self.values[layer].write_at(0, &values[layer][..copy_len]);
        }

        self.seq_len = seq_len;
        self.stored_len = seq_len;
        Ok(())
    }

    /// Truncate the KV cache to `n` tokens.
    ///
    /// After this call `seq_len()` returns `n` (clamped to the current
    /// `seq_len` if `n` is already beyond it — truncate never extends the
    /// cache).  The underlying buffers are **not** zeroed; the truncated
    /// region is simply considered invalid and will be overwritten on the
    /// next `store_kv` call.
    ///
    /// This is the low-level primitive for speculative-decoding rollback: the
    /// target engine calls `truncate(divergence_pos)` after rejecting a draft
    /// token, then continues generating from `divergence_pos`.
    pub fn truncate(&mut self, n: usize) {
        let n = n.min(self.seq_len);
        self.seq_len = n;
        self.stored_len = n;
    }

    /// Capture a snapshot of the current KV state.
    ///
    /// Only the data up to `seq_len * kv_dim` is copied per layer, keeping
    /// the snapshot compact.
    pub fn snapshot(&self) -> KvCacheSnapshot {
        self.snapshot_truncated(self.seq_len)
    }

    /// Capture a snapshot covering only the first `n` token positions.
    ///
    /// `n` is clamped to the current `seq_len`.  This is what a prefix cache
    /// wants: the trie key is the *prompt*, so retaining the completion's KV
    /// alongside it wastes memory and breaks the "snapshot length == key
    /// length" invariant that [`restore_from_snapshot`](Self::restore_from_snapshot)
    /// now enforces.
    pub fn snapshot_truncated(&self, n: usize) -> KvCacheSnapshot {
        let seq_len = n.min(self.seq_len);
        let copy_len = seq_len * self.kv_dim;
        let keys = self
            .keys
            .iter()
            .map(|k| k.to_f32_prefix_vec(copy_len))
            .collect();
        let values = self
            .values
            .iter()
            .map(|v| v.to_f32_prefix_vec(copy_len))
            .collect();
        KvCacheSnapshot {
            keys,
            values,
            seq_len,
        }
    }

    /// Build a serializable [`crate::snapshot::KvStatePayload`] from the current state.
    pub fn to_payload(&self) -> crate::snapshot::KvStatePayload {
        let copy_len = self.seq_len * self.kv_dim;
        let keys = self
            .keys
            .iter()
            .map(|k| k.to_f32_prefix_vec(copy_len))
            .collect();
        let values = self
            .values
            .iter()
            .map(|v| v.to_f32_prefix_vec(copy_len))
            .collect();
        crate::snapshot::KvStatePayload {
            keys,
            values,
            seq_len: self.seq_len,
            num_layers: self.num_layers,
            max_seq_len: self.max_seq_len,
            kv_dim: self.kv_dim,
        }
    }

    /// Restore cache state from a [`crate::snapshot::KvStatePayload`].
    ///
    /// Validates that layer count and dimensions match the cache configuration,
    /// then restores the key/value buffers and sequence length.
    pub fn restore_from_payload(
        &mut self,
        payload: &crate::snapshot::KvStatePayload,
    ) -> crate::error::RuntimeResult<()> {
        if payload.num_layers != self.num_layers {
            return Err(RuntimeError::SnapshotIncompatible {
                detail: format!(
                    "layer count mismatch: snapshot has {}, cache has {}",
                    payload.num_layers, self.num_layers
                ),
            });
        }
        if payload.kv_dim != self.kv_dim {
            return Err(RuntimeError::SnapshotIncompatible {
                detail: format!(
                    "kv_dim mismatch: snapshot has {}, cache has {}",
                    payload.kv_dim, self.kv_dim
                ),
            });
        }
        self.restore_from_snapshot(&payload.keys, &payload.values, payload.seq_len)
    }

    /// Copy every cached key for `layer` into `dst` as `f32`.
    ///
    /// Works in **both** dtypes, unlike
    /// [`get_keys`](KvCacheAccess::get_keys), which can only hand out a borrow
    /// when the storage is already FP32.
    ///
    /// # Errors
    ///
    /// [`ArchError::ForwardPassError`] when `layer` is out of range.
    pub fn copy_keys_into(&self, layer: usize, dst: &mut Vec<f32>) -> ArchResult<()> {
        if layer >= self.num_layers {
            return Err(ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        self.keys[layer].copy_f32_prefix_into(self.stored_len * self.kv_dim, dst);
        Ok(())
    }

    /// Copy every cached value for `layer` into `dst` as `f32`.
    ///
    /// # Errors
    ///
    /// [`ArchError::ForwardPassError`] when `layer` is out of range.
    pub fn copy_values_into(&self, layer: usize, dst: &mut Vec<f32>) -> ArchResult<()> {
        if layer >= self.num_layers {
            return Err(ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        self.values[layer].copy_f32_prefix_into(self.stored_len * self.kv_dim, dst);
        Ok(())
    }

    /// Shared body of `for_each_key` / `for_each_value`.
    ///
    /// One reusable `kv_dim`-sized row buffer is allocated per call rather than
    /// one per position, and FP16 storage is converted into it on the way out.
    fn for_each_row(
        &self,
        layer: usize,
        keys: bool,
        f: &mut dyn FnMut(usize, &[f32]),
    ) -> ArchResult<()> {
        if layer >= self.num_layers {
            return Err(ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        if self.kv_dim == 0 {
            return Ok(());
        }
        let buf = if keys {
            &self.keys[layer]
        } else {
            &self.values[layer]
        };
        // FP32 storage can hand the caller the real slice with no copy at all.
        if let Some(all) = buf.as_f32_prefix(self.stored_len * self.kv_dim) {
            for (pos, row) in all.chunks_exact(self.kv_dim).enumerate() {
                f(pos, row);
            }
            return Ok(());
        }
        let mut row = vec![0.0f32; self.kv_dim];
        for pos in 0..self.stored_len {
            buf.read_row_into(pos * self.kv_dim, &mut row);
            f(pos, &row);
        }
        Ok(())
    }
}

/// The error returned when a borrowed `&[f32]` is requested from non-FP32
/// storage.
///
/// Deliberately verbose: it names the dtype, the accessor that failed, and the
/// two APIs that do work, because this is the error an architecture author will
/// hit the first time they run against an FP16 cache.
fn dtype_borrow_error(layer: usize, dtype: KvCacheDtype, accessor: &str, what: &str) -> ArchError {
    ArchError::ForwardPassError {
        layer,
        message: format!(
            "KV cache stores {what} as {} — `{accessor}()` can only borrow `&[f32]` from f32 \
             storage; use `for_each_{}()` or `copy_{}_into(&mut buf)` instead",
            dtype.as_str(),
            if what == "keys" { "key" } else { "value" },
            what,
        ),
    }
}

impl KvCacheAccess for KvCache {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Store one token's K/V for `layer` at the current position.
    ///
    /// # Errors
    ///
    /// * The position is at or past `max_seq_len`.  This used to be silent: the
    ///   body was guarded by `if end <= buffer.len()` and fell through to
    ///   `Ok(())`, so once the context filled up every subsequent token's K/V
    ///   was **discarded** while generation carried on producing logits from a
    ///   frozen cache.
    /// * `key` or `value` is shorter than `kv_dim`.  That used to panic inside
    ///   `copy_from_slice`, taking the process down on an architecture bug
    ///   instead of surfacing one.
    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        if layer >= self.num_layers {
            return Err(ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        if self.seq_len >= self.max_seq_len {
            return Err(ArchError::ForwardPassError {
                layer,
                message: format!(
                    "KV cache full: position {} reaches the maximum context length {}",
                    self.seq_len, self.max_seq_len
                ),
            });
        }
        if key.len() < self.kv_dim || value.len() < self.kv_dim {
            return Err(ArchError::ForwardPassError {
                layer,
                message: format!(
                    "store_kv received a {}-element key and a {}-element value, but kv_dim is {}",
                    key.len(),
                    value.len(),
                    self.kv_dim
                ),
            });
        }

        self.ensure_capacity(self.seq_len + 1);
        let offset = self.seq_len * self.kv_dim;
        self.keys[layer].write_at(offset, &key[..self.kv_dim]);
        self.values[layer].write_at(offset, &value[..self.kv_dim]);
        // Ensure get_keys/get_values can see the entry we just wrote even
        // before advance() is called (advance is called once per token
        // after ALL layers have written their K/V, but attention reads
        // back during the same forward pass).
        if self.stored_len <= self.seq_len {
            self.stored_len = self.seq_len + 1;
        }

        Ok(())
    }

    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        if layer >= self.num_layers {
            return Err(ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        let end = self.stored_len * self.kv_dim;
        self.keys[layer]
            .as_f32_prefix(end)
            .ok_or_else(|| dtype_borrow_error(layer, self.dtype, "get_keys", "keys"))
    }

    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        if layer >= self.num_layers {
            return Err(ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        let end = self.stored_len * self.kv_dim;
        self.values[layer]
            .as_f32_prefix(end)
            .ok_or_else(|| dtype_borrow_error(layer, self.dtype, "get_values", "values"))
    }

    fn for_each_key(&self, layer: usize, f: &mut dyn FnMut(usize, &[f32])) -> ArchResult<()> {
        self.for_each_row(layer, true, f)
    }

    fn for_each_value(&self, layer: usize, f: &mut dyn FnMut(usize, &[f32])) -> ArchResult<()> {
        self.for_each_row(layer, false, f)
    }

    fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
            // stored_len must always be >= seq_len.
            if self.stored_len < self.seq_len {
                self.stored_len = self.seq_len;
            }
        }
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_at_zero_seq_len() {
        let cache = KvCache::new(4, 128, 64);
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_new_stores_dimensions() {
        let cache = KvCache::new(8, 512, 128);
        assert_eq!(cache.num_layers(), 8);
        assert_eq!(cache.max_seq_len(), 512);
        assert_eq!(cache.kv_dim(), 128);
    }

    // ── advance ──────────────────────────────────────────────────────────────

    #[test]
    fn test_advance_increments_seq_len() {
        let mut cache = KvCache::new(2, 8, 4);
        assert_eq!(cache.seq_len(), 0);
        cache.advance();
        assert_eq!(cache.seq_len(), 1);
        cache.advance();
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn test_advance_capped_at_max_seq_len() {
        let max = 3;
        let mut cache = KvCache::new(1, max, 4);
        for _ in 0..max + 5 {
            cache.advance();
        }
        assert_eq!(cache.seq_len(), max, "seq_len must not exceed max_seq_len");
    }

    #[test]
    fn test_kvcache_access_advance_also_increments() {
        let mut cache = KvCache::new(2, 8, 4);
        // KvCacheAccess::advance should behave identically.
        <KvCache as KvCacheAccess>::advance(&mut cache);
        assert_eq!(cache.seq_len(), 1);
    }

    // ── clear ────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_seq_len_to_zero() {
        let mut cache = KvCache::new(2, 8, 4);
        cache.advance();
        cache.advance();
        assert_eq!(cache.seq_len(), 2);
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }

    /// `clear()` no longer zeroes the buffers — it is an `O(1)` bookkeeping
    /// reset — so what is asserted here is *unreachability*, not zeroing.
    #[test]
    fn test_clear_makes_stored_data_unreachable() {
        let kv_dim = 4;
        let mut cache = KvCache::new(1, 8, kv_dim);

        // Write some data and advance.
        let key = vec![1.0f32, 2.0, 3.0, 4.0];
        let val = vec![5.0f32, 6.0, 7.0, 8.0];
        cache
            .store_kv(0, &key, &val)
            .expect("store_kv must succeed");
        cache.advance();

        cache.clear();

        // After clear the seq_len is 0, so get_keys returns empty slice.
        let keys = cache.get_keys(0).expect("get_keys must succeed");
        assert!(
            keys.is_empty(),
            "after clear, get_keys should return empty slice"
        );
    }

    // ── store_kv / get_keys / get_values round-trip ───────────────────────

    #[test]
    fn test_store_kv_and_get_keys_round_trip() {
        let kv_dim = 8;
        let mut cache = KvCache::new(2, 16, kv_dim);

        let key: Vec<f32> = (0..kv_dim as i32).map(|i| i as f32 * 0.1).collect();
        let val: Vec<f32> = (0..kv_dim as i32).map(|i| i as f32 * -0.1).collect();

        cache.store_kv(0, &key, &val).expect("store_kv layer 0");
        cache.advance();

        let stored_keys = cache.get_keys(0).expect("get_keys layer 0");
        assert_eq!(stored_keys.len(), kv_dim, "should have kv_dim floats");
        for (i, (&got, &expected)) in stored_keys.iter().zip(key.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-7,
                "key[{i}]: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_store_kv_and_get_values_round_trip() {
        let kv_dim = 4;
        let mut cache = KvCache::new(1, 8, kv_dim);

        let key = vec![0.0f32; kv_dim];
        let val = vec![1.1f32, 2.2, 3.3, 4.4];

        cache.store_kv(0, &key, &val).expect("store_kv");
        cache.advance();

        let stored_vals = cache.get_values(0).expect("get_values");
        assert_eq!(stored_vals.len(), kv_dim);
        for (i, (&got, &expected)) in stored_vals.iter().zip(val.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "value[{i}]: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_store_kv_accumulates_across_tokens() {
        let kv_dim = 2;
        let mut cache = KvCache::new(1, 8, kv_dim);

        for t in 0..3u32 {
            let key = vec![t as f32, t as f32 + 0.5];
            let val = vec![0.0f32; kv_dim];
            cache.store_kv(0, &key, &val).expect("store_kv");
            cache.advance();
        }

        let keys = cache.get_keys(0).expect("get_keys");
        assert_eq!(
            keys.len(),
            3 * kv_dim,
            "should have 3 tokens × kv_dim floats"
        );
        // Verify first token keys.
        assert!((keys[0] - 0.0).abs() < 1e-7);
        assert!((keys[1] - 0.5).abs() < 1e-7);
        // Verify second token keys.
        assert!((keys[2] - 1.0).abs() < 1e-7);
    }

    // ── out-of-range layer errors ────────────────────────────────────────────

    #[test]
    fn test_store_kv_out_of_range_layer_returns_error() {
        let mut cache = KvCache::new(2, 8, 4);
        let key = vec![0.0f32; 4];
        let val = vec![0.0f32; 4];
        let result = cache.store_kv(99, &key, &val);
        assert!(result.is_err(), "out-of-range layer should return error");
    }

    #[test]
    fn test_get_keys_out_of_range_layer_returns_error() {
        let cache = KvCache::new(2, 8, 4);
        let result = cache.get_keys(99);
        assert!(result.is_err(), "out-of-range layer should return error");
    }

    #[test]
    fn test_get_values_out_of_range_layer_returns_error() {
        let cache = KvCache::new(2, 8, 4);
        let result = cache.get_values(99);
        assert!(result.is_err(), "out-of-range layer should return error");
    }

    // ── multi-layer independence ─────────────────────────────────────────────

    #[test]
    fn test_store_kv_different_layers_independent() {
        let kv_dim = 4;
        let mut cache = KvCache::new(2, 8, kv_dim);

        let key0 = vec![1.0f32; kv_dim];
        let key1 = vec![2.0f32; kv_dim];
        let val0 = vec![3.0f32; kv_dim];
        let val1 = vec![4.0f32; kv_dim];

        cache.store_kv(0, &key0, &val0).expect("layer 0 store");
        cache.store_kv(1, &key1, &val1).expect("layer 1 store");
        cache.advance();

        let stored0 = cache.get_keys(0).expect("layer 0 keys");
        let stored1 = cache.get_keys(1).expect("layer 1 keys");

        for &v in stored0 {
            assert!((v - 1.0).abs() < 1e-7, "layer 0 key should be 1.0");
        }
        for &v in stored1 {
            assert!((v - 2.0).abs() < 1e-7, "layer 1 key should be 2.0");
        }
    }

    // ── for_each_key / for_each_value iteration ─────────────────────────────

    #[test]
    fn kv_cache_for_each_key_contiguous() {
        use oxillama_arch::traits::KvCacheAccess;

        let kv_dim = 4usize;
        let mut cache = KvCache::new(1, 16, kv_dim);

        // Store 4 tokens in layer 0.
        for t in 0..4u32 {
            let key: Vec<f32> = (0..kv_dim).map(|d| t as f32 * 10.0 + d as f32).collect();
            let val: Vec<f32> = (0..kv_dim).map(|d| t as f32 * 100.0 + d as f32).collect();
            cache.store_kv(0, &key, &val).expect("store_kv");
            cache.advance();
        }

        // Collect callbacks via for_each_key.
        let mut positions_seen: Vec<usize> = Vec::new();
        let mut keys_seen: Vec<Vec<f32>> = Vec::new();
        cache
            .for_each_key(0, &mut |pos, slice| {
                positions_seen.push(pos);
                keys_seen.push(slice.to_vec());
            })
            .expect("for_each_key must succeed");

        assert_eq!(positions_seen.len(), 4, "must visit all 4 positions");
        assert_eq!(
            positions_seen,
            vec![0, 1, 2, 3],
            "positions must be in order"
        );

        // Check key data for each position.
        for (t, key_row) in keys_seen.iter().enumerate() {
            assert_eq!(key_row.len(), kv_dim, "key row must have kv_dim elements");
            for (d, &v) in key_row.iter().enumerate() {
                let expected = t as f32 * 10.0 + d as f32;
                assert!(
                    (v - expected).abs() < 1e-6,
                    "token {t} dim {d}: expected {expected}, got {v}"
                );
            }
        }
    }

    #[test]
    fn kv_cache_for_each_value_contiguous() {
        use oxillama_arch::traits::KvCacheAccess;

        let kv_dim = 3usize;
        let mut cache = KvCache::new(1, 8, kv_dim);

        for t in 0..3u32 {
            let key = vec![0.0f32; kv_dim];
            let val: Vec<f32> = (0..kv_dim).map(|d| t as f32 + d as f32 * 0.1).collect();
            cache.store_kv(0, &key, &val).expect("store_kv");
            cache.advance();
        }

        let mut count = 0usize;
        cache
            .for_each_value(0, &mut |_pos, slice| {
                assert_eq!(slice.len(), kv_dim);
                count += 1;
            })
            .expect("for_each_value must succeed");
        assert_eq!(count, 3, "must visit 3 value rows");
    }
}

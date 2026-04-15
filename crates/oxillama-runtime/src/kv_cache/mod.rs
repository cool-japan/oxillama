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

use oxillama_arch::traits::KvCacheAccess;
use oxillama_arch::ArchResult;

pub use paged::PagedKvCache;
pub use prefix::{PrefixCacheConfig, PrefixKvCache};

/// Simple contiguous KV cache implementation.
///
/// Stores key and value tensors for all layers in contiguous FP32 buffers.
/// Each layer has a separate key buffer and value buffer, sized for the
/// maximum context length.
pub struct KvCache {
    /// Key buffers: one per layer, each of size [max_seq_len * kv_dim].
    keys: Vec<Vec<f32>>,
    /// Value buffers: one per layer, each of size [max_seq_len * kv_dim].
    values: Vec<Vec<f32>>,
    /// Current sequence length (number of fully-committed tokens).
    seq_len: usize,
    /// Number of token positions that have had K/V data written.
    ///
    /// Invariant: `stored_len >= seq_len`.  Between a `store_kv` call at
    /// position `seq_len` and the subsequent `advance()`, `stored_len ==
    /// seq_len + 1` so that attention can immediately read the just-written
    /// entry without requiring `advance()` to have been called first.
    stored_len: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// KV dimension per token (num_kv_heads * head_dim).
    kv_dim: usize,
    /// Number of layers.
    num_layers: usize,
}

impl KvCache {
    /// Allocate a new KV cache.
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers.
    /// * `max_seq_len` - Maximum context length.
    /// * `kv_dim` - KV dimension per token (num_kv_heads * head_dim).
    pub fn new(num_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        let keys = (0..num_layers)
            .map(|_| vec![0.0f32; max_seq_len * kv_dim])
            .collect();
        let values = (0..num_layers)
            .map(|_| vec![0.0f32; max_seq_len * kv_dim])
            .collect();

        Self {
            keys,
            values,
            seq_len: 0,
            stored_len: 0,
            max_seq_len,
            kv_dim,
            num_layers,
        }
    }

    /// Reset the cache, clearing all stored KV pairs.
    pub fn clear(&mut self) {
        self.seq_len = 0;
        self.stored_len = 0;
        for k in &mut self.keys {
            k.fill(0.0);
        }
        for v in &mut self.values {
            v.fill(0.0);
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
    /// Copies the provided per-layer key/value data into internal buffers
    /// and sets `seq_len` to the snapshot's length.  The caller must ensure
    /// that `keys.len() == values.len() == num_layers` and that each inner
    /// vec has `seq_len * kv_dim` elements.
    pub fn restore_from_snapshot(
        &mut self,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        seq_len: usize,
    ) {
        let layers = keys.len().min(values.len()).min(self.num_layers);
        let copy_len = seq_len * self.kv_dim;

        for layer in 0..layers {
            let src_k = &keys[layer];
            let src_v = &values[layer];
            let n = copy_len.min(src_k.len()).min(self.keys[layer].len());
            self.keys[layer][..n].copy_from_slice(&src_k[..n]);
            let n = copy_len.min(src_v.len()).min(self.values[layer].len());
            self.values[layer][..n].copy_from_slice(&src_v[..n]);
        }

        self.seq_len = seq_len.min(self.max_seq_len);
        self.stored_len = self.seq_len;
    }
}

impl KvCacheAccess for KvCache {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }

        let offset = self.seq_len * self.kv_dim;
        let end = offset + self.kv_dim;

        if end <= self.keys[layer].len() {
            self.keys[layer][offset..end].copy_from_slice(&key[..self.kv_dim]);
            self.values[layer][offset..end].copy_from_slice(&value[..self.kv_dim]);
            // Ensure get_keys/get_values can see the entry we just wrote even
            // before advance() is called (advance is called once per token
            // after ALL layers have written their K/V, but attention reads
            // back during the same forward pass).
            if self.stored_len <= self.seq_len {
                self.stored_len = self.seq_len + 1;
            }
        }

        Ok(())
    }

    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        let end = self.stored_len * self.kv_dim;
        Ok(&self.keys[layer][..end])
    }

    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        let end = self.stored_len * self.kv_dim;
        Ok(&self.values[layer][..end])
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

    #[test]
    fn test_clear_zeros_stored_data() {
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
}

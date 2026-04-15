//! Paged KV cache implementation.
//!
//! Uses fixed-size blocks (pages) for memory-efficient KV storage.
//! Each page holds `PAGE_SIZE` tokens worth of KV data. Pages are
//! allocated on demand from a shared pool and can be freed when
//! a sequence is discarded.
//!
//! Benefits over contiguous cache:
//! - Memory-efficient for variable-length sequences
//! - No wasted memory for sequences shorter than max context
//! - Foundation for continuous batching (share pool across sequences)

use oxillama_arch::traits::KvCacheAccess;
use oxillama_arch::ArchResult;

/// Number of tokens per page. Chosen to balance allocation overhead
/// (fewer, larger pages) vs memory waste (smaller pages = less waste).
const PAGE_SIZE: usize = 16;

/// A single page of KV data for one layer.
///
/// Stores `PAGE_SIZE` tokens worth of key or value data.
/// Each token occupies `kv_dim` floats.
struct Page {
    data: Vec<f32>,
}

impl Page {
    fn new(kv_dim: usize) -> Self {
        Self {
            data: vec![0.0f32; PAGE_SIZE * kv_dim],
        }
    }

    /// Write one token's data at the given slot within this page.
    fn write_token(&mut self, slot: usize, kv_dim: usize, src: &[f32]) {
        let offset = slot * kv_dim;
        self.data[offset..offset + kv_dim].copy_from_slice(&src[..kv_dim]);
    }

    /// Read one token's data at the given slot within this page.
    fn read_token(&self, slot: usize, kv_dim: usize) -> &[f32] {
        let offset = slot * kv_dim;
        &self.data[offset..offset + kv_dim]
    }
}

/// Per-layer page table: maps logical page index → physical page.
struct LayerCache {
    key_pages: Vec<Page>,
    value_pages: Vec<Page>,
}

impl LayerCache {
    fn new() -> Self {
        Self {
            key_pages: Vec::new(),
            value_pages: Vec::new(),
        }
    }

    /// Ensure enough pages are allocated to cover `token_pos` (0-based).
    fn ensure_capacity(&mut self, token_pos: usize, kv_dim: usize) {
        let needed_pages = token_pos / PAGE_SIZE + 1;
        while self.key_pages.len() < needed_pages {
            self.key_pages.push(Page::new(kv_dim));
            self.value_pages.push(Page::new(kv_dim));
        }
    }

    /// Store a KV pair at the given token position.
    fn store(&mut self, token_pos: usize, kv_dim: usize, key: &[f32], value: &[f32]) {
        self.ensure_capacity(token_pos, kv_dim);
        let page_idx = token_pos / PAGE_SIZE;
        let slot = token_pos % PAGE_SIZE;
        self.key_pages[page_idx].write_token(slot, kv_dim, key);
        self.value_pages[page_idx].write_token(slot, kv_dim, value);
    }

    /// Get the number of allocated pages.
    fn num_pages(&self) -> usize {
        self.key_pages.len()
    }

    /// Free all pages beyond what's needed for `seq_len` tokens.
    fn shrink_to(&mut self, seq_len: usize) {
        let needed = if seq_len == 0 {
            0
        } else {
            seq_len / PAGE_SIZE + 1
        };
        self.key_pages.truncate(needed);
        self.value_pages.truncate(needed);
    }
}

/// Paged KV cache.
///
/// Memory is allocated in fixed-size pages (blocks) of `PAGE_SIZE` tokens.
/// Pages grow on demand — short sequences don't waste memory for unused
/// context positions. The `assemble_*` methods reconstruct contiguous
/// slices for attention computation.
pub struct PagedKvCache {
    /// Per-layer caches.
    layers: Vec<LayerCache>,
    /// Current sequence length.
    seq_len: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// KV dimension per token (num_kv_heads * head_dim).
    kv_dim: usize,
    /// Number of layers.
    num_layers: usize,
}

impl PagedKvCache {
    /// Create a new paged KV cache.
    ///
    /// Unlike the contiguous cache, this does NOT pre-allocate all memory.
    /// Pages are allocated on demand as tokens are processed.
    pub fn new(num_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        let layers = (0..num_layers).map(|_| LayerCache::new()).collect();

        Self {
            layers,
            seq_len: 0,
            max_seq_len,
            kv_dim,
            num_layers,
        }
    }

    /// Returns the page size (tokens per page).
    pub fn page_size(&self) -> usize {
        PAGE_SIZE
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

    /// Returns total number of allocated pages across all layers.
    pub fn total_pages(&self) -> usize {
        self.layers.iter().map(|l| l.num_pages()).sum()
    }

    /// Returns total memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        self.total_pages() * PAGE_SIZE * self.kv_dim * 4 * 2 // *2 for K+V, *4 for f32
    }

    /// Reset the cache, freeing all pages.
    pub fn clear(&mut self) {
        self.seq_len = 0;
        for layer in &mut self.layers {
            layer.key_pages.clear();
            layer.value_pages.clear();
        }
    }

    /// Shrink allocated pages to fit the current sequence length.
    /// Useful after trimming context.
    pub fn shrink_to_fit(&mut self) {
        for layer in &mut self.layers {
            layer.shrink_to(self.seq_len);
        }
    }

    /// Assemble contiguous key data for a layer into the provided buffer.
    ///
    /// Copies from paged storage into a flat `[seq_len * kv_dim]` buffer.
    fn assemble_keys(&self, layer: usize, buf: &mut Vec<f32>) {
        let total = self.seq_len * self.kv_dim;
        buf.clear();
        buf.reserve(total);

        let layer_cache = &self.layers[layer];
        for pos in 0..self.seq_len {
            let page_idx = pos / PAGE_SIZE;
            let slot = pos % PAGE_SIZE;
            let token_data = layer_cache.key_pages[page_idx].read_token(slot, self.kv_dim);
            buf.extend_from_slice(token_data);
        }
    }

    /// Assemble contiguous value data for a layer into the provided buffer.
    fn assemble_values(&self, layer: usize, buf: &mut Vec<f32>) {
        let total = self.seq_len * self.kv_dim;
        buf.clear();
        buf.reserve(total);

        let layer_cache = &self.layers[layer];
        for pos in 0..self.seq_len {
            let page_idx = pos / PAGE_SIZE;
            let slot = pos % PAGE_SIZE;
            let token_data = layer_cache.value_pages[page_idx].read_token(slot, self.kv_dim);
            buf.extend_from_slice(token_data);
        }
    }
}

impl KvCacheAccess for PagedKvCache {
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

        if self.seq_len >= self.max_seq_len {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!(
                    "sequence length {} exceeds max {}",
                    self.seq_len, self.max_seq_len
                ),
            });
        }

        self.layers[layer].store(self.seq_len, self.kv_dim, key, value);
        Ok(())
    }

    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }

        // For now, we need to return a contiguous slice. The paged layout
        // means we can't return a zero-copy slice if data spans multiple pages.
        // This is a known limitation — the trait will need to evolve for
        // truly zero-copy paged access (page-aware attention kernels).
        //
        // SAFETY: We use interior mutability via the assemble buffer approach.
        // Since we can't mutate &self, we return a reference to assembled data
        // that lives in the pages themselves when seq_len fits in one page.
        if self.seq_len == 0 {
            return Ok(&[]);
        }

        // Fast path: all data fits in a single page — return zero-copy slice
        let pages_used = (self.seq_len - 1) / PAGE_SIZE + 1;
        if pages_used == 1 {
            let end = self.seq_len * self.kv_dim;
            return Ok(&self.layers[layer].key_pages[0].data[..end]);
        }

        // Multi-page: we can't return a contiguous &[f32] without copying.
        // This is a fundamental limitation of returning &[f32] from paged storage.
        // For now, panic with a message pointing to the solution.
        // TODO: Change trait to support page-aware iteration or accept a callback.
        Err(oxillama_arch::ArchError::ForwardPassError {
            layer,
            message: format!(
                "paged KV cache: sequence length {} spans {} pages; \
                 use get_keys_into() for multi-page access",
                self.seq_len, pages_used
            ),
        })
    }

    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }

        if self.seq_len == 0 {
            return Ok(&[]);
        }

        let pages_used = (self.seq_len - 1) / PAGE_SIZE + 1;
        if pages_used == 1 {
            let end = self.seq_len * self.kv_dim;
            return Ok(&self.layers[layer].value_pages[0].data[..end]);
        }

        Err(oxillama_arch::ArchError::ForwardPassError {
            layer,
            message: format!(
                "paged KV cache: sequence length {} spans {} pages; \
                 use get_values_into() for multi-page access",
                self.seq_len, pages_used
            ),
        })
    }

    fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }
}

/// Extended paged KV cache operations (not part of the base trait).
impl PagedKvCache {
    /// Copy all cached keys for a layer into a contiguous buffer.
    ///
    /// This is the multi-page alternative to `get_keys()`. The caller
    /// provides a reusable buffer to avoid repeated allocation.
    pub fn get_keys_into(&self, layer: usize, buf: &mut Vec<f32>) -> ArchResult<()> {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        self.assemble_keys(layer, buf);
        Ok(())
    }

    /// Copy all cached values for a layer into a contiguous buffer.
    pub fn get_values_into(&self, layer: usize, buf: &mut Vec<f32>) -> ArchResult<()> {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        self.assemble_values(layer, buf);
        Ok(())
    }

    /// Read a specific token's key data from the cache.
    pub fn get_key_token(&self, layer: usize, pos: usize) -> ArchResult<&[f32]> {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        if pos >= self.seq_len {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("position {pos} out of range (seq_len {})", self.seq_len),
            });
        }
        let page_idx = pos / PAGE_SIZE;
        let slot = pos % PAGE_SIZE;
        Ok(self.layers[layer].key_pages[page_idx].read_token(slot, self.kv_dim))
    }

    /// Read a specific token's value data from the cache.
    pub fn get_value_token(&self, layer: usize, pos: usize) -> ArchResult<&[f32]> {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        if pos >= self.seq_len {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("position {pos} out of range (seq_len {})", self.seq_len),
            });
        }
        let page_idx = pos / PAGE_SIZE;
        let slot = pos % PAGE_SIZE;
        Ok(self.layers[layer].value_pages[page_idx].read_token(slot, self.kv_dim))
    }

    /// Iterate over key tokens for a layer, calling `f` for each (pos, key_data).
    pub fn iter_keys<F>(&self, layer: usize, mut f: F) -> ArchResult<()>
    where
        F: FnMut(usize, &[f32]),
    {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        let layer_cache = &self.layers[layer];
        for pos in 0..self.seq_len {
            let page_idx = pos / PAGE_SIZE;
            let slot = pos % PAGE_SIZE;
            let data = layer_cache.key_pages[page_idx].read_token(slot, self.kv_dim);
            f(pos, data);
        }
        Ok(())
    }

    /// Iterate over value tokens for a layer.
    pub fn iter_values<F>(&self, layer: usize, mut f: F) -> ArchResult<()>
    where
        F: FnMut(usize, &[f32]),
    {
        if layer >= self.num_layers {
            return Err(oxillama_arch::ArchError::ForwardPassError {
                layer,
                message: format!("layer index {layer} out of range (max {})", self.num_layers),
            });
        }
        let layer_cache = &self.layers[layer];
        for pos in 0..self.seq_len {
            let page_idx = pos / PAGE_SIZE;
            let slot = pos % PAGE_SIZE;
            let data = layer_cache.value_pages[page_idx].read_token(slot, self.kv_dim);
            f(pos, data);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_basic_store_retrieve() {
        let mut cache = PagedKvCache::new(2, 64, 4);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.total_pages(), 0);

        // Store first token in layer 0
        let key = [1.0, 2.0, 3.0, 4.0];
        let val = [5.0, 6.0, 7.0, 8.0];
        cache.store_kv(0, &key, &val).unwrap();
        cache.advance();

        assert_eq!(cache.seq_len(), 1);
        // Should have allocated 1 page per layer for layer 0 only
        assert_eq!(cache.layers[0].num_pages(), 1);
        assert_eq!(cache.layers[1].num_pages(), 0);

        // Retrieve via single-page fast path
        let keys = cache.get_keys(0).unwrap();
        assert_eq!(keys, &[1.0, 2.0, 3.0, 4.0]);

        let vals = cache.get_values(0).unwrap();
        assert_eq!(vals, &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_paged_multi_token_single_page() {
        let mut cache = PagedKvCache::new(1, 64, 2);

        // Store PAGE_SIZE tokens (all fit in one page)
        for i in 0..PAGE_SIZE {
            let key = [i as f32, (i * 10) as f32];
            let val = [(i + 100) as f32, (i + 200) as f32];
            cache.store_kv(0, &key, &val).unwrap();
            cache.advance();
        }

        assert_eq!(cache.seq_len(), PAGE_SIZE);
        assert_eq!(cache.layers[0].num_pages(), 1);

        // Should still work via fast path (single page)
        let keys = cache.get_keys(0).unwrap();
        assert_eq!(keys.len(), PAGE_SIZE * 2);
        assert_eq!(keys[0], 0.0);
        assert_eq!(keys[1], 0.0);
        assert_eq!(keys[2], 1.0);
        assert_eq!(keys[3], 10.0);
    }

    #[test]
    fn test_paged_multi_page_assembly() {
        let mut cache = PagedKvCache::new(1, 64, 2);

        // Store PAGE_SIZE + 1 tokens (spans 2 pages)
        for i in 0..=PAGE_SIZE {
            let key = [i as f32, (i * 10) as f32];
            let val = [(i + 100) as f32, (i + 200) as f32];
            cache.store_kv(0, &key, &val).unwrap();
            cache.advance();
        }

        assert_eq!(cache.seq_len(), PAGE_SIZE + 1);
        assert_eq!(cache.layers[0].num_pages(), 2);

        // get_keys returns error for multi-page
        assert!(cache.get_keys(0).is_err());

        // Use get_keys_into for multi-page access
        let mut buf = Vec::new();
        cache.get_keys_into(0, &mut buf).unwrap();
        assert_eq!(buf.len(), (PAGE_SIZE + 1) * 2);

        // Verify first and last tokens
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[1], 0.0);
        let last_off = PAGE_SIZE * 2;
        assert_eq!(buf[last_off], PAGE_SIZE as f32);
        assert_eq!(buf[last_off + 1], (PAGE_SIZE * 10) as f32);
    }

    #[test]
    fn test_paged_per_token_access() {
        let mut cache = PagedKvCache::new(1, 64, 3);

        for i in 0..20 {
            let key = [i as f32, (i * 2) as f32, (i * 3) as f32];
            let val = [(i + 50) as f32, (i + 60) as f32, (i + 70) as f32];
            cache.store_kv(0, &key, &val).unwrap();
            cache.advance();
        }

        // Access specific tokens
        let k5 = cache.get_key_token(0, 5).unwrap();
        assert_eq!(k5, &[5.0, 10.0, 15.0]);

        let v17 = cache.get_value_token(0, 17).unwrap();
        assert_eq!(v17, &[67.0, 77.0, 87.0]);

        // Out of range
        assert!(cache.get_key_token(0, 20).is_err());
    }

    #[test]
    fn test_paged_iteration() {
        let mut cache = PagedKvCache::new(1, 64, 2);

        for i in 0..20 {
            let key = [i as f32, (i + 1) as f32];
            let val = [(i + 100) as f32, (i + 101) as f32];
            cache.store_kv(0, &key, &val).unwrap();
            cache.advance();
        }

        let mut count = 0;
        cache
            .iter_keys(0, |pos, data| {
                assert_eq!(data[0], pos as f32);
                assert_eq!(data[1], (pos + 1) as f32);
                count += 1;
            })
            .unwrap();
        assert_eq!(count, 20);
    }

    #[test]
    fn test_paged_clear() {
        let mut cache = PagedKvCache::new(2, 64, 4);

        for i in 0..20 {
            let key = [i as f32; 4];
            let val = [i as f32; 4];
            cache.store_kv(0, &key, &val).unwrap();
            cache.store_kv(1, &key, &val).unwrap();
            cache.advance();
        }

        assert!(cache.total_pages() > 0);
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.total_pages(), 0);
    }

    #[test]
    fn test_paged_shrink_to_fit() {
        let mut cache = PagedKvCache::new(1, 128, 4);

        // Fill 40 tokens (3 pages)
        for i in 0..40 {
            cache.store_kv(0, &[i as f32; 4], &[i as f32; 4]).unwrap();
            cache.advance();
        }
        assert_eq!(cache.layers[0].num_pages(), 3);

        // Manually reduce seq_len to simulate context trimming
        cache.seq_len = 10;
        cache.shrink_to_fit();
        // 10 tokens → 1 page needed
        assert_eq!(cache.layers[0].num_pages(), 1);
    }

    #[test]
    fn test_paged_memory_efficiency() {
        // Compare memory: contiguous pre-allocates everything,
        // paged only allocates what's used.
        let num_layers = 32;
        let max_seq = 4096;
        let kv_dim = 128;

        let contiguous_bytes = num_layers * max_seq * kv_dim * 4 * 2; // K+V

        let mut cache = PagedKvCache::new(num_layers, max_seq, kv_dim);
        // Store just 10 tokens
        for i in 0..10 {
            for layer in 0..num_layers {
                cache
                    .store_kv(layer, &vec![i as f32; kv_dim], &vec![i as f32; kv_dim])
                    .unwrap();
            }
            cache.advance();
        }

        let paged_bytes = cache.memory_bytes();
        // Paged should use much less memory than contiguous
        assert!(
            paged_bytes < contiguous_bytes / 10,
            "paged={paged_bytes} should be << contiguous={contiguous_bytes}"
        );
    }

    #[test]
    fn test_paged_max_seq_len_error() {
        let mut cache = PagedKvCache::new(1, 2, 2);

        cache.store_kv(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.advance();
        cache.store_kv(0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        cache.advance();

        // Should fail — at max
        let result = cache.store_kv(0, &[9.0, 10.0], &[11.0, 12.0]);
        assert!(result.is_err());
    }
}

//! Radix-tree based prefix KV cache.
//!
//! Stores KV cache states indexed by token prefix sequences.  When a new prompt
//! shares a prefix with a previously-cached sequence, the matching KV state is
//! reused and only the remaining tokens need prefill.
//!
//! ## How it works
//!
//! 1. Token sequences are stored in a radix tree (trie with path compression).
//! 2. Each node stores a segment of tokens and optional KV cache data.
//! 3. On lookup, the tree walks down matching token prefixes.
//! 4. The longest matching prefix's KV state can be directly restored.
//! 5. LRU eviction removes least-recently-used entries when capacity is exceeded.

use std::collections::HashMap;
use std::time::Instant;

use oxillama_arch::traits::KvCacheAccess;

use super::KvCache;

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for prefix KV caching.
#[derive(Debug, Clone)]
pub struct PrefixCacheConfig {
    /// Maximum number of cached prefixes (nodes with KV data).
    pub max_entries: usize,
    /// Maximum total memory for cached KV states (bytes).
    pub max_memory_bytes: usize,
    /// Minimum prefix length to cache (tokens).
    pub min_prefix_len: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_memory_bytes: 512 * 1024 * 1024, // 512 MiB
            min_prefix_len: 4,
        }
    }
}

// ── Cached KV state ──────────────────────────────────────────────────────────

/// Snapshot of KV cache state for a prefix.
#[derive(Clone)]
pub struct CachedKvState {
    /// Per-layer key tensors flattened: `[layer][seq_pos * kv_dim]`.
    keys: Vec<Vec<f32>>,
    /// Per-layer value tensors flattened.
    values: Vec<Vec<f32>>,
    /// Number of tokens this state covers.
    seq_len: usize,
}

impl CachedKvState {
    /// Construct a new `CachedKvState` from pre-built KV buffers.
    ///
    /// This is the public constructor used when re-assembling a state from
    /// cloned data (e.g. after releasing a `Mutex` lock on a `PrefixKvCache`).
    /// `keys` and `values` must each have one inner `Vec<f32>` per layer.
    pub fn new(keys: Vec<Vec<f32>>, values: Vec<Vec<f32>>, seq_len: usize) -> Self {
        Self {
            keys,
            values,
            seq_len,
        }
    }

    /// Number of tokens this snapshot covers.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Per-layer key buffers.
    pub fn keys(&self) -> &[Vec<f32>] {
        &self.keys
    }

    /// Per-layer value buffers.
    pub fn values(&self) -> &[Vec<f32>] {
        &self.values
    }

    /// Estimated memory usage in bytes.
    fn memory_bytes(&self) -> usize {
        let float_count: usize = self
            .keys
            .iter()
            .chain(self.values.iter())
            .map(|v| v.len())
            .sum();
        float_count * std::mem::size_of::<f32>()
    }
}

// ── Radix tree node ──────────────────────────────────────────────────────────

/// A node in the radix tree.
struct RadixNode {
    /// Token segment stored at this node (compressed path).
    tokens: Vec<u32>,
    /// Children keyed by the first token of their segment.
    children: HashMap<u32, Box<RadixNode>>,
    /// Cached KV data for this prefix (`None` for internal-only nodes).
    cached_kv: Option<CachedKvState>,
    /// Last access timestamp for LRU eviction.
    last_access: Instant,
    /// Reference count (how many active sequences use this prefix).
    ref_count: u32,
}

impl RadixNode {
    /// Create a new node with the given token segment.
    fn new(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            children: HashMap::new(),
            cached_kv: None,
            last_access: Instant::now(),
            ref_count: 0,
        }
    }

    /// Walk down the tree, returning the best (deepest) node that has cached
    /// KV data whose prefix matches the query tokens.
    ///
    /// Returns `(matched_token_count, reference_to_node)`.
    fn lookup<'a>(
        &'a mut self,
        query: &[u32],
        matched_so_far: usize,
    ) -> Option<(usize, &'a CachedKvState)> {
        // Match this node's segment against the beginning of `query`.
        let common = common_prefix_len(&self.tokens, query);
        if common < self.tokens.len() {
            // Partial match only — cannot descend further.
            // Return the cached KV at this node only if the segment fully matched
            // (it didn't, so nothing from this node).
            return None;
        }

        let total_matched = matched_so_far + common;
        let remaining = &query[common..];

        // Update access time since we're visiting this node.
        self.last_access = Instant::now();

        // Try to descend into a child.
        let mut best: Option<(usize, &'a CachedKvState)> = None;

        if let Some(&first_token) = remaining.first() {
            if let Some(child) = self.children.get_mut(&first_token) {
                best = child.lookup(remaining, total_matched);
            }
        }

        // If no deeper match found, use this node's cache (if any).
        if best.is_none() {
            if let Some(ref kv) = self.cached_kv {
                best = Some((total_matched, kv));
            }
        }

        best
    }

    /// Insert KV data at the leaf matching `tokens`, splitting nodes as needed.
    fn insert(&mut self, tokens: &[u32], kv: CachedKvState) {
        if tokens.is_empty() {
            self.cached_kv = Some(kv);
            self.last_access = Instant::now();
            return;
        }

        let common = common_prefix_len(&self.tokens, tokens);

        if common < self.tokens.len() {
            // Need to split this node.
            self.split_at(common);
        }

        let remaining = &tokens[common..];
        if remaining.is_empty() {
            self.cached_kv = Some(kv);
            self.last_access = Instant::now();
            return;
        }

        let first = remaining[0];
        let child = self
            .children
            .entry(first)
            .or_insert_with(|| Box::new(RadixNode::new(remaining.to_vec())));

        // If the child already exists, recurse into it.
        if child.tokens == remaining {
            child.cached_kv = Some(kv);
            child.last_access = Instant::now();
        } else {
            child.insert(remaining, kv);
        }
    }

    /// Split this node at position `pos`, pushing the suffix (and all
    /// children / cached data) into a new child node.
    fn split_at(&mut self, pos: usize) {
        let suffix = self.tokens[pos..].to_vec();
        let first_of_suffix = suffix[0];

        let mut new_child = RadixNode::new(suffix);
        new_child.children = std::mem::take(&mut self.children);
        new_child.cached_kv = self.cached_kv.take();
        new_child.last_access = self.last_access;
        new_child.ref_count = self.ref_count;

        self.tokens.truncate(pos);
        self.children.insert(first_of_suffix, Box::new(new_child));
    }

    /// Count the number of nodes that carry cached KV data.
    fn count_entries(&self) -> usize {
        let mine = usize::from(self.cached_kv.is_some());
        let children_count: usize = self.children.values().map(|c| c.count_entries()).sum();
        mine + children_count
    }

    /// Sum the estimated memory of all cached KV states in this subtree.
    fn total_memory(&self) -> usize {
        let mine = self.cached_kv.as_ref().map_or(0, |kv| kv.memory_bytes());
        let children_mem: usize = self.children.values().map(|c| c.total_memory()).sum();
        mine + children_mem
    }

    /// Find and remove the LRU eviction candidate in this subtree.
    ///
    /// Returns the memory freed (0 if nothing was evicted).
    fn evict_lru_one(&mut self) -> usize {
        // Collect candidates: this node and all descendants.
        let mut oldest_time = Instant::now();
        let mut oldest_path: Option<Vec<u32>> = None;
        let mut oldest_mem: usize = 0;

        self.find_lru_candidate(&mut oldest_time, &mut oldest_path, &mut oldest_mem, &[]);

        if let Some(path) = oldest_path {
            self.remove_cached_at(&path)
        } else {
            0
        }
    }

    /// Recursively find the LRU candidate with `ref_count == 0`.
    fn find_lru_candidate(
        &self,
        oldest_time: &mut Instant,
        oldest_path: &mut Option<Vec<u32>>,
        oldest_mem: &mut usize,
        prefix: &[u32],
    ) {
        if self.cached_kv.is_some() && self.ref_count == 0 && self.last_access < *oldest_time {
            *oldest_time = self.last_access;
            let mut path = prefix.to_vec();
            path.extend_from_slice(&self.tokens);
            *oldest_path = Some(path);
            *oldest_mem = self.cached_kv.as_ref().map_or(0, |kv| kv.memory_bytes());
        }

        for child in self.children.values() {
            let mut child_prefix = prefix.to_vec();
            child_prefix.extend_from_slice(&self.tokens);
            child.find_lru_candidate(oldest_time, oldest_path, oldest_mem, &child_prefix);
        }
    }

    /// Remove cached KV data at the node reached by following `path` tokens.
    ///
    /// Returns the memory freed.
    fn remove_cached_at(&mut self, path: &[u32]) -> usize {
        let common = common_prefix_len(&self.tokens, path);
        if common < self.tokens.len() {
            return 0;
        }

        let remaining = &path[common..];
        if remaining.is_empty() {
            // This is the target node.
            let freed = self.cached_kv.as_ref().map_or(0, |kv| kv.memory_bytes());
            self.cached_kv = None;
            return freed;
        }

        if let Some(&first) = remaining.first() {
            if let Some(child) = self.children.get_mut(&first) {
                let freed = child.remove_cached_at(remaining);
                // If the child is now empty (no cache, no children), prune it.
                if child.cached_kv.is_none() && child.children.is_empty() {
                    self.children.remove(&first);
                }
                return freed;
            }
        }
        0
    }

    /// Clear all cached data in this subtree.
    fn clear_all(&mut self) {
        self.cached_kv = None;
        self.children.clear();
    }
}

/// Returns the length of the common prefix between two slices.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

// ── PrefixKvCache ────────────────────────────────────────────────────────────

/// A radix-tree based prefix KV cache.
///
/// Stores KV cache states indexed by token prefix sequences. When a new prompt
/// shares a prefix with a previously-cached sequence, the matching KV state is
/// reused and only the remaining tokens need prefill.
pub struct PrefixKvCache {
    /// Root of the radix tree (has an empty token segment).
    root: RadixNode,
    /// Configuration.
    config: PrefixCacheConfig,
    /// Cache hit counter.
    hit_count: u64,
    /// Cache miss counter.
    miss_count: u64,
}

impl PrefixKvCache {
    /// Create a new prefix KV cache with the given configuration.
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            root: RadixNode::new(Vec::new()),
            config,
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Look up the longest matching prefix for the given tokens.
    ///
    /// Returns `(matching_prefix_length, cached_kv_state_ref)`.  Returns `None`
    /// if no prefix matches or the match is shorter than `min_prefix_len`.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<(usize, &CachedKvState)> {
        if tokens.is_empty() {
            self.miss_count += 1;
            return None;
        }

        let result = self.root.lookup(tokens, 0);

        match result {
            Some((matched, kv)) if matched >= self.config.min_prefix_len => {
                self.hit_count += 1;
                Some((matched, kv))
            }
            _ => {
                self.miss_count += 1;
                None
            }
        }
    }

    /// Store KV cache state for a token prefix.
    ///
    /// Extracts the relevant KV data from the live cache via the
    /// [`KvCacheAccess`] trait.  If the prefix is shorter than
    /// `min_prefix_len`, the store is silently skipped.
    pub fn store(
        &mut self,
        tokens: &[u32],
        kv_cache: &dyn KvCacheAccess,
        seq_len: usize,
        kv_dim: usize,
        num_layers: usize,
    ) {
        if tokens.len() < self.config.min_prefix_len {
            return;
        }

        // Snapshot the KV state from the live cache.
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let k = kv_cache.get_keys(layer).unwrap_or(&[]);
            let v = kv_cache.get_values(layer).unwrap_or(&[]);
            let end = seq_len * kv_dim;
            keys.push(k[..end.min(k.len())].to_vec());
            values.push(v[..end.min(v.len())].to_vec());
        }

        let snapshot = CachedKvState {
            keys,
            values,
            seq_len,
        };

        self.root.insert(tokens, snapshot);

        // Evict if over limits.
        self.evict_lru();
    }

    /// Store a pre-built [`CachedKvState`] directly for a token prefix.
    ///
    /// This is useful when the caller has already constructed the snapshot.
    pub fn store_snapshot(&mut self, tokens: &[u32], snapshot: CachedKvState) {
        if tokens.len() < self.config.min_prefix_len {
            return;
        }
        self.root.insert(tokens, snapshot);
        self.evict_lru();
    }

    /// Restore a cached prefix into a live KV cache.
    ///
    /// Copies the cached KV data into the target cache's buffers and resets
    /// the target's sequence position to match the snapshot.
    pub fn restore(cached: &CachedKvState, target: &mut KvCache) {
        target.restore_from_snapshot(&cached.keys, &cached.values, cached.seq_len);
    }

    /// Evict least-recently-used entries until memory is under the limit.
    fn evict_lru(&mut self) {
        // Evict by entry count.
        while self.root.count_entries() > self.config.max_entries {
            if self.root.evict_lru_one() == 0 {
                break; // No more evictable entries.
            }
        }
        // Evict by memory.
        while self.root.total_memory() > self.config.max_memory_bytes {
            if self.root.evict_lru_one() == 0 {
                break;
            }
        }
    }

    /// Current number of cached prefixes (nodes with KV data).
    pub fn len(&self) -> usize {
        self.root.count_entries()
    }

    /// Whether the cache is empty (no cached KV data).
    pub fn is_empty(&self) -> bool {
        self.root.count_entries() == 0
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.root.clear_all();
        self.hit_count = 0;
        self.miss_count = 0;
    }

    /// Current estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.root.total_memory()
    }

    /// Number of cache hits since creation.
    pub fn hits(&self) -> u64 {
        self.hit_count
    }

    /// Number of cache misses since creation.
    pub fn misses(&self) -> u64 {
        self.miss_count
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_arch::traits::KvCacheAccess;

    /// Helper: build a tiny KV cache, fill it with deterministic data for
    /// `num_tokens` tokens, and return it along with the tokens used.
    fn make_filled_cache(
        num_layers: usize,
        kv_dim: usize,
        num_tokens: usize,
    ) -> (KvCache, Vec<u32>) {
        let mut cache = KvCache::new(num_layers, 128, kv_dim);
        let tokens: Vec<u32> = (0..num_tokens as u32).collect();

        for t in 0..num_tokens {
            for layer in 0..num_layers {
                let base = (layer * 1000 + t) as f32;
                let key: Vec<f32> = (0..kv_dim).map(|d| base + d as f32 * 0.01).collect();
                let val: Vec<f32> = (0..kv_dim).map(|d| base + d as f32 * 0.02).collect();
                cache
                    .store_kv(layer, &key, &val)
                    .expect("store_kv should succeed");
            }
            cache.advance();
        }

        (cache, tokens)
    }

    fn default_config() -> PrefixCacheConfig {
        PrefixCacheConfig {
            max_entries: 64,
            max_memory_bytes: 16 * 1024 * 1024,
            min_prefix_len: 1,
        }
    }

    // ── Basic insert / lookup ────────────────────────────────────────────

    #[test]
    fn test_insert_and_lookup_exact() {
        let mut pcache = PrefixKvCache::new(default_config());
        let (cache, tokens) = make_filled_cache(2, 4, 5);

        pcache.store(&tokens, &cache, 5, 4, 2);
        assert_eq!(pcache.len(), 1);

        let result = pcache.lookup(&tokens);
        assert!(result.is_some());
        let (matched, kv) = result.expect("lookup should succeed");
        assert_eq!(matched, 5);
        assert_eq!(kv.seq_len(), 5);
    }

    #[test]
    fn test_lookup_longer_query_returns_cached_prefix() {
        let mut pcache = PrefixKvCache::new(default_config());
        let (cache, tokens) = make_filled_cache(2, 4, 5);

        pcache.store(&tokens, &cache, 5, 4, 2);

        // Query with more tokens — should still match the cached 5-token prefix.
        let longer: Vec<u32> = (0..10).collect();
        let result = pcache.lookup(&longer);
        assert!(result.is_some());
        let (matched, _) = result.expect("lookup should succeed");
        assert_eq!(matched, 5);
    }

    #[test]
    fn test_lookup_no_match_returns_none() {
        let mut pcache = PrefixKvCache::new(default_config());
        let (cache, tokens) = make_filled_cache(1, 4, 5);
        pcache.store(&tokens, &cache, 5, 4, 1);

        // Completely different tokens.
        let other = vec![100, 200, 300];
        let result = pcache.lookup(&other);
        assert!(result.is_none());
    }

    #[test]
    fn test_empty_cache_lookup_returns_none() {
        let mut pcache = PrefixKvCache::new(default_config());
        let result = pcache.lookup(&[1, 2, 3]);
        assert!(result.is_none());
    }

    #[test]
    fn test_empty_query_returns_none() {
        let mut pcache = PrefixKvCache::new(default_config());
        let result = pcache.lookup(&[]);
        assert!(result.is_none());
    }

    // ── Multiple prefixes with shared prefix ─────────────────────────────

    #[test]
    fn test_multiple_prefixes_with_shared_root() {
        let mut pcache = PrefixKvCache::new(default_config());

        // Two sequences that share tokens [0,1,2] but diverge after.
        let tokens_a = vec![0u32, 1, 2, 3, 4];
        let tokens_b = vec![0u32, 1, 2, 10, 11];

        let (cache_a, _) = make_filled_cache(1, 4, 5);
        let (cache_b, _) = make_filled_cache(1, 4, 5);

        pcache.store(&tokens_a, &cache_a, 5, 4, 1);
        pcache.store(&tokens_b, &cache_b, 5, 4, 1);

        assert_eq!(pcache.len(), 2);

        // Lookup each — should get exact match.
        let (m_a, _) = pcache.lookup(&tokens_a).expect("lookup A");
        assert_eq!(m_a, 5);

        let (m_b, _) = pcache.lookup(&tokens_b).expect("lookup B");
        assert_eq!(m_b, 5);

        // Lookup shared prefix only — should match A or B (both have 5-len
        // prefix starting with [0,1,2,…]; the shared subset is [0,1,2]).
        // Since neither has a cached node at exactly 3 tokens, this should
        // return None (no node at depth 3 has cached_kv).
        let shared_only = vec![0u32, 1, 2];
        let result = pcache.lookup(&shared_only);
        assert!(result.is_none());
    }

    // ── LRU eviction ─────────────────────────────────────────────────────

    #[test]
    fn test_lru_eviction_by_entries() {
        let config = PrefixCacheConfig {
            max_entries: 2,
            max_memory_bytes: usize::MAX,
            min_prefix_len: 1,
        };
        let mut pcache = PrefixKvCache::new(config);

        for i in 0u32..3 {
            let tokens = vec![100 + i, 200 + i];
            let snapshot = CachedKvState {
                keys: vec![vec![i as f32; 4]],
                values: vec![vec![i as f32; 4]],
                seq_len: 2,
            };
            pcache.store_snapshot(&tokens, snapshot);
        }

        // Should have evicted one entry to stay at max_entries=2.
        assert!(pcache.len() <= 2);
    }

    #[test]
    fn test_lru_eviction_by_memory() {
        // Each entry: 1 layer, 4 floats for keys + 4 floats for values = 32 bytes.
        let config = PrefixCacheConfig {
            max_entries: 100,
            max_memory_bytes: 64, // room for ~2 entries
            min_prefix_len: 1,
        };
        let mut pcache = PrefixKvCache::new(config);

        for i in 0u32..5 {
            let tokens = vec![100 + i, 200 + i];
            let snapshot = CachedKvState {
                keys: vec![vec![i as f32; 4]],
                values: vec![vec![i as f32; 4]],
                seq_len: 2,
            };
            pcache.store_snapshot(&tokens, snapshot);
        }

        assert!(pcache.memory_usage() <= 64);
    }

    // ── Clear ────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_everything() {
        let mut pcache = PrefixKvCache::new(default_config());
        let (cache, tokens) = make_filled_cache(1, 4, 5);
        pcache.store(&tokens, &cache, 5, 4, 1);

        // Trigger a hit.
        let _ = pcache.lookup(&tokens);

        pcache.clear();

        assert!(pcache.is_empty());
        assert_eq!(pcache.len(), 0);
        assert_eq!(pcache.memory_usage(), 0);
        assert_eq!(pcache.hits(), 0);
        assert_eq!(pcache.misses(), 0);
    }

    // ── Store and restore round-trip ─────────────────────────────────────

    #[test]
    fn test_store_and_restore_round_trip() {
        let num_layers = 2;
        let kv_dim = 4;
        let num_tokens = 5;

        let mut pcache = PrefixKvCache::new(default_config());
        let (source_cache, tokens) = make_filled_cache(num_layers, kv_dim, num_tokens);

        pcache.store(&tokens, &source_cache, num_tokens, kv_dim, num_layers);

        let (_, cached_kv) = pcache.lookup(&tokens).expect("lookup must succeed");
        let cached_kv_clone = cached_kv.clone();

        // Restore into a fresh KvCache.
        let mut target = KvCache::new(num_layers, 128, kv_dim);
        PrefixKvCache::restore(&cached_kv_clone, &mut target);

        assert_eq!(target.seq_len(), num_tokens);

        // Verify all data matches the source.
        for layer in 0..num_layers {
            let src_keys = source_cache.get_keys(layer).expect("get_keys");
            let tgt_keys = target.get_keys(layer).expect("get_keys");
            assert_eq!(src_keys.len(), tgt_keys.len(), "layer {layer} key length");
            for (i, (&s, &t)) in src_keys.iter().zip(tgt_keys.iter()).enumerate() {
                assert!(
                    (s - t).abs() < 1e-7,
                    "layer {layer} key[{i}]: source={s}, target={t}"
                );
            }

            let src_vals = source_cache.get_values(layer).expect("get_values");
            let tgt_vals = target.get_values(layer).expect("get_values");
            assert_eq!(src_vals.len(), tgt_vals.len(), "layer {layer} value length");
            for (i, (&s, &t)) in src_vals.iter().zip(tgt_vals.iter()).enumerate() {
                assert!(
                    (s - t).abs() < 1e-7,
                    "layer {layer} value[{i}]: source={s}, target={t}"
                );
            }
        }
    }

    // ── Memory tracking ──────────────────────────────────────────────────

    #[test]
    fn test_memory_usage_tracking() {
        let mut pcache = PrefixKvCache::new(default_config());
        assert_eq!(pcache.memory_usage(), 0);

        // 1 layer, kv_dim=4, 2 tokens → keys: 8 floats, values: 8 floats = 64 bytes.
        let snapshot = CachedKvState {
            keys: vec![vec![0.0f32; 8]], // 2 tokens * kv_dim=4
            values: vec![vec![0.0f32; 8]],
            seq_len: 2,
        };
        pcache.store_snapshot(&[1, 2], snapshot);

        // 8 floats * 4 bytes * 2 (keys + values) = 64 bytes.
        assert_eq!(pcache.memory_usage(), 64);
    }

    // ── Hit / miss counters ──────────────────────────────────────────────

    #[test]
    fn test_hit_miss_counters() {
        let mut pcache = PrefixKvCache::new(default_config());
        assert_eq!(pcache.hits(), 0);
        assert_eq!(pcache.misses(), 0);

        // Miss on empty cache.
        let _ = pcache.lookup(&[1, 2, 3]);
        assert_eq!(pcache.misses(), 1);
        assert_eq!(pcache.hits(), 0);

        // Store something.
        let snapshot = CachedKvState {
            keys: vec![vec![0.0; 4]],
            values: vec![vec![0.0; 4]],
            seq_len: 2,
        };
        pcache.store_snapshot(&[1, 2], snapshot);

        // Hit.
        let _ = pcache.lookup(&[1, 2]);
        assert_eq!(pcache.hits(), 1);
        assert_eq!(pcache.misses(), 1);

        // Another miss (different tokens).
        let _ = pcache.lookup(&[99, 100]);
        assert_eq!(pcache.hits(), 1);
        assert_eq!(pcache.misses(), 2);
    }

    // ── min_prefix_len filter ────────────────────────────────────────────

    #[test]
    fn test_min_prefix_len_filters_short_store() {
        let config = PrefixCacheConfig {
            max_entries: 64,
            max_memory_bytes: 16 * 1024 * 1024,
            min_prefix_len: 5,
        };
        let mut pcache = PrefixKvCache::new(config);

        // Try to store a 3-token prefix with min_prefix_len=5.
        let (cache, _) = make_filled_cache(1, 4, 3);
        pcache.store(&[0, 1, 2], &cache, 3, 4, 1);

        // Should not have been stored.
        assert!(pcache.is_empty());
    }

    #[test]
    fn test_min_prefix_len_filters_short_lookup() {
        let config = PrefixCacheConfig {
            max_entries: 64,
            max_memory_bytes: 16 * 1024 * 1024,
            min_prefix_len: 5,
        };
        let mut pcache = PrefixKvCache::new(config);

        // Store a long prefix.
        let (cache, tokens) = make_filled_cache(1, 4, 10);
        pcache.store(&tokens, &cache, 10, 4, 1);
        assert_eq!(pcache.len(), 1);

        // Lookup with a 3-token query. Even though 3 tokens match, the
        // matched length (3) is below min_prefix_len (5), so it returns None.
        let short_query = vec![0u32, 1, 2];
        let result = pcache.lookup(&short_query);
        assert!(result.is_none());
    }

    // ── is_empty / len ───────────────────────────────────────────────────

    #[test]
    fn test_is_empty_and_len() {
        let mut pcache = PrefixKvCache::new(default_config());
        assert!(pcache.is_empty());
        assert_eq!(pcache.len(), 0);

        let snapshot = CachedKvState {
            keys: vec![vec![0.0; 4]],
            values: vec![vec![0.0; 4]],
            seq_len: 2,
        };
        pcache.store_snapshot(&[1, 2], snapshot);

        assert!(!pcache.is_empty());
        assert_eq!(pcache.len(), 1);
    }

    // ── common_prefix_len helper ─────────────────────────────────────────

    #[test]
    fn test_common_prefix_len() {
        assert_eq!(common_prefix_len(&[], &[]), 0);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[]), 0);
        assert_eq!(common_prefix_len(&[], &[1, 2, 3]), 0);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[4, 5, 6]), 0);
        assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3, 4]), 2);
    }

    // ── Radix tree node splitting ────────────────────────────────────────

    #[test]
    fn test_node_split_preserves_data() {
        let mut pcache = PrefixKvCache::new(default_config());

        // Insert [1,2,3,4] then [1,2,5,6]. This forces a split at [1,2].
        let snap_a = CachedKvState {
            keys: vec![vec![1.0; 4]],
            values: vec![vec![2.0; 4]],
            seq_len: 4,
        };
        let snap_b = CachedKvState {
            keys: vec![vec![3.0; 4]],
            values: vec![vec![4.0; 4]],
            seq_len: 4,
        };

        pcache.store_snapshot(&[1, 2, 3, 4], snap_a);
        pcache.store_snapshot(&[1, 2, 5, 6], snap_b);

        assert_eq!(pcache.len(), 2);

        // Both lookups should still succeed.
        let (m_a, kv_a) = pcache.lookup(&[1, 2, 3, 4]).expect("lookup A");
        assert_eq!(m_a, 4);
        assert_eq!(kv_a.keys()[0][0], 1.0);

        let (m_b, kv_b) = pcache.lookup(&[1, 2, 5, 6]).expect("lookup B");
        assert_eq!(m_b, 4);
        assert_eq!(kv_b.keys()[0][0], 3.0);
    }
}

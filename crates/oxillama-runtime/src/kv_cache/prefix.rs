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
use std::sync::Arc;
use std::time::Instant;

use oxillama_arch::traits::KvCacheAccess;

use crate::error::RuntimeResult;

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
///
/// The per-layer buffers sit behind an [`Arc`], so cloning a `CachedKvState` —
/// which the radix tree does whenever a node is split and a caller does when
/// lifting a hit out from under a lock — is a refcount bump rather than a copy
/// of `num_layers × seq_len × kv_dim × 4 × 2` bytes.  For a 32-layer model with
/// a 2000-token prompt at `kv_dim` 1024 that is ~520 MB per clone avoided.
#[derive(Clone)]
pub struct CachedKvState {
    /// Per-layer key tensors flattened: `[layer][seq_pos * kv_dim]`.
    keys: Arc<Vec<Vec<f32>>>,
    /// Per-layer value tensors flattened.
    values: Arc<Vec<Vec<f32>>>,
    /// Number of tokens this state covers.
    seq_len: usize,
    /// Byte footprint, computed once at construction.
    ///
    /// The cache's running memory counter reads this; recomputing it by summing
    /// every layer on each eviction-loop iteration is what made `evict_lru`
    /// quadratic.
    memory_bytes: usize,
}

impl CachedKvState {
    /// Construct a new `CachedKvState` from pre-built KV buffers.
    ///
    /// This is the public constructor used when re-assembling a state from
    /// cloned data (e.g. after releasing a `Mutex` lock on a `PrefixKvCache`).
    /// `keys` and `values` must each have one inner `Vec<f32>` per layer.
    pub fn new(keys: Vec<Vec<f32>>, values: Vec<Vec<f32>>, seq_len: usize) -> Self {
        let float_count: usize = keys.iter().chain(values.iter()).map(|v| v.len()).sum();
        Self {
            keys: Arc::new(keys),
            values: Arc::new(values),
            seq_len,
            memory_bytes: float_count * std::mem::size_of::<f32>(),
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
    ///
    /// Counts the buffers once regardless of how many `CachedKvState` handles
    /// share them, which is the right accounting for the cache's budget: an
    /// entry is only freed when the last handle to it goes away.
    pub fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    /// The number of layers this snapshot covers.
    pub fn num_layers(&self) -> usize {
        self.keys.len()
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
}

// NOTE: this node used to carry a `ref_count` field described as "how many
// active sequences use this prefix", consulted by `find_lru_candidate` as an
// "in use, don't evict" guard.  Nothing ever incremented it, so the guard was
// inert — but it was also unnecessary.  `PrefixKvCache::lookup` takes
// `&mut self` and hands back a `&CachedKvState` borrowed from it, so the borrow
// checker already forbids any `store`/evict call while a caller holds a hit,
// and every consumer that needs the data beyond that borrow (the server's
// `try_prefix_cache_hit`) clones it — now an `Arc` bump — before releasing the
// lock.  The field is gone rather than left as a lie.

impl RadixNode {
    /// Create a new node with the given token segment.
    fn new(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            children: HashMap::new(),
            cached_kv: None,
            last_access: Instant::now(),
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
    ///
    /// Returns the entry that was displaced, if any, so the cache can keep its
    /// running entry/byte counters exact without re-walking the tree.
    fn insert(&mut self, tokens: &[u32], kv: CachedKvState) -> Option<CachedKvState> {
        if tokens.is_empty() {
            self.last_access = Instant::now();
            return self.cached_kv.replace(kv);
        }

        let common = common_prefix_len(&self.tokens, tokens);

        if common < self.tokens.len() {
            // Need to split this node.
            self.split_at(common);
        }

        let remaining = &tokens[common..];
        if remaining.is_empty() {
            self.last_access = Instant::now();
            return self.cached_kv.replace(kv);
        }

        let first = remaining[0];
        let child = self
            .children
            .entry(first)
            .or_insert_with(|| Box::new(RadixNode::new(remaining.to_vec())));

        // If the child already exists, recurse into it.
        if child.tokens == remaining {
            child.last_access = Instant::now();
            child.cached_kv.replace(kv)
        } else {
            child.insert(remaining, kv)
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

        self.tokens.truncate(pos);
        self.children.insert(first_of_suffix, Box::new(new_child));
    }

    /// Count the number of nodes that carry cached KV data.
    ///
    /// Only the audit path uses this now — the cache keeps a running counter.
    #[cfg(test)]
    fn count_entries(&self) -> usize {
        let mine = usize::from(self.cached_kv.is_some());
        let children_count: usize = self.children.values().map(|c| c.count_entries()).sum();
        mine + children_count
    }

    /// Sum the estimated memory of all cached KV states in this subtree.
    ///
    /// Only the audit path uses this now — the cache keeps a running counter.
    #[cfg(test)]
    fn total_memory(&self) -> usize {
        let mine = self.cached_kv.as_ref().map_or(0, |kv| kv.memory_bytes());
        let children_mem: usize = self.children.values().map(|c| c.total_memory()).sum();
        mine + children_mem
    }

    /// Find and remove the LRU eviction candidate in this subtree.
    ///
    /// Returns the number of bytes freed, or `None` when the subtree holds no
    /// evictable entry.  `Some(0)` is a real outcome (a zero-length snapshot
    /// still occupies an entry slot), which is why this is an `Option` rather
    /// than a bare count — the old `0 means nothing happened` convention made
    /// `evict_lru` spin forever on such an entry.
    fn evict_lru_one(&mut self) -> Option<usize> {
        // Collect candidates: this node and all descendants.
        let mut oldest_time = Instant::now();
        let mut oldest_path: Option<Vec<u32>> = None;

        self.find_lru_candidate(&mut oldest_time, &mut oldest_path, &[]);

        let path = oldest_path?;
        self.remove_cached_at(&path)
    }

    /// Recursively find the least-recently-used node that carries cached data.
    fn find_lru_candidate(
        &self,
        oldest_time: &mut Instant,
        oldest_path: &mut Option<Vec<u32>>,
        prefix: &[u32],
    ) {
        if self.cached_kv.is_some() && (oldest_path.is_none() || self.last_access < *oldest_time) {
            *oldest_time = self.last_access;
            let mut path = prefix.to_vec();
            path.extend_from_slice(&self.tokens);
            *oldest_path = Some(path);
        }

        for child in self.children.values() {
            let mut child_prefix = prefix.to_vec();
            child_prefix.extend_from_slice(&self.tokens);
            child.find_lru_candidate(oldest_time, oldest_path, &child_prefix);
        }
    }

    /// Remove cached KV data at the node reached by following `path` tokens.
    ///
    /// Returns `Some(bytes_freed)` when an entry was actually removed.
    fn remove_cached_at(&mut self, path: &[u32]) -> Option<usize> {
        let common = common_prefix_len(&self.tokens, path);
        if common < self.tokens.len() {
            return None;
        }

        let remaining = &path[common..];
        if remaining.is_empty() {
            // This is the target node.
            return self.cached_kv.take().map(|kv| kv.memory_bytes());
        }

        let &first = remaining.first()?;
        let child = self.children.get_mut(&first)?;
        let freed = child.remove_cached_at(remaining);
        // If the child is now empty (no cache, no children), prune it.
        if child.cached_kv.is_none() && child.children.is_empty() {
            self.children.remove(&first);
        }
        freed
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
    /// Running count of nodes carrying cached KV data.
    ///
    /// Maintained incrementally.  `evict_lru` used to call `count_entries()`
    /// and `total_memory()` — each a full tree traversal — on *every* iteration
    /// of both eviction loops, and `evict_lru_one` traversed twice more, so a
    /// single `store()` that had to evict `k` entries from a tree of `n` nodes
    /// cost `O(k · n)`.
    entry_count: usize,
    /// Running total of [`CachedKvState::memory_bytes`] across all entries.
    memory_bytes: usize,
}

impl PrefixKvCache {
    /// Create a new prefix KV cache with the given configuration.
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            root: RadixNode::new(Vec::new()),
            config,
            hit_count: 0,
            miss_count: 0,
            entry_count: 0,
            memory_bytes: 0,
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
    /// [`KvCacheAccess`] trait.  Returns `true` when an entry was stored.
    ///
    /// # The length invariant
    ///
    /// The trie key is `tokens`, so the snapshot must cover **exactly**
    /// `tokens.len()` positions:
    ///
    /// * If `seq_len > tokens.len()` the snapshot is **truncated** to the key
    ///   length.  This is the normal case for a server that stores after the
    ///   decode loop has run: `InferenceEngine::store_kv_in_prefix_cache` used
    ///   to pass `kv.seq_len()` — prompt *plus everything generated* — and the
    ///   entry then retained the completion's KV for a key that never mentions
    ///   it.  Positions past the prompt are simply not part of this prefix.
    /// * If `seq_len < tokens.len()` the store is **refused**.  Storing it
    ///   would let a later `lookup` report `matched = tokens.len()` against a
    ///   shorter snapshot, and `prime_with_prefix` would then mark unwritten
    ///   positions valid.
    ///
    /// A prefix shorter than `min_prefix_len` is skipped, as before.
    pub fn store(
        &mut self,
        tokens: &[u32],
        kv_cache: &dyn KvCacheAccess,
        seq_len: usize,
        kv_dim: usize,
        num_layers: usize,
    ) -> bool {
        if tokens.len() < self.config.min_prefix_len {
            return false;
        }
        if seq_len < tokens.len() {
            tracing::warn!(
                tokens = tokens.len(),
                seq_len,
                "prefix cache store refused: the KV state is shorter than its trie key"
            );
            return false;
        }
        let store_len = tokens.len();

        // Snapshot the KV state from the live cache.
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        let end = store_len * kv_dim;

        for layer in 0..num_layers {
            // `fetch_keys`/`fetch_values` borrow FP32 contiguous storage and
            // gather anything else (FP16 elements, paged layouts), so a prefix
            // cache in front of an f16 KV cache stores real data instead of
            // refusing every layer.
            let (k, v) = match (
                oxillama_arch::common::fetch_keys(kv_cache, layer),
                oxillama_arch::common::fetch_values(kv_cache, layer),
            ) {
                (Ok(k), Ok(v)) => (k, v),
                (Err(e), _) | (_, Err(e)) => {
                    // Previously `unwrap_or(&[])`, which stored an entry with
                    // empty layers and turned an unreadable cache into a
                    // silently corrupt cache hit later on.
                    tracing::warn!(
                        layer,
                        error = %e,
                        "prefix cache store refused: KV cache layer is not readable"
                    );
                    return false;
                }
            };
            if k.len() < end || v.len() < end {
                tracing::warn!(
                    layer,
                    keys = k.len(),
                    values = v.len(),
                    needed = end,
                    "prefix cache store refused: KV cache layer is shorter than the prefix"
                );
                return false;
            }
            keys.push(k[..end].to_vec());
            values.push(v[..end].to_vec());
        }

        self.insert_entry(tokens, CachedKvState::new(keys, values, store_len));
        true
    }

    /// Store a pre-built [`CachedKvState`] directly for a token prefix.
    ///
    /// This is useful when the caller has already constructed the snapshot.
    /// Returns `true` when an entry was stored; the same length invariant as
    /// [`store`](Self::store) applies, except that a snapshot longer than the
    /// key cannot be truncated here (the caller built it, so a mismatch is a
    /// caller bug) and is refused.
    pub fn store_snapshot(&mut self, tokens: &[u32], snapshot: CachedKvState) -> bool {
        if tokens.len() < self.config.min_prefix_len {
            return false;
        }
        if snapshot.seq_len() != tokens.len() {
            tracing::warn!(
                tokens = tokens.len(),
                seq_len = snapshot.seq_len(),
                "prefix cache store_snapshot refused: snapshot length must equal the key length"
            );
            return false;
        }
        self.insert_entry(tokens, snapshot);
        true
    }

    /// Insert `snapshot` under `tokens`, keeping the running counters exact.
    fn insert_entry(&mut self, tokens: &[u32], snapshot: CachedKvState) {
        let added = snapshot.memory_bytes();
        match self.root.insert(tokens, snapshot) {
            Some(replaced) => {
                self.memory_bytes = self.memory_bytes + added - replaced.memory_bytes();
            }
            None => {
                self.entry_count += 1;
                self.memory_bytes += added;
            }
        }
        self.evict_lru();
    }

    /// Restore a cached prefix into a live KV cache.
    ///
    /// Copies the cached KV data into the target cache's buffers and resets
    /// the target's sequence position to match the snapshot.
    ///
    /// # Errors
    ///
    /// Propagates [`KvCache::restore_from_snapshot`]'s validation: the target's
    /// layer count must match and every layer must actually carry
    /// `seq_len * kv_dim` floats.
    pub fn restore(cached: &CachedKvState, target: &mut KvCache) -> RuntimeResult<()> {
        target.restore_from_snapshot(&cached.keys, &cached.values, cached.seq_len)
    }

    /// Evict least-recently-used entries until both limits are satisfied.
    fn evict_lru(&mut self) {
        while self.entry_count > self.config.max_entries
            || self.memory_bytes > self.config.max_memory_bytes
        {
            match self.root.evict_lru_one() {
                Some(freed) => {
                    self.entry_count = self.entry_count.saturating_sub(1);
                    self.memory_bytes = self.memory_bytes.saturating_sub(freed);
                }
                // No evictable entry left; the limits cannot be met.
                None => break,
            }
        }
    }

    /// Current number of cached prefixes (nodes with KV data).
    pub fn len(&self) -> usize {
        self.entry_count
    }

    /// Whether the cache is empty (no cached KV data).
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.root.clear_all();
        self.hit_count = 0;
        self.miss_count = 0;
        self.entry_count = 0;
        self.memory_bytes = 0;
    }

    /// Current estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.memory_bytes
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
            let snapshot = CachedKvState::new(vec![vec![i as f32; 4]], vec![vec![i as f32; 4]], 2);
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
            let snapshot = CachedKvState::new(vec![vec![i as f32; 4]], vec![vec![i as f32; 4]], 2);
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
        PrefixKvCache::restore(&cached_kv_clone, &mut target).expect("restore must succeed");

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
        let snapshot = CachedKvState::new(vec![vec![0.0f32; 8]], vec![vec![0.0f32; 8]], 2);
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
        let snapshot = CachedKvState::new(vec![vec![0.0; 4]], vec![vec![0.0; 4]], 2);
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

        let snapshot = CachedKvState::new(vec![vec![0.0; 4]], vec![vec![0.0; 4]], 2);
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
        let snap_a = CachedKvState::new(vec![vec![1.0; 4]], vec![vec![2.0; 4]], 4);
        let snap_b = CachedKvState::new(vec![vec![3.0; 4]], vec![vec![4.0; 4]], 4);

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

    // ── Running counters vs. tree traversal ──────────────────────────────

    /// `len()` and `memory_usage()` are maintained incrementally now instead of
    /// being recomputed by a full tree walk on every eviction-loop iteration.
    /// That is only sound if they stay exactly equal to what the walk reports,
    /// so this test compares them after insert / replace / split / evict
    /// traffic.
    #[test]
    fn running_counters_match_a_full_traversal() {
        let mut pcache = PrefixKvCache::new(PrefixCacheConfig {
            max_entries: 4,
            max_memory_bytes: 16 * 1024,
            min_prefix_len: 1,
        });

        let mk =
            |n: usize| CachedKvState::new(vec![vec![0.0f32; 4]; 2], vec![vec![0.0f32; 4]; 2], n);

        let keys: Vec<Vec<u32>> = vec![
            vec![1, 2, 3, 4],
            vec![1, 2, 5, 6], // forces a split at [1,2]
            vec![1, 2],       // lands on the split node itself
            vec![9, 9, 9, 9],
            vec![1, 2, 3, 4], // a replacement, not a new entry
            vec![7, 7],
            vec![8, 8], // pushes past max_entries, forcing eviction
        ];

        for key in &keys {
            let n = key.len();
            pcache.store_snapshot(key, mk(n));
            assert_eq!(
                pcache.len(),
                pcache.root.count_entries(),
                "entry counter drifted from the tree after storing {key:?}"
            );
            assert_eq!(
                pcache.memory_usage(),
                pcache.root.total_memory(),
                "memory counter drifted from the tree after storing {key:?}"
            );
        }

        pcache.clear();
        assert_eq!(pcache.len(), pcache.root.count_entries());
        assert_eq!(pcache.memory_usage(), pcache.root.total_memory());
    }
}

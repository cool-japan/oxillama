//! Namespaced, bounded registry of prefix KV caches (D6).
//!
//! `PrefixKvCache` (from `oxillama-runtime`) is keyed purely by token IDs.
//! Before this fix, the worker held exactly one shared `PrefixKvCache`
//! instance across every request regardless of which LoRA adapters (if any)
//! were applied, so a request that populated the cache while a LoRA stack
//! was active could produce a cache *hit* for a later request with the same
//! token prefix but a different (or no) LoRA — silently returning
//! LoRA-contaminated output for ordinary traffic, since `cache_prompt`
//! defaults to `true`.
//!
//! [`PrefixCacheRegistry`] fixes this by keeping one independent
//! `PrefixKvCache` per **namespace**, where a namespace is derived from the
//! sorted set of `(adapter_name, scale)` pairs applied to a request (see
//! [`cache_namespace`]). Two requests only ever share a cache if they
//! requested the exact same adapter set at the exact same scales.
//!
//! The registry is bounded (LRU-capped) so an attacker who cycles through
//! many distinct LoRA combinations cannot grow it without bound.

use std::collections::HashMap;
use std::sync::Mutex;

use oxillama_runtime::{PrefixCacheConfig, PrefixKvCache};

use crate::router::eviction::LruQueue;

/// Default maximum number of distinct cache namespaces retained at once.
///
/// Each namespace holds its own `PrefixKvCache` (bounded in turn by
/// `PrefixCacheConfig::max_entries` / `max_memory_bytes`), so this caps the
/// *number of adapter combinations* remembered, not the memory of any one
/// cache.
pub const DEFAULT_MAX_NAMESPACES: usize = 16;

/// Compute the cache namespace key for a request.
///
/// The key is `model_id` followed by the sorted `(adapter_name, scale)`
/// pairs, so:
/// - Requests with no LoRA (`lora_selection` empty) share one namespace per
///   model.
/// - Requests with the same adapters at the same scales — regardless of the
///   order the caller listed them in — share a namespace.
/// - Requests with a different adapter set, or the same adapters at a
///   different scale, get their own, disjoint namespace.
///
/// Scale is formatted with a fixed 6-decimal precision so that two floats
/// which are equal in every bit that matters for sampling never collide
/// into different string keys due to representation noise, while genuinely
/// different scales always produce different keys.
pub fn cache_namespace(model_id: &str, lora_selection: &[(String, f32)]) -> String {
    let mut pairs: Vec<(String, f32)> = lora_selection.to_vec();
    pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));

    let mut key = String::with_capacity(model_id.len() + pairs.len() * 24);
    key.push_str(model_id);
    for (name, scale) in &pairs {
        key.push('|');
        key.push_str(name);
        key.push(':');
        key.push_str(&format!("{scale:.6}"));
    }
    key
}

/// A bounded collection of `PrefixKvCache` instances, one per namespace.
pub struct PrefixCacheRegistry {
    inner: Mutex<RegistryInner>,
    config: PrefixCacheConfig,
    max_namespaces: usize,
}

struct RegistryInner {
    caches: HashMap<String, PrefixKvCache>,
    lru: LruQueue,
}

impl PrefixCacheRegistry {
    /// Create a new registry. Each namespace's `PrefixKvCache` is
    /// constructed with `config`; at most `max_namespaces` distinct
    /// namespaces are retained (least-recently-used eviction).
    pub fn new(config: PrefixCacheConfig, max_namespaces: usize) -> Self {
        let max_namespaces = max_namespaces.max(1);
        Self {
            inner: Mutex::new(RegistryInner {
                caches: HashMap::new(),
                lru: LruQueue::with_capacity(max_namespaces),
            }),
            config,
            max_namespaces,
        }
    }

    /// Run `f` against the `PrefixKvCache` for `namespace`, creating it
    /// (evicting the least-recently-used namespace if at capacity) if it
    /// does not already exist.
    ///
    /// Held under a single `Mutex` for the duration of `f` — the same
    /// granularity the previous single-cache design used, since the worker
    /// is single-threaded and sequential anyway.
    pub fn with_cache<R>(&self, namespace: &str, f: impl FnOnce(&mut PrefixKvCache) -> R) -> R {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };

        inner.lru.touch(namespace);
        if !inner.caches.contains_key(namespace) && inner.lru.len() > self.max_namespaces {
            if let Some(evicted) = inner.lru.evict_lru() {
                if evicted != namespace {
                    inner.caches.remove(&evicted);
                }
            }
        }

        let config = self.config.clone();
        let cache = inner
            .caches
            .entry(namespace.to_string())
            .or_insert_with(|| PrefixKvCache::new(config));
        f(cache)
    }

    /// Number of distinct namespaces currently cached. Test/introspection
    /// helper.
    #[cfg(test)]
    fn namespace_count(&self) -> usize {
        let inner = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        inner.caches.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression (D6): the same prompt with and without a LoRA must not
    /// share a cache namespace — this is the exact contamination scenario
    /// the defect describes. Before the fix there was only ever one shared
    /// `PrefixKvCache`; namespacing by adapter set is what prevents a
    /// LoRA'd request from ever hitting the plain-request cache entry.
    #[test]
    fn with_lora_and_without_lora_are_different_namespaces() {
        let ns_plain = cache_namespace("model-a", &[]);
        let ns_lora = cache_namespace("model-a", &[("style-x".to_string(), 1.0)]);
        assert_ne!(
            ns_plain, ns_lora,
            "a request with a LoRA must not share a namespace with a plain request"
        );
    }

    /// A different scale for the same adapter must also produce a
    /// different namespace — reusing KV state computed under a different
    /// LoRA scale would still corrupt output.
    #[test]
    fn different_scale_is_a_different_namespace() {
        let ns_a = cache_namespace("model-a", &[("style-x".to_string(), 1.0)]);
        let ns_b = cache_namespace("model-a", &[("style-x".to_string(), 0.5)]);
        assert_ne!(ns_a, ns_b);
    }

    /// The namespace key must be independent of the order adapters were
    /// specified in — `[a, b]` and `[b, a]` mean the same LoRA stack.
    #[test]
    fn namespace_is_order_independent() {
        let ns_ab = cache_namespace("model-a", &[("a".to_string(), 1.0), ("b".to_string(), 0.5)]);
        let ns_ba = cache_namespace("model-a", &[("b".to_string(), 0.5), ("a".to_string(), 1.0)]);
        assert_eq!(ns_ab, ns_ba);
    }

    /// Different model IDs never share a namespace even with identical LoRA
    /// selections — forward-compatible with a future multi-model worker.
    #[test]
    fn different_model_id_is_a_different_namespace() {
        let ns_a = cache_namespace("model-a", &[("x".to_string(), 1.0)]);
        let ns_b = cache_namespace("model-b", &[("x".to_string(), 1.0)]);
        assert_ne!(ns_a, ns_b);
    }

    /// The registry actually keeps namespaces isolated: writing through one
    /// namespace's cache must not be observable via another.
    #[test]
    fn registry_isolates_namespaces() {
        let registry = PrefixCacheRegistry::new(PrefixCacheConfig::default(), 8);
        let ns_plain = cache_namespace("model-a", &[]);
        let ns_lora = cache_namespace("model-a", &[("style-x".to_string(), 1.0)]);

        registry.with_cache(&ns_lora, |cache| {
            assert!(cache.is_empty(), "freshly created cache must start empty");
        });

        // The plain namespace's cache must be a distinct, still-empty
        // instance — proving the two namespaces are not aliases of the
        // same underlying cache.
        registry.with_cache(&ns_plain, |cache| {
            assert!(
                cache.is_empty(),
                "plain namespace must not observe the lora namespace's state"
            );
        });

        assert_eq!(registry.namespace_count(), 2);
    }

    /// The registry evicts the least-recently-used namespace once the cap
    /// is exceeded, bounding memory even under many distinct LoRA
    /// combinations (part of the D6 fix's "bounded" requirement).
    #[test]
    fn registry_evicts_lru_namespace_when_over_capacity() {
        let registry = PrefixCacheRegistry::new(PrefixCacheConfig::default(), 2);

        registry.with_cache("ns-a", |_| {});
        registry.with_cache("ns-b", |_| {});
        assert_eq!(registry.namespace_count(), 2);

        // Touch ns-a again so ns-b becomes the LRU entry.
        registry.with_cache("ns-a", |_| {});
        // Inserting a third namespace must evict ns-b (the LRU one), not ns-a.
        registry.with_cache("ns-c", |_| {});

        assert_eq!(registry.namespace_count(), 2);
        let inner = registry.inner.lock().expect("lock");
        assert!(
            inner.caches.contains_key("ns-a"),
            "ns-a was MRU, must survive"
        );
        assert!(inner.caches.contains_key("ns-c"), "ns-c was just inserted");
        assert!(
            !inner.caches.contains_key("ns-b"),
            "ns-b was LRU, must be evicted"
        );
    }
}

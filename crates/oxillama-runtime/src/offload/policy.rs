//! Offload policy configuration.
//!
//! [`OffloadPolicy`] is the declarative entry-point that callers use to
//! express how aggressively the runtime should evict model weights to disk.
//! It is stored in [`EngineConfig`][crate::engine::EngineConfig] and
//! interpreted by [`InferenceEngine`][crate::engine::InferenceEngine] during
//! model loading to build (or skip) the [`LayerPager`][super::LayerPager].

/// Configures which weights stay in RAM vs. get evicted to disk.
///
/// # Variants
///
/// - `None` — all layer weights remain in RAM. This is the default and matches
///   the classic llama.cpp in-memory-only behaviour.
/// - `Budget` — evict weights until the resident set fits within `ram_bytes`.
///   The LRU pager is activated; any tensor not pinned can be evicted.
/// - `PinnedHotSet` — like `Budget` but a named hot-set (embeddings, output
///   head, last N attention layers) is pinned and never evicted. Cold layers
///   cycle in and out of RAM as they are needed.
#[derive(Debug, Clone, Default)]
pub enum OffloadPolicy {
    /// All layer weights remain in RAM (default, current behaviour).
    #[default]
    None,

    /// Evict weights until resident set fits within `ram_bytes`.
    Budget {
        /// Maximum number of bytes allowed to be resident simultaneously.
        ram_bytes: u64,
    },

    /// Pinned hot-set: evict cold layers but keep embeddings, output head,
    /// and the last N attention layers always resident.
    PinnedHotSet {
        /// Maximum number of bytes allowed to be resident simultaneously.
        ram_bytes: u64,
        /// Keep token-embedding table pinned (never evicted).
        pin_embeddings: bool,
        /// Keep output LM head pinned (never evicted).
        pin_output_head: bool,
        /// Number of trailing (deepest) attention layers to always keep resident.
        pin_last_n_layers: usize,
        /// How many layers ahead to prefetch in the background.
        ///
        /// Currently advisory — the pager does not spawn a prefetch thread
        /// automatically; callers can use this value to decide whether to
        /// pre-acquire tensors ahead of the current layer index.
        prefetch_n_ahead: usize,
    },
}

impl OffloadPolicy {
    /// Return the RAM budget in bytes, if any eviction limit is set.
    ///
    /// Returns `None` for [`OffloadPolicy::None`] (unlimited RAM usage).
    pub fn ram_budget_bytes(&self) -> Option<u64> {
        match self {
            Self::None => None,
            Self::Budget { ram_bytes } => Some(*ram_bytes),
            Self::PinnedHotSet { ram_bytes, .. } => Some(*ram_bytes),
        }
    }

    /// Returns `true` if offloading is disabled (i.e. the default in-RAM path).
    pub fn is_disabled(&self) -> bool {
        matches!(self, Self::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_none() {
        assert!(matches!(OffloadPolicy::default(), OffloadPolicy::None));
    }

    #[test]
    fn none_has_no_budget() {
        assert_eq!(OffloadPolicy::None.ram_budget_bytes(), None);
    }

    #[test]
    fn budget_returns_bytes() {
        let policy = OffloadPolicy::Budget { ram_bytes: 1024 * 1024 * 1024 };
        assert_eq!(policy.ram_budget_bytes(), Some(1024 * 1024 * 1024));
    }

    #[test]
    fn pinned_hot_set_returns_bytes() {
        let policy = OffloadPolicy::PinnedHotSet {
            ram_bytes: 512 * 1024 * 1024,
            pin_embeddings: true,
            pin_output_head: true,
            pin_last_n_layers: 4,
            prefetch_n_ahead: 2,
        };
        assert_eq!(policy.ram_budget_bytes(), Some(512 * 1024 * 1024));
    }

    #[test]
    fn none_is_disabled() {
        assert!(OffloadPolicy::None.is_disabled());
        assert!(!OffloadPolicy::Budget { ram_bytes: 1 }.is_disabled());
    }

    #[test]
    fn policy_clone_is_independent() {
        let original = OffloadPolicy::PinnedHotSet {
            ram_bytes: 100,
            pin_embeddings: false,
            pin_output_head: true,
            pin_last_n_layers: 2,
            prefetch_n_ahead: 1,
        };
        let cloned = original.clone();
        // Both should format without panic
        let _ = format!("{original:?}");
        let _ = format!("{cloned:?}");
    }
}

//! LRU eviction tracker for the model pool.
//!
//! This module manages the Least-Recently-Used (LRU) ordering of loaded model
//! identifiers independently from the HashMap storage, keeping eviction
//! decisions O(1) under a single `Mutex<VecDeque>`.

use std::collections::VecDeque;

/// LRU ordered queue of model identifiers.
///
/// The *front* of the deque is the least-recently-used model (first eviction
/// candidate); the *back* is the most-recently-used.
pub struct LruQueue {
    inner: VecDeque<String>,
}

impl LruQueue {
    /// Create an empty LRU queue with the given initial capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: VecDeque::with_capacity(cap),
        }
    }

    /// Record that `model_id` was just used, promoting it to the MRU position.
    ///
    /// If `model_id` already exists in the queue it is moved rather than
    /// duplicated. If it is new it is appended at the back.
    pub fn touch(&mut self, model_id: &str) {
        // Remove existing entry (linear scan is acceptable — capacity is tiny).
        self.inner.retain(|id| id != model_id);
        // Push to the MRU end.
        self.inner.push_back(model_id.to_string());
    }

    /// Remove the least-recently-used entry and return it.
    ///
    /// Returns `None` when the queue is empty.
    pub fn evict_lru(&mut self) -> Option<String> {
        self.inner.pop_front()
    }

    /// Remove `model_id` from the queue (used when a model is explicitly
    /// unloaded rather than evicted by pressure).
    pub fn remove(&mut self, model_id: &str) {
        self.inner.retain(|id| id != model_id);
    }

    /// Number of models currently tracked.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over all tracked IDs from LRU to MRU.
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.inner.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_appends_new_entry() {
        let mut q = LruQueue::with_capacity(4);
        q.touch("a");
        q.touch("b");
        q.touch("c");
        assert_eq!(q.len(), 3);
        // a is LRU — should evict first
        assert_eq!(q.evict_lru().as_deref(), Some("a"));
    }

    #[test]
    fn touch_existing_promotes_to_mru() {
        let mut q = LruQueue::with_capacity(4);
        q.touch("a");
        q.touch("b");
        q.touch("c");
        // Promote a to MRU
        q.touch("a");
        // Now b is LRU
        assert_eq!(q.evict_lru().as_deref(), Some("b"));
        // Then c
        assert_eq!(q.evict_lru().as_deref(), Some("c"));
        // Then a (was MRU)
        assert_eq!(q.evict_lru().as_deref(), Some("a"));
    }

    #[test]
    fn evict_lru_from_empty_returns_none() {
        let mut q = LruQueue::with_capacity(4);
        assert!(q.evict_lru().is_none());
    }

    #[test]
    fn remove_explicit_entry() {
        let mut q = LruQueue::with_capacity(4);
        q.touch("a");
        q.touch("b");
        q.remove("a");
        assert_eq!(q.len(), 1);
        assert_eq!(q.evict_lru().as_deref(), Some("b"));
    }

    #[test]
    fn is_empty_reflects_state() {
        let mut q = LruQueue::with_capacity(4);
        assert!(q.is_empty());
        q.touch("x");
        assert!(!q.is_empty());
        let _ = q.evict_lru();
        assert!(q.is_empty());
    }
}

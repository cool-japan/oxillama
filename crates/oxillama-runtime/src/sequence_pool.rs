//! SSM runtime bridge — polymorphic sequence-state pool.
//!
//! # Overview
//!
//! OxiLLaMa supports two categories of model architectures:
//!
//! 1. **Attention-based** (LLaMA, Qwen3, Mistral, Gemma, Phi, …): per-sequence
//!    state is the KV cache (a contiguous K/V buffer per layer).
//! 2. **SSM-based** (Mamba-2, …): per-sequence state is a set of per-layer
//!    recurrent hidden vectors; there is no KV cache.
//!
//! The [`SequencePool`] enum abstracts over both kinds via the
//! [`oxillama_arch::common::sequence_state::SequenceState`] trait.  The engine
//! picks the right pool variant at load time by examining the loaded
//! architecture; both variants expose the same `alloc` / `release` / `slot`
//! interface so the rest of the engine stays arch-agnostic.
//!
//! ## Design notes
//!
//! - Slots are identified by a `usize` index (same as `Sequence::slot_id`).
//! - A slot is "live" when it holds a `Box<dyn SequenceState>`.
//! - On `release` the state is **reset** (zeroed) and returned to the free pool.
//! - Neither variant interacts with the KV cache from `kv_cache/mod.rs`; the
//!   KV-based pool manages its own separate per-slot state.
//! - The SSM state pool owns the `Box<dyn SequenceState>` objects outright;
//!   the KV-based pool keeps a `KvCachePool` from which page indices are lent.
//!
//! ## Thread safety
//!
//! `SequencePool` is **not** `Send` + `Sync` by itself; it is intended to be
//! owned by a single-threaded engine or wrapped in a `Mutex` by the caller.

use crate::kv_pool::KvCachePool;
use oxillama_arch::common::sequence_state::SequenceState;
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by pool operations.
#[derive(Debug, Error)]
pub enum PoolError {
    /// The pool has no free slots; the request must wait or be rejected.
    #[error("sequence pool exhausted: no free slots available")]
    Exhausted,
    /// A slot index passed to a pool operation does not identify a live slot.
    #[error("invalid slot index {0}: slot is not live or out of range")]
    InvalidSlot(usize),
}

/// Convenience alias.
pub type PoolResult<T> = Result<T, PoolError>;

// ─── SequenceSlot ─────────────────────────────────────────────────────────────

/// A live sequence slot in the [`SsmStatePool`].
///
/// Each slot carries:
/// - `state`: the arch-specific [`SequenceState`] (SSM hidden state or an
///   attention position counter wrapped by the arch crate).
/// - `position`: current token position in the sequence (mirrors
///   `state.step_position()`, but accessible without a vtable call).
/// - `request_id`: the logical request ID associated with this slot (matches
///   [`Sequence::id`](crate::scheduler::Sequence::id)); `0` = unassigned.
pub struct SequenceSlot {
    /// Arch-specific sequence state (SSM hidden vectors, or attention counter).
    pub state: Box<dyn SequenceState>,
    /// Current token position (0-indexed).
    pub position: usize,
    /// Request ID bound to this slot (0 = none).
    pub request_id: u64,
}

impl SequenceSlot {
    /// Create a new slot with the given state and a zero request ID.
    pub fn new(state: Box<dyn SequenceState>) -> Self {
        Self {
            position: state.step_position(),
            state,
            request_id: 0,
        }
    }

    /// Advance the internal position counter by one (after each forward step).
    ///
    /// Also calls `state.advance()` so both the slot's cached `position` and
    /// the underlying trait object stay in sync.
    pub fn step(&mut self) {
        self.state.advance();
        self.position = self.state.step_position();
    }

    /// Reset the slot to position 0 and clear the state.
    ///
    /// The slot's `request_id` is reset to 0 as well so that stale IDs are
    /// not accidentally read after re-allocation.
    pub fn reset(&mut self) {
        self.state.reset();
        self.position = 0;
        self.request_id = 0;
    }
}

// ─── SsmStatePool ─────────────────────────────────────────────────────────────

/// A free-list pool of [`SequenceSlot`]s for SSM-based models.
///
/// Pre-allocates `capacity` slots at construction time.  Slots are identified
/// by their index into the internal `slots` vector.
///
/// Free slots are tracked by a `free_list: Vec<usize>`.  `alloc` pops from the
/// list; `release` resets the slot and pushes back.
pub struct SsmStatePool {
    /// All slots, indexed by slot ID.  `Some` = live (allocated); `None` = never
    /// initialised (only possible during construction before the pool is fully
    /// initialised — in practice every index 0..capacity is always `Some`).
    slots: Vec<Option<SequenceSlot>>,
    /// Indices of slots currently on the free list.
    free_list: Vec<usize>,
    /// Total capacity (never changes after construction).
    capacity: usize,
}

impl SsmStatePool {
    /// Create a pool by calling `ForwardPass::allocate_sequence_state` for each slot.
    ///
    /// This is the preferred construction path because it delegates state
    /// allocation to the architecture implementation, rather than hard-coding
    /// the state type at the call site. The runtime calls this once at model
    /// load time, after `ForwardPass` is available.
    ///
    /// ```ignore
    /// let pool = SsmStatePool::from_forward_pass(fwd_pass.as_ref(), capacity, max_ctx);
    /// ```
    pub fn from_forward_pass(
        forward_pass: &dyn oxillama_arch::traits::ForwardPass,
        capacity: usize,
        max_context_length: usize,
    ) -> Self {
        Self::new(capacity, |_| {
            forward_pass.allocate_sequence_state(max_context_length)
        })
    }

    /// Create a new pool using a factory closure to produce each slot's state.
    ///
    /// The closure is called once per slot with the slot index.  Use it to
    /// initialise arch-specific state (e.g. `Mamba2SequenceState::new(...)`).
    ///
    /// ```ignore
    /// let pool = SsmStatePool::new(8, |_| {
    ///     Box::new(Mamba2SequenceState::new(24, 16, 256, 4096))
    /// });
    /// ```
    pub fn new<F>(capacity: usize, mut make_state: F) -> Self
    where
        F: FnMut(usize) -> Box<dyn SequenceState>,
    {
        let mut slots = Vec::with_capacity(capacity);
        let mut free_list = Vec::with_capacity(capacity);

        for i in 0..capacity {
            let state = make_state(i);
            slots.push(Some(SequenceSlot::new(state)));
            free_list.push(i);
        }

        Self {
            slots,
            free_list,
            capacity,
        }
    }

    /// Allocate a free slot and bind it to `request_id`.
    ///
    /// Returns `Ok(slot_idx)` on success.
    ///
    /// # Errors
    ///
    /// Returns [`PoolError::Exhausted`] when no free slots are available.
    pub fn alloc(&mut self, request_id: u64) -> PoolResult<usize> {
        let idx = self.free_list.pop().ok_or(PoolError::Exhausted)?;
        if let Some(slot) = self.slots[idx].as_mut() {
            slot.request_id = request_id;
        }
        Ok(idx)
    }

    /// Release slot `idx` back to the free list.
    ///
    /// The slot's state is reset (zeroed) and its `request_id` cleared.
    ///
    /// # Errors
    ///
    /// Returns [`PoolError::InvalidSlot`] if `idx` is out of range or already free.
    pub fn release(&mut self, idx: usize) -> PoolResult<()> {
        if idx >= self.slots.len() {
            return Err(PoolError::InvalidSlot(idx));
        }
        // Check that the slot is not already on the free list.
        if self.free_list.contains(&idx) {
            return Err(PoolError::InvalidSlot(idx));
        }
        if let Some(slot) = self.slots[idx].as_mut() {
            slot.reset();
        }
        self.free_list.push(idx);
        Ok(())
    }

    /// Get a shared reference to slot `idx`.
    ///
    /// Returns `None` if the slot has never been initialised (should not happen
    /// for a correctly-constructed pool) or the index is out of range.
    pub fn slot(&self, idx: usize) -> Option<&SequenceSlot> {
        self.slots.get(idx)?.as_ref()
    }

    /// Get a mutable reference to slot `idx`.
    ///
    /// Returns `None` if the slot is uninitialised or out of range.
    pub fn slot_mut(&mut self, idx: usize) -> Option<&mut SequenceSlot> {
        self.slots.get_mut(idx)?.as_mut()
    }

    /// Total pool capacity (never changes).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of currently free (unallocated) slots.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Number of currently allocated (live) slots.
    pub fn used_count(&self) -> usize {
        self.capacity.saturating_sub(self.free_list.len())
    }
}

// ─── SequencePool ─────────────────────────────────────────────────────────────

/// Dispatch-enum over the two pool backends.
///
/// At model-load time the engine inspects the loaded architecture and
/// constructs either a `KvBased` pool (for any transformer) or an `Ssm` pool
/// (for Mamba-2 and similar).  Both variants expose the same interface through
/// [`SequencePool`]'s methods.
///
/// # KV-based pooling
///
/// The KV-based variant stores state in a [`KvCachePool`] of page-sized slabs.
/// Slots are identified by page indices returned by `KvCachePool::alloc`.
///
/// # SSM pooling
///
/// The SSM variant stores the full per-layer recurrent state in an
/// [`SsmStatePool`].  The `alloc_ssm` / `release_ssm` helpers delegate to it.
pub enum SequencePool {
    /// Attention-transformer pool (KV cache pages).
    KvBased(KvCachePool),
    /// SSM pool (per-layer recurrent hidden states).
    Ssm(SsmStatePool),
}

impl SequencePool {
    /// Allocate a slot from the KV-based pool.
    ///
    /// Returns the page index on success.
    ///
    /// # Errors
    ///
    /// `PoolError::Exhausted` if the pool is full.
    /// `PoolError::InvalidSlot(usize::MAX)` if called on an `Ssm` variant.
    pub fn alloc_kv(&mut self) -> PoolResult<usize> {
        match self {
            SequencePool::KvBased(pool) => pool.alloc().ok_or(PoolError::Exhausted),
            SequencePool::Ssm(_) => Err(PoolError::InvalidSlot(usize::MAX)),
        }
    }

    /// Free a page in the KV-based pool.
    ///
    /// # Errors
    ///
    /// `PoolError::InvalidSlot` if called on an `Ssm` variant.
    pub fn free_kv(&mut self, page_idx: usize) -> PoolResult<()> {
        match self {
            SequencePool::KvBased(pool) => {
                pool.free(page_idx);
                Ok(())
            }
            SequencePool::Ssm(_) => Err(PoolError::InvalidSlot(page_idx)),
        }
    }

    /// Allocate an SSM slot bound to `request_id`.
    ///
    /// # Errors
    ///
    /// `PoolError::Exhausted` if the pool is full.
    /// `PoolError::InvalidSlot(usize::MAX)` if called on a `KvBased` variant.
    pub fn alloc_ssm(&mut self, request_id: u64) -> PoolResult<usize> {
        match self {
            SequencePool::Ssm(pool) => pool.alloc(request_id),
            SequencePool::KvBased(_) => Err(PoolError::InvalidSlot(usize::MAX)),
        }
    }

    /// Release an SSM slot by index.
    ///
    /// # Errors
    ///
    /// `PoolError::InvalidSlot` if `idx` is invalid or already free, or if
    /// called on a `KvBased` variant.
    pub fn release_ssm(&mut self, idx: usize) -> PoolResult<()> {
        match self {
            SequencePool::Ssm(pool) => pool.release(idx),
            SequencePool::KvBased(_) => Err(PoolError::InvalidSlot(idx)),
        }
    }

    /// Get an immutable reference to an SSM slot.
    ///
    /// Returns `None` for KV-based pools or out-of-range indices.
    pub fn ssm_slot(&self, idx: usize) -> Option<&SequenceSlot> {
        match self {
            SequencePool::Ssm(pool) => pool.slot(idx),
            SequencePool::KvBased(_) => None,
        }
    }

    /// Get a mutable reference to an SSM slot.
    ///
    /// Returns `None` for KV-based pools or out-of-range indices.
    pub fn ssm_slot_mut(&mut self, idx: usize) -> Option<&mut SequenceSlot> {
        match self {
            SequencePool::Ssm(pool) => pool.slot_mut(idx),
            SequencePool::KvBased(_) => None,
        }
    }

    /// Returns `true` if this pool uses the KV-cache backend.
    pub fn is_kv_based(&self) -> bool {
        matches!(self, SequencePool::KvBased(_))
    }

    /// Returns `true` if this pool uses the SSM backend.
    pub fn is_ssm(&self) -> bool {
        matches!(self, SequencePool::Ssm(_))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_arch::common::sequence_state::{AttentionSequenceState, Mamba2SequenceState};

    // ── SequenceSlot ──────────────────────────────────────────────────────────

    /// Slot step() advances both the cached position and the inner state.
    #[test]
    fn sequence_slot_position_advances() {
        let state = Box::new(AttentionSequenceState::new(512));
        let mut slot = SequenceSlot::new(state);

        assert_eq!(slot.position, 0, "initial position must be 0");
        assert_eq!(slot.state.step_position(), 0);

        slot.step();
        assert_eq!(slot.position, 1, "position after one step must be 1");
        assert_eq!(slot.state.step_position(), 1);

        slot.step();
        slot.step();
        assert_eq!(slot.position, 3);
        assert_eq!(slot.state.step_position(), 3);
    }

    /// Slot reset() clears position and request_id.
    #[test]
    fn sequence_slot_reset_clears_position() {
        let state = Box::new(AttentionSequenceState::new(64));
        let mut slot = SequenceSlot::new(state);
        slot.request_id = 42;
        slot.step();
        slot.step();
        assert_eq!(slot.position, 2);

        slot.reset();
        assert_eq!(slot.position, 0, "position must be 0 after reset");
        assert_eq!(slot.state.step_position(), 0);
        assert_eq!(slot.request_id, 0, "request_id must be cleared by reset");
    }

    // ── SsmStatePool ──────────────────────────────────────────────────────────

    /// Allocate and release cycles must recycle slot indices.
    #[test]
    fn sequence_pool_allocate_release() {
        let mut pool = SsmStatePool::new(4, |_| {
            Box::new(AttentionSequenceState::new(256)) as Box<dyn SequenceState>
        });

        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.free_count(), 4);
        assert_eq!(pool.used_count(), 0);

        let idx_a = pool.alloc(1).expect("first alloc must succeed");
        let idx_b = pool.alloc(2).expect("second alloc must succeed");
        assert_ne!(idx_a, idx_b);
        assert_eq!(pool.used_count(), 2);
        assert_eq!(pool.free_count(), 2);

        // Release one and re-allocate — must reuse the freed slot.
        pool.release(idx_a).expect("release must succeed");
        assert_eq!(pool.free_count(), 3);

        let idx_c = pool.alloc(3).expect("alloc after release");
        assert_eq!(idx_c, idx_a, "freed slot must be reused");
        assert_eq!(pool.used_count(), 2);
    }

    /// Exhausted pool must return PoolError::Exhausted.
    #[test]
    fn ssm_pool_exhaustion_returns_error() {
        let mut pool = SsmStatePool::new(2, |_| {
            Box::new(AttentionSequenceState::new(64)) as Box<dyn SequenceState>
        });

        pool.alloc(10).expect("first");
        pool.alloc(11).expect("second");
        let err = pool.alloc(12);
        assert!(
            matches!(err, Err(PoolError::Exhausted)),
            "exhausted pool must return Exhausted, got {err:?}"
        );
    }

    /// Releasing an already-free slot must return InvalidSlot.
    #[test]
    fn ssm_pool_double_release_errors() {
        let mut pool = SsmStatePool::new(2, |_| {
            Box::new(AttentionSequenceState::new(64)) as Box<dyn SequenceState>
        });

        let idx = pool.alloc(1).expect("alloc");
        pool.release(idx).expect("first release");
        let err = pool.release(idx);
        assert!(
            matches!(err, Err(PoolError::InvalidSlot(_))),
            "double-release must return InvalidSlot, got {err:?}"
        );
    }

    /// Release resets the underlying state (zeroes position and h vectors).
    #[test]
    fn ssm_pool_release_resets_state() {
        let n_layers = 3;
        let d_state = 4;
        let d_inner = 8;
        let mut pool = SsmStatePool::new(2, |_| {
            Box::new(Mamba2SequenceState::new(n_layers, d_state, d_inner, 256))
                as Box<dyn SequenceState>
        });

        let idx = pool.alloc(99).expect("alloc");

        // Advance the slot's position.
        if let Some(slot) = pool.slot_mut(idx) {
            slot.step();
            slot.step();
            assert_eq!(slot.position, 2, "position must be 2 before release");
        }

        pool.release(idx).expect("release");

        // After re-allocation the slot must be fresh (position 0).
        let idx2 = pool.alloc(100).expect("re-alloc");
        assert_eq!(idx2, idx, "must reuse the released slot");
        let slot = pool.slot(idx2).expect("slot must exist");
        assert_eq!(
            slot.position, 0,
            "position must be 0 after re-alloc following release"
        );
        assert_eq!(
            slot.state.step_position(),
            0,
            "state.step_position() must be 0 after release"
        );
        assert_eq!(slot.request_id, 100, "request_id must be updated on alloc");
    }

    // ── SequencePool enum ─────────────────────────────────────────────────────

    /// KvBased pool's alloc_kv / free_kv round-trip works.
    #[test]
    fn sequence_pool_kv_based_alloc_free() {
        let kv_pool = KvCachePool::new(16, 4);
        let mut pool = SequencePool::KvBased(kv_pool);

        assert!(pool.is_kv_based());
        assert!(!pool.is_ssm());

        let idx = pool.alloc_kv().expect("alloc_kv must succeed");
        // Should be a valid page index (0..3).
        assert!(idx < 4, "page index must be in range 0..4, got {idx}");

        pool.free_kv(idx).expect("free_kv must succeed");
    }

    /// Calling alloc_ssm on a KvBased pool must return an error.
    #[test]
    fn sequence_pool_kv_rejects_ssm_ops() {
        let kv_pool = KvCachePool::new(16, 4);
        let mut pool = SequencePool::KvBased(kv_pool);
        let err = pool.alloc_ssm(1);
        assert!(
            matches!(err, Err(PoolError::InvalidSlot(_))),
            "alloc_ssm on KvBased must fail, got {err:?}"
        );
    }

    /// Ssm pool's alloc_ssm / release_ssm round-trip works.
    #[test]
    fn sequence_pool_ssm_alloc_release() {
        let inner = SsmStatePool::new(4, |_| {
            Box::new(AttentionSequenceState::new(256)) as Box<dyn SequenceState>
        });
        let mut pool = SequencePool::Ssm(inner);

        assert!(pool.is_ssm());
        assert!(!pool.is_kv_based());

        let idx = pool.alloc_ssm(7).expect("alloc_ssm");
        let slot = pool.ssm_slot(idx).expect("slot must exist after alloc");
        assert_eq!(slot.request_id, 7);

        pool.release_ssm(idx).expect("release_ssm");
        // After release the slot is on the free list; ssm_slot still returns it
        // (it's physically present), but request_id should have been cleared.
        let slot = pool.ssm_slot(idx).expect("slot still accessible");
        assert_eq!(slot.request_id, 0, "request_id must be 0 after release");
    }

    /// Calling alloc_kv on an Ssm pool must return an error.
    #[test]
    fn sequence_pool_ssm_rejects_kv_ops() {
        let inner = SsmStatePool::new(2, |_| {
            Box::new(AttentionSequenceState::new(64)) as Box<dyn SequenceState>
        });
        let mut pool = SequencePool::Ssm(inner);
        let err = pool.alloc_kv();
        assert!(
            matches!(err, Err(PoolError::InvalidSlot(_))),
            "alloc_kv on Ssm must fail, got {err:?}"
        );
    }

    /// Two independent SSM requests must not share state (isolation test).
    #[test]
    fn mixed_pool_isolation() {
        let n_layers = 2;
        let d_state = 2;
        let d_inner = 4;
        let inner = SsmStatePool::new(4, |_| {
            Box::new(Mamba2SequenceState::new(n_layers, d_state, d_inner, 128))
                as Box<dyn SequenceState>
        });
        let mut pool = SequencePool::Ssm(inner);

        let idx_a = pool.alloc_ssm(1).expect("alloc A");
        let idx_b = pool.alloc_ssm(2).expect("alloc B");
        assert_ne!(idx_a, idx_b, "two requests must occupy different slots");

        // Advance slot A twice.
        if let Some(slot_a) = pool.ssm_slot_mut(idx_a) {
            slot_a.step();
            slot_a.step();
        }

        // Slot B must remain at position 0.
        let slot_b = pool.ssm_slot(idx_b).expect("slot B must exist");
        assert_eq!(
            slot_b.position, 0,
            "slot B position must not be affected by slot A's steps"
        );
    }

    /// Out-of-range slot index must return PoolError::InvalidSlot.
    #[test]
    fn ssm_pool_out_of_range_slot_errors() {
        let mut pool = SsmStatePool::new(2, |_| {
            Box::new(AttentionSequenceState::new(64)) as Box<dyn SequenceState>
        });
        pool.alloc(1).expect("alloc to make slot 0 live");

        let err = pool.release(99); // way out of range
        assert!(
            matches!(err, Err(PoolError::InvalidSlot(99))),
            "out-of-range release must return InvalidSlot(99), got {err:?}"
        );
    }

    /// `slot_reset_on_eos_for_ssm`: when a slot is released (simulating EOS),
    /// the underlying SSM state must be all-zero on next allocation.
    #[test]
    fn slot_reset_on_eos_for_ssm() {
        let n_layers = 2;
        let d_state = 4;
        let d_inner = 8;
        let inner = SsmStatePool::new(2, |_| {
            Box::new(Mamba2SequenceState::new(n_layers, d_state, d_inner, 256))
                as Box<dyn SequenceState>
        });
        let mut pool = SequencePool::Ssm(inner);

        // Allocate, advance a few steps, then simulate EOS by releasing.
        let idx = pool.alloc_ssm(5).expect("alloc");
        if let Some(slot) = pool.ssm_slot_mut(idx) {
            for _ in 0..10 {
                slot.step();
            }
            assert_eq!(slot.position, 10, "must have 10 steps before release");
        }
        pool.release_ssm(idx).expect("release on EOS");

        // Re-allocate: state must be fresh.
        let idx2 = pool.alloc_ssm(6).expect("re-alloc");
        let slot = pool.ssm_slot(idx2).expect("slot must exist");
        assert_eq!(
            slot.position, 0,
            "position must be 0 on fresh re-alloc (EOS reset)"
        );
        assert_eq!(slot.state.step_position(), 0);
    }
}

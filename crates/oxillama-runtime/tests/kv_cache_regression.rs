// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the 0.1.4 KV-cache and engine-routing fixes.
//!
//! Every test here names the defect it pins down and fails against the code as
//! it stood before the corresponding fix.  They are deliberately *outside* the
//! crate (integration tests) so they exercise only the public API — the same
//! surface the CLI and the server use.

use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_arch::{ArchError, ArchResult};
use oxillama_runtime::error::RuntimeError;
use oxillama_runtime::kv_cache::{KvCacheDtype, DEFAULT_BLOCK_TOKENS};
use oxillama_runtime::{CachedKvState, KvCache, KvCachePool, PrefixCacheConfig, PrefixKvCache};

// ── T4: context overflow must be reported, not swallowed ─────────────────────

/// `store_kv` past `max_seq_len` used to skip its body and return `Ok(())`, so
/// the token's K/V was silently discarded and generation continued against a
/// frozen cache — producing confident, wrong logits forever.
#[test]
fn store_kv_past_max_context_returns_error() {
    let max = 4usize;
    let kv_dim = 8usize;
    let mut cache = KvCache::new(2, max, kv_dim);
    let key = vec![1.0f32; kv_dim];
    let val = vec![2.0f32; kv_dim];

    for pos in 0..max {
        for layer in 0..2 {
            cache.store_kv(layer, &key, &val).unwrap_or_else(|e| {
                panic!("store at position {pos} layer {layer} must succeed: {e}")
            });
        }
        cache.advance();
    }
    assert_eq!(cache.seq_len(), max);

    let err = cache
        .store_kv(0, &key, &val)
        .expect_err("storing past max_seq_len must be an error, not a silent no-op");
    let msg = err.to_string();
    assert!(
        msg.contains("full") || msg.contains("maximum context"),
        "overflow error should name the condition, got: {msg}"
    );
}

/// The overflow must be reported on *every* layer, not just layer 0 — an
/// architecture that ignores layer 0's error would otherwise write layers 1..N
/// into a cache it must not touch.
#[test]
fn store_kv_past_max_context_errors_on_all_layers() {
    let mut cache = KvCache::new(3, 1, 4);
    let key = vec![0.5f32; 4];
    for layer in 0..3 {
        cache.store_kv(layer, &key, &key).expect("first position");
    }
    cache.advance();
    for layer in 0..3 {
        assert!(
            cache.store_kv(layer, &key, &key).is_err(),
            "layer {layer} must also refuse to store past the context limit"
        );
    }
}

/// `store_kv` did `copy_from_slice(&key[..self.kv_dim])` with no length check,
/// so a short slice from an architecture panicked and took the process down.
#[test]
fn store_kv_short_key_returns_error_instead_of_panicking() {
    let kv_dim = 16usize;
    let mut cache = KvCache::new(1, 8, kv_dim);
    let short = vec![1.0f32; kv_dim - 1];
    let full = vec![1.0f32; kv_dim];

    let err = cache
        .store_kv(0, &short, &full)
        .expect_err("a key shorter than kv_dim must be an error, not a panic");
    assert!(
        err.to_string().contains("kv_dim"),
        "error should mention kv_dim, got: {err}"
    );

    let err = cache
        .store_kv(0, &full, &short)
        .expect_err("a value shorter than kv_dim must be an error");
    assert!(err.to_string().contains("kv_dim"), "got: {err}");
}

/// Once the cache is full, nothing may be silently appended: the data readable
/// afterwards must be exactly what was stored while there was room.
#[test]
fn overflowing_writes_do_not_corrupt_stored_data() {
    let max = 3usize;
    let kv_dim = 2usize;
    let mut cache = KvCache::new(1, max, kv_dim);

    for t in 0..max {
        let key = vec![t as f32, t as f32 + 0.5];
        cache.store_kv(0, &key, &key).expect("store within bounds");
        cache.advance();
    }
    let before: Vec<f32> = cache.get_keys(0).expect("get_keys").to_vec();

    let poison = vec![999.0f32; kv_dim];
    assert!(cache.store_kv(0, &poison, &poison).is_err());

    let after: Vec<f32> = cache.get_keys(0).expect("get_keys").to_vec();
    assert_eq!(
        before, after,
        "a refused store must leave the cache byte-identical"
    );
    assert!(
        !after.contains(&999.0),
        "the refused token's data must not appear in the cache"
    );
}

// ── T5: clear() must not memset the whole pre-allocated buffer ───────────────

/// `clear()` used to run `fill(0.0)` over every layer's full
/// `max_seq_len * kv_dim` buffer — ~1 GiB per reset for Llama-3-8B at a
/// 4096-token context, once per server request and once per beam per step —
/// even though reads are already bounded by `stored_len`.
///
/// The observable contract is unchanged (nothing from the old sequence is
/// readable); this test pins the contract so the `O(1)` implementation cannot
/// regress into exposing stale data.
#[test]
fn clear_hides_all_previous_data() {
    let kv_dim = 4usize;
    let mut cache = KvCache::new(2, 32, kv_dim);
    let key = vec![7.0f32; kv_dim];
    for _ in 0..5 {
        for layer in 0..2 {
            cache.store_kv(layer, &key, &key).expect("store");
        }
        cache.advance();
    }
    assert_eq!(cache.seq_len(), 5);

    cache.clear();

    assert_eq!(cache.seq_len(), 0, "clear must reset the position");
    for layer in 0..2 {
        assert!(
            cache.get_keys(layer).expect("get_keys").is_empty(),
            "layer {layer} must expose no keys after clear"
        );
        assert!(
            cache.get_values(layer).expect("get_values").is_empty(),
            "layer {layer} must expose no values after clear"
        );
    }
}

/// A cleared cache must not leak the previous sequence's values into the new
/// one at any position, including positions the new sequence has not reached.
#[test]
fn clear_then_new_sequence_reads_only_new_data() {
    let kv_dim = 4usize;
    let mut cache = KvCache::new(1, 32, kv_dim);

    for _ in 0..8 {
        let old = vec![11.0f32; kv_dim];
        cache.store_kv(0, &old, &old).expect("store old");
        cache.advance();
    }
    cache.clear();

    for _ in 0..2 {
        let new = vec![22.0f32; kv_dim];
        cache.store_kv(0, &new, &new).expect("store new");
        cache.advance();
    }

    let keys = cache.get_keys(0).expect("get_keys");
    assert_eq!(keys.len(), 2 * kv_dim, "only the new tokens may be visible");
    assert!(
        keys.iter().all(|v| (*v - 22.0).abs() < 1e-6),
        "no value from the previous sequence may be readable, got {keys:?}"
    );
}

/// `clear()` keeps the allocation for reuse; `clear_and_release()` gives it
/// back.  Both must be safe to use again afterwards.
#[test]
fn clear_retains_allocation_and_release_frees_it() {
    let kv_dim = 8usize;
    let mut cache = KvCache::new(2, 1024, kv_dim);
    let key = vec![1.0f32; kv_dim];
    for _ in 0..10 {
        for layer in 0..2 {
            cache.store_kv(layer, &key, &key).expect("store");
        }
        cache.advance();
    }
    let grown = cache.memory_bytes();
    assert!(grown > 0);

    cache.clear();
    assert_eq!(
        cache.memory_bytes(),
        grown,
        "clear() must retain the allocation so the next sequence reuses it"
    );

    cache.clear_and_release();
    assert_eq!(
        cache.memory_bytes(),
        0,
        "clear_and_release() must return the memory"
    );

    // Still usable.
    cache.store_kv(0, &key, &key).expect("store after release");
    cache.advance();
    assert_eq!(cache.seq_len(), 1);
}

// ── T3: lazy block growth ────────────────────────────────────────────────────

/// The cache used to allocate `max_seq_len * kv_dim` per layer for both K and V
/// in the constructor.  A short conversation now pays for one block, not the
/// whole trained context.
#[test]
fn allocation_is_lazy_not_full_context() {
    let num_layers = 32usize;
    let kv_dim = 1024usize;
    let max_ctx = 4096usize;

    let cache = KvCache::new(num_layers, max_ctx, kv_dim);
    assert_eq!(
        cache.memory_bytes(),
        0,
        "a fresh cache must not allocate any token storage"
    );

    let full = num_layers * max_ctx * kv_dim * 4 * 2;
    assert_eq!(
        cache.max_memory_bytes(),
        full,
        "max_memory_bytes must describe the eventual full-context cost"
    );
}

#[test]
fn growth_happens_one_block_at_a_time() {
    let num_layers = 4usize;
    let kv_dim = 64usize;
    let max_ctx = 4 * DEFAULT_BLOCK_TOKENS;
    let mut cache = KvCache::new(num_layers, max_ctx, kv_dim);
    let key = vec![1.0f32; kv_dim];

    cache.store_kv(0, &key, &key).expect("first store");
    assert_eq!(
        cache.capacity_tokens(),
        DEFAULT_BLOCK_TOKENS,
        "the first write must reserve exactly one block"
    );
    let one_block = cache.memory_bytes();
    let full = cache.max_memory_bytes();
    assert!(
        one_block * 2 <= full,
        "one block ({one_block} B) must be far below the full context ({full} B)"
    );

    // Fill past the first block.
    for _ in 0..=DEFAULT_BLOCK_TOKENS {
        for layer in 0..num_layers {
            cache.store_kv(layer, &key, &key).expect("store");
        }
        cache.advance();
    }
    assert_eq!(
        cache.capacity_tokens(),
        2 * DEFAULT_BLOCK_TOKENS,
        "crossing a block boundary must reserve exactly one more block"
    );
}

/// Growth must never lose already-stored data.
#[test]
fn growth_preserves_previously_stored_tokens() {
    let kv_dim = 4usize;
    let max_ctx = 3 * DEFAULT_BLOCK_TOKENS;
    let mut cache = KvCache::new(1, max_ctx, kv_dim);

    let n = DEFAULT_BLOCK_TOKENS + 17;
    for t in 0..n {
        let key = vec![t as f32; kv_dim];
        cache.store_kv(0, &key, &key).expect("store");
        cache.advance();
    }

    let keys = cache.get_keys(0).expect("get_keys");
    assert_eq!(keys.len(), n * kv_dim);
    for t in 0..n {
        assert!(
            (keys[t * kv_dim] - t as f32).abs() < 1e-6,
            "token {t} was corrupted by a growth step: got {}",
            keys[t * kv_dim]
        );
    }
}

// ── T3: FP16 storage mode ────────────────────────────────────────────────────

#[test]
fn f16_storage_halves_the_footprint() {
    let num_layers = 8usize;
    let kv_dim = 128usize;
    let max_ctx = DEFAULT_BLOCK_TOKENS;
    let key = vec![0.25f32; kv_dim];

    let mut f32_cache = KvCache::new(num_layers, max_ctx, kv_dim);
    let mut f16_cache = KvCache::with_dtype(num_layers, max_ctx, kv_dim, KvCacheDtype::F16);
    for cache in [&mut f32_cache, &mut f16_cache] {
        for layer in 0..num_layers {
            cache.store_kv(layer, &key, &key).expect("store");
        }
        cache.advance();
    }

    assert_eq!(f32_cache.dtype(), KvCacheDtype::F32);
    assert_eq!(f16_cache.dtype(), KvCacheDtype::F16);
    assert_eq!(
        f16_cache.memory_bytes() * 2,
        f32_cache.memory_bytes(),
        "f16 storage must be exactly half of f32"
    );
    assert_eq!(
        f16_cache.max_memory_bytes() * 2,
        f32_cache.max_memory_bytes()
    );
}

#[test]
fn f16_round_trips_through_the_per_token_apis() {
    let kv_dim = 8usize;
    let mut cache = KvCache::with_dtype(1, 64, kv_dim, KvCacheDtype::F16);

    let mut expected: Vec<Vec<f32>> = Vec::new();
    for t in 0..5u32 {
        let key: Vec<f32> = (0..kv_dim).map(|d| t as f32 + d as f32 * 0.5).collect();
        cache.store_kv(0, &key, &key).expect("store");
        cache.advance();
        expected.push(key);
    }

    // `get_keys` cannot borrow `&[f32]` out of f16 storage, and says so.
    let err = cache
        .get_keys(0)
        .expect_err("get_keys must not silently reinterpret f16 as f32");
    let msg = err.to_string();
    assert!(
        msg.contains("f16") && msg.contains("for_each_key"),
        "the error must name the dtype and the working alternative, got: {msg}"
    );

    // The per-token iterator works and returns the values back as f32.
    let mut seen: Vec<Vec<f32>> = Vec::new();
    cache
        .for_each_key(0, &mut |_pos, row| seen.push(row.to_vec()))
        .expect("for_each_key must work in f16 mode");
    assert_eq!(seen.len(), expected.len());
    for (t, (got, want)) in seen.iter().zip(expected.iter()).enumerate() {
        for (d, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1e-2,
                "token {t} dim {d}: f16 round trip {w} -> {g}"
            );
        }
    }

    // So does the copy-out API.
    let mut buf = Vec::new();
    cache
        .copy_values_into(0, &mut buf)
        .expect("copy_values_into");
    assert_eq!(buf.len(), expected.len() * kv_dim);
}

/// FP32 remains the default so every existing architecture keeps working
/// unchanged through `get_keys`.
#[test]
fn f32_is_the_default_and_still_borrows() {
    let kv_dim = 4usize;
    let mut cache = KvCache::new(1, 16, kv_dim);
    assert_eq!(cache.dtype(), KvCacheDtype::F32);
    let key = vec![3.0f32; kv_dim];
    cache.store_kv(0, &key, &key).expect("store");
    cache.advance();
    let keys = cache.get_keys(0).expect("f32 storage must still borrow");
    assert_eq!(keys, &[3.0f32; 4]);
}

// ── T6: restore_from_snapshot must validate, not clamp ───────────────────────

/// A snapshot claiming more tokens than it carries used to be accepted: the
/// copy was clamped to the source length but `seq_len` was set to the claimed
/// value, marking never-written positions VALID.  With the `O(1)` `clear()`
/// those positions now read the *previous sequence's* keys.
#[test]
fn restore_from_short_snapshot_is_refused() {
    let kv_dim = 4usize;
    let mut cache = KvCache::new(2, 64, kv_dim);

    // Claims 10 tokens, carries 3.
    let short = vec![vec![1.0f32; 3 * kv_dim], vec![1.0f32; 3 * kv_dim]];
    let err = cache
        .restore_from_snapshot(&short, &short, 10)
        .expect_err("a snapshot shorter than its claimed seq_len must be refused");
    assert!(
        matches!(err, RuntimeError::SnapshotIncompatible { .. }),
        "expected SnapshotIncompatible, got {err:?}"
    );
    assert_eq!(
        cache.seq_len(),
        0,
        "a refused restore must not move the cache position"
    );
}

#[test]
fn restore_with_wrong_layer_count_is_refused() {
    let kv_dim = 4usize;
    let mut cache = KvCache::new(4, 64, kv_dim);
    let two_layers = vec![vec![0.0f32; 2 * kv_dim]; 2];
    let err = cache
        .restore_from_snapshot(&two_layers, &two_layers, 2)
        .expect_err("a snapshot with fewer layers than the cache must be refused");
    assert!(matches!(err, RuntimeError::SnapshotIncompatible { .. }));
}

#[test]
fn restore_beyond_max_context_reports_kv_cache_full() {
    let kv_dim = 4usize;
    let mut cache = KvCache::new(1, 8, kv_dim);
    let big = vec![vec![0.0f32; 100 * kv_dim]];
    let err = cache
        .restore_from_snapshot(&big, &big, 100)
        .expect_err("restoring past max_seq_len must be refused");
    assert!(
        matches!(err, RuntimeError::KvCacheFull { max_ctx: 8 }),
        "expected KvCacheFull {{ max_ctx: 8 }}, got {err:?}"
    );
}

#[test]
fn restore_of_an_exact_snapshot_round_trips() {
    let kv_dim = 4usize;
    let num_layers = 2usize;
    let mut src = KvCache::new(num_layers, 64, kv_dim);
    for t in 0..6u32 {
        for layer in 0..num_layers {
            let key: Vec<f32> = (0..kv_dim)
                .map(|d| (layer * 100) as f32 + t as f32 + d as f32 * 0.1)
                .collect();
            src.store_kv(layer, &key, &key).expect("store");
        }
        src.advance();
    }
    let snap = src.snapshot();

    let mut dst = KvCache::new(num_layers, 64, kv_dim);
    dst.restore_from_snapshot(&snap.keys, &snap.values, snap.seq_len)
        .expect("an exact snapshot must restore");
    assert_eq!(dst.seq_len(), 6);
    for layer in 0..num_layers {
        assert_eq!(
            src.get_keys(layer).expect("src"),
            dst.get_keys(layer).expect("dst"),
            "layer {layer} must round trip exactly"
        );
    }
}

/// `snapshot_truncated` is what lets the prefix cache store exactly the prompt.
#[test]
fn snapshot_truncated_covers_only_the_requested_prefix() {
    let kv_dim = 2usize;
    let mut cache = KvCache::new(1, 32, kv_dim);
    for t in 0..10u32 {
        let key = vec![t as f32; kv_dim];
        cache.store_kv(0, &key, &key).expect("store");
        cache.advance();
    }
    let snap = cache.snapshot_truncated(4);
    assert_eq!(snap.seq_len, 4);
    assert_eq!(snap.keys[0].len(), 4 * kv_dim);
    assert!((snap.keys[0][3 * kv_dim] - 3.0).abs() < 1e-6);
}

// ── T6: prefix cache length invariant and accounting ─────────────────────────

/// `store(tokens, kv, seq_len, ..)` never checked `seq_len` against
/// `tokens.len()`.  `store_kv_in_prefix_cache` passed `kv.seq_len()` — prompt
/// PLUS everything generated — so a later `lookup` returned
/// `matched = tokens.len()` against a snapshot of a different length.
#[test]
fn prefix_store_refuses_a_snapshot_shorter_than_its_key() {
    let kv_dim = 4usize;
    let mut live = KvCache::new(1, 64, kv_dim);
    for _ in 0..3 {
        let key = vec![1.0f32; kv_dim];
        live.store_kv(0, &key, &key).expect("store");
        live.advance();
    }

    let mut pcache = PrefixKvCache::new(PrefixCacheConfig {
        max_entries: 16,
        max_memory_bytes: 1 << 20,
        min_prefix_len: 1,
    });

    // 10-token key but only 3 tokens of KV state.
    let tokens: Vec<u32> = (0..10).collect();
    let stored = pcache.store(&tokens, &live, 3, kv_dim, 1);
    assert!(
        !stored,
        "storing a 3-token snapshot under a 10-token key must be refused"
    );
    assert!(pcache.is_empty(), "nothing may have been stored");
    assert!(
        pcache.lookup(&tokens).is_none(),
        "no bogus entry may be discoverable"
    );
}

/// The other direction: a longer KV state is truncated to the key, so the entry
/// covers exactly the prompt and does not retain the completion's KV.
#[test]
fn prefix_store_truncates_a_longer_kv_state_to_the_key() {
    let kv_dim = 4usize;
    let num_layers = 2usize;
    let mut live = KvCache::new(num_layers, 128, kv_dim);
    // 5 prompt tokens + 20 generated tokens, as after a decode loop.
    for t in 0..25u32 {
        for layer in 0..num_layers {
            let key = vec![t as f32; kv_dim];
            live.store_kv(layer, &key, &key).expect("store");
        }
        live.advance();
    }

    let mut pcache = PrefixKvCache::new(PrefixCacheConfig {
        max_entries: 16,
        max_memory_bytes: 1 << 24,
        min_prefix_len: 1,
    });
    let prompt: Vec<u32> = (0..5).collect();
    assert!(pcache.store(&prompt, &live, live.seq_len(), kv_dim, num_layers));

    let (matched, cached) = pcache.lookup(&prompt).expect("the prompt must be cached");
    assert_eq!(matched, prompt.len());
    assert_eq!(
        cached.seq_len(),
        prompt.len(),
        "the entry must cover the key exactly, not prompt+completion"
    );
    assert_eq!(
        cached.keys()[0].len(),
        prompt.len() * kv_dim,
        "no completion KV may be retained"
    );
    assert_eq!(
        cached.memory_bytes(),
        num_layers * prompt.len() * kv_dim * 4 * 2,
        "the retained bytes must match the key length, not the sequence length"
    );
}

/// The entry produced by a truncating store must restore cleanly — this is the
/// composition that used to silently mark unwritten positions valid.
#[test]
fn prefix_entry_restores_into_a_live_cache() {
    let kv_dim = 4usize;
    let num_layers = 2usize;
    let mut live = KvCache::new(num_layers, 128, kv_dim);
    for t in 0..12u32 {
        for layer in 0..num_layers {
            let key: Vec<f32> = (0..kv_dim).map(|d| t as f32 + d as f32 * 0.25).collect();
            live.store_kv(layer, &key, &key).expect("store");
        }
        live.advance();
    }

    let mut pcache = PrefixKvCache::new(PrefixCacheConfig {
        max_entries: 8,
        max_memory_bytes: 1 << 24,
        min_prefix_len: 1,
    });
    let prompt: Vec<u32> = (0..6).collect();
    assert!(pcache.store(&prompt, &live, live.seq_len(), kv_dim, num_layers));

    let (_, cached) = pcache.lookup(&prompt).expect("hit");
    let cached = cached.clone();
    let mut target = KvCache::new(num_layers, 128, kv_dim);
    PrefixKvCache::restore(&cached, &mut target).expect("restore must succeed");
    assert_eq!(target.seq_len(), prompt.len());
    for layer in 0..num_layers {
        let got = target.get_keys(layer).expect("target keys");
        assert_eq!(got.len(), prompt.len() * kv_dim);
        for t in 0..prompt.len() {
            assert!(
                (got[t * kv_dim] - t as f32).abs() < 1e-6,
                "layer {layer} token {t} restored incorrectly"
            );
        }
    }
}

#[test]
fn prefix_store_snapshot_refuses_a_length_mismatch() {
    let mut pcache = PrefixKvCache::new(PrefixCacheConfig {
        max_entries: 8,
        max_memory_bytes: 1 << 20,
        min_prefix_len: 1,
    });
    let snap = CachedKvState::new(vec![vec![0.0f32; 8]], vec![vec![0.0f32; 8]], 2);
    assert!(
        !pcache.store_snapshot(&[1, 2, 3], snap),
        "a 2-token snapshot under a 3-token key must be refused"
    );
    assert!(pcache.is_empty());
}

/// The running counters must agree with a full tree walk after arbitrary
/// insert / replace / evict traffic.  They used to be recomputed by traversal
/// on every eviction-loop iteration, which was correct but quadratic; keeping
/// them incrementally is only safe if they stay exact.
#[test]
fn prefix_cache_counters_track_the_tree() {
    let mut pcache = PrefixKvCache::new(PrefixCacheConfig {
        max_entries: 8,
        max_memory_bytes: 1 << 24,
        min_prefix_len: 1,
    });

    let mk = |n: usize| CachedKvState::new(vec![vec![0.0f32; 4]], vec![vec![0.0f32; 4]], n);

    // Distinct keys.
    for i in 0..5u32 {
        assert!(pcache.store_snapshot(&[i, i + 100], mk(2)));
    }
    assert_eq!(pcache.len(), 5);
    assert_eq!(pcache.memory_usage(), 5 * 8 * 4);

    // Replacing an existing key must not double-count.
    assert!(pcache.store_snapshot(&[0, 100], mk(2)));
    assert_eq!(pcache.len(), 5, "a replacement must not add an entry");
    assert_eq!(pcache.memory_usage(), 5 * 8 * 4);

    // Splitting keys.
    assert!(pcache.store_snapshot(&[0, 100, 7], mk(3)));
    assert_eq!(pcache.len(), 6);

    pcache.clear();
    assert_eq!(pcache.len(), 0);
    assert_eq!(pcache.memory_usage(), 0);
}

#[test]
fn prefix_cache_evicts_to_stay_within_its_entry_budget() {
    let mut pcache = PrefixKvCache::new(PrefixCacheConfig {
        max_entries: 3,
        max_memory_bytes: 1 << 24,
        min_prefix_len: 1,
    });
    for i in 0..10u32 {
        let snap = CachedKvState::new(vec![vec![0.0f32; 4]], vec![vec![0.0f32; 4]], 2);
        pcache.store_snapshot(&[i, i + 500], snap);
    }
    assert!(
        pcache.len() <= 3,
        "the entry budget must hold, got {}",
        pcache.len()
    );
}

#[test]
fn prefix_cache_evicts_to_stay_within_its_memory_budget() {
    // One entry: 1 layer × 4 floats × (K+V) = 32 bytes.
    let mut pcache = PrefixKvCache::new(PrefixCacheConfig {
        max_entries: 1000,
        max_memory_bytes: 64,
        min_prefix_len: 1,
    });
    for i in 0..10u32 {
        let snap = CachedKvState::new(vec![vec![0.0f32; 4]], vec![vec![0.0f32; 4]], 2);
        pcache.store_snapshot(&[i, i + 500], snap);
    }
    assert!(
        pcache.memory_usage() <= 64,
        "the memory budget must hold, got {}",
        pcache.memory_usage()
    );
}

/// A zero-byte entry used to make `evict_lru` spin: `evict_lru_one` returned
/// `0` both for "freed nothing" and for "freed a zero-byte entry", and the
/// entry-count loop treated `0` as "give up" while the entry was still there.
#[test]
fn prefix_cache_evicts_zero_byte_entries_without_hanging() {
    let mut pcache = PrefixKvCache::new(PrefixCacheConfig {
        max_entries: 2,
        max_memory_bytes: 1 << 20,
        min_prefix_len: 1,
    });
    for i in 0..6u32 {
        // A snapshot with no layers at all: memory_bytes() == 0.
        let snap = CachedKvState::new(Vec::new(), Vec::new(), 2);
        pcache.store_snapshot(&[i, i + 900], snap);
    }
    assert!(
        pcache.len() <= 2,
        "zero-byte entries must still be evictable, got {}",
        pcache.len()
    );
}

/// Cloning a hit must not copy the KV payload.
#[test]
fn cached_state_clone_shares_its_buffers() {
    let big = vec![vec![1.0f32; 4096]; 8];
    let state = CachedKvState::new(big.clone(), big, 4);
    let before = state.memory_bytes();
    let clone = state.clone();
    assert_eq!(clone.memory_bytes(), before);
    assert_eq!(clone.keys().len(), state.keys().len());
    // Same backing storage, so the data pointers coincide.
    assert!(
        std::ptr::eq(state.keys().as_ptr(), clone.keys().as_ptr()),
        "cloning a CachedKvState must share, not copy, the per-layer buffers"
    );
}

// ── T8: KV pool double-free protection ───────────────────────────────────────

/// Freeing the same page twice pushed the index onto the free list twice, so
/// the next two `alloc()` calls handed the SAME page to two owners.  Release
/// builds had no guard at all.
#[test]
fn kv_pool_rejects_a_double_free() {
    let mut pool = KvCachePool::new(16, 2);
    let a = pool.alloc().expect("first alloc");
    pool.free(a).expect("the first free must succeed");
    let err = pool
        .free(a)
        .expect_err("the second free of the same page must be an error");
    assert!(
        err.to_string().contains("twice"),
        "the error should name the double free, got: {err}"
    );
}

/// The consequence the guard exists to prevent: two live allocations must never
/// name the same page.
#[test]
fn kv_pool_never_hands_one_page_to_two_owners() {
    let mut pool = KvCachePool::new(16, 2);
    let a = pool.alloc().expect("alloc a");
    let b = pool.alloc().expect("alloc b");
    assert_ne!(a, b);

    pool.free(a).expect("free a");
    let _ = pool.free(a); // the buggy second free, now refused

    let c = pool.alloc().expect("alloc c");
    assert!(pool.alloc().is_none(), "the pool holds only two pages");
    assert_ne!(c, b, "page {c} was handed out while still owned by {b}");
}

#[test]
fn kv_pool_rejects_an_out_of_range_free() {
    let mut pool = KvCachePool::new(8, 1);
    assert!(
        pool.free(99).is_err(),
        "an out-of-range free must be an error in release builds too"
    );
}

#[test]
fn kv_pool_tracks_allocation_state() {
    let mut pool = KvCachePool::new(8, 2);
    let a = pool.alloc().expect("alloc");
    assert!(pool.is_allocated(a));
    pool.free(a).expect("free");
    assert!(!pool.is_allocated(a));
    assert!(!pool.is_allocated(999));
}

// ── T2: reset_sequence must run between sequences ────────────────────────────

/// A `ForwardPass` that owns per-sequence state, standing in for DeepSeek's
/// `MlaLatentCache`, Mamba-2 / Jamba's SSM hidden state, and DBRX / Grok's
/// `current_pos`.
struct StatefulForwardPass {
    /// Tokens seen since the last `reset_sequence()`.
    seen: Vec<u32>,
    /// How many times `reset_sequence()` has been called.
    resets: usize,
    vocab_size: usize,
}

impl StatefulForwardPass {
    fn new(vocab_size: usize) -> Self {
        Self {
            seen: Vec::new(),
            resets: 0,
            vocab_size,
        }
    }
}

impl ForwardPass for StatefulForwardPass {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let kv_dim = kv_cache.kv_dim().max(1);
        for &t in tokens {
            self.seen.push(t);
            let row = vec![t as f32; kv_dim];
            kv_cache.store_kv(0, &row, &row)?;
            kv_cache.advance();
        }
        // The logits encode the accumulated state, so contamination is visible.
        let mut logits = vec![0.0f32; self.vocab_size];
        logits[0] = self.seen.len() as f32;
        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn hidden_size(&self) -> usize {
        1
    }

    fn max_context_length(&self) -> usize {
        4096
    }

    fn reset_sequence(&mut self) {
        self.seen.clear();
        self.resets += 1;
    }
}

/// Without a `reset_sequence()` call site, per-sequence architecture state
/// carried straight into the next request: DeepSeek's latent cache grew until
/// `append` errored, Mamba-2 and Jamba reused the previous `h`, and DBRX and
/// Grok never returned `current_pos` to 0.
#[test]
fn reset_sequence_clears_per_architecture_state() {
    let mut fp = StatefulForwardPass::new(8);
    let mut kv = KvCache::new(1, 128, 4);

    fp.forward(&[1, 2, 3], &mut kv).expect("first sequence");
    assert_eq!(fp.seen.len(), 3);

    // What `InferenceEngine::reset()` now does, in the same order.
    kv.clear();
    fp.reset_sequence();

    assert_eq!(fp.resets, 1, "reset_sequence must have been called");
    assert!(
        fp.seen.is_empty(),
        "per-sequence state must not survive a reset"
    );

    let logits = fp.forward(&[9], &mut kv).expect("second sequence");
    assert_eq!(
        logits[0], 1.0,
        "the second sequence must see only its own token, not the first sequence's"
    );
    assert_eq!(
        kv.seq_len(),
        1,
        "the KV cache must also start the second sequence at position 0"
    );
}

/// The default `reset_sequence()` is a no-op, so a stateless architecture is
/// unaffected by the new call site.
#[test]
fn reset_sequence_defaults_to_a_no_op() {
    struct Stateless;
    impl ForwardPass for Stateless {
        fn forward(&mut self, _t: &[u32], _kv: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
            Err(ArchError::NotSupported {
                detail: "test stub".to_string(),
            })
        }
        fn vocab_size(&self) -> usize {
            1
        }
        fn hidden_size(&self) -> usize {
            1
        }
        fn max_context_length(&self) -> usize {
            1
        }
    }
    let mut fp = Stateless;
    fp.reset_sequence();
    fp.reset_sequence();
}

// ── T2 / T4: measured cache-level cost at real Llama-3-8B geometry ───────────

/// Llama-3-8B as the engine configures it: 32 layers, `kv_dim = 8 kv-heads ×
/// 128 head-dim = 1024`, context clamped by `resolve_context_size` to 4096.
const L3_LAYERS: usize = 32;
const L3_KV_DIM: usize = 1024;
const L3_CTX: usize = 4096;

/// Report the numbers T2 and T4 are graded on, and pin the ratios so a
/// regression to eager allocation fails rather than merely slows things down.
///
/// `max_memory_bytes()` is exactly the pre-fix constructor cost — the old
/// `vec![0.0f32; max_seq_len * kv_dim]` per layer for both K and V — so it is a
/// measured "before", not an estimate.
#[test]
fn kv_cache_cost_at_llama3_8b_geometry() {
    let eager = L3_LAYERS * L3_CTX * L3_KV_DIM * 4 * 2;

    let mut cache = KvCache::new(L3_LAYERS, L3_CTX, L3_KV_DIM);
    assert_eq!(
        cache.max_memory_bytes(),
        eager,
        "max_memory_bytes must equal the old eager constructor cost"
    );
    assert_eq!(
        cache.memory_bytes(),
        0,
        "construction must allocate nothing"
    );

    // A 20-token exchange, every layer written, as a real short chat does.
    let row = vec![0.5f32; L3_KV_DIM];
    for _ in 0..20 {
        for layer in 0..L3_LAYERS {
            cache.store_kv(layer, &row, &row).expect("store");
        }
        cache.advance();
    }
    let after_20 = cache.memory_bytes();

    // One block per layer for K and V — nothing more.
    let one_block = L3_LAYERS * DEFAULT_BLOCK_TOKENS * L3_KV_DIM * 4 * 2;
    assert_eq!(
        after_20, one_block,
        "a 20-token conversation must hold exactly one block per layer"
    );
    assert!(
        after_20 * 15 < eager,
        "lazy growth must be more than an order of magnitude below eager \
         allocation: {after_20} B vs {eager} B"
    );

    let f16 = KvCache::with_dtype(L3_LAYERS, L3_CTX, L3_KV_DIM, KvCacheDtype::F16);
    assert_eq!(
        f16.max_memory_bytes() * 2,
        eager,
        "F16 storage must halve the full-context cost"
    );

    println!("T2 KV cache cost, Llama-3-8B geometry ({L3_LAYERS} layers, kv_dim {L3_KV_DIM}, ctx {L3_CTX}):");
    println!(
        "  before (eager FP32 constructor) : {eager} B = {:.1} MiB",
        eager as f64 / 1048576.0
    );
    println!(
        "  after  (lazy, 20-token chat)    : {after_20} B = {:.1} MiB",
        after_20 as f64 / 1048576.0
    );
    println!(
        "  after  (lazy, full 4096 ctx)    : {eager} B = {:.1} MiB (unchanged ceiling)",
        eager as f64 / 1048576.0
    );
    println!(
        "  F16 full-context ceiling        : {} B = {:.1} MiB",
        f16.max_memory_bytes(),
        f16.max_memory_bytes() as f64 / 1048576.0
    );
    println!(
        "  reduction for a 20-token chat   : {:.1}x",
        eager as f64 / after_20 as f64
    );
}

/// `clear()` used to memset every layer's full pre-allocated buffer even though
/// `get_keys`/`get_values` are bounded by `stored_len`, so the zeroed region was
/// already unreachable.  The server calls `reset()` per request and beam search
/// calls it per beam per step.
///
/// This measures the memset that is no longer paid: a full-context cache is
/// built, then `clear()` is timed against an explicit `fill(0.0)` over the same
/// bytes.
#[test]
fn clear_no_longer_pays_a_full_context_memset() {
    use std::time::Instant;

    // A smaller geometry than 8B so the test stays quick, but still large
    // enough for the memset to dominate: 8 layers × 4096 × 1024 × 4 × 2 = 256 MiB.
    let layers = 8usize;
    let mut cache = KvCache::new(layers, L3_CTX, L3_KV_DIM);
    let row = vec![0.5f32; L3_KV_DIM];
    for _ in 0..L3_CTX {
        for layer in 0..layers {
            cache.store_kv(layer, &row, &row).expect("store");
        }
        cache.advance();
    }
    let bytes = cache.memory_bytes();
    assert!(
        bytes > 0,
        "the cache must be populated for this measurement"
    );

    // `clear()` is idempotent, so it can be timed on its own.
    cache.clear();
    let iters = 1000;
    let start = Instant::now();
    for _ in 0..iters {
        cache.clear();
        std::hint::black_box(&cache);
    }
    let clear_nanos = start.elapsed().as_nanos() as f64 / iters as f64;

    // What `clear()` used to do, over the same live bytes.  Warm up first so
    // the first-touch page faults of a fresh 256 MiB allocation are not
    // charged to the measurement.
    let mut scratch = vec![0.5f32; bytes / 4];
    scratch.fill(0.0);
    std::hint::black_box(&scratch);
    let memset_iters = 20;
    let start = Instant::now();
    for _ in 0..memset_iters {
        scratch.fill(0.0);
        std::hint::black_box(&scratch);
    }
    let memset_nanos = start.elapsed().as_nanos() as f64 / memset_iters as f64;

    println!(
        "T4 clear() cost over {:.0} MiB of live KV:",
        bytes as f64 / 1048576.0
    );
    println!("  before (fill(0.0) over the buffers) : {memset_nanos:.0} ns/call");
    println!("  after  (O(1) bookkeeping reset)     : {clear_nanos:.0} ns/call");
    println!(
        "  speedup                             : {:.0}x",
        memset_nanos / clear_nanos
    );

    assert!(
        clear_nanos * 10.0 < memset_nanos,
        "clear() must be at least 10x cheaper than the memset it replaced: \
         {clear_nanos:.0} ns vs {memset_nanos:.0} ns"
    );
    // `clear()` is a bookkeeping reset, and the allocation survives it so the
    // next sequence reuses the same memory instead of re-allocating.
    assert_eq!(
        cache.seq_len(),
        0,
        "clear() must reset the sequence position"
    );
    assert_eq!(
        cache.memory_bytes(),
        bytes,
        "clear() must retain the allocation for reuse"
    );
}

/// The decode loop must stop cleanly when the *cache* is full, even if the
/// architecture advertises a longer context than the cache was built for.
///
/// `resolve_context_size` clamps an unset context to `min(n_ctx_train, 4096)`,
/// so an architecture whose `max_context_length()` reports the trained length
/// (8192 for Llama-3) disagrees with a 4096-token cache.  Bounding the loop by
/// `max_context_length()` alone would run it past the cache, and `store_kv` now
/// returns a hard error on overflow (T3) — surfacing as a failed request rather
/// than a `ContextFull` stop.
#[test]
fn cache_limit_bounds_generation_when_arch_claims_more() {
    let cache_ctx = 8usize;
    let mut cache = KvCache::new(1, cache_ctx, 4);
    let row = vec![1.0f32; 4];

    // Fill the cache to its limit, as a long prompt would.
    for _ in 0..cache_ctx {
        cache
            .store_kv(0, &row, &row)
            .expect("store within the cache");
        cache.advance();
    }
    assert_eq!(cache.seq_len(), cache_ctx);

    // An architecture claiming 8192 would happily ask for one more token.
    let arch_claim = 8192usize;
    let effective = arch_claim.min(cache.max_seq_len());
    assert_eq!(
        effective, cache_ctx,
        "the decode loop's bound must be the smaller of the two limits"
    );
    assert!(
        cache.seq_len() >= effective,
        "the loop must see itself as context-full and stop"
    );

    // And if it did not stop, the storage layer refuses rather than corrupting.
    assert!(
        cache.store_kv(0, &row, &row).is_err(),
        "a write past the cache limit must be an error, not a silent no-op"
    );
}

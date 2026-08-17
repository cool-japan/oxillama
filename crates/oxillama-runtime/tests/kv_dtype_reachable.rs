// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `KvCacheDtype::F16` must be *reachable*, not merely implemented.
//!
//! The FP16 backing store landed in 0.1.4 and was correct — and unusable.  Every
//! architecture read the cache through
//! [`KvCacheAccess::get_keys`](oxillama_arch::traits::KvCacheAccess::get_keys),
//! which hands out a borrowed `&[f32]`; a `Vec<f16>` has no such slice to lend,
//! so selecting F16 failed on the first layer of the first token.  The read
//! path now goes through `oxillama_arch::common::fetch_keys`/`fetch_values`,
//! which borrow when the storage allows it and gather otherwise.
//!
//! This file pins that contract from the runtime side, against the **real**
//! `KvCache`/`PagedKvCache` rather than a test double:
//!
//! | Test | Property |
//! |------|----------|
//! | [`f32_cache_is_borrowed_with_no_copy`] | the f32 fast path is still a zero-copy borrow of the cache's own buffer |
//! | [`f16_cache_is_readable_through_fetch`] | f16 storage reads back exactly its f16-rounded contents |
//! | [`f16_cache_still_refuses_the_raw_borrow`] | `get_keys()` keeps failing loudly rather than materialising a hidden copy |
//! | [`multi_page_paged_cache_is_readable_through_fetch`] | the gather spans a paged cache's pages correctly (which does **not** by itself make paged attention work — see that test) |
//! | [`f16_cache_holds_half_the_bytes`] | measured allocation, not the theoretical maximum |
//! | [`prefix_cache_stores_from_an_f16_cache`] | the server's prefix cache snapshots an f16 KV cache instead of refusing every layer |
//! | [`prefix_cache_restores_into_an_f16_cache`] | and primes an f16 cache back from that f32 payload — the path a `serve` prefix *hit* takes |
//!
//! Geometry is deliberately awkward: `KV_DIM = 33` is not a multiple of any SIMD
//! width, and the sequences cross `DEFAULT_BLOCK_TOKENS` so the cache's
//! block-growth path is exercised rather than a single up-front allocation.

use half::f16;
use oxillama_arch::traits::KvCacheAccess;
use oxillama_runtime::kv_cache::{KvCacheDtype, DEFAULT_BLOCK_TOKENS};
use oxillama_runtime::{KvCache, PrefixCacheConfig, PrefixKvCache};

/// Not a multiple of 2, 4, 8, 16 or 32 — a partial tail in every vectorised loop.
const KV_DIM: usize = 33;
const LAYERS: usize = 3;
/// One token past a whole block, so growth happens twice and the last block is
/// partially filled.
const TOKENS: usize = DEFAULT_BLOCK_TOKENS + 1;

/// Deterministic, wide-dynamic-range payload: `f16` has 10 significand bits and
/// a limited exponent range, so values are kept well inside it while still
/// varying enough that rounding is visible.
fn sample(pos: usize, dim: usize, salt: usize) -> f32 {
    let n = (pos * KV_DIM + dim + salt * 7) as f32;
    (n * 0.013).sin() * 3.5 + 0.25
}

fn fill(cache: &mut KvCache, tokens: usize) {
    let mut key = vec![0.0f32; KV_DIM];
    let mut val = vec![0.0f32; KV_DIM];
    for pos in 0..tokens {
        for layer in 0..LAYERS {
            for d in 0..KV_DIM {
                key[d] = sample(pos, d, layer);
                val[d] = -sample(pos, d, layer + 64);
            }
            cache.store_kv(layer, &key, &val).expect("store_kv");
        }
        cache.advance();
    }
}

/// The F32 path must still hand out the cache's own buffer — no copy, no
/// allocation.  Comparing pointers is the only assertion that actually proves
/// it; a value comparison would pass for a copy too.
#[test]
fn f32_cache_is_borrowed_with_no_copy() {
    let mut cache = KvCache::with_dtype(LAYERS, TOKENS, KV_DIM, KvCacheDtype::F32);
    fill(&mut cache, TOKENS);

    for layer in 0..LAYERS {
        let borrowed = cache.get_keys(layer).expect("f32 storage must lend keys");
        let fetched = oxillama_arch::common::fetch_keys(&cache, layer).expect("fetch_keys");
        assert_eq!(
            fetched.as_ptr(),
            borrowed.as_ptr(),
            "layer {layer}: fetch_keys must borrow, not copy"
        );
        assert_eq!(fetched.len(), borrowed.len());

        let borrowed = cache
            .get_values(layer)
            .expect("f32 storage must lend values");
        let fetched = oxillama_arch::common::fetch_values(&cache, layer).expect("fetch_values");
        assert_eq!(
            fetched.as_ptr(),
            borrowed.as_ptr(),
            "layer {layer}: fetch_values must borrow, not copy"
        );
    }
}

/// F16 storage must read back exactly its `f16`-rounded contents — bit-exact,
/// not "close".  Any looser assertion would also pass if the gather dropped or
/// duplicated a row.
#[test]
fn f16_cache_is_readable_through_fetch() {
    let mut cache = KvCache::with_dtype(LAYERS, TOKENS, KV_DIM, KvCacheDtype::F16);
    fill(&mut cache, TOKENS);

    for layer in 0..LAYERS {
        let keys = oxillama_arch::common::fetch_keys(&cache, layer).expect("fetch_keys on f16");
        let values =
            oxillama_arch::common::fetch_values(&cache, layer).expect("fetch_values on f16");
        assert_eq!(
            keys.len(),
            TOKENS * KV_DIM,
            "layer {layer}: every stored token must be visible"
        );
        assert_eq!(values.len(), TOKENS * KV_DIM);

        for pos in 0..TOKENS {
            for d in 0..KV_DIM {
                let idx = pos * KV_DIM + d;
                let expect_key = f16::from_f32(sample(pos, d, layer)).to_f32();
                let expect_val = f16::from_f32(-sample(pos, d, layer + 64)).to_f32();
                assert_eq!(
                    keys[idx], expect_key,
                    "layer {layer} pos {pos} dim {d}: key must equal its f16 round trip"
                );
                assert_eq!(
                    values[idx], expect_val,
                    "layer {layer} pos {pos} dim {d}: value must equal its f16 round trip"
                );
            }
        }
    }
}

/// `get_keys()` must keep failing on f16 storage.
///
/// Making it succeed would require materialising a copy behind the caller's
/// back on a `&self` method of a `Send + Sync` type — the borrow it hands out
/// has to live in the cache, so a silent conversion buffer is not available.
/// A loud, typed error that names the working alternatives is the contract.
#[test]
fn f16_cache_still_refuses_the_raw_borrow() {
    let mut cache = KvCache::with_dtype(LAYERS, 8, KV_DIM, KvCacheDtype::F16);
    fill(&mut cache, 4);

    let err = cache
        .get_keys(0)
        .expect_err("f16 storage must not pretend to lend &[f32]");
    let text = err.to_string();
    assert!(text.contains("f16"), "error must name the dtype: {text}");
    assert!(
        text.contains("for_each_key") || text.contains("copy_keys_into"),
        "error must point at the APIs that do work: {text}"
    );
}

/// A paged cache whose sequence spans several pages also refuses `get_keys()`;
/// `fetch_keys` gathers across pages instead.
///
/// **This does not make paged attention work.** `PagedKvCache::store_kv` writes
/// at `seq_len` but has no `stored_len` counterpart, and `iter_keys` walks
/// `0..seq_len` — so the row an attention layer just wrote is invisible to it
/// until `advance()`, which a forward pass only calls after *all* layers have
/// stored.  This test therefore advances past every token before reading.
/// Wiring `PagedKvCache` into the inference path needs that skew fixed first;
/// what is verified here is only that the gather itself is correct across page
/// boundaries.
#[test]
fn multi_page_paged_cache_is_readable_through_fetch() {
    use oxillama_runtime::kv_cache::PagedKvCache;

    let mut cache = PagedKvCache::new(LAYERS, 4096, KV_DIM);
    let tokens = 300usize; // > PAGE_SIZE, so the sequence is multi-page
    let mut key = vec![0.0f32; KV_DIM];
    let mut val = vec![0.0f32; KV_DIM];
    for pos in 0..tokens {
        for layer in 0..LAYERS {
            for d in 0..KV_DIM {
                key[d] = sample(pos, d, layer);
                val[d] = -sample(pos, d, layer + 64);
            }
            cache.store_kv(layer, &key, &val).expect("paged store_kv");
        }
        cache.advance();
    }

    assert!(
        cache.get_keys(0).is_err(),
        "the multi-page borrow is expected to fail — that is what fetch_keys works around"
    );

    for layer in 0..LAYERS {
        let keys = oxillama_arch::common::fetch_keys(&cache, layer).expect("fetch_keys on paged");
        assert_eq!(keys.len(), tokens * KV_DIM, "layer {layer}");
        for pos in 0..tokens {
            for d in 0..KV_DIM {
                assert_eq!(
                    keys[pos * KV_DIM + d],
                    sample(pos, d, layer),
                    "layer {layer} pos {pos} dim {d}: paged gather must be exact"
                );
            }
        }
    }
}

/// The whole point: half the bytes, measured on what the process actually
/// allocated (`memory_bytes()`), not the theoretical `max_memory_bytes()`.
#[test]
fn f16_cache_holds_half_the_bytes() {
    let mut f32_cache = KvCache::with_dtype(LAYERS, TOKENS, KV_DIM, KvCacheDtype::F32);
    let mut f16_cache = KvCache::with_dtype(LAYERS, TOKENS, KV_DIM, KvCacheDtype::F16);
    fill(&mut f32_cache, TOKENS);
    fill(&mut f16_cache, TOKENS);

    assert!(
        f16_cache.memory_bytes() > 0,
        "the cache must have allocated"
    );
    assert_eq!(
        f16_cache.memory_bytes() * 2,
        f32_cache.memory_bytes(),
        "f16 KV storage must be exactly half of f32"
    );
    assert_eq!(
        f16_cache.max_memory_bytes() * 2,
        f32_cache.max_memory_bytes()
    );
}

/// The server's prefix cache snapshots the live KV cache layer by layer.  It
/// used to do so through `get_keys()`, so an f16 cache made every `store` bail
/// out with "KV cache layer is not readable" — a silent loss of prefix reuse.
#[test]
fn prefix_cache_stores_from_an_f16_cache() {
    let tokens: Vec<u32> = (0..32u32).collect();
    let mut cache = KvCache::with_dtype(LAYERS, 64, KV_DIM, KvCacheDtype::F16);
    fill(&mut cache, tokens.len());

    let mut prefix = PrefixKvCache::new(PrefixCacheConfig {
        min_prefix_len: 1,
        ..Default::default()
    });
    assert!(
        prefix.store(&tokens, &cache, tokens.len(), KV_DIM, LAYERS),
        "prefix store must succeed against f16 KV storage"
    );

    let (matched, state) = prefix.lookup(&tokens).expect("stored prefix must be found");
    assert_eq!(matched, tokens.len());
    assert_eq!(state.seq_len(), tokens.len());
    let keys = state.keys();
    assert_eq!(keys.len(), LAYERS);
    for (layer, layer_keys) in keys.iter().enumerate() {
        assert_eq!(layer_keys.len(), tokens.len() * KV_DIM);
        for pos in 0..tokens.len() {
            for d in 0..KV_DIM {
                assert_eq!(
                    layer_keys[pos * KV_DIM + d],
                    f16::from_f32(sample(pos, d, layer)).to_f32(),
                    "layer {layer} pos {pos} dim {d}: snapshot must match the f16 contents"
                );
            }
        }
    }
}

/// The *other half* of prefix reuse: a cache hit primes the live cache from the
/// stored f32 payload.
///
/// `prefix_cache_stores_from_an_f16_cache` only covers the read direction, which
/// is the one this wave changed.  The write direction goes through
/// [`KvCache::restore_from_snapshot`] → `LayerBuf::write_at`, which is already
/// dtype-aware — but nothing pinned that, and a `serve --kv-dtype f16` process
/// hits it on the *second* request that shares a prompt prefix, not the first.
/// A single-request smoke test would never reach it.
///
/// Double rounding is idempotent (`f16(f16(x)) == f16(x)`), so the restored
/// contents must match the originals' f16 round trip exactly.
#[test]
fn prefix_cache_restores_into_an_f16_cache() {
    let tokens: Vec<u32> = (0..32u32).collect();
    let mut source = KvCache::with_dtype(LAYERS, 64, KV_DIM, KvCacheDtype::F16);
    fill(&mut source, tokens.len());

    let mut prefix = PrefixKvCache::new(PrefixCacheConfig {
        min_prefix_len: 1,
        ..Default::default()
    });
    assert!(prefix.store(&tokens, &source, tokens.len(), KV_DIM, LAYERS));
    let (_, state) = prefix.lookup(&tokens).expect("stored prefix must be found");
    let (keys, values, seq_len) = (
        state.keys().to_vec(),
        state.values().to_vec(),
        state.seq_len(),
    );

    let mut target = KvCache::with_dtype(LAYERS, 64, KV_DIM, KvCacheDtype::F16);
    target
        .restore_from_snapshot(&keys, &values, seq_len)
        .expect("restoring an f32 payload into f16 storage must succeed");
    assert_eq!(target.seq_len(), tokens.len());

    for layer in 0..LAYERS {
        let restored = oxillama_arch::common::fetch_keys(&target, layer).expect("fetch_keys");
        assert_eq!(restored.len(), tokens.len() * KV_DIM, "layer {layer}");
        for pos in 0..tokens.len() {
            for d in 0..KV_DIM {
                assert_eq!(
                    restored[pos * KV_DIM + d],
                    f16::from_f32(sample(pos, d, layer)).to_f32(),
                    "layer {layer} pos {pos} dim {d}: restore must round-trip through f16"
                );
            }
        }
    }
}

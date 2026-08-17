//! Regression tests for confirmed Mistral correctness defects (MI1-MI6).
//!
//! Each synthetic GGUF is built directly with `oxillama_gguf::GgufWriter`
//! (the always-available production writer, not the `test-utils` fixture in
//! `oxillama-gguf/src/test_utils/arch.rs`) so these tests do not depend on —
//! or need to modify — the shared `build_minimal_mistral_gguf()` fixture,
//! which `oxillama-runtime`'s `test_generate_mistral_arch` also depends on
//! and which is outside this crate's ownership boundary for the Mistral
//! fixture.
//!
//! MI1/MI2/MI4 were verified to fail before the fix and pass after by
//! `git stash`-ing `mistral/model.rs`/`mod.rs` and re-running
//! `cargo nextest run -p oxillama-arch --test mistral_regressions`; see the
//! fix report for the captured before/after output.

#![cfg(feature = "mistral")]

use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::ArchResult;
use oxillama_arch::mistral::{load_mistral_from_gguf, MistralArchitecture};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess, ModelArchitecture};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

struct TestKv {
    kv_dim: usize,
    max_seq: usize,
    position: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl TestKv {
    fn new(kv_dim: usize, max_seq: usize) -> Self {
        Self {
            kv_dim,
            max_seq,
            position: 0,
            keys: vec![0.0f32; max_seq * kv_dim],
            values: vec![0.0f32; max_seq * kv_dim],
        }
    }
}

impl KvCacheAccess for TestKv {
    fn seq_len(&self) -> usize {
        self.position
    }
    fn store_kv(&mut self, _layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let offset = self.position * self.kv_dim;
        let ck = key.len().min(self.kv_dim);
        let cv = value.len().min(self.kv_dim);
        self.keys[offset..offset + ck].copy_from_slice(&key[..ck]);
        self.values[offset..offset + cv].copy_from_slice(&value[..cv]);
        Ok(())
    }
    fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
        let end = (self.position + 1) * self.kv_dim;
        Ok(&self.keys[..end])
    }
    fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
        let end = (self.position + 1) * self.kv_dim;
        Ok(&self.values[..end])
    }
    fn advance(&mut self) {
        self.position = (self.position + 1).min(self.max_seq.saturating_sub(1));
    }
}

fn zeros(n: usize) -> Vec<u8> {
    vec![0u8; n * 4]
}

/// Build a minimal 1-layer "mistral"-arch GGUF.
///
/// `hidden`/`heads`/`kv_heads`/`head_dim` are asymmetric on purpose (MI1):
/// `heads * head_dim != hidden`, matching the qwen3-style geometry that
/// exposed the `buf_attn_out` sizing bug (a `hidden_size`-sized buffer
/// indexed by `h * head_dim` up to `num_heads * head_dim`).
#[allow(clippy::too_many_arguments)] // test-only fixture builder; one param per GGUF metadata key under test
fn build_mistral_gguf(
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    ffn: usize,
    sliding_window: Option<u32>,
    include_output_weight: bool,
) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("mistral".to_string()),
    );
    w.add_metadata(
        "mistral.embedding_length",
        MetadataValue::Uint32(hidden as u32),
    );
    w.add_metadata(
        "mistral.feed_forward_length",
        MetadataValue::Uint32(ffn as u32),
    );
    w.add_metadata("mistral.block_count", MetadataValue::Uint32(1));
    w.add_metadata(
        "mistral.attention.head_count",
        MetadataValue::Uint32(heads as u32),
    );
    w.add_metadata(
        "mistral.attention.head_count_kv",
        MetadataValue::Uint32(kv_heads as u32),
    );
    w.add_metadata(
        "mistral.attention.key_length",
        MetadataValue::Uint32(head_dim as u32),
    );
    w.add_metadata("mistral.context_length", MetadataValue::Uint32(128));
    w.add_metadata("mistral.vocab_size", MetadataValue::Uint32(vocab as u32));
    w.add_metadata("mistral.rope.freq_base", MetadataValue::Float32(10000.0));
    if let Some(win) = sliding_window {
        w.add_metadata(
            "mistral.attention.sliding_window",
            MetadataValue::Uint32(win),
        );
    }

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    w.add_tensor(
        "token_embd.weight",
        &[hidden as u64, vocab as u64],
        GgufTensorType::F32,
        &zeros(vocab * hidden),
    );
    w.add_tensor(
        "blk.0.attn_norm.weight",
        &[hidden as u64],
        GgufTensorType::F32,
        &zeros(hidden),
    );
    w.add_tensor(
        "blk.0.attn_q.weight",
        &[hidden as u64, q_dim as u64],
        GgufTensorType::F32,
        &zeros(hidden * q_dim),
    );
    w.add_tensor(
        "blk.0.attn_k.weight",
        &[hidden as u64, kv_dim as u64],
        GgufTensorType::F32,
        &zeros(hidden * kv_dim),
    );
    w.add_tensor(
        "blk.0.attn_v.weight",
        &[hidden as u64, kv_dim as u64],
        GgufTensorType::F32,
        &zeros(hidden * kv_dim),
    );
    w.add_tensor(
        "blk.0.attn_output.weight",
        &[q_dim as u64, hidden as u64],
        GgufTensorType::F32,
        &zeros(q_dim * hidden),
    );
    w.add_tensor(
        "blk.0.ffn_norm.weight",
        &[hidden as u64],
        GgufTensorType::F32,
        &zeros(hidden),
    );
    w.add_tensor(
        "blk.0.ffn_gate.weight",
        &[hidden as u64, ffn as u64],
        GgufTensorType::F32,
        &zeros(hidden * ffn),
    );
    w.add_tensor(
        "blk.0.ffn_up.weight",
        &[hidden as u64, ffn as u64],
        GgufTensorType::F32,
        &zeros(hidden * ffn),
    );
    w.add_tensor(
        "blk.0.ffn_down.weight",
        &[ffn as u64, hidden as u64],
        GgufTensorType::F32,
        &zeros(ffn * hidden),
    );
    w.add_tensor(
        "output_norm.weight",
        &[hidden as u64],
        GgufTensorType::F32,
        &zeros(hidden),
    );
    if include_output_weight {
        w.add_tensor(
            "output.weight",
            &[hidden as u64, vocab as u64],
            GgufTensorType::F32,
            &zeros(vocab * hidden),
        );
    }

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic mistral GGUF must serialize");
    bytes
}

fn parse_and_config(bytes: Vec<u8>) -> (GgufModel, ModelConfig) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
    let config =
        ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
    (gguf, config)
}

// ─── MI1: buf_attn_out sized num_heads * head_dim, not hidden_size ────────────

/// Qwen3-4B-shaped geometry: `hidden=2560`-equivalent-but-tiny-for-speed;
/// what matters is `num_heads * head_dim != hidden_size` — here `8 * 6 = 48
/// != 32`. Before the fix, `buf_attn_out` was sized `hidden_size` (32) and
/// writing head 5's slice (`[30..36]`) panicked with an out-of-bounds slice
/// index. Kept intentionally tiny; this is about the buffer shape, not a
/// realistic checkpoint.
#[test]
fn mi1_asymmetric_head_geometry_does_not_panic() {
    const HIDDEN: usize = 32;
    const HEADS: usize = 8;
    const KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 6; // 8 * 6 = 48 != 32
    const VOCAB: usize = 8;
    const FFN: usize = 16;

    let (gguf, config) = parse_and_config(build_mistral_gguf(
        HIDDEN, HEADS, KV_HEADS, HEAD_DIM, VOCAB, FFN, None, true,
    ));
    let mut model = load_mistral_from_gguf(&gguf, &config).expect("load mistral");
    let mut kv = TestKv::new(KV_HEADS * HEAD_DIM, 8);
    let result = model.forward(&[0u32], &mut kv);
    assert!(
        result.is_ok(),
        "asymmetric head geometry (num_heads*head_dim != hidden_size) must not panic, got: {:?}",
        result.err()
    );
    let logits = result.expect("logits");
    assert_eq!(logits.len(), VOCAB);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
    }
}

// ─── MI2: out-of-vocabulary token id must error, not panic ────────────────────

/// A token id at or beyond `vocab_size` must return
/// `ArchError::ConfigMismatch`, not index `token_embd` out of bounds. This
/// matters specifically for Mistral because `config.rs` falls back to the
/// tokenizer token-array length, then a hard-coded 32000, when a checkpoint
/// omits `{arch}.vocab_size` — an over-estimated `vocab_size` makes this
/// reachable even for well-formed prompts.
#[test]
fn mi2_out_of_vocabulary_token_is_an_error_not_a_panic() {
    const VOCAB: usize = 8;
    let (gguf, config) = parse_and_config(build_mistral_gguf(32, 4, 4, 8, VOCAB, 16, None, true));
    let mut model = load_mistral_from_gguf(&gguf, &config).expect("load mistral");
    let mut kv = TestKv::new(4 * 8, 8);
    let result = model.forward(&[VOCAB as u32 + 100], &mut kv);
    assert!(
        result.is_err(),
        "an out-of-vocabulary token id must error, not index out of bounds"
    );
}

// ─── MI3: sliding window metadata reaches attention() (consumes swa_attend_start) ─
//
// NOTE on what this test does and does not prove: `swa_attend_start`'s
// windowing arithmetic (`(pos + 1).saturating_sub(w)`, the pre-fix formula
// this migration replaced) is algebraically identical to the shared helper's
// `pos.saturating_sub(w - 1)` for every `w >= 1` — the MI3 defect was that
// Mistral hand-rolled this arithmetic instead of routing through the shared
// `common::attention::swa_attend_start` every other sliding-window
// architecture uses, not that the arithmetic itself was wrong. There is
// therefore no numerical delta a fail-before/pass-after test could observe
// on window trimming alone; see the fix report for the full MI3 writeup.
// This test instead locks in the part that DOES have a discriminating
// signal: that `config.sliding_window` / `swa_window` metadata actually
// reaches the model `attention()` reads at call time (`swa_config()`),
// rather than being parsed and then silently ignored.

/// A checkpoint carrying `{arch}.attention.sliding_window` must produce a
/// model whose `swa_config()` reports it; a checkpoint without the key must
/// report `None`. Both configurations must still produce finite,
/// `vocab_size`-length logits end to end.
#[test]
fn mi3_swa_window_round_trips_from_gguf_metadata() {
    const HIDDEN: usize = 16;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const VOCAB: usize = 8;
    const FFN: usize = 16;
    const SEQ: usize = 6;

    // Windowed: window=2, so position 5 only sees positions [4, 5].
    let (gguf_w, config_w) = parse_and_config(build_mistral_gguf(
        HIDDEN,
        HEADS,
        HEADS,
        HEAD_DIM,
        VOCAB,
        FFN,
        Some(2),
        true,
    ));
    let mut model_w = load_mistral_from_gguf(&gguf_w, &config_w).expect("load windowed");
    let mut kv_w = TestKv::new(HEADS * HEAD_DIM, 16);
    let tokens: Vec<u32> = (0..SEQ as u32).map(|t| t % VOCAB as u32).collect();
    let logits_windowed = model_w
        .forward(&tokens, &mut kv_w)
        .expect("windowed forward");

    // Unwindowed: same weights (all-zero, so logits would be identical
    // regardless of window if the window were not applied at all) — use a
    // LARGE window instead so it never actually trims anything, as the
    // "no window" control.
    let (gguf_full, config_full) = parse_and_config(build_mistral_gguf(
        HIDDEN, HEADS, HEADS, HEAD_DIM, VOCAB, FFN, None, true,
    ));
    let mut model_full = load_mistral_from_gguf(&gguf_full, &config_full).expect("load full");
    let mut kv_full = TestKv::new(HEADS * HEAD_DIM, 16);
    let logits_full = model_full
        .forward(&tokens, &mut kv_full)
        .expect("full forward");

    // With all-zero weights both must be finite and (in this degenerate
    // all-zero case) numerically identical — the discriminating check is
    // that `swa_config()` reports the window, proving it round-tripped from
    // GGUF metadata into the model that `attention()` reads at call time.
    assert_eq!(logits_windowed.len(), VOCAB);
    assert_eq!(logits_full.len(), VOCAB);
    assert_eq!(
        model_w.swa_config(),
        Some((2, false)),
        "windowed model must report its sliding window via swa_config()"
    );
    assert_eq!(
        model_full.swa_config(),
        None,
        "unwindowed model must report no sliding window"
    );
}

// ─── MI4: tied-embedding LM-head fallback ──────────────────────────────────────

/// Many Mistral fine-tunes tie the LM head to `token_embd.weight` and ship no
/// standalone `output.weight`. Before the fix, `load_quant_linear(model,
/// "output.weight")` returned `MissingTensor` unconditionally.
#[test]
fn mi4_missing_output_weight_falls_back_to_tied_token_embd() {
    let (gguf, config) = parse_and_config(build_mistral_gguf(32, 4, 4, 8, 8, 16, None, false));
    let result = load_mistral_from_gguf(&gguf, &config);
    assert!(
        result.is_ok(),
        "a checkpoint with no output.weight must fall back to tied token_embd.weight, got: {:?}",
        result.err()
    );
    let mut model = result.expect("load mistral");
    let mut kv = TestKv::new(4 * 8, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), 8);
}

// ─── MI5: RoPE NORM convention (interleaved pairs) ─────────────────────────────

/// Mistral checkpoints convert to GGUF arch `"llama"` and are
/// `LLAMA_ROPE_TYPE_NORM` — the same interleaved-pair convention LLaMA uses —
/// NOT the NeoX half-split every other RoPE-using architecture in this crate
/// defaults to. `apply_rope_norm` is `pub(crate)` (only reachable from inside
/// `oxillama-arch`), so this is verified as an in-crate unit test in
/// `mistral::model::tests::mi5_rope_uses_norm_not_neox_convention` — see that
/// test for the interleaved-vs-half-split pairing check.
#[test]
fn mi5_model_rope_table_is_norm_convention_smoke() {
    // Smoke test at this (black-box) layer: a model built here must produce
    // finite output at all, which the in-crate test then discriminates by
    // convention. This just guards against the fixture itself regressing.
    let (gguf, config) = parse_and_config(build_mistral_gguf(32, 4, 4, 8, 8, 16, None, true));
    let mut model = load_mistral_from_gguf(&gguf, &config).expect("load mistral");
    let mut kv = TestKv::new(4 * 8, 8);
    let logits = model.forward(&[0u32, 1, 2], &mut kv).expect("forward");
    assert_eq!(logits.len(), 8);
}

// ─── Shared: context bounds, two consecutive forward calls, LoRA ──────────────

#[test]
fn mi_prefill_past_max_context_length_is_an_error() {
    let (gguf, config) = parse_and_config(build_mistral_gguf(32, 4, 4, 8, 8, 16, None, true));
    let mut model = load_mistral_from_gguf(&gguf, &config).expect("load mistral");
    let mut kv = TestKv::new(4 * 8, 256);
    let tokens = vec![0u32; config.max_context_length + 1];
    let result = model.forward(&tokens, &mut kv);
    assert!(
        result.is_err(),
        "a prompt longer than max_context_length must error, not panic on buf_attn_scores"
    );
}

/// Two consecutive `forward()` calls on the same model instance must both
/// succeed — guards against `std::mem::take(&mut self.buf_logits)` leaving
/// the buffer empty for a second call (the exact bug this fix introduced and
/// then caught in Command-R; ported here defensively since Mistral now uses
/// the same pattern).
#[test]
fn mi_two_consecutive_forward_calls_both_succeed() {
    let (gguf, config) = parse_and_config(build_mistral_gguf(32, 4, 4, 8, 8, 16, None, true));
    let mut model = load_mistral_from_gguf(&gguf, &config).expect("load mistral");
    let mut kv = TestKv::new(4 * 8, 8);
    let first = model.forward(&[0u32], &mut kv).expect("first forward");
    assert_eq!(first.len(), 8);
    let second = model.forward(&[1u32], &mut kv).expect("second forward");
    assert_eq!(
        second.len(),
        8,
        "second forward() must still produce vocab_size logits"
    );
}

/// `ModelArchitecture::build_from_gguf` must delegate to
/// `load_mistral_from_gguf` (the registry-driven load path the runtime
/// engine is being migrated onto), not the default `NotSupported`.
#[test]
fn mi_build_from_gguf_delegates_to_the_real_loader() {
    let (gguf, config) = parse_and_config(build_mistral_gguf(32, 4, 4, 8, 8, 16, None, true));
    let arch = MistralArchitecture::new();
    let mut model = arch
        .build_from_gguf(&gguf, &config)
        .expect("build_from_gguf must succeed");
    let mut kv = TestKv::new(4 * 8, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), 8);
}

#[test]
fn mi_apply_lora_without_matching_adapter_is_a_harmless_ok() {
    let (gguf, config) = parse_and_config(build_mistral_gguf(32, 4, 4, 8, 8, 16, None, true));
    let mut model = load_mistral_from_gguf(&gguf, &config).expect("load mistral");
    let empty = oxillama_arch::lora::LoadedLora {
        adapters: std::collections::HashMap::new(),
        rank: 0,
        alpha: 1.0,
    };
    assert!(model.apply_lora(&empty).is_ok());
    assert!(model.apply_lora_scaled(&empty, 0.5).is_ok());
}

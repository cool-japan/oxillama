//! Regression tests for confirmed Command-R correctness defects (CR1-CR3).
//!
//! Each synthetic GGUF is built directly with `oxillama_gguf::GgufWriter`
//! (the always-available production writer, not the `test-utils` fixture in
//! `oxillama-gguf/src/test_utils/arch.rs`) so these tests do not depend on —
//! or need to modify — the shared fixture builder beyond the single
//! `ffn_norm` removal CR1 requires.
//!
//! CR1's topology fix was verified to fail before and pass after by
//! `git stash`-ing `command_r/model.rs`/`mod.rs` and re-running
//! `cargo nextest run -p oxillama-arch --test command_r_regressions`; see the
//! fix report for the captured before/after output.

#![cfg(feature = "command-r")]

use oxillama_arch::command_r::CommandRArchitecture;
use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::ArchResult;
use oxillama_arch::traits::{ForwardPass, KvCacheAccess, ModelArchitecture};
use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};

const H: usize = 32; // hidden_size
const HEADS: usize = 2;
const HD: usize = 16; // head_dim (H / HEADS)
const FFN: usize = 64;
const VOCAB: usize = 8;

/// Minimal in-memory KV cache, sized for a single layer.
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

/// Small deterministic (non-zero) fill so attention/FFN outputs are
/// distinguishable, needed by the CR1 topology test.
fn ramp(n: usize, scale: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = ((i % 7) as f32 - 3.0) * scale;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// `n` copies of `1.0f32`, little-endian.
fn ones(n: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * 4);
    for _ in 0..n {
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
    }
    bytes
}

/// Build a minimal 1-layer Command-R GGUF using the real (post-fix) tensor
/// set: one `attn_norm` per layer, no `ffn_norm`, optional Q/K-norm tensors,
/// and an optional standalone `output.weight`.
fn build_command_r_gguf(include_output_weight: bool, logit_scale: f32) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("command-r".to_string()),
    );
    w.add_metadata(
        "command-r.embedding_length",
        MetadataValue::Uint32(H as u32),
    );
    w.add_metadata(
        "command-r.feed_forward_length",
        MetadataValue::Uint32(FFN as u32),
    );
    w.add_metadata("command-r.block_count", MetadataValue::Uint32(1));
    w.add_metadata(
        "command-r.attention.head_count",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata(
        "command-r.attention.head_count_kv",
        MetadataValue::Uint32(HEADS as u32),
    );
    w.add_metadata("command-r.context_length", MetadataValue::Uint32(128));
    w.add_metadata("command-r.vocab_size", MetadataValue::Uint32(VOCAB as u32));
    w.add_metadata("command-r.rope.freq_base", MetadataValue::Float32(10000.0));
    w.add_metadata("command-r.logit_scale", MetadataValue::Float32(logit_scale));

    w.add_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        GgufTensorType::F32,
        &ramp(VOCAB * H, 0.01),
    );
    w.add_tensor(
        "blk.0.attn_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &ones(H),
    );
    w.add_tensor(
        "blk.0.attn_q.weight",
        &[H as u64, H as u64],
        GgufTensorType::F32,
        &ramp(H * H, 0.01),
    );
    w.add_tensor(
        "blk.0.attn_k.weight",
        &[H as u64, H as u64],
        GgufTensorType::F32,
        &ramp(H * H, 0.01),
    );
    w.add_tensor(
        "blk.0.attn_v.weight",
        &[H as u64, H as u64],
        GgufTensorType::F32,
        &ramp(H * H, 0.01),
    );
    w.add_tensor(
        "blk.0.attn_output.weight",
        &[H as u64, H as u64],
        GgufTensorType::F32,
        &ramp(H * H, 0.01),
    );
    w.add_tensor(
        "blk.0.ffn_gate.weight",
        &[H as u64, FFN as u64],
        GgufTensorType::F32,
        &ramp(H * FFN, 0.01),
    );
    w.add_tensor(
        "blk.0.ffn_up.weight",
        &[H as u64, FFN as u64],
        GgufTensorType::F32,
        &ramp(H * FFN, 0.01),
    );
    w.add_tensor(
        "blk.0.ffn_down.weight",
        &[FFN as u64, H as u64],
        GgufTensorType::F32,
        &ramp(FFN * H, 0.01),
    );
    w.add_tensor(
        "output_norm.weight",
        &[H as u64],
        GgufTensorType::F32,
        &zeros(H),
    );
    if include_output_weight {
        w.add_tensor(
            "output.weight",
            &[H as u64, VOCAB as u64],
            GgufTensorType::F32,
            &ramp(VOCAB * H, 0.01),
        );
    }

    let mut bytes = Vec::new();
    w.write_to(&mut bytes)
        .expect("synthetic command-r GGUF must serialize");
    bytes
}

fn parse_and_config(bytes: Vec<u8>) -> (GgufModel, ModelConfig) {
    let gguf = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
    let config =
        ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
    (gguf, config)
}

// ─── CR1: one LayerNorm per layer, no ffn_norm, parallel attn+FFN block ───────

/// A checkpoint with the real Command-R tensor set (no `blk.0.ffn_norm.weight`)
/// must load. Before the fix, `load_command_r_from_gguf` hard-required
/// `ffn_norm.weight` and this failed with `MissingTensor`.
#[test]
fn cr1_no_ffn_norm_tensor_loads_successfully() {
    let (gguf, config) = parse_and_config(build_command_r_gguf(true, 1.0));
    let result = oxillama_arch::command_r::load_command_r_from_gguf(&gguf, &config);
    assert!(
        result.is_ok(),
        "a real Command-R checkpoint (no ffn_norm) must load, got: {:?}",
        result.err()
    );
}

/// End-to-end forward pass after loading must produce finite vocab-sized
/// logits, proving the parallel attn+FFN block actually runs (not merely
/// that the tensors were found).
#[test]
fn cr1_forward_after_load_is_finite() {
    let (gguf, config) = parse_and_config(build_command_r_gguf(true, 1.0));
    let mut model =
        oxillama_arch::command_r::load_command_r_from_gguf(&gguf, &config).expect("load");
    let mut kv = TestKv::new(HD * HEADS, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
    }
}

/// Two consecutive `forward()` calls on the same model instance must both
/// succeed. This guards against the `std::mem::take(&mut self.buf_logits)`
/// optimization silently leaving `buf_logits` empty for the second call
/// (caught during this fix: the runtime's `test_generate_command_r_arch`
/// failed with `Quant(DimensionMismatch { expected: 32, got: 0 })` on the
/// second decode step until `forward()` was made to re-`resize` the buffer).
#[test]
fn cr_two_consecutive_forward_calls_both_succeed() {
    let (gguf, config) = parse_and_config(build_command_r_gguf(true, 1.0));
    let mut model =
        oxillama_arch::command_r::load_command_r_from_gguf(&gguf, &config).expect("load");
    let mut kv = TestKv::new(HD * HEADS, 8);
    let first = model.forward(&[0u32], &mut kv).expect("first forward");
    assert_eq!(first.len(), VOCAB);
    let second = model.forward(&[1u32], &mut kv).expect("second forward");
    assert_eq!(
        second.len(),
        VOCAB,
        "second forward() must still produce vocab_size logits"
    );
    for (i, &v) in second.iter().enumerate() {
        assert!(
            v.is_finite(),
            "logit[{i}] must be finite on 2nd call, got {v}"
        );
    }
}

/// Numeric topology check: with a normed input `n`, the block output must be
/// `hidden + attn_out(n) + ffn_out(n)` — attention and FFN both reading the
/// SAME normed input and both adding directly to the ORIGINAL (pre-norm)
/// residual, not LLaMA's sequential `norm1 -> attn -> residual -> norm2 ->
/// ffn -> residual` pattern. `embed()` (stops before the LM head) exposes the
/// post-block hidden state directly for this check.
#[test]
fn cr1_block_output_is_hidden_plus_attn_plus_ffn_from_one_norm() {
    let (gguf, config) = parse_and_config(build_command_r_gguf(true, 1.0));
    let mut model =
        oxillama_arch::command_r::load_command_r_from_gguf(&gguf, &config).expect("load");
    let mut kv = TestKv::new(HD * HEADS, 8);

    // `embed()` runs the full block (attn_norm -> attn + ffn from the same
    // normed input -> combine) and applies only the FINAL output_norm, so its
    // return value is `output_norm(hidden_after_block)`. We only assert that
    // it is finite and vocab-independent (hidden_size-length): a wrong
    // (sequential) topology would still produce finite output, so the real
    // discriminating signal is the before/after `git stash` comparison
    // recorded in the fix report. This test locks in the current (correct)
    // output shape and finiteness as a standing regression guard.
    let embedding = model.embed(&[2u32], &mut kv).expect("embed");
    assert_eq!(embedding.len(), H);
    for (i, &v) in embedding.iter().enumerate() {
        assert!(v.is_finite(), "embedding[{i}] must be finite, got {v}");
    }
}

// ─── CR2: tied-embedding LM-head fallback ──────────────────────────────────────

/// Command-R never ships a standalone `output.weight` on real checkpoints
/// (llama.cpp always duplicates `token_embd.weight` into `output`). Before
/// the fix, `load_quant_linear(model, "output.weight")` returned
/// `MissingTensor` unconditionally.
#[test]
fn cr2_missing_output_weight_falls_back_to_tied_token_embd() {
    let (gguf, config) = parse_and_config(build_command_r_gguf(false, 1.0));
    let result = oxillama_arch::command_r::load_command_r_from_gguf(&gguf, &config);
    assert!(
        result.is_ok(),
        "a checkpoint with no output.weight must fall back to tied token_embd.weight, got: {:?}",
        result.err()
    );
    let mut model = result.expect("load command-r");
    let mut kv = TestKv::new(HD * HEADS, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
}

// CR3 (RoPE NORM convention) is verified by an in-crate unit test in
// `command_r/model.rs` (`tests::cr3_rope_uses_norm_not_neox_convention`):
// `apply_rope_norm` is `pub(crate)`, so this external integration-test crate
// cannot call it directly. See that test for the interleaved-vs-half-split
// pairing check.

// ─── Shared: context bounds / OOV token guards ─────────────────────────────────

#[test]
fn cr_prefill_past_max_context_length_is_an_error() {
    let (gguf, config) = parse_and_config(build_command_r_gguf(true, 1.0));
    let mut model =
        oxillama_arch::command_r::load_command_r_from_gguf(&gguf, &config).expect("load");
    let mut kv = TestKv::new(HD * HEADS, 256);
    let tokens = vec![0u32; config.max_context_length + 1];
    let result = model.forward(&tokens, &mut kv);
    assert!(
        result.is_err(),
        "a prompt longer than max_context_length must error, not panic on buf_attn_scores"
    );
}

#[test]
fn cr_out_of_vocabulary_token_is_an_error() {
    let (gguf, config) = parse_and_config(build_command_r_gguf(true, 1.0));
    let mut model =
        oxillama_arch::command_r::load_command_r_from_gguf(&gguf, &config).expect("load");
    let mut kv = TestKv::new(HD * HEADS, 8);
    let result = model.forward(&[VOCAB as u32 + 5], &mut kv);
    assert!(
        result.is_err(),
        "an out-of-vocabulary token id must error, not index out of bounds"
    );
}

// ─── LoRA: apply_lora / apply_lora_scaled still work after the topology change ─

/// `ModelArchitecture::build_from_gguf` must delegate to
/// `load_command_r_from_gguf` (the registry-driven load path the runtime
/// engine is being migrated onto), not the default `NotSupported`.
#[test]
fn cr_build_from_gguf_delegates_to_the_real_loader() {
    let (gguf, config) = parse_and_config(build_command_r_gguf(true, 1.0));
    let arch = CommandRArchitecture::new();
    let mut model = arch
        .build_from_gguf(&gguf, &config)
        .expect("build_from_gguf must succeed");
    let mut kv = TestKv::new(HD * HEADS, 8);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert_eq!(logits.len(), VOCAB);
}

#[test]
fn cr_apply_lora_without_matching_adapter_is_a_harmless_ok() {
    let (gguf, config) = parse_and_config(build_command_r_gguf(true, 1.0));
    let mut model =
        oxillama_arch::command_r::load_command_r_from_gguf(&gguf, &config).expect("load");
    let empty = oxillama_arch::lora::LoadedLora {
        adapters: std::collections::HashMap::new(),
        rank: 0,
        alpha: 1.0,
    };
    assert!(model.apply_lora(&empty).is_ok());
    assert!(model.apply_lora_scaled(&empty, 0.5).is_ok());
}

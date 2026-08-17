//! Integration tests for the DBRX architecture.
//!
//! These drive the model through the **GGUF fixture** rather than a
//! hand-assembled struct, because the fixture is where the tensor-set
//! regressions live: DBRX ships a fused `attn_qkv` and an `attn_output_norm`
//! and has neither separate `attn_q`/`attn_k`/`attn_v` nor an `ffn_norm`.
//!
//! The KV cache used here actually stores rows.  The previous `NullKv` stub
//! returned `seq_len() == 0` and `get_keys() == &[]`, which made every attention
//! score `-inf` and `attn_out` exactly zero — no cache-indexing bug could change
//! a logit, which is why a per-*layer* `kv_cache.advance()` survived.

#![cfg(feature = "dbrx")]

use oxillama_arch::dbrx::load_dbrx_from_gguf;
use oxillama_arch::error::ArchResult;
use oxillama_arch::registry::ArchitectureRegistry;
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};

/// A KV cache that records what each layer wrote and where.
struct RecordingKv {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    kv_dim: usize,
    seq_len: usize,
    stored_len: usize,
    advances: usize,
    writes: Vec<usize>,
}

impl RecordingKv {
    fn new(num_layers: usize, kv_dim: usize, max_seq_len: usize) -> Self {
        Self {
            keys: vec![vec![0.0; kv_dim * max_seq_len]; num_layers],
            values: vec![vec![0.0; kv_dim * max_seq_len]; num_layers],
            kv_dim,
            seq_len: 0,
            stored_len: 0,
            advances: 0,
            writes: vec![0; num_layers],
        }
    }
}

impl KvCacheAccess for RecordingKv {
    fn seq_len(&self) -> usize {
        self.seq_len
    }
    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let off = self.seq_len * self.kv_dim;
        let end = off + self.kv_dim;
        self.keys[layer][off..end].copy_from_slice(&key[..self.kv_dim]);
        self.values[layer][off..end].copy_from_slice(&value[..self.kv_dim]);
        self.writes[layer] += 1;
        if self.stored_len <= self.seq_len {
            self.stored_len = self.seq_len + 1;
        }
        Ok(())
    }
    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.keys[layer][..self.stored_len * self.kv_dim])
    }
    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.values[layer][..self.stored_len * self.kv_dim])
    }
    fn advance(&mut self) {
        self.seq_len += 1;
        self.advances += 1;
        if self.stored_len < self.seq_len {
            self.stored_len = self.seq_len;
        }
    }
    fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

fn load() -> oxillama_arch::dbrx::DbrxModel {
    let bytes = oxillama_gguf::test_utils::build_minimal_dbrx_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("DBRX GGUF fixture must parse");
    load_dbrx_from_gguf(&gguf).expect("load_dbrx_from_gguf must succeed")
}

fn cache_for(model: &oxillama_arch::dbrx::DbrxModel) -> RecordingKv {
    RecordingKv::new(
        model.layers.len(),
        model.config.num_kv_heads * model.config.head_dim,
        model.config.max_context_length,
    )
}

/// DBRX is registered in the architecture registry under "dbrx".
#[test]
fn test_dbrx_registered_in_registry() {
    let reg = ArchitectureRegistry::with_builtins();
    assert!(reg.contains("dbrx"), "registry must contain 'dbrx'");
    let arch = reg.get("dbrx").expect("get dbrx must succeed");
    assert_eq!(arch.arch_id(), "dbrx");
}

/// The registry can now actually *build* DBRX.  Before `build_from_gguf` was
/// overridden the default returned `NotSupported` and the engine could reach
/// only 7 of the 27 registered architectures.
#[test]
fn test_dbrx_reachable_through_registry() {
    let bytes = oxillama_gguf::test_utils::build_minimal_dbrx_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("fixture parses");
    let config = oxillama_arch::config::ModelConfig::from_metadata(&gguf.file.metadata)
        .expect("config parses");
    let reg = ArchitectureRegistry::with_builtins();
    let arch = reg.get("dbrx").expect("get dbrx");
    let model = arch
        .build_from_gguf(&gguf, &config)
        .expect("registry must be able to build DBRX from a GGUF");
    assert_eq!(model.vocab_size(), 32);
}

/// Forward pass produces logits of the correct shape.
#[test]
fn test_dbrx_forward_shape() {
    let mut model = load();
    let mut kv = cache_for(&model);
    let logits = model
        .forward(&[1u32, 2, 3], &mut kv)
        .expect("forward must succeed");
    assert_eq!(logits.len(), 32, "logits must have vocab_size=32 elements");
}

/// All forward pass outputs are finite.
#[test]
fn test_dbrx_forward_finite() {
    let mut model = load();
    let mut kv = cache_for(&model);
    let logits = model
        .forward(&[0u32], &mut kv)
        .expect("forward must succeed");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "all DBRX logits must be finite"
    );
}

/// GGUF fixture parses correctly.
#[test]
fn test_dbrx_gguf_fixture_parses() {
    let bytes = oxillama_gguf::test_utils::build_minimal_dbrx_gguf();
    let model = oxillama_gguf::GgufModel::from_bytes(bytes).expect("DBRX GGUF fixture must parse");
    assert_eq!(model.architecture().expect("arch must be present"), "dbrx");
}

/// `advance()` is a per-**token** operation, not a per-layer one.
///
/// It used to be the last statement of `attention_single_token`, so a 2-layer
/// model consumed 2 cache slots per token, layer 1 read the rows layer 0 wrote,
/// and the usable context was `max_ctx / n_layers`.
#[test]
fn test_dbrx_advances_once_per_token() {
    let mut model = load();
    let n_layers = model.layers.len();
    assert!(n_layers >= 2, "the regression needs >= 2 layers to show");
    let mut kv = cache_for(&model);
    model
        .forward(&[1u32, 2, 3, 0], &mut kv)
        .expect("forward must succeed");
    assert_eq!(
        kv.advances, 4,
        "4 tokens through {n_layers} layers must advance the cache 4 times"
    );
    assert_eq!(kv.seq_len(), 4);
    for l in 0..n_layers {
        assert_eq!(kv.writes[l], 4, "layer {l} must write one row per token");
    }
}

/// Positions come from the cache, so a second request in the same slot starts
/// at 0 again.  The model used to keep a monotonic `current_pos` that was never
/// reset, so every RoPE angle of the second request was wrong.
#[test]
fn test_dbrx_second_sequence_restarts_at_position_zero() {
    let mut model = load();
    let mut kv1 = cache_for(&model);
    let first = model.forward(&[1u32, 2], &mut kv1).expect("first request");

    model.reset_sequence();
    let mut kv2 = cache_for(&model);
    let second = model.forward(&[1u32, 2], &mut kv2).expect("second request");

    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit {i} differs between two identical requests: {a} vs {b}"
        );
    }
}

/// An over-long prompt is reported instead of indexing `buf_attn_scores` past
/// its end.
#[test]
fn test_dbrx_rejects_overlong_prompt() {
    let mut model = load();
    let max = model.max_context_length();
    let mut kv = cache_for(&model);
    let tokens: Vec<u32> = (0..=max as u32).map(|t| t % 4).collect();
    assert!(
        model.forward(&tokens, &mut kv).is_err(),
        "a prompt of {} tokens must be rejected for a {max}-token context",
        tokens.len()
    );
}

/// A token id past the end of the embedding matrix is an error, not a panic.
#[test]
fn test_dbrx_rejects_out_of_vocab_token() {
    let mut model = load();
    let mut kv = cache_for(&model);
    assert!(model.forward(&[10_000u32], &mut kv).is_err());
}

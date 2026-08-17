//! Integration tests for the Grok-1 architecture.
//!
//! Driven through the GGUF fixture, with a KV cache that actually stores rows.
//! The previous tests built a hand-assembled all-zero model behind a `NullKv`
//! stub, which made every attention score `-inf` and every logit exactly 0.

#![cfg(feature = "grok")]

use oxillama_arch::error::ArchResult;
use oxillama_arch::grok::{load_grok_from_gguf, GrokModel};
use oxillama_arch::registry::ArchitectureRegistry;
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};

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

fn load() -> GrokModel {
    let bytes = oxillama_gguf::test_utils::build_minimal_grok_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("Grok GGUF fixture must parse");
    load_grok_from_gguf(&gguf).expect("load_grok_from_gguf must succeed")
}

fn cache_for(model: &GrokModel) -> RecordingKv {
    RecordingKv::new(
        model.layers.len(),
        model.config.num_kv_heads * model.config.head_dim,
        model.config.max_context_length,
    )
}

#[test]
fn test_grok_registered_in_registry() {
    let reg = ArchitectureRegistry::with_builtins();
    assert!(reg.contains("grok"), "registry must contain 'grok'");
    assert_eq!(reg.get("grok").expect("get grok").arch_id(), "grok");
}

/// The registry can now actually build Grok; the default `build_from_gguf`
/// returned `NotSupported`.
#[test]
fn test_grok_reachable_through_registry() {
    let bytes = oxillama_gguf::test_utils::build_minimal_grok_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("fixture parses");
    let config = oxillama_arch::config::ModelConfig::from_metadata(&gguf.file.metadata)
        .expect("config parses");
    let reg = ArchitectureRegistry::with_builtins();
    let model = reg
        .get("grok")
        .expect("get grok")
        .build_from_gguf(&gguf, &config)
        .expect("registry must be able to build Grok from a GGUF");
    assert_eq!(model.vocab_size(), 32);
}

#[test]
fn test_grok_forward_shape() {
    let mut model = load();
    let mut kv = cache_for(&model);
    let logits = model.forward(&[1u32, 2, 3], &mut kv).expect("forward");
    assert_eq!(logits.len(), 32);
}

#[test]
fn test_grok_forward_finite() {
    let mut model = load();
    let mut kv = cache_for(&model);
    let logits = model.forward(&[0u32], &mut kv).expect("forward");
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_grok_gguf_fixture_parses() {
    let bytes = oxillama_gguf::test_utils::build_minimal_grok_gguf();
    let model = oxillama_gguf::GgufModel::from_bytes(bytes).expect("Grok GGUF fixture must parse");
    assert_eq!(model.architecture().expect("arch"), "grok");
}

#[test]
fn test_grok_default_rope_theta_is_1e6() {
    let model = load();
    assert!(
        (model.grok_config.rope_theta - 1_000_000.0).abs() < 1.0,
        "Grok-1 rope_theta is 1e6, got {}",
        model.grok_config.rope_theta
    );
}

/// Grok's distinctive scalars must survive the loader.  None of them existed as
/// fields before, so the embedding was ~78x too small and the logits ~1.73x too
/// large.
#[test]
fn test_grok_scales_are_loaded() {
    let model = load();
    assert!((model.grok_config.embedding_scale - 78.383_67).abs() < 1e-3);
    assert!((model.grok_config.logit_scale - 0.577_350_26).abs() < 1e-6);
    assert!((model.grok_config.attn_output_scale - 0.088_388_35).abs() < 1e-7);
    assert!((model.grok_config.attn_logit_softcapping - 30.0).abs() < 1e-6);
}

/// `advance()` is per-token, not per-layer.
#[test]
fn test_grok_advances_once_per_token() {
    let mut model = load();
    let n_layers = model.layers.len();
    assert!(n_layers >= 2);
    let mut kv = cache_for(&model);
    model.forward(&[1u32, 2, 3, 0], &mut kv).expect("forward");
    assert_eq!(
        kv.advances, 4,
        "4 tokens through {n_layers} layers must advance the cache 4 times"
    );
    for l in 0..n_layers {
        assert_eq!(kv.writes[l], 4, "layer {l} writes one row per token");
    }
}

/// Two identical requests must produce identical logits.  With the old
/// never-reset `current_pos`, the second request's RoPE angles were wrong.
#[test]
fn test_grok_second_sequence_restarts_at_position_zero() {
    let mut model = load();
    let mut kv1 = cache_for(&model);
    let first = model.forward(&[1u32, 2], &mut kv1).expect("first");
    model.reset_sequence();
    let mut kv2 = cache_for(&model);
    let second = model.forward(&[1u32, 2], &mut kv2).expect("second");
    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "logit {i}: {a} vs {b}");
    }
}

#[test]
fn test_grok_rejects_overlong_prompt() {
    let mut model = load();
    let max = model.max_context_length();
    let mut kv = cache_for(&model);
    let tokens: Vec<u32> = (0..=max as u32).map(|t| t % 4).collect();
    assert!(model.forward(&tokens, &mut kv).is_err());
}

#[test]
fn test_grok_rejects_out_of_vocab_token() {
    let mut model = load();
    let mut kv = cache_for(&model);
    assert!(model.forward(&[10_000u32], &mut kv).is_err());
}

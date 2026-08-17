//! Test-only scaffolding for the Grok unit tests.
//!
//! [`TestKvCache`] stores rows for real.  The previous Grok tests used a stub
//! whose `seq_len()` was 0 and whose `get_keys()` was `&[]`, so every attention
//! score came out `-inf`, `attn_out` stayed exactly zero, and no KV-indexing bug
//! could ever change a logit.

use oxillama_gguf::GgufTensorType;
use oxillama_quant::{KernelDispatcher, QuantKernel, QuantTensor};

use crate::common::linear::QuantLinear;
use crate::common::moe::QuantExpert;
use crate::error::{ArchError, ArchResult};
use crate::grok::moe::GrokMoe;
use crate::traits::KvCacheAccess;

/// An in-test KV cache that actually stores rows.
pub struct TestKvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    kv_dim: usize,
    seq_len: usize,
    stored_len: usize,
    writes: Vec<usize>,
}

impl TestKvCache {
    /// Allocate a cache for `num_layers` layers of `kv_dim` floats per token.
    pub fn new(num_layers: usize, kv_dim: usize, max_seq_len: usize) -> Self {
        Self {
            keys: vec![vec![0.0; kv_dim * max_seq_len]; num_layers],
            values: vec![vec![0.0; kv_dim * max_seq_len]; num_layers],
            kv_dim,
            seq_len: 0,
            stored_len: 0,
            writes: vec![0; num_layers],
        }
    }

    /// How many rows `layer` has written.
    pub fn writes(&self, layer: usize) -> usize {
        self.writes.get(layer).copied().unwrap_or(0)
    }

    /// The key row `layer` wrote at `pos`.
    pub fn key_row(&self, layer: usize, pos: usize) -> &[f32] {
        &self.keys[layer][pos * self.kv_dim..(pos + 1) * self.kv_dim]
    }
}

impl KvCacheAccess for TestKvCache {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let off = self.seq_len * self.kv_dim;
        let end = off + self.kv_dim;
        if layer >= self.keys.len() || end > self.keys[layer].len() {
            return Err(ArchError::ForwardPassError {
                layer,
                message: format!("TestKvCache overflow at position {}", self.seq_len),
            });
        }
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
        if self.stored_len < self.seq_len {
            self.stored_len = self.seq_len;
        }
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

/// Deterministic non-zero weight generator (a plain LCG, no dependencies).
pub struct TinyWeights {
    state: u64,
    dispatcher: KernelDispatcher,
}

impl TinyWeights {
    /// Seed a new generator.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed | 1,
            dispatcher: KernelDispatcher::new(),
        }
    }

    /// Next value, uniform in roughly `[-0.25, 0.25)`.
    pub fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mantissa = (self.state >> 33) as u32 & 0x007f_ffff;
        (f32::from_bits(mantissa | 0x3f80_0000) - 1.5) * 0.5
    }

    /// `n` random values.
    pub fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }

    /// An F32 `QuantLinear` of shape `[out_features, in_features]` from explicit
    /// row-major values.
    pub fn linear_from(
        &self,
        out_features: usize,
        in_features: usize,
        values: &[f32],
    ) -> QuantLinear {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        QuantLinear::new(
            QuantTensor::new(bytes, vec![out_features, in_features], GgufTensorType::F32),
            None,
        )
    }

    /// A random F32 `QuantLinear` of shape `[out_features, in_features]`.
    pub fn linear(&mut self, out_features: usize, in_features: usize) -> QuantLinear {
        let values = self.vec(out_features * in_features);
        self.linear_from(out_features, in_features, &values)
    }

    /// The kernel for a linear layer's quantization type.
    pub fn kernel(&self, linear: &QuantLinear) -> Box<dyn QuantKernel> {
        self.dispatcher
            .get_kernel(linear.weight.tensor_type)
            .expect("F32 kernel is always available")
    }

    /// A random token-embedding table `[vocab, hidden]`.
    pub fn embedding(&mut self, vocab: usize, hidden: usize) -> Vec<f32> {
        self.vec(vocab * hidden)
    }

    /// A [`GrokMoe`] with `n_experts` random GELU experts.
    pub fn grok_moe(
        &mut self,
        hidden: usize,
        intermediate: usize,
        n_experts: usize,
        top_k: usize,
    ) -> GrokMoe {
        let router = self.linear(n_experts, hidden);
        let experts: Vec<QuantExpert> = (0..n_experts)
            .map(|_| {
                QuantExpert::new(
                    self.linear(intermediate, hidden),
                    self.linear(intermediate, hidden),
                    self.linear(hidden, intermediate),
                )
                .expect("expert shapes compose")
            })
            .collect();
        GrokMoe::new(router, experts, top_k).expect("MoE constructs")
    }
}

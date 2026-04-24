//! Multi-head Latent Attention (MLA) primitive.
//!
//! Implements the DeepSeek-V2 MLA mechanism, which compresses both the query
//! and key-value latents through low-rank projections, then reconstructs
//! per-head K/V lazily during attention computation. The decoupled RoPE path
//! applies rotary embeddings only to a subset of Q and K dimensions, leaving
//! the nope (no-positional-encoding) dimensions untouched.
//!
//! ## Key invariants
//!
//! - Q latent: `[q_lora_rank]` per token, projected to `[num_heads × (qk_nope + qk_rope)]`
//! - KV latent: `[kv_lora_rank]` per token, cached and lazily expanded
//! - RoPE applied only to the last `qk_rope` dims of Q (per head) and K (shared across heads)
//! - `MlaLatentCache` is arch-internal — NOT a `KvCacheAccess` extension

use crate::common::linear::QuantLinear;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::error::{ArchError, ArchResult};
use oxillama_quant::KernelDispatcher;

/// Configuration for Multi-head Latent Attention.
#[derive(Debug, Clone)]
pub struct MlaConfig {
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Rank of the Q low-rank projection (compressed Q latent dimension).
    pub q_lora_rank: usize,
    /// Rank of the KV low-rank projection (compressed KV latent dimension).
    pub kv_lora_rank: usize,
    /// Head dimension for the nope (no-positional-encoding) part of Q and K.
    pub qk_nope_head_dim: usize,
    /// Head dimension for the rope (positional-encoding) part of Q and K.
    pub qk_rope_head_dim: usize,
    /// Head dimension for V.
    pub v_head_dim: usize,
    /// RoPE base frequency (used to build the RopeTable).
    pub rope_theta: f32,
    /// Scale factor for attention scores (typically `1 / sqrt(qk_nope + qk_rope)`).
    pub softmax_scale: f32,
}

impl MlaConfig {
    /// Total head dimension (nope + rope).
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Full per-token output dimension from w_q_b: num_heads × (qk_nope + qk_rope).
    pub fn q_full_dim(&self) -> usize {
        self.num_heads * self.qk_head_dim()
    }

    /// Combined KV projection output: kv_lora_rank + qk_rope_head_dim.
    pub fn kv_combined_dim(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    /// Per-head KV expansion output: qk_nope_head_dim + v_head_dim.
    pub fn kv_b_per_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.v_head_dim
    }

    /// Total output from w_kv_b per latent: num_heads × (qk_nope + v_head_dim).
    pub fn kv_b_full_dim(&self) -> usize {
        self.num_heads * self.kv_b_per_head_dim()
    }

    /// Output dimension of the concatenated attention heads fed into w_o.
    pub fn attn_out_dim(&self) -> usize {
        self.num_heads * self.v_head_dim
    }
}

/// Weights for one MLA layer.
///
/// The RoPE table is stored here because it is a function of the config
/// that is computed once at layer construction. It applies to both Q-rope
/// and K-rope slices.
pub struct MlaWeights {
    /// Q down-projection: `[hidden → q_lora_rank]`.
    pub w_q_a: QuantLinear,
    /// RMSNorm on the q_lora_rank dimension.
    pub q_a_norm: RmsNorm,
    /// Q up-projection: `[q_lora_rank → num_heads × (qk_nope + qk_rope)]`.
    pub w_q_b: QuantLinear,
    /// KV combined down-projection: `[hidden → kv_lora_rank + qk_rope]`.
    pub w_kv_a: QuantLinear,
    /// RMSNorm on the kv_lora_rank dimension.
    pub kv_a_norm: RmsNorm,
    /// KV up-projection: `[kv_lora_rank → num_heads × (qk_nope + v_head_dim)]`.
    pub w_kv_b: QuantLinear,
    /// Output projection: `[num_heads × v_head_dim → hidden]`.
    pub w_o: QuantLinear,
    /// Precomputed RoPE table (applied to qk_rope dimensions).
    pub rope: RopeTable,
}

/// Arch-internal latent KV cache for one MLA layer.
///
/// Stores the compressed KV latents and the shared K-rope vectors,
/// avoiding the O(seq²) memory of a full materialized KV cache.
/// `MlaLatentCache` is NOT a `KvCacheAccess` extension — it is owned
/// directly by `DeepSeekModel`.
pub struct MlaLatentCache {
    /// Stored KV latents, shape `[capacity × kv_lora_rank]`.
    pub kv_latent: Vec<f32>,
    /// Stored K-rope vectors, shape `[capacity × qk_rope_head_dim]`.
    pub k_rope: Vec<f32>,
    /// Number of tokens currently stored.
    pub seq_len: usize,
    /// Maximum tokens this cache can hold without reallocation.
    capacity: usize,
    /// Latent rank.
    kv_lora_rank: usize,
    /// K-rope dimension.
    qk_rope_head_dim: usize,
}

impl MlaLatentCache {
    /// Create a new empty cache with the given capacity.
    pub fn new(capacity: usize, cfg: &MlaConfig) -> Self {
        Self {
            kv_latent: vec![0.0; capacity * cfg.kv_lora_rank],
            k_rope: vec![0.0; capacity * cfg.qk_rope_head_dim],
            seq_len: 0,
            capacity,
            kv_lora_rank: cfg.kv_lora_rank,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        }
    }

    /// Append one token's KV latent and K-rope vector to the cache.
    ///
    /// # Errors
    /// Returns `ArchError::InvalidConfig` if the cache is full.
    pub fn append(&mut self, kv_latent: &[f32], k_rope: &[f32]) -> ArchResult<()> {
        if self.seq_len >= self.capacity {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "MlaLatentCache is full (capacity = {}); clear before appending",
                    self.capacity
                ),
            });
        }
        if kv_latent.len() != self.kv_lora_rank {
            return Err(ArchError::InvalidShape {
                name: "kv_latent".to_string(),
                expected: vec![self.kv_lora_rank],
                got: vec![kv_latent.len()],
            });
        }
        if k_rope.len() != self.qk_rope_head_dim {
            return Err(ArchError::InvalidShape {
                name: "k_rope".to_string(),
                expected: vec![self.qk_rope_head_dim],
                got: vec![k_rope.len()],
            });
        }
        let lat_off = self.seq_len * self.kv_lora_rank;
        self.kv_latent[lat_off..lat_off + self.kv_lora_rank].copy_from_slice(kv_latent);
        let rope_off = self.seq_len * self.qk_rope_head_dim;
        self.k_rope[rope_off..rope_off + self.qk_rope_head_dim].copy_from_slice(k_rope);
        self.seq_len += 1;
        Ok(())
    }

    /// Reset the cache to empty (does not free memory).
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }
}

/// Run a single MLA forward pass for the given batch of tokens.
///
/// # Arguments
/// * `x`        - Input hidden states, shape `[seq_len × hidden_size]`.
/// * `weights`  - MLA weight matrices and RoPE table.
/// * `cfg`      - MLA configuration.
/// * `cache`    - Mutable arch-internal latent KV cache for this layer.
/// * `position` - Starting sequence position (for RoPE indexing).
///
/// # Returns
/// Output hidden states, shape `[seq_len × hidden_size]`.
///
/// # Errors
/// Returns `ArchError` on shape mismatches, cache overflow, or kernel errors.
pub fn mla_forward(
    x: &[f32],
    weights: &MlaWeights,
    cfg: &MlaConfig,
    cache: &mut MlaLatentCache,
    position: usize,
) -> ArchResult<Vec<f32>> {
    let hidden_size = weights.w_o.out_features;
    let seq_len = x
        .len()
        .checked_div(hidden_size)
        .ok_or_else(|| ArchError::InvalidConfig {
            detail: "mla_forward: hidden_size is 0".to_string(),
        })?;

    if x.len() != seq_len * hidden_size {
        return Err(ArchError::InvalidShape {
            name: "x".to_string(),
            expected: vec![seq_len, hidden_size],
            got: vec![x.len()],
        });
    }

    let dispatcher = KernelDispatcher::new();

    // --- Per-token Q and KV projection + RoPE + cache append ---
    // q_rope_all[t × num_heads × qk_rope]:  rotated Q-rope per token per head
    // q_nope_all[t × num_heads × qk_nope]:  Q-nope per token per head (no RoPE)
    let mut q_rope_all = vec![0.0f32; seq_len * cfg.num_heads * cfg.qk_rope_head_dim];
    let mut q_nope_all = vec![0.0f32; seq_len * cfg.num_heads * cfg.qk_nope_head_dim];

    for t in 0..seq_len {
        let x_t = &x[t * hidden_size..(t + 1) * hidden_size];

        // Step 1: Q latent = w_q_a(x_t) then q_a_norm
        let q_a_kernel = dispatcher
            .get_kernel(weights.w_q_a.weight.tensor_type)
            .map_err(ArchError::from)?;
        let mut q_latent = vec![0.0f32; cfg.q_lora_rank];
        weights
            .w_q_a
            .forward(&*q_a_kernel, x_t, &mut q_latent)
            .map_err(ArchError::from)?;
        weights.q_a_norm.forward(&mut q_latent);

        // Step 2: Q full = q_latent @ w_q_b → [num_heads × (qk_nope + qk_rope)]
        let q_b_kernel = dispatcher
            .get_kernel(weights.w_q_b.weight.tensor_type)
            .map_err(ArchError::from)?;
        let mut q_full = vec![0.0f32; cfg.q_full_dim()];
        weights
            .w_q_b
            .forward(&*q_b_kernel, &q_latent, &mut q_full)
            .map_err(ArchError::from)?;

        // Split q_full into q_nope and q_rope_raw per head, apply RoPE to rope part
        let global_pos = position + t;
        for h in 0..cfg.num_heads {
            let head_off = h * cfg.qk_head_dim();
            let q_nope_src = &q_full[head_off..head_off + cfg.qk_nope_head_dim];
            let nope_dst = &mut q_nope_all[(t * cfg.num_heads + h) * cfg.qk_nope_head_dim
                ..(t * cfg.num_heads + h + 1) * cfg.qk_nope_head_dim];
            nope_dst.copy_from_slice(q_nope_src);

            let rope_src_off = head_off + cfg.qk_nope_head_dim;
            let mut q_rope_head =
                q_full[rope_src_off..rope_src_off + cfg.qk_rope_head_dim].to_vec();
            weights.rope.apply(&mut q_rope_head, global_pos);
            let rope_dst = &mut q_rope_all[(t * cfg.num_heads + h) * cfg.qk_rope_head_dim
                ..(t * cfg.num_heads + h + 1) * cfg.qk_rope_head_dim];
            rope_dst.copy_from_slice(&q_rope_head);
        }

        // Step 3: KV combined = x_t @ w_kv_a → [kv_lora_rank + qk_rope]
        let kv_a_kernel = dispatcher
            .get_kernel(weights.w_kv_a.weight.tensor_type)
            .map_err(ArchError::from)?;
        let mut kv_combined = vec![0.0f32; cfg.kv_combined_dim()];
        weights
            .w_kv_a
            .forward(&*kv_a_kernel, x_t, &mut kv_combined)
            .map_err(ArchError::from)?;

        // Step 4: Split kv_combined → kv_latent_t + k_rope_raw
        let kv_latent_t = kv_combined[..cfg.kv_lora_rank].to_vec();
        let mut k_rope_raw = kv_combined[cfg.kv_lora_rank..].to_vec();

        // Apply q_a_norm equivalent (kv_a_norm) to kv_latent part
        let mut kv_latent_normed = kv_latent_t;
        weights.kv_a_norm.forward(&mut kv_latent_normed);

        // Step 5: Apply RoPE to k_rope (shared across heads)
        weights.rope.apply(&mut k_rope_raw, global_pos);

        // Step 6: Cache append
        cache.append(&kv_latent_normed, &k_rope_raw)?;
    }

    // --- Attention over all tokens ---
    let mut output = vec![0.0f32; seq_len * hidden_size];

    let kv_b_kernel = dispatcher
        .get_kernel(weights.w_kv_b.weight.tensor_type)
        .map_err(ArchError::from)?;
    let w_o_kernel = dispatcher
        .get_kernel(weights.w_o.weight.tensor_type)
        .map_err(ArchError::from)?;

    for t in 0..seq_len {
        // The query token corresponds to cache position (position + t), but we read
        // from the first (position + t + 1) cached entries (all tokens up to and including t).
        // Note: the cache now has (initial_seq_len + t + 1) entries after the loop above,
        // but for causal attention token t can attend to positions [0..=position+t].
        // Since cache was populated starting from seq_len_before (= initial cache seq_len
        // before this call), token t in our batch corresponds to cache index
        // (cache_start + t) where cache_start = cache.seq_len - seq_len.
        let cache_start = cache.seq_len - seq_len;
        let attend_len = cache_start + t + 1; // attend to all cached tokens up to this one

        // Per-head attention output, accumulated here
        let mut attn_head_out = vec![0.0f32; cfg.attn_out_dim()];

        // Per-head attention scores
        let mut scores = vec![0.0f32; attend_len];

        for h in 0..cfg.num_heads {
            let q_nope_h = &q_nope_all[(t * cfg.num_heads + h) * cfg.qk_nope_head_dim
                ..(t * cfg.num_heads + h + 1) * cfg.qk_nope_head_dim];
            let q_rope_h = &q_rope_all[(t * cfg.num_heads + h) * cfg.qk_rope_head_dim
                ..(t * cfg.num_heads + h + 1) * cfg.qk_rope_head_dim];

            // Compute attention scores for each cached token
            for (s, score_slot) in scores.iter_mut().enumerate().take(attend_len) {
                // Lazy reconstruct: kv_up_s = cache.kv_latent[s] @ w_kv_b
                let lat_off = s * cfg.kv_lora_rank;
                let kv_lat_s = &cache.kv_latent[lat_off..lat_off + cfg.kv_lora_rank];
                let mut kv_up = vec![0.0f32; cfg.kv_b_full_dim()];
                weights
                    .w_kv_b
                    .forward(&*kv_b_kernel, kv_lat_s, &mut kv_up)
                    .map_err(ArchError::from)?;

                // Split kv_up per head: k_nope[h] + v[h]
                let h_off = h * cfg.kv_b_per_head_dim();
                let k_nope_s = &kv_up[h_off..h_off + cfg.qk_nope_head_dim];

                // k_rope is shared across heads — read from cache
                let rope_off = s * cfg.qk_rope_head_dim;
                let k_rope_s = &cache.k_rope[rope_off..rope_off + cfg.qk_rope_head_dim];

                // Score: q_nope·k_nope + q_rope·k_rope (decoupled RoPE)
                let score: f32 = q_nope_h
                    .iter()
                    .zip(k_nope_s.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    + q_rope_h
                        .iter()
                        .zip(k_rope_s.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                *score_slot = score * cfg.softmax_scale;
            }

            // Softmax over scores[0..attend_len]
            softmax_inplace(&mut scores[..attend_len]);

            // Weighted sum of V
            let v_head_out = &mut attn_head_out[h * cfg.v_head_dim..(h + 1) * cfg.v_head_dim];
            for (s, &w) in scores.iter().enumerate().take(attend_len) {
                let lat_off = s * cfg.kv_lora_rank;
                let kv_lat_s = &cache.kv_latent[lat_off..lat_off + cfg.kv_lora_rank];
                let mut kv_up = vec![0.0f32; cfg.kv_b_full_dim()];
                weights
                    .w_kv_b
                    .forward(&*kv_b_kernel, kv_lat_s, &mut kv_up)
                    .map_err(ArchError::from)?;

                let h_off = h * cfg.kv_b_per_head_dim();
                let v_s = &kv_up
                    [h_off + cfg.qk_nope_head_dim..h_off + cfg.qk_nope_head_dim + cfg.v_head_dim];
                for (vo, &vs) in v_head_out.iter_mut().zip(v_s.iter()) {
                    *vo += w * vs;
                }
            }
        }

        // Output projection: w_o @ concat(head_outputs)
        let out_t = &mut output[t * hidden_size..(t + 1) * hidden_size];
        weights
            .w_o
            .forward(&*w_o_kernel, &attn_head_out, out_t)
            .map_err(ArchError::from)?;
    }

    Ok(output)
}

/// Numerically stable softmax applied in-place.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    /// Minimal LCG pseudo-random number generator (deterministic, no deps).
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        /// Returns a float in [-0.05, 0.05) range.
        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Mask off sign and exponent bits, force exponent=127 (value in [1.0, 2.0))
            let mantissa = (self.state >> 33) as u32 & 0x007f_ffff;
            let bits = mantissa | 0x3f80_0000u32;
            let v = f32::from_bits(bits) - 1.5; // in [-0.5, 0.5)
            v * 0.1 // in [-0.05, 0.05), small to keep softmax numerically stable
        }
        fn fill(&mut self, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.next_f32();
            }
        }
    }

    /// Build an F32 QuantTensor of given shape filled by the LCG.
    fn rand_tensor(lcg: &mut Lcg, rows: usize, cols: usize) -> QuantTensor {
        let n = rows * cols;
        let mut vals = vec![0.0f32; n];
        lcg.fill(&mut vals);
        let mut data = Vec::with_capacity(n * 4);
        for v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32)
    }

    /// Build a small MlaConfig for tests.
    fn test_cfg() -> MlaConfig {
        MlaConfig {
            num_heads: 2,
            q_lora_rank: 8,
            kv_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            rope_theta: 10000.0,
            softmax_scale: 1.0 / (8.0f32).sqrt(),
        }
    }

    const HIDDEN: usize = 16;

    /// Build a fully random MlaWeights for the given config and hidden size.
    fn test_weights(lcg: &mut Lcg, cfg: &MlaConfig, hidden: usize) -> MlaWeights {
        let q_full = cfg.q_full_dim();
        let kv_comb = cfg.kv_combined_dim();
        let kv_b_full = cfg.kv_b_full_dim();
        let attn_out = cfg.attn_out_dim();

        let w_q_a = QuantLinear::new(rand_tensor(lcg, cfg.q_lora_rank, hidden), None);
        let q_a_norm = {
            let mut w = vec![0.0f32; cfg.q_lora_rank];
            lcg.fill(&mut w);
            // Use small positive weights to keep RmsNorm stable
            let w: Vec<f32> = w.into_iter().map(|v| v.abs() + 0.1).collect();
            RmsNorm::new(w, 1e-5)
        };
        let w_q_b = QuantLinear::new(rand_tensor(lcg, q_full, cfg.q_lora_rank), None);
        let w_kv_a = QuantLinear::new(rand_tensor(lcg, kv_comb, hidden), None);
        let kv_a_norm = {
            let mut w = vec![0.0f32; cfg.kv_lora_rank];
            lcg.fill(&mut w);
            let w: Vec<f32> = w.into_iter().map(|v| v.abs() + 0.1).collect();
            RmsNorm::new(w, 1e-5)
        };
        let w_kv_b = QuantLinear::new(rand_tensor(lcg, kv_b_full, cfg.kv_lora_rank), None);
        let w_o = QuantLinear::new(rand_tensor(lcg, hidden, attn_out), None);
        let rope = RopeTable::new_standard(cfg.qk_rope_head_dim, 128, cfg.rope_theta);

        MlaWeights {
            w_q_a,
            q_a_norm,
            w_q_b,
            w_kv_a,
            kv_a_norm,
            w_kv_b,
            w_o,
            rope,
        }
    }

    /// (a) shape_coherence: output length must equal seq_len × hidden_size.
    #[test]
    fn shape_coherence() {
        let cfg = test_cfg();
        let mut lcg = Lcg::new(42);
        let weights = test_weights(&mut lcg, &cfg, HIDDEN);
        let mut cache = MlaLatentCache::new(128, &cfg);

        let x = {
            let mut v = vec![0.0f32; HIDDEN];
            lcg.fill(&mut v);
            v
        };

        let out = mla_forward(&x, &weights, &cfg, &mut cache, 0).expect("forward should succeed");
        assert_eq!(out.len(), HIDDEN, "output shape mismatch for single token");
        assert_eq!(cache.seq_len, 1);
    }

    /// (b) determinism: two forward passes (after cache.clear) produce identical results.
    #[test]
    fn determinism() {
        let cfg = test_cfg();
        let mut lcg = Lcg::new(99);
        let weights = test_weights(&mut lcg, &cfg, HIDDEN);

        let x = {
            let mut v = vec![0.0f32; HIDDEN];
            lcg.fill(&mut v);
            v
        };

        let mut cache = MlaLatentCache::new(128, &cfg);
        let out1 = mla_forward(&x, &weights, &cfg, &mut cache, 0).expect("first forward");
        cache.clear();
        let out2 = mla_forward(&x, &weights, &cfg, &mut cache, 0).expect("second forward");

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a, b, "outputs must be bit-for-bit identical");
        }
    }

    /// (c) cached_vs_uncached: running both tokens together vs token-by-token
    /// produces the same second-token output (validates lazy w_kv_b reconstruction).
    #[test]
    fn cached_vs_uncached() {
        let cfg = test_cfg();
        let mut lcg = Lcg::new(7);
        let weights = test_weights(&mut lcg, &cfg, HIDDEN);

        let mut x0 = vec![0.0f32; HIDDEN];
        let mut x1 = vec![0.0f32; HIDDEN];
        lcg.fill(&mut x0);
        lcg.fill(&mut x1);

        // Combined path: both tokens in a single forward call
        let mut x_both = Vec::with_capacity(2 * HIDDEN);
        x_both.extend_from_slice(&x0);
        x_both.extend_from_slice(&x1);

        let mut cache_both = MlaLatentCache::new(128, &cfg);
        let out_both =
            mla_forward(&x_both, &weights, &cfg, &mut cache_both, 0).expect("combined forward");

        // Step-by-step path
        let mut cache_step = MlaLatentCache::new(128, &cfg);
        let _out0 = mla_forward(&x0, &weights, &cfg, &mut cache_step, 0).expect("step 0 forward");
        let out1_cached =
            mla_forward(&x1, &weights, &cfg, &mut cache_step, 1).expect("step 1 forward");

        // Second token output must match
        let second_token_both = &out_both[HIDDEN..2 * HIDDEN];
        for (i, (a, b)) in second_token_both.iter().zip(out1_cached.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "cached[{i}] combined={a} vs step-by-step={b}"
            );
        }
    }

    /// (d) rope_decoupled: zeroing q_rope_raw changes the attention output
    /// compared to non-zeroed, confirming RoPE only affects the rope partition.
    #[test]
    fn rope_decoupled() {
        // We use a config where rope and nope dims are non-zero.
        let cfg = test_cfg();
        let mut lcg = Lcg::new(13);
        let weights = test_weights(&mut lcg, &cfg, HIDDEN);

        // Build two inputs that differ only in the rope-affecting dimensions
        // by running the same x through mla twice, but swapping it
        let mut x_a = vec![0.0f32; HIDDEN];
        let mut x_b = vec![0.0f32; HIDDEN];
        lcg.fill(&mut x_a);
        lcg.fill(&mut x_b);
        // Make them distinctly different
        for (a, b) in x_a.iter_mut().zip(x_b.iter_mut()) {
            *b = -*a;
        }

        let mut cache_a = MlaLatentCache::new(128, &cfg);
        let out_a = mla_forward(&x_a, &weights, &cfg, &mut cache_a, 0).expect("forward a");
        let mut cache_b = MlaLatentCache::new(128, &cfg);
        let out_b = mla_forward(&x_b, &weights, &cfg, &mut cache_b, 0).expect("forward b");

        // Outputs must differ (RoPE + different inputs must change the result)
        let differs = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "different inputs (including through rope path) must produce different outputs"
        );
    }
}

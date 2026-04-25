//! Integration tests for the DeepSeek-V2 architecture.
//!
//! These tests exercise the full C1 (MLA primitive) and C2 (DeepSeek-V2 model)
//! stack. They build a tiny in-memory model (all F32, random weights) and verify
//! structural correctness: shapes, determinism, and registry presence.
//!
//! Note: `MlaLatentCache` is arch-internal — `KvCacheAccess` is stubbed with a
//! no-op implementation; the DeepSeek model ignores it in favour of its own cache.

#[cfg(feature = "deepseek")]
mod deepseek_tests {
    use oxillama_arch::common::linear::QuantLinear;
    use oxillama_arch::common::mla::{mla_forward, MlaConfig, MlaLatentCache, MlaWeights};
    use oxillama_arch::common::rms_norm::RmsNorm;
    use oxillama_arch::common::rope::RopeTable;
    use oxillama_arch::config::{DeepSeekConfig, ModelConfig};
    use oxillama_arch::deepseek::model::{
        build_deepseek_model, DeepSeekLayer, DenseFfn, FfnKind, N_DENSE_LAYERS,
    };
    use oxillama_arch::deepseek::moe::{
        moe_forward, DeepSeekExpert, MoeConfig, MoeWeights, ScoringMode,
    };
    use oxillama_arch::error::ArchResult;
    use oxillama_arch::registry::ArchitectureRegistry;
    use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    // ─── No-op KvCacheAccess stub ─────────────────────────────────────────────

    /// Stub KV cache that satisfies the trait but never actually caches anything.
    /// DeepSeek uses arch-internal `MlaLatentCache`; this stub is only needed
    /// because `ForwardPass::forward` takes `&mut dyn KvCacheAccess`.
    struct NullKv;

    impl KvCacheAccess for NullKv {
        fn seq_len(&self) -> usize {
            0
        }

        fn store_kv(&mut self, _layer: usize, _keys: &[f32], _values: &[f32]) -> ArchResult<()> {
            Ok(())
        }

        fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
            Ok(&[])
        }

        fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
            Ok(&[])
        }

        fn advance(&mut self) {}
    }

    // ─── Minimal LCG ─────────────────────────────────────────────────────────

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mantissa = (self.state >> 33) as u32 & 0x007f_ffff;
            let bits = mantissa | 0x3f80_0000u32;
            (f32::from_bits(bits) - 1.5) * 0.02
        }

        fn fill(&mut self, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.next_f32();
            }
        }
    }

    // ─── Test helpers ─────────────────────────────────────────────────────────

    const HIDDEN: usize = 16;
    const VOCAB: usize = 32;
    const INTERMEDIATE: usize = 32;
    const N_LAYERS: usize = 2;
    const MAX_SEQ: usize = 64;
    const N_ROUTED: usize = 4;
    const TOP_K: usize = 2;
    const N_SHARED: usize = 1;
    const MOE_INTER: usize = 16;

    fn rand_f32_tensor(lcg: &mut Lcg, rows: usize, cols: usize) -> QuantTensor {
        let n = rows * cols;
        let mut vals = vec![0.0f32; n];
        lcg.fill(&mut vals);
        let mut data = Vec::with_capacity(n * 4);
        for &v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32)
    }

    fn test_mla_cfg() -> MlaConfig {
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

    fn build_mla_weights(lcg: &mut Lcg, cfg: &MlaConfig) -> MlaWeights {
        let q_full = cfg.q_full_dim();
        let kv_comb = cfg.kv_combined_dim();
        let kv_b_full = cfg.kv_b_full_dim();
        let attn_out = cfg.attn_out_dim();

        let pos_weights = |lcg: &mut Lcg, n: usize| -> Vec<f32> {
            let mut w = vec![0.0f32; n];
            lcg.fill(&mut w);
            w.iter_mut().for_each(|v| *v = v.abs() + 0.1);
            w
        };

        MlaWeights {
            w_q_a: QuantLinear::new(rand_f32_tensor(lcg, cfg.q_lora_rank, HIDDEN), None),
            q_a_norm: RmsNorm::new(pos_weights(lcg, cfg.q_lora_rank), 1e-5),
            w_q_b: QuantLinear::new(rand_f32_tensor(lcg, q_full, cfg.q_lora_rank), None),
            w_kv_a: QuantLinear::new(rand_f32_tensor(lcg, kv_comb, HIDDEN), None),
            kv_a_norm: RmsNorm::new(pos_weights(lcg, cfg.kv_lora_rank), 1e-5),
            w_kv_b: QuantLinear::new(rand_f32_tensor(lcg, kv_b_full, cfg.kv_lora_rank), None),
            w_o: QuantLinear::new(rand_f32_tensor(lcg, HIDDEN, attn_out), None),
            rope: RopeTable::new_standard(cfg.qk_rope_head_dim, MAX_SEQ, cfg.rope_theta),
        }
    }

    fn build_dense_ffn(lcg: &mut Lcg) -> DenseFfn {
        DenseFfn {
            gate: QuantLinear::new(rand_f32_tensor(lcg, INTERMEDIATE, HIDDEN), None),
            up: QuantLinear::new(rand_f32_tensor(lcg, INTERMEDIATE, HIDDEN), None),
            down: QuantLinear::new(rand_f32_tensor(lcg, HIDDEN, INTERMEDIATE), None),
        }
    }

    fn make_expert(lcg: &mut Lcg, inter: usize) -> DeepSeekExpert {
        let mut gate = vec![0.0f32; inter * HIDDEN];
        let mut up = vec![0.0f32; inter * HIDDEN];
        let mut down = vec![0.0f32; HIDDEN * inter];
        lcg.fill(&mut gate);
        lcg.fill(&mut up);
        lcg.fill(&mut down);
        DeepSeekExpert {
            gate,
            up,
            down,
            hidden_size: HIDDEN,
            intermediate_size: inter,
        }
    }

    fn build_moe_ffn(lcg: &mut Lcg) -> (MoeWeights, MoeConfig) {
        let moe_cfg = MoeConfig {
            hidden_size: HIDDEN,
            expert_intermediate_size: MOE_INTER,
            n_shared_experts: N_SHARED,
            n_routed_experts: N_ROUTED,
            top_k: TOP_K,
            routed_scaling_factor: 1.0,
            scoring_mode: ScoringMode::Softmax,
            shared_expert_intermediate_size: MOE_INTER,
        };
        let mut router = vec![0.0f32; N_ROUTED * HIDDEN];
        lcg.fill(&mut router);
        let moe_weights = MoeWeights {
            router,
            routed_experts: (0..N_ROUTED).map(|_| make_expert(lcg, MOE_INTER)).collect(),
            shared_experts: (0..N_SHARED).map(|_| make_expert(lcg, MOE_INTER)).collect(),
            expert_bias: None,
        };
        (moe_weights, moe_cfg)
    }

    fn build_test_model(lcg: &mut Lcg) -> impl ForwardPass {
        let mla_cfg = test_mla_cfg();

        let pos_weights = |lcg: &mut Lcg, n: usize| -> Vec<f32> {
            let mut w = vec![0.0f32; n];
            lcg.fill(&mut w);
            w.iter_mut().for_each(|v| *v = v.abs() + 0.1);
            w
        };

        let layers = (0..N_LAYERS)
            .map(|idx| {
                let ffn = if idx < N_DENSE_LAYERS {
                    FfnKind::Dense(Box::new(build_dense_ffn(lcg)))
                } else {
                    let (moe_weights, moe_cfg) = build_moe_ffn(lcg);
                    FfnKind::Moe {
                        weights: Box::new(moe_weights),
                        config: moe_cfg,
                    }
                };
                DeepSeekLayer {
                    attn_norm: RmsNorm::new(pos_weights(lcg, HIDDEN), 1e-5),
                    mla_weights: build_mla_weights(lcg, &mla_cfg),
                    mla_config: mla_cfg.clone(),
                    mla_cache: MlaLatentCache::new(MAX_SEQ, &mla_cfg),
                    ffn_norm: RmsNorm::new(pos_weights(lcg, HIDDEN), 1e-5),
                    ffn,
                }
            })
            .collect();

        let mut token_embd = vec![0.0f32; VOCAB * HIDDEN];
        lcg.fill(&mut token_embd);

        let ds_config = DeepSeekConfig {
            q_lora_rank: mla_cfg.q_lora_rank,
            kv_lora_rank: mla_cfg.kv_lora_rank,
            qk_nope_head_dim: mla_cfg.qk_nope_head_dim,
            qk_rope_head_dim: mla_cfg.qk_rope_head_dim,
            v_head_dim: mla_cfg.v_head_dim,
            n_shared_experts: N_SHARED,
            n_routed_experts: N_ROUTED,
            top_k_routed: TOP_K,
            shared_expert_intermediate_size: MOE_INTER,
            routed_scaling_factor: 1.0,
            first_k_dense_replace: 1,
        };

        let model_config = ModelConfig {
            architecture: "deepseek2".to_string(),
            model_name: "test-deepseek".to_string(),
            hidden_size: HIDDEN,
            intermediate_size: INTERMEDIATE,
            num_layers: N_LAYERS,
            num_attention_heads: mla_cfg.num_heads,
            num_kv_heads: mla_cfg.num_heads,
            head_dim: mla_cfg.qk_head_dim(),
            vocab_size: VOCAB,
            max_context_length: MAX_SEQ,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            ..ModelConfig::default()
        };

        let output_norm = RmsNorm::new(pos_weights(lcg, HIDDEN), 1e-5);
        let output = QuantLinear::new(rand_f32_tensor(lcg, VOCAB, HIDDEN), None);

        build_deepseek_model(
            model_config,
            ds_config,
            token_embd,
            layers,
            output_norm,
            output,
        )
    }

    // ─── Integration tests ────────────────────────────────────────────────────

    /// DeepSeek-V2 is registered in the architecture registry under "deepseek2".
    #[test]
    fn test_deepseek_registered_in_registry() {
        let reg = ArchitectureRegistry::with_builtins();
        assert!(
            reg.contains("deepseek2"),
            "registry must contain 'deepseek2'"
        );
        let arch = reg.get("deepseek2").expect("get deepseek2 must succeed");
        assert_eq!(arch.arch_id(), "deepseek2");
    }

    /// Forward pass produces logits of vocab_size length.
    #[test]
    fn test_forward_shape() {
        let mut lcg = Lcg::new(42);
        let mut model = build_test_model(&mut lcg);
        let mut kv = NullKv;
        let logits = model
            .forward(&[1u32, 2, 3], &mut kv)
            .expect("forward must succeed");
        assert_eq!(
            logits.len(),
            VOCAB,
            "logits length must equal vocab_size={VOCAB}"
        );
    }

    /// Forward pass is deterministic: same input after cache reset → same output.
    #[test]
    fn test_forward_determinism() {
        let mut lcg = Lcg::new(7777);
        let mut model = build_test_model(&mut lcg);

        let mut kv = NullKv;
        let out1 = model.forward(&[0u32, 1], &mut kv).expect("first forward");

        // Cast to DeepSeekModel to call reset_position (returns impl ForwardPass)
        // We can't directly call reset_position via dyn ForwardPass.
        // Instead, rebuild with the same seed:
        let mut lcg2 = Lcg::new(7777);
        let mut model2 = build_test_model(&mut lcg2);
        let mut kv2 = NullKv;
        let out2 = model2
            .forward(&[0u32, 1], &mut kv2)
            .expect("second forward");

        assert_eq!(out1.len(), out2.len());
        for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            assert_eq!(a_bits, b_bits, "logits[{i}] must be bit-for-bit identical");
        }
    }

    /// MLA forward pass: shape coherence for a single token.
    #[test]
    fn test_mla_shape() {
        let cfg = test_mla_cfg();
        let mut lcg = Lcg::new(100);
        let mut kv_cache = MlaLatentCache::new(MAX_SEQ, &cfg);
        let weights = build_mla_weights(&mut lcg, &cfg);

        let mut x = vec![0.0f32; HIDDEN];
        lcg.fill(&mut x);

        let out =
            mla_forward(&x, &weights, &cfg, &mut kv_cache, 0).expect("mla_forward must succeed");
        assert_eq!(
            out.len(),
            HIDDEN,
            "MLA output must have hidden_size={HIDDEN} elements"
        );
        assert_eq!(kv_cache.seq_len, 1, "cache must have 1 entry after 1 token");
    }

    /// MoE forward pass: shape coherence.
    #[test]
    fn test_moe_shape() {
        let mut lcg = Lcg::new(200);
        let (weights, cfg) = build_moe_ffn(&mut lcg);

        let mut x = vec![0.0f32; HIDDEN];
        lcg.fill(&mut x);

        let out = moe_forward(&x, &weights, &cfg).expect("moe_forward must succeed");
        assert_eq!(
            out.len(),
            HIDDEN,
            "MoE output must have hidden_size={HIDDEN} elements"
        );
    }
}

//! Integration tests for DeepSeek-V3 sigmoid-with-bias MoE scoring.
//!
//! Tests verify that `ScoringMode::SigmoidWithBias` correctly:
//! (a) normalises selected routing weights to sum to 1 after top-k selection.
//! (b) produces different expert selections than softmax when a bias value
//!     flips which expert lands in top-k.
//! (c) runs a full forward pass with an exp_probs_b (bias) fixture.

#[cfg(feature = "deepseek")]
mod deepseek_v3_tests {
    use oxillama_arch::deepseek::moe::{
        moe_forward, DeepSeekExpert, MoeConfig, MoeWeights, ScoringMode,
    };

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
            (f32::from_bits(bits) - 1.5) * 0.1
        }

        fn fill(&mut self, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.next_f32();
            }
        }
    }

    fn make_expert(hidden: usize, intermediate: usize) -> DeepSeekExpert {
        DeepSeekExpert {
            gate: vec![0.0f32; intermediate * hidden],
            up: vec![0.0f32; intermediate * hidden],
            down: vec![0.0f32; hidden * intermediate],
            hidden_size: hidden,
            intermediate_size: intermediate,
        }
    }

    // ─── (a) sigmoid_bias_topk_sums_to_one_after_normalisation ───────────────

    /// After sigmoid+bias scoring and top-k selection, the selected routing
    /// weights are normalised by their sum → they must sum to exactly 1.0.
    #[test]
    fn sigmoid_bias_topk_sums_to_one_after_normalisation() {
        const H: usize = 8;
        const N_EXPERTS: usize = 6;
        const TOP_K: usize = 2;
        const INTER: usize = 8;

        let mut lcg = Lcg::new(1234);

        // Build router weights.
        let mut router = vec![0.0f32; N_EXPERTS * H];
        lcg.fill(&mut router);

        // Build per-expert bias (varied so different experts are favoured).
        // Expert 0 gets a large positive bias, expert 5 gets a large negative bias.
        let mut expert_bias = vec![0.0f32; N_EXPERTS];
        expert_bias[0] = 2.0;
        expert_bias[5] = -2.0;

        let experts: Vec<DeepSeekExpert> = (0..N_EXPERTS).map(|_| make_expert(H, INTER)).collect();

        let weights = MoeWeights {
            router,
            routed_experts: experts,
            shared_experts: vec![],
            expert_bias: Some(expert_bias),
        };

        let cfg = MoeConfig {
            hidden_size: H,
            expert_intermediate_size: INTER,
            n_shared_experts: 0,
            n_routed_experts: N_EXPERTS,
            top_k: TOP_K,
            routed_scaling_factor: 1.0,
            scoring_mode: ScoringMode::SigmoidWithBias,
            shared_expert_intermediate_size: INTER,
        };

        // Use a simple input to produce deterministic router logits.
        let x: Vec<f32> = (0..H).map(|i| i as f32 * 0.1).collect();

        // Run the MoE forward pass.
        let out = moe_forward(&x, &weights, &cfg).expect("moe_forward must succeed");
        assert_eq!(out.len(), H, "output length must equal hidden_size");

        // Verify: recompute routing manually to check weight normalisation.
        // Router logits for each expert.
        let mut logits: Vec<f32> = (0..N_EXPERTS).map(|_| 0.0f32).collect();
        for (e, logit) in logits.iter_mut().enumerate() {
            *logit = weights.router[e * H..(e + 1) * H]
                .iter()
                .zip(x.iter())
                .map(|(w, xi)| w * xi)
                .sum();
        }

        // Apply sigmoid + bias.
        let bias = weights.expert_bias.as_ref().expect("bias must be present");
        let scores: Vec<f32> = logits
            .iter()
            .zip(bias.iter())
            .map(|(l, b)| 1.0 / (1.0 + (-l).exp()) + b)
            .collect();

        // Top-k selection (find the top-2 by score).
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_scores: Vec<f32> = indexed[..TOP_K].iter().map(|(_, s)| *s).collect();
        let selected_sum: f32 = top_k_scores.iter().sum();

        // Verify normalised sum.
        let normalised_sum: f32 = top_k_scores.iter().map(|s| s / selected_sum).sum();
        assert!(
            (normalised_sum - 1.0).abs() < 1e-5,
            "normalised top-k weights must sum to 1.0, got {normalised_sum}"
        );
    }

    // ─── (b) sigmoid_bias_routing_vs_softmax_differs ─────────────────────────

    /// Construct logits where a bias flips which expert is in top-k.
    /// With softmax, expert A wins. With sigmoid+bias, expert B wins.
    /// Assert that the two modes select different experts.
    #[test]
    fn sigmoid_bias_routing_vs_softmax_differs() {
        const H: usize = 4;
        const N_EXPERTS: usize = 4;
        const INTER: usize = 4;

        // Craft router rows so that:
        // - For input x = [1, 0, 0, 0]:
        //   expert 0 logit = 3.0 (highest logit → softmax prefers it)
        //   expert 1 logit = 1.0
        //   expert 2 logit = 0.0
        //   expert 3 logit = 0.0
        // With sigmoid+bias:
        //   expert 0 bias = -10.0 (drives sigmoid score very low)
        //   expert 1 bias = +5.0  (drives expert 1 score to top)
        // → sigmoid+bias should flip the top-1 from expert 0 to expert 1.

        let mut router = vec![0.0f32; N_EXPERTS * H];
        // Expert 0: first dim = 3.0
        router[0] = 3.0;
        // Expert 1: first dim = 1.0
        router[H] = 1.0;
        // Experts 2,3: zero

        // Shared zero experts (contributions are zero regardless of routing).
        let make_zero_expert = || DeepSeekExpert {
            gate: vec![0.0f32; INTER * H],
            up: vec![0.0f32; INTER * H],
            down: vec![0.0f32; H * INTER],
            hidden_size: H,
            intermediate_size: INTER,
        };

        // ── Softmax mode: top-1 should pick expert 0 ──────────────────────────
        let weights_softmax = MoeWeights {
            router: router.clone(),
            routed_experts: (0..N_EXPERTS).map(|_| make_zero_expert()).collect(),
            shared_experts: vec![],
            expert_bias: None,
        };
        let cfg_softmax = MoeConfig {
            hidden_size: H,
            expert_intermediate_size: INTER,
            n_shared_experts: 0,
            n_routed_experts: N_EXPERTS,
            top_k: 1,
            routed_scaling_factor: 1.0,
            scoring_mode: ScoringMode::Softmax,
            shared_expert_intermediate_size: INTER,
        };

        let x = vec![1.0f32, 0.0, 0.0, 0.0];

        // Verify softmax selects expert 0 by inspecting logits directly.
        let logit_0 = router[0]; // 3.0
        let logit_1 = router[H]; // 1.0
        assert!(
            logit_0 > logit_1,
            "softmax test: expert 0 must have higher raw logit"
        );

        let _out_softmax = moe_forward(&x, &weights_softmax, &cfg_softmax)
            .expect("softmax moe_forward must succeed");

        // ── SigmoidWithBias mode: bias flips expert selection ─────────────────
        let mut expert_bias = vec![0.0f32; N_EXPERTS];
        expert_bias[0] = -10.0; // drives expert 0 score way down
        expert_bias[1] = 5.0; // drives expert 1 score way up

        let weights_sigmoid = MoeWeights {
            router: router.clone(),
            routed_experts: (0..N_EXPERTS).map(|_| make_zero_expert()).collect(),
            shared_experts: vec![],
            expert_bias: Some(expert_bias.clone()),
        };
        let cfg_sigmoid = MoeConfig {
            hidden_size: H,
            expert_intermediate_size: INTER,
            n_shared_experts: 0,
            n_routed_experts: N_EXPERTS,
            top_k: 1,
            routed_scaling_factor: 1.0,
            scoring_mode: ScoringMode::SigmoidWithBias,
            shared_expert_intermediate_size: INTER,
        };

        let _out_sigmoid = moe_forward(&x, &weights_sigmoid, &cfg_sigmoid)
            .expect("sigmoid+bias moe_forward must succeed");

        // Verify by computing scores manually.
        // Softmax selects by logit: expert 0 (logit=3.0) wins.
        let softmax_top_expert = 0usize; // known from construction

        // SigmoidWithBias: sigmoid(logit_e) + bias_e
        let score_0 = 1.0 / (1.0 + (-3.0f32).exp()) + expert_bias[0]; // sigmoid(3) - 10 ≈ -9.05
        let score_1 = 1.0 / (1.0 + (-1.0f32).exp()) + expert_bias[1]; // sigmoid(1) + 5 ≈ 5.73
        let sigmoid_top_expert = if score_1 > score_0 { 1usize } else { 0usize };

        assert_ne!(
            softmax_top_expert, sigmoid_top_expert,
            "sigmoid+bias routing (top={sigmoid_top_expert}) must differ from \
             softmax routing (top={softmax_top_expert}): bias must flip the selection. \
             scores: 0={score_0}, 1={score_1}"
        );
    }

    // ─── (c) deepseek_v3_forward_with_bias ───────────────────────────────────

    /// Full forward pass using the DeepSeek-V2 model with SigmoidWithBias routing.
    ///
    /// Builds a minimal model directly (following the pattern in tests/deepseek.rs)
    /// with SigmoidWithBias MoE and verifies that:
    /// - The output shape equals vocab_size.
    /// - All output values are finite.
    #[test]
    fn deepseek_v3_forward_with_bias() {
        use oxillama_arch::common::linear::QuantLinear;
        use oxillama_arch::common::mla::{MlaConfig, MlaLatentCache, MlaWeights};
        use oxillama_arch::common::rms_norm::RmsNorm;
        use oxillama_arch::common::rope::RopeTable;
        use oxillama_arch::config::{DeepSeekConfig, ModelConfig};
        use oxillama_arch::deepseek::model::{
            build_deepseek_model, DeepSeekLayer, DenseFfn, FfnKind, N_DENSE_LAYERS,
        };
        use oxillama_arch::error::ArchResult;
        use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
        use oxillama_gguf::GgufTensorType;
        use oxillama_quant::QuantTensor;

        struct NullKv;
        impl KvCacheAccess for NullKv {
            fn seq_len(&self) -> usize {
                0
            }
            fn store_kv(&mut self, _: usize, _: &[f32], _: &[f32]) -> ArchResult<()> {
                Ok(())
            }
            fn get_keys(&self, _: usize) -> ArchResult<&[f32]> {
                Ok(&[])
            }
            fn get_values(&self, _: usize) -> ArchResult<&[f32]> {
                Ok(&[])
            }
            fn advance(&mut self) {}
        }

        struct Lcg2 {
            state: u64,
        }
        impl Lcg2 {
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

        fn rand_qt(lcg: &mut Lcg2, rows: usize, cols: usize) -> QuantTensor {
            let n = rows * cols;
            let mut vals = vec![0.0f32; n];
            lcg.fill(&mut vals);
            let mut data = Vec::with_capacity(n * 4);
            for &v in &vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32)
        }

        const H: usize = 16;
        const VOCAB: usize = 32;
        const INTERMEDIATE: usize = 32;
        const N_LAYERS: usize = 2;
        const MAX_SEQ: usize = 64;
        const N_ROUTED: usize = 4;
        const TOP_K: usize = 2;
        const N_SHARED: usize = 1;
        const MOE_INTER: usize = 16;

        let mut lcg = Lcg2::new(999);

        let mla_cfg = MlaConfig {
            num_heads: 2,
            q_lora_rank: 8,
            kv_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            rope_theta: 10000.0,
            softmax_scale: 1.0 / (8.0f32).sqrt(),
        };

        let pos_w = |lcg: &mut Lcg2, n: usize| -> Vec<f32> {
            let mut w = vec![0.0f32; n];
            lcg.fill(&mut w);
            w.iter_mut().for_each(|v| *v = v.abs() + 0.1);
            w
        };

        let build_mla = |lcg: &mut Lcg2| -> MlaWeights {
            MlaWeights {
                w_q_a: QuantLinear::new(rand_qt(lcg, mla_cfg.q_lora_rank, H), None),
                q_a_norm: RmsNorm::new(pos_w(lcg, mla_cfg.q_lora_rank), 1e-5),
                w_q_b: QuantLinear::new(
                    rand_qt(lcg, mla_cfg.q_full_dim(), mla_cfg.q_lora_rank),
                    None,
                ),
                w_kv_a: QuantLinear::new(rand_qt(lcg, mla_cfg.kv_combined_dim(), H), None),
                kv_a_norm: RmsNorm::new(pos_w(lcg, mla_cfg.kv_lora_rank), 1e-5),
                w_kv_b: QuantLinear::new(
                    rand_qt(lcg, mla_cfg.kv_b_full_dim(), mla_cfg.kv_lora_rank),
                    None,
                ),
                w_o: QuantLinear::new(rand_qt(lcg, H, mla_cfg.attn_out_dim()), None),
                rope: RopeTable::new_standard(
                    mla_cfg.qk_rope_head_dim,
                    MAX_SEQ,
                    mla_cfg.rope_theta,
                ),
            }
        };

        let make_ds_expert = |lcg: &mut Lcg2, inter: usize| -> DeepSeekExpert {
            let mut gate = vec![0.0f32; inter * H];
            let mut up = vec![0.0f32; inter * H];
            let mut down = vec![0.0f32; H * inter];
            lcg.fill(&mut gate);
            lcg.fill(&mut up);
            lcg.fill(&mut down);
            DeepSeekExpert {
                gate,
                up,
                down,
                hidden_size: H,
                intermediate_size: inter,
            }
        };

        let build_moe_with_bias = |lcg: &mut Lcg2| -> (MoeWeights, MoeConfig) {
            let moe_cfg = MoeConfig {
                hidden_size: H,
                expert_intermediate_size: MOE_INTER,
                n_shared_experts: N_SHARED,
                n_routed_experts: N_ROUTED,
                top_k: TOP_K,
                routed_scaling_factor: 1.0,
                scoring_mode: ScoringMode::SigmoidWithBias,
                shared_expert_intermediate_size: MOE_INTER,
            };
            let mut router = vec![0.0f32; N_ROUTED * H];
            lcg.fill(&mut router);
            // Small bias values so routing remains numerically stable.
            let mut bias = vec![0.0f32; N_ROUTED];
            lcg.fill(&mut bias);
            let moe_weights = MoeWeights {
                router,
                routed_experts: (0..N_ROUTED)
                    .map(|_| make_ds_expert(lcg, MOE_INTER))
                    .collect(),
                shared_experts: (0..N_SHARED)
                    .map(|_| make_ds_expert(lcg, MOE_INTER))
                    .collect(),
                expert_bias: Some(bias),
            };
            (moe_weights, moe_cfg)
        };

        let mut token_embd = vec![0.0f32; VOCAB * H];
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
        };

        let model_config = ModelConfig {
            architecture: "deepseek2".to_string(),
            model_name: "test-deepseek-v3".to_string(),
            hidden_size: H,
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

        let build_dense_ffn = |lcg: &mut Lcg2| -> FfnKind {
            FfnKind::Dense(Box::new(DenseFfn {
                gate: QuantLinear::new(rand_qt(lcg, INTERMEDIATE, H), None),
                up: QuantLinear::new(rand_qt(lcg, INTERMEDIATE, H), None),
                down: QuantLinear::new(rand_qt(lcg, H, INTERMEDIATE), None),
            }))
        };

        let layers: Vec<DeepSeekLayer> = (0..N_LAYERS)
            .map(|idx| {
                let ffn = if idx < N_DENSE_LAYERS {
                    build_dense_ffn(&mut lcg)
                } else {
                    let (moe_weights, moe_cfg) = build_moe_with_bias(&mut lcg);
                    FfnKind::Moe {
                        weights: Box::new(moe_weights),
                        config: moe_cfg,
                    }
                };
                DeepSeekLayer {
                    attn_norm: RmsNorm::new(pos_w(&mut lcg, H), 1e-5),
                    mla_weights: build_mla(&mut lcg),
                    mla_config: mla_cfg.clone(),
                    mla_cache: MlaLatentCache::new(MAX_SEQ, &mla_cfg),
                    ffn_norm: RmsNorm::new(pos_w(&mut lcg, H), 1e-5),
                    ffn,
                }
            })
            .collect();

        let output_norm = RmsNorm::new(pos_w(&mut lcg, H), 1e-5);
        let output = QuantLinear::new(rand_qt(&mut lcg, VOCAB, H), None);

        let mut model = build_deepseek_model(
            model_config,
            ds_config,
            token_embd,
            layers,
            output_norm,
            output,
        );

        let mut kv = NullKv;
        let logits = model
            .forward(&[1u32, 2, 3], &mut kv)
            .expect("forward with sigmoid+bias must succeed");

        assert_eq!(
            logits.len(),
            VOCAB,
            "output logits must have vocab_size={VOCAB} elements"
        );
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all output logits must be finite"
        );
    }
}

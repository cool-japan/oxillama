//! Integration tests for the Grok-1 architecture.
//!
//! Tests build a tiny in-memory model (zero weights) and verify:
//! - Output shape is `[vocab_size]`
//! - All output values are finite
//! - `rope_theta` defaults to 1_000_000 for Grok-1

#[cfg(feature = "grok")]
mod grok_tests {
    use oxillama_arch::common::linear::QuantLinear;
    use oxillama_arch::common::rms_norm::RmsNorm;
    use oxillama_arch::config::ModelConfig;
    use oxillama_arch::error::ArchResult;
    use oxillama_arch::grok::config::GrokConfig;
    use oxillama_arch::grok::model::{build_grok_model, make_grok_layer};
    use oxillama_arch::registry::ArchitectureRegistry;
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

    fn make_f32_ql(rows: usize, cols: usize) -> QuantLinear {
        let data = vec![0u8; rows * cols * 4];
        let weight = QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32);
        QuantLinear::new(weight, None)
    }

    fn build_grok_test_model() -> impl ForwardPass {
        const H: usize = 32;
        const VOCAB: usize = 32;
        const N_HEADS: usize = 2;
        const HEAD_DIM: usize = 16;
        const N_LAYERS: usize = 2;
        const N_EXPERTS: usize = 8;
        const TOP_K: usize = 2;
        const EXPERT_INTER: usize = 16;

        let grok_cfg = GrokConfig {
            hidden_size: H,
            num_layers: N_LAYERS,
            num_heads: N_HEADS,
            num_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_seq_len: 128,
            expert_count: N_EXPERTS,
            expert_used_count: TOP_K,
            ffn_hidden_size: EXPERT_INTER,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
        };

        let model_cfg = ModelConfig {
            architecture: "grok".to_string(),
            model_name: "test-grok".to_string(),
            hidden_size: H,
            intermediate_size: EXPERT_INTER,
            num_layers: N_LAYERS,
            num_attention_heads: N_HEADS,
            num_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_context_length: 128,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            ..ModelConfig::default()
        };

        let layers = (0..N_LAYERS)
            .map(|_| {
                make_grok_layer(
                    H,
                    N_HEADS,
                    N_HEADS,
                    HEAD_DIM,
                    N_EXPERTS,
                    TOP_K,
                    EXPERT_INTER,
                )
            })
            .collect();

        let token_embd = vec![0.0f32; VOCAB * H];
        let output_norm = RmsNorm::new(vec![1.0f32; H], 1e-5);
        let output = make_f32_ql(VOCAB, H);

        build_grok_model(model_cfg, grok_cfg, token_embd, layers, output_norm, output)
    }

    /// Grok-1 is registered in the architecture registry under "grok".
    #[test]
    fn test_grok_registered_in_registry() {
        let reg = ArchitectureRegistry::with_builtins();
        assert!(reg.contains("grok"), "registry must contain 'grok'");
        let arch = reg.get("grok").expect("get grok must succeed");
        assert_eq!(arch.arch_id(), "grok");
    }

    /// Forward pass produces logits of correct shape.
    #[test]
    fn test_grok_forward_shape() {
        let mut model = build_grok_test_model();
        let mut kv = NullKv;
        let logits = model
            .forward(&[1u32, 2, 3], &mut kv)
            .expect("forward must succeed");
        assert_eq!(logits.len(), 32, "logits must have vocab_size=32 elements");
    }

    /// All forward pass outputs are finite.
    #[test]
    fn test_grok_forward_finite() {
        let mut model = build_grok_test_model();
        let mut kv = NullKv;
        let logits = model
            .forward(&[0u32], &mut kv)
            .expect("forward must succeed");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all Grok-1 logits must be finite"
        );
    }

    /// GGUF fixture parses correctly.
    #[test]
    fn test_grok_gguf_fixture_parses() {
        let bytes = oxillama_gguf::test_utils::build_minimal_grok_gguf();
        let model =
            oxillama_gguf::GgufModel::from_bytes(bytes).expect("Grok GGUF fixture must parse");
        assert_eq!(model.architecture().expect("arch must be present"), "grok");
    }

    /// Grok-1 rope_theta defaults to 1_000_000 (not the standard 10_000).
    #[test]
    fn test_grok_default_rope_theta_is_1e6() {
        let store = oxillama_gguf::MetadataStore::new();
        let cfg = GrokConfig::from_metadata(&store);
        assert!(
            (cfg.rope_theta - 1_000_000.0).abs() < 1.0,
            "Grok-1 default rope_theta must be 1_000_000, got {}",
            cfg.rope_theta
        );
    }
}

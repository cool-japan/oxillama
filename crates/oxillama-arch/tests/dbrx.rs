//! Integration tests for the DBRX architecture.
//!
//! Tests build a tiny in-memory model (zero weights) and verify:
//! - Output shape is `[vocab_size]`
//! - All output values are finite

#[cfg(feature = "dbrx")]
mod dbrx_tests {
    use oxillama_arch::common::linear::QuantLinear;
    use oxillama_arch::common::rms_norm::RmsNorm;
    use oxillama_arch::config::ModelConfig;
    use oxillama_arch::dbrx::config::DbrxConfig;
    use oxillama_arch::dbrx::model::{build_dbrx_model, make_dbrx_layer};
    use oxillama_arch::error::ArchResult;
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

    fn build_dbrx_test_model() -> impl ForwardPass {
        const H: usize = 32;
        const VOCAB: usize = 32;
        const N_HEADS: usize = 2;
        const HEAD_DIM: usize = 16;
        const N_LAYERS: usize = 2;
        const N_EXPERTS: usize = 4;
        const TOP_K: usize = 2;
        const EXPERT_INTER: usize = 16;

        let dbrx_cfg = DbrxConfig {
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
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        };

        let model_cfg = ModelConfig {
            architecture: "dbrx".to_string(),
            model_name: "test-dbrx".to_string(),
            hidden_size: H,
            intermediate_size: EXPERT_INTER,
            num_layers: N_LAYERS,
            num_attention_heads: N_HEADS,
            num_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_context_length: 128,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            ..ModelConfig::default()
        };

        let layers = (0..N_LAYERS)
            .map(|_| {
                make_dbrx_layer(
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

        build_dbrx_model(model_cfg, dbrx_cfg, token_embd, layers, output_norm, output)
    }

    /// DBRX is registered in the architecture registry under "dbrx".
    #[test]
    fn test_dbrx_registered_in_registry() {
        let reg = ArchitectureRegistry::with_builtins();
        assert!(reg.contains("dbrx"), "registry must contain 'dbrx'");
        let arch = reg.get("dbrx").expect("get dbrx must succeed");
        assert_eq!(arch.arch_id(), "dbrx");
    }

    /// Forward pass produces logits of correct shape.
    #[test]
    fn test_dbrx_forward_shape() {
        let mut model = build_dbrx_test_model();
        let mut kv = NullKv;
        let logits = model
            .forward(&[1u32, 2, 3], &mut kv)
            .expect("forward must succeed");
        assert_eq!(logits.len(), 32, "logits must have vocab_size=32 elements");
    }

    /// All forward pass outputs are finite.
    #[test]
    fn test_dbrx_forward_finite() {
        let mut model = build_dbrx_test_model();
        let mut kv = NullKv;
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
        let model =
            oxillama_gguf::GgufModel::from_bytes(bytes).expect("DBRX GGUF fixture must parse");
        assert_eq!(model.architecture().expect("arch must be present"), "dbrx");
    }
}

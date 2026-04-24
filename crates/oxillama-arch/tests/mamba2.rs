//! Integration tests for the Mamba-2 architecture.
//!
//! Tests build a tiny in-memory model (zero weights) and verify:
//! - Output shape is `[vocab_size]`
//! - All output values are finite
//! - Sequence-state reset produces identical output on re-run

#[cfg(feature = "mamba2")]
mod mamba2_tests {
    use oxillama_arch::common::rms_norm::RmsNorm;
    use oxillama_arch::error::ArchResult;
    use oxillama_arch::mamba2::model::{build_mamba2_model, make_zero_mamba2_layer, Mamba2Config};
    use oxillama_arch::registry::ArchitectureRegistry;
    use oxillama_arch::traits::{ForwardPass, KvCacheAccess};

    /// Null KV cache — Mamba-2 never calls KV cache, but the trait is required.
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

    /// Dimensions (small but structurally valid):
    ///   d_model=16, d_state=8, expand=1 → d_inner=16, d_conv=4, n_layer=2, vocab=64
    fn build_mamba2_test_model() -> impl ForwardPass {
        const D_MODEL: usize = 16;
        const D_STATE: usize = 8;
        const D_CONV: usize = 4;
        const EXPAND: usize = 1;
        const N_LAYER: usize = 2;
        const VOCAB: usize = 64;
        const MAX_SEQ: usize = 256;

        let cfg = Mamba2Config {
            d_model: D_MODEL,
            n_layer: N_LAYER,
            d_state: D_STATE,
            d_conv: D_CONV,
            expand: EXPAND,
            vocab_size: VOCAB,
            max_seq_len: MAX_SEQ,
        };

        let token_embd = vec![0.0f32; VOCAB * D_MODEL];
        let layers = (0..N_LAYER).map(|_| make_zero_mamba2_layer(&cfg)).collect();
        let output_norm = RmsNorm::new(vec![1.0f32; D_MODEL], 1e-5);
        let lm_head = vec![0.0f32; VOCAB * D_MODEL];

        build_mamba2_model(cfg, token_embd, layers, output_norm, lm_head)
    }

    /// For the reset-roundtrip test we need direct access to `Mamba2Model::reset_state`.
    fn build_mamba2_concrete_model() -> oxillama_arch::mamba2::model::Mamba2Model {
        const D_MODEL: usize = 16;
        const D_STATE: usize = 8;
        const D_CONV: usize = 4;
        const EXPAND: usize = 1;
        const N_LAYER: usize = 2;
        const VOCAB: usize = 64;
        const MAX_SEQ: usize = 256;

        let cfg = Mamba2Config {
            d_model: D_MODEL,
            n_layer: N_LAYER,
            d_state: D_STATE,
            d_conv: D_CONV,
            expand: EXPAND,
            vocab_size: VOCAB,
            max_seq_len: MAX_SEQ,
        };

        let token_embd = vec![0.0f32; VOCAB * D_MODEL];
        let layers = (0..N_LAYER).map(|_| make_zero_mamba2_layer(&cfg)).collect();
        let output_norm = RmsNorm::new(vec![1.0f32; D_MODEL], 1e-5);
        let lm_head = vec![0.0f32; VOCAB * D_MODEL];

        build_mamba2_model(cfg, token_embd, layers, output_norm, lm_head)
    }

    /// Mamba-2 is registered in the architecture registry under "mamba2".
    #[test]
    fn test_mamba2_registered_in_registry() {
        let reg = ArchitectureRegistry::with_builtins();
        assert!(reg.contains("mamba2"), "registry must contain 'mamba2'");
        let arch = reg.get("mamba2").expect("get mamba2 must succeed");
        assert_eq!(arch.arch_id(), "mamba2");
    }

    /// Forward pass with a 4-token input produces logits of correct shape.
    #[test]
    fn mamba2_forward_shape_and_finite() {
        let mut model = build_mamba2_test_model();
        let mut kv = NullKv;
        let logits = model
            .forward(&[1u32, 2, 3, 4], &mut kv)
            .expect("forward must succeed");
        assert_eq!(
            logits.len(),
            64,
            "logits must have vocab_size=64 elements, got {}",
            logits.len()
        );
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all Mamba-2 logits must be finite"
        );
    }

    /// `reset_state()` zeroes all per-layer hidden states and resets the position counter.
    ///
    /// This directly inspects the recurrent state tensors to prove that `reset_state()`
    /// is not a no-op — even when forward passes produce zero logits (all-zero weights),
    /// manually writing non-zero junk into `h` before reset lets us observe the clear.
    #[test]
    fn sequence_state_reset_roundtrip() {
        use oxillama_arch::common::sequence_state::SequenceState;

        let mut model = build_mamba2_concrete_model();
        let mut kv = NullKv;

        let tokens = [0u32, 5, 3, 7];

        // First pass — establishes a valid position counter.
        let logits_first = model
            .forward(&tokens, &mut kv)
            .expect("first forward must succeed");

        // Position counter must equal the number of tokens processed.
        assert_eq!(
            model.state.step_position(),
            tokens.len(),
            "position must equal token count after first forward"
        );

        // Write non-zero values into all SSM hidden states to prove reset clears them.
        for layer in &mut model.state.layers {
            layer.h.iter_mut().enumerate().for_each(|(i, v)| {
                *v = (i + 1) as f32 * 0.5;
            });
        }

        // Verify the junk is actually there before reset.
        assert!(
            model
                .state
                .layers
                .iter()
                .all(|l| l.h.iter().any(|&v| v != 0.0)),
            "h must be non-zero before reset"
        );

        // Reset clears hidden states AND position counter.
        model.reset_state();

        assert_eq!(
            model.state.step_position(),
            0,
            "position must be 0 after reset"
        );
        for (idx, layer) in model.state.layers.iter().enumerate() {
            assert!(
                layer.h.iter().all(|&v| v == 0.0),
                "layer {idx} h must be all-zero after reset"
            );
        }

        // Second pass on the same tokens — with a clean state, must match the first.
        let logits_second = model
            .forward(&tokens, &mut kv)
            .expect("second forward must succeed");

        assert_eq!(
            logits_first.len(),
            logits_second.len(),
            "output length must be identical after reset"
        );
        for (i, (a, b)) in logits_first.iter().zip(logits_second.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "logit[{i}] differs after state reset: {a} vs {b}"
            );
        }
    }

    /// GGUF fixture parses correctly with architecture == "mamba2".
    #[test]
    fn test_mamba2_gguf_fixture_parses() {
        let bytes = oxillama_gguf::test_utils::build_minimal_mamba2_gguf();
        let model =
            oxillama_gguf::GgufModel::from_bytes(bytes).expect("Mamba-2 GGUF fixture must parse");
        assert_eq!(
            model.architecture().expect("arch must be present"),
            "mamba2"
        );
    }
}

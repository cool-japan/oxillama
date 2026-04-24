//! Integration tests for the Jamba hybrid (LLaMA × Mamba-2) architecture.

#[cfg(feature = "jamba")]
mod jamba_tests {
    use oxillama_arch::common::sequence_state::SequenceState;
    use oxillama_arch::jamba::config::{JambaConfig, LayerKind};
    use oxillama_arch::jamba::{build_zero_jamba_model, JambaArchitecture, JambaSequenceState};
    use oxillama_arch::traits::{ForwardPass, ModelArchitecture};

    // ─── Helper ───────────────────────────────────────────────────────────────

    fn minimal_config() -> JambaConfig {
        JambaConfig {
            n_layers: 4,
            attn_layer_period: 2,
            attn_layer_offset: 0,
            expert_count: 0,
            expert_top_k: 1,
            d_state: 4,
            d_inner: 8,
            d_conv: 2,
            hidden_size: 8,
            num_attention_heads: 2,
            num_kv_heads: 2,
            intermediate_size: 16,
            vocab_size: 32,
            max_context_length: 64,
            rms_norm_eps: 1e-5,
        }
    }

    // ─── Config / layer-pattern tests ─────────────────────────────────────────

    /// 16-layer model with period=8: layers 0 and 8 are attention.
    #[test]
    fn jamba_layer_pattern_default_n8() {
        let cfg = JambaConfig {
            n_layers: 16,
            attn_layer_period: 8,
            attn_layer_offset: 0,
            ..minimal_config()
        };

        let pattern = cfg.layer_pattern();
        assert_eq!(
            pattern[0],
            LayerKind::Attention,
            "layer 0 must be attention"
        );
        assert_eq!(pattern[1], LayerKind::Ssm, "layer 1 must be SSM");
        assert_eq!(
            pattern[8],
            LayerKind::Attention,
            "layer 8 must be attention"
        );
        assert_eq!(cfg.attn_layer_count(), 2, "2 attention layers out of 16");
        assert_eq!(cfg.ssm_layer_count(), 14, "14 SSM layers out of 16");
    }

    // ─── JambaSequenceState tests ─────────────────────────────────────────────

    /// Reset clears position and zeros SSM layer states.
    #[test]
    fn jamba_sequence_state_reset_clears_ssm_state() {
        let cfg = minimal_config();
        let mut state = JambaSequenceState::new(&cfg, 64);

        // Advance position.
        state.advance();
        state.advance();
        assert_eq!(
            state.step_position(),
            2,
            "position must be 2 after 2 advances"
        );

        // Set some SSM h values.
        use oxillama_arch::jamba::JambaLayerState;
        for layer in &mut state.layers {
            if let JambaLayerState::Ssm(ssm) = layer {
                ssm.h.iter_mut().for_each(|v| *v = 1.0);
            }
        }

        state.reset();
        assert_eq!(state.step_position(), 0, "position must be 0 after reset");
        for layer in &state.layers {
            if let JambaLayerState::Ssm(ssm) = layer {
                assert!(
                    ssm.h.iter().all(|&v| v == 0.0),
                    "SSM h must be zeroed after reset"
                );
            }
        }
    }

    /// Two `JambaSequenceState` instances do not share state.
    #[test]
    fn jamba_mixed_state_isolation() {
        let cfg = minimal_config();
        let mut state_a = JambaSequenceState::new(&cfg, 64);
        let mut state_b = JambaSequenceState::new(&cfg, 64);

        state_a.advance();
        state_a.advance();

        assert_eq!(state_b.step_position(), 0, "state_b must remain at 0");
        state_b.advance();
        assert_eq!(state_a.step_position(), 2, "state_a must still be 2");
    }

    /// `capacity()` returns `max_context_length`.
    #[test]
    fn jamba_sequence_state_capacity() {
        let cfg = minimal_config();
        let state = JambaSequenceState::new(&cfg, 128);
        assert_eq!(state.capacity(), 128);
    }

    // ─── Architecture trait tests ─────────────────────────────────────────────

    /// JambaArchitecture::arch_id() returns "jamba".
    #[test]
    fn jamba_arch_id_is_jamba() {
        assert_eq!(JambaArchitecture::new().arch_id(), "jamba");
    }

    /// tensor_names() returns expected patterns for both block types.
    #[test]
    fn jamba_tensor_names() {
        let arch = JambaArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();

        assert!(
            patterns.contains(&"token_embd.weight"),
            "must have token_embd.weight"
        );
        assert!(
            patterns.contains(&"output.weight"),
            "must have output.weight"
        );
        assert!(
            patterns.iter().any(|p| p.contains("attn_q")),
            "must have attn_q pattern"
        );
        assert!(
            patterns.iter().any(|p| p.contains("ssm_in")),
            "must have ssm_in pattern"
        );
    }

    /// `allocate_sequence_state` on a zero Jamba model returns a Jamba state.
    #[test]
    fn jamba_allocate_sequence_state_correct_capacity() {
        let cfg = minimal_config();
        let model = build_zero_jamba_model(cfg.clone());
        let state = model.allocate_sequence_state(cfg.max_context_length);

        assert_eq!(
            state.step_position(),
            0,
            "fresh state must be at position 0"
        );
        assert_eq!(
            state.capacity(),
            cfg.max_context_length,
            "state capacity must match config"
        );
    }

    // ─── A1↔B1 trait seam test ────────────────────────────────────────────────

    /// `allocate_sequence_state(512)` on an all-attention Jamba model returns a state at position 0.
    ///
    /// This exercises the default `AttentionSequenceState` path — since Jamba with
    /// `attn_layer_period=1` makes every layer an attention block.
    #[test]
    fn arch_allocate_sequence_state_default_kv() {
        let cfg = JambaConfig {
            n_layers: 2,
            attn_layer_period: 1, // all attention
            ..minimal_config()
        };
        let model = build_zero_jamba_model(cfg);
        let state = model.allocate_sequence_state(512);
        assert_eq!(
            state.step_position(),
            0,
            "allocate_sequence_state must return a state at position 0"
        );
    }

    /// Jamba is registered in the default registry.
    #[test]
    fn jamba_registered_in_registry() {
        let registry = oxillama_arch::ArchitectureRegistry::with_builtins();
        assert!(
            registry.contains("jamba"),
            "jamba must be in the default registry"
        );
    }
}

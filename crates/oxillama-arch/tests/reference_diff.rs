//! Integration tests for the full-precision f32 reference path (`reference-f32` feature).

#[cfg(feature = "reference-f32")]
mod tests {
    use oxillama_arch::reference::ReferenceLoader;
    use oxillama_gguf::test_utils::build_minimal_llama_gguf;
    use oxillama_gguf::GgufModel;

    /// All tensors in a synthetic GGUF should dequantize to finite f32 values.
    #[test]
    fn reference_dequantizes_tensors_to_f32() {
        let bytes = build_minimal_llama_gguf();
        let model = GgufModel::from_bytes(bytes).expect("parse synthetic GGUF");
        let weights = ReferenceLoader::dequantize_all(&model)
            .expect("dequantize_all must succeed for a valid GGUF");

        assert!(
            !weights.tensors.is_empty(),
            "weights map must not be empty after dequantization"
        );

        for (name, data) in &weights.tensors {
            assert!(
                data.iter().all(|v| v.is_finite()),
                "tensor '{name}' contains non-finite f32 values after dequantization"
            );
        }
    }

    /// Shape metadata is preserved after dequantization.
    #[test]
    fn reference_shapes_match_tensor_count() {
        let bytes = build_minimal_llama_gguf();
        let model = GgufModel::from_bytes(bytes).expect("parse synthetic GGUF");
        let weights = ReferenceLoader::dequantize_all(&model).expect("dequantize_all must succeed");

        assert_eq!(
            weights.tensors.len(),
            weights.shapes.len(),
            "tensors and shapes maps must have the same number of entries"
        );

        // Every shape entry must have at least one dimension.
        for (name, shape) in &weights.shapes {
            assert!(!shape.is_empty(), "tensor '{name}' has empty shape");
        }
    }

    /// `load_as_reference` constructs a `ForwardPass` box without panicking.
    #[test]
    fn reference_forward_constructs_successfully() {
        let bytes = build_minimal_llama_gguf();
        let model = GgufModel::from_bytes(bytes).expect("parse synthetic GGUF");

        // synthetic LLaMA: vocab=32, hidden=32, max_ctx=512
        let fwd = oxillama_arch::reference::load_as_reference(&model, 32, 512, 32)
            .expect("load_as_reference must succeed for a valid GGUF");

        assert_eq!(fwd.vocab_size(), 32);
        assert_eq!(fwd.hidden_size(), 32);
        assert_eq!(fwd.max_context_length(), 512);
    }
}

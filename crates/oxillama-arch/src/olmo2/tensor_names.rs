//! GGUF tensor name patterns for OLMo2.
//!
//! OLMo2 uses post-norm style with per-head QK-norm tensors.  The naming
//! follows the llama.cpp GGUF convention for OLMo2.

use crate::traits::TensorNamePattern;

/// Return all tensor name patterns used by OLMo2 models.
///
/// Patterns use `{i}` as a placeholder for the layer index.
pub fn olmo2_tensor_name_patterns() -> Vec<TensorNamePattern> {
    let mut patterns = vec![
        TensorNamePattern {
            pattern: "token_embd.weight".to_string(),
            description: "Token embedding matrix [vocab_size, hidden_size]".to_string(),
            required: true,
        },
        TensorNamePattern {
            pattern: "output_norm.weight".to_string(),
            description: "Final RMSNorm scale weights".to_string(),
            required: true,
        },
        TensorNamePattern {
            pattern: "output.weight".to_string(),
            description: "LM head / unembedding matrix [vocab_size, hidden_size]".to_string(),
            required: true,
        },
    ];

    let layer_patterns: &[(&str, &str, bool)] = &[
        // Per-head QK-norm (OLMo2 unique)
        (
            "blk.{i}.attn_q_norm.weight",
            "Per-head query RMSNorm scale",
            true,
        ),
        (
            "blk.{i}.attn_k_norm.weight",
            "Per-head key RMSNorm scale",
            true,
        ),
        // Attention projections (no pre-attn norm on x — norms applied after)
        ("blk.{i}.attn_q.weight", "Query projection weight", true),
        ("blk.{i}.attn_k.weight", "Key projection weight", true),
        ("blk.{i}.attn_v.weight", "Value projection weight", true),
        (
            "blk.{i}.attn_output.weight",
            "Attention output projection weight",
            true,
        ),
        // Post-attn norm (unique to OLMo2)
        (
            "blk.{i}.attn_post_norm.weight",
            "Post-attention RMSNorm scale",
            true,
        ),
        // FFN projections (SwiGLU)
        (
            "blk.{i}.ffn_gate.weight",
            "FFN gate projection weight (SwiGLU)",
            true,
        ),
        ("blk.{i}.ffn_up.weight", "FFN up projection weight", true),
        (
            "blk.{i}.ffn_down.weight",
            "FFN down projection weight",
            true,
        ),
        // Post-FFN norm (unique to OLMo2)
        (
            "blk.{i}.ffn_post_norm.weight",
            "Post-FFN RMSNorm scale",
            true,
        ),
    ];

    for (pat, desc, required) in layer_patterns {
        patterns.push(TensorNamePattern {
            pattern: pat.to_string(),
            description: desc.to_string(),
            required: *required,
        });
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patterns_non_empty() {
        let p = olmo2_tensor_name_patterns();
        assert!(!p.is_empty());
    }

    #[test]
    fn test_token_embd_present() {
        let p = olmo2_tensor_name_patterns();
        assert!(p.iter().any(|t| t.pattern == "token_embd.weight"));
    }

    #[test]
    fn test_output_weight_present() {
        let p = olmo2_tensor_name_patterns();
        assert!(p.iter().any(|t| t.pattern == "output.weight"));
    }

    #[test]
    fn test_attn_post_norm_present() {
        let p = olmo2_tensor_name_patterns();
        assert!(
            p.iter().any(|t| t.pattern.contains("attn_post_norm")),
            "OLMo2 must have attn_post_norm"
        );
    }

    #[test]
    fn test_ffn_post_norm_present() {
        let p = olmo2_tensor_name_patterns();
        assert!(
            p.iter().any(|t| t.pattern.contains("ffn_post_norm")),
            "OLMo2 must have ffn_post_norm"
        );
    }

    #[test]
    fn test_qk_norm_tensors_present() {
        let p = olmo2_tensor_name_patterns();
        assert!(p.iter().any(|t| t.pattern.contains("attn_q_norm")));
        assert!(p.iter().any(|t| t.pattern.contains("attn_k_norm")));
    }
}

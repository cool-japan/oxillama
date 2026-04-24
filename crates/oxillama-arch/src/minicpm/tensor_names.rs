//! GGUF tensor name patterns for MiniCPM.
//!
//! MiniCPM uses the same GGUF tensor naming scheme as LLaMA. The only
//! architectural difference (scaled embedding) is a runtime computation, not
//! a separate tensor.

use crate::traits::TensorNamePattern;

/// Return all tensor name patterns used by MiniCPM models.
///
/// Patterns use `{i}` as a placeholder for the layer index.
pub fn minicpm_tensor_name_patterns() -> Vec<TensorNamePattern> {
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
        (
            "blk.{i}.attn_norm.weight",
            "Pre-attention RMSNorm scale",
            true,
        ),
        ("blk.{i}.ffn_norm.weight", "Pre-FFN RMSNorm scale", true),
        ("blk.{i}.attn_q.weight", "Query projection weight", true),
        ("blk.{i}.attn_k.weight", "Key projection weight", true),
        ("blk.{i}.attn_v.weight", "Value projection weight", true),
        (
            "blk.{i}.attn_output.weight",
            "Attention output projection weight",
            true,
        ),
        (
            "blk.{i}.ffn_gate.weight",
            "FFN gate projection weight (SwiGLU)",
            true,
        ),
        (
            "blk.{i}.ffn_up.weight",
            "FFN up projection weight (SwiGLU)",
            true,
        ),
        (
            "blk.{i}.ffn_down.weight",
            "FFN down projection weight",
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
        let p = minicpm_tensor_name_patterns();
        assert!(!p.is_empty());
    }

    #[test]
    fn test_token_embd_present() {
        let p = minicpm_tensor_name_patterns();
        assert!(p.iter().any(|t| t.pattern == "token_embd.weight"));
    }

    #[test]
    fn test_output_weight_present() {
        let p = minicpm_tensor_name_patterns();
        assert!(p.iter().any(|t| t.pattern == "output.weight"));
    }

    #[test]
    fn test_attn_q_present() {
        let p = minicpm_tensor_name_patterns();
        assert!(p.iter().any(|t| t.pattern.contains("attn_q.weight")));
    }

    #[test]
    fn test_ffn_gate_present() {
        let p = minicpm_tensor_name_patterns();
        assert!(p.iter().any(|t| t.pattern.contains("ffn_gate")));
    }

    #[test]
    fn test_all_required_tensors_marked() {
        let p = minicpm_tensor_name_patterns();
        let required: Vec<_> = p.iter().filter(|t| t.required).collect();
        assert!(!required.is_empty(), "some tensors must be required");
    }
}

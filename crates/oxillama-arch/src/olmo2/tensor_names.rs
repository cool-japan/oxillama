//! GGUF tensor name patterns for OLMo2.
//!
//! OLMo2 uses post-norm style with QK-norm tensors.  The naming follows the
//! llama.cpp GGUF convention for OLMo2.
//!
//! # The post-norm names are NOT `attn_post_norm` / `ffn_post_norm`
//!
//! `LLM_TENSOR_NAMES` in `~/work/refs/llama.cpp/src/llama-arch.cpp` is a single
//! table shared by every architecture, and it resolves the two post-norm slots
//! to Gemma-2's original spellings:
//!
//! ```text
//! llama-arch.cpp:355  { LLM_TENSOR_ATTN_POST_NORM, "blk.%d.post_attention_norm" }
//! llama-arch.cpp:359  { LLM_TENSOR_FFN_POST_NORM,  "blk.%d.post_ffw_norm"       }
//! ```
//!
//! `gguf-py/gguf/constants.py` agrees verbatim (lines 964 / 969:
//! `MODEL_TENSOR.ATTN_POST_NORM: "blk.{bid}.post_attention_norm"`,
//! `MODEL_TENSOR.FFN_POST_NORM: "blk.{bid}.post_ffw_norm"`), and
//! `gguf-py/gguf/tensor_mapping.py` maps HF's `post_attention_layernorm` /
//! `post_feedforward_layernorm` onto them with the comment `# gemma2 olmo2`.
//! `LLM_ARCH_OLMO2` in `llama-arch.cpp:1507` lists both enum slots, so a real
//! OLMo2 checkpoint ships exactly those two literal names.
//!
//! This module previously declared `blk.{i}.attn_post_norm.weight` and
//! `blk.{i}.ffn_post_norm.weight`, which match no checkpoint that has ever been
//! produced.

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
        // QK-norm (OLMo2 unique).  Applied to the WHOLE projected vector, so
        // the widths differ under GQA: Q is `n_embd`, K is `n_head_kv *
        // n_embd_head` (llama-model.cpp, `case LLM_ARCH_OLMO2:`).
        (
            "blk.{i}.attn_q_norm.weight",
            "Query RMSNorm scale [n_embd]",
            true,
        ),
        (
            "blk.{i}.attn_k_norm.weight",
            "Key RMSNorm scale [n_head_kv * n_embd_head]",
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
        // Post-attn norm.  `LLM_TENSOR_ATTN_POST_NORM` →
        // `"blk.%d.post_attention_norm"` (llama-arch.cpp:355).
        (
            "blk.{i}.post_attention_norm.weight",
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
        // Post-FFN norm.  `LLM_TENSOR_FFN_POST_NORM` →
        // `"blk.%d.post_ffw_norm"` (llama-arch.cpp:359).
        (
            "blk.{i}.post_ffw_norm.weight",
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

    /// O4: the post-attention norm is `blk.{i}.post_attention_norm.weight`.
    ///
    /// This assertion FAILS against the pre-fix table, which declared
    /// `blk.{i}.attn_post_norm.weight` — a name no OLMo2 checkpoint contains.
    #[test]
    fn test_attn_post_norm_uses_llama_cpp_name() {
        let p = olmo2_tensor_name_patterns();
        assert!(
            p.iter()
                .any(|t| t.pattern == "blk.{i}.post_attention_norm.weight"),
            "OLMo2 post-attention norm is 'blk.%d.post_attention_norm' (llama-arch.cpp:355)"
        );
        assert!(
            !p.iter()
                .any(|t| t.pattern == "blk.{i}.attn_post_norm.weight"),
            "'attn_post_norm' is not a GGUF tensor name for any architecture"
        );
    }

    /// O4: the post-FFN norm is `blk.{i}.post_ffw_norm.weight`.
    #[test]
    fn test_ffn_post_norm_uses_llama_cpp_name() {
        let p = olmo2_tensor_name_patterns();
        assert!(
            p.iter()
                .any(|t| t.pattern == "blk.{i}.post_ffw_norm.weight"),
            "OLMo2 post-FFN norm is 'blk.%d.post_ffw_norm' (llama-arch.cpp:359)"
        );
        assert!(
            !p.iter()
                .any(|t| t.pattern == "blk.{i}.ffn_post_norm.weight"),
            "'ffn_post_norm' is not a GGUF tensor name for any architecture"
        );
    }

    #[test]
    fn test_qk_norm_tensors_present() {
        let p = olmo2_tensor_name_patterns();
        assert!(p.iter().any(|t| t.pattern.contains("attn_q_norm")));
        assert!(p.iter().any(|t| t.pattern.contains("attn_k_norm")));
    }
}

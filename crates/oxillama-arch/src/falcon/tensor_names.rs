//! GGUF tensor name patterns for the Falcon model family.
//!
//! Covers both the old Falcon-1 naming (llama.cpp convention) and the
//! updated Falcon-2 naming used in more recent GGUF files.

use crate::traits::TensorNamePattern;

/// Return all tensor name patterns used by Falcon models.
///
/// Patterns use `{i}` as a placeholder for the layer index.
pub fn falcon_tensor_name_patterns() -> Vec<TensorNamePattern> {
    let mut patterns = vec![
        // ── Global tensors ──────────────────────────────────────────────────
        TensorNamePattern {
            pattern: "token_embd.weight".to_string(),
            description: "Token embedding matrix [vocab_size, hidden_size]".to_string(),
            required: true,
        },
        TensorNamePattern {
            pattern: "output_norm.weight".to_string(),
            description: "Final LayerNorm scale weights".to_string(),
            required: true,
        },
        TensorNamePattern {
            pattern: "output_norm.bias".to_string(),
            description: "Final LayerNorm bias".to_string(),
            required: false,
        },
        TensorNamePattern {
            pattern: "output.weight".to_string(),
            description: "LM head / unembedding matrix [vocab_size, hidden_size]".to_string(),
            required: true,
        },
    ];

    // ── Per-layer tensors ────────────────────────────────────────────────────
    //
    // Old Falcon-1 naming (parallel attention): one shared LayerNorm feeds
    // both the attention and FFN branches.  Newer Falcon-2 files may have
    // separate `attn_norm` and `ffn_norm` per layer.
    let layer_patterns: &[(&str, &str, bool)] = &[
        // Attention normalisation (present in all Falcon variants)
        (
            "blk.{i}.attn_norm.weight",
            "Pre-attention LayerNorm scale",
            true,
        ),
        (
            "blk.{i}.attn_norm.bias",
            "Pre-attention LayerNorm bias",
            false,
        ),
        // FFN normalisation — Falcon-1 (parallel) uses a shared attn_norm;
        // Falcon-2 (sequential) has a separate ffn_norm.
        (
            "blk.{i}.ffn_norm.weight",
            "Pre-FFN LayerNorm scale (Falcon-2 / sequential mode)",
            false,
        ),
        (
            "blk.{i}.ffn_norm.bias",
            "Pre-FFN LayerNorm bias (Falcon-2 / sequential mode)",
            false,
        ),
        // Fused QKV projection — Falcon-1 packs Q, K, V into one weight.
        (
            "blk.{i}.attn_qkv.weight",
            "Fused QKV projection weight",
            true,
        ),
        ("blk.{i}.attn_qkv.bias", "Fused QKV projection bias", false),
        // Attention output projection
        (
            "blk.{i}.attn_output.weight",
            "Attention output projection weight",
            true,
        ),
        (
            "blk.{i}.attn_output.bias",
            "Attention output projection bias",
            false,
        ),
        // FFN projections (Falcon uses a simple up → GELU/SiLU → down FFN,
        // not SwiGLU, so there is no gate weight)
        ("blk.{i}.ffn_up.weight", "FFN up-projection weight", true),
        ("blk.{i}.ffn_up.bias", "FFN up-projection bias", false),
        (
            "blk.{i}.ffn_down.weight",
            "FFN down-projection weight",
            true,
        ),
        ("blk.{i}.ffn_down.bias", "FFN down-projection bias", false),
    ];

    for (pattern, description, required) in layer_patterns {
        patterns.push(TensorNamePattern {
            pattern: pattern.to_string(),
            description: description.to_string(),
            required: *required,
        });
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_global_tensors_present() {
        let pats = falcon_tensor_name_patterns();
        let required_globals = ["token_embd.weight", "output_norm.weight", "output.weight"];
        for name in required_globals {
            assert!(
                pats.iter().any(|p| p.pattern == name && p.required),
                "Required tensor {name:?} missing from patterns"
            );
        }
    }

    #[test]
    fn test_required_per_layer_tensors_present() {
        let pats = falcon_tensor_name_patterns();
        let required_layer = [
            "blk.{i}.attn_norm.weight",
            "blk.{i}.attn_qkv.weight",
            "blk.{i}.attn_output.weight",
            "blk.{i}.ffn_up.weight",
            "blk.{i}.ffn_down.weight",
        ];
        for name in required_layer {
            assert!(
                pats.iter().any(|p| p.pattern == name && p.required),
                "Required layer tensor {name:?} missing from patterns"
            );
        }
    }

    #[test]
    fn test_bias_tensors_are_optional() {
        let pats = falcon_tensor_name_patterns();
        let optional_biases = [
            "blk.{i}.attn_norm.bias",
            "blk.{i}.attn_qkv.bias",
            "blk.{i}.attn_output.bias",
            "blk.{i}.ffn_up.bias",
            "blk.{i}.ffn_down.bias",
        ];
        for name in optional_biases {
            assert!(
                pats.iter().any(|p| p.pattern == name && !p.required),
                "Bias tensor {name:?} should be optional"
            );
        }
    }

    #[test]
    fn test_no_duplicate_patterns() {
        let pats = falcon_tensor_name_patterns();
        let mut seen = std::collections::HashSet::new();
        for p in &pats {
            let inserted = seen.insert(p.pattern.clone());
            assert!(inserted, "Duplicate pattern found: {}", p.pattern);
        }
    }

    #[test]
    fn test_fused_qkv_not_split() {
        // Falcon uses fused QKV, not separate attn_q / attn_k / attn_v
        let pats = falcon_tensor_name_patterns();
        for p in &pats {
            assert!(
                !p.pattern.contains("attn_q."),
                "Falcon should not have separate attn_q tensor; found: {}",
                p.pattern
            );
            assert!(
                !p.pattern.contains("attn_k."),
                "Falcon should not have separate attn_k tensor; found: {}",
                p.pattern
            );
            assert!(
                !p.pattern.contains("attn_v."),
                "Falcon should not have separate attn_v tensor; found: {}",
                p.pattern
            );
        }
    }
}

//! GGUF tensor name patterns for the Falcon model family.
//!
//! # Reference
//!
//! `~/work/refs/llama.cpp/src/llama-model.cpp`, `case LLM_ARCH_FALCON:` in
//! `load_tensors()`:
//!
//! ```text
//! tok_embd      = create_tensor(TOKEN_EMBD  "weight", {n_embd, n_vocab}, 0);
//! output_norm   = create_tensor(OUTPUT_NORM "weight", {n_embd}, 0);
//! output_norm_b = create_tensor(OUTPUT_NORM "bias",   {n_embd}, 0);
//! output        = create_tensor(OUTPUT      "weight", {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
//! if (!output) output = create_tensor(TOKEN_EMBD "weight", ..., TENSOR_DUPLICATED);
//! // per layer:
//! attn_norm     = create_tensor(ATTN_NORM   "weight", i, {n_embd}, 0);
//! attn_norm_b   = create_tensor(ATTN_NORM   "bias",   i, {n_embd}, 0);
//! attn_norm_2   = create_tensor(ATTN_NORM_2 "weight", i, {n_embd}, TENSOR_NOT_REQUIRED);
//! attn_norm_2_b = create_tensor(ATTN_NORM_2 "bias",   i, {n_embd}, TENSOR_NOT_REQUIRED);
//! wqkv          = create_tensor(ATTN_QKV    "weight", i, {n_embd, n_embd + 2*n_embd_gqa}, 0);
//! wo            = create_tensor(ATTN_OUT    "weight", i, {n_embd, n_embd}, 0);
//! ffn_down      = create_tensor(FFN_DOWN    "weight", i, {n_ff, n_embd}, 0);
//! ffn_up        = create_tensor(FFN_UP      "weight", i, {n_embd, n_ff}, 0);
//! ```
//!
//! `LLM_TENSOR_NAMES` (`src/llama-arch.cpp`) resolves `ATTN_NORM_2` to the
//! literal `"blk.%d.attn_norm_2"`.
//!
//! Two consequences the previous version of this table got wrong:
//!
//! * Falcon has **no linear biases at all** — no `attn_qkv.bias`, no
//!   `attn_output.bias`, no `ffn_{up,down}.bias`.  The only biases in the
//!   architecture are the LayerNorm shifts, and those are **required**.
//! * Falcon has **no `ffn_norm`**.  The FFN reads the same `attn_norm` output
//!   the attention branch was built from.
//!
//! The optional bias/`ffn_norm` patterns those two facts retire are not listed
//! here any more; the loader still accepts them if a third-party conversion
//! ships one (`load_bias` returns `None` when a tensor is absent), it just no
//! longer advertises tensors llama.cpp never writes.

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
            description: "Final LayerNorm scale".to_string(),
            required: true,
        },
        TensorNamePattern {
            pattern: "output_norm.bias".to_string(),
            // `create_tensor(OUTPUT_NORM "bias", {n_embd}, 0)` — flags 0, i.e.
            // required.  Falcon's final norm is a true LayerNorm, not RMSNorm.
            description: "Final LayerNorm shift".to_string(),
            required: true,
        },
        TensorNamePattern {
            pattern: "output.weight".to_string(),
            // TENSOR_NOT_REQUIRED with a TENSOR_DUPLICATED fallback onto
            // `token_embd.weight`; `load_lm_head` implements that fallback.
            description: "LM head / unembedding matrix [vocab_size, hidden_size] \
                          (absent on tied checkpoints — falls back to token_embd.weight)"
                .to_string(),
            required: false,
        },
    ];

    // ── Per-layer tensors ────────────────────────────────────────────────────
    let layer_patterns: &[(&str, &str, bool)] = &[
        // Pre-attention / pre-FFN LayerNorm.  Its output feeds the FFN
        // unconditionally, and the attention branch too unless `attn_norm_2`
        // is present.  Weight *and* bias are required.
        (
            "blk.{i}.attn_norm.weight",
            "Pre-attention LayerNorm scale (also the FFN's input norm)",
            true,
        ),
        (
            "blk.{i}.attn_norm.bias",
            "Pre-attention LayerNorm shift",
            true,
        ),
        // Falcon-40B's second LayerNorm: when present the attention branch
        // reads *this* norm's output instead of `attn_norm`'s.
        (
            "blk.{i}.attn_norm_2.weight",
            "Second pre-attention LayerNorm scale (Falcon-40B only)",
            false,
        ),
        (
            "blk.{i}.attn_norm_2.bias",
            "Second pre-attention LayerNorm shift (Falcon-40B only)",
            false,
        ),
        // Fused QKV projection: [n_embd, n_embd + 2*n_embd_gqa], no bias.
        (
            "blk.{i}.attn_qkv.weight",
            "Fused QKV projection weight [n_embd, n_embd + 2*n_embd_gqa]",
            true,
        ),
        // Attention output projection, no bias.
        (
            "blk.{i}.attn_output.weight",
            "Attention output projection weight [n_embd, n_embd]",
            true,
        ),
        // FFN: plain up → GELU → down.  No gate (not SwiGLU), no biases.
        ("blk.{i}.ffn_up.weight", "FFN up-projection weight", true),
        (
            "blk.{i}.ffn_down.weight",
            "FFN down-projection weight",
            true,
        ),
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
        let required_globals = [
            "token_embd.weight",
            "output_norm.weight",
            // `create_tensor(OUTPUT_NORM "bias", ..., 0)` — required, not optional.
            "output_norm.bias",
        ];
        for name in required_globals {
            assert!(
                pats.iter().any(|p| p.pattern == name && p.required),
                "Required tensor {name:?} missing from patterns"
            );
        }
    }

    /// `output.weight` is `TENSOR_NOT_REQUIRED` in llama.cpp with a tied
    /// `token_embd.weight` fallback, so advertising it as required would
    /// reject every tied Falcon checkpoint.
    #[test]
    fn test_output_weight_is_optional_because_it_can_be_tied() {
        let pats = falcon_tensor_name_patterns();
        let out = pats
            .iter()
            .find(|p| p.pattern == "output.weight")
            .expect("output.weight pattern");
        assert!(
            !out.required,
            "output.weight must be optional (tied fallback)"
        );
    }

    #[test]
    fn test_required_per_layer_tensors_present() {
        let pats = falcon_tensor_name_patterns();
        let required_layer = [
            "blk.{i}.attn_norm.weight",
            "blk.{i}.attn_norm.bias",
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

    /// F2 regression: `attn_norm_2` (Falcon-40B's second LayerNorm) must be
    /// declared, and declared *optional* — Falcon-7B/Falcon-2 have no such
    /// tensor.  Before the fix neither pattern existed at all, so a Falcon-40B
    /// checkpoint's second norm was invisible to the loader.
    #[test]
    fn test_attn_norm_2_is_declared_and_optional() {
        let pats = falcon_tensor_name_patterns();
        for name in ["blk.{i}.attn_norm_2.weight", "blk.{i}.attn_norm_2.bias"] {
            let pat = pats
                .iter()
                .find(|p| p.pattern == name)
                .unwrap_or_else(|| panic!("{name:?} must be declared (Falcon-40B)"));
            assert!(
                !pat.required,
                "{name:?} is TENSOR_NOT_REQUIRED in llama.cpp"
            );
        }
    }

    /// Falcon creates no linear bias tensors at all (`llama-model.cpp`,
    /// `LLM_ARCH_FALCON`), so the table must not advertise any.
    #[test]
    fn test_no_linear_bias_patterns() {
        let pats = falcon_tensor_name_patterns();
        for name in [
            "blk.{i}.attn_qkv.bias",
            "blk.{i}.attn_output.bias",
            "blk.{i}.ffn_up.bias",
            "blk.{i}.ffn_down.bias",
        ] {
            assert!(
                !pats.iter().any(|p| p.pattern == name),
                "Falcon has no linear biases; {name:?} must not be advertised"
            );
        }
    }

    /// Falcon has no `ffn_norm`: the FFN reuses `attn_norm`'s output.
    #[test]
    fn test_no_ffn_norm_pattern() {
        let pats = falcon_tensor_name_patterns();
        assert!(
            !pats.iter().any(|p| p.pattern.contains("ffn_norm")),
            "Falcon's FFN reads attn_norm; llama.cpp creates no ffn_norm tensor"
        );
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

    /// Falcon's FFN is up → GELU → down; there is no gate projection.
    #[test]
    fn test_no_ffn_gate_pattern() {
        let pats = falcon_tensor_name_patterns();
        assert!(
            !pats.iter().any(|p| p.pattern.contains("ffn_gate")),
            "Falcon's FFN is not SwiGLU; there is no ffn_gate tensor"
        );
    }
}

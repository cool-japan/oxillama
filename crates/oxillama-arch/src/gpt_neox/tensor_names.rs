//! GGUF tensor name patterns for the GPT-NeoX (Pythia) model family.
//!
//! # Provenance
//!
//! Every pattern below is transcribed from `~/work/refs/llama.cpp`:
//!
//! * `src/llama-model.cpp`, `case LLM_ARCH_GPTNEOX:` inside `load_tensors()`
//!   — the authoritative list of tensors llama.cpp creates for this
//!   architecture, together with their `flags`.  **Every** GPT-NeoX tensor is
//!   created with `flags = 0` (i.e. `TENSOR_NOT_REQUIRED` is never passed), so
//!   every pattern here is `required: true` — including the bias of every
//!   single linear projection, which is the defining quirk of the GPT-NeoX
//!   block.
//! * `gguf-py/gguf/constants.py`'s `TENSOR_NAMES` table, which turns the
//!   `LLM_TENSOR_*` enum used at that call site into the on-disk strings:
//!   `LLM_TENSOR_ATTN_NORM` → `blk.{bid}.attn_norm`,
//!   `LLM_TENSOR_ATTN_QKV` → `blk.{bid}.attn_qkv`,
//!   `LLM_TENSOR_ATTN_OUT` → `blk.{bid}.attn_output`,
//!   `LLM_TENSOR_FFN_NORM` / `FFN_UP` / `FFN_DOWN` →
//!   `blk.{bid}.ffn_norm` / `ffn_up` / `ffn_down`.
//!
//! # The `ln1` / `ln2` trap (defect N1)
//!
//! A previous revision of this module declared `blk.{i}.ln1.weight`,
//! `blk.{i}.ln2.weight` and three *separate* `attn_q` / `attn_k` / `attn_v`
//! projections.  **None of those names exists in any real GGUF file.**  There
//! is no `ln1`/`ln2` spelling anywhere in llama.cpp or `gguf-py` — the
//! LayerNorms are `attn_norm` and `ffn_norm` like every other architecture —
//! and GPT-NeoX's attention projection is a single *fused* `attn_qkv` tensor.
//! Any loader driven by the old list could only ever report
//! [`MissingTensor`](crate::error::ArchError::MissingTensor).
//!
//! # Fused QKV layout
//!
//! `blk.{i}.attn_qkv.weight` has GGUF shape
//! `{n_embd, n_embd + 2 * n_embd_gqa}` (in-features first), i.e. the
//! mathematical matrix is `[n_embd + 2 * n_embd_gqa, n_embd]`.  Its output
//! vector is three **contiguous** blocks:
//!
//! ```text
//! rows [0 .. n_embd)                              → Q
//! rows [n_embd .. n_embd + n_embd_gqa)            → K
//! rows [n_embd + n_embd_gqa .. + n_embd_gqa)      → V
//! ```
//!
//! confirmed by the `ggml_view_3d` offsets in
//! `src/models/gptneox.cpp` (`0*n_embd`, `1*n_embd`, `1*(n_embd + n_embd_gqa)`).
//! The HuggingFace checkpoint interleaves Q/K/V *per head*
//! (`[n_head, 3, head_dim, n_embd]`); `convert_hf_to_gguf.py`'s
//! `GPTNeoXModel.modify_tensors` de-interleaves it **at conversion time**, so a
//! GGUF loader must not re-permute anything.

use crate::traits::TensorNamePattern;

/// Every GGUF tensor name pattern a GPT-NeoX checkpoint ships.
///
/// `{i}` is the layer-index placeholder.  All patterns are required; see the
/// module documentation for the `flags = 0` citation.
pub fn gpt_neox_tensor_name_patterns() -> Vec<TensorNamePattern> {
    let mut patterns = Vec::with_capacity(4 + 12);

    // ── Global tensors ───────────────────────────────────────────────────────
    for (pattern, description) in [
        (
            "token_embd.weight",
            "Token embedding matrix [vocab_size, hidden_size]",
        ),
        ("output_norm.weight", "Final LayerNorm scale (gamma)"),
        ("output_norm.bias", "Final LayerNorm shift (beta)"),
        (
            "output.weight",
            "LM head / unembedding projection [vocab_size, hidden_size]",
        ),
    ] {
        patterns.push(TensorNamePattern {
            pattern: pattern.to_string(),
            description: description.to_string(),
            required: true,
        });
    }

    // ── Per-layer tensors ────────────────────────────────────────────────────
    for (pattern, description) in [
        (
            "blk.{i}.attn_norm.weight",
            "Pre-attention LayerNorm scale (gamma)",
        ),
        (
            "blk.{i}.attn_norm.bias",
            "Pre-attention LayerNorm shift (beta)",
        ),
        (
            "blk.{i}.attn_qkv.weight",
            "Fused Q/K/V projection [n_embd + 2*n_embd_gqa, n_embd] \
             (contiguous Q, K, V row blocks — never separate attn_q/attn_k/attn_v)",
        ),
        (
            "blk.{i}.attn_qkv.bias",
            "Fused Q/K/V bias [n_embd + 2*n_embd_gqa]",
        ),
        (
            "blk.{i}.attn_output.weight",
            "Attention output projection [n_embd, n_embd]",
        ),
        ("blk.{i}.attn_output.bias", "Attention output bias [n_embd]"),
        ("blk.{i}.ffn_norm.weight", "Pre-FFN LayerNorm scale (gamma)"),
        ("blk.{i}.ffn_norm.bias", "Pre-FFN LayerNorm shift (beta)"),
        (
            "blk.{i}.ffn_up.weight",
            "FFN up projection [n_ff, n_embd] (GELU, gate-free)",
        ),
        ("blk.{i}.ffn_up.bias", "FFN up bias [n_ff]"),
        (
            "blk.{i}.ffn_down.weight",
            "FFN down projection [n_embd, n_ff]",
        ),
        ("blk.{i}.ffn_down.bias", "FFN down bias [n_embd]"),
    ] {
        patterns.push(TensorNamePattern {
            pattern: pattern.to_string(),
            description: description.to_string(),
            required: true,
        });
    }

    patterns
}

/// Expand `{i}` in a pattern to a concrete layer index.
///
/// Convenience for validators and tests that want the literal tensor name a
/// checkpoint must contain.
pub fn expand_layer_pattern(pattern: &str, layer: usize) -> String {
    pattern.replace("{i}", &layer.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Defect N1: the fused-QKV name must be declared and the phantom
    /// `ln1`/`ln2`/`attn_q`/`attn_k`/`attn_v` names must be gone for good.
    #[test]
    fn gptneox_declares_fused_qkv_and_no_phantom_names() {
        let patterns = gpt_neox_tensor_name_patterns();
        let names: Vec<&str> = patterns.iter().map(|p| p.pattern.as_str()).collect();

        for required in [
            "token_embd.weight",
            "output_norm.weight",
            "output_norm.bias",
            "output.weight",
            "blk.{i}.attn_norm.weight",
            "blk.{i}.attn_norm.bias",
            "blk.{i}.attn_qkv.weight",
            "blk.{i}.attn_qkv.bias",
            "blk.{i}.attn_output.weight",
            "blk.{i}.attn_output.bias",
            "blk.{i}.ffn_norm.weight",
            "blk.{i}.ffn_norm.bias",
            "blk.{i}.ffn_up.weight",
            "blk.{i}.ffn_up.bias",
            "blk.{i}.ffn_down.weight",
            "blk.{i}.ffn_down.bias",
        ] {
            assert!(
                names.contains(&required),
                "GPT-NeoX must declare '{required}'; got {names:?}"
            );
        }

        for name in &names {
            for phantom in ["ln1", "ln2", "attn_q.", "attn_k.", "attn_v."] {
                assert!(
                    !name.contains(phantom),
                    "'{name}' contains '{phantom}', which never appears in a real GPT-NeoX GGUF"
                );
            }
        }
    }

    /// Every GPT-NeoX tensor is created with `flags = 0` in llama.cpp.
    #[test]
    fn gptneox_tensors_are_all_required() {
        for p in gpt_neox_tensor_name_patterns() {
            assert!(
                p.required,
                "'{}' must be required (llama.cpp passes flags = 0)",
                p.pattern
            );
            assert!(!p.description.is_empty(), "'{}' needs a doc", p.pattern);
        }
    }

    #[test]
    fn layer_pattern_expands() {
        assert_eq!(
            expand_layer_pattern("blk.{i}.attn_qkv.weight", 7),
            "blk.7.attn_qkv.weight"
        );
        assert_eq!(expand_layer_pattern("output.weight", 7), "output.weight");
    }
}

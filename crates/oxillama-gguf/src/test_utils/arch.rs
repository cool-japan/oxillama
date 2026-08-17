//! Architecture-specific synthetic GGUF builders.
//!
//! This module is split out from `test_utils` to keep each file under 2000
//! lines.  All items here are re-exported from the parent module so the
//! public API is unchanged.

use super::{build_gguf_v3, KvEntry, TensorDesc};

// ─── Qwen3 builder ────────────────────────────────────────────────────────────

/// Tensors for a minimal 1-layer Qwen3 model.
///
/// Qwen3 is structurally identical to LLaMA — same tensor names, same shapes.
/// The loader (`load_qwen3_from_gguf`) uses `load_quant_linear_with_bias` for
/// attn_q/k/v/output, but the bias tensors are optional (checked with
/// `model.file.tensors.contains()`), so we omit them here.
const QWEN3_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.attn_q.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_k.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.ffn_gate.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer Qwen3 model.
///
/// Uses the same tiny dimensions as [`super::build_minimal_llama_gguf`].
/// Qwen3 tensor names are identical to LLaMA; only `general.architecture`
/// differs.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_qwen3_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "qwen3"),
            KvEntry::U32("qwen3.embedding_length", 32),
            KvEntry::U32("qwen3.feed_forward_length", 64),
            KvEntry::U32("qwen3.block_count", 1),
            KvEntry::U32("qwen3.attention.head_count", 2),
            KvEntry::U32("qwen3.attention.head_count_kv", 2),
            KvEntry::U32("qwen3.context_length", 128),
            KvEntry::U32("qwen3.vocab_size", 32),
            KvEntry::F32("qwen3.rope.freq_base", 10000.0),
            KvEntry::Str("tokenizer.ggml.model", "qwen"),
        ],
        QWEN3_TENSORS,
    )
}

// ─── Mistral builder ──────────────────────────────────────────────────────────

/// Tensors for a minimal 1-layer Mistral model.
///
/// Mistral is identical to LLaMA in tensor names.  The loader
/// (`load_mistral_from_gguf`) uses `load_quant_linear` (no bias) for all
/// projection weights.
const MISTRAL_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.attn_q.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_k.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.ffn_gate.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer Mistral model.
///
/// Includes `attention.sliding_window = 64` which exercises the sliding-window
/// attention path in `MistralModel`.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_mistral_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "mistral"),
            KvEntry::U32("mistral.embedding_length", 32),
            KvEntry::U32("mistral.feed_forward_length", 64),
            KvEntry::U32("mistral.block_count", 1),
            KvEntry::U32("mistral.attention.head_count", 2),
            KvEntry::U32("mistral.attention.head_count_kv", 2),
            KvEntry::U32("mistral.context_length", 128),
            KvEntry::U32("mistral.vocab_size", 32),
            KvEntry::F32("mistral.rope.freq_base", 10000.0),
            KvEntry::U32("mistral.attention.sliding_window", 64),
            KvEntry::Str("tokenizer.ggml.model", "llama"),
        ],
        MISTRAL_TENSORS,
    )
}

// ─── Gemma builder ────────────────────────────────────────────────────────────

/// Tensors for a minimal 1-layer Gemma model.
///
/// Gemma adds optional `attn_post_norm.weight` and `ffn_post_norm.weight`
/// per block.  The loader uses `load_optional_rms_norm` so these are optional;
/// we include them to exercise the Gemma-2 post-norm code path.
/// The `output.weight` projection is also optional (weight-tied if absent);
/// we include it to exercise the explicit-output path.
const GEMMA_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.attn_post_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.ffn_post_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.attn_q.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_k.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.ffn_gate.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer Gemma model.
///
/// Includes `attention.logit_softcap` and `final_logit_softcap` to exercise
/// Gemma-2 soft-capping.  Also includes per-block `attn_post_norm.weight` and
/// `ffn_post_norm.weight` to cover the post-norm code path.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_gemma_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "gemma"),
            KvEntry::U32("gemma.embedding_length", 32),
            KvEntry::U32("gemma.feed_forward_length", 64),
            KvEntry::U32("gemma.block_count", 1),
            KvEntry::U32("gemma.attention.head_count", 2),
            KvEntry::U32("gemma.attention.head_count_kv", 2),
            KvEntry::U32("gemma.context_length", 128),
            KvEntry::U32("gemma.vocab_size", 32),
            KvEntry::F32("gemma.rope.freq_base", 10000.0),
            KvEntry::F32("gemma.attention.logit_softcap", 50.0),
            KvEntry::F32("gemma.final_logit_softcap", 30.0),
            KvEntry::Str("tokenizer.ggml.model", "llama"),
        ],
        GEMMA_TENSORS,
    )
}

// ─── Phi builder ──────────────────────────────────────────────────────────────

/// Tensors for a minimal 1-layer Phi (Phi-3) model.
///
/// Phi uses a merged QKV projection `blk.{i}.attn_qkv.weight` of shape
/// `[(num_heads + 2*num_kv_heads) * head_dim, hidden_size]`.
/// With heads=2, kv_heads=2, head_dim=16, hidden=32:
/// `(2 + 2*2) * 16 = 96` rows, so the tensor shape (GGUF order, in-features
/// first) is `[32, 96]` with 3072 elements.
const PHI_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // merged QKV: (num_heads + 2*kv_heads) * head_dim out-features = (2+4)*16 = 96
    TensorDesc {
        name: "blk.0.attn_qkv.weight",
        dims: &[32, 96],
        n_elements: 3072,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.ffn_gate.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer Phi-3 model.
///
/// Phi is the most architecturally distinct from LLaMA:
/// - Merged QKV (`attn_qkv.weight`) instead of separate q/k/v weights.
/// - Partial RoPE (`phi3.rope.partial_rotary_factor = 0.5`).
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_phi3_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "phi3"),
            KvEntry::U32("phi3.embedding_length", 32),
            KvEntry::U32("phi3.feed_forward_length", 64),
            KvEntry::U32("phi3.block_count", 1),
            KvEntry::U32("phi3.attention.head_count", 2),
            KvEntry::U32("phi3.attention.head_count_kv", 2),
            KvEntry::U32("phi3.context_length", 128),
            KvEntry::U32("phi3.vocab_size", 32),
            KvEntry::F32("phi3.rope.freq_base", 10000.0),
            KvEntry::F32("phi3.rope.partial_rotary_factor", 0.5),
            KvEntry::Str("tokenizer.ggml.model", "llama"),
        ],
        PHI_TENSORS,
    )
}

// ─── Command-R builder ────────────────────────────────────────────────────────

/// Tensors for a minimal 1-layer Command-R model.
///
/// Command-R's projection tensor names are identical to LLaMA, but the norm
/// topology is not: there is exactly ONE `attn_norm` per layer (a LayerNorm,
/// not RMSNorm) that feeds both attention and FFN, and no `ffn_norm` tensor
/// at all — `LLM_ARCH_COMMAND_R`'s tensor table in llama.cpp's
/// `src/llama-arch.cpp` never lists `FFN_NORM`. Optional Q/K-norm weights are
/// absent here (they are loaded conditionally with `.ok()`).
const COMMAND_R_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.attn_q.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_k.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.ffn_gate.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer Command-R model.
///
/// Includes `logit_scale = 0.0625` to exercise the logit-scaling path.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_command_r_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "command-r"),
            KvEntry::U32("command-r.embedding_length", 32),
            KvEntry::U32("command-r.feed_forward_length", 64),
            KvEntry::U32("command-r.block_count", 1),
            KvEntry::U32("command-r.attention.head_count", 2),
            KvEntry::U32("command-r.attention.head_count_kv", 2),
            KvEntry::U32("command-r.context_length", 128),
            KvEntry::U32("command-r.vocab_size", 32),
            KvEntry::F32("command-r.rope.freq_base", 10000.0),
            KvEntry::F32("command-r.logit_scale", 0.0625),
            KvEntry::Str("tokenizer.ggml.model", "llama"),
        ],
        COMMAND_R_TENSORS,
    )
}

// ─── StarCoder builder ────────────────────────────────────────────────────────

/// Tensors for a minimal 1-layer StarCoder (GPT-BigCode / MQA) model.
///
/// StarCoder uses:
/// - Absolute position embeddings (`position_embd.weight`)
/// - Fused QKV: `[(num_heads + 2) * head_dim, hidden_size]` =
///   `[(2 + 2) * 16, 32]` = `[64, 32]` = 2048 elements.
///   (MQA: 1 shared K and 1 shared V head, plus num_heads Q heads)
/// - `attn_output.weight` — the standard GGUF name used by every architecture
///   (`gguf-py`'s `TENSOR_NAMES[MODEL_TENSOR.ATTN_OUT] = "blk.{bid}.attn_output"`)
/// - Per-layer biases stored as 1-D F32 tensors
/// - LayerNorm with separate `.bias` tensors (not RMSNorm)
/// - `output_norm.bias` in addition to `output_norm.weight`
const STARCODER_TENSORS: &[TensorDesc] = &[
    // Token embeddings
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // Absolute position embeddings, GGUF order: [hidden_size, context_len]
    TensorDesc {
        name: "position_embd.weight",
        dims: &[32, 128],
        n_elements: 4096,
    },
    // Layer 0 — pre-attention LayerNorm
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.attn_norm.bias",
        dims: &[32],
        n_elements: 32,
    },
    // Fused QKV: (num_heads + 2) * head_dim = (2+2)*16 = 64 out-features
    TensorDesc {
        name: "blk.0.attn_qkv.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.attn_qkv.bias",
        dims: &[64],
        n_elements: 64,
    },
    // Attention output projection
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_output.bias",
        dims: &[32],
        n_elements: 32,
    },
    // Pre-FFN LayerNorm
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.ffn_norm.bias",
        dims: &[32],
        n_elements: 32,
    },
    // FFN up projection
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.bias",
        dims: &[64],
        n_elements: 64,
    },
    // FFN down projection
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.bias",
        dims: &[32],
        n_elements: 32,
    },
    // Final LayerNorm
    TensorDesc {
        name: "output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "output_norm.bias",
        dims: &[32],
        n_elements: 32,
    },
    // LM head
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer StarCoder model.
///
/// StarCoder (GPT-BigCode) is the most structurally distinct from LLaMA:
/// - Absolute position embeddings instead of RoPE.
/// - Multi-Query Attention (MQA): `num_kv_heads = 1`.
/// - Fused `attn_qkv.weight/bias` of shape `[(num_heads+2)*head_dim, hidden]`.
/// - LayerNorm (not RMSNorm) with separate bias tensors everywhere.
/// - GELU activation (gate-free FFN).
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_starcoder_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "starcoder"),
            KvEntry::U32("starcoder.embedding_length", 32),
            KvEntry::U32("starcoder.feed_forward_length", 64),
            KvEntry::U32("starcoder.block_count", 1),
            KvEntry::U32("starcoder.attention.head_count", 2),
            // MQA: 1 shared K/V head
            KvEntry::U32("starcoder.attention.head_count_kv", 1),
            KvEntry::U32("starcoder.context_length", 128),
            KvEntry::U32("starcoder.vocab_size", 32),
            KvEntry::Str("tokenizer.ggml.model", "gpt2"),
        ],
        STARCODER_TENSORS,
    )
}

// ─── LoRA adapter builder ─────────────────────────────────────────────────────

/// Tensors for a minimal LoRA adapter covering 3 layers of a 1-layer LLaMA model.
///
/// All tensors are F32, zero-initialised.
///
/// GGUF stores dimensions in column-major (fastest-changing-first) order:
/// - `lora_a` of math shape `[rank × in_features]` → `dims = [in_features, rank]`
/// - `lora_b` of math shape `[out_features × rank]` → `dims = [rank, out_features]`
///
/// Parameters:
/// - hidden_size  = 32
/// - rank         = 4
/// - intermediate_size = 64   (for ffn_gate only)
const LORA_TENSORS: &[TensorDesc] = &[
    // blk.0.attn_q — in=32, out=32, rank=4
    TensorDesc {
        name: "blk.0.attn_q.weight.lora_a",
        dims: &[32, 4],  // GGUF col-major: [in_features=32, rank=4]
        n_elements: 128, // 4 × 32
    },
    TensorDesc {
        name: "blk.0.attn_q.weight.lora_b",
        dims: &[4, 32],  // GGUF col-major: [rank=4, out_features=32]
        n_elements: 128, // 32 × 4
    },
    // blk.0.attn_v — in=32, out=32, rank=4
    TensorDesc {
        name: "blk.0.attn_v.weight.lora_a",
        dims: &[32, 4],
        n_elements: 128,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight.lora_b",
        dims: &[4, 32],
        n_elements: 128,
    },
    // blk.0.ffn_gate — in=32, out=64, rank=4
    TensorDesc {
        name: "blk.0.ffn_gate.weight.lora_a",
        dims: &[32, 4],
        n_elements: 128, // 4 × 32
    },
    TensorDesc {
        name: "blk.0.ffn_gate.weight.lora_b",
        dims: &[4, 64],  // GGUF col-major: [rank=4, out_features=64]
        n_elements: 256, // 64 × 4
    },
];

/// Build a minimal valid LoRA adapter GGUF v3 binary.
///
/// Contains 3 LoRA pairs for layer 0:
/// - `blk.0.attn_q.weight.lora_a/b`
/// - `blk.0.attn_v.weight.lora_a/b`
/// - `blk.0.ffn_gate.weight.lora_a/b`
///
/// Metadata: `lora.r = 4`, `lora.alpha = 8.0`, `general.architecture = "llama"`.
/// All tensors are F32, zero-initialised.
///
/// Dimension conventions (GGUF col-major, i.e. fastest dimension first):
/// - A matrices `[in_features=32, rank=4]`   → 128 f32 = 512 bytes each
/// - B matrices for attn_q/v `[rank=4, out_features=32]` → 128 f32 = 512 bytes
/// - B matrix  for ffn_gate `[rank=4, out_features=64]`  → 256 f32 = 1024 bytes
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_lora_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "llama"),
            KvEntry::U32("lora.r", 4),
            KvEntry::F32("lora.alpha", 8.0),
        ],
        LORA_TENSORS,
    )
}

// ─── DBRX builder ────────────────────────────────────────────────────────────

/// Tensor catalogue for a minimal 2-layer DBRX model.
///
/// Dimensions (small but structurally valid):
/// - hidden=32, heads=2, kv_heads=2, head_dim=16, vocab=32
/// - 4 MoE experts (small, not 16), top-2 from 4
const DBRX_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // Layer 0
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // Fused QKV: [in = hidden 32, out = hidden 32 + 2 * kv_dim 32] = [32, 96].
    TensorDesc {
        name: "blk.0.attn_qkv.weight",
        dims: &[32, 96],
        n_elements: 3072,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // DBRX has no `ffn_norm`; `attn_output_norm` occupies the pre-FFN slot.
    TensorDesc {
        name: "blk.0.attn_output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // Router: [n_experts=4, hidden=32]
    TensorDesc {
        name: "blk.0.ffn_gate_inp.weight",
        dims: &[32, 4],
        n_elements: 128,
    },
    // Stacked exps, GGUF `ne` order (fastest first): [in, out, n_expert].
    TensorDesc {
        name: "blk.0.ffn_gate_exps.weight",
        dims: &[32, 64, 4],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.0.ffn_up_exps.weight",
        dims: &[32, 64, 4],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.0.ffn_down_exps.weight",
        dims: &[64, 32, 4],
        n_elements: 8192,
    },
    // Layer 1
    TensorDesc {
        name: "blk.1.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // Fused QKV: [in = hidden 32, out = hidden 32 + 2 * kv_dim 32] = [32, 96].
    TensorDesc {
        name: "blk.1.attn_qkv.weight",
        dims: &[32, 96],
        n_elements: 3072,
    },
    TensorDesc {
        name: "blk.1.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // DBRX has no `ffn_norm`; `attn_output_norm` occupies the pre-FFN slot.
    TensorDesc {
        name: "blk.1.attn_output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.1.ffn_gate_inp.weight",
        dims: &[32, 4],
        n_elements: 128,
    },
    TensorDesc {
        name: "blk.1.ffn_gate_exps.weight",
        dims: &[32, 64, 4],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.1.ffn_up_exps.weight",
        dims: &[32, 64, 4],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.1.ffn_down_exps.weight",
        dims: &[64, 32, 4],
        n_elements: 8192,
    },
    // Output
    TensorDesc {
        name: "output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
];

/// Build a valid GGUF v3 binary for a minimal 2-layer DBRX model.
///
/// Uses 4 experts (not 16) for speed. The binary can be parsed with
/// [`crate::GgufModel::from_bytes`] and will satisfy structural tensor checks.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_dbrx_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "dbrx"),
            KvEntry::Str("general.name", "test-dbrx"),
            KvEntry::U32("dbrx.embedding_length", 32),
            KvEntry::U32("dbrx.feed_forward_length", 64),
            KvEntry::U32("dbrx.block_count", 2),
            KvEntry::U32("dbrx.attention.head_count", 2),
            KvEntry::U32("dbrx.attention.head_count_kv", 2),
            KvEntry::U32("dbrx.context_length", 128),
            KvEntry::U32("dbrx.vocab_size", 32),
            KvEntry::F32("dbrx.rope.freq_base", 10000.0),
            KvEntry::U32("dbrx.expert_count", 4),
            KvEntry::U32("dbrx.expert_used_count", 2),
            // `convert_hf_to_gguf.py::DbrxModel` writes both of these.
            KvEntry::F32("dbrx.attention.clamp_kqv", 8.0),
            KvEntry::F32("dbrx.attention.layer_norm_epsilon", 1e-5),
            KvEntry::Str("tokenizer.ggml.model", "dbrx"),
        ],
        DBRX_TENSORS,
    )
}

// ─── Grok builder ─────────────────────────────────────────────────────────────

/// Tensor catalogue for a minimal 2-layer Grok-1 model.
///
/// Uses 2 experts (top-2 from 2) for speed. Real Grok-1 has 8 experts.
const GROK_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // Layer 0
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.attn_q.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_k.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // `LLM_TENSOR_ATTN_OUT_NORM` — post-attention, applied before the residual.
    TensorDesc {
        name: "blk.0.attn_output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // Router: [n_experts=2, hidden=32]
    TensorDesc {
        name: "blk.0.ffn_gate_inp.weight",
        dims: &[32, 2],
        n_elements: 64,
    },
    // Stacked exps, GGUF `ne` order (fastest first): [in, out, n_expert].
    TensorDesc {
        name: "blk.0.ffn_gate_exps.weight",
        dims: &[32, 64, 2],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.0.ffn_up_exps.weight",
        dims: &[32, 64, 2],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.0.ffn_down_exps.weight",
        dims: &[64, 32, 2],
        n_elements: 4096,
    },
    // `LLM_TENSOR_LAYER_OUT_NORM` — post-FFN, applied before the residual.
    TensorDesc {
        name: "blk.0.layer_output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // Layer 1
    TensorDesc {
        name: "blk.1.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.1.attn_q.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.1.attn_k.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.1.attn_v.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.1.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // `LLM_TENSOR_ATTN_OUT_NORM` — post-attention, applied before the residual.
    TensorDesc {
        name: "blk.1.attn_output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.1.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.1.ffn_gate_inp.weight",
        dims: &[32, 2],
        n_elements: 64,
    },
    TensorDesc {
        name: "blk.1.ffn_gate_exps.weight",
        dims: &[32, 64, 2],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.1.ffn_up_exps.weight",
        dims: &[32, 64, 2],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.1.ffn_down_exps.weight",
        dims: &[64, 32, 2],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.1.layer_output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // Output
    TensorDesc {
        name: "output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
];

/// Build a valid GGUF v3 binary for a minimal 2-layer Grok-1 model.
///
/// Uses 2 experts for speed. The Grok-1 rope_theta of 1_000_000 is encoded.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_grok_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "grok"),
            KvEntry::Str("general.name", "test-grok"),
            KvEntry::U32("grok.embedding_length", 32),
            KvEntry::U32("grok.feed_forward_length", 64),
            KvEntry::U32("grok.block_count", 2),
            KvEntry::U32("grok.attention.head_count", 2),
            KvEntry::U32("grok.attention.head_count_kv", 2),
            KvEntry::U32("grok.context_length", 128),
            KvEntry::U32("grok.vocab_size", 32),
            KvEntry::F32("grok.rope.freq_base", 1_000_000.0),
            KvEntry::U32("grok.expert_count", 2),
            KvEntry::U32("grok.expert_used_count", 2),
            KvEntry::Str("tokenizer.ggml.model", "grok"),
        ],
        GROK_TENSORS,
    )
}

// ─── Mamba-2 builder ──────────────────────────────────────────────────────────

/// Tensor catalogue for a minimal 1-layer Mamba-2 model.
///
/// Mirrors the `LLM_ARCH_MAMBA2` tensor set of `llama.cpp`
/// (`src/llama-arch.cpp:1388`, shapes at `src/llama-model.cpp:4585`).
///
/// Dimensions:
/// - n_embd=16, d_inner=32, d_state=8, d_conv=4, n_head=4, n_group=2,
///   n_layer=1, vocab=256
/// - Derived: conv_dim = d_inner + 2*n_group*d_state = 64,
///   d_in_proj = 2*d_inner + 2*n_group*d_state + n_head = 100
const MAMBA2_TENSORS: &[TensorDesc] = &[
    // Token embedding: [vocab=256, d_model=16]
    TensorDesc {
        name: "token_embd.weight",
        dims: &[16, 256],
        n_elements: 4096,
    },
    // Layer 0: norm
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[16],
        n_elements: 16,
    },
    // Fused zxBCdt projection: {n_embd=16, d_in_proj=100}
    // d_in_proj = 2*d_inner + 2*n_group*d_state + n_head = 64 + 32 + 4
    TensorDesc {
        name: "blk.0.ssm_in.weight",
        dims: &[16, 100],
        n_elements: 1600,
    },
    // Depthwise conv: {d_conv=4, conv_dim=64}
    // conv_dim = d_inner + 2*n_group*d_state = 32 + 32 (x, B and C share it)
    TensorDesc {
        name: "blk.0.ssm_conv1d.weight",
        dims: &[4, 64],
        n_elements: 256,
    },
    TensorDesc {
        name: "blk.0.ssm_conv1d.bias",
        dims: &[64],
        n_elements: 64,
    },
    // Per-head Δ bias: {n_head=4}. MAMBA2 has no ssm_dt *weight*.
    TensorDesc {
        name: "blk.0.ssm_dt.bias",
        dims: &[4],
        n_elements: 4,
    },
    // Per-head A = -exp(A_log): {1, n_head=4}
    TensorDesc {
        name: "blk.0.ssm_a",
        dims: &[1, 4],
        n_elements: 4,
    },
    // Per-head D (skip): {1, n_head=4}
    TensorDesc {
        name: "blk.0.ssm_d",
        dims: &[1, 4],
        n_elements: 4,
    },
    // Gated group-RMSNorm: {d_inner/n_group=16, n_group=2}
    TensorDesc {
        name: "blk.0.ssm_norm.weight",
        dims: &[16, 2],
        n_elements: 32,
    },
    // Output projection: {d_inner=32, n_embd=16}
    TensorDesc {
        name: "blk.0.ssm_out.weight",
        dims: &[32, 16],
        n_elements: 512,
    },
    // Output head
    TensorDesc {
        name: "output_norm.weight",
        dims: &[16],
        n_elements: 16,
    },
    TensorDesc {
        name: "output.weight",
        dims: &[16, 256],
        n_elements: 4096,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer Mamba-2 model.
///
/// Emits the real `LLM_ARCH_MAMBA2` tensor set and the canonical
/// `{arch}.ssm.*` metadata keys, so a loader that passes against this fixture
/// also passes against a converted checkpoint.  See `MAMBA2_TENSORS` for the
/// dimensions.  All weights are F32, zero-initialised.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_mamba2_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "mamba2"),
            KvEntry::Str("general.name", "test-mamba2"),
            KvEntry::U32("mamba2.embedding_length", 16),
            KvEntry::U32("mamba2.block_count", 1),
            KvEntry::U32("mamba2.context_length", 512),
            KvEntry::U32("mamba2.ssm.conv_kernel", 4),
            KvEntry::U32("mamba2.ssm.inner_size", 32),
            KvEntry::U32("mamba2.ssm.state_size", 8),
            KvEntry::U32("mamba2.ssm.time_step_rank", 4),
            KvEntry::U32("mamba2.ssm.group_count", 2),
            KvEntry::F32("mamba2.attention.layer_norm_rms_epsilon", 1e-5),
            KvEntry::U32("mamba2.vocab_size", 256),
            KvEntry::Str("tokenizer.ggml.model", "mamba2"),
        ],
        MAMBA2_TENSORS,
    )
}

// ─── Qwen2-VL builder ─────────────────────────────────────────────────────────

/// Tensor catalogue for a minimal 1-layer Qwen2-VL model.
///
/// Dimensions:
/// - LLM: hidden=32, heads=2, kv_heads=2, head_dim=16, ffn=64, vocab=32
/// - Vision encoder: vis_hidden=8, patch_size=4, num_heads=2
/// - MM merger: mm.0.weight [32, 4*8=32]
/// - v.patch_embd.weight [8, 4*4*3=48]
/// - v.post_ln.weight [8]
const QWEN2VL_TENSORS: &[TensorDesc] = &[
    // ── LLM backbone ─────────────────────────────────────────────────────────
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.attn_q.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_k.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.ffn_gate.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "output_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // ── MM Merger ─────────────────────────────────────────────────────────────
    // mm.0.weight: [llm_hidden=32, 4*vis_hidden=4*8=32]
    TensorDesc {
        name: "mm.0.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // ── Vision encoder ────────────────────────────────────────────────────────
    // v.patch_embd.weight: [vis_hidden=8, patch_size²×3=4×4×3=48]
    TensorDesc {
        name: "v.patch_embd.weight",
        dims: &[48, 8],
        n_elements: 384,
    },
    // v.post_ln.weight: [vis_hidden=8]
    TensorDesc {
        name: "v.post_ln.weight",
        dims: &[8],
        n_elements: 8,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer Qwen2-VL model.
///
/// Includes both LLM backbone tensors and vision encoder tensors (`v.*` prefix)
/// plus the MM merger weight (`mm.0.weight`).
///
/// # Dimensions
///
/// | Component      | Hyper-parameter   | Value |
/// |----------------|-------------------|-------|
/// | LLM backbone   | `hidden_size`     | 32    |
/// |                | `heads`           | 2     |
/// |                | `kv_heads`        | 2     |
/// |                | `head_dim`        | 16    |
/// |                | `layers`          | 1     |
/// |                | `vocab_size`      | 32    |
/// |                | `ffn_size`        | 64    |
/// |                | `context_len`     | 128   |
/// | Vision encoder | `vis_hidden`      | 8     |
/// |                | `patch_size`      | 4     |
/// |                | `vis_num_heads`   | 2     |
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_qwen2vl_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "qwen2vl"),
            KvEntry::Str("general.name", "test-qwen2vl"),
            KvEntry::U32("qwen2vl.embedding_length", 32),
            KvEntry::U32("qwen2vl.feed_forward_length", 64),
            KvEntry::U32("qwen2vl.block_count", 1),
            KvEntry::U32("qwen2vl.attention.head_count", 2),
            KvEntry::U32("qwen2vl.attention.head_count_kv", 2),
            KvEntry::U32("qwen2vl.context_length", 128),
            KvEntry::U32("qwen2vl.vocab_size", 32),
            KvEntry::F32("qwen2vl.rope.freq_base", 10000.0),
            KvEntry::F32("qwen2vl.attention.layer_norm_rms_epsilon", 1e-5),
            // Vision encoder metadata
            KvEntry::U32("clip.vision.image_size", 16),
            KvEntry::U32("clip.vision.patch_size", 4),
            KvEntry::U32("clip.vision.embedding_length", 8),
            KvEntry::U32("clip.vision.attention.head_count", 2),
            KvEntry::U32("clip.vision.n_layers", 0),
            KvEntry::U32("clip.vision.attention.window_size", 8),
            KvEntry::Str("tokenizer.ggml.model", "qwen"),
        ],
        QWEN2VL_TENSORS,
    )
}

// ─── BLOOM builder ────────────────────────────────────────────────────────────

/// Tensor catalogue for a minimal 1-layer BLOOM model.
///
/// BLOOM uses:
/// - Biased LayerNorm instead of RMSNorm
/// - Fused QKV: `[(num_heads * 3) * head_dim, hidden_size]` with bias
/// - All linear layers have bias vectors
/// - No RoPE (uses ALiBi bias instead)
/// - Standard GELU FFN (not SwiGLU)
///
/// Dimensions: hidden=64, heads=8, head_dim=8, vocab=32, ffn=256 (4×hidden), layers=1
const BLOOM_TENSORS: &[TensorDesc] = &[
    // Token embedding: [vocab=32, hidden=64]
    TensorDesc {
        name: "token_embd.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    // Final LayerNorm
    TensorDesc {
        name: "output_norm.weight",
        dims: &[64],
        n_elements: 64,
    },
    TensorDesc {
        name: "output_norm.bias",
        dims: &[64],
        n_elements: 64,
    },
    // Layer 0: pre-attention LayerNorm
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[64],
        n_elements: 64,
    },
    TensorDesc {
        name: "blk.0.attn_norm.bias",
        dims: &[64],
        n_elements: 64,
    },
    // Fused QKV: (num_heads * 3) * head_dim = 8*3*8 = 192 rows, hidden=64 cols
    TensorDesc {
        name: "blk.0.attn_qkv.weight",
        dims: &[192, 64],
        n_elements: 12288,
    },
    TensorDesc {
        name: "blk.0.attn_qkv.bias",
        dims: &[192],
        n_elements: 192,
    },
    // Attention output projection: [hidden=64, num_heads*head_dim=64]
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[64, 64],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.0.attn_output.bias",
        dims: &[64],
        n_elements: 64,
    },
    // Pre-FFN LayerNorm
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[64],
        n_elements: 64,
    },
    TensorDesc {
        name: "blk.0.ffn_norm.bias",
        dims: &[64],
        n_elements: 64,
    },
    // FFN up (W1): [ffn=256, hidden=64]
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[256, 64],
        n_elements: 16384,
    },
    TensorDesc {
        name: "blk.0.ffn_up.bias",
        dims: &[256],
        n_elements: 256,
    },
    // FFN down (W2): [hidden=64, ffn=256]
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[64, 256],
        n_elements: 16384,
    },
    TensorDesc {
        name: "blk.0.ffn_down.bias",
        dims: &[64],
        n_elements: 64,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer BLOOM model.
///
/// | Hyper-parameter | Value |
/// |-----------------|-------|
/// | `hidden_size`   | 64    |
/// | `heads`         | 8     |
/// | `head_dim`      | 8     |
/// | `layers`        | 1     |
/// | `vocab_size`    | 32    |
/// | `ffn_size`      | 256   |
/// | `context_len`   | 128   |
///
/// BLOOM uses standard LayerNorm (with bias) and ALiBi positional biases.
/// No RoPE. No separate `output.weight` tensor (weight-tied to `token_embd`).
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_bloom_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "bloom"),
            KvEntry::Str("general.name", "test-bloom"),
            KvEntry::U32("bloom.embedding_length", 64),
            KvEntry::U32("bloom.feed_forward_length", 256),
            KvEntry::U32("bloom.block_count", 1),
            KvEntry::U32("bloom.attention.head_count", 8),
            KvEntry::U32("bloom.attention.head_count_kv", 8),
            KvEntry::U32("bloom.context_length", 128),
            KvEntry::U32("bloom.vocab_size", 32),
            KvEntry::F32("bloom.attention.layer_norm_rms_epsilon", 1e-5),
            KvEntry::Str("tokenizer.ggml.model", "bloom"),
        ],
        BLOOM_TENSORS,
    )
}

// ─── Phi-MoE builder ──────────────────────────────────────────────────────────

/// Tensor catalogue for a minimal 1-layer Phi-3.5-MoE model.
///
/// Phi-MoE uses Phi-3 style attention (fused QKV, partial RoPE, GQA)
/// combined with a sparse MoE FFN:
/// - 4 experts (reduced from 16 for test speed), top-2 routing
/// - RMSNorm (not LayerNorm)
/// - No bias on attention projections (like Phi-3)
///
/// Dimensions: hidden=64, attn_heads=8, kv_heads=4, head_dim=8, vocab=32
///             intermediate=64, num_experts=4, top-2 routing
const PHI_MOE_TENSORS: &[TensorDesc] = &[
    // Token embedding: [vocab=32, hidden=64]
    TensorDesc {
        name: "token_embd.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    // Final RMSNorm
    TensorDesc {
        name: "output_norm.weight",
        dims: &[64],
        n_elements: 64,
    },
    // LM head
    TensorDesc {
        name: "output.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    // Layer 0: pre-attention RMSNorm
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[64],
        n_elements: 64,
    },
    // Fused QKV: (num_heads + 2*kv_heads) * head_dim = (8+2*4)*8 = 128 rows
    TensorDesc {
        name: "blk.0.attn_qkv.weight",
        dims: &[128, 64],
        n_elements: 8192,
    },
    // Attention output projection: [hidden=64, num_heads*head_dim=64]
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[64, 64],
        n_elements: 4096,
    },
    // Pre-FFN RMSNorm
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[64],
        n_elements: 64,
    },
    // Router: [num_experts=4, hidden=64]
    TensorDesc {
        name: "blk.0.ffn_gate_inp.weight",
        dims: &[64, 4],
        n_elements: 256,
    },
    // Stacked gate exps: [num_experts=4, ffn=64, hidden=64]
    TensorDesc {
        name: "blk.0.ffn_gate_exps.weight",
        dims: &[4, 64, 64],
        n_elements: 16384,
    },
    // Stacked up exps: [num_experts=4, ffn=64, hidden=64]
    TensorDesc {
        name: "blk.0.ffn_up_exps.weight",
        dims: &[4, 64, 64],
        n_elements: 16384,
    },
    // Stacked down exps: [num_experts=4, hidden=64, ffn=64]
    TensorDesc {
        name: "blk.0.ffn_down_exps.weight",
        dims: &[4, 64, 64],
        n_elements: 16384,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer Phi-3.5-MoE model.
///
/// | Hyper-parameter          | Value |
/// |--------------------------|-------|
/// | `hidden_size`            | 64    |
/// | `attn_heads`             | 8     |
/// | `kv_heads`               | 4     |
/// | `head_dim`               | 8     |
/// | `layers`                 | 1     |
/// | `vocab_size`             | 32    |
/// | `intermediate_size`      | 64    |
/// | `num_experts`            | 4     |
/// | `num_experts_per_tok`    | 2     |
/// | `partial_rotary_factor`  | 0.5   |
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_phi_moe_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "phimoe"),
            KvEntry::Str("general.name", "test-phimoe"),
            KvEntry::U32("phimoe.embedding_length", 64),
            KvEntry::U32("phimoe.feed_forward_length", 64),
            KvEntry::U32("phimoe.block_count", 1),
            KvEntry::U32("phimoe.attention.head_count", 8),
            KvEntry::U32("phimoe.attention.head_count_kv", 4),
            KvEntry::U32("phimoe.context_length", 128),
            KvEntry::U32("phimoe.vocab_size", 32),
            KvEntry::F32("phimoe.rope.freq_base", 10000.0),
            KvEntry::F32("phimoe.attention.layer_norm_rms_epsilon", 1e-5),
            KvEntry::F32("phimoe.rope.partial_rotary_factor", 0.5),
            KvEntry::U32("phimoe.expert_count", 4),
            KvEntry::U32("phimoe.expert_used_count", 2),
            KvEntry::Str("tokenizer.ggml.model", "phi"),
        ],
        PHI_MOE_TENSORS,
    )
}

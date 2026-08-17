//! DeepSeek-V2 / V2-Lite / V3 synthetic GGUF builders.
//!
//! Split out of `arch.rs` to keep that file under the workspace's
//! 2000-line-per-file limit — the same reason `arch_ext` exists.  Three agents
//! append to `arch.rs` concurrently, so growing it further is not an option.
//!
//! All three fixtures emit the **real** `LLM_ARCH_DEEPSEEK2` tensor names from
//! `llama_model::load_tensors`.  The previous fixture invented
//! `blk.N.ffn_exp.{e}.*` / `blk.N.ffn_shared_exp.{e}.*`, names that appear
//! nowhere in llama.cpp or `gguf-py`, which is why the loader written against it
//! could not read a real checkpoint.

use super::{build_gguf_v3, KvEntry, TensorDesc};

// ─── DeepSeek-V2 builder ──────────────────────────────────────────────────────

/// Tensor catalogue for a minimal 2-layer DeepSeek-V2 model.
///
/// Uses the **real** `LLM_ARCH_DEEPSEEK2` tensor names from
/// `llama_model::load_tensors`, not the `blk.N.ffn_exp.{e}.*` /
/// `blk.N.ffn_shared_exp.{e}.*` spelling this fixture used to invent — those
/// names appear nowhere in llama.cpp or `gguf-py`, so a loader written against
/// the old fixture could not read a single real DeepSeek checkpoint.
///
/// Geometry: hidden = 32, heads = 2, `q_lora_rank` = 8, `kv_lora_rank` = 8,
/// `key_length` = 8, `rope.dimension_count` = 4 (so `qk_nope` = 4, `qk_rope` = 4,
/// `v_head_dim` = 4), dense `n_ff` = 64, expert `n_ff_exp` = 16, 2 routed
/// experts, 1 shared expert.  Layer 0 is dense
/// (`leading_dense_block_count = 1`), layer 1 is MoE.
const DEEPSEEK_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // ── Layer 0: MLA + dense FFN ──
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.0.attn_q_a.weight",
        dims: &[32, 8],
        n_elements: 256,
    },
    TensorDesc {
        name: "blk.0.attn_q_a_norm.weight",
        dims: &[8],
        n_elements: 8,
    },
    TensorDesc {
        name: "blk.0.attn_q_b.weight",
        dims: &[8, 16],
        n_elements: 128,
    },
    TensorDesc {
        name: "blk.0.attn_kv_a_mqa.weight",
        dims: &[32, 12],
        n_elements: 384,
    },
    TensorDesc {
        name: "blk.0.attn_kv_a_norm.weight",
        dims: &[8],
        n_elements: 8,
    },
    TensorDesc {
        name: "blk.0.attn_kv_b.weight",
        dims: &[8, 16],
        n_elements: 128,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[8, 32],
        n_elements: 256,
    },
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
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
    // ── Layer 1: MLA + routed MoE + shared expert ──
    TensorDesc {
        name: "blk.1.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    TensorDesc {
        name: "blk.1.attn_q_a.weight",
        dims: &[32, 8],
        n_elements: 256,
    },
    TensorDesc {
        name: "blk.1.attn_q_a_norm.weight",
        dims: &[8],
        n_elements: 8,
    },
    TensorDesc {
        name: "blk.1.attn_q_b.weight",
        dims: &[8, 16],
        n_elements: 128,
    },
    TensorDesc {
        name: "blk.1.attn_kv_a_mqa.weight",
        dims: &[32, 12],
        n_elements: 384,
    },
    TensorDesc {
        name: "blk.1.attn_kv_a_norm.weight",
        dims: &[8],
        n_elements: 8,
    },
    TensorDesc {
        name: "blk.1.attn_kv_b.weight",
        dims: &[8, 16],
        n_elements: 128,
    },
    TensorDesc {
        name: "blk.1.attn_output.weight",
        dims: &[8, 32],
        n_elements: 256,
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
        dims: &[32, 16, 2],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.1.ffn_up_exps.weight",
        dims: &[32, 16, 2],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.1.ffn_down_exps.weight",
        dims: &[16, 32, 2],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.1.ffn_gate_shexp.weight",
        dims: &[32, 16],
        n_elements: 512,
    },
    TensorDesc {
        name: "blk.1.ffn_up_shexp.weight",
        dims: &[32, 16],
        n_elements: 512,
    },
    TensorDesc {
        name: "blk.1.ffn_down_shexp.weight",
        dims: &[16, 32],
        n_elements: 512,
    },
    // ── Output ──
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

/// Build a valid GGUF v3 binary for a minimal 2-layer DeepSeek-V2 model.
///
/// The binary includes all required MLA and MoE metadata keys prefixed with
/// `deepseek2.*`. All tensors are F32 and zero-initialised.
///
/// The resulting bytes can be parsed with [`crate::GgufModel::from_bytes`]
/// and verified to contain all expected tensor names.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_deepseek_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            // General
            KvEntry::Str("general.architecture", "deepseek2"),
            KvEntry::Str("general.name", "test-deepseek"),
            // Base model hyperparameters
            KvEntry::U32("deepseek2.embedding_length", 32),
            KvEntry::U32("deepseek2.feed_forward_length", 64),
            KvEntry::U32("deepseek2.block_count", 2),
            KvEntry::U32("deepseek2.attention.head_count", 2),
            KvEntry::U32("deepseek2.attention.head_count_kv", 2),
            KvEntry::U32("deepseek2.context_length", 128),
            KvEntry::U32("deepseek2.vocab_size", 32),
            KvEntry::F32("deepseek2.rope.freq_base", 10000.0),
            KvEntry::F32("deepseek2.attention.layer_norm_rms_epsilon", 1e-5),
            // MLA hyperparameters
            KvEntry::U32("deepseek2.attention.q_lora_rank", 8),
            KvEntry::U32("deepseek2.attention.kv_lora_rank", 8),
            // `add_key_length(qk_nope + qk_rope)` and
            // `add_rope_dimension_count(qk_rope)` — the keys the converter
            // actually writes.  `deepseek2.attention.rope_head_dim` does not
            // exist in gguf-py and used to be invented here, which hid the fact
            // that `qk_nope_head_dim` was being read straight off `key_length`.
            KvEntry::U32("deepseek2.attention.key_length", 8),
            KvEntry::U32("deepseek2.rope.dimension_count", 4),
            KvEntry::U32("deepseek2.attention.value_length", 4),
            // MoE hyperparameters
            KvEntry::U32("deepseek2.expert_count", 2),
            KvEntry::U32("deepseek2.expert_used_count", 1),
            KvEntry::U32("deepseek2.expert_shared_count", 1),
            KvEntry::U32("deepseek2.expert_feed_forward_length", 16),
            KvEntry::U32("deepseek2.expert_shared_feed_forward_length", 16),
            KvEntry::U32("deepseek2.leading_dense_block_count", 1),
            KvEntry::F32("deepseek2.expert_weights_scale", 1.0),
            // Tokenizer stub
            KvEntry::Str("tokenizer.ggml.model", "deepseek"),
        ],
        DEEPSEEK_TENSORS,
    )
}

// ─── DeepSeek-V3 builder ──────────────────────────────────────────────────────

/// Build a valid GGUF v3 binary for a minimal 2-layer DeepSeek-V3 model.
///
/// Same geometry as [`build_minimal_deepseek_gguf`], plus everything that makes
/// a checkpoint V3 rather than V2 and that this crate previously ignored:
///
/// * `blk.1.exp_probs_b.bias` — the load-balancing selection bias.  llama.cpp
///   writes it with a **`.bias`** suffix (`tn(LLM_TENSOR_FFN_EXP_PROBS_B,
///   "bias", i)`); the old fixture used `.weight`.
/// * `deepseek2.expert_gating_func = 2` (sigmoid).
/// * `deepseek2.expert_weights_norm = true` — V2 ships `false`.
/// * `deepseek2.expert_group_count` / `expert_group_used_count` — group-limited
///   routing, which had no implementation at all.
/// * `deepseek2.expert_weights_scale = 2.5`.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_deepseek_v3_gguf() -> Vec<u8> {
    const DEEPSEEK_V3_TENSORS: &[TensorDesc] = &[
        TensorDesc {
            name: "token_embd.weight",
            dims: &[32, 32],
            n_elements: 1024,
        },
        // ── Layer 0: MLA + dense FFN ──
        TensorDesc {
            name: "blk.0.attn_norm.weight",
            dims: &[32],
            n_elements: 32,
        },
        TensorDesc {
            name: "blk.0.attn_q_a.weight",
            dims: &[32, 8],
            n_elements: 256,
        },
        TensorDesc {
            name: "blk.0.attn_q_a_norm.weight",
            dims: &[8],
            n_elements: 8,
        },
        TensorDesc {
            name: "blk.0.attn_q_b.weight",
            dims: &[8, 16],
            n_elements: 128,
        },
        TensorDesc {
            name: "blk.0.attn_kv_a_mqa.weight",
            dims: &[32, 12],
            n_elements: 384,
        },
        TensorDesc {
            name: "blk.0.attn_kv_a_norm.weight",
            dims: &[8],
            n_elements: 8,
        },
        TensorDesc {
            name: "blk.0.attn_kv_b.weight",
            dims: &[8, 16],
            n_elements: 128,
        },
        TensorDesc {
            name: "blk.0.attn_output.weight",
            dims: &[8, 32],
            n_elements: 256,
        },
        TensorDesc {
            name: "blk.0.ffn_norm.weight",
            dims: &[32],
            n_elements: 32,
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
        // ── Layer 1: MLA + routed MoE (sigmoid + bias) + shared expert ──
        TensorDesc {
            name: "blk.1.attn_norm.weight",
            dims: &[32],
            n_elements: 32,
        },
        TensorDesc {
            name: "blk.1.attn_q_a.weight",
            dims: &[32, 8],
            n_elements: 256,
        },
        TensorDesc {
            name: "blk.1.attn_q_a_norm.weight",
            dims: &[8],
            n_elements: 8,
        },
        TensorDesc {
            name: "blk.1.attn_q_b.weight",
            dims: &[8, 16],
            n_elements: 128,
        },
        TensorDesc {
            name: "blk.1.attn_kv_a_mqa.weight",
            dims: &[32, 12],
            n_elements: 384,
        },
        TensorDesc {
            name: "blk.1.attn_kv_a_norm.weight",
            dims: &[8],
            n_elements: 8,
        },
        TensorDesc {
            name: "blk.1.attn_kv_b.weight",
            dims: &[8, 16],
            n_elements: 128,
        },
        TensorDesc {
            name: "blk.1.attn_output.weight",
            dims: &[8, 32],
            n_elements: 256,
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
            name: "blk.1.exp_probs_b.bias",
            dims: &[2],
            n_elements: 2,
        },
        TensorDesc {
            name: "blk.1.ffn_gate_exps.weight",
            dims: &[32, 16, 2],
            n_elements: 1024,
        },
        TensorDesc {
            name: "blk.1.ffn_up_exps.weight",
            dims: &[32, 16, 2],
            n_elements: 1024,
        },
        TensorDesc {
            name: "blk.1.ffn_down_exps.weight",
            dims: &[16, 32, 2],
            n_elements: 1024,
        },
        TensorDesc {
            name: "blk.1.ffn_gate_shexp.weight",
            dims: &[32, 16],
            n_elements: 512,
        },
        TensorDesc {
            name: "blk.1.ffn_up_shexp.weight",
            dims: &[32, 16],
            n_elements: 512,
        },
        TensorDesc {
            name: "blk.1.ffn_down_shexp.weight",
            dims: &[16, 32],
            n_elements: 512,
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

    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "deepseek2"),
            KvEntry::Str("general.name", "test-deepseek-v3"),
            KvEntry::U32("deepseek2.embedding_length", 32),
            KvEntry::U32("deepseek2.feed_forward_length", 64),
            KvEntry::U32("deepseek2.block_count", 2),
            KvEntry::U32("deepseek2.attention.head_count", 2),
            KvEntry::U32("deepseek2.attention.head_count_kv", 2),
            KvEntry::U32("deepseek2.context_length", 128),
            KvEntry::U32("deepseek2.vocab_size", 32),
            KvEntry::F32("deepseek2.rope.freq_base", 10000.0),
            KvEntry::F32("deepseek2.attention.layer_norm_rms_epsilon", 1e-5),
            KvEntry::U32("deepseek2.attention.q_lora_rank", 8),
            KvEntry::U32("deepseek2.attention.kv_lora_rank", 8),
            KvEntry::U32("deepseek2.attention.key_length", 8),
            KvEntry::U32("deepseek2.rope.dimension_count", 4),
            KvEntry::U32("deepseek2.attention.value_length", 4),
            KvEntry::U32("deepseek2.expert_count", 2),
            KvEntry::U32("deepseek2.expert_used_count", 1),
            KvEntry::U32("deepseek2.expert_shared_count", 1),
            KvEntry::U32("deepseek2.expert_feed_forward_length", 16),
            KvEntry::U32("deepseek2.expert_shared_feed_forward_length", 16),
            KvEntry::U32("deepseek2.leading_dense_block_count", 1),
            // ── V3-only routing ──
            KvEntry::U32("deepseek2.expert_gating_func", 2),
            // `KvEntry` has no `Bool` variant (test_utils/mod.rs is owned
            // elsewhere); the loader accepts the integer encoding too.
            KvEntry::U32("deepseek2.expert_weights_norm", 1),
            KvEntry::U32("deepseek2.expert_group_count", 2),
            KvEntry::U32("deepseek2.expert_group_used_count", 1),
            KvEntry::F32("deepseek2.expert_weights_scale", 2.5),
            KvEntry::Str("tokenizer.ggml.model", "deepseek"),
        ],
        DEEPSEEK_V3_TENSORS,
    )
}

// ─── DeepSeek-V2-Lite builder ────────────────────────────────────────────────

/// Build a valid GGUF v3 binary for a minimal 1-layer **DeepSeek-V2-Lite** model.
///
/// Lite variants (DeepSeek-V2-Lite, DeepSeek-Coder-V2-Lite) ship
/// `q_lora_rank = null` and therefore a single `blk.N.attn_q.weight` instead of
/// `attn_q_a` + `attn_q_a_norm` + `attn_q_b`.  `llm_build_deepseek2` branches on
/// exactly that (`const bool is_lite = model.layers[il].wq;`).
///
/// No fixture covered this shape, and the loader unconditionally demanded the
/// low-rank tensors — so these checkpoints could not be loaded at all.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_deepseek_lite_gguf() -> Vec<u8> {
    const DEEPSEEK_LITE_TENSORS: &[TensorDesc] = &[
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
            dims: &[32, 16],
            n_elements: 512,
        },
        TensorDesc {
            name: "blk.0.attn_kv_a_mqa.weight",
            dims: &[32, 12],
            n_elements: 384,
        },
        TensorDesc {
            name: "blk.0.attn_kv_a_norm.weight",
            dims: &[8],
            n_elements: 8,
        },
        TensorDesc {
            name: "blk.0.attn_kv_b.weight",
            dims: &[8, 16],
            n_elements: 128,
        },
        TensorDesc {
            name: "blk.0.attn_output.weight",
            dims: &[8, 32],
            n_elements: 256,
        },
        TensorDesc {
            name: "blk.0.ffn_norm.weight",
            dims: &[32],
            n_elements: 32,
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

    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "deepseek2"),
            KvEntry::Str("general.name", "test-deepseek-v2-lite"),
            KvEntry::U32("deepseek2.embedding_length", 32),
            KvEntry::U32("deepseek2.feed_forward_length", 64),
            KvEntry::U32("deepseek2.block_count", 1),
            KvEntry::U32("deepseek2.attention.head_count", 2),
            KvEntry::U32("deepseek2.attention.head_count_kv", 2),
            KvEntry::U32("deepseek2.context_length", 128),
            KvEntry::U32("deepseek2.vocab_size", 32),
            KvEntry::F32("deepseek2.rope.freq_base", 10000.0),
            KvEntry::F32("deepseek2.attention.layer_norm_rms_epsilon", 1e-5),
            // No `deepseek2.attention.q_lora_rank`: this is the Lite shape.
            KvEntry::U32("deepseek2.attention.kv_lora_rank", 8),
            KvEntry::U32("deepseek2.attention.key_length", 8),
            KvEntry::U32("deepseek2.rope.dimension_count", 4),
            KvEntry::U32("deepseek2.attention.value_length", 4),
            KvEntry::U32("deepseek2.expert_count", 2),
            KvEntry::U32("deepseek2.expert_used_count", 1),
            KvEntry::U32("deepseek2.expert_shared_count", 1),
            KvEntry::U32("deepseek2.expert_feed_forward_length", 16),
            // Every layer dense, so the fixture stays a single block.
            KvEntry::U32("deepseek2.leading_dense_block_count", 1),
            KvEntry::Str("tokenizer.ggml.model", "deepseek"),
        ],
        DEEPSEEK_LITE_TENSORS,
    )
}

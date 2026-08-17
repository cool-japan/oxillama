//! Architecture-specific synthetic GGUF builders, continued.
//!
//! Split out of `arch.rs` to keep that file under the workspace's 2000-line
//! policy (both files were approaching/over the limit purely from the volume
//! of concurrently-landing per-architecture fixtures — this is a mechanical
//! split, not a functional change). All items here are re-exported from the
//! parent `test_utils` module so the public API is unchanged.

use super::{build_gguf_v3, KvEntry, TensorDesc};

// ─── Falcon builder ───────────────────────────────────────────────────────────
//
// Verified against `~/work/refs/llama.cpp`:
// - `src/llama-model.cpp` (`case LLM_ARCH_FALCON:` in `load_tensors`, ~line 835):
//   `attn_norm{,.bias}` required; `attn_norm_2{,.bias}` `TENSOR_NOT_REQUIRED`
//   (Falcon-40B only); fused `attn_qkv.weight` shape
//   `[n_embd, n_embd + 2*n_embd_gqa]`, **no** qkv/output/ffn bias tensors are
//   created at all (Falcon has zero linear biases, only LayerNorm biases);
//   `output.weight` `TENSOR_NOT_REQUIRED` with a tied `token_embd` fallback.
// - `src/models/falcon.cpp` (`llm_build_falcon`): attention and FFN are
//   UNCONDITIONALLY parallel (`cur = ffn_out + attn_out + inpL`) and RoPE is
//   unconditional — there is no ALiBi branch anywhere in this build function.
///
/// Tensors for a minimal 1-layer Falcon-2 style model (RoPE, GQA, single
/// `attn_norm`, no `attn_norm_2`).  Falcon-40B's `attn_norm_2` path is
/// exercised by a separate hand-built fixture in the falcon crate's own test
/// module (kept there because `KvEntry`/`TensorDesc` here cannot express
/// per-fixture-variant optionality without a second whole tensor table).
///
/// Dimensions: hidden=32, heads=4, head_dim=8, kv_heads=2 (GQA, n_embd_gqa=16),
/// ffn=64, vocab=32, layers=1.
const FALCON_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
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
    TensorDesc {
        name: "output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
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
    // Fused QKV: n_embd + 2*n_embd_gqa = 32 + 2*16 = 64 out-features.
    TensorDesc {
        name: "blk.0.attn_qkv.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // GELU FFN, no gate, no bias.
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
];

/// Build a valid GGUF v3 binary for a minimal 1-layer Falcon-2 style model
/// (GQA + RoPE, single `attn_norm`, no `attn_norm_2`).
///
/// | Hyper-parameter | Value |
/// |------------------|-------|
/// | `hidden_size`    | 32    |
/// | `heads`          | 4     |
/// | `head_dim`       | 8     |
/// | `kv_heads`       | 2 (GQA) |
/// | `layers`         | 1     |
/// | `vocab_size`     | 32    |
/// | `ffn_size`       | 64    |
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_falcon_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "falcon"),
            KvEntry::Str("general.name", "test-falcon"),
            KvEntry::U32("falcon.embedding_length", 32),
            KvEntry::U32("falcon.feed_forward_length", 64),
            KvEntry::U32("falcon.block_count", 1),
            KvEntry::U32("falcon.attention.head_count", 4),
            KvEntry::U32("falcon.attention.head_count_kv", 2),
            KvEntry::U32("falcon.context_length", 128),
            KvEntry::U32("falcon.vocab_size", 32),
            KvEntry::F32("falcon.attention.layer_norm_epsilon", 1e-5),
            KvEntry::F32("falcon.rope.freq_base", 10000.0),
            KvEntry::Str("tokenizer.ggml.model", "gpt2"),
        ],
        FALCON_TENSORS,
    )
}

// ─── GPT-NeoX builder ─────────────────────────────────────────────────────────
//
// Verified against `~/work/refs/llama.cpp`:
// - `src/llama-model.cpp` (`case LLM_ARCH_GPTNEOX:` in `load_tensors`,
//   ~line 1642): `attn_norm`/`ffn_norm` (with bias), fused `attn_qkv.weight`
//   shape `[n_embd, n_embd + 2*n_embd_gqa]` **with** required `bqkv` bias,
//   `bo`, `ffn_up_b`, `ffn_down_b` all required (flags=0, not
//   `TENSOR_NOT_REQUIRED`) — GPT-NeoX has bias on every linear projection.
//   `output.weight` has flags=0 (no tied-embedding fallback for this arch).
// - `src/models/gptneox.cpp` (`llm_build_gptneox`): `hparams.use_par_res`
//   branches between the parallel formula (`x + attn(ln1(x)) + ffn(ln2(x))`)
//   and the sequential formula — both read `attn_qkv`/`bqkv` as one fused
//   tensor, split into contiguous Q / K / V row-blocks (the interleave-undo
//   happens once at `convert_hf_to_gguf.py::GPTNeoXModel` conversion time, not
//   at load time), never separate `attn_q`/`attn_k`/`attn_v` tensors.
// - `gguf-py/gguf/constants.py`: `USE_PARALLEL_RESIDUAL = "{arch}.use_parallel_residual"`,
//   `DIMENSION_COUNT = "{arch}.rope.dimension_count"` (written by the
//   converter as `rotary_pct * head_dim`, so the loader must use this value
//   directly as `n_rot`, never re-derive it from a hard-coded 0.25 factor).
///
/// Tensors for a minimal 1-layer GPT-NeoX model.  `use_parallel_residual` is
/// deliberately omitted from the metadata (real checkpoints always write it,
/// but omitting it here exercises the loader's documented default of `true`,
/// matching `convert_hf_to_gguf.py`'s `hparams.get("use_parallel_residual",
/// True)`).  The `false` (sequential) branch is exercised by a hand-built
/// fixture with an explicit `Bool` metadata entry in the crate's own test
/// module, since `KvEntry` here has no `Bool` variant.
///
/// Dimensions: hidden=64, heads=4, head_dim=16, kv_heads=4 (MHA), ffn=128,
/// vocab=32, layers=1, rope.dimension_count=4 (rotary_pct 0.25 * head_dim 16).
const GPT_NEOX_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
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
    TensorDesc {
        name: "output.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
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
    // Fused QKV: n_embd + 2*n_embd_gqa = 64 + 2*64 = 192 out-features (MHA).
    TensorDesc {
        name: "blk.0.attn_qkv.weight",
        dims: &[64, 192],
        n_elements: 12288,
    },
    TensorDesc {
        name: "blk.0.attn_qkv.bias",
        dims: &[192],
        n_elements: 192,
    },
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
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[64, 128],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.0.ffn_up.bias",
        dims: &[128],
        n_elements: 128,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[128, 64],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.0.ffn_down.bias",
        dims: &[64],
        n_elements: 64,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer GPT-NeoX model.
///
/// | Hyper-parameter          | Value |
/// |---------------------------|-------|
/// | `hidden_size`              | 64    |
/// | `heads` / `kv_heads`       | 4 (MHA) |
/// | `head_dim`                 | 16    |
/// | `layers`                   | 1     |
/// | `vocab_size`                | 32    |
/// | `ffn_size`                  | 128   |
/// | `rope.dimension_count`      | 4 (partial RoPE: rotary_pct 0.25) |
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_gpt_neox_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "gptneox"),
            KvEntry::Str("general.name", "test-gptneox"),
            KvEntry::U32("gptneox.embedding_length", 64),
            KvEntry::U32("gptneox.feed_forward_length", 128),
            KvEntry::U32("gptneox.block_count", 1),
            KvEntry::U32("gptneox.attention.head_count", 4),
            KvEntry::U32("gptneox.attention.head_count_kv", 4),
            KvEntry::U32("gptneox.context_length", 128),
            KvEntry::U32("gptneox.vocab_size", 32),
            KvEntry::F32("gptneox.attention.layer_norm_epsilon", 1e-5),
            KvEntry::U32("gptneox.rope.dimension_count", 4),
            KvEntry::F32("gptneox.rope.freq_base", 10000.0),
            KvEntry::Str("tokenizer.ggml.model", "gpt2"),
        ],
        GPT_NEOX_TENSORS,
    )
}

// ─── StableLM builder ─────────────────────────────────────────────────────────
//
// Verified against `~/work/refs/llama.cpp`:
// - `src/llama-model.cpp` (`case LLM_ARCH_STABLELM:` in `load_tensors`,
//   ~line 3752): separate `wq`/`wk`/`wv`/`wo`; `bq`/`bk`/`bv`
//   `TENSOR_NOT_REQUIRED` (StableLM-2 1.6B); `attn_q_norm`/`attn_k_norm`
//   `TENSOR_NOT_REQUIRED` shaped `{n_embd_head_k, n_head}` /
//   `{n_embd_head_k, n_head_kv}` — i.e. genuinely **per-head** weights
//   (head_dim × n_head elements total), NOT a single head_dim-wide vector
//   shared across heads like Qwen3's QK-norm; `ffn_norm`/`ffn_norm_b`
//   `TENSOR_NOT_REQUIRED` (absent only in StableLM-2 12B, which uses parallel
//   residual); `output_norm`/`output_norm_b` and `output.weight` all
//   required (flags=0).
// - `src/models/stablelm.cpp` (`llm_build_stablelm`): `if (ffn_norm)` →
//   sequential (`ffn_norm(ffn_inp)` feeds the FFN); `else` → parallel, and the
//   FFN reads `inpSA` (the *attention input*, i.e. `attn_norm`'s output) —
//   never a second norm of the raw residual stream.  All three norm kinds use
//   `LLM_NORM` (plain LayerNorm, mean-centered), matching `attn_norm`'s
//   `.bias` tensor; the QK-norms pass `NULL` for bias (biasless LayerNorm).
///
/// Tensors for a minimal 1-layer StableLM model using the SEQUENTIAL
/// (`ffn_norm` present) branch — the common case for real checkpoints
/// (`use_parallel_residual=false`).  No q/k biases, no QK-norm (StableLM-1
/// style); the parallel-residual and QK-norm/bias branches are exercised by
/// hand-built fixtures in the crate's own test module.
///
/// Dimensions: hidden=64, heads=4, head_dim=16, kv_heads=2 (GQA, n_embd_gqa=32),
/// ffn=128, vocab=32, layers=1, rope.dimension_count=4 (rotary_pct 0.25).
const STABLELM_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
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
    TensorDesc {
        name: "output.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
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
    TensorDesc {
        name: "blk.0.attn_q.weight",
        dims: &[64, 64],
        n_elements: 4096,
    },
    // n_embd_gqa = kv_heads(2) * head_dim(16) = 32.
    TensorDesc {
        name: "blk.0.attn_k.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[64, 64],
        n_elements: 4096,
    },
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
    TensorDesc {
        name: "blk.0.ffn_gate.weight",
        dims: &[64, 128],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[128, 64],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[64, 128],
        n_elements: 8192,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer StableLM model
/// (sequential residual: `ffn_norm` present, no QK-norm, no q/k/v bias).
///
/// | Hyper-parameter          | Value |
/// |---------------------------|-------|
/// | `hidden_size`              | 64    |
/// | `heads`                    | 4     |
/// | `head_dim`                 | 16    |
/// | `kv_heads`                 | 2 (GQA) |
/// | `layers`                   | 1     |
/// | `vocab_size`                | 32    |
/// | `ffn_size`                  | 128   |
/// | `rope.dimension_count`      | 4 (partial RoPE: rotary_pct 0.25) |
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_stablelm_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "stablelm"),
            KvEntry::Str("general.name", "test-stablelm"),
            KvEntry::U32("stablelm.embedding_length", 64),
            KvEntry::U32("stablelm.feed_forward_length", 128),
            KvEntry::U32("stablelm.block_count", 1),
            KvEntry::U32("stablelm.attention.head_count", 4),
            KvEntry::U32("stablelm.attention.head_count_kv", 2),
            KvEntry::U32("stablelm.context_length", 128),
            KvEntry::U32("stablelm.vocab_size", 32),
            KvEntry::F32("stablelm.attention.layer_norm_epsilon", 1e-5),
            KvEntry::U32("stablelm.rope.dimension_count", 4),
            KvEntry::F32("stablelm.rope.freq_base", 10000.0),
            KvEntry::Str("tokenizer.ggml.model", "gpt2"),
        ],
        STABLELM_TENSORS,
    )
}

// ─── OLMo2 builder ────────────────────────────────────────────────────────────
//
// Verified against `~/work/refs/llama.cpp`:
// - `src/llama-model.cpp` (`case LLM_ARCH_OLMO2:` in `load_tensors`,
//   ~line 4909): separate `wq`/`wk`/`wv`/`wo`, no attention biases;
//   `attn_q_norm` shaped `{n_embd}` (full Q width) but `attn_k_norm` shaped
//   `{n_head_kv * n_embd_head}` — the **K width**, which is `n_embd_gqa`, NOT
//   `n_embd`, whenever `n_head_kv != n_head` (GQA).  `attn_post_norm` and
//   `ffn_post_norm` are both required (flags=0).
// - `src/llama-arch.cpp`'s `LLM_TENSOR_NAMES` map (~line 331): the *string*
//   for `LLM_TENSOR_ATTN_POST_NORM` is `"blk.%d.post_attention_norm"` and for
//   `LLM_TENSOR_FFN_POST_NORM` is `"blk.%d.post_ffw_norm"` — this table is
//   global (not per-architecture), so OLMo2's GGUF tensor names really are
//   `blk.{i}.post_attention_norm.weight` / `blk.{i}.post_ffw_norm.weight`,
//   **not** `attn_post_norm` / `ffn_post_norm` (confirmed independently via
//   `gguf-py/gguf/tensor_mapping.py`'s `ATTN_POST_NORM`/`FFN_POST_NORM`
//   entries, both commented `# gemma2 olmo2`).
// - `src/models/olmo2.cpp` (`llm_build_olmo2`): Q/K are RMSNorm'd as whole
//   vectors *before* the head reshape (`build_norm(Qcur, attn_q_norm, ...)`
//   then `ggml_reshape_3d`) — i.e. one normalization statistic over the
//   entire projected vector, not per-head.  `n_rot == n_embd_head` always
//   (`GGML_ASSERT`), so there is no partial RoPE.  Post-norm placement
//   (`attn_post_norm` after attention, `ffn_post_norm` after FFN, both before
//   their residual add) matches this repo's existing implementation.
///
/// Tensors for a minimal 1-layer OLMo2 model with GQA, so `attn_k_norm`'s
/// narrower shape (`n_embd_gqa` rather than `n_embd`) is actually exercised.
///
/// Dimensions: hidden=32, heads=4, head_dim=8, kv_heads=2 (GQA, n_embd_gqa=16),
/// ffn=64, vocab=32, layers=1.
const OLMO2_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
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
    TensorDesc {
        name: "blk.0.attn_q.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // n_embd_gqa = kv_heads(2) * head_dim(8) = 16.
    TensorDesc {
        name: "blk.0.attn_k.weight",
        dims: &[32, 16],
        n_elements: 512,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight",
        dims: &[32, 16],
        n_elements: 512,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // Full Q width: n_embd = 32.
    TensorDesc {
        name: "blk.0.attn_q_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // K width: n_embd_gqa = 16 — deliberately narrower than attn_q_norm so a
    // loader that reuses the Q-norm width for K would panic on this fixture.
    TensorDesc {
        name: "blk.0.attn_k_norm.weight",
        dims: &[16],
        n_elements: 16,
    },
    TensorDesc {
        name: "blk.0.post_attention_norm.weight",
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
        name: "blk.0.post_ffw_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
];

/// Build a valid GGUF v3 binary for a minimal 1-layer OLMo2 model (GQA, so
/// `attn_k_norm` legitimately differs in width from `attn_q_norm`).
///
/// | Hyper-parameter          | Value |
/// |---------------------------|-------|
/// | `hidden_size`              | 32    |
/// | `heads`                    | 4     |
/// | `head_dim`                 | 8     |
/// | `kv_heads`                 | 2 (GQA) |
/// | `layers`                   | 1     |
/// | `vocab_size`                | 32    |
/// | `ffn_size`                  | 64    |
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_olmo2_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "olmo2"),
            KvEntry::Str("general.name", "test-olmo2"),
            KvEntry::U32("olmo2.embedding_length", 32),
            KvEntry::U32("olmo2.feed_forward_length", 64),
            KvEntry::U32("olmo2.block_count", 1),
            KvEntry::U32("olmo2.attention.head_count", 4),
            KvEntry::U32("olmo2.attention.head_count_kv", 2),
            KvEntry::U32("olmo2.context_length", 128),
            KvEntry::U32("olmo2.vocab_size", 32),
            KvEntry::F32("olmo2.attention.layer_norm_rms_epsilon", 1e-5),
            KvEntry::F32("olmo2.rope.freq_base", 10000.0),
            KvEntry::Str("tokenizer.ggml.model", "gpt2"),
        ],
        OLMO2_TENSORS,
    )
}

// ─── MiniCPM builder ──────────────────────────────────────────────────────────
//
// Verified against `~/work/refs/llama.cpp`:
// - `src/llama-model.cpp` (`case LLM_ARCH_MINICPM:` shares the `LLM_ARCH_LLAMA`
//   tensor-creation block in `load_tensors`, ~line 2995): MiniCPM (base, not
//   MiniCPM3) is tensor-for-tensor identical to LLaMA — separate `wq`/`wk`/
//   `wv`/`wo`, `attn_norm`, `ffn_norm`, `ffn_gate`/`ffn_up`/`ffn_down`, tied
//   `output.weight` fallback. No MiniCPM-specific tensors exist; the three
//   scale factors below are hparams-only (no extra weight tensors).
// - `src/llama-model.cpp` (`case LLM_ARCH_MINICPM:` in `load_hparams`,
//   ~line 769): defaults `f_embedding_scale=12.0`, `f_residual_scale =
//   1.4/sqrt(n_layer)`, `f_logit_scale = 256/n_embd`, each overridable by the
//   generic `{arch}.embedding_scale` / `{arch}.residual_scale` /
//   `{arch}.logit_scale` GGUF keys (`gguf-py/gguf/constants.py`).
// - `src/models/granite.cpp` (`llm_build_granite`, the graph MiniCPM (base)
//   shares via `case LLM_ARCH_MINICPM: llm = make_unique<llm_build_granite>`):
//   `residual_scale` multiplies BOTH the attention output and the FFN output
//   before each is added to its residual; `logit_scale` is applied as
//   `logits *= 1.0/f_logit_scale` after the LM head.  `embedding_scale` is
//   applied generically in `llm_graph_context::build_inp_embd`
//   (`src/llama-graph.cpp`) as `hidden *= f_embedding_scale` right after the
//   embedding lookup.
// - `convert_hf_to_gguf.py::MiniCPMModel.set_gguf_parameters`: confirms the
//   GGUF-side formulas — `embedding_scale = hparams["scale_emb"]` (NOT
//   `hidden_size/dim_model_base`, which is only used for `logit_scale`),
//   `residual_scale = scale_depth / sqrt(num_hidden_layers)`, `logit_scale =
//   hidden_size / dim_model_base`.
///
/// Tensors for a minimal 1-layer MiniCPM (base) model — structurally
/// identical to LLaMA.  Uses distinctive (non-1.0, non-default) scale values
/// so a round-trip test can assert they were actually read from the GGUF
/// metadata rather than silently defaulted.
///
/// Dimensions: hidden=32, heads=4, head_dim=8, kv_heads=2 (GQA, n_embd_gqa=16),
/// ffn=64, vocab=32, layers=1.
const MINICPM_TENSORS: &[TensorDesc] = &[
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
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
    // n_embd_gqa = kv_heads(2) * head_dim(8) = 16.
    TensorDesc {
        name: "blk.0.attn_k.weight",
        dims: &[32, 16],
        n_elements: 512,
    },
    TensorDesc {
        name: "blk.0.attn_v.weight",
        dims: &[32, 16],
        n_elements: 512,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
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
];

/// Build a valid GGUF v3 binary for a minimal 1-layer MiniCPM (base) model.
///
/// Includes non-default `minicpm.embedding_scale = 2.0`,
/// `minicpm.residual_scale = 0.5`, `minicpm.logit_scale = 4.0` so a
/// round-trip test can assert all three are actually wired up (the pre-fix
/// code applied none of them: `embedding_scale` was hard-coded to `1.0`, and
/// `residual_scale`/`logit_scale` did not exist at all).
///
/// | Hyper-parameter          | Value |
/// |---------------------------|-------|
/// | `hidden_size`              | 32    |
/// | `heads`                    | 4     |
/// | `head_dim`                 | 8     |
/// | `kv_heads`                 | 2 (GQA) |
/// | `layers`                   | 1     |
/// | `vocab_size`                | 32    |
/// | `ffn_size`                  | 64    |
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_minicpm_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "minicpm"),
            KvEntry::Str("general.name", "test-minicpm"),
            KvEntry::U32("minicpm.embedding_length", 32),
            KvEntry::U32("minicpm.feed_forward_length", 64),
            KvEntry::U32("minicpm.block_count", 1),
            KvEntry::U32("minicpm.attention.head_count", 4),
            KvEntry::U32("minicpm.attention.head_count_kv", 2),
            KvEntry::U32("minicpm.context_length", 128),
            KvEntry::U32("minicpm.vocab_size", 32),
            KvEntry::F32("minicpm.attention.layer_norm_rms_epsilon", 1e-5),
            KvEntry::F32("minicpm.rope.freq_base", 10000.0),
            KvEntry::F32("minicpm.embedding_scale", 2.0),
            KvEntry::F32("minicpm.residual_scale", 0.5),
            KvEntry::F32("minicpm.logit_scale", 4.0),
            KvEntry::Str("tokenizer.ggml.model", "llama"),
        ],
        MINICPM_TENSORS,
    )
}

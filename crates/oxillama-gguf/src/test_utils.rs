//! Utilities for building synthetic GGUF models in tests.
//!
//! Enabled with `features = ["test-utils"]`. This module is **stable** as of v0.1.1:
//! builder function signatures will not change in a breaking way within the 0.1.x
//! series.
//!
//! Each builder returns a `Vec<u8>` containing a valid GGUF v3 binary. The
//! binaries can be parsed with [`crate::GgufModel::from_bytes`] and satisfy all
//! tensor lookups performed by `oxillama-arch`.
//!
//! # Example
//!
//! ```rust,ignore
//! # #[cfg(feature = "test-utils")]
//! # {
//! use oxillama_gguf::test_utils::{build_minimal_llama_gguf, minimal_tokenizer_json};
//!
//! let bytes = build_minimal_llama_gguf();
//! let model = oxillama_gguf::GgufModel::from_bytes(bytes).expect("parse synthetic GGUF");
//! assert_eq!(model.architecture().expect("arch"), "llama");
//!
//! let _tok_json = minimal_tokenizer_json();
//! # }
//! ```

/// Minimal BPE tokenizer JSON string compatible with `tokenizers 0.22.x`.
///
/// The vocabulary contains 32 entries (IDs 0–31), matching the `vocab_size=32`
/// baked into the synthetic GGUF produced by [`build_minimal_llama_gguf`].
/// Special tokens: `<unk>`=0, `<s>`=1, `</s>`=2.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn minimal_tokenizer_json() -> &'static str {
    r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7, "f": 8, "g": 9, "h": 10,
      "i": 11, "j": 12, "k": 13, "l": 14, "m": 15, "n": 16, "o": 17, "p": 18,
      "q": 19, "r": 20, "s": 21, "t": 22, "u": 23, "v": 24, "w": 25, "x": 26,
      "y": 27, "z": 28, " ": 29, ".": 30, "?": 31
    },
    "merges": []
  }
}"#
}

// ─── GGUF binary builder internals ────────────────────────────────────────────

/// Append a little-endian u32 to a byte vector.
fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Append a little-endian u64 to a byte vector.
fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Append a little-endian f32 to a byte vector.
fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Append a GGUF-encoded string: `[u64 len][UTF-8 bytes]`.
fn push_str(buf: &mut Vec<u8>, s: &str) {
    push_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

/// Append a KV pair whose value is a string.
fn push_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    push_str(buf, key);
    push_u32(buf, 8); // GgufValueType::String = 8
    push_str(buf, value);
}

/// Append a KV pair whose value is a u32.
fn push_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    push_str(buf, key);
    push_u32(buf, 4); // GgufValueType::Uint32 = 4
    push_u32(buf, value);
}

/// Append a KV pair whose value is an f32.
fn push_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
    push_str(buf, key);
    push_u32(buf, 6); // GgufValueType::Float32 = 6
    push_f32(buf, value);
}

/// Append a tensor-info record.
///
/// `dims` must be in GGUF order: innermost (cols) first, e.g. `[32, 32]` for a
/// 32×32 matrix.
fn push_tensor_info(buf: &mut Vec<u8>, name: &str, dims: &[u64], tensor_type: u32, offset: u64) {
    push_str(buf, name);
    push_u32(buf, dims.len() as u32); // n_dims
    for &d in dims {
        push_u64(buf, d);
    }
    push_u32(buf, tensor_type); // F32 = 0
    push_u64(buf, offset);
}

/// Pad `buf` up to the next multiple of `align` bytes by appending zero bytes.
fn align_to(buf: &mut Vec<u8>, align: usize) {
    let rem = buf.len() % align;
    if rem != 0 {
        buf.resize(buf.len() + align - rem, 0u8);
    }
}

// ─── Tensor catalogue ─────────────────────────────────────────────────────────

/// Descriptor for a single tensor in the synthetic model.
struct TensorDesc {
    name: &'static str,
    /// GGUF-order dims: [cols, rows] for 2-D, [len] for 1-D.
    dims: &'static [u64],
    /// Number of f32 elements = product of dims.
    n_elements: usize,
}

/// The 12 tensors required for a 1-layer LLaMA model (all F32).
const TENSORS: &[TensorDesc] = &[
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
        // shape[0]=out_features=intermediate_size=64, shape[1]=in_features=hidden=32
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        // shape[0]=out_features=intermediate_size=64, shape[1]=in_features=hidden=32
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        // shape[0]=out_features=hidden=32, shape[1]=in_features=intermediate_size=64
        dims: &[32, 64],
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

// ─── Public builder ───────────────────────────────────────────────────────────

/// Build a valid GGUF v3 binary for a minimal 1-layer LLaMA model.
///
/// All weight tensors are F32 and zero-initialised.  The resulting binary can
/// be parsed with [`crate::GgufModel::from_bytes`] and will satisfy every
/// tensor lookup performed by `oxillama-arch`'s `load_llama_from_gguf`.
///
/// # Dimensions (tiny but structurally valid)
///
/// | Hyper-parameter | Value |
/// |-----------------|-------|
/// | `hidden_size`   | 32    |
/// | `heads`         | 2     |
/// | `kv_heads`      | 2     |
/// | `head_dim`      | 16    |
/// | `layers`        | 1     |
/// | `vocab_size`    | 32    |
/// | `ffn_size`      | 64    |
/// | `context_len`   | 128   |
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_llama_gguf() -> Vec<u8> {
    const GGUF_MAGIC: u32 = 0x4655_4747; // b"GGUF" little-endian
    const TENSOR_COUNT: u64 = 12;
    const KV_COUNT: u64 = 10;
    const F32_TYPE: u32 = 0; // GgufTensorType::F32
    const ALIGN: usize = 32;

    let mut buf: Vec<u8> = Vec::with_capacity(128 * 1024);

    // ── Header ────────────────────────────────────────────────────────────────
    push_u32(&mut buf, GGUF_MAGIC);
    push_u32(&mut buf, 3); // version = 3
    push_u64(&mut buf, TENSOR_COUNT);
    push_u64(&mut buf, KV_COUNT);

    // ── KV metadata (10 pairs) ────────────────────────────────────────────────
    push_kv_string(&mut buf, "general.architecture", "llama");
    push_kv_u32(&mut buf, "llama.embedding_length", 32);
    push_kv_u32(&mut buf, "llama.feed_forward_length", 64);
    push_kv_u32(&mut buf, "llama.block_count", 1);
    push_kv_u32(&mut buf, "llama.attention.head_count", 2);
    push_kv_u32(&mut buf, "llama.attention.head_count_kv", 2);
    push_kv_u32(&mut buf, "llama.context_length", 128);
    push_kv_u32(&mut buf, "llama.vocab_size", 32);
    push_kv_f32(&mut buf, "llama.rope.freq_base", 10000.0);
    push_kv_string(&mut buf, "tokenizer.ggml.model", "llama");

    // ── Tensor infos ──────────────────────────────────────────────────────────
    // Pre-compute byte offsets by walking the tensor list once.
    let mut offsets = Vec::with_capacity(TENSORS.len());
    let mut running_offset: u64 = 0;
    for td in TENSORS {
        offsets.push(running_offset);
        running_offset += (td.n_elements as u64) * 4; // F32 = 4 bytes each
    }

    for (i, td) in TENSORS.iter().enumerate() {
        push_tensor_info(&mut buf, td.name, td.dims, F32_TYPE, offsets[i]);
    }

    // ── Alignment padding ─────────────────────────────────────────────────────
    align_to(&mut buf, ALIGN);

    // ── Tensor data section ───────────────────────────────────────────────────
    // All weights are zero-initialised F32. Written in the same order as the
    // tensor infos (offsets are cumulative from the start of this section).
    for td in TENSORS {
        let zero_bytes = vec![0u8; td.n_elements * 4];
        buf.extend_from_slice(&zero_bytes);
    }

    buf
}

// ─── Generic GGUF builder ─────────────────────────────────────────────────────

/// A single KV metadata entry in a synthetic GGUF.
enum KvEntry {
    Str(&'static str, &'static str),
    U32(&'static str, u32),
    F32(&'static str, f32),
}

/// Build a GGUF v3 binary from a list of KV entries and tensor descriptors.
///
/// All tensors are F32, zero-filled.  `kv` must be in the order the loaders
/// expect; the count is derived automatically.
fn build_gguf_v3(kv: &[KvEntry], tensors: &[TensorDesc]) -> Vec<u8> {
    const GGUF_MAGIC: u32 = 0x4655_4747;
    const F32_TYPE: u32 = 0;
    const ALIGN: usize = 32;

    let kv_count = kv.len() as u64;
    let tensor_count = tensors.len() as u64;

    let mut buf: Vec<u8> = Vec::with_capacity(256 * 1024);

    push_u32(&mut buf, GGUF_MAGIC);
    push_u32(&mut buf, 3);
    push_u64(&mut buf, tensor_count);
    push_u64(&mut buf, kv_count);

    for entry in kv {
        match entry {
            KvEntry::Str(k, v) => push_kv_string(&mut buf, k, v),
            KvEntry::U32(k, v) => push_kv_u32(&mut buf, k, *v),
            KvEntry::F32(k, v) => push_kv_f32(&mut buf, k, *v),
        }
    }

    // Pre-compute tensor data offsets.
    let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut running: u64 = 0;
    for td in tensors {
        offsets.push(running);
        running += (td.n_elements as u64) * 4;
    }

    for (i, td) in tensors.iter().enumerate() {
        push_tensor_info(&mut buf, td.name, td.dims, F32_TYPE, offsets[i]);
    }

    align_to(&mut buf, ALIGN);

    for td in tensors {
        buf.extend_from_slice(&vec![0u8; td.n_elements * 4]);
    }

    buf
}

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
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[32, 64],
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
/// Uses the same tiny dimensions as [`build_minimal_llama_gguf`].
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
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[32, 64],
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
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[32, 64],
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
/// `(2 + 2*2) * 16 = 96` rows, so the tensor shape (GGUF order) is `[96, 32]`
/// with 3072 elements.
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
    // merged QKV: (num_heads + 2*kv_heads) * head_dim rows = (2+4)*16 = 96
    TensorDesc {
        name: "blk.0.attn_qkv.weight",
        dims: &[96, 32],
        n_elements: 3072,
    },
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.ffn_gate.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[32, 64],
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
/// Command-R tensor names are identical to LLaMA.  Optional Q/K-norm weights
/// are absent here (they are loaded conditionally with `.ok()`).
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
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[32, 64],
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
/// - `attn_out.weight` (not `attn_output.weight`)
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
    // Absolute position embeddings: [context_len, hidden_size]
    TensorDesc {
        name: "position_embd.weight",
        dims: &[128, 32],
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
    // Fused QKV: (num_heads + 2) * head_dim = (2+2)*16 = 64 rows
    TensorDesc {
        name: "blk.0.attn_qkv.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.attn_qkv.bias",
        dims: &[64],
        n_elements: 64,
    },
    // Attention output projection
    TensorDesc {
        name: "blk.0.attn_out.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    TensorDesc {
        name: "blk.0.attn_out.bias",
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
        dims: &[64, 32],
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
        dims: &[32, 64],
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

// ─── DeepSeek-V2 builder ──────────────────────────────────────────────────────

/// Tensor catalogue for a minimal 1-layer DeepSeek-V2 model.
///
/// Dimensions (tiny but structurally valid):
///
/// | Hyper-parameter         | Value |
/// |-------------------------|-------|
/// | `hidden_size`           | 32    |
/// | `num_heads`             | 2     |
/// | `q_lora_rank`           | 8     |
/// | `kv_lora_rank`          | 8     |
/// | `qk_nope_head_dim`      | 4     |
/// | `qk_rope_head_dim`      | 4     |
/// | `v_head_dim`            | 4     |
/// | `expert_count` (routed) | 2     |
/// | `shared_experts`        | 1     |
/// | `layers`                | 1     |
/// | `vocab_size`            | 32    |
const DEEPSEEK_TENSORS: &[TensorDesc] = &[
    // Token embedding: [vocab=32, hidden=32]
    TensorDesc {
        name: "token_embd.weight",
        dims: &[32, 32],
        n_elements: 1024,
    },
    // ── Layer 0: pre-attention norm ──
    TensorDesc {
        name: "blk.0.attn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // ── Layer 0: MLA weights ──
    // w_q_a: [q_lora_rank=8, hidden=32] — row-major: [out_features, in_features]
    TensorDesc {
        name: "blk.0.attn_q_a_proj.weight",
        dims: &[8, 32],
        n_elements: 256,
    },
    // q_a_norm: [q_lora_rank=8]
    TensorDesc {
        name: "blk.0.attn_q_a_norm.weight",
        dims: &[8],
        n_elements: 8,
    },
    // w_q_b: [q_full=2*(4+4)=16, q_lora_rank=8] — row-major: [out, in]
    TensorDesc {
        name: "blk.0.attn_q_b_proj.weight",
        dims: &[16, 8],
        n_elements: 128,
    },
    // w_kv_a: [kv_comb=8+4=12, hidden=32] — row-major: [out, in]
    TensorDesc {
        name: "blk.0.attn_kv_a_proj.weight",
        dims: &[12, 32],
        n_elements: 384,
    },
    // kv_a_norm: [kv_lora_rank=8]
    TensorDesc {
        name: "blk.0.attn_kv_a_norm.weight",
        dims: &[8],
        n_elements: 8,
    },
    // w_kv_b: [kv_b_full=2*(4+4)=16, kv_lora_rank=8] — row-major: [out, in]
    TensorDesc {
        name: "blk.0.attn_kv_b_proj.weight",
        dims: &[16, 8],
        n_elements: 128,
    },
    // w_o: [hidden=32, attn_out=2*4=8] — row-major: [out, in]
    TensorDesc {
        name: "blk.0.attn_output.weight",
        dims: &[32, 8],
        n_elements: 256,
    },
    // ── Layer 0: pre-FFN norm ──
    TensorDesc {
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // ── Layer 0: Dense FFN (layer 0 is dense in DeepSeek-V2) ──
    // gate/up: [intermediate=64, hidden=32] — row-major: [out, in]
    TensorDesc {
        name: "blk.0.ffn_gate.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    // down: [hidden=32, intermediate=64] — row-major: [out, in]
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    // ── Layer 0: Router (for completeness; dense layer, router unused) ──
    // Router: [n_routed_experts=2, hidden=32] — row-major: [out, in]
    TensorDesc {
        name: "blk.0.ffn_gate_inp.weight",
        dims: &[2, 32],
        n_elements: 64,
    },
    // ── Layer 0: Routed expert 0 (2D layout) ──
    TensorDesc {
        name: "blk.0.ffn_exp.0.ffn_gate.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_exp.0.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_exp.0.ffn_down.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    // ── Layer 0: Routed expert 1 ──
    TensorDesc {
        name: "blk.0.ffn_exp.1.ffn_gate.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_exp.1.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_exp.1.ffn_down.weight",
        dims: &[32, 64],
        n_elements: 2048,
    },
    // ── Layer 0: Shared expert 0 ──
    TensorDesc {
        name: "blk.0.ffn_shared_exp.0.ffn_gate.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_shared_exp.0.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_shared_exp.0.ffn_down.weight",
        dims: &[32, 64],
        n_elements: 2048,
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

/// Build a valid GGUF v3 binary for a minimal 1-layer DeepSeek-V2 model.
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
            KvEntry::U32("deepseek2.block_count", 1),
            KvEntry::U32("deepseek2.attention.head_count", 2),
            KvEntry::U32("deepseek2.attention.head_count_kv", 2),
            KvEntry::U32("deepseek2.context_length", 128),
            KvEntry::U32("deepseek2.vocab_size", 32),
            KvEntry::F32("deepseek2.rope.freq_base", 10000.0),
            KvEntry::F32("deepseek2.attention.layer_norm_rms_epsilon", 1e-5),
            // MLA hyperparameters
            KvEntry::U32("deepseek2.attention.q_lora_rank", 8),
            KvEntry::U32("deepseek2.attention.kv_lora_rank", 8),
            KvEntry::U32("deepseek2.attention.key_length", 4),
            KvEntry::U32("deepseek2.attention.rope_head_dim", 4),
            KvEntry::U32("deepseek2.attention.value_length", 4),
            // MoE hyperparameters
            KvEntry::U32("deepseek2.expert_count", 2),
            KvEntry::U32("deepseek2.expert_used_count", 1),
            KvEntry::U32("deepseek2.expert_shared_count", 1),
            KvEntry::U32("deepseek2.expert_shared_feed_forward_length", 64),
            KvEntry::F32("deepseek2.expert_weights_scale", 1.0),
            // Tokenizer stub
            KvEntry::Str("tokenizer.ggml.model", "deepseek"),
        ],
        DEEPSEEK_TENSORS,
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
        name: "blk.0.ffn_norm.weight",
        dims: &[32],
        n_elements: 32,
    },
    // Router: [n_experts=4, hidden=32]
    TensorDesc {
        name: "blk.0.ffn_gate_inp.weight",
        dims: &[32, 4],
        n_elements: 128,
    },
    // Stacked gate exps: [n_experts=4, ffn_hidden=64, hidden=32]
    TensorDesc {
        name: "blk.0.ffn_gate_exps.weight",
        dims: &[4, 64, 32],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.0.ffn_up_exps.weight",
        dims: &[4, 64, 32],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.0.ffn_down_exps.weight",
        dims: &[4, 32, 64],
        n_elements: 8192,
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
    TensorDesc {
        name: "blk.1.ffn_norm.weight",
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
        dims: &[4, 64, 32],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.1.ffn_up_exps.weight",
        dims: &[4, 64, 32],
        n_elements: 8192,
    },
    TensorDesc {
        name: "blk.1.ffn_down_exps.weight",
        dims: &[4, 32, 64],
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
    // Stacked expert tensors: [n_experts=2, ffn_hidden=64, hidden=32]
    TensorDesc {
        name: "blk.0.ffn_gate_exps.weight",
        dims: &[2, 64, 32],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.0.ffn_up_exps.weight",
        dims: &[2, 64, 32],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.0.ffn_down_exps.weight",
        dims: &[2, 32, 64],
        n_elements: 4096,
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
        dims: &[2, 64, 32],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.1.ffn_up_exps.weight",
        dims: &[2, 64, 32],
        n_elements: 4096,
    },
    TensorDesc {
        name: "blk.1.ffn_down_exps.weight",
        dims: &[2, 32, 64],
        n_elements: 4096,
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

// ─── DeepSeek-V3 builder ──────────────────────────────────────────────────────

/// Build a valid GGUF v3 binary for a minimal 1-layer DeepSeek-V3 model
/// (same as V2 but adds `exp_probs_b.weight` tensor for sigmoid+bias routing).
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_deepseek_v3_gguf() -> Vec<u8> {
    // Re-use the same tensor layout as V2 but append the exp_probs_b tensor.
    // exp_probs_b: [n_routed_experts=2] — one bias per routed expert.
    const DEEPSEEK_V3_TENSORS: &[TensorDesc] = &[
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
            name: "blk.0.attn_q_a_proj.weight",
            dims: &[8, 32],
            n_elements: 256,
        },
        TensorDesc {
            name: "blk.0.attn_q_a_norm.weight",
            dims: &[8],
            n_elements: 8,
        },
        TensorDesc {
            name: "blk.0.attn_q_b_proj.weight",
            dims: &[16, 8],
            n_elements: 128,
        },
        TensorDesc {
            name: "blk.0.attn_kv_a_proj.weight",
            dims: &[12, 32],
            n_elements: 384,
        },
        TensorDesc {
            name: "blk.0.attn_kv_a_norm.weight",
            dims: &[8],
            n_elements: 8,
        },
        TensorDesc {
            name: "blk.0.attn_kv_b_proj.weight",
            dims: &[16, 8],
            n_elements: 128,
        },
        TensorDesc {
            name: "blk.0.attn_output.weight",
            dims: &[32, 8],
            n_elements: 256,
        },
        TensorDesc {
            name: "blk.0.ffn_norm.weight",
            dims: &[32],
            n_elements: 32,
        },
        TensorDesc {
            name: "blk.0.ffn_gate.weight",
            dims: &[64, 32],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_up.weight",
            dims: &[64, 32],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_down.weight",
            dims: &[32, 64],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_gate_inp.weight",
            dims: &[2, 32],
            n_elements: 64,
        },
        TensorDesc {
            name: "blk.0.ffn_exp.0.ffn_gate.weight",
            dims: &[64, 32],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_exp.0.ffn_up.weight",
            dims: &[64, 32],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_exp.0.ffn_down.weight",
            dims: &[32, 64],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_exp.1.ffn_gate.weight",
            dims: &[64, 32],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_exp.1.ffn_up.weight",
            dims: &[64, 32],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_exp.1.ffn_down.weight",
            dims: &[32, 64],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_shared_exp.0.ffn_gate.weight",
            dims: &[64, 32],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_shared_exp.0.ffn_up.weight",
            dims: &[64, 32],
            n_elements: 2048,
        },
        TensorDesc {
            name: "blk.0.ffn_shared_exp.0.ffn_down.weight",
            dims: &[32, 64],
            n_elements: 2048,
        },
        // Per-expert bias for sigmoid+bias routing: [n_routed_experts=2]
        TensorDesc {
            name: "blk.0.exp_probs_b.weight",
            dims: &[2],
            n_elements: 2,
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
            KvEntry::U32("deepseek2.block_count", 1),
            KvEntry::U32("deepseek2.attention.head_count", 2),
            KvEntry::U32("deepseek2.attention.head_count_kv", 2),
            KvEntry::U32("deepseek2.context_length", 128),
            KvEntry::U32("deepseek2.vocab_size", 32),
            KvEntry::F32("deepseek2.rope.freq_base", 10000.0),
            KvEntry::F32("deepseek2.attention.layer_norm_rms_epsilon", 1e-5),
            KvEntry::U32("deepseek2.attention.q_lora_rank", 8),
            KvEntry::U32("deepseek2.attention.kv_lora_rank", 8),
            KvEntry::U32("deepseek2.attention.key_length", 4),
            KvEntry::U32("deepseek2.attention.rope_head_dim", 4),
            KvEntry::U32("deepseek2.attention.value_length", 4),
            KvEntry::U32("deepseek2.expert_count", 2),
            KvEntry::U32("deepseek2.expert_used_count", 1),
            KvEntry::U32("deepseek2.expert_shared_count", 1),
            KvEntry::U32("deepseek2.expert_shared_feed_forward_length", 64),
            // Non-unit expert_weights_scale for SigmoidWithBias mode.
            KvEntry::F32("deepseek2.expert_weights_scale", 2.5),
            // All layers are MoE (no leading dense block) so the single layer
            // exercises the MoE path.
            KvEntry::U32("deepseek2.leading_dense_block_count", 0),
            KvEntry::Str("tokenizer.ggml.model", "deepseek"),
        ],
        DEEPSEEK_V3_TENSORS,
    )
}

// ─── Mamba-2 builder ──────────────────────────────────────────────────────────

/// Tensor catalogue for a minimal 1-layer Mamba-2 model.
///
/// Dimensions:
/// - d_model=16, d_state=8, d_inner=16 (expand=1), d_conv=4, n_layer=1, vocab=256
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
    // Combined gate+input projection: [2*d_inner=32, d_model=16]
    TensorDesc {
        name: "blk.0.ssm_in.weight",
        dims: &[16, 32],
        n_elements: 512,
    },
    // Depthwise conv: [d_inner=16, d_conv=4]
    TensorDesc {
        name: "blk.0.ssm_conv1d.weight",
        dims: &[4, 16],
        n_elements: 64,
    },
    TensorDesc {
        name: "blk.0.ssm_conv1d.bias",
        dims: &[16],
        n_elements: 16,
    },
    // B projection: [d_state=8, d_inner=16]
    TensorDesc {
        name: "blk.0.ssm_x.weight",
        dims: &[16, 8],
        n_elements: 128,
    },
    // C projection: [d_state=8, d_inner=16] (separate for clarity)
    // Δ projection: [d_inner=16, d_inner=16]
    TensorDesc {
        name: "blk.0.ssm_dt.weight",
        dims: &[16, 16],
        n_elements: 256,
    },
    TensorDesc {
        name: "blk.0.ssm_dt.bias",
        dims: &[16],
        n_elements: 16,
    },
    // Log-A: [d_state=8, d_inner=16]
    TensorDesc {
        name: "blk.0.ssm_A",
        dims: &[16, 8],
        n_elements: 128,
    },
    // D (skip): [d_inner=16]
    TensorDesc {
        name: "blk.0.ssm_D",
        dims: &[16],
        n_elements: 16,
    },
    // Output projection: [d_model=16, d_inner=16]
    TensorDesc {
        name: "blk.0.ssm_out.weight",
        dims: &[16, 16],
        n_elements: 256,
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
/// d_model=16, d_state=8, d_inner=16, d_conv=4, n_layer=1, vocab=256.
/// All weights are F32, zero-initialised.
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub fn build_minimal_mamba2_gguf() -> Vec<u8> {
    build_gguf_v3(
        &[
            KvEntry::Str("general.architecture", "mamba2"),
            KvEntry::Str("general.name", "test-mamba2"),
            KvEntry::U32("mamba2.d_model", 16),
            KvEntry::U32("mamba2.n_layer", 1),
            KvEntry::U32("mamba2.d_state", 8),
            KvEntry::U32("mamba2.d_conv", 4),
            KvEntry::U32("mamba2.expand", 1),
            KvEntry::U32("mamba2.vocab_size", 256),
            KvEntry::U32("mamba2.context_length", 512),
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
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        dims: &[64, 32],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        dims: &[32, 64],
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

// ─── Self-tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod self_tests {
    use super::*;
    use crate::GgufModel;

    /// The builder must produce a buffer that GgufModel::from_bytes accepts.
    #[test]
    fn test_build_parses_successfully() {
        let bytes = build_minimal_llama_gguf();
        let model = GgufModel::from_bytes(bytes).expect("synthetic GGUF must parse");
        assert_eq!(model.file.header.version, 3, "version must be 3");
        assert_eq!(model.file.header.tensor_count, 12, "must have 12 tensors");
        assert_eq!(
            model.file.header.metadata_kv_count, 10,
            "must have 10 KV pairs"
        );
    }

    /// Architecture metadata must resolve to "llama".
    #[test]
    fn test_architecture_is_llama() {
        let bytes = build_minimal_llama_gguf();
        let model = GgufModel::from_bytes(bytes).expect("parse synthetic GGUF");
        assert_eq!(
            model.architecture().expect("architecture must be present"),
            "llama"
        );
    }

    /// Every tensor name in the catalogue must be accessible by name.
    #[test]
    fn test_all_tensor_names_accessible() {
        let bytes = build_minimal_llama_gguf();
        let model = GgufModel::from_bytes(bytes).expect("parse synthetic GGUF");
        for td in TENSORS {
            let result = model.tensor_data(td.name);
            assert!(
                result.is_ok(),
                "tensor '{}' must be accessible, got: {:?}",
                td.name,
                result.err()
            );
        }
    }

    /// F32 tensor data must have the expected byte length (4 bytes per element).
    #[test]
    fn test_tensor_data_byte_sizes() {
        let bytes = build_minimal_llama_gguf();
        let model = GgufModel::from_bytes(bytes).expect("parse synthetic GGUF");
        for td in TENSORS {
            let data = model
                .tensor_data(td.name)
                .unwrap_or_else(|e| panic!("tensor '{}' must load: {e}", td.name));
            assert_eq!(
                data.len(),
                td.n_elements * 4,
                "tensor '{}' must have {} bytes, got {}",
                td.name,
                td.n_elements * 4,
                data.len()
            );
        }
    }

    /// The tokenizer JSON must be parseable (smoke-test string validity).
    #[test]
    fn test_minimal_tokenizer_json_is_valid_json() {
        let json = minimal_tokenizer_json();
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(json);
        assert!(
            parsed.is_ok(),
            "minimal_tokenizer_json() must be valid JSON: {:?}",
            parsed.err()
        );
    }
}

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
pub(crate) fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Append a little-endian u64 to a byte vector.
pub(crate) fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Append a little-endian f32 to a byte vector.
pub(crate) fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Append a GGUF-encoded string: `[u64 len][UTF-8 bytes]`.
pub(crate) fn push_str(buf: &mut Vec<u8>, s: &str) {
    push_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

/// Append a KV pair whose value is a string.
pub(crate) fn push_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    push_str(buf, key);
    push_u32(buf, 8); // GgufValueType::String = 8
    push_str(buf, value);
}

/// Append a KV pair whose value is a u32.
pub(crate) fn push_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    push_str(buf, key);
    push_u32(buf, 4); // GgufValueType::Uint32 = 4
    push_u32(buf, value);
}

/// Append a KV pair whose value is an f32.
pub(crate) fn push_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
    push_str(buf, key);
    push_u32(buf, 6); // GgufValueType::Float32 = 6
    push_f32(buf, value);
}

/// Append a tensor-info record.
///
/// `dims` must be in GGUF order: innermost (cols) first, e.g. `[32, 32]` for a
/// 32×32 matrix.
pub(crate) fn push_tensor_info(
    buf: &mut Vec<u8>,
    name: &str,
    dims: &[u64],
    tensor_type: u32,
    offset: u64,
) {
    push_str(buf, name);
    push_u32(buf, dims.len() as u32); // n_dims
    for &d in dims {
        push_u64(buf, d);
    }
    push_u32(buf, tensor_type); // F32 = 0
    push_u64(buf, offset);
}

/// Pad `buf` up to the next multiple of `align` bytes by appending zero bytes.
pub(crate) fn align_to(buf: &mut Vec<u8>, align: usize) {
    let rem = buf.len() % align;
    if rem != 0 {
        buf.resize(buf.len() + align - rem, 0u8);
    }
}

// ─── Tensor catalogue ─────────────────────────────────────────────────────────

/// Descriptor for a single tensor in the synthetic model.
pub(crate) struct TensorDesc {
    pub(crate) name: &'static str,
    /// GGUF-order dims: [in_features, out_features] for 2-D, [len] for 1-D.
    ///
    /// GGUF writes `ne` fastest-changing-first, so a weight that maps
    /// `in_features → out_features` is stored with `in_features` first — the
    /// reverse of the `[out_features, in_features]` order the loaders hand to
    /// `QuantLinear`.
    pub(crate) dims: &'static [u64],
    /// Number of f32 elements = product of dims.
    pub(crate) n_elements: usize,
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
        // GGUF order: [in_features=hidden=32, out_features=intermediate_size=64]
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_up.weight",
        // GGUF order: [in_features=hidden=32, out_features=intermediate_size=64]
        dims: &[32, 64],
        n_elements: 2048,
    },
    TensorDesc {
        name: "blk.0.ffn_down.weight",
        // GGUF order: [in_features=intermediate_size=64, out_features=hidden=32]
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
pub(crate) enum KvEntry {
    Str(&'static str, &'static str),
    U32(&'static str, u32),
    F32(&'static str, f32),
}

/// Build a GGUF v3 binary from a list of KV entries and tensor descriptors.
///
/// All tensors are F32, zero-filled.  `kv` must be in the order the loaders
/// expect; the count is derived automatically.
pub(crate) fn build_gguf_v3(kv: &[KvEntry], tensors: &[TensorDesc]) -> Vec<u8> {
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

// ─── Architecture-specific builders (split into submodule) ───────────────────

pub(crate) mod arch;
pub use arch::{
    build_minimal_bloom_gguf, build_minimal_command_r_gguf, build_minimal_dbrx_gguf,
    build_minimal_gemma_gguf, build_minimal_grok_gguf, build_minimal_lora_gguf,
    build_minimal_mamba2_gguf, build_minimal_mistral_gguf, build_minimal_phi3_gguf,
    build_minimal_phi_moe_gguf, build_minimal_qwen2vl_gguf, build_minimal_qwen3_gguf,
    build_minimal_starcoder_gguf,
};

// `arch.rs` was at 2571 lines with all five of falcon/gpt_neox/stablelm/olmo2/
// minicpm's fixtures appended (the workspace's 2000-line-per-file policy) —
// split those five out into `arch_ext` rather than restructuring the shared
// `arch` module other agents are concurrently appending to.
// The three DeepSeek fixtures live in their own file: appending them to `arch`
// pushed it past the workspace's 2000-line-per-file limit, the same reason
// `arch_ext` was split out.
pub(crate) mod arch_deepseek;
pub use arch_deepseek::{
    build_minimal_deepseek_gguf, build_minimal_deepseek_lite_gguf, build_minimal_deepseek_v3_gguf,
};

pub(crate) mod arch_ext;
pub use arch_ext::{
    build_minimal_falcon_gguf, build_minimal_gpt_neox_gguf, build_minimal_minicpm_gguf,
    build_minimal_olmo2_gguf, build_minimal_stablelm_gguf,
};

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

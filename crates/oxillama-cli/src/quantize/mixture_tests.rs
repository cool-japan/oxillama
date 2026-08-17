//! Model-level golden test: does the port reproduce llama.cpp's real mixtures?
//!
//! The two fixtures in `testdata/` are the tensor-info sections of GGUF files
//! **llama.cpp itself produced** at `Q4_K_M`:
//!
//! * `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` — 291 tensors, untied embeddings
//!   (`output.weight` present), 32 layers, GQA 32/8;
//! * `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` — 398 tensors, tied embeddings (no
//!   `output.weight`), 36 layers, GQA 32/8.
//!
//! Because those files came out of `llama_model_quantize`, every tensor's
//! stored type *is* `llama_tensor_get_type`'s answer for that model and
//! ftype. Replaying the tensor lists through [`super::type_select`] and
//! demanding an exact match on all 689 tensors is therefore a golden test
//! against llama.cpp's own output — not against a reimplementation of it.
//!
//! Between them the two models cover the decisions that actually vary:
//! the tied/untied `token_embd` split, `use_more_bits` on two different layer
//! counts, `attn_v` and `ffn_down` promotion, and the long tail of tensors
//! (`*_norm.weight`, `attn_k_norm`, `attn_q_norm`) that must be excluded from
//! quantization entirely.

use oxillama_gguf::GgufTensorType;

use super::type_select::{
    fallback_for_incompatible_row, should_quantize, Fallback, Ftype, ModelInfo, TypeSelector,
};

const LLAMA3: &str = include_str!("../../testdata/mixture_llama3_q4_k_m.txt");
const QWEN3: &str = include_str!("../../testdata/mixture_qwen3_q4_k_m.txt");

struct Recorded {
    name: String,
    n_dims: usize,
    ne0: i64,
    ty: GgufTensorType,
}

fn type_from_name(name: &str) -> GgufTensorType {
    match name {
        "F32" => GgufTensorType::F32,
        "F16" => GgufTensorType::F16,
        "BF16" => GgufTensorType::Bf16,
        "Q4_0" => GgufTensorType::Q4_0,
        "Q5_0" => GgufTensorType::Q5_0,
        "Q5_1" => GgufTensorType::Q5_1,
        "Q8_0" => GgufTensorType::Q8_0,
        "Q2_K" => GgufTensorType::Q2K,
        "Q3_K" => GgufTensorType::Q3K,
        "Q4_K" => GgufTensorType::Q4K,
        "Q5_K" => GgufTensorType::Q5K,
        "Q6_K" => GgufTensorType::Q6K,
        other => panic!("fixture names an unhandled tensor type: {other}"),
    }
}

fn ftype_from_name(name: &str) -> Ftype {
    match name {
        "Q4_K_M" => Ftype::Q4KM,
        other => panic!("fixture names an unhandled ftype: {other}"),
    }
}

fn parse(fixture: &str) -> (String, Ftype, ModelInfo, Vec<Recorded>) {
    let mut label = String::new();
    let mut ftype = Ftype::Q4KM;
    let mut arch = String::new();
    let (mut n_layer, mut n_head, mut n_head_kv, mut n_expert) = (0i32, 0u32, 0u32, 0u32);
    let mut has_output = false;
    let mut tensors = Vec::new();

    for line in fixture.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("MODEL") => {
                label = f[1].to_string();
                ftype = ftype_from_name(f[2]);
                arch = f[3].to_string();
                n_layer = f[4].parse().expect("n_layer");
                n_head = f[5].parse().expect("n_head");
                n_head_kv = f[6].parse().expect("n_head_kv");
                n_expert = f[7].parse().expect("n_expert");
                has_output = f[8] == "true";
            }
            Some("TENSOR") => tensors.push(Recorded {
                name: f[1].to_string(),
                n_dims: f[2].parse().expect("n_dims"),
                ne0: f[3].parse().expect("ne0"),
                ty: type_from_name(f[4]),
            }),
            other => panic!("unexpected fixture record: {other:?}"),
        }
    }

    // Same counting rule as the driver's `read_model_info`.
    let n_attention_wv = tensors
        .iter()
        .filter(|t| {
            t.name.contains("attn_v.weight")
                || t.name.contains("attn_qkv.weight")
                || t.name.contains("attn_kv_b.weight")
        })
        .count() as i32;

    let info = ModelInfo {
        arch,
        n_layer,
        n_head,
        n_head_kv,
        n_expert,
        has_output,
        n_attention_wv,
    };
    (label, ftype, info, tensors)
}

/// Replay one recorded model and require an exact match on every tensor.
fn check(fixture: &str, expected_tensors: usize) {
    let (label, ftype, info, tensors) = parse(fixture);
    assert_eq!(tensors.len(), expected_tensors, "{label}: fixture size");

    let mut selector = TypeSelector::new(&info, ftype);
    let mut quantized = 0usize;
    let mut passthrough = 0usize;

    for t in &tensors {
        if should_quantize(&t.name, t.n_dims) {
            let mut got = selector.select(&t.name, t.ne0).expect("select");
            match fallback_for_incompatible_row(got, t.ne0) {
                None => {}
                Some(Fallback::Use(f)) => got = f,
                Some(Fallback::NeedsIq4Nl) => {
                    panic!("{label}: {} unexpectedly needs IQ4_NL", t.name)
                }
                Some(Fallback::Unsupported) => {
                    panic!("{label}: {} unexpectedly has an unsupported shape", t.name)
                }
            }
            assert_eq!(
                got, t.ty,
                "{label}: tensor '{}' (ne0={}) — llama.cpp stored {}, the port chose {got}",
                t.name, t.ne0, t.ty
            );
            quantized += 1;
        } else {
            assert!(
                matches!(
                    t.ty,
                    GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::Bf16
                ),
                "{label}: tensor '{}' is excluded from quantization by the port, but \
                 llama.cpp stored it as {} — the exclusion list is wrong",
                t.name,
                t.ty
            );
            passthrough += 1;
        }
    }

    assert_eq!(quantized + passthrough, expected_tensors);
    assert!(
        quantized > 0 && passthrough > 0,
        "{label}: expected both quantized and passed-through tensors, got {quantized}/{passthrough}"
    );
}

/// Untied embeddings: `output.weight` exists, so `token_embd.weight` stays at
/// the bulk Q4_K and only `output.weight` is promoted to Q6_K.
#[test]
fn llama3_8b_q4_k_m_mixture_matches_llama_cpp() {
    check(LLAMA3, 291);
}

/// Tied embeddings: no `output.weight`, so `token_embd.weight` *is* the output
/// head and llama.cpp stores it as Q6_K.
#[test]
fn qwen3_4b_q4_k_m_mixture_matches_llama_cpp() {
    check(QWEN3, 398);
}

/// A guard on the guard: if the selector always returned the bulk type, the
/// checks above would still have something to compare, so make sure the
/// fixtures really do contain a mixture.
#[test]
fn fixtures_contain_more_than_one_quantized_type() {
    for (fixture, label) in [(LLAMA3, "llama3"), (QWEN3, "qwen3")] {
        let (_, _, _, tensors) = parse(fixture);
        let mut kinds: Vec<GgufTensorType> = tensors.iter().map(|t| t.ty).collect();
        kinds.sort_by_key(|t| *t as u32);
        kinds.dedup();
        let quantized: Vec<_> = kinds
            .iter()
            .filter(|t| {
                !matches!(
                    t,
                    GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::Bf16
                )
            })
            .collect();
        assert!(
            quantized.len() >= 2,
            "{label}: fixture has only {quantized:?}; it would not detect a constant selector"
        );
    }
}

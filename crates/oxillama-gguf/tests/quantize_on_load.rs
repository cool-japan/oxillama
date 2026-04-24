//! Integration tests for the quantize-on-load module.

use half::f16;
use oxillama_gguf::{GgufError, GgufModel, GgufTensorType, GgufValueType, QuantPlan, QuantTarget};

fn write_string(data: &mut Vec<u8>, s: &str) {
    data.extend_from_slice(&(s.len() as u64).to_le_bytes());
    data.extend_from_slice(s.as_bytes());
}

/// Build a minimal GGUF with a single F16 tensor of `n` weights (all = 1.0).
fn build_f16_gguf(tensor_name: &str, n: u64) -> Vec<u8> {
    use oxillama_gguf::types::GGUF_MAGIC;

    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV

    write_string(&mut data, "general.architecture");
    data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    write_string(&mut data, "llama");

    write_string(&mut data, tensor_name);
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&n.to_le_bytes());
    data.extend_from_slice(&(GgufTensorType::F16 as u32).to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    let rem = data.len() % 32;
    if rem != 0 {
        data.resize(data.len() + 32 - rem, 0u8);
    }

    for _ in 0..n {
        data.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
    }
    data
}

/// Build a minimal GGUF with a single F32 tensor of `n` weights (all = 2.0).
fn build_f32_gguf(tensor_name: &str, n: u64) -> Vec<u8> {
    use oxillama_gguf::types::GGUF_MAGIC;

    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    write_string(&mut data, "general.architecture");
    data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    write_string(&mut data, "llama");

    write_string(&mut data, tensor_name);
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&n.to_le_bytes());
    data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let rem = data.len() % 32;
    if rem != 0 {
        data.resize(data.len() + 32 - rem, 0u8);
    }

    for _ in 0..n {
        data.extend_from_slice(&2.0f32.to_le_bytes());
    }
    data
}

/// Build a minimal GGUF with a pre-quantized Q8_0 tensor.
fn build_q8_gguf(tensor_name: &str) -> Vec<u8> {
    use oxillama_gguf::types::GGUF_MAGIC;
    const N: u64 = 32;

    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    write_string(&mut data, "general.architecture");
    data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    write_string(&mut data, "llama");

    write_string(&mut data, tensor_name);
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&N.to_le_bytes());
    data.extend_from_slice(&(GgufTensorType::Q8_0 as u32).to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let rem = data.len() % 32;
    if rem != 0 {
        data.resize(data.len() + 32 - rem, 0u8);
    }
    data.resize(data.len() + 34, 0u8); // Q8_0: 34 bytes/block

    data
}

// ── Q4_0 tests ───────────────────────────────────────────────────────────────

#[test]
fn quantize_on_load_f16_to_q4_0() {
    let raw = build_f16_gguf("embed.weight", 32);
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    let plan = QuantPlan::uniform(QuantTarget::Q4_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    let out = model
        .tensor_data("embed.weight")
        .expect("test: tensor_data");
    // Q4_0: 32 weights → 1 block × 18 bytes
    assert_eq!(out.len(), 18, "Q4_0 for 32 weights = 18 bytes");
}

#[test]
fn quantize_on_load_f32_to_q4_0() {
    let raw = build_f32_gguf("embed.weight", 64); // 2 blocks
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    let plan = QuantPlan::uniform(QuantTarget::Q4_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    let out = model
        .tensor_data("embed.weight")
        .expect("test: tensor_data");
    // Q4_0: 64 weights → 2 blocks × 18 bytes = 36 bytes
    assert_eq!(out.len(), 36, "Q4_0 for 64 weights = 36 bytes");
}

// ── Q8_0 tests ───────────────────────────────────────────────────────────────

#[test]
fn quantize_on_load_f16_to_q8_0() {
    let raw = build_f16_gguf("embed.weight", 32);
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    let plan = QuantPlan::uniform(QuantTarget::Q8_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    let out = model
        .tensor_data("embed.weight")
        .expect("test: tensor_data");
    // Q8_0: 32 weights → 1 block × 34 bytes
    assert_eq!(out.len(), 34, "Q8_0 for 32 weights = 34 bytes");
}

#[test]
fn quantize_on_load_f32_to_q8_0() {
    let raw = build_f32_gguf("embed.weight", 96); // 3 blocks
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    let plan = QuantPlan::uniform(QuantTarget::Q8_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    let out = model
        .tensor_data("embed.weight")
        .expect("test: tensor_data");
    // Q8_0: 96 weights → 3 blocks × 34 bytes = 102 bytes
    assert_eq!(out.len(), 102, "Q8_0 for 96 weights = 102 bytes");
}

// ── Re-quantize rejection ─────────────────────────────────────────────────────

#[test]
fn quantize_on_load_rejects_requantize() {
    let raw = build_q8_gguf("embed.weight");
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    let plan = QuantPlan::uniform(QuantTarget::Q4_0);
    let result = model.apply_quant_plan(&plan);
    assert!(
        matches!(result, Err(GgufError::CannotRequantize { .. })),
        "re-quantizing Q8_0 should return CannotRequantize, got: {result:?}"
    );
}

// ── Per-tensor override ──────────────────────────────────────────────────────

#[test]
fn quantize_on_load_override_per_tensor() {
    let raw = build_f16_gguf("embed.weight", 32);
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    // Default Q8_0 but override embed.weight → Q4_0
    let plan =
        QuantPlan::uniform(QuantTarget::Q8_0).with_override("embed.weight", QuantTarget::Q4_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    let out = model
        .tensor_data("embed.weight")
        .expect("test: tensor_data");
    // Q4_0: 18 bytes (not Q8_0: 34 bytes)
    assert_eq!(out.len(), 18, "override Q4_0 for 32 weights = 18 bytes");
}

// ── load_with_quant_plan from file ───────────────────────────────────────────

#[test]
fn load_with_quant_plan_from_temp_file() {
    let raw = build_f32_gguf("output.weight", 32);
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, &raw).expect("test: write gguf");

    let plan = QuantPlan::uniform(QuantTarget::Q8_0);
    let model = GgufModel::load_with_quant_plan(&path, &plan).expect("test: load_with_quant_plan");

    let out = model
        .tensor_data("output.weight")
        .expect("test: tensor_data");
    // Q8_0 for 32 weights = 34 bytes
    assert_eq!(out.len(), 34, "Q8_0 for 32 weights = 34 bytes");
}

// ── No plan = no quantization ─────────────────────────────────────────────────

#[test]
fn empty_plan_leaves_f16_unchanged() {
    let raw = build_f16_gguf("embed.weight", 32);
    let original_len = {
        let model = GgufModel::from_bytes(raw.clone()).expect("test: from_bytes");
        model
            .tensor_data("embed.weight")
            .expect("test: tensor_data")
            .len()
    };

    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");
    let plan = QuantPlan::new(); // empty — no quantization
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    let out = model
        .tensor_data("embed.weight")
        .expect("test: tensor_data");
    // F16: 32 weights × 2 bytes = 64 bytes, unchanged
    assert_eq!(
        out.len(),
        original_len,
        "empty plan should leave tensor unchanged"
    );
}

// ── Multi-tensor model ────────────────────────────────────────────────────────

fn build_two_tensor_f16_gguf() -> Vec<u8> {
    use oxillama_gguf::types::GGUF_MAGIC;

    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV

    write_string(&mut data, "general.architecture");
    data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    write_string(&mut data, "llama");

    // Tensor 1: embed.weight F16 [32]
    write_string(&mut data, "embed.weight");
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&32u64.to_le_bytes());
    data.extend_from_slice(&(GgufTensorType::F16 as u32).to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // offset 0

    // Tensor 2: output.weight F16 [32], immediately after embed.weight
    let embed_size: u64 = 32 * 2; // F16: 2 bytes per weight
    write_string(&mut data, "output.weight");
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&32u64.to_le_bytes());
    data.extend_from_slice(&(GgufTensorType::F16 as u32).to_le_bytes());
    data.extend_from_slice(&embed_size.to_le_bytes()); // offset after embed

    // Pad
    let rem = data.len() % 32;
    if rem != 0 {
        data.resize(data.len() + 32 - rem, 0u8);
    }

    // Tensor data: 32 × f16(0.5) for embed, 32 × f16(2.0) for output
    for _ in 0..32 {
        data.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
    }
    for _ in 0..32 {
        data.extend_from_slice(&f16::from_f32(2.0).to_le_bytes());
    }

    data
}

#[test]
fn quant_plan_applies_uniform_to_all_f16_tensors() {
    let raw = build_two_tensor_f16_gguf();
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    let plan = QuantPlan::uniform(QuantTarget::Q4_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    let embed = model.tensor_data("embed.weight").expect("test: embed");
    assert_eq!(embed.len(), 18, "Q4_0 for 32 weights = 18 bytes");

    let output = model.tensor_data("output.weight").expect("test: output");
    assert_eq!(output.len(), 18, "Q4_0 for 32 weights = 18 bytes");
}

#[test]
fn quant_plan_override_applies_only_to_named_tensor() {
    let raw = build_two_tensor_f16_gguf();
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    // No default, only override for embed.weight
    let plan = QuantPlan::new().with_override("embed.weight", QuantTarget::Q8_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    // embed.weight should be Q8_0 (34 bytes)
    let embed = model.tensor_data("embed.weight").expect("test: embed");
    assert_eq!(
        embed.len(),
        34,
        "embed.weight: Q8_0 for 32 weights = 34 bytes"
    );

    // output.weight should still be raw F16 (32 × 2 = 64 bytes)
    let output = model.tensor_data("output.weight").expect("test: output");
    assert_eq!(output.len(), 64, "output.weight: unchanged F16 = 64 bytes");
}

// ── TensorStore type consistency after quantization ───────────────────────────
//
// This guards against the correctness bug where `tensor_data` returns quantized
// bytes but `tensor_type` still reports the old F16/F32 dtype.  Downstream
// consumers that read both fields (e.g., dequantize kernels) must see a
// consistent view.

#[test]
fn tensor_type_updated_to_q4_0_after_quantize() {
    let raw = build_f16_gguf("embed.weight", 32);
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    let plan = QuantPlan::uniform(QuantTarget::Q4_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    let info = model
        .file
        .tensors
        .get("embed.weight")
        .expect("test: get TensorInfo");
    assert_eq!(
        info.tensor_type,
        GgufTensorType::Q4_0,
        "TensorStore.tensor_type must be Q4_0 after on-load quantization"
    );
}

#[test]
fn tensor_type_updated_to_q8_0_after_quantize() {
    let raw = build_f32_gguf("embed.weight", 32);
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    let plan = QuantPlan::uniform(QuantTarget::Q8_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    let info = model
        .file
        .tensors
        .get("embed.weight")
        .expect("test: get TensorInfo");
    assert_eq!(
        info.tensor_type,
        GgufTensorType::Q8_0,
        "TensorStore.tensor_type must be Q8_0 after on-load quantization"
    );
}

#[test]
fn tensor_type_unchanged_for_non_quantized_tensor() {
    let raw = build_two_tensor_f16_gguf();
    let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

    // Only quantize embed.weight; output.weight has no override and no default
    let plan = QuantPlan::new().with_override("embed.weight", QuantTarget::Q4_0);
    model
        .apply_quant_plan(&plan)
        .expect("test: apply_quant_plan");

    // embed.weight should be Q4_0
    let embed_info = model
        .file
        .tensors
        .get("embed.weight")
        .expect("test: embed info");
    assert_eq!(
        embed_info.tensor_type,
        GgufTensorType::Q4_0,
        "embed.weight must be updated to Q4_0"
    );

    // output.weight should remain F16
    let output_info = model
        .file
        .tensors
        .get("output.weight")
        .expect("test: output info");
    assert_eq!(
        output_info.tensor_type,
        GgufTensorType::F16,
        "output.weight must remain F16 when not in the plan"
    );
}

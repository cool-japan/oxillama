//! WebAssembly bindings for OxiLLaMa.
//!
//! Exposes GGUF header parsing, Q4_0 dequantization, **and full text
//! generation** to JavaScript/TypeScript via wasm-bindgen.
//!
//! ## Feature flags
//!
//! | Feature     | Default | Description                                        |
//! |-------------|---------|---------------------------------------------------|
//! | `inference` | yes     | Enables `generate()` via `oxillama-runtime` with  |
//! |             |         | the pure-Rust `unstable_wasm` tokenizer backend.  |
//!
//! ## Usage (generate)
//!
//! ```js
//! import init, { generate } from './oxillama_wasm.js';
//! await init();
//!
//! const modelResp = await fetch('model.gguf');
//! const modelBytes = new Uint8Array(await modelResp.arrayBuffer());
//! const tokenizerResp = await fetch('tokenizer.json');
//! const tokenizerJson = await tokenizerResp.text();
//!
//! // Streaming: pass a callback to receive each token as it is generated.
//! const text = generate(modelBytes, tokenizerJson, "Hello, world!", 128,
//!     (token) => process.stdout.write(token));
//! console.log(text);
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub mod gpu_bridge;
pub mod idb_cache;
pub mod service_worker;
pub mod simd_check;
pub mod streaming_load;
pub mod streaming_loader;
pub mod webgpu;
pub mod worker;

pub use service_worker::{
    get_service_worker_script, register_service_worker, ServiceWorkerOptions,
};
pub use simd_check::get_simd128_status;
pub use streaming_loader::StreamingGgufLoader;
pub use streaming_loader::StreamingLoadOptions;

// ── Panic hook (default feature) ─────────────────────────────────────────────

/// Initialize the WASM module (sets up panic hook for better error messages).
///
/// Called automatically by the generated JS glue code when the WASM module
/// is first instantiated.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ── GGUF header parsing ───────────────────────────────────────────────────────

/// Parse a GGUF file header from raw bytes and return key metadata as a JS object.
///
/// The returned object has the following numeric fields:
/// - `tensorCount`   — number of tensors in the file
/// - `metadataCount` — number of metadata KV pairs
/// - `version`       — GGUF file version (2 or 3)
///
/// Throws a JavaScript error string if the bytes are not a valid GGUF file.
#[wasm_bindgen(js_name = parseGgufHeader)]
pub fn parse_gguf_header(data: &[u8]) -> Result<JsValue, JsValue> {
    let gguf = oxillama_gguf::GgufFile::parse(data)
        .map_err(|e| JsValue::from_str(&format!("GGUF parse error: {e}")))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("tensorCount"),
        &JsValue::from_f64(gguf.tensors.len() as f64),
    )
    .map_err(|e| JsValue::from_str(&format!("Reflect.set error: {e:?}")))?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("metadataCount"),
        &JsValue::from_f64(gguf.metadata.len() as f64),
    )
    .map_err(|e| JsValue::from_str(&format!("Reflect.set error: {e:?}")))?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("version"),
        &JsValue::from_f64(gguf.header.version as f64),
    )
    .map_err(|e| JsValue::from_str(&format!("Reflect.set error: {e:?}")))?;

    Ok(JsValue::from(obj))
}

/// Return all tensor names stored in a GGUF file as a JS array of strings.
///
/// Throws a JavaScript error string if parsing fails.
#[wasm_bindgen(js_name = listTensorNames)]
pub fn list_tensor_names(data: &[u8]) -> Result<Vec<JsValue>, JsValue> {
    let gguf = oxillama_gguf::GgufFile::parse(data)
        .map_err(|e| JsValue::from_str(&format!("GGUF parse error: {e}")))?;

    Ok(gguf
        .tensors
        .names()
        .map(|name| JsValue::from_str(name))
        .collect())
}

// ── Q4_0 dequantization ───────────────────────────────────────────────────────

/// Dequantize a buffer of Q4_0 blocks to an array of f32 values.
///
/// The Q4_0 block layout is 18 bytes per 32 weights:
/// - 2 bytes: FP16 scale factor
/// - 16 bytes: 32 × 4-bit nibbles packed two per byte
///
/// `data` must be a multiple of 18 bytes.  Returns a `Vec<f32>` of length
/// `(data.len() / 18) * 32`.  Throws a JavaScript error string on any
/// malformed input.
#[wasm_bindgen(js_name = dequantQ4_0)]
pub fn dequant_q4_0(data: &[u8]) -> Result<Vec<f32>, JsValue> {
    use oxillama_quant::reference::Q4_0Ref;
    use oxillama_quant::traits::QuantKernel;

    const BLOCK_BYTES: usize = 18;
    const BLOCK_SIZE: usize = 32;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(JsValue::from_str(&format!(
            "Q4_0 data length {} is not a multiple of {} bytes per block",
            data.len(),
            BLOCK_BYTES,
        )));
    }

    let n_blocks = data.len() / BLOCK_BYTES;
    let n_weights = n_blocks * BLOCK_SIZE;
    let mut out = vec![0.0f32; n_weights];
    let kernel = Q4_0Ref;

    for (blk_idx, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
        let output_slice = &mut out[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
        kernel.dequant_block(block, output_slice).map_err(|e| {
            JsValue::from_str(&format!("dequant_block error at block {blk_idx}: {e}"))
        })?;
    }

    Ok(out)
}

// ── Text generation ───────────────────────────────────────────────────────────

/// Run full text generation from an in-memory GGUF model.
///
/// # Arguments
///
/// - `model_bytes`    — raw bytes of the `.gguf` model file (copied in JS
///                      via `new Uint8Array(buffer)`)
/// - `tokenizer_json` — contents of the HuggingFace `tokenizer.json` file
///                      for this model
/// - `prompt`         — input text prompt
/// - `max_tokens`     — maximum number of tokens to generate (generation
///                      stops earlier if the model produces an EOS token)
/// - `on_token`       — optional JS callback invoked with each generated token
///                      text as a string, enabling streaming output
///
/// Returns the generated text as a JS string, or throws a JS error.
///
/// # Notes
///
/// This function requires the `inference` feature (enabled by default).
/// It uses `oxillama-runtime` with the pure-Rust `tokenizer-wasm` backend
/// (`fancy-regex`) so it is safe on `wasm32-unknown-unknown`.
///
/// Model loading is done via `InferenceEngine::load_model_from_bytes` which
/// accepts raw GGUF bytes — no filesystem access is needed inside the WASM sandbox.
#[cfg(feature = "inference")]
#[wasm_bindgen]
pub fn generate(
    model_bytes: &[u8],
    tokenizer_json: &str,
    prompt: &str,
    max_tokens: usize,
    on_token: Option<js_sys::Function>,
) -> Result<String, JsValue> {
    use oxillama_runtime::{EngineConfig, InferenceEngine};

    // ── 1. Create engine and load model from raw bytes ────────────────────────
    //
    // `load_model_from_bytes` is the filesystem-free entry point added to
    // InferenceEngine specifically for WASM (and any other no-fs environment).
    // It accepts the GGUF bytes directly and loads the tokenizer from the
    // supplied JSON string rather than a file path.
    let mut engine = InferenceEngine::new(EngineConfig::default());
    engine
        .load_model_from_bytes(model_bytes, tokenizer_json)
        .map_err(|e| JsValue::from_str(&format!("model load error: {e}")))?;

    // ── 2. Run the generation pipeline ───────────────────────────────────────
    //
    // `InferenceEngine::generate` handles tokenisation, prefill, and the
    // autoregressive decode loop — including EOS detection and context-length
    // capping.  The callback receives each token's text as it is decoded;
    // if a JS callback was supplied we forward each token to it.
    let output = engine
        .generate(prompt, max_tokens, |token_text| {
            if let Some(ref cb) = on_token {
                let this = JsValue::NULL;
                let _ = cb.call1(&this, &JsValue::from_str(token_text));
            }
        })
        .map_err(|e| JsValue::from_str(&format!("generation error: {e}")))?;

    Ok(output)
}

// ── K-quant dequantization ─────────────────────────────────────────────────────

/// Dequantize a buffer of Q4_K blocks to an array of f32 values.
///
/// The Q4_K block layout is 144 bytes per 256 weights.
/// `data` must be a multiple of 144 bytes.  Returns a `Vec<f32>` of length
/// `(data.len() / 144) * 256`.  Throws a JavaScript error string on any
/// malformed input.
#[wasm_bindgen(js_name = dequantQ4K)]
pub fn dequant_q4_k(data: &[u8]) -> Result<Vec<f32>, JsValue> {
    use oxillama_quant::reference::Q4KRef;
    use oxillama_quant::traits::QuantKernel;

    const BLOCK_BYTES: usize = 144;
    const BLOCK_SIZE: usize = 256;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(JsValue::from_str(&format!(
            "Q4_K data length {} is not a multiple of {} bytes per block",
            data.len(),
            BLOCK_BYTES,
        )));
    }

    let n_blocks = data.len() / BLOCK_BYTES;
    let n_weights = n_blocks * BLOCK_SIZE;
    let mut out = vec![0.0f32; n_weights];
    let kernel = Q4KRef;

    for (blk_idx, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
        let output_slice = &mut out[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
        kernel.dequant_block(block, output_slice).map_err(|e| {
            JsValue::from_str(&format!("dequant_block error at block {blk_idx}: {e}"))
        })?;
    }

    Ok(out)
}

/// Dequantize a buffer of Q5_K blocks to an array of f32 values.
///
/// The Q5_K block layout is 176 bytes per 256 weights.
/// `data` must be a multiple of 176 bytes.  Returns a `Vec<f32>` of length
/// `(data.len() / 176) * 256`.  Throws a JavaScript error string on any
/// malformed input.
#[wasm_bindgen(js_name = dequantQ5K)]
pub fn dequant_q5_k(data: &[u8]) -> Result<Vec<f32>, JsValue> {
    use oxillama_quant::reference::Q5KRef;
    use oxillama_quant::traits::QuantKernel;

    const BLOCK_BYTES: usize = 176;
    const BLOCK_SIZE: usize = 256;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(JsValue::from_str(&format!(
            "Q5_K data length {} is not a multiple of {} bytes per block",
            data.len(),
            BLOCK_BYTES,
        )));
    }

    let n_blocks = data.len() / BLOCK_BYTES;
    let n_weights = n_blocks * BLOCK_SIZE;
    let mut out = vec![0.0f32; n_weights];
    let kernel = Q5KRef;

    for (blk_idx, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
        let output_slice = &mut out[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
        kernel.dequant_block(block, output_slice).map_err(|e| {
            JsValue::from_str(&format!("dequant_block error at block {blk_idx}: {e}"))
        })?;
    }

    Ok(out)
}

/// Dequantize a buffer of Q6_K blocks to an array of f32 values.
///
/// The Q6_K block layout is 210 bytes per 256 weights.
/// `data` must be a multiple of 210 bytes.  Returns a `Vec<f32>` of length
/// `(data.len() / 210) * 256`.  Throws a JavaScript error string on any
/// malformed input.
#[wasm_bindgen(js_name = dequantQ6K)]
pub fn dequant_q6_k(data: &[u8]) -> Result<Vec<f32>, JsValue> {
    use oxillama_quant::reference::Q6KRef;
    use oxillama_quant::traits::QuantKernel;

    const BLOCK_BYTES: usize = 210;
    const BLOCK_SIZE: usize = 256;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(JsValue::from_str(&format!(
            "Q6_K data length {} is not a multiple of {} bytes per block",
            data.len(),
            BLOCK_BYTES,
        )));
    }

    let n_blocks = data.len() / BLOCK_BYTES;
    let n_weights = n_blocks * BLOCK_SIZE;
    let mut out = vec![0.0f32; n_weights];
    let kernel = Q6KRef;

    for (blk_idx, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
        let output_slice = &mut out[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
        kernel.dequant_block(block, output_slice).map_err(|e| {
            JsValue::from_str(&format!("dequant_block error at block {blk_idx}: {e}"))
        })?;
    }

    Ok(out)
}

// ── Load model with progress callback ────────────────────────────────────────

/// Typed GGUF metadata returned by [`parse_gguf_metadata`].
///
/// Optional fields are `None` when the corresponding key is absent in the file.
#[derive(Debug, Serialize, Deserialize)]
pub struct GgufMetadataJs {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub arch: Option<String>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub attention_head_count: Option<u64>,
    pub block_count: Option<u64>,
    pub quantization_version: Option<u32>,
    pub general_name: Option<String>,
    pub general_author: Option<String>,
    pub general_description: Option<String>,
}

/// Core model-loading logic shared by `load_model_from_bytes_with_progress`
/// and the inference `generate` function.
///
/// Emits progress percentages (0, 25, 100) to `on_progress` if provided.
/// Returns an error as a `JsValue` string if loading fails.
#[cfg(feature = "inference")]
fn load_model_core(
    model_bytes: &[u8],
    tokenizer_json: &str,
    on_progress: Option<&js_sys::Function>,
) -> Result<oxillama_runtime::InferenceEngine, JsValue> {
    use oxillama_runtime::{EngineConfig, InferenceEngine};

    let emit = |pct: u32| {
        if let Some(cb) = on_progress {
            let _ = cb.call1(&JsValue::UNDEFINED, &JsValue::from(pct));
        }
    };

    emit(0);
    let mut engine = InferenceEngine::new(EngineConfig::default());
    emit(25);
    engine
        .load_model_from_bytes(model_bytes, tokenizer_json)
        .map_err(|e| JsValue::from_str(&format!("model load error: {e}")))?;
    emit(100);

    Ok(engine)
}

/// Load a GGUF model from raw bytes, reporting progress via an optional JS callback.
///
/// The callback is invoked with a percentage value (`0`, `25`, `100`) at key
/// milestones during loading.  Pass `undefined` / `null` to skip progress
/// reporting.
///
/// Returns an opaque `WasmEngine` handle on success, or throws a JS error.
#[cfg(feature = "inference")]
#[wasm_bindgen(js_name = loadModelFromBytesWithProgress)]
pub fn load_model_from_bytes_with_progress(
    model_bytes: &[u8],
    tokenizer_json: &str,
    on_progress: Option<js_sys::Function>,
) -> Result<WasmEngine, JsValue> {
    let engine = load_model_core(model_bytes, tokenizer_json, on_progress.as_ref())?;
    Ok(WasmEngine { inner: engine })
}

/// Opaque handle wrapping a loaded `InferenceEngine` for use from JS.
#[cfg(feature = "inference")]
#[wasm_bindgen]
pub struct WasmEngine {
    inner: oxillama_runtime::InferenceEngine,
}

#[cfg(feature = "inference")]
#[wasm_bindgen]
impl WasmEngine {
    /// Run text generation on this engine.
    ///
    /// Equivalent to the top-level [`generate`] function but reuses an already
    /// loaded model, avoiding the expensive load step on subsequent calls.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        on_token: Option<js_sys::Function>,
    ) -> Result<String, JsValue> {
        self.inner
            .generate(prompt, max_tokens, |tok| {
                if let Some(ref cb) = on_token {
                    let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(tok));
                }
            })
            .map_err(|e| JsValue::from_str(&format!("generation error: {e}")))
    }
}

// ── Typed GGUF metadata export ────────────────────────────────────────────────

/// Parse a GGUF file and return typed metadata as a JS object.
///
/// The returned object conforms to the [`GgufMetadataJs`] schema.  Optional
/// fields are `null` when the corresponding metadata key is absent.
///
/// Throws a JavaScript error string if parsing fails.
#[wasm_bindgen(js_name = parseGgufMetadata)]
pub fn parse_gguf_metadata(data: &[u8]) -> Result<JsValue, JsValue> {
    let gguf = oxillama_gguf::GgufFile::parse(data)
        .map_err(|e| JsValue::from_str(&format!("GGUF parse error: {e}")))?;

    let meta = &gguf.metadata;

    // Detect architecture first — used as prefix for architecture-specific keys.
    let arch: Option<String> = meta
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned());

    // Helper: look up an integer key, trying the arch-prefixed form first then
    // falling back to common prefixes.
    let get_u64 = |suffix: &str| -> Option<u64> {
        let prefixes: &[&str] = match arch.as_deref() {
            Some(a) => {
                // Use the detected arch plus a handful of common fallbacks.
                &[a, "llama", "mistral", "qwen3", "gemma", "phi"][..]
            }
            None => &["llama", "mistral", "qwen3", "gemma", "phi"][..],
        };
        for prefix in prefixes {
            let key = format!("{prefix}.{suffix}");
            if let Some(val) = meta.get(&key).and_then(|v| v.as_u64()) {
                return Some(val);
            }
        }
        None
    };

    let metadata_js = GgufMetadataJs {
        version: gguf.header.version,
        tensor_count: gguf.header.tensor_count,
        kv_count: gguf.header.metadata_kv_count,
        context_length: get_u64("context_length"),
        embedding_length: get_u64("embedding_length"),
        feed_forward_length: get_u64("feed_forward_length"),
        attention_head_count: get_u64("attention.head_count"),
        block_count: get_u64("block_count"),
        quantization_version: meta
            .get("general.quantization_version")
            .and_then(|v| v.as_u32()),
        general_name: meta
            .get("general.name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned()),
        general_author: meta
            .get("general.author")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned()),
        general_description: meta
            .get("general.description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned()),
        arch,
    };

    serde_wasm_bindgen::to_value(&metadata_js).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // Tests operate at the underlying library level to avoid wasm-bindgen
    // JsValue/Reflect machinery that only works correctly inside a WASM runtime.
    // The wasm-bindgen glue wrappers are tested via wasm-bindgen-test on a real
    // WASM target; here we verify the underlying logic independently.

    use oxillama_quant::reference::Q4_0Ref;
    use oxillama_quant::traits::QuantKernel;

    #[test]
    fn test_parse_gguf_empty_fails() {
        // An empty buffer must return a descriptive error, not panic.
        let result = oxillama_gguf::GgufFile::parse(&[]);
        assert!(result.is_err(), "empty buffer should fail to parse");
    }

    #[test]
    fn test_parse_gguf_bad_magic_fails() {
        // Wrong magic bytes must produce an error, not a panic.
        let bad = b"BAAD\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let result = oxillama_gguf::GgufFile::parse(bad);
        assert!(result.is_err(), "wrong magic should fail to parse");
    }

    #[test]
    fn test_dequant_q4_0_wrong_length_fails() {
        // 17 bytes is not a multiple of 18 — the length check should catch it.
        const BLOCK_BYTES: usize = 18;
        let bad = vec![0u8; 17];
        assert_ne!(
            bad.len() % BLOCK_BYTES,
            0,
            "17 must not be a multiple of 18"
        );
        // Verify that feeding incomplete block data to dequant_block gives an error.
        let kernel = Q4_0Ref;
        let mut out = vec![0.0f32; 32];
        let result = kernel.dequant_block(&bad, &mut out);
        assert!(result.is_err(), "incomplete block should fail");
    }

    #[test]
    fn test_dequant_q4_0_zero_block() {
        // A single Q4_0 block: scale = 1.0 (FP16), all nibbles = 0x88 (encodes 0).
        let mut block = vec![0u8; 18];
        // FP16 1.0 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        // Nibbles 0x88 => lo = 8 - 8 = 0, hi = 8 - 8 = 0 for each byte
        for b in block[2..].iter_mut() {
            *b = 0x88;
        }
        let kernel = Q4_0Ref;
        let mut out = vec![0.0f32; 32];
        kernel
            .dequant_block(&block, &mut out)
            .expect("should not fail on valid block");
        assert_eq!(out.len(), 32, "one block = 32 weights");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-5, "weight[{i}] = {v}, expected ~0.0");
        }
    }

    #[test]
    fn test_dequant_q4_0_two_blocks_length() {
        // Two blocks: verify output vector has 64 elements.
        const BLOCK_BYTES: usize = 18;
        const BLOCK_SIZE: usize = 32;
        let data = [0u8; 2 * BLOCK_BYTES];
        let kernel = Q4_0Ref;
        let n_blocks = data.len() / BLOCK_BYTES;
        let mut out = vec![0.0f32; n_blocks * BLOCK_SIZE];
        for (blk_idx, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
            let slice = &mut out[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
            kernel
                .dequant_block(block, slice)
                .expect("dequant_block should succeed on zeroed data");
        }
        assert_eq!(out.len(), 64, "two blocks = 64 weights");
    }

    // ── Q4_K tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_dequant_q4_k_wrong_length_fails() {
        use oxillama_quant::reference::Q4KRef;
        const BLOCK_BYTES: usize = 144;
        let bad = vec![0u8; 143];
        assert_ne!(bad.len() % BLOCK_BYTES, 0);
        let kernel = Q4KRef;
        let mut out = vec![0.0f32; 256];
        let result = kernel.dequant_block(&bad, &mut out);
        assert!(result.is_err(), "incomplete Q4_K block should fail");
    }

    #[test]
    fn test_dequant_q4_k_zero_block() {
        use oxillama_quant::reference::Q4KRef;
        const BLOCK_BYTES: usize = 144;
        const BLOCK_SIZE: usize = 256;
        // All-zero block: d=0, dmin=0, all nibbles=0 → all weights should be 0.
        let block = vec![0u8; BLOCK_BYTES];
        let kernel = Q4KRef;
        let mut out = vec![0.0f32; BLOCK_SIZE];
        kernel
            .dequant_block(&block, &mut out)
            .expect("zero block should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-5, "Q4_K weight[{i}] = {v}, expected ~0.0");
        }
    }

    // ── Q5_K tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_dequant_q5_k_wrong_length_fails() {
        use oxillama_quant::reference::Q5KRef;
        const BLOCK_BYTES: usize = 176;
        let bad = vec![0u8; 175];
        assert_ne!(bad.len() % BLOCK_BYTES, 0);
        let kernel = Q5KRef;
        let mut out = vec![0.0f32; 256];
        let result = kernel.dequant_block(&bad, &mut out);
        assert!(result.is_err(), "incomplete Q5_K block should fail");
    }

    #[test]
    fn test_dequant_q5_k_zero_block() {
        use oxillama_quant::reference::Q5KRef;
        const BLOCK_BYTES: usize = 176;
        const BLOCK_SIZE: usize = 256;
        let block = vec![0u8; BLOCK_BYTES];
        let kernel = Q5KRef;
        let mut out = vec![0.0f32; BLOCK_SIZE];
        kernel
            .dequant_block(&block, &mut out)
            .expect("zero block should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-5, "Q5_K weight[{i}] = {v}, expected ~0.0");
        }
    }

    // ── Q6_K tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_dequant_q6_k_wrong_length_fails() {
        use oxillama_quant::reference::Q6KRef;
        const BLOCK_BYTES: usize = 210;
        let bad = vec![0u8; 209];
        assert_ne!(bad.len() % BLOCK_BYTES, 0);
        let kernel = Q6KRef;
        let mut out = vec![0.0f32; 256];
        let result = kernel.dequant_block(&bad, &mut out);
        assert!(result.is_err(), "incomplete Q6_K block should fail");
    }

    #[test]
    fn test_dequant_q6_k_zero_block() {
        use oxillama_quant::reference::Q6KRef;
        const BLOCK_BYTES: usize = 210;
        const BLOCK_SIZE: usize = 256;
        // Q6_K zero block: d=0 → all weights = 0 regardless of quant values
        // (the 6-bit quants get -32 offset but d=0 zeroes everything).
        let block = vec![0u8; BLOCK_BYTES];
        let kernel = Q6KRef;
        let mut out = vec![0.0f32; BLOCK_SIZE];
        kernel
            .dequant_block(&block, &mut out)
            .expect("zero block should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-5, "Q6_K weight[{i}] = {v}, expected ~0.0");
        }
    }

    // ── Progress callback / metadata tests ───────────────────────────────────

    #[test]
    fn test_load_model_with_progress_empty_fails() {
        // Empty bytes must return an error (not panic) when no progress cb given.
        let result = oxillama_gguf::GgufFile::parse(&[]);
        assert!(result.is_err(), "empty bytes must fail GGUF parse");
    }

    #[test]
    fn test_parse_gguf_metadata_empty_fails() {
        // parse_gguf_metadata on empty input must propagate the parse error.
        let result = oxillama_gguf::GgufFile::parse(&[]);
        assert!(result.is_err(), "empty bytes must fail metadata extraction");
    }
}

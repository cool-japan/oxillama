//! WebGPU compute bridge for accelerating dequantization in the browser.
//!
//! Since we target `wasm32-unknown-unknown`, native WebGPU APIs are not
//! available.  Instead this module uses `js-sys` and `wasm-bindgen` to call
//! the browser's WebGPU JavaScript API from Rust.
//!
//! The CPU fallback paths (`dequant_*_cpu_fallback`, `gemv_cpu_fallback`) are
//! always available and functional — they use `oxillama-quant` reference
//! kernels.  The WebGPU pipeline struct holds JS handles for future async
//! dispatch once `wasm_bindgen_futures` is wired up.

use wasm_bindgen::prelude::*;

// ── WGSL shader source ──────────────────────────────────────────────────────

/// Embedded Q4_0 dequantization compute shader for WebGPU.
const Q4_0_DEQUANT_WGSL: &str = r#"
// Q4_0 dequantization compute shader for WebGPU
// Block layout: 2B FP16 scale + 16B nibbles = 18 bytes per 32 weights
// Workgroup size: 256 threads (each handles one weight)
@group(0) @binding(0) var<storage, read> input_blocks: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    n_blocks: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn fp16_to_f32(bits: u32) -> f32 {
    let exp = (bits >> 10u) & 0x1Fu;
    let mantissa = bits & 0x3FFu;
    let sign = bits >> 15u;
    if exp == 0u { return 0.0; }
    let f = f32(1u + mantissa) * pow(2.0, f32(i32(exp) - 25));
    return select(f, -f, sign != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let weight_idx = gid.x;
    let block_idx = weight_idx / 32u;
    let local_idx = weight_idx % 32u;
    if block_idx >= params.n_blocks { return; }

    // Each block is 18 bytes = 4.5 u32 words; use byte-addressable offset
    let block_base_u32 = block_idx * 5u; // approximate; scale at word 0 bits [0:15]
    let scale_raw = input_blocks[block_base_u32] & 0xFFFFu;
    let d = fp16_to_f32(scale_raw);

    // Nibble extraction: 16 bytes of nibbles start at byte 2
    let nibble_word = (local_idx / 8u) + 1u;
    let nibble_shift = (local_idx % 8u) * 4u;
    let nibble_val = (input_blocks[block_base_u32 + nibble_word] >> nibble_shift) & 0xFu;
    let q = i32(nibble_val) - 8;
    output[weight_idx] = d * f32(q);
}
"#;

// ── WebGpuContext ────────────────────────────────────────────────────────────

/// Holds JS handles to a `GPUDevice` and `GPUQueue`.
///
/// The `JsValue` fields are opaque browser handles accessed reflectively from
/// the JS side; they are intentionally stored but not read from Rust code.
#[allow(dead_code)] // fields are JS handles, accessed reflectively
pub struct WebGpuContext {
    device: JsValue,
    queue: JsValue,
}

impl WebGpuContext {
    /// Construct from a JS `GPUDevice` object.
    ///
    /// Extracts the `.queue` property via `Reflect::get`.
    pub fn from_device(device: JsValue) -> Self {
        let queue = js_sys::Reflect::get(&device, &JsValue::from_str("queue"))
            .unwrap_or(JsValue::UNDEFINED);
        Self { device, queue }
    }

    /// Check whether `navigator.gpu` exists in the current JS global scope.
    ///
    /// Returns `false` on native targets (no `navigator`) or in browsers
    /// without WebGPU support.
    pub fn is_webgpu_available() -> bool {
        let global = js_sys::global();
        let navigator = js_sys::Reflect::get(&global, &JsValue::from_str("navigator"))
            .unwrap_or(JsValue::UNDEFINED);
        if navigator.is_undefined() || navigator.is_null() {
            return false;
        }
        let gpu = js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"))
            .unwrap_or(JsValue::UNDEFINED);
        !gpu.is_undefined() && !gpu.is_null()
    }

    /// Create a `GPUBuffer` mapped at creation with float32 data.
    ///
    /// The buffer is marked with `STORAGE | COPY_SRC` usage flags so it can
    /// be used as a compute shader binding and read back.
    ///
    /// Returns `JsValue::UNDEFINED` if the `createBuffer` call fails.
    pub fn create_buffer_with_data(&self, data: &[f32]) -> JsValue {
        // GPUBufferDescriptor: { size, usage, mappedAtCreation }
        let descriptor = js_sys::Object::new();
        let byte_len = std::mem::size_of_val(data) as f64;
        // STORAGE (0x80) | COPY_SRC (0x04) | COPY_DST (0x08)
        let usage = 0x80 | 0x04 | 0x08;

        let _ = js_sys::Reflect::set(
            &descriptor,
            &JsValue::from_str("size"),
            &JsValue::from_f64(byte_len),
        );
        let _ = js_sys::Reflect::set(
            &descriptor,
            &JsValue::from_str("usage"),
            &JsValue::from_f64(usage as f64),
        );
        let _ = js_sys::Reflect::set(
            &descriptor,
            &JsValue::from_str("mappedAtCreation"),
            &JsValue::from_bool(true),
        );

        // device.createBuffer(descriptor)
        let create_buffer_fn =
            match js_sys::Reflect::get(&self.device, &JsValue::from_str("createBuffer")) {
                Ok(f) if f.is_function() => js_sys::Function::from(f),
                _ => return JsValue::UNDEFINED,
            };

        let buffer = match create_buffer_fn.call1(&self.device, &descriptor) {
            Ok(b) => b,
            Err(_) => return JsValue::UNDEFINED,
        };

        // Write data into the mapped range:
        // buffer.getMappedRange() → ArrayBuffer → new Float32Array(ab).set(data)
        let get_mapped = match js_sys::Reflect::get(&buffer, &JsValue::from_str("getMappedRange")) {
            Ok(f) if f.is_function() => js_sys::Function::from(f),
            _ => return buffer,
        };
        if let Ok(ab) = get_mapped.call0(&buffer) {
            let typed = js_sys::Float32Array::new(&ab);
            for (i, &v) in data.iter().enumerate() {
                typed.set_index(i as u32, v);
            }
        }

        // buffer.unmap()
        let unmap_fn = match js_sys::Reflect::get(&buffer, &JsValue::from_str("unmap")) {
            Ok(f) if f.is_function() => js_sys::Function::from(f),
            _ => return buffer,
        };
        let _ = unmap_fn.call0(&buffer);

        buffer
    }

    /// Read back from a GPU buffer.
    ///
    /// Since `mapAsync` is genuinely async and cannot be synchronously awaited
    /// from WASM without `wasm_bindgen_futures`, this is bridge scaffolding
    /// that returns an empty vector.  The real async path would be wired
    /// through a `Future` / JS `Promise`.
    pub fn read_buffer(&self, _buffer: &JsValue, _size: usize) -> Vec<f32> {
        // Placeholder: actual implementation needs wasm_bindgen_futures +
        // mapAsync → getMappedRange → Float32Array → Vec<f32>.
        Vec::new()
    }
}

// ── WebGpuDequantPipeline ───────────────────────────────────────────────────

/// Holds a [`WebGpuContext`] and the embedded WGSL shader source for
/// dispatching dequantization work to the GPU.
#[allow(dead_code)] // fields used for future async dispatch
pub struct WebGpuDequantPipeline {
    ctx: WebGpuContext,
    shader_source: &'static str,
}

impl WebGpuDequantPipeline {
    /// Create a new pipeline with the embedded Q4_0 WGSL shader.
    pub fn new(ctx: WebGpuContext) -> Self {
        Self {
            ctx,
            shader_source: Q4_0_DEQUANT_WGSL,
        }
    }

    /// CPU fallback: dequantize Q4_0 blocks using the reference kernel.
    ///
    /// `data` must be a multiple of 18 bytes (Q4_0 block size).
    pub fn dequant_q4_0_cpu_fallback(data: &[u8]) -> Result<Vec<f32>, String> {
        use oxillama_quant::reference::Q4_0Ref;
        use oxillama_quant::traits::QuantKernel;

        const BLOCK_BYTES: usize = 18;
        const BLOCK_SIZE: usize = 32;

        if !data.len().is_multiple_of(BLOCK_BYTES) {
            return Err(format!(
                "Q4_0 data length {} is not a multiple of {BLOCK_BYTES} bytes per block",
                data.len(),
            ));
        }

        let n_blocks = data.len() / BLOCK_BYTES;
        let n_weights = n_blocks * BLOCK_SIZE;
        let mut out = vec![0.0f32; n_weights];
        let kernel = Q4_0Ref;

        for (blk_idx, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
            let output_slice = &mut out[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
            kernel
                .dequant_block(block, output_slice)
                .map_err(|e| format!("dequant_block error at block {blk_idx}: {e}"))?;
        }

        Ok(out)
    }

    /// CPU fallback: dequantize Q8_0 blocks using the reference kernel.
    ///
    /// `data` must be a multiple of 34 bytes (Q8_0 block size: 2B scale + 32B quants).
    pub fn dequant_q8_0_cpu_fallback(data: &[u8]) -> Result<Vec<f32>, String> {
        use oxillama_quant::reference::Q8_0Ref;
        use oxillama_quant::traits::QuantKernel;

        const BLOCK_BYTES: usize = 34;
        const BLOCK_SIZE: usize = 32;

        if !data.len().is_multiple_of(BLOCK_BYTES) {
            return Err(format!(
                "Q8_0 data length {} is not a multiple of {BLOCK_BYTES} bytes per block",
                data.len(),
            ));
        }

        let n_blocks = data.len() / BLOCK_BYTES;
        let n_weights = n_blocks * BLOCK_SIZE;
        let mut out = vec![0.0f32; n_weights];
        let kernel = Q8_0Ref;

        for (blk_idx, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
            let output_slice = &mut out[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
            kernel
                .dequant_block(block, output_slice)
                .map_err(|e| format!("dequant_block error at block {blk_idx}: {e}"))?;
        }

        Ok(out)
    }

    /// CPU fallback GEMV: dot product of each row of `weights` with `input`.
    ///
    /// `weights` is row-major with dimensions `rows × cols`.
    /// Returns a vector of length `rows`.
    pub fn gemv_cpu_fallback(
        weights: &[f32],
        input: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, String> {
        if input.len() != cols {
            return Err(format!(
                "GEMV dimension mismatch: input.len()={} but cols={cols}",
                input.len(),
            ));
        }
        if weights.len() != rows * cols {
            return Err(format!(
                "GEMV dimension mismatch: weights.len()={} but rows*cols={}",
                weights.len(),
                rows * cols,
            ));
        }

        let mut output = vec![0.0f32; rows];
        for (r, out_val) in output.iter_mut().enumerate() {
            let row_start = r * cols;
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += weights[row_start + c] * input[c];
            }
            *out_val = acc;
        }

        Ok(output)
    }
}

// ── Constant for module-existence checks in tests ────────────────────────────

/// Module version tag (used by tests to verify the module is linked).
pub const WEBGPU_MODULE_VERSION: &str = "0.1.0";

// ── wasm-bindgen exports ────────────────────────────────────────────────────

/// Check whether WebGPU is available in the current browser environment.
#[wasm_bindgen(js_name = webgpuAvailable)]
pub fn webgpu_available() -> bool {
    WebGpuContext::is_webgpu_available()
}

/// Dequantize Q4_0 blocks using the CPU fallback path.
///
/// Always works in WASM — does not require WebGPU.
#[wasm_bindgen(js_name = dequantQ4_0WithFallback)]
pub fn dequant_q4_0_with_fallback(data: &[u8]) -> Result<Vec<f32>, JsValue> {
    WebGpuDequantPipeline::dequant_q4_0_cpu_fallback(data).map_err(|e| JsValue::from_str(&e))
}

/// Dequantize Q8_0 blocks using the CPU fallback path.
///
/// Always works in WASM — does not require WebGPU.
#[wasm_bindgen(js_name = dequantQ8_0WithFallback)]
pub fn dequant_q8_0_with_fallback(data: &[u8]) -> Result<Vec<f32>, JsValue> {
    WebGpuDequantPipeline::dequant_q8_0_cpu_fallback(data).map_err(|e| JsValue::from_str(&e))
}

/// GEMV (matrix-vector multiply) using the CPU fallback path.
///
/// `weights` is a row-major `rows × cols` matrix.
/// `input` is a vector of length `cols`.
/// Returns a vector of length `rows`.
#[wasm_bindgen(js_name = gemvCpuFallback)]
pub fn gemv_cpu_fallback(
    weights: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, JsValue> {
    WebGpuDequantPipeline::gemv_cpu_fallback(weights, input, rows, cols)
        .map_err(|e| JsValue::from_str(&e))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webgpu_module_exists() {
        // Verify the module is linked and the const is accessible.
        assert_eq!(WEBGPU_MODULE_VERSION, "0.1.0");
    }

    #[test]
    fn test_dequant_q4_0_wrong_length() {
        // 17 bytes is not a multiple of 18 — should error.
        let bad = vec![0u8; 17];
        let result = WebGpuDequantPipeline::dequant_q4_0_cpu_fallback(&bad);
        let err = result.expect_err("expected error message");
        assert!(
            err.contains("not a multiple of 18"),
            "error should mention block size: {err}"
        );
    }

    #[test]
    fn test_dequant_q4_0_one_zero_block() {
        // 18 zero bytes → scale = 0 → all 32 floats should be 0.
        let data = vec![0u8; 18];
        let result = WebGpuDequantPipeline::dequant_q4_0_cpu_fallback(&data)
            .expect("zero block should succeed");
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "weight[{i}] = {v}, expected ~0.0");
        }
    }

    #[test]
    fn test_dequant_q4_0_scale_1_all_center() {
        // Scale = FP16 1.0 = 0x3C00, all nibbles = 0x88 (value 8, 8-8=0).
        let mut block = vec![0u8; 18];
        block[0] = 0x00; // FP16 1.0 low byte
        block[1] = 0x3C; // FP16 1.0 high byte
        for b in block[2..].iter_mut() {
            *b = 0x88;
        }
        let result = WebGpuDequantPipeline::dequant_q4_0_cpu_fallback(&block)
            .expect("valid block should succeed");
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-5, "weight[{i}] = {v}, expected ~0.0");
        }
    }

    #[test]
    fn test_dequant_q8_0_wrong_length() {
        // 33 bytes is not a multiple of 34 — should error.
        let bad = vec![0u8; 33];
        let result = WebGpuDequantPipeline::dequant_q8_0_cpu_fallback(&bad);
        let err = result.expect_err("expected error message");
        assert!(
            err.contains("not a multiple of 34"),
            "error should mention block size: {err}"
        );
    }

    #[test]
    fn test_dequant_q8_0_one_zero_block() {
        // 34 zero bytes → scale = 0 → all 32 floats should be 0.
        let data = vec![0u8; 34];
        let result = WebGpuDequantPipeline::dequant_q8_0_cpu_fallback(&data)
            .expect("zero block should succeed");
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "weight[{i}] = {v}, expected ~0.0");
        }
    }

    #[test]
    fn test_gemv_scalar_correctness() {
        // 2×2 identity matrix · [3, 4] = [3, 4]
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let input = vec![3.0, 4.0];
        let result = WebGpuDequantPipeline::gemv_cpu_fallback(&weights, &input, 2, 2)
            .expect("gemv should succeed");
        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 1e-6, "row 0: {}", result[0]);
        assert!((result[1] - 4.0).abs() < 1e-6, "row 1: {}", result[1]);
    }

    #[test]
    fn test_gemv_dimension_mismatch() {
        // input.len() = 3 but cols = 2 → should error.
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let input = vec![1.0, 2.0, 3.0];
        let result = WebGpuDequantPipeline::gemv_cpu_fallback(&weights, &input, 2, 2);
        let err = result.expect_err("expected error message");
        assert!(
            err.contains("dimension mismatch"),
            "error should mention dimension: {err}"
        );
    }

    #[test]
    fn test_dequant_q4_0_two_blocks() {
        // 36 bytes (2 blocks × 18) → 64 floats.
        let data = vec![0u8; 36];
        let result = WebGpuDequantPipeline::dequant_q4_0_cpu_fallback(&data)
            .expect("two zero blocks should succeed");
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_gemv_3x3_identity() {
        // 3×3 identity matrix · [1, 2, 3] = [1, 2, 3]
        #[rustfmt::skip]
        let weights = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let input = vec![1.0, 2.0, 3.0];
        let result = WebGpuDequantPipeline::gemv_cpu_fallback(&weights, &input, 3, 3)
            .expect("3x3 gemv should succeed");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }
}

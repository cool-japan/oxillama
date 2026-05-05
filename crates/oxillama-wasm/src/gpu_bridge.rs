//! Async WebGPU bridge for OxiLLaMa WASM.
//!
//! This module provides an async initialization path for WebGPU compute.
//! Use `init_webgpu_device()` to request a GPU device, then pass the
//! returned `JsValue` (GPUDevice) to `webgpu_dequant_q4_0_async()` or
//! `webgpu_gemv_async()`.
//!
//! All functions return `js_sys::Promise` to be awaited from JavaScript:
//! ```js
//! const device = await initWebGpuDevice();
//! const output = await webgpuDequantQ4_0Async(device, quantData, n_blocks);
//! ```

use js_sys::{Function, Object, Promise, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{future_to_promise, JsFuture};

// ── WGSL shaders ─────────────────────────────────────────────────────────────

/// Q4_0 dequantization WGSL compute shader.
///
/// Block layout: 2B FP16 scale + 16B nibbles = 18 bytes per 32 weights.
/// The input buffer is treated as u32 words (4 bytes each). Each block
/// occupies 5 u32 words (18 bytes rounded up). Only the first 16-bit word of
/// the first u32 carries the FP16 scale; bytes 2-17 contain the nibbles.
const Q4_0_DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       input_blocks : array<u32>;
@group(0) @binding(1) var<storage, read_write> output       : array<f32>;

struct Params { n_blocks: u32 }
@group(0) @binding(2) var<uniform> params: Params;

fn fp16_to_f32(bits: u32) -> f32 {
    let sign     = (bits >> 15u) & 1u;
    let exp      = (bits >> 10u) & 0x1Fu;
    let mantissa = bits & 0x3FFu;
    if exp == 0u { return 0.0; }
    if exp == 31u { return select(1e38, -1e38, sign != 0u); }
    let value = (1.0 + f32(mantissa) / 1024.0) * pow(2.0, f32(i32(exp) - 15));
    return select(value, -value, sign != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let weight_idx = gid.x;
    let block_idx  = weight_idx / 32u;
    let local_idx  = weight_idx % 32u;
    if block_idx >= params.n_blocks { return; }

    // Each block is 18 bytes = 4.5 u32s; pack into 5 u32s per block.
    let base   = block_idx * 5u;
    let scale_raw = input_blocks[base] & 0xFFFFu;
    let d = fp16_to_f32(scale_raw);

    // Nibbles start at byte 2, i.e. the high 16 bits of word[base] plus
    // words base+1 … base+4. We compute which byte (offset from byte 2)
    // holds our nibble pair.
    let byte_offset  = local_idx / 2u;            // 0-15
    let nibble_half  = local_idx % 2u;            // 0 = low nibble, 1 = high
    // Convert byte_offset into a u32-word index and bit shift.
    // Bytes 0-1 are the FP16 scale (base word); bytes 2-17 are nibbles:
    //   nibble byte_offset 0 → global byte 2 → word base+0 bits [23:16] ... actually
    //   treat the 18 raw bytes packed into 5 u32s (little-endian).
    // byte 0 = bits[7:0] of word[base], byte 1 = bits[15:8], etc.
    let global_byte  = byte_offset + 2u;           // skip 2 scale bytes
    let word_idx     = global_byte / 4u;
    let byte_in_word = global_byte % 4u;
    let byte_val     = (input_blocks[base + word_idx] >> (byte_in_word * 8u)) & 0xFFu;
    let nibble       = select(byte_val & 0xFu, (byte_val >> 4u) & 0xFu, nibble_half != 0u);
    let q = i32(nibble) - 8;
    output[weight_idx] = d * f32(q);
}
"#;

/// GEMV (Q4_0 weights × f32 input) WGSL compute shader.
///
/// Each invocation handles one output row. The workgroup dispatches `rows`
/// threads (up to 256 per workgroup). Weights are stored in Q4_0 format
/// (18 bytes per block of 32 values); `n_cols` must be a multiple of 32.
const GEMV_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       weight_blocks : array<u32>;
@group(0) @binding(1) var<storage, read>       input_vec     : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_vec    : array<f32>;

struct Params {
    rows   : u32,
    n_cols : u32,   // must be multiple of 32
}
@group(0) @binding(3) var<uniform> params: Params;

fn fp16_to_f32(bits: u32) -> f32 {
    let sign     = (bits >> 15u) & 1u;
    let exp      = (bits >> 10u) & 0x1Fu;
    let mantissa = bits & 0x3FFu;
    if exp == 0u { return 0.0; }
    if exp == 31u { return select(1e38, -1e38, sign != 0u); }
    let value = (1.0 + f32(mantissa) / 1024.0) * pow(2.0, f32(i32(exp) - 15));
    return select(value, -value, sign != 0u);
}

fn get_q4_0_weight(block_base: u32, local_idx: u32) -> f32 {
    let scale_raw    = weight_blocks[block_base] & 0xFFFFu;
    let d            = fp16_to_f32(scale_raw);
    let byte_offset  = local_idx / 2u;
    let nibble_half  = local_idx % 2u;
    let global_byte  = byte_offset + 2u;
    let word_idx     = global_byte / 4u;
    let byte_in_word = global_byte % 4u;
    let byte_val     = (weight_blocks[block_base + word_idx] >> (byte_in_word * 8u)) & 0xFFu;
    let nibble       = select(byte_val & 0xFu, (byte_val >> 4u) & 0xFu, nibble_half != 0u);
    return d * f32(i32(nibble) - 8);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }

    let n_blocks_per_row = params.n_cols / 32u;
    var acc = 0.0f;

    for (var blk = 0u; blk < n_blocks_per_row; blk++) {
        // Block index in the full weight tensor: row * n_blocks_per_row + blk
        let block_idx  = row * n_blocks_per_row + blk;
        // Each block occupies 5 u32 words.
        let block_base = block_idx * 5u;
        let col_base   = blk * 32u;

        for (var li = 0u; li < 32u; li++) {
            let w = get_q4_0_weight(block_base, li);
            acc += w * input_vec[col_base + li];
        }
    }

    output_vec[row] = acc;
}
"#;

// ── helper: call a method on a JS object by name ────────────────────────────

/// Call `obj.method_name(…args)` via `js_sys::Reflect`.
///
/// Returns an error `JsValue` if the property is not a function or the call
/// itself throws.
fn reflect_call(obj: &JsValue, method: &str, args: &js_sys::Array) -> Result<JsValue, JsValue> {
    let fn_val = Reflect::get(obj, &JsValue::from_str(method))
        .map_err(|_| JsValue::from_str(&format!("property '{method}' not found")))?;
    if !fn_val.is_function() {
        return Err(JsValue::from_str(&format!("'{method}' is not a function")));
    }
    Function::from(fn_val)
        .apply(obj, args)
        .map_err(|e| JsValue::from_str(&format!("'{method}' threw: {e:?}")))
}

/// Call `obj.method_name()` with no arguments.
fn reflect_call0(obj: &JsValue, method: &str) -> Result<JsValue, JsValue> {
    reflect_call(obj, method, &js_sys::Array::new())
}

/// Call `obj.method_name(arg)` with one argument.
fn reflect_call1(obj: &JsValue, method: &str, arg: &JsValue) -> Result<JsValue, JsValue> {
    let args = js_sys::Array::new();
    args.push(arg);
    reflect_call(obj, method, &args)
}

// ── helper: build a plain JS object from key-value pairs ────────────────────

/// Build a `JSObject` from a slice of `(key, value)` pairs.
fn js_obj(pairs: &[(&str, JsValue)]) -> Result<JsValue, JsValue> {
    let obj = Object::new();
    for (key, val) in pairs {
        Reflect::set(&obj, &JsValue::from_str(key), val)
            .map_err(|e| JsValue::from_str(&format!("Reflect::set({key}) failed: {e:?}")))?;
    }
    Ok(obj.into())
}

// ── GPU buffer utilities ─────────────────────────────────────────────────────

/// Create a `GPUBuffer` on `device` with the given `size` (bytes) and `usage`
/// flags. `mapped_at_creation` controls whether `mappedAtCreation` is set.
fn create_gpu_buffer(
    device: &JsValue,
    size: u64,
    usage: u32,
    mapped_at_creation: bool,
) -> Result<JsValue, JsValue> {
    let descriptor = js_obj(&[
        ("size", JsValue::from_f64(size as f64)),
        ("usage", JsValue::from_f64(usage as f64)),
        ("mappedAtCreation", JsValue::from_bool(mapped_at_creation)),
    ])?;
    reflect_call1(device, "createBuffer", &descriptor)
}

/// Write `bytes` into a `GPUBuffer` that was created with `mappedAtCreation =
/// true`, then unmap it.
///
/// The buffer must be at least `bytes.len()` bytes in size.
fn write_mapped_buffer(buffer: &JsValue, bytes: &[u8]) -> Result<(), JsValue> {
    let ab = reflect_call0(buffer, "getMappedRange")?;
    // SAFETY: We are operating inside wasm32-unknown-unknown. The ArrayBuffer
    // `ab` is a live JS heap object. `Uint8Array::new` takes a reference and
    // does not transfer ownership. We only write to it before unmapping.
    let typed = js_sys::Uint8Array::new(&ab);
    for (i, &b) in bytes.iter().enumerate() {
        typed.set_index(i as u32, b);
    }
    reflect_call0(buffer, "unmap")?;
    Ok(())
}

// ── init_webgpu_device ───────────────────────────────────────────────────────

/// Request a `GPUDevice` from the browser's WebGPU API.
///
/// Returns a `Promise` that resolves to the `GPUDevice` `JsValue`, or rejects
/// with a descriptive error string if WebGPU is unavailable or no adapter
/// could be obtained.
///
/// # JavaScript usage
///
/// ```js
/// const device = await initWebGpuDevice();
/// ```
#[wasm_bindgen(js_name = initWebGpuDevice)]
pub fn init_webgpu_device() -> Promise {
    future_to_promise(async {
        let global = js_sys::global();

        let navigator = Reflect::get(&global, &JsValue::from_str("navigator"))
            .map_err(|_| JsValue::from_str("no navigator object in global scope"))?;
        if navigator.is_undefined() || navigator.is_null() {
            return Err(JsValue::from_str("navigator is not available"));
        }

        let gpu = Reflect::get(&navigator, &JsValue::from_str("gpu"))
            .map_err(|_| JsValue::from_str("navigator.gpu property not found"))?;
        if gpu.is_undefined() || gpu.is_null() {
            return Err(JsValue::from_str(
                "WebGPU is not available in this browser (navigator.gpu is undefined)",
            ));
        }

        // requestAdapter() → Promise<GPUAdapter|null>
        let adapter_promise = reflect_call0(&gpu, "requestAdapter")?;
        let adapter = JsFuture::from(Promise::from(adapter_promise)).await?;
        if adapter.is_null() || adapter.is_undefined() {
            return Err(JsValue::from_str(
                "No suitable WebGPU adapter found (requestAdapter returned null)",
            ));
        }

        // requestDevice() → Promise<GPUDevice>
        let device_promise = reflect_call0(&adapter, "requestDevice")?;
        let device = JsFuture::from(Promise::from(device_promise)).await?;
        if device.is_null() || device.is_undefined() {
            return Err(JsValue::from_str(
                "requestDevice returned null — GPU device unavailable",
            ));
        }

        Ok(device)
    })
}

// ── webgpu_dequant_q4_0_async ────────────────────────────────────────────────

/// Dequantize Q4_0 blocks asynchronously on the GPU.
///
/// - `device` — a `GPUDevice` obtained from [`init_webgpu_device`].
/// - `data` — raw Q4_0 bytes (must be a multiple of 18 bytes per block).
/// - `n_blocks` — number of Q4_0 blocks in `data`.
///
/// Returns a `Promise` that resolves to a `Float32Array` of length
/// `n_blocks * 32`.
///
/// # JavaScript usage
///
/// ```js
/// const result = await webgpuDequantQ4_0Async(device, uint8Data, nBlocks);
/// // result is a Float32Array
/// ```
#[wasm_bindgen(js_name = webgpuDequantQ4_0Async)]
pub fn webgpu_dequant_q4_0_async(device: JsValue, data: Vec<u8>, n_blocks: u32) -> Promise {
    future_to_promise(async move { dequant_q4_0_on_gpu(&device, &data, n_blocks).await })
}

async fn dequant_q4_0_on_gpu(
    device: &JsValue,
    data: &[u8],
    n_blocks: u32,
) -> Result<JsValue, JsValue> {
    // Validate
    const BLOCK_BYTES: usize = 18;
    if data.len() != n_blocks as usize * BLOCK_BYTES {
        return Err(JsValue::from_str(&format!(
            "data length {} does not match n_blocks={n_blocks} × {BLOCK_BYTES}",
            data.len()
        )));
    }

    let n_weights = n_blocks as usize * 32;
    let output_bytes = (n_weights * 4) as u64;

    // Each Q4_0 block is 18 bytes; we pack into 5 u32 words (20 bytes) per
    // block for GPU-friendly alignment.
    let padded_per_block = 20usize; // 5 × 4 bytes
    let input_bytes_padded = n_blocks as usize * padded_per_block;
    let mut packed = vec![0u8; input_bytes_padded];
    for (blk, chunk) in data.chunks_exact(BLOCK_BYTES).enumerate() {
        let dst = &mut packed[blk * padded_per_block..blk * padded_per_block + BLOCK_BYTES];
        dst.copy_from_slice(chunk);
    }

    // GPU usage flags
    // STORAGE = 0x80, COPY_DST = 0x04, COPY_SRC = 0x08, MAP_READ = 0x01
    const STORAGE_COPY_DST: u32 = 0x80 | 0x04;
    const STORAGE_COPY_SRC: u32 = 0x80 | 0x08;
    const MAP_READ_COPY_DST: u32 = 0x01 | 0x04;

    // Create input buffer (mappedAtCreation = true → write data → unmap)
    let input_buf = create_gpu_buffer(device, packed.len() as u64, STORAGE_COPY_DST, true)?;
    write_mapped_buffer(&input_buf, &packed)?;

    // Create output buffer (STORAGE | COPY_SRC)
    let output_buf = create_gpu_buffer(device, output_bytes, STORAGE_COPY_SRC, false)?;

    // Uniform params buffer: n_blocks (u32, 4 bytes, aligned to 16)
    let mut uniform_data = [0u8; 16];
    uniform_data[..4].copy_from_slice(&n_blocks.to_le_bytes());
    const UNIFORM_COPY_DST: u32 = 0x40 | 0x04; // UNIFORM = 0x40
    let uniform_buf = create_gpu_buffer(device, 16, UNIFORM_COPY_DST, true)?;
    write_mapped_buffer(&uniform_buf, &uniform_data)?;

    // Create readback buffer (MAP_READ | COPY_DST)
    let readback_buf = create_gpu_buffer(device, output_bytes, MAP_READ_COPY_DST, false)?;

    // Create shader module
    let shader_desc = js_obj(&[("code", JsValue::from_str(Q4_0_DEQUANT_WGSL))])?;
    let shader_module = reflect_call1(device, "createShaderModule", &shader_desc)?;

    // Create compute pipeline
    let compute_stage = js_obj(&[
        ("module", shader_module),
        ("entryPoint", JsValue::from_str("main")),
    ])?;
    let pipeline_desc = js_obj(&[
        ("layout", JsValue::from_str("auto")),
        ("compute", compute_stage),
    ])?;
    let pipeline = reflect_call1(device, "createComputePipeline", &pipeline_desc)?;

    // Get bind group layout
    let layout = reflect_call1(&pipeline, "getBindGroupLayout", &JsValue::from_f64(0.0))?;

    // Build bind group entries
    let make_entry = |binding: u32, resource: JsValue| -> Result<JsValue, JsValue> {
        js_obj(&[
            ("binding", JsValue::from_f64(binding as f64)),
            ("resource", resource),
        ])
    };
    let buf_binding =
        |buf: &JsValue| -> Result<JsValue, JsValue> { js_obj(&[("buffer", buf.clone())]) };

    let entries = js_sys::Array::new();
    entries.push(&make_entry(0, buf_binding(&input_buf)?)?);
    entries.push(&make_entry(1, buf_binding(&output_buf)?)?);
    entries.push(&make_entry(2, buf_binding(&uniform_buf)?)?);

    let bg_desc = js_obj(&[("layout", layout), ("entries", entries.into())])?;
    let bind_group = reflect_call1(device, "createBindGroup", &bg_desc)?;

    // Command encoding
    let encoder = reflect_call0(device, "createCommandEncoder")?;
    let pass = reflect_call0(&encoder, "beginComputePass")?;
    reflect_call1(&pass, "setPipeline", &pipeline)?;
    let set_bg_args = js_sys::Array::new();
    set_bg_args.push(&JsValue::from_f64(0.0));
    set_bg_args.push(&bind_group);
    reflect_call(&pass, "setBindGroup", &set_bg_args)?;

    // Dispatch: ceil(n_weights / 256) workgroups
    let workgroups = (n_weights as u32).div_ceil(256);
    let dispatch_args = js_sys::Array::new();
    dispatch_args.push(&JsValue::from_f64(workgroups as f64));
    reflect_call(&pass, "dispatchWorkgroups", &dispatch_args)?;
    reflect_call0(&pass, "end")?;

    // Copy output → readback
    let copy_args = js_sys::Array::new();
    copy_args.push(&output_buf);
    copy_args.push(&JsValue::from_f64(0.0));
    copy_args.push(&readback_buf);
    copy_args.push(&JsValue::from_f64(0.0));
    copy_args.push(&JsValue::from_f64(output_bytes as f64));
    reflect_call(&encoder, "copyBufferToBuffer", &copy_args)?;

    let command_buffer = reflect_call0(&encoder, "finish")?;

    // Submit
    let queue = Reflect::get(device, &JsValue::from_str("queue"))
        .map_err(|_| JsValue::from_str("device.queue not found"))?;
    let submit_list = js_sys::Array::new();
    submit_list.push(&command_buffer);
    reflect_call1(&queue, "submit", &submit_list.into())?;

    // Map readback buffer for reading (MAP_READ = 1)
    let map_args = js_sys::Array::new();
    map_args.push(&JsValue::from_f64(1.0));
    let map_promise = reflect_call(&readback_buf, "mapAsync", &map_args)?;
    JsFuture::from(Promise::from(map_promise)).await?;

    // Read mapped range
    let ab = reflect_call0(&readback_buf, "getMappedRange")?;
    let float_arr = js_sys::Float32Array::new(&ab);

    // Copy to a new owned Float32Array (so we can unmap the readback buffer)
    let result_len = float_arr.length();
    let result_ab = js_sys::ArrayBuffer::new(result_len * 4);
    let result_arr = js_sys::Float32Array::new(&result_ab);
    result_arr.set(&float_arr, 0);

    reflect_call0(&readback_buf, "unmap")?;

    Ok(result_arr.into())
}

// ── webgpu_gemv_async ────────────────────────────────────────────────────────

/// GEMV on GPU: multiply Q4_0-quantized weight matrix by an f32 input vector.
///
/// - `device` — a `GPUDevice` obtained from [`init_webgpu_device`].
/// - `weights` — raw Q4_0 bytes for the weight matrix, laid out as
///   `rows` × `(cols/32)` blocks of 18 bytes each. `cols` must be a multiple
///   of 32.
/// - `input` — f32 input vector of length `cols`.
/// - `rows` — number of output rows.
/// - `cols` — number of columns (= input vector length, multiple of 32).
///
/// Returns a `Promise` that resolves to a `Float32Array` of length `rows`.
///
/// # JavaScript usage
///
/// ```js
/// const output = await webgpuGemvAsync(device, weightBytes, inputF32, rows, cols);
/// ```
#[wasm_bindgen(js_name = webgpuGemvAsync)]
pub fn webgpu_gemv_async(
    device: JsValue,
    weights: Vec<u8>,
    input: Vec<f32>,
    rows: u32,
    cols: u32,
) -> Promise {
    future_to_promise(async move { gemv_on_gpu(&device, &weights, &input, rows, cols).await })
}

async fn gemv_on_gpu(
    device: &JsValue,
    weights: &[u8],
    input: &[f32],
    rows: u32,
    cols: u32,
) -> Result<JsValue, JsValue> {
    // Validate
    if !cols.is_multiple_of(32) {
        return Err(JsValue::from_str(&format!(
            "cols={cols} must be a multiple of 32 for Q4_0 GEMV"
        )));
    }
    if input.len() as u32 != cols {
        return Err(JsValue::from_str(&format!(
            "input length {} does not match cols={cols}",
            input.len()
        )));
    }
    let n_blocks_per_row = cols / 32;
    let expected_weight_bytes = rows as usize * n_blocks_per_row as usize * 18;
    if weights.len() != expected_weight_bytes {
        return Err(JsValue::from_str(&format!(
            "weights length {} expected {expected_weight_bytes} (rows={rows}, cols={cols})",
            weights.len()
        )));
    }

    // Pack Q4_0 blocks into 5-u32 (20 byte) aligned blocks for GPU.
    const BLOCK_BYTES: usize = 18;
    let n_blocks_total = rows as usize * n_blocks_per_row as usize;
    let padded_per_block = 20usize;
    let weight_buf_size = n_blocks_total * padded_per_block;
    let mut packed_weights = vec![0u8; weight_buf_size];
    for (blk, chunk) in weights.chunks_exact(BLOCK_BYTES).enumerate() {
        let dst = &mut packed_weights[blk * padded_per_block..blk * padded_per_block + BLOCK_BYTES];
        dst.copy_from_slice(chunk);
    }

    // GPU buffers
    const STORAGE_COPY_DST: u32 = 0x80 | 0x04;
    const STORAGE_COPY_SRC: u32 = 0x80 | 0x08;
    const MAP_READ_COPY_DST: u32 = 0x01 | 0x04;

    // Weight buffer (input, read-only on GPU)
    let weight_buf = create_gpu_buffer(device, weight_buf_size as u64, STORAGE_COPY_DST, true)?;
    write_mapped_buffer(&weight_buf, &packed_weights)?;

    // Input vector buffer
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input_buf = create_gpu_buffer(device, input_bytes.len() as u64, STORAGE_COPY_DST, true)?;
    write_mapped_buffer(&input_buf, &input_bytes)?;

    // Output buffer (STORAGE | COPY_SRC, written by shader)
    let output_bytes_size = rows as u64 * 4;
    let output_buf = create_gpu_buffer(device, output_bytes_size, STORAGE_COPY_SRC, false)?;

    // Uniform buffer: rows (u32) + n_cols (u32) = 8 bytes, padded to 16
    let mut uniform_data = [0u8; 16];
    uniform_data[..4].copy_from_slice(&rows.to_le_bytes());
    uniform_data[4..8].copy_from_slice(&cols.to_le_bytes());
    const UNIFORM_COPY_DST: u32 = 0x40 | 0x04;
    let uniform_buf = create_gpu_buffer(device, 16, UNIFORM_COPY_DST, true)?;
    write_mapped_buffer(&uniform_buf, &uniform_data)?;

    // Readback buffer
    let readback_buf = create_gpu_buffer(device, output_bytes_size, MAP_READ_COPY_DST, false)?;

    // Shader module
    let shader_desc = js_obj(&[("code", JsValue::from_str(GEMV_WGSL))])?;
    let shader_module = reflect_call1(device, "createShaderModule", &shader_desc)?;

    // Compute pipeline
    let compute_stage = js_obj(&[
        ("module", shader_module),
        ("entryPoint", JsValue::from_str("main")),
    ])?;
    let pipeline_desc = js_obj(&[
        ("layout", JsValue::from_str("auto")),
        ("compute", compute_stage),
    ])?;
    let pipeline = reflect_call1(device, "createComputePipeline", &pipeline_desc)?;

    let layout = reflect_call1(&pipeline, "getBindGroupLayout", &JsValue::from_f64(0.0))?;

    let make_entry = |binding: u32, resource: JsValue| -> Result<JsValue, JsValue> {
        js_obj(&[
            ("binding", JsValue::from_f64(binding as f64)),
            ("resource", resource),
        ])
    };
    let buf_binding =
        |buf: &JsValue| -> Result<JsValue, JsValue> { js_obj(&[("buffer", buf.clone())]) };

    let entries = js_sys::Array::new();
    entries.push(&make_entry(0, buf_binding(&weight_buf)?)?);
    entries.push(&make_entry(1, buf_binding(&input_buf)?)?);
    entries.push(&make_entry(2, buf_binding(&output_buf)?)?);
    entries.push(&make_entry(3, buf_binding(&uniform_buf)?)?);

    let bg_desc = js_obj(&[("layout", layout), ("entries", entries.into())])?;
    let bind_group = reflect_call1(device, "createBindGroup", &bg_desc)?;

    // Command encoding
    let encoder = reflect_call0(device, "createCommandEncoder")?;
    let pass = reflect_call0(&encoder, "beginComputePass")?;
    reflect_call1(&pass, "setPipeline", &pipeline)?;

    let set_bg_args = js_sys::Array::new();
    set_bg_args.push(&JsValue::from_f64(0.0));
    set_bg_args.push(&bind_group);
    reflect_call(&pass, "setBindGroup", &set_bg_args)?;

    // Dispatch: ceil(rows / 256) workgroups
    let workgroups = rows.div_ceil(256);
    let dispatch_args = js_sys::Array::new();
    dispatch_args.push(&JsValue::from_f64(workgroups as f64));
    reflect_call(&pass, "dispatchWorkgroups", &dispatch_args)?;
    reflect_call0(&pass, "end")?;

    // Copy output → readback
    let copy_args = js_sys::Array::new();
    copy_args.push(&output_buf);
    copy_args.push(&JsValue::from_f64(0.0));
    copy_args.push(&readback_buf);
    copy_args.push(&JsValue::from_f64(0.0));
    copy_args.push(&JsValue::from_f64(output_bytes_size as f64));
    reflect_call(&encoder, "copyBufferToBuffer", &copy_args)?;

    let command_buffer = reflect_call0(&encoder, "finish")?;

    // Submit
    let queue = Reflect::get(device, &JsValue::from_str("queue"))
        .map_err(|_| JsValue::from_str("device.queue not found"))?;
    let submit_list = js_sys::Array::new();
    submit_list.push(&command_buffer);
    reflect_call1(&queue, "submit", &submit_list.into())?;

    // mapAsync(MAP_READ = 1)
    let map_args = js_sys::Array::new();
    map_args.push(&JsValue::from_f64(1.0));
    let map_promise = reflect_call(&readback_buf, "mapAsync", &map_args)?;
    JsFuture::from(Promise::from(map_promise)).await?;

    // Read result
    let ab = reflect_call0(&readback_buf, "getMappedRange")?;
    let float_arr = js_sys::Float32Array::new(&ab);

    let result_len = float_arr.length();
    let result_ab = js_sys::ArrayBuffer::new(result_len * 4);
    let result_arr = js_sys::Float32Array::new(&result_ab);
    result_arr.set(&float_arr, 0);

    reflect_call0(&readback_buf, "unmap")?;

    Ok(result_arr.into())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the WGSL shader strings are non-empty (compile-time check).
    #[test]
    fn shaders_are_non_empty() {
        assert!(!Q4_0_DEQUANT_WGSL.is_empty());
        assert!(!GEMV_WGSL.is_empty());
    }

    /// Verify Q4_0 packing logic: 18-byte blocks padded to 20 bytes.
    #[test]
    fn pack_q4_0_blocks_alignment() {
        // Two blocks → 40 padded bytes
        let data = [0xABu8; 36]; // 2 × 18
        let n_blocks = 2usize;
        const BLOCK_BYTES: usize = 18;
        let padded_per_block = 20usize;
        let mut packed = vec![0u8; n_blocks * padded_per_block];
        for (blk, chunk) in data.chunks_exact(BLOCK_BYTES).enumerate() {
            let dst = &mut packed[blk * padded_per_block..blk * padded_per_block + BLOCK_BYTES];
            dst.copy_from_slice(chunk);
        }
        assert_eq!(packed.len(), 40);
        // First 18 bytes of each block should match input data.
        assert_eq!(&packed[0..18], &data[0..18]);
        assert_eq!(&packed[20..38], &data[18..36]);
        // Padding bytes should be zero.
        assert_eq!(&packed[18..20], &[0u8, 0u8]);
        assert_eq!(&packed[38..40], &[0u8, 0u8]);
    }

    /// Verify uniform buffer layout for GEMV params.
    #[test]
    fn gemv_uniform_layout() {
        let rows: u32 = 64;
        let cols: u32 = 128;
        let mut uniform = [0u8; 16];
        uniform[..4].copy_from_slice(&rows.to_le_bytes());
        uniform[4..8].copy_from_slice(&cols.to_le_bytes());
        assert_eq!(
            u32::from_le_bytes(uniform[..4].try_into().expect("4 bytes")),
            rows
        );
        assert_eq!(
            u32::from_le_bytes(uniform[4..8].try_into().expect("4 bytes")),
            cols
        );
    }

    /// Verify workgroup dispatch count calculation.
    #[test]
    fn workgroup_count_ceiling() {
        assert_eq!(256u32.div_ceil(256), 1);
        assert_eq!(257u32.div_ceil(256), 2);
        assert_eq!(512u32.div_ceil(256), 2);
        assert_eq!(513u32.div_ceil(256), 3);
    }
}

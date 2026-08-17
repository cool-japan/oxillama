// gemv_q4_0_resident.wgsl — Q4_0 GEMV with in-shader dequantisation.
//
// Unlike gemv_f32.wgsl (a format-agnostic dot product over an already
// f32-dequantised weight matrix), this shader consumes the QUANTISED weight
// data directly: an f32 scale per block plus the raw 4-bit nibble bytes
// (packed 8 per u32 word — already word-aligned, since Q4_0's 16-byte
// nibble section is exactly 4 u32 words). Dequantisation and the multiply
// happen together in registers, so the upload payload is the quantised byte
// count (0.5 byte/weight for `qs`, plus one f32 scale per 32 weights) rather
// than 4 bytes/weight for a pre-dequantised f32 buffer.
//
// Split-half nibble layout (matches `dequantize_row_q4_0` in
// `llama.cpp/ggml/src/ggml-quants.c` and
// `oxillama_quant::reference::q4_0`): within a block's 16-byte nibble
// section, byte `i` carries weight `i` in its low nibble and weight
// `i + 16` in its high nibble — NOT interleaved pairs.
//
// Computes: output[row] = sum_{j=0}^{cols-1}( weight[row, j] * input[j] )

@group(0) @binding(0) var<storage, read>       qs:     array<u32>;  // 4 words per block
@group(0) @binding(1) var<storage, read>       scales: array<f32>;  // 1 per block
@group(0) @binding(2) var<storage, read>       input:  array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
    rows: u32,
    cols: u32,
    blocks_per_row: u32,
    _pad: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }

    var acc: f32 = 0.0;

    for (var blk: u32 = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let block_idx = row * params.blocks_per_row + blk;
        let d = scales[block_idx];
        let word_base = block_idx * 4u;
        let col_base = blk * 32u;

        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let word = qs[word_base + w];
            for (var bi: u32 = 0u; bi < 4u; bi = bi + 1u) {
                let byte = (word >> (bi * 8u)) & 0xFFu;
                let i = w * 4u + bi; // byte index within the block's 16-byte qs section, 0..16

                let lo = i32(byte & 0xFu) - 8;
                let hi = i32((byte >> 4u) & 0xFu) - 8;

                let col0 = col_base + i;
                let col1 = col0 + 16u;

                if col0 < params.cols {
                    acc += d * f32(lo) * input[col0];
                }
                if col1 < params.cols {
                    acc += d * f32(hi) * input[col1];
                }
            }
        }
    }

    output[row] = acc;
}

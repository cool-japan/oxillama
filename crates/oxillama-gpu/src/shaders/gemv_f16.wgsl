// gemv_f16.wgsl — GEMV with f16 weights (packed as u32) and f32 input/output.
//
// Weights are stored as packed u32 values, each containing two f16 values.
// The shader unpacks pairs via unpack2x16float(), multiplies by f32 input,
// and accumulates in f32 precision.  This halves weight-memory bandwidth
// compared to the gemv_f32 path.
//
// Dispatch with ceil(rows / 256) workgroups of size 256.

struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read>       weights: array<u32>;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<uniform>             params:  Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) { return; }

    let cols = params.cols;
    let half_cols = (cols + 1u) / 2u;
    var sum: f32 = 0.0;

    for (var j = 0u; j < half_cols; j = j + 1u) {
        let packed = weights[row * half_cols + j];
        let unpacked = unpack2x16float(packed);
        let col0 = j * 2u;
        let col1 = col0 + 1u;

        if (col0 < cols) {
            sum = sum + unpacked.x * input[col0];
        }
        if (col1 < cols) {
            sum = sum + unpacked.y * input[col1];
        }
    }

    output[row] = sum;
}

// gemv_f32.wgsl — generic f32 GEMV
//
// Computes: output[row] = sum_{j=0}^{cols-1}( weight[row * cols + j] * input[j] )
//
// Each workgroup thread handles one output row.  Dispatch with (rows, 1, 1)
// groups of size 64; unused threads are masked by the bounds check on line 1.

@group(0) @binding(0) var<storage, read>       weight: array<f32>;
@group(0) @binding(1) var<storage, read>       input:  array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    rows: u32,
    cols: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < params.cols; j = j + 1u) {
        acc += weight[row * params.cols + j] * input[j];
    }
    output[row] = acc;
}

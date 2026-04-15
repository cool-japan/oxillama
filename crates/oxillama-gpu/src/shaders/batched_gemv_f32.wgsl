// batched_gemv_f32.wgsl — batched f32 GEMV
//
// Computes: output[batch * rows + row] =
//   sum_{c=0}^{cols-1}( matrix[row * cols + c] * vectors[batch * cols + c] )
//
// Each thread handles one (row, batch) output element.
// Dispatch: (ceil(rows / 64), batch_size, 1)

@group(0) @binding(0) var<storage, read>       matrix:  array<f32>;
@group(0) @binding(1) var<storage, read>       vectors: array<f32>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;

struct Params {
    rows: u32,
    cols: u32,
    batch_size: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row   = gid.x;
    let batch = gid.y;

    if row >= params.rows || batch >= params.batch_size {
        return;
    }

    var acc: f32 = 0.0;
    let mat_offset = row * params.cols;
    let vec_offset = batch * params.cols;

    for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
        acc += matrix[mat_offset + c] * vectors[vec_offset + c];
    }

    output[batch * params.rows + row] = acc;
}

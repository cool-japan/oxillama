// gemm_f32.wgsl — tiled f32 GEMM with workgroup shared memory
//
// Computes: C[M × N] = A[M × K] × B[K × N]  (row-major)
//
// Tile parameters:
//   TILE_M = 32  (rows handled per workgroup in Y dimension)
//   TILE_N = 32  (cols handled per workgroup in X dimension)
//   TILE_K = 16  (K-dimension tile depth)
//
// Workgroup: 16 × 16 = 256 threads
//   local_id.x ∈ [0, 16)   → selects column within tile
//   local_id.y ∈ [0, 16)   → selects row within tile
//
// Shared memory:
//   A_tile[TILE_M × TILE_K] = 32 × 16 = 512 f32  (2 KiB)
//   B_tile[TILE_K × TILE_N] = 16 × 32 = 512 f32  (2 KiB)
//   Total = 4 KiB — well within the 16 KiB WebGPU minimum.
//
// Cooperative loading:
//   256 threads, 512 elements per tile → each thread loads 2 elements.
//   Thread tid = local_id.y * 16 + local_id.x  (∈ [0, 256))
//   For A_tile: element at flat index tid and tid + 256.
//   Same for B_tile.

struct Params {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read>       A:      array<f32>;  // [M × K]
@group(0) @binding(1) var<storage, read>       B:      array<f32>;  // [K × N]
@group(0) @binding(2) var<storage, read_write> C:      array<f32>;  // [M × N]
@group(0) @binding(3) var<uniform>             params: Params;

// Shared tiles — 512 elements each = 2 KiB each = 4 KiB total.
var<workgroup> A_tile: array<f32, 512>; // TILE_M × TILE_K = 32 × 16
var<workgroup> B_tile: array<f32, 512>; // TILE_K × TILE_N = 16 × 32

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    // Output element this thread is responsible for.
    let row = wg_id.y * 32u + local_id.y * 2u;
    let col = wg_id.x * 32u + local_id.x * 2u;

    // Thread index within the workgroup (0..255).
    let tid = local_id.y * 16u + local_id.x;

    // Each thread accumulates a 2×2 output sub-tile.
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    // Number of K-dimension tiles (ceiling division).
    let k_tiles = (params.K + 15u) / 16u;

    for (var kt: u32 = 0u; kt < k_tiles; kt++) {
        // ── Cooperative load A_tile[TILE_M × TILE_K] ──────────────────────
        // A_tile is stored row-major: A_tile[r * 16 + k_local].
        // 256 threads load 512 elements: thread tid loads indices tid and tid+256.
        {
            // Element 0: flat index = tid → row = tid / 16, k_local = tid % 16
            let r0 = tid / 16u;
            let k0 = tid % 16u;
            let a_row0 = wg_id.y * 32u + r0;
            let a_col0 = kt * 16u + k0;
            if a_row0 < params.M && a_col0 < params.K {
                A_tile[r0 * 16u + k0] = A[a_row0 * params.K + a_col0];
            } else {
                A_tile[r0 * 16u + k0] = 0.0;
            }

            // Element 1: flat index = tid + 256
            let r1 = (tid + 256u) / 16u;
            let k1 = (tid + 256u) % 16u;
            let a_row1 = wg_id.y * 32u + r1;
            let a_col1 = kt * 16u + k1;
            if a_row1 < params.M && a_col1 < params.K {
                A_tile[r1 * 16u + k1] = A[a_row1 * params.K + a_col1];
            } else {
                A_tile[r1 * 16u + k1] = 0.0;
            }
        }

        // ── Cooperative load B_tile[TILE_K × TILE_N] ──────────────────────
        // B_tile is stored row-major: B_tile[k_local * 32 + c].
        // Thread tid loads elements tid and tid+256.
        {
            // Element 0
            let k0 = tid / 32u;
            let c0 = tid % 32u;
            let b_row0 = kt * 16u + k0;
            let b_col0 = wg_id.x * 32u + c0;
            if b_row0 < params.K && b_col0 < params.N {
                B_tile[k0 * 32u + c0] = B[b_row0 * params.N + b_col0];
            } else {
                B_tile[k0 * 32u + c0] = 0.0;
            }

            // Element 1: flat index = tid + 256
            let k1 = (tid + 256u) / 32u;
            let c1 = (tid + 256u) % 32u;
            let b_row1 = kt * 16u + k1;
            let b_col1 = wg_id.x * 32u + c1;
            if b_row1 < params.K && b_col1 < params.N {
                B_tile[k1 * 32u + c1] = B[b_row1 * params.N + b_col1];
            } else {
                B_tile[k1 * 32u + c1] = 0.0;
            }
        }

        workgroupBarrier();

        // ── Accumulate 2×2 sub-tile ────────────────────────────────────────
        // Thread (local_id.x, local_id.y) owns output rows [2*ly, 2*ly+1]
        // and output cols [2*lx, 2*lx+1].
        let ly = local_id.y;
        let lx = local_id.x;

        for (var k: u32 = 0u; k < 16u; k++) {
            let a0 = A_tile[(2u * ly + 0u) * 16u + k];
            let a1 = A_tile[(2u * ly + 1u) * 16u + k];
            let b0 = B_tile[k * 32u + 2u * lx + 0u];
            let b1 = B_tile[k * 32u + 2u * lx + 1u];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        workgroupBarrier();
    }

    // ── Write 2×2 result to C ──────────────────────────────────────────────
    if row + 0u < params.M && col + 0u < params.N {
        C[(row + 0u) * params.N + col + 0u] = acc00;
    }
    if row + 0u < params.M && col + 1u < params.N {
        C[(row + 0u) * params.N + col + 1u] = acc01;
    }
    if row + 1u < params.M && col + 0u < params.N {
        C[(row + 1u) * params.N + col + 0u] = acc10;
    }
    if row + 1u < params.M && col + 1u < params.N {
        C[(row + 1u) * params.N + col + 1u] = acc11;
    }
}

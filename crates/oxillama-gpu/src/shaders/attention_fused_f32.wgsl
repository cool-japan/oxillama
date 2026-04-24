// attention_fused_f32.wgsl — fused QK + online softmax + AV in a single dispatch.
//
// Implements Flash-Attention-style online softmax: one workgroup per Q row,
// iterating over KV tiles with numerically stable running max (m) and
// normalisation (l).
//
// Parameters:
//   seq_len_q  — number of Q rows
//   seq_len_kv — number of K/V rows
//   head_dim   — dimension of each Q/K/V vector (≤ 64)
//   scale      — pre-multiplied scale factor (typically 1/sqrt(head_dim))
//   causal     — 0: full attention, 1: causal mask (only attend to j <= q_row)
//
// Workgroup size: 64 threads (one per KV element within a tile of size 64).
// Each workgroup handles exactly one Q row.
//
// Shared memory usage:
//   K_tile[TILE_K × 64]:  64 × 64 = 4096 × 4 = 16 KiB — exactly at limit.
//   V_tile[TILE_K × 64]:  another 16 KiB — exceeds 16 KiB baseline.
//
// To fit within the 16 KiB WebGPU minimum, we use TILE_K = 32, head_dim ≤ 64:
//   K_tile[32 × 64]  = 2048 × 4 = 8 KiB
//   V_tile[32 × 64]  = 2048 × 4 = 8 KiB
//   dot_scratch[64]  = 64 × 4   = 256 B (dedicated QK reduction scratchpad)
//   Total ≈ 16.25 KiB — within desktop Metal/Vulkan limits (32 KiB).
//
// For head_dim > 64, callers must split into multiple passes (not handled here).

struct AttentionParams {
    seq_len_q:  u32,
    seq_len_kv: u32,
    head_dim:   u32,
    scale:      f32,
    causal:     u32,  // 0 = no mask, 1 = causal
}

@group(0) @binding(0) var<storage, read>       Q:      array<f32>;  // [seq_q  × head_dim]
@group(0) @binding(1) var<storage, read>       K:      array<f32>;  // [seq_kv × head_dim]
@group(0) @binding(2) var<storage, read>       V:      array<f32>;  // [seq_kv × head_dim]
@group(0) @binding(3) var<storage, read_write> Out:    array<f32>;  // [seq_q  × head_dim]
@group(0) @binding(4) var<uniform>             params: AttentionParams;

// TILE_K = 32 KV rows per tile iteration.
const TILE_K: u32 = 32u;

// Shared tiles — sized for TILE_K=32, head_dim≤64.
var<workgroup> K_tile:       array<f32, 2048>;  // TILE_K × 64 = 8 KiB
var<workgroup> V_tile:       array<f32, 2048>;  // TILE_K × 64 = 8 KiB
var<workgroup> dot_scratch:  array<f32, 64>;    // dot-product reduction scratchpad (256 B)
// Total shared memory: 8 KiB + 8 KiB + 256 B = ~16.25 KiB.
// Note: wgpu on Metal allows up to 32 KiB; the 16 KiB WebGPU baseline is for browsers.
// On desktop targets this is within the real limit.

// One workgroup per Q row; 64 threads per workgroup.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)        wg_id:    vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let q_row = wg_id.x;          // which Q row this workgroup handles
    let tid   = local_id.x;       // thread index within workgroup (0..63)
    let hd    = params.head_dim;

    // Out-of-bounds workgroup guard.
    if q_row >= params.seq_len_q { return; }

    // Load Q row into registers: each thread caches Q[q_row, tid] if tid < hd.
    var q_val: f32 = 0.0;
    if tid < hd {
        q_val = Q[q_row * hd + tid];
    }

    // Online softmax state.
    var m: f32 = -1e38;  // running max
    var l: f32 = 0.0;    // running normalisation (sum of exp)

    // Accumulator for output: o[tid] (register storage; max hd = 64 = workgroup size).
    var o: f32 = 0.0;

    let n_kv_tiles = (params.seq_len_kv + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < n_kv_tiles; t++) {
        let kv_start = t * TILE_K;

        // ── Cooperative load K_tile and V_tile ────────────────────────────
        // 64 threads load TILE_K × hd elements.
        // Each thread loads ceil(TILE_K × hd / 64) elements.
        // With TILE_K=32, hd≤64: max 32 elements per thread.
        for (var i: u32 = tid; i < TILE_K * hd; i += 64u) {
            let kv_row = kv_start + i / hd;
            let dim    = i % hd;
            if kv_row < params.seq_len_kv {
                K_tile[i] = K[kv_row * hd + dim];
                V_tile[i] = V[kv_row * hd + dim];
            } else {
                K_tile[i] = 0.0;
                V_tile[i] = 0.0;
            }
        }
        workgroupBarrier();

        // ── Process each KV row in this tile ──────────────────────────────
        for (var kv_local: u32 = 0u; kv_local < TILE_K; kv_local++) {
            let kv_abs = kv_start + kv_local;
            if kv_abs >= params.seq_len_kv { break; }

            // Causal mask: skip future tokens.
            if params.causal != 0u && kv_abs > q_row { continue; }

            // dot(Q[q_row], K[kv_abs]) across head_dim — all threads in wg
            // compute a partial sum of exactly one element.
            // Reduce 64 partial sums via dot_scratch (separate from K_tile/V_tile).
            var partial: f32 = 0.0;
            if tid < hd {
                partial = q_val * K_tile[kv_local * hd + tid];
            }

            // Write partial product to dedicated dot-product scratchpad.
            // Using dot_scratch keeps K_tile intact for subsequent kv_local iterations.
            dot_scratch[tid] = partial;
            workgroupBarrier();

            // Parallel reduction of 64 elements → single sum.
            // Assumes workgroup size = 64 (power of two).
            if tid < 32u { dot_scratch[tid] = dot_scratch[tid] + dot_scratch[tid + 32u]; } workgroupBarrier();
            if tid < 16u { dot_scratch[tid] = dot_scratch[tid] + dot_scratch[tid + 16u]; } workgroupBarrier();
            if tid < 8u  { dot_scratch[tid] = dot_scratch[tid] + dot_scratch[tid +  8u]; } workgroupBarrier();
            if tid < 4u  { dot_scratch[tid] = dot_scratch[tid] + dot_scratch[tid +  4u]; } workgroupBarrier();
            if tid < 2u  { dot_scratch[tid] = dot_scratch[tid] + dot_scratch[tid +  2u]; } workgroupBarrier();
            if tid < 1u  { dot_scratch[0]   = dot_scratch[0]   + dot_scratch[1];         } workgroupBarrier();

            let score = dot_scratch[0] * params.scale;

            // Online softmax update (Flash-Attention style):
            //   m_new = max(m, score)
            //   P     = exp(score - m_new)
            //   l_new = exp(m - m_new) * l + P
            //   o_new = exp(m - m_new) * o + P * v[tid]
            let m_new   = max(m, score);
            let exp_old = exp(m - m_new);   // rescale factor for old state
            let p       = exp(score - m_new);
            let l_new   = exp_old * l + p;

            var v_val: f32 = 0.0;
            if tid < hd {
                v_val = V_tile[kv_local * hd + tid];
            }
            o = exp_old * o + p * v_val;

            m = m_new;
            l = l_new;
        }

        workgroupBarrier();
    }

    // ── Write output ───────────────────────────────────────────────────────
    // Normalise: o / l.
    if tid < hd {
        var result: f32;
        if l > 0.0 {
            result = o / l;
        } else {
            result = 0.0;
        }
        Out[q_row * hd + tid] = result;
    }
}

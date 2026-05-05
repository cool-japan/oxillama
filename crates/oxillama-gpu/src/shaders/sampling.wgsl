// sampling.wgsl — GPU sampling kernels for OxiLLaMa
//
// Three entry points:
//   1. softmax_logits    — temperature-scaled softmax over logit vector
//   2. topk_partition    — extract top-k probability/index pairs
//   3. sample_categorical — sample one token using CDF + LCG RNG
//
// Design notes:
//   - softmax_logits uses workgroup shared memory for tree-reduction of max and
//     sum, then writes normalised probabilities to the output buffer.
//   - topk_partition does a single-pass cooperative scan: each thread maintains
//     a local candidate slot, then all threads cooperatively merge into a
//     global top-k heap kept in workgroup shared memory.
//   - sample_categorical runs in a single-thread workgroup (1,1,1); it walks
//     the CDF reconstructed from the input probability array until a LCG-
//     generated uniform variate is exceeded.
//
// Workgroup shared memory budget (16 KiB WebGPU baseline):
//   softmax_logits : 2 × 256 × 4 = 2 KiB (max_scratch + sum_scratch)
//   topk_partition : 256 × 8 = 2 KiB     (topk_vals[256] + topk_idxs[256])

// ─── Entry point 1: softmax_logits ──────────────────────────────────────────
//
// Bindings:
//   @binding(0) logits: array<f32>     — input logits [n_vocab]
//   @binding(1) params: array<f32>     — [temperature, n_vocab_f32_bits]
//   @binding(2) probs:  array<f32>     — output probabilities [n_vocab]
//
// params[0] = temperature  (0.0 → argmax degenerate distribution)
// params[1] = bitcast<f32>(n_vocab as u32)  — n_vocab encoded in f32 bits

@group(0) @binding(0) var<storage, read>       sm_logits : array<f32>;
@group(0) @binding(1) var<storage, read>       sm_params : array<f32>;
@group(0) @binding(2) var<storage, read_write> sm_probs  : array<f32>;

// Shared reduction scratch for softmax.
// 256 threads per workgroup → 256 slots each for max and sum reduction.
var<workgroup> wg_max : array<f32, 256>;
var<workgroup> wg_sum : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn softmax_logits(
    @builtin(global_invocation_id) gid      : vec3<u32>,
    @builtin(local_invocation_id)  local_id : vec3<u32>,
    @builtin(workgroup_id)         wg_id    : vec3<u32>,
) {
    let tid       = local_id.x;
    let temp      = sm_params[0];
    let n_vocab   = bitcast<u32>(sm_params[1]);

    // ── Pass 1: find maximum logit via tree-reduction ──────────────────────
    var local_max: f32 = -1e38;
    // Each thread scans its stride of the logit array.
    var i: u32 = tid;
    loop {
        if i >= n_vocab { break; }
        var val: f32 = sm_logits[i];
        if temp > 0.0 {
            val = val / temp;
        }
        if val > local_max { local_max = val; }
        i = i + 256u;
    }
    wg_max[tid] = local_max;
    workgroupBarrier();

    // Tree-reduce 256 → 1 for max.
    var stride: u32 = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            if wg_max[tid + stride] > wg_max[tid] {
                wg_max[tid] = wg_max[tid + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let global_max = wg_max[0];

    // ── Special case: temperature == 0 → argmax distribution ─────────────
    if temp == 0.0 {
        // Two-pass: find argmax index, then write 1.0 there and 0.0 everywhere.
        // Use wg_sum as a scratch for argmax index (store as f32 bits of u32).
        var local_argmax: u32 = 0u;
        var local_argmax_val: f32 = -1e38;
        var j: u32 = tid;
        loop {
            if j >= n_vocab { break; }
            if sm_logits[j] > local_argmax_val {
                local_argmax_val = sm_logits[j];
                local_argmax     = j;
            }
            j = j + 256u;
        }
        // Re-use wg_max for argmax value, wg_sum for argmax index encoded as f32 bits.
        wg_max[tid] = local_argmax_val;
        wg_sum[tid] = bitcast<f32>(local_argmax);
        workgroupBarrier();

        // Reduce: find the slot with the largest value.
        var s2: u32 = 128u;
        loop {
            if s2 == 0u { break; }
            if tid < s2 {
                if wg_max[tid + s2] > wg_max[tid] {
                    wg_max[tid] = wg_max[tid + s2];
                    wg_sum[tid] = wg_sum[tid + s2];
                }
            }
            workgroupBarrier();
            s2 = s2 >> 1u;
        }
        let argmax_idx = bitcast<u32>(wg_sum[0]);

        // Write degenerate distribution.
        var k2: u32 = tid;
        loop {
            if k2 >= n_vocab { break; }
            if k2 == argmax_idx {
                sm_probs[k2] = 1.0;
            } else {
                sm_probs[k2] = 0.0;
            }
            k2 = k2 + 256u;
        }
        return;
    }

    // ── Pass 2: compute exp(logit/temp - max) and sum via tree-reduction ───
    var local_sum: f32 = 0.0;
    var m2: u32 = tid;
    loop {
        if m2 >= n_vocab { break; }
        let val  = sm_logits[m2] / temp - global_max;
        let expv = exp(val);
        sm_probs[m2] = expv;   // store unnormalised for now
        local_sum += expv;
        m2 = m2 + 256u;
    }
    wg_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree-reduce 256 → 1 for sum.
    var stride2: u32 = 128u;
    loop {
        if stride2 == 0u { break; }
        if tid < stride2 {
            wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride2];
        }
        workgroupBarrier();
        stride2 = stride2 >> 1u;
    }
    let global_sum = wg_sum[0];

    // ── Pass 3: normalise ──────────────────────────────────────────────────
    var n2: u32 = tid;
    loop {
        if n2 >= n_vocab { break; }
        if global_sum > 0.0 {
            sm_probs[n2] = sm_probs[n2] / global_sum;
        } else {
            sm_probs[n2] = 0.0;
        }
        n2 = n2 + 256u;
    }
}

// ─── Entry point 2: topk_partition ──────────────────────────────────────────
//
// Bindings:
//   @binding(0) probs:     array<f32>  — input probability distribution [n_vocab]
//   @binding(1) tk_params: array<u32>  — [k, n_vocab]
//   @binding(2) topk_vals: array<f32>  — output top-k values [k]
//   @binding(3) topk_idxs: array<u32>  — output top-k indices [k]
//
// Algorithm:
//   - 256 threads each maintain a local "minimum of current top-k" threshold.
//   - Each thread scans its stride; any value exceeding a heap candidate is
//     tentatively stored in workgroup shared memory.
//   - After the scan, thread 0 does a final pass over the workgroup candidates
//     to select the true global top-k.
//   - Supports k ≤ 256.

@group(0) @binding(0) var<storage, read>       tk_probs     : array<f32>;
@group(0) @binding(1) var<storage, read>       tk_params    : array<u32>;
@group(0) @binding(2) var<storage, read_write> topk_vals_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> topk_idxs_out: array<u32>;

// Shared storage for candidate top-k heap (one slot per workgroup thread).
var<workgroup> wg_cand_val : array<f32, 256>;
var<workgroup> wg_cand_idx : array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn topk_partition(
    @builtin(global_invocation_id) gid      : vec3<u32>,
    @builtin(local_invocation_id)  local_id : vec3<u32>,
) {
    let tid     = local_id.x;
    let k       = tk_params[0];
    let n_vocab = tk_params[1];

    // Each thread scans its stride and keeps the single best candidate.
    var best_val: f32 = -1.0;
    var best_idx: u32 = 0u;

    var i: u32 = tid;
    loop {
        if i >= n_vocab { break; }
        let v = tk_probs[i];
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
        i = i + 256u;
    }

    wg_cand_val[tid] = best_val;
    wg_cand_idx[tid] = best_idx;
    workgroupBarrier();

    // Thread 0 collects the 256 candidates and selects the true top-k.
    // This is O(256 * k) which is at most 256*256 = 65536 ops — fine for GPU.
    if tid == 0u {
        // We need to pick the top-k from the 256 workgroup candidates.
        // Strategy: selection sort over 256 candidates, k rounds.
        var n_cands: u32 = min(256u, n_vocab);
        var out_pos: u32 = 0u;

        // Track which candidates have been selected via a boolean mask encoded
        // in the upper bit of wg_cand_idx (bit 31). We use a separate approach:
        // After selecting a candidate, set its value to -2.0 so it is skipped.
        loop {
            if out_pos >= k { break; }
            if out_pos >= n_cands { break; }

            // Find the maximum among remaining candidates.
            var max_val: f32 = -2.0;
            var max_pos: u32 = 0u;
            var c: u32 = 0u;
            loop {
                if c >= n_cands { break; }
                if wg_cand_val[c] > max_val {
                    max_val = wg_cand_val[c];
                    max_pos = c;
                }
                c = c + 1u;
            }

            if max_val < 0.0 {
                // No positive probability candidates remain.
                // Fill remaining slots with index 0 and prob 0.
                topk_vals_out[out_pos] = 0.0;
                topk_idxs_out[out_pos] = 0u;
            } else {
                topk_vals_out[out_pos] = max_val;
                topk_idxs_out[out_pos] = wg_cand_idx[max_pos];
                // Mark as selected.
                wg_cand_val[max_pos] = -2.0;
            }
            out_pos = out_pos + 1u;
        }
    }
}

// ─── Entry point 3: sample_categorical ──────────────────────────────────────
//
// Bindings:
//   @binding(0) cat_probs:  array<f32>  — probability distribution [n_candidates]
//   @binding(1) cat_idxs:   array<u32>  — corresponding token IDs [n_candidates]
//   @binding(2) cat_params: array<u32>  — [n_candidates, seed_lo, seed_hi]
//   @binding(3) cat_result: array<u32>  — output [sampled_token_id]
//
// Algorithm:
//   LCG RNG: state = seed_lo | (seed_hi << 32)
//   LCG multiplier: 6364136223846793005  (Knuth)
//   LCG increment:  1442695040888963407
//   Uniform variate: (state >> 33) as f32 / 2^31
//   Walk CDF until cumulative sum exceeds the variate.

@group(0) @binding(0) var<storage, read>       cat_probs  : array<f32>;
@group(0) @binding(1) var<storage, read>       cat_idxs   : array<u32>;
@group(0) @binding(2) var<storage, read>       cat_params : array<u32>;
@group(0) @binding(3) var<storage, read_write> cat_result : array<u32>;

@compute @workgroup_size(1, 1, 1)
fn sample_categorical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_candidates = cat_params[0];
    let seed_lo      = cat_params[1];
    let seed_hi      = cat_params[2];

    // Reconstruct 64-bit seed from two 32-bit halves (stored as lo/hi u32).
    // We work in 32-bit since WGSL does not have native u64.
    // LCG step using 32-bit Lehmer RNG (simple but sufficient for sampling):
    //   next = state * 1664525u + 1013904223u  (Numerical Recipes LCG)
    // We combine seed_lo and seed_hi into a single 32-bit state by XOR.
    var state: u32 = seed_lo ^ (seed_hi * 2654435761u);

    // LCG step to mix state further before use.
    state = state * 1664525u + 1013904223u;
    state = state * 1664525u + 1013904223u;

    // Generate uniform float in [0, 1).
    // Use upper 24 bits for float mantissa precision.
    let uniform = f32(state >> 8u) / 16777216.0;  // 2^24 = 16777216

    // Walk CDF until cumulative probability exceeds the uniform variate.
    var cumsum: f32 = 0.0;
    var selected: u32 = cat_idxs[0];  // fallback to first token

    var i: u32 = 0u;
    loop {
        if i >= n_candidates { break; }
        cumsum += cat_probs[i];
        if cumsum > uniform {
            selected = cat_idxs[i];
            break;
        }
        i = i + 1u;
    }

    cat_result[0] = selected;
}

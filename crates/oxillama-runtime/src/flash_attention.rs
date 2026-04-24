//! Tiled flash-attention kernel for memory-efficient attention computation.
//!
//! Implements the FlashAttention algorithm (Dao et al. 2022) in pure Rust.
//! Processes Q, K, V in tiles to avoid materializing the full N×N
//! attention matrix, reducing memory from O(N²) to O(N·d).

use crate::error::{RuntimeError, RuntimeResult};
use rayon::prelude::*;

/// Configuration for the flash attention kernel.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size along the query (row) dimension.
    pub block_size_q: usize,
    /// Block size along the key/value (column) dimension.
    pub block_size_kv: usize,
    /// Scaling factor applied to QK^T. Defaults to `1 / sqrt(head_dim)`.
    pub scale: Option<f32>,
    /// Whether to apply causal masking (upper-triangular mask).
    pub causal: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            scale: None,
            causal: true,
        }
    }
}

/// Compute the scaling factor: explicit value or `1 / sqrt(head_dim)`.
fn resolve_scale(config: &FlashAttentionConfig, head_dim: usize) -> f32 {
    config
        .scale
        .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt())
}

/// Validate that slice lengths are consistent with the declared dimensions.
fn validate_single_head(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) -> RuntimeResult<()> {
    let expected = seq_len * head_dim;
    if query.len() < expected {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "query length {} too small for seq_len={} head_dim={}",
                query.len(),
                seq_len,
                head_dim
            ),
        });
    }
    if key.len() < expected {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "key length {} too small for seq_len={} head_dim={}",
                key.len(),
                seq_len,
                head_dim
            ),
        });
    }
    if value.len() < expected {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "value length {} too small for seq_len={} head_dim={}",
                value.len(),
                seq_len,
                head_dim
            ),
        });
    }
    if output.len() < expected {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "output length {} too small for seq_len={} head_dim={}",
                output.len(),
                seq_len,
                head_dim
            ),
        });
    }
    if seq_len == 0 || head_dim == 0 {
        return Err(RuntimeError::AttentionError {
            message: "seq_len and head_dim must be > 0".to_string(),
        });
    }
    Ok(())
}

/// Tiled flash-attention for a single head.
///
/// All tensors are row-major with shape `[seq_len, head_dim]` stored as flat
/// slices of length `seq_len * head_dim`.
///
/// # Algorithm
///
/// For each block of query rows (`B_r` rows at a time) the kernel streams
/// over blocks of key/value rows (`B_c` at a time), maintaining a running
/// log-sum-exp accumulator so that the full `N×N` attention matrix is never
/// materialised.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    config: &FlashAttentionConfig,
) -> RuntimeResult<()> {
    validate_single_head(query, key, value, output, seq_len, head_dim)?;

    let scale = resolve_scale(config, head_dim);
    let br = config.block_size_q.min(seq_len);
    let bc = config.block_size_kv.min(seq_len);

    // Iterate over Q tiles.
    let mut i = 0;
    while i < seq_len {
        let tile_rows = br.min(seq_len - i);

        // Per-row running max, running sum, and output accumulator.
        let mut row_max = vec![f32::NEG_INFINITY; tile_rows];
        let mut row_sum = vec![0.0f32; tile_rows];
        let mut out_acc = vec![0.0f32; tile_rows * head_dim];

        // Iterate over KV tiles.
        let mut j = 0;
        while j < seq_len {
            let tile_cols = bc.min(seq_len - j);

            // If causal and entire tile is above diagonal, skip.
            if config.causal && j > i + tile_rows - 1 {
                j += bc;
                continue;
            }

            // Compute S = Q_tile @ K_tile^T * scale  (tile_rows x tile_cols).
            let mut s_tile = vec![0.0f32; tile_rows * tile_cols];
            for ri in 0..tile_rows {
                let q_off = (i + ri) * head_dim;
                for ci in 0..tile_cols {
                    let k_off = (j + ci) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += query[q_off + d] * key[k_off + d];
                    }
                    s_tile[ri * tile_cols + ci] = dot * scale;
                }
            }

            // Causal mask: set S[ri, ci] = -inf where (j + ci) > (i + ri).
            if config.causal {
                for ri in 0..tile_rows {
                    let global_row = i + ri;
                    for ci in 0..tile_cols {
                        let global_col = j + ci;
                        if global_col > global_row {
                            s_tile[ri * tile_cols + ci] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Per-row maximum of current S tile.
            let mut tile_max = vec![f32::NEG_INFINITY; tile_rows];
            for ri in 0..tile_rows {
                for ci in 0..tile_cols {
                    let v = s_tile[ri * tile_cols + ci];
                    if v > tile_max[ri] {
                        tile_max[ri] = v;
                    }
                }
            }

            // new_max = max(running_max, tile_max)
            let mut new_max = vec![0.0f32; tile_rows];
            for ri in 0..tile_rows {
                new_max[ri] = row_max[ri].max(tile_max[ri]);
            }

            // P = exp(S - new_max)  (tile_rows x tile_cols)
            let mut p_tile = vec![0.0f32; tile_rows * tile_cols];
            for ri in 0..tile_rows {
                for ci in 0..tile_cols {
                    let v = s_tile[ri * tile_cols + ci] - new_max[ri];
                    // Clamp to avoid extreme underflow.
                    p_tile[ri * tile_cols + ci] = if v < -88.0 { 0.0 } else { v.exp() };
                }
            }

            // correction = exp(old_max - new_max)
            let mut correction = vec![0.0f32; tile_rows];
            for ri in 0..tile_rows {
                let diff = row_max[ri] - new_max[ri];
                correction[ri] = if diff < -88.0 || diff <= f32::NEG_INFINITY {
                    0.0
                } else {
                    diff.exp()
                };
            }

            // Update running_sum and out_acc.
            for ri in 0..tile_rows {
                row_sum[ri] *= correction[ri];

                // Accumulate P row sum.
                let mut p_row_sum = 0.0f32;
                for ci in 0..tile_cols {
                    p_row_sum += p_tile[ri * tile_cols + ci];
                }
                row_sum[ri] += p_row_sum;

                // Rescale existing output accumulator.
                let out_base = ri * head_dim;
                for d in 0..head_dim {
                    out_acc[out_base + d] *= correction[ri];
                }

                // Accumulate P @ V_tile for this row.
                for ci in 0..tile_cols {
                    let p_val = p_tile[ri * tile_cols + ci];
                    if p_val != 0.0 {
                        let v_off = (j + ci) * head_dim;
                        for d in 0..head_dim {
                            out_acc[out_base + d] += p_val * value[v_off + d];
                        }
                    }
                }
            }

            // Update running_max.
            row_max[..tile_rows].copy_from_slice(&new_max[..tile_rows]);

            j += bc;
        }

        // Normalize: output = out_acc / running_sum.
        for (ri, &sum) in row_sum.iter().enumerate().take(tile_rows) {
            let out_base = ri * head_dim;
            let denom = if sum == 0.0 { 1.0 } else { sum };
            let dst_base = (i + ri) * head_dim;
            for d in 0..head_dim {
                output[dst_base + d] = out_acc[out_base + d] / denom;
            }
        }

        i += br;
    }

    Ok(())
}

/// Multi-head flash attention.
///
/// Tensors are laid out as `[num_heads, seq_len, head_dim]` in row-major order.
/// Each head is processed independently through the tiled kernel.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_multi_head(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    config: &FlashAttentionConfig,
) -> RuntimeResult<()> {
    let head_size = seq_len * head_dim;
    let total = num_heads * head_size;
    if query.len() < total {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "query length {} too small for {} heads × seq_len={} × head_dim={}",
                query.len(),
                num_heads,
                seq_len,
                head_dim
            ),
        });
    }
    if key.len() < total {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "key length {} too small for {} heads × seq_len={} × head_dim={}",
                key.len(),
                num_heads,
                seq_len,
                head_dim
            ),
        });
    }
    if value.len() < total {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "value length {} too small for {} heads × seq_len={} × head_dim={}",
                value.len(),
                num_heads,
                seq_len,
                head_dim
            ),
        });
    }
    if output.len() < total {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "output length {} too small for {} heads × seq_len={} × head_dim={}",
                output.len(),
                num_heads,
                seq_len,
                head_dim
            ),
        });
    }

    for h in 0..num_heads {
        let offset = h * head_size;
        flash_attention(
            &query[offset..offset + head_size],
            &key[offset..offset + head_size],
            &value[offset..offset + head_size],
            &mut output[offset..offset + head_size],
            seq_len,
            head_dim,
            config,
        )?;
    }

    Ok(())
}

/// Grouped-query attention (GQA) with flash attention.
///
/// Multiple query heads share the same key/value head. `num_q_heads` must be
/// an exact multiple of `num_kv_heads`.
///
/// Layout:
/// - `query`:  `[num_q_heads,  seq_len, head_dim]`
/// - `key`:    `[num_kv_heads, seq_len, head_dim]`
/// - `value`:  `[num_kv_heads, seq_len, head_dim]`
/// - `output`: `[num_q_heads,  seq_len, head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_gqa(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    config: &FlashAttentionConfig,
) -> RuntimeResult<()> {
    if num_kv_heads == 0 {
        return Err(RuntimeError::AttentionError {
            message: "num_kv_heads must be > 0".to_string(),
        });
    }
    if num_q_heads % num_kv_heads != 0 {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "num_q_heads ({}) must be divisible by num_kv_heads ({})",
                num_q_heads, num_kv_heads
            ),
        });
    }

    let head_size = seq_len * head_dim;
    let q_total = num_q_heads * head_size;
    let kv_total = num_kv_heads * head_size;

    if query.len() < q_total {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "query length {} too small for {} Q heads × seq_len={} × head_dim={}",
                query.len(),
                num_q_heads,
                seq_len,
                head_dim
            ),
        });
    }
    if key.len() < kv_total {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "key length {} too small for {} KV heads × seq_len={} × head_dim={}",
                key.len(),
                num_kv_heads,
                seq_len,
                head_dim
            ),
        });
    }
    if value.len() < kv_total {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "value length {} too small for {} KV heads × seq_len={} × head_dim={}",
                value.len(),
                num_kv_heads,
                seq_len,
                head_dim
            ),
        });
    }
    if output.len() < q_total {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "output length {} too small for {} Q heads × seq_len={} × head_dim={}",
                output.len(),
                num_q_heads,
                seq_len,
                head_dim
            ),
        });
    }

    let group_size = num_q_heads / num_kv_heads;

    for kv_h in 0..num_kv_heads {
        let kv_offset = kv_h * head_size;
        let k_slice = &key[kv_offset..kv_offset + head_size];
        let v_slice = &value[kv_offset..kv_offset + head_size];

        for g in 0..group_size {
            let q_h = kv_h * group_size + g;
            let q_offset = q_h * head_size;
            flash_attention(
                &query[q_offset..q_offset + head_size],
                k_slice,
                v_slice,
                &mut output[q_offset..q_offset + head_size],
                seq_len,
                head_dim,
                config,
            )?;
        }
    }

    Ok(())
}

/// Block size along the query dimension for the public API kernel.
const BQ: usize = 64;
/// Block size along the key/value dimension for the public API kernel.
const BK: usize = 64;

/// Compute tiled flash attention for multiple heads in parallel.
///
/// This is the primary public-facing entry point for flash attention.  Tensors
/// use **seq-major** layout:
///   - `q`:      `[seq_len_q,  num_heads, head_dim]`
///   - `k`:      `[seq_len_kv, num_heads, head_dim]`
///   - `v`:      `[seq_len_kv, num_heads, head_dim]`
///   - returns:  `[seq_len_q,  num_heads, head_dim]`
///
/// Internally the tensors are transposed to **head-major** layout so that each
/// head's data is contiguous; per-head processing then runs in parallel via
/// `rayon`.  The tile loop (Q-tile × K-tile) is sequential within each head
/// because the online softmax state carries cross-tile state.
///
/// # Arguments
///
/// * `q`, `k`, `v` – flat slices matching the seq-major shapes above.
/// * `num_heads`    – number of attention heads (`H`).
/// * `head_dim`     – dimension per head (`D`); must divide the hidden dim.
/// * `softmax_scale`– multiplier applied to `Q @ K^T` before softmax.
/// * `causal_mask`  – if `true`, tokens cannot attend to future positions.
///
/// # Errors
///
/// Returns [`RuntimeError::AttentionError`] if the slice lengths are
/// inconsistent with the declared dimensions.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
    causal_mask: bool,
) -> RuntimeResult<Vec<f32>> {
    if num_heads == 0 || head_dim == 0 {
        return Err(RuntimeError::AttentionError {
            message: "num_heads and head_dim must be > 0".to_string(),
        });
    }

    let q_total = q.len();
    let kv_total = k.len();

    if q_total % (num_heads * head_dim) != 0 {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "q length {} is not divisible by num_heads * head_dim = {}",
                q_total,
                num_heads * head_dim
            ),
        });
    }
    if kv_total % (num_heads * head_dim) != 0 {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "k length {} is not divisible by num_heads * head_dim = {}",
                kv_total,
                num_heads * head_dim
            ),
        });
    }
    if k.len() != v.len() {
        return Err(RuntimeError::AttentionError {
            message: format!("k length {} and v length {} must match", k.len(), v.len()),
        });
    }

    let seq_len_q = q_total / (num_heads * head_dim);
    let seq_len_kv = kv_total / (num_heads * head_dim);

    if seq_len_q == 0 || seq_len_kv == 0 {
        return Err(RuntimeError::AttentionError {
            message: "seq_len_q and seq_len_kv must be > 0".to_string(),
        });
    }

    // Transpose Q from [seq_len_q, num_heads, head_dim]
    //              to  [num_heads, seq_len_q, head_dim]  (head-major)
    let mut q_head = vec![0.0f32; num_heads * seq_len_q * head_dim];
    for s in 0..seq_len_q {
        for h in 0..num_heads {
            let src = s * num_heads * head_dim + h * head_dim;
            let dst = h * seq_len_q * head_dim + s * head_dim;
            q_head[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
        }
    }

    // Transpose K from [seq_len_kv, num_heads, head_dim]
    //              to  [num_heads, seq_len_kv, head_dim]
    let mut k_head = vec![0.0f32; num_heads * seq_len_kv * head_dim];
    for s in 0..seq_len_kv {
        for h in 0..num_heads {
            let src = s * num_heads * head_dim + h * head_dim;
            let dst = h * seq_len_kv * head_dim + s * head_dim;
            k_head[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
        }
    }

    // Transpose V from [seq_len_kv, num_heads, head_dim]
    //              to  [num_heads, seq_len_kv, head_dim]
    let mut v_head = vec![0.0f32; num_heads * seq_len_kv * head_dim];
    for s in 0..seq_len_kv {
        for h in 0..num_heads {
            let src = s * num_heads * head_dim + h * head_dim;
            let dst = h * seq_len_kv * head_dim + s * head_dim;
            v_head[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
        }
    }

    // Output buffer in head-major: [num_heads, seq_len_q, head_dim]
    let mut out_head = vec![0.0f32; num_heads * seq_len_q * head_dim];

    let q_stride = seq_len_q * head_dim;
    let kv_stride = seq_len_kv * head_dim;

    // Process each head in parallel.
    out_head
        .par_chunks_mut(q_stride)
        .enumerate()
        .try_for_each(|(h, out_slice)| {
            let q_off = h * q_stride;
            let kv_off = h * kv_stride;
            flash_attention_forward_single_head(
                &q_head[q_off..q_off + q_stride],
                &k_head[kv_off..kv_off + kv_stride],
                &v_head[kv_off..kv_off + kv_stride],
                out_slice,
                seq_len_q,
                seq_len_kv,
                head_dim,
                softmax_scale,
                causal_mask,
            )
        })?;

    // Transpose output from [num_heads, seq_len_q, head_dim]
    //                    to [seq_len_q, num_heads, head_dim]
    let mut output = vec![0.0f32; num_heads * seq_len_q * head_dim];
    for h in 0..num_heads {
        for s in 0..seq_len_q {
            let src = h * seq_len_q * head_dim + s * head_dim;
            let dst = s * num_heads * head_dim + h * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&out_head[src..src + head_dim]);
        }
    }

    Ok(output)
}

/// Single-head flash attention with asymmetric seq_len_q / seq_len_kv.
///
/// All inputs and output are in **head-major** layout (i.e. contiguous for
/// the single head being processed).
///
/// Q shape: `[seq_len_q,  head_dim]`
/// K shape: `[seq_len_kv, head_dim]`
/// V shape: `[seq_len_kv, head_dim]`
/// O shape: `[seq_len_q,  head_dim]`
///
/// For causal masking with `seq_len_q < seq_len_kv` (decode step with a
/// growing KV cache), the query position `a` is mapped to global position
/// `seq_len_kv - seq_len_q + a` so that the newest query attends only to
/// key positions `<= seq_len_kv - seq_len_q + a`.
#[allow(clippy::too_many_arguments)]
fn flash_attention_forward_single_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len_q: usize,
    seq_len_kv: usize,
    head_dim: usize,
    softmax_scale: f32,
    causal_mask: bool,
) -> RuntimeResult<()> {
    let br = BQ.min(seq_len_q);
    let bc = BK.min(seq_len_kv);

    // Offset so that q-row `a` is at global position `q_offset + a` in the
    // key sequence.  For prefill seq_len_q == seq_len_kv and q_offset = 0.
    // For decode seq_len_q < seq_len_kv and q_offset = seq_len_kv - seq_len_q.
    let q_offset = seq_len_kv.saturating_sub(seq_len_q);

    let mut qi = 0;
    while qi < seq_len_q {
        let tile_rows = br.min(seq_len_q - qi);

        // Running statistics for online softmax.
        let mut row_max = vec![f32::NEG_INFINITY; tile_rows];
        let mut row_sum = vec![0.0f32; tile_rows];
        let mut out_acc = vec![0.0f32; tile_rows * head_dim];

        let mut kj = 0;
        while kj < seq_len_kv {
            let tile_cols = bc.min(seq_len_kv - kj);

            // Skip entire tile if it would be fully masked.
            if causal_mask {
                // The last query row in this tile attends from global position
                // `q_offset + qi + tile_rows - 1`.  The first key column is at
                // global position `kj`.  If kj > q_offset + qi + tile_rows - 1
                // every cell is masked → skip.
                let last_q_global = q_offset + qi + tile_rows - 1;
                if kj > last_q_global {
                    kj += bc;
                    continue;
                }
            }

            // S = Q_tile @ K_tile^T * softmax_scale  (tile_rows × tile_cols)
            let mut s_tile = vec![0.0f32; tile_rows * tile_cols];
            for ri in 0..tile_rows {
                let q_row_off = (qi + ri) * head_dim;
                for ci in 0..tile_cols {
                    let k_row_off = (kj + ci) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_row_off + d] * k[k_row_off + d];
                    }
                    s_tile[ri * tile_cols + ci] = dot * softmax_scale;
                }
            }

            // Causal mask using absolute positions.
            if causal_mask {
                for ri in 0..tile_rows {
                    let q_global = q_offset + qi + ri;
                    for ci in 0..tile_cols {
                        let k_global = kj + ci;
                        if k_global > q_global {
                            s_tile[ri * tile_cols + ci] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Per-row max of the current tile.
            let mut tile_max = vec![f32::NEG_INFINITY; tile_rows];
            for ri in 0..tile_rows {
                for ci in 0..tile_cols {
                    let s = s_tile[ri * tile_cols + ci];
                    if s > tile_max[ri] {
                        tile_max[ri] = s;
                    }
                }
            }

            // new_max = max(running_max, tile_max)
            let mut new_max = vec![0.0f32; tile_rows];
            for ri in 0..tile_rows {
                new_max[ri] = row_max[ri].max(tile_max[ri]);
            }

            // correction = exp(old_max - new_max)
            let mut correction = vec![0.0f32; tile_rows];
            for ri in 0..tile_rows {
                let diff = row_max[ri] - new_max[ri];
                correction[ri] = if diff <= f32::NEG_INFINITY || diff < -88.0 {
                    0.0
                } else {
                    diff.exp()
                };
            }

            // P = exp(S - new_max)
            let mut p_tile = vec![0.0f32; tile_rows * tile_cols];
            for ri in 0..tile_rows {
                for ci in 0..tile_cols {
                    let v = s_tile[ri * tile_cols + ci] - new_max[ri];
                    p_tile[ri * tile_cols + ci] = if v < -88.0 { 0.0 } else { v.exp() };
                }
            }

            // Update row_sum and out_acc.
            for ri in 0..tile_rows {
                // Rescale existing running sum.
                row_sum[ri] *= correction[ri];

                // Add row sum of P.
                let mut p_row_sum = 0.0f32;
                for ci in 0..tile_cols {
                    p_row_sum += p_tile[ri * tile_cols + ci];
                }
                row_sum[ri] += p_row_sum;

                // Rescale existing output accumulator.
                let out_base = ri * head_dim;
                for d in 0..head_dim {
                    out_acc[out_base + d] *= correction[ri];
                }

                // Accumulate P @ V_tile.
                for ci in 0..tile_cols {
                    let p_val = p_tile[ri * tile_cols + ci];
                    if p_val != 0.0 {
                        let v_off = (kj + ci) * head_dim;
                        for d in 0..head_dim {
                            out_acc[out_base + d] += p_val * v[v_off + d];
                        }
                    }
                }
            }

            // Update running max.
            row_max[..tile_rows].copy_from_slice(&new_max[..tile_rows]);

            kj += bc;
        }

        // Normalize.
        for (ri, &sum) in row_sum.iter().enumerate().take(tile_rows) {
            let out_base = ri * head_dim;
            let denom = if sum == 0.0 { 1.0 } else { sum };
            let dst_base = (qi + ri) * head_dim;
            for d in 0..head_dim {
                output[dst_base + d] = out_acc[out_base + d] / denom;
            }
        }

        qi += br;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive single-head attention for reference:
    ///   S = Q @ K^T * scale
    ///   if causal: mask upper triangle to -inf
    ///   P = softmax(S, dim=-1)
    ///   O = P @ V
    fn naive_attention(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let n = seq_len;
        let d = head_dim;

        // S = Q @ K^T * scale  (n x n)
        let mut s = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += query[i * d + k] * key[j * d + k];
                }
                s[i * n + j] = dot * scale;
            }
        }

        // Causal mask.
        if causal {
            for i in 0..n {
                for j in (i + 1)..n {
                    s[i * n + j] = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax per row.
        let mut p = vec![0.0f32; n * n];
        for i in 0..n {
            let row = &s[i * n..(i + 1) * n];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for j in 0..n {
                let e = (row[j] - max_val).exp();
                p[i * n + j] = e;
                sum_exp += e;
            }
            if sum_exp > 0.0 {
                for j in 0..n {
                    p[i * n + j] /= sum_exp;
                }
            }
        }

        // O = P @ V  (n x d)
        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            for k in 0..d {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += p[i * n + j] * value[j * d + k];
                }
                out[i * d + k] = acc;
            }
        }

        out
    }

    /// Generate deterministic pseudo-random data using a simple LCG.
    fn pseudo_random_data(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // LCG: state = (a * state + c) mod m
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                // Map to [-1, 1].
                let bits = (state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (idx, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(
                diff <= tol,
                "{label}: mismatch at index {idx}: flash={x} naive={y} diff={diff} tol={tol}"
            );
        }
    }

    #[test]
    fn test_flash_attention_single_head() {
        let seq_len = 8;
        let head_dim = 4;
        let n = seq_len * head_dim;

        let q = pseudo_random_data(n, 42);
        let k = pseudo_random_data(n, 123);
        let v = pseudo_random_data(n, 456);

        let config = FlashAttentionConfig {
            block_size_q: 4,
            block_size_kv: 4,
            scale: None,
            causal: true,
        };
        let scale = resolve_scale(&config, head_dim);

        let expected = naive_attention(&q, &k, &v, seq_len, head_dim, scale, true);

        let mut output = vec![0.0f32; n];
        flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, &config)
            .expect("flash_attention failed");

        assert_close(&output, &expected, 1e-4, "single_head");
    }

    #[test]
    fn test_flash_attention_causal_mask() {
        let seq_len = 8;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = pseudo_random_data(n, 10);
        let k = pseudo_random_data(n, 20);
        let v = pseudo_random_data(n, 30);

        // Causal.
        let config_causal = FlashAttentionConfig {
            block_size_q: 4,
            block_size_kv: 4,
            scale: Some(scale),
            causal: true,
        };
        let mut out_causal = vec![0.0f32; n];
        flash_attention(
            &q,
            &k,
            &v,
            &mut out_causal,
            seq_len,
            head_dim,
            &config_causal,
        )
        .expect("causal attention failed");

        // Non-causal.
        let config_full = FlashAttentionConfig {
            block_size_q: 4,
            block_size_kv: 4,
            scale: Some(scale),
            causal: false,
        };
        let mut out_full = vec![0.0f32; n];
        flash_attention(&q, &k, &v, &mut out_full, seq_len, head_dim, &config_full)
            .expect("full attention failed");

        // Row 0 with causal: attends only to col 0, differs from full.
        // The last row attends to all positions in both cases, so should match.
        let last_row_causal = &out_causal[(seq_len - 1) * head_dim..seq_len * head_dim];
        let last_row_full = &out_full[(seq_len - 1) * head_dim..seq_len * head_dim];
        assert_close(last_row_causal, last_row_full, 1e-4, "last_row");

        // Middle rows should differ (causal vs full).
        let mid = seq_len / 2;
        let mid_causal = &out_causal[mid * head_dim..(mid + 1) * head_dim];
        let mid_full = &out_full[mid * head_dim..(mid + 1) * head_dim];
        let has_diff = mid_causal
            .iter()
            .zip(mid_full.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(has_diff, "middle row should differ between causal and full");

        // Verify against naive references.
        let naive_causal = naive_attention(&q, &k, &v, seq_len, head_dim, scale, true);
        let naive_full = naive_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert_close(&out_causal, &naive_causal, 1e-4, "causal_vs_naive");
        assert_close(&out_full, &naive_full, 1e-4, "full_vs_naive");
    }

    #[test]
    fn test_flash_attention_multi_head() {
        let num_heads = 4;
        let seq_len = 16;
        let head_dim = 8;
        let head_size = seq_len * head_dim;
        let total = num_heads * head_size;

        let q = pseudo_random_data(total, 100);
        let k = pseudo_random_data(total, 200);
        let v = pseudo_random_data(total, 300);

        let config = FlashAttentionConfig {
            block_size_q: 8,
            block_size_kv: 8,
            scale: None,
            causal: true,
        };
        let scale = resolve_scale(&config, head_dim);

        let mut output = vec![0.0f32; total];
        flash_attention_multi_head(
            &q,
            &k,
            &v,
            &mut output,
            num_heads,
            seq_len,
            head_dim,
            &config,
        )
        .expect("multi_head attention failed");

        // Compare each head independently.
        for h in 0..num_heads {
            let off = h * head_size;
            let expected = naive_attention(
                &q[off..off + head_size],
                &k[off..off + head_size],
                &v[off..off + head_size],
                seq_len,
                head_dim,
                scale,
                true,
            );
            assert_close(
                &output[off..off + head_size],
                &expected,
                1e-4,
                &format!("head_{h}"),
            );
        }
    }

    #[test]
    fn test_flash_attention_gqa() {
        let num_q_heads = 8;
        let num_kv_heads = 2;
        let seq_len = 16;
        let head_dim = 8;
        let head_size = seq_len * head_dim;
        let group_size = num_q_heads / num_kv_heads;

        let q = pseudo_random_data(num_q_heads * head_size, 500);
        let k = pseudo_random_data(num_kv_heads * head_size, 600);
        let v = pseudo_random_data(num_kv_heads * head_size, 700);

        let config = FlashAttentionConfig {
            block_size_q: 8,
            block_size_kv: 8,
            scale: None,
            causal: true,
        };
        let scale = resolve_scale(&config, head_dim);

        let mut output = vec![0.0f32; num_q_heads * head_size];
        flash_attention_gqa(
            &q,
            &k,
            &v,
            &mut output,
            num_q_heads,
            num_kv_heads,
            seq_len,
            head_dim,
            &config,
        )
        .expect("gqa attention failed");

        // Each Q head in a group shares the same KV head.
        for kv_h in 0..num_kv_heads {
            let kv_off = kv_h * head_size;
            for g in 0..group_size {
                let q_h = kv_h * group_size + g;
                let q_off = q_h * head_size;
                let expected = naive_attention(
                    &q[q_off..q_off + head_size],
                    &k[kv_off..kv_off + head_size],
                    &v[kv_off..kv_off + head_size],
                    seq_len,
                    head_dim,
                    scale,
                    true,
                );
                assert_close(
                    &output[q_off..q_off + head_size],
                    &expected,
                    1e-4,
                    &format!("gqa_kv{kv_h}_g{g}"),
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_numerical_stability() {
        let seq_len = 16;
        let head_dim = 8;
        let n = seq_len * head_dim;

        // Large values that would cause naive exp() to overflow without
        // the log-sum-exp trick.
        let q: Vec<f32> = pseudo_random_data(n, 999)
            .iter()
            .map(|x| x * 50.0)
            .collect();
        let k: Vec<f32> = pseudo_random_data(n, 888)
            .iter()
            .map(|x| x * 50.0)
            .collect();
        let v = pseudo_random_data(n, 777);

        let config = FlashAttentionConfig {
            block_size_q: 4,
            block_size_kv: 4,
            scale: None,
            causal: true,
        };
        let scale = resolve_scale(&config, head_dim);

        let mut output = vec![0.0f32; n];
        flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, &config)
            .expect("numerically-stable attention failed");

        // Verify no NaN or Inf in output.
        for (idx, val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "output[{idx}] = {val} is not finite (NaN or Inf)"
            );
        }

        // Compare against naive (which also uses max-subtraction softmax).
        let expected = naive_attention(&q, &k, &v, seq_len, head_dim, scale, true);
        assert_close(&output, &expected, 1e-3, "numerical_stability");
    }

    #[test]
    fn test_flash_attention_various_block_sizes() {
        let seq_len = 32;
        let head_dim = 8;
        let n = seq_len * head_dim;

        let q = pseudo_random_data(n, 1111);
        let k = pseudo_random_data(n, 2222);
        let v = pseudo_random_data(n, 3333);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let expected = naive_attention(&q, &k, &v, seq_len, head_dim, scale, true);

        for &bs in &[4usize, 8, 16, 32] {
            let config = FlashAttentionConfig {
                block_size_q: bs,
                block_size_kv: bs,
                scale: Some(scale),
                causal: true,
            };
            let mut output = vec![0.0f32; n];
            flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, &config)
                .unwrap_or_else(|e| panic!("block_size={bs} failed: {e}"));
            assert_close(&output, &expected, 1e-4, &format!("block_size_{bs}"));
        }
    }

    // ── Tests for flash_attention_forward (seq-major layout, asymmetric KV) ──

    /// Reference attention for `flash_attention_forward` tests.
    ///
    /// Layout: `[seq_len_q, num_heads, head_dim]` (seq-major).
    /// Supports `seq_len_q != seq_len_kv`; causal mask uses absolute positions
    /// `q_global = q_offset + qi` where `q_offset = seq_len_kv - seq_len_q`.
    fn naive_attention_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let seq_len_q = q.len() / (num_heads * head_dim);
        let seq_len_kv = k.len() / (num_heads * head_dim);
        let q_offset = seq_len_kv.saturating_sub(seq_len_q);
        let mut output = vec![0.0f32; seq_len_q * num_heads * head_dim];

        for h in 0..num_heads {
            // S[qi, kj] = dot(q[qi,h,:], k[kj,h,:]) * scale
            for qi in 0..seq_len_q {
                let q_base = qi * num_heads * head_dim + h * head_dim;

                // Compute scores for each key position.
                let mut scores: Vec<f32> = (0..seq_len_kv)
                    .map(|kj| {
                        let k_base = kj * num_heads * head_dim + h * head_dim;
                        let dot: f32 = (0..head_dim).map(|d| q[q_base + d] * k[k_base + d]).sum();
                        dot * scale
                    })
                    .collect();

                // Causal mask: positions beyond q_global attend to -inf.
                if causal {
                    let q_global = q_offset + qi;
                    for (kj, s) in scores.iter_mut().enumerate() {
                        if kj > q_global {
                            *s = f32::NEG_INFINITY;
                        }
                    }
                }

                // Softmax.
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                if sum_exp > 0.0 {
                    for e in &mut exp_scores {
                        *e /= sum_exp;
                    }
                }

                // O[qi, h, :] = sum_kj exp_scores[kj] * v[kj, h, :]
                let o_base = qi * num_heads * head_dim + h * head_dim;
                for (kj, &w) in exp_scores.iter().enumerate() {
                    let v_base = kj * num_heads * head_dim + h * head_dim;
                    for d in 0..head_dim {
                        output[o_base + d] += w * v[v_base + d];
                    }
                }
            }
        }

        output
    }

    /// Deterministic data with `f32::sin(i as f32 * 0.1) * 0.1` pattern.
    fn sin_data(len: usize) -> Vec<f32> {
        (0..len).map(|i| f32::sin(i as f32 * 0.1) * 0.1).collect()
    }

    /// (a) Causal short: 8 heads, 16 head_dim, seq 32×32, tolerance 1e-5.
    #[test]
    fn flash_matches_reference_causal_short() {
        let num_heads = 8;
        let head_dim = 16;
        let seq_len_q = 32;
        let seq_len_kv = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = sin_data(seq_len_q * num_heads * head_dim);
        let k = sin_data(seq_len_kv * num_heads * head_dim);
        let v = sin_data(seq_len_kv * num_heads * head_dim);

        let expected = naive_attention_reference(&q, &k, &v, num_heads, head_dim, scale, true);
        let got = flash_attention_forward(&q, &k, &v, num_heads, head_dim, scale, true)
            .expect("flash_attention_forward failed");

        assert_close(&got, &expected, 1e-5, "causal_short");
    }

    /// (b) Causal long: 8 heads, 32 head_dim, seq 512×1024, tolerance 1e-4.
    #[test]
    fn flash_matches_reference_causal_long() {
        let num_heads = 8;
        let head_dim = 32;
        let seq_len_q = 512;
        let seq_len_kv = 1024;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = sin_data(seq_len_q * num_heads * head_dim);
        let k = sin_data(seq_len_kv * num_heads * head_dim);
        let v = sin_data(seq_len_kv * num_heads * head_dim);

        let expected = naive_attention_reference(&q, &k, &v, num_heads, head_dim, scale, true);
        let got = flash_attention_forward(&q, &k, &v, num_heads, head_dim, scale, true)
            .expect("flash_attention_forward failed");

        assert_close(&got, &expected, 1e-4, "causal_long");
    }

    /// (c) Non-causal: 4 heads, 16 head_dim, seq 64×64, tolerance 1e-5.
    #[test]
    fn flash_matches_reference_non_causal() {
        let num_heads = 4;
        let head_dim = 16;
        let seq_len_q = 64;
        let seq_len_kv = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = sin_data(seq_len_q * num_heads * head_dim);
        let k = sin_data(seq_len_kv * num_heads * head_dim);
        let v = sin_data(seq_len_kv * num_heads * head_dim);

        let expected = naive_attention_reference(&q, &k, &v, num_heads, head_dim, scale, false);
        let got = flash_attention_forward(&q, &k, &v, num_heads, head_dim, scale, false)
            .expect("flash_attention_forward failed");

        assert_close(&got, &expected, 1e-5, "non_causal");
    }

    /// (d) Determinism: same inputs → bit-equal outputs on two calls.
    #[test]
    fn flash_determinism() {
        let num_heads = 4;
        let head_dim = 16;
        let seq_len = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = sin_data(seq_len * num_heads * head_dim);
        let k = sin_data(seq_len * num_heads * head_dim);
        let v = sin_data(seq_len * num_heads * head_dim);

        let out1 = flash_attention_forward(&q, &k, &v, num_heads, head_dim, scale, true)
            .expect("first call failed");
        let out2 = flash_attention_forward(&q, &k, &v, num_heads, head_dim, scale, true)
            .expect("second call failed");

        // Bit-equal: use assert_eq! on the raw bits via integer conversion.
        assert_eq!(out1.len(), out2.len(), "output length mismatch");
        for (idx, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit mismatch at index {idx}: {a} vs {b}"
            );
        }
    }

    /// (e) Single-token decode: seq_len_q=1, seq_len_kv=1024, causal=false, tol 1e-5.
    #[test]
    fn flash_single_token_decode() {
        let num_heads = 4;
        let head_dim = 16;
        let seq_len_q = 1;
        let seq_len_kv = 1024;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = sin_data(seq_len_q * num_heads * head_dim);
        let k = sin_data(seq_len_kv * num_heads * head_dim);
        let v = sin_data(seq_len_kv * num_heads * head_dim);

        let expected = naive_attention_reference(&q, &k, &v, num_heads, head_dim, scale, false);
        let got = flash_attention_forward(&q, &k, &v, num_heads, head_dim, scale, false)
            .expect("flash_attention_forward failed");

        assert_close(&got, &expected, 1e-5, "single_token_decode");
    }
}

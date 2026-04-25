//! Batched decode-phase attention for continuous batching.
//!
//! During the decode step every in-flight request produces exactly one query
//! vector (the current token embedding) but reads from a potentially long KV
//! cache that was built up during prefill and earlier decode steps.
//!
//! `batched_flash_attention` iterates over the slots in `kv_view`, calls the
//! single-token flash-attention kernel for each slot independently, and
//! concatenates the per-slot results into a single output tensor.
//!
//! The kernel re-uses `flash_attention_forward` from the flash-attention
//! module with `seq_len_q = 1` per slot, which keeps memory at O(BK × D)
//! per head per slot — independent of KV-cache depth.
//!
//! # Layout conventions
//!
//! * `q_batch`: `[batch_size, num_heads, head_dim]` — **head-major** within
//!   each batch element (same layout as [`flash_attention_forward`]'s
//!   seq-major format with `seq_len_q = 1`).
//! * Output: `[batch_size, num_heads, head_dim]`.
//!
//! # Integration with ForwardPass
//!
//! `ForwardPass::forward_batched` in `oxillama-arch` delegates to this
//! kernel for LLaMA and other architectures that support continuous batching.
//! Non-overriding architectures return `ArchError::NotSupported`.

use crate::error::{RuntimeError, RuntimeResult};
use crate::flash_attention::flash_attention_forward;
use crate::kv_cache::BatchedKvView;

/// Run batched decode-phase flash attention over `batch_size` request slots.
///
/// Each slot has `seq_len_q = 1` (one new query token) and reads from a
/// per-slot KV cache of variable depth via `kv_view`.
///
/// # Arguments
///
/// * `q_batch`     – Query tensor, layout `[batch_size, num_heads, head_dim]`.
/// * `kv_view`     – Provides per-slot keys and values via [`BatchedKvView`].
/// * `num_heads`   – Number of attention heads.
/// * `head_dim`    – Dimension of each head.
/// * `softmax_scale` – Pre-computed `1 / sqrt(head_dim)` (or override).
///
/// # Returns
///
/// Output tensor with layout `[batch_size, num_heads, head_dim]`.
///
/// # Errors
///
/// Returns [`RuntimeError::AttentionError`] if:
/// - slice lengths are inconsistent with declared dimensions,
/// - `kv_view.slot_count()` does not equal `batch_size`, or
/// - a per-slot flash-attention call fails.
pub fn batched_flash_attention<V: BatchedKvView>(
    q_batch: &[f32],
    kv_view: &V,
    num_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
) -> RuntimeResult<Vec<f32>> {
    if num_heads == 0 || head_dim == 0 {
        return Err(RuntimeError::AttentionError {
            message: "num_heads and head_dim must be > 0".to_string(),
        });
    }

    let head_stride = num_heads * head_dim;
    if head_stride == 0 || q_batch.len() % head_stride != 0 {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "q_batch length {} is not a multiple of num_heads * head_dim = {}",
                q_batch.len(),
                head_stride
            ),
        });
    }

    let batch_size = q_batch.len() / head_stride;
    let slot_count = kv_view.slot_count();

    if batch_size != slot_count {
        return Err(RuntimeError::AttentionError {
            message: format!(
                "batch_size ({batch_size}) must equal kv_view.slot_count() ({slot_count})"
            ),
        });
    }

    let mut output = vec![0.0f32; batch_size * head_stride];

    for slot in 0..batch_size {
        let seq_len_kv = kv_view.position(slot);
        if seq_len_kv == 0 {
            // No KV context yet — output is all-zero for this slot (undefined
            // but safe; callers should ensure seq_len_kv >= 1 in practice).
            continue;
        }

        let (k_flat, v_flat) = kv_view.kv_for_slot(slot);

        // Validate KV slice shapes.
        // k_flat and v_flat are [seq_len_kv, num_heads, head_dim].
        let kv_expected = seq_len_kv * head_stride;
        if k_flat.len() < kv_expected {
            return Err(RuntimeError::AttentionError {
                message: format!(
                    "slot {slot}: k_flat length {} < expected {} (seq_len_kv={seq_len_kv}, \
                     num_heads={num_heads}, head_dim={head_dim})",
                    k_flat.len(),
                    kv_expected,
                ),
            });
        }
        if v_flat.len() < kv_expected {
            return Err(RuntimeError::AttentionError {
                message: format!(
                    "slot {slot}: v_flat length {} < expected {} (seq_len_kv={seq_len_kv}, \
                     num_heads={num_heads}, head_dim={head_dim})",
                    v_flat.len(),
                    kv_expected,
                ),
            });
        }

        // Query for this slot: shape [1, num_heads, head_dim] (seq_len_q = 1).
        let q_off = slot * head_stride;
        let q_slot = &q_batch[q_off..q_off + head_stride];

        // flash_attention_forward takes q: [seq_len_q, H, D], causal=false
        // since in decode mode the single query attends to all cached KV.
        let slot_out = flash_attention_forward(
            q_slot,
            &k_flat[..kv_expected],
            &v_flat[..kv_expected],
            num_heads,
            head_dim,
            softmax_scale,
            false, // decode: attend to all past tokens, no causal restriction needed
        )?;

        let o_off = slot * head_stride;
        output[o_off..o_off + head_stride].copy_from_slice(&slot_out);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{BatchedKvView, KvSlot, VecBatchedKvView};

    /// Deterministic test data.
    fn sin_data(len: usize) -> Vec<f32> {
        (0..len).map(|i| f32::sin(i as f32 * 0.1) * 0.1).collect()
    }

    /// (a) Basic `BatchedKvView` correctness.
    ///
    /// Construct a `VecBatchedKvView` with 3 slots and verify that the trait
    /// methods return the expected values.
    #[test]
    fn batched_kv_view_basic() {
        let num_heads = 2usize;
        let head_dim = 4usize;
        let kv_dim = num_heads * head_dim;

        // Slot 0: position 3, slot 1: position 7, slot 2: position 1.
        let positions = [3usize, 7, 1];
        let mut slots = Vec::new();
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for (i, &pos) in positions.iter().enumerate() {
            slots.push(KvSlot::new(i as u64 + 1, i, pos));
            // key = [slot_idx * 10 + token_pos] for each element.
            let k: Vec<f32> = (0..pos * kv_dim)
                .map(|j| (i as f32 * 10.0) + j as f32)
                .collect();
            let v: Vec<f32> = (0..pos * kv_dim)
                .map(|j| (i as f32 * 100.0) + j as f32)
                .collect();
            keys.push(k);
            values.push(v);
        }

        let view = VecBatchedKvView::new(slots, keys, values);

        // slot_count
        assert_eq!(view.slot_count(), 3, "slot_count must be 3");

        // position
        assert_eq!(view.position(0), 3, "slot 0 position must be 3");
        assert_eq!(view.position(1), 7, "slot 1 position must be 7");
        assert_eq!(view.position(2), 1, "slot 2 position must be 1");

        // kv_for_slot lengths
        let (k0, v0) = view.kv_for_slot(0);
        assert_eq!(k0.len(), 3 * kv_dim, "slot 0 key length must be pos*kv_dim");
        assert_eq!(
            v0.len(),
            3 * kv_dim,
            "slot 0 value length must be pos*kv_dim"
        );

        let (k1, v1) = view.kv_for_slot(1);
        assert_eq!(k1.len(), 7 * kv_dim, "slot 1 key length");
        assert_eq!(v1.len(), 7 * kv_dim, "slot 1 value length");

        let (k2, v2) = view.kv_for_slot(2);
        assert_eq!(k2.len(), kv_dim, "slot 2 key length");
        assert_eq!(v2.len(), kv_dim, "slot 2 value length");

        // Spot-check values.
        // Slot 0: k[0] = 0*10 + 0 = 0.0, k[1] = 1.0
        assert!(
            (k0[0] - 0.0f32).abs() < 1e-7,
            "slot 0 k[0] should be 0.0, got {}",
            k0[0]
        );
        assert!(
            (k0[1] - 1.0f32).abs() < 1e-7,
            "slot 0 k[1] should be 1.0, got {}",
            k0[1]
        );
        // Slot 0: v[0] = 0*100 + 0 = 0.0, v[1] = 1.0
        assert!(
            (v0[0] - 0.0f32).abs() < 1e-7,
            "slot 0 v[0] should be 0.0, got {}",
            v0[0]
        );

        // Slot 1: k[0] = 1*10 + 0 = 10.0
        assert!(
            (k1[0] - 10.0f32).abs() < 1e-7,
            "slot 1 k[0] should be 10.0, got {}",
            k1[0]
        );

        // Slot 2: v[0] = 2*100 + 0 = 200.0
        assert!(
            (v2[0] - 200.0f32).abs() < 1e-7,
            "slot 2 v[0] should be 200.0, got {}",
            v2[0]
        );
    }

    /// (b) `batched_flash_decode_matches_serial`.
    ///
    /// Two slots with different KV caches.  Compute batched output and verify
    /// each row matches a single-slot call to `flash_attention_forward` with
    /// the same inputs.
    #[test]
    fn batched_flash_decode_matches_serial() {
        let num_heads = 2usize;
        let head_dim = 8usize;
        let kv_dim = num_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Slot 0: 16-token KV cache.
        let seq_kv_0 = 16usize;
        // Slot 1: 24-token KV cache.
        let seq_kv_1 = 24usize;

        // Query for each slot: [1, num_heads, head_dim].
        let q0 = sin_data(kv_dim);
        let q1: Vec<f32> = (0..kv_dim)
            .map(|i| f32::cos(i as f32 * 0.07) * 0.12)
            .collect();

        // KV caches in [seq_kv, num_heads, head_dim] layout.
        let k0 = sin_data(seq_kv_0 * kv_dim);
        let v0: Vec<f32> = (0..seq_kv_0 * kv_dim)
            .map(|i| f32::cos(i as f32 * 0.13) * 0.1)
            .collect();
        let k1: Vec<f32> = (0..seq_kv_1 * kv_dim)
            .map(|i| f32::sin(i as f32 * 0.05 + 1.0) * 0.08)
            .collect();
        let v1: Vec<f32> = (0..seq_kv_1 * kv_dim)
            .map(|i| f32::cos(i as f32 * 0.09 + 0.5) * 0.09)
            .collect();

        // Build batched view.
        let slots = vec![KvSlot::new(1, 0, seq_kv_0), KvSlot::new(2, 1, seq_kv_1)];
        let keys_vec = vec![k0.clone(), k1.clone()];
        let vals_vec = vec![v0.clone(), v1.clone()];
        let view = VecBatchedKvView::new(slots, keys_vec, vals_vec);

        // Batched query: stack q0 and q1.
        let mut q_batch = Vec::with_capacity(2 * kv_dim);
        q_batch.extend_from_slice(&q0);
        q_batch.extend_from_slice(&q1);

        let batched_out = batched_flash_attention(&q_batch, &view, num_heads, head_dim, scale)
            .expect("batched_flash_attention failed");

        assert_eq!(
            batched_out.len(),
            2 * kv_dim,
            "output must be batch_size * num_heads * head_dim"
        );

        // Serial reference for slot 0.
        let serial_out_0 =
            flash_attention_forward(&q0, &k0, &v0, num_heads, head_dim, scale, false)
                .expect("serial slot 0 failed");

        // Serial reference for slot 1.
        let serial_out_1 =
            flash_attention_forward(&q1, &k1, &v1, num_heads, head_dim, scale, false)
                .expect("serial slot 1 failed");

        // Compare batched[slot 0] vs serial slot 0.
        for (idx, (&b, &s)) in batched_out[..kv_dim]
            .iter()
            .zip(serial_out_0.iter())
            .enumerate()
        {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-5,
                "slot 0, index {idx}: batched={b} serial={s} diff={diff}"
            );
        }

        // Compare batched[slot 1] vs serial slot 1.
        for (idx, (&b, &s)) in batched_out[kv_dim..]
            .iter()
            .zip(serial_out_1.iter())
            .enumerate()
        {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-5,
                "slot 1, index {idx}: batched={b} serial={s} diff={diff}"
            );
        }
    }
}

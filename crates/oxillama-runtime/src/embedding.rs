//! Embedding pooling modes and the `pool_hidden_states` kernel.
//!
//! An embedding model produces a per-token hidden state matrix of shape
//! `[seq_len, hidden_size]` (stored row-major as a flat `Vec<f32>`). This
//! module provides four standard strategies to collapse that matrix into a
//! single `hidden_size`-dimensional vector suitable for similarity search,
//! retrieval, and reranking.
//!
//! # Modes
//!
//! | Mode   | Description |
//! |--------|-------------|
//! | `Last` | Return the hidden state of the **last** token (default). Appropriate for causal / decoder-only models such as LLaMA. |
//! | `Mean` | Elementwise arithmetic mean across **all** tokens. Standard choice for BERT-style models. |
//! | `Max`  | Elementwise maximum across all tokens. Captures the "most activated" feature in the sequence. |
//! | `Cls`  | Return the hidden state of the **first** token (CLS). Used by BERT and its variants. |
//!
//! # Usage
//!
//! ```ignore
//! use oxillama_runtime::embedding::{pool_hidden_states, PoolingMode};
//!
//! // hidden is a flat [seq_len × hidden_size] matrix.
//! let pooled = pool_hidden_states(&hidden, seq_len, hidden_size, PoolingMode::Mean)?;
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{RuntimeError, RuntimeResult};

/// Strategy for collapsing a sequence of hidden states into a single vector.
///
/// The default mode is [`PoolingMode::Last`], which is appropriate for causal
/// LLMs (the last token's hidden state encodes the full context).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PoolingMode {
    /// Return the hidden state of the **last** token in the sequence.
    ///
    /// Shape: `states[(seq_len - 1) * hidden_size .. seq_len * hidden_size]`.
    /// This is the natural pooling choice for causal / decoder-only models.
    #[default]
    Last,

    /// Elementwise arithmetic mean across **all** `seq_len` token hidden states.
    ///
    /// `result[j] = (1 / seq_len) * Σ_{i=0}^{seq_len-1} states[i * hidden_size + j]`
    Mean,

    /// Elementwise maximum across **all** `seq_len` token hidden states.
    ///
    /// `result[j] = max_{i=0..seq_len-1} states[i * hidden_size + j]`
    Max,

    /// Return the hidden state of the **first** token (CLS position).
    ///
    /// Shape: `states[0 .. hidden_size]`.
    /// This is the standard pooling choice for BERT-style encoder models.
    Cls,
}

/// Pool a sequence of per-token hidden states into a single vector.
///
/// # Arguments
///
/// * `states` — Flat `[seq_len × hidden_size]` matrix in row-major order.
///   Total length must equal `seq_len * hidden_size`.
/// * `seq_len` — Number of tokens in the sequence (rows). Must be ≥ 1.
/// * `hidden_size` — Dimensionality of each token's hidden state (columns).
/// * `mode` — Pooling strategy; see [`PoolingMode`].
///
/// # Returns
///
/// A `Vec<f32>` of length `hidden_size`.
///
/// # Errors
///
/// * [`RuntimeError::EmptySequence`] — when `seq_len == 0`.
/// * [`RuntimeError::SamplingError`] — when `states.len() != seq_len * hidden_size`
///   (indicates a mismatched buffer from the forward pass).
pub fn pool_hidden_states(
    states: &[f32],
    seq_len: usize,
    hidden_size: usize,
    mode: PoolingMode,
) -> RuntimeResult<Vec<f32>> {
    if seq_len == 0 {
        return Err(RuntimeError::EmptySequence);
    }

    let expected_len = seq_len * hidden_size;
    if states.len() != expected_len {
        return Err(RuntimeError::SamplingError {
            message: format!(
                "pool_hidden_states: states.len()={} != seq_len({}) * hidden_size({}) = {}",
                states.len(),
                seq_len,
                hidden_size,
                expected_len,
            ),
        });
    }

    if hidden_size == 0 {
        return Ok(Vec::new());
    }

    match mode {
        PoolingMode::Last => pool_last(states, seq_len, hidden_size),
        PoolingMode::Mean => pool_mean(states, seq_len, hidden_size),
        PoolingMode::Max => pool_max(states, seq_len, hidden_size),
        PoolingMode::Cls => pool_cls(states, hidden_size),
    }
}

// ── Per-mode kernels ──────────────────────────────────────────────────────────

/// Return a copy of the last row: `states[(seq_len-1)*hidden_size..]`.
fn pool_last(states: &[f32], seq_len: usize, hidden_size: usize) -> RuntimeResult<Vec<f32>> {
    let start = (seq_len - 1) * hidden_size;
    Ok(states[start..start + hidden_size].to_vec())
}

/// Elementwise arithmetic mean across all `seq_len` rows.
fn pool_mean(states: &[f32], seq_len: usize, hidden_size: usize) -> RuntimeResult<Vec<f32>> {
    let mut result = vec![0.0f32; hidden_size];
    for row in 0..seq_len {
        let offset = row * hidden_size;
        for j in 0..hidden_size {
            result[j] += states[offset + j];
        }
    }
    let inv_n = 1.0 / seq_len as f32;
    for v in &mut result {
        *v *= inv_n;
    }
    Ok(result)
}

/// Elementwise maximum across all `seq_len` rows.
fn pool_max(states: &[f32], seq_len: usize, hidden_size: usize) -> RuntimeResult<Vec<f32>> {
    // Initialise with the first row so that we handle negative values correctly.
    let mut result = states[0..hidden_size].to_vec();
    for row in 1..seq_len {
        let offset = row * hidden_size;
        for j in 0..hidden_size {
            if states[offset + j] > result[j] {
                result[j] = states[offset + j];
            }
        }
    }
    Ok(result)
}

/// Return a copy of the first row (CLS token): `states[0..hidden_size]`.
fn pool_cls(states: &[f32], hidden_size: usize) -> RuntimeResult<Vec<f32>> {
    Ok(states[0..hidden_size].to_vec())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat `[seq_len × hidden_size]` matrix where `states[i][j] = (i+1) * (j+1)`.
    fn make_states(seq_len: usize, hidden_size: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(seq_len * hidden_size);
        for i in 0..seq_len {
            for j in 0..hidden_size {
                v.push(((i + 1) * (j + 1)) as f32);
            }
        }
        v
    }

    /// `PoolingMode::Last` must return exactly the last row of the matrix.
    #[test]
    fn pooling_last_matches_last_row() {
        let seq_len = 4;
        let hidden_size = 3;
        let states = make_states(seq_len, hidden_size);
        let pooled = pool_hidden_states(&states, seq_len, hidden_size, PoolingMode::Last)
            .expect("Last pooling must succeed");

        // Last row (i=3): values are (3+1)*(j+1) = 4*(j+1)
        let expected: Vec<f32> = (0..hidden_size).map(|j| 4.0 * (j + 1) as f32).collect();
        assert_eq!(pooled, expected, "Last pooling should return the last row");
    }

    /// `PoolingMode::Mean` must return the elementwise arithmetic mean.
    #[test]
    fn pooling_mean_is_arithmetic_mean() {
        let seq_len = 3;
        let hidden_size = 2;
        // states[i][j] = (i+1)*(j+1)
        // For j=0: values are 1*1=1, 2*1=2, 3*1=3 → mean = 2.0
        // For j=1: values are 1*2=2, 2*2=4, 3*2=6 → mean = 4.0
        let states = make_states(seq_len, hidden_size);
        let pooled = pool_hidden_states(&states, seq_len, hidden_size, PoolingMode::Mean)
            .expect("Mean pooling must succeed");

        assert_eq!(
            pooled.len(),
            hidden_size,
            "output length must equal hidden_size"
        );
        assert!(
            (pooled[0] - 2.0).abs() < 1e-5,
            "Mean at j=0 should be 2.0, got {}",
            pooled[0]
        );
        assert!(
            (pooled[1] - 4.0).abs() < 1e-5,
            "Mean at j=1 should be 4.0, got {}",
            pooled[1]
        );
    }

    /// `PoolingMode::Max` must return the elementwise maximum across rows.
    #[test]
    fn pooling_max_is_elementwise_max() {
        let seq_len = 4;
        let hidden_size = 3;
        // states[i][j] = (i+1)*(j+1), so the max across i is at i=3 (last row).
        // Max[j] = 4*(j+1).
        let states = make_states(seq_len, hidden_size);
        let pooled = pool_hidden_states(&states, seq_len, hidden_size, PoolingMode::Max)
            .expect("Max pooling must succeed");

        let expected: Vec<f32> = (0..hidden_size).map(|j| 4.0 * (j + 1) as f32).collect();
        assert_eq!(
            pooled, expected,
            "Max pooling should return elementwise max"
        );
    }

    /// `PoolingMode::Cls` must return exactly the first row of the matrix.
    #[test]
    fn pooling_cls_matches_first_row() {
        let seq_len = 5;
        let hidden_size = 4;
        let states = make_states(seq_len, hidden_size);
        let pooled = pool_hidden_states(&states, seq_len, hidden_size, PoolingMode::Cls)
            .expect("Cls pooling must succeed");

        // First row (i=0): values are (0+1)*(j+1) = j+1
        let expected: Vec<f32> = (0..hidden_size).map(|j| (j + 1) as f32).collect();
        assert_eq!(pooled, expected, "Cls pooling should return the first row");
    }

    /// `pool_hidden_states` with `seq_len == 0` must return `EmptySequence`.
    #[test]
    fn pooling_empty_sequence_errors() {
        let result = pool_hidden_states(&[], 0, 4, PoolingMode::Last);
        assert!(
            matches!(result, Err(RuntimeError::EmptySequence)),
            "empty seq_len must produce EmptySequence error"
        );
    }

    /// Mismatched buffer length must return a `SamplingError`.
    #[test]
    fn pooling_wrong_buffer_length_errors() {
        let states = vec![0.0f32; 10]; // 10 ≠ 2*3=6
        let result = pool_hidden_states(&states, 2, 3, PoolingMode::Mean);
        assert!(
            matches!(result, Err(RuntimeError::SamplingError { .. })),
            "mismatched buffer must produce SamplingError"
        );
    }

    /// All four modes must work correctly for a single-token sequence.
    #[test]
    fn pooling_single_token_sequence() {
        let seq_len = 1;
        let hidden_size = 3;
        let states = vec![1.0f32, 2.0, 3.0];

        for mode in [
            PoolingMode::Last,
            PoolingMode::Mean,
            PoolingMode::Max,
            PoolingMode::Cls,
        ] {
            let pooled = pool_hidden_states(&states, seq_len, hidden_size, mode)
                .unwrap_or_else(|e| panic!("mode {mode:?} failed: {e}"));
            assert_eq!(
                pooled, states,
                "single-token pooling with {mode:?} must return the only row unchanged"
            );
        }
    }

    /// Mean pooling of a matrix where all values in each column are the same
    /// must return those column values.
    #[test]
    fn pooling_mean_constant_columns() {
        // 3 rows, 2 columns. Col 0 all 5.0, col 1 all 7.0.
        let states = vec![5.0f32, 7.0, 5.0, 7.0, 5.0, 7.0];
        let pooled = pool_hidden_states(&states, 3, 2, PoolingMode::Mean)
            .expect("mean pooling must succeed");
        assert!(
            (pooled[0] - 5.0).abs() < 1e-6,
            "mean of constant column 0 must be 5.0"
        );
        assert!(
            (pooled[1] - 7.0).abs() < 1e-6,
            "mean of constant column 1 must be 7.0"
        );
    }
}

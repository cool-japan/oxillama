//! Parallel (multi-threaded) wrappers for quantized matrix operations.
//!
//! When the `parallel` feature is enabled (default), uses rayon to parallelize
//! across output rows in GEMV/GEMM, which is the primary source of latency in
//! LLM inference. Each row's dot product is independent, making this
//! embarrassingly parallel.
//!
//! When `parallel` is disabled (e.g. for `wasm32-unknown-unknown`), falls back
//! to identical single-threaded logic so the same API compiles everywhere.
//!
//! The underlying single-row computation still uses the same scalar
//! (or SIMD) kernels — this module only adds thread-level parallelism.
//!
//! # Thread pool ownership
//!
//! Row splitting runs on a **dedicated** rayon [`ThreadPool`](rayon::ThreadPool)
//! owned by this crate, *not* on rayon's global pool.  A library consumer that
//! uses rayon for its own work therefore sees no change in the global pool's
//! thread count and no interference from oxillama's GEMVs.
//!
//! The pool is built lazily on the first parallel GEMV and its width is
//! resolved, in order of precedence:
//!
//! 1. the `OXILLAMA_NUM_THREADS` environment variable, when it parses to a
//!    non-zero `usize`;
//! 2. the value most recently passed to [`set_num_threads`] *before* the first
//!    parallel GEMV (this is how `oxillama-runtime` forwards
//!    `EngineConfig::num_threads`);
//! 3. [`std::thread::available_parallelism`].
//!
//! If the pool cannot be built, every operation degrades to the identical
//! single-threaded loop rather than failing.

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "parallel")]
use std::sync::OnceLock;

use crate::error::{QuantError, QuantResult};
use crate::types::QuantTensor;

/// Environment variable that overrides the GEMV thread-pool width.
pub const NUM_THREADS_ENV: &str = "OXILLAMA_NUM_THREADS";

/// Requested pool width; `0` means "not set, fall back to auto-detection".
#[cfg(feature = "parallel")]
static REQUESTED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// The dedicated GEMV thread pool, built at most once.
///
/// `None` means the pool could not be constructed and callers must run the
/// serial fallback.
#[cfg(feature = "parallel")]
static GEMV_POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// Request a width for the GEMV thread pool.
///
/// Returns `true` when the request was recorded *and* the pool has not been
/// built yet (so the value will be honoured), `false` when the pool is already
/// running with a fixed width or when `n_threads` is zero.  Passing `0` is a
/// no-op that always returns `false`.
///
/// `OXILLAMA_NUM_THREADS` still wins over this value — an operator can always
/// override what the embedding application asked for.
///
/// This has no effect when the `parallel` feature is disabled.
pub fn set_num_threads(n_threads: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        if n_threads == 0 {
            return false;
        }
        REQUESTED_THREADS.store(n_threads, Ordering::Relaxed);
        GEMV_POOL.get().is_none()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = n_threads;
        false
    }
}

/// Number of threads the GEMV pool is (or would be) built with.
///
/// Returns `1` when the `parallel` feature is disabled.
pub fn num_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        match gemv_pool() {
            Some(pool) => pool.current_num_threads(),
            None => 1,
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Resolve the pool width from env → explicit request → `available_parallelism`.
#[cfg(feature = "parallel")]
fn resolve_num_threads() -> usize {
    if let Ok(raw) = std::env::var(NUM_THREADS_ENV) {
        if let Ok(n) = raw.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    let requested = REQUESTED_THREADS.load(Ordering::Relaxed);
    if requested > 0 {
        return requested;
    }
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

/// Borrow the lazily built GEMV thread pool.
#[cfg(feature = "parallel")]
fn gemv_pool() -> Option<&'static rayon::ThreadPool> {
    GEMV_POOL
        .get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(resolve_num_threads())
                .thread_name(|i| format!("oxillama-gemv-{i}"))
                .build()
                .ok()
        })
        .as_ref()
}

/// Evaluate one GEMV output row per `output` slot, in parallel when worthwhile.
///
/// This is the single place where every quantization kernel's `gemv` /
/// `matvec_q8_fused` row loop obtains thread-level parallelism.  `row_fn` is
/// handed `(row_index, &mut output[row_index])` and must be free of
/// cross-row state, which is exactly the shape of every GEMV row dot product.
///
/// Rows are never split, so the floating-point accumulation order inside a row
/// is unchanged and results are **bit-identical** to the serial loop
/// regardless of the thread count.
///
/// Falls back to a plain serial loop when the `parallel` feature is off, when
/// [`should_parallelize`] says the matrix is too small to pay for scheduling,
/// or when the thread pool could not be built.
///
/// At most `output.len()` rows are visited, mirroring the `.take(n_rows)` of
/// the serial loop it replaces.
#[inline]
pub fn for_each_row<F>(output: &mut [f32], n_rows: usize, n_cols: usize, row_fn: F)
where
    F: Fn(usize, &mut f32) + Send + Sync,
{
    let n = n_rows.min(output.len());

    #[cfg(feature = "parallel")]
    {
        if should_parallelize(n, n_cols) {
            if let Some(pool) = gemv_pool() {
                if pool.current_num_threads() > 1 {
                    // Rayon's *adaptive* splitting (halve-until-a-steal-fails)
                    // beats every fixed row-band size measured on this
                    // big.LITTLE CPU: fixed bands hand the slow efficiency
                    // cores a share they cannot finish on time, and the whole
                    // GEMV waits for them.  Measured on Apple M3 (4P+4E),
                    // Qwen3-4B Q4_K_M decode: adaptive 6.37 tok/s vs 5.08
                    // (4 bands/thread) and 4.34 (32 bands/thread).
                    let rows = &mut output[..n];
                    pool.install(move || {
                        rows.par_iter_mut()
                            .enumerate()
                            .for_each(|(row, out)| row_fn(row, out));
                    });
                    return;
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    let _ = n_cols;

    for (row, out) in output.iter_mut().enumerate().take(n) {
        row_fn(row, out);
    }
}

/// Batched sibling of [`for_each_row`]: evaluate one weight row against `m`
/// activation vectors at a time.
///
/// `output` is the feature-major `[n_rows][m]` accumulator a batched matmul
/// writes, so weight row `row` owns the **contiguous** slice
/// `output[row * m .. (row + 1) * m]`.  `row_fn` is handed
/// `(row_index, &mut output[row_index * m ..][.. m])`.
///
/// The invariants of [`for_each_row`] carry over verbatim and are what make a
/// batched prefill numerically indistinguishable from replaying the per-token
/// path:
///
/// * the same **dedicated** pool (`gemv_pool`) is used — never rayon's global
///   pool;
/// * a weight row is never split across threads, so the per-`(row, token)` dot
///   product keeps a single accumulation order;
/// * every `m` slot of a row is produced by the same closure invocation, so
///   the batch dimension is not a source of reassociation either.
///
/// Work per task is `m` times a plain GEMV row, so the
/// [`should_parallelize`] test is applied to the *scaled* column count: a
/// matmul with few rows still deserves threads once the batch makes each row
/// expensive.
#[inline]
pub fn for_each_row_batch<F>(output: &mut [f32], n_rows: usize, n_cols: usize, m: usize, row_fn: F)
where
    F: Fn(usize, &mut [f32]) + Send + Sync,
{
    if m == 0 {
        return;
    }
    let n = n_rows.min(output.len() / m);
    for_each_chunk_init(
        &mut output[..n * m],
        m,
        n_cols.saturating_mul(m),
        || (),
        |(), row, out| row_fn(row, out),
    );
}

/// Split `output` into equal `chunk_len` pieces and evaluate one closure call
/// per piece, in parallel when the work justifies it.
///
/// The general primitive behind [`for_each_row`] and [`for_each_row_batch`],
/// and the one entry point a *non*-GEMV hot loop (batched-prefill attention,
/// say) should use so that all of oxillama's CPU parallelism stays on a single
/// pool.  Its guarantees are the same three:
///
/// * the **dedicated** pool (`gemv_pool`) is used, never rayon's global pool;
/// * a chunk is never split, so whatever accumulation the closure performs
///   keeps one order and one thread — results are bit-identical to the serial
///   loop at any thread count;
/// * chunks are disjoint, so the closure needs no synchronisation.
///
/// `state` is per-thread scratch built by `init`.  Rayon calls `init` once per
/// worker that actually receives chunks (once total on the serial path), which
/// is how a closure that needs a scratch buffer avoids allocating per chunk.
///
/// `work_per_chunk` is a rough operation count for one chunk; together with the
/// chunk count it decides whether threading is worth its scheduling cost (see
/// [`should_parallelize`]).
#[inline]
pub fn for_each_chunk_init<S, I, F>(
    output: &mut [f32],
    chunk_len: usize,
    work_per_chunk: usize,
    init: I,
    f: F,
) where
    I: Fn() -> S + Send + Sync,
    S: Send,
    F: Fn(&mut S, usize, &mut [f32]) + Send + Sync,
{
    if chunk_len == 0 {
        return;
    }
    let n_chunks = output.len() / chunk_len;

    #[cfg(feature = "parallel")]
    {
        if should_parallelize(n_chunks, work_per_chunk) {
            if let Some(pool) = gemv_pool() {
                if pool.current_num_threads() > 1 {
                    let chunks = &mut output[..n_chunks * chunk_len];
                    let init = &init;
                    let f = &f;
                    pool.install(move || {
                        chunks
                            .par_chunks_mut(chunk_len)
                            .enumerate()
                            .for_each_init(init, |state, (i, c)| f(state, i, c));
                    });
                    return;
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    let _ = work_per_chunk;

    let mut state = init();
    for (i, chunk) in output[..n_chunks * chunk_len]
        .chunks_mut(chunk_len)
        .enumerate()
    {
        f(&mut state, i, chunk);
    }
}

/// Parallel GEMV: compute `output = quant_matrix @ input` using rayon.
///
/// Each output row is computed independently in parallel. This is the
/// primary optimization for autoregressive decode (single-token inference).
///
/// Delegates to [`for_each_row`], which is what makes this function honour
/// the module's dedicated-pool guarantee: an earlier version of this
/// function called `par_iter_mut()` directly, which runs on rayon's
/// *global* pool rather than `gemv_pool` — the one place the module doc
/// promises every GEMV/GEMM uses. `for_each_row` also gets the
/// [`should_parallelize`] size threshold for free, which this function
/// previously lacked (it engaged rayon unconditionally, even for tiny
/// matrices where the scheduling overhead outweighs the work).
///
/// # Arguments
/// * `quant_matrix` - Quantized weight matrix [n_rows x n_cols].
/// * `input` - FP32 input vector of length n_cols.
/// * `output` - FP32 output vector of length n_rows (written in parallel).
/// * `block_size` - Number of weights per quantized block.
/// * `block_bytes` - Number of bytes per quantized block.
/// * `row_dot` - Function that computes the dot product for one row.
///   Signature: `fn(row_data: &[u8], input: &[f32], n_cols: usize) -> f32`
pub fn parallel_gemv<F>(
    quant_matrix: &QuantTensor,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
    block_bytes: usize,
    row_dot: F,
) -> QuantResult<()>
where
    F: Fn(&[u8], &[f32], usize) -> f32 + Send + Sync,
{
    let n_rows = quant_matrix.shape[0];
    let n_cols = if quant_matrix.shape.len() > 1 {
        quant_matrix.shape[1]
    } else {
        quant_matrix.n_elements() / n_rows
    };

    if input.len() < n_cols {
        return Err(QuantError::DimensionMismatch {
            expected: n_cols,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(QuantError::DimensionMismatch {
            expected: n_rows,
            got: output.len(),
        });
    }

    let blocks_per_row = n_cols.div_ceil(block_size);
    let row_bytes = blocks_per_row * block_bytes;
    let data = &quant_matrix.data;

    for_each_row(output, n_rows, n_cols, |row, out| {
        let row_start = row * row_bytes;
        let row_data = &data[row_start..row_start + row_bytes];
        *out = row_dot(row_data, input, n_cols);
    });

    Ok(())
}

/// Dimensions for a parallel GEMM operation.
pub struct GemmDims {
    /// Number of input rows (batch size).
    pub m: usize,
    /// Number of output columns (weight rows).
    pub n: usize,
    /// Shared inner dimension.
    pub k: usize,
    /// Number of weights per quantized block.
    pub block_size: usize,
    /// Number of bytes per quantized block.
    pub block_bytes: usize,
}

/// Parallel GEMM: compute `output = input_matrix @ quant_matrix^T` using rayon.
///
/// Parallelizes across rows of the input matrix (batch dimension), each
/// chunk of `dims.n` output slots being one batch row's full set of weight-row
/// dot products.
///
/// Delegates to [`for_each_chunk_init`], for the same reason
/// [`parallel_gemv`] delegates to [`for_each_row`]: an earlier version called
/// `par_chunks_mut()` directly, which runs on rayon's *global* pool instead
/// of the dedicated `gemv_pool` the module doc promises, and had no
/// [`should_parallelize`] size gate.
pub fn parallel_gemm<F>(
    quant_matrix: &QuantTensor,
    input: &[f32],
    output: &mut [f32],
    dims: &GemmDims,
    row_dot: F,
) -> QuantResult<()>
where
    F: Fn(&[u8], &[f32], usize) -> f32 + Send + Sync,
{
    let blocks_per_row = dims.k.div_ceil(dims.block_size);
    let weight_row_bytes = blocks_per_row * dims.block_bytes;
    let data = &quant_matrix.data;
    let n = dims.n;
    let k = dims.k;

    // Only the first `dims.m` chunks are meaningful — matches the previous
    // `.take(dims.m)` on the chunk iterator. `for_each_chunk_init` processes
    // `output.len() / chunk_len` whole chunks, so truncating the slice here
    // is what keeps a caller-provided `output` longer than `m * n` from
    // having chunks past `m` written.
    let usable_len = dims.m.saturating_mul(n).min(output.len());

    for_each_chunk_init(
        &mut output[..usable_len],
        n,
        n.saturating_mul(k),
        || (),
        |(), batch_row, out_row| {
            let inp_row = &input[batch_row * k..(batch_row + 1) * k];
            for (weight_row, out) in out_row.iter_mut().enumerate().take(n) {
                let row_start = weight_row * weight_row_bytes;
                let row_data = &data[row_start..row_start + weight_row_bytes];
                *out = row_dot(row_data, inp_row, k);
            }
        },
    );

    Ok(())
}

/// Minimum number of rows before engaging parallel execution.
/// For small matrices, thread overhead exceeds the benefit.
pub const PARALLEL_ROW_THRESHOLD: usize = 64;

/// Check whether parallel execution is worthwhile for the given dimensions.
///
/// Returns `true` if the matrix is large enough to benefit from parallelism.
pub fn should_parallelize(n_rows: usize, n_cols: usize) -> bool {
    // Heuristic: parallelize when total work exceeds threshold.
    // Each row does O(n_cols) work, so total is O(n_rows * n_cols).
    n_rows >= PARALLEL_ROW_THRESHOLD && n_cols >= 256
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::Q8_0Ref;
    use crate::traits::QuantKernel;

    fn make_q8_0_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        let d_bits = half::f16::from_f32(d).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    #[cfg_attr(miri, ignore)] // half crate aarch64 asm not supported in Miri
    #[test]
    fn test_parallel_gemv_matches_sequential() {
        // Build a 4x32 matrix (4 rows, each 1 block of Q8_0)
        let n_rows = 4;
        let n_cols = 32;
        let mut data = Vec::new();
        for row in 0..n_rows {
            let mut qs = [0i8; 32];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((row as i16 * 7 + i as i16 * 3 - 48).clamp(-128, 127)) as i8;
            }
            data.extend_from_slice(&make_q8_0_block(0.5, &qs));
        }

        let tensor = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32 * 0.1) - 1.6).collect();

        // Sequential reference
        let kernel = Q8_0Ref;
        let mut seq_output = vec![0.0f32; n_rows];
        kernel.gemv(&tensor, &input, &mut seq_output).unwrap();

        // Parallel
        let mut par_output = vec![0.0f32; n_rows];
        parallel_gemv(
            &tensor,
            &input,
            &mut par_output,
            32,
            34,
            |row_data, inp, _n_cols| {
                // Q8_0 row dot: d * sum(qs[i] * inp[i])
                let d =
                    half::f16::from_bits(u16::from_le_bytes([row_data[0], row_data[1]])).to_f32();
                let qs = &row_data[2..34];
                let mut sum = 0.0f32;
                for (i, &q) in qs.iter().enumerate() {
                    sum += (q as i8) as f32 * inp[i];
                }
                d * sum
            },
        )
        .unwrap();

        for (i, (&s, &p)) in seq_output.iter().zip(par_output.iter()).enumerate() {
            assert!(
                (s - p).abs() < 1e-4,
                "row {i}: sequential={s}, parallel={p}"
            );
        }
    }

    /// `for_each_row` visits exactly `min(n_rows, output.len())` slots, matching
    /// the `.take(n_rows)` semantics of the serial loop it replaced.
    #[test]
    fn test_for_each_row_clamps_to_output_len() {
        let mut out = vec![-1.0f32; 4];
        for_each_row(&mut out, 10, 4096, |row, slot| *slot = row as f32);
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0]);
    }

    /// Below [`should_parallelize`] the serial path still fills every slot.
    #[test]
    fn test_for_each_row_small_input_is_serial_and_complete() {
        let mut out = vec![0.0f32; 8];
        for_each_row(&mut out, 8, 8, |row, slot| *slot = (row * row) as f32);
        assert_eq!(out, vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]);
    }

    /// A parallel-sized workload produces the same values as the serial loop.
    #[test]
    fn test_for_each_row_parallel_matches_serial() {
        let n = 4097usize; // not a multiple of any lane or thread count
        let f = |row: usize| ((row % 97) as f32) * 0.25 - 3.0;

        let mut par = vec![0.0f32; n];
        for_each_row(&mut par, n, 4096, |row, slot| *slot = f(row));

        let mut serial = vec![0.0f32; n];
        for (row, slot) in serial.iter_mut().enumerate() {
            *slot = f(row);
        }
        assert_eq!(par, serial);
    }

    #[test]
    fn test_set_num_threads_rejects_zero() {
        assert!(
            !set_num_threads(0),
            "0 must not be accepted as a pool width"
        );
        assert!(num_threads() >= 1);
    }

    #[test]
    fn test_should_parallelize() {
        assert!(!should_parallelize(1, 256));
        assert!(!should_parallelize(32, 256));
        assert!(should_parallelize(64, 256));
        assert!(should_parallelize(4096, 4096));
        assert!(!should_parallelize(128, 32));
    }

    #[cfg_attr(miri, ignore)] // half crate aarch64 asm not supported in Miri
    #[test]
    fn test_parallel_gemv_input_too_small_errors() {
        let tensor = QuantTensor::new(
            make_q8_0_block(1.0, &[0i8; 32]),
            vec![1, 32],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let input = vec![0.0f32; 4]; // need 32
        let mut output = vec![0.0f32; 1];
        let result = parallel_gemv(&tensor, &input, &mut output, 32, 34, |_, _, _| 0.0);
        assert!(result.is_err(), "too-small input should error");
    }

    #[cfg_attr(miri, ignore)] // half crate aarch64 asm not supported in Miri
    #[test]
    fn test_parallel_gemv_output_too_small_errors() {
        let tensor = QuantTensor::new(
            make_q8_0_block(1.0, &[0i8; 32]),
            vec![2, 32],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let input = vec![0.0f32; 32];
        let mut output = vec![0.0f32; 1]; // need 2
        let result = parallel_gemv(&tensor, &input, &mut output, 32, 34, |_, _, _| 0.0);
        assert!(result.is_err(), "too-small output should error");
    }

    #[cfg_attr(miri, ignore)] // half crate aarch64 asm not supported in Miri
    #[test]
    fn test_parallel_gemm_basic() {
        // 2 weight rows of 32 cols Q8_0; 1 batch input row
        let n_rows = 2usize;
        let n_cols = 32usize;
        let mut data = Vec::new();
        for row in 0..n_rows {
            let mut qs = [0i8; 32];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((row as i16 + i as i16) % 10) as i8;
            }
            data.extend_from_slice(&make_q8_0_block(0.25, &qs));
        }
        let tensor = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let m = 1usize;
        let k = n_cols;
        let input = vec![1.0f32; k]; // batch of 1 input row
        let mut output = vec![0.0f32; m * n_rows]; // [m x n_rows]

        let dims = GemmDims {
            m,
            n: n_rows,
            k,
            block_size: 32,
            block_bytes: 34,
        };

        let result = parallel_gemm(&tensor, &input, &mut output, &dims, |row_data, inp, _nc| {
            let d = half::f16::from_bits(u16::from_le_bytes([row_data[0], row_data[1]])).to_f32();
            let qs = &row_data[2..34];
            let mut sum = 0.0f32;
            for (i, &q) in qs.iter().enumerate() {
                if i < inp.len() {
                    sum += (q as i8) as f32 * inp[i];
                }
            }
            d * sum
        });
        assert!(result.is_ok(), "parallel_gemm should succeed: {result:?}");
    }

    /// Regression test for `parallel_gemv` using rayon's *global* pool
    /// instead of this module's dedicated [`gemv_pool`]: build a workload
    /// large enough to cross [`should_parallelize`]'s threshold (`>= 64` rows,
    /// `>= 256` cols), record which thread each row's `row_dot` closure runs
    /// on, and assert at least one row ran on a `gemv_pool` worker (named
    /// `oxillama-gemv-*`, see [`gemv_pool`]) rather than an unnamed
    /// rayon-global-pool thread.
    ///
    /// One weight block per row (`block_size == n_cols`) keeps the fixture
    /// small — `row_dot` never reads the bytes, it only needs to run.
    ///
    /// This can only demonstrate the fix on a host with more than one core
    /// (a single-thread pool takes the serial fallback either way and proves
    /// nothing), so it skips rather than fails there.
    #[test]
    fn test_parallel_gemv_uses_dedicated_pool_not_global() {
        if num_threads() <= 1 {
            return; // Nothing to distinguish on a single-thread pool.
        }

        let n_rows = 4096usize; // well past PARALLEL_ROW_THRESHOLD (64)
        let n_cols = 256usize; // meets should_parallelize's column floor
        let block_size = n_cols; // one block per row
        let block_bytes = 1usize; // row_dot never reads it
        let data = vec![0u8; n_rows * block_bytes];
        let tensor = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::F32,
        );
        let input = vec![1.0f32; n_cols];
        let mut output = vec![0.0f32; n_rows];

        let saw_dedicated_pool_thread = std::sync::atomic::AtomicBool::new(false);
        let saw_any_worker_thread = std::sync::atomic::AtomicBool::new(false);

        let result = parallel_gemv(
            &tensor,
            &input,
            &mut output,
            block_size,
            block_bytes,
            |_row_data, _inp, _n_cols| {
                let name = std::thread::current().name().unwrap_or("").to_string();
                if !name.is_empty() {
                    saw_any_worker_thread.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                if name.starts_with("oxillama-gemv-") {
                    saw_dedicated_pool_thread.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                1.0f32
            },
        );
        assert!(result.is_ok(), "parallel_gemv should succeed: {result:?}");

        assert!(
            saw_any_worker_thread.load(std::sync::atomic::Ordering::Relaxed),
            "workload should have run on a named worker thread, proving the parallel path engaged"
        );
        assert!(
            saw_dedicated_pool_thread.load(std::sync::atomic::Ordering::Relaxed),
            "row_dot must run on the dedicated gemv_pool (oxillama-gemv-*), not rayon's global pool \
             — parallel_gemv used to call par_iter_mut() directly, bypassing gemv_pool() entirely"
        );
    }
}

//! Token embedding table — kept quantized, dequantized one row at a time.
//!
//! Shared by every architecture that keeps its `token_embd` table in GGUF
//! form instead of bulk-dequantizing it to `f32` at load time (currently
//! `llama` and `qwen3`; the Mixtral module reuses `llama`'s via
//! `crate::llama::TokenEmbedding`).  DBRX does not: it still bulk-dequantizes
//! `token_embd` into a plain `Vec<f32>` in `dbrx::loader`, a separate
//! memory-footprint gap this module does not close.
//!
//! # Why this is not a `Vec<f32>`
//!
//! `token_embd.weight` is routinely the single largest tensor in a
//! checkpoint: `[vocab, hidden]` is 128256 × 4096 for Llama-3-8B and
//! 151936 × 2560 for Qwen3-4B.  Materialising either as f32 costs **2.10 GB**
//! / **1.556 GB** of anonymous memory — against ~295 MB / a bit over half of
//! a 2.38 GB `Q4_K_M` file on disk — and every byte of it is dead weight,
//! because a forward pass reads exactly **one** row per token.
//!
//! [`TokenEmbedding::Quantized`] keeps the GGUF block bytes (a
//! [`SharedBytes`][oxillama_gguf::SharedBytes] view straight into the backing
//! store, so they cost nothing on top of it) and dequantizes the looked-up
//! row into the destination buffer on demand.  For Llama-3-8B that is 16
//! Q4_K blocks once per token, against the ~225 GEMVs of ≥4096 weights each
//! that the same token then triggers; for Qwen3-4B it is 10 Q4_K blocks
//! against ~400 GEMVs of ≥2560 weights.  It does not register.
//!
//! The values are bit-identical to the bulk-dequantized table: the same
//! [`QuantKernel::dequant_block`][oxillama_quant::QuantKernel::dequant_block]
//! runs over the same block bytes, just fewer of them.  Row `t` occupies
//! elements `[t * hidden, (t + 1) * hidden)`, which is a whole number of
//! blocks whenever `hidden % block_size == 0` — always true for GGUF, whose
//! row length must be a multiple of the block size.  When it somehow is not,
//! the table falls back to [`TokenEmbedding::Dense`] and the old bulk
//! dequantization, so correctness never depends on the assumption.
//!
//! # Tied LM head
//!
//! Several checkpoints (Llama-3.2-1B/3B, Qwen3 ≤4B) ship no `output.weight`
//! and reuse `token_embd.weight` as the output projection.  That path is
//! untouched by this module: the LM head is loaded separately as a
//! [`QuantLinear`][crate::common::linear::QuantLinear] over the *same*
//! tensor and keeps running through the fused/quantized GEMV kernels.  Both
//! views share one `SharedBytes`, so tying costs no extra memory either.
//!
//! # Measured (Qwen3-4B `Q4_K_M`, Apple M3, 2026-08-04)
//!
//! Together with the mmap-backed projection weights, dropping the f32 table
//! took the CLI from 6.736 GB RSS / 4.204 GB peak footprint to 2.684 GB /
//! 0.148 GB, and cut weight-load wall time from 0.60 s (cache-warm 0.45 s) to
//! 0.015 s.  Decode throughput was unchanged — median-of-5 13.22 → 14.15
//! tok/s on the differencing harness, 13.01 → 13.19 tok/s on a lower-noise
//! 8→88-token variant over 15 interleaved pairs — and a fixed-seed generation
//! stayed character-identical, at 1, 4 and 8 threads.

use oxillama_gguf::GgufTensorType;
use oxillama_quant::{KernelDispatcher, QuantTensor};

use crate::error::{ArchError, ArchResult};

/// The token embedding table, in whichever form the checkpoint allows.
pub enum TokenEmbedding {
    /// A fully materialised `[vocab * hidden]` f32 table.
    ///
    /// Used by callers that already hold f32 weights, and as the fallback for
    /// exotic layouts where a row is not a whole number of quantization blocks.
    Dense {
        /// Row-major `[vocab][hidden]` weights.
        weights: Vec<f32>,
        /// Row stride, i.e. the model's hidden size.
        hidden: usize,
    },
    /// The GGUF block bytes, dequantized one row per lookup.
    Quantized {
        /// `[vocab, hidden]` quantized tensor, typically a shared mmap view.
        tensor: QuantTensor,
        /// Row stride in elements (hidden size).
        hidden: usize,
        /// Row stride in bytes (`hidden / block_size * block_bytes`).
        row_bytes: usize,
    },
}

impl TokenEmbedding {
    /// Wrap an already-dequantized table.
    ///
    /// `hidden` is the row stride; a zero stride yields an empty table that
    /// fails every lookup rather than dividing by zero.
    pub fn dense(weights: Vec<f32>, hidden: usize) -> Self {
        Self::Dense { weights, hidden }
    }

    /// Build a table over a quantized `[vocab, hidden]` tensor.
    ///
    /// Returns `None` when the tensor's rows do not align to block boundaries or
    /// the payload is short — the caller should then dequantize in bulk.
    pub fn quantized(tensor: QuantTensor) -> Option<Self> {
        let [vocab, hidden] = tensor.shape[..] else {
            return None;
        };
        if hidden == 0 || vocab == 0 {
            return None;
        }

        // Row stride from the block geometry.  The float types report
        // `block_size == 1`, so this covers them too.
        let block_size = tensor.tensor_type.block_size();
        let block_bytes = tensor.tensor_type.block_bytes();
        if block_size == 0 || block_bytes == 0 || hidden % block_size != 0 {
            return None;
        }
        let row_bytes = (hidden / block_size).checked_mul(block_bytes)?;

        if tensor.data.len() < vocab.checked_mul(row_bytes)? {
            return None;
        }

        Some(Self::Quantized {
            tensor,
            hidden,
            row_bytes,
        })
    }

    /// Row stride in elements (the model's hidden size).
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Dense { hidden, .. } | Self::Quantized { hidden, .. } => *hidden,
        }
    }

    /// Number of rows the table holds.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Dense { weights, hidden } => {
                if *hidden == 0 {
                    0
                } else {
                    weights.len() / hidden
                }
            }
            Self::Quantized { tensor, .. } => tensor.shape.first().copied().unwrap_or(0),
        }
    }

    /// Write the embedding row for `token` into `out`.
    ///
    /// `out.len()` must be at least [`Self::hidden_size`]; only that prefix is
    /// written.  Out-of-range tokens and short buffers are reported as errors —
    /// this path must never panic, it runs on every decoded token.
    pub fn row_into(
        &self,
        dispatcher: &KernelDispatcher,
        token: u32,
        out: &mut [f32],
    ) -> ArchResult<()> {
        let hidden = self.hidden_size();
        if hidden == 0 || out.len() < hidden {
            return Err(ArchError::InvalidShape {
                name: "token_embd row".to_string(),
                expected: vec![hidden],
                got: vec![out.len()],
            });
        }
        let row = token as usize;
        if row >= self.vocab_size() {
            return Err(ArchError::ConfigMismatch {
                param: "token id".to_string(),
                expected: format!("< {}", self.vocab_size()),
                got: row.to_string(),
            });
        }
        let dst = &mut out[..hidden];

        match self {
            Self::Dense { weights, .. } => {
                let start = row * hidden;
                let src =
                    weights
                        .get(start..start + hidden)
                        .ok_or_else(|| ArchError::InvalidShape {
                            name: "token_embd table".to_string(),
                            expected: vec![start + hidden],
                            got: vec![weights.len()],
                        })?;
                dst.copy_from_slice(src);
                Ok(())
            }
            Self::Quantized {
                tensor, row_bytes, ..
            } => {
                let start = row * *row_bytes;
                let bytes = tensor.data.get(start..start + *row_bytes).ok_or_else(|| {
                    ArchError::InvalidShape {
                        name: "token_embd payload".to_string(),
                        expected: vec![start + *row_bytes],
                        got: vec![tensor.data.len()],
                    }
                })?;
                dequant_row(tensor.tensor_type, bytes, dst, dispatcher)
            }
        }
    }
}

/// Dequantize one whole row out of `bytes` into `out`.
///
/// `bytes` is exactly one row's payload and `out` exactly one row wide.
///
/// F32/F16 are decoded inline to mirror the bulk loader's own special cases byte
/// for byte; everything else goes block by block through the dispatched kernel,
/// exactly as the bulk path does.
fn dequant_row(
    tensor_type: GgufTensorType,
    bytes: &[u8],
    out: &mut [f32],
    dispatcher: &KernelDispatcher,
) -> ArchResult<()> {
    match tensor_type {
        GgufTensorType::F32 => {
            for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(())
        }
        GgufTensorType::F16 => {
            for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            }
            Ok(())
        }
        other => {
            let kernel = dispatcher.get_kernel(other)?;
            let block_size = other.block_size();
            let block_bytes = other.block_bytes();
            if block_size == 0 || block_bytes == 0 {
                return Err(ArchError::NotSupported {
                    detail: format!("token_embd tensor type {other:?} has no block geometry"),
                });
            }
            // `bytes` is exactly one row and `out` exactly `hidden` wide, so the
            // zip runs `hidden / block_size` times and neither side can fall
            // short — expressed as an iterator so it cannot panic even if a
            // caller ever violates that.
            for (src, dst) in bytes
                .chunks_exact(block_bytes)
                .zip(out.chunks_exact_mut(block_size))
            {
                kernel.dequant_block(src, dst)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_quant::dequantize_to_f32;

    const HIDDEN: usize = 256;
    const VOCAB: usize = 7;

    /// Deterministic Q8_0 payload for a `[VOCAB, HIDDEN]` table.
    fn q8_0_table() -> Vec<u8> {
        let mut out = Vec::new();
        for row in 0..VOCAB {
            for blk in 0..HIDDEN / 32 {
                out.extend_from_slice(
                    &half::f16::from_f32(0.0125 * (blk + 1) as f32).to_le_bytes(),
                );
                for col in 0..32 {
                    out.push((((row * 31 + blk * 7 + col) % 251) as i8) as u8);
                }
            }
        }
        out
    }

    fn q8_0_tensor() -> QuantTensor {
        QuantTensor::new(q8_0_table(), vec![VOCAB, HIDDEN], GgufTensorType::Q8_0)
    }

    /// A per-row lookup must be bit-identical to the bulk table it replaces —
    /// otherwise swapping the representation is a silent numerics change.
    #[test]
    fn quantized_row_matches_bulk_dequantization_bitwise() {
        let bulk = dequantize_to_f32(&q8_0_table(), GgufTensorType::Q8_0, VOCAB * HIDDEN)
            .expect("test: bulk dequantize");
        let embd = TokenEmbedding::quantized(q8_0_tensor()).expect("test: block-aligned rows");
        let dispatcher = KernelDispatcher::new();

        let mut row = vec![0.0f32; HIDDEN];
        for token in 0..VOCAB {
            embd.row_into(&dispatcher, token as u32, &mut row)
                .expect("test: row lookup");
            let expected = &bulk[token * HIDDEN..(token + 1) * HIDDEN];
            assert_eq!(
                row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row {token} must be bit-identical to the bulk table"
            );
        }
    }

    /// The dense fallback and the quantized path agree, including on the
    /// metadata queries (`vocab_size`/`hidden_size`) callers use to size
    /// buffers before ever calling `row_into`.
    #[test]
    fn dense_and_quantized_agree() {
        let bulk = dequantize_to_f32(&q8_0_table(), GgufTensorType::Q8_0, VOCAB * HIDDEN)
            .expect("test: bulk dequantize");
        let dense = TokenEmbedding::dense(bulk, HIDDEN);
        let quant = TokenEmbedding::quantized(q8_0_tensor()).expect("test: block-aligned rows");
        let dispatcher = KernelDispatcher::new();

        let mut a = vec![0.0f32; HIDDEN];
        let mut b = vec![0.0f32; HIDDEN];
        for token in 0..VOCAB as u32 {
            dense
                .row_into(&dispatcher, token, &mut a)
                .expect("test: dense lookup");
            quant
                .row_into(&dispatcher, token, &mut b)
                .expect("test: quantized lookup");
            assert_eq!(a, b, "dense and quantized lookups must agree");
        }
        assert_eq!(dense.vocab_size(), VOCAB);
        assert_eq!(quant.vocab_size(), VOCAB);
        assert_eq!(dense.hidden_size(), HIDDEN);
        assert_eq!(quant.hidden_size(), HIDDEN);
    }

    /// F32 and F16 tensors take the inline-decode branch, not the dispatched
    /// kernel loop — both architectures' checkpoints can ship an unquantized
    /// `token_embd.weight`.
    #[test]
    fn f32_and_f16_rows_round_trip() {
        let values: Vec<f32> = (0..VOCAB * 4).map(|i| i as f32 * 0.25).collect();
        let mut f32_bytes = Vec::new();
        let mut f16_bytes = Vec::new();
        for v in &values {
            f32_bytes.extend_from_slice(&v.to_le_bytes());
            f16_bytes.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        let dispatcher = KernelDispatcher::new();

        let t32 = TokenEmbedding::quantized(QuantTensor::new(
            f32_bytes,
            vec![VOCAB, 4],
            GgufTensorType::F32,
        ))
        .expect("test: f32 table");
        let t16 = TokenEmbedding::quantized(QuantTensor::new(
            f16_bytes,
            vec![VOCAB, 4],
            GgufTensorType::F16,
        ))
        .expect("test: f16 table");

        let mut row = vec![0.0f32; 4];
        t32.row_into(&dispatcher, 2, &mut row)
            .expect("test: f32 lookup");
        assert_eq!(row, values[8..12]);
        t16.row_into(&dispatcher, 2, &mut row)
            .expect("test: f16 lookup");
        for (got, want) in row.iter().zip(&values[8..12]) {
            assert_eq!(*got, half::f16::from_f32(*want).to_f32());
        }
    }

    /// Rows that are not whole blocks fall back instead of mis-addressing.
    #[test]
    fn misaligned_rows_are_rejected_so_the_caller_falls_back() {
        // 100 is not a multiple of the Q8_0 block size (32).
        let tensor = QuantTensor::new(vec![0u8; 4096], vec![2, 100], GgufTensorType::Q8_0);
        assert!(TokenEmbedding::quantized(tensor).is_none());
    }

    /// A short payload is rejected at construction, not indexed past at lookup.
    #[test]
    fn short_payload_is_rejected() {
        let tensor = QuantTensor::new(vec![0u8; 8], vec![VOCAB, HIDDEN], GgufTensorType::Q8_0);
        assert!(TokenEmbedding::quantized(tensor).is_none());
    }

    /// The decode hot path must never panic: bad ids and short buffers error.
    #[test]
    fn out_of_range_token_and_short_buffer_error_instead_of_panicking() {
        let embd = TokenEmbedding::quantized(q8_0_tensor()).expect("test: table");
        let dispatcher = KernelDispatcher::new();
        let mut row = vec![0.0f32; HIDDEN];
        assert!(embd.row_into(&dispatcher, VOCAB as u32, &mut row).is_err());
        assert!(embd.row_into(&dispatcher, u32::MAX, &mut row).is_err());
        let mut short = vec![0.0f32; HIDDEN - 1];
        assert!(embd.row_into(&dispatcher, 0, &mut short).is_err());
    }

    /// A zero vocab, zero hidden, or missing hidden dimension must be
    /// rejected at construction rather than dividing by zero or panicking on
    /// a short `shape` slice later.
    #[test]
    fn zero_dimensions_are_rejected() {
        assert!(TokenEmbedding::quantized(QuantTensor::new(
            Vec::new(),
            vec![0, HIDDEN],
            GgufTensorType::Q8_0
        ))
        .is_none());
        assert!(TokenEmbedding::quantized(QuantTensor::new(
            Vec::new(),
            vec![VOCAB, 0],
            GgufTensorType::Q8_0
        ))
        .is_none());
        assert!(TokenEmbedding::quantized(QuantTensor::new(
            Vec::new(),
            vec![VOCAB],
            GgufTensorType::Q8_0
        ))
        .is_none());
    }
}

# oxillama-quant

Quantization kernels for all GGUF quantization types used in LLM inference.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.4 — **Tests:** 502 passing

## What's New in v0.1.4 (2026-08-17)

- **Correctness fixes across seven formats** — `Q4_0`, `Q4_1`, `IQ4_NL`, `IQ4_XS` and `Q5_K` decoded to the wrong byte/bit-position mapping; `TQ1_0`/`TQ2_0` decoded digit-minor instead of digit-major, and `TQ1_0`'s `qh` used four 2-bit fields instead of upstream's base-3 fixed-point scheme — its *values* were wrong, not merely their order. Fixed across the scalar reference, AVX2, AVX-512 and NEON tiers, and both `Q4_0` encoders. Every fix is now golden-tested against values produced by compiling and running upstream llama.cpp's own `dequantize_row_*` C code, not the (previously also wrong) scalar reference other tiers were checked against.
- **K-quant encoders** (`src/kquant/`) — byte-for-byte ports of llama.cpp's `quantize_row_q{2,3,4,5,6}_K_ref`; `oxillama quantize --target` now accepts 13 formats instead of 2, byte-identical to compiled upstream C on 22 golden inputs.
- **`QuantTensor::data` changed type from `Vec<u8>` to `SharedBytes`** — a pre-1.0 API break enabling zero-copy mmap loading. `QuantTensor::new(Vec<u8>, ...)` is unchanged and still accepts a `Vec<u8>`; only direct field access/pattern-matching on `.data` breaks.
- **Threaded GEMV via a dedicated rayon pool** — all 122 serial per-row GEMV loops now route through `parallel::for_each_row`, bit-identical to serial at any thread count.
- **NEON SDOT integer dot products** via stable `asm!("sdot …")` (the `vdotq_s32` intrinsic is still unstable on current rustc); +11.8% decode on Apple M3.
- Test count: 389 → **502 tests passing**.

## What's New in v0.1.3 (2026-05-05)

- **AVX-512 IQ kernels** — IQ2_XXS, IQ2_XS, IQ3_S, IQ4_XS with AVX-512BW (`_mm512_permutexvar_epi8`); 2× throughput vs AVX2. Runtime-guarded via `is_x86_feature_detected!("avx512bw")`. 8 new tests.
- **Fused `matvec_q8` for Q5_0 / Q5_1 / Q8_1** — single-pass dequant+dot in registers with no scratch allocation; AVX2 + NEON + scalar reference; 6 new tests (tol 1e-5 on 64×1024 GEMV).
- Test count: 382 → **389 tests passing**.

## What's New in v0.1.2

- IQ3_S and IQ3_XXS codebook tables added in v0.1.2

## What It Provides

- **24 quantized block formats** (plus F16/BF16/F32 passthrough — 27 `GgufTensorType` variants in total): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, Q1_0_G128, TQ1_0, TQ2_0, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
- SIMD-accelerated paths: AVX2 (24 kernels), AVX-512 (19 kernels), ARM NEON (24 kernels), and a portable scalar fallback covering all 27 types
- Fused dequant+GEMM for Q4_0 and Q4_K on AVX2 and NEON — single-pass matmul with no intermediate f32 scratch buffer
- `matvec_q8_fused` trait method on `QuantKernel` with scalar default impl and SIMD overrides
- **The AVX-512 tier covers every fused format the AVX2 tier does.** Enabling `simd-avx512` used to *remove* the fused Q8_0-activation GEMV, because `dispatch.rs` picks AVX-512 first and no AVX-512 kernel overrode `matvec_q8_fused`/`q8_fused_acts_blocks`. Q4_0/Q5_0/Q5_1/Q8_0/Q8_1 now have native 512-bit fused kernels (`simd/avx512/{int_dot,fused}.rs`, AVX-512BW `_mm512_madd_epi16` with an AVX-512F-only fallback that is bit-identical); Q2_K/Q3_K/Q4_K/Q5_K/Q6_K delegate to their AVX2 kernels as a documented decision.
  *Verification:* the lane arithmetic is golden-tested against constants produced by executing llama.cpp's C reference (`tests/avx512_fused_goldens.rs`), and the code cross-compiles clean for `x86_64`. **Execution on AVX-512 hardware is still pending — no speedup has been measured.**
- `oxiblas` float GEMM fallback for F16/BF16/F32 tensors via `oxiblas::gemv_f32/f16`
- Parallel dequantization via `rayon` (feature-gated)
- Property-based test coverage for all kernels

## Encoders (FP32 → quantized blocks)

Decoding and encoding are separate capabilities, and this crate can decode far
more types than it can write. The **encodable** set is:

| Type | Function | Block |
|------|----------|-------|
| Q4_0 | `quantize_f32_to_q4_0` | 18 B / 32 w |
| Q5_0 | `quantize_f32_to_q5_0` | 22 B / 32 w |
| Q5_1 | `quantize_f32_to_q5_1` | 24 B / 32 w |
| Q8_0 | `quantize_f32_to_q8_0` | 34 B / 32 w |
| Q2_K | `quantize_f32_to_q2_k` | 84 B / 256 w |
| Q3_K | `quantize_f32_to_q3_k` | 110 B / 256 w |
| Q4_K | `quantize_f32_to_q4_k` | 144 B / 256 w |
| Q5_K | `quantize_f32_to_q5_k` | 176 B / 256 w |
| Q6_K | `quantize_f32_to_q6_k` | 210 B / 256 w |

`quantize_f32_row` dispatches over all of them (plus F16/F32 passthrough) and
`quantize_f32_rows` encodes a whole 2-D tensor row by row (rayon-parallel under
the `parallel` feature); `can_encode` reports whether a `GgufTensorType` is in
the set above. Everything else — the I-quants, Q4_1, Q8_1, Q1_0_G128, Q8_K — is
**decode-only**, and `quantize_f32_row` returns `QuantError::UnsupportedType`
rather than producing something plausible-looking.

K-quant rows must be a whole number of 256-weight super-blocks; a partial row
returns `QuantError::RowNotBlockAligned` rather than being zero-padded, because
padding the tail of one row would shift every row after it. Model-level callers
handle indivisible rows the way llama.cpp does, by choosing a different type for
that tensor.

*Verification:* every encoder is held **byte-for-byte** against llama.cpp's C
reference in `tests/golden_kquant_encoders.rs`. The expected bytes were produced
by compiling `quantize_row_q{2,3,4,5,6}_K_ref` (and `make_qx_quants` /
`make_qkx2_quants` / `nearest_int`) verbatim out of llama.cpp `ba7e817ee` and
running them on 21 deterministic inputs — plateaus, all-zero rows, huge
outliers, exact rounding ties, and four slices of real Meta-Llama-3-8B weights
dequantized by ggml's own decoders. Q4_0 and Q8_0 are asserted as controls, and
`half::f16` rounding is pinned against ggml's software fp32→fp16 converter.

## Key Types

| Type | Description |
|------|-------------|
| `QuantKernel` | Trait implemented by every quantization format |
| `QuantTensor` | Tensor wrapper: `SharedBytes` block data + shape + `GgufTensorType` |
| `KernelDispatcher` | Runtime dispatch to the correct kernel for a `GgufTensorType` |
| `dequantize_to_f32` | Dispatches and dequantizes a raw block buffer into an owned `Vec<f32>` |

## Usage

```rust
use oxillama_quant::{QuantResult, dequantize_to_f32};
use oxillama_gguf::GgufTensorType;

fn dequantize_tensor(raw: &[u8], dtype: GgufTensorType, n_elements: usize) -> QuantResult<Vec<f32>> {
    dequantize_to_f32(raw, dtype, n_elements)
}
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | yes | Enable Rayon-based parallel dequantization |
| `simd-avx2` | no | AVX2 256-bit SIMD kernel path |
| `simd-avx512` | no | AVX-512 512-bit SIMD kernel path. **Implies `simd-avx2`** — the AVX-512 tier is a superset: `dispatch.rs` falls through to AVX2 for the five types with no AVX-512 kernel (IQ1_S, IQ1_M, IQ2_S, IQ3_XXS, IQ4_NL), and the AVX-512 K-quant kernels delegate their fused GEMV to the AVX2 ones on purpose. |
| `simd-neon` | no | ARM NEON 128-bit SIMD kernel path |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

# oxillama-quant

Quantization kernels for all GGUF quantization types used in LLM inference.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.3 — **Tests:** 389 passing

## What's New in v0.1.3 (2026-05-05)

- **AVX-512 IQ kernels** — IQ2_XXS, IQ2_XS, IQ3_S, IQ4_XS with AVX-512BW (`_mm512_permutexvar_epi8`); 2× throughput vs AVX2. Runtime-guarded via `is_x86_feature_detected!("avx512bw")`. 8 new tests.
- **Fused `matvec_q8` for Q5_0 / Q5_1 / Q8_1** — single-pass dequant+dot in registers with no scratch allocation; AVX2 + NEON + scalar reference; 6 new tests (tol 1e-5 on 64×1024 GEMV).
- Test count: 382 → **389 tests passing**.

## What's New in v0.1.2

- IQ3_S and IQ3_XXS codebook tables added in v0.1.2

## What It Provides

- **25 quantization types**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q1_0_G128
- SIMD-accelerated paths: AVX-512, AVX2 (27 kernels), ARM NEON (27 kernels), and portable scalar fallback; 11 AVX-512 kernels
- Fused dequant+GEMM for Q4_0 and Q4_K on AVX2 and NEON — single-pass matmul with no intermediate f32 scratch buffer
- `matvec_q8_fused` trait method on `QuantKernel` with scalar default impl and SIMD overrides
- `oxiblas` float GEMM fallback for F16/BF16/F32 tensors via `oxiblas::gemv_f32/f16`
- Parallel dequantization via `rayon` (feature-gated)
- Property-based test coverage for all kernels

## Key Types

| Type | Description |
|------|-------------|
| `QuantKernel` | Trait implemented by every quantization format |
| `QuantTensor` | Owned buffer of quantized blocks + element type tag |
| `dispatch_kernel` | Runtime dispatch to the correct kernel for a `GgmlType` |
| `DequantizeSlice` | Zero-allocation dequantize into a caller-provided `f32` slice |

## Usage

```rust
use oxillama_quant::{QuantResult, dispatch_dequantize};
use oxillama_gguf::GgmlType;

fn dequantize_tensor(raw: &[u8], dtype: GgmlType, out: &mut Vec<f32>) -> QuantResult<()> {
    dispatch_dequantize(dtype, raw, out)?;
    Ok(())
}
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | yes | Enable Rayon-based parallel dequantization |
| `simd-avx2` | no | AVX2 256-bit SIMD kernel path |
| `simd-avx512` | no | AVX-512 512-bit SIMD kernel path |
| `simd-neon` | no | ARM NEON 128-bit SIMD kernel path |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)

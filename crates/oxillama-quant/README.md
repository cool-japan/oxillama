# oxillama-quant

Quantization kernels for all GGUF quantization types used in LLM inference.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- **25 quantization types**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q1_0_G128
- SIMD-accelerated paths: AVX-512, AVX2, ARM NEON, and portable scalar fallback
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

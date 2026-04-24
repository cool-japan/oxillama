# oxillama-quant — TODO

## 1. Overview

`oxillama-quant` is the numeric core of OxiLLaMa: a full-coverage, Pure Rust
quantization kernel suite that dequantizes and matrix-multiplies every tensor
format produced by the GGUF ecosystem. It lives between `oxillama-gguf`
(byte-level block parsing) and `oxillama-arch` (attention, FFN, RMSNorm),
supplying the hot-path dequant + GEMV primitives that every forward pass
depends on.

Scope in v0.1.0:

- 27 tensor types across five families:
  - Legacy scalar block formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
  - K-quants (super-block, scale-of-scales): Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
  - I-quants (lookup-grid based): IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
    IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
  - Ternary: TQ1_0, TQ2_0
  - Reference floats (passthrough): F16, BF16, F32
  - Bonsai absorption: Q1_0_G128 (1-bit signed + group-128 scales)
- Runtime SIMD dispatcher: AVX-512 → AVX2+FMA → NEON → scalar
- Feature-gated Rayon parallel GEMV for row-parallel matmul
- `QuantLinear` + optional `LoraAdapter` fused into a single kernel call
- Criterion microbenchmarks covering every shipped kernel

Design invariants: zero `unwrap()` in production paths, zero C/FFI, no
OpenBLAS / MKL / FFTW. CPU feature detection uses `scirs2-core` when a
consumer pulls it in, otherwise falls back to `std::arch::is_x86_feature_detected!`
wrapped behind `SimdCapabilities::detect` with a `OnceLock` cache.

The crate sits on a single downstream dependency — `oxillama-gguf` — for
block-layout constants and `GgufTensorType` discrimination. It carries
`half`, `thiserror`, `tracing`, and an optional `rayon` as its only runtime
dependencies, keeping the binary size and build graph of consumers
(`oxillama-arch`, `oxillama-runtime`) minimal. All kernel output is
`Vec<f32>` dequantized into the architecture-layer scratch buffer; the
activation side stays untouched by this crate except where a Q8 matmul
re-quantizes activations internally.

## 2. Status Snapshot

| Field | Value |
|---|---|
| Version | 0.1.1 |
| Completion | ~99% |
| Source files | ~67 under `src/` |
| Top-level modules | `dispatch`, `error`, `lora`, `parallel`, `quantize`, `reference`, `simd`, `traits`, `types` |
| Default features | `parallel` (Rayon) |
| Optional features | `simd-avx2`, `simd-avx512`, `simd-neon` |
| Dispatch pyramid | AVX-512F → AVX2+FMA → AArch64 NEON → scalar reference |
| Detection | `SimdCapabilities::detect()` cached via `OnceLock` in `simd::cached_capabilities()` |
| Production `unwrap()` | 0 |
| Bench harness | Criterion (`benches/quant_kernels.rs`) |

SIMD coverage matrix (v0.1.0):

| Type | Scalar | AVX2 | AVX-512 | NEON |
|---|:-:|:-:|:-:|:-:|
| Q4_0 | yes | yes | yes | yes |
| Q4_K | yes | yes | yes | yes |
| Q5_K | yes | yes | yes | yes |
| Q6_K | yes | yes | yes | yes |
| Q8_0 | yes | yes | yes | yes |
| Q1_0_G128 | yes | yes | yes | yes |
| Q2_K | yes | yes | — | — |
| Q3_K | yes | yes | — | — |
| Q4_1 | yes | — | — | — |
| Q5_0 | yes | yes | — | — |
| Q5_1 | yes | — | — | — |
| Q8_1 | yes | — | — | — |
| Q8_K | yes (dequant only) | — | — | yes |
| TQ1_0 | yes | — | — | — |
| TQ2_0 | yes | — | — | — |
| IQ1_S, IQ1_M | yes | — | — | — |
| IQ2_XXS, IQ2_XS, IQ2_S | yes | IQ2_XXS only | — | — |
| IQ3_XXS, IQ3_S | yes | IQ3_S ✅ | — | — |
| IQ4_NL, IQ4_XS | yes | IQ4_XS ✅ | — | — |
| F16, BF16, F32 | yes (passthrough) | — | — | — |

Six block formats have a full four-tier SIMD ladder (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q1_0_G128). Q2_K and Q3_K have scalar + AVX2. Q8_K has scalar + NEON. IQ2_XXS has scalar + AVX2. TQ1_0 and TQ2_0 have scalar reference kernels. Thirteen remaining formats ship with scalar-only paths. That gap is the v0.1.1 roadmap.

Feature flag behaviour:

- `parallel` (default on) — enables `rayon` and the row-parallel GEMV path
  in `parallel.rs`. Disabling it produces a single-threaded build with zero
  transitive thread-pool dependencies.
- `simd-avx2` — compiles `simd/avx2/`.
  Kernels are still runtime-guarded by `SimdCapabilities.avx2 && fma`.
- `simd-avx512` — compiles `simd/avx512/`. Runtime-guarded by AVX-512F.
- `simd-neon` — compiles `simd/neon/`. Target-arch-gated to `aarch64`;
  runtime detection is a compile-time constant (`true` under `aarch64`).

The feature flags are additive and orthogonal; a single build binary can
contain AVX-512, AVX2, and NEON kernels, selecting the appropriate tier per
host at process start via `simd::cached_capabilities()`.

## 3. Module Map

Top-level layout under `crates/oxillama-quant/src/`:

Shared infrastructure:

- `lib.rs` — crate root, module declarations, public re-exports
- `dispatch.rs` — `KernelDispatcher`, `SimdCapabilities`, runtime tier selection
- `error.rs` — `QuantError` (thiserror) + `QuantResult<T>`
- `traits.rs` — `QuantKernel` trait (dequant + matvec contract)
- `types.rs` — `QuantTensor`, `BlockInfo` descriptors
- `parallel.rs` — Rayon row-parallel GEMV (feature `parallel`)
- `lora.rs` — `LoraAdapter`, `QuantLinear` fused wrapper

Scalar reference kernels (`src/reference/`):

- Floats: `f32.rs`, `f16.rs`, `bf16.rs`
- Legacy: `q4_0.rs`, `q4_1.rs`, `q5_0.rs`, `q5_1.rs`, `q8_0.rs`, `q8_1.rs`
- K-quants: `q2_k.rs`, `q3_k.rs`, `q4_k.rs`, `q5_k.rs`, `q6_k.rs`, `q8_k.rs`
- I-quants: `iq1_s.rs`, `iq1_m.rs`, `iq2_xxs.rs`, `iq2_xs.rs`, `iq2_s.rs`,
  `iq3_xxs.rs`, `iq3_s.rs`, `iq4_nl.rs`, `iq4_xs.rs`, `iq_shared.rs`
- Bonsai: `q1_0_g128.rs`
- Ternary: `tq1_0.rs`, `tq2_0.rs`
- Lookup tables (split to honour the 2000-line policy):
  - `iq_grids.rs`
  - `iq1s_grid/mod.rs` + `iq1s_grid/data_a.rs` + `iq1s_grid/data_b.rs`
  - `iq1s_table_hi.rs`, `iq1s_table_lo.rs`
  - `iq2s_table.rs`

Platform SIMD kernels (`src/simd/`):

- `simd/mod.rs` — `cached_capabilities()` + platform gates
- `simd/avx2/` — `q4_0.rs`, `q5_0.rs`, `q4_k.rs`, `q5_k.rs`, `q6_k.rs`, `q8_0.rs`,
  `q1_0_g128.rs`, `q2_k.rs`, `q3_k.rs`, `util.rs`
- `simd/avx512/` — `q4_0.rs`, `q4_k.rs`, `q5_k.rs`, `q6_k.rs`, `q8_0.rs`, `q1_0_g128.rs`, `util.rs`
- `simd/neon/` — `q4_0.rs`, `q4_k.rs`, `q5_k.rs`, `q6_k.rs`, `q8_0.rs`, `q1_0_g128.rs`

Quantize API:

- `quantize.rs` — Quantize-on-the-fly conversion (F32/F16 → Q4_0/Q8_0, generic dequant)

No single source file exceeds the 2000-line splitrs ceiling; the largest
(`iq1s_grid/data_*.rs`) are mechanical grid tables and were pre-split at
absorption time.

## 4. Shipped in v0.1.0

Kernels and formats:

- 27 quantization types fully decoded against upstream GGUF block layouts,
  each with a `reference::*` scalar path that doubles as the correctness oracle
- `QuantKernel` trait providing uniform `dequantize_block` and `matvec_q8`
  signatures across every format
- `KernelDispatcher::new(kind, caps)` returns a `Box<dyn QuantKernel>`
  specialized for the detected tier

Runtime SIMD dispatch:

- Three-tier selection in `dispatch.rs`: AVX-512F → AVX2+FMA → NEON → scalar
- `SimdCapabilities { avx2, avx512f, fma, neon }` with per-target
  `detect_*` functions gated by `#[cfg(target_arch = …)]`
- Result cached through `simd::cached_capabilities()` (`OnceLock`), so the
  dispatcher pays the CPUID cost exactly once per process
- `best_tier()` returns a stable display string for `oxillama info`

AVX-512 kernels (`simd/avx512/`): Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q1_0_G128.
AVX2+FMA kernels (`simd/avx2/`): Q4_0, Q5_0, Q4_K, Q5_K, Q6_K, Q8_0, Q1_0_G128, Q2_K, Q3_K.
NEON kernels (`simd/neon/`): Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q1_0_G128, Q8_K.

Q8_K NEON-optimized GEMV (int8→f32 via NEON widening + vfmaq_f32 FMA).

Parallelism:

- `parallel.rs` — Rayon row-parallel GEMV, feature-gated on `parallel`
  (default on), sharded at row granularity to keep KV-cache locality intact
- Scalar fallback path remains single-threaded so builds with
  `--no-default-features` stay dependency-free

Fine-tuning support:

- `lora.rs` — `LoraAdapter { a: Vec<f32>, b: Vec<f32>, rank, alpha }`
- `QuantLinear` composes a quantized base weight with an optional LoRA field
  and fuses the `x @ (W + α·A·B)` path into a single dispatcher call

Tables and data:

- `iq_grids.rs` plus the split `iq1s_grid/` / `iq1s_table_*` / `iq2s_table.rs`
  modules carry the full IQ lookup tables, sized so every file stays under
  the 2000-line splitrs ceiling
- All grids are `const` slices — no lazy initialization, no runtime allocation

Ternary quantization:

- TQ1_0 (base-3 packed, 54 bytes/256 weights) and
  TQ2_0 (2-bit codes, 66 bytes/256 weights) with scalar reference kernels

Quantize-on-the-fly:

- `quantize_f32_to_q4_0`, `quantize_f32_to_q8_0`,
  `quantize_f16_to_q4_0`, `quantize_f16_to_q8_0`, `dequantize_to_f32`

Benchmarks and tests:

- Criterion harness `benches/quant_kernels.rs` iterating over all 27 kernels
- Singleton kernel dispatcher: `CachedDispatcher` + `global_dispatcher()` static table keyed by `GgufTensorType`, zero-allocation after first lookup per type
- Proptest suites for every scalar kernel (dequant round-trip bounds)
- SIMD kernels cross-validated byte-for-byte against the scalar reference
- Contribution to the workspace-wide total (oxillama-quant: 373 tests): all green with
  `cargo nextest run -p oxillama-quant`

## 5. Known Gaps / Incomplete

Tracked against the remaining ~1% to 100%:

- **SIMD coverage breadth:** ~2 of 27 types still run scalar-only (Q4_1, Q5_1). All IQ* types now have full AVX2+NEON coverage; TQ1_0/TQ2_0/Q5_0/Q8_K have AVX-512+NEON.
- ~~**No IQ SIMD beyond IQ2_XXS:**~~ ✅ IQ2_XS, IQ3_S, IQ4_XS, and Q4_1 AVX2 kernels now ship; IQ2_XXS was already done.
- ~~**Q8_K is dequant-only:** there is no `matvec_q8` fast path. The
  activation-side Q8_K is currently materialized via dequant→Q8_0 GEMV.~~ ✅ Fixed: Q8_K now has a true fused GEMV in scalar, AVX2, and NEON tiers.
- **No fused dequant+GEMM:** every matmul performs dequant into a scratch
  buffer and then hands off to GEMV, doubling the memory traffic of the hot
  path on large models.
- **No group-calibrated (activation-aware) quantization.** The kernels
  consume GGUF blocks as-shipped and assume calibration was done upstream.
- **No WASM or RISC-V SIMD.** Both targets currently run scalar-only.
- **No `no_std` story:** the crate assumes `std` for `OnceLock` and Rayon.

## 6. v0.1.1 Roadmap

Ordered by production impact.

- ~~**IQ2_XXS AVX2:**~~ ✅ Shipped: `dequant_block` + `gemv` AVX2 SIMD kernel
  for IQ2_XXS, registered in the dispatcher. The most common I-quant in
  Hugging Face uploads for long-context models is now off the scalar path.
- ~~**Q8_K GEMV:** promote Q8_K from dequant-only to a first-class matvec
  target, removing the Q8_0 staging buffer on the activation side.~~ ✅ Done — scalar reference, AVX2+FMA, and NEON fused kernels all ship; 278 tests pass.
- **Bench extension:** ~~Criterion scenarios for short (seq=1, decode) vs
  long (seq=512, prefill) matmul shapes across every kernel tier so
  regressions get caught by size, not just format.~~ ✅ Done — `benches/quant_shapes.rs` ships parametric (seq=1/64/512 × hidden=2048/4096) benchmarks for Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q8_K.

## 7. v0.1.2+ Vision

- ~~**Ternary SIMD acceleration:** AVX-512 VPOPCNTDQ and NEON vcntq_u8 paths
  for TQ1_0/TQ2_0, lifting them from scalar to hardware-accelerated popcount
  paths.~~ ✅ Fully shipped: TQ1_0 and TQ2_0 AVX2 kernels (previous run), plus NEON kernels (`src/simd/neon/tq1_0.rs`, `tq2_0.rs`) and AVX-512 kernels (`src/simd/avx512/tq1_0.rs`, `tq2_0.rs`) — all registered in the dispatcher. Also added AVX-512 Q5_0 (`avx512/q5_0.rs`) and Q8_K (`avx512/q8_k.rs`).
- ~~**Complete IQ SIMD matrix:**~~ ✅ All 11 IQ types have full AVX2 + NEON coverage: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS all have NEON AArch64 kernels in `src/simd/neon/`. Wired into `dispatch.rs` NEON branch.
- **Activation-aware weights:** per-group calibrated quantization where the
  group scale absorbs activation statistics (AWQ / GPTQ-style). Requires a
  calibration pass and a compatible block layout extension in `oxillama-gguf`.
- **Fused dequant + GEMM:** single-pass matmul that pulls quantized blocks
  through registers into the FMA lane without an intermediate F32 buffer.
  Removes the largest remaining memory-bandwidth tax in the runtime.
- **`simd-riscv` feature:** RVV 1.0 kernels for Q4_0, Q8_0, Q4_K, Q1_0_G128,
  matching the NEON tier. Blocked on stable `std::arch::riscv64::*` intrinsics.
- ~~**`simd-wasm` feature:** WebAssembly SIMD128 kernels for browser deploys of small models~~ ✅ Partially shipped: `.cargo/config.toml` enables `+simd128` for `wasm32-unknown-unknown` target, enabling SIMD-accelerated dequant in all modern browsers; full SIMD128 Rust kernel implementation remains v2.0+.
- **Activation quantization (A8W4):** when activations are Q8 and weights
  are Q4, the GEMM becomes compute-bound rather than memory-bound. Demands
  a new `QuantKernel::matvec_q8_activations` contract and careful rounding
  semantics.
- **Block-sparse + quantized hybrid:** composition with sparse-attention
  runtimes so MoE routers can prune inactive experts before dequant.
- **`no_std` + `alloc`-only build:** strip `OnceLock` and Rayon behind
  feature flags so the scalar kernels compile for embedded targets.
- **`oxiblas` GEMM fallback:** for tensors stored in F16 / BF16 / F32,
  route GEMM through `oxiblas` instead of the passthrough path. Moves the
  float tier from a dequant-shaped contract to a true BLAS integration.

*Last updated: 2026-04-20 (v0.1.1)*

## 8. Planned Kernels (A1–A8)

NEON and AVX2 coverage gaps identified for v0.1.1. Each item is `[~]` (planned, not yet implemented).

### AVX2 Coverage Gaps

- [x] **A1 — Q5_1 AVX2 kernel** (done 2026-04-19)
  - **Goal:** `Q5_1Avx2` kernel with full `QuantKernel` trait, registered in dispatch.rs AVX2 branch, parity with `Q5_1Ref`.
  - **Design:** Q5_1 block = 24 bytes (f16 d, f16 m, u32 qh, 16×u8 qs). Decode low-nibble via `_mm256_and_si256(0x0F)` + high-bit via qh shift-and-mask, unsigned range [0..31], FMA `_mm256_fmadd_ps(q_f32, d_v, m_v)`.
  - **Files:** `src/simd/avx2/q5_1.rs` (new), `src/simd/avx2/mod.rs`, `src/dispatch.rs`.
  - **Tests:** `avx2_dequant_matches_reference` + `avx2_matvec_q8_matches_reference`, tolerance `< 1e-6`, guarded by `is_x86_feature_detected!("avx2")`.
  - **Risk:** Q5_1 is unsigned (range [0..31]); Q5_0 is signed — don't apply Q5_0's -16 bias.

- [x] **A2 — Q8_1 AVX2 kernel** (done 2026-04-19)
  - **Goal:** `Q8_1Avx2` kernel dispatched from AVX2 branch, parity with reference.
  - **Design:** Q8_1 block = 36 bytes (f16 d, f16 s, 32×i8 qs). Read `reference/q8_1.rs::matvec_q8` first and match exactly its activation-side pairing (Q8_1 vs Q8_1). AVX2 loads 32×i8 via `_mm256_loadu_si256`, widens via `_mm256_cvtepi8_epi16`, FMA with `d_broadcast`. Apply `s` bias correction matching the reference.
  - **Files:** `src/simd/avx2/q8_1.rs` (new), `src/simd/avx2/mod.rs`, `src/dispatch.rs`.
  - **Tests:** `avx2_dequant_matches_reference` + `avx2_matvec_q8_matches_reference`, tolerance `< 1e-5`.
  - **Risk:** Must read reference first; any mis-pairing caught by parity test.

### NEON Coverage Gaps

- [x] **A3 — Q4_1 NEON kernel** (done 2026-04-19)
  - **Goal:** `Q4_1Neon` kernel dispatched from NEON branch, parity with reference.
  - **Design:** Q4_1 block = 20 bytes (f16 d, f16 m, 16×u8 qs), unsigned 4-bit. Split nibbles via `vandq_u8` + `vshrq_n_u8(4)`, widen to f32, FMA with `vfmaq_f32(m_v, q_v, d_v)`.
  - **Files:** `src/simd/neon/q4_1.rs` (new), `src/simd/neon/mod.rs`, `src/dispatch.rs`.
  - **Tests:** dequant + matvec parity, tolerance `< 1e-6`.
  - **Risk:** unsigned range [0..15]; do not apply Q4_0's -8 bias.

- [x] **A4 — Q5_0 NEON kernel** (done 2026-04-19)
  - **Goal:** `Q5_0Neon` kernel dispatched from NEON branch.
  - **Design:** Q5_0 = 22 bytes (f16 d, u32 qh, 16×u8 qs), signed 5-bit. NEON: low nibble from qs, high bit from qh via shift/mask, combine, bias -16 for signed 5-bit, multiply by d.
  - **Files:** `src/simd/neon/q5_0.rs` (new), `src/simd/neon/mod.rs`, `src/dispatch.rs`.
  - **Tests:** dequant + matvec parity.
  - **Risk:** qh bit order must match GGUF spec.

- [x] **A5 — Q5_1 NEON kernel** (done 2026-04-19)
  - **Goal:** `Q5_1Neon` kernel; combines Q4_1 affine and Q5_0 high-bit layout.
  - **Design:** Block = 24 bytes (f16 d, f16 m, u32 qh, 16×u8 qs), unsigned 5-bit. Decode like Q5_0 but no -16 bias; FMA with `vfmaq_f32(m_v, q_v, d_v)`.
  - **Files:** `src/simd/neon/q5_1.rs` (new), `src/simd/neon/mod.rs`, `src/dispatch.rs`.
  - **Tests:** standard.
  - **Risk:** unsigned range — no -16 bias.

- [x] **A6 — Q8_1 NEON kernel** (done 2026-04-19)
  - **Goal:** `Q8_1Neon` with dequant + matvec on NEON.
  - **Design:** Block = 36 bytes, signed 8-bit. Load 32×i8 as `int8x16_t × 2`, widen to i16, dequant with vcvtq_f32. For matvec: `vdotq_s32` if dotprod available, else pairwise `vmull_s8 → vpadal_s16`. Apply Q8_1 `s` correction.
  - **Files:** `src/simd/neon/q8_1.rs` (new), `src/simd/neon/mod.rs`, `src/dispatch.rs`.
  - **Tests:** standard.
  - **Risk:** Guard `vdotq_s32` with `#[cfg(target_feature = "dotprod")]`, provide fallback.

- [x] **A7 — Q2_K NEON kernel** (done 2026-04-19)
  - **Goal:** `Q2_KNeon` kernel.
  - **Design:** Q2_K super-block = 84 bytes, 256 weights. Unpack 2-bit weights via `vshrq_n_u8` + `vandq_u8`, broadcast per-sub-block scale and min, FMA-combine. Follow AVX2 Q2_K template.
  - **Files:** `src/simd/neon/q2_k.rs` (new), `src/simd/neon/mod.rs`, `src/dispatch.rs`.
  - **Tests:** 256-weight dequant parity + 64×256 matvec parity.
  - **Risk:** nibble-packed scales are fiddly — cross-reference scalar reference.

- [x] **A8 — Q3_K NEON kernel** (done 2026-04-19)
  - **Goal:** `Q3_KNeon` kernel.
  - **Design:** Q3_K = 110 bytes, 3-bit weights from 32-byte qs + 8-byte hmask, 6-bit scales. Combine 2-bit low + 1-bit high, subtract 4 for signed [-4..3]. Keep `fn unpack_scales(src: &[u8; 12]) -> [i8; 16]` as testable standalone helper.
  - **Files:** `src/simd/neon/q3_k.rs` (new), `src/simd/neon/mod.rs`, `src/dispatch.rs`.
  - **Tests:** standard + standalone `unpack_scales` unit test.
  - **Risk:** 6-bit scale unpack is error-prone.

## 9. Planned Enhancements (B1–B3)

Fused matmul paths and float GEMM integration. Each item is `[~]` (planned, not yet implemented).

- [x] **B1 — Fused dequant+GEMM for Q4_0 (AVX2 + NEON) (done 2026-04-20)**
  - **Goal:** `matvec_q8_fused` on `Q4_0Avx2` and `Q4_0Neon` — dequant + dot product in registers, no scratch f32 buffer. ~30% win on decode loops where Q4_0 is bottleneck. Byte-equal to two-pass path up to rounding.
  - **Design:** New trait method on `QuantKernel`: `fn matvec_q8_fused(&self, weights: &[u8], acts_q8: &[BlockQ8_0], out: &mut [f32]) -> QuantResult<()>;`. Default impl: dequant-into-scratch then matvec_q8. AVX2: load nibbles into `__m256i`, decode to signed 4-bit in i32 lanes, load Q8_0 as i8×32→i32 widening, accumulate `d_w * d_a * dot_q(i32_w, i32_a)`, horizontal-sum at end. NEON: `vld1q_s8 → vmull_s8 → vpaddlq_s16 → vaddvq_s32`, FMA with `d_w * d_a`.
  - **Files:** `src/simd/avx2/q4_0.rs` (extend); `src/simd/neon/q4_0.rs` (extend); `src/kernel.rs` (extend trait with default impl); `src/reference/q4_0.rs` (scalar reference as parity oracle).
  - **Prerequisites:** existing Q4_0 AVX2/NEON kernels.
  - **Tests:** (a) `avx2_fused_matches_reference` — 64×1024 GEMV, tol 1e-5; (b) `neon_fused_matches_reference` — same; (c) `fused_matches_unfused` — both SIMD variants agree with old two-pass path, tol 1e-5.
  - **Risk:** NEON pairwise adds saturating if widened incorrectly — use vpaddlq_s16→vaddvq_s32, not saturating adds.

- [x] **B2 — Fused dequant+GEMM for Q4_K (AVX2 + NEON) (done 2026-04-20)**
  - **Goal:** Same as B1 for Q4_K. Most-used K-quant in modern GGUFs; fused path is the largest remaining memory-bandwidth win on CPU.
  - **Design:** Q4_K super-block = 144 bytes, 256 4-bit weights with 6-bit scale + 6-bit min per 16-weight sub-block. AVX2: broadcast per-sub-block `(d * scale_i)` and `(dmin * min_i)` as f32 regs; decode 16 4-bit weights into f32 lanes; FMA into accumulator. NEON: `vfmaq_f32(acc, (d·scale - dmin·min) broadcast, q_f32)`. Reuse existing `unpack_q4_k_scales` helper.
  - **Files:** `src/simd/avx2/q4_k.rs` (extend); `src/simd/neon/q4_k.rs` (extend); `src/reference/q4_k.rs` (scalar fused reference).
  - **Prerequisites:** existing Q4_K AVX2/NEON kernels.
  - **Tests:** parity tests at tol 1e-5 on 64×1024 GEMV.
  - **Risk:** Q4_K scale/min unpacking error-prone — factor the decode helper, unit-test standalone, reuse in fused path.

- [x] **B3 — oxiblas GEMM fallback for F16/BF16/F32 (done 2026-04-20)**
  - **Goal:** Route F16/BF16/F32 tensor matmuls through `oxiblas::gemm` instead of passthrough scalar path. Workspace sovereignty posture (oxiblas already a workspace dep).
  - **Design:** New `QuantKernel` impls `F16Kernel`, `Bf16Kernel`, `F32Kernel` with `matvec_q8_fused` overrides calling `oxiblas::gemv_f32` / `oxiblas::gemv_f16`. Dispatch wiring in `dispatch.rs`. Fallback: bubble `QuantError::FloatGemmFailed`.
  - **Files:** `src/simd/float_gemm.rs` (new, ~200 LoC); `src/dispatch.rs` (route F16/BF16/F32); `Cargo.toml` (verify `oxiblas = { workspace = true }`, add if missing).
  - **Prerequisites:** oxiblas in workspace deps (confirmed in root Cargo.toml line 47).
  - **Tests:** (a) `f32_gemv_matches_scalar_reference` tol 1e-6; (b) `f16_gemv_matches_scalar_reference` tol 1e-3; (c) `bf16_gemv_matches_scalar_reference` tol 1e-2.
  - **Risk:** oxiblas row/column major convention mismatch — verify before declaring done. If oxiblas lacks a feature, flag as deviated and fall back to scalar.

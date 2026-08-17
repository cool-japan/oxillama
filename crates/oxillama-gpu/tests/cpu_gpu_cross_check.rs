//! CPU-vs-GPU cross-check: every quant type the GPU dispatcher supports must
//! agree with the upstream-pinned CPU reference kernel on real GPU hardware.
//!
//! # Why this test exists
//!
//! Every `test_gpu_gemv_*_matches_cpu*` test that already existed in this
//! crate computed its "expected" value by calling the *same* CPU-side
//! `dequant_*_to_f32` helper the kernel itself calls before uploading to the
//! GPU.  That makes those tests self-consistent, not correct: a systematic
//! layout bug present in the helper is invisible to a test that uses the
//! same helper on both sides of the comparison.  That is exactly the class
//! of bug this pass fixed (see `src/kernels/golden_tests.rs`).
//!
//! This test closes that gap by using `oxillama_quant::reference` — an
//! independent implementation pinned to upstream GGML, maintained in a
//! different crate — as the oracle, and dispatching the *actual* GPU
//! compute pipeline (not a CPU-side stand-in) via [`oxillama_gpu::GpuDispatcher`].
//!
//! # Hardware requirement
//!
//! If no GPU adapter is available (`GpuDispatcher::has_gpu() == false`), this
//! test prints a message and returns without asserting anything — it cannot
//! claim to have exercised GPU execution it did not get.  Whoever runs this
//! locally on an environment with a Vulkan/Metal/DX12 adapter (this pass was
//! verified against an Apple M3 / Metal) gets the full cross-check.

use oxillama_gguf::GgufTensorType;
use oxillama_gpu::GpuDispatcher;
use oxillama_quant::reference::{
    Iq1MRef, Iq1SRef, Iq2SRef, Iq2XsRef, Iq2XxsRef, Iq3SRef, Iq3XxsRef, Iq4NlRef, Iq4XsRef,
    Q1_0G128Ref, Q2KRef, Q3KRef, Q4KRef, Q4_0Ref, Q4_1Ref, Q5KRef, Q5_0Ref, Q5_1Ref, Q6KRef,
    Q8KRef, Q8_0Ref, Q8_1Ref, Tq1_0Ref, Tq2_0Ref,
};
use oxillama_quant::{QuantKernel, QuantTensor};

/// A small, dependency-free xorshift64 PRNG — deterministic across runs so a
/// failure is reproducible without needing to capture the seed.
struct Xorshift64(u64);

impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn fill_bytes(&mut self, buf: &mut [u8]) {
        let mut i = 0;
        while i < buf.len() {
            let word = self.next_u64().to_le_bytes();
            let n = (buf.len() - i).min(8);
            buf[i..i + n].copy_from_slice(&word[..n]);
            i += n;
        }
    }
}

/// One case: a dispatcher-supported quant type paired with its
/// upstream-pinned CPU reference kernel.
struct Case {
    ty: GgufTensorType,
    name: &'static str,
    cpu: Box<dyn QuantKernel>,
}

/// Every type `GpuDispatcher::get_kernel` supports (see
/// `crates/oxillama-gpu/src/lib.rs`'s `match tensor_type` arm), paired with
/// the reference kernel from `oxillama-quant` — a different crate, so this
/// is a genuine cross-check rather than a self-comparison.
fn all_cases() -> Vec<Case> {
    vec![
        Case {
            ty: GgufTensorType::Q2K,
            name: "Q2K",
            cpu: Box::new(Q2KRef),
        },
        Case {
            ty: GgufTensorType::Q3K,
            name: "Q3K",
            cpu: Box::new(Q3KRef),
        },
        Case {
            ty: GgufTensorType::Q4_0,
            name: "Q4_0",
            cpu: Box::new(Q4_0Ref),
        },
        Case {
            ty: GgufTensorType::Q4_1,
            name: "Q4_1",
            cpu: Box::new(Q4_1Ref),
        },
        Case {
            ty: GgufTensorType::Q4K,
            name: "Q4K",
            cpu: Box::new(Q4KRef),
        },
        Case {
            ty: GgufTensorType::Q5_0,
            name: "Q5_0",
            cpu: Box::new(Q5_0Ref),
        },
        Case {
            ty: GgufTensorType::Q5_1,
            name: "Q5_1",
            cpu: Box::new(Q5_1Ref),
        },
        Case {
            ty: GgufTensorType::Q5K,
            name: "Q5K",
            cpu: Box::new(Q5KRef),
        },
        Case {
            ty: GgufTensorType::Q6K,
            name: "Q6K",
            cpu: Box::new(Q6KRef),
        },
        Case {
            ty: GgufTensorType::Q8_0,
            name: "Q8_0",
            cpu: Box::new(Q8_0Ref),
        },
        Case {
            ty: GgufTensorType::Q8_1,
            name: "Q8_1",
            cpu: Box::new(Q8_1Ref),
        },
        Case {
            ty: GgufTensorType::Q8K,
            name: "Q8K",
            cpu: Box::new(Q8KRef),
        },
        Case {
            ty: GgufTensorType::Q1_0G128,
            name: "Q1_0G128",
            cpu: Box::new(Q1_0G128Ref),
        },
        Case {
            ty: GgufTensorType::Iq4Xs,
            name: "Iq4Xs",
            cpu: Box::new(Iq4XsRef),
        },
        Case {
            ty: GgufTensorType::Iq2Xxs,
            name: "Iq2Xxs",
            cpu: Box::new(Iq2XxsRef),
        },
        Case {
            ty: GgufTensorType::Iq2S,
            name: "Iq2S",
            cpu: Box::new(Iq2SRef),
        },
        Case {
            ty: GgufTensorType::Iq2Xs,
            name: "Iq2Xs",
            cpu: Box::new(Iq2XsRef),
        },
        Case {
            ty: GgufTensorType::Iq3Xxs,
            name: "Iq3Xxs",
            cpu: Box::new(Iq3XxsRef),
        },
        Case {
            ty: GgufTensorType::Iq3S,
            name: "Iq3S",
            cpu: Box::new(Iq3SRef),
        },
        Case {
            ty: GgufTensorType::Iq1S,
            name: "Iq1S",
            cpu: Box::new(Iq1SRef),
        },
        Case {
            ty: GgufTensorType::Iq1M,
            name: "Iq1M",
            cpu: Box::new(Iq1MRef),
        },
        Case {
            ty: GgufTensorType::Iq4Nl,
            name: "Iq4Nl",
            cpu: Box::new(Iq4NlRef),
        },
        Case {
            ty: GgufTensorType::Tq1_0,
            name: "Tq1_0",
            cpu: Box::new(Tq1_0Ref),
        },
        Case {
            ty: GgufTensorType::Tq2_0,
            name: "Tq2_0",
            cpu: Box::new(Tq2_0Ref),
        },
    ]
}

/// Rows of quantised data to generate per type: enough to exercise more than
/// one block per row is unnecessary here (each row is exactly one block —
/// `cols == block_size()`), but multiple *rows* exercise the row-stride
/// arithmetic on both sides.
const ROWS: usize = 4;

#[test]
fn cpu_gpu_cross_check_all_dispatcher_types() {
    let gpu = GpuDispatcher::new();
    if !gpu.has_gpu() {
        eprintln!(
            "cpu_gpu_cross_check: no GPU adapter available in this environment — \
             skipping real-hardware cross-check. This test asserts nothing when \
             skipped; it is not evidence of correctness on its own."
        );
        return;
    }
    let ctx = gpu
        .context()
        .expect("has_gpu() true implies context() is Some");

    let mut rng = Xorshift64(0xC0FF_EE15_5EED_1234);
    let mut failures = Vec::new();
    let mut checked = Vec::new();

    for case in all_cases() {
        let block_bytes = case.cpu.block_bytes();
        let block_size = case.cpu.block_size();
        let cols = block_size; // one block per row
        let mut weight_bytes = vec![0u8; ROWS * block_bytes];
        rng.fill_bytes(&mut weight_bytes);

        // Ramp-like input, scaled down so dot products stay well within f32
        // range for every format (Q6_K/Q8_K weights can be large).
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 / cols as f32) - 0.5).collect();

        let tensor = QuantTensor::new(weight_bytes.clone(), vec![ROWS, cols], case.ty);
        let mut cpu_out = vec![0.0f32; ROWS];
        case.cpu
            .gemv(&tensor, &input, &mut cpu_out)
            .unwrap_or_else(|e| panic!("{}: CPU reference gemv failed: {e}", case.name));

        let gpu_kernel = gpu.get_kernel(case.ty).unwrap_or_else(|| {
            panic!(
                "{}: has_gpu() is true but GpuDispatcher::get_kernel returned None \
                 for a type the dispatcher's own match arm claims to support",
                case.name
            )
        });
        let mut gpu_out = vec![0.0f32; ROWS];
        gpu_kernel
            .gemv(ctx, &weight_bytes, &input, &mut gpu_out, ROWS, cols)
            .unwrap_or_else(|e| panic!("{}: GPU gemv failed: {e}", case.name));

        checked.push(case.name);

        for row in 0..ROWS {
            let want = cpu_out[row];
            let got = gpu_out[row];
            let scale = want.abs().max(1.0);
            let err = (got - want).abs() / scale;
            if err > 1e-3 {
                failures.push(format!(
                    "{} row {row}: gpu={got}, cpu_reference={want} (rel err {err})",
                    case.name
                ));
            }
        }
    }

    eprintln!(
        "cpu_gpu_cross_check: verified {} types on real GPU hardware: {:?}",
        checked.len(),
        checked
    );

    assert!(
        failures.is_empty(),
        "CPU-vs-GPU cross-check failures ({} of {} types had at least one mismatching row):\n{}",
        failures.len(),
        all_cases().len(),
        failures.join("\n")
    );
}

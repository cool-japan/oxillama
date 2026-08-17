//! Bit-exactness of the K-quant encoders against llama.cpp's C reference.
//!
//! ## Where the expected values come from
//!
//! Every byte asserted here was produced by **executing llama.cpp's own C
//! code**, not by this crate. The generator harness copies
//! `nearest_int`, `make_qx_quants`, `make_q3_quants`, `make_qkx2_quants`,
//! `get_scale_min_k4`, `quantize_row_q{2,3,4,5,6}_K_ref`,
//! `quantize_row_q{4_0,5_0,5_1,8_0}_ref`, `dequantize_row_q{4,5,6}_K` and the
//! `ggml_compute_fp{16_to_32,32_to_16}` converters *verbatim* out of
//! llama.cpp commit `ba7e817ee` (`ggml/src/ggml-quants.c`,
//! `ggml/src/ggml-impl.h`, `ggml/src/ggml-common.h`), runs them on
//! deterministic inputs, and dumps inputs and outputs as hex into
//! `tests/data/kquant_golden.txt`.
//!
//! Build flags matter and are recorded in the fixture's header:
//!
//! ```text
//! clang -O2 -std=c11 -DNDEBUG -ffp-contract=off -fno-fast-math oracle_kquant.c -lm
//! ```
//!
//! `-ffp-contract=off` is not cosmetic. With clang's default contraction on
//! aarch64 the inner accumulator updates `sumlx += w*x[i]*l` become `fmadd`,
//! which rustc never emits; the fused build disagrees with the unfused one on
//! 38 of the 84 Q2_K bytes of the `all_positive` case. The unfused build is
//! the correct target for a Rust port, and `-O0` and `-O2` agree with each
//! other byte-for-byte once contraction is off.
//!
//! ## Why this file exists
//!
//! Comparing an encoder against this crate's own decoder proves nothing: a
//! self-consistent but wrong layout round-trips perfectly. Seven quantization
//! formats shipped broken in this repository while its test suite was green,
//! for exactly that reason. So:
//!
//! * the assertions are on **bytes**, not on reconstruction error;
//! * `Q4_0` and `Q8_0` — encoders that predate this work — are asserted as
//!   **controls**: if a control fails, the harness is wrong, not the new code;
//! * `half::f16::from_f32` is pinned against ggml's software converter, so a
//!   divergence in the fp16 rounding cannot hide inside a passing encoder test;
//! * inputs include four slices of genuine Meta-Llama-3-8B-Instruct weights,
//!   dequantized by ggml's own `dequantize_row_q4_K` / `dequantize_row_q6_K`.

use std::collections::HashMap;

use oxillama_quant::{
    dequantize_to_f32, quantize_f32_to_q2_k, quantize_f32_to_q3_k, quantize_f32_to_q4_0,
    quantize_f32_to_q4_k, quantize_f32_to_q5_0, quantize_f32_to_q5_1, quantize_f32_to_q5_k,
    quantize_f32_to_q6_k, quantize_f32_to_q8_0,
};

const GOLDEN: &str = include_str!("data/kquant_golden.txt");

/// One record from the fixture.
struct Case {
    name: String,
    n: usize,
    input: Vec<f32>,
    fields: HashMap<String, Vec<u8>>,
}

impl Case {
    fn bytes(&self, tag: &str) -> &[u8] {
        self.fields
            .get(tag)
            .unwrap_or_else(|| panic!("case {}: missing field {tag}", self.name))
    }

    fn f32s(&self, tag: &str) -> Vec<f32> {
        let raw = self.bytes(tag);
        assert_eq!(
            raw.len() % 4,
            0,
            "case {}: {tag} is not f32-sized",
            self.name
        );
        raw.chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_be_bytes([c[0], c[1], c[2], c[3]])))
            .collect()
    }
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    assert_eq!(s.len() % 2, 0, "odd-length hex payload");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("valid hex"))
        .collect()
}

fn parse_golden() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut cur: Option<Case> = None;
    for line in GOLDEN.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("CASE ") {
            let mut it = rest.split_whitespace();
            let name = it.next().expect("case name").to_string();
            let n = it
                .next()
                .expect("case length")
                .parse::<usize>()
                .expect("numeric length");
            cur = Some(Case {
                name,
                n,
                input: Vec::new(),
                fields: HashMap::new(),
            });
            continue;
        }
        if line == "END" {
            cases.push(cur.take().expect("END without CASE"));
            continue;
        }
        let case = cur.as_mut().expect("payload outside a CASE");
        let (tag, payload) = match line.split_once(' ') {
            Some((t, p)) => (t, p),
            // A field with no bytes (e.g. Q4K for a sub-superblock case).
            None => (line, ""),
        };
        let raw = hex_to_bytes(payload);
        if tag == "IN" {
            assert_eq!(raw.len(), case.n * 4, "case {}: IN length", case.name);
            case.input = raw
                .chunks_exact(4)
                .map(|c| f32::from_bits(u32::from_be_bytes([c[0], c[1], c[2], c[3]])))
                .collect();
        }
        case.fields.insert(tag.to_string(), raw);
    }
    assert!(cur.is_none(), "fixture ends inside a CASE");
    assert!(!cases.is_empty(), "fixture parsed to zero cases");
    cases
}

/// Report the first differing byte with enough context to debug it.
fn assert_bytes_eq(case: &str, tag: &str, got: &[u8], want: &[u8]) {
    if got == want {
        return;
    }
    assert_eq!(
        got.len(),
        want.len(),
        "case {case} / {tag}: length {} != golden {}",
        got.len(),
        want.len()
    );
    let idx = got
        .iter()
        .zip(want.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(0);
    let lo = idx.saturating_sub(8);
    let hi = (idx + 8).min(got.len());
    panic!(
        "case {case} / {tag}: first mismatch at byte {idx}: got 0x{:02x}, llama.cpp 0x{:02x}\n  \
         got   [{lo}..{hi}] = {:02x?}\n  golden[{lo}..{hi}] = {:02x?}",
        got[idx],
        want[idx],
        &got[lo..hi],
        &want[lo..hi]
    );
}

// ── Controls ─────────────────────────────────────────────────────────────
//
// These two encoders predate this work and are known to match llama.cpp. If
// they fail, the oracle harness (input generation, hex plumbing, byte order)
// is broken and no conclusion may be drawn from the K-quant assertions.

#[test]
fn control_q4_0_matches_llama_cpp() {
    for case in parse_golden() {
        if case.n == 0 || !case.n.is_multiple_of(32) {
            continue;
        }
        let got = quantize_f32_to_q4_0(&case.input).expect("q4_0 encode");
        assert_bytes_eq(&case.name, "Q40", &got, case.bytes("Q40"));
    }
}

#[test]
fn control_q8_0_matches_llama_cpp() {
    for case in parse_golden() {
        if case.n == 0 || !case.n.is_multiple_of(32) {
            continue;
        }
        let got = quantize_f32_to_q8_0(&case.input).expect("q8_0 encode");
        assert_bytes_eq(&case.name, "Q80", &got, case.bytes("Q80"));
    }
}

/// `half::f16` must round exactly like ggml's software fp32→fp16 converter.
///
/// Every K-quant super-scale goes through this conversion, and the encoders'
/// *second* pass reads the rounded value back, so a one-ULP difference here
/// would change quantized codes, not just the stored scale.
#[test]
fn control_f16_rounding_matches_ggml() {
    let cases = parse_golden();
    let sweep = cases
        .iter()
        .find(|c| c.name == "fp16_sweep")
        .expect("fp16_sweep case present");
    let inputs = sweep.f32s("F16IN");
    let expected = sweep.bytes("F16OUT");
    assert_eq!(expected.len(), inputs.len() * 2);
    let back = sweep.f32s("F16BACK");
    for (i, &v) in inputs.iter().enumerate() {
        let want = u16::from_be_bytes([expected[2 * i], expected[2 * i + 1]]);
        let got = half::f16::from_f32(v).to_bits();
        assert_eq!(
            got,
            want,
            "fp16 rounding of {v:e} (bits {:08x}): got {got:04x}, ggml {want:04x}",
            v.to_bits()
        );
        let got_back = half::f16::from_bits(got).to_f32();
        assert_eq!(
            got_back.to_bits(),
            back[i].to_bits(),
            "fp16 -> f32 of {v:e}: got {got_back:e}, ggml {:e}",
            back[i]
        );
    }
}

// ── K-quant encoders ─────────────────────────────────────────────────────

macro_rules! golden_encoder_test {
    ($test:ident, $encode:path, $tag:literal) => {
        #[test]
        fn $test() {
            let mut checked = 0usize;
            for case in parse_golden() {
                if case.n == 0 || !case.n.is_multiple_of(256) {
                    continue;
                }
                let got = $encode(&case.input).expect(concat!($tag, " encode"));
                assert_bytes_eq(&case.name, $tag, &got, case.bytes($tag));
                checked += 1;
            }
            assert!(checked >= 19, "expected >= 19 golden cases, ran {checked}");
        }
    };
}

golden_encoder_test!(q4_k_matches_llama_cpp, quantize_f32_to_q4_k, "Q4K");
golden_encoder_test!(q5_k_matches_llama_cpp, quantize_f32_to_q5_k, "Q5K");
golden_encoder_test!(q6_k_matches_llama_cpp, quantize_f32_to_q6_k, "Q6K");
golden_encoder_test!(q2_k_matches_llama_cpp, quantize_f32_to_q2_k, "Q2K");
golden_encoder_test!(q3_k_matches_llama_cpp, quantize_f32_to_q3_k, "Q3K");

// ── Legacy fallback encoders ─────────────────────────────────────────────
//
// llama.cpp downgrades K-quants to these when a tensor's row length is not a
// multiple of 256, so the re-quantization pipeline can emit them.

#[test]
fn q5_0_matches_llama_cpp() {
    for case in parse_golden() {
        if case.n == 0 || !case.n.is_multiple_of(32) {
            continue;
        }
        let got = quantize_f32_to_q5_0(&case.input).expect("q5_0 encode");
        assert_bytes_eq(&case.name, "Q50", &got, case.bytes("Q50"));
    }
}

#[test]
fn q5_1_matches_llama_cpp() {
    for case in parse_golden() {
        if case.n == 0 || !case.n.is_multiple_of(32) {
            continue;
        }
        let got = quantize_f32_to_q5_1(&case.input).expect("q5_1 encode");
        assert_bytes_eq(&case.name, "Q51", &got, case.bytes("Q51"));
    }
}

// ── Round-trip: this crate's decoders vs ggml's ──────────────────────────

/// Encoding with the new encoders and decoding with this crate's kernels must
/// reproduce, bit for bit, what ggml's `dequantize_row_*_K` produces from the
/// same blocks. This closes the loop: byte-exact encode plus byte-exact decode
/// means the reconstruction error is byte-exactly llama.cpp's too.
#[test]
fn round_trip_matches_ggml_dequantization() {
    use oxillama_gguf::GgufTensorType;

    let mut checked = 0usize;
    for case in parse_golden() {
        if case.n == 0 || !case.n.is_multiple_of(256) {
            continue;
        }
        for (tag, encode, ty) in [
            (
                "RT_Q4K",
                quantize_f32_to_q4_k as fn(&[f32]) -> oxillama_quant::QuantResult<Vec<u8>>,
                GgufTensorType::Q4K,
            ),
            (
                "RT_Q5K",
                quantize_f32_to_q5_k as fn(&[f32]) -> oxillama_quant::QuantResult<Vec<u8>>,
                GgufTensorType::Q5K,
            ),
            (
                "RT_Q6K",
                quantize_f32_to_q6_k as fn(&[f32]) -> oxillama_quant::QuantResult<Vec<u8>>,
                GgufTensorType::Q6K,
            ),
        ] {
            let blocks = encode(&case.input).expect("encode");
            let got = dequantize_to_f32(&blocks, ty, case.n).expect("decode");
            let want = case.f32s(tag);
            assert_eq!(got.len(), want.len(), "case {} / {tag}: length", case.name);
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "case {}: {tag} element {i}: got {g:e} ({:08x}), ggml {w:e} ({:08x})",
                    case.name,
                    g.to_bits(),
                    w.to_bits()
                );
            }
            checked += 1;
        }
    }
    assert!(checked >= 57, "expected >= 57 round trips, ran {checked}");
}

/// The mixture of shapes actually exercised by the fixture, asserted so a
/// truncated or mis-generated fixture cannot silently weaken every test above.
#[test]
fn fixture_covers_the_intended_shapes() {
    let cases = parse_golden();
    let names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
    for expected in [
        "lcg_mixed",           // mixed signs and magnitudes, 2 super-blocks
        "plateau_pos",         // max == min: make_qkx2_quants' degenerate branch
        "all_zero",            // amax < GROUP_MAX_EPS everywhere
        "outlier",             // one huge value among near-zeros
        "near_group_eps",      // magnitudes straddling 1e-15
        "all_positive",        // min > 0 forced to 0
        "all_negative",        // scales derived from a negative extreme
        "two_level",           // exact +-1 plateau, ties in nearest_int
        "ramp",                // linear ramp: many exact .5 ties
        "fp16_edge",           // super-scale near the fp16 maximum
        "tiny",                // 1e-7 magnitudes
        "sparse",              // zero sub-blocks -> the `if (!d) continue` path
        "real_ffn_gate_q4k",   // genuine Llama-3 weights
        "real_token_embd_q4k", // genuine Llama-3 weights
        "real_ffn_down_q6k",   // genuine Llama-3 weights
        "real_output_q6k",     // genuine Llama-3 weights
        "q6k_tie_sensitive",   // the one input whose Q6_K bytes need ties-to-even
        "half_integer_neg",    // exact .5 products: ties-to-even vs away-from-zero
        "half_integer_pos",    // same, positive extreme
        "legacy_96",           // 96 elements: legacy encoders only
        "legacy_32",           // one 32-weight block
        "fp16_sweep",
    ] {
        assert!(names.contains(&expected), "fixture is missing {expected}");
    }
    assert_eq!(names.len(), 22, "unexpected fixture case count");
}

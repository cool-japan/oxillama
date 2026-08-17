// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Whole-model logit parity against llama.cpp, through the real binary.
//!
//! # Where the golden values come from
//!
//! Every constant in [`CASES`] was produced by **executing llama.cpp**, not by
//! any OxiLLaMa code — a parity test against this repository's own reference
//! proves nothing.
//!
//! * upstream commit `ba7e817eecea53595fa9ee5ff4dce2d7febbd26b` (ggml 0.9.7),
//!   checkout `~/work/refs/llama.cpp`, unmodified;
//! * built CPU-only with **every matmul fast path disabled** —
//!   `-DGGML_METAL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
//!   -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF -DGGML_LLAMAFILE=OFF
//!   -DGGML_CPU_REPACK=OFF -DGGML_OPENMP=OFF` — so llama.cpp runs its plain
//!   `ggml_vec_dot_q4_K_q8_K` / `ggml_vec_dot_q6_K_q8_K` kernels;
//! * a driver linked against `libllama` dumped the full float32 logit vector
//!   at each of 8 greedy decode steps; `greedy` there is the argmax of the
//!   exact bytes written to disk, lowest index winning ties;
//! * runtime: `n_ctx = n_batch = n_ubatch = 512`, `n_threads = 8`,
//!   `n_gpu_layers = 0`, flash attention explicitly disabled,
//!   `type_k = type_v = GGML_TYPE_F16`, `llama_tokenize(add_special = true,
//!   parse_special = false)`, no chat template.
//!
//! `MARGINS` is the reference's own top1−top2 gap at each step, from the same
//! dumps.
//!
//! # Why the gate is margin-conditioned rather than "all 8 tokens must match"
//!
//! Two llama.cpp builds of that same commit — the one above versus llama.cpp's
//! *default* CPU build (`q4_K_8x8`/`q6_K_8x8` repacking + tinyBLAS +
//! Accelerate) — differ by up to **0.626 logits** on these very prompts, and
//! disagree on 1 greedy token out of 29 comparable steps: `qwen3/p2` step 4,
//! where the margin is only 0.1246. An exact-token gate therefore fails
//! llama.cpp for not being llama.cpp. So this test **requires** agreement at
//! every step whose reference margin exceeds that measured 0.626-logit spread,
//! and merely reports the sub-margin steps.
//!
//! # Why this is skipped by default
//!
//! It needs multi-gigabyte GGUF files that cannot live in the repository.
//! Point the env vars at them to run it:
//!
//! ```text
//! OXILLAMA_PARITY_LLAMA3_GGUF=/path/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
//! OXILLAMA_PARITY_QWEN3_GGUF=/path/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
//!   cargo nextest run -p oxillama-cli llamacpp_logit_parity
//! ```
//!
//! Expected file sha256:
//! `ab9e4eec7e80892fd78f74d9a15d0299f1e22121cea44efd68a7a02a3fe9a1da` (llama3),
//! `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` (qwen3).

use std::path::PathBuf;
use std::process::Command;

/// The measured spread between two llama.cpp builds of the same commit on the
/// same weights and prompts (max absolute logit difference over 29 comparable
/// steps). A reference step whose top1−top2 margin is below this is a coin
/// flip that llama.cpp itself does not resolve consistently.
const LLAMACPP_CROSS_KERNEL_SPREAD: f32 = 0.626;

/// One (model, prompt) reference record.
struct Case {
    /// Env var naming the GGUF file.
    env: &'static str,
    /// Human-readable id used in assertion messages.
    id: &'static str,
    /// `vocab_size`, i.e. the expected `4 * n` file size divisor.
    vocab_size: usize,
    /// Prompt token ids exactly as llama.cpp tokenized them (no BOS: both
    /// GGUFs resolve to `add_bos = false` under llama.cpp).
    prompt: &'static [u32],
    /// The 8 greedy tokens llama.cpp emitted.
    greedy: &'static [u32],
    /// Reference top1−top2 margin at each step.
    margins: &'static [f32],
}

/// Reference records — see the module docs for provenance.
const CASES: &[Case] = &[
    Case {
        env: "OXILLAMA_PARITY_LLAMA3_GGUF",
        id: "llama3-8b-instruct-q4km/p1",
        vocab_size: 128_256,
        prompt: &[791, 6864, 315, 9822, 374],
        greedy: &[12366, 627, 791, 6864, 315, 279, 3723, 4273],
        margins: &[
            3.7382, 0.3951, 2.7757, 1.0413, 7.0401, 0.6494, 3.8394, 1.5883,
        ],
    },
    Case {
        env: "OXILLAMA_PARITY_LLAMA3_GGUF",
        id: "llama3-8b-instruct-q4km/p2",
        vocab_size: 128_256,
        prompt: &[
            791, 3925, 315, 25213, 374, 264, 3446, 315, 24981, 59851, 13, 23591, 12933, 1051,
            56168, 555, 79722, 6322, 7106, 46121, 11, 1243, 555, 62018, 7563, 11, 323, 3010, 555,
            36396, 14956, 15823, 13, 9062, 502, 6324, 31894, 279, 3649, 315, 279, 832, 3770, 433,
            11, 20806, 25175, 2944, 922, 8294, 5435, 1418, 279, 12035, 14264, 10819, 30456, 13,
        ],
        greedy: &[578, 10205, 315, 1579, 11852, 15823, 1093, 356],
        margins: &[
            0.1467, 0.191, 8.9348, 2.2329, 8.5061, 1.7548, 0.8217, 0.7386,
        ],
    },
    Case {
        env: "OXILLAMA_PARITY_QWEN3_GGUF",
        id: "qwen3-4b-instruct-2507-q4km/p1",
        vocab_size: 151_936,
        prompt: &[785, 6722, 315, 9625, 374],
        greedy: &[12095, 13, 576, 6722, 315, 9856, 374, 19846],
        margins: &[2.742, 1.0286, 0.2288, 4.2119, 7.5393, 1.177, 9.3932, 6.6142],
    },
    Case {
        env: "OXILLAMA_PARITY_QWEN3_GGUF",
        id: "qwen3-4b-instruct-2507-q4km/p2",
        vocab_size: 151_936,
        prompt: &[
            785, 3840, 315, 24231, 374, 264, 3364, 315, 24020, 58751, 13, 22752, 12645, 1033,
            55068, 553, 78622, 6191, 6961, 45021, 11, 1221, 553, 60918, 7411, 11, 323, 2937, 553,
            35296, 14614, 15459, 13, 8886, 501, 6193, 30794, 279, 3565, 315, 279, 825, 3685, 432,
            11, 20194, 24198, 2874, 911, 8131, 5322, 1393, 279, 11773, 13938, 10596, 29356, 13,
        ],
        greedy: &[1096, 58751, 702, 1012, 773, 6849, 429, 582],
        margins: &[
            0.9993, 0.8455, 1.512, 0.6981, 0.1246, 1.5998, 3.4596, 0.2096,
        ],
    },
];

/// Locate the compiled `oxillama` binary next to this test executable.
fn oxillama_bin() -> PathBuf {
    let mut path = std::env::current_exe()
        .expect("current_exe")
        .parent()
        .expect("parent")
        .to_path_buf();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join("oxillama")
}

/// Comma-separate token ids for `--prompt-tokens` / `--force-tokens`.
fn csv(ids: &[u32]) -> String {
    ids.iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Argmax with the lowest index winning ties, matching llama.cpp's greedy
/// sampler and OxiLLaMa's own `--temp 0` shortcut.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = index;
        }
    }
    best
}

/// Read a `step<k>.logits.f32.bin` produced by `--dump-logits`.
fn read_logits(path: &std::path::Path, vocab_size: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    assert_eq!(
        bytes.len(),
        vocab_size * 4,
        "{} must be exactly 4 * vocab_size bytes",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
        .collect()
}

/// Run one reference case and return `(decisive_mismatches, near_tie_mismatches)`.
fn run_case(case: &Case, model: &str) -> (Vec<String>, Vec<String>) {
    let out_dir = std::env::temp_dir().join(format!(
        "oxillama_parity_{}_{}",
        case.id.replace('/', "_"),
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&out_dir);

    let output = Command::new(oxillama_bin())
        .args([
            "run",
            "--model",
            model,
            "--dump-logits",
            &out_dir.to_string_lossy(),
            "--prompt-tokens",
            &csv(case.prompt),
            // Teacher forcing: every step is conditioned on llama.cpp's own
            // prefix, so one divergence cannot invalidate the later steps.
            "--force-tokens",
            &csv(case.greedy),
            "--max-tokens",
            "8",
            "--ctx-size",
            "512",
            // Greedy, with the two sampler stages that run *before* the greedy
            // shortcut neutralised.
            "--temp",
            "0",
            "--repeat-penalty",
            "1.0",
            "--min-p",
            "0.0",
            // llama.cpp recorded these logits with an f16 KV cache; OxiLLaMa
            // defaults to f32, so match the reference explicitly.
            "--kv-dtype",
            "f16",
            // Never let a stray ~/.config/oxillama profile change a knob.
            "--profile",
            "__oxillama_parity_no_profile__",
        ])
        .output()
        .expect("spawn oxillama");
    assert!(
        output.status.success(),
        "{}: oxillama run failed: {}",
        case.id,
        String::from_utf8_lossy(&output.stderr)
    );

    // The manifest must confirm the model saw exactly llama.cpp's ids.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out_dir.join("manifest.json")).expect("read"),
    )
    .expect("parse manifest.json");
    assert_eq!(
        manifest["vocab_size"].as_u64(),
        Some(case.vocab_size as u64),
        "{}: vocab size",
        case.id
    );
    assert_eq!(
        manifest["add_bos_applied"].as_bool(),
        Some(false),
        "{}: llama.cpp applied no BOS to either of these GGUFs, so neither may we",
        case.id
    );
    let seen: Vec<u32> = manifest["prompt_token_ids"]
        .as_array()
        .expect("prompt_token_ids")
        .iter()
        .map(|v| v.as_u64().expect("u64") as u32)
        .collect();
    assert_eq!(seen, case.prompt, "{}: prompt token ids", case.id);

    let mut decisive = Vec::new();
    let mut near_tie = Vec::new();
    for (step, (&expected, &margin)) in case.greedy.iter().zip(case.margins).enumerate() {
        let logits = read_logits(
            &out_dir.join(format!("step{step}.logits.f32.bin")),
            case.vocab_size,
        );
        let got = argmax(&logits) as u32;
        if got == expected {
            continue;
        }
        let message = format!(
            "{} step{step}: llama.cpp picked {expected}, OxiLLaMa picked {got} (reference margin {margin})",
            case.id
        );
        if margin > LLAMACPP_CROSS_KERNEL_SPREAD {
            decisive.push(message);
        } else {
            near_tie.push(message);
        }
    }
    let _ = std::fs::remove_dir_all(&out_dir);
    (decisive, near_tie)
}

/// Every decisive step must reproduce llama.cpp's greedy token.
///
/// Skipped (with a printed notice) unless the GGUF env vars are set — see the
/// module docs.
#[test]
fn greedy_tokens_match_llamacpp_on_every_decisive_step() {
    let mut ran = 0usize;
    let mut decisive = Vec::new();
    let mut near_tie = Vec::new();
    for case in CASES {
        let Ok(model) = std::env::var(case.env) else {
            continue;
        };
        if !std::path::Path::new(&model).exists() {
            eprintln!("{} points at a missing file ({model}); skipping", case.env);
            continue;
        }
        let (d, n) = run_case(case, &model);
        decisive.extend(d);
        near_tie.extend(n);
        ran += 1;
    }
    if ran == 0 {
        eprintln!(
            "skipped: set OXILLAMA_PARITY_LLAMA3_GGUF / OXILLAMA_PARITY_QWEN3_GGUF \
             to the reference checkpoints to run this test"
        );
        return;
    }
    for message in &near_tie {
        // Reported, never fatal: llama.cpp disagrees with llama.cpp here.
        eprintln!("near-tie divergence (not a failure): {message}");
    }
    assert!(
        decisive.is_empty(),
        "{ran} case(s) run; {} decisive step(s) diverged from llama.cpp:\n{}",
        decisive.len(),
        decisive.join("\n")
    );
}

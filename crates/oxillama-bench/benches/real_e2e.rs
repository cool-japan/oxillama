//! Real end-to-end benchmark binary: model load wall-time, prefill tok/s,
//! decode tok/s (fixed 64-token greedy), and peak RSS against an actual GGUF
//! file loaded through `oxillama-runtime`.
//!
//! This is deliberately **not** a Criterion benchmark. Criterion's
//! `bench_function` re-runs its closure many times to reach statistical
//! significance, which is the right tool for a microsecond-scale kernel and
//! the wrong one for "load an 8B-parameter GGUF file" — repeating that dozens
//! of times to satisfy a statistics engine would turn a benchmark run into a
//! multi-hour one for no benefit. `[[bench]] harness = false` only means
//! "cargo does not inject libtest's harness"; it does not require using
//! Criterion, so this is a plain `fn main()` that loads the model once, runs
//! one prefill + 64-token greedy decode pass, and prints a report.
//!
//! `cargo bench` compiles bench targets with the (release-derived) `bench`
//! profile, which satisfies the "release build" requirement without a
//! separate manual `cargo build --release` step.
//!
//! # Usage
//!
//! ```text
//! OXILLAMA_BENCH_MODEL=/path/to/model.gguf \
//!     cargo bench -p oxillama-bench --bench real_e2e
//! ```
//!
//! Skips gracefully (prints a message, exits `0`) when `OXILLAMA_BENCH_MODEL`
//! is unset — this binary is built and (attempted-)run as part of `cargo
//! bench -p oxillama-bench` in CI, which has no model file on disk.

use oxillama_bench::{run_real_e2e_bench, RealE2eConfig};

fn main() {
    // `cargo bench --bench real_e2e -- <extra args>` forwards `<extra args>`
    // (and sometimes `--bench` itself) to this binary's argv. This benchmark
    // takes no arguments of its own (model path is the env var, decode
    // length is mission-fixed) — argv is intentionally ignored rather than
    // parsed strictly, so `cargo bench`'s own flags never cause a spurious
    // "unrecognized argument" failure here.
    let config = RealE2eConfig::default();

    match run_real_e2e_bench(&config) {
        Ok(Some(report)) => {
            println!("{}", report.display());
        }
        Ok(None) => {
            // Skip message already printed to stderr by run_real_e2e_bench.
        }
        Err(err) => {
            eprintln!("[oxillama-bench] real_e2e benchmark failed: {err}");
            std::process::exit(1);
        }
    }
}

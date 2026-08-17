//! `--help` coverage for every subcommand (C11).
//!
//! Each test invokes the real compiled binary and checks that `--help`
//! exits 0 and documents a handful of flags specific to that subcommand —
//! a regression net against a subcommand silently losing a flag (e.g. a
//! `clap` attribute typo) or the binary panicking while building the help
//! text, without needing a real GGUF model file.

use std::path::PathBuf;
use std::process::{Command, Output};

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

fn help_for(args: &[&str]) -> Output {
    let mut full_args: Vec<&str> = args.to_vec();
    full_args.push("--help");
    Command::new(oxillama_bin())
        .args(&full_args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run oxillama {}: {e}", full_args.join(" ")))
}

fn assert_help_ok_and_contains(args: &[&str], expected: &[&str]) {
    let output = help_for(args);
    assert!(
        output.status.success(),
        "`oxillama {} --help` should exit 0, stderr: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for needle in expected {
        assert!(
            stdout.contains(needle),
            "`oxillama {} --help` should mention '{needle}', got:\n{stdout}",
            args.join(" ")
        );
    }
}

#[test]
fn top_level_help() {
    assert_help_ok_and_contains(&[], &["Usage", "--config"]);
}

#[test]
fn run_help() {
    assert_help_ok_and_contains(
        &["run"],
        &[
            "--model",
            "--prompt",
            "--file",
            "--stdin",
            "--profile",
            "--max-tokens",
            "--kv-dtype",
            "--add-bos",
            "--dump-logits",
            "--prompt-tokens",
            "--force-tokens",
        ],
    );
}

#[cfg(feature = "server")]
#[test]
fn serve_help() {
    assert_help_ok_and_contains(
        &["serve"],
        &[
            "--model",
            "--host",
            "--port",
            "--profile",
            "--api-key",
            "--admin-token",
            "--allowed-model-dir",
            "--disable-cors",
            "--max-concurrent",
        ],
    );
}

#[test]
fn info_help() {
    assert_help_ok_and_contains(&["info"], &["--model", "--tensors", "--metadata"]);
}

#[cfg(feature = "bench")]
#[test]
fn bench_help() {
    assert_help_ok_and_contains(
        &["bench"],
        &["--model", "--warmup", "--iterations", "--n-predict"],
    );
}

#[test]
fn chat_help() {
    let output = help_for(&["chat"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for needle in ["--model", "--profile", "--system", "--model-id"] {
        assert!(
            stdout.contains(needle),
            "chat --help should mention '{needle}', got:\n{stdout}"
        );
    }
    // `--tui` is only advertised when the binary was actually built with the
    // `tui` feature (C5) — check both directions explicitly rather than
    // assuming either.
    if cfg!(feature = "tui") {
        assert!(
            stdout.contains("--tui"),
            "a tui-feature build must advertise --tui in chat --help"
        );
    } else {
        assert!(
            !stdout.contains("--tui"),
            "a non-tui build must not advertise a flag it will refuse"
        );
    }
}

#[test]
fn hub_help() {
    assert_help_ok_and_contains(&["hub"], &["pull", "list", "rm"]);
}

#[test]
fn hub_pull_help() {
    assert_help_ok_and_contains(
        &["hub", "pull"],
        &[
            "--file",
            "--revision",
            "--force",
            "--cache",
            "--verify-sha256",
        ],
    );
}

#[test]
fn hub_list_help() {
    assert_help_ok_and_contains(&["hub", "list"], &["--cache"]);
}

#[test]
fn hub_rm_help() {
    assert_help_ok_and_contains(&["hub", "rm"], &["--cache"]);
}

#[test]
fn quantize_help() {
    // Every K-quant mixture is encodable and advertised, spelled as
    // llama.cpp's `quantize` tool spells it, alongside the flags that control
    // re-quantizing an already-quantized source.
    assert_help_ok_and_contains(
        &["quantize"],
        &[
            "--target",
            "Q4_0",
            "Q8_0",
            "Q2_K",
            "Q3_K_M",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_K_M",
            "Q6_K",
            "--allow-requantize",
            "--force-requantize",
            "--pure",
            "--dry-run",
        ],
    );
}

#[test]
fn convert_help() {
    let output = help_for(&["convert"]);
    assert!(output.status.success());
}

#[test]
fn verify_help() {
    assert_help_ok_and_contains(&["verify"], &["--sha256"]);
}

#[test]
fn tokenize_help() {
    assert_help_ok_and_contains(&["tokenize"], &["--model", "--format"]);
}

#[test]
fn detokenize_help() {
    assert_help_ok_and_contains(&["detokenize"], &["--model", "--ids"]);
}

#[test]
fn completions_help() {
    assert_help_ok_and_contains(&["completions"], &["bash", "zsh", "fish"]);
}

#[test]
fn generate_manpage_help() {
    assert_help_ok_and_contains(&["generate-manpage"], &["--output-dir"]);
}

#[test]
fn version_help() {
    let output = help_for(&["version"]);
    assert!(output.status.success());
}

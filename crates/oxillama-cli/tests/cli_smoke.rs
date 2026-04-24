//! CLI smoke tests — verify subcommands compile and display help correctly.
//!
//! These tests do NOT require a real GGUF model file. They exercise the
//! argument parsing, help text, and exit codes without loading any weights.

use std::path::PathBuf;
use std::process::Command;

fn oxillama_bin() -> PathBuf {
    // Use cargo's output directory to find the compiled binary.
    let mut path = std::env::current_exe()
        .expect("current_exe")
        .parent()
        .expect("parent")
        .to_path_buf();
    // From tests/deps/ go up to the target dir
    if path.ends_with("deps") {
        path.pop();
    }
    path.join("oxillama")
}

#[test]
fn help_exits_zero() {
    let output = Command::new(oxillama_bin())
        .arg("--help")
        .output()
        .expect("failed to run oxillama --help");
    assert!(output.status.success(), "oxillama --help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage") || stdout.contains("usage") || stdout.len() > 10,
        "help should print usage text"
    );
}

#[test]
fn version_exits_zero() {
    let output = Command::new(oxillama_bin())
        .arg("--version")
        .output()
        .expect("failed to run oxillama --version");
    assert!(output.status.success(), "oxillama --version should exit 0");
}

#[test]
fn run_missing_model_exits_nonzero() {
    let output = Command::new(oxillama_bin())
        .args([
            "run",
            "--model",
            "/nonexistent/model.gguf",
            "--prompt",
            "hi",
        ])
        .output()
        .expect("failed to run oxillama run with bad model");
    // Should fail (nonzero exit) because the model file doesn't exist
    assert!(
        !output.status.success(),
        "run with nonexistent model should exit nonzero"
    );
}

#[test]
fn completions_bash_exits_zero() {
    let output = Command::new(oxillama_bin())
        .args(["completions", "bash"])
        .output()
        .expect("failed to run oxillama completions bash");
    assert!(output.status.success(), "completions bash should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.is_empty(), "completions should print something");
}

#[test]
fn info_missing_model_exits_nonzero() {
    let output = Command::new(oxillama_bin())
        .args(["info", "--model", "/nonexistent/model.gguf"])
        .output()
        .expect("failed to run oxillama info with bad model");
    assert!(
        !output.status.success(),
        "info with nonexistent model should exit nonzero"
    );
}

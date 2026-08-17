//! Exit-code classification, exercised through the real compiled binary
//! (C9 + C11).
//!
//! `exit_codes.rs` has its own unit tests calling `classify()` directly on
//! synthetic errors; these tests instead drive the actual failure paths end
//! to end, so a regression in *how* an error reaches `classify()` (e.g. a
//! call site losing the typed source by switching back to
//! `anyhow!("...: {e}")`) is caught too.

use std::path::PathBuf;
use std::process::Command;

const ERR_MODEL_NOT_FOUND: i32 = 2;
const ERR_IO: i32 = 6;

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

#[test]
fn info_missing_model_exits_model_not_found() {
    let output = Command::new(oxillama_bin())
        .args(["info", "--model", "/nonexistent/oxillama_test_model.gguf"])
        .output()
        .expect("failed to run oxillama info");
    assert_eq!(
        output.status.code(),
        Some(ERR_MODEL_NOT_FOUND),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_missing_prompt_file_exits_io_not_model_not_found() {
    // Regression test for the pre-fix substring-matching bug (C9): the
    // prompt file is read *before* the model is ever touched, so a failed
    // `--file` read must classify as ERR_IO, never ERR_MODEL_NOT_FOUND —
    // even though the model path given here also doesn't exist, and even
    // though the io::Error's message contains "No such file or directory"
    // (which used to trigger the ERR_MODEL_NOT_FOUND substring match).
    let output = Command::new(oxillama_bin())
        .args([
            "run",
            "--model",
            "/nonexistent/oxillama_test_model.gguf",
            "--file",
            "/nonexistent/oxillama_test_prompt.txt",
        ])
        .output()
        .expect("failed to run oxillama run");
    assert!(!output.status.success());
    assert_eq!(
        output.status.code(),
        Some(ERR_IO),
        "a failed --file prompt read must exit ERR_IO ({ERR_IO}), not ERR_MODEL_NOT_FOUND \
         ({ERR_MODEL_NOT_FOUND}) — no model was ever touched. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn tokenize_missing_model_exits_model_not_found() {
    let output = Command::new(oxillama_bin())
        .args([
            "tokenize",
            "--model",
            "/nonexistent/oxillama_test_model.gguf",
            "hello",
        ])
        .output()
        .expect("failed to run oxillama tokenize");
    assert!(!output.status.success());
    assert_eq!(
        output.status.code(),
        Some(ERR_MODEL_NOT_FOUND),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

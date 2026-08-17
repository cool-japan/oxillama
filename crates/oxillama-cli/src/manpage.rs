//! `oxillama generate-manpage` — write man pages for `oxillama` and every
//! subcommand.
//!
//! ## Why this exists
//!
//! The previous implementation called `clap_mangen::Man::new(cmd).render(..)`
//! on the *top-level* `Cli` command only, writing a single `oxillama.1`. That
//! page's `SUBCOMMANDS` section cross-referenced `oxillama-run(1)`,
//! `oxillama-serve(1)`, and a dozen more pages that were never generated, so
//! `man oxillama-run` failed for every one of them, and no subcommand's own
//! flags were documented anywhere.
//!
//! This module recurses through the full `clap::Command` tree (including
//! nested subcommands such as `oxillama hub pull`) and renders one page per
//! node, so every cross-reference the top-level page makes actually
//! resolves. Each page's title, synopsis, options, and `.SH VERSION` banner
//! are derived from the `clap::Command` at generation time, so they can
//! never drift from the binary's actual version or the flags it actually
//! parses the way a hand-maintained page can.

use std::path::{Path, PathBuf};

use clap::Command;

/// Generate `oxillama.1` plus one page per subcommand (recursively) into
/// `output_dir`.
///
/// Returns the list of file paths written, sorted for stable, reproducible
/// output.
///
/// # Errors
///
/// Returns an `io::Error` if `output_dir` cannot be created or a page cannot
/// be written.
pub fn generate_all(cmd: Command, output_dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    std::fs::create_dir_all(output_dir)?;

    // Matches `clap_mangen::generate_to`'s own setup: disable the synthetic
    // "help" pseudo-subcommand (it has no useful man page of its own) and
    // force a full build pass so every subcommand's qualified display name
    // (`oxillama-run`, `oxillama-hub-pull`, …) is resolved before we read it.
    let mut cmd = cmd.disable_help_subcommand(true);
    cmd.build();

    let mut written = Vec::new();
    generate_recursive(cmd, output_dir, &mut written)?;
    written.sort();
    Ok(written)
}

fn generate_recursive(
    cmd: Command,
    output_dir: &Path,
    written: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    let subcommands: Vec<Command> = cmd
        .get_subcommands()
        .filter(|s| !s.is_hide_set())
        .cloned()
        .collect();
    for sub in subcommands {
        generate_recursive(sub, output_dir, written)?;
    }
    let path = clap_mangen::Man::new(cmd).generate_to(output_dir)?;
    written.push(path);
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser, Subcommand};

    #[derive(Parser)]
    #[command(name = "testcli", version = "9.9.9")]
    struct TestCli {
        #[command(subcommand)]
        command: TestCommands,
    }

    #[derive(Subcommand)]
    enum TestCommands {
        Alpha {
            #[arg(long)]
            flag: bool,
        },
        Beta {
            #[command(subcommand)]
            command: BetaSub,
        },
    }

    #[derive(Subcommand)]
    enum BetaSub {
        Inner,
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oxillama_manpage_tests_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn generates_one_page_per_subcommand_including_nested() {
        let dir = temp_dir("nested");
        let written = generate_all(TestCli::command(), &dir).expect("generate_all should succeed");

        // Top-level + Alpha + Beta + Beta's nested Inner = 4 pages.
        assert_eq!(written.len(), 4, "expected 4 man pages, got {written:?}");
        assert!(dir.join("testcli.1").exists());
        assert!(dir.join("testcli-alpha.1").exists());
        assert!(dir.join("testcli-beta.1").exists());
        assert!(dir.join("testcli-beta-inner.1").exists());
    }

    #[test]
    fn subcommand_page_cross_references_resolve() {
        let dir = temp_dir("crossref");
        generate_all(TestCli::command(), &dir).expect("generate_all should succeed");

        let top_level = std::fs::read_to_string(dir.join("testcli.1")).expect("read top page");
        // The top-level page must reference subcommands by their qualified
        // name, and that qualified name must actually exist on disk.
        assert!(top_level.contains("testcli\\-alpha") || top_level.contains("testcli-alpha"));

        let alpha_page =
            std::fs::read_to_string(dir.join("testcli-alpha.1")).expect("read alpha page");
        assert!(alpha_page.contains("flag"));
    }

    #[test]
    fn version_banner_matches_command_version() {
        let dir = temp_dir("version");
        generate_all(TestCli::command(), &dir).expect("generate_all should succeed");
        let top_level = std::fs::read_to_string(dir.join("testcli.1")).expect("read top page");
        assert!(
            top_level.contains("9.9.9"),
            "man page should embed the command's actual version, got: {top_level}"
        );
    }
}

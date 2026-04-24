//! Cross-compilation validation for all WGSL shaders.
//!
//! Parses every `*.wgsl` file in `src/shaders/` as WGSL, then cross-compiles
//! each one to:
//!   - Metal MSL  (macOS / iOS backend)
//!   - Vulkan SPIR-V (Linux / Android backend)
//!
//! A failing test here means a shader regression is caught in CI without
//! needing a physical GPU of each flavour.

use naga::{
    back::{
        msl::{self, PipelineOptions},
        spv::{self, WriterFlags},
    },
    front::wgsl::Frontend,
    valid::{Capabilities, ValidationFlags, Validator},
};
use std::path::PathBuf;

fn shaders_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("shaders")
}

/// Collect `(filename, source)` pairs for every `.wgsl` in `src/shaders/`.
fn collect_shaders() -> Vec<(String, String)> {
    let dir = shaders_dir();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(err) => panic!("Cannot read shaders dir {dir:?}: {err}"),
    };

    let mut shaders = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unknown>")
            .to_string();
        match std::fs::read_to_string(&path) {
            Ok(src) => shaders.push((name, src)),
            Err(err) => eprintln!("WARNING: could not read {path:?}: {err}"),
        }
    }
    shaders
}

// ---------------------------------------------------------------------------
// Helper: parse + validate, returning (Module, ModuleInfo) or panicking.
// ---------------------------------------------------------------------------

fn parse_and_validate(name: &str, source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    let mut frontend = Frontend::new();
    let module = frontend
        .parse(source)
        .unwrap_or_else(|e| panic!("WGSL parse failed for {name}: {e:?}"));

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation failed for {name}: {e:?}"));

    (module, info)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn all_shaders_parse_as_wgsl() {
    let shaders = collect_shaders();
    assert!(
        !shaders.is_empty(),
        "No .wgsl shaders found in src/shaders/ — check CARGO_MANIFEST_DIR"
    );

    for (name, source) in &shaders {
        let _module = {
            let mut frontend = Frontend::new();
            frontend
                .parse(source)
                .unwrap_or_else(|e| panic!("WGSL parse failed for {name}: {e:?}"))
        };
        eprintln!("[ok] {name} — WGSL parse");
    }
}

#[test]
fn all_shaders_cross_compile_to_msl() {
    let shaders = collect_shaders();
    assert!(
        !shaders.is_empty(),
        "No .wgsl shaders found in src/shaders/"
    );

    for (name, source) in &shaders {
        let (module, info) = parse_and_validate(name, source);

        let options = msl::Options::default();
        let pipeline = PipelineOptions::default();
        let (msl_src, _translation_info) = msl::write_string(&module, &info, &options, &pipeline)
            .unwrap_or_else(|e| panic!("MSL emit failed for {name}: {e:?}"));

        assert!(!msl_src.is_empty(), "MSL output empty for {name}");
        eprintln!("[ok] {name} — MSL emit ({} chars)", msl_src.len());
    }
}

#[test]
fn all_shaders_cross_compile_to_spirv() {
    let shaders = collect_shaders();
    assert!(
        !shaders.is_empty(),
        "No .wgsl shaders found in src/shaders/"
    );

    for (name, source) in &shaders {
        let (module, info) = parse_and_validate(name, source);

        let options = spv::Options {
            flags: WriterFlags::empty(),
            ..Default::default()
        };
        let spv_words = spv::write_vec(&module, &info, &options, None)
            .unwrap_or_else(|e| panic!("SPIR-V emit failed for {name}: {e:?}"));

        assert!(!spv_words.is_empty(), "SPIR-V output empty for {name}");
        eprintln!("[ok] {name} — SPIR-V emit ({} words)", spv_words.len());
    }
}

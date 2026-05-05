//! `oxillama verify <model.gguf> [--sha256 <hex>]` — integrity checker.
//!
//! Validates:
//! 1. File starts with GGUF magic bytes.
//! 2. GGUF version is 1, 2, or 3.
//! 3. `GgufFile::parse` succeeds without error.
//! 4. Optionally: SHA-256 of the file matches a user-supplied hex digest.
//! 5. For each tensor: the data offset + data_size is within the file bounds,
//!    and `dimensions.iter().product::<u64>() * element_size` is consistent
//!    with the declared data length.

use std::path::PathBuf;

use anyhow::Context;
use sha2::{Digest, Sha256};

/// Arguments for the `verify` subcommand.
#[derive(clap::Args, Clone, Debug)]
pub struct VerifyArgs {
    /// Path to the GGUF model file to verify.
    pub model: PathBuf,

    /// Expected SHA-256 hex digest of the file (64 hex characters).
    #[arg(long, value_name = "HEX")]
    pub sha256: Option<String>,
}

/// Run model verification.
///
/// Returns `Ok(())` if all checks pass; returns an error describing the
/// first failing check otherwise.
pub fn run_verify(args: &VerifyArgs) -> anyhow::Result<()> {
    let path = &args.model;

    // ── 1. Read the file ───────────────────────────────────────────────
    let data = std::fs::read(path)
        .with_context(|| format!("cannot read model file '{}'", path.display()))?;

    // ── 2. Check GGUF magic bytes ──────────────────────────────────────
    if data.len() < 4 {
        anyhow::bail!(
            "file '{}' is too short to contain GGUF magic ({} bytes)",
            path.display(),
            data.len()
        );
    }
    if &data[..4] != b"GGUF" {
        anyhow::bail!(
            "file '{}' does not start with GGUF magic (got {:02x} {:02x} {:02x} {:02x})",
            path.display(),
            data[0],
            data[1],
            data[2],
            data[3]
        );
    }

    // ── 3. Check GGUF version ──────────────────────────────────────────
    if data.len() < 8 {
        anyhow::bail!(
            "file '{}' is too short to contain GGUF version field",
            path.display()
        );
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if !(1..=3).contains(&version) {
        anyhow::bail!(
            "file '{}' has unsupported GGUF version {version} (expected 1, 2, or 3)",
            path.display()
        );
    }

    // ── 4. Full parse ──────────────────────────────────────────────────
    let gguf_file = oxillama_gguf::GgufFile::parse(&data)
        .with_context(|| format!("GGUF parse failed for '{}'", path.display()))?;

    // ── 5. Optional SHA-256 check ──────────────────────────────────────
    if let Some(ref expected_hex) = args.sha256 {
        let digest = compute_sha256(&data);
        if digest != expected_hex.to_ascii_lowercase() {
            anyhow::bail!(
                "SHA-256 mismatch for '{}': expected {}, got {}",
                path.display(),
                expected_hex.to_ascii_lowercase(),
                digest
            );
        }
    }

    // ── 6. Tensor bounds checking ──────────────────────────────────────
    let data_offset = gguf_file.tensors.data_offset() as usize;

    for (name, info) in gguf_file.tensors.iter() {
        let tensor_offset = info.offset as usize;
        let tensor_data_size = info.data_size() as usize;

        // Check that the tensor data region is within the file.
        let abs_start = data_offset
            .checked_add(tensor_offset)
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}': offset arithmetic overflow"))?;
        let abs_end = abs_start
            .checked_add(tensor_data_size)
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}': data end arithmetic overflow"))?;

        if abs_end > data.len() {
            anyhow::bail!(
                "tensor '{name}': data region [{abs_start}, {abs_end}) exceeds file size {}",
                data.len()
            );
        }

        // Cross-check: dimensions.product() * element_size_for_type == tensor_data_size.
        let n_elements = info.n_elements();
        let block_size = info.tensor_type.block_size() as u64;
        let block_bytes = info.tensor_type.block_bytes() as u64;
        let expected_size = if block_size == 0 {
            0u64
        } else {
            n_elements.div_ceil(block_size) * block_bytes
        };

        if expected_size != tensor_data_size as u64 {
            anyhow::bail!(
                "tensor '{name}': expected data size {expected_size} bytes (from shape {:?} × {block_bytes} bytes/block), \
                 but declared data_size is {tensor_data_size} bytes",
                info.dimensions
            );
        }
    }

    println!(
        "ok: '{}' — version {}, {} tensors, {} metadata KV pairs",
        path.display(),
        gguf_file.header.version,
        gguf_file.header.tensor_count,
        gguf_file.header.metadata_kv_count,
    );

    Ok(())
}

/// Compute the lowercase hex-encoded SHA-256 digest of a byte slice.
fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the minimal valid GGUF v3 bytes (zero tensors, zero metadata).
    fn minimal_gguf_v3() -> Vec<u8> {
        let mut buf = Vec::new();
        // magic "GGUF"
        buf.extend_from_slice(b"GGUF");
        // version = 3 (LE u32)
        buf.extend_from_slice(&3u32.to_le_bytes());
        // tensor_count = 0 (LE u64)
        buf.extend_from_slice(&0u64.to_le_bytes());
        // metadata_kv_count = 0 (LE u64)
        buf.extend_from_slice(&0u64.to_le_bytes());
        // Alignment pad to 32 bytes
        let rem = buf.len() % 32;
        if rem != 0 {
            buf.resize(buf.len() + 32 - rem, 0u8);
        }
        buf
    }

    /// Write bytes to a temp file and return the path.
    fn write_temp(name: &str, data: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join("oxillama_verify_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(name);
        std::fs::write(&path, data).expect("write temp file");
        path
    }

    #[test]
    fn verify_valid_gguf_passes() {
        let data = minimal_gguf_v3();
        let path = write_temp("verify_valid.gguf", &data);
        let args = VerifyArgs {
            model: path,
            sha256: None,
        };
        assert!(run_verify(&args).is_ok(), "valid minimal GGUF should pass");
    }

    #[test]
    fn verify_corrupt_magic_fails() {
        let mut data = minimal_gguf_v3();
        // Corrupt the magic bytes.
        data[0] = 0xFF;
        data[1] = 0xFE;
        data[2] = 0xFD;
        data[3] = 0xFC;
        let path = write_temp("verify_corrupt.gguf", &data);
        let args = VerifyArgs {
            model: path,
            sha256: None,
        };
        assert!(
            run_verify(&args).is_err(),
            "corrupt magic should fail verification"
        );
    }

    #[test]
    fn verify_sha256_correct_passes() {
        let data = minimal_gguf_v3();
        let expected = compute_sha256(&data);
        let path = write_temp("verify_sha256_ok.gguf", &data);
        let args = VerifyArgs {
            model: path,
            sha256: Some(expected),
        };
        assert!(
            run_verify(&args).is_ok(),
            "correct SHA-256 should pass verification"
        );
    }

    #[test]
    fn verify_sha256_mismatch_fails() {
        let data = minimal_gguf_v3();
        let wrong_hex = "0".repeat(64);
        let path = write_temp("verify_sha256_bad.gguf", &data);
        let args = VerifyArgs {
            model: path,
            sha256: Some(wrong_hex),
        };
        assert!(
            run_verify(&args).is_err(),
            "wrong SHA-256 should fail verification"
        );
    }
}

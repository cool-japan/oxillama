//! Tensor data integrity validation using Blake3 hashing.
//!
//! Provides per-tensor and whole-model hash computation for data integrity
//! verification. Useful for detecting corrupted downloads, storage bit-flips,
//! or tampered model files.

use crate::error::{GgufError, GgufResult};
use crate::tensor_info::{TensorInfo, TensorStore};

/// Hash of a single tensor's data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHash {
    /// Tensor name.
    pub name: String,
    /// Blake3 hash (32 bytes).
    pub hash: [u8; 32],
    /// Size of the hashed data in bytes.
    pub data_size: u64,
}

/// Hash manifest for an entire model.
#[derive(Debug, Clone)]
pub struct ModelHashManifest {
    /// Per-tensor hashes, sorted by tensor name.
    pub tensor_hashes: Vec<TensorHash>,
    /// Combined hash of all tensor data (hash of per-tensor hashes).
    pub model_hash: [u8; 32],
    /// Total bytes hashed.
    pub total_bytes: u64,
}

/// A single integrity check failure.
#[derive(Debug, Clone)]
pub struct IntegrityFailure {
    /// Name of the tensor that failed verification.
    pub tensor_name: String,
    /// Expected hash from the manifest.
    pub expected: [u8; 32],
    /// Actual hash computed from the data.
    pub actual: [u8; 32],
}

/// Compute the Blake3 hash of a single tensor's data.
///
/// # Arguments
/// * `data` — full file data (or mmap'd region)
/// * `tensor_info` — tensor metadata (offset, size)
/// * `data_section_offset` — base offset of the data section in the file
///
/// # Errors
/// Returns `GgufError::IntegrityError` if the tensor data slice is out of bounds.
pub fn hash_tensor(
    data: &[u8],
    tensor_info: &TensorInfo,
    data_section_offset: u64,
) -> GgufResult<TensorHash> {
    let slice = tensor_data_slice(data, tensor_info, data_section_offset)?;
    let hash = blake3::hash(slice);
    Ok(TensorHash {
        name: tensor_info.name.clone(),
        hash: *hash.as_bytes(),
        data_size: tensor_info.data_size(),
    })
}

/// Compute hashes for all tensors in a model and produce a manifest.
///
/// Tensors are sorted by name for deterministic ordering.
///
/// # Errors
/// Returns an error if any tensor data slice is out of bounds.
pub fn compute_model_manifest(data: &[u8], tensors: &TensorStore) -> GgufResult<ModelHashManifest> {
    let data_section_offset = tensors.data_offset();

    // Collect and sort by name for deterministic ordering
    let mut entries: Vec<(&String, &TensorInfo)> = tensors.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut tensor_hashes = Vec::with_capacity(entries.len());
    let mut total_bytes: u64 = 0;

    for (_name, info) in &entries {
        let th = hash_tensor(data, info, data_section_offset)?;
        total_bytes = total_bytes.saturating_add(th.data_size);
        tensor_hashes.push(th);
    }

    // Model hash = blake3(concatenation of all per-tensor hashes)
    let mut combined_hasher = blake3::Hasher::new();
    for th in &tensor_hashes {
        combined_hasher.update(&th.hash);
    }
    let model_hash = *combined_hasher.finalize().as_bytes();

    Ok(ModelHashManifest {
        tensor_hashes,
        model_hash,
        total_bytes,
    })
}

/// Verify a tensor's data against an expected hash.
///
/// Returns `Ok(true)` if the hash matches, `Ok(false)` if it does not.
///
/// # Errors
/// Returns an error if the tensor data slice is out of bounds.
pub fn verify_tensor(
    data: &[u8],
    tensor_info: &TensorInfo,
    data_section_offset: u64,
    expected: &[u8; 32],
) -> GgufResult<bool> {
    let computed = hash_tensor(data, tensor_info, data_section_offset)?;
    Ok(&computed.hash == expected)
}

/// Verify all tensors in a model against a manifest.
///
/// Returns a list of failures. An empty list means all tensors passed.
///
/// # Errors
/// Returns an error if any tensor data slice is out of bounds.
pub fn verify_model(
    data: &[u8],
    tensors: &TensorStore,
    manifest: &ModelHashManifest,
) -> GgufResult<Vec<IntegrityFailure>> {
    let actual_manifest = compute_model_manifest(data, tensors)?;

    let mut failures = Vec::new();

    // Build a lookup from the expected manifest
    let expected_map: std::collections::HashMap<&str, &[u8; 32]> = manifest
        .tensor_hashes
        .iter()
        .map(|th| (th.name.as_str(), &th.hash))
        .collect();

    for actual_th in &actual_manifest.tensor_hashes {
        if let Some(expected_hash) = expected_map.get(actual_th.name.as_str()) {
            if &actual_th.hash != *expected_hash {
                failures.push(IntegrityFailure {
                    tensor_name: actual_th.name.clone(),
                    expected: **expected_hash,
                    actual: actual_th.hash,
                });
            }
        }
    }

    Ok(failures)
}

/// Format a 32-byte hash as a lowercase hex string (64 characters).
pub fn hash_to_hex(hash: &[u8; 32]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Format a manifest as a human-readable report.
pub fn format_manifest(manifest: &ModelHashManifest) -> String {
    let mut report = String::new();
    use std::fmt::Write;

    let _ = writeln!(report, "Model Hash Manifest");
    let _ = writeln!(report, "===================");
    let _ = writeln!(
        report,
        "Model hash:    {}",
        hash_to_hex(&manifest.model_hash)
    );
    let _ = writeln!(report, "Total bytes:   {}", manifest.total_bytes);
    let _ = writeln!(report, "Tensor count:  {}", manifest.tensor_hashes.len());
    let _ = writeln!(report);

    for th in &manifest.tensor_hashes {
        let _ = writeln!(
            report,
            "  {} {} ({} bytes)",
            hash_to_hex(&th.hash),
            th.name,
            th.data_size
        );
    }

    report
}

/// Extract the data slice for a tensor, validating bounds.
fn tensor_data_slice<'a>(
    data: &'a [u8],
    tensor_info: &TensorInfo,
    data_section_offset: u64,
) -> GgufResult<&'a [u8]> {
    let abs_offset = data_section_offset
        .checked_add(tensor_info.offset)
        .ok_or_else(|| GgufError::IntegrityError {
            tensor_name: tensor_info.name.clone(),
            reason: "offset overflow".to_string(),
        })?;
    let size = tensor_info.data_size();
    let end = abs_offset
        .checked_add(size)
        .ok_or_else(|| GgufError::IntegrityError {
            tensor_name: tensor_info.name.clone(),
            reason: "offset + size overflow".to_string(),
        })?;

    let abs_offset_usize = usize::try_from(abs_offset).map_err(|_| GgufError::IntegrityError {
        tensor_name: tensor_info.name.clone(),
        reason: format!("offset {abs_offset} exceeds platform address space"),
    })?;
    let end_usize = usize::try_from(end).map_err(|_| GgufError::IntegrityError {
        tensor_name: tensor_info.name.clone(),
        reason: format!("end offset {end} exceeds platform address space"),
    })?;

    if end_usize > data.len() {
        return Err(GgufError::IntegrityError {
            tensor_name: tensor_info.name.clone(),
            reason: format!(
                "tensor data [{abs_offset_usize}..{end_usize}] out of bounds (data len: {})",
                data.len()
            ),
        });
    }

    Ok(&data[abs_offset_usize..end_usize])
}

// ─── TensorHashValidator ─────────────────────────────────────────────────────
//
// Validates tensor blobs against Blake3 hashes stored in GGUF metadata under
// the key `general.tensor_hashes`.  The value must be an array of strings,
// each formatted as `"<tensor_name>:<hex_hash>"` (64 hex chars = 32 bytes).

use crate::metadata::MetadataStore;

/// Expected hash for a single tensor blob, sourced from GGUF metadata.
#[derive(Debug, Clone)]
pub struct TensorHashEntry {
    /// Name of the tensor this hash covers.
    pub tensor_name: String,
    /// Expected Blake3 hash (32 bytes).
    pub expected: [u8; 32],
}

/// Validates tensor blobs against Blake3 hashes stored in GGUF metadata.
///
/// Build one via [`TensorHashValidator::from_metadata`], then call
/// [`TensorHashValidator::validate`] for each tensor blob to check.
pub struct TensorHashValidator {
    entries: std::collections::HashMap<String, [u8; 32]>,
}

impl TensorHashValidator {
    /// Build from GGUF metadata.
    ///
    /// Reads `general.tensor_hashes` — an array of `"name:hexhash"` strings.
    /// Returns `Ok(None)` when the key is absent (validation opt-out).
    /// Returns `Err` if the key is present but malformed.
    pub fn from_metadata(metadata: &MetadataStore) -> GgufResult<Option<Self>> {
        let Some(value) = metadata.get("general.tensor_hashes") else {
            return Ok(None);
        };

        let arr = value.as_array().ok_or_else(|| GgufError::InvalidMetadata {
            key: "general.tensor_hashes".to_string(),
            reason: "expected array value".to_string(),
        })?;

        let mut entries = std::collections::HashMap::with_capacity(arr.len());

        for item in arr {
            let s = item.as_str().ok_or_else(|| GgufError::InvalidMetadata {
                key: "general.tensor_hashes".to_string(),
                reason: "array element is not a string".to_string(),
            })?;

            // Format: "<tensor_name>:<64-char-hex>"
            let colon = s.rfind(':').ok_or_else(|| GgufError::InvalidMetadata {
                key: "general.tensor_hashes".to_string(),
                reason: format!("entry missing ':' separator: {s}"),
            })?;
            let tensor_name = s[..colon].to_string();
            let hex = &s[colon + 1..];
            if hex.len() != 64 {
                return Err(GgufError::InvalidMetadata {
                    key: "general.tensor_hashes".to_string(),
                    reason: format!(
                        "hash for '{tensor_name}' must be 64 hex chars, got {}",
                        hex.len()
                    ),
                });
            }
            let hash = hex_to_bytes32(hex).map_err(|e| GgufError::InvalidMetadata {
                key: "general.tensor_hashes".to_string(),
                reason: format!("invalid hex in entry for '{tensor_name}': {e}"),
            })?;
            entries.insert(tensor_name, hash);
        }

        Ok(Some(Self { entries }))
    }

    /// Validate a tensor blob against its expected hash.
    ///
    /// Returns `Ok(())` if the hash matches or if no entry exists for this
    /// tensor name (unknown tensors are allowed through).
    /// Returns `Err(GgufError::HashMismatch)` on mismatch.
    pub fn validate(&self, tensor_name: &str, data: &[u8]) -> GgufResult<()> {
        let Some(expected) = self.entries.get(tensor_name) else {
            return Ok(());
        };

        let actual = *blake3::hash(data).as_bytes();
        if &actual == expected {
            return Ok(());
        }

        Err(GgufError::HashMismatch {
            name: tensor_name.to_string(),
            expected: hash_to_hex(expected),
            actual: hash_to_hex(&actual),
        })
    }

    /// Number of hash entries loaded from metadata.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no hash entries were loaded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Parse a 64-char hex string into 32 bytes.
fn hex_to_bytes32(hex: &str) -> Result<[u8; 32], String> {
    let mut out = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex char: {b}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GgufTensorType;

    /// Build a mock TensorInfo for F32 tensors (block_size=1, block_bytes=4).
    fn mock_tensor_info(name: &str, dims: Vec<u64>, offset: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            n_dims: dims.len() as u32,
            dimensions: dims,
            tensor_type: GgufTensorType::F32,
            offset,
        }
    }

    /// Build a mock data buffer and TensorStore with several tensors.
    /// Returns (data, tensor_store).
    fn mock_model(tensor_specs: &[(&str, Vec<u64>)]) -> (Vec<u8>, TensorStore) {
        let data_section_offset: u64 = 256; // pretend header occupies 256 bytes
        let mut store = TensorStore::new();
        store.set_data_offset(data_section_offset);

        // Lay tensors out consecutively after the header
        let mut current_offset: u64 = 0;
        let mut total_data_size: u64 = 0;
        let mut infos = Vec::new();

        for (name, dims) in tensor_specs {
            let info = mock_tensor_info(name, dims.clone(), current_offset);
            let size = info.data_size();
            current_offset += size;
            total_data_size += size;
            infos.push(info);
        }

        // Create data buffer: header zeros + tensor data with deterministic pattern
        let buf_len = data_section_offset as usize + total_data_size as usize;
        let mut data = vec![0u8; buf_len];

        // Fill tensor data region with a deterministic pattern
        for (i, byte) in data[data_section_offset as usize..].iter_mut().enumerate() {
            *byte = (i % 251) as u8; // prime modulus for variety
        }

        for info in infos {
            store.insert(info);
        }

        (data, store)
    }

    #[test]
    fn test_hash_tensor_basic() {
        let (data, store) = mock_model(&[("weight", vec![4, 4])]);
        let info = store.get("weight").expect("tensor must exist");
        let result = hash_tensor(&data, info, store.data_offset());
        assert!(result.is_ok());
        let th = result.expect("hash should succeed");
        assert_eq!(th.name, "weight");
        assert_ne!(th.hash, [0u8; 32]);
        assert_eq!(th.data_size, 4 * 4 * 4); // 16 F32 elements = 64 bytes
    }

    #[test]
    fn test_hash_deterministic() {
        let (data, store) = mock_model(&[("w", vec![8])]);
        let info = store.get("w").expect("tensor must exist");
        let h1 = hash_tensor(&data, info, store.data_offset())
            .expect("first hash")
            .hash;
        let h2 = hash_tensor(&data, info, store.data_offset())
            .expect("second hash")
            .hash;
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_changes_with_data() {
        let (mut data, store) = mock_model(&[("w", vec![8])]);
        let info = store.get("w").expect("tensor must exist");
        let h1 = hash_tensor(&data, info, store.data_offset())
            .expect("original hash")
            .hash;

        // Flip one byte in the tensor data region
        let flip_idx = store.data_offset() as usize + info.offset as usize;
        data[flip_idx] ^= 0xFF;

        let h2 = hash_tensor(&data, info, store.data_offset())
            .expect("modified hash")
            .hash;
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_manifest_sorted_by_name() {
        let (data, store) = mock_model(&[("zebra", vec![4]), ("alpha", vec![4])]);
        let manifest = compute_model_manifest(&data, &store).expect("manifest");
        assert_eq!(manifest.tensor_hashes.len(), 2);
        assert_eq!(manifest.tensor_hashes[0].name, "alpha");
        assert_eq!(manifest.tensor_hashes[1].name, "zebra");
    }

    #[test]
    fn test_verify_tensor_success() {
        let (data, store) = mock_model(&[("w", vec![8])]);
        let info = store.get("w").expect("tensor must exist");
        let th = hash_tensor(&data, info, store.data_offset()).expect("hash");
        let ok = verify_tensor(&data, info, store.data_offset(), &th.hash).expect("verify");
        assert!(ok);
    }

    #[test]
    fn test_verify_tensor_tampered() {
        let (mut data, store) = mock_model(&[("w", vec![8])]);
        let info = store.get("w").expect("tensor must exist");
        let th = hash_tensor(&data, info, store.data_offset()).expect("hash");

        // Tamper
        let flip_idx = store.data_offset() as usize + info.offset as usize + 1;
        data[flip_idx] ^= 0x01;

        let ok = verify_tensor(&data, info, store.data_offset(), &th.hash).expect("verify");
        assert!(!ok);
    }

    #[test]
    fn test_verify_model_all_good() {
        let (data, store) = mock_model(&[("a", vec![4]), ("b", vec![8])]);
        let manifest = compute_model_manifest(&data, &store).expect("manifest");
        let failures = verify_model(&data, &store, &manifest).expect("verify");
        assert!(failures.is_empty());
    }

    #[test]
    fn test_verify_model_detects_corruption() {
        let (mut data, store) = mock_model(&[("a", vec![4]), ("b", vec![8])]);
        let manifest = compute_model_manifest(&data, &store).expect("manifest");

        // Corrupt tensor "b" data
        let b_info = store.get("b").expect("tensor b");
        let flip_idx = store.data_offset() as usize + b_info.offset as usize;
        data[flip_idx] ^= 0xFF;

        let failures = verify_model(&data, &store, &manifest).expect("verify");
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].tensor_name, "b");
        assert_ne!(failures[0].expected, failures[0].actual);
    }

    #[test]
    fn test_hash_to_hex_format() {
        let hash = [
            0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55,
            0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x10, 0x20, 0x30, 0x40,
            0x50, 0x60, 0x70, 0x80,
        ];
        let hex = hash_to_hex(&hash);
        assert_eq!(hex.len(), 64);
        assert_eq!(hex, hex.to_lowercase());
        assert_eq!(&hex[..8], "abcdef01");
    }

    #[test]
    fn test_out_of_bounds_returns_error() {
        let data = vec![0u8; 100]; // too small
        let info = mock_tensor_info("big", vec![1024], 0);
        let result = hash_tensor(&data, &info, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_manifest_output() {
        let (data, store) = mock_model(&[("layer.0.weight", vec![4])]);
        let manifest = compute_model_manifest(&data, &store).expect("manifest");
        let report = format_manifest(&manifest);
        assert!(report.contains("Model Hash Manifest"));
        assert!(report.contains("layer.0.weight"));
        assert!(report.contains("Tensor count:  1"));
    }

    // ── TensorHashValidator tests ──────────────────────────────────────────

    #[test]
    fn test_validator_no_metadata_key_returns_none() {
        let metadata = crate::metadata::MetadataStore::new();
        let result = TensorHashValidator::from_metadata(&metadata);
        assert!(result.is_ok());
        assert!(result.expect("ok").is_none());
    }

    #[test]
    fn test_validator_matching_hash_passes() {
        let data = b"hello world tensor data";
        let actual_hash = *blake3::hash(data).as_bytes();
        let hex = hash_to_hex(&actual_hash);
        let entry = format!("my_tensor:{hex}");

        let mut metadata = crate::metadata::MetadataStore::new();
        metadata.insert(
            "general.tensor_hashes".to_string(),
            crate::metadata::MetadataValue::Array(vec![crate::metadata::MetadataValue::String(
                entry,
            )]),
        );

        let validator = TensorHashValidator::from_metadata(&metadata)
            .expect("ok")
            .expect("some");
        assert_eq!(validator.len(), 1);
        assert!(validator.validate("my_tensor", data).is_ok());
    }

    #[test]
    fn test_validator_mismatched_hash_returns_error() {
        let real_data = b"real tensor bytes";
        let wrong_hash = [0xABu8; 32];
        let hex = hash_to_hex(&wrong_hash);
        let entry = format!("bad_tensor:{hex}");

        let mut metadata = crate::metadata::MetadataStore::new();
        metadata.insert(
            "general.tensor_hashes".to_string(),
            crate::metadata::MetadataValue::Array(vec![crate::metadata::MetadataValue::String(
                entry,
            )]),
        );

        let validator = TensorHashValidator::from_metadata(&metadata)
            .expect("ok")
            .expect("some");
        let result = validator.validate("bad_tensor", real_data);
        assert!(result.is_err());
        match result.expect_err("should be hash mismatch") {
            GgufError::HashMismatch { name, .. } => assert_eq!(name, "bad_tensor"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_validator_unknown_tensor_passes() {
        let real_data = b"some data";
        let hex = hash_to_hex(&[0xFFu8; 32]);
        let entry = format!("other_tensor:{hex}");

        let mut metadata = crate::metadata::MetadataStore::new();
        metadata.insert(
            "general.tensor_hashes".to_string(),
            crate::metadata::MetadataValue::Array(vec![crate::metadata::MetadataValue::String(
                entry,
            )]),
        );

        let validator = TensorHashValidator::from_metadata(&metadata)
            .expect("ok")
            .expect("some");
        // "unknown_tensor" is not in the validator's entries → passes through
        assert!(validator.validate("unknown_tensor", real_data).is_ok());
    }
}

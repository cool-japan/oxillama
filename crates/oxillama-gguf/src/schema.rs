//! Pluggable metadata schema validators.
//!
//! Each architecture family defines required metadata keys and value ranges.
//! `validate_schema()` checks the loaded metadata against the applicable
//! validator and returns a list of violations.  Violations are returned rather
//! than raised as errors so that callers can choose between hard-fail and
//! warn-and-continue policies.

use crate::metadata::MetadataStore;

// ─── Violation type ──────────────────────────────────────────────────────────

/// A single schema validation violation.
#[derive(Debug, Clone)]
pub struct SchemaViolation {
    /// The metadata key that was missing or invalid.
    pub key: String,
    /// Human-readable description of the problem.
    pub message: String,
}

impl SchemaViolation {
    fn new(key: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SchemaViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.key, self.message)
    }
}

// ─── Validator trait ─────────────────────────────────────────────────────────

/// Pluggable schema validator for a specific model architecture.
pub trait SchemaValidator: Send + Sync {
    /// The architecture name this validator handles (e.g. `"llama"`).
    fn architecture_name(&self) -> &str;

    /// Validate the metadata store and return any violations found.
    fn validate(&self, metadata: &MetadataStore) -> Vec<SchemaViolation>;
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

/// Check that a u32 key is present and > 0.
fn require_u32_positive(
    metadata: &MetadataStore,
    key: &str,
    violations: &mut Vec<SchemaViolation>,
) {
    match metadata.get(key).and_then(|v| v.as_u32()) {
        None => violations.push(SchemaViolation::new(key, "required u32 key is missing")),
        Some(0) => violations.push(SchemaViolation::new(key, "value must be > 0")),
        Some(_) => {}
    }
}

/// Check that a key is merely present (any type).
fn require_present(metadata: &MetadataStore, key: &str, violations: &mut Vec<SchemaViolation>) {
    if metadata.get(key).is_none() {
        violations.push(SchemaViolation::new(key, "required key is missing"));
    }
}

/// Validate the four keys common to all LLM architectures.
fn validate_common(arch: &str, metadata: &MetadataStore, violations: &mut Vec<SchemaViolation>) {
    require_u32_positive(metadata, &format!("llm.{arch}.block_count"), violations);
    require_u32_positive(
        metadata,
        &format!("llm.{arch}.embedding_length"),
        violations,
    );
    require_u32_positive(metadata, &format!("llm.{arch}.head_count"), violations);
    require_present(metadata, &format!("llm.{arch}.head_count_kv"), violations);
}

// ─── Built-in arch validators ─────────────────────────────────────────────────

/// Schema validator for LLaMA architecture models.
pub struct LlamaSchemaValidator;

impl SchemaValidator for LlamaSchemaValidator {
    fn architecture_name(&self) -> &str {
        "llama"
    }

    fn validate(&self, metadata: &MetadataStore) -> Vec<SchemaViolation> {
        let mut v = Vec::new();
        validate_common("llama", metadata, &mut v);
        // LLaMA-specific: RoPE frequency base should be present.
        require_present(metadata, "llm.llama.rope.freq_base", &mut v);
        v
    }
}

/// Schema validator for Qwen3 architecture models.
pub struct Qwen3SchemaValidator;

impl SchemaValidator for Qwen3SchemaValidator {
    fn architecture_name(&self) -> &str {
        "qwen3"
    }

    fn validate(&self, metadata: &MetadataStore) -> Vec<SchemaViolation> {
        let mut v = Vec::new();
        validate_common("qwen3", metadata, &mut v);
        v
    }
}

/// Schema validator for Mistral architecture models.
pub struct MistralSchemaValidator;

impl SchemaValidator for MistralSchemaValidator {
    fn architecture_name(&self) -> &str {
        "mistral"
    }

    fn validate(&self, metadata: &MetadataStore) -> Vec<SchemaViolation> {
        let mut v = Vec::new();
        validate_common("mistral", metadata, &mut v);
        v
    }
}

/// Schema validator for Gemma architecture models.
pub struct GemmaSchemaValidator;

impl SchemaValidator for GemmaSchemaValidator {
    fn architecture_name(&self) -> &str {
        "gemma"
    }

    fn validate(&self, metadata: &MetadataStore) -> Vec<SchemaViolation> {
        let mut v = Vec::new();
        validate_common("gemma", metadata, &mut v);
        v
    }
}

/// Schema validator for Phi architecture models.
pub struct PhiSchemaValidator;

impl SchemaValidator for PhiSchemaValidator {
    fn architecture_name(&self) -> &str {
        "phi"
    }

    fn validate(&self, metadata: &MetadataStore) -> Vec<SchemaViolation> {
        let mut v = Vec::new();
        validate_common("phi", metadata, &mut v);
        v
    }
}

/// Schema validator for Command-R architecture models.
pub struct CommandRSchemaValidator;

impl SchemaValidator for CommandRSchemaValidator {
    fn architecture_name(&self) -> &str {
        "command-r"
    }

    fn validate(&self, metadata: &MetadataStore) -> Vec<SchemaViolation> {
        let mut v = Vec::new();
        validate_common("command-r", metadata, &mut v);
        v
    }
}

/// Schema validator for StarCoder architecture models.
pub struct StarCoderSchemaValidator;

impl SchemaValidator for StarCoderSchemaValidator {
    fn architecture_name(&self) -> &str {
        "starcoder"
    }

    fn validate(&self, metadata: &MetadataStore) -> Vec<SchemaViolation> {
        let mut v = Vec::new();
        validate_common("starcoder", metadata, &mut v);
        v
    }
}

// ─── Dispatch ────────────────────────────────────────────────────────────────

/// Validate metadata against all registered architecture validators.
///
/// Dispatches based on the `general.architecture` value.  If the architecture
/// is unknown, a single `SchemaViolation` is returned noting this.
/// If the key is absent entirely, that violation is also returned.
pub fn validate_schema(metadata: &MetadataStore) -> Vec<SchemaViolation> {
    let arch = match metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
    {
        Some(a) => a.to_string(),
        None => {
            return vec![SchemaViolation::new(
                "general.architecture",
                "required key is missing",
            )];
        }
    };

    let validators: &[&dyn SchemaValidator] = &[
        &LlamaSchemaValidator,
        &Qwen3SchemaValidator,
        &MistralSchemaValidator,
        &GemmaSchemaValidator,
        &PhiSchemaValidator,
        &CommandRSchemaValidator,
        &StarCoderSchemaValidator,
    ];

    for validator in validators {
        if validator.architecture_name() == arch {
            return validator.validate(metadata);
        }
    }

    vec![SchemaViolation::new(
        "general.architecture",
        format!("unknown architecture '{arch}'; no schema validator registered"),
    )]
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{MetadataStore, MetadataValue};

    fn store_with_full_llama() -> MetadataStore {
        let mut m = MetadataStore::new();
        m.insert(
            "general.architecture".to_string(),
            MetadataValue::String("llama".to_string()),
        );
        m.insert(
            "llm.llama.block_count".to_string(),
            MetadataValue::Uint32(32),
        );
        m.insert(
            "llm.llama.embedding_length".to_string(),
            MetadataValue::Uint32(4096),
        );
        m.insert(
            "llm.llama.head_count".to_string(),
            MetadataValue::Uint32(32),
        );
        m.insert(
            "llm.llama.head_count_kv".to_string(),
            MetadataValue::Uint32(8),
        );
        m.insert(
            "llm.llama.rope.freq_base".to_string(),
            MetadataValue::Float32(10000.0),
        );
        m
    }

    #[test]
    fn test_llama_full_schema_passes() {
        let m = store_with_full_llama();
        let violations = validate_schema(&m);
        assert!(
            violations.is_empty(),
            "expected no violations, got: {violations:?}"
        );
    }

    #[test]
    fn test_llama_missing_block_count() {
        let mut m = store_with_full_llama();
        // Remove block_count
        let mut store2 = MetadataStore::new();
        // rebuild without block_count
        store2.insert(
            "general.architecture".to_string(),
            MetadataValue::String("llama".to_string()),
        );
        store2.insert(
            "llm.llama.embedding_length".to_string(),
            MetadataValue::Uint32(4096),
        );
        store2.insert(
            "llm.llama.head_count".to_string(),
            MetadataValue::Uint32(32),
        );
        store2.insert(
            "llm.llama.head_count_kv".to_string(),
            MetadataValue::Uint32(8),
        );
        store2.insert(
            "llm.llama.rope.freq_base".to_string(),
            MetadataValue::Float32(10000.0),
        );
        // Silence unused warning
        let _ = &mut m;

        let violations = validate_schema(&store2);
        let keys: Vec<&str> = violations.iter().map(|v| v.key.as_str()).collect();
        assert!(
            keys.contains(&"llm.llama.block_count"),
            "expected block_count violation, got: {violations:?}"
        );
    }

    #[test]
    fn test_llama_zero_embedding_length() {
        let mut m = store_with_full_llama();
        m.insert(
            "llm.llama.embedding_length".to_string(),
            MetadataValue::Uint32(0),
        );
        let violations = validate_schema(&m);
        let keys: Vec<&str> = violations.iter().map(|v| v.key.as_str()).collect();
        assert!(
            keys.contains(&"llm.llama.embedding_length"),
            "expected embedding_length violation, got: {violations:?}"
        );
    }

    #[test]
    fn test_missing_architecture_key() {
        let m = MetadataStore::new();
        let violations = validate_schema(&m);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].key, "general.architecture");
    }

    #[test]
    fn test_unknown_architecture_returns_violation() {
        let mut m = MetadataStore::new();
        m.insert(
            "general.architecture".to_string(),
            MetadataValue::String("mystery-arch-9000".to_string()),
        );
        let violations = validate_schema(&m);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].key, "general.architecture");
    }
}

//! Model configuration extracted from GGUF metadata.

use oxillama_gguf::{GgufTensorType, MetadataStore};

use crate::error::{ArchError, ArchResult};

/// Model configuration parsed from GGUF metadata.
///
/// Contains all architecture-level hyperparameters needed to construct
/// and run a model. Populated from GGUF KV entries like
/// `general.architecture`, `llama.embedding_length`, etc.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Architecture identifier (e.g., "llama", "qwen3", "mistral").
    pub architecture: String,
    /// Model name or path.
    pub model_name: String,
    /// Hidden size / embedding dimension.
    pub hidden_size: usize,
    /// Intermediate size (FFN hidden dim, e.g., for SwiGLU).
    pub intermediate_size: usize,
    /// Number of transformer layers (blocks).
    pub num_layers: usize,
    /// Number of attention heads (query).
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA; equals num_attention_heads if MHA).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context / sequence length.
    pub max_context_length: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    pub rope_freq_base: f32,
    /// Predominant quantization type for weights.
    pub quant_type: Option<GgufTensorType>,
    /// Whether the model uses bias in attention projections.
    pub attention_bias: bool,
    /// Whether the model uses bias in FFN layers.
    pub ffn_bias: bool,
    /// Activation function name (e.g., "silu", "gelu", "swiglu").
    pub activation: String,
    /// Sliding window size for local attention (None = full causal attention).
    pub sliding_window: Option<usize>,
    /// Logit scaling factor (used by Command-R and similar; 1.0 = no scaling).
    pub logit_scale: f32,
    /// Number of MoE experts (0 = standard dense FFN, no MoE).
    pub num_experts: usize,
    /// Number of experts used per token (top-K routing; 0 = no MoE).
    pub num_experts_used: usize,
}

impl ModelConfig {
    /// Construct a `ModelConfig` from GGUF metadata.
    ///
    /// Reads standard GGUF keys like `general.architecture`,
    /// `{arch}.embedding_length`, `{arch}.block_count`, etc.
    pub fn from_metadata(metadata: &MetadataStore) -> ArchResult<Self> {
        let architecture = metadata
            .get_string("general.architecture")
            .map(String::from)
            .map_err(|_| ArchError::UnknownArchitecture {
                arch_id: "<missing>".to_string(),
            })?;

        let arch = &architecture;
        let model_name = metadata
            .get_string("general.name")
            .unwrap_or("unknown")
            .to_string();

        let hidden_size = metadata
            .get_u32(&format!("{arch}.embedding_length"))
            .map(|v| v as usize)
            .unwrap_or(4096);

        let intermediate_size = metadata
            .get_u32(&format!("{arch}.feed_forward_length"))
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4);

        let num_layers = metadata
            .get_u32(&format!("{arch}.block_count"))
            .map(|v| v as usize)
            .unwrap_or(32);

        let num_attention_heads = metadata
            .get_u32(&format!("{arch}.attention.head_count"))
            .map(|v| v as usize)
            .unwrap_or(32);

        let num_kv_heads = metadata
            .get_u32(&format!("{arch}.attention.head_count_kv"))
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let head_dim = hidden_size.checked_div(num_attention_heads).unwrap_or(128);

        let vocab_size = metadata
            .get_u32(&format!("{arch}.vocab_size"))
            .or_else(|_| metadata.get_u32("tokenizer.ggml.tokens.length"))
            .map(|v| v as usize)
            .unwrap_or(32000);

        let max_context_length = metadata
            .get_u32(&format!("{arch}.context_length"))
            .map(|v| v as usize)
            .unwrap_or(4096);

        let rms_norm_eps = metadata
            .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let rope_freq_base = metadata
            .get_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10000.0);

        let sliding_window = metadata
            .get_u32(&format!("{arch}.attention.sliding_window"))
            .ok()
            .map(|v| v as usize);

        let logit_scale = metadata
            .get_f32(&format!("{arch}.logit_scale"))
            .unwrap_or(1.0);

        let num_experts = metadata
            .get_u32(&format!("{arch}.expert_count"))
            .map(|v| v as usize)
            .unwrap_or(0);

        let num_experts_used = metadata
            .get_u32(&format!("{arch}.expert_used_count"))
            .map(|v| v as usize)
            .unwrap_or(0);

        Ok(Self {
            architecture,
            model_name,
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_context_length,
            rms_norm_eps,
            rope_freq_base,
            quant_type: None,
            attention_bias: false,
            ffn_bias: false,
            activation: "silu".to_string(),
            sliding_window,
            logit_scale,
            num_experts,
            num_experts_used,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::MetadataValue;

    fn minimal_store(arch: &str) -> MetadataStore {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String(arch.to_string()),
        );
        store
    }

    #[test]
    fn test_from_metadata_requires_architecture() {
        let store = MetadataStore::new();
        let result = ModelConfig::from_metadata(&store);
        assert!(result.is_err(), "missing architecture should return error");
    }

    #[test]
    fn test_from_metadata_minimal_succeeds() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("minimal store should succeed");
        assert_eq!(cfg.architecture, "llama");
    }

    #[test]
    fn test_from_metadata_default_model_name() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.model_name, "unknown");
    }

    #[test]
    fn test_from_metadata_custom_name() {
        let mut store = minimal_store("llama");
        store.insert(
            "general.name".to_string(),
            MetadataValue::String("MyModel".to_string()),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.model_name, "MyModel");
    }

    #[test]
    fn test_from_metadata_default_hidden_size() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_from_metadata_custom_hidden_size() {
        let mut store = minimal_store("llama");
        store.insert(
            "llama.embedding_length".to_string(),
            MetadataValue::Uint32(2048),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.hidden_size, 2048);
    }

    #[test]
    fn test_from_metadata_default_intermediate_size_is_4x_hidden() {
        let mut store = minimal_store("llama");
        store.insert(
            "llama.embedding_length".to_string(),
            MetadataValue::Uint32(1024),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        // No feed_forward_length set → defaults to hidden_size * 4
        assert_eq!(cfg.intermediate_size, 4096);
    }

    #[test]
    fn test_from_metadata_custom_intermediate_size() {
        let mut store = minimal_store("llama");
        store.insert(
            "llama.embedding_length".to_string(),
            MetadataValue::Uint32(4096),
        );
        store.insert(
            "llama.feed_forward_length".to_string(),
            MetadataValue::Uint32(11008),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.intermediate_size, 11008);
    }

    #[test]
    fn test_from_metadata_default_num_layers() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.num_layers, 32);
    }

    #[test]
    fn test_from_metadata_custom_num_layers() {
        let mut store = minimal_store("llama");
        store.insert("llama.block_count".to_string(), MetadataValue::Uint32(16));
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.num_layers, 16);
    }

    #[test]
    fn test_from_metadata_default_attention_heads() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.num_attention_heads, 32);
        // num_kv_heads defaults to num_attention_heads when not set
        assert_eq!(cfg.num_kv_heads, 32);
    }

    #[test]
    fn test_from_metadata_gqa_kv_heads() {
        let mut store = minimal_store("llama");
        store.insert(
            "llama.attention.head_count".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert(
            "llama.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(8),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
    }

    #[test]
    fn test_from_metadata_head_dim_computed() {
        let mut store = minimal_store("llama");
        store.insert(
            "llama.embedding_length".to_string(),
            MetadataValue::Uint32(4096),
        );
        store.insert(
            "llama.attention.head_count".to_string(),
            MetadataValue::Uint32(32),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.head_dim, 128); // 4096 / 32
    }

    #[test]
    fn test_from_metadata_default_vocab_size() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.vocab_size, 32000);
    }

    #[test]
    fn test_from_metadata_custom_vocab_size() {
        let mut store = minimal_store("llama");
        store.insert(
            "llama.vocab_size".to_string(),
            MetadataValue::Uint32(128256),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.vocab_size, 128256);
    }

    #[test]
    fn test_from_metadata_default_context_length() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.max_context_length, 4096);
    }

    #[test]
    fn test_from_metadata_default_rms_norm_eps() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert!(
            (cfg.rms_norm_eps - 1e-5).abs() < 1e-10,
            "default eps should be 1e-5"
        );
    }

    #[test]
    fn test_from_metadata_default_rope_freq_base() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert!(
            (cfg.rope_freq_base - 10000.0).abs() < 1.0,
            "default rope_freq_base should be 10000"
        );
    }

    #[test]
    fn test_from_metadata_logit_scale_default_is_one() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert!(
            (cfg.logit_scale - 1.0).abs() < 1e-7,
            "default logit_scale should be 1.0, got {}",
            cfg.logit_scale
        );
    }

    #[test]
    fn test_from_metadata_custom_logit_scale() {
        let mut store = minimal_store("command-r");
        store.insert(
            "command-r.logit_scale".to_string(),
            MetadataValue::Float32(0.0625),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert!(
            (cfg.logit_scale - 0.0625).abs() < 1e-7,
            "custom logit_scale"
        );
    }

    #[test]
    fn test_from_metadata_sliding_window_absent_is_none() {
        let store = minimal_store("mistral");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert!(
            cfg.sliding_window.is_none(),
            "sliding_window should default to None"
        );
    }

    #[test]
    fn test_from_metadata_sliding_window_present() {
        let mut store = minimal_store("mistral");
        store.insert(
            "mistral.attention.sliding_window".to_string(),
            MetadataValue::Uint32(4096),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.sliding_window, Some(4096));
    }

    #[test]
    fn test_from_metadata_activation_defaults_to_silu() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.activation, "silu");
    }

    #[test]
    fn test_from_metadata_quant_type_defaults_to_none() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert!(cfg.quant_type.is_none());
    }

    #[test]
    fn test_from_metadata_bias_defaults_to_false() {
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert!(!cfg.attention_bias);
        assert!(!cfg.ffn_bias);
    }

    #[test]
    fn test_from_metadata_qwen3_architecture() {
        let store = minimal_store("qwen3");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.architecture, "qwen3");
    }

    #[test]
    fn test_from_metadata_default_num_experts_is_zero() {
        // Dense models have no expert_count metadata → defaults to 0.
        let store = minimal_store("llama");
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(
            cfg.num_experts, 0,
            "default num_experts should be 0 (dense model)"
        );
        assert_eq!(
            cfg.num_experts_used, 0,
            "default num_experts_used should be 0 (dense model)"
        );
    }

    #[test]
    fn test_from_metadata_custom_num_experts_mixtral() {
        // Mixtral-style MoE: 8 experts, top-2 routing.
        let mut store = minimal_store("llama");
        store.insert("llama.expert_count".to_string(), MetadataValue::Uint32(8));
        store.insert(
            "llama.expert_used_count".to_string(),
            MetadataValue::Uint32(2),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.num_experts, 8, "num_experts should be 8 for Mixtral");
        assert_eq!(
            cfg.num_experts_used, 2,
            "num_experts_used should be 2 (top-2)"
        );
    }
}

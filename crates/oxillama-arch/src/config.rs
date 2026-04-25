//! Model configuration extracted from GGUF metadata.

use oxillama_gguf::{GgufTensorType, MetadataStore};

use crate::common::rope::RopeScalingType;
use crate::error::{ArchError, ArchResult};

/// DeepSeek-V2/V3 specific configuration for Multi-head Latent Attention (MLA)
/// and Mixture-of-Experts (MoE) layers.
///
/// Populated from GGUF KV entries prefixed with `deepseek2.*`.
#[derive(Debug, Clone)]
pub struct DeepSeekConfig {
    /// Rank of the Q low-rank projection (compressed Q latent dimension).
    pub q_lora_rank: usize,
    /// Rank of the KV low-rank projection (compressed KV latent dimension).
    pub kv_lora_rank: usize,
    /// Head dimension for the nope (no-positional-encoding) part of Q and K.
    pub qk_nope_head_dim: usize,
    /// Head dimension for the rope (positional-encoding) part of Q and K.
    pub qk_rope_head_dim: usize,
    /// Head dimension for V.
    pub v_head_dim: usize,
    /// Number of shared experts (always active, applied every token).
    pub n_shared_experts: usize,
    /// Total number of routed experts.
    pub n_routed_experts: usize,
    /// Number of experts activated per token via top-k routing.
    pub top_k_routed: usize,
    /// Intermediate size for each shared expert's SwiGLU FFN.
    pub shared_expert_intermediate_size: usize,
    /// Scaling factor for the routing score normalisation (if sigmoid mode).
    pub routed_scaling_factor: f32,
    /// Number of leading dense layers before MoE layers begin.
    ///
    /// Layers `0..first_k_dense_replace` use a standard dense SwiGLU FFN;
    /// layers `first_k_dense_replace..num_layers` use the DeepSeek sparse MoE FFN.
    pub first_k_dense_replace: usize,
}

impl DeepSeekConfig {
    /// Parse a `DeepSeekConfig` from GGUF metadata.
    ///
    /// Reads `deepseek2.*` keys with sensible defaults for any missing entries.
    ///
    /// # Errors
    /// Returns `ArchError::UnknownArchitecture` if `general.architecture` is absent
    /// (delegated to the outer `ModelConfig::from_metadata` call).
    pub fn from_metadata(metadata: &MetadataStore, hidden_size: usize) -> Self {
        let q_lora_rank = metadata
            .get_u32("deepseek2.attention.q_lora_rank")
            .map(|v| v as usize)
            .unwrap_or(1536);

        let kv_lora_rank = metadata
            .get_u32("deepseek2.attention.kv_lora_rank")
            .map(|v| v as usize)
            .unwrap_or(512);

        let qk_nope_head_dim = metadata
            .get_u32("deepseek2.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(128);

        let qk_rope_head_dim = metadata
            .get_u32("deepseek2.attention.rope_head_dim")
            .map(|v| v as usize)
            .unwrap_or(64);

        let v_head_dim = metadata
            .get_u32("deepseek2.attention.value_length")
            .map(|v| v as usize)
            .unwrap_or(128);

        let n_shared_experts = metadata
            .get_u32("deepseek2.expert_shared_count")
            .map(|v| v as usize)
            .unwrap_or(1);

        let n_routed_experts = metadata
            .get_u32("deepseek2.expert_count")
            .map(|v| v as usize)
            .unwrap_or(64);

        let top_k_routed = metadata
            .get_u32("deepseek2.expert_used_count")
            .map(|v| v as usize)
            .unwrap_or(6);

        let shared_expert_intermediate_size = metadata
            .get_u32("deepseek2.expert_shared_feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 2);

        let routed_scaling_factor = metadata
            .get_f32("deepseek2.expert_weights_scale")
            .unwrap_or(1.0);

        let first_k_dense_replace = metadata
            .get_u32("deepseek2.leading_dense_block_count")
            .map(|v| v as usize)
            .unwrap_or(1);

        Self {
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            n_shared_experts,
            n_routed_experts,
            top_k_routed,
            shared_expert_intermediate_size,
            routed_scaling_factor,
            first_k_dense_replace,
        }
    }
}

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
    /// Sliding-window attention span in tokens, unified across arch families.
    /// `None` means global attention on all layers.
    /// Alias kept alongside `sliding_window` for cross-arch consistency;
    /// both are populated from the same GGUF key.
    pub swa_window: Option<u32>,
    /// Interleaved SWA pattern: `true` means SWA alternates with global
    /// attention (Gemma style). `false` means all layers use `swa_window`
    /// (Mistral style).
    pub swa_interleaved: bool,
    /// RoPE scaling strategy for extending context beyond the training length.
    pub rope_scaling_type: RopeScalingType,
    /// RoPE scaling factor (`1.0` = no scaling).
    pub rope_scaling_factor: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: String::new(),
            model_name: String::new(),
            hidden_size: 4096,
            intermediate_size: 16384,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            vocab_size: 32000,
            max_context_length: 4096,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            quant_type: None,
            attention_bias: false,
            ffn_bias: false,
            activation: "silu".to_string(),
            sliding_window: None,
            logit_scale: 1.0,
            num_experts: 0,
            num_experts_used: 0,
            swa_window: None,
            swa_interleaved: false,
            rope_scaling_type: RopeScalingType::Standard,
            rope_scaling_factor: 1.0,
        }
    }
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

        let swa_window = sliding_window.map(|v| v as u32);

        // Gemma-family models use interleaved SWA (even layers = global).
        // Detect by architecture name prefix.
        let swa_interleaved = architecture.starts_with("gemma");

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

        let rope_scaling_factor = metadata
            .get_f32(&format!("{arch}.rope.scaling.factor"))
            .unwrap_or(1.0);

        let rope_scaling_type = metadata
            .get_string(&format!("{arch}.rope.scaling.type"))
            .map(|s| match s {
                "linear" => RopeScalingType::Linear,
                "yarn" => RopeScalingType::Yarn,
                _ => RopeScalingType::Standard,
            })
            .unwrap_or(RopeScalingType::Standard);

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
            swa_window,
            swa_interleaved,
            rope_scaling_type,
            rope_scaling_factor,
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

#[cfg(test)]
mod swa_tests {
    use super::*;
    use crate::common::effective_attention_span;

    #[test]
    fn test_global_attention() {
        let config = ModelConfig {
            swa_window: None,
            ..ModelConfig::default()
        };
        assert_eq!(effective_attention_span(&config, 0), u32::MAX);
        assert_eq!(effective_attention_span(&config, 5), u32::MAX);
    }

    #[test]
    fn test_mistral_swa_all_layers() {
        let config = ModelConfig {
            swa_window: Some(4096),
            swa_interleaved: false,
            ..ModelConfig::default()
        };
        assert_eq!(effective_attention_span(&config, 0), 4096);
        assert_eq!(effective_attention_span(&config, 3), 4096);
    }

    #[test]
    fn test_gemma_interleaved_swa() {
        let config = ModelConfig {
            swa_window: Some(4096),
            swa_interleaved: true,
            ..ModelConfig::default()
        };
        // Layer 0 is global (interleaved, even index)
        assert_eq!(effective_attention_span(&config, 0), u32::MAX);
        // Layer 1 is sliding window
        assert_eq!(effective_attention_span(&config, 1), 4096);
        // Layer 2 is global again
        assert_eq!(effective_attention_span(&config, 2), u32::MAX);
    }

    #[test]
    fn test_mistral_swa_from_metadata() {
        use oxillama_gguf::MetadataValue;
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("mistral".to_string()),
        );
        store.insert(
            "mistral.attention.sliding_window".to_string(),
            MetadataValue::Uint32(4096),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.swa_window, Some(4096));
        assert!(!cfg.swa_interleaved, "mistral is not interleaved");
    }

    #[test]
    fn test_gemma_swa_interleaved_from_metadata() {
        use oxillama_gguf::MetadataValue;
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("gemma".to_string()),
        );
        store.insert(
            "gemma.attention.sliding_window".to_string(),
            MetadataValue::Uint32(2048),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.swa_window, Some(2048));
        assert!(cfg.swa_interleaved, "gemma should be interleaved");
    }
}

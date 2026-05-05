//! Qwen2-VL multimodal architecture.
//!
//! Qwen2-VL is Alibaba's second-generation Vision-Language Model.
//! Key differences from LLaVA:
//! - **Native ViT** (not CLIP): no CLS token, dynamic resolution, window attention.
//! - **M-RoPE**: joint text+vision position encoding partitioning head_dim into
//!   three axes (time, height, width).
//! - **MM merger**: 2×2 spatial patch blocks collapsed to a single LLM token.
//!
//! ## Feature flag
//! Compiled when `features = ["qwen2-vl"]` is enabled (default-on).
//!
//! ## Architecture ID
//! Registered under the GGUF key `"qwen2vl"`.

pub mod model;
pub mod vision;

pub use model::{load_qwen2vl_from_gguf, MmMerger, Qwen2Layer, Qwen2VlModel};
pub use vision::{Qwen2VlVisionEncoder, VisionBlock};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// Qwen2-VL architecture plugin.
///
/// Registered under the GGUF architecture ID `"qwen2vl"`.
pub struct Qwen2VlArchitecture;

impl Qwen2VlArchitecture {
    /// Create a new architecture plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Qwen2VlArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for Qwen2VlArchitecture {
    fn arch_id(&self) -> &str {
        "qwen2vl"
    }

    fn build(
        &self,
        config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        if config.num_attention_heads == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "num_attention_heads".to_string(),
                expected: ">0".to_string(),
                got: "0".to_string(),
            });
        }
        if config.hidden_size == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "hidden_size".to_string(),
                expected: ">0".to_string(),
                got: "0".to_string(),
            });
        }

        // Full loading is handled via load_qwen2vl_from_gguf(GgufModel, config).
        // This registry path validates config parameters only.
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use load_qwen2vl_from_gguf for full loading)".to_string(),
        })
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        let mut patterns = vec![
            // ── LLM backbone ────────────────────────────────────────────────
            TensorNamePattern {
                pattern: "token_embd.weight".to_string(),
                description: "Token embedding matrix".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.weight".to_string(),
                description: "Final RMSNorm".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head projection".to_string(),
                required: true,
            },
            // ── MM Merger ────────────────────────────────────────────────────
            TensorNamePattern {
                pattern: "mm.0.weight".to_string(),
                description: "MM merger projection weight [llm_hidden, 4*vis_hidden]".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "mm.0.bias".to_string(),
                description: "MM merger projection bias".to_string(),
                required: false,
            },
            // ── Vision encoder ───────────────────────────────────────────────
            TensorNamePattern {
                pattern: "v.patch_embd.weight".to_string(),
                description: "ViT patch embedding weight".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "v.patch_embd.bias".to_string(),
                description: "ViT patch embedding bias".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "v.post_ln.weight".to_string(),
                description: "ViT post-normalization RMSNorm weight".to_string(),
                required: false,
            },
        ];

        // ── Per-layer LLM backbone tensors ────────────────────────────────
        let layer_tensors = [
            ("blk.{i}.attn_norm.weight", "Pre-attention RMSNorm", true),
            ("blk.{i}.attn_q.weight", "Query projection weight", true),
            ("blk.{i}.attn_q.bias", "Query projection bias", false),
            ("blk.{i}.attn_k.weight", "Key projection weight", true),
            ("blk.{i}.attn_k.bias", "Key projection bias", false),
            ("blk.{i}.attn_v.weight", "Value projection weight", true),
            ("blk.{i}.attn_v.bias", "Value projection bias", false),
            (
                "blk.{i}.attn_output.weight",
                "Attention output projection",
                true,
            ),
            ("blk.{i}.attn_output.bias", "Attention output bias", false),
            ("blk.{i}.ffn_norm.weight", "Pre-FFN RMSNorm", true),
            ("blk.{i}.ffn_gate.weight", "FFN gate projection", true),
            ("blk.{i}.ffn_up.weight", "FFN up projection", true),
            ("blk.{i}.ffn_down.weight", "FFN down projection", true),
        ];

        for (pat, desc, required) in layer_tensors {
            patterns.push(TensorNamePattern {
                pattern: pat.to_string(),
                description: desc.to_string(),
                required,
            });
        }

        // ── Per-layer vision encoder tensors ──────────────────────────────
        let vis_layer_tensors = [
            ("v.blk.{i}.attn_norm.weight", "ViT pre-attention RMSNorm"),
            ("v.blk.{i}.ffn_norm.weight", "ViT pre-FFN RMSNorm"),
            ("v.blk.{i}.attn_q.weight", "ViT query projection"),
            ("v.blk.{i}.attn_k.weight", "ViT key projection"),
            ("v.blk.{i}.attn_v.weight", "ViT value projection"),
            (
                "v.blk.{i}.attn_out.weight",
                "ViT attention output projection",
            ),
            ("v.blk.{i}.ffn_up.weight", "ViT FFN up projection"),
            ("v.blk.{i}.ffn_down.weight", "ViT FFN down projection"),
        ];

        for (pat, desc) in vis_layer_tensors {
            patterns.push(TensorNamePattern {
                pattern: pat.to_string(),
                description: desc.to_string(),
                required: false,
            });
        }

        patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ArchitectureRegistry;
    use oxillama_gguf::test_utils::build_minimal_qwen2vl_gguf;
    use oxillama_gguf::{GgufModel, MetadataStore, MetadataValue, TensorStore};

    fn make_config() -> ModelConfig {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("qwen2vl".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal qwen2vl config should parse")
    }

    /// arch_id() must return "qwen2vl".
    #[test]
    fn test_arch_id() {
        let arch = Qwen2VlArchitecture::new();
        assert_eq!(arch.arch_id(), "qwen2vl");
    }

    /// tensor_names() must be non-empty.
    #[test]
    fn test_tensor_names_non_empty() {
        let arch = Qwen2VlArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty());
    }

    /// tensor_names() must include token_embd.weight.
    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = Qwen2VlArchitecture::new();
        let names = arch.tensor_names();
        let pats: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(pats.contains(&"token_embd.weight"));
    }

    /// tensor_names() must contain mm.0.weight (MM merger) and vision patterns.
    #[test]
    fn qwen2vl_tensor_names_complete() {
        let arch = Qwen2VlArchitecture::new();
        let names = arch.tensor_names();
        let pats: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();

        // Required backbone patterns
        assert!(
            pats.contains(&"token_embd.weight"),
            "missing token_embd.weight"
        );
        assert!(
            pats.contains(&"output_norm.weight"),
            "missing output_norm.weight"
        );
        assert!(pats.contains(&"output.weight"), "missing output.weight");

        // MM merger
        assert!(pats.contains(&"mm.0.weight"), "missing mm.0.weight");

        // Vision encoder
        assert!(
            pats.iter().any(|p| p.starts_with("v.")),
            "missing vision encoder patterns (v.*)"
        );

        // Per-layer backbone patterns
        assert!(
            pats.iter().any(|p| p.contains("blk.{i}.attn_q.weight")),
            "missing per-layer attn_q.weight"
        );
        assert!(
            pats.iter().any(|p| p.contains("blk.{i}.ffn_gate.weight")),
            "missing per-layer ffn_gate.weight"
        );
    }

    /// build() with zero heads must return ConfigMismatch.
    #[test]
    fn test_build_zero_heads_errors() {
        let arch = Qwen2VlArchitecture::new();
        let mut config = make_config();
        config.num_attention_heads = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(matches!(result, Err(ArchError::ConfigMismatch { .. })));
    }

    /// build() with zero hidden_size must return ConfigMismatch.
    #[test]
    fn test_build_zero_hidden_errors() {
        let arch = Qwen2VlArchitecture::new();
        let mut config = make_config();
        config.hidden_size = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(matches!(result, Err(ArchError::ConfigMismatch { .. })));
    }

    /// build() with valid config but empty tensors must return MissingTensor.
    #[test]
    fn test_build_valid_config_missing_tensor() {
        let arch = Qwen2VlArchitecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(matches!(result, Err(ArchError::MissingTensor { .. })));
    }

    /// ArchitectureRegistry::with_builtins() must contain "qwen2vl".
    #[test]
    fn qwen2vl_registry_lookup() {
        let reg = ArchitectureRegistry::with_builtins();
        assert!(
            reg.contains("qwen2vl"),
            "registry must contain qwen2vl architecture"
        );
        let arch = reg.get("qwen2vl").expect("get qwen2vl must succeed");
        assert_eq!(arch.arch_id(), "qwen2vl");
    }

    /// Full GGUF fixture round-trip: forward pass must produce finite logits.
    #[test]
    fn qwen2vl_forward_shape_and_finite() {
        use crate::traits::KvCacheAccess;
        use oxillama_gguf::MetadataValue;

        let bytes = build_minimal_qwen2vl_gguf();
        let gguf = GgufModel::from_bytes(bytes).expect("parse qwen2vl gguf");

        let mut meta = MetadataStore::new();
        meta.insert(
            "general.architecture".to_string(),
            MetadataValue::String("qwen2vl".to_string()),
        );
        meta.insert(
            "qwen2vl.embedding_length".to_string(),
            MetadataValue::Uint32(32),
        );
        meta.insert(
            "qwen2vl.feed_forward_length".to_string(),
            MetadataValue::Uint32(64),
        );
        meta.insert("qwen2vl.block_count".to_string(), MetadataValue::Uint32(1));
        meta.insert(
            "qwen2vl.attention.head_count".to_string(),
            MetadataValue::Uint32(2),
        );
        meta.insert(
            "qwen2vl.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(2),
        );
        meta.insert(
            "qwen2vl.context_length".to_string(),
            MetadataValue::Uint32(128),
        );
        meta.insert("qwen2vl.vocab_size".to_string(), MetadataValue::Uint32(32));

        let config = crate::config::ModelConfig::from_metadata(&meta).expect("config");
        let mut model = crate::qwen2_vl::model::load_qwen2vl_from_gguf(&gguf, &config)
            .expect("load qwen2vl model");

        struct FlatKv {
            keys: Vec<Vec<f32>>,
            vals: Vec<Vec<f32>>,
            pos: usize,
        }
        impl KvCacheAccess for FlatKv {
            fn store_kv(&mut self, layer: usize, k: &[f32], v: &[f32]) -> ArchResult<()> {
                while self.keys.len() <= layer {
                    self.keys.push(Vec::new());
                    self.vals.push(Vec::new());
                }
                self.keys[layer].extend_from_slice(k);
                self.vals[layer].extend_from_slice(v);
                Ok(())
            }
            fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
                Ok(self.keys.get(layer).map_or(&[], |v| v.as_slice()))
            }
            fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
                Ok(self.vals.get(layer).map_or(&[], |v| v.as_slice()))
            }
            fn seq_len(&self) -> usize {
                self.pos
            }
            fn advance(&mut self) {
                self.pos += 1;
            }
            fn kv_dim(&self) -> usize {
                0
            }
        }

        let mut kv = FlatKv {
            keys: vec![],
            vals: vec![],
            pos: 0,
        };
        let tokens = [1u32, 2];
        let logits = model.forward(&tokens, &mut kv).expect("forward");

        assert_eq!(logits.len(), 32, "should have vocab_size=32 logits");
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}]={v} must be finite");
        }
    }
}

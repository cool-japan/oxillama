//! MiniCPM-specific configuration parsed from GGUF metadata.
//!
//! MiniCPM (base) is **architecturally identical to LLaMA**: llama.cpp puts
//! `LLM_ARCH_MINICPM` in the same tensor-creation block as `LLM_ARCH_LLAMA`
//! and `LLM_ARCH_GRANITE` (`src/llama-model.cpp`, `load_tensors()`), so it has
//! no MiniCPM-specific tensors at all.  The entire difference is **three
//! scalar hyper-parameters** applied at three points in the graph
//! (`src/models/granite.cpp`, which `LLM_ARCH_MINICPM` dispatches to):
//!
//! | Scalar            | GGUF key                    | Applied |
//! |-------------------|-----------------------------|---------|
//! | `embedding_scale` | `minicpm.embedding_scale`   | once, on the embedded token vector before layer 0 (`llama-graph.cpp`, `build_inp_embd`) |
//! | `residual_scale`  | `minicpm.residual_scale`    | twice per layer, on the attention output and on the FFN output, immediately before each residual add (`granite.cpp::build_layer_ffn`) |
//! | `logit_scale`     | `minicpm.logit_scale`       | once, after the LM head: `logits *= 1.0 / logit_scale` (`granite.cpp`, `ggml_scale(cur, 1.0f / hparams.f_logit_scale)`) |
//!
//! `convert_hf_to_gguf.py::MiniCPMModel::set_gguf_parameters` writes all three
//! keys for every real checkpoint:
//!
//! ```text
//! embedding_scale = hparams["scale_emb"]
//! residual_scale  = scale_depth / sqrt(num_hidden_layers)
//! logit_scale     = hidden_size / dim_model_base
//! ```
//!
//! Note that `hidden_size / dim_model_base` is **`logit_scale`'s** formula —
//! it is *not* `embedding_scale`'s, which is the raw HF `scale_emb` value.
//! An earlier revision of this file documented `embedding_scale` with the
//! `logit_scale` formula and then hard-coded the value to `1.0`, which
//! disabled all three scales.
//!
//! The [`DEFAULT_EMBEDDING_SCALE`] / [`default_residual_scale`] /
//! [`default_logit_scale`] fallbacks reproduce llama.cpp's
//! backward-compatibility defaults for *old* GGUFs that predate those keys
//! (`src/llama-model.cpp`, `case LLM_ARCH_MINICPM:` in `load_hparams()`).

use oxillama_gguf::MetadataStore;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};

/// Canonical GGUF architecture id for MiniCPM (base).
pub const MINICPM_ARCH: &str = "minicpm";

/// llama.cpp's backward-compatible `f_embedding_scale` for MiniCPM GGUFs that
/// predate the `{arch}.embedding_scale` key.
pub const DEFAULT_EMBEDDING_SCALE: f32 = 12.0;

/// llama.cpp's backward-compatible `f_residual_scale`: `1.4 / sqrt(n_layer)`.
///
/// Used only when `{arch}.residual_scale` is absent.
pub fn default_residual_scale(n_layers: usize) -> f32 {
    let n = (n_layers.max(1)) as f32;
    1.4 / n.sqrt()
}

/// llama.cpp's backward-compatible `f_logit_scale`: `256 / n_embd`.
///
/// Mirrors `hparams.n_embd ? (256.0f / float(hparams.n_embd)) : 1.0f`, so a
/// zero `hidden_size` yields `1.0` instead of a division by zero.  Used only
/// when `{arch}.logit_scale` is absent.
///
/// This is deliberately *not* [`ModelConfig::logit_scale`]'s generic default
/// of `1.0`: that field cannot distinguish "key present with value 1.0" from
/// "key absent", and `1.0` is the wrong fallback for MiniCPM.
pub fn default_logit_scale(hidden_size: usize) -> f32 {
    if hidden_size > 0 {
        256.0 / hidden_size as f32
    } else {
        1.0
    }
}

/// MiniCPM-specific hyperparameters extracted from GGUF metadata.
///
/// The generic [`ModelConfig`] fields hold the shared shape parameters; this
/// struct resolves them together with MiniCPM's three scale factors.
#[derive(Debug, Clone)]
pub struct MiniCpmConfig {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Hidden size / model dimension.
    pub hidden_size: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (GQA).
    pub n_kv_heads: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_context_length: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// RoPE base frequency.
    pub rope_freq_base: f32,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Input embedding scale, GGUF `{arch}.embedding_scale`.
    ///
    /// Multiplied into the embedded token vector once, before layer 0.
    /// Falls back to [`DEFAULT_EMBEDDING_SCALE`] when the key is absent.
    pub embedding_scale: f32,
    /// Residual branch scale, GGUF `{arch}.residual_scale`.
    ///
    /// Multiplied into the attention output *and* the FFN output, each
    /// immediately before it is added back to the residual stream.  Falls back
    /// to [`default_residual_scale`] when the key is absent.
    pub residual_scale: f32,
    /// Output logit scale, GGUF `{arch}.logit_scale`.
    ///
    /// The LM head's logits are multiplied by `1.0 / logit_scale`.  Falls back
    /// to [`default_logit_scale`] when the key is absent.
    pub logit_scale: f32,
}

impl MiniCpmConfig {
    /// Parse a [`MiniCpmConfig`] from a [`ModelConfig`] plus the raw GGUF
    /// metadata the three MiniCPM scale keys live in.
    ///
    /// This is the constructor the GGUF loader uses.  The scales are read
    /// straight out of `metadata` rather than through
    /// [`ModelConfig::logit_scale`], whose generic `1.0` default is wrong for
    /// MiniCPM and indistinguishable from a genuine `1.0`.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] when a scale is non-finite, or when
    /// `logit_scale` is zero (the graph divides by it).
    pub fn from_metadata(config: &ModelConfig, metadata: &MetadataStore) -> ArchResult<Self> {
        Self::resolve(config, Some(metadata))
    }

    /// Parse a [`MiniCpmConfig`] from a [`ModelConfig`] alone.
    ///
    /// No GGUF metadata is available here, so all three scales take the
    /// llama.cpp backward-compatibility defaults.  Prefer
    /// [`Self::from_metadata`] whenever the GGUF is on hand.
    ///
    /// # Errors
    ///
    /// See [`Self::from_metadata`].
    pub fn from_model_config(config: &ModelConfig) -> ArchResult<Self> {
        Self::resolve(config, None)
    }

    fn resolve(config: &ModelConfig, metadata: Option<&MetadataStore>) -> ArchResult<Self> {
        let n_heads = config.num_attention_heads.max(1);
        let n_kv_heads = config.num_kv_heads.max(1);
        let n_layers = config.num_layers.max(1);
        let hidden_size = config.hidden_size.max(1);
        let vocab_size = config.vocab_size.max(1);
        let intermediate_size = config.intermediate_size.max(1);
        let max_context_length = config.max_context_length.max(1);
        let head_dim = if config.head_dim > 0 {
            config.head_dim
        } else {
            hidden_size.checked_div(n_heads).unwrap_or(64).max(1)
        };
        let norm_eps = config.rms_norm_eps.max(1e-9);

        let rope_freq_base = if config.rope_freq_base > 0.0 {
            config.rope_freq_base
        } else {
            10_000.0
        };

        // llama.cpp seeds the three scales with backward-compatible defaults
        // and then lets an optional GGUF key override each one
        // (`src/llama-model.cpp`, `case LLM_ARCH_MINICPM:`).
        let embedding_scale =
            read_scale(metadata, config, "embedding_scale").unwrap_or(DEFAULT_EMBEDDING_SCALE);
        let residual_scale = read_scale(metadata, config, "residual_scale")
            .unwrap_or_else(|| default_residual_scale(config.num_layers));
        let logit_scale = read_scale(metadata, config, "logit_scale")
            .unwrap_or_else(|| default_logit_scale(config.hidden_size));

        finite("embedding_scale", embedding_scale)?;
        finite("residual_scale", residual_scale)?;
        finite("logit_scale", logit_scale)?;
        if logit_scale == 0.0 {
            return Err(ArchError::InvalidConfig {
                detail: "minicpm.logit_scale must be non-zero: the LM head output is \
                         multiplied by 1.0 / logit_scale"
                    .to_string(),
            });
        }

        Ok(Self {
            n_layers,
            hidden_size,
            n_heads,
            n_kv_heads,
            intermediate_size,
            vocab_size,
            max_context_length,
            norm_eps,
            rope_freq_base,
            head_dim,
            embedding_scale,
            residual_scale,
            logit_scale,
        })
    }
}

/// Reject a non-finite scale instead of letting `NaN`/`inf` reach the kernels.
fn finite(name: &str, value: f32) -> ArchResult<()> {
    if value.is_finite() {
        return Ok(());
    }
    Err(ArchError::InvalidConfig {
        detail: format!("minicpm.{name} must be finite, got {value}"),
    })
}

/// Look up `{arch}.{suffix}`, falling back to the canonical `minicpm.` prefix.
///
/// llama.cpp spells these keys `"%s.<suffix>"` with the file's own
/// architecture id (`src/llama-arch.cpp`: `LLM_KV_EMBEDDING_SCALE`,
/// `LLM_KV_RESIDUAL_SCALE`, `LLM_KV_LOGIT_SCALE`), so the file's architecture
/// is tried first; the `minicpm.` retry only matters for a `ModelConfig` whose
/// `architecture` field was not populated from the same file.
fn read_scale(metadata: Option<&MetadataStore>, config: &ModelConfig, suffix: &str) -> Option<f32> {
    let metadata = metadata?;
    let arch = if config.architecture.is_empty() {
        MINICPM_ARCH
    } else {
        config.architecture.as_str()
    };
    if let Ok(v) = metadata.get_f32(&format!("{arch}.{suffix}")) {
        return Some(v);
    }
    if arch != MINICPM_ARCH {
        if let Ok(v) = metadata.get_f32(&format!("{MINICPM_ARCH}.{suffix}")) {
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use oxillama_gguf::{MetadataStore, MetadataValue};

    fn make_minicpm_metadata() -> MetadataStore {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("minicpm".to_string()),
        );
        store.insert(
            "minicpm.embedding_length".to_string(),
            MetadataValue::Uint32(2048),
        );
        store.insert("minicpm.block_count".to_string(), MetadataValue::Uint32(40));
        store.insert(
            "minicpm.attention.head_count".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert(
            "minicpm.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert(
            "minicpm.feed_forward_length".to_string(),
            MetadataValue::Uint32(5504),
        );
        store.insert(
            "minicpm.rope.freq_base".to_string(),
            MetadataValue::Float32(10000.0),
        );
        store
    }

    #[test]
    fn test_config_from_metadata() {
        let store = make_minicpm_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = MiniCpmConfig::from_metadata(&model_cfg, &store).expect("minicpm config");

        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.n_layers, 40);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 32);
        assert_eq!(cfg.intermediate_size, 5504);
        assert!(cfg.rope_freq_base > 0.0);
        assert!(cfg.norm_eps > 0.0);
        assert_eq!(cfg.head_dim, 64);
    }

    /// M2: with no scale keys the three fields take llama.cpp's
    /// backward-compatibility defaults, **not** `1.0`.
    #[test]
    fn test_scales_fall_back_to_llama_cpp_defaults() {
        let store = make_minicpm_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = MiniCpmConfig::from_metadata(&model_cfg, &store).expect("minicpm config");

        assert!(
            (cfg.embedding_scale - 12.0).abs() < 1e-6,
            "embedding_scale default is 12.0, got {}",
            cfg.embedding_scale
        );
        let expected_residual = 1.4 / 40.0f32.sqrt();
        assert!(
            (cfg.residual_scale - expected_residual).abs() < 1e-6,
            "residual_scale default is 1.4/sqrt(40), got {}",
            cfg.residual_scale
        );
        assert!(
            (cfg.logit_scale - 256.0 / 2048.0).abs() < 1e-6,
            "logit_scale default is 256/n_embd, got {}",
            cfg.logit_scale
        );
    }

    /// M2: an explicit key overrides the default.
    #[test]
    fn test_scales_read_from_metadata() {
        let mut store = make_minicpm_metadata();
        store.insert(
            "minicpm.embedding_scale".to_string(),
            MetadataValue::Float32(2.0),
        );
        store.insert(
            "minicpm.residual_scale".to_string(),
            MetadataValue::Float32(0.5),
        );
        store.insert(
            "minicpm.logit_scale".to_string(),
            MetadataValue::Float32(4.0),
        );

        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = MiniCpmConfig::from_metadata(&model_cfg, &store).expect("minicpm config");

        assert!((cfg.embedding_scale - 2.0).abs() < 1e-6);
        assert!((cfg.residual_scale - 0.5).abs() < 1e-6);
        assert!((cfg.logit_scale - 4.0).abs() < 1e-6);
    }

    /// Without metadata every scale takes its backward-compatible default.
    #[test]
    fn test_from_model_config_uses_defaults() {
        let store = make_minicpm_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = MiniCpmConfig::from_model_config(&model_cfg).expect("minicpm config");
        assert!((cfg.embedding_scale - 12.0).abs() < 1e-6);
        assert!((cfg.logit_scale - 256.0 / 2048.0).abs() < 1e-6);
    }

    /// `logit_scale == 0` would make the graph divide by zero.
    #[test]
    fn test_zero_logit_scale_is_rejected() {
        let mut store = make_minicpm_metadata();
        store.insert(
            "minicpm.logit_scale".to_string(),
            MetadataValue::Float32(0.0),
        );
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        assert!(matches!(
            MiniCpmConfig::from_metadata(&model_cfg, &store),
            Err(ArchError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn test_config_has_positive_head_dim() {
        let store = make_minicpm_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = MiniCpmConfig::from_metadata(&model_cfg, &store).expect("minicpm config");
        assert!(cfg.head_dim > 0);
    }

    #[test]
    fn test_default_logit_scale_guards_zero_hidden() {
        assert!((default_logit_scale(0) - 1.0).abs() < 1e-9);
        assert!((default_logit_scale(256) - 1.0).abs() < 1e-9);
    }
}

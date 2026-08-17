//! Model configuration extracted from GGUF metadata.

use oxillama_gguf::{GgufTensorType, MetadataStore, MetadataValue, TensorStore};

use crate::common::attention::SwaPattern;
use crate::common::rope::{RopeScalingType, RopeStyle};
use crate::error::{ArchError, ArchResult};

/// The RoPE element-pairing convention llama.cpp assigns to `arch_id`.
///
/// Reproduces `llama_model_rope_type()` in `src/llama-model.cpp`, keyed by the
/// GGUF `general.architecture` string from `LLM_ARCH_NAMES` in
/// `src/llama-arch.cpp`.
///
/// Architectures that do not use RoPE at all (`LLAMA_ROPE_TYPE_NONE`: bloom,
/// mamba, mamba2, jamba, gpt2, mpt, rwkv*, t5, …) and the M-RoPE families
/// (qwen2vl, qwen3vl) are **not** in either table; they fall through to the
/// [`RopeStyle::Neox`] default, which is inert for them because they never
/// build a [`RopeTable`](crate::common::rope::RopeTable).
///
/// Aliases that are not llama.cpp architecture ids but *are* registered by this
/// crate (`mistral`, `mixtral`, `yi`, `internlm3`, `llava`, `llava16`) are
/// mapped to the family their checkpoints actually ship as — all of which are
/// written as `"llama"` and are therefore `Norm`.
///
/// # Name traps
///
/// `starcoder` is `Norm` but `starcoder2` is `Neox`; `minicpm` is `Norm` but
/// `minicpm3` is `Neox`; `olmo` is `Norm` but `olmo2` is `Neox`.
pub fn rope_style_for_arch(arch_id: &str) -> RopeStyle {
    match arch_id {
        // ── LLAMA_ROPE_TYPE_NORM: rotate (x[2i], x[2i+1]) ────────────────
        // Q/K rows were permuted by convert_hf_to_gguf.py::permute().
        "arcee"
        | "arctic"
        | "baichuan"
        | "bailingmoe"
        | "chameleon"
        | "chatglm"
        | "cohere2"
        | "command-r"
        | "deci"
        | "deepseek"
        | "deepseek2"
        | "ernie4_5"
        | "ernie4_5-moe"
        | "glm-dsa"
        | "granite"
        | "granitehybrid"
        | "granitemoe"
        | "internlm2"
        | "llada"
        | "llama"
        | "llama-embed"
        | "llama4"
        | "maincoder"
        | "minicpm"
        | "mistral3"
        | "neo-bert"
        | "olmo"
        | "plm"
        | "smollm3"
        | "starcoder"
        | "xverse"
        // Registry aliases whose checkpoints ship as `llama`.
        | "mistral"
        | "mixtral"
        | "yi"
        | "internlm3"
        | "llava"
        | "llava16" => RopeStyle::Norm,

        // ── LLAMA_ROPE_TYPE_NEOX: rotate (x[i], x[i + n/2]) ──────────────
        // Everything else that ropes.  Listed explicitly so that an unknown
        // architecture is visibly falling through to the default rather than
        // matching by accident.
        "afmoe" | "apertus" | "bailingmoe2" | "bert" | "bitnet" | "codeshell" | "cogvlm"
        | "dbrx" | "dots1" | "dream" | "eurobert" | "exaone" | "exaone-moe" | "exaone4"
        | "falcon" | "falcon-h1" | "gemma" | "gemma-embedding" | "gemma2" | "gemma3"
        | "gemma3n" | "gpt-oss" | "gptneox" | "grok" | "grovemoe" | "hunyuan-dense"
        | "hunyuan-moe" | "jais2" | "jina-bert-v3" | "lfm2" | "lfm2moe" | "llada-moe"
        | "mimo2" | "minicpm3" | "minimax-m2" | "modern-bert" | "nemotron" | "nomic-bert"
        | "nomic-bert-moe" | "olmo2" | "olmoe" | "openelm" | "orion" | "pangu-embedded"
        | "phi2" | "phi3" | "phimoe" | "plamo" | "plamo2" | "plamo3" | "qwen" | "qwen2"
        | "qwen2moe" | "qwen3" | "qwen3moe" | "qwen3next" | "rnd1" | "seed_oss"
        | "smallthinker" | "stablelm" | "starcoder2" | "step35" => RopeStyle::Neox,

        _ => RopeStyle::Neox,
    }
}

/// The sliding-window layer pattern llama.cpp assigns to `arch_id`.
///
/// Reproduces the `hparams.set_swa_pattern(..)` call each architecture makes in
/// `llama_model::load_hparams`.  Returns `None` for architectures that have no
/// SWA at all (which includes Gemma **v1** — only Gemma-2 and later interleave).
pub fn swa_pattern_for_arch(arch_id: &str) -> Option<SwaPattern> {
    match arch_id {
        // hparams.set_swa_pattern(2)
        "gemma2" | "gpt-oss" => Some(SwaPattern::EveryNth(2)),
        // hparams.set_swa_pattern(4) — "3 sliding, 1 full"
        "cohere2" | "olmo2" | "exaone4" | "exaone-moe" | "llama4" | "afmoe" => {
            Some(SwaPattern::EveryNth(4))
        }
        // hparams.set_swa_pattern(5)
        "gemma3n" => Some(SwaPattern::EveryNth(5)),
        // hparams.set_swa_pattern(6)
        "gemma3" | "gemma-embedding" => Some(SwaPattern::EveryNth(6)),
        // hparams.set_swa_pattern(4, dense_first = true)
        "smallthinker" => Some(SwaPattern::EveryNthDenseFirst(4)),
        // llama.cpp *disables* Phi SWA: `swa_type = NONE`, `n_swa = 0`,
        // `set_swa_pattern(1)` — and `il % 1 < 0` is false for every layer, so
        // nothing slides.  See the comment at `LLM_ARCH_PHI3` referencing
        // ggml-org/llama.cpp#13676.
        "phi3" => Some(SwaPattern::EveryNth(1)),
        // Mistral ships as the `llama` architecture id; llama.cpp applies its
        // sliding window uniformly (no `set_swa_pattern` call at all).  The
        // `mistral*` ids are this crate's own.
        "mistral" | "mistral3" => Some(SwaPattern::All),
        _ => None,
    }
}

/// Architectures whose FFN activation is GELU rather than SiLU/SwiGLU.
///
/// Derived from the `build_*` graphs in `src/llama-model.cpp`
/// (`ggml_gelu` vs `ggml_silu`).  Only consulted when the checkpoint carries no
/// explicit `{arch}.activation` / `general.activation` key — GGUF has no
/// standard activation key, so this table is the only signal available.
///
/// # `falcon` is deliberately absent
///
/// llama.cpp's `falcon` graph does use GELU, but `falcon/config.rs` treats
/// `ModelConfig::activation == "gelu"` as its **out-of-band discriminator**
/// between Falcon-1 (ALiBi + parallel attention) and Falcon-2 (RoPE + GQA);
/// both ship under the same `falcon` architecture id, so the activation cannot
/// actually distinguish them.  Returning `"gelu"` here would flip every Falcon-2
/// checkpoint onto the ALiBi path.  Falcon's owner must re-key that heuristic
/// on the tensor set (`attn_norm_2` presence) before `falcon` can be added.
fn default_activation_for_arch(arch_id: &str) -> &'static str {
    match arch_id {
        "gemma" | "gemma2" | "gemma3" | "gemma3n" | "gemma-embedding" => "gelu_pytorch_tanh",
        "bloom" | "gpt2" | "gptneox" | "starcoder" | "starcoder2" | "phi2" | "codeshell"
        | "jais" | "bert" | "nomic-bert" | "mpt" | "refact" => "gelu",
        _ => "silu",
    }
}

/// Map a GGUF `general.file_type` (llama.cpp's `LLAMA_FTYPE`) onto the
/// predominant tensor quantization type it denotes.
///
/// Mirrors `gguf-py`'s `LlamaFileType` enum.  Returns `None` for
/// `GUESSED` (1024) and for any value that is not a recognised file type.
fn quant_type_from_file_type(file_type: u32) -> Option<GgufTensorType> {
    Some(match file_type {
        0 => GgufTensorType::F32,
        1 => GgufTensorType::F16,
        2 => GgufTensorType::Q4_0,
        3 => GgufTensorType::Q4_1,
        7 => GgufTensorType::Q8_0,
        8 => GgufTensorType::Q5_0,
        9 => GgufTensorType::Q5_1,
        10 | 21 => GgufTensorType::Q2K,
        11..=13 => GgufTensorType::Q3K,
        14 | 15 => GgufTensorType::Q4K,
        16 | 17 => GgufTensorType::Q5K,
        18 => GgufTensorType::Q6K,
        19 => GgufTensorType::Iq2Xxs,
        20 => GgufTensorType::Iq2Xs,
        22 | 23 => GgufTensorType::Iq3Xxs,
        24 => GgufTensorType::Iq1S,
        25 => GgufTensorType::Iq4Nl,
        26 | 27 => GgufTensorType::Iq3S,
        28 | 29 => GgufTensorType::Iq2S,
        30 => GgufTensorType::Iq4Xs,
        31 => GgufTensorType::Iq1M,
        32 => GgufTensorType::Bf16,
        _ => return None,
    })
}

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
    /// Vision encoder configuration (only populated for multimodal models like
    /// Qwen2-VL).  `None` for text-only models.
    pub vision_config: Option<VisionConfig>,
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
            vision_config: None,
        }
    }
}

/// Vision encoder configuration for multimodal models (e.g. Qwen2-VL).
///
/// Populated from `vision.*` GGUF metadata keys when available.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Input image size (square; width == height).
    pub image_size: usize,
    /// Patch size in pixels (square patches).
    pub patch_size: usize,
    /// Vision encoder hidden size.
    pub hidden_size: usize,
    /// Number of attention heads in the vision encoder.
    pub num_heads: usize,
    /// Number of transformer layers in the vision encoder.
    pub num_layers: usize,
    /// Window size for windowed attention (0 = full attention).
    pub window_size: usize,
}

impl VisionConfig {
    /// Parse a `VisionConfig` from GGUF metadata.
    ///
    /// Accepts both the current `clip.vision.*` key namespace written by
    /// `gguf-py`'s `Keys.ClipVision` and the legacy `vision.*` prefix that
    /// `ModelConfig`'s documentation has always promised.  Returns `None` when
    /// neither namespace carries a block count — i.e. the checkpoint has no
    /// vision tower.
    ///
    /// Keys (first match wins):
    ///
    /// | Field | `clip.vision.*` | legacy `vision.*` |
    /// |-------|-----------------|-------------------|
    /// | `num_layers`  | `clip.vision.block_count`           | `vision.block_count` |
    /// | `hidden_size` | `clip.vision.embedding_length`      | `vision.embedding_length` |
    /// | `num_heads`   | `clip.vision.attention.head_count`  | `vision.attention.head_count` |
    /// | `image_size`  | `clip.vision.image_size`            | `vision.image_size` |
    /// | `patch_size`  | `clip.vision.patch_size`            | `vision.patch_size` |
    /// | `window_size` | `clip.vision.window_size`           | `vision.window_size` |
    pub fn from_metadata(metadata: &MetadataStore) -> Option<Self> {
        let read = |suffix: &str| -> Option<usize> {
            metadata
                .get_u32(&format!("clip.vision.{suffix}"))
                .or_else(|_| metadata.get_u32(&format!("vision.{suffix}")))
                .ok()
                .map(|v| v as usize)
        };

        // The block count is the discriminator: without layers there is no tower.
        let num_layers = read("block_count").filter(|&v| v > 0)?;

        Some(Self {
            image_size: read("image_size").unwrap_or(224),
            patch_size: read("patch_size").unwrap_or(14),
            hidden_size: read("embedding_length").unwrap_or(1024),
            num_heads: read("attention.head_count").unwrap_or(16),
            num_layers,
            window_size: read("window_size").unwrap_or(0),
        })
    }
}

/// Hyperparameters that [`ModelConfig`] has no field for.
///
/// These keys exist in `gguf-py/gguf/constants.py` and are consumed by
/// individual architectures.  They live in a side-car struct rather than as
/// extra `ModelConfig` fields so that the per-architecture
/// `ModelConfig { .. }` struct literals keep compiling; adding fields to
/// `ModelConfig` would break 18 files owned by other architecture agents.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExtraHparams {
    /// `{arch}.rope.dimension_count` — the rotary dimension count `n_rot`,
    /// which is **not** always `head_dim` (partial-rotary models such as
    /// GPT-NeoX, Phi-2 and StableLM rotate only a prefix of each head).
    pub rope_dimension_count: Option<usize>,
    /// `{arch}.rope.dimension_sections` — M-RoPE per-axis dimension split
    /// (`[t, h, w, e]`), e.g. `[16, 24, 24, 0]` for Qwen2-VL.
    pub rope_sections: Option<Vec<usize>>,
    /// `{arch}.attn_logit_softcapping` — Gemma-2 attention logit soft cap.
    pub attn_logit_softcapping: Option<f32>,
    /// `{arch}.final_logit_softcapping` — Gemma-2/3 output logit soft cap.
    pub final_logit_softcapping: Option<f32>,
    /// `{arch}.attention.query_pre_attn_scalar` — Gemma's pre-attention query
    /// scale (overrides `1 / sqrt(head_dim)`).
    pub query_pre_attn_scalar: Option<f32>,
    /// `{arch}.expert_group_count` — DeepSeek-V3 routing group count.
    pub expert_group_count: Option<usize>,
    /// `{arch}.expert_group_used_count` — DeepSeek-V3 groups selected per token.
    pub expert_group_used_count: Option<usize>,
    /// `{arch}.expert_weights_norm` — whether routed expert weights are
    /// re-normalised after top-k selection.
    pub expert_weights_norm: Option<bool>,
    /// `{arch}.expert_weights_scale` — routed expert weight multiplier.
    pub expert_weights_scale: Option<f32>,
    /// `{arch}.attention.clamp_kqv` — MPT/DBRX QKV clamp magnitude.
    pub clamp_kqv: Option<f32>,
    /// `{arch}.attention.max_alibi_bias` — ALiBi maximum bias (BLOOM, MPT).
    pub max_alibi_bias: Option<f32>,
    /// `{arch}.attention.layer_norm_epsilon` — the **non**-RMS LayerNorm
    /// epsilon needed by BLOOM / Falcon / StarCoder / GPT-NeoX / StableLM.
    /// `ModelConfig::rms_norm_eps` only ever reads the `_rms_` variant.
    pub layer_norm_eps: Option<f32>,
    /// `{arch}.use_parallel_residual` — GPT-NeoX / Falcon parallel attention.
    pub use_parallel_residual: Option<bool>,
    /// `{arch}.attention.sliding_window_pattern` — explicit SWA period.
    pub sliding_window_pattern: Option<u32>,
    /// `{arch}.attention.scale` — explicit attention softmax scale.
    pub attention_scale: Option<f32>,
    /// `{arch}.attention.value_length` — V head dimension when it differs
    /// from the K head dimension.
    pub value_length: Option<usize>,
}

impl ExtraHparams {
    /// Read every supplementary hyperparameter key for `arch`.
    ///
    /// All keys are optional; absent keys leave their field `None` so callers
    /// can distinguish "not present" from "present and zero".
    pub fn from_metadata(metadata: &MetadataStore, arch: &str) -> Self {
        let u = |k: &str| metadata.get_u32(k).ok().map(|v| v as usize);
        let f = |k: &str| metadata.get_f32(k).ok();
        let b = |k: &str| metadata.get(k).and_then(MetadataValue::as_bool);

        let rope_sections = metadata
            .get(&format!("{arch}.rope.dimension_sections"))
            .and_then(MetadataValue::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(MetadataValue::as_u32)
                    .map(|v| v as usize)
                    .collect::<Vec<_>>()
            })
            .filter(|v: &Vec<usize>| !v.is_empty());

        Self {
            rope_dimension_count: u(&format!("{arch}.rope.dimension_count")),
            rope_sections,
            attn_logit_softcapping: f(&format!("{arch}.attn_logit_softcapping")),
            final_logit_softcapping: f(&format!("{arch}.final_logit_softcapping")),
            query_pre_attn_scalar: f(&format!("{arch}.attention.query_pre_attn_scalar")),
            expert_group_count: u(&format!("{arch}.expert_group_count")),
            expert_group_used_count: u(&format!("{arch}.expert_group_used_count")),
            expert_weights_norm: b(&format!("{arch}.expert_weights_norm")),
            expert_weights_scale: f(&format!("{arch}.expert_weights_scale")),
            clamp_kqv: f(&format!("{arch}.attention.clamp_kqv")),
            max_alibi_bias: f(&format!("{arch}.attention.max_alibi_bias")),
            layer_norm_eps: f(&format!("{arch}.attention.layer_norm_epsilon")),
            use_parallel_residual: b(&format!("{arch}.use_parallel_residual")),
            sliding_window_pattern: metadata
                .get_u32(&format!("{arch}.attention.sliding_window_pattern"))
                .ok(),
            attention_scale: f(&format!("{arch}.attention.scale")),
            value_length: u(&format!("{arch}.attention.value_length")),
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

        // `head_dim` is NOT always `hidden_size / num_attention_heads`.  Qwen3
        // decouples the two: Qwen3-4B has `embedding_length = 2560` and
        // `head_count = 32` (which would give 80) but ships 128-wide heads, so
        // `attn_q.weight` projects 2560 → 32 × 128.  The GGUF states the true
        // width in `{arch}.attention.key_length`; only fall back to the
        // division when the checkpoint omits it.
        let head_dim = metadata
            .get_u32(&format!("{arch}.attention.key_length"))
            .ok()
            .map(|v| v as usize)
            .filter(|&v| v > 0)
            .or_else(|| hidden_size.checked_div(num_attention_heads))
            .filter(|&v| v > 0)
            .unwrap_or(128);

        // `{arch}.vocab_size` is optional; most converters only ship the
        // tokenizer token array, whose length is the authoritative vocabulary
        // size (and matches the LM head's row count, padding included).
        let vocab_size = metadata
            .get_u32(&format!("{arch}.vocab_size"))
            .map(|v| v as usize)
            .ok()
            .or_else(|| {
                metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())
                    .map(<[MetadataValue]>::len)
            })
            .filter(|&v| v > 0)
            .unwrap_or(32000);

        let max_context_length = metadata
            .get_u32(&format!("{arch}.context_length"))
            .map(|v| v as usize)
            .unwrap_or(4096);

        // `_rms_` is the LLaMA-family key; BLOOM / Falcon / StarCoder /
        // GPT-NeoX / StableLM ship the plain LayerNorm epsilon instead.  Fall
        // back to it so those architectures stop silently normalising with
        // 1e-5 when the checkpoint says otherwise.
        let rms_norm_eps = metadata
            .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .or_else(|_| metadata.get_f32(&format!("{arch}.attention.layer_norm_epsilon")))
            .unwrap_or(1e-5);

        let rope_freq_base = metadata
            .get_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10000.0);

        let sliding_window = metadata
            .get_u32(&format!("{arch}.attention.sliding_window"))
            .ok()
            .map(|v| v as usize);

        let swa_window = sliding_window.map(|v| v as u32);

        // Interleaved SWA is decided by llama.cpp's per-architecture
        // `set_swa_pattern` call, not by a name prefix.  Gemma **v1** has no
        // SWA at all, so `starts_with("gemma")` wrongly interleaved it.
        let swa_interleaved =
            swa_pattern_for_arch(&architecture).is_some_and(SwaPattern::is_interleaved);

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

        // An unrecognised scaling type is an error rather than a silent
        // downgrade to "no scaling": Phi-3.5's `longrope`/`su` and
        // Llama-3.1's `llama3` used to fall through here and produce garbage
        // past the training context with no warning.
        let rope_scaling_type = match metadata.get_string(&format!("{arch}.rope.scaling.type")) {
            Ok(s) => RopeScalingType::parse(s)?,
            Err(_) => RopeScalingType::Standard,
        };

        let quant_type = metadata
            .get_u32("general.file_type")
            .ok()
            .and_then(quant_type_from_file_type);

        let activation = metadata
            .get_string(&format!("{arch}.activation"))
            .or_else(|_| metadata.get_string("general.activation"))
            .map(str::to_string)
            .unwrap_or_else(|_| default_activation_for_arch(arch).to_string());

        let vision_config = VisionConfig::from_metadata(metadata);

        validate_attention_shape(
            arch,
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_dim,
        )?;

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
            quant_type,
            // GGUF has no bias key; presence of `blk.N.attn_q.bias` /
            // `blk.N.ffn_up.bias` is the only signal, so these stay false here
            // and are filled in by `apply_tensor_hints` /
            // `from_metadata_and_tensors`.
            attention_bias: false,
            ffn_bias: false,
            activation,
            sliding_window,
            logit_scale,
            num_experts,
            num_experts_used,
            swa_window,
            swa_interleaved,
            rope_scaling_type,
            rope_scaling_factor,
            vision_config,
        })
    }

    /// Construct a `ModelConfig` from GGUF metadata **and** the tensor table.
    ///
    /// Preferred over [`Self::from_metadata`] whenever the `TensorStore` is
    /// available: `attention_bias`, `ffn_bias` and `quant_type` can only be
    /// determined from the tensors themselves.
    ///
    /// # Errors
    ///
    /// Propagates every error from [`Self::from_metadata`].
    pub fn from_metadata_and_tensors(
        metadata: &MetadataStore,
        tensors: &TensorStore,
    ) -> ArchResult<Self> {
        let mut config = Self::from_metadata(metadata)?;
        config.apply_tensor_hints(tensors);
        Ok(config)
    }

    /// Fill in the fields that can only be derived from the tensor table.
    ///
    /// * `attention_bias` — any `attn_q.bias` / `attn_k.bias` / `attn_v.bias` /
    ///   `attn_qkv.bias` / `attn_output.bias` entry.
    /// * `ffn_bias` — any `ffn_up.bias` / `ffn_down.bias` / `ffn_gate.bias`.
    /// * `quant_type` — the most common non-`F32` tensor type, which is what
    ///   `general.file_type` is supposed to record but frequently omits.
    pub fn apply_tensor_hints(&mut self, tensors: &TensorStore) {
        const ATTN_BIAS_SUFFIXES: [&str; 5] = [
            "attn_q.bias",
            "attn_k.bias",
            "attn_v.bias",
            "attn_qkv.bias",
            "attn_output.bias",
        ];
        const FFN_BIAS_SUFFIXES: [&str; 3] = ["ffn_up.bias", "ffn_down.bias", "ffn_gate.bias"];

        let mut counts: std::collections::HashMap<GgufTensorType, usize> =
            std::collections::HashMap::new();

        for (name, info) in tensors.iter() {
            if ATTN_BIAS_SUFFIXES.iter().any(|s| name.ends_with(s)) {
                self.attention_bias = true;
            }
            if FFN_BIAS_SUFFIXES.iter().any(|s| name.ends_with(s)) {
                self.ffn_bias = true;
            }
            *counts.entry(info.tensor_type).or_insert(0) += 1;
        }

        if self.quant_type.is_none() {
            self.quant_type = counts
                .into_iter()
                .filter(|(ty, _)| !matches!(ty, GgufTensorType::F32 | GgufTensorType::F16))
                .max_by_key(|&(_, n)| n)
                .map(|(ty, _)| ty);
        }
    }

    /// The RoPE element-pairing convention this architecture requires.
    ///
    /// Delegates to [`rope_style_for_arch`].  Architectures build their
    /// [`RopeTable`](crate::common::rope::RopeTable) with
    /// `RopeTable::new_with_style(head_dim, ctx, base, ty, factor, config.rope_style())`.
    pub fn rope_style(&self) -> RopeStyle {
        rope_style_for_arch(&self.architecture)
    }

    /// The per-layer sliding-window pattern, or `None` when every layer is global.
    ///
    /// Resolution order:
    /// 1. `None` when [`Self::swa_window`] is `None` (no window at all).
    /// 2. The architecture's llama.cpp pattern via [`swa_pattern_for_arch`].
    /// 3. Otherwise the legacy [`Self::swa_interleaved`] flag, which maps to
    ///    `EveryNth(2)` when set and `All` when clear.
    pub fn swa_pattern(&self) -> Option<SwaPattern> {
        self.swa_window?;
        Some(
            swa_pattern_for_arch(&self.architecture).unwrap_or(if self.swa_interleaved {
                SwaPattern::EveryNth(2)
            } else {
                SwaPattern::All
            }),
        )
    }
}

/// Validate the attention-shape invariants that every architecture divides by.
///
/// Without these checks a checkpoint (or a hand-built `MetadataStore`) with
/// `head_count_kv = 0` reaches `num_heads / num_kv_heads` and aborts the
/// process; `num_kv_heads > num_heads` yields `heads_per_kv == 0` and then
/// `h / 0`.
fn validate_attention_shape(
    arch: &str,
    hidden_size: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> ArchResult<()> {
    let mismatch = |param: &str, expected: String, got: String| ArchError::ConfigMismatch {
        param: format!("{arch}.{param}"),
        expected,
        got,
    };

    if num_attention_heads == 0 {
        return Err(mismatch(
            "attention.head_count",
            "> 0".to_string(),
            "0".to_string(),
        ));
    }
    if num_kv_heads == 0 {
        return Err(mismatch(
            "attention.head_count_kv",
            "> 0".to_string(),
            "0".to_string(),
        ));
    }
    if num_kv_heads > num_attention_heads {
        return Err(mismatch(
            "attention.head_count_kv",
            format!("<= head_count ({num_attention_heads})"),
            num_kv_heads.to_string(),
        ));
    }
    if !num_attention_heads.is_multiple_of(num_kv_heads) {
        return Err(mismatch(
            "attention.head_count_kv",
            format!("a divisor of head_count ({num_attention_heads})"),
            num_kv_heads.to_string(),
        ));
    }
    if head_dim == 0 {
        return Err(mismatch(
            "attention.key_length",
            "> 0".to_string(),
            "0".to_string(),
        ));
    }
    if hidden_size == 0 {
        return Err(mismatch(
            "embedding_length",
            "> 0".to_string(),
            "0".to_string(),
        ));
    }
    Ok(())
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

    /// `attention.key_length` wins over `hidden_size / head_count`.
    ///
    /// Qwen3-4B is exactly this shape: 2560 / 32 = 80, but the checkpoint's
    /// heads are 128 wide and `attn_q.weight` projects 2560 → 32 × 128.
    #[test]
    fn test_from_metadata_head_dim_prefers_key_length() {
        let mut store = minimal_store("qwen3");
        store.insert(
            "qwen3.embedding_length".to_string(),
            MetadataValue::Uint32(2560),
        );
        store.insert(
            "qwen3.attention.head_count".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert(
            "qwen3.attention.key_length".to_string(),
            MetadataValue::Uint32(128),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(
            cfg.head_dim, 128,
            "head_dim must come from key_length, not 2560 / 32 = 80"
        );
    }

    /// A zero `key_length` is ignored in favour of the division.
    #[test]
    fn test_from_metadata_head_dim_ignores_zero_key_length() {
        let mut store = minimal_store("llama");
        store.insert(
            "llama.embedding_length".to_string(),
            MetadataValue::Uint32(4096),
        );
        store.insert(
            "llama.attention.head_count".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert(
            "llama.attention.key_length".to_string(),
            MetadataValue::Uint32(0),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.head_dim, 128);
    }

    /// Without `{arch}.vocab_size`, the tokenizer token array supplies it.
    ///
    /// Most converters ship only the array; falling through to the 32000
    /// default under-sizes the logit buffer and the LM head GEMV fails.
    #[test]
    fn test_from_metadata_vocab_size_from_tokenizer_tokens() {
        let mut store = minimal_store("qwen3");
        store.insert(
            "tokenizer.ggml.tokens".to_string(),
            MetadataValue::Array(vec![MetadataValue::String("tok".to_string()); 7]),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.vocab_size, 7, "vocab_size must follow the token array");
    }

    /// An explicit `{arch}.vocab_size` still wins over the token array.
    #[test]
    fn test_from_metadata_vocab_size_prefers_explicit_key() {
        let mut store = minimal_store("qwen3");
        store.insert("qwen3.vocab_size".to_string(), MetadataValue::Uint32(11));
        store.insert(
            "tokenizer.ggml.tokens".to_string(),
            MetadataValue::Array(vec![MetadataValue::String("tok".to_string()); 7]),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.vocab_size, 11);
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

    /// llama.cpp's `set_swa_pattern(2)` marks `il % 2 < 1`, i.e. **even**
    /// layers, as sliding.  This test previously asserted the inverse.
    #[test]
    fn test_gemma_interleaved_swa() {
        let config = ModelConfig {
            architecture: "gemma2".to_string(),
            swa_window: Some(4096),
            swa_interleaved: true,
            ..ModelConfig::default()
        };
        // Layer 0 is the sliding-window layer (even index).
        assert_eq!(effective_attention_span(&config, 0), 4096);
        // Layer 1 is global.
        assert_eq!(effective_attention_span(&config, 1), u32::MAX);
        // Layer 2 is sliding again.
        assert_eq!(effective_attention_span(&config, 2), 4096);
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

    /// Gemma-**2** interleaves; Gemma **v1** does not.  Detecting by the
    /// `"gemma"` name prefix wrongly interleaved v1 as well.
    #[test]
    fn test_gemma_swa_interleaved_from_metadata() {
        use oxillama_gguf::MetadataValue;
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("gemma2".to_string()),
        );
        store.insert(
            "gemma2.attention.sliding_window".to_string(),
            MetadataValue::Uint32(2048),
        );
        let cfg = ModelConfig::from_metadata(&store).expect("should succeed");
        assert_eq!(cfg.swa_window, Some(2048));
        assert!(cfg.swa_interleaved, "gemma2 should be interleaved");
    }

    #[test]
    fn test_gemma_v1_is_not_interleaved() {
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
        assert!(
            !cfg.swa_interleaved,
            "Gemma v1 has no SWA pattern in llama.cpp"
        );
    }
}

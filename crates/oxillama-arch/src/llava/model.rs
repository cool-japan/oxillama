//! LLaVA model: LLaMA backbone + CLIP tower + MM projector.
//!
//! ## How a caller selects the multimodal path
//!
//! **A LLaVA GGUF is not a distinct architecture.**  `convert_hf_to_gguf.py`
//! writes the language model with `general.architecture = "llama"` and the
//! vision half into a *separate* file whose `general.architecture` is
//! `"clip"` (`MODEL_ARCH_NAMES[MODEL_ARCH.MMPROJ] = "clip"`,
//! `gguf-py/gguf/constants.py:805`; written unconditionally by
//! `GGUFWriter.add_architecture`, `gguf_writer.py:497`).  So `"llava"` can
//! never appear in `general.architecture` and the registry entry for it is
//! unreachable by that route — exactly as llama.cpp, which takes a separate
//! `--mmproj` path.
//!
//! The selection point is therefore [`LlavaModel::load_with_mmproj`]: the
//! caller passes both files.  [`LlavaModel::load`] remains for the legacy
//! single-file layout where `mm.*`/`v.*` sit alongside the backbone.
//!
//! ## Tensor naming (GGUF)
//!
//! Backbone (main file) — identical to LLaMA.
//! Projector (mmproj) — `mm.0.{weight,bias}`, `mm.2.{weight,bias}`
//! (`TN_LLAVA_PROJ = "mm.%d.%s"`, `clip-impl.h:92`).
//! Tower (mmproj) — `v.class_embd`, `v.patch_embd.{weight,bias}`,
//! `v.position_embd.weight`, `v.pre_ln.*`, `v.post_ln.*`,
//! `v.blk.{i}.{ln1,ln2,attn_q,attn_k,attn_v,attn_out,ffn_up,ffn_down}.*`.

use crate::common::loader::{load_bias, load_dequant_tensor};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::llama::{load_llama_from_gguf, LlamaModel};
use crate::llava::clip::{ClipEncoder, ClipEncoderLayer, ClipFfnOp, ClipVisionParams};
use crate::llava::inject::{Prompt, VisualTokens};
use crate::llava::projector::MmProjector;
use crate::lora::LoadedLora;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::{GgufModel, MetadataStore, MetadataValue};

/// Combined LLaVA model: CLIP vision tower + MLP projector + LLaMA backbone.
pub struct LlavaModel {
    /// CLIP vision tower, from the mmproj file.
    pub vision_encoder: ClipEncoder,
    /// Multi-modal projector (CLIP dim → LLM dim).
    pub mm_projector: MmProjector,
    /// LLaMA language backbone, from the main file.
    pub language_model: LlamaModel,
    /// LLM hidden size.
    pub llm_hidden_size: usize,
    /// CLIP hidden size.
    pub clip_hidden_size: usize,
}

impl LlavaModel {
    /// Load a LLaVA model from a main GGUF plus a `clip`-architecture mmproj.
    ///
    /// # Errors
    ///
    /// Propagates backbone loading failures, and reports every missing vision
    /// tensor as [`ArchError::MissingTensor`].  Nothing is zero-filled: the
    /// previous loader used `.unwrap_or_default()` on every `v.*` tensor, so a
    /// mmproj with a single renamed tensor produced an all-zero tower and no
    /// diagnostic.
    pub fn load_with_mmproj(
        gguf: &GgufModel,
        mmproj: &GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Self> {
        let language_model = load_llama_from_gguf(gguf, config)?;
        let hidden_size = config.hidden_size;

        let params = clip_params_from_metadata(&mmproj.file.metadata);
        params.validate()?;

        let mm_projector = load_mm_projector(mmproj, params.hidden_size, hidden_size)?;
        let vision_encoder = load_clip_tower(mmproj, params)?;
        let clip_hidden_size = vision_encoder.hidden_size();

        Ok(Self {
            vision_encoder,
            mm_projector,
            language_model,
            llm_hidden_size: hidden_size,
            clip_hidden_size,
        })
    }

    /// Load from a single GGUF that carries backbone, projector and tower.
    ///
    /// # Errors
    ///
    /// See [`Self::load_with_mmproj`].
    pub fn load(gguf: &GgufModel, config: &ModelConfig) -> ArchResult<Self> {
        Self::load_with_mmproj(gguf, gguf, config)
    }

    /// Encode an image into visual tokens in the LLM embedding space.
    ///
    /// # Arguments
    /// * `pixels` — flat `[3 × image_size × image_size]`, channels-first,
    ///   normalized by the tower's `image_mean` / `image_std`.
    ///
    /// # Returns
    /// [`VisualTokens`] of `num_patches` rows (576 for LLaVA-1.5).
    ///
    /// # Errors
    ///
    /// Propagates tower and projector failures.
    pub fn encode_image(&self, pixels: &[f32]) -> ArchResult<VisualTokens> {
        let clip_features = self.vision_encoder.encode(pixels)?;
        let projected = self.mm_projector.project(&clip_features)?;
        VisualTokens::new(projected, self.llm_hidden_size)
    }

    /// Build the spliced input-embedding buffer for a multimodal prompt.
    ///
    /// Text ids are looked up in the backbone's embedding table; image slots are
    /// filled with the projected patch embeddings.  The result is a flat
    /// `[seq_len × hidden_size]` buffer — the `llama_batch.embd` payload of the
    /// reference (`mtmd-helper.cpp:129-145`).
    ///
    /// # Errors
    ///
    /// Propagates embedding-table lookup failures (out-of-vocabulary ids) and
    /// splicing errors.
    pub fn build_input_embeddings(&self, prompt: &Prompt<'_>) -> ArchResult<Vec<f32>> {
        let hidden = self.llm_hidden_size;
        let embd = &self.language_model.token_embd;
        let dispatcher = &self.language_model.dispatcher;
        prompt.build_embeddings(hidden, |tok, row| embd.row_into(dispatcher, tok, row))
    }

    /// Run a multimodal forward pass and return the next-token logits.
    ///
    /// Splices `prompt` into an input-embedding sequence and evaluates the
    /// backbone over it.
    ///
    /// # Implementation note
    ///
    /// The backbone's residual stream is loaded per position by
    /// `token_embd.row_into(id)`.  With no `ForwardPass` method that accepts
    /// hidden states, this method installs the spliced sequence *as* the lookup
    /// table and walks the identity id sequence `0..seq_len`, so row `i` of the
    /// table is position `i`'s embedding.  Both backbone paths — the per-token
    /// loop and the batched prefill — resolve embeddings through that one call
    /// (`llama/model.rs:326`, `llama/batch.rs:293`), so the substitution is
    /// exact.  The original table is restored before returning, including on
    /// the error path.
    ///
    /// This is a bridge, not the destination: see the crate report for the
    /// `ForwardPass::forward_embeds` signature that removes the swap.
    ///
    /// # Errors
    ///
    /// Propagates splicing and backbone failures, including context overflow.
    pub fn forward_multimodal(
        &mut self,
        prompt: &Prompt<'_>,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let embeds = self.build_input_embeddings(prompt)?;
        let seq_len = prompt.seq_len();
        self.forward_embeds(&embeds, seq_len, kv_cache)
    }

    /// Evaluate the backbone over precomputed input embeddings.
    ///
    /// `embeds` is `[seq_len × hidden_size]` in row-major order.
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidShape`] — `embeds.len() != seq_len * hidden_size`.
    /// * [`ArchError::InvalidConfig`] — `seq_len` is zero.
    /// * whatever the backbone reports.
    pub fn forward_embeds(
        &mut self,
        embeds: &[f32],
        seq_len: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let hidden = self.llm_hidden_size;
        if seq_len == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "forward_embeds: seq_len must be > 0".to_string(),
            });
        }
        let need = seq_len
            .checked_mul(hidden)
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: "forward_embeds: seq_len * hidden_size overflows".to_string(),
            })?;
        if embeds.len() != need {
            return Err(ArchError::InvalidShape {
                name: "input embeddings".to_string(),
                expected: vec![seq_len, hidden],
                got: vec![embeds.len()],
            });
        }
        if u32::try_from(seq_len).is_err() {
            return Err(ArchError::InvalidConfig {
                detail: format!("forward_embeds: seq_len {seq_len} exceeds u32 token-id range"),
            });
        }

        let ids: Vec<u32> = (0..seq_len as u32).collect();
        let saved = std::mem::replace(
            &mut self.language_model.token_embd,
            crate::llama::TokenEmbedding::dense(embeds.to_vec(), hidden),
        );
        let result = self.language_model.forward(&ids, kv_cache);
        self.language_model.token_embd = saved;
        result
    }
}

// ---------------------------------------------------------------------------
// mmproj loading
// ---------------------------------------------------------------------------

/// Read `clip.vision.*` geometry from an mmproj's metadata.
///
/// Falls back to CLIP ViT-L/14-336 only for keys the file omits; the reference
/// treats these as required (`clip.cpp:1009-1018`), but a synthetic fixture may
/// legitimately carry a subset.
pub fn clip_params_from_metadata(meta: &MetadataStore) -> ClipVisionParams {
    let defaults = ClipVisionParams::default();
    let u = |k: &str| meta.get_u32(k).ok().map(|v| v as usize);

    // `clip.use_gelu` / `clip.use_silu` have no `.vision.` segment
    // (`clip-impl.h:26-27`); neither set means QuickGELU (`clip.cpp:1085`).
    let ffn_op = match (
        meta.get("clip.use_gelu").and_then(MetadataValue::as_bool),
        meta.get("clip.use_silu").and_then(MetadataValue::as_bool),
    ) {
        (Some(true), _) => ClipFfnOp::Gelu,
        (_, Some(true)) => ClipFfnOp::Silu,
        _ => ClipFfnOp::QuickGelu,
    };

    ClipVisionParams {
        hidden_size: u("clip.vision.embedding_length").unwrap_or(defaults.hidden_size),
        num_heads: u("clip.vision.attention.head_count").unwrap_or(defaults.num_heads),
        num_layers: u("clip.vision.block_count").unwrap_or(defaults.num_layers),
        patch_size: u("clip.vision.patch_size").unwrap_or(defaults.patch_size),
        image_size: u("clip.vision.image_size").unwrap_or(defaults.image_size),
        eps: meta
            .get_f32("clip.vision.attention.layer_norm_epsilon")
            .unwrap_or(defaults.eps),
        ffn_op,
        feature_layer: u("clip.vision.feature_layer"),
    }
}

/// Load `mm.0.*` / `mm.2.*`.
fn load_mm_projector(
    mmproj: &GgufModel,
    clip_hidden_size: usize,
    llm_hidden_size: usize,
) -> ArchResult<MmProjector> {
    let fc1_weight = load_dequant_tensor(mmproj, "mm.0.weight")?;
    let fc1_bias = load_bias(mmproj, "mm.0.bias")?.unwrap_or_default();
    let fc2_weight = load_dequant_tensor(mmproj, "mm.2.weight")?;
    let fc2_bias = load_bias(mmproj, "mm.2.bias")?.unwrap_or_default();

    // `mm.0` maps clip_hidden → mm_hidden; infer the middle width from its
    // element count rather than assuming it equals the LLM hidden size.
    if clip_hidden_size == 0 {
        return Err(ArchError::InvalidConfig {
            detail: "mm projector: clip hidden size must be > 0".to_string(),
        });
    }
    let mm_hidden_size = fc1_weight.len() / clip_hidden_size;
    if mm_hidden_size == 0 {
        return Err(ArchError::InvalidShape {
            name: "mm.0.weight".to_string(),
            expected: vec![clip_hidden_size],
            got: vec![fc1_weight.len()],
        });
    }

    let projector = MmProjector {
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        clip_hidden_size,
        mm_hidden_size,
        llm_hidden_size,
    };
    projector.validate()?;
    Ok(projector)
}

/// Load the `v.*` tower.
///
/// Every per-block tensor is **required**: the reference marks `attn_out`,
/// `ffn_up` and `ffn_down` weights required (`clip.cpp:1403,1421,1425`) and a
/// missing norm or bias would silently change the arithmetic.
pub fn load_clip_tower(mmproj: &GgufModel, params: ClipVisionParams) -> ArchResult<ClipEncoder> {
    let class_embd = if mmproj.file.tensors.contains("v.class_embd") {
        load_dequant_tensor(mmproj, "v.class_embd")?
    } else {
        Vec::new()
    };
    let patch_embd_weight = load_dequant_tensor(mmproj, "v.patch_embd.weight")?;
    let patch_embd_bias = load_bias(mmproj, "v.patch_embd.bias")?.unwrap_or_default();
    let position_embd = load_dequant_tensor(mmproj, "v.position_embd.weight")?;
    let pre_ln_weight = optional(mmproj, "v.pre_ln.weight")?;
    let pre_ln_bias = optional(mmproj, "v.pre_ln.bias")?;
    let post_ln_weight = optional(mmproj, "v.post_ln.weight")?;
    let post_ln_bias = optional(mmproj, "v.post_ln.bias")?;

    let mut layers = Vec::with_capacity(params.num_layers);
    for i in 0..params.num_layers {
        let pfx = format!("v.blk.{i}");
        layers.push(ClipEncoderLayer {
            ln1_weight: load_dequant_tensor(mmproj, &format!("{pfx}.ln1.weight"))?,
            ln1_bias: optional(mmproj, &format!("{pfx}.ln1.bias"))?,
            ln2_weight: load_dequant_tensor(mmproj, &format!("{pfx}.ln2.weight"))?,
            ln2_bias: optional(mmproj, &format!("{pfx}.ln2.bias"))?,
            q_weight: load_dequant_tensor(mmproj, &format!("{pfx}.attn_q.weight"))?,
            k_weight: load_dequant_tensor(mmproj, &format!("{pfx}.attn_k.weight"))?,
            v_weight: load_dequant_tensor(mmproj, &format!("{pfx}.attn_v.weight"))?,
            out_weight: load_dequant_tensor(mmproj, &format!("{pfx}.attn_out.weight"))?,
            q_bias: optional(mmproj, &format!("{pfx}.attn_q.bias"))?,
            k_bias: optional(mmproj, &format!("{pfx}.attn_k.bias"))?,
            v_bias: optional(mmproj, &format!("{pfx}.attn_v.bias"))?,
            out_bias: optional(mmproj, &format!("{pfx}.attn_out.bias"))?,
            fc1_weight: load_dequant_tensor(mmproj, &format!("{pfx}.ffn_up.weight"))?,
            fc1_bias: optional(mmproj, &format!("{pfx}.ffn_up.bias"))?,
            fc2_weight: load_dequant_tensor(mmproj, &format!("{pfx}.ffn_down.weight"))?,
            fc2_bias: optional(mmproj, &format!("{pfx}.ffn_down.bias"))?,
        });
    }

    Ok(ClipEncoder {
        params,
        class_embd,
        patch_embd_weight,
        patch_embd_bias,
        position_embd,
        pre_ln_weight,
        pre_ln_bias,
        post_ln_weight,
        post_ln_bias,
        layers,
    })
}

/// Load a genuinely optional tensor, propagating read errors for one that exists.
fn optional(model: &GgufModel, name: &str) -> ArchResult<Vec<f32>> {
    if model.file.tensors.contains(name) {
        load_dequant_tensor(model, name)
    } else {
        Ok(Vec::new())
    }
}

impl ForwardPass for LlavaModel {
    /// Text-only forward pass — delegates to the LLaMA backbone.
    ///
    /// For multimodal inference use [`LlavaModel::encode_image`] then
    /// [`LlavaModel::forward_multimodal`].
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.language_model.forward(tokens, kv_cache)
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.language_model.embed(tokens, kv_cache)
    }

    fn embed_all(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.language_model.embed_all(tokens, kv_cache)
    }

    fn vocab_size(&self) -> usize {
        self.language_model.vocab_size()
    }

    fn max_context_length(&self) -> usize {
        self.language_model.max_context_length()
    }

    fn hidden_size(&self) -> usize {
        self.language_model.hidden_size()
    }

    fn apply_lora(&mut self, lora: &LoadedLora) -> ArchResult<()> {
        self.language_model.apply_lora(lora)
    }

    fn apply_lora_scaled(&mut self, lora: &LoadedLora, scale: f32) -> ArchResult<()> {
        self.language_model.apply_lora_scaled(lora, scale)
    }

    fn reset_sequence(&mut self) {
        self.language_model.reset_sequence();
    }

    fn unapply_all_loras(&mut self) {
        self.language_model.unapply_all_loras();
    }
}

/// Convenience entry-point used by the architecture registry.
///
/// # Errors
///
/// See [`LlavaModel::load`].
pub fn load_llava_from_gguf(gguf: &GgufModel, config: &ModelConfig) -> ArchResult<LlavaModel> {
    LlavaModel::load(gguf, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::MetadataStore;

    #[test]
    fn metadata_drives_geometry_not_hardcoded_constants() {
        let mut meta = MetadataStore::new();
        meta.insert(
            "clip.vision.embedding_length".to_string(),
            MetadataValue::Uint32(768),
        );
        meta.insert(
            "clip.vision.attention.head_count".to_string(),
            MetadataValue::Uint32(12),
        );
        meta.insert(
            "clip.vision.block_count".to_string(),
            MetadataValue::Uint32(12),
        );
        meta.insert(
            "clip.vision.patch_size".to_string(),
            MetadataValue::Uint32(16),
        );
        meta.insert(
            "clip.vision.image_size".to_string(),
            MetadataValue::Uint32(224),
        );

        let p = clip_params_from_metadata(&meta);
        assert_eq!(p.hidden_size, 768, "ViT-B/16, not the ViT-L default");
        assert_eq!(p.num_heads, 12);
        assert_eq!(p.num_layers, 12);
        assert_eq!(p.patch_size, 16);
        assert_eq!(p.image_size, 224);
        assert_eq!(p.num_patches(), 196, "(224/16)^2");
        assert_eq!(p.executed_layers(), 11, "n_layer - 1");
        assert_eq!(
            p.ffn_op,
            ClipFfnOp::QuickGelu,
            "neither clip.use_gelu nor clip.use_silu ⇒ QuickGELU"
        );
        p.validate().expect("ViT-B/16 geometry is valid");
    }

    #[test]
    fn use_gelu_and_use_silu_select_the_activation() {
        let mut meta = MetadataStore::new();
        meta.insert("clip.use_gelu".to_string(), MetadataValue::Bool(true));
        assert_eq!(clip_params_from_metadata(&meta).ffn_op, ClipFfnOp::Gelu);

        let mut meta = MetadataStore::new();
        meta.insert("clip.use_silu".to_string(), MetadataValue::Bool(true));
        assert_eq!(clip_params_from_metadata(&meta).ffn_op, ClipFfnOp::Silu);
    }

    #[test]
    fn empty_metadata_falls_back_to_vit_l_14_336() {
        let p = clip_params_from_metadata(&MetadataStore::new());
        assert_eq!(p.hidden_size, 1024);
        assert_eq!(p.num_layers, 24);
        assert_eq!(p.num_patches(), 576);
        assert_eq!(p.executed_layers(), 23);
    }

    #[test]
    fn feature_layer_metadata_overrides_the_default() {
        let mut meta = MetadataStore::new();
        meta.insert(
            "clip.vision.feature_layer".to_string(),
            MetadataValue::Uint32(20),
        );
        assert_eq!(clip_params_from_metadata(&meta).executed_layers(), 20);
    }
}

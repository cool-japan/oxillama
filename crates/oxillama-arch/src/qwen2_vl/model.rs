//! Qwen2-VL full multimodal model.
//!
//! ```text
//! image pixels
//!   └─► Qwen2VlVisionEncoder ─► [N_patches, vis_hidden]   (2×2-block order)
//!         └─► MmMerger        ─► [N_patches/4, llm_hidden]
//! text tokens
//!   └─► token_embd            ─► [seq_len, llm_hidden]
//!            (spliced together) ▼
//!                      Qwen2 backbone (M-RoPE)
//!                                 │
//!                         logits [vocab_size]
//! ```
//!
//! ## The vision tower is optional, and never fabricated
//!
//! A Qwen2-VL checkpoint splits the same way a LLaVA one does: the text model
//! is `general.architecture = "qwen2vl"` and the ViT lives in a separate
//! `clip`-architecture mmproj file.  The main file therefore has **no** `v.*`
//! or `mm.*` tensors at all, and [`load_qwen2vl_from_gguf`] leaves
//! [`Qwen2VlModel::vision`] as `None`.
//!
//! What it no longer does is fabricate one.  The previous loader read
//! `vis_num_layers = vis_cfg.map_or(0, …)` and
//! `load_optional_f32(…, "v.patch_embd.weight").unwrap_or_else(|| vec![0.0; …])`,
//! so — because `ModelConfig::from_metadata` hard-coded `vision_config: None`
//! at the time — production always got a tower with a single all-zeros patch
//! projection and **zero** transformer blocks, which returned all zeros with no
//! error.  Now:
//!
//! * no vision metadata ⇒ `vision = None` and [`Qwen2VlModel::encode_image`]
//!   returns [`ArchError::NotSupported`];
//! * vision metadata present ⇒ every `v.*` / `mm.*` tensor is **required** and
//!   a missing one is [`ArchError::MissingTensor`].
//!
//! ## Tensor naming (GGUF)
//!
//! Backbone — as Qwen2/3: `token_embd.weight`, `blk.{i}.*`, `output_norm.weight`,
//! `output.weight`.
//!
//! Merger — `mm.0.{weight,bias}` and `mm.2.{weight,bias}` with GELU between
//! (`clip.cpp:1573-1576` loads `mm_1_w` from `TN_LLAVA_PROJ` index **2**;
//! `models/qwen2vl.cpp:158-164` runs `build_ffn(…, FFN_GELU, -1)`).
//!
//! Tower — `v.patch_embd.weight`, `v.blk.{i}.{ln1,ln2,attn_*,ffn_*}.*`,
//! `v.post_ln.*`.

use crate::common::attention::{validate_context_bounds, validate_token_ids};
use crate::common::linear::QuantLinear;
use crate::common::mrope::MRopeTable;
use crate::common::rms_norm::RmsNorm;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::lora::LoadedLora;
use crate::qwen2_vl::vision::Qwen2VlVisionEncoder;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::KernelDispatcher;

// ---------------------------------------------------------------------------
// M-RoPE positions
// ---------------------------------------------------------------------------

/// A single token's M-RoPE position triple.
///
/// Text tokens use `t == h == w`, which the fixed [`MRopeTable`] reduces to
/// exactly standard 1-D NeoX RoPE.  Visual tokens carry the patch's grid row in
/// `h` and column in `w`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MRopePos {
    /// Temporal / text position.
    pub t: usize,
    /// Vision patch row.
    pub h: usize,
    /// Vision patch column.
    pub w: usize,
}

impl MRopePos {
    /// The text-token triple `(pos, pos, pos)`.
    pub fn text(pos: usize) -> Self {
        Self {
            t: pos,
            h: pos,
            w: pos,
        }
    }
}

// ---------------------------------------------------------------------------
// MM Merger: 2×2 → 1 spatial compression
// ---------------------------------------------------------------------------

/// Compresses 2×2 spatial patch blocks into single LLM tokens.
///
/// The vision encoder already emits tokens in 2×2-block order (see
/// [`Qwen2VlVisionEncoder::token_grid_pos`]), so merging is a plain "group four
/// consecutive tokens" reshape followed by a two-layer GELU MLP.
#[derive(Debug, Clone)]
pub struct MmMerger {
    /// Vision encoder hidden size.
    pub vis_hidden_size: usize,
    /// LLM hidden size.
    pub llm_hidden_size: usize,
    /// Width between the two projections.
    pub mm_hidden_size: usize,
    /// `mm.0.weight`, row-major `[mm_hidden, 4 × vis_hidden]`.
    pub fc1_weight: Vec<f32>,
    /// `mm.0.bias`.
    pub fc1_bias: Vec<f32>,
    /// `mm.2.weight`, row-major `[llm_hidden, mm_hidden]`.
    pub fc2_weight: Vec<f32>,
    /// `mm.2.bias`.
    pub fc2_bias: Vec<f32>,
}

impl MmMerger {
    /// Merge and project vision patches.
    ///
    /// # Arguments
    /// * `patch_features` — flat `[num_patches × vis_hidden_size]` in 2×2-block
    ///   order.
    /// * `patches_h` / `patches_w` — the patch grid, both even.
    ///
    /// # Returns
    /// Flat `[(patches_h/2 × patches_w/2) × llm_hidden_size]`.
    ///
    /// # Errors
    /// * [`ArchError::InvalidConfig`] — zero `vis_hidden_size`, or an odd grid
    ///   dimension (the reference asserts `img.nx % (patch_size * 2) == 0`).
    /// * [`ArchError::InvalidShape`] — feature or weight length mismatch.
    pub fn merge(
        &self,
        patch_features: &[f32],
        patches_h: usize,
        patches_w: usize,
    ) -> ArchResult<Vec<f32>> {
        if self.vis_hidden_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "MmMerger: vis_hidden_size must be > 0".to_string(),
            });
        }
        if !patches_h.is_multiple_of(2) || !patches_w.is_multiple_of(2) {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "MmMerger: patch grid {patches_w}×{patches_h} must have even dimensions"
                ),
            });
        }
        let expected = patches_h * patches_w * self.vis_hidden_size;
        if patch_features.len() != expected {
            return Err(ArchError::InvalidShape {
                name: "patch_features".to_string(),
                expected: vec![expected],
                got: vec![patch_features.len()],
            });
        }

        let merged_in_dim = 4 * self.vis_hidden_size;
        let num_merged = (patches_h / 2) * (patches_w / 2);
        let mut out = Vec::with_capacity(num_merged * self.llm_hidden_size);
        let mut h = vec![0.0f32; self.mm_hidden_size];

        for block in 0..num_merged {
            let src = &patch_features[block * merged_in_dim..(block + 1) * merged_in_dim];

            for (i, hv) in h.iter_mut().enumerate() {
                let start = i * merged_in_dim;
                let row = self
                    .fc1_weight
                    .get(start..start + merged_in_dim)
                    .ok_or_else(|| ArchError::InvalidShape {
                        name: "mm.0.weight".to_string(),
                        expected: vec![self.mm_hidden_size, merged_in_dim],
                        got: vec![self.fc1_weight.len()],
                    })?;
                *hv = row.iter().zip(src.iter()).map(|(w, x)| w * x).sum::<f32>()
                    + self.fc1_bias.get(i).copied().unwrap_or(0.0);
                *hv = crate::common::gelu::gelu(*hv);
            }

            for i in 0..self.llm_hidden_size {
                let start = i * self.mm_hidden_size;
                let row = self
                    .fc2_weight
                    .get(start..start + self.mm_hidden_size)
                    .ok_or_else(|| ArchError::InvalidShape {
                        name: "mm.2.weight".to_string(),
                        expected: vec![self.llm_hidden_size, self.mm_hidden_size],
                        got: vec![self.fc2_weight.len()],
                    })?;
                out.push(
                    row.iter().zip(h.iter()).map(|(w, x)| w * x).sum::<f32>()
                        + self.fc2_bias.get(i).copied().unwrap_or(0.0),
                );
            }
        }

        Ok(out)
    }
}

/// The optional vision half of a Qwen2-VL model.
pub struct Qwen2VlVision {
    /// Native ViT encoder.
    pub encoder: Qwen2VlVisionEncoder,
    /// 2×2 spatial patch compressor + projector.
    pub merger: MmMerger,
}

// ---------------------------------------------------------------------------
// Backbone
// ---------------------------------------------------------------------------

/// A single Qwen2-backbone transformer layer.
pub struct Qwen2Layer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Query projection (biased).
    pub attn_q: QuantLinear,
    /// Key projection (biased).
    pub attn_k: QuantLinear,
    /// Value projection (biased).
    pub attn_v: QuantLinear,
    /// Attention output projection.
    pub attn_output: QuantLinear,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN gate projection.
    pub ffn_gate: QuantLinear,
    /// FFN up projection.
    pub ffn_up: QuantLinear,
    /// FFN down projection.
    pub ffn_down: QuantLinear,
}

/// Combined Qwen2-VL model.
pub struct Qwen2VlModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Vision half — `None` for a text-only checkpoint.
    pub vision: Option<Qwen2VlVision>,
    /// Token embedding table `[vocab_size, hidden_size]`.
    pub token_embd: Vec<f32>,
    /// Backbone layers.
    pub layers: Vec<Qwen2Layer>,
    /// Final RMSNorm.
    pub output_norm: RmsNorm,
    /// LM head `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Multimodal RoPE table, built from `qwen2vl.rope.dimension_sections`.
    pub mrope: MRopeTable,
    /// Quantization kernel dispatcher.
    dispatcher: KernelDispatcher,

    // Scratch buffers reused across forward calls.
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl Qwen2VlModel {
    /// Construct from preloaded components.
    ///
    /// `rope_sections` is `qwen2vl.rope.dimension_sections` in **rotation-pair**
    /// units (`[16, 24, 24, 0]` for a 128-wide head).  `None` falls back to the
    /// equal-thirds split, which no real checkpoint uses.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] when the sections sum past `head_dim / 2`,
    /// or [`ArchError::InvalidConfig`] when they are all zero.
    pub fn new(
        config: ModelConfig,
        vision: Option<Qwen2VlVision>,
        token_embd: Vec<f32>,
        layers: Vec<Qwen2Layer>,
        output_norm: RmsNorm,
        output: QuantLinear,
        rope_sections: Option<&[usize]>,
    ) -> ArchResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let rope_base = config.rope_freq_base;

        let mrope = match rope_sections {
            Some(sections) => {
                MRopeTable::new_with_sections(head_dim, max_ctx, rope_base, sections)?
            }
            None => MRopeTable::new(head_dim, max_ctx, rope_base),
        };

        Ok(Self {
            config,
            vision,
            token_embd,
            layers,
            output_norm,
            output,
            mrope,
            dispatcher: KernelDispatcher::new(),
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_q: vec![0.0; num_heads * head_dim],
            buf_k: vec![0.0; num_kv_heads * head_dim],
            buf_v: vec![0.0; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0; num_heads * head_dim],
            buf_gate: vec![0.0; intermediate_size],
            buf_up: vec![0.0; intermediate_size],
            buf_ffn_out: vec![0.0; hidden_size],
            buf_logits: vec![0.0; vocab_size],
            buf_attn_scores: vec![0.0; max_ctx],
        })
    }

    /// Encode an image into visual tokens ready for injection.
    ///
    /// # Returns
    /// `(embeddings, merged_h, merged_w)` where `embeddings` is flat
    /// `[(merged_h × merged_w) × llm_hidden_size]` and the two counts are the
    /// merged grid, needed to assign each visual token its `(h, w)` M-RoPE
    /// position.
    ///
    /// # Errors
    ///
    /// [`ArchError::NotSupported`] when this checkpoint has no vision tower —
    /// this used to be an all-zeros buffer returned as if it were a real
    /// encoding.  Otherwise propagates tower and merger failures.
    pub fn encode_image(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> ArchResult<(Vec<f32>, usize, usize)> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| ArchError::NotSupported {
                detail: "this Qwen2-VL checkpoint has no vision tower: the main GGUF carries no \
                     `clip.vision.*` metadata and no `v.*` tensors. Load the model's separate \
                     mmproj file (general.architecture = \"clip\") to enable image input."
                    .to_string(),
            })?;

        let patch_features = vision.encoder.forward(pixel_values, height, width)?;
        let (patches_h, patches_w) = vision.encoder.patch_grid(height, width);
        let merged = vision.merger.merge(&patch_features, patches_h, patches_w)?;
        Ok((merged, patches_h / 2, patches_w / 2))
    }

    /// Whether a vision tower was loaded.
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Run a forward pass over precomputed input embeddings with explicit
    /// M-RoPE positions.
    ///
    /// This is the injection entry point: `embeds` is `[seq_len × hidden_size]`
    /// and `positions[i]` is token `i`'s `(t, h, w)` triple.  Text tokens use
    /// [`MRopePos::text`]; visual tokens carry their patch grid coordinates.
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidShape`] — length mismatch between `embeds`,
    ///   `positions` and `seq_len`.
    /// * [`ArchError::ConfigMismatch`] — the sequence exceeds the context window.
    pub fn forward_embeds_mrope(
        &mut self,
        embeds: &[f32],
        positions: &[MRopePos],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let seq_len = positions.len();
        if seq_len == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "forward_embeds_mrope: positions must not be empty".to_string(),
            });
        }
        if embeds.len() != seq_len * hidden {
            return Err(ArchError::InvalidShape {
                name: "input embeddings".to_string(),
                expected: vec![seq_len, hidden],
                got: vec![embeds.len()],
            });
        }
        validate_context_bounds(&self.config, kv_cache.seq_len(), seq_len)?;

        for (i, pos) in positions.iter().enumerate() {
            self.buf_hidden
                .copy_from_slice(&embeds[i * hidden..(i + 1) * hidden]);
            self.run_layers_for_token(*pos, kv_cache)?;
            kv_cache.advance();
        }

        self.output_norm.forward(&mut self.buf_hidden);
        self.project_logits()?;
        Ok(std::mem::take(&mut self.buf_logits))
    }

    /// Run a forward pass over precomputed input embeddings with sequential
    /// 1-D positions.
    ///
    /// Convenience wrapper over [`Self::forward_embeds_mrope`] that assigns
    /// `MRopePos::text(start + i)` to every row.
    ///
    /// # Errors
    ///
    /// See [`Self::forward_embeds_mrope`].
    pub fn forward_embeds(
        &mut self,
        embeds: &[f32],
        seq_len: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let start = kv_cache.seq_len();
        let positions: Vec<MRopePos> = (0..seq_len).map(|i| MRopePos::text(start + i)).collect();
        self.forward_embeds_mrope(embeds, &positions, kv_cache)
    }

    /// Build the spliced embedding sequence and matching M-RoPE positions for a
    /// prompt whose `placeholder` token id marks one image.
    ///
    /// Visual tokens receive `(t, h, w)` where `t` is the shared temporal
    /// position of the whole image (llama.cpp advances `n_past` by
    /// `max(nx, ny)` for an M-RoPE image, `mtmd.cpp:1109-1116`), and `(h, w)`
    /// are the merged-grid coordinates.
    ///
    /// # Errors
    ///
    /// [`ArchError::ConfigMismatch`] when the placeholder count is not 1, or an
    /// out-of-vocabulary token id is present.
    pub fn splice_image_prompt(
        &self,
        tokens: &[u32],
        placeholder: u32,
        image: &[f32],
        merged_h: usize,
        merged_w: usize,
        start_pos: usize,
    ) -> ArchResult<(Vec<f32>, Vec<MRopePos>)> {
        let hidden = self.config.hidden_size;
        let n_visual = merged_h * merged_w;
        if image.len() != n_visual * hidden {
            return Err(ArchError::InvalidShape {
                name: "visual tokens".to_string(),
                expected: vec![n_visual, hidden],
                got: vec![image.len()],
            });
        }
        let count = tokens.iter().filter(|&&t| t == placeholder).count();
        if count != 1 {
            return Err(ArchError::ConfigMismatch {
                param: format!("occurrences of placeholder token {placeholder}"),
                expected: "1".to_string(),
                got: count.to_string(),
            });
        }

        let mut embeds = Vec::with_capacity((tokens.len() - 1 + n_visual) * hidden);
        let mut positions = Vec::with_capacity(tokens.len() - 1 + n_visual);
        let mut pos = start_pos;

        for &tok in tokens {
            if tok == placeholder {
                // The whole image shares one temporal position; h/w vary.
                for r in 0..merged_h {
                    for c in 0..merged_w {
                        let idx = r * merged_w + c;
                        embeds.extend_from_slice(&image[idx * hidden..(idx + 1) * hidden]);
                        positions.push(MRopePos {
                            t: pos,
                            h: pos + r,
                            w: pos + c,
                        });
                    }
                }
                // Advance by max(h, w), as the reference does.
                pos += merged_h.max(merged_w);
                continue;
            }
            if tok as usize >= self.config.vocab_size {
                return Err(ArchError::ConfigMismatch {
                    param: "token id".to_string(),
                    expected: format!("< vocab_size ({})", self.config.vocab_size),
                    got: tok.to_string(),
                });
            }
            let offset = tok as usize * hidden;
            let row = self
                .token_embd
                .get(offset..offset + hidden)
                .ok_or_else(|| ArchError::InvalidShape {
                    name: "token_embd".to_string(),
                    expected: vec![offset + hidden],
                    got: vec![self.token_embd.len()],
                })?;
            embeds.extend_from_slice(row);
            positions.push(MRopePos::text(pos));
            pos += 1;
        }

        Ok((embeds, positions))
    }

    fn kernel_for(&self, linear: &QuantLinear) -> ArchResult<Box<dyn oxillama_quant::QuantKernel>> {
        self.dispatcher
            .get_kernel(linear.weight.tensor_type)
            .map_err(ArchError::from)
    }

    /// Load the residual stream with `token`'s embedding row.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let hidden_size = self.config.hidden_size;
        let offset = (token as usize) * hidden_size;
        let row = self
            .token_embd
            .get(offset..offset + hidden_size)
            .ok_or_else(|| ArchError::ConfigMismatch {
                param: "token id".to_string(),
                expected: format!("< {}", self.token_embd.len() / hidden_size.max(1)),
                got: token.to_string(),
            })?;
        self.buf_hidden.copy_from_slice(row);
        Ok(())
    }

    /// Run every transformer layer over the current `buf_hidden`.
    fn run_layers_for_token(
        &mut self,
        pos: MRopePos,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        for layer_idx in 0..self.layers.len() {
            self.layers[layer_idx]
                .attn_norm
                .forward_to(&self.buf_hidden, &mut self.buf_norm);
            self.attention(layer_idx, pos, kv_cache)?;

            self.layers[layer_idx]
                .ffn_norm
                .forward_to(&self.buf_hidden, &mut self.buf_norm);
            self.feed_forward(layer_idx)?;
        }
        Ok(())
    }

    fn attention(
        &mut self,
        layer_idx: usize,
        pos: MRopePos,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let heads_per_kv = num_heads.max(1) / num_kv_heads.max(1);

        {
            let layer = &self.layers[layer_idx];
            let q_kernel = self.kernel_for(&layer.attn_q)?;
            let k_kernel = self.kernel_for(&layer.attn_k)?;
            let v_kernel = self.kernel_for(&layer.attn_v)?;
            layer
                .attn_q
                .forward(&*q_kernel, &self.buf_norm, &mut self.buf_q)?;
            layer
                .attn_k
                .forward(&*k_kernel, &self.buf_norm, &mut self.buf_k)?;
            layer
                .attn_v
                .forward(&*v_kernel, &self.buf_norm, &mut self.buf_v)?;
        }

        for h in 0..num_heads {
            let q_head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
            self.mrope
                .try_apply_mrope(q_head, pos.t, pos.h, pos.w, head_dim)?;
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            self.mrope
                .try_apply_mrope(k_head, pos.t, pos.h, pos.w, head_dim)?;
        }

        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let seq_len = cached_keys.len().checked_div(kv_dim).unwrap_or(0);
        if seq_len > self.buf_attn_scores.len() {
            self.buf_attn_scores.resize(seq_len, 0.0);
        }
        let scale = 1.0 / (head_dim as f32).sqrt();

        self.buf_attn_out.fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv.max(1);
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for p in 0..seq_len {
                let k_offset = p * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_offset..k_offset + head_dim];
                self.buf_attn_scores[p] = q_head
                    .iter()
                    .zip(k_vec.iter())
                    .map(|(&q, &k)| q * k)
                    .sum::<f32>()
                    * scale;
            }

            softmax_inplace(&mut self.buf_attn_scores[..seq_len]);

            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for p in 0..seq_len {
                let v_offset = p * kv_dim + kv_head * head_dim;
                let v_vec = &cached_values[v_offset..v_offset + head_dim];
                let w = self.buf_attn_scores[p];
                for (o, &vv) in out_head.iter_mut().zip(v_vec.iter()) {
                    *o += w * vv;
                }
            }
        }

        let o_kernel = self.kernel_for(&self.layers[layer_idx].attn_output)?;
        let mut proj_out = vec![0.0f32; self.config.hidden_size];
        self.layers[layer_idx].attn_output.forward(
            &*o_kernel,
            &self.buf_attn_out,
            &mut proj_out,
        )?;

        for (h, &p) in self.buf_hidden.iter_mut().zip(proj_out.iter()) {
            *h += p;
        }
        Ok(())
    }

    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];
        let gate_kernel = self.kernel_for(&layer.ffn_gate)?;
        let up_kernel = self.kernel_for(&layer.ffn_up)?;
        let down_kernel = self.kernel_for(&layer.ffn_down)?;

        layer
            .ffn_gate
            .forward(&*gate_kernel, &self.buf_norm, &mut self.buf_gate)?;
        layer
            .ffn_up
            .forward(&*up_kernel, &self.buf_norm, &mut self.buf_up)?;

        swiglu_inplace(&mut self.buf_gate, &self.buf_up);

        layer
            .ffn_down
            .forward(&*down_kernel, &self.buf_gate, &mut self.buf_ffn_out)?;

        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }
        Ok(())
    }

    fn project_logits(&mut self) -> ArchResult<()> {
        if self.buf_logits.len() != self.config.vocab_size {
            self.buf_logits.resize(self.config.vocab_size, 0.0);
        }
        let output_kernel = self.kernel_for(&self.output)?;
        self.output
            .forward(&*output_kernel, &self.buf_hidden, &mut self.buf_logits)?;
        Ok(())
    }

    /// Shared body of `forward` / `embed`.
    fn run_tokens(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<()> {
        validate_token_ids(&self.config, tokens)?;
        let start_pos = kv_cache.seq_len();
        validate_context_bounds(&self.config, start_pos, tokens.len())?;

        for (i, &token) in tokens.iter().enumerate() {
            self.embed_token(token)?;
            self.run_layers_for_token(MRopePos::text(start_pos + i), kv_cache)?;
            kv_cache.advance();
        }
        Ok(())
    }
}

impl ForwardPass for Qwen2VlModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_tokens(tokens, kv_cache)?;
        self.output_norm.forward(&mut self.buf_hidden);
        self.project_logits()?;
        Ok(std::mem::take(&mut self.buf_logits))
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.run_tokens(tokens, kv_cache)?;
        self.output_norm.forward(&mut self.buf_hidden);
        Ok(self.buf_hidden.clone())
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn max_context_length(&self) -> usize {
        self.config.max_context_length
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn apply_lora(&mut self, _lora: &LoadedLora) -> ArchResult<()> {
        Err(ArchError::NotSupported {
            detail: "apply_lora() is not implemented for qwen2vl; an adapter would be silently \
                     ignored"
                .to_string(),
        })
    }
}

/// Numerically stable in-place softmax.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_v = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

pub use crate::qwen2_vl::loader::load_qwen2vl_from_gguf;

#[cfg(test)]
mod tests {
    use super::*;

    fn merger(vis: usize, mm: usize, llm: usize) -> MmMerger {
        MmMerger {
            vis_hidden_size: vis,
            llm_hidden_size: llm,
            mm_hidden_size: mm,
            fc1_weight: vec![0.1f32; mm * 4 * vis],
            fc1_bias: vec![0.0f32; mm],
            fc2_weight: vec![0.1f32; llm * mm],
            fc2_bias: vec![0.0f32; llm],
        }
    }

    #[test]
    fn mm_merger_2x2_compression() {
        let m = merger(4, 16, 8);
        let out = m.merge(&[1.0f32; 4 * 4], 2, 2).expect("merge");
        assert_eq!(out.len(), 8, "2×2 patches → 1 token of llm_hidden = 8");
    }

    #[test]
    fn mm_merger_output_finite() {
        let mut m = merger(4, 16, 8);
        m.fc1_weight = (0..16 * 16).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let feats: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
        for v in m.merge(&feats, 2, 2).expect("merge") {
            assert!(v.is_finite(), "got {v}");
        }
    }

    #[test]
    fn mm_merger_wrong_size_errors() {
        assert!(merger(4, 16, 8).merge(&[0.0f32; 5], 2, 2).is_err());
    }

    /// Odd patch grids cannot be 2×2-merged; the reference asserts this.
    #[test]
    fn mm_merger_rejects_odd_grid() {
        assert!(merger(4, 16, 8).merge(&[0.0f32; 3 * 3 * 4], 3, 3).is_err());
    }

    /// The merger groups four *consecutive* tokens, matching the encoder's
    /// 2×2-block emission order.
    #[test]
    fn mm_merger_groups_consecutive_tokens() {
        // vis=1, mm=4, llm=1, fc1 = identity-ish so the output reveals grouping.
        let m = MmMerger {
            vis_hidden_size: 1,
            llm_hidden_size: 1,
            mm_hidden_size: 1,
            fc1_weight: vec![1.0, 1.0, 1.0, 1.0], // sum of the 4 grouped values
            fc1_bias: vec![0.0],
            fc2_weight: vec![1.0],
            fc2_bias: vec![0.0],
        };
        // 2×4 grid → 1×2 merged. Tokens 0..3 form block 0, 4..7 form block 1.
        let feats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = m.merge(&feats, 2, 4).expect("merge");
        assert_eq!(out.len(), 2);
        // gelu(1+2+3+4) = gelu(10) ≈ 10; gelu(100) ≈ 100.
        assert!((out[0] - 10.0).abs() < 0.01, "block 0 sum, got {}", out[0]);
        assert!((out[1] - 100.0).abs() < 0.01, "block 1 sum, got {}", out[1]);
    }

    #[test]
    fn mrope_pos_text_is_diagonal() {
        let p = MRopePos::text(7);
        assert_eq!((p.t, p.h, p.w), (7, 7, 7));
    }
}

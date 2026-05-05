//! Qwen2-VL full multimodal model.
//!
//! Combines three components:
//! 1. [`Qwen2VlVisionEncoder`] — native ViT that encodes image pixels into
//!    one feature vector per patch.
//! 2. [`MmMerger`] — compresses 2×2 spatial patch blocks into a single LLM
//!    token by reshaping `[N, hidden]` → `[N/4, 4*hidden]` then projecting.
//! 3. `Qwen2Backbone` — Qwen-shaped decoder LLM with M-RoPE instead of
//!    standard 1-D RoPE.
//!
//! ## Forward flow
//!
//! ```text
//! image pixels
//!   └─► Qwen2VlVisionEncoder  ─► [N_patches, vis_hidden]
//!         └─► MmMerger         ─► [N_patches/4, llm_hidden]
//! text tokens
//!   └─► token_embd             ─► [seq_len, llm_hidden]
//!                                      │
//!   (concatenate visual + text tokens) ▼
//!                           Qwen2Backbone (M-RoPE)
//!                                      │
//!                              output logits [vocab_size]
//! ```
//!
//! For text-only inference, skip the vision path entirely and pass tokens
//! directly to the backbone.
//!
//! ## Tensor naming (GGUF)
//!
//! **LLM backbone** — identical to Qwen3:
//! `token_embd.weight`, `blk.{i}.*`, `output_norm.weight`, `output.weight`
//!
//! **MM merger** (2×2 spatial compression → LLM dim):
//! `mm.0.weight` `[llm_hidden, 4*vis_hidden]`
//! `mm.0.bias`   `[llm_hidden]`
//!
//! **Vision encoder** — same `v.*` prefix as LLaVA (different tensor shapes):
//! `v.patch_embd.weight`, `v.blk.{i}.*`, `v.post_ln.weight`

use crate::common::linear::QuantLinear;
use crate::common::mrope::MRopeTable;
use crate::common::rms_norm::RmsNorm;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::lora::LoadedLora;
use crate::qwen2_vl::vision::{Qwen2VlVisionEncoder, VisionBlock};
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::GgufModel;
use oxillama_quant::{KernelDispatcher, QuantTensor};

// ---------------------------------------------------------------------------
// MM Merger: 2×2 → 1 spatial compression
// ---------------------------------------------------------------------------

/// Compresses 2×2 spatial patch blocks into single LLM tokens.
///
/// Reshapes `[N, vis_hidden]` → `[N/4, 4*vis_hidden]` by grouping adjacent
/// 2×2 patch groups, then projects through a linear layer to `llm_hidden`.
pub struct MmMerger {
    /// Vision encoder hidden size.
    pub vis_hidden_size: usize,
    /// LLM hidden size.
    pub llm_hidden_size: usize,
    /// Projection weight: `[llm_hidden, 4 * vis_hidden]`.
    pub proj_weight: Vec<f32>,
    /// Optional projection bias: `[llm_hidden]`.
    pub proj_bias: Vec<f32>,
}

impl MmMerger {
    /// Merge vision patches from a 2×2 spatial neighbourhood.
    ///
    /// # Arguments
    /// * `patch_features` — Flat `[num_patches, vis_hidden_size]` features from
    ///   the vision encoder.
    /// * `patches_h` — Number of patch rows in the spatial grid.
    /// * `patches_w` — Number of patch columns.
    ///
    /// # Returns
    /// Flat `[num_merged, llm_hidden_size]` where `num_merged = patches_h/2 × patches_w/2`.
    /// If either spatial dimension is odd, the last row/column is dropped.
    ///
    /// # Errors
    /// * `ArchError::InvalidShape` if `patch_features.len() != patches_h * patches_w * vis_hidden_size`.
    /// * `ArchError::InvalidConfig` if `vis_hidden_size == 0`.
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
        let expected = patches_h * patches_w * self.vis_hidden_size;
        if patch_features.len() != expected {
            return Err(ArchError::InvalidShape {
                name: "patch_features".to_string(),
                expected: vec![expected],
                got: vec![patch_features.len()],
            });
        }

        // Merged grid size: floor(h/2) × floor(w/2)
        let merged_h = patches_h / 2;
        let merged_w = patches_w / 2;
        let num_merged = merged_h * merged_w;
        let merged_in_dim = 4 * self.vis_hidden_size;
        let mut out = Vec::with_capacity(num_merged * self.llm_hidden_size);

        for mr in 0..merged_h {
            for mc in 0..merged_w {
                // Collect 2×2 neighbourhood: (2r, 2c), (2r, 2c+1), (2r+1, 2c), (2r+1, 2c+1).
                let mut merged_vec = Vec::with_capacity(merged_in_dim);
                for dr in 0..2usize {
                    for dc in 0..2usize {
                        let pr = 2 * mr + dr;
                        let pc = 2 * mc + dc;
                        let offset = (pr * patches_w + pc) * self.vis_hidden_size;
                        merged_vec.extend_from_slice(
                            &patch_features[offset..offset + self.vis_hidden_size],
                        );
                    }
                }

                // Project: [4*vis_hidden → llm_hidden]
                let projected = linear_proj_f32(
                    &merged_vec,
                    &self.proj_weight,
                    &self.proj_bias,
                    self.llm_hidden_size,
                    merged_in_dim,
                );
                out.extend_from_slice(&projected);
            }
        }

        Ok(out)
    }
}

/// Simple f32 matrix-vector multiply + optional bias.
fn linear_proj_f32(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for (i, yi) in y.iter_mut().enumerate() {
        let row_start = i * in_dim;
        let row_end = row_start + in_dim;
        if row_end <= weight.len() {
            *yi = weight[row_start..row_end]
                .iter()
                .zip(x.iter())
                .map(|(&w, &xv)| w * xv)
                .sum::<f32>();
        }
        *yi += bias.get(i).copied().unwrap_or(0.0);
    }
    y
}

// ---------------------------------------------------------------------------
// Qwen2 backbone layer (structurally identical to Qwen3)
// ---------------------------------------------------------------------------

/// A single Qwen2-backbone transformer layer.
pub struct Qwen2Layer {
    pub attn_norm: RmsNorm,
    pub attn_q: QuantLinear,
    pub attn_k: QuantLinear,
    pub attn_v: QuantLinear,
    pub attn_output: QuantLinear,
    pub ffn_norm: RmsNorm,
    pub ffn_gate: QuantLinear,
    pub ffn_up: QuantLinear,
    pub ffn_down: QuantLinear,
}

// ---------------------------------------------------------------------------
// Full Qwen2-VL model
// ---------------------------------------------------------------------------

/// Combined Qwen2-VL model.
///
/// Holds the vision encoder, MM merger, Qwen2 language backbone, and the
/// shared M-RoPE table that handles both text and vision position encoding.
pub struct Qwen2VlModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Native ViT vision encoder.
    pub vision_encoder: Qwen2VlVisionEncoder,
    /// 2×2 spatial patch compressor.
    pub mm_merger: MmMerger,
    /// Token embedding table: `[vocab_size, hidden_size]`.
    pub token_embd: Vec<f32>,
    /// Qwen2 language backbone layers.
    pub layers: Vec<Qwen2Layer>,
    /// Final RMSNorm.
    pub output_norm: RmsNorm,
    /// LM head projection: `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Multimodal RoPE table.
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
    pub fn new(
        config: ModelConfig,
        vision_encoder: Qwen2VlVisionEncoder,
        mm_merger: MmMerger,
        token_embd: Vec<f32>,
        layers: Vec<Qwen2Layer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let rope_base = config.rope_freq_base;

        // Build M-RoPE table — the head_dim must be divisible by 6 for exact
        // three-way split.  We use the configured head_dim directly.
        let mrope = MRopeTable::new(head_dim, max_ctx, rope_base);
        let dispatcher = KernelDispatcher::new();

        Self {
            config,
            vision_encoder,
            mm_merger,
            token_embd,
            layers,
            output_norm,
            output,
            mrope,
            dispatcher,
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_q: vec![0.0; num_heads * head_dim],
            buf_k: vec![0.0; num_kv_heads * head_dim],
            buf_v: vec![0.0; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0; hidden_size],
            buf_gate: vec![0.0; intermediate_size],
            buf_up: vec![0.0; intermediate_size],
            buf_ffn_out: vec![0.0; hidden_size],
            buf_logits: vec![0.0; vocab_size],
            buf_attn_scores: vec![0.0; max_ctx],
        }
    }

    /// Encode an image into visual tokens ready to be injected into the LLM.
    ///
    /// # Returns
    /// Flat `[num_visual_tokens, llm_hidden_size]` where
    /// `num_visual_tokens = (patches_h/2) × (patches_w/2)`.
    ///
    /// # Arguments
    /// * `pixel_values` — `[3 × height × width]` normalized f32.
    /// * `height` / `width` — image dimensions in pixels.
    pub fn encode_image(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> ArchResult<Vec<f32>> {
        // Run ViT encoder.
        let patch_features = self.vision_encoder.forward(pixel_values, height, width)?;
        let (patches_h, patches_w) = self.vision_encoder.patch_grid(height, width);

        // Merge 2×2 blocks → single LLM tokens.
        self.mm_merger.merge(&patch_features, patches_h, patches_w)
    }

    fn kernel_for(&self, linear: &QuantLinear) -> ArchResult<Box<dyn oxillama_quant::QuantKernel>> {
        self.dispatcher
            .get_kernel(linear.weight.tensor_type)
            .map_err(ArchError::from)
    }

    fn embed_token(&mut self, token: u32) {
        let hidden_size = self.config.hidden_size;
        let offset = token as usize * hidden_size;
        if offset + hidden_size <= self.token_embd.len() {
            self.buf_hidden
                .copy_from_slice(&self.token_embd[offset..offset + hidden_size]);
        } else {
            self.buf_hidden.fill(0.0);
        }
    }

    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
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

        // Apply M-RoPE: text path uses (pos, pos, pos) → degrades to 1D RoPE.
        for h in 0..num_heads {
            let q_head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
            self.mrope
                .apply_mrope(q_head, position, position, position, head_dim);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            self.mrope
                .apply_mrope(k_head, position, position, position, head_dim);
        }

        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = kv_cache.get_keys(layer_idx)?;
        let cached_values = kv_cache.get_values(layer_idx)?;
        let seq_len = position + 1;
        let scale = 1.0 / (head_dim as f32).sqrt();

        self.buf_attn_out.fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in 0..seq_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_offset..k_offset + head_dim];
                let score = q_head
                    .iter()
                    .zip(k_vec.iter())
                    .map(|(&q, &k)| q * k)
                    .sum::<f32>()
                    * scale;
                self.buf_attn_scores[pos] = score;
            }

            softmax_inplace(&mut self.buf_attn_scores[..seq_len]);

            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for pos in 0..seq_len {
                let v_offset = pos * kv_dim + kv_head * head_dim;
                let v_vec = &cached_values[v_offset..v_offset + head_dim];
                let w = self.buf_attn_scores[pos];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
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
}

impl ForwardPass for Qwen2VlModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            self.embed_token(token);

            for layer_idx in 0..self.layers.len() {
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&self.buf_hidden.clone(), &mut self.buf_norm);

                self.attention(layer_idx, position, kv_cache)?;

                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&self.buf_hidden.clone(), &mut self.buf_norm);

                self.feed_forward(layer_idx)?;
            }

            kv_cache.advance();
        }

        self.output_norm.forward(&mut self.buf_hidden);

        let output_kernel = self.kernel_for(&self.output)?;
        self.output
            .forward(&*output_kernel, &self.buf_hidden, &mut self.buf_logits)?;

        Ok(self.buf_logits.clone())
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            self.embed_token(token);

            for layer_idx in 0..self.layers.len() {
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&self.buf_hidden.clone(), &mut self.buf_norm);

                self.attention(layer_idx, position, kv_cache)?;

                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&self.buf_hidden.clone(), &mut self.buf_norm);

                self.feed_forward(layer_idx)?;
            }

            kv_cache.advance();
        }

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
        // LoRA application for Qwen2-VL follows the same pattern as other archs.
        // Deferred for this initial implementation — returns Ok for compatibility.
        Ok(())
    }

    fn unapply_all_loras(&mut self) {}
}

// ---------------------------------------------------------------------------
// GGUF loader
// ---------------------------------------------------------------------------

/// Load a Qwen2-VL model from a `GgufModel`.
///
/// Falls back gracefully when optional vision tensors are absent.
pub fn load_qwen2vl_from_gguf(model: &GgufModel, config: &ModelConfig) -> ArchResult<Qwen2VlModel> {
    let dispatcher = KernelDispatcher::new();

    // ── Token embeddings ──────────────────────────────────────────────────
    let embd_info = model.file.tensors.get("token_embd.weight")?;
    let embd_data = model.tensor_data("token_embd.weight")?;
    let token_embd = dequant_to_f32_local(embd_info, embd_data, &dispatcher)?;

    // ── LLM backbone layers ───────────────────────────────────────────────
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;

        let attn_q = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_q.weight"),
            &format!("{prefix}.attn_q.bias"),
        )?;
        let attn_k = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_k.weight"),
            &format!("{prefix}.attn_k.bias"),
        )?;
        let attn_v = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_v.weight"),
            &format!("{prefix}.attn_v.bias"),
        )?;
        let attn_output = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_output.weight"),
            &format!("{prefix}.attn_output.bias"),
        )?;
        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        layers.push(Qwen2Layer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    let output_norm_weight = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_weight, config.rms_norm_eps);
    let output = load_quant_linear(model, "output.weight")?;

    // ── Vision encoder parameters ─────────────────────────────────────────
    // Infer vision config from the loaded model config.
    let vis_cfg = config.vision_config.as_ref();
    let vis_patch_size = vis_cfg.map_or(14, |v| v.patch_size);
    let vis_hidden = vis_cfg.map_or(config.hidden_size, |v| v.hidden_size);
    let vis_num_heads = vis_cfg.map_or(16, |v| v.num_heads);
    let vis_num_layers = vis_cfg.map_or(0, |v| v.num_layers);
    let vis_window_size = vis_cfg.map_or(8, |v| v.window_size);

    let patch_flat = vis_patch_size * vis_patch_size * 3;

    let patch_embd_weight = load_optional_f32(model, &dispatcher, "v.patch_embd.weight")
        .unwrap_or_else(|| vec![0.0f32; vis_hidden * patch_flat]);
    let patch_embd_bias =
        load_optional_f32(model, &dispatcher, "v.patch_embd.bias").unwrap_or_default();
    let post_ln_weight =
        load_optional_f32(model, &dispatcher, "v.post_ln.weight").unwrap_or_default();

    let mut vis_layers = Vec::with_capacity(vis_num_layers);
    for i in 0..vis_num_layers {
        let pfx = format!("v.blk.{i}");
        vis_layers.push(VisionBlock {
            attn_norm_weight: load_optional_f32(
                model,
                &dispatcher,
                &format!("{pfx}.attn_norm.weight"),
            )
            .unwrap_or_else(|| vec![1.0f32; vis_hidden]),
            ffn_norm_weight: load_optional_f32(
                model,
                &dispatcher,
                &format!("{pfx}.ffn_norm.weight"),
            )
            .unwrap_or_else(|| vec![1.0f32; vis_hidden]),
            attn_q_weight: load_optional_f32(model, &dispatcher, &format!("{pfx}.attn_q.weight"))
                .unwrap_or_else(|| vec![0.0f32; vis_hidden * vis_hidden]),
            attn_k_weight: load_optional_f32(model, &dispatcher, &format!("{pfx}.attn_k.weight"))
                .unwrap_or_else(|| vec![0.0f32; vis_hidden * vis_hidden]),
            attn_v_weight: load_optional_f32(model, &dispatcher, &format!("{pfx}.attn_v.weight"))
                .unwrap_or_else(|| vec![0.0f32; vis_hidden * vis_hidden]),
            attn_out_weight: load_optional_f32(
                model,
                &dispatcher,
                &format!("{pfx}.attn_out.weight"),
            )
            .unwrap_or_else(|| vec![0.0f32; vis_hidden * vis_hidden]),
            ffn_up_weight: load_optional_f32(model, &dispatcher, &format!("{pfx}.ffn_up.weight"))
                .unwrap_or_else(|| vec![0.0f32; vis_hidden * vis_hidden * 4]),
            ffn_down_weight: load_optional_f32(
                model,
                &dispatcher,
                &format!("{pfx}.ffn_down.weight"),
            )
            .unwrap_or_else(|| vec![0.0f32; vis_hidden * vis_hidden * 4]),
        });
    }

    let vision_encoder = Qwen2VlVisionEncoder {
        patch_size: vis_patch_size,
        window_size: vis_window_size,
        hidden_size: vis_hidden,
        num_heads: vis_num_heads,
        layers: vis_layers,
        patch_embd_weight,
        patch_embd_bias,
        post_ln_weight,
    };

    // ── MM Merger ─────────────────────────────────────────────────────────
    let merged_in_dim = 4 * vis_hidden;
    let mm0_weight = load_optional_f32(model, &dispatcher, "mm.0.weight")
        .unwrap_or_else(|| vec![0.0f32; config.hidden_size * merged_in_dim]);
    let mm0_bias = load_optional_f32(model, &dispatcher, "mm.0.bias").unwrap_or_default();

    let mm_merger = MmMerger {
        vis_hidden_size: vis_hidden,
        llm_hidden_size: config.hidden_size,
        proj_weight: mm0_weight,
        proj_bias: mm0_bias,
    };

    Ok(Qwen2VlModel::new(
        config.clone(),
        vision_encoder,
        mm_merger,
        token_embd,
        layers,
        output_norm,
        output,
    ))
}

// ---------------------------------------------------------------------------
// Private tensor loading helpers
// ---------------------------------------------------------------------------

fn load_quant_linear(model: &GgufModel, name: &str) -> ArchResult<QuantLinear> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model.tensor_data(name)?;
    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let tensor = QuantTensor::new(data.to_vec(), shape, info.tensor_type);
    Ok(QuantLinear::new(tensor, None))
}

fn load_quant_linear_with_bias(
    model: &GgufModel,
    weight_name: &str,
    bias_name: &str,
) -> ArchResult<QuantLinear> {
    let info = model
        .file
        .tensors
        .get(weight_name)
        .map_err(|_| ArchError::MissingTensor {
            name: weight_name.to_string(),
        })?;
    let data = model.tensor_data(weight_name)?;
    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let tensor = QuantTensor::new(data.to_vec(), shape, info.tensor_type);

    let bias = if model.file.tensors.contains(bias_name) {
        let bd = model.tensor_data(bias_name)?;
        let bi = model.file.tensors.get(bias_name)?;
        let n = bi.n_elements() as usize;
        let mut bv = vec![0.0f32; n];
        for (i, chunk) in bd.chunks_exact(4).enumerate().take(n) {
            bv[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Some(bv)
    } else {
        None
    };

    Ok(QuantLinear::new(tensor, bias))
}

fn load_rms_norm_weight(model: &GgufModel, name: &str) -> ArchResult<Vec<f32>> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model.tensor_data(name)?;
    let dispatcher = KernelDispatcher::new();
    dequant_to_f32_local(info, data, &dispatcher)
}

fn load_optional_f32(
    model: &GgufModel,
    dispatcher: &KernelDispatcher,
    name: &str,
) -> Option<Vec<f32>> {
    let info = model.file.tensors.get(name).ok()?;
    let data = model.tensor_data(name).ok()?;
    dequant_to_f32_local(info, data, dispatcher).ok()
}

fn dequant_to_f32_local(
    info: &oxillama_gguf::TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    let n_elements = info.n_elements() as usize;
    let tensor_type = info.tensor_type;

    if tensor_type == oxillama_gguf::GgufTensorType::F32 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(4).enumerate().take(n_elements) {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    if tensor_type == oxillama_gguf::GgufTensorType::F16 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(2).enumerate().take(n_elements) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            out[i] = half::f16::from_bits(bits).to_f32();
        }
        return Ok(out);
    }

    let kernel = dispatcher.get_kernel(tensor_type)?;
    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();
    let n_blocks = n_elements.div_ceil(block_size);

    let mut out = vec![0.0f32; n_elements];
    for blk in 0..n_blocks {
        let data_offset = blk * block_bytes;
        let out_offset = blk * block_size;
        let block_data = &data[data_offset..data_offset + block_bytes];
        let out_slice = &mut out[out_offset..out_offset.saturating_add(block_size).min(n_elements)];
        kernel.dequant_block(block_data, out_slice)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::mrope::MRopeTable;
    use crate::config::VisionConfig;
    use oxillama_gguf::test_utils::build_minimal_qwen2vl_gguf;
    use oxillama_gguf::GgufModel;

    // ── MmMerger tests ──────────────────────────────────────────────────────

    /// 2×2 patch block → 1 merged vector of llm_hidden_size.
    #[test]
    fn qwen2vl_mm_merger_2x2_compression() {
        let llm_hidden = 8usize;
        let vis_hidden = 4usize;
        let merger = MmMerger {
            vis_hidden_size: vis_hidden,
            llm_hidden_size: llm_hidden,
            proj_weight: vec![0.1f32; llm_hidden * (4 * vis_hidden)],
            proj_bias: vec![0.0f32; llm_hidden],
        };

        // 2×2 grid of patches, each of vis_hidden=4 dims → 4 patches total.
        let patch_features = vec![1.0f32; 4 * vis_hidden];
        let out = merger
            .merge(&patch_features, 2, 2)
            .expect("merge should succeed");
        assert_eq!(
            out.len(),
            llm_hidden,
            "2×2 patches should produce 1 merged token of llm_hidden_size={llm_hidden}"
        );
    }

    /// Merger output must be non-empty and finite for non-trivial inputs.
    #[test]
    fn qwen2vl_mm_merger_output_finite() {
        let llm_hidden = 8usize;
        let vis_hidden = 4usize;
        let merger = MmMerger {
            vis_hidden_size: vis_hidden,
            llm_hidden_size: llm_hidden,
            proj_weight: (0..llm_hidden * 4 * vis_hidden)
                .map(|i| (i as f32 + 1.0) * 0.01)
                .collect(),
            proj_bias: vec![0.0f32; llm_hidden],
        };
        let patch_features: Vec<f32> = (0..4 * vis_hidden)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let out = merger.merge(&patch_features, 2, 2).expect("merge ok");
        for &v in &out {
            assert!(v.is_finite(), "merger output must be finite, got {v}");
        }
    }

    /// Merging wrong-size input must return Err.
    #[test]
    fn qwen2vl_mm_merger_wrong_size_errors() {
        let merger = MmMerger {
            vis_hidden_size: 4,
            llm_hidden_size: 8,
            proj_weight: vec![0.0; 8 * 16],
            proj_bias: vec![],
        };
        // Provide 5 elements instead of 4*4 = 16
        let bad = vec![0.0f32; 5];
        assert!(merger.merge(&bad, 2, 2).is_err());
    }

    // ── Full model tests ────────────────────────────────────────────────────

    /// Forward pass on text-only input (no image) must succeed and return
    /// a finite logit vector of shape [vocab_size].
    #[test]
    fn qwen2vl_no_vision_input_text_only_works() {
        use crate::traits::KvCacheAccess;

        let bytes = build_minimal_qwen2vl_gguf();
        let gguf = GgufModel::from_bytes(bytes).expect("parse qwen2vl gguf");
        let mut meta = oxillama_gguf::MetadataStore::new();
        meta.insert(
            "general.architecture".to_string(),
            oxillama_gguf::MetadataValue::String("qwen2vl".to_string()),
        );
        meta.insert(
            "qwen2vl.embedding_length".to_string(),
            oxillama_gguf::MetadataValue::Uint32(32),
        );
        meta.insert(
            "qwen2vl.feed_forward_length".to_string(),
            oxillama_gguf::MetadataValue::Uint32(64),
        );
        meta.insert(
            "qwen2vl.block_count".to_string(),
            oxillama_gguf::MetadataValue::Uint32(1),
        );
        meta.insert(
            "qwen2vl.attention.head_count".to_string(),
            oxillama_gguf::MetadataValue::Uint32(2),
        );
        meta.insert(
            "qwen2vl.attention.head_count_kv".to_string(),
            oxillama_gguf::MetadataValue::Uint32(2),
        );
        meta.insert(
            "qwen2vl.context_length".to_string(),
            oxillama_gguf::MetadataValue::Uint32(128),
        );
        meta.insert(
            "qwen2vl.vocab_size".to_string(),
            oxillama_gguf::MetadataValue::Uint32(32),
        );

        let config = crate::config::ModelConfig::from_metadata(&meta).expect("config parse");
        let mut model = load_qwen2vl_from_gguf(&gguf, &config).expect("load qwen2vl model");

        struct FlatKv {
            keys: Vec<Vec<f32>>,
            vals: Vec<Vec<f32>>,
            pos: usize,
        }
        impl KvCacheAccess for FlatKv {
            fn store_kv(
                &mut self,
                layer: usize,
                k: &[f32],
                v: &[f32],
            ) -> crate::error::ArchResult<()> {
                while self.keys.len() <= layer {
                    self.keys.push(Vec::new());
                    self.vals.push(Vec::new());
                }
                self.keys[layer].extend_from_slice(k);
                self.vals[layer].extend_from_slice(v);
                Ok(())
            }
            fn get_keys(&self, layer: usize) -> crate::error::ArchResult<&[f32]> {
                Ok(self.keys.get(layer).map_or(&[], |v| v.as_slice()))
            }
            fn get_values(&self, layer: usize) -> crate::error::ArchResult<&[f32]> {
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
        let tokens = [1u32, 2, 3];
        let logits = model.forward(&tokens, &mut kv).expect("text-only forward");

        assert_eq!(logits.len(), 32, "output should have vocab_size=32 logits");
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] must be finite, got {v}");
        }
    }

    /// Vision encoder dynamic resolution: output shape must scale with image size.
    #[test]
    fn qwen2vl_vision_dynamic_resolution() {
        let patch_size = 4usize;
        let hidden_size = 8usize;
        let enc = Qwen2VlVisionEncoder {
            patch_size,
            window_size: 8,
            hidden_size,
            num_heads: 2,
            layers: vec![],
            patch_embd_weight: vec![0.0f32; hidden_size * patch_size * patch_size * 3],
            patch_embd_bias: vec![0.0f32; hidden_size],
            post_ln_weight: vec![1.0f32; hidden_size],
        };

        let h1 = 8usize;
        let w1 = 8usize;
        let h2 = 16usize;
        let w2 = 16usize;

        let out1 = enc
            .forward(&vec![0.0f32; 3 * h1 * w1], h1, w1)
            .expect("encode 8×8");
        let out2 = enc
            .forward(&vec![0.0f32; 3 * h2 * w2], h2, w2)
            .expect("encode 16×16");

        // 8×8 with patch_size=4 → (8/4)² = 4 patches
        assert_eq!(
            out1.len(),
            4 * hidden_size,
            "8×8 image should give 4 patches"
        );
        // 16×16 with patch_size=4 → (16/4)² = 16 patches
        assert_eq!(
            out2.len(),
            16 * hidden_size,
            "16×16 image should give 16 patches"
        );
    }

    // ── MRopeTable tests (also in mrope.rs; extras here for model context) ──

    /// MRopeTable applied to zero vector must return all zeros.
    #[test]
    fn mrope_zero_input_stays_zero() {
        let mrope = MRopeTable::new(24, 32, 10000.0);
        let mut x = vec![0.0f32; 24];
        mrope.apply_mrope(&mut x, 0, 0, 0, 24);
        for v in &x {
            assert_eq!(*v, 0.0f32, "zero input must stay zero");
        }
    }

    // ── Vision config and tensor names ─────────────────────────────────────

    /// VisionConfig must be constructable with sensible defaults.
    #[test]
    fn vision_config_constructable() {
        let vc = VisionConfig {
            image_size: 448,
            patch_size: 14,
            hidden_size: 1152,
            num_heads: 16,
            num_layers: 32,
            window_size: 8,
        };
        assert_eq!(vc.patch_size, 14);
        assert_eq!(vc.window_size, 8);
        assert_eq!(vc.num_layers, 32);
    }
}

//! CLIP ViT vision tower, as loaded from a `clip`-architecture mmproj GGUF.
//!
//! Normative reference: `tools/mtmd/models/llava.cpp` (`clip_graph_llava::build`)
//! plus the shared loader in `tools/mtmd/clip.cpp` around line 1400.
//!
//! ## What this module gets from the reference
//!
//! * **Learned class token.** `v.class_embd` is a real tensor
//!   (`clip-impl.h:70`, loaded at `clip.cpp:1377`) and `n_pos = n_patches +
//!   (class_embedding ? 1 : 0)` (`llava.cpp:7`).  A zero vector is *not* a
//!   stand-in: the CLS row participates in every self-attention layer and a
//!   zero row changes every patch's output.  When the checkpoint has no
//!   `v.class_embd`, the reference runs with **no** CLS row at all — it does
//!   not fabricate one.
//! * **QuickGELU by default.** `clip.cpp:1070-1088` reads `clip.use_gelu` /
//!   `clip.use_silu` and falls back to `FFN_GELU_QUICK` when neither is set.
//!   OpenAI CLIP ViT-L/14 — the LLaVA-1.5 tower — ships neither key, so the
//!   activation is `x · sigmoid(1.702 x)`, not the tanh approximation.
//! * **Geometry from metadata.** `clip.vision.{image_size, patch_size,
//!   embedding_length, block_count, attention.head_count,
//!   attention.layer_norm_epsilon}` (`clip-impl.h:29-45`) — all *required* on
//!   the reference read path (`clip.cpp:1009-1018`).
//! * **`n_layer - 1` blocks are executed.** `llava.cpp:11-30` computes
//!   `max_feature_layer = hparams.n_layer - 1` unless
//!   `clip.vision.feature_layer` overrides it, while `model.layers` is sized to
//!   the full `n_layer` (`clip.cpp:1399`).  The final block is loaded and never
//!   run.
//! * **`post_ln` *is* applied.** `llava.cpp:128-131` applies it unconditionally
//!   after the loop, including when only `n_layer - 1` blocks ran.
//!
//! ## Where this module deliberately differs from llama.cpp HEAD
//!
//! HEAD **appends** the class embedding (`llava.cpp:34-37`,
//! `ggml_concat(ctx0, inp, model.class_embedding, 1)` puts it at row
//! `n_patches`) while still gathering output rows `1..=n_patches`
//! (`clip.cpp:3866-3872`, `patches[i] = i + patch_offset`).  That combination
//! drops patch 0 and keeps CLS, which contradicts both
//!
//! * the pre-refactor llama.cpp code (commit `32916a490^`, `clip.cpp:1133-1136`)
//!   which `ggml_acc`-ed `class_embedding` to byte offset **0**, and
//! * HuggingFace `CLIPVisionModel`, whose `embeddings` are
//!   `cat([class_embeds, patch_embeds], dim=1)` with the LLaVA feature taken as
//!   `hidden_states[:, 1:]`.
//!
//! The LLaVA-1.5 weights were trained with CLS **first**, so this module
//! prepends it, aligns position-embedding row 0 with CLS, and drops row 0 from
//! the output.

use crate::error::{ArchError, ArchResult};

/// Feed-forward activation of the CLIP tower.
///
/// Selected by `clip.use_gelu` / `clip.use_silu`; neither key present means
/// [`ClipFfnOp::QuickGelu`] (`clip.cpp:1070-1088`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClipFfnOp {
    /// `x · sigmoid(1.702 x)` — OpenAI CLIP's activation and clip.cpp's default.
    #[default]
    QuickGelu,
    /// tanh-approximation GELU (`clip.use_gelu = true`).
    Gelu,
    /// SiLU / swish (`clip.use_silu = true`).
    Silu,
}

impl ClipFfnOp {
    /// Apply the activation element-wise.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::QuickGelu => quick_gelu(x),
            Self::Gelu => crate::common::gelu::gelu(x),
            Self::Silu => x / (1.0 + (-x).exp()),
        }
    }
}

/// QuickGELU: `x · sigmoid(1.702 x)`.
///
/// This is `ggml_gelu_quick`, the activation OpenAI's CLIP ViT-L/14 was trained
/// with.  It is **not** interchangeable with the tanh GELU approximation: the
/// two differ by up to ~0.02 absolute around `x ≈ ±2`, which compounds across
/// 23 transformer blocks.
#[inline]
pub fn quick_gelu(x: f32) -> f32 {
    x / (1.0 + (-1.702_f32 * x).exp())
}

/// Geometry and hyper-parameters of a CLIP vision tower.
///
/// Every field maps to a `clip.vision.*` metadata key; nothing here is
/// hard-coded to ViT-L/14 any more.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipVisionParams {
    /// `clip.vision.embedding_length` (1024 for ViT-L/14).
    pub hidden_size: usize,
    /// `clip.vision.attention.head_count` (16 for ViT-L/14).
    pub num_heads: usize,
    /// `clip.vision.block_count` (24 for ViT-L/14).
    pub num_layers: usize,
    /// `clip.vision.patch_size` (14 for ViT-L/14).
    pub patch_size: usize,
    /// `clip.vision.image_size` (336 for LLaVA-1.5).
    pub image_size: usize,
    /// `clip.vision.attention.layer_norm_epsilon`.
    pub eps: f32,
    /// Activation from `clip.use_gelu` / `clip.use_silu`.
    pub ffn_op: ClipFfnOp,
    /// `clip.vision.feature_layer`, the deepest block to execute.
    ///
    /// `None` selects the LLaVA default `num_layers - 1` — i.e. the last block
    /// is loaded but skipped, the GGUF spelling of
    /// `vision_feature_layer = -2`.
    pub feature_layer: Option<usize>,
}

impl Default for ClipVisionParams {
    /// OpenAI CLIP ViT-L/14-336, the LLaVA-1.5 tower.
    ///
    /// Only a fallback for checkpoints that omit the keys; the loader prefers
    /// metadata in every case.
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_heads: 16,
            num_layers: 24,
            patch_size: 14,
            image_size: 336,
            eps: 1e-5,
            ffn_op: ClipFfnOp::QuickGelu,
            feature_layer: None,
        }
    }
}

impl ClipVisionParams {
    /// Number of visual patches (excludes the class token).
    pub fn num_patches(&self) -> usize {
        if self.patch_size == 0 {
            return 0;
        }
        let side = self.image_size / self.patch_size;
        side * side
    }

    /// Number of transformer blocks actually executed.
    ///
    /// `llava.cpp:11-30`: `max_feature_layer = feature_layer.unwrap_or(n_layer - 1)`.
    pub fn executed_layers(&self) -> usize {
        match self.feature_layer {
            Some(l) => l.min(self.num_layers),
            None => self.num_layers.saturating_sub(1),
        }
    }

    /// Validate that the geometry is self-consistent.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] for a zero dimension, a `num_heads` that
    /// does not divide `hidden_size`, or an `image_size` that is not a whole
    /// number of patches.
    pub fn validate(&self) -> ArchResult<()> {
        if self.hidden_size == 0 || self.patch_size == 0 || self.image_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "ClipVisionParams: hidden_size, patch_size and image_size must all be > 0"
                    .to_string(),
            });
        }
        if self.num_heads == 0 || !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "ClipVisionParams: hidden_size ({}) must be a non-zero multiple of num_heads ({})",
                    self.hidden_size, self.num_heads
                ),
            });
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "ClipVisionParams: image_size ({}) must be a whole number of patch_size ({})",
                    self.image_size, self.patch_size
                ),
            });
        }
        Ok(())
    }
}

/// Single CLIP transformer encoder layer.
///
/// Tensor names, all under `v.blk.{i}.` — see `clip-impl.h:76-87`.  The
/// per-block norms are `ln1` / `ln2`; the CLIP namespace has no `attn_norm`.
#[derive(Debug, Clone, Default)]
pub struct ClipEncoderLayer {
    /// `ln1.weight`.
    pub ln1_weight: Vec<f32>,
    /// `ln1.bias`.
    pub ln1_bias: Vec<f32>,
    /// `ln2.weight`.
    pub ln2_weight: Vec<f32>,
    /// `ln2.bias`.
    pub ln2_bias: Vec<f32>,
    /// `attn_q.weight`.
    pub q_weight: Vec<f32>,
    /// `attn_k.weight`.
    pub k_weight: Vec<f32>,
    /// `attn_v.weight`.
    pub v_weight: Vec<f32>,
    /// `attn_out.weight`.
    pub out_weight: Vec<f32>,
    /// `attn_q.bias`.
    pub q_bias: Vec<f32>,
    /// `attn_k.bias`.
    pub k_bias: Vec<f32>,
    /// `attn_v.bias`.
    pub v_bias: Vec<f32>,
    /// `attn_out.bias`.
    pub out_bias: Vec<f32>,
    /// `ffn_up.weight`, row-major `[ffn_dim, hidden]`.
    pub fc1_weight: Vec<f32>,
    /// `ffn_up.bias`.
    pub fc1_bias: Vec<f32>,
    /// `ffn_down.weight`, row-major `[hidden, ffn_dim]`.
    pub fc2_weight: Vec<f32>,
    /// `ffn_down.bias`.
    pub fc2_bias: Vec<f32>,
}

/// CLIP Vision Transformer encoder.
#[derive(Debug, Clone)]
pub struct ClipEncoder {
    /// Geometry read from `clip.vision.*`.
    pub params: ClipVisionParams,
    /// `v.class_embd` — the learned CLS token, `[hidden_size]`.
    ///
    /// Empty means the checkpoint has no class token, in which case no CLS row
    /// is created (matching `n_pos = n_patches + (class_embedding ? 1 : 0)`).
    pub class_embd: Vec<f32>,
    /// `v.patch_embd.weight`, row-major `[hidden_size, patch_size² × 3]`.
    pub patch_embd_weight: Vec<f32>,
    /// `v.patch_embd.bias` (optional).
    pub patch_embd_bias: Vec<f32>,
    /// `v.position_embd.weight`, `[num_positions, hidden_size]`.
    pub position_embd: Vec<f32>,
    /// `v.pre_ln.weight` (optional).
    pub pre_ln_weight: Vec<f32>,
    /// `v.pre_ln.bias`.
    pub pre_ln_bias: Vec<f32>,
    /// `v.post_ln.weight` (optional).
    pub post_ln_weight: Vec<f32>,
    /// `v.post_ln.bias`.
    pub post_ln_bias: Vec<f32>,
    /// Transformer blocks — all `num_layers` of them, even though
    /// [`ClipVisionParams::executed_layers`] of them run.
    pub layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    /// Number of visual patches (excludes the class token).
    pub fn num_patches(&self) -> usize {
        self.params.num_patches()
    }

    /// CLIP hidden size.
    pub fn hidden_size(&self) -> usize {
        self.params.hidden_size
    }

    /// Whether the tower carries a learned class token.
    pub fn has_class_token(&self) -> bool {
        !self.class_embd.is_empty()
    }

    /// Encode an image into patch feature vectors.
    ///
    /// # Arguments
    /// * `pixels` — flat `[3 × image_size × image_size]`, channels-first,
    ///   already normalized by `clip.vision.image_mean` / `image_std`.
    ///
    /// # Returns
    /// Flat `[num_patches × hidden_size]`; the CLS row, when present, is
    /// dropped.
    ///
    /// # Errors
    /// * [`ArchError::InvalidConfig`] — inconsistent geometry.
    /// * [`ArchError::InvalidShape`] — wrong pixel count, or a weight buffer
    ///   too short for its declared shape.
    /// * [`ArchError::MissingTensor`] — an absent required tensor.
    pub fn encode(&self, pixels: &[f32]) -> ArchResult<Vec<f32>> {
        self.params.validate()?;
        let hidden = self.params.hidden_size;
        let patch_size = self.params.patch_size;
        let image_size = self.params.image_size;

        let expected_pixels = 3 * image_size * image_size;
        if pixels.len() != expected_pixels {
            return Err(ArchError::InvalidShape {
                name: "clip pixels".to_string(),
                expected: vec![expected_pixels],
                got: vec![pixels.len()],
            });
        }

        let num_patches = self.params.num_patches();
        let patch_flat = patch_size * patch_size * 3;
        if self.patch_embd_weight.is_empty() {
            return Err(ArchError::MissingTensor {
                name: "v.patch_embd.weight".to_string(),
            });
        }

        let side = image_size / patch_size;
        let has_cls = self.has_class_token();
        let num_positions = num_patches + usize::from(has_cls);

        // ── Step 1: CLS (row 0) + patch projection ───────────────────────────
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(num_positions);
        if has_cls {
            if self.class_embd.len() < hidden {
                return Err(ArchError::InvalidShape {
                    name: "v.class_embd".to_string(),
                    expected: vec![hidden],
                    got: vec![self.class_embd.len()],
                });
            }
            embeddings.push(self.class_embd[..hidden].to_vec());
        }

        let mut patch = vec![0.0f32; patch_flat];
        for row in 0..side {
            for col in 0..side {
                for c in 0..3usize {
                    for pr in 0..patch_size {
                        let pixel_row = row * patch_size + pr;
                        for pc in 0..patch_size {
                            let pixel_col = col * patch_size + pc;
                            let src =
                                c * image_size * image_size + pixel_row * image_size + pixel_col;
                            let dst = pr * patch_size * 3 + pc * 3 + c;
                            patch[dst] = pixels[src];
                        }
                    }
                }
                embeddings.push(linear_proj(
                    &patch,
                    &self.patch_embd_weight,
                    &self.patch_embd_bias,
                    hidden,
                    patch_flat,
                )?);
            }
        }

        // ── Step 2: position embeddings (row i ↔ position i) ─────────────────
        if !self.position_embd.is_empty() {
            let need = num_positions * hidden;
            if self.position_embd.len() < need {
                return Err(ArchError::InvalidShape {
                    name: "v.position_embd.weight".to_string(),
                    expected: vec![num_positions, hidden],
                    got: vec![self.position_embd.len()],
                });
            }
            for (pos, emb) in embeddings.iter_mut().enumerate() {
                let start = pos * hidden;
                for (e, &p) in emb
                    .iter_mut()
                    .zip(self.position_embd[start..start + hidden].iter())
                {
                    *e += p;
                }
            }
        }

        // ── Step 3: optional pre-LN ──────────────────────────────────────────
        if !self.pre_ln_weight.is_empty() {
            for emb in embeddings.iter_mut() {
                *emb = layer_norm(emb, &self.pre_ln_weight, &self.pre_ln_bias, self.params.eps);
            }
        }

        // ── Step 4: transformer blocks (only `executed_layers` of them) ──────
        let run = self.params.executed_layers().min(self.layers.len());
        let head_dim = hidden / self.params.num_heads;
        for layer_idx in 0..run {
            let layer = &self.layers[layer_idx];

            let ln1_out: Vec<Vec<f32>> = embeddings
                .iter()
                .map(|emb| layer_norm(emb, &layer.ln1_weight, &layer.ln1_bias, self.params.eps))
                .collect();

            let attn_out =
                self.mhsa(&ln1_out, layer, head_dim)
                    .map_err(|e| ArchError::ForwardPassError {
                        layer: layer_idx,
                        message: format!("clip block {layer_idx} attention: {e}"),
                    })?;

            for (emb, attn) in embeddings.iter_mut().zip(attn_out.iter()) {
                for (e, &a) in emb.iter_mut().zip(attn.iter()) {
                    *e += a;
                }
            }

            for emb in embeddings.iter_mut() {
                let normed = layer_norm(emb, &layer.ln2_weight, &layer.ln2_bias, self.params.eps);
                let ffn = self
                    .ffn(&normed, layer)
                    .map_err(|e| ArchError::ForwardPassError {
                        layer: layer_idx,
                        message: format!("clip block {layer_idx} ffn: {e}"),
                    })?;
                for (e, &f) in emb.iter_mut().zip(ffn.iter()) {
                    *e += f;
                }
            }
        }

        // ── Step 5: post-LN (applied even when the last block was skipped) ───
        if !self.post_ln_weight.is_empty() {
            for emb in embeddings.iter_mut() {
                *emb = layer_norm(
                    emb,
                    &self.post_ln_weight,
                    &self.post_ln_bias,
                    self.params.eps,
                );
            }
        }

        // ── Step 6: drop the CLS row ─────────────────────────────────────────
        let mut result = Vec::with_capacity(num_patches * hidden);
        for tok in embeddings.iter().skip(usize::from(has_cls)) {
            result.extend_from_slice(tok);
        }
        Ok(result)
    }

    /// FFN: `fc2(act(fc1(x)))`.
    fn ffn(&self, x: &[f32], layer: &ClipEncoderLayer) -> ArchResult<Vec<f32>> {
        let hidden = self.params.hidden_size;
        if layer.fc1_weight.is_empty() {
            return Err(ArchError::MissingTensor {
                name: "v.blk.*.ffn_up.weight".to_string(),
            });
        }
        let ffn_dim = layer.fc1_weight.len() / hidden;
        if ffn_dim == 0 {
            return Err(ArchError::InvalidShape {
                name: "v.blk.*.ffn_up.weight".to_string(),
                expected: vec![hidden],
                got: vec![layer.fc1_weight.len()],
            });
        }
        let mut h = linear_proj(x, &layer.fc1_weight, &layer.fc1_bias, ffn_dim, hidden)?;
        for v in h.iter_mut() {
            *v = self.params.ffn_op.apply(*v);
        }
        linear_proj(&h, &layer.fc2_weight, &layer.fc2_bias, hidden, ffn_dim)
    }

    /// Scaled dot-product multi-head self-attention.
    fn mhsa(
        &self,
        x: &[Vec<f32>],
        layer: &ClipEncoderLayer,
        head_dim: usize,
    ) -> ArchResult<Vec<Vec<f32>>> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Ok(Vec::new());
        }
        let hidden = self.params.hidden_size;

        let mut q_all = Vec::with_capacity(seq_len);
        let mut k_all = Vec::with_capacity(seq_len);
        let mut v_all = Vec::with_capacity(seq_len);
        for tok in x {
            q_all.push(linear_proj(
                tok,
                &layer.q_weight,
                &layer.q_bias,
                hidden,
                hidden,
            )?);
            k_all.push(linear_proj(
                tok,
                &layer.k_weight,
                &layer.k_bias,
                hidden,
                hidden,
            )?);
            v_all.push(linear_proj(
                tok,
                &layer.v_weight,
                &layer.v_bias,
                hidden,
                hidden,
            )?);
        }

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut concat_heads: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; seq_len];
        let mut scores = vec![0.0f32; seq_len];

        for h in 0..self.params.num_heads {
            let start = h * head_dim;
            for i in 0..seq_len {
                let qi = &q_all[i][start..start + head_dim];
                for (j, score) in scores.iter_mut().enumerate() {
                    let kj = &k_all[j][start..start + head_dim];
                    *score = qi.iter().zip(kj.iter()).map(|(&q, &k)| q * k).sum::<f32>() * scale;
                }
                softmax_inplace(&mut scores);
                let out = &mut concat_heads[i][start..start + head_dim];
                for (j, &w) in scores.iter().enumerate() {
                    let vj = &v_all[j][start..start + head_dim];
                    for (o, &vv) in out.iter_mut().zip(vj.iter()) {
                        *o += w * vv;
                    }
                }
            }
        }

        concat_heads
            .iter()
            .map(|tok| linear_proj(tok, &layer.out_weight, &layer.out_bias, hidden, hidden))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Shared numeric helpers
// ---------------------------------------------------------------------------

/// LayerNorm: `y = (x - mean) / sqrt(var + eps) * w + b`.
pub(crate) fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0f32 / (var + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            (xi - mean) * inv_std * w.get(i).copied().unwrap_or(1.0)
                + b.get(i).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Matrix-vector multiply + optional bias, row-major `[out_dim, in_dim]`.
///
/// # Errors
///
/// [`ArchError::InvalidShape`] when `weight` is shorter than `out_dim * in_dim`
/// or `x` is shorter than `in_dim`.  The previous implementation left the row
/// at `0.0` when the weight buffer was short, so a truncated or misnamed
/// tensor produced a silently zeroed projection.
pub(crate) fn linear_proj(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> ArchResult<Vec<f32>> {
    let need = out_dim
        .checked_mul(in_dim)
        .ok_or_else(|| ArchError::InvalidConfig {
            detail: "clip linear: out_dim * in_dim overflows".to_string(),
        })?;
    if weight.len() < need {
        return Err(ArchError::InvalidShape {
            name: "clip linear weight".to_string(),
            expected: vec![out_dim, in_dim],
            got: vec![weight.len()],
        });
    }
    if x.len() < in_dim {
        return Err(ArchError::InvalidShape {
            name: "clip linear input".to_string(),
            expected: vec![in_dim],
            got: vec![x.len()],
        });
    }
    let mut y = vec![0.0f32; out_dim];
    for (i, yi) in y.iter_mut().enumerate() {
        let row = &weight[i * in_dim..(i + 1) * in_dim];
        *yi = row
            .iter()
            .zip(x[..in_dim].iter())
            .map(|(&w, &xv)| w * xv)
            .sum::<f32>()
            + bias.get(i).copied().unwrap_or(0.0);
    }
    Ok(y)
}

/// Numerically stable in-place softmax.
pub(crate) fn softmax_inplace(x: &mut [f32]) {
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

    fn tiny_params(hidden: usize, heads: usize, layers: usize) -> ClipVisionParams {
        ClipVisionParams {
            hidden_size: hidden,
            num_heads: heads,
            num_layers: layers,
            patch_size: 14,
            image_size: 28,
            eps: 1e-5,
            ffn_op: ClipFfnOp::QuickGelu,
            feature_layer: None,
        }
    }

    fn tiny_encoder(params: ClipVisionParams, class_embd: Vec<f32>) -> ClipEncoder {
        let hidden = params.hidden_size;
        let patch_flat = params.patch_size * params.patch_size * 3;
        let num_positions = params.num_patches() + usize::from(!class_embd.is_empty());
        ClipEncoder {
            class_embd,
            patch_embd_weight: vec![0.01f32; hidden * patch_flat],
            patch_embd_bias: vec![0.0f32; hidden],
            position_embd: vec![0.01f32; num_positions * hidden],
            pre_ln_weight: vec![],
            pre_ln_bias: vec![],
            post_ln_weight: vec![1.0f32; hidden],
            post_ln_bias: vec![0.0f32; hidden],
            layers: vec![],
            params,
        }
    }

    /// Deterministic, **non-uniform** weights.
    ///
    /// Uniform weights make the tower degenerate: LayerNorm maps every constant
    /// row to zero, so a constant class token would be indistinguishable from a
    /// zero one no matter how the attention is wired.
    fn varied(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 7 + seed * 13) % 23) as f32 / 23.0 - 0.5) * 0.4)
            .collect()
    }

    fn tiny_layer(hidden: usize, seed: usize) -> ClipEncoderLayer {
        let ffn = hidden * 4;
        ClipEncoderLayer {
            ln1_weight: vec![1.0f32; hidden],
            ln1_bias: vec![0.0f32; hidden],
            ln2_weight: vec![1.0f32; hidden],
            ln2_bias: vec![0.0f32; hidden],
            q_weight: varied(hidden * hidden, seed + 1),
            k_weight: varied(hidden * hidden, seed + 2),
            v_weight: varied(hidden * hidden, seed + 3),
            out_weight: varied(hidden * hidden, seed + 4),
            q_bias: vec![0.0f32; hidden],
            k_bias: vec![0.0f32; hidden],
            v_bias: vec![0.0f32; hidden],
            out_bias: vec![0.0f32; hidden],
            fc1_weight: varied(ffn * hidden, seed + 5),
            fc1_bias: vec![0.0f32; ffn],
            fc2_weight: varied(hidden * ffn, seed + 6),
            fc2_bias: vec![0.0f32; hidden],
        }
    }

    // ── QuickGELU (V5) ──────────────────────────────────────────────────────

    /// QuickGELU is a distinct function from the tanh GELU approximation.
    ///
    /// The old tower used tanh-GELU throughout; this asserts the value the
    /// reference actually computes.
    #[test]
    fn quick_gelu_matches_x_sigmoid_1702x() {
        for &x in &[-3.0f32, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0] {
            let want = x * (1.0 / (1.0 + (-1.702f32 * x).exp()));
            assert!(
                (quick_gelu(x) - want).abs() < 1e-6,
                "quick_gelu({x}) = {}, expected {want}",
                quick_gelu(x)
            );
        }
        assert_eq!(quick_gelu(0.0), 0.0);
    }

    /// QuickGELU and tanh-GELU disagree materially — swapping them is not free.
    #[test]
    fn quick_gelu_differs_from_tanh_gelu() {
        let x = 1.5f32;
        let tanh_gelu = crate::common::gelu::gelu(x);
        assert!(
            (quick_gelu(x) - tanh_gelu).abs() > 1e-3,
            "quick_gelu and tanh gelu must differ at x = {x}: {} vs {tanh_gelu}",
            quick_gelu(x)
        );
    }

    // ── Class token (V5) ────────────────────────────────────────────────────

    /// A learned CLS token changes every patch output.
    ///
    /// The previous tower pushed `vec![0.0; hidden]` for the class token and
    /// never loaded `v.class_embd`.  This asserts that the loaded value
    /// actually participates: with attention active, a non-zero CLS row must
    /// move the patch features.
    #[test]
    fn learned_class_token_changes_patch_features() {
        let hidden = 8usize;
        // `num_layers: 2` so `executed_layers()` is 1 — with a single declared
        // block the LLaVA `n_layer - 1` rule would run none at all and the CLS
        // token could not reach the patches through attention.
        let params = tiny_params(hidden, 2, 2);
        assert_eq!(params.executed_layers(), 1);

        let mut zero_cls = tiny_encoder(params.clone(), vec![0.0f32; hidden]);
        zero_cls.layers = vec![tiny_layer(hidden, 0), tiny_layer(hidden, 9)];
        // A real `v.class_embd` is not a constant vector either.
        let learned: Vec<f32> = (0..hidden).map(|i| 0.3 + i as f32 * 0.11).collect();
        let mut real_cls = tiny_encoder(params, learned);
        real_cls.layers = vec![tiny_layer(hidden, 0), tiny_layer(hidden, 9)];

        let pixels: Vec<f32> = (0..3 * 28 * 28).map(|i| (i as f32) * 0.001).collect();
        let a = zero_cls.encode(&pixels).expect("zero-cls encode");
        let b = real_cls.encode(&pixels).expect("real-cls encode");

        assert_eq!(a.len(), b.len());
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "a learned class token must change the patch features (max diff {max_diff})"
        );
    }

    /// With no `v.class_embd` there is no CLS row at all — the reference does
    /// not fabricate one (`n_pos = n_patches + (class_embedding ? 1 : 0)`).
    #[test]
    fn absent_class_token_means_no_cls_row() {
        let hidden = 8usize;
        let enc = tiny_encoder(tiny_params(hidden, 2, 1), vec![]);
        assert!(!enc.has_class_token());
        let pixels = vec![0.0f32; 3 * 28 * 28];
        let out = enc.encode(&pixels).expect("encode");
        assert_eq!(out.len(), enc.num_patches() * hidden);
    }

    // ── Geometry from metadata (V5) ─────────────────────────────────────────

    /// `executed_layers` implements `max_feature_layer = n_layer - 1`.
    #[test]
    fn executed_layers_defaults_to_n_layer_minus_one() {
        let mut p = ClipVisionParams::default();
        assert_eq!(p.num_layers, 24);
        assert_eq!(p.executed_layers(), 23, "LLaVA runs n_layer - 1 blocks");

        p.feature_layer = Some(20);
        assert_eq!(p.executed_layers(), 20);

        p.num_layers = 1;
        p.feature_layer = None;
        assert_eq!(p.executed_layers(), 0);
    }

    /// `post_ln` is applied **even when the last block was skipped**.
    ///
    /// The task brief claimed the reference does *not* apply `post_ln` when a
    /// feature layer other than the last is selected.  `models/llava.cpp:128-131`
    /// contradicts that: the norm is applied unconditionally after the loop,
    /// which runs only `max_feature_layer = n_layer - 1` blocks
    /// (`llava.cpp:11-30`).  This pins the behaviour we kept.
    #[test]
    fn post_ln_runs_even_when_the_final_block_is_skipped() {
        let hidden = 8usize;
        let params = tiny_params(hidden, 2, 2);
        assert!(
            params.executed_layers() < params.num_layers,
            "the last block must be skipped for this test to mean anything"
        );

        let cls: Vec<f32> = (0..hidden).map(|i| 0.2 + i as f32 * 0.09).collect();
        let mut with_post = tiny_encoder(params.clone(), cls.clone());
        with_post.layers = vec![tiny_layer(hidden, 0), tiny_layer(hidden, 9)];
        // A non-identity post-LN scale so its effect is visible.
        with_post.post_ln_weight = (0..hidden).map(|i| 1.0 + i as f32 * 0.25).collect();
        with_post.post_ln_bias = vec![0.5f32; hidden];

        let mut without_post = tiny_encoder(params, cls);
        without_post.layers = vec![tiny_layer(hidden, 0), tiny_layer(hidden, 9)];
        without_post.post_ln_weight = vec![];
        without_post.post_ln_bias = vec![];

        let pixels: Vec<f32> = (0..3 * 28 * 28).map(|i| (i as f32) * 0.0007).collect();
        let a = with_post.encode(&pixels).expect("with post_ln");
        let b = without_post.encode(&pixels).expect("without post_ln");

        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "post_ln must still affect the output after a skipped final block (max diff {max_diff})"
        );
    }

    /// Geometry is no longer hard-coded: a non-ViT-L tower is honoured.
    #[test]
    fn non_default_geometry_is_honoured() {
        let params = ClipVisionParams {
            hidden_size: 12,
            num_heads: 3,
            num_layers: 2,
            patch_size: 7,
            image_size: 21,
            eps: 1e-6,
            ffn_op: ClipFfnOp::Gelu,
            feature_layer: None,
        };
        assert_eq!(params.num_patches(), 9, "(21/7)^2");
        let enc = tiny_encoder(params, vec![0.1f32; 12]);
        let pixels = vec![0.25f32; 3 * 21 * 21];
        let out = enc.encode(&pixels).expect("encode");
        assert_eq!(out.len(), 9 * 12);
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        let p = ClipVisionParams {
            num_heads: 7, // does not divide 1024
            ..ClipVisionParams::default()
        };
        assert!(p.validate().is_err());

        let p = ClipVisionParams {
            image_size: 300, // not a whole number of 14-px patches
            ..ClipVisionParams::default()
        };
        assert!(p.validate().is_err());

        let p = ClipVisionParams {
            patch_size: 0,
            ..ClipVisionParams::default()
        };
        assert!(p.validate().is_err());
    }

    // ── No silent zero fallbacks (V5) ───────────────────────────────────────

    /// A missing patch-embedding weight must be reported, not zero-filled.
    #[test]
    fn missing_patch_embd_errors() {
        let mut enc = tiny_encoder(tiny_params(8, 2, 1), vec![0.0f32; 8]);
        enc.patch_embd_weight = vec![];
        let pixels = vec![0.0f32; 3 * 28 * 28];
        assert!(
            matches!(enc.encode(&pixels), Err(ArchError::MissingTensor { .. })),
            "an absent v.patch_embd.weight must be MissingTensor"
        );
    }

    /// A truncated weight buffer must be reported, not silently zeroed.
    #[test]
    fn short_weight_buffer_errors() {
        let mut enc = tiny_encoder(tiny_params(8, 2, 1), vec![0.0f32; 8]);
        enc.patch_embd_weight.truncate(10);
        let pixels = vec![0.0f32; 3 * 28 * 28];
        assert!(
            matches!(enc.encode(&pixels), Err(ArchError::InvalidShape { .. })),
            "a truncated v.patch_embd.weight must be InvalidShape"
        );

        assert!(linear_proj(&[1.0, 1.0], &[1.0; 3], &[], 2, 2).is_err());
        assert!(linear_proj(&[1.0], &[1.0; 4], &[], 2, 2).is_err());
    }

    /// A short position-embedding table must be reported.
    #[test]
    fn short_position_embd_errors() {
        let mut enc = tiny_encoder(tiny_params(8, 2, 1), vec![0.0f32; 8]);
        enc.position_embd.truncate(8);
        let pixels = vec![0.0f32; 3 * 28 * 28];
        assert!(matches!(
            enc.encode(&pixels),
            Err(ArchError::InvalidShape { .. })
        ));
    }

    // ── Shape / error basics ────────────────────────────────────────────────

    #[test]
    fn wrong_pixel_count_errors() {
        let enc = tiny_encoder(tiny_params(8, 2, 1), vec![0.0f32; 8]);
        assert!(enc.encode(&[0.0f32; 100]).is_err());
    }

    #[test]
    fn vit_l_14_336_has_576_patches() {
        assert_eq!(ClipVisionParams::default().num_patches(), 576);
    }

    #[test]
    fn layer_norm_centres_and_scales() {
        let input = vec![1.0f32, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let out = layer_norm(&input, &[1.0f32; 8], &[0.0f32; 8], 1e-5);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-4, "mean {mean}");
        let var: f32 = out.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / out.len() as f32;
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
    }
}

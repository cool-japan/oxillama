//! Qwen2-VL native ViT vision encoder.
//!
//! This mirrors `tools/mtmd/models/qwen2vl.cpp` (`clip_graph_qwen2vl::build`)
//! from llama.cpp, which is the normative description of the tower:
//!
//! * **No CLS token and no patch bias** — the reference opens with
//!   `GGML_ASSERT(model.patch_bias == nullptr)` and
//!   `GGML_ASSERT(model.class_embedding == nullptr)`.
//! * **LayerNorm, not RMSNorm.** `norm_t` is `NORM_TYPE_NORMAL` for Qwen2-VL and
//!   only `NORM_TYPE_RMS` for Qwen2.5-VL.  Both `ln1`/`ln2` carry a bias.
//! * **Biased Q/K/V/O projections** — `ggml_add(ggml_mul_mat(layer.q_w, cur), layer.q_b)`.
//! * **2-D M-RoPE inside the ViT**, applied through `ggml_rope_multi` with
//!   `GGML_ROPE_TYPE_VISION`; see [`apply_vision_rope`] for the exact layout.
//! * **2×2 spatial-merge token order** — tokens are emitted in 2×2 blocks so the
//!   downstream merger is a plain "group four consecutive tokens" reshape.
//!
//! ## Tensor naming convention (GGUF)
//!
//! Every vision tower in llama.cpp — Qwen2-VL included — is loaded by the *same*
//! loop in `clip.cpp` (around line 1400), so the per-block norms are
//! `TN_LN_1` / `TN_LN_2`:
//!
//! ```text
//! v.patch_embd.weight      v.patch_embd.weight.1   (second conv, summed)
//! v.blk.{i}.ln1.weight     v.blk.{i}.ln1.bias
//! v.blk.{i}.ln2.weight     v.blk.{i}.ln2.bias
//! v.blk.{i}.attn_q.weight  v.blk.{i}.attn_q.bias     (likewise k / v / out)
//! v.blk.{i}.ffn_up.weight  v.blk.{i}.ffn_up.bias
//! v.blk.{i}.ffn_down.weight v.blk.{i}.ffn_down.bias
//! v.post_ln.weight         v.post_ln.bias
//! ```
//!
//! There is **no** `v.blk.{i}.attn_norm.weight` in the CLIP namespace — that
//! name was invented by an earlier revision of this file and matched nothing in
//! any real checkpoint.

use crate::error::{ArchError, ArchResult};

/// Normalisation flavour used by the ViT blocks.
///
/// `clip_graph_qwen2vl::build` selects `NORM_TYPE_RMS` only for
/// `PROJECTOR_TYPE_QWEN25VL`; plain Qwen2-VL uses `NORM_TYPE_NORMAL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisionNorm {
    /// Standard LayerNorm with weight and bias (Qwen2-VL).
    #[default]
    Layer,
    /// RMSNorm with weight only (Qwen2.5-VL).
    Rms,
}

/// Feed-forward activation, chosen by `clip.use_gelu` / `clip.use_silu`.
///
/// `clip.cpp` defaults to **QuickGELU** when neither key is present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisionFfnOp {
    /// `x * sigmoid(1.702 x)` — the OpenAI CLIP activation, and clip.cpp's default.
    #[default]
    GeluQuick,
    /// tanh-approximation GELU (`clip.use_gelu = true`).
    Gelu,
    /// SiLU / swish (`clip.use_silu = true`).
    Silu,
}

impl VisionFfnOp {
    /// Apply the activation element-wise.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::GeluQuick => crate::llava::clip::quick_gelu(x),
            Self::Gelu => crate::common::gelu::gelu(x),
            Self::Silu => x / (1.0 + (-x).exp()),
        }
    }
}

/// A single Qwen2-VL ViT transformer block.
///
/// Field names track the reference tensor names exactly (`ln1`, `ln2`), not the
/// text-decoder names (`attn_norm`, `ffn_norm`) an earlier revision used.
#[derive(Debug, Clone, Default)]
pub struct VisionBlock {
    /// `v.blk.{i}.ln1.weight` — pre-attention norm scale.
    pub ln1_weight: Vec<f32>,
    /// `v.blk.{i}.ln1.bias` — empty for RMSNorm towers.
    pub ln1_bias: Vec<f32>,
    /// `v.blk.{i}.ln2.weight` — pre-FFN norm scale.
    pub ln2_weight: Vec<f32>,
    /// `v.blk.{i}.ln2.bias` — empty for RMSNorm towers.
    pub ln2_bias: Vec<f32>,
    /// `v.blk.{i}.attn_q.weight`, row-major `[hidden, hidden]`.
    pub attn_q_weight: Vec<f32>,
    /// `v.blk.{i}.attn_q.bias`.
    pub attn_q_bias: Vec<f32>,
    /// `v.blk.{i}.attn_k.weight`.
    pub attn_k_weight: Vec<f32>,
    /// `v.blk.{i}.attn_k.bias`.
    pub attn_k_bias: Vec<f32>,
    /// `v.blk.{i}.attn_v.weight`.
    pub attn_v_weight: Vec<f32>,
    /// `v.blk.{i}.attn_v.bias`.
    pub attn_v_bias: Vec<f32>,
    /// `v.blk.{i}.attn_out.weight`.
    pub attn_out_weight: Vec<f32>,
    /// `v.blk.{i}.attn_out.bias`.
    pub attn_out_bias: Vec<f32>,
    /// `v.blk.{i}.ffn_up.weight`, row-major `[ffn_dim, hidden]`.
    pub ffn_up_weight: Vec<f32>,
    /// `v.blk.{i}.ffn_up.bias`.
    pub ffn_up_bias: Vec<f32>,
    /// `v.blk.{i}.ffn_down.weight`, row-major `[hidden, ffn_dim]`.
    pub ffn_down_weight: Vec<f32>,
    /// `v.blk.{i}.ffn_down.bias`.
    pub ffn_down_bias: Vec<f32>,
}

/// Qwen2-VL native ViT encoder.
///
/// Accepts pixel values at dynamic resolution and returns one feature vector
/// per patch, ordered in 2×2 spatial-merge blocks (see [`Self::token_grid_pos`]).
#[derive(Debug, Clone)]
pub struct Qwen2VlVisionEncoder {
    /// Number of pixels on each side of a square patch (14 for Qwen2-VL).
    pub patch_size: usize,
    /// Window size in patches for local self-attention (0 / 1 = full attention).
    pub window_size: usize,
    /// Hidden size of ViT layers (1280 for Qwen2-VL).
    pub hidden_size: usize,
    /// Number of attention heads (16 for Qwen2-VL → `head_dim = 80`).
    pub num_heads: usize,
    /// Normalisation flavour.
    pub norm: VisionNorm,
    /// Feed-forward activation.
    pub ffn_op: VisionFfnOp,
    /// Norm epsilon (`clip.vision.attention.layer_norm_epsilon`).
    pub norm_eps: f32,
    /// RoPE base for the in-tower 2-D M-RoPE (10000 in the reference).
    pub rope_base: f32,
    /// ViT transformer blocks.
    pub layers: Vec<VisionBlock>,
    /// Linear patch projection weight: `[hidden_size, patch_size² × 3]`.
    ///
    /// When the checkpoint carries `v.patch_embd.weight.1` the loader sums it
    /// into this buffer, exactly as the reference sums the two `ggml_conv_2d`
    /// results.
    pub patch_embd_weight: Vec<f32>,
    /// Optional post-normalisation weight (`v.post_ln.weight`).
    pub post_ln_weight: Vec<f32>,
    /// Optional post-normalisation bias (`v.post_ln.bias`).
    pub post_ln_bias: Vec<f32>,
}

impl Qwen2VlVisionEncoder {
    /// Per-head dimension.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] when `num_heads` is zero or does not divide
    /// `hidden_size`.
    pub fn head_dim(&self) -> ArchResult<usize> {
        if self.num_heads == 0 || !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "Qwen2VlVisionEncoder: hidden_size ({}) must be a non-zero multiple of \
                     num_heads ({})",
                    self.hidden_size, self.num_heads
                ),
            });
        }
        Ok(self.hidden_size / self.num_heads)
    }

    /// Compute patch grid dimensions for an image.
    ///
    /// Returns `(patches_h, patches_w)`.  The reference asserts
    /// `img.nx % (patch_size * 2) == 0`, so both counts are rounded **down** to
    /// an even number here rather than rounded up with `div_ceil` — a partially
    /// covered patch row has no position in the 2×2 merge grid.
    pub fn patch_grid(&self, height: usize, width: usize) -> (usize, usize) {
        if self.patch_size == 0 {
            return (0, 0);
        }
        let ph = (height / self.patch_size) & !1usize;
        let pw = (width / self.patch_size) & !1usize;
        (ph, pw)
    }

    /// Grid `(row, col)` of the `token_index`-th output token.
    ///
    /// The reference emits tokens in 2×2 blocks:
    ///
    /// ```text
    /// for y in (0..ph).step_by(2) { for x in (0..pw).step_by(2) {
    ///     for dy in 0..2 { for dx in 0..2 { emit (y + dy, x + dx) } } } }
    /// ```
    ///
    /// (`clip.cpp`, `PROJECTOR_TYPE_QWEN2VL` branch of the `positions` input.)
    /// Emitting patches in this order is what makes the downstream merger a
    /// plain "group four consecutive tokens" reshape.
    ///
    /// Returns `None` when `token_index` is outside the grid.
    pub fn token_grid_pos(
        token_index: usize,
        patches_h: usize,
        patches_w: usize,
    ) -> Option<(usize, usize)> {
        if patches_h == 0 || patches_w == 0 || token_index >= patches_h * patches_w {
            return None;
        }
        let blocks_w = patches_w / 2;
        if blocks_w == 0 {
            return None;
        }
        let block = token_index / 4;
        let within = token_index % 4;
        let block_row = block / blocks_w;
        let block_col = block % blocks_w;
        Some((2 * block_row + within / 2, 2 * block_col + within % 2))
    }

    /// Run the vision encoder forward pass.
    ///
    /// # Arguments
    /// * `pixel_values` — Flat `[3 × height × width]` f32, channels-first.
    /// * `height` / `width` — image dimensions in pixels.
    ///
    /// # Returns
    /// Flat `[num_patches × hidden_size]`, tokens in 2×2 spatial-merge order.
    ///
    /// # Errors
    /// * [`ArchError::InvalidConfig`] — zero dimension, or a patch grid that is
    ///   not at least 2×2 (the 2×2 merge has nothing to group otherwise).
    /// * [`ArchError::InvalidShape`] — `pixel_values.len() != 3 × height × width`,
    ///   or a weight buffer that is too short for its declared shape.
    pub fn forward(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> ArchResult<Vec<f32>> {
        let head_dim = self.head_dim()?;
        if self.patch_size == 0 || self.hidden_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "Qwen2VlVisionEncoder: patch_size and hidden_size must be > 0".to_string(),
            });
        }
        if height == 0 || width == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "Qwen2VlVisionEncoder: image height and width must be > 0".to_string(),
            });
        }

        let expected = 3usize
            .checked_mul(height)
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: "Qwen2VlVisionEncoder: image dimensions overflow".to_string(),
            })?;
        if pixel_values.len() != expected {
            return Err(ArchError::InvalidShape {
                name: "pixel_values".to_string(),
                expected: vec![expected],
                got: vec![pixel_values.len()],
            });
        }

        let (patches_h, patches_w) = self.patch_grid(height, width);
        if patches_h < 2 || patches_w < 2 {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "Qwen2VlVisionEncoder: image {width}×{height} yields a {patches_w}×{patches_h} \
                     patch grid; Qwen2-VL requires both dimensions to be at least \
                     2 × patch_size ({})",
                    2 * self.patch_size
                ),
            });
        }
        let num_patches = patches_h * patches_w;
        let patch_flat = self.patch_size * self.patch_size * 3;

        let need = self.hidden_size * patch_flat;
        if self.patch_embd_weight.len() < need {
            return Err(ArchError::InvalidShape {
                name: "v.patch_embd.weight".to_string(),
                expected: vec![need],
                got: vec![self.patch_embd_weight.len()],
            });
        }

        // ── Step 1: patch extraction + linear projection, in 2×2 block order ──
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(num_patches);
        let mut patch = vec![0.0f32; patch_flat];
        for token_index in 0..num_patches {
            let (prow, pcol) =
                Self::token_grid_pos(token_index, patches_h, patches_w).ok_or_else(|| {
                    ArchError::InvalidConfig {
                        detail: format!(
                            "Qwen2VlVisionEncoder: token {token_index} outside patch grid"
                        ),
                    }
                })?;
            patch.fill(0.0);
            for c in 0..3usize {
                for py in 0..self.patch_size {
                    let img_y = prow * self.patch_size + py;
                    if img_y >= height {
                        continue;
                    }
                    for px in 0..self.patch_size {
                        let img_x = pcol * self.patch_size + px;
                        if img_x >= width {
                            continue;
                        }
                        let src = c * height * width + img_y * width + img_x;
                        let dst = (py * self.patch_size + px) * 3 + c;
                        patch[dst] = pixel_values[src];
                    }
                }
            }
            embeddings.push(linear_proj(
                &patch,
                &self.patch_embd_weight,
                &[],
                self.hidden_size,
                patch_flat,
            )?);
        }

        // ── Step 2: transformer blocks ───────────────────────────────────────
        for (layer_idx, block) in self.layers.iter().enumerate() {
            // LN1 → attention → residual
            let normed: Vec<Vec<f32>> = embeddings
                .iter()
                .map(|emb| self.norm_vec(emb, &block.ln1_weight, &block.ln1_bias))
                .collect();

            let attn_out = self
                .attention(&normed, block, head_dim, patches_h, patches_w)
                .map_err(|e| ArchError::ForwardPassError {
                    layer: layer_idx,
                    message: format!("vision block {layer_idx} attention: {e}"),
                })?;

            for (emb, attn) in embeddings.iter_mut().zip(attn_out.iter()) {
                for (e, &a) in emb.iter_mut().zip(attn.iter()) {
                    *e += a;
                }
            }

            // LN2 → FFN → residual
            let normed_ffn: Vec<Vec<f32>> = embeddings
                .iter()
                .map(|emb| self.norm_vec(emb, &block.ln2_weight, &block.ln2_bias))
                .collect();

            for (emb, tok) in embeddings.iter_mut().zip(normed_ffn.iter()) {
                let ffn =
                    self.feed_forward(tok, block)
                        .map_err(|e| ArchError::ForwardPassError {
                            layer: layer_idx,
                            message: format!("vision block {layer_idx} ffn: {e}"),
                        })?;
                for (e, &f) in emb.iter_mut().zip(ffn.iter()) {
                    *e += f;
                }
            }
        }

        // ── Step 3: optional post-normalisation ──────────────────────────────
        if !self.post_ln_weight.is_empty() {
            for emb in embeddings.iter_mut() {
                *emb = self.norm_vec(emb, &self.post_ln_weight, &self.post_ln_bias);
            }
        }

        let mut out = Vec::with_capacity(num_patches * self.hidden_size);
        for emb in &embeddings {
            out.extend_from_slice(emb);
        }
        Ok(out)
    }

    /// Apply the configured normalisation.
    fn norm_vec(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        if weight.is_empty() {
            return x.to_vec();
        }
        match self.norm {
            VisionNorm::Layer => layer_norm(x, weight, bias, self.norm_eps),
            VisionNorm::Rms => rms_norm(x, weight, self.norm_eps),
        }
    }

    /// FFN: `down(act(up(x)))`, both projections biased.
    fn feed_forward(&self, x: &[f32], block: &VisionBlock) -> ArchResult<Vec<f32>> {
        if self.hidden_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "vision ffn: hidden_size must be > 0".to_string(),
            });
        }
        if block.ffn_up_weight.is_empty() {
            return Err(ArchError::MissingTensor {
                name: "v.blk.*.ffn_up.weight".to_string(),
            });
        }
        let ffn_dim = block.ffn_up_weight.len() / self.hidden_size;
        if ffn_dim == 0 {
            return Err(ArchError::InvalidShape {
                name: "v.blk.*.ffn_up.weight".to_string(),
                expected: vec![self.hidden_size],
                got: vec![block.ffn_up_weight.len()],
            });
        }
        let mut h = linear_proj(
            x,
            &block.ffn_up_weight,
            &block.ffn_up_bias,
            ffn_dim,
            self.hidden_size,
        )?;
        for v in h.iter_mut() {
            *v = self.ffn_op.apply(*v);
        }
        linear_proj(
            &h,
            &block.ffn_down_weight,
            &block.ffn_down_bias,
            self.hidden_size,
            ffn_dim,
        )
    }

    /// Multi-head self-attention with the reference 2-D M-RoPE.
    fn attention(
        &self,
        x: &[Vec<f32>],
        block: &VisionBlock,
        head_dim: usize,
        patches_h: usize,
        patches_w: usize,
    ) -> ArchResult<Vec<Vec<f32>>> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Ok(Vec::new());
        }
        let hidden = self.hidden_size;

        let mut q_all = Vec::with_capacity(seq_len);
        let mut k_all = Vec::with_capacity(seq_len);
        let mut v_all = Vec::with_capacity(seq_len);
        for tok in x {
            q_all.push(linear_proj(
                tok,
                &block.attn_q_weight,
                &block.attn_q_bias,
                hidden,
                hidden,
            )?);
            k_all.push(linear_proj(
                tok,
                &block.attn_k_weight,
                &block.attn_k_bias,
                hidden,
                hidden,
            )?);
            v_all.push(linear_proj(
                tok,
                &block.attn_v_weight,
                &block.attn_v_bias,
                hidden,
                hidden,
            )?);
        }

        for (token_index, (q, k)) in q_all.iter_mut().zip(k_all.iter_mut()).enumerate() {
            let (row, col) =
                Self::token_grid_pos(token_index, patches_h, patches_w).ok_or_else(|| {
                    ArchError::InvalidConfig {
                        detail: format!("vision rope: token {token_index} outside patch grid"),
                    }
                })?;
            for h in 0..self.num_heads {
                let start = h * head_dim;
                apply_vision_rope(&mut q[start..start + head_dim], row, col, self.rope_base)?;
                apply_vision_rope(&mut k[start..start + head_dim], row, col, self.rope_base)?;
            }
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut concat_heads: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; seq_len];
        let mut scores = vec![0.0f32; seq_len];

        for h in 0..self.num_heads {
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
            .map(|tok| {
                linear_proj(
                    tok,
                    &block.attn_out_weight,
                    &block.attn_out_bias,
                    hidden,
                    hidden,
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 2-D vision RoPE
// ---------------------------------------------------------------------------

/// Apply Qwen2-VL's in-ViT 2-D rotary embedding to one attention head, in place.
///
/// # What the reference does
///
/// `clip_graph_qwen2vl::build` calls
///
/// ```text
/// ggml_rope_multi(ctx0, Qcur, positions, nullptr,
///                 d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
///                 32768, 10000, 1, 0, 1, 32, 1);
/// int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};
/// ```
///
/// With `ne0 = d_head` and `n_dims = d_head/2`, `ggml_compute_forward_rope_flt`
/// takes the `GGML_ROPE_TYPE_VISION` branch:
///
/// ```text
/// rotate_pairs(ne0 /* = d_head */, n_dims /* = d_head/2 */, cache, src, dst);
///   → for i0 in (0..d_head).step(2): j = i0/2;
///       (x[j], x[j + d_head/2]) rotated by cache[i0], cache[i0+1]
/// ```
///
/// so the pairing is **global GPT-NeoX** across the whole head, with
/// `d_head/2` rotation pairs.  `ggml_mrope_cache_init` is called with
/// `indep_sects = is_vision = true`, `sect_dims = d_head`, and assigns
///
/// ```text
/// sector = j;                     // j < d_head/2 < sect_dims, so no wraparound
/// j <  d_head/4  → theta_t, restarted at j = 0
/// j >= d_head/4  → theta_h, restarted at j = d_head/4
/// theta_scale    = base^(-2 / n_dims) = base^(-4 / d_head)
/// ```
///
/// The `positions` buffer is filled with the patch **row** in the `t` slot and
/// the patch **column** in the `h` slot (`clip.cpp`, `PROJECTOR_TYPE_QWEN2VL`),
/// so the first quarter of the head encodes the row and the second quarter the
/// column.
///
/// # Why this replaces the previous implementation
///
/// The earlier code rotated `(x[head_start + off + i], x[head_start + off + half + i])`
/// with `off ∈ {0, half/2}` and bounds-checked against the length of the whole
/// hidden vector.  With `off = half/2` the second index reached
/// `head_start + head_dim + half/2 - 1`, i.e. **inside the next head**, and the
/// index range `[half/2, half)` was rotated twice.  Every head except the last
/// was corrupted.
///
/// # Errors
///
/// [`ArchError::InvalidShape`] when `head` is empty or has odd length — a
/// rotation pair needs `x[j]` and `x[j + len/2]` to both exist.
pub fn apply_vision_rope(head: &mut [f32], row: usize, col: usize, base: f32) -> ArchResult<()> {
    let head_dim = head.len();
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(ArchError::InvalidShape {
            name: "vision rope head".to_string(),
            expected: vec![2],
            got: vec![head_dim],
        });
    }
    let half = head_dim / 2;
    // `sections[0] = d_head/4` rotation pairs on the row axis, the rest on the
    // column axis. For an odd `half` (head_dim ≡ 2 mod 4) ggml's integer
    // division rounds down identically.
    let row_pairs = head_dim / 4;
    let inv_n_dims = if half == 0 { 0.0 } else { 2.0 / half as f32 };

    for j in 0..half {
        let (pos, freq_index) = if j < row_pairs {
            (row, j)
        } else {
            (col, j - row_pairs)
        };
        let theta = pos as f32 * base.powf(-(freq_index as f32) * inv_n_dims);
        let (sin, cos) = theta.sin_cos();
        let x0 = head[j];
        let x1 = head[j + half];
        head[j] = x0 * cos - x1 * sin;
        head[j + half] = x0 * sin + x1 * cos;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// LayerNorm: `y = (x - mean) / sqrt(var + eps) * w + b`.
fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            (xi - mean) * inv_std * w.get(i).copied().unwrap_or(1.0)
                + b.get(i).copied().unwrap_or(0.0)
        })
        .collect()
}

/// RMSNorm: `y = x / rms(x) * w` (Qwen2.5-VL towers).
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let inv_rms = 1.0 / (x.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &xi)| xi * inv_rms * w.get(i).copied().unwrap_or(1.0))
        .collect()
}

/// Matrix-vector multiply + optional bias, row-major `[out_dim, in_dim]`.
///
/// # Errors
///
/// [`ArchError::InvalidShape`] when `weight` is shorter than `out_dim * in_dim`
/// or `x` is shorter than `in_dim`.  The previous version silently produced a
/// zero row for a short weight buffer.
fn linear_proj(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> ArchResult<Vec<f32>> {
    let need = out_dim
        .checked_mul(in_dim)
        .ok_or_else(|| ArchError::InvalidConfig {
            detail: "vision linear: out_dim * in_dim overflows".to_string(),
        })?;
    if weight.len() < need {
        return Err(ArchError::InvalidShape {
            name: "vision linear weight".to_string(),
            expected: vec![out_dim, in_dim],
            got: vec![weight.len()],
        });
    }
    if x.len() < in_dim {
        return Err(ArchError::InvalidShape {
            name: "vision linear input".to_string(),
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

    fn make_encoder(
        hidden_size: usize,
        patch_size: usize,
        num_heads: usize,
    ) -> Qwen2VlVisionEncoder {
        let patch_flat = patch_size * patch_size * 3;
        Qwen2VlVisionEncoder {
            patch_size,
            window_size: 0,
            hidden_size,
            num_heads,
            norm: VisionNorm::Layer,
            ffn_op: VisionFfnOp::GeluQuick,
            norm_eps: 1e-6,
            rope_base: 10000.0,
            layers: vec![],
            patch_embd_weight: vec![0.0f32; hidden_size * patch_flat],
            post_ln_weight: vec![1.0f32; hidden_size],
            post_ln_bias: vec![0.0f32; hidden_size],
        }
    }

    #[test]
    fn vision_encoder_basic_shape() {
        let enc = make_encoder(8, 4, 2);
        let pixels = vec![0.0f32; 3 * 8 * 8];
        let out = enc.forward(&pixels, 8, 8).expect("forward should succeed");
        assert_eq!(out.len(), 4 * 8, "expected 4 patches × 8 dims");
    }

    #[test]
    fn vision_encoder_dynamic_resolution_scales_patches() {
        let enc = make_encoder(8, 4, 2);
        let small = enc
            .forward(&vec![0.0f32; 3 * 8 * 8], 8, 8)
            .expect("small forward");
        let large = enc
            .forward(&vec![0.0f32; 3 * 16 * 16], 16, 16)
            .expect("large forward");
        assert_eq!(small.len(), 4 * 8);
        assert_eq!(large.len(), 16 * 8);
    }

    #[test]
    fn vision_encoder_wrong_pixel_size_errors() {
        let enc = make_encoder(8, 4, 2);
        assert!(enc.forward(&[0.0f32; 10], 8, 8).is_err());
    }

    #[test]
    fn vision_encoder_zero_patch_size_errors() {
        let mut enc = make_encoder(8, 4, 2);
        enc.patch_size = 0;
        let pixels = vec![0.0f32; 3 * 8 * 8];
        assert!(enc.forward(&pixels, 8, 8).is_err());
    }

    /// A grid narrower than 2×2 patches cannot feed the 2×2 merger.
    #[test]
    fn vision_encoder_rejects_sub_2x2_grid() {
        let enc = make_encoder(8, 4, 2);
        let pixels = vec![0.0f32; 3 * 4 * 4];
        assert!(
            enc.forward(&pixels, 4, 4).is_err(),
            "a 1×1 patch grid must be rejected"
        );
    }

    /// Token order is 2×2 blocks, matching clip.cpp's `positions` fill loop.
    #[test]
    fn token_grid_pos_matches_reference_block_order() {
        // 4×4 patch grid → 4 blocks of 4 tokens.
        let expected = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1), // block (0,0)
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3), // block (0,1)
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1), // block (1,0)
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3), // block (1,1)
        ];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(
                Qwen2VlVisionEncoder::token_grid_pos(i, 4, 4),
                Some(*want),
                "token {i}"
            );
        }
        assert_eq!(Qwen2VlVisionEncoder::token_grid_pos(16, 4, 4), None);
    }

    // ── 2-D vision RoPE regressions (V3) ────────────────────────────────────

    /// The rotation must stay inside the head it was handed.
    ///
    /// The previous implementation indexed
    /// `head_start + half/2 + half + i`, reaching `head_start + head_dim +
    /// half/2 - 1` — inside head `h + 1`.  Rotating one head of a multi-head
    /// buffer therefore corrupted its neighbour.
    #[test]
    fn vision_rope_does_not_write_past_the_head_boundary() {
        let head_dim = 8usize;
        let num_heads = 3usize;
        let mut buf: Vec<f32> = (0..head_dim * num_heads)
            .map(|i| (i as f32 + 1.0) * 0.25)
            .collect();
        let before = buf.clone();

        apply_vision_rope(&mut buf[head_dim..2 * head_dim], 3, 5, 10000.0).expect("rope");

        for i in 0..head_dim {
            assert_eq!(buf[i], before[i], "head 0 dim {i} must be untouched");
        }
        for i in 2 * head_dim..3 * head_dim {
            assert_eq!(buf[i], before[i], "head 2 dim {i} must be untouched");
        }
        assert!(
            buf[head_dim..2 * head_dim] != before[head_dim..2 * head_dim],
            "head 1 must actually be rotated"
        );
    }

    /// Every index is rotated exactly once, so negating the angle restores the
    /// input.  The old code double-rotated `[half/2, half)`, which this catches.
    #[test]
    fn vision_rope_is_an_exact_rotation() {
        let head_dim = 16usize;
        let x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let mut rotated = x.clone();
        apply_vision_rope(&mut rotated, 2, 7, 10000.0).expect("rope");

        // Undo: an exact rotation composed with its transpose is the identity.
        let half = head_dim / 2;
        let row_pairs = head_dim / 4;
        let inv_n_dims = 2.0 / half as f32;
        let mut restored = rotated.clone();
        for j in 0..half {
            let (pos, fi) = if j < row_pairs {
                (2usize, j)
            } else {
                (7usize, j - row_pairs)
            };
            let theta = pos as f32 * 10000.0f32.powf(-(fi as f32) * inv_n_dims);
            let (sin, cos) = theta.sin_cos();
            let y0 = rotated[j];
            let y1 = rotated[j + half];
            restored[j] = y0 * cos + y1 * sin;
            restored[j + half] = -y0 * sin + y1 * cos;
        }
        for (i, (a, b)) in restored.iter().zip(x.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "dim {i}: round-trip gave {a}, expected {b} (index rotated more than once?)"
            );
        }
    }

    /// The row position drives only the first quarter of the head and the
    /// column position only the second quarter — the reference's section split.
    #[test]
    fn vision_rope_row_and_col_own_disjoint_quarters() {
        let head_dim = 16usize;
        let half = head_dim / 2;
        let row_pairs = head_dim / 4;
        let x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let mut a = x.clone();
        apply_vision_rope(&mut a, 1, 1, 10000.0).expect("rope");
        let mut b = x.clone();
        apply_vision_rope(&mut b, 5, 1, 10000.0).expect("rope");

        for j in 0..half {
            let changed = (a[j] - b[j]).abs() > 1e-6 || (a[j + half] - b[j + half]).abs() > 1e-6;
            assert_eq!(
                changed,
                j < row_pairs,
                "pair {j}: only pairs < {row_pairs} may depend on the row position"
            );
        }
    }

    /// Position (0, 0) is the identity rotation.
    #[test]
    fn vision_rope_origin_is_identity() {
        let x: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut y = x.clone();
        apply_vision_rope(&mut y, 0, 0, 10000.0).expect("rope");
        for (i, (a, b)) in y.iter().zip(x.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "dim {i}: {a} != {b}");
        }
    }

    /// A short weight buffer must be reported, not silently zero-filled.
    #[test]
    fn linear_proj_rejects_short_weight() {
        assert!(linear_proj(&[1.0, 2.0], &[1.0, 2.0], &[], 2, 2).is_err());
        assert!(linear_proj(&[1.0], &[1.0; 4], &[], 2, 2).is_err());
        assert!(linear_proj(&[1.0, 1.0], &[1.0; 4], &[], 2, 2).is_ok());
    }
}

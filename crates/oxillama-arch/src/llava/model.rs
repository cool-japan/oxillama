//! LLaVA model implementation.
//!
//! Provides the CLIP vision encoder, multi-modal projector (MLP), and
//! LLaMA language backbone needed for LLaVA-1.5 and LLaVA-NeXT inference.
//!
//! ## Tensor naming (GGUF)
//!
//! Language backbone — identical to LLaMA:
//! - `token_embd.weight`, `blk.{i}.*`, `output_norm.weight`, `output.weight`
//!
//! MM projector (2-layer MLP, FC → GeLU → FC):
//! - `mm.0.weight` `[mm_hidden_size, clip_hidden_size]`
//! - `mm.0.bias`   `[mm_hidden_size]`
//! - `mm.2.weight` `[llm_hidden_size, mm_hidden_size]`
//! - `mm.2.bias`   `[llm_hidden_size]`
//!
//! CLIP vision encoder:
//! - `v.patch_embd.weight`, `v.position_embd.weight`
//! - `v.pre_ln.weight`, `v.pre_ln.bias`
//! - `v.post_ln.weight`, `v.post_ln.bias`
//! - `v.blk.{i}.ln1.*`, `v.blk.{i}.ln2.*`
//! - `v.blk.{i}.attn_q.*`, `v.blk.{i}.attn_k.*`, `v.blk.{i}.attn_v.*`, `v.blk.{i}.attn_out.*`
//! - `v.blk.{i}.ffn_up.*`, `v.blk.{i}.ffn_down.*`

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::llama::{dequant_to_f32, load_dequant_tensor, load_llama_from_gguf, LlamaModel};
use crate::lora::LoadedLora;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::GgufModel;
use oxillama_quant::KernelDispatcher;

/// Multi-modal projector: maps CLIP features to LLM embedding space.
///
/// For LLaVA-1.5 this is a 2-layer MLP: Linear → GeLU → Linear.
/// Tensor names: `mm.0.weight`, `mm.0.bias`, `mm.2.weight`, `mm.2.bias`.
pub struct MmProjector {
    /// FC1 weight [mm_hidden_size, clip_hidden_size].
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    /// FC2 weight [llm_hidden_size, mm_hidden_size].
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
    pub clip_hidden_size: usize,
    pub mm_hidden_size: usize,
    pub llm_hidden_size: usize,
}

impl MmProjector {
    /// Project CLIP image features to LLM embedding space.
    ///
    /// # Arguments
    /// * `input` — flat `[num_patches * clip_hidden_size]`
    ///
    /// # Returns
    /// Flat `[num_patches * llm_hidden_size]`
    pub fn project(&self, input: &[f32]) -> ArchResult<Vec<f32>> {
        if self.clip_hidden_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "clip_hidden_size must be > 0".to_string(),
            });
        }
        if input.len() % self.clip_hidden_size != 0 {
            return Err(ArchError::InvalidShape {
                name: "mm_projector input".to_string(),
                expected: vec![self.clip_hidden_size],
                got: vec![input.len()],
            });
        }

        let num_patches = input.len() / self.clip_hidden_size;
        let mut out = Vec::with_capacity(num_patches * self.llm_hidden_size);

        for patch_idx in 0..num_patches {
            let patch =
                &input[patch_idx * self.clip_hidden_size..(patch_idx + 1) * self.clip_hidden_size];

            // FC1: [clip_hidden_size → mm_hidden_size]
            let mut h = vec![0.0f32; self.mm_hidden_size];
            for (i, h_val) in h.iter_mut().enumerate().take(self.mm_hidden_size) {
                let row =
                    &self.fc1_weight[i * self.clip_hidden_size..(i + 1) * self.clip_hidden_size];
                *h_val = row
                    .iter()
                    .zip(patch.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
                    + self.fc1_bias.get(i).copied().unwrap_or(0.0);
            }

            // GeLU activation (tanh approximation — numerically stable for all inputs)
            for v in h.iter_mut() {
                let x = *v;
                let inner = 0.797_884_6_f32 * (x + 0.044_715 * x * x * x);
                *v = x * 0.5 * (1.0 + inner.tanh());
            }

            // FC2: [mm_hidden_size → llm_hidden_size]
            let mut proj = vec![0.0f32; self.llm_hidden_size];
            for (i, proj_val) in proj.iter_mut().enumerate().take(self.llm_hidden_size) {
                let row = &self.fc2_weight[i * self.mm_hidden_size..(i + 1) * self.mm_hidden_size];
                *proj_val = row.iter().zip(h.iter()).map(|(w, x)| w * x).sum::<f32>()
                    + self.fc2_bias.get(i).copied().unwrap_or(0.0);
            }

            out.extend_from_slice(&proj);
        }

        Ok(out)
    }
}

/// Single CLIP transformer encoder layer.
pub struct ClipEncoderLayer {
    // Layer norms
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
    // Self-attention projections
    pub q_weight: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub v_weight: Vec<f32>,
    pub out_weight: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub out_bias: Vec<f32>,
    // Feed-forward
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
}

/// CLIP Vision Transformer encoder.
///
/// For LLaVA-1.5: CLIP ViT-L/14 — 336×336 images, 14×14 patches → 576 patches,
/// 1024-dim hidden size, 16 attention heads, 23 transformer layers.
pub struct ClipEncoder {
    /// Patch convolution embedding: [clip_hidden_size, patch_size*patch_size*3].
    pub patch_embd_weight: Vec<f32>,
    pub patch_embd_bias: Vec<f32>,
    /// Learnable position embeddings: [num_positions, clip_hidden_size].
    pub position_embd: Vec<f32>,
    /// Pre-normalization (applied before patch projection in some CLIP variants).
    pub pre_ln_weight: Vec<f32>,
    pub pre_ln_bias: Vec<f32>,
    /// Post-normalization (applied after last transformer block).
    pub post_ln_weight: Vec<f32>,
    pub post_ln_bias: Vec<f32>,
    /// Transformer encoder layers.
    pub layers: Vec<ClipEncoderLayer>,
    /// Hidden dimension of the CLIP model.
    pub hidden_size: usize,
    /// Number of self-attention heads.
    pub num_heads: usize,
    /// Patch size (pixels, square).
    pub patch_size: usize,
    /// Input image size (pixels, square).
    pub image_size: usize,
}

impl ClipEncoder {
    /// Number of visual patches (excludes CLS token).
    pub fn num_patches(&self) -> usize {
        let side = self.image_size.checked_div(self.patch_size).unwrap_or(0);
        side * side
    }

    /// Encode an image into patch feature vectors.
    ///
    /// Runs the full CLIP ViT forward pass:
    /// patch extraction → linear projection → position embeddings →
    /// optional pre-LN → N × (LN₁ + MHSA + LN₂ + FFN) → post-LN.
    ///
    /// # Arguments
    /// * `pixels` — flat `[3 * H * W]` normalized floats (channels-first, H=W=image_size).
    ///
    /// # Returns
    /// Flat `[num_patches * hidden_size]` (CLS token is dropped).
    ///
    /// # Errors
    /// * `ArchError::InvalidConfig` — if any dimension is 0.
    /// * `ArchError::InvalidShape` — if `pixels.len() != 3 * image_size * image_size`.
    pub fn encode(&self, pixels: &[f32]) -> ArchResult<Vec<f32>> {
        // --- Validate dimensions ---
        if self.hidden_size == 0 || self.patch_size == 0 || self.image_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "ClipEncoder: hidden_size, patch_size, and image_size must all be > 0"
                    .to_string(),
            });
        }

        let expected_pixels = 3 * self.image_size * self.image_size;
        if pixels.len() != expected_pixels {
            return Err(ArchError::InvalidShape {
                name: "clip pixels".to_string(),
                expected: vec![expected_pixels],
                got: vec![pixels.len()],
            });
        }

        let num_patches = self.num_patches();
        let patch_flat = self.patch_size * self.patch_size * 3;
        let num_positions = num_patches + 1; // +1 for CLS token

        // --- Step 1: Patch extraction ---
        // Input pixels: channels-first [3, H, W]
        // We extract num_patches patches of size patch_size×patch_size×3.
        // Patches are ordered row-major (top-left to bottom-right).
        let side = self.image_size / self.patch_size;
        let mut patches: Vec<Vec<f32>> = Vec::with_capacity(num_patches);

        for row in 0..side {
            for col in 0..side {
                let mut patch = vec![0.0f32; patch_flat];
                // Each channel is laid out as [H, W] in channels-first format
                for c in 0..3usize {
                    for pr in 0..self.patch_size {
                        for pc in 0..self.patch_size {
                            let pixel_row = row * self.patch_size + pr;
                            let pixel_col = col * self.patch_size + pc;
                            let src_idx = c * self.image_size * self.image_size
                                + pixel_row * self.image_size
                                + pixel_col;
                            // Destination: channels interleaved — [pr, pc, c] ordering
                            // (standard HWC within patch, which is how CLIP reads patches)
                            let dst_idx = pr * self.patch_size * 3 + pc * 3 + c;
                            patch[dst_idx] = pixels[src_idx];
                        }
                    }
                }
                patches.push(patch);
            }
        }

        // --- Step 2: Patch embedding: linear projection [patch_flat → hidden_size] ---
        // patch_embd_weight layout: [hidden_size, patch_flat]
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(num_positions);

        // CLS token: zero vector prepended at position 0
        embeddings.push(vec![0.0f32; self.hidden_size]);

        for patch in &patches {
            let emb = Self::linear_proj(
                patch,
                &self.patch_embd_weight,
                &self.patch_embd_bias,
                self.hidden_size,
                patch_flat,
            );
            embeddings.push(emb);
        }

        // --- Step 3: Add position embeddings ---
        // position_embd: [num_positions, hidden_size]
        if !self.position_embd.is_empty() {
            for (pos, emb) in embeddings.iter_mut().enumerate() {
                let start = pos * self.hidden_size;
                let end = start + self.hidden_size;
                if end <= self.position_embd.len() {
                    for (e, &p) in emb.iter_mut().zip(self.position_embd[start..end].iter()) {
                        *e += p;
                    }
                }
            }
        }

        // --- Step 4: Optional pre-normalization ---
        if !self.pre_ln_weight.is_empty() {
            for emb in embeddings.iter_mut() {
                *emb = Self::layer_norm(emb, &self.pre_ln_weight, &self.pre_ln_bias);
            }
        }

        // --- Step 5: Transformer encoder layers ---
        // x shape logically: [seq_len, hidden_size]  (seq_len = num_positions)
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if self.num_heads == 0 {
                return Err(ArchError::InvalidConfig {
                    detail: format!("ClipEncoder layer {layer_idx}: num_heads must be > 0"),
                });
            }

            let head_dim = self.hidden_size / self.num_heads;
            if head_dim == 0 {
                return Err(ArchError::InvalidConfig {
                    detail: format!(
                        "ClipEncoder layer {layer_idx}: head_dim = hidden_size / num_heads = 0"
                    ),
                });
            }

            // LN₁ on each token
            let ln1_out: Vec<Vec<f32>> = embeddings
                .iter()
                .map(|emb| {
                    if layer.ln1_weight.is_empty() {
                        emb.clone()
                    } else {
                        Self::layer_norm(emb, &layer.ln1_weight, &layer.ln1_bias)
                    }
                })
                .collect();

            // Multi-head self-attention
            let attn_out = Self::mhsa(&ln1_out, layer, self.hidden_size, self.num_heads, head_dim)?;

            // Residual connection: x = x + attn_out
            for (emb, attn) in embeddings.iter_mut().zip(attn_out.iter()) {
                for (e, &a) in emb.iter_mut().zip(attn.iter()) {
                    *e += a;
                }
            }

            // LN₂ on each token
            let ln2_out: Vec<Vec<f32>> = embeddings
                .iter()
                .map(|emb| {
                    if layer.ln2_weight.is_empty() {
                        emb.clone()
                    } else {
                        Self::layer_norm(emb, &layer.ln2_weight, &layer.ln2_bias)
                    }
                })
                .collect();

            // Feed-forward network: fc2(gelu(fc1(x)))
            let ffn_out: Vec<Vec<f32>> = ln2_out
                .iter()
                .map(|tok| Self::ffn(tok, layer, self.hidden_size))
                .collect();

            // Residual connection: x = x + ffn_out
            for (emb, ffn) in embeddings.iter_mut().zip(ffn_out.iter()) {
                for (e, &f) in emb.iter_mut().zip(ffn.iter()) {
                    *e += f;
                }
            }
        }

        // --- Step 6: Post-normalization ---
        if !self.post_ln_weight.is_empty() {
            for emb in embeddings.iter_mut() {
                *emb = Self::layer_norm(emb, &self.post_ln_weight, &self.post_ln_bias);
            }
        }

        // --- Step 7: Drop CLS token (index 0), return visual tokens 1..=num_patches ---
        let mut result = Vec::with_capacity(num_patches * self.hidden_size);
        for tok in embeddings.iter().skip(1) {
            result.extend_from_slice(tok);
        }

        Ok(result)
    }

    // ---- Private helper: LayerNorm ----

    /// Standard LayerNorm with eps=1e-5.
    ///
    /// `y[i] = (x[i] - mean) / sqrt(var + 1e-5) * w[i] + b[i]`
    ///
    /// If `b` is empty, the bias term is omitted (treated as zero).
    fn layer_norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        const EPS: f32 = 1e-5;
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }

        let mean: f32 = x.iter().sum::<f32>() / n as f32;
        let var: f32 = x
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        let inv_std = 1.0_f32 / (var + EPS).sqrt();

        x.iter()
            .enumerate()
            .map(|(i, &xi)| {
                let normalized = (xi - mean) * inv_std;
                let scale = w.get(i).copied().unwrap_or(1.0);
                let shift = b.get(i).copied().unwrap_or(0.0);
                normalized * scale + shift
            })
            .collect()
    }

    // ---- Private helper: GELU activation ----

    /// Tanh-approximation GELU.
    ///
    /// `gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
    #[inline]
    fn gelu(x: f32) -> f32 {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6_f32;
        const COEFF: f32 = 0.044_715_f32;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    // ---- Private helper: linear projection ----

    /// Matrix-vector multiply + bias: `y[i] = sum_j(w[i,j] * x[j]) + b[i]`
    ///
    /// Weight layout: row-major `[out_dim, in_dim]`.
    fn linear_proj(
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

    // ---- Private helper: feed-forward network ----

    /// FFN: `fc2(gelu(fc1(x)))`.
    ///
    /// fc1: `[hidden_size → ffn_dim]`, fc2: `[ffn_dim → hidden_size]`.
    /// Weight layout: row-major `[out_dim, in_dim]`.
    fn ffn(x: &[f32], layer: &ClipEncoderLayer, hidden_size: usize) -> Vec<f32> {
        // Infer ffn_dim from fc1_weight: rows = ffn_dim
        let ffn_dim = if hidden_size > 0 && !layer.fc1_weight.is_empty() {
            layer.fc1_weight.len() / hidden_size
        } else {
            0
        };

        if ffn_dim == 0 {
            return vec![0.0f32; hidden_size];
        }

        // FC1 + GELU
        let mut h = Self::linear_proj(x, &layer.fc1_weight, &layer.fc1_bias, ffn_dim, hidden_size);
        for v in h.iter_mut() {
            *v = Self::gelu(*v);
        }

        // FC2
        Self::linear_proj(&h, &layer.fc2_weight, &layer.fc2_bias, hidden_size, ffn_dim)
    }

    // ---- Private helper: multi-head self-attention ----

    /// Scaled dot-product multi-head self-attention.
    ///
    /// For each head `h`:
    /// - Q_h, K_h, V_h ∈ ℝ^{seq_len × head_dim}
    /// - A_h = softmax(Q_h K_h^T / sqrt(head_dim))
    /// - out_h = A_h V_h
    ///
    /// Heads are concatenated and projected by `out_weight`.
    fn mhsa(
        x: &[Vec<f32>],
        layer: &ClipEncoderLayer,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> ArchResult<Vec<Vec<f32>>> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        // Project Q, K, V for all tokens: result shape [seq_len, hidden_size]
        let q_all: Vec<Vec<f32>> = x
            .iter()
            .map(|tok| {
                Self::linear_proj(
                    tok,
                    &layer.q_weight,
                    &layer.q_bias,
                    hidden_size,
                    hidden_size,
                )
            })
            .collect();
        let k_all: Vec<Vec<f32>> = x
            .iter()
            .map(|tok| {
                Self::linear_proj(
                    tok,
                    &layer.k_weight,
                    &layer.k_bias,
                    hidden_size,
                    hidden_size,
                )
            })
            .collect();
        let v_all: Vec<Vec<f32>> = x
            .iter()
            .map(|tok| {
                Self::linear_proj(
                    tok,
                    &layer.v_weight,
                    &layer.v_bias,
                    hidden_size,
                    hidden_size,
                )
            })
            .collect();

        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // Concatenated head output, shape [seq_len, hidden_size]
        let mut concat_heads: Vec<Vec<f32>> = vec![vec![0.0f32; hidden_size]; seq_len];

        for h in 0..num_heads {
            let head_start = h * head_dim;

            // Compute attention scores for this head: [seq_len, seq_len]
            let mut attn_scores: Vec<Vec<f32>> = vec![vec![0.0f32; seq_len]; seq_len];
            for i in 0..seq_len {
                let qi = &q_all[i][head_start..head_start + head_dim];
                for j in 0..seq_len {
                    let kj = &k_all[j][head_start..head_start + head_dim];
                    attn_scores[i][j] =
                        qi.iter().zip(kj.iter()).map(|(&q, &k)| q * k).sum::<f32>() * scale;
                }
            }

            // Stable softmax over each row
            for scores_row in attn_scores.iter_mut() {
                let max_val = scores_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores_row.iter_mut() {
                    *s = (*s - max_val).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    for s in scores_row.iter_mut() {
                        *s /= sum;
                    }
                }
            }

            // Weighted sum of V for each query token, write into concat_heads
            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for j in 0..seq_len {
                        val += attn_scores[i][j] * v_all[j][head_start + d];
                    }
                    concat_heads[i][head_start + d] = val;
                }
            }
        }

        // Output projection: [hidden_size, hidden_size]
        let out: Vec<Vec<f32>> = concat_heads
            .iter()
            .map(|tok| {
                Self::linear_proj(
                    tok,
                    &layer.out_weight,
                    &layer.out_bias,
                    hidden_size,
                    hidden_size,
                )
            })
            .collect();

        Ok(out)
    }
}

/// Combined LLaVA model: CLIP vision encoder + MLP projector + LLaMA backbone.
pub struct LlavaModel {
    /// CLIP vision encoder.
    pub vision_encoder: ClipEncoder,
    /// Multi-modal projector (MLP mapping CLIP dim → LLM dim).
    pub mm_projector: MmProjector,
    /// LLaMA language backbone.
    pub language_model: LlamaModel,
    /// LLM hidden size (= language_model.config.hidden_size).
    pub llm_hidden_size: usize,
    /// CLIP hidden size (inferred from mm projector weight shapes).
    pub clip_hidden_size: usize,
}

impl LlavaModel {
    /// Load a LLaVA model from a GGUF file.
    ///
    /// Loads the LLaMA backbone (standard tensors), the MM projector
    /// (`mm.0.*` / `mm.2.*`), and the CLIP vision encoder (`v.*` prefix).
    pub fn load(gguf: &GgufModel, config: &ModelConfig) -> ArchResult<Self> {
        let language_model = load_llama_from_gguf(gguf, config)?;
        let dispatcher = KernelDispatcher::new();
        let hidden_size = config.hidden_size;

        // ---- MM Projector ----
        // mm.0.weight shape: [mm_hidden_size, clip_hidden_size]
        let mm0_weight = load_dequant_tensor(gguf, &dispatcher, "mm.0.weight")?;
        let mm0_bias = load_f32_tensor_opt(gguf, &dispatcher, "mm.0.bias").unwrap_or_default();
        let mm2_weight = load_dequant_tensor(gguf, &dispatcher, "mm.2.weight")?;
        let mm2_bias = load_f32_tensor_opt(gguf, &dispatcher, "mm.2.bias")
            .unwrap_or_else(|| vec![0.0; hidden_size]);

        // Infer shapes:
        //   mm_hidden_size = number of rows in mm2_weight = hidden_size (LLaVA-1.5)
        //   clip_hidden_size = mm0_weight_len / mm_hidden_size
        let mm_hidden_size = hidden_size;
        let clip_hidden_size = if mm_hidden_size > 0 && !mm0_weight.is_empty() {
            mm0_weight.len() / mm_hidden_size
        } else {
            1024 // CLIP ViT-L default
        };

        let mm_projector = MmProjector {
            fc1_weight: mm0_weight,
            fc1_bias: mm0_bias,
            fc2_weight: mm2_weight,
            fc2_bias: mm2_bias,
            clip_hidden_size,
            mm_hidden_size,
            llm_hidden_size: hidden_size,
        };

        // ---- CLIP Vision Encoder ----
        let patch_embd_weight =
            load_dequant_tensor(gguf, &dispatcher, "v.patch_embd.weight").unwrap_or_default();
        let patch_embd_bias =
            load_f32_tensor_opt(gguf, &dispatcher, "v.patch_embd.bias").unwrap_or_default();
        let position_embd =
            load_f32_tensor_opt(gguf, &dispatcher, "v.position_embd.weight").unwrap_or_default();
        let pre_ln_weight =
            load_f32_tensor_opt(gguf, &dispatcher, "v.pre_ln.weight").unwrap_or_default();
        let pre_ln_bias =
            load_f32_tensor_opt(gguf, &dispatcher, "v.pre_ln.bias").unwrap_or_default();
        let post_ln_weight =
            load_f32_tensor_opt(gguf, &dispatcher, "v.post_ln.weight").unwrap_or_default();
        let post_ln_bias =
            load_f32_tensor_opt(gguf, &dispatcher, "v.post_ln.bias").unwrap_or_default();

        // CLIP ViT-L/14 (LLaVA-1.5 default): 23 encoder layers
        const CLIP_NUM_LAYERS: usize = 23;
        let mut clip_layers = Vec::with_capacity(CLIP_NUM_LAYERS);
        for i in 0..CLIP_NUM_LAYERS {
            let pfx = format!("v.blk.{i}");
            let layer = ClipEncoderLayer {
                ln1_weight: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.ln1.weight"))
                    .unwrap_or_default(),
                ln1_bias: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.ln1.bias"))
                    .unwrap_or_default(),
                ln2_weight: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.ln2.weight"))
                    .unwrap_or_default(),
                ln2_bias: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.ln2.bias"))
                    .unwrap_or_default(),
                q_weight: load_dequant_tensor(gguf, &dispatcher, &format!("{pfx}.attn_q.weight"))
                    .unwrap_or_default(),
                k_weight: load_dequant_tensor(gguf, &dispatcher, &format!("{pfx}.attn_k.weight"))
                    .unwrap_or_default(),
                v_weight: load_dequant_tensor(gguf, &dispatcher, &format!("{pfx}.attn_v.weight"))
                    .unwrap_or_default(),
                out_weight: load_dequant_tensor(
                    gguf,
                    &dispatcher,
                    &format!("{pfx}.attn_out.weight"),
                )
                .unwrap_or_default(),
                q_bias: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.attn_q.bias"))
                    .unwrap_or_default(),
                k_bias: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.attn_k.bias"))
                    .unwrap_or_default(),
                v_bias: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.attn_v.bias"))
                    .unwrap_or_default(),
                out_bias: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.attn_out.bias"))
                    .unwrap_or_default(),
                fc1_weight: load_dequant_tensor(gguf, &dispatcher, &format!("{pfx}.ffn_up.weight"))
                    .unwrap_or_default(),
                fc1_bias: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.ffn_up.bias"))
                    .unwrap_or_default(),
                fc2_weight: load_dequant_tensor(
                    gguf,
                    &dispatcher,
                    &format!("{pfx}.ffn_down.weight"),
                )
                .unwrap_or_default(),
                fc2_bias: load_f32_tensor_opt(gguf, &dispatcher, &format!("{pfx}.ffn_down.bias"))
                    .unwrap_or_default(),
            };
            clip_layers.push(layer);
        }

        let vision_encoder = ClipEncoder {
            patch_embd_weight,
            patch_embd_bias,
            position_embd,
            pre_ln_weight,
            pre_ln_bias,
            post_ln_weight,
            post_ln_bias,
            layers: clip_layers,
            hidden_size: clip_hidden_size,
            num_heads: 16,   // CLIP ViT-L: 16 heads
            patch_size: 14,  // 14×14 pixel patches
            image_size: 336, // LLaVA-1.5: 336×336 input
        };

        Ok(Self {
            vision_encoder,
            mm_projector,
            language_model,
            llm_hidden_size: hidden_size,
            clip_hidden_size,
        })
    }

    /// Encode an image into visual token embeddings for the LLM.
    ///
    /// Runs the CLIP encoder then the MM projector.
    ///
    /// # Arguments
    /// * `pixels` — flat `[3 * H * W]` normalized floats (channels-first, H=W=336).
    ///
    /// # Returns
    /// Flat `[num_visual_tokens * llm_hidden_size]` — 576 × hidden_size for LLaVA-1.5.
    pub fn encode_image(&self, pixels: &[f32]) -> ArchResult<Vec<f32>> {
        let clip_features = self.vision_encoder.encode(pixels)?;
        self.mm_projector.project(&clip_features)
    }
}

impl ForwardPass for LlavaModel {
    /// Text-only forward pass — delegates to the LLaMA language backbone.
    ///
    /// For multimodal inference, call `encode_image()` first to obtain visual
    /// embeddings, then inject them into the prompt at the `<image>` placeholder
    /// positions before calling this method.
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
}

/// Convenience entry-point used by the architecture registry.
pub fn load_llava_from_gguf(gguf: &GgufModel, config: &ModelConfig) -> ArchResult<LlavaModel> {
    LlavaModel::load(gguf, config)
}

// ---- Internal tensor loading helpers ----

/// Load a tensor as f32 (for non-quantized tensors like norms and biases).
/// Returns `None` if the tensor is absent — callers use `.unwrap_or_default()`.
fn load_f32_tensor_opt(
    gguf: &GgufModel,
    dispatcher: &KernelDispatcher,
    name: &str,
) -> Option<Vec<f32>> {
    let info = gguf.file.tensors.get(name).ok()?;
    let data = gguf.tensor_data(name).ok()?;
    dequant_to_f32(info, data, dispatcher).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MmProjector tests ----

    #[test]
    fn test_mm_projector_shape_output() {
        // fc1: [8, 4] maps 4-dim clip to 8-dim hidden
        // fc2: [4, 8] maps 8-dim hidden back to 4-dim llm
        let proj = MmProjector {
            fc1_weight: vec![1.0; 8 * 4],
            fc1_bias: vec![0.0; 8],
            fc2_weight: vec![1.0; 4 * 8],
            fc2_bias: vec![0.0; 4],
            clip_hidden_size: 4,
            mm_hidden_size: 8,
            llm_hidden_size: 4,
        };
        let input = vec![1.0f32; 4]; // 1 patch
        let output = proj.project(&input).expect("project should succeed");
        assert_eq!(
            output.len(),
            4,
            "output must be [num_patches * llm_hidden_size]"
        );
    }

    #[test]
    fn test_mm_projector_two_patches() {
        // 2 patches of clip_hidden_size=4 → 2 * llm_hidden_size=4 = 8 elements
        let proj = MmProjector {
            fc1_weight: vec![0.1; 8 * 4],
            fc1_bias: vec![0.0; 8],
            fc2_weight: vec![0.1; 4 * 8],
            fc2_bias: vec![0.0; 4],
            clip_hidden_size: 4,
            mm_hidden_size: 8,
            llm_hidden_size: 4,
        };
        let input = vec![1.0f32; 8]; // 2 patches × 4 dims
        let output = proj.project(&input).expect("two-patch project");
        assert_eq!(output.len(), 8, "2 patches × llm_hidden=4");
    }

    #[test]
    fn test_mm_projector_wrong_input_size_errors() {
        let proj = MmProjector {
            fc1_weight: vec![0.0; 8],
            fc1_bias: vec![0.0; 2],
            fc2_weight: vec![0.0; 8],
            fc2_bias: vec![0.0; 4],
            clip_hidden_size: 4,
            mm_hidden_size: 2,
            llm_hidden_size: 4,
        };
        // 3 elements is not divisible by clip_hidden_size=4
        let input = vec![1.0f32; 3];
        assert!(
            proj.project(&input).is_err(),
            "must error on bad input size"
        );
    }

    #[test]
    fn test_mm_projector_zero_clip_hidden_errors() {
        let proj = MmProjector {
            fc1_weight: vec![],
            fc1_bias: vec![],
            fc2_weight: vec![],
            fc2_bias: vec![],
            clip_hidden_size: 0,
            mm_hidden_size: 4,
            llm_hidden_size: 4,
        };
        let input = vec![1.0f32; 4];
        assert!(
            proj.project(&input).is_err(),
            "zero clip_hidden_size must error"
        );
    }

    #[test]
    fn test_mm_projector_gelu_output_is_nonnegative_for_positive_input() {
        // GeLU(x) ≈ x for large positive x.
        // With all-ones weights and positive input, output should be positive.
        let proj = MmProjector {
            fc1_weight: vec![1.0; 2 * 2],
            fc1_bias: vec![0.0; 2],
            fc2_weight: vec![1.0; 2 * 2],
            fc2_bias: vec![0.0; 2],
            clip_hidden_size: 2,
            mm_hidden_size: 2,
            llm_hidden_size: 2,
        };
        let input = vec![1.0f32, 1.0f32];
        let output = proj.project(&input).expect("project ok");
        for v in &output {
            assert!(
                *v > 0.0,
                "all-positive input through GeLU should stay positive"
            );
        }
    }

    // ---- ClipEncoder tests ----

    #[test]
    fn test_clip_encoder_returns_correct_patch_count() {
        // image_size=28, patch_size=14 → (28/14)^2 = 4 patches, hidden_size=8
        // Provide correctly-sized weights for the forward pass:
        //   patch_embd_weight: [hidden_size=8, patch_flat=14*14*3=588]
        //   patch_embd_bias: [hidden_size=8]
        //   position_embd: [(num_patches+1=5) * hidden_size=8 = 40]
        //   post_ln_weight / post_ln_bias: [hidden_size=8]
        let hidden_size = 8usize;
        let patch_size = 14usize;
        let image_size = 28usize;
        let patch_flat = patch_size * patch_size * 3;
        let num_patches = (image_size / patch_size).pow(2);
        let num_positions = num_patches + 1;

        let enc = ClipEncoder {
            patch_embd_weight: vec![0.0f32; hidden_size * patch_flat],
            patch_embd_bias: vec![0.0f32; hidden_size],
            position_embd: vec![0.0f32; num_positions * hidden_size],
            pre_ln_weight: vec![],
            pre_ln_bias: vec![],
            post_ln_weight: vec![1.0f32; hidden_size],
            post_ln_bias: vec![0.0f32; hidden_size],
            layers: vec![],
            hidden_size,
            num_heads: 4,
            patch_size,
            image_size,
        };
        let pixels = vec![0.0f32; 3 * image_size * image_size];
        let features = enc.encode(&pixels).expect("encode should not fail");
        // (28 / 14)^2 = 4 patches × 8 dims = 32
        assert_eq!(
            features.len(),
            num_patches * hidden_size,
            "expected 32 features for 2×2 patches"
        );
    }

    #[test]
    fn test_clip_encoder_num_patches() {
        let enc = ClipEncoder {
            patch_embd_weight: vec![],
            patch_embd_bias: vec![],
            position_embd: vec![],
            pre_ln_weight: vec![],
            pre_ln_bias: vec![],
            post_ln_weight: vec![],
            post_ln_bias: vec![],
            layers: vec![],
            hidden_size: 1024,
            num_heads: 16,
            patch_size: 14,
            image_size: 336, // LLaVA-1.5 default
        };
        // (336/14)^2 = 24^2 = 576 patches
        assert_eq!(enc.num_patches(), 576, "LLaVA-1.5 should have 576 patches");
    }

    #[test]
    fn test_clip_encoder_zero_patch_size_errors() {
        // patch_size=0 is an invalid configuration — must return ArchError::InvalidConfig.
        let enc = ClipEncoder {
            patch_embd_weight: vec![],
            patch_embd_bias: vec![],
            position_embd: vec![],
            pre_ln_weight: vec![],
            pre_ln_bias: vec![],
            post_ln_weight: vec![],
            post_ln_bias: vec![],
            layers: vec![],
            hidden_size: 64,
            num_heads: 4,
            patch_size: 0, // invalid
            image_size: 336,
        };
        // num_patches() is 0 for zero patch_size (no-panic, just zero)
        assert_eq!(enc.num_patches(), 0);
        // But encode() must return Err for zero patch_size
        let pixels = vec![0.0f32; 3];
        assert!(
            enc.encode(&pixels).is_err(),
            "zero patch_size must return Err from encode()"
        );
    }

    // ---- LlavaArchitecture tests ----

    #[test]
    fn test_llava_arch_id() {
        use crate::llava::LlavaArchitecture;
        use crate::traits::ModelArchitecture;
        let arch = LlavaArchitecture::new();
        assert_eq!(arch.arch_id(), "llava");
    }

    #[test]
    fn test_llava_architecture_default() {
        use crate::llava::LlavaArchitecture;
        use crate::traits::ModelArchitecture;
        let arch = LlavaArchitecture;
        assert_eq!(arch.arch_id(), "llava");
    }

    #[test]
    fn test_llava_tensor_names_includes_mm_projector() {
        use crate::llava::LlavaArchitecture;
        use crate::traits::ModelArchitecture;
        let arch = LlavaArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|n| n.pattern.as_str()).collect();
        assert!(
            patterns.contains(&"mm.0.weight"),
            "tensor_names must include mm.0.weight"
        );
        assert!(
            patterns.contains(&"mm.2.weight"),
            "tensor_names must include mm.2.weight"
        );
    }

    #[test]
    fn test_llava_tensor_names_includes_vision_encoder() {
        use crate::llava::LlavaArchitecture;
        use crate::traits::ModelArchitecture;
        let arch = LlavaArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|n| n.pattern.as_str()).collect();
        assert!(
            patterns.contains(&"v.patch_embd.weight"),
            "tensor_names must include v.patch_embd.weight"
        );
    }

    // ---- New CLIP ViT forward pass tests ----

    /// Passing wrong-length pixel array must return Err(ArchError::InvalidShape).
    #[test]
    fn test_clip_encode_wrong_pixel_size_errors() {
        let hidden_size = 8usize;
        let patch_size = 14usize;
        let image_size = 28usize;
        let patch_flat = patch_size * patch_size * 3;
        let num_patches = (image_size / patch_size).pow(2);
        let num_positions = num_patches + 1;

        let enc = ClipEncoder {
            patch_embd_weight: vec![0.0f32; hidden_size * patch_flat],
            patch_embd_bias: vec![0.0f32; hidden_size],
            position_embd: vec![0.0f32; num_positions * hidden_size],
            pre_ln_weight: vec![],
            pre_ln_bias: vec![],
            post_ln_weight: vec![1.0f32; hidden_size],
            post_ln_bias: vec![0.0f32; hidden_size],
            layers: vec![],
            hidden_size,
            num_heads: 4,
            patch_size,
            image_size,
        };

        // Correct size is 3 * 28 * 28 = 2352; pass 100 instead.
        let wrong_pixels = vec![0.0f32; 100];
        let result = enc.encode(&wrong_pixels);
        assert!(result.is_err(), "wrong pixel length must return Err");
    }

    /// With 0 layers the encoder should still return correctly-shaped output
    /// (patch embedding + positional embeddings only, no attention layers).
    #[test]
    fn test_clip_encode_zero_layers_returns_positional_only() {
        let hidden_size = 8usize;
        let patch_size = 14usize;
        let image_size = 28usize;
        let patch_flat = patch_size * patch_size * 3;
        let num_patches = (image_size / patch_size).pow(2); // 4
        let num_positions = num_patches + 1;

        let mut pos_embd = vec![0.0f32; num_positions * hidden_size];
        for (i, v) in pos_embd.iter_mut().enumerate() {
            *v = (i as f32) * 0.01;
        }

        let enc = ClipEncoder {
            patch_embd_weight: vec![0.1f32; hidden_size * patch_flat],
            patch_embd_bias: vec![0.0f32; hidden_size],
            position_embd: pos_embd,
            pre_ln_weight: vec![],
            pre_ln_bias: vec![],
            post_ln_weight: vec![1.0f32; hidden_size],
            post_ln_bias: vec![0.0f32; hidden_size],
            layers: vec![],
            hidden_size,
            num_heads: 4,
            patch_size,
            image_size,
        };

        let pixels = vec![1.0f32; 3 * image_size * image_size];
        let out = enc
            .encode(&pixels)
            .expect("zero-layer encode should succeed");
        assert_eq!(
            out.len(),
            num_patches * hidden_size,
            "zero-layer encode must return num_patches * hidden_size = {} elements, got {}",
            num_patches * hidden_size,
            out.len()
        );
    }

    /// LayerNorm output should have near-zero mean and unit variance.
    #[test]
    fn test_clip_layer_norm_centering() {
        let input = vec![1.0f32, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let w = vec![1.0f32; 8];
        let b = vec![0.0f32; 8];

        let out = ClipEncoder::layer_norm(&input, &w, &b);
        assert_eq!(out.len(), input.len(), "layernorm must preserve length");

        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(
            mean.abs() < 1e-4,
            "LayerNorm output should have near-zero mean, got {mean}"
        );

        let var: f32 = out.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / out.len() as f32;
        assert!(
            (var - 1.0).abs() < 1e-3,
            "LayerNorm output variance should be near 1, got {var}"
        );
    }

    /// Full encode() with 1 layer and small dims: output shape must be
    /// (image_size/patch_size)^2 * hidden_size = (28/14)^2 * 8 = 32.
    #[test]
    fn test_clip_encode_output_shape_correct() {
        let hidden_size = 8usize;
        let num_heads = 2usize;
        let patch_size = 14usize;
        let image_size = 28usize;
        let patch_flat = patch_size * patch_size * 3;
        let num_patches = (image_size / patch_size).pow(2); // 4
        let num_positions = num_patches + 1;
        let ffn_dim = hidden_size * 4; // typical 4× expansion

        let attn_w = vec![0.01f32; hidden_size * hidden_size];
        let attn_b = vec![0.0f32; hidden_size];

        let fc1_w = vec![0.01f32; ffn_dim * hidden_size];
        let fc1_b = vec![0.0f32; ffn_dim];
        let fc2_w = vec![0.01f32; hidden_size * ffn_dim];
        let fc2_b = vec![0.0f32; hidden_size];

        let layer = ClipEncoderLayer {
            ln1_weight: vec![1.0f32; hidden_size],
            ln1_bias: vec![0.0f32; hidden_size],
            ln2_weight: vec![1.0f32; hidden_size],
            ln2_bias: vec![0.0f32; hidden_size],
            q_weight: attn_w.clone(),
            k_weight: attn_w.clone(),
            v_weight: attn_w.clone(),
            out_weight: attn_w,
            q_bias: attn_b.clone(),
            k_bias: attn_b.clone(),
            v_bias: attn_b.clone(),
            out_bias: attn_b,
            fc1_weight: fc1_w,
            fc1_bias: fc1_b,
            fc2_weight: fc2_w,
            fc2_bias: fc2_b,
        };

        let enc = ClipEncoder {
            patch_embd_weight: vec![0.01f32; hidden_size * patch_flat],
            patch_embd_bias: vec![0.0f32; hidden_size],
            position_embd: vec![0.01f32; num_positions * hidden_size],
            pre_ln_weight: vec![],
            pre_ln_bias: vec![],
            post_ln_weight: vec![1.0f32; hidden_size],
            post_ln_bias: vec![0.0f32; hidden_size],
            layers: vec![layer],
            hidden_size,
            num_heads,
            patch_size,
            image_size,
        };

        let pixels: Vec<f32> = (0..(3 * image_size * image_size))
            .map(|i| (i as f32) * 0.001)
            .collect();

        let out = enc.encode(&pixels).expect("1-layer encode should succeed");

        // Expected: (28/14)^2 * 8 = 4 * 8 = 32
        let expected_len = num_patches * hidden_size;
        assert_eq!(
            out.len(),
            expected_len,
            "encode output shape: expected {expected_len}, got {}",
            out.len()
        );
    }
}

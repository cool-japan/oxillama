//! LLaVA-1.6 / LLaVA-NeXT model with anyres tiling.
//!
//! ## Visual token order
//!
//! ```text
//! pixels → split_into_tiles() → [tile₀ … tileₙ] + thumbnail
//!            ↓ ClipEncoder (each)          ↓ ClipEncoder
//!          grid features                base feature
//!            ↓ MmProjector                ↓ MmProjector
//!          [rows × cols × patches, llm]  [patches, llm]
//!            ↓ unpad + image_newline
//!      final order:  [ base ][ row₀ ⏎ ][ row₁ ⏎ ] …
//! ```
//!
//! **The base/thumbnail feature comes first.**  Three independent confirmations:
//!
//! 1. `llava_uhd::slice_image` (`tools/mtmd/clip.cpp:2708-2717`) pushes the
//!    resized overview into `output` *before* any slice.
//! 2. The deleted legacy path memcpy'd the base to offset 0 —
//!    `b4726345a^:tools/mtmd/llava.cpp:220-223`, `"main image as global context"`.
//! 3. The upstream Python reference, transcribed in that same file, ends with
//!    `torch.cat((base_image_feature, image_feature), dim=0)`.
//!
//! The previous implementation appended the thumbnail **last**, so every visual
//! token was at the wrong position relative to the model's training.
//!
//! ## `model.image_newline` and unpad — a divergence worth stating
//!
//! llama.cpp at HEAD loads `model.image_newline`
//! (`TN_IMAGE_NEWLINE = "model.image_newline"`, `clip-impl.h:100`; loaded at
//! `clip.cpp:1495`) and exposes `clip_get_newline_tensor` (`clip.cpp:3297`) —
//! but that accessor has **zero callers**, and `unpad_image` does not exist
//! anywhere in the tree.  `clip_llava_handle_patches`, which did the rearrange,
//! was deleted in `b4726345a`.  So llama.cpp currently concatenates
//! `[base][grid…]` verbatim.
//!
//! The upstream LLaVA-NeXT reference — quoted verbatim as a comment inside the
//! deleted file, `b4726345a^:tools/mtmd/llava.cpp:144-162` — does unpad the
//! grid and append `image_newline` after every feature row.  This module
//! implements that whenever the checkpoint actually carries the tensor, and
//! falls back to llama.cpp's plain concatenation when it does not.  Both orders
//! put the base first.

use crate::common::loader::load_dequant_tensor;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::llama::{load_llama_from_gguf, LlamaModel};
use crate::llava::clip::ClipEncoder;
use crate::llava::inject::{Prompt, VisualTokens};
use crate::llava::model::{clip_params_from_metadata, load_clip_tower};
use crate::llava::projector::MmProjector;
use crate::llava_next::tiler::AnyresTileConfig;
use crate::lora::LoadedLora;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::GgufModel;

/// LLaVA-1.6 (NeXT) anyres multimodal model.
pub struct LlavaNextModel {
    /// CLIP vision tower (shared across tiles and thumbnail).
    vision_encoder: ClipEncoder,
    /// Multi-modal projector: CLIP features → LLM hidden dim.
    mm_projector: MmProjector,
    /// LLaMA language backbone.
    language_model: LlamaModel,
    /// `model.image_newline`, `[llm_hidden]`.  Empty when the checkpoint omits it.
    image_newline: Vec<f32>,
    /// LLM hidden size.
    llm_hidden_size: usize,
    /// CLIP output hidden size.
    clip_hidden_size: usize,
    /// Anyres tile splitter configuration.
    tiler: AnyresTileConfig,
}

impl LlavaNextModel {
    /// Load from a main GGUF plus a `clip`-architecture mmproj.
    ///
    /// # Errors
    ///
    /// Propagates backbone and tower loading failures; every vision tensor is
    /// required, so a renamed or absent one is reported rather than zero-filled.
    pub fn load_with_mmproj(
        gguf: &GgufModel,
        mmproj: &GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Self> {
        let language_model = load_llama_from_gguf(gguf, config)?;
        let hidden_size = config.hidden_size;

        let params = clip_params_from_metadata(&mmproj.file.metadata);
        params.validate()?;
        let clip_hidden_size = params.hidden_size;

        let fc1_weight = load_dequant_tensor(mmproj, "mm.0.weight")?;
        let mm_hidden_size = fc1_weight.len().checked_div(clip_hidden_size).unwrap_or(0);
        let mm_projector = MmProjector {
            fc1_weight,
            fc1_bias: crate::common::loader::load_bias(mmproj, "mm.0.bias")?.unwrap_or_default(),
            fc2_weight: load_dequant_tensor(mmproj, "mm.2.weight")?,
            fc2_bias: crate::common::loader::load_bias(mmproj, "mm.2.bias")?.unwrap_or_default(),
            clip_hidden_size,
            mm_hidden_size,
            llm_hidden_size: hidden_size,
        };
        mm_projector.validate()?;

        let vision_encoder = load_clip_tower(mmproj, params)?;

        // `model.image_newline` carries no `v.`/`mm.` prefix — it is the one
        // odd name out in `clip-impl.h`.  It may live in either file.
        let image_newline = if mmproj.file.tensors.contains("model.image_newline") {
            load_dequant_tensor(mmproj, "model.image_newline")?
        } else if gguf.file.tensors.contains("model.image_newline") {
            load_dequant_tensor(gguf, "model.image_newline")?
        } else {
            Vec::new()
        };
        if !image_newline.is_empty() && image_newline.len() < hidden_size {
            return Err(ArchError::InvalidShape {
                name: "model.image_newline".to_string(),
                expected: vec![hidden_size],
                got: vec![image_newline.len()],
            });
        }

        let tiler = Self::parse_tiler_config(mmproj, vision_encoder.params.image_size);

        Ok(Self {
            vision_encoder,
            mm_projector,
            language_model,
            image_newline,
            llm_hidden_size: hidden_size,
            clip_hidden_size,
            tiler,
        })
    }

    /// Load from a single GGUF carrying backbone, projector and tower.
    ///
    /// # Errors
    ///
    /// See [`Self::load_with_mmproj`].
    pub fn load(gguf: &GgufModel, config: &ModelConfig) -> ArchResult<Self> {
        Self::load_with_mmproj(gguf, gguf, config)
    }

    /// Parse anyres config from GGUF metadata.
    ///
    /// `clip.vision.image_grid_pinpoints` (`clip-impl.h:52`) is a flat array of
    /// pixel pairs; `llava16.*` is accepted as a legacy alias.
    fn parse_tiler_config(gguf: &GgufModel, tile_size_default: usize) -> AnyresTileConfig {
        let meta = &gguf.file.metadata;
        let tile_size = meta
            .get_u32("llava16.tile_size")
            .map(|v| v as usize)
            .unwrap_or(tile_size_default.max(1));
        let max_tiles = meta
            .get_u32("llava16.max_tiles")
            .map(|v| v as usize)
            .unwrap_or(6);

        let pinpoints = meta
            .get("clip.vision.image_grid_pinpoints")
            .or_else(|| meta.get("llava16.image_grid_pinpoints"));

        let grid_pinpoints = match pinpoints {
            Some(oxillama_gguf::MetadataValue::Array(arr)) => {
                let flat: Vec<u32> = arr.iter().filter_map(|v| v.as_u32()).collect();
                let parsed: Vec<(usize, usize)> = flat
                    .chunks_exact(2)
                    .filter_map(|pair| {
                        let (w_px, h_px) = (pair[0] as usize, pair[1] as usize);
                        if tile_size == 0 || w_px == 0 || h_px == 0 {
                            return None;
                        }
                        let (cols, rows) = (w_px / tile_size, h_px / tile_size);
                        (cols > 0 && rows > 0).then_some((cols, rows))
                    })
                    .collect();
                if parsed.is_empty() {
                    AnyresTileConfig::default_llava16().grid_pinpoints
                } else {
                    parsed
                }
            }
            _ => AnyresTileConfig::default_llava16().grid_pinpoints,
        };

        AnyresTileConfig {
            tile_size,
            max_tiles,
            grid_pinpoints,
        }
    }

    /// Patches along one side of a single tile.
    fn patches_per_side(&self) -> usize {
        self.tiler
            .tile_size
            .checked_div(self.vision_encoder.params.patch_size)
            .unwrap_or(0)
    }

    /// Encode an image with anyres tiling into LLM-space visual tokens.
    ///
    /// Output order is `[base][grid rows, each followed by image_newline]`.
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidConfig`] — zero image dimensions, or a tile size
    ///   that is not a whole number of patches.
    /// * Propagates tower / projector failures.
    pub fn encode_image(
        &self,
        pixels: &[f32],
        img_w: usize,
        img_h: usize,
    ) -> ArchResult<VisualTokens> {
        if img_w == 0 || img_h == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "encode_image: img_w and img_h must be > 0".to_string(),
            });
        }
        let side = self.patches_per_side();
        if side == 0 {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "encode_image: tile_size ({}) is smaller than patch_size ({})",
                    self.tiler.tile_size, self.vision_encoder.params.patch_size
                ),
            });
        }

        let (tile_pixels, thumbnail) = self.tiler.split_into_tiles(pixels, img_w, img_h)?;
        let (grid_cols, grid_rows) = self.tiler.select_grid(img_w, img_h);

        // ── Base / thumbnail feature — FIRST ─────────────────────────────────
        let base_clip =
            self.vision_encoder
                .encode(&thumbnail)
                .map_err(|e| ArchError::ForwardPassError {
                    layer: 0,
                    message: format!("CLIP thumbnail encode: {e}"),
                })?;
        let mut out = self.mm_projector.project(&base_clip)?;

        // ── Grid tiles, projected then spatially rearranged ──────────────────
        let mut grid: Vec<f32> = Vec::new();
        for (idx, tile) in tile_pixels.iter().enumerate() {
            let feats =
                self.vision_encoder
                    .encode(tile)
                    .map_err(|e| ArchError::ForwardPassError {
                        layer: idx,
                        message: format!("CLIP tile {idx} encode: {e}"),
                    })?;
            grid.extend_from_slice(&self.mm_projector.project(&feats)?);
        }

        if grid.is_empty() {
            return VisualTokens::new(out, self.llm_hidden_size);
        }

        if self.image_newline.is_empty() {
            // llama.cpp HEAD behaviour: plain concatenation after the base.
            out.extend_from_slice(&grid);
            return VisualTokens::new(out, self.llm_hidden_size);
        }

        let rows = self.assemble_grid_rows(&grid, grid_rows, grid_cols, side, img_w, img_h)?;
        out.extend_from_slice(&rows);
        VisualTokens::new(out, self.llm_hidden_size)
    }

    /// Merge the per-tile features into one feature map, unpad it, and append
    /// `image_newline` after each row.
    ///
    /// Mirrors the upstream reference quoted at
    /// `b4726345a^:tools/mtmd/llava.cpp:144-162`:
    ///
    /// ```text
    /// view(num_patch_height, num_patch_width, height, width, -1)
    ///   .permute(4, 0, 2, 1, 3).flatten(1,2).flatten(2,3)
    /// unpad_image(..., original_size)
    /// cat(feature, image_newline.expand(..., 1), dim=-1)
    /// flatten(1,2).transpose(0,1)
    /// ```
    fn assemble_grid_rows(
        &self,
        grid: &[f32],
        grid_rows: usize,
        grid_cols: usize,
        side: usize,
        img_w: usize,
        img_h: usize,
    ) -> ArchResult<Vec<f32>> {
        assemble_grid_rows(
            grid,
            grid_rows,
            grid_cols,
            side,
            self.llm_hidden_size,
            &self.image_newline,
            img_w,
            img_h,
        )
    }

    /// Splice image embeddings into a prompt — see [`crate::llava::inject`].
    ///
    /// # Errors
    ///
    /// Propagates embedding-table lookup and splicing failures.
    pub fn build_input_embeddings(&self, prompt: &Prompt<'_>) -> ArchResult<Vec<f32>> {
        let embd = &self.language_model.token_embd;
        let dispatcher = &self.language_model.dispatcher;
        prompt.build_embeddings(self.llm_hidden_size, |tok, row| {
            embd.row_into(dispatcher, tok, row)
        })
    }

    /// Run a multimodal forward pass — see [`crate::llava::LlavaModel::forward_multimodal`].
    ///
    /// # Errors
    ///
    /// Propagates splicing and backbone failures.
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
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] on a length mismatch; otherwise whatever the
    /// backbone reports.
    pub fn forward_embeds(
        &mut self,
        embeds: &[f32],
        seq_len: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let hidden = self.llm_hidden_size;
        if seq_len == 0 || embeds.len() != seq_len * hidden {
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

    /// Tile config accessor (for tests and diagnostics).
    pub fn tiler(&self) -> &AnyresTileConfig {
        &self.tiler
    }

    /// CLIP hidden size (for diagnostics).
    pub fn clip_hidden_size(&self) -> usize {
        self.clip_hidden_size
    }

    /// Whether the checkpoint carries `model.image_newline`.
    pub fn has_image_newline(&self) -> bool {
        !self.image_newline.is_empty()
    }
}

/// Merge tile features into one feature map, unpad it, and append
/// `image_newline` after each row.
///
/// Free function so tests exercise the *production* index arithmetic rather
/// than a reimplementation of it.
///
/// Mirrors the upstream reference quoted at
/// `b4726345a^:tools/mtmd/llava.cpp:144-162`:
///
/// ```text
/// view(num_patch_height, num_patch_width, height, width, -1)
///   .permute(4, 0, 2, 1, 3).flatten(1,2).flatten(2,3)
/// unpad_image(..., original_size)
/// cat(feature, image_newline.expand(..., 1), dim=-1)
/// flatten(1,2).transpose(0,1)
/// ```
///
/// # Errors
///
/// [`ArchError::InvalidShape`] when `grid` does not hold
/// `grid_rows * side × grid_cols * side` rows of `hidden`, or `newline` is
/// shorter than `hidden`.
#[allow(clippy::too_many_arguments)]
pub fn assemble_grid_rows(
    grid: &[f32],
    grid_rows: usize,
    grid_cols: usize,
    side: usize,
    hidden: usize,
    newline: &[f32],
    img_w: usize,
    img_h: usize,
) -> ArchResult<Vec<f32>> {
    {
        let map_h = grid_rows * side;
        let map_w = grid_cols * side;
        let expected = map_h * map_w * hidden;
        if grid.len() != expected {
            return Err(ArchError::InvalidShape {
                name: "anyres grid features".to_string(),
                expected: vec![map_h, map_w, hidden],
                got: vec![grid.len()],
            });
        }

        if newline.len() < hidden {
            return Err(ArchError::InvalidShape {
                name: "model.image_newline".to_string(),
                expected: vec![hidden],
                got: vec![newline.len()],
            });
        }

        let (crop_top, crop_left, crop_h, crop_w) = unpad_extent(map_h, map_w, img_h, img_w);

        let mut out = Vec::with_capacity(crop_h * (crop_w + 1) * hidden);
        for r in 0..crop_h {
            let map_row = crop_top + r;
            for c in 0..crop_w {
                let map_col = crop_left + c;
                // Tile (tr, tc) occupies map rows [tr*side, (tr+1)*side).
                let tile_row = map_row / side;
                let tile_col = map_col / side;
                let in_row = map_row % side;
                let in_col = map_col % side;
                let tile_index = tile_row * grid_cols + tile_col;
                let token = tile_index * side * side + in_row * side + in_col;
                let start = token * hidden;
                out.extend_from_slice(&grid[start..start + hidden]);
            }
            out.extend_from_slice(&newline[..hidden]);
        }
        Ok(out)
    }
}

/// Compute the unpadded crop of an anyres feature map.
///
/// Returns `(top, left, height, width)` in feature-map units.  Port of the
/// upstream `unpad_image`: the tiling resizes the image to fit the grid canvas
/// while preserving aspect ratio and centres it, so the padding is the
/// difference between the canvas and the scaled extent, split evenly.
///
/// The previous implementation had no unpad step at all, so padding rows and
/// columns became visual tokens.
pub fn unpad_extent(
    map_h: usize,
    map_w: usize,
    orig_h: usize,
    orig_w: usize,
) -> (usize, usize, usize, usize) {
    if map_h == 0 || map_w == 0 || orig_h == 0 || orig_w == 0 {
        return (0, 0, map_h, map_w);
    }
    let orig_ratio = orig_w as f64 / orig_h as f64;
    let cur_ratio = map_w as f64 / map_h as f64;

    if orig_ratio > cur_ratio {
        // Width-limited: the image spans the full width, padding is top/bottom.
        let scale = map_w as f64 / orig_w as f64;
        let new_h = (orig_h as f64 * scale).round() as usize;
        let new_h = new_h.min(map_h);
        let pad = (map_h - new_h) / 2;
        (pad, 0, map_h.saturating_sub(2 * pad), map_w)
    } else {
        // Height-limited: padding is left/right.
        let scale = map_h as f64 / orig_h as f64;
        let new_w = (orig_w as f64 * scale).round() as usize;
        let new_w = new_w.min(map_w);
        let pad = (map_w - new_w) / 2;
        (0, pad, map_h, map_w.saturating_sub(2 * pad))
    }
}

impl ForwardPass for LlavaNextModel {
    /// Text-only forward pass — delegates to the LLaMA backbone.
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
/// See [`LlavaNextModel::load`].
pub fn load_llava_next_from_gguf(
    gguf: &GgufModel,
    config: &ModelConfig,
) -> ArchResult<LlavaNextModel> {
    LlavaNextModel::load(gguf, config)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── unpad (V6) ──────────────────────────────────────────────────────────

    /// A square image on a square canvas needs no crop.
    #[test]
    fn unpad_square_on_square_is_a_no_op() {
        assert_eq!(unpad_extent(48, 48, 100, 100), (0, 0, 48, 48));
    }

    /// A wide image on a square canvas is padded top and bottom.
    #[test]
    fn unpad_wide_image_crops_rows() {
        // 2:1 image on a 48×48 map ⇒ scaled height 24, 12 rows padding each side.
        let (top, left, h, w) = unpad_extent(48, 48, 100, 200);
        assert_eq!((top, left), (12, 0));
        assert_eq!((h, w), (24, 48));
    }

    /// A tall image on a square canvas is padded left and right.
    #[test]
    fn unpad_tall_image_crops_columns() {
        let (top, left, h, w) = unpad_extent(48, 48, 200, 100);
        assert_eq!((top, left), (0, 12));
        assert_eq!((h, w), (48, 24));
    }

    #[test]
    fn unpad_degenerate_inputs_are_safe() {
        assert_eq!(unpad_extent(0, 0, 10, 10), (0, 0, 0, 0));
        assert_eq!(unpad_extent(4, 4, 0, 0), (0, 0, 4, 4));
    }

    // ── Ordering + newline (V6) ─────────────────────────────────────────────

    /// The tile-major → row-major remap must read the feature map in raster
    /// order across tile boundaries, not tile by tile.
    ///
    /// This calls the production [`assemble_grid_rows`], not a copy of it.
    #[test]
    fn grid_rearrange_reads_the_feature_map_in_raster_order() {
        // 2×2 tiles, each 2×2 patches ⇒ a 4×4 feature map, hidden = 1.
        // Token t of tile k has value (k * 4 + t).
        let grid: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // Square image ⇒ no crop.
        let out = assemble_grid_rows(&grid, 2, 2, 2, 1, &[-1.0], 100, 100).expect("assemble");

        let expected = [
            0.0, 1.0, 4.0, 5.0, -1.0, // map row 0
            2.0, 3.0, 6.0, 7.0, -1.0, // map row 1
            8.0, 9.0, 12.0, 13.0, -1.0, // map row 2
            10.0, 11.0, 14.0, 15.0, -1.0, // map row 3
        ];
        assert_eq!(out, expected);
    }

    /// One newline token is appended per feature row, so the token count is
    /// `crop_h * (crop_w + 1)`, not `rows * cols * patches`.
    #[test]
    fn newline_adds_exactly_one_token_per_row() {
        let grid: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = assemble_grid_rows(&grid, 2, 2, 2, 1, &[-1.0], 100, 100).expect("assemble");
        assert_eq!(out.len(), 4 * (4 + 1), "4 rows × (4 features + 1 newline)");
        for row in 0..4 {
            assert_eq!(out[row * 5 + 4], -1.0, "row {row} must end with newline");
        }
    }

    /// Unpadding drops the padded rows before newlines are inserted.
    #[test]
    fn unpad_reduces_the_visual_token_count() {
        let grid: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // 2:1 wide image on a 4×4 map ⇒ 2 rows kept.
        let out = assemble_grid_rows(&grid, 2, 2, 2, 1, &[-1.0], 200, 100).expect("assemble");
        assert_eq!(out.len(), 2 * (4 + 1), "2 unpadded rows × (4 + newline)");
        // The surviving rows are the middle two of the feature map.
        assert_eq!(out[0], 2.0, "first kept row starts at map row 1");
    }

    /// A grid whose length disagrees with the declared shape is reported.
    #[test]
    fn grid_rearrange_rejects_a_short_grid() {
        assert!(assemble_grid_rows(&[0.0; 10], 2, 2, 2, 1, &[-1.0], 100, 100).is_err());
    }

    /// A newline shorter than the hidden size is reported.
    #[test]
    fn grid_rearrange_rejects_a_short_newline() {
        let grid: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert!(assemble_grid_rows(&grid, 2, 2, 2, 2, &[-1.0], 100, 100).is_err());
    }

    // ── Tiler ───────────────────────────────────────────────────────────────

    #[test]
    fn llava_next_default_tiler() {
        let tiler = AnyresTileConfig::default_llava16();
        assert_eq!(tiler.tile_size, 336);
        assert_eq!(tiler.max_tiles, 6);
        assert!(!tiler.grid_pinpoints.is_empty());
    }

    /// The tiler still splits into `cols × rows` tiles plus a thumbnail.
    #[test]
    fn tiler_produces_grid_plus_thumbnail() {
        let tiler = AnyresTileConfig {
            tile_size: 28,
            max_tiles: 6,
            grid_pinpoints: vec![(2, 2)],
        };
        let pixels = vec![0.5f32; 3 * 56 * 56];
        let (tiles, thumb) = tiler.split_into_tiles(&pixels, 56, 56).expect("split");
        assert_eq!(tiles.len(), 4);
        assert_eq!(thumb.len(), 3 * 28 * 28);
    }
}

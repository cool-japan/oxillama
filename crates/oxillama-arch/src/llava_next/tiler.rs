//! Anyres tile splitter for LLaVA-1.6 / LLaVA-NeXT.
//!
//! LLaVA-NeXT divides an input image into a variable grid of tiles (e.g. 2×2,
//! 1×3, 3×1) plus a global 336×336 thumbnail.  Each tile is independently
//! processed by the CLIP vision encoder, giving the model much higher effective
//! resolution than the fixed 336×336 crop used by LLaVA-1.5.
//!
//! ## Tile selection algorithm
//!
//! Given image dimensions `(img_w, img_h)` and a list of supported grid
//! configurations `(cols, rows)` (stored in `grid_pinpoints`), we pick the grid
//! that minimises wasted padding while keeping the total tile count within
//! `max_tiles` (thumbnail counts as 1).
//!
//! For each candidate grid `(cols, rows)`:
//! 1. Scale the image to fit inside `(cols * tile_size, rows * tile_size)`
//!    while preserving aspect ratio.
//! 2. Compute the fraction of the bounding box that is actual image pixels
//!    (1.0 = perfect fit, 0.0 = total waste).
//! 3. The candidate with the highest fill fraction wins.  Ties are broken by
//!    preferring smaller total tile count (fewest wasted encoder calls).

use crate::error::{ArchError, ArchResult};

/// Configuration for anyres tiling used by LLaVA-NeXT.
#[derive(Debug, Clone)]
pub struct AnyresTileConfig {
    /// Pixels per tile side (e.g. 336).  Must match the CLIP encoder's
    /// expected input size.
    pub tile_size: usize,
    /// Maximum total tile count **including** the global thumbnail.
    /// E.g. `6` means at most 5 grid tiles + 1 thumbnail.
    pub max_tiles: usize,
    /// Supported grid configurations as `(cols, rows)` pairs.
    ///
    /// `cols * rows + 1` must not exceed `max_tiles` for any entry in this
    /// list (enforced by the LLaVA-NeXT configuration).
    ///
    /// Typical LLaVA-1.6 pinpoints (in tile units, each tile = 336 px):
    /// - (1, 2), (2, 1), (2, 2), (3, 1), (1, 3)
    pub grid_pinpoints: Vec<(usize, usize)>,
}

impl AnyresTileConfig {
    /// Default LLaVA-1.6 configuration (tile_size=336, max_tiles=6, standard
    /// pinpoints).
    pub fn default_llava16() -> Self {
        Self {
            tile_size: 336,
            max_tiles: 6,
            grid_pinpoints: vec![(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)],
        }
    }

    /// Choose the best grid for an image of `(img_w, img_h)` pixels.
    ///
    /// Returns `(n_cols, n_rows)` for the selected grid.
    ///
    /// # Strategy
    ///
    /// For each candidate `(cols, rows)` we compute the **fill fraction** —
    /// the ratio of image pixels to the total canvas area after fitting the
    /// image inside the grid.  The grid with the highest fill fraction is
    /// selected; ties go to the grid with the smaller total tile count.
    ///
    /// If `grid_pinpoints` is empty, returns `(1, 1)` (single tile).
    pub fn select_grid(&self, img_w: usize, img_h: usize) -> (usize, usize) {
        if self.grid_pinpoints.is_empty() {
            return (1, 1);
        }

        // Guard against divide-by-zero: treat a zero-sized image as 1×1.
        let img_w = img_w.max(1);
        let img_h = img_h.max(1);

        let mut best_grid = self.grid_pinpoints[0];
        let mut best_fill = -1.0_f64;
        let mut best_tiles = usize::MAX;

        for &(cols, rows) in &self.grid_pinpoints {
            if cols == 0 || rows == 0 {
                continue;
            }
            // Total tile count including thumbnail must stay within max_tiles.
            let total_tiles = cols * rows + 1;
            if total_tiles > self.max_tiles {
                continue;
            }

            let canvas_w = (cols * self.tile_size) as f64;
            let canvas_h = (rows * self.tile_size) as f64;

            // Scale image to fit in canvas preserving aspect ratio.
            let scale = (canvas_w / img_w as f64).min(canvas_h / img_h as f64);

            let scaled_w = img_w as f64 * scale;
            let scaled_h = img_h as f64 * scale;

            // Fill fraction: how much of the canvas is covered by the image.
            let fill = (scaled_w * scaled_h) / (canvas_w * canvas_h);

            // Prefer higher fill; break ties by fewer total tiles.
            if fill > best_fill + 1e-9 || (fill > best_fill - 1e-9 && total_tiles < best_tiles) {
                best_fill = fill;
                best_grid = (cols, rows);
                best_tiles = total_tiles;
            }
        }

        best_grid
    }

    /// Split a flat pixel buffer into tile buffers plus a global thumbnail.
    ///
    /// # Arguments
    ///
    /// * `pixels` — Channels-first `[3, img_h, img_w]` normalized float buffer.
    ///   Length must equal `3 * img_h * img_w`.
    /// * `img_w` — Image width in pixels.
    /// * `img_h` — Image height in pixels.
    ///
    /// # Returns
    ///
    /// `(tile_pixels, thumbnail_pixels)` where:
    ///
    /// * `tile_pixels` — `n_cols * n_rows` buffers, each `3 * tile_size *
    ///   tile_size` elements, containing the cropped + bilinearly-scaled tile.
    ///   Ordered row-major (left-to-right, top-to-bottom).
    /// * `thumbnail_pixels` — A single `3 * tile_size * tile_size` buffer
    ///   containing the globally bilinearly-downsampled image (the "overview"
    ///   tile).
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidShape`] if `pixels.len() != 3 * img_w * img_h`.
    /// * [`ArchError::InvalidConfig`] if `tile_size == 0` or `img_w == 0` or
    ///   `img_h == 0`.
    pub fn split_into_tiles(
        &self,
        pixels: &[f32],
        img_w: usize,
        img_h: usize,
    ) -> ArchResult<(Vec<Vec<f32>>, Vec<f32>)> {
        // ── Validate inputs ──────────────────────────────────────────────────
        if self.tile_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "AnyresTileConfig: tile_size must be > 0".to_string(),
            });
        }
        if img_w == 0 || img_h == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "AnyresTileConfig: img_w and img_h must be > 0".to_string(),
            });
        }
        let expected_len = 3 * img_w * img_h;
        if pixels.len() != expected_len {
            return Err(ArchError::InvalidShape {
                name: "anyres pixels".to_string(),
                expected: vec![expected_len],
                got: vec![pixels.len()],
            });
        }

        let (n_cols, n_rows) = self.select_grid(img_w, img_h);

        // ── Compute source region for each grid cell ─────────────────────────
        //
        // We first fit the image inside the full grid canvas
        // (n_cols * tile_size × n_rows * tile_size) using "aspect-ratio
        // preserving resize".  The resized image is then padded to the exact
        // canvas size (centred), and each grid cell crops out exactly one
        // tile_size×tile_size region.
        //
        // Rather than materializing the padded resized image in memory, we
        // compute per-tile source coordinates and sample directly from the
        // original `pixels` via bilinear interpolation.  The padding regions
        // map to 0.0 (black).

        let canvas_w = n_cols * self.tile_size;
        let canvas_h = n_rows * self.tile_size;

        let scale = ((canvas_w as f64) / (img_w as f64)).min((canvas_h as f64) / (img_h as f64));

        let scaled_w = (img_w as f64 * scale).round() as usize;
        let scaled_h = (img_h as f64 * scale).round() as usize;

        // Centre the scaled image on the canvas.
        let pad_left = (canvas_w.saturating_sub(scaled_w)) / 2;
        let pad_top = (canvas_h.saturating_sub(scaled_h)) / 2;

        let tile_buf_len = 3 * self.tile_size * self.tile_size;
        let mut tile_pixels: Vec<Vec<f32>> = Vec::with_capacity(n_cols * n_rows);

        for row_idx in 0..n_rows {
            for col_idx in 0..n_cols {
                // Canvas-space origin of this tile.
                let tile_canvas_x = col_idx * self.tile_size;
                let tile_canvas_y = row_idx * self.tile_size;

                let mut tile = vec![0.0_f32; tile_buf_len];

                for ty in 0..self.tile_size {
                    for tx in 0..self.tile_size {
                        let cx = tile_canvas_x + tx;
                        let cy = tile_canvas_y + ty;

                        // Map canvas pixel (cx, cy) → scaled-image pixel.
                        // If outside the scaled image region (i.e. in the
                        // padding), leave the tile value as 0.0.
                        if cx < pad_left || cy < pad_top {
                            continue;
                        }
                        let sx_f64 = cx.saturating_sub(pad_left) as f64;
                        let sy_f64 = cy.saturating_sub(pad_top) as f64;
                        if sx_f64 >= scaled_w as f64 || sy_f64 >= scaled_h as f64 {
                            continue;
                        }

                        // Map scaled-image coordinate → original-image
                        // coordinate via bilinear interpolation.
                        let src_x =
                            sx_f64 * (img_w as f64 - 1.0) / (scaled_w as f64 - 1.0).max(1.0);
                        let src_y =
                            sy_f64 * (img_h as f64 - 1.0) / (scaled_h as f64 - 1.0).max(1.0);

                        let tile_idx = ty * self.tile_size + tx;

                        for c in 0..3usize {
                            let v = bilinear_sample(pixels, img_w, img_h, c, src_x, src_y);
                            // Tile layout: channels-first [C, H, W]
                            tile[c * self.tile_size * self.tile_size + tile_idx] = v;
                        }
                    }
                }

                tile_pixels.push(tile);
            }
        }

        // ── Global thumbnail (full image downsampled to tile_size × tile_size) ──
        let thumbnail = bilinear_resize(pixels, img_w, img_h, self.tile_size, self.tile_size);

        Ok((tile_pixels, thumbnail))
    }
}

/// Sample a single channel from a channels-first pixel buffer using bilinear
/// interpolation.
///
/// `pixels` has layout `[C, H, W]` (channels-first).
/// Coordinates `(src_x, src_y)` are in pixel space `[0, W-1] × [0, H-1]`.
/// Out-of-bounds coordinates are clamped.
fn bilinear_sample(
    pixels: &[f32],
    img_w: usize,
    img_h: usize,
    channel: usize,
    src_x: f64,
    src_y: f64,
) -> f32 {
    // Clamp to valid range.
    let src_x = src_x.clamp(0.0, (img_w as f64 - 1.0).max(0.0));
    let src_y = src_y.clamp(0.0, (img_h as f64 - 1.0).max(0.0));

    let x0 = src_x.floor() as usize;
    let y0 = src_y.floor() as usize;
    let x1 = (x0 + 1).min(img_w - 1);
    let y1 = (y0 + 1).min(img_h - 1);

    let wx = src_x - x0 as f64;
    let wy = src_y - y0 as f64;

    let ch_offset = channel * img_w * img_h;

    let p00 = pixels[ch_offset + y0 * img_w + x0];
    let p01 = pixels[ch_offset + y0 * img_w + x1];
    let p10 = pixels[ch_offset + y1 * img_w + x0];
    let p11 = pixels[ch_offset + y1 * img_w + x1];

    let top = p00 * (1.0 - wx as f32) + p01 * wx as f32;
    let bot = p10 * (1.0 - wx as f32) + p11 * wx as f32;
    top * (1.0 - wy as f32) + bot * wy as f32
}

/// Bilinearly resize a channels-first `[3, src_h, src_w]` pixel buffer to
/// `[3, dst_h, dst_w]`.
///
/// The output buffer is channels-first (`[3, dst_h, dst_w]`), length
/// `3 * dst_w * dst_h`.
fn bilinear_resize(
    pixels: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; 3 * dst_w * dst_h];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let src_x = dx as f64 * (src_w as f64 - 1.0) / (dst_w as f64 - 1.0).max(1.0);
            let src_y = dy as f64 * (src_h as f64 - 1.0) / (dst_h as f64 - 1.0).max(1.0);

            for c in 0..3usize {
                let v = bilinear_sample(pixels, src_w, src_h, c, src_x, src_y);
                out[c * dst_h * dst_w + dy * dst_w + dx] = v;
            }
        }
    }

    out
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a config with standard LLaVA-1.6 pinpoints.
    fn llava16_config() -> AnyresTileConfig {
        AnyresTileConfig {
            tile_size: 336,
            max_tiles: 6,
            // pinpoints in (cols, rows)
            grid_pinpoints: vec![(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)],
        }
    }

    // ── Grid selection ───────────────────────────────────────────────────────

    /// 800×400 landscape image should select the 2×1 grid (wide layout).
    #[test]
    fn anyres_grid_selection_landscape() {
        let cfg = llava16_config();
        let (cols, rows) = cfg.select_grid(800, 400);
        assert_eq!(
            (cols, rows),
            (2, 1),
            "800×400 landscape should select 2×1 grid, got {cols}×{rows}"
        );
    }

    /// 672×672 square image should select the 2×2 grid.
    #[test]
    fn anyres_grid_selection_square() {
        let cfg = llava16_config();
        let (cols, rows) = cfg.select_grid(672, 672);
        assert_eq!(
            (cols, rows),
            (2, 2),
            "672×672 square should select 2×2 grid, got {cols}×{rows}"
        );
    }

    /// Portrait 400×800 image should select the 1×2 grid.
    #[test]
    fn anyres_grid_selection_portrait() {
        let cfg = llava16_config();
        let (cols, rows) = cfg.select_grid(400, 800);
        assert_eq!(
            (cols, rows),
            (1, 2),
            "400×800 portrait should select 1×2 grid, got {cols}×{rows}"
        );
    }

    /// Empty pinpoints list falls back to (1, 1).
    #[test]
    fn anyres_grid_selection_empty_pinpoints_returns_1x1() {
        let cfg = AnyresTileConfig {
            tile_size: 336,
            max_tiles: 6,
            grid_pinpoints: vec![],
        };
        assert_eq!(cfg.select_grid(1024, 512), (1, 1));
    }

    // ── Tile splitting ───────────────────────────────────────────────────────

    /// A 2×2 grid should produce 4 tile buffers + 1 thumbnail = 5 total.
    #[test]
    fn anyres_tile_split_correct_count() {
        let cfg = AnyresTileConfig {
            tile_size: 28, // small for speed
            max_tiles: 6,
            grid_pinpoints: vec![(2, 2)], // force 2×2 grid
        };
        let img_w = 56usize;
        let img_h = 56usize;
        let pixels = vec![0.5_f32; 3 * img_w * img_h];
        let (tiles, _thumbnail) = cfg
            .split_into_tiles(&pixels, img_w, img_h)
            .expect("split ok");
        assert_eq!(
            tiles.len(),
            4,
            "2×2 grid must produce exactly 4 tile buffers"
        );
    }

    /// Every tile buffer must be exactly tile_size × tile_size × 3 elements.
    #[test]
    fn anyres_tile_dimensions_correct() {
        let tile_size = 14usize;
        let cfg = AnyresTileConfig {
            tile_size,
            max_tiles: 6,
            grid_pinpoints: vec![(2, 2)],
        };
        let img_w = 28usize;
        let img_h = 28usize;
        let pixels = vec![0.3_f32; 3 * img_w * img_h];
        let (tiles, thumbnail) = cfg
            .split_into_tiles(&pixels, img_w, img_h)
            .expect("split ok");

        let expected = 3 * tile_size * tile_size;
        for (i, tile) in tiles.iter().enumerate() {
            assert_eq!(
                tile.len(),
                expected,
                "tile {i} should have {expected} elements, got {}",
                tile.len()
            );
        }
        assert_eq!(
            thumbnail.len(),
            expected,
            "thumbnail should have {expected} elements, got {}",
            thumbnail.len()
        );
    }

    /// Passing wrong-length pixel array must return Err(InvalidShape).
    #[test]
    fn anyres_tile_split_wrong_pixel_count_errors() {
        let cfg = AnyresTileConfig {
            tile_size: 28,
            max_tiles: 6,
            grid_pinpoints: vec![(1, 1)],
        };
        let pixels = vec![0.0_f32; 100]; // wrong length
        assert!(
            cfg.split_into_tiles(&pixels, 10, 10).is_err(),
            "wrong pixel count must return Err"
        );
    }

    /// Zero tile_size must return Err(InvalidConfig).
    #[test]
    fn anyres_tile_split_zero_tile_size_errors() {
        let cfg = AnyresTileConfig {
            tile_size: 0,
            max_tiles: 6,
            grid_pinpoints: vec![(1, 1)],
        };
        let pixels = vec![0.0_f32; 3 * 10 * 10];
        assert!(
            cfg.split_into_tiles(&pixels, 10, 10).is_err(),
            "tile_size=0 must return Err"
        );
    }

    /// Bilinear resize should preserve pixel values for a uniform image.
    #[test]
    fn bilinear_resize_uniform_image_stays_uniform() {
        let src_w = 10usize;
        let src_h = 10usize;
        let dst_w = 5usize;
        let dst_h = 5usize;
        let pixels = vec![0.7_f32; 3 * src_w * src_h];
        let out = bilinear_resize(&pixels, src_w, src_h, dst_w, dst_h);
        for v in &out {
            assert!(
                (v - 0.7).abs() < 1e-5,
                "uniform image must remain uniform after resize, got {v}"
            );
        }
    }
}

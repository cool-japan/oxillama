//! Image-embedding injection: splicing projected visual tokens into a prompt.
//!
//! Before this module existed, the whole vision stack was dead code:
//! `LlavaModel::forward` ignored the tower entirely, `encode_image` had no
//! callers anywhere in the workspace, and no API accepted hidden states in
//! place of token ids — so a caller could not have injected anything even if it
//! wanted to.
//!
//! ## What the reference does
//!
//! `tools/mtmd/mtmd.cpp` splits the prompt text on a media marker and keeps the
//! marker as its own element (`split_text`, `mtmd.cpp:764-782`), producing an
//! alternating chunk list.  Text chunks are tokenized; image chunks carry
//! embeddings.  Decoding an image chunk builds a `llama_batch` with
//! `tokens = nullptr` and `embd = <float*>` (`mtmd-helper.cpp:129-145`) and
//! calls `llama_decode` on it (`mtmd-helper.cpp:275-289`).
//!
//! Crucially, **no placeholder token ever enters the token stream** in the mtmd
//! path — the marker string is never passed to `llama_tokenize`.
//!
//! ## What this module offers
//!
//! Two entry points, because oxillama's tokenizer lives in `oxillama-runtime`
//! and this crate only sees token ids:
//!
//! * [`Prompt::from_segments`] — the mtmd shape.  The caller has already split
//!   the text on its marker and tokenized each part.
//! * [`Prompt::from_placeholder`] — the HuggingFace LLaVA-1.5 shape, where the
//!   prompt genuinely contains an `<image>` placeholder id (32000 for
//!   `llava-1.5-7b-hf`) that is *replaced* by the projected patch embeddings.
//!
//! Both produce the same thing: a flat `[seq_len × hidden_size]` input
//! embedding buffer, which is exactly the `llama_batch.embd` payload the
//! reference decodes.

use crate::error::{ArchError, ArchResult};

/// Projected visual tokens for one image, `[n_tokens × hidden_size]`.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualTokens {
    data: Vec<f32>,
    hidden_size: usize,
}

impl VisualTokens {
    /// Wrap a projector output.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] when `hidden_size` is zero;
    /// [`ArchError::InvalidShape`] when `data` is not a whole number of rows.
    pub fn new(data: Vec<f32>, hidden_size: usize) -> ArchResult<Self> {
        if hidden_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "VisualTokens: hidden_size must be > 0".to_string(),
            });
        }
        if !data.len().is_multiple_of(hidden_size) {
            return Err(ArchError::InvalidShape {
                name: "visual tokens".to_string(),
                expected: vec![hidden_size],
                got: vec![data.len()],
            });
        }
        Ok(Self { data, hidden_size })
    }

    /// Number of visual tokens (576 for LLaVA-1.5).
    pub fn len(&self) -> usize {
        self.data.len() / self.hidden_size
    }

    /// Whether this image contributes no tokens.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Row width — must equal the language model's hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// The flat `[n_tokens × hidden_size]` buffer.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// One piece of a multimodal prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment<'a> {
    /// A run of ordinary token ids.
    Text(&'a [u32]),
    /// The image at this index in the caller's image list.
    Image(usize),
}

/// A prompt with image slots resolved to positions in a flat embedding buffer.
#[derive(Debug, Clone)]
pub struct Prompt<'a> {
    segments: Vec<Segment<'a>>,
    images: &'a [VisualTokens],
}

impl<'a> Prompt<'a> {
    /// Build a prompt from an explicit segment list (the mtmd chunk shape).
    ///
    /// # Errors
    ///
    /// [`ArchError::ConfigMismatch`] when a [`Segment::Image`] names an index
    /// outside `images`.
    pub fn from_segments(
        segments: Vec<Segment<'a>>,
        images: &'a [VisualTokens],
    ) -> ArchResult<Self> {
        for seg in &segments {
            if let Segment::Image(idx) = seg {
                if *idx >= images.len() {
                    return Err(ArchError::ConfigMismatch {
                        param: "image index".to_string(),
                        expected: format!("< {}", images.len()),
                        got: idx.to_string(),
                    });
                }
            }
        }
        Ok(Self { segments, images })
    }

    /// Split `tokens` on every occurrence of `placeholder`, substituting the
    /// images in order.
    ///
    /// This is the HuggingFace LLaVA-1.5 convention: the tokenized prompt holds
    /// one `<image>` id per image, and each is *replaced* — not surrounded — by
    /// that image's projected patch embeddings.
    ///
    /// # Errors
    ///
    /// [`ArchError::ConfigMismatch`] when the number of placeholders does not
    /// equal `images.len()`.  A mismatch is always a caller bug and silently
    /// dropping either side would produce a subtly wrong sequence.
    pub fn from_placeholder(
        tokens: &'a [u32],
        placeholder: u32,
        images: &'a [VisualTokens],
    ) -> ArchResult<Self> {
        let count = tokens.iter().filter(|&&t| t == placeholder).count();
        if count != images.len() {
            return Err(ArchError::ConfigMismatch {
                param: format!("occurrences of placeholder token {placeholder}"),
                expected: images.len().to_string(),
                got: count.to_string(),
            });
        }

        let mut segments = Vec::with_capacity(2 * count + 1);
        let mut start = 0usize;
        let mut image_idx = 0usize;
        for (i, &tok) in tokens.iter().enumerate() {
            if tok != placeholder {
                continue;
            }
            if i > start {
                segments.push(Segment::Text(&tokens[start..i]));
            }
            segments.push(Segment::Image(image_idx));
            image_idx += 1;
            start = i + 1;
        }
        if start < tokens.len() {
            segments.push(Segment::Text(&tokens[start..]));
        }

        Ok(Self { segments, images })
    }

    /// The segment list.
    pub fn segments(&self) -> &[Segment<'a>] {
        &self.segments
    }

    /// Total length of the spliced sequence in positions.
    pub fn seq_len(&self) -> usize {
        self.segments
            .iter()
            .map(|seg| match seg {
                Segment::Text(t) => t.len(),
                Segment::Image(idx) => self.images.get(*idx).map_or(0, VisualTokens::len),
            })
            .sum()
    }

    /// Materialise the flat `[seq_len × hidden_size]` input embedding buffer.
    ///
    /// `embed_token` writes one token's embedding row into the supplied slice —
    /// for LLaVA that is the backbone's `token_embd` lookup.
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidConfig`] — `hidden_size` is zero.
    /// * [`ArchError::ConfigMismatch`] — an image's row width disagrees with
    ///   `hidden_size`, or a segment names a missing image.
    /// * whatever `embed_token` returns for an out-of-vocabulary id.
    pub fn build_embeddings<F>(
        &self,
        hidden_size: usize,
        mut embed_token: F,
    ) -> ArchResult<Vec<f32>>
    where
        F: FnMut(u32, &mut [f32]) -> ArchResult<()>,
    {
        if hidden_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "build_embeddings: hidden_size must be > 0".to_string(),
            });
        }

        let mut out = vec![0.0f32; self.seq_len() * hidden_size];
        let mut cursor = 0usize;

        for seg in &self.segments {
            match seg {
                Segment::Text(tokens) => {
                    for &tok in tokens.iter() {
                        let row = &mut out[cursor..cursor + hidden_size];
                        embed_token(tok, row)?;
                        cursor += hidden_size;
                    }
                }
                Segment::Image(idx) => {
                    let img = self
                        .images
                        .get(*idx)
                        .ok_or_else(|| ArchError::ConfigMismatch {
                            param: "image index".to_string(),
                            expected: format!("< {}", self.images.len()),
                            got: idx.to_string(),
                        })?;
                    if img.hidden_size() != hidden_size {
                        return Err(ArchError::ConfigMismatch {
                            param: "visual token width".to_string(),
                            expected: hidden_size.to_string(),
                            got: img.hidden_size().to_string(),
                        });
                    }
                    let n = img.as_slice().len();
                    out[cursor..cursor + n].copy_from_slice(img.as_slice());
                    cursor += n;
                }
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Embed by repeating the token id — makes provenance obvious in asserts.
    fn id_embed(hidden: usize) -> impl FnMut(u32, &mut [f32]) -> ArchResult<()> {
        move |tok, row| {
            for (i, v) in row.iter_mut().enumerate() {
                *v = tok as f32 + i as f32 * 0.01;
            }
            let _ = hidden;
            Ok(())
        }
    }

    fn image(n_tokens: usize, hidden: usize, fill: f32) -> VisualTokens {
        VisualTokens::new(vec![fill; n_tokens * hidden], hidden).expect("visual tokens")
    }

    #[test]
    fn placeholder_split_produces_alternating_segments() {
        let tokens = [1u32, 2, 99, 3, 4];
        let images = [image(3, 4, 7.0)];
        let prompt = Prompt::from_placeholder(&tokens, 99, &images).expect("split");
        assert_eq!(
            prompt.segments(),
            &[
                Segment::Text(&[1, 2]),
                Segment::Image(0),
                Segment::Text(&[3, 4]),
            ]
        );
        // 2 text + 3 visual + 2 text
        assert_eq!(prompt.seq_len(), 7);
    }

    #[test]
    fn placeholder_at_the_edges_is_handled() {
        let tokens = [99u32, 1, 2];
        let images = [image(2, 4, 1.0)];
        let p = Prompt::from_placeholder(&tokens, 99, &images).expect("split");
        assert_eq!(p.segments(), &[Segment::Image(0), Segment::Text(&[1, 2])]);

        let tokens = [1u32, 2, 99];
        let p = Prompt::from_placeholder(&tokens, 99, &images).expect("split");
        assert_eq!(p.segments(), &[Segment::Text(&[1, 2]), Segment::Image(0)]);

        let tokens = [99u32];
        let p = Prompt::from_placeholder(&tokens, 99, &images).expect("split");
        assert_eq!(p.segments(), &[Segment::Image(0)]);
    }

    #[test]
    fn multiple_images_are_substituted_in_order() {
        let tokens = [1u32, 99, 2, 99, 3];
        let images = [image(1, 2, 10.0), image(1, 2, 20.0)];
        let p = Prompt::from_placeholder(&tokens, 99, &images).expect("split");
        let embeds = p.build_embeddings(2, id_embed(2)).expect("build");
        assert_eq!(embeds.len(), 5 * 2);
        // rows: [tok1][img0][tok2][img1][tok3]
        assert_eq!(&embeds[2..4], &[10.0, 10.0]);
        assert_eq!(&embeds[6..8], &[20.0, 20.0]);
    }

    #[test]
    fn placeholder_count_mismatch_is_reported() {
        let tokens = [1u32, 99, 2];
        let two = [image(1, 2, 1.0), image(1, 2, 2.0)];
        assert!(Prompt::from_placeholder(&tokens, 99, &two).is_err());

        let none: [VisualTokens; 0] = [];
        assert!(Prompt::from_placeholder(&tokens, 99, &none).is_err());
    }

    /// The whole point of V1: image embeddings actually land in the buffer at
    /// the placeholder's position, replacing it.
    #[test]
    fn image_embeddings_replace_the_placeholder_row() {
        let hidden = 4usize;
        let tokens = [5u32, 99, 6];
        let images = [image(2, hidden, 3.5)];
        let p = Prompt::from_placeholder(&tokens, 99, &images).expect("split");

        assert_eq!(p.seq_len(), 4, "1 text + 2 visual + 1 text");
        let embeds = p.build_embeddings(hidden, id_embed(hidden)).expect("build");
        assert_eq!(embeds.len(), 4 * hidden);

        // Row 0 is token 5.
        assert_eq!(embeds[0], 5.0);
        // Rows 1 and 2 are the image — and note 99 never appears anywhere.
        for v in &embeds[hidden..3 * hidden] {
            assert_eq!(*v, 3.5);
        }
        // Row 3 is token 6.
        assert_eq!(embeds[3 * hidden], 6.0);
        assert!(
            !embeds.iter().any(|v| (*v - 99.0).abs() < 1e-6),
            "the placeholder id must never be embedded"
        );
    }

    #[test]
    fn width_mismatch_is_reported() {
        let tokens = [99u32];
        let images = [image(1, 8, 1.0)];
        let p = Prompt::from_placeholder(&tokens, 99, &images).expect("split");
        assert!(p.build_embeddings(4, id_embed(4)).is_err());
    }

    #[test]
    fn segment_form_matches_placeholder_form() {
        let tokens = [1u32, 2, 3];
        let images = [image(2, 4, 9.0)];
        let by_seg = Prompt::from_segments(
            vec![
                Segment::Text(&tokens[..2]),
                Segment::Image(0),
                Segment::Text(&tokens[2..]),
            ],
            &images,
        )
        .expect("segments");

        let with_ph = [1u32, 2, 99, 3];
        let by_ph = Prompt::from_placeholder(&with_ph, 99, &images).expect("placeholder");

        let a = by_seg.build_embeddings(4, id_embed(4)).expect("a");
        let b = by_ph.build_embeddings(4, id_embed(4)).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn out_of_range_image_index_is_reported() {
        let tokens = [1u32];
        let images = [image(1, 4, 1.0)];
        assert!(Prompt::from_segments(vec![Segment::Image(7)], &images).is_err());
        let _ = tokens;
    }

    #[test]
    fn visual_tokens_reject_ragged_data() {
        assert!(VisualTokens::new(vec![0.0; 5], 4).is_err());
        assert!(VisualTokens::new(vec![0.0; 8], 0).is_err());
        assert_eq!(
            VisualTokens::new(vec![0.0; 8], 4).expect("ok").len(),
            2,
            "8 floats / width 4 = 2 tokens"
        );
    }

    #[test]
    fn text_only_prompt_is_a_plain_lookup() {
        let tokens = [1u32, 2, 3];
        let none: [VisualTokens; 0] = [];
        let p = Prompt::from_placeholder(&tokens, 99, &none).expect("no images");
        assert_eq!(p.seq_len(), 3);
        let e = p.build_embeddings(2, id_embed(2)).expect("build");
        assert_eq!(e.len(), 6);
        assert_eq!(e[0], 1.0);
        assert_eq!(e[2], 2.0);
        assert_eq!(e[4], 3.0);
    }
}

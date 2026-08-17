//! LLaVA multi-modal projector: CLIP features → LLM embedding space.
//!
//! Reference: `tools/mtmd/models/llava.cpp:165-176` (`PROJECTOR_TYPE_MLP`):
//!
//! ```text
//! embeddings = mul_mat(mm_0_w, embeddings) + mm_0_b
//! embeddings = ggml_gelu(embeddings)
//! if (mm_2_w) embeddings = mul_mat(mm_2_w, embeddings) + mm_2_b
//! ```
//!
//! Note the projector uses plain `ggml_gelu` — the **tanh approximation** —
//! regardless of `hparams.ffn_op`.  This is deliberately *different* from the
//! vision tower, which defaults to QuickGELU; see [`super::clip`].
//!
//! Tensor names come from `TN_LLAVA_PROJ = "mm.%d.%s"` (`clip-impl.h:92`), so
//! LLaVA-1.5's two-layer MLP is `mm.0.*` and `mm.2.*` (index 1 is the LayerNorm
//! slot used only by the Yi-type `MLP_NORM` projector).

use crate::error::{ArchError, ArchResult};

/// Multi-modal projector: a 2-layer MLP mapping CLIP features to LLM hidden dim.
#[derive(Debug, Clone)]
pub struct MmProjector {
    /// `mm.0.weight`, row-major `[mm_hidden_size, clip_hidden_size]`.
    pub fc1_weight: Vec<f32>,
    /// `mm.0.bias`, `[mm_hidden_size]`.
    pub fc1_bias: Vec<f32>,
    /// `mm.2.weight`, row-major `[llm_hidden_size, mm_hidden_size]`.
    pub fc2_weight: Vec<f32>,
    /// `mm.2.bias`, `[llm_hidden_size]`.
    pub fc2_bias: Vec<f32>,
    /// Input width — the CLIP tower's hidden size.
    pub clip_hidden_size: usize,
    /// Hidden width between the two linears.
    pub mm_hidden_size: usize,
    /// Output width — the language model's hidden size.
    pub llm_hidden_size: usize,
}

impl MmProjector {
    /// Validate that the weight buffers match the declared shapes.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidConfig`] for a zero dimension;
    /// [`ArchError::InvalidShape`] for a buffer shorter than its shape.
    pub fn validate(&self) -> ArchResult<()> {
        if self.clip_hidden_size == 0 || self.mm_hidden_size == 0 || self.llm_hidden_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "MmProjector: clip/mm/llm hidden sizes must all be > 0".to_string(),
            });
        }
        let fc1_need = self.mm_hidden_size * self.clip_hidden_size;
        if self.fc1_weight.len() < fc1_need {
            return Err(ArchError::InvalidShape {
                name: "mm.0.weight".to_string(),
                expected: vec![self.mm_hidden_size, self.clip_hidden_size],
                got: vec![self.fc1_weight.len()],
            });
        }
        let fc2_need = self.llm_hidden_size * self.mm_hidden_size;
        if self.fc2_weight.len() < fc2_need {
            return Err(ArchError::InvalidShape {
                name: "mm.2.weight".to_string(),
                expected: vec![self.llm_hidden_size, self.mm_hidden_size],
                got: vec![self.fc2_weight.len()],
            });
        }
        Ok(())
    }

    /// Project CLIP image features into the LLM embedding space.
    ///
    /// # Arguments
    /// * `input` — flat `[num_patches × clip_hidden_size]`.
    ///
    /// # Returns
    /// Flat `[num_patches × llm_hidden_size]`.
    ///
    /// # Errors
    /// * [`ArchError::InvalidConfig`] — zero dimension.
    /// * [`ArchError::InvalidShape`] — `input` is not a whole number of patches,
    ///   or a weight buffer is short.  The previous implementation indexed
    ///   `fc1_weight[i * clip .. (i+1) * clip]` unguarded, which panicked on a
    ///   truncated tensor.
    pub fn project(&self, input: &[f32]) -> ArchResult<Vec<f32>> {
        self.validate()?;
        if !input.len().is_multiple_of(self.clip_hidden_size) {
            return Err(ArchError::InvalidShape {
                name: "mm_projector input".to_string(),
                expected: vec![self.clip_hidden_size],
                got: vec![input.len()],
            });
        }

        let num_patches = input.len() / self.clip_hidden_size;
        let mut out = Vec::with_capacity(num_patches * self.llm_hidden_size);
        let mut h = vec![0.0f32; self.mm_hidden_size];

        for patch_idx in 0..num_patches {
            let patch =
                &input[patch_idx * self.clip_hidden_size..(patch_idx + 1) * self.clip_hidden_size];

            for (i, h_val) in h.iter_mut().enumerate() {
                let row =
                    &self.fc1_weight[i * self.clip_hidden_size..(i + 1) * self.clip_hidden_size];
                *h_val = row
                    .iter()
                    .zip(patch.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
                    + self.fc1_bias.get(i).copied().unwrap_or(0.0);
                // `ggml_gelu` — the tanh approximation, not QuickGELU.
                *h_val = crate::common::gelu::gelu(*h_val);
            }

            for i in 0..self.llm_hidden_size {
                let row = &self.fc2_weight[i * self.mm_hidden_size..(i + 1) * self.mm_hidden_size];
                out.push(
                    row.iter().zip(h.iter()).map(|(w, x)| w * x).sum::<f32>()
                        + self.fc2_bias.get(i).copied().unwrap_or(0.0),
                );
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proj(clip: usize, mm: usize, llm: usize) -> MmProjector {
        MmProjector {
            fc1_weight: vec![0.1; mm * clip],
            fc1_bias: vec![0.0; mm],
            fc2_weight: vec![0.1; llm * mm],
            fc2_bias: vec![0.0; llm],
            clip_hidden_size: clip,
            mm_hidden_size: mm,
            llm_hidden_size: llm,
        }
    }

    #[test]
    fn projector_output_shape() {
        let p = proj(4, 8, 4);
        assert_eq!(p.project(&[1.0f32; 4]).expect("one patch").len(), 4);
        assert_eq!(p.project(&[1.0f32; 8]).expect("two patches").len(), 8);
    }

    #[test]
    fn projector_rejects_ragged_input() {
        assert!(proj(4, 2, 4).project(&[1.0f32; 3]).is_err());
    }

    #[test]
    fn projector_rejects_zero_dims() {
        let mut p = proj(4, 4, 4);
        p.clip_hidden_size = 0;
        assert!(p.project(&[1.0f32; 4]).is_err());
    }

    /// A truncated `mm.0.weight` used to panic on an unguarded slice; it must
    /// now be reported.
    #[test]
    fn projector_rejects_short_weights() {
        let mut p = proj(4, 8, 4);
        p.fc1_weight.truncate(5);
        assert!(matches!(
            p.project(&[1.0f32; 4]),
            Err(ArchError::InvalidShape { .. })
        ));

        let mut p = proj(4, 8, 4);
        p.fc2_weight.truncate(5);
        assert!(matches!(
            p.project(&[1.0f32; 4]),
            Err(ArchError::InvalidShape { .. })
        ));
    }

    #[test]
    fn projector_positive_input_stays_positive() {
        let p = proj(2, 2, 2);
        for v in p.project(&[1.0f32, 1.0]).expect("project") {
            assert!(v > 0.0, "got {v}");
        }
    }
}

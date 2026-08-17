//! GGUF loading for Mamba-2.
//!
//! The tensor set is exactly `LLM_ARCH_MAMBA2`'s entry in `LLM_TENSOR_NAMES`
//! (`src/llama-arch.cpp:1388`):
//!
//! ```text
//! TOKEN_EMBD, OUTPUT_NORM, OUTPUT, ATTN_NORM,
//! SSM_IN, SSM_CONV1D, SSM_DT, SSM_A, SSM_D, SSM_NORM, SSM_OUT
//! ```
//!
//! with the shapes created at `src/llama-model.cpp:4585`:
//!
//! ```text
//! ssm_in       {n_embd, 2*d_inner + 2*n_group*d_state + n_head}
//! ssm_conv1d   {d_conv, d_inner + 2*n_group*d_state}
//! ssm_conv1d_b {d_inner + 2*n_group*d_state}
//! ssm_dt_b     {n_head}                       // bias only — no SSM_DT weight
//! ssm_a        {1, n_head}
//! ssm_d        {1, n_head}
//! ssm_norm     {d_inner / n_group, n_group}
//! ssm_out      {d_inner, n_embd}
//! ```
//!
//! # Name casing
//!
//! `LLM_TENSOR_SSM_A` and `LLM_TENSOR_SSM_D` are spelled **lowercase** —
//! `blk.%d.ssm_a` / `blk.%d.ssm_d` (`src/llama-arch.cpp:388-389`,
//! `gguf-py/gguf/constants.py:1008,1011`).  The uppercase `ssm_A` / `ssm_D`
//! spellings are accepted as a fallback only so that files written by this
//! crate's pre-0.1.4 fixture still load.
//!
//! # Every read is bounds-checked
//!
//! All dequantization goes through [`crate::common::loader`], which validates
//! the payload length against the declared element count instead of slicing a
//! possibly-truncated mmap.

use oxillama_gguf::GgufModel;

use crate::common::loader::{load_dequant_tensor, load_rms_norm_weight};
use crate::common::rms_norm::RmsNorm;
use crate::error::{ArchError, ArchResult};
use crate::mamba2::config::Mamba2Config;
use crate::mamba2::model::{Mamba2LayerWeights, Mamba2Model};

/// Load a tensor by its canonical name, falling back to a legacy spelling.
fn load_aliased(model: &GgufModel, primary: &str, fallback: &str) -> ArchResult<Vec<f32>> {
    if model.file.tensors.contains(primary) {
        return load_dequant_tensor(model, primary);
    }
    if model.file.tensors.contains(fallback) {
        return load_dequant_tensor(model, fallback);
    }
    Err(ArchError::MissingTensor {
        name: format!("{primary} (no fallback '{fallback}' either)"),
    })
}

/// Assert a loaded tensor's element count.
fn expect_len(name: &str, data: &[f32], expected: usize) -> ArchResult<()> {
    if data.len() == expected {
        Ok(())
    } else {
        Err(ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![expected],
            got: vec![data.len()],
        })
    }
}

/// Load a Mamba-2 model from a parsed GGUF file.
///
/// `vocab_size` is taken from the `token_embd.weight` row count rather than
/// from metadata: llama.cpp derives `n_vocab` from the embedding tensor too,
/// and an over-estimated `{arch}.vocab_size` would otherwise turn into an
/// out-of-bounds embedding lookup on the first forward pass.  A metadata value
/// that disagrees with the tensor is an error, not a silent override.
///
/// # Errors
///
/// * [`ArchError::InvalidConfig`] when the hyper-parameters are inconsistent.
/// * [`ArchError::MissingTensor`] naming any required tensor that is absent.
/// * [`ArchError::InvalidShape`] naming the first tensor whose element count
///   disagrees with the config, or whose payload is truncated.
pub fn load_mamba2_from_gguf(model: &GgufModel) -> ArchResult<Mamba2Model> {
    let mut cfg = Mamba2Config::from_metadata(&model.file.metadata)?;

    let d_model = cfg.d_model;
    let d_inner = cfg.d_inner;
    let d_conv = cfg.d_conv;
    let n_head = cfg.n_head;
    let conv_dim = cfg.conv_dim();
    let d_in_proj = cfg.d_in_proj();

    // ── Token embeddings decide the real vocabulary size ──────────────────────
    let token_embd = load_dequant_tensor(model, "token_embd.weight")?;
    if d_model == 0 || token_embd.len() % d_model != 0 {
        return Err(ArchError::InvalidShape {
            name: "token_embd.weight".to_string(),
            expected: vec![d_model],
            got: vec![token_embd.len()],
        });
    }
    let rows = token_embd.len() / d_model;
    if rows == 0 {
        return Err(ArchError::InvalidShape {
            name: "token_embd.weight".to_string(),
            expected: vec![1, d_model],
            got: vec![token_embd.len()],
        });
    }
    // A metadata vocab of 1 means "absent" (see `Mamba2Config::from_metadata`).
    if cfg.vocab_size > 1 && cfg.vocab_size != rows {
        return Err(ArchError::InvalidShape {
            name: "token_embd.weight".to_string(),
            expected: vec![cfg.vocab_size, d_model],
            got: vec![rows, d_model],
        });
    }
    cfg.vocab_size = rows;
    cfg.validate()?;

    // ── Per-layer weights ─────────────────────────────────────────────────────
    let mut layers = Vec::with_capacity(cfg.n_layer);
    for i in 0..cfg.n_layer {
        let pfx = format!("blk.{i}");

        let norm_name = format!("{pfx}.attn_norm.weight");
        let norm_weights = if model.file.tensors.contains(&norm_name) {
            load_rms_norm_weight(model, &norm_name)?
        } else {
            load_rms_norm_weight(model, &format!("{pfx}.norm.weight"))?
        };
        expect_len(&norm_name, &norm_weights, d_model)?;

        let w_in_name = format!("{pfx}.ssm_in.weight");
        let w_in = load_dequant_tensor(model, &w_in_name)?;
        expect_len(&w_in_name, &w_in, d_in_proj * d_model)?;

        let w_conv_name = format!("{pfx}.ssm_conv1d.weight");
        let w_conv = load_dequant_tensor(model, &w_conv_name)?;
        expect_len(&w_conv_name, &w_conv, conv_dim * d_conv)?;

        // Required in llama.cpp (`create_tensor(..., 0)`); a zero-filled default
        // would silently change the convolution's output.
        let b_conv_name = format!("{pfx}.ssm_conv1d.bias");
        let b_conv = load_dequant_tensor(model, &b_conv_name)?;
        expect_len(&b_conv_name, &b_conv, conv_dim)?;

        // MAMBA2 has no `ssm_dt` *weight*: dt comes straight out of `ssm_in`.
        let dt_bias_name = format!("{pfx}.ssm_dt.bias");
        let dt_bias = load_dequant_tensor(model, &dt_bias_name)?;
        expect_len(&dt_bias_name, &dt_bias, n_head)?;

        let a_name = format!("{pfx}.ssm_a");
        let a = load_aliased(model, &a_name, &format!("{pfx}.ssm_A"))?;
        expect_len(&a_name, &a, n_head)?;

        let d_name = format!("{pfx}.ssm_d");
        let d_skip = load_aliased(model, &d_name, &format!("{pfx}.ssm_D"))?;
        expect_len(&d_name, &d_skip, n_head)?;

        let ssm_norm_name = format!("{pfx}.ssm_norm.weight");
        let ssm_norm = load_rms_norm_weight(model, &ssm_norm_name)?;
        expect_len(&ssm_norm_name, &ssm_norm, d_inner)?;

        let w_out_name = format!("{pfx}.ssm_out.weight");
        let w_out = load_dequant_tensor(model, &w_out_name)?;
        expect_len(&w_out_name, &w_out, d_model * d_inner)?;

        layers.push(Mamba2LayerWeights {
            norm: RmsNorm::new(norm_weights, cfg.rms_norm_eps),
            w_in,
            w_conv,
            b_conv,
            dt_bias,
            a,
            d_skip,
            ssm_norm,
            w_out,
        });
    }

    // ── Final norm and LM head ────────────────────────────────────────────────
    let output_norm_weights = load_rms_norm_weight(model, "output_norm.weight")?;
    expect_len("output_norm.weight", &output_norm_weights, d_model)?;
    let output_norm = RmsNorm::new(output_norm_weights, cfg.rms_norm_eps);

    // `output.weight` is TENSOR_NOT_REQUIRED for MAMBA2; llama.cpp duplicates
    // `token_embd.weight` when it is absent.  Presence is tested explicitly so
    // that a *corrupt* `output.weight` reports its own error instead of
    // silently falling back to the tied embedding.
    let lm_head = if model.file.tensors.contains("output.weight") {
        let head = load_dequant_tensor(model, "output.weight")?;
        expect_len("output.weight", &head, cfg.vocab_size * d_model)?;
        head
    } else {
        token_embd.clone()
    };

    Mamba2Model::new(cfg, token_embd, layers, output_norm, lm_head)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> GgufModel {
        let bytes = oxillama_gguf::test_utils::build_minimal_mamba2_gguf();
        GgufModel::from_bytes(bytes).expect("fixture must parse")
    }

    /// The fixture must carry the real llama.cpp MAMBA2 tensor set.
    #[test]
    fn fixture_has_the_llama_cpp_mamba2_tensors() {
        let gguf = fixture();
        for name in [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ssm_in.weight",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_conv1d.bias",
            "blk.0.ssm_dt.bias",
            "blk.0.ssm_a",
            "blk.0.ssm_d",
            "blk.0.ssm_norm.weight",
            "blk.0.ssm_out.weight",
        ] {
            assert!(
                gguf.file.tensors.contains(name),
                "fixture is missing '{name}'"
            );
        }
        for name in ["blk.0.ssm_x.weight", "blk.0.ssm_dt.weight"] {
            assert!(
                !gguf.file.tensors.contains(name),
                "fixture still carries the Mamba-1 tensor '{name}'"
            );
        }
    }

    /// `mamba2.ssm.group_count` must reach the config.
    #[test]
    fn group_count_is_read_from_metadata() {
        let gguf = fixture();
        let cfg = Mamba2Config::from_metadata(&gguf.file.metadata).expect("config");
        assert_eq!(cfg.n_group, 2, "mamba2.ssm.group_count");
        assert_eq!(cfg.n_head, 4, "mamba2.ssm.time_step_rank");
        assert_eq!(cfg.d_inner, 32, "mamba2.ssm.inner_size");
        assert_eq!(cfg.d_state, 8, "mamba2.ssm.state_size");
        assert_eq!(cfg.d_conv, 4, "mamba2.ssm.conv_kernel");
        assert!(
            (cfg.rms_norm_eps - 1e-5).abs() < 1e-12,
            "mamba2.attention.layer_norm_rms_epsilon: {}",
            cfg.rms_norm_eps
        );
    }

    /// A missing required tensor is reported by name.
    #[test]
    fn missing_tensor_is_named() {
        let err = load_aliased(&fixture(), "blk.0.nope", "blk.0.also_nope")
            .expect_err("absent tensor must error");
        assert!(format!("{err}").contains("blk.0.nope"), "{err}");
    }

    /// The loader derives `vocab_size` from `token_embd`.
    #[test]
    fn vocab_size_comes_from_the_embedding_table() {
        let model = load_mamba2_from_gguf(&fixture()).expect("load");
        assert_eq!(model.config.vocab_size, 256);
        assert_eq!(
            model.token_embd.len(),
            256 * 16,
            "token_embd must be vocab_size × d_model"
        );
    }
}
